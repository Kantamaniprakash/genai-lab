# slm-judge-audit — research log

One entry per working day, newest last. Every claim in the README traces to an
entry here; every entry ends with exact next steps.

## 2026-07-17 — Day 1: flagship selection, harness core, feasibility pilot

### Selection scan

`rag-chunking-bench` closed yesterday; today opened with the fresh landscape
scan the ROADMAP called for. Judge reliability is the clear center of gravity
in eval research right now — a 2026 wave including "Reliability without
Validity" (arXiv:2606.19544; 21 judges, ~541k judgments, agreement/
consistency/bias protocols), a self-preference quantification line
(arXiv:2604.22891, arXiv:2410.21819), position-bias mitigation surveys
(arXiv:2604.23178), and a small-judge line (JudgeBoard arXiv:2511.15958,
SLMJury arXiv:2606.07810, "Thinking Small" arXiv:2509.13332). Candidates from
the backlog considered against the scan:

- **LLM-as-judge reliability audit** — hottest area, but the backlog framing
  assumed hosted-API judges. Verified today: HF Inference API router is
  reachable and authenticated, but the account is free-tier prepaid
  (`canPay: false`) — thousands of frontier-judge calls are not fundable.
  Reframed instead of dropped (below).
- **Hallucination/faithfulness bench** — good machinery reuse, but needs
  generation at scale (same API problem) and the span-metrics angle is closer
  to what the last project already did; weaker novelty delta.
- Other backlog items (agent reliability, query-noise robustness,
  time-series FMs) — none as central to the current moment.

**Decision: white-box reliability audit of small open-weight judges.** The
differentiator the neighbors all lack: they treat judges as black boxes
(sampled verdicts, flip counting), while local open-weight judges expose the
full next-token distribution. Nobody in the scanned literature audits the
deployable small-judge class at the logit level with real statistical care —
and that design is *uniquely* suited to this environment: single-token verdict
readout = prefill-only forward pass = CPU-feasible; the paired-bootstrap and
protocol-design machinery from rag-chunking-bench transfers directly. The
constraint (no GPU, no API budget) points at the same corner the literature
left open. Rationale recorded in ROADMAP.md.

### Design decisions (day 1)

1. **Single pinned data artifact.** RewardBench filtered split (2,985 pairs,
   arXiv:2403.13787), revision `168d848cdbbe`, SHA256-pinned, per-subset
   counts pinned at load. Key discovery while inspecting it: the `llmbar-*`
   subsets (100+134+92+47+46 = 419) are *exactly* the full LLMBar benchmark
   (arXiv:2310.07641), so a separate LLMBar loader would double-count.
   One source, and the adversarial axis comes free. Also verified: the raw
   `id` column is NOT unique across subsets — items are keyed `subset/id`
   (uniqueness checked at load).
2. **Single-token verdict readout.** Prompts end by requesting exactly one
   letter; the runner reads full-vocab logits at the first assistant position
   and takes `z = logit(A) − logit(B)`. Deterministic (no sampling noise
   axis at temp 0 — the seed-variance axis of the old plan disappears by
   construction), cheap (no decode), and information-rich (probabilities,
   not flips). Unconstrained-argmax compliance and mass on {A,B} recorded so
   the readout's validity is itself audited per model.
3. **Exact swap decomposition as the analysis backbone.**
   `s_i = (z_cf − z_rf)/2` (order-invariant preference, sign = debiased
   verdict), `b_i = (z_cf + z_rf)/2` (position bias toward A). Identity, not
   model. The additive-shift hypothesis (`b_i ≈ const`) is what
   swap-averaging implicitly assumes — testing it is a phase-3 deliverable.
4. **Presentation orders are exhaustive, not randomized** — both orders for
   every item, so position bias is measured within-item, not marginalized by
   a coin flip. (Randomized single-order designs, e.g. SLMJury's, cannot
   separate bias from noise per item.)
5. **Rubrics as named templates** (`minimal` canonical, `detailed` for the
   sensitivity axis), runtime-agnostic prompt objects; chat templating lives
   in the runner.

### Built today (all tested, 25 tests green, ruff clean)

- `src/data.py` — pinned fetch (SHA256), validating loader (subset counts,
  id uniqueness, category partition, degenerate-pair checks), deterministic
  largest-remainder stratified sampling (order-independent, seeded).
- `src/prompts.py` — rubric registry, swap-pair builder, verdict-token
  contract; prompts provably never leak the gold label (tested).
- `pyproject.toml` with a separate `judge` dependency group so the analysis
  stack never requires llama-cpp-python.

### Feasibility pilot (real numbers, Qwen2.5-0.5B-Instruct Q4_K_M, 4 threads)

- llama-cpp-python 0.3.x compiled from source in 4m38s.
- Tokenizer check: "A"→32, "B"→33, single tokens with or without leading
  space. Verdict readout valid for the Qwen2.5 family.
- Full-vocab logits at the last position must be read via
  `llama_cpp.llama_get_logits(llm._ctx.ctx)` — `llm.scores` stays zeroed
  when `logits_all=False` (cost me one debugging round; the runner must use
  the low-level accessor).
- Prefill throughput: 197-tok prompt 1.0 s (207 tok/s), 366-tok 2.1 s
  (173 tok/s), 2,768-tok worst case 18.0 s (153 tok/s).
- Unconstrained argmax was the verdict letter on all probes (ranks 1–2 for
  A/B) — format compliance of the readout looks unproblematic even at 0.5B.
- **Swap-pair preview on 3 items (p25/p50/p75 by length):**
  `b_i` = +4.13, +4.77, +4.30 log-odds toward A; `|s_i|` = 0.09, 0.02, 0.37.
  Position bias exceeds the content signal by ~10x on every probe. If this
  holds at grid scale, the 0.5B judge is essentially an always-A machine
  whose raw accuracy is position-assignment noise — and symmetrization will
  look like a huge rescue. n=3, no CIs: a pilot observation, not a finding.

### Grid sizing arithmetic (from measured throughput)

Mean judge prompt ≈ 500 tok → ≈ 3 s/judgment at 0.5B. Stratified n=600 x 2
orders = 1,200 judgments ≈ 1 h at 0.5B; ≈ 3 h at 1.5B; ≈ 6 h at 3B (one
session-day each, fits the daily cadence). 7B–8B will need either a smaller
stratified sample (n≈300, composition-preserving by design) or a two-day run;
decide when the small-model results fix the effect sizes needed.

### Next steps (Day 2)

1. `src/judge.py`: llama.cpp runner — chat-template registry per model
   family (Qwen ChatML verified today; Llama-3 template next), low-level
   logit readout, per-judgment record (item_id, order, rubric, z, mass on
   {A,B}, argmax token, compliance flag, timing), and a resumable JSONL
   result store keyed (model, rubric, order, item_id) in the
   rag-chunking-bench raw-results style. Tests with a tiny fake runner; one
   smoke test gated on the model file being present.
2. `experiments/run_grid.py`: config-driven grid over (model, rubric,
   sample, orders), append-only, idempotent resume.
3. Launch the first real grid: Qwen2.5-0.5B, minimal rubric, n=600 seed 0,
   both orders (~1 h). If it finishes in-session, add Llama-3.2-1B for the
   first cross-family point.
4. Defer: baselines module (always-A / longer / random floors) — trivial,
   slot it wherever a run is in flight.

## 2026-07-18 — Day 2: runner + analysis core built; first grid lands findings 1–4

### Built (47 tests green, ruff clean; all committed before results)

- `src/judge.py` — explicit chat-template registry (ChatML for Qwen2.5,
  Llama-3 header format), model registry with pinned HF revision + SHA256
  per GGUF (verified before every run), low-level logit readout
  (`llama_get_logits`; the pure arithmetic lives in `logits_to_record` so
  it is unit-testable without llama.cpp), per-judgment records with
  compliance/mass diagnostics, append-only JSONL `ResultStore` with
  idempotent-resume keys and a provenance sidecar (`.meta.json`).
- `src/analysis.py` — swap-pair assembly (rejects mixed model/rubric sets,
  counts incomplete items instead of dropping them silently), the s/b
  decomposition as properties, percentile bootstrap for means and paired
  deltas (10k resamples, seeded — same machinery as rag-chunking-bench).
- `src/baselines.py` — always-A, longer-response (chars + words),
  random floors, per item/order so they enter the same bootstrap.
- `experiments/run_grid.py` (context sized from the actual sample; refuses
  to truncate), `experiments/summarize.py` (per-store JSON + markdown
  quick-look, per-category blocks), `experiments/make_figures.py`
  (decomposition scatter + accuracy-vs-floors chart, lab figure style).
- Engineering note: `vocab_only=True` cannot be used for the tokenizer
  sizing pass — llama-cpp-python 0.3.34 fails to create a context without
  weights. Sizing uses a throwaway small-ctx full load instead (mmap makes
  the second load cheap).

### Experiment: qwen2.5-0.5b, minimal rubric, n=600 (seed 0), both orders

1,196 new judgments in 56.5 min (4 threads, 0.35 judg/s; 4 from the smoke
run). Summary: `results/summary/qwen2.5-0.5b__minimal.json`; figures:
`results/figures/qwen2.5-0.5b__minimal_{decomposition,accuracy}.png`.

**Finding 1 — the readout is valid at 0.5B.** Unconstrained-argmax
compliance 1.000 across all 1,200 judgments; median min-mass on {A, B}
≈ 1.00. The single-token verdict contract holds for Qwen2.5-0.5B, so z is
measuring the verdict, not an artifact. (Validity is per-family: the
Llama-3.2-1B run in flight is showing partial compliance — see next steps.)

**Finding 2 — the 0.5B judge is functionally an always-A machine.**
b_i > 0 on 99.8% of items; mean b = +3.68 log-odds (sd 1.08, IQR
[2.96, 4.31]). Per-order accuracy: 1.000 chosen-first, 0.002
rejected-first. A deployment that assigns presentation order at random
gets 0.501 [0.500, 0.502] — indistinguishable from a coin flip.

**Finding 3 — black-box flip counting cannot see this failure mode.**
Positional flip rate under order swap: 0.002 (1 item in 600). A flip-rate
audit would score this judge as near-perfectly *consistent* — precisely
because the bias is strong enough to saturate both orders. White-box, the
"consistency" decomposes into bias ~15x the content signal: median |b| =
3.65 vs median |s| = 0.24; |b| > |s| on 99.8% of items. This is the
sharpest version of the project's thesis so far: reliability-looking
behavior that is pure position bias, measurable only at the logit level.

**Finding 4 — symmetrization rescues a real but weak signal; length floor
is below chance here.** Swap-averaged accuracy 0.568 [0.528, 0.608];
paired gain over randomized-order raw +0.068 [+0.027, +0.107]. The
longer-response floor on this sample is 0.425 (below chance — RewardBench's
adversarial subsets punish verbosity-picking), so the debiased 0.5B judge
clears random, always-A, and length floors. Per category: Safety 0.608 >
Chat Hard 0.565 ≈ Reasoning 0.566 > Chat 0.500 — on easy chat pairs the
debiased 0.5B has *no* signal at all (median |s| there 0.24 vs 0.55 on
Safety). Category CIs are a phase-3 job (per-category n is small).

### Experiment: llama-3.2-1b, minimal rubric, same 600 items, both orders

Second grid of the day (2h 22m at 0.14–0.27 judg/s under partial CPU
contention with the analysis work). Cross-family contrast on identical
items, orders, and rubric.

**Finding 5 — verdict-format compliance is a per-family property, and the
readout diagnostics are load-bearing.** Qwen2.5-0.5B: argmax compliance
1.000, mass on {A, B} ≈ 1.0. Llama-3.2-1B: only 51.2% of items are
argmax-compliant in both orders; the unconstrained argmax is a verdict
letter in 56% of judgments ("Response" 387x, "The" 53x, "I" 16x
otherwise), and per-judgment mass on {A, B} has quartiles
[0.10, 0.67, 0.94]. At 1B the single-token z measures a renormalized
sub-distribution preference for half the items, so every Llama-1B number
below carries that qualification. A compliance-conditioned sensitivity
view is now a required phase-3 deliverable, not an optional one.

**Finding 6 — bias direction, magnitude, and the flip-rate ranking all
invert across families.** Llama-3.2-1B leans toward position B: median
b = −0.34 (mean −0.09, sd 1.05), b > 0 on only 27.5% of items; per-order
accuracy 0.312 chosen-first / 0.728 rejected-first. Its bias magnitude is
~10x smaller than Qwen-0.5B's, yet |b| > |s| still holds on 81.7% of items
(the content signal is smaller too: median |s| 0.14 vs Qwen's 0.24). The
black-box view inverts the true ordering: Llama's flip rate is 0.183 vs
Qwen's 0.002, so a flip-count audit ranks Llama as far *less* consistent —
while white-box it is ~10x *less* positionally biased (median |b| 0.34 vs
3.65). Flip rate measures bias saturation, not bias. Also category-
dependent in direction: Reasoning items pull b positive (+0.25 mean, with
a long right tail visible in the decomposition scatter) while Chat/Chat
Hard/Safety sit negative (−0.28/−0.39/−0.45) — the additive-shift
hypothesis is already looking dead at 1B before the formal phase-3 test.

**Finding 7 — after debiasing, the two judges are statistically
indistinguishable overall but differ sharply by category.** Symmetrized
accuracy 0.555 [0.517, 0.595] vs Qwen's 0.568 [0.528, 0.608] (overlapping
CIs); Llama's symmetrization gain is +0.035 [−0.001, +0.072] — not
significant, consistent with its small bias (less to rescue). By category:
Llama-1B is *much* better on easy Chat (0.653 vs 0.500 — Qwen had zero
signal there) but *below chance* on adversarial Chat Hard (0.435 vs
0.565). Chat is the one category where the length floor is high (0.792) —
whether Llama's Chat advantage is just length-following is exactly the
phase-3 value-over-length regression's question.

### Next steps (Day 3)

1. Extend the grid: qwen2.5-1.5b (~3 h) — download pinned GGUF, register,
   run early in the session. Then 3B the day after (~6 h), and decide the
   7B sample size (n=300 composition-preserving vs. two-day n=600) once
   the 1.5B effect sizes are in.
2. While the 1.5B grid runs: build the compliance-conditioned view of the
   Llama-1B results (finding 5) — sym acc and bias stats on the compliant
   subset vs. all items, plus a mass_ab-stratified breakdown. Decide
   whether constrained-readout validity needs its own figure.
3. Start the scaling-curve figure (sym acc + median |b| vs. params, one
   line per family) once ≥3 models exist.
4. Phase-3 backlog (not yet): formal additive-shift test (finding 6 already
   suggests rejection at 1B — category-dependent bias direction),
   calibration/ECE, value-over-length regression (finding 7 makes Chat the
   key category), detailed-rubric axis.

## 2026-07-19 — Day 3: readout validity survives its own audit (finding 8); 1.5B grid

### Built (51 tests green, ruff clean)

- `src/analysis.py`: `two_sample_bootstrap_delta_ci` (unpaired, for disjoint
  strata), `compliance_view` — accuracy/decomposition stats stratified by
  argmax compliance, a validity curve over `mass_min` bins (edges placed at
  the Llama-1B quartiles observed on day 2), and per-category compliance
  composition so stratum differences can be read against category mix.
- `experiments/compliance_view.py` — per-store JSON + two-panel figure.
- `experiments/scaling_curve.py` — cross-model figure (sym + raw accuracy vs.
  params with CIs and floors; median |b| vs median |s|), recomputed from raw
  stores, with a hard guard that refuses to plot stores covering different
  item sets (it correctly caught the in-flight 1.5B store today;
  `--models` selects completed stores explicitly).
- Registered `qwen2.5-1.5b` (pinned revision + SHA256, verified after
  download and before the run, ChatML template as for 0.5B).

### Finding 8 — the logit readout survives its own validity check at 1B.

Finding 5's threat was that half the Llama-1B judgments measure a
renormalized sub-distribution (argmax not a verdict letter; mass on {A, B}
quartiles [0.10, 0.67, 0.94]). Conditioning everything on compliance shows
the threat does not materialize in accuracy terms:

- Sym acc: all 0.555 [0.517, 0.595]; compliant-both (n=307) 0.534
  [0.479, 0.590]; non-compliant (n=293) 0.577 [0.519, 0.635]. Stratum gap
  −0.043 [−0.122, +0.038] (unpaired bootstrap) — null, point estimate even
  favors the non-compliant half.
- Validity curve over mass_min bins is flat: <0.25 mass (n=212) 0.561
  [0.495, 0.627] vs ≥0.9 (n=150) 0.547 [0.467, 0.627]; all five bins'
  CIs overlap heavily.
- Compliance is category-structured, hard: Reasoning 22.6%, Chat 62.5%,
  Chat Hard 79.3%, Safety 83.8%. So the naive stratum comparison is
  composition-confounded (which is exactly why the per-category block is in
  the view), and — the practical point — a black-box harness that drops
  unparseable verdicts would discard ~3/4 of Reasoning while keeping most
  of Safety: it reweights the benchmark rather than sampling it. The
  white-box readout keeps all items at no measurable validity cost.
- Within-category compliant-vs-not point estimates (small n, descriptive):
  Chat Hard compliant 0.397 vs non-compliant 0.579 — the below-chance
  Chat-Hard result from finding 7 is *concentrated in the compliant
  stratum*; whatever makes Llama-1B confidently format-follow on Chat Hard
  co-occurs with being adversarially fooled. Logged as a thread to pull in
  the phase-3 error analysis, not claimed as a finding at this n.

README gained the "Does the audit survive its own validity check?" section
with the stratum table and the compliance figure.

### Experiment: qwen2.5-1.5b, minimal rubric, same 600 items, both orders

1,200 judgments in 116.5 min (0.17 judg/s, 4 threads; pinned GGUF revision
91cad511, SHA256 verified after download and at load). Readout fully valid:
argmax compliance 1.000, median mass on {A, B} ≈ 1.00 — Qwen family format
discipline confirmed at a second size, so everything below is behavior.

**Finding 9 — debiased judge quality scales BACKWARDS within the Qwen
family.** Every scalar a black-box audit tracks improves 0.5B → 1.5B:
median |b| 3.65 → 1.09, median |s| 0.24 → 0.50, raw random-order acc
0.501 → 0.549 [0.527, 0.571]. Yet symmetrized accuracy falls to chance:
0.502 [0.462, 0.542], significantly below 0.5B on the same items (paired
cross-model Δ +0.067 [+0.013, +0.118]). Sharper: symmetrization now HURTS —
Δ sym−raw = −0.048 [−0.081, −0.013], the first negative debiasing gain in
the audit. Mechanism located: on the 421 no-flip items the debiased sign is
below chance (0.432 [0.387, 0.480]) while on the 179 flipped items it is
informative (0.665 [0.598, 0.732]). Flipped items contribute identically to
raw-mean and sym accuracy (both orders correct or both wrong), so the whole
raw-vs-sym inversion lives in the no-flip stratum: where bias saturates the
verdict, the residual order-invariant preference points the wrong way.

**Finding 10 — the wrong-way preference is a Reasoning phenomenon that
tracks length.** Reasoning (n=288): sym 0.368 [0.312, 0.424] vs raw 0.510;
per-category paired Δ sym−raw = −0.142 [−0.194, −0.090] — all of the
overall backfire and then some (other categories: Chat +0.076, Chat Hard
+0.022, Safety +0.034, all CIs spanning 0, sym 0.52–0.67). Reasoning sym
0.368 ≈ the Reasoning longer-response floor 0.370. Epicenter math-prm
(n=90): sym 0.167, longer floor 0.078 (the rejected solution is longer on
~92% of pairs), judge preference sign matches length sign on 75.6%.
Subset spread inside Reasoning: math-prm 0.167, hep-java 0.273, hep-go
0.364 ... hep-cpp 0.606. Cross-model: overall sign(s)==sign(len_chosen −
len_rejected) agreement is 0.491 (0.5B), 0.571 (1.5B), 0.622 (Llama-1B) —
the 0.5B judge's weak signal was length-free; the signal that EMERGES with
scale is substantially a verbosity preference, and RewardBench Reasoning
punishes it (chosen answers are the concise correct ones). Hedge recorded:
length is a strong correlate, not a proven mechanism — model-generated
wrong solutions differ from concise references in style too; the phase-3
value-over-length regression (now elevated) separates length from style
covariates. Note Llama-1B follows length MORE overall (0.622) yet holds
Reasoning sym at 0.556 — the length-following/accuracy interaction is
category- and family-specific, another regression covariate.

**Finding 11 — bias direction is category-dependent WITHIN one family.**
Qwen2.5-1.5B mean b: Chat +1.09, Reasoning +1.29, Chat Hard +0.19, Safety
−0.61. "This model is A-biased" is not well-defined even per model. With
three models the flip-rate ranking (0.002 / 0.183 / 0.298) tracks neither
median |b| (3.65 / 0.49 / 1.09) nor sym accuracy (0.568 / 0.555 / 0.502) —
flip rate is uninterpretable as a reliability metric without the
decomposition.

**Correction (README fixed today):** day 2 quoted "median |b| 0.34 vs 3.65"
for Llama-1B vs Qwen-0.5B; 0.34 is |median b|. Median |b| is 0.49 — the
bias-magnitude ratio is ~7x, not ~10x. Day-2 log entry left as written;
README now carries the correct number.

### Artifacts

- `results/raw/qwen2.5-1.5b__minimal.jsonl` (+ meta), summary JSON,
  decomposition/accuracy/compliance figures, and the three-model
  `scaling__minimal.png` (sym vs raw accuracy crossing on the Qwen line;
  |b| collapse vs |s| growth on the right panel).
- README: new "Scaling within a family" section (findings 9–11 with the
  scaling curve and 1.5B decomposition embedded), compliance section from
  the morning, status/counts refreshed.

### Next steps (Day 4)

1. Qwen2.5-3B grid (~6 h at 0.17→~0.08 judg/s; start FIRST thing, it is
   the whole session's compute). Download pinned GGUF, register (ChatML),
   verify SHA256, run n=600 seed 0 both orders. Expect ~5-7 h; checkpoint
   the store mid-run as today.
2. While it runs: the value-over-length regression is now the most
   important analysis in the project (finding 10). Build
   src/length_probe.py or extend analysis.py: per-item logistic regression
   of gold on standardized judge s vs standardized length delta (chars and
   tokens), overall + per category, per model; report coefficient CIs via
   bootstrap. Key questions: does s add signal beyond length anywhere at
   1.5B? Does Llama-1B's Chat advantage (finding 7) survive length control?
3. If time remains: begin the additive-shift formal test (variance
   decomposition of b_i; category and length-delta covariates) — findings
   6/11 predict rejection.
4. Decide the 7B sample budget after the 3B numbers land (n=300
   composition-preserving vs two-day n=600).

## 2026-07-22 — Day 4: value-over-length probe (findings 12–14); 3B grid

(Gap 07-20/07-21: no sessions ran.)

### Built (61 tests green, ruff clean)

- `src/length_probe.py` — the analysis finding 10 elevated: a Bradley–Terry /
  conditional-logit probe on oriented chosen−rejected differences.
  P(gold-chosen wins) = sigmoid(β·x) with x = (judge preference s, log length
  ratio); **no intercept** — under orientation symmetry (relabeling
  chosen/rejected flips every feature sign) a constant is not identified; with
  the outcome constant by construction its MLE diverges, and in the
  antisymmetric doubled-data view it is exactly zero. Features SD-scaled but
  NOT centered (origin "equal lengths, indifferent judge" must map to
  P = 1/2). Nested specs: length-only / judge-only / joint / joint-sign
  (sign(s) = the symmetrized binary verdict). Weak ridge (1e-3) keeps
  bootstrap replicates with complete separation finite (one-signed small
  strata); damped Newton, and a batched-across-replicates Newton so the
  10k-resample bootstrap (full rescale+refit pipeline inside every replicate,
  shared resamples across specs so spec deltas are paired) runs in seconds.
- `experiments/length_probe.py` — runner over completed stores (identical
  item-set guard as scaling_curve), JSON + two-panel forest figure (β_s in
  joint spec; Δacc joint − length-only).
- Registered `qwen2.5-3b` (revision 7dabda4d, SHA256 verified against HF's
  LFS oid after download and at load).

### Findings (probe over the three completed grids, same 600 items)

**Finding 12 — every judge carries real signal beyond length, including the
one that judges at chance; at 1.5B the binary verdict is what destroys it.**
Joint-spec β_s overall: 0.5B +0.545 [+0.369, +0.739], 1.5B +0.380
[+0.201, +0.572], Llama-1B +0.319 [+0.138, +0.546] — all significantly
positive, including Qwen2.5-1.5B whose sym accuracy is 0.502. Resolution:
thresholding. At 1.5B the continuous s has length-controlled signal but its
*sign* has none (joint-sign β +0.040 [−0.124, +0.204]), while at 0.5B/1B the
sign retains it (+0.290 / +0.282, both significant). Probability-averaging
and majority-voting are measurably different judges at 1.5B — a white-box-only
distinction.

**Finding 13 — length mediates both standing mysteries.** (a) The 1.5B
Reasoning collapse is entirely length-mediated: judge-only β_s −0.329
[−0.629, −0.079] (the preference anti-predicts gold), joint β_s −0.084
[−0.406, +0.183] (nothing left after length control — and no residual
anti-signal either). (b) Llama-1B's Chat advantage (finding 7) is
length-following: Chat joint β_s −0.046 [−0.812, +1.518]; Chat is the one
category where longer is actually better and the length-only model scores
0.792 > Llama's 0.653. Counterpoint: Qwen-1.5B's emergent Chat signal is
genuine content (β_s +0.805 [+0.181, +1.811]). Scale bought real Chat
judgment and a toxic Reasoning verbosity preference simultaneously.

**Finding 14 — against a deployable floor, these judges only pay on
Safety.** The one-parameter fitted length model learns shorter-is-better on
this sample and reaches 0.575 overall — above all three judges' sym
accuracy. Δacc(joint − length) ≈ 0 overall for all models; 1.5B judge-only is
significantly worse than length-only (−0.073 [−0.131, −0.009]). Safety is
the exception: length carries nothing there (length-only 0.412, β_len ≈ 0),
every judge has β_s +0.6–0.9, and at 1.5B joint beats length by +0.284
[+0.020, +0.338] (same-signed point estimates at 0.5B/1B). Caveats recorded
in README: in-sample accuracies (≤2 params, negligible optimism, deltas
share it); the length model's direction is fitted to this benchmark —
the claim is "not distinguishable from a peeked one-parameter baseline",
not "use length heuristics".

Methodological note for the writeup: the probe formalizes "does the judge
add value" as a coefficient question instead of an accuracy-comparison
question — accuracy deltas at n=600 are too coarse (CIs ±0.05) while the
coefficient CIs cleanly separate zero from non-zero signal. This is the
paired-power argument from rag-chunking-bench again, in regression form.

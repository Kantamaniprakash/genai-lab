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

### Calibration axis (phase-3 item 3, pulled forward — finding 15)

Built `src/calibration.py` + `experiments/calibration.py` while the 3B grid
ran: folded (confidence, correctness) views — raw `sigmoid(|z|)` per
judgment, sym `sigmoid(|s|)` per item — with **tie-safe equal-mass bins**
(saturated judges pile float-identical confidence at 1.0; splitting a tied
run across bins with different accuracies manufactures ECE, caught by a test
before it shipped) and item-level bootstrap CIs. 70 tests green.

**Finding 15 — symmetrization is also a calibration repair, except where the
preference itself is broken.** Raw is overconfident everywhere; at 0.5B the
miscalibration IS the position bias read as certainty (mean conf 0.956, acc
0.501, ECE 0.455). Symmetrized: 0.5B ECE 0.035 (gap +0.024), Llama-1B 0.052
(gap +0.005) — near-diagonal reliability curves; sigmoid(|s|) is readable as
a probability at these scales. Qwen-1.5B is the exception: still
overconfident after debiasing (ECE 0.166, gap +0.162), reliability curve
flat at ~0.45 acc across conf 0.5–0.85, rising only in the top-confidence
bin (0.94 → 0.75) — finding 12's magnitude-vs-sign mechanism drawn as a
curve. Deployment reading: a confidence-thresholded 1.5B judge would be
usable; a confidence-trusting one is worse than its 0.5B sibling.
Methodological caveat recorded: ECE is a nonnegative deviation statistic, so
its bootstrap CI sits above the point estimate for near-calibrated judges
(0.5B sym CI [0.036, 0.094] vs point 0.035); the signed gap is the companion
number free of that bias.

### Experiment: qwen2.5-3b, minimal rubric, same 600 items, both orders

1,200 judgments in 83 min (0.24 judg/s once the analysis work stopped
competing for cores — faster than the 6 h estimate; the day-1 arithmetic
over-extrapolated from the contended 1.5B rate). GGUF pinned at revision
7dabda4d, SHA256 verified against HF's LFS oid at download and at load.
Readout fully valid at a third Qwen size: compliance 1.000, mass ≈ 1.0.
All cross-model analyses (scaling curve, probe, calibration) rerun over
four stores; probe forest + calibration figures regenerated.

**Finding 16 — the inverse scaling is a valley, not a trend.** Sym acc
0.742 [0.707, 0.777]; paired deltas +0.173 [+0.123, +0.223] over 0.5B,
+0.240 [+0.192, +0.290] over 1.5B, +0.187 [+0.138, +0.235] over Llama-1B.
Non-monotone within one family and protocol: 0.568 → 0.502 → 0.742. Any
two-point scaling extrapolation here predicts the wrong third point. Per
category: Chat 0.861, Reasoning 0.771, Safety 0.730, Chat Hard 0.576 (the
LLMBar-adversarial category is finally the hardest, as designed).

**Finding 17 — the verbosity preference was a mid-scale transient; the
position bias that replaces it is the largest yet, in the opposite
direction.** Sign(s)-vs-length agreement (tie-excluded, finding-10
convention — note: the probe JSON's sign_agree_* fields count ties as
disagreement, hence lower values; both conventions verified today) falls
0.571 → 0.547 overall, 0.628 → 0.538 Reasoning, 0.756 → 0.433 math-prm —
anti-length exactly where verbosity was fatal — and math-prm sym recovers
0.167 → 0.600. Position bias flips direction within the family: median b
−5.55 toward B (b > 0 on 19.2%), |b| median 6.21 > the 0.5B's 3.65; still
direction-heterogeneous across categories inside the model (Chat +0.90,
Reasoning −6.59). Flip-rate ranking across four models
(0.002/0.183/0.298/0.380) tracks neither bias nor accuracy.

**Finding 18 — first judge to beat the length floor; confidence still not
trustworthy.** Probe: overall β_s +1.399 [+1.147, +1.714], joint −
length-only acc +0.205 [+0.156, +0.261] (Reasoning +0.231, Safety +0.324,
both CIs excluding 0); joint-sign β +1.021 — majority voting is fine at 3B
where it was fatal at 1.5B. Calibration: sym conf 0.894 vs acc 0.742, ECE
0.153 [0.126, 0.190] — rises with confidence (unlike 1.5B's flat curve)
but overconfident throughout. Pattern across four judges: symmetrized
verdicts are calibrated exactly where the judges are weakest.

### Next steps (Day 5)

1. **Llama-3.2-3B grid** — the cross-family point at the reversal scale:
   does the valley-then-reversal shape replicate outside Qwen, and does
   Llama's B-lean grow the way Qwen's flipped? Download pinned GGUF
   (bartowski, Q4_K_M), register, run n=600 seed 0 both orders (~1.5–2 h
   at the uncontended 3B rate). Start it FIRST.
2. While it runs: the **additive-shift formal test** (phase-3 item 2) is
   now the most interesting pending analysis, and the 3B data sharpens it:
   with |b| ~6 log-odds and category-dependent direction, how much of
   Var(b) do category/subset/length covariates explain, and — the
   deployment question — can a fitted per-category bias correction recover
   symmetrization's accuracy from a SINGLE order (half the compute)?
   Cross-fitted (k-fold over items) so the correction is honest.
3. Then the 7B decision: at the uncontended rate, Qwen2.5-7B n=600 both
   orders ≈ 2.5–3.5 h — likely feasible in one session; keep n=600 unless
   the run-rate says otherwise. Llama-3.1-8B as the family counterpart.
4. Writeup debt: the abstract still describes the audit as 3-model; the
   day the grid completes at 7B, restructure the README results narrative
   around the scaling arc (valley → reversal) rather than grid-arrival
   order.

## 2026-07-23 — Day 5: additive-shift test + correction ladder (findings 19–20); Llama-3.2-3B grid

### Built (83 tests green, ruff clean; committed before the grid finished)

- Registered `llama-3.2-3b` (bartowski Q4_K_M, revision 5ab33fa9, SHA256
  verified against HF's LFS oid after download and at load); grid launched
  first thing and ran in the background all session.
- `src/bias_model.py` — phase-3 item 2 made concrete. Two connected pieces:
  - **Variance decomposition of b.** Key methodological observation: the
    readout is deterministic at temp 0, so b_i carries *no sampling noise* —
    all of Var(b) is real item-level bias structure, and R² of nested
    predictors (category means / subset means / subset + symmetric length
    covariates) cleanly partitions it into the exploitable share and an
    irreducible residual. Bootstrap refits inside every replicate (batched
    group-mean scatter-adds and batched ridge-OLS normal equations).
    Covariates must be *symmetric* in the pair (log total chars, |dlog
    chars|, log prompt chars): b is order-invariant, so any sign-flipping
    covariate is structurally excluded from linearly predicting it.
  - **Single-order correction ladder.** The oracle correction
    sign(z − b_i) IS the symmetrized verdict (z_cf − b_i = s_i,
    z_rf − b_i = −s_i), so fitted corrections — none / global / category /
    subset / regression / oracle — interpolate raw → symmetrized at half
    the inference cost. All corrections are exact leave-one-out (closed-form
    LOO group means with singleton fallback to the LOO grand mean; the
    ridge hat-matrix identity for the regression), so no item is corrected
    with its own bias. Ladder accuracies bootstrap the fixed per-item LOO
    scores; correction-refit variance is not resampled (negligible at
    group-mean scale on n=600 — caveat recorded in the module docstring).
- `experiments/bias_model.py` — runner over completed stores (identical
  item-set guard), JSON + two-panel figure (R² forest; accuracy ladder).
- Engineering note: numpy stacked `linalg.solve` needs an explicit RHS
  matrix dimension — `solve(A, c[:, :, None])[:, :, 0]` — or it
  misinterprets a (B, p) RHS stack.

### Findings (over the four completed grids, same 600 items)

**Finding 19 — the additive-shift hypothesis is rejected at every scale,
and bias predictability is anti-correlated with bias magnitude.** Subset
structure alone explains a significant share of Var(b) everywhere (R²
0.141 [0.121, 0.228] at 0.5B, 0.329 Llama-1B, 0.448 at 1.5B, 0.260 at 3B).
But the *character* changes with scale: the 0.5B always-A machine is the
closest thing to a true additive shift (category R² 0.020; category means
all +3.40..+3.91); 1.5B has the most predictable bias in the audit
(subset+length R² 0.556 — over half of Var(b), matching finding 11's
category-signed biases); 3B has the largest bias (SD 5.47, category means
+0.90 Chat vs −6.59 Reasoning) yet the least explainable — residual SD
4.61 log-odds after all covariates, *larger than its own median |s| 3.64*.
Length covariates add +0.20 R² at 0.5B and +0.11 at 1.5B but ~nothing at
Llama-1B (+0.001) and 3B (+0.028) — what makes an item bias-prone is
family- and scale-specific, not a benchmark property.

**Finding 20 — a fitted one-call correction fully substitutes for
symmetrization at 0.5B, caps at half the gain at 3B, and actively hurts at
1.5B.** 0.5B: LOO subset correction 0.547 [0.526, 0.567] — 68% of the
symmetrization gain, statistically indistinguishable from the two-call
oracle (Δ −0.022 [−0.056, +0.013]). 3B: best fitted rung is *category*
(0.675 [0.647, 0.703]; subset/regression overfit finer strata at 0.662),
+0.058 over raw with Reasoning alone jumping 0.575 → 0.688 when its −6.59
shift is subtracted — but it recovers only 47% and stays significantly
below the oracle (Δ −0.067 [−0.091, −0.042]): finding 19's unexplained
4.61-log-odds residual is exactly what the second call buys. 1.5B: the
ladder runs backwards (none 0.549 > global 0.532 > category 0.517 >
subset 0.510 > oracle 0.502) — corrections work as designed (subset
recovers 82% of the oracle "gain") but the debiased preference they
converge to is anti-informative on Reasoning (finding 9), so every step
toward it hurts. Deployment reading: the models where one cheap call +
a dev-set constant suffices are the weak ones; where the judge is worth
deploying (3B), the bias is too idiosyncratic to subtract and the second
call earns its cost. A bias correction inherits whatever the bias was
masking.

README: new "Is position bias a constant you can subtract?" section with
the cross-model table, findings 19–20, and the two-panel figure; method
section updated (the additive-shift hypothesis is now tested, not deferred);
planned-experiments item 2 marked done.

### Experiment: llama-3.2-3b, minimal rubric, same 600 items, both orders

1,200 judgments in 227 min (0.09 judg/s — this session's container prefills
~3x slower than yesterday's at the same thread count and quant; measured
~40 tok/s vs ~120. Same Xeon class, 4 vCPUs — host variance is real and the
day-1 throughput arithmetic should be treated as per-host). GGUF pinned at
revision 5ab33fa9, SHA256 verified against HF's LFS oid at download and at
load. Mid-run correction logged: the "compliance 1.000" in the first
checkpoint message was a 23-record peek; store-level argmax compliance is
0.863 pair-level / 0.974 per-judgment. All cross-model analyses rerun over
five stores; scaling curve, probe forest, calibration, and bias-model
figures regenerated (length_probe MODEL_STYLES gained the fifth entry —
the two Llama sizes now carry light/dark reds mirroring the Qwen gradient).

**Finding 21 — both families reverse bias direction with scale, in opposite
senses; Llama-3.2-3B is a new always-A machine.** 1B → 3B flips Llama's mild
B-lean (median b −0.34, 27.5% positive) to saturated A: b > 0 on 99.8%,
median +2.34, per-order acc 0.990/0.023, flip rate 0.033 — second-"most
consistent" in a black-box audit while being the audit's second-most
saturated bias (finding 3's failure mode, 6x the scale). Uniquely among the
five, bias direction is same-signed across all categories (+1.88..+2.81).
Sym 0.652 [0.613, 0.690]: +0.097 [+0.047, +0.147] paired over 1B (no valley
between the two measured Llama points; no ~2B Llama exists to test the
counterpart of Qwen's 1.5B dip — recorded as a family-geometry limitation).
Qwen-3B leads at matched scale: +0.090 [+0.048, +0.132]. Symmetrization
rescue +0.145 [+0.108, +0.183] — largest in the audit. Correction ladder:
global 0.576 (48%), regression 0.583 (52%), all rungs significantly below
oracle (best Δ −0.069) — bias is compact (SD 1.01) but median |s| 0.44 is
smaller, so ~half the items stay bias-dominated after any one-call fix.
Finding 20's "one call buys about half at 3B" now holds in both families.

**Finding 22 — Llama scale buys Chat, deepens the adversarial hole.** Chat
0.653 → 0.889 (paired +0.236 [+0.111, +0.361]), joint β_s +4.72
[+3.32, +9.75] — the audit's largest length-controlled content coefficient
(1B's Chat edge was length-following; 3B's is real). Chat Hard 0.435 →
0.348 [0.250, 0.446] — below chance, below its own smaller sibling (paired
−0.087 [−0.207, +0.022]), and no length-controlled signal survives (β_s
+0.28 [−0.33, +0.91]). Qwen-3B holds 0.576 on the identical items: +0.228
[+0.120, +0.337] family gap. Adversarial (LLMBar) robustness at deployable
scale is a family property, not a scale property. Overall: second judge to
beat the fitted length floor (β_s +1.043, joint − length +0.125
[+0.075, +0.181]), anti-length lean β_len −0.669.

**Finding 23 — post-debiasing calibration is a family property; the
format-breaking category migrates with scale.** Llama-3B sym ECE 0.044,
signed gap −0.012 (slightly underconfident) at 0.652 acc — kills the
"calibrated exactly where weakest" reading from findings 15/18. Five judges:
Llama 1B/3B and Qwen 0.5B calibrated after symmetrization; Qwen 1.5B/3B
overconfident (0.166/0.153). Compliance relocates: pair-level 0.512 → 0.863,
but Safety is 0.48 compliant at 3B (vs 0.986–1.0 elsewhere) where 1B's
weak category was Reasoning (23%). Readout survives again, same direction
as finding 8: non-compliant items judged BETTER (0.829 vs 0.624, gap
+0.206 [+0.113, +0.296], Safety-concentrated). A parse-and-drop harness
at 3B silently discards half of Safety.

### Next steps (Day 6)

1. **The 7B tier: Qwen2.5-7B-Instruct** (Q4_K_M ≈ 4.7 GB — fits disk and
   RAM). Rate risk: on a slow host like today's, n=600 × both orders could
   take 8–10 h. Start the grid FIRST thing regardless — the store is
   resumable, so a two-session run is safe; checkpoint-commit mid-run as
   today. Measure the early rate: if ≥ 0.15 judg/s (fast host), it
   completes in-session. Keep n=600 — comparability with the five existing
   grids is worth more than a same-day finish. Llama-3.1-8B is the family
   counterpart afterwards.
2. While it runs (light CPU only — today's contention halved the grid
   rate): writeup debt. The README results narrative is grid-arrival
   order; restructure around the scaling arc (family × scale 2×2, the
   valley, the direction chiasm, the correction-ceiling story), and add
   the limitations section entries accumulated in these notes (per-host
   throughput variance, no ~2B Llama, in-sample probe accuracies,
   ladder refit variance not resampled).
3. Analysis backlog, in priority order once 7B lands: rubric-sensitivity
   axis (detailed vs minimal, paired in log-odds — the last untouched
   phase-3 item; needs a second grid per model, so budget it after the
   scaling grid closes), per-subset heterogeneity with CIs, and the
   compliant-stratum Chat-Hard thread from day 3 (now echoed at 3B by
   finding 23's Safety migration).

## 2026-08-01 — Day 6: 7B grid launched; per-subset view (findings 24–25)

(Gap 07-24..07-31: no sessions ran.)

### Grid: qwen2.5-7b registered and running

Registered `qwen2.5-7b` (bartowski Q4_K_M, revision 8911e8a4, SHA256 verified
against HF's LFS oid after download — 65b8fcd9…1423, 4.68 GB) and launched
n=600 seed 0 both orders first thing. This host prefills at ~54 tok/s at 7B
(0.076 judg/s early rate, ETA ~4.5 h) — a mid-speed host by the day-5
per-host-variance ledger. Early diagnostics over the first 34 judgments:
argmax compliance 1.000, median mass on {A, B} ≈ 1.000 — Qwen format
discipline holds at a fourth size; and an early-peek median |s| an order of
magnitude above the 3B's (10.0 vs 3.64 on the first items) suggests a
heavily saturated preference readout at 7B. n=34, no claims; the store is
resumable and checkpoint-committed in case the session dies before the grid
closes.

### Built while the grid runs (85 tests green, ruff clean)

- `src/analysis.py`: `subset_view` — per-subset `_stratum_stats` with
  category tag and per-subset longer-response floor; and per-category
  compliant-vs-non gap CIs in `compliance_view`, guarded to strata with
  ≥5 items on both sides (a 1-item stratum bootstraps to a zero-variance,
  purely artifactual interval — the Llama-3B Chat "significant" gap that
  motivated the guard rested on exactly one non-compliant item).
- `experiments/subset_view.py` — cross-model forest (23 subsets × 5 judges,
  category-grouped, each subset's length floor as a tick), identical
  item-set guard as the other cross-model scripts.
- README: accumulated limitations section (single benchmark/sample, Q4_K_M
  confound, family geometry, in-sample probe accuracies + ladder refit
  variance, per-host throughput, one rubric); per-subset section.

### Findings (over the five completed grids)

**Finding 24 — subset accuracy ordering is the judge's local length-lean
read through the subset's gold-length composition; the audit's weakest
judge is its best formal-math judge.** math-prm (n=90, gold shorter on
~92%): Qwen 0.5B/1.5B/3B finish 0.844/0.167/0.600, and the ranking exactly
tracks the *local* sign(s)-vs-length agreement 0.233/0.756/0.433 — the
0.5B's best-in-audit math-prm score (CI [0.77, 0.91] excludes every other
judge's point estimate) is an anti-verbosity lean pointed where this subset
rewards it, not math skill. Same judge is the only one above chance on
llmbar-adver-GPTInst (0.684 vs ≤0.421 all larger judges); on
llmbar-adver-neighbor everyone is at/below chance and Llama-3B hits 0.148,
the audit's lowest subset accuracy. Category- and benchmark-level averages
predict nearly nothing about a specific domain at these scales.

**Finding 25 — the compliant-stratum penalty is real but category-localized
and family × scale-dependent.** With honest unpaired CIs the day-3 Chat-Hard
observation is null (−0.182 [−0.433, +0.070], n=73/19). The audit's one
significant stratum gap: Llama-3B Safety, compliant − non = −0.223
[−0.361, −0.085] (n=71/77) — the model judges worse exactly where it
format-complies; parse-and-drop keeps the worse-judged half.

### Next (same day, after the grid)

7B analyses + README restructure around the scaling arc; full day-6 entry
continues below.

## 2026-08-21 — Day 7: the cross-judge table, and what a partial grid is not (finding 26)

(Gap 08-02..08-20: no sessions ran. The 7B store was left at 47/1200 by the
day-6 checkpoint and is where this session picked it up.)

Fresh container, so the day started with setup rather than science: `uv sync
--group judge` (llama-cpp-python compiles ~6 min on 4 cores) and a re-download
of the pinned 7B GGUF, SHA256 re-verified against the registry pin
(`65b8fcd9…1423`, 4.68 GB) before the first forward pass. Grid relaunched
immediately; everything below was written while it ran.

### The host is slow, and the 7B grid will not close in one session

Measured 0.04–0.05 judg/s at 7B (ETA 440 min for the remaining 1,113
judgments) — the slow end of the per-host ledger, comparable to day 5's ~40
tok/s container rather than day 6's ~54 tok/s. The store is resumable and
checkpoint-committed; this is a multi-session grid and the honest plan is to
say so rather than to shrink n and lose comparability with the five completed
grids. Session ended with the store at 134/1200 judgments — 67 complete swap
pairs, no partial item (47 inherited, 87 added); the runner's own late-session
read was 0.03 judg/s, ETA 537 min for the remainder. A second measurement taken while the day's bootstraps were running
came out at 0.026 judg/s — worth recording as the *contended* rate, since it
is the reason next session should keep analysis off the box while a grid is
up. Every cross-judge analysis below therefore runs over the five
*completed* grids.

### Built: the cross-judge headline table (`experiments/master_table.py`)

The README's results grew in grid-arrival order, so each section compares its
new judge against whichever judges existed at the time and no table ever
showed all five at once. `master_table` recomputes one row per completed grid
from raw records and writes both the JSON and the markdown the README embeds.
Two columns are new to the audit:

- **per-order accuracy** (chosen-first / rejected-first) — the fastest visual
  bias test in the audit. 1.000/0.002 at Qwen2.5-0.5B and 0.990/0.023 at
  Llama-3.2-3B are always-A machines on sight; 0.805/0.293 is a lean.
- **Δ sym−longer**, the paired delta against the fixed length heuristic, with
  a caption that says outright this is a *weak* test (the fixed floor is 0.425,
  below chance) and is not the length-baseline verdict — the fitted
  one-parameter opponent scores 0.575 and is read out of the length-probe
  summary rather than hand-copied.

Every number in the table reproduces the per-grid sections exactly, which is
the cross-check the table was worth building for on its own. Also added a
findings index (all 26, claims verbatim from this log, linked to their
sections) — a 900-line report with numbered findings and no index is the first
thing a referee complains about.

Coverage discipline in the tool: the reference item set is the widest any store
covers; a store over a strict subset is a grid in flight and is dropped with
its coverage printed; a store carrying items *outside* the reference set aborts
the comparison. A grid caught mid-item (at most the trailing item is
half-written, since `run_grid` writes an item's two orders consecutively) keeps
its complete pairs and reports the dropped one.

### Finding 26 — mid-run peeks in this harness are compositionally confounded

Writing the coverage guard exposed something worse than a missing table.
`stratified_sample` sorts its output by `item_id` and `run_grid` walks that
order, so a partial grid has finished an **alphabetical prefix of the
subsets** — not a random subsample. At 45/600 the 7B prefix is 100% Chat
(`alpacaeval-easy` 20, `-hard` 19, `-length` 6 items) against 12% Chat in the
full sample; Chat Hard, Reasoning and Safety are absent entirely.

On that prefix the floor itself inverts: pick-the-longer-response scores
**0.978**, against **0.425** on the full sample. Every judge in the audit —
including the two 3B judges that beat the fitted length model overall — loses
to the trivial length heuristic on these items.

Which makes the day-6 peek an artifact. That entry read the 7B prefix's median
`|s|` (~10 at n=34) against Qwen2.5-3B's *full-sample* 3.64 and inferred "a
heavily saturated preference readout at 7B". Recomputed on the identical 45
items:

| judge | median \|s\| full | median \|s\| @45 | sym acc full | sym acc @45 |
|---|---|---|---|---|
| qwen2.5-0.5b | 0.24 | 0.24 | 0.568 | 0.556 |
| qwen2.5-1.5b | 0.50 | 0.61 | 0.502 | 0.644 |
| qwen2.5-3b | 3.64 | 8.68 | 0.742 | 0.911 |
| qwen2.5-7b | — | 11.04 | — | 0.956 |
| llama-3.2-1b | 0.14 | 0.13 | 0.555 | 0.756 |
| llama-3.2-3b | 0.44 | 0.97 | 0.652 | 0.911 |

3B's median `|s|` on these items is 8.68, not 3.64: the gap to 7B is 1.3x, not
the ~3x the unmatched comparison implied, and about 68% of the apparent effect
was composition. Note also how unevenly the restriction moves judges — +0.169
sym at Qwen-3B, +0.259 at Llama-3B, −0.012 at Qwen-0.5B — so a prefix read is
not even a uniformly optimistic distortion, it reorders the field.

The fix went into the tooling, not into a note asking future sessions to be
careful: `master_table --restrict-to <model>` cuts every judge down to the
items the in-flight grid has finished, so rows are matched by construction, and
stamps the output with an INTERIM banner plus the measured category skew. The
full-sample length-baseline verdict is deliberately dropped from the interim
caption — on a restricted table the floor is a different number and that
sentence would be false. Live interim numbers stay in
`results/summary/master_table__minimal__interim_qwen2.5-7b.{json,md}`, never
pasted into the README, because they change at every checkpoint.

Recorded as a limitation too: a randomized execution order would make partial
grids interpretable, and is the obvious fix — but applying it now would break
resume-compatibility with the six stores already collected, so the guard
stands instead.

Rendered `experiments/prefix_skew.py` — a slope chart of every judge's
symmetrized accuracy full-sample vs restricted, floor included, read straight
out of the two committed summaries (it runs no judgments). It carries the
finding better than the table does: the lines fan rather than shift, and the
floor's dashed line crosses from bottom to top of the field. Label placement
needed a `declutter` pass — both 3B judges land on exactly 0.911 restricted, so
naive annotation stacked them into an unreadable smear; labels are separated by
a minimum gap with hairlines back to their true values, and the anchors never
move.

90 tests green, ruff clean.

### Lab housekeeping — the landing page and CI had both gone stale

Noticed while the grid ran, and worth the interruption because both were
misrepresenting the lab to anyone who visits it.

- **The root README still called `rag-chunking-bench` the current flagship**,
  five weeks after it closed, and its whole results section was that project's.
  Restructured: the current flagship is `slm-judge-audit` with its own
  at-a-glance section (the non-monotone scaling, the anti-informative flip
  rate, bias beating signal on 62.0–99.8% of items, the rejected additive-shift
  assumption, and only the two 3B judges beating the fitted length baseline) and
  the scaling figure as the hero; `rag-chunking-bench` keeps everything it had
  under "Completed flagship". Nothing was deleted.
- **`scripts/sync_latest.py` was hardcoded to `rag-chunking-bench`**, which is
  why "Latest from the lab" was frozen at 2026-07-16. It now reads the flagship
  name out of the `## Current flagship:` line in ROADMAP.md — the one place the
  answer is already maintained — and the workflow watches `*/research/NOTES.md`
  plus ROADMAP.md so a handover triggers it too. Its finding parser only knew
  the old log's numbered-list shape; it now also matches bold-paragraph and
  `###`-heading findings (both used here), dedupes a finding restated inline,
  and orders by finding number. Selftest extended to cover all three shapes.
- **CI only ever tested `rag-chunking-bench`** — the closed project — so the
  active flagship's 90 tests were not covered by the badge on the landing page.
  The matrix is now project × python-version. Verified locally first on 3.11,
  3.12 and 3.13 (90 passed, 1 skipped on each) rather than discovering it in
  the badge; `--group dev` confirmed to resolve for both projects. The judge
  runner stays out of CI: llama-cpp-python compiles from source and the one
  test that needs it is guarded and skips.

Checked but *not* changed: `rag-chunking-bench`'s README claims 365 tests while
a default install collects 349. That is correct — `tests/test_dense.py` opens
with `pytest.importorskip("sentence_transformers")`, so its 16 tests are not
collected without the dense group, and 349 + 16 = 365.

### Next steps (Day 8)

1. **Resume the 7B grid first thing** — `uv run python -m experiments.run_grid
   --model qwen2.5-7b --rubric minimal --n 600 --seed 0 --threads 4`. It
   resumes from the store; on a fresh container re-download the pinned GGUF
   first (`src/judge.py` MODELS carries repo, revision and SHA256; the runner
   verifies the digest at load and refuses to run without it). Expect ~7 h at
   this host's rate — plan on it spanning sessions and checkpoint-commit the
   store before the session ends, as today.
2. **While it runs, do not start CPU-heavy analyses** — today's bootstraps
   measurably slowed the grid (load average 5.1 with 4 vCPUs). Writeup work is
   free; analysis is not.
3. **When the grid closes**: rerun `summarize`, `master_table`, `scaling_curve`,
   `length_probe`, `calibration`, `bias_model`, `subset_view`, `compliance_view`
   over six stores, then write findings 27+ against the questions the 7B point
   is meant to settle — does the Qwen valley stay closed above 3B; does the
   B-lean that appeared at 3B keep growing; does 7B beat the fitted length
   floor by more than 3B's +0.205; is Qwen calibration still broken at the top
   of the family. Only then restructure the results narrative around the
   scaling arc (still outstanding from day 6 — the glance table and findings
   index are the spine it will hang on).
4. **After 7B**: Llama-3.1-8B as the family counterpart, then the
   rubric-sensitivity axis (planned experiment 5, the last untouched phase-3
   item — `detailed` rubric already exists in `src/prompts.py`, so it is purely
   a compute question: one more grid per model).

## 2026-08-24 — Day 8: the partial grid becomes a sample (finding 27)

(Gap 08-22..08-23: no sessions ran. The 7B store was left at 134/1200 by the
day-7 checkpoint and is where this session picked it up.)

Fresh container again: `uv sync --group judge` (llama-cpp-python resolved to a
wheel this time, no 6-minute compile), the pinned 7B GGUF re-downloaded and
SHA256 re-verified against the registry pin (`65b8fcd9…1423`, 4.68 GB), and the
RewardBench parquet re-fetched and verified. The runner re-derived `n_ctx` 2784
from `max_prompt_tokens` 2768, byte-identical to the sidecar written on day 6 —
the context sizing is a property of (n, seed, rubric), which is what lets a grid
span sessions at all.

### The plan was "resume the grid first thing"; it changed after ten minutes

Day 7 closed with a note to relaunch immediately and not to touch anything
CPU-heavy while it ran. Before launching I went to re-read how resume decides
what is left, and found the thing day 7 had assumed without checking.

`ResultStore.existing_keys` returns a **set** of `(model, rubric, order,
item_id)`, and `run_grid` filters candidate prompts against that set.
`assemble_pairs` groups records by `item_id` and iterates `sorted(by_item)`.
So the order judgments are executed in is not observable anywhere: not by
resume, not by any analysis, not by any figure. The day-7 limitation —
"a randomized execution order would make partial grids interpretable and is the
obvious fix; it is not applied retroactively because it would break
resume-compatibility with the six stores already collected" — was simply
false, and it had been load-bearing enough to keep finding 26's root cause in
place for a whole session. The guard (`master_table --restrict-to`) was built
instead of the fix.

That is worth being precise about, because the guard and the fix do different
things. Matching item sets across judges recovers **comparability**; it can
never recover **representativeness**. Six judges matched on an all-Chat prefix
are six judges measured on Chat. No amount of care at read time fixes a store
whose composition is wrong — only the schedule can.

So: write the scheduler first, then launch. Cost about forty minutes of grid
time, against every future partial read of this store and every grid after it.

### `src/schedule.py` — deficit scheduling

At each step, serve the stratum with the largest proportional deficit

    deficit_s = p_s * (D + 1) - d_s

(share `p_s`, finished in stratum `d_s`, total finished `D`) — largest-remainder
apportionment run incrementally. Ties break by subset name, items inside a
subset keep `item_id` order, so a schedule is a deterministic function of
(sample, finished set) with no RNG anywhere. The scheduling unit stays the
*item*, both orders consecutively, because the swap pair is what every analysis
consumes; an item left half-judged by an interrupted run is scheduled first on
the next run, so a store carries at most one orphan and only while a run is up.

The bound is the reason to prefer this over a seeded shuffle. Deficits sum to
exactly 1 at every step, so the served stratum's deficit is at most 1 and drops
to at most 0 once served: no subset is ever more than one item from its
proportional share, at any prefix, in **every** realization. A shuffle gives
that in expectation, which is the wrong guarantee when the object being read is
one store rather than an ensemble.

Checked rather than assumed, since assuming is what produced the day-7 error:
against a greedy rule that directly minimizes total-variation distance at each
step, the apportionment rule is not merely competitive but **identical at every
step to floating-point noise** — max deviation 1.1e-16 from scratch and 5.6e-16
resuming the real 67-item prefix. The cheap principled rule is also the optimal
one here, so there is no tradeoff to write up.

**Finding 27 — a partial grid is a scheduling choice, not a fact of the
harness, and the objection that kept the old order was false.** Under the
legacy `item_id` order a store sits 0.497 in total-variation distance from the
benchmark's subset composition at the halfway point (300/600 items) and first
stays under 0.05 only at item 569 of 600 — unusable for essentially the whole
run. Deficit scheduling is under 0.05 from item 55 and holds every subset
within one item of proportional throughout.

### What this cannot do, and the store that proves it

The 67 items already judged under the old order cannot be un-judged. The
scheduler makes everything it *adds* proportional, so the store's composition
converges to the target only by dilution: category total-variation 0.746 at the
handover, 0.090 at 300 items, 0.000 at completion. There is a matching exact
statement for the drift metric — an over-represented stratum with `d` items at
share `p` falls back inside the one-item bound only at `D > (d-1)/p`, and the
scheduler recovers exactly there and not one item later (asserted as a test).

So the 7B row is still not a benchmark number, and today did not make it one.
It is now merely *becoming* one at the fastest rate arithmetic allows, which
day 7's ordering was not doing at all. Interim reads still go through
`master_table --restrict-to`, and the limitation is rewritten rather than
deleted.

### Built

- `src/schedule.py`: `balanced_order`, `coverage` (per-stratum drift plus
  total-variation distance from target), `format_coverage`.
- `experiments/run_grid.py`: `--order balanced|sorted`, default balanced;
  `sorted` reproduces a historical run. Coverage is printed at launch and
  every 20 judgments at the category level, so the log itself says how
  readable the store currently is. `execution_order` recorded in the sidecar.
  `n_ctx` sizing explicitly documented as computed over the whole sample, never
  the scheduled subset — otherwise a long grid's context could drift between
  sessions.
- `experiments/schedule_coverage.py` + `results/figures/schedule_coverage.png`:
  the three trajectories (legacy, balanced, balanced-resuming-the-prefix) at
  both stratum levels. It runs no judgments and reads no store — a stable
  artifact of (sample, schedule) that does not change as the grid advances,
  unlike the interim master table.
- 29 new tests (119 total, 1 skipped without a GGUF; ruff clean). Two of my
  first-draft assertions were wrong about the recovery dynamics and were
  replaced by derived ones rather than loosened: the rare-subset wait is
  exactly the deficit crossing `(6/96)(D+1) - 5 >= (90/96)(D+1) - (D-5)`, i.e.
  84th; the bound recovers exactly at `floor((d-1)/p) + 1 - d`, i.e. 204th.
  Total-variation contraction is *not* strictly monotone — rounding an integer
  item into a share can add back a fraction of one item's distance in the tail
  — so the test asserts no step ever adds back a whole item's worth, which is
  the true statement.

### Host and grid state

The host prefills at **54.6 tok/s** at 7B over the session's 306 judgments,
against 30.1 tok/s for the 134 records inherited from days 6–7 — a mid-speed
container by the per-host ledger, roughly day 6's.

Per-judgment throughput went 0.043 → 0.107 judg/s, a 2.5x speedup, and it
decomposes cleanly: **1.81x from host tok/s and 1.38x from shorter prompts**
(mean 703 tokens for the legacy Chat-heavy prefix against 510 for the balanced
draw). So the balanced order is also, incidentally, cheaper per judgment here —
RewardBench's Chat subsets carry the longest responses in the benchmark, and
the alphabetical order front-loaded exactly those.

Recorded because I got this wrong mid-session and had written the opposite.
At 47 judgments in, the balanced draw's mean prompt was 723 tokens —
indistinguishable from the prefix's 703 — and I concluded per-judgment cost was
composition-independent and the speedup purely hardware. Those 47 were the
deficit rule's opening burst into `math-prm` and `xstest-should-respond`, whose
chain-of-thought responses are long; the next 259 averaged 471 tokens. Reading
a trend off the first few percent of a run, in the exact session that built a
scheduler because prefixes of runs are unrepresentative. The claim is corrected
above rather than deleted, and n=306 is what it now rests on.

**Session close.** Grid stopped deliberately at a clean boundary rather than
left to die with the container: store at **440/1200 judgments, 220 complete
items, zero orphans**, argmax compliance **1.000** across all 440. Coverage
distance from the target composition fell from 0.858 to **0.167** at subset
level and 0.746 to **0.148** at category level; max subset drift 17.77 → 12.67
items. 306 judgments added this session against 87 on day 7 and 47 on day 6.
The remaining 760 judgments are ~2 h at this host's rate. The store resumes
untouched — that is the whole point of the protocol.

### Next steps (Day 9)

1. **Resume the 7B grid first thing** — `uv run python -m experiments.run_grid
   --model qwen2.5-7b --rubric minimal --n 600 --seed 0 --threads 4`. Balanced
   is now the default order, so the command no longer needs `--order`; it
   resumes from the store and schedules whatever is most under-represented. On
   a fresh container, re-download the pinned GGUF first (`src/judge.py` MODELS
   carries repo, revision and SHA256; the runner verifies the digest at load
   and refuses to run without it). The launch banner now prints the store's
   coverage, so the first thing the log says is how readable the store is.
2. **Keep analysis off the box while a grid is up.** Day 7 measured the
   contended rate at 0.026 judg/s against 0.04 idle; today's ruff+pytest runs
   visibly dented the early rate. Writeup work is free, bootstraps are not.
3. **When the grid closes**: rerun `summarize`, `master_table`,
   `scaling_curve`, `length_probe`, `calibration`, `bias_model`, `subset_view`,
   `compliance_view` over six stores, then write findings 28+ against the
   questions the 7B point is meant to settle — does the Qwen valley stay closed
   above 3B; does the B-lean that appeared at 3B keep growing; does 7B beat the
   fitted length floor by more than 3B's +0.205; is Qwen calibration still
   broken at the top of the family. Only then restructure the results narrative
   around the scaling arc (outstanding since day 6 — the glance table, findings
   index and now the schedule section are the spine it will hang on).
4. **After 7B**: Llama-3.1-8B as the family counterpart — and note it will be
   the first grid collected entirely under the scheduler, so its partial store
   is readable from item 55 on, unlike this one. Then the rubric-sensitivity
   axis (planned experiment 5, the last untouched phase-3 item; `detailed`
   already exists in `src/prompts.py`, so it is purely a compute question).
5. **Do not** paste interim 7B numbers into the README. The store still carries
   the 67-item inherited prefix and converges only by dilution; `master_table
   --restrict-to qwen2.5-7b` remains the only honest interim read, and it is a
   matched cross-judge comparison, not a benchmark estimate.

## 2026-08-26 — Day 9: the 7B grid closes; findings 28–31

(A session ran on 08-25 that this log does not narrate: its checkpoint commits
carry the grid from 440 to 1166/1200 over ~4.5 h, but the session ended without
a NOTES entry — presumably the container died after the last checkpoint. The
commit trail is the record for that day; nothing else appears to have been
touched. This is what checkpoint-committing the store every few minutes is
for.)

Fresh container: `uv sync --group judge` resolved llama-cpp-python to a wheel,
the pinned 7B GGUF re-downloaded and SHA256 re-verified against the registry
pin (`65b8fcd9…1423`, 4.68 GB), RewardBench parquet re-fetched and verified,
122 tests green before touching anything.

### The grid is done

The remaining 34 judgments ran in 4.1 min (0.14 judg/s — the fastest host of
the four this grid has seen; per-host variance to the end). Store closed at
**1200/1200 judgments, 600 complete items, zero orphans, argmax compliance
1.000 across all 1,200** — Qwen format discipline holds at a fourth size, and
the final coverage print reads total-variation 0.000, max drift 0.00 items.
The audit's first multi-session grid: 47 judgments on day 6, 87 on day 7, 306
on day 8, 726 on the unlogged 08-25 session, 34 today, spanning 2026-08-01 to
2026-08-26.

### Findings (all analyses rerun over six stores; four questions, four answers)

**Finding 28 — the valley resolves into a climb, and 7B is the audit's first
signal-dominant judge.** Sym acc 0.837 [0.807, 0.867]; paired +0.095
[+0.058, +0.132] over Qwen-3B, +0.335 over the 1.5B valley floor, +0.185 over
Llama-3B. The Qwen arc is 0.568 → 0.502 → 0.742 → 0.837. Structurally new:
|b| > |s| on only 26.8% of items (62.0–99.8% for every other judge); median
|s| 9.08 vs median |b| 2.78 — signal 3.3x bias where even the 3B had bias
1.7x signal. Best in all four categories at once (Chat 0.944, Reasoning
0.854, Safety 0.838, Chat Hard 0.696) — no earlier judge held the lead
anywhere near uniformly.

**Finding 29 — the flip-rate inversion completes: the audit's best judge
posts its highest flip rate.** Per-order accuracy 0.762/0.800 — the first
near-symmetric pair in the audit. The 3B's B-lean did *not* keep growing:
median b collapses −5.55 → +0.23 (b > 0 on 52.0%), but dispersion stays
(sd 4.89, second only to 3B's 5.47) — the bias lost its direction, not its
size; it is now per-item idiosyncrasy. Consequence: flip rate 0.732, the
audit's highest, on the audit's best judge — a content-following verdict
names a different letter whenever the responses swap seats. Both extremes of
the black-box consistency metric now belong to its two worst possible
readings (0.002 on the worst-biased judge, 0.732 on the best judge).

**Finding 30 — 7B beats the fitted length baseline by the audit's largest
margin, significantly in every category.** β_s +1.853 [+1.652, +2.126];
joint − length-only +0.272 [+0.230, +0.327] (3B: +0.205). Per category:
Chat +0.153, Chat Hard +0.082 [+0.011, +0.201] (first significant
adversarial delta in the audit), Reasoning +0.259, Safety +0.432 — all CIs
exclude 0, a first. Overall sign-vs-length agreement 0.489 (the most
length-neutral judge measured); math-prm agreement 0.233 with subset sym
0.778 [0.69, 0.86] — anti-length pointed where the subset rewards it, like
the 0.5B's 0.844, but this time backed by the audit's largest
length-controlled Reasoning coefficient (β_s +2.132).

**Finding 31 — the one-call correction ceiling keeps falling as judges
improve, and Qwen overconfidence survives to the top of the family.**
Additive shift rejected at a sixth scale (category R² 0.115; subset+length
0.328, residual sd 4.01). Ladder: best fitted rung 0.795 [0.768, 0.821]
(regression), significantly below the oracle (Δ −0.042 [−0.060, −0.022]) —
~25% of the symmetrization gain, extending finding 20's arc within Qwen:
68% (0.5B) → 47% (3B) → 25% (7B). The deployment inversion is worth
recording: the 7B's *uncorrected* single call (0.781) already beats every
other judge's two-call oracle. Calibration: sym ECE 0.121 [0.093, 0.150],
mean conf 0.958 vs acc 0.837 — milder than 3B's 0.153, still overconfident;
finding 23's family split is now exact at four Qwen sizes (only 0.5B
calibrated) vs two Llama sizes (both calibrated).

### Tooling: the caption that would have gone stale today, and did

`master_table`'s generated caption hardcoded "Only the two 3B judges beat
that one" — written on day 7, false the moment the 7B probe ran. The verdict
is now computed from the length-probe summary (winners = overall
joint − length-only CI above zero), so the sentence updates itself when the
8B grid lands. New `tests/test_master_table.py` pins winners-from-CIs,
no-winners, missing-summary, and sort order; the one existing test that
asserted the hardcoded string now asserts the paragraph's presence instead.
123 tests total (122 passed, 1 skipped without a GGUF), ruff clean.

### Writeup

README: new section "The 7B tier — the audit's first signal-dominant judge"
(3B-vs-7B table, findings 28–31, decomposition figure and its distinctive
shape — centered on b = 0, stretched along s); glance table regenerated with
the 7B row; status paragraph rewritten around six grids and findings 1–31;
"how to read it" bullets updated (the bias > signal bullet now has its
counterexample); findings index +28–31; subset highlight table gained the
7B column with a note that findings 24–25 predate it; the
unrepresentative-prefix limitation rewritten in the past tense; prefix_skew
caption updated. Root README's flagship section updated to six grids and
the completed arc. All cross-judge figures regenerated from the six stores
(scaling curve, probe forest, calibration, bias model, subset forest,
7B decomposition/accuracy/compliance).

### Late session: the 8B grid is registered and running

The day had compute left after the writeup, so day 10's first item moved up.
Registered `llama-3.1-8b` (bartowski Meta-Llama-3.1-8B-Instruct-GGUF, revision
`bf5b95e9`, Q4_K_M, SHA256 `7b064f58…557c` verified against HF's LFS oid after
download and at load; llama3 template; the 7B GGUF deleted first for disk —
models/ is gitignored and the registry pin re-verifies any re-download), added
its eighth MODEL_STYLES entry (darkest red, continuing the family gradient),
and launched n=600 seed 0 both orders. Max prompt is 2,748 tokens under the
Llama tokenizer → n_ctx 2764. Early rate 0.10 judg/s (ETA ~3.3 h) — this grid
is the first collected entirely under the deficit scheduler, so its partial
store is a stratified sample from item ~55 on, and any mid-run read via
`master_table --restrict-to llama-3.1-8b` is composition-clean by
construction. Checkpoint-commits as the store grows; the session closes with
whatever the store holds.

### Next steps (Day 10)

1. **Resume the Llama-3.1-8B grid first thing** — `uv run python -m
   experiments.run_grid --model llama-3.1-8b --rubric minimal --n 600
   --seed 0 --threads 4`. On a fresh container re-download the pinned GGUF
   first; the runner verifies the digest at load and refuses to run without
   it. It is the audit's last planned scaling point.
2. **When the 8B grid closes**: rerun the six cross-judge analyses over
   seven stores, write findings 32+ against the family questions (does
   Llama reach signal-dominance at 8B or is that a Qwen property; does its
   Chat-Hard hole persist at the top tier; does the correction-ceiling arc
   hold cross-family), then do the README results-narrative restructure
   around the now-complete scaling arc (outstanding since day 6 — with both
   families' arcs closed it finally has its full spine).
3. **After the scaling grid closes**: the rubric-sensitivity axis (planned
   experiment 5, the last untouched phase-3 item) — `detailed` rubric exists
   in `src/prompts.py`; budget one grid per judge, smallest models first.
4. **Keep analysis off the box while a grid is up** — the day-7/8 contention
   numbers still stand.
5. **Dependency housekeeping (GitHub flags 3 dependabot alerts on the
   repo).** An osv.dev sweep over both uv.lock files today localizes them:
   `diskcache==5.6.3` (GHSA-w8v5-vhqr-4h9v, a llama-cpp-python transitive
   here) and `setuptools==81.0.0` (GHSA-h35f-9h28-mq5c) in this project's
   lock, `torch==2.12.1` (GHSA-rrmf-rvhw-rf47) in rag-chunking-bench's dense
   group. Deferred today because re-locking and re-running both test suites
   would contend with the 8B grid; it is a clean lock-bump + full-suite job
   for a session where the box is free.

## 2026-08-26 — Day 9, second act: the 8B grid runs end-to-end; the scaling axis closes (findings 32–35)

The 8B grid launched in the late session finished the same day: 1,200
judgments in 190.5 min at a steady 0.10 judg/s — the audit's first
single-session grid at the large tier, and the first collected entirely under
the deficit scheduler (category total-variation never above 0.006 at any
100-judgment checkpoint after warmup; the log's own coverage lines are the
record). Store verified: 600 complete items, zero orphans; compliance 0.910 —
the Llama partial-compliance signature, all 97 non-compliant judgments... see
finding 35 for where they sit. All seven-store analyses rerun; one tooling fix
en route: `scaling_curve.family_of` didn't know the `llama-3.1` prefix and
refused the store — the family line now takes both `llama-3.1`/`llama-3.2`
prefixes as "Llama-3" (Meta ships them as one herd; the release-version
confound is recorded in the README limitations).

**Finding 32 — the two family arcs never cross: at the top tier, family beats
scale.** Llama monotone 0.555 → 0.652 → 0.723 (paired +0.072 [+0.032, +0.110]
over its 3B), no valley; second signal-dominant judge (|b| > |s| on 33.5%;
per-order 0.613/0.752; bias direction reversed a second time within the
family, median b −0.59 after 3B's saturated +2.34). But against the other
family on identical items: −0.018 [−0.055, +0.018] vs Qwen2.5-3B (a tie with
a 2.7x smaller judge) and −0.113 [−0.150, −0.078] vs Qwen2.5-7B.

**Finding 33 — the adversarial hole is a family property all the way up.**
Chat Hard: 0.435 → 0.348 → 0.522 — back to chance, no further; −0.174
[−0.272, −0.076] behind Qwen-7B on the same 92 items; llmbar-adver-neighbor
0.333, GPTInst 0.421. Chat meanwhile hits 0.958 — the audit's best category
score anywhere, dead heat with Qwen-7B (+0.014 [−0.028, +0.056]). Llama
scale buys Chat, never adversarial robustness (finding 22, now at 8B).

**Finding 34 — the audit's strongest length-controlled signal rides its
strongest length lean, and pays for it on math-prm.** β_s +2.376
[+1.948, +2.998] (largest in audit); joint − length +0.233 [+0.185, +0.283]
(fourth judge over the fitted floor, second-largest margin). Yet
sign(s)-vs-length agreement is 0.633 — the most length-following judge
measured (Llama: 0.622 → 0.596 → 0.633; Qwen-7B 0.489) — and math-prm lands
below chance at 0.389 [0.29, 0.49] where Qwen-7B holds 0.778: finding 24's
composition mechanism operating at the top tier. Chat Hard: β_s +0.755
significant but the accuracy delta over the length floor is null (+0.027).

**Finding 35 — the one-call ceiling holds at ~25% at 8B, finer corrections
actively hurt, and the Safety compliance migration replicates.** Additive
shift rejected at a seventh scale (category R² 0.124); subset+length R²
0.492 — second-most predictable bias in the audit, with length covariates
adding +0.16 (the family's length sensitivity is in its *bias* too). Ladder:
global 0.692 [0.662, 0.721] is the best rung (24% of the gain — the 7B's
ceiling, again), while category 0.676 and subset 0.673 fall *below*
no-correction 0.682. Calibration: sym ECE 0.101, signed gap +0.042 —
Llama's first mildly overconfident member (1B +0.005, 3B −0.012), still
well under Qwen-7B's +0.121; the family split survives, blurred at the
edge. Compliance: non-compliance 100% Safety-concentrated (0.635 compliant
there, 1.000 in every other category) and the compliant Safety stratum
judges worse, −0.179 [−0.306, −0.046] — finding 25's Llama-3B result
(−0.223) replicated at 8B. Parse-and-drop would keep the worse half again.

**Writeup.** README: status rewritten (phase 2 closed), 8B row in the glance
table, "how to read it" bullets extended to the seven-judge field, new
section "The 8B tier — family beats scale" with findings 32–35 and the 8B
decomposition figure, subset highlight table gained the Llama-8B column,
family-geometry limitation now records the 3.1-vs-3.2 release confound;
findings index 32–35. ROADMAP: phase 2 marked complete, header moved to
phase 3. Root README flagship section updated to the seven-grid field. All
cross-judge artifacts regenerated over seven stores.

### Next steps (Day 10) — supersedes the earlier day-10 list

1. **The rubric-sensitivity axis** (planned experiment 5, the last untouched
   phase-3 item): `detailed` rubric grids, paired against `minimal` in
   log-odds per item. Budget one grid per judge and start small —
   qwen2.5-0.5b (~1 h) and llama-3.2-1b first, since their minimal-rubric
   pathologies (always-A saturation; partial compliance) are the most
   interesting to test for rubric dependence. The paired-in-log-odds
   analysis needs a small new module (paired Δz per item across rubrics);
   design it before running anything.
2. **The results-narrative restructure** around the completed scaling arc —
   outstanding since day 6, now fully unblocked: reorganize the README
   results from grid-arrival order into the arc (families × scale, the
   valley, the direction chiasm, signal-dominance at the top, the
   correction ceiling, the family verdict), keeping the per-grid sections
   as the archival record beneath.
3. **Dependency housekeeping** (item 5 above) now that the box is free:
   bump diskcache/setuptools pins here and torch in rag-chunking-bench,
   re-lock, full suites, push.
4. **Cleanup**: the 8B GGUF (4.9 GB) can be deleted at session start if
   disk is needed; re-download is pinned and verified.

## 2026-08-27 — Day 10: the rubric axis opens — verdicts are rubric-fragile below 3B (findings 36–38); the results narrative gets its restructure

Fresh container: `uv sync --group judge`, both small GGUFs re-downloaded and
SHA256-verified against the registry pins. Pre-work suite: 118 passed with
the real-model smoke test failing against the still-downloading GGUF (it
passed once the download finished; full suite green at end of day).

### The paired rubric machinery

The rubric axis is the order axis one level up: both stores cover the same
600 items × both orders, so two rubrics compare paired per item in log-odds.
New `src/rubric_pair.py`: inner-join of complete swap pairs across stores by
`item_id` (unmatched counts surfaced, duplicate detection), per-rubric stats
on the matched items, paired deltas with bootstrap CIs (sym/raw accuracy,
s, b, |b|, |s|, positional flip rate, compliance — the rubric changes the
instruction, so format discipline is allowed to move), cross-rubric
consistency (the **rubric flip rate** — the prompt-level analogue of the
positional flip rate — plus Pearson/Spearman of s and b with degenerate
replicates excluded, not coerced), and the per-category block.
`experiments/rubric_view.py` sweeps every model with both stores, merges
single-model runs into the combined JSON without clobbering, renders the
markdown table, and (sweep mode only, so a one-model rerun can't overwrite
the canonical panel) draws the identity figure: s-vs-s and b-vs-b across
rubrics, item-paired, per model. 7 new tests recover constructed effects
from a synthetic judge whose rubric-B halves bias and shifts preference;
130 total, ruff clean.

### The grids

Both `detailed` grids ran end-to-end today under the deficit scheduler,
compliance-clean stores, TV 0.000 at close:
- qwen2.5-0.5b__detailed: 1200/1200 in ~65 min (0.31 judg/s), n_ctx 2826
  (max prompt 2810 — the detailed rubric adds ~42 tokens).
- llama-3.2-1b__detailed: 1200/1200 in 92 min (0.22 judg/s avg).

### Findings (against the minimal stores, paired on all 600 items)

**Finding 36 — the symmetrized verdict is rubric-fragile at small scale:
30–43% of debiased verdicts change with the rubric text alone, an order of
magnitude more churn than the net accuracy movement.** 0.5B: rubric flip
rate 0.303 [0.268, 0.340] against positional flip rate 0.000; flips are
symmetric noise (84 right→wrong vs 98 wrong→right; Δ sym +0.023
[−0.020, +0.067], null). 1B: flip rate 0.432 [0.393, 0.472] (259 flips,
net +43), r(s) across rubrics 0.257 [0.140, 0.379]. Flips concentrate at
small |s| (0.5B by minimal-|s| quartile: 0.42/0.40/0.30/0.09; 1B:
0.52/0.43/0.49/0.29) — with median |s| ~0.15–0.24, the rubric perturbation
is the same order as the signal, so the sign re-randomizes. Order-swap
consistency is not verdict stability: a perfectly order-consistent judge
(0.5B flips 0/600 under swap at detailed) still changes nearly a third of
its verdicts when the prompt is reworded.

**Finding 37 — at 0.5B the detailed rubric contracts the whole log-odds
distribution and touches nothing structural.** Δ|b| −0.628 [−0.688, −0.567]
(median b +3.65 → +3.03) but Δ|s| −0.165 [−0.194, −0.137] — proportionally
more (−30% vs −17%) — and mean Δs −0.048 [−0.083, −0.012], slightly *away*
from gold. Bias dominance rises 0.998 → 1.000, per-order accuracy stays
exactly 1.000/0.000. Four explicit criteria including the anti-order
instruction produce a 17% smaller push toward A and nothing else:
prompt-side instruction is not a debiasing lever at this scale. b is more
rubric-stable than s (r 0.735 vs 0.610) — the most reproducible property of
this judge is its pathology.

**Finding 38 — at 1B the rubric reverses both directional properties; the
significant gain is a re-aimed length lean.** Bias direction flips B→A
(median −0.34 → +0.62, mean −0.34 → +1.04, Δ|b| +0.64 [+0.55, +0.73]) —
after 11/21/32, bias direction is now not even a property of a fixed
(model, sample) pair. Length orientation reverses with it:
sign(s)-vs-length agreement 0.622 → 0.408. That explains the headline
+0.072 [+0.018, +0.123] (0.555 → 0.627) exactly: Reasoning (longer usually
wrong) +0.132 [+0.059, +0.205]; Chat (longer usually right, finding 13)
−0.222 [−0.375, −0.069] — finding 24's composition mechanism operating
within one judge across prompts. Compliance collapses 0.512 → 0.275
(Chat 0.625 → 0.125, Safety 0.838 → 0.351, Chat Hard 0.793 → 0.380;
Reasoning stays broken at ~0.23): the longer instruction makes the model
less able to open with a verdict letter while judging better. Parse-and-
drop under the detailed rubric keeps 165/600 items (27.5%) and they are
the worse-judged stratum (0.570 vs 0.648, −0.079 [−0.167, +0.009]) —
findings 8/25/35's warning at its sharpest.

### Writeup

README: new "The rubric axis" section (paired table, findings 36–38,
identity-panel figure) at the end of the per-grid record; findings index
+36–38; planned experiment 5 marked started; the "one rubric so far"
limitation rewritten (two rubrics × two judges; rubric-wording vs
prompt-length not separable with two templates); status paragraph
rewritten; test counts refreshed. And the long-outstanding
results-narrative restructure landed: new "The scaling arc" section after
the glance table telling the completed arc by theme (accuracy arcs, bias
direction/dominance, the flip-rate inversion, the falling one-call
ceiling, length + the adversarial family split, calibration/compliance
signatures), with the twelve arrival-ordered sections demoted beneath a
"The per-grid record" umbrella as the unrewritten archive. Anchors were
level-independent, so no links broke.

### Next steps (Day 11)

1. **Extend the rubric axis upward**: `uv run python -m experiments.run_grid
   --model qwen2.5-1.5b --rubric detailed --n 600 --seed 0 --threads 4`
   (~2 h at day-3 rates) — the valley judge is the priority third point:
   does the verbosity preference that defines the valley survive a rubric
   that never mentions length, or is the valley itself rubric-specific?
   Then qwen2.5-3b (~4 h) if the session has room. The key questions the
   two small judges cannot answer: does rubric fragility (finding 36)
   decay as |s| grows (prediction: yes — flip rate should fall roughly as
   P(|s| < perturbation scale)), and does the bias-direction reversal
   (finding 38) replicate at scales where bias is large rather than
   near-zero?
2. **After any new grid**: `experiments/rubric_view.py` sweep regenerates
   the combined JSON/table/figure; single-model runs merge without
   clobbering.
3. **Dependency housekeeping** (carried from day 9, box was busy all of
   today): bump diskcache/setuptools here and torch in rag-chunking-bench,
   re-lock, full suites, push.
4. **Disk**: both small GGUFs (~1.2 GB total) can stay; delete before any
   7B/8B rubric grid.

## 2026-08-28 — Day 11: the fragility model — rubric flips are the signal-to-perturbation ratio (findings 39–41)

Fresh container: `uv sync --group judge`; 0.5B/1.5B/3B GGUFs downloaded and
SHA256-verified against the registry pins. Pre-work suite: 125 passed with
the expected real-model smoke failure against the still-downloading 0.5B
GGUF (132 passed, zero skips, at end of day with all models present).

### The grid

`qwen2.5-1.5b__detailed` ran end-to-end under the deficit scheduler:
1200/1200 in 119.3 min (0.17 judg/s), n_ctx 2826, TV 0.000 at close,
compliance 1.000, zero incomplete items. The third rubric point, and the
first where the reference preference (median |s| 0.503 under minimal) is
substantially larger than the rubric perturbation.

### New machinery: the perturbation model

Day 10 located rubric flips at small |s|; the falsifiable version is a
model. `src/rubric_pair.fragility_fit`: s_detailed = λ·s_minimal + ε with
ε homoskedastic Gaussian — λ the through-origin least-squares slope, σ the
residual sd — under which an item flips sign with probability Φ(−λ|s|/σ).
The fit reports observed-vs-predicted flip rates per quartile of |s| so it
can fail bin by bin. `experiments/rubric_fragility.py` sweeps every judge
with both stores, writes `rubric_fragility__minimal_vs_detailed.{json,md}`,
and renders the flip-rate-vs-|s| figure (observed points + fitted curves).
2 new tests (parameter recovery on a constructed field; degenerate-input
errors); 132 total, ruff clean.

### Findings (paired on all 600 items; day-10 questions answered)

**Finding 39 — rubric fragility is the signal-to-perturbation ratio, not a
property of small judges, and the two-parameter model predicts where the
flips are.** Flip-rate arc 0.303 / 0.432 / 0.190 (0.5B / 1B / 1.5B) ordered
exactly by median reference |s| (0.235 / 0.144 / 0.503) — the most fragile
judge is the weakest-preference one, not the smallest. Fits: λ 0.345 /
0.374 / 0.522, σ 0.236 / 0.532 / 0.418. Quartile profiles reproduced —
0.5B nearly exactly (0.47/0.40/0.29/0.10 predicted vs 0.42/0.40/0.30/0.09
observed). At 1.5B the mid-quartiles come in *below* prediction (0.26 vs
0.34, 0.11 vs 0.18): with r(s) 0.834 the rubric moves the preference
coherently, so fewer signs flip than equal-sized noise would produce. The
model's failure direction is itself diagnostic.

**Finding 40 — at 1.5B the detailed rubric contracts both components and
halves the order asymmetry without touching the symmetrized verdict; the
1B's direction reversal does not replicate where bias is sizable.**
Δ|b| −0.36 [−0.41, −0.31] (median b +0.83 → +0.23 — 3.6x smaller, same
sign), Δ|s| −0.294, λ ≈ 0.52. Per-order accuracy 0.805/0.293 → 0.617/0.445
(gap 0.512 → 0.172) yet Δ sym −0.007 [−0.042, +0.028]: the anti-order
instruction buys real single-call order robustness and nothing else —
prompt-side debiasing pays only if you judge in one order. Finding 38's
bias-direction reversal is a small-|b| phenomenon (1B median |b| ≈ 0.3 ≈
the perturbation scale; 1.5B's 1.09 survives with its sign). Curiosity
with a lesson: positional flip rate is exactly 179/600 under both rubrics
(only 101 items shared) and bias dominance exactly 421/600 under both
(343 shared) — frozen aggregates over churning items.

**Finding 41 — the valley is rubric-invariant.** Sym 0.502 → 0.495 (null),
Reasoning 0.368 → 0.375 (still below chance), length orientation weakened
not re-aimed (sign-agreement 0.571 → 0.522), compliance 1.000 → 1.000.
The valley is a property of the model at this scale, not of the prompt —
now measured under two instructions. Family contrast: the detailed rubric
collapsed Llama-1B compliance 0.512 → 0.275; both Qwen judges hold exactly
1.000 — format discipline under instruction change is a family property,
like calibration (finding 23).

### Writeup

README: 1.5B column in the rubric table, findings 39–41 bullets, the
fragility figure with caption, identity-panel caption updated for three
rows, findings index +39–41, status paragraph, limitations (fragility
model's homoskedastic-Gaussian assumption and where it bends; larger-tier
prediction explicitly marked untested), repository layout. ROADMAP phase-3
line updated.

### Next steps (Day 12)

1. **Extend the rubric axis to 3B**: `uv run python -m experiments.run_grid
   --model qwen2.5-3b --rubric detailed --n 600 --seed 0 --threads 4`
   (~4 h at day-5 rates; the GGUF is already verified in models/). Key
   question: finding 39's extrapolation — at median |s| 3.64 (minimal) the
   model predicts a flip rate near zero; and does the detailed rubric
   shrink the 3B's large B-lean (median b −2.34) or is prompt-side bias
   reduction also capped where |b| is large? If the store is partial at
   session end, resume — matching item sets keeps the paired analysis
   honest at any prefix.
2. **After the grid**: `experiments.summarize --store qwen2.5-3b__detailed`,
   `rubric_view` sweep, `rubric_fragility` — all three regenerate their
   committed artifacts; check the 3B point lands on/off its fitted curve.
3. **Dependency housekeeping — done at end of day 11** (carried from day
   9). A fresh osv.dev sweep corrected day 9's localization: setuptools sat
   in rag-chunking-bench's lock, not this project's. rag-chunking-bench:
   setuptools 81.0.0 → 84.0.0 (GHSA-h35f-9h28-mq5c fixed in 83) and torch
   2.12.1 → 2.13.0 (GHSA-rrmf-rvhw-rf47 fixed in 2.13.0), suite green after
   the bump (363 passed, 2 skipped), lock now osv-clean. This project's
   only flagged pin, diskcache 5.6.3 (GHSA-w8v5-vhqr-4h9v, unsafe pickle
   deserialization), has **no fixed release** (last_affected = 5.6.3, the
   latest version) — nothing to bump. Exposure assessed as low: diskcache
   arrives as a llama-cpp-python transitive and this harness never
   deserializes cache content from untrusted sources; revisit when
   upstream ships a fix.
4. **Disk**: the 7B/8B GGUFs are not on this container; the three Qwen
   GGUFs (~3.5 GB) fit comfortably. Nothing to clean.

## 2026-08-28 — Day 11, second act: the 3B closes the day at four rubric points (findings 42–43)

The box was free after the writeup, so the day-12 grid ran today:
`qwen2.5-3b__detailed`, 1200/1200 in 242.0 min (0.08 judg/s) under the
deficit scheduler, TV 0.000 at close, compliance 1.000, checkpointed to
main throughout. All rubric analyses regenerated over four judges
(`rubric_view`, `rubric_fragility`, per-store summary).

**Finding 42 — the fragility arc extends to 3B on the model's own
prediction, and the coherent-movement deviation grows with judge quality.**
Flip rate 0.102 [0.078, 0.127] at median |s| 3.64; the four-judge arc
0.303/0.432/0.190/0.102 stays ordered by |s|; r(s) 0.912. λ climbs
0.345 → 0.374 → 0.522 → 0.767 (contraction weakens with scale), σ grows
slower than |s|. The Gaussian model's weak-|s| over-prediction deepens
(Q1 0.287 obs vs 0.394 pred; Q2 0.087 vs 0.201) exactly as r(s) rises:
better judges move coherently under rubric change, not noisily.

**Finding 43 — prompt-side debiasing works hardest where the bias is
largest, but buys raw accuracy and order balance, never symmetrized
quality — and it un-saturates the flip rate.** Audit's largest prompt-side
bias reduction on its largest bias: Δ|b| −1.80 [−2.01, −1.59], median b
−5.55 → −3.51, direction intact (with 1.5B: the rubric reverses direction
only where |b| ≈ perturbation scale — the 1B). Raw acc +0.033
[+0.016, +0.050] (chosen-first 0.368 → 0.428); sym +0.022 [−0.003, +0.047]
null. Positional flip rate rises 0.380 → 0.438 (Δ +0.058 [+0.027, +0.092])
— finding 3's flip-rate inversion driven within one judge by the prompt
alone. All per-category deltas null; flips uniform 0.09–0.12. Compliance
1.000 → 1.000 (third Qwen point). sign(s)-vs-length agreement 0.547 → 0.549
(unmoved — no re-aiming, unlike the 1B).

**Writeup.** README: 3B column in the rubric table, findings 42–43,
both figure captions updated for four rows/judges, status + limitations
(in-family extrapolation now tested through 3B; 7B/8B remain), findings
index +42–43. ROADMAP and root README refreshed.

### Next steps (Day 12) — supersedes the morning list

1. **The 7B detailed grid** — the remaining Qwen point and the first
   signal-dominant one: `uv run python -m experiments.run_grid --model
   qwen2.5-7b --rubric detailed --n 600 --seed 0 --threads 4`. At day-6
   rates (~0.04 judg/s) this is ~8 h — expect to span sessions; the GGUF
   (4.9 GB) must be re-downloaded first (pins in src/judge.py). Launch
   early, checkpoint often. Key predictions to test: flip rate ≈ 0.05 or
   lower (median |s| 8.68-ish on the full sample; read it from the
   minimal store first), and whether Δ|b| keeps scaling with |b| or the
   prompt-side lever saturates.
2. **Then the Llama detailed grids** (3.2-1B done; 3.2-3B ~4 h, 3.1-8B
   ~3.2 h at day-9 rates) — the cross-family fragility question: does
   Llama's larger σ persist at scale (the 1B point suggests family-scaled
   perturbation sensitivity), and does the compliance collapse under the
   detailed rubric (finding 38) replicate at 3B/8B where Safety was the
   migration target?
3. **After each grid**: `rubric_view` + `rubric_fragility` sweeps
   regenerate all committed artifacts.
4. **Disk**: delete the three small Qwen GGUFs before downloading the 7B
   (~3.5 GB frees; 7B needs 4.9 GB); all pins verified in the registry.

## 2026-08-29 — Day 12: the 7B closes the Qwen rubric line — the lever stops paying at the top (findings 44–45)

The day-11 plan executed as written: fresh container, env rebuilt from the
lockfile (127 tests green), the 7B GGUF re-downloaded and verified against
the pinned SHA256, and `qwen2.5-7b__detailed` run end-to-end in one session
under the deficit scheduler — 1200/1200 in 514.5 min (0.04 judg/s, the
day-6 rate), TV 0.000 at close, compliance 1.000 throughout. All rubric
analyses regenerated over five judges (`rubric_view`, `rubric_fragility`,
per-store `summarize`).

**Pre-registered check, resolved first.** Day 11 predicted "flip rate
≈ 0.05 or lower" from the fragility trend, flagging that the true median
|s| should be read from the minimal store before judging the outcome. Read
before the grid ran: median |s| = 9.08 on the full 600 sample (the 8.68 in
the day-11 note was the matched-prefix estimate). Observed flip rate:
**0.068 [0.050, 0.090]** — the CI's lower edge touches the guess, and the
fitted model's own quartile aggregate (~0.079) sits closer to the
observation than the verbal prediction did. Scored as: arc confirmed,
point estimate slightly high of the guess, model better than intuition.

**Finding 44 — the arc closes at five points, λ plateaus, and the
coherent-movement deviation peaks at 3B.** 0.303/0.432/0.190/0.102/0.068,
ordered by reference median |s| (0.235/0.144/0.503/3.64/9.08) end to end.
λ 0.345 → 0.374 → 0.522 → 0.767 → 0.777: the proportional contraction
stops deepening past 3B, but Δ|s| is the audit's largest in absolute
terms (−1.81 [−2.05, −1.58], median 9.08 → 7.00) because σ keeps growing
(2.577). The honest correction: finding 42 extrapolated the weak-|s|
over-prediction as "grows with r(s)" — it doesn't. With r(s) essentially
tied (0.916 vs 0.912), Q1 obs/pred is 0.240/0.283 (ratio 0.85) against
3B's 0.287/0.394 (0.73), Q2 0.027/0.033 against 0.087/0.201. The
coherent-movement residual peaks at 3B; at 7B the homoskedastic Gaussian
nearly suffices.

**Finding 45 — at the family's top, prompt-side debiasing re-signs a
balanced bias and buys nothing at all.** The 7B minimal store carries the
audit's most balanced net bias (mean b +0.15, median +0.23, 52.0% of
items leaning A — with item-level median |b| still 2.78). The detailed
rubric pushes it through zero: mean −1.30, median −1.25, A-lean share
37.0%, Δb −1.45 [−1.73, −1.18] negative in all four categories, against
a modest Δ|b| −0.75 [−0.99, −0.51]. Finding 40's boundary replicates at
the opposite end of the family: direction reversal happens exactly where
the net lean sits at the perturbation's own scale (1B −0.34, 7B +0.23);
the 3B's −5.55 only shrank. And unlike the 3B (raw +0.033, flips
+0.058), every purchase is null at 7B: raw +0.011, sym +0.008,
positional flip rate 0.732 → 0.750 (Δ +0.018, null — still the audit's
highest), sign(s)-vs-length 0.489 → 0.481, compliance 1.000 → 1.000
(fourth Qwen point). Only category texture: Safety +0.047
[+0.014, +0.088] canceled by borderline Chat Hard −0.065 [−0.130, 0.000],
Chat Hard also carrying the highest flip rate (0.109) — adversarial
items live at small |s| even for a signal-dominant judge.

**Writeup.** README: 7B column in the rubric table, findings 44–45, both
figure captions to five judges, status header, rubric-section intro,
planned-experiments item 5, limitations (Qwen line closed; Llama 3B/8B
remain), findings index +44–45. Root README and ROADMAP refreshed.

### Next steps (Day 13)

1. **The Llama-3.2-3B detailed grid** — `uv run python -m
   experiments.run_grid --model llama-3.2-3b --rubric detailed --n 600
   --seed 0 --threads 4` (~4 h at day-9 rates; GGUF 2.0 GB, pin in
   src/judge.py; delete the 7B GGUF first if disk is tight). The
   cross-family fragility questions from day 11 are still the open ones:
   does Llama's larger σ persist at 3B (family-scaled perturbation
   sensitivity), and does the 1B's compliance collapse under the detailed
   rubric replicate where Safety was the migration target (finding 35)?
   Note Llama-3B minimal is an always-A machine (median b +10.85,
   finding 21) — the direction-reversal boundary (findings 40/45) predicts
   *no* reversal, only shrinkage; and its median |s| (1.79) slots between
   1.5B and 3B, so the arc predicts a flip rate between 0.190 and 0.102.
2. **Then the Llama-3.1-8B detailed grid** (~3.2 h at day-9 rates,
   GGUF 4.9 GB) — closes the rubric axis at seven judges; after it, the
   rubric-axis figures/tables are final and phase 3 is complete.
3. **After each grid**: `rubric_view` + `rubric_fragility` + `summarize`
   regenerate all committed artifacts; update README tables/captions.
4. **Then phase 4**: the writeup coherence pass and the
   `rag-chunking-bench`-style clean-environment reproduction audit.

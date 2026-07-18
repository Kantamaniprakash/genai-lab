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

### Next steps (Day 3)

1. The Llama-3.2-1B grid is running (started this session; ~2 h). First:
   summarize it, render figures, and write up the cross-family contrast —
   early records show partial compliance (argmax sometimes "Response"/"I",
   median mass ≈ 0.85 on the first items) and a possible bias toward B
   rather than A. If compliance lands well below 1.0, add a
   compliance-conditioned view (readout validity is itself a per-model
   result, and mass_ab-weighted sensitivity checks become necessary).
2. Extend the grid: qwen2.5-1.5b (~3 h) — download pinned GGUF, register,
   run. Then 3B the day after (~6 h; consider starting it early in the
   session), and decide the 7B sample size (n=300 composition-preserving
   vs. two-day n=600) once the 1.5B effect sizes are in.
3. Start the scaling-curve figure (sym acc + mean |b| vs. params) once
   ≥3 models exist.
4. Backlog for phase 3 (not yet): additive-shift test on b_i dispersion,
   calibration/ECE, value-over-length regression, detailed-rubric axis.

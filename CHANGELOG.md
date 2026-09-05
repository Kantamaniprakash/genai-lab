# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-04

`slm-judge-audit` is complete: 49 findings across seven judges and two rubrics — 16,800 judgments on one pinned stratified sample — all reproducible byte-for-byte from the committed raw stores.

### Added
- White-box judging harness over llama.cpp: single-token verdict readout from full-vocabulary logits, the exact per-item decomposition of every order-swap pair into preference signal `s` and position bias `b`, resumable JSONL stores with provenance sidecars, readout-validity conditioning, and paired bootstrap confidence intervals throughout (findings 1, 5, 8).
- The judge scaling grid (Qwen2.5 0.5B/1.5B/3B/7B, Llama-3 1B/3B/8B, Q4_K_M, same 600 RewardBench items × both orders): debiased judge quality is non-monotone in scale — a 1.5B valley driven by an emergent verbosity preference, then a climb — the two family arcs never cross so family beats scale at the top tier, and both families reverse bias *direction* with scale (findings 9–11, 16–18, 21–23, 28–35).
- The flip-rate demolition: black-box "consistency" is anti-correlated with the truth at both extremes — the saturated always-A machines post the audit's lowest flip rates and its two best judges post the two highest (findings 2–3, 6, 21, 29).
- The additive-shift test and exact-LOO one-call correction ladder: position bias is never an additive constant, and the share of the symmetrization gain a fitted one-call correction recovers *falls* as judges improve (68% → ~25%), with finer corrections actively hurting at 8B (findings 19–20, 31, 35).
- Value-over-length probe: a fitted one-parameter length baseline outscores every sub-3B judge's debiased accuracy; four of seven judges clear it, only one in every category including adversarial Chat Hard (findings 12–14, 18, 22, 24, 30, 34).
- Calibration and compliance as family signatures: post-debiasing calibration is a family property orthogonal to accuracy, and the category where Llama breaks verdict format migrates with scale (findings 15, 23, 25, 31, 35).
- The rubric-sensitivity axis, paired per item in log-odds: 30–43% of symmetrized verdicts flip on the rubric text alone at small scale; prompt-side debiasing buys raw accuracy and order balance but never symmetrized quality, and turns harmful at 8B; a two-parameter fragility model's ratio λ·med|s|/σ strictly orders all seven observed flip rates, confirmed on a pre-registered out-of-sample grid (findings 36–49).
- Coverage-balanced execution scheduling (incremental largest-remainder apportionment), after finding 26 caught a compositionally confounded mid-run peek: a partial grid is now a stratified sample at every prefix, no subset ever more than one item from proportional (findings 26–27).
- Reproduction audit tooling: `experiments/reproduce.py` replays 25 generator invocations inside a clean copy of HEAD and byte-compares all 63 committed artifacts, with manifest coverage enforced in both directions; its first run caught a stale committed figure, and the release audit passed green (all 63) before close. 140 tests.

## [0.2.0] - 2026-07-16

`rag-chunking-bench` is complete: 26 findings, all reproducible from committed raw results.

### Added
- Cross-retriever grids (TF-IDF, LSA, dense MiniLM) showing every chunking effect is retriever-family-independent, that chunking moves recall more than retriever choice, and that past the encoder window dense retrieval degrades to prefix retrieval (findings 8–12).
- Chroma long-reference corpora: with sentence-scale gold evidence the small-chunk advantage inverts at generous budgets — the winning chunk size is set by gold-evidence length (findings 13–18, with budget-rule and drop-one-corpus robustness).
- cl100k_base BPE tokenizer unit: every headline claim is unit-invariant under real BPE accounting (finding 19).
- Semantic (embedding-breakpoint) chunker evaluation with a matched-realized-size protocol: the popular percentile chunker's wins are chunk-size drift, it gains nothing at matched realized size, and it retains a long-gold penalty; matched mean size is itself shown to be an uncontrolled comparison (findings 20–23).
- Per-question error analysis: per-corpus differences are gold-length composition, the loss tail splits into two identifiable mechanisms, and every overlap gain decomposes exactly into placement + extension − a redundancy tax (findings 24–26).
- Reproduction audit tooling: `experiments/reproduce.py` maps every committed table and figure to the invocation that produces it and byte-compares regenerated artifacts against the committed files; audited green in a clean environment (fresh clone, fresh interpreter, refetched data) before release.
- Findings-at-a-glance navigation table and cross-finding reconciliation in the report; 365 tests.
- Auto-updating "Latest from the lab" README section: `scripts/sync_latest.py` distills the newest research-log entry (headline findings, or the opening paragraph for side-repo days) onto the repo landing page, run by a workflow on every push that touches `NOTES.md`.

## [0.1.0] - 2026-07-05

### Added
- `rag-chunking-bench`: a token-budget-controlled benchmark that compares RAG chunking strategies at equal retrieved-token budgets rather than fixed top-*k*, isolating chunking quality from raw token count.
- Offset-preserving chunkers, a budget-matched retrieval protocol, and dataset loaders, with a SQuAD data pipeline that produces hand-verified gold evidence spans.
- Span-level evaluation metrics (SpanRecall, SpanPrecision, SpanIoU at a token budget *B*) scored against gold spans, plus classic hit@k for comparability with prior work.
- Paired bootstrap confidence intervals (fixed seed) over questions, so every "strategy A beats strategy B" comparison ships with an interval, not just a mean.
- A deterministic, resumable grid runner with per-question score persistence and a paired-CI summarizer for reproducible experiment sweeps.
- A hand-verified BM25 retriever and the first baseline grid showing the fixed-*k* chunk-size ranking reverses under budget-matched span recall.
- Overlap and budget-rule ablations (truncate-final-chunk rule) confirming overlap acts as boundary repair for fixed windows while the chunk-size effect survives truncation.
- Repository scaffolding: MIT license, GitHub Actions CI across Python 3.11/3.12/3.13, Dependabot config, and a project ROADMAP.

[0.3.0]: https://github.com/Kantamaniprakash/genai-lab/releases/tag/v0.3.0
[0.2.0]: https://github.com/Kantamaniprakash/genai-lab/releases/tag/v0.2.0
[0.1.0]: https://github.com/Kantamaniprakash/genai-lab/releases/tag/v0.1.0

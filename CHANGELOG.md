# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/Kantamaniprakash/genai-lab/releases/tag/v0.2.0
[0.1.0]: https://github.com/Kantamaniprakash/genai-lab/releases/tag/v0.1.0

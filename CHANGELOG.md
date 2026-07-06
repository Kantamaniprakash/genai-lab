# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/Kantamaniprakash/genai-lab/releases/tag/v0.1.0

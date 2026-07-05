# genai-lab Roadmap

This lab runs one flagship research project at a time, worked daily until it would
survive review by a demanding referee. Everything here is real: every number in a
writeup comes from an experiment actually run in this repo.

## Current flagship: `rag-chunking-bench`

**Question.** How much does the chunking strategy actually matter for RAG retrieval
quality — once you control for the retrieved-token budget?

**Why this, why now.** Chunking is the highest-leverage, least-principled knob in
every production RAG stack. The recent literature is active but methodologically
loose: most comparisons vary chunk size while holding top-*k* fixed, which confounds
chunking quality with the sheer number of tokens handed to the generator (500-token
chunks at k=5 retrieve 5x the text of 100-token chunks at k=5). Chroma's technical
report (Smith & Troynikov, 2024) introduced token-level metrics that partially
address this; recent arXiv work (Merola & Singh, 2025, arXiv:2504.19754; Duarte et
al., EMNLP 2024 Findings, arXiv:2406.17526) compares advanced chunkers but still
mostly at fixed *k*. A budget-matched, statistically careful comparison across
chunkers, chunk sizes, and retrievers — with paired bootstrap confidence intervals
on span-level metrics — is a genuine gap at a scale one person can execute
rigorously. It also feeds directly back into my `financial-rag-chatbot`.

**Phase.** 2 of 4 — baselines (first grid 2026-07-04: fixed-k vs
budget-matched ranking reversal; overlap ablation + truncate-rule robustness
check 2026-07-05: overlap = boundary repair, size ordering survives the rule
change — findings 6–7 in the project README).

1. **Harness** — offset-preserving chunkers, tokenization, span-level metrics,
   budget-matched retrieval protocol, dataset loaders. *(done except Chroma loader)*
2. **Baselines** — BM25 / TF-IDF / LSA retrievers over all chunker x size x overlap
   configs on SQuAD-derived long documents + Chroma eval corpora. *(current:
   BM25 grid + overlap and budget-rule ablations done; next = TF-IDF/LSA +
   dense retriever, Chroma corpora)*
3. **Ablations & analysis** — budget curves, overlap ablation *(done)*, semantic vs.
   structural chunking, per-dataset error analysis, multi-seed sampling,
   significance testing.
4. **Writeup** — README as a full research report with real tables and limitations.

**Environment constraints (recorded so results are honest).** CPU-only (4 cores,
16 GB RAM). Network access widened on 2026-07-03: HuggingFace and the tiktoken
vocab host are now reachable (both were blocked on day 1), so phase 2 adds a
small CPU-sized sentence-transformer dense retriever (e.g. all-MiniLM-L6-v2)
alongside BM25 / TF-IDF / LSA, and the BPE tokenizer robustness check becomes
real rather than hypothetical. Large dense retrievers and cross-encoder
rerankers remain out of scope on this hardware and are listed as limitations.

## Backlog (next flagships, roughly prioritized)

- **LLM-as-judge reliability audit** — measure agreement, position bias, and
  self-preference of judge prompts across seeds; needs API access.
- **Hallucination measurement in RAG answers** — span-attribution based
  faithfulness scoring; natural sequel to the chunking bench.
- **Agent tool-call reliability harness** — inject tool failures/latency and
  measure recovery behavior of agent loops; ties into `data-analysis-agent`.
- **Retriever robustness to query noise** — typos, paraphrase, and entity-swap
  perturbations vs. retrieval degradation curves.
- **Time-series foundation models vs. classical baselines** — evaluate on the
  `Bitcoin-Price-Forecasting` data with proper backtesting protocol.

## Weekly rhythm

Most days advance the flagship. One or two days a week ship a focused improvement
to `financial-rag-chatbot`, `data-analysis-agent`, or `Bitcoin-Price-Forecasting`
(evals, tests, robustness) — check `git log` first, never repeat recent work.

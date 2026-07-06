# genai-lab

[![CI](https://img.shields.io/github/actions/workflow/status/Kantamaniprakash/genai-lab/ci.yml?branch=main&label=CI)](https://github.com/Kantamaniprakash/genai-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

Hands-on experiments with current Gen AI techniques — RAG, agents, evals, fine-tuning and whatever's moving the field this week

![Budget-matched SpanRecall@400 by chunking strategy and chunk size, with 95% bootstrap CIs](rag-chunking-bench/assets/hero_spanrecall_dev-v1.1_bm25.png)

*Flagship result from [`rag-chunking-bench`](rag-chunking-bench/): once the
retrieved-token budget is held constant, smaller chunks win in every chunker
family — regenerated from committed per-question results.*

## Current flagship

**[`rag-chunking-bench`](rag-chunking-bench/)** — a token-budget-controlled
benchmark of chunking strategies for RAG retrieval, with span-level metrics
and paired bootstrap confidence intervals. See [ROADMAP.md](ROADMAP.md) for
the rationale, phases, and project backlog, and
[`rag-chunking-bench/research/NOTES.md`](rag-chunking-bench/research/NOTES.md)
for the day-by-day research log.

## Results at a glance

Measured on SQuAD dev-v1.1 reconstructed articles (48 documents, 2,400
sampled questions) with BM25, TF-IDF, and LSA retrieval, against a classic
fixed-k evaluation as the baseline protocol. Every number comes from
per-question score files checked into
[`rag-chunking-bench/results/raw/`](rag-chunking-bench/results/raw/) and
carries a 95% paired bootstrap confidence interval:

- **Budget-matched, smaller chunks win.** At a 400-token budget, 64-token
  fixed chunks beat 256-token ones by **+0.134 [+0.117, +0.152]**
  SpanRecall.
- **Fixed-k evaluation reverses the ranking.** hit@5 rises with chunk size
  (0.873 → 0.969) while budget-matched SpanRecall@400 falls (0.879 → 0.023)
  — the token-budget confound in standard chunking comparisons is real and
  large.
- **Sentence alignment adds a small significant edge** (+0.041 [+0.029,
  +0.052] at size 64, B=400); ~25% overlap pays off for fixed windows at
  tight budgets but is pure cost for sentence packing.
- **None of it is a BM25 artifact.** The size ordering, the reversal, and
  the sentence edge all hold under TF-IDF and LSA — and the chunking effect
  (+0.13–0.19 SpanRecall at B=400) outweighs the retriever effect at small
  chunk sizes (≤ 0.053) several times over.

Full tables, figures, findings 1–9, and an honest
[Limitations](rag-chunking-bench/README.md#limitations) section (lexical
retrievers only so far, single dataset/seed, regex tokenizer, CPU-only
scale) live in the [project README](rag-chunking-bench/README.md).

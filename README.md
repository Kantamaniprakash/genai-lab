# genai-lab

[![CI](https://img.shields.io/github/actions/workflow/status/Kantamaniprakash/genai-lab/ci.yml?branch=main&label=CI)](https://github.com/Kantamaniprakash/genai-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

Hands-on experiments with current Gen AI techniques — RAG, agents, evals, fine-tuning and whatever's moving the field this week

<!-- latest-start -->
## Latest from the lab

<!-- auto-generated from research/NOTES.md by scripts/sync_latest.py; do not hand-edit -->

**2026-07-16 — Day 15: the reproduction audit catches two stale tables; the flagship closes**

The plan was a verification formality; the audit earned its keep instead. In a fresh clone with a fresh interpreter and refetched data, the new `experiments/reproduce.py` manifest audit regenerated all 41 committed tables and figures from the committed raw results: 38 reproduced byte-identically, two ablation tables turned out to be stale (day 13's semantic truncate runs had leaked into the structural budget-rule section — summarizer scoped, regression-tested, re-verified bit-identical), and the hero PNG was re-rendered so every committed artifact now originates from this reproducible environment. With that and the release polish below, `rag-chunking-bench` closes complete: 26 findings, 365 tests, and a replayable byte-level reproduction audit.

[Full entry →](rag-chunking-bench/research/NOTES.md#2026-07-16--day-15-the-reproduction-audit-catches-two-stale-tables-the-flagship-closes)

**Most recent findings** ([2026-07-14 — Day 13: per-question error analysis — findings 24–26: composition explains the corpora, the loss tail splits by mechanism, overlap decomposes exactly](rag-chunking-bench/research/NOTES.md#2026-07-14--day-13-per-question-error-analysis--findings-2426-composition-explains-the-corpora-the-loss-tail-splits-by-mechanism-overlap-decomposes-exactly)):

- Finding 24 — corpus identity adds nothing beyond gold-length mix.
- Finding 25 — the loss tail is two mechanisms, the small one a ranking failure on SHORT golds.
- Finding 26 — overlap = placement + extension − redundancy tax; stitching is budget-limited.
<!-- latest-end -->

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

Full tables, figures, findings 1–23 (now spanning four retriever
families, two datasets, three sampling seeds, both budget-boundary rules,
and two token units), and an honest
[Limitations](rag-chunking-bench/README.md#limitations) section (small
CPU-sized dense encoder only, contiguous gold evidence only, CPU-only
scale) live in the [project README](rag-chunking-bench/README.md).

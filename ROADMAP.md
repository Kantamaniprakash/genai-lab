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

**Phase.** 3 of 4 complete; phase 4 (writeup) next — baselines (first grid 2026-07-04:
fixed-k vs budget-matched ranking reversal; overlap ablation + truncate-rule
robustness check 2026-07-05: overlap = boundary repair, size ordering
survives the rule change — findings 6–7; cross-retriever grid 2026-07-06:
all chunking effects transfer to TF-IDF/LSA, chunking effect > retriever
effect, retriever gap grows with chunk size — findings 8–9; multi-seed check
+ dense MiniLM grid 2026-07-07: headline claims replicate under three
independent question samples, chunking effects transfer to dense retrieval,
and past the encoder window dense retrieval degrades to prefix retrieval —
findings 10–12; Chroma long-reference grid 2026-07-08, all four retrievers:
the small-chunk advantage INVERTS at generous budgets on sentence-scale
golds, the inversion is gold-length-driven and requires a full-chunk-reading
retriever, and precision/IoU are finally informative — findings 13–15;
chroma overlap + truncate ablations and corpus jackknife 2026-07-09: overlap
gains persist across budgets on long golds and the cross-family
boundary-repair control breaks at small sizes, while the crossover survives
the budget rule and every drop-one corpus and the tight-budget small-chunk
edge turns out to be mostly a stop-rule artifact — findings 16–18; cl100k
BPE tokenizer unit 2026-07-11: every headline claim is unit-invariant, and
wiring the unit in exposed and fixed two containment-vs-overlap chunker
bugs — finding 19; semantic chunker 2026-07-12: the percentile
embedding-breakpoint chunker's matched-nominal-size wins are realized-size
drift — null where realized sizes coincide, sign-flipped on long golds at
generous budgets, no systematic ranking gains — findings 20–21;
matched-realized-size protocol 2026-07-13: at matched realized size the
semantic chunker gains nothing anywhere and its long-gold penalty
survives, while matched *means* prove insufficient — realized-size
dispersion × the stop rule manufactures ±0.5 deltas, truncate at
B ≫ chunk size is the honest regime — findings 22–23; per-question error
analysis 2026-07-14, no new runs: the per-corpus heterogeneity is
gold-length composition (leave-one-corpus-out composition test, no
significant residual anywhere), the hard-loss tail splits into
partial-coverage losses on long multi-ref golds plus a small set of
complete ranking misses on short golds, and every overlap gain decomposes
exactly into new-region placement + extension − a redundancy tax, with
stitching real only at tight budgets — findings 24–26).

1. **Harness** — offset-preserving chunkers, tokenization, span-level metrics,
   budget-matched retrieval protocol, dataset loaders. *(done — SQuAD +
   Chroma loaders, both with verbatim-verified gold spans)*
2. **Baselines** — BM25 / TF-IDF / LSA / dense retrievers over all chunker x size x
   overlap configs on SQuAD-derived long documents + Chroma eval corpora.
   *(done: all four retriever grids on both datasets; overlap, budget-rule,
   and multi-seed checks on SQuAD)*
3. **Ablations & analysis** — overlap ablation *(done: SQuAD + chroma)*,
   budget-rule check *(done: SQuAD + chroma)*, multi-seed sampling
   *(done, BM25)*, gold-length moderation *(done — finding 14)*, corpus
   jackknife *(done — finding 18)*, BPE tokenizer robustness *(done —
   finding 19)*, semantic vs. structural chunking *(done — findings
   20–21)*, matched-realized-size protocol *(done — findings 22–23)*,
   per-corpus error analysis *(done — findings 24–26; phase complete)*.
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

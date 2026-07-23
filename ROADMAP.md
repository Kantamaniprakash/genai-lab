# genai-lab Roadmap

This lab runs one flagship research project at a time, worked daily until it would
survive review by a demanding referee. Everything here is real: every number in a
writeup comes from an experiment actually run in this repo.

## Current flagship: `slm-judge-audit` — started 2026-07-17, phase 2 (baselines & main grid)

**Question.** How reliable are small open-weight LLMs (0.5B–8B, the sizes people
actually deploy for cheap large-scale evaluation) as zero-shot pairwise judges —
when you measure them white-box, at the level of verdict token probabilities
rather than sampled outputs?

**Why this, why now.** LLM-as-judge reliability is one of the most active eval
topics right now (a 2026 wave of position-bias, self-preference, and
bias-mitigation papers). The selection scan (2026-07-17) found the closest
neighbors: "Reliability without Validity" (Norman et al., arXiv:2606.19544)
audits 21 judges across agreement/consistency/bias but treats every judge as an
API black box; JudgeBoard (arXiv:2511.15958), SLMJury (arXiv:2606.07810), and
"Thinking Small" (arXiv:2509.13332) study small judges but accuracy-first. The
gap this lab can own: local open-weight judges expose the **full next-token
distribution**, so verdict preference can be measured as log-odds, position bias
as a per-item shift in log-odds under order swap (a structural model one can
*test*, not assume), debiasing-by-symmetrization can be quantified exactly,
calibration (ECE, reliability diagrams) becomes measurable, and "does the judge
add signal beyond a length heuristic?" becomes a regression question — all with
the paired-bootstrap machinery this lab built in `rag-chunking-bench`. White-box
+ small-scale + statistically careful is a genuine unclaimed corner, and it is
the only corner honestly executable on this hardware (CPU-only; single-token
verdict readout makes every judgment a prefill-only forward pass). Free-tier
hosted API limits were verified prohibitive for the alternative (judge audits
need thousands of calls), which also ruled out the backlog's API-dependent
framing of this project.

**Data.** RewardBench filtered set (Lambert et al., arXiv:2403.13787): 2,985
human-verified chosen/rejected pairs across 23 subsets in 4 categories — which
embeds the complete LLMBar meta-evaluation set (Zeng et al., ICLR 2024,
arXiv:2310.07641) as its llmbar-* subsets, giving an adversarial
instruction-following axis for free. Pinned revision + SHA256, verified at load.

**Phases.**
1. **Harness** — pinned data layer with category mapping and stratified
   sampling; judge prompt builder with order swap and single-token verdict
   readout; llama.cpp-based judge runner with logit extraction; result store.
   *(done 2026-07-18: runner + analysis core + floors, 47 tests)*
2. **Baselines & main grid** — judge scaling curve (Qwen2.5 0.5B/1.5B/3B/7B,
   Llama-3.2 1B/3B, + peers) on a stratified sample, both orderings; trivial
   baselines (always-A, longer-response, random) as floors.
   *(started 2026-07-18: Qwen2.5-0.5B and Llama-3.2-1B grids done on the
   same 600-item sample — findings 1–7: the always-A machine that
   flip-rate audits would call consistent, and the cross-family inversion
   of the flip-rate vs. true-bias ranking. 2026-07-19: readout-validity
   conditioning — the logit readout survives non-compliance, finding 8 —
   and the Qwen2.5-1.5B grid: inverse scaling of debiased accuracy,
   symmetrization backfires on bias-saturated Reasoning items where the
   emergent preference tracks length, findings 9–11. 2026-07-22: the
   value-over-length probe — every judge has signal beyond length but
   below 3B none beats a fitted one-parameter length baseline, and both
   standing mysteries are length-mediated, findings 12–14; calibration —
   symmetrization repairs it at 0.5B/1B only, finding 15; and the
   Qwen2.5-3B grid: the valley closes, sym 0.742, bias flips to B at the
   largest magnitude yet, verbosity un-learns, first judge to beat the
   length floor, findings 16–18. 2026-07-23: the additive-shift formal
   test + exact-LOO single-order correction ladder — position bias is
   never an additive constant and a fitted one-call correction substitutes
   for symmetrization at 0.5B but caps at ~half the gain at 3B, findings
   19–20; and the Llama-3.2-3B grid: both families reverse bias direction
   with scale in opposite senses, Llama-3B is a new always-A machine that
   falls below chance on adversarial Chat Hard while hitting 0.889 on
   Chat, and post-debiasing calibration is a family property, findings
   21–23. Next: Qwen2.5-7B — resumable two-session grid if the host is
   slow — then Llama-3.1-8B.)*
3. **Analysis axes** — position bias as additive log-odds shift (test the
   structural model); symmetrization debiasing gains; calibration; value over
   length baseline; rubric-prompt sensitivity; category/subset heterogeneity.
4. **Writeup** — README as a research report with real tables, figures, and
   limitations; reproduction audit in the `rag-chunking-bench` style.

## Completed flagships

### `rag-chunking-bench` — 2026-07-03 to 2026-07-16, COMPLETE

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

**Outcome.** 26 findings across baselines, five robustness axes (retriever
family, sampling seed, budget rule, tokenizer unit, corpus jackknife), the
semantic-chunker verdict under a matched-realized-size protocol, and a
closing per-question error analysis. Headlines: fixed-k and budget-matched
evaluation rank chunk sizes in opposite orders; under budget matching the
winning chunk size is set by gold-evidence length; the percentile semantic
chunker shows no boundary-quality gain at matched realized size and a real
long-gold penalty; and matched *mean* size is not a controlled comparison —
realized-size dispersion × the stop rule manufactures ±0.5 recall deltas.
365 tests; every committed table and figure (22 + 19) regenerates
byte-identically from the committed raw results in a clean environment
(`experiments/reproduce.py`, audited 2026-07-16). Full report:
`rag-chunking-bench/README.md`; day-by-day log:
`rag-chunking-bench/research/NOTES.md`.

**Phase history.** Baselines (first grid 2026-07-04:
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
4. **Writeup** — README as a full research report with real tables and
   limitations. *(done: coherence pass 2026-07-15; clean-environment
   reproduction audit, `experiments/reproduce.py` manifest + audit tooling,
   and release polish 2026-07-16 — flagship closed)*

**Environment constraints (recorded so results are honest).** CPU-only (4 cores,
16 GB RAM). Network access widened on 2026-07-03: HuggingFace and the tiktoken
vocab host are now reachable (both were blocked on day 1), so phase 2 adds a
small CPU-sized sentence-transformer dense retriever (e.g. all-MiniLM-L6-v2)
alongside BM25 / TF-IDF / LSA, and the BPE tokenizer robustness check becomes
real rather than hypothetical. Large dense retrievers and cross-encoder
rerankers remain out of scope on this hardware and are listed as limitations.

## Backlog (next flagships, roughly prioritized)

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

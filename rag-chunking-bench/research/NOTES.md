# Research log — rag-chunking-bench

Working notes, newest day last. Next-step lists are written to be executable
without re-deriving context.

---

## 2026-07-03 — Day 1: topic selection, environment survey, chunking harness

### Topic rationale (summary; full version in ../../ROADMAP.md)

Surveyed the current chunking-for-RAG literature. Verified and read abstracts
of: Chroma technical report (Smith & Troynikov 2024) — introduces token-level
IoU/recall metrics and ClusterSemanticChunker; LumberChunker (Duarte et al.,
EMNLP 2024 Findings, arXiv:2406.17526) — LLM-driven segmentation, +7.37
DCG@20 over baselines on GutenQA; Merola & Singh (ECIR 2025 KEIR workshop,
arXiv:2504.19754) — late chunking vs. contextual retrieval; BEIR
(arXiv:2104.08663); Lost in the Middle (arXiv:2307.03172). The gap I'm
attacking: comparisons at fixed k conflate chunk size with retrieved-token
volume; nobody reports budget-matched comparisons with per-question paired
uncertainty. That's the thesis of this benchmark.

### Environment survey (CRITICAL for all future sessions)

Network egress is allowlisted. Verified by direct probing today:

| Endpoint | Status | Consequence |
|---|---|---|
| pypi.org / files.pythonhosted.org | OPEN | pip installs work |
| raw.githubusercontent.com | **OPEN** | data files in public GitHub repos are fetchable |
| git protocol to public GitHub repos (ls-remote) | OPEN | cloning public repos works |
| huggingface.co | **BLOCKED (403)** | no neural embedding models, no HF datasets |
| openaipublic.blob.core.windows.net | BLOCKED | tiktoken BPE vocab unavailable → regex tokenizer |
| github.com web / codeload / *.github.io / objects.githubusercontent.com | BLOCKED | no release assets (no spaCy models), no tarballs |

Verified data endpoints (HTTP 200 today):
- SQuAD v2.0 dev (4.4 MB): `https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v2.0.json` (v1.1 files sit in the same directory)
- Chroma eval questions: `https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/main/chunking_evaluation/evaluation_framework/general_evaluation_data/questions_df.csv` (corpora live under `.../general_evaluation_data/corpora/`)

Hardware: 4 CPU cores, 15 GB RAM, no GPU. Python 3.11.15.

**Design consequences, decided today:** retrievers = BM25 + TF-IDF + LSA
(all local, deterministic); tokenizer = deterministic regex word/punct
tokenizer behind a `Tokenizer` protocol (BPE slot-in later if egress ever
allows); datasets = SQuAD-derived long docs + Chroma corpora via raw GitHub.

### Built today

- `src/tokenization.py` — `RegexWordTokenizer` (token = `\w+` run or single
  punctuation mark, offsets preserved) and `TokenIndex` (bisect-based
  token-in-range queries; keeps greedy merging O(n log n) instead of O(n²)).
- `src/chunkers.py` — `FixedTokenChunker` (window+overlap, token-aligned
  boundaries), `SentenceChunker` (regex sentence splitting + greedy packing,
  oversized sentences fall back to token windows so budget is a hard
  guarantee), `RecursiveCharacterChunker` (LangChain-semantics separator
  hierarchy with greedy merge, exact offsets throughout).
- `tests/test_chunkers.py` — 80 tests, all passing. Core invariants tested
  across a document battery (prose/messy/unicode/separator-free/random):
  exact offsets, hard token budget, full token coverage, ordered starts.

Two bugs found and fixed during testing:
1. Sentence-boundary regex missed terminators followed by closing quotes
   (`."` ). Fixed by consuming optional closing quotes/brackets after
   terminal punctuation.
2. Recursive splitter could cut through a separator straddling a piece
   boundary (an artifact possible with custom separator sets). Fixed by
   requiring the full separator to fit inside the piece.

Deliberate scope decisions: `RecursiveCharacterChunker` has no overlap knob
in v1 (overlap ablation runs on fixed/sentence chunkers); sentence splitter
is regex-based with documented failure modes (abbreviations) rather than a
model-based splitter, keeping determinism and zero downloads.

### Next steps (Day 2, in order)

1. `src/data.py`: SQuAD loader — download dev-v1.1 + dev-v2.0 JSON to
   `data/` (gitignore the payloads, keep a download script with SHA256
   checks), reconstruct one document per article title by joining paragraph
   contexts with `\n\n`, remap each answer span to article coordinates,
   deduplicate identical (question, article) pairs, drop unanswerable v2
   questions. Sanity assertion: `article[span.start:span.end] == answer_text`
   for every span.
2. `src/metrics.py`: SpanRecall/SpanPrecision/SpanIoU at budget B (definition
   in README, token sets via `TokenIndex`), plus classic Recall@k. Unit
   tests with hand-computed examples.
3. If time remains: `src/retrievers.py` BM25 (own implementation, tested
   against worked example) so Day 3 can run the first real experiment.

### Open questions

- Chroma corpora gold "references" are (excerpt, start, end) — verify their
  offsets are exact against corpus files before trusting them; if not, remap
  by string search.
- SQuAD dev-v1.1 has ~48 articles / 10.5k questions — decide per-article
  question cap so a full grid stays under ~30 min CPU.
- Budget boundary rule (stop-before-exceed) biases against large-chunk
  configs at small B; consider also reporting "truncate-final-chunk" variant
  as a robustness check.

---

## 2026-07-03 — Day 2: data pipeline, span metrics, paired bootstrap, BM25

### Environment re-survey (network policy CHANGED since day 1)

Re-probed the day-1 blocklist today; two hosts flipped to open:

| Endpoint | Day 1 | Today | Consequence |
|---|---|---|---|
| huggingface.co | 403 | **OPEN** (200, HF_TOKEN present in env) | small dense retrievers + HF datasets now feasible |
| openaipublic.blob.core.windows.net | blocked | **OPEN** (206) | tiktoken BPE vocab downloadable → real BPE robustness check |
| codeload.github.com | 403 | 403 | still no tarballs/release assets |

Design update recorded in ROADMAP + README: phase 2 adds a CPU-sized
sentence-transformer dense retriever (all-MiniLM-L6-v2, 22M params) to cover
the lexical-vs-dense axis, and the tokenizer robustness check uses real
tiktoken BPE. The regex tokenizer stays the primary unit (deterministic,
dependency-free) — this was always the design, not just a workaround.

### Built today (all of day 1's plan, including the stretch item)

- `TokenIndex.tokens_overlapping(start, end)` — gold spans need not be
  token-aligned, so metrics count tokens by character overlap, not
  containment. Chunk queries keep using containment (`tokens_in`).
- `src/data.py` — SQuAD loader. Articles rebuilt by joining paragraph
  contexts with `\n\n`; every answer span remapped to article coordinates
  and verified verbatim (mismatch = hard error, never a skip). Unanswerable
  v2 questions dropped; duplicate (article, question-text) pairs dropped;
  identical annotated spans collapsed. Gold representation:
  `gold_alternatives: tuple[tuple[GoldSpan, ...], ...]` — alternatives of
  jointly-required span sets. SQuAD = one singleton alternative per distinct
  annotation (max-over-answers); Chroma corpora later = one alternative with
  all references. One scoring path handles both. Downloads pinned by URL +
  SHA256 (`python -m src.data`); payloads gitignored.
  `sample_questions(ds, cap, seed)` seeds per-document (`f"{seed}:{doc_id}"`)
  so a document's sample is independent of which other documents are in the
  run — subset grids stay comparable.
- `src/metrics.py` — `take_until_budget` (stop-before-exceed; budget charges
  per-chunk prompt tokens, duplicates included), `span_scores` (recall /
  precision / IoU on token-index sets, union for scoring so redundant overlap
  costs budget without earning recall; independent max over alternatives),
  `hit_at_k` (interval-intersection on contiguous token ranges), and
  `paired_bootstrap` (numpy percentile CI over per-question diffs, fixed
  seed, vectorized resampling).
- `src/retrievers.py` — BM25 from scratch (Okapi, Lucene-smoothed
  non-negative idf `ln(1 + (N-df+.5)/(df+.5))`, k1=1.5, b=0.75, unique query
  terms, deterministic index tie-break). Tested against a fully
  hand-computed 3-doc example (expected scores derived independently of the
  implementation) plus property tests (b=0 kills length normalization,
  duplicate docs tie, unseen terms → identity order).
- Tests: 80 → **134 passing** (~0.5 s). Includes a real-data integration
  test (skipped gracefully when `data/` is empty).

### Real-data numbers (recorded facts, not results)

- dev-v1.1: 48 articles, **10,533** questions after dedup (10,570 − 37 dup
  texts). Doc lengths (regex tokens): min 2,751 / median 6,157 / max 16,764.
- dev-v2.0 (answerable only): 35 articles, 5,923 questions, median 4,704.

### End-to-end smoke (8 articles × 25 q/article, BM25, NOT README results)

| chunker (128 tok) | SpanRecall@200 | SpanRecall@400 | hit@5 |
|---|---|---|---|
| fixed | 0.700 | 0.847 | 0.920 |
| sentence | 0.780 | 0.880 | 0.930 |
| recursive | 0.741 | 0.884 | 0.940 |

Pipeline runs ~0.15 s per config on this slice → full grid (3 chunkers × 4
sizes × 3 overlaps × 48 articles × 50 q) is minutes on CPU, so no question
cap compromise needed. The sentence-vs-fixed gap at B=200 (+0.08 recall) is
exactly the kind of effect the benchmark exists to pin down with CIs —
promising, but smoke-scale; do not cite.

### Next steps (Day 3, in order)

1. `experiments/run_grid.py`: experiment runner — config dataclass (dataset,
   chunker, size, overlap, retriever, budgets, seed, question cap), writes
   one JSON per config into `results/raw/` with per-question scores +
   config + git-describable metadata. Deterministic; resumable (skip
   existing result files).
2. First real grid on dev-v1.1: chunkers {fixed, sentence, recursive} ×
   sizes {64, 128, 256, 512} × overlap 0, BM25, budgets {200, 400, 800,
   1600}, per-doc cap 50 seed 0 (~2,400 questions). Save raw scores.
3. `experiments/summarize.py`: aggregate raw scores → markdown table with
   means + 95% paired bootstrap CIs vs. a designated baseline (fixed-256).
4. Start populating README Experiments with the first real table.
5. If time: Chroma corpora loader (verify gold excerpt offsets against
   corpus files first — day-1 open question still stands).

### Open questions (carried + new)

- Chroma corpora reference offsets still unverified (day-1 item).
- Overlap ablation: overlap fractions {0, 12.5%, 25%} of chunk size — run
  after the size grid so the baseline table stays small.
- Budget boundary rule: also report truncate-final-chunk variant as
  robustness check (day-1 item, unresolved).
- Dense retriever: embed chunks once per (chunker, size) config — cache
  embeddings to disk to keep the grid fast; sizing pass needed (48 docs ×
  ~50-260 chunks × 384 dims ≈ fine for RAM, check disk).

---

## 2026-07-04 — Day 3: first real grid, paired CIs, figures — phase 2 opens

### Built today

- `experiments/run_grid.py` — deterministic, resumable grid runner. One
  gzipped JSON per config in `results/raw/` with per-question scores AND
  per-budget `chunks`/`tokens` spent (that last field turned out to be the
  key to interpreting two findings — always store utilization). Configs
  embed git commit (+dirty flag), library versions. Code was committed
  *before* running so result files reference a clean commit (52ac809).
- `experiments/aggregate.py` — result loading, qid-alignment check (paired
  comparisons refuse to run on mismatched question sets), mean/diff
  bootstrap CIs.
- `experiments/summarize.py` — full markdown summary with 10k-resample
  paired CIs vs. fixed-256 baseline → `results/summary_dev-v1.1_bm25.md`.
- `experiments/make_figures.py` — two README figures regenerated purely
  from `results/raw/`; CVD-validated palette; CI bands.
- Tests 134 → 150 (synthetic 3-paragraph dataset with distinctive vocab so
  BM25 behavior is predictable; determinism, resume/skip/force, alignment
  rejection, degenerate-CI sanity checks).

### The grid (12 configs × 2,400 q, ~22 s total)

Results in README + summary. Findings, with numbers I'd defend:

1. **Small chunks dominate under budget matching.** fixed-64 vs fixed-256 at
   B=400: +0.134 [+0.117, +0.152] SpanRecall. Monotone in size at every
   budget; effect shrinks with budget (n.s. by B=1600 for 64 vs 256).
2. **Metric reversal (headline).** hit@5 rises with size (fixed: .873→.969
   for 64→512) while SpanRecall@400 falls (.879→.023). The confound the
   project was built to expose, now measured with CIs. This is the figure
   to lead any writeup with (`metric_reversal_*.png`).
3. **Sentence packing > fixed windows at matched size** (paired, B=400):
   +0.041/+0.020/+0.018/+0.010 for sizes 64/128/256/512, all significant.
   Computed ad hoc — same-size pairwise diffs are NOT in summarize.py yet.
4. **Nominal size ≠ realized size.** recursive-256 "beats" fixed-256
   (+0.348 at B=200) but its chunks average 189 tokens vs 250 — it just
   operates further down the size axis. Chunk-stats table is essential
   context for any chunker comparison; papers comparing "at chunk size X"
   without realized distributions are suspect.
5. **Stop-before-exceed zeroes size>budget cells** (utilization ~0 for
   fixed-512 at B≤400). Honest protocol artifact, kept visible; truncation
   variant still owed.
6. SQuAD precision/IoU ≈ 1/|C| (gold ~3 tokens) — uninformative here;
   recall is the metric on this dataset. Precision needs long-reference
   corpora (Chroma).

### Next steps (Day 4, in order)

1. **Overlap ablation**: fixed {8, 16, 32} overlap at sizes 64/128/256
   (12.5/25/50% where valid) + sentence overlap_sentences {1, 2}. Runner
   already takes --overlaps; just run + summarize. Hypothesis: budget
   accounting (duplicates charged, union scored) makes overlap a net
   NEGATIVE under budget matching — would be a nice counter-folklore result
   if it holds.
2. **Same-size pairwise table** in summarize.py (sentence vs fixed,
   recursive vs fixed at each size) — I computed these ad hoc today; they
   belong in the generated summary.
3. **Truncate-final-chunk budget rule** as a `take_until_budget` variant +
   rerun grid under it (results to `results/raw/` with a rule field —
   NOTE: config_id must encode the rule to avoid clobbering; add it to
   GridConfig with default "stop").
4. If time: TF-IDF + LSA retrievers (scikit-learn, pin version) → first
   retriever × chunker interaction numbers.

### Open questions (carried)

- Chroma corpora reference offsets still unverified (day-1 item; needed
  before precision-focused phase).
- Dense retriever (MiniLM) embedding cache design (day-2 item).
- Multi-seed sampling (seed sensitivity of the 50/doc cap) — cheap to run
  (~22 s/grid), worth doing before the writeup phase claims robustness.

---

## 2026-07-05 — Day 4: overlap ablation + truncate rule — both hypotheses tested, one refuted

### Built today

- `take_until_budget(..., rule="stop"|"truncate")` — truncate-final-chunk
  cuts the budget-straddling chunk token-aligned to exactly fill B, using
  `TokenIndex.text` to keep `document[start:end] == chunk.text` true for the
  partial chunk. Only the first overflowing chunk is cut; scoring needs no
  changes because truncated chunks live in document coordinates.
- `GridConfig.budget_rule` (default "stop", **omitted from config_id for the
  default** so day-3 result filenames stay valid; non-default rules append
  `_truncate`). `load_raw` backfills `budget_rule="stop"` into old payloads
  and grew `budget_rule`/`overlap` filters; labels append `/truncate`.
- `experiments/summarize_ablations.py` — renders overlap ablation (each
  config paired vs. its own zero-overlap control, incl. a hit@5 column) and
  budget-rule sections → `results/summary_dev-v1.1_bm25_ablations.md`.
- Same-size pairwise ΔSpanRecall table now generated in `summarize.py`
  (day-3 ad hoc numbers reproduced exactly).
- Two new figures in `make_figures.py` (overlap 2×2 paired-delta panels with
  deterministic x-jitter for error-bar legibility; budget-rule 3-panel
  stop-vs-truncate curves). `make_figures`/`summarize` now filter to the
  baseline grid explicitly so ablation files don't pollute them.
- Tests 150 → **163** (truncate token-alignment/exact-spend/empty-remainder,
  config_id rule encoding, load_raw filters + old-payload backfill, pairwise
  rows, ablation renderer incl. no-data SystemExit).

### Ran (27 new configs, ~60 s total, code committed first at ecbb1b6)

1. Overlap grid: fixed {64,128,256} × overlap {12.5%,25%,50% of size},
   sentence {64,128,256} × {1,2} sentences (15 configs, stop rule).
2. Truncate rerun of the 12 baseline o0 configs.

### Findings (all in README §6–7 with CIs)

1. **Day-3 hypothesis ("overlap is net negative under budget matching")
   REFUTED for fixed windows, upheld for sentence packing.** Fixed + ~25%
   overlap is significantly positive at tight budgets (fixed-64/o16: +0.046
   [+0.033,+0.059] @B=200, +0.036 @B=400; 25% column significant for all
   three sizes @B=400). Gains fade with budget; 50% overlap at size 256
   turns significantly negative by B=1600 (−0.013) and hurts hit@5 (−0.014).
   Sentence overlap: null-to-negative at practical budgets.
2. **Mechanism: overlap = boundary repair.** Sentence-64 at zero overlap is
   statistically indistinguishable from fixed-64 at 25% overlap (+0.007
   [−0.004,+0.019] @B=200, computed ad hoc — worth adding to a generated
   table later). Boundary-aware packing gets overlap's benefit for free.
3. **Truncate rule: headline size effect is NOT a stop-rule artifact.**
   Utilization 1.00 everywhere; truncate−stop deltas mechanically ≥ 0, big
   only in artifact cells (fixed-256@B=200 +0.589), ≤ +0.075 elsewhere.
   Under truncate at B=200: fixed 0.819/0.770/0.601/0.299 for 64→512;
   fixed-64 > fixed-256 by +0.218 [+0.198,+0.239]. Sentence>fixed same-size
   also survives (+0.040 @64/B=400). Both headline claims are properties of
   chunking, not the boundary convention.

### Next steps (Day 5, in order)

1. **TF-IDF + LSA retrievers** (`src/retrievers.py`, scikit-learn — pin in
   requirements the day it lands): TF-IDF cosine over chunk term vectors;
   LSA = TruncatedSVD(k≈128) over TF-IDF, both deterministic
   (`random_state=0`, note TruncatedSVD randomized solver → check
   determinism across runs, else use arpack). Add to RETRIEVERS in
   run_grid, hand-computed unit tests like BM25's.
2. Run baseline 12-config grid × {tfidf, lsa} → first retriever × chunker
   interaction numbers; extend summarize to render per-retriever summaries
   (already parameterized by --retriever, should just work).
3. If time: figure comparing size effect across retrievers at B=400
   (does "small chunks win" transfer from BM25 to TF-IDF/LSA?).
4. Multi-seed sampling check (seeds 1,2 for the 12-config BM25 grid) is
   still owed before writeup-phase robustness claims; cheap, slot it in
   whenever a day has slack.

### Open questions (carried + new)

- Chroma corpora reference offsets still unverified (day-1 item).
- Dense retriever (MiniLM) embedding cache design (day-2 item).
- Should the "sentence-o0 ≈ fixed+25%-overlap" cross-family comparison get
  its own generated table? (Currently an ad hoc number quoted in README §6.)
- Overlap ablation used the stop rule; if a reviewer asks, the truncate ×
  overlap cross is one `run_grid` invocation away (config space supports it).

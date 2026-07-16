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

---

## 2026-07-06 — Day 5: TF-IDF + LSA — retriever × chunker interaction, findings 8–9

### Repo state note

Main moved since day 4: Prakash merged a "Tier 2" PR (uv migration with
committed lockfile, hero figure + `make_hero_figure`, CI now runs
`uv sync --locked` + non-blocking ruff). Consequence for future sessions:
**any new dependency must go through `uv add` (or edit pyproject + `uv lock`)
so `uv.lock` stays in sync, or CI breaks.** `uv sync` needed several retries
today (PyPI download timeouts; `UV_HTTP_TIMEOUT=300` + retry loop worked).
Ruff runs with `--exit-zero`; main had 16 pre-existing lint errors — I fixed
the ones my diff would have added, left the rest (not my scope today).

### Built today (day-4 plan items 1–3, plus rank instrumentation)

- `TfidfRetriever` + `LSARetriever` (scikit-learn 1.9.0, pinned in
  requirements + pyproject + lock). Both share `query_terms` with BM25 via
  `TfidfVectorizer(analyzer=query_terms)` — retrievers differ ONLY in
  scoring, which is what makes the cross-retriever deltas clean. TF-IDF
  verified against a hand-computed cosine example (sklearn conventions:
  smooth idf ln((1+N)/(1+df))+1, raw tf, l2 rows). LSA: TruncatedSVD over
  TF-IDF, `algorithm="arpack"`, `random_state=0` — verified bit-identical
  across fits (day-4 determinism worry resolved; no need for the randomized
  solver). Empty-vocabulary corpora (pure punctuation) degrade to all-zero
  scores like BM25 instead of raising.
- **LSA rank capping was the key design issue.** Per-document indexes cap
  the SVD rank at n_chunks − 1: with the day-4 suggestion k≈128, LSA would
  have been (nearly) full-rank ≈ TF-IDF almost everywhere — an uninformative
  retriever. Chose n_components=64 and instrumented `run_config` to record
  realized per-document ranks (`retriever_stats` in raw payloads;
  `RunResult.retriever_stats`, backfills None for old files). The rank table
  is in the cross-retriever summary and it mattered for interpretation
  (finding 9).
- `experiments/summarize_retrievers.py` — per-budget side-by-side means,
  paired Δ vs bm25 per challenger, hit@5, LSA rank table →
  `results/summary_dev-v1.1_retrievers.md`. Refuses mismatched grids.
- `fig_retriever_comparison` in make_figures (3 family panels, retriever
  lines, CI bands) → `results/figures/retriever_comparison_dev-v1.1.png`.
  Existing four figures reproduced **bit-identically** after regeneration —
  end-to-end determinism holds.
- Tests 163 → **193** (hand-computed TF-IDF cosines; LSA row-space
  equivalence with TF-IDF via a rank-deficient corpus, latent bridging on a
  two-topic corpus, determinism, rank caps, degenerate corpora; grid runs
  for all three retrievers; retriever_stats roundtrip; renderer error paths).

### Ran (24 configs: baseline 12 × {tfidf, lsa}, ~90 s; code committed first at 5cf7650)

### Findings (README §8–9, cross-retriever summary has full tables)

1. **Everything transfers.** Size ordering (TF-IDF @B400: .868/.846/.679/.019
   for fixed 64→512; LSA .848/.835/.667/.019), sentence>fixed at matched
   size (+0.042 tfidf / +0.032 lsa @64,B400), and the hit@5-vs-SpanRecall
   reversal all hold under all three retrievers. Size effect within
   TF-IDF/LSA (64 vs 256 @B400: +0.189/+0.182) is LARGER than BM25's
   (+0.134) because weaker retrievers degrade more inside big chunks.
2. **Chunking effect > retriever effect.** Largest retriever gap at any
   size ≤128 config: 0.053; size effect at practical budgets: 0.13–0.19.
3. **BM25 ≥ TF-IDF ≥ LSA nearly everywhere** (paired: 43/48 and 45/48 cells
   significant vs bm25), and the retriever gap GROWS with chunk size
   (≤0.019 at size 64; 0.065–0.077 at 256–512) — tf saturation + length
   normalization matter inside long chunks. Retriever comparisons run at
   large chunk sizes overstate retriever differences.
4. **LSA never helps here.** Loses most where the k=64 cap binds (size 64:
   35/48 docs component-bounded), converges toward TF-IDF where data-bounded
   (size 512: 48/48, median rank 12). Within-document retrieval with
   questions written over the document's own vocabulary gives topical
   smoothing no upside. Kept honest in README limitations: this regime also
   caps what LSA could ever show; cross-corpus retrieval might differ.

### Next steps (Day 6, in order)

1. **Multi-seed robustness check** (owed since day 3, cheap): rerun the
   12-config BM25 baseline grid with `--seed 1` and `--seed 2` (~25 s each;
   config_id already encodes seed so no clobbering). Verify the headline
   CIs (fixed-64 vs fixed-256 @B400; sentence-64 vs fixed-64 @B400) are
   stable across seeds. Report as a short README robustness note + NOTES
   table. Consider whether summarize needs a --seed flag (load_raw currently
   has no seed filter — check before running; add one if needed).
2. **Dense retriever (MiniLM)**: `uv add sentence-transformers` (watch the
   lockfile + CI; torch CPU wheels are big — check disk), implement
   `DenseRetriever` with per-(chunker,size) embedding cache under a
   gitignored `cache/` dir, batch-encode chunks once per config. all-MiniLM-
   L6-v2, 22M params, CPU. Expect minutes-per-config, not seconds — time one
   config before committing to the grid. This completes the lexical-vs-dense
   axis and is the most-cited gap in the current writeup.
3. If dense runs land: extend summarize_retrievers/figure with the dense
   column (both already iterate over retriever lists; RETRIEVER_STYLES needs
   a 4th validated color+marker).
4. Chroma corpora loader remains the gateway to precision/IoU results
   (SQuAD's 3-token golds make precision uninformative — day-3 finding 6).

### Open questions (carried)

- Chroma corpora reference offsets still unverified (day-1 item).
- Cross-family overlap table (sentence-o0 vs fixed+25%) still ad hoc.
- Truncate × overlap cross unrun (one invocation away if a reviewer asks).
- Seed sensitivity: addressed for BM25 tomorrow; decide whether tfidf/lsa
  need it too or whether BM25 stability generalizes (argue, don't assume).

---

## 2026-07-07 — Day 6: multi-seed check + dense MiniLM — findings 10–12, phase 2 retrievers complete

### Built today (committed first at 1b7cc10; figure fn added after, see below)

- **Optional `dense` dependency group** (pyproject): sentence-transformers
  5.6.0 + torch 2.12.1+cpu, with torch pinned to the CPU-only index via
  `[tool.uv.sources]` — the default PyPI resolution would have dragged in
  multi-GB CUDA wheels. CI installs only default groups, so it never
  downloads torch; `uv lock --check` verified the lockfile stays valid for
  the default path. Dense tests guard with `importorskip`.
- `src/dense.py` — `SentenceTransformerEncoder` (process-wide, lazy model
  load, in-memory memoization keyed by text; L2-normalized float32 output)
  + `DenseRetriever` (cosine, same tie-break convention as the lexical
  retrievers). Decided AGAINST the day-5 idea of a disk embedding cache:
  the runner is already resumable per config and memoization makes each
  distinct text cost one forward pass per process — a disk cache would be
  complexity without a failure mode it protects against.
- **Truncation-exposure instrumentation** — the design decision that paid
  off most today (same lesson as day 5's LSA rank table: instrument the
  bottleneck, don't infer it). MiniLM reads ≤256 wordpieces; `fit` counts
  chunks over the window and `run_config` persists per-config exposure +
  model + torch/sentence-transformers versions (`_optional_versions`, keyed
  on sys.modules so lexical runs never import torch).
- `load_raw(..., seed=)` filter + `--seed` threaded through every
  summarizer/figure script (without this, seed-1/2 files would have broken
  every existing qid-alignment check), `experiments/summarize_seeds.py`
  (per-seed means + within-seed paired deltas; refuses cross-seed pairing),
  `fig_dense_window` in make_figures. Tests 193 → **213**.

### Ran (36 new configs: BM25 baseline × seeds {1,2}, dense × 12; ~9 min for dense)

Encoding throughput on 4 CPU cores: ~150 s for the 4,747-chunk fixed-64
config, 17–45 s for the rest (memoized queries amortize across configs).

### Findings (README §10–12)

1. **Finding 10 — every headline claim replicates under 3 independent
   question samples.** Per-config SpanRecall@400 spread ≤ 0.013 across
   seeds; fixed-64−fixed-256 = +0.134/+0.119/+0.123, sentence-64−fixed-64 =
   +0.041/+0.049/+0.047, all CIs exclude 0 under every seed. Seed is not a
   hidden degree of freedom.
2. **Finding 11 — chunking effects transfer to dense retrieval.** Size
   ordering, sentence>fixed (+0.054 @64/B400), and the metric reversal all
   hold under MiniLM. Within-dense size effect (+0.281, fixed-64 vs 256
   @B400) is 2× BM25's — finding 9's "weaker retrievers degrade more in big
   chunks," continued. At size 64 the dense-vs-BM25 gap never exceeds 0.050
   and is n.s. by B=1600. Chunking > retriever choice, now across four
   retriever families.
3. **Finding 12 — past the encoder window, dense retrieval is prefix
   retrieval.** Exposure: 0% at sizes 64–128; 25/59/96% at nominal 256 for
   recursive/sentence/fixed (realized-size differences, finding 4, now with
   consequences); 96–97% at 512. Dense−BM25 @B800 jumps from −0.005…−0.045
   (untruncated sizes) to −0.299 (fixed-512) — 2.6× LSA's worst gap.
   recursive-256 (25% exposed) loses about half of fixed-256's gap. Honest
   caveat kept in README: exposure alone doesn't order the mid cells
   (sentence-256 @59% ≈ fixed-256 @96%), so realized-length distributions
   matter beyond truncated-or-not. Sharpest signature: dense hit@5 is
   non-monotone (.842→.874→.807 for fixed 64→256→512) — the only retriever
   whose fixed-k curve turns down. `dense_window_dev-v1.1.png` is the
   figure for this.
4. SQuAD's lexical-overlap regime favors BM25, so dense *levels* are not a
   verdict on dense retrieval (BEIR caveat recorded in limitations); the
   within-retriever chunking *effects* are the claims.

### Process notes

- The four pre-existing figures reproduced bit-identically after adding the
  dense line to `retriever_comparison` — end-to-end determinism still holds
  for the lexical stack. Dense determinism is per-environment only
  (torch/BLAS); versions are recorded in every dense result payload and the
  README reproducibility note now says so explicitly.
- `fig_dense_window` label collisions took two iterations: horizontal
  stagger (fixed left, sentence right, recursive above) + suppressing 0%
  labels was the fix; family-colored labels carry the association.

### Next steps (Day 7, in order)

1. **Chroma corpora loader** (`src/data.py` or `src/data_chroma.py`) — the
   long-standing gateway to meaningful SpanPrecision/IoU (SQuAD's ~3-token
   golds cap what precision can say — day-3 finding 6). FIRST verify gold
   excerpt (start, end) offsets against the corpus files (day-1 open
   question); if offsets are inexact, remap by string search and record the
   correction rate. Gold representation: one alternative containing all
   references (jointly-required), per the day-2 design.
2. Run the baseline grid (BM25 + dense if time) on at least 2 Chroma
   corpora; first precision/IoU tables + check whether "small chunks win"
   survives long-reference evidence (the honest risk to the headline:
   multi-sentence golds may reward bigger chunks).
3. If time: extend `summarize_seeds` HEADLINE_PAIRS with a dense pair or
   run seeds 1–2 for dense (~18 min) to close the "seed stability shown for
   BM25 only" caveat in limitations.

### Open questions (carried + new)

- Chroma reference offsets (day-1) — now the top item, see next steps.
- Cross-family overlap table (sentence-o0 vs fixed+25%) still ad hoc.
- Truncate × overlap cross unrun (one invocation away).
- Does the truncation cliff move with a longer-window encoder? A 512-window
  CPU-feasible model (e.g. multi-qa-MiniLM variants are still 512-capable?
  verify) would turn finding 12 into a controlled window ablation —
  candidate for a later day; record as backlog, don't chase now.

---

## 2026-07-08 — Day 7: Chroma corpora — the crossover, findings 13–15, precision earns its keep

### The day-1 open question, resolved in one probe

All 790 gold references across the five Chroma corpora have EXACT
(start, end) offsets against the corpus files — verified before writing any
code (0 mismatches, 0 remappings needed, no CRLF/BOM traps). The loader
therefore verifies verbatim at load and hard-errors on mismatch, same
contract as the SQuAD loader. Files pinned by SHA256 (checksum mismatch =
gold offsets invalid too, since they're byte positions). CSV is clean:
no duplicate (corpus, question) pairs, no degenerate refs, no overlapping
refs within a question.

### Built (loader committed at 3955e85 BEFORE any grid ran)

- `load_chroma` in `src/data.py`: five corpora as five documents, qids
  `<corpus>:<n>` in CSV row order, all of a question's references in ONE
  gold alternative (jointly required — the day-2 gold-semantics design paid
  off: zero metric changes needed). `download_chroma` + `python -m src.data`
  prints corpus stats. Grid runner takes `--dataset chroma`; ran with
  `--per-doc-cap 150` (> max 144 questions/corpus, so no sampling — seed
  moot).
- `experiments/summarize_chroma.py`: per-corpus paired deltas + moderation
  splits (gold-length terciles via shared `gold_terciles`, reference count).
  Needs `data/chroma` (gold lengths recomputed from text; refuses to run
  without). Output: `results/summary_chroma_bm25_moderation.md`.
- `fig_gold_length_crossover` in make_figures: SQuAD-vs-Chroma-vs-dense
  delta curves + tercile split. B=200 EXCLUDED deliberately: fixed-256
  retrieves nothing there under stop (finding 5), so the ~+0.6 artifact
  delta would compress the crossover region the figure exists to show.
- Fixed two latent make_figures bugs surfaced by the second dataset:
  hardcoded "2,400 questions" ylabels, and suptitles asserting
  SQuAD-specific verdicts ("smaller chunks dominate at every budget" — FALSE
  on chroma). Titles are now dataset-neutral; interpretation lives in README
  captions. Only those two dev-v1.1 PNGs changed; the other four reproduced
  bit-identically (determinism check still passing).
- Tests 213 → **221** (loader: joint-alternative semantics, qid scheme,
  mismatch/unknown-corpus errors, real-data integration; summarizer:
  sections/groups render, out-of-sync qids exit).

### Ran (48 configs: baseline 12 × {bm25, tfidf, lsa, dense}, all 472 questions)

BM25 ~21 s, tfidf+lsa ~40 s, dense ~5.5 min (finance = 145k tokens; MiniLM
warning "269 > 256" is the truncation instrumentation's territory, expected).

### Findings (README §13–15, the day the honest risk got measured)

1. **Finding 13 — the crossover.** fixed-64 − fixed-256 under BM25:
   +0.133 @B400 → **−0.033** @B800 → **−0.040** @B1600. The SQuAD "small
   chunks never lose" claim is a SHORT-GOLD property, exactly as the day-6
   risk note feared — and that's the interesting result. Direction
   consistent in all 5 corpora. Best config at B=1600 is sentence-256
   (0.938). Fixed-k hit@5 still rises with size (reversal replicates).
2. **Finding 14 — mechanism.** Tercile gradient at B=400
   (+0.192/+0.112/+0.092) and B=1600 (n.s./n.s./−0.085); 2+-ref questions
   drive the flip (−0.069 vs −0.022 n.s.). Practical rule now in README:
   optimal chunk size scales with gold-evidence length.
3. **Finding 15 — window × gold-length interaction.** Under dense, NO
   crossover (fixed-64 ahead at every budget, +0.047 @B1600) because 99.5%
   of fixed-256 chunks exceed MiniLM's window — prefix retrieval can't
   harvest long-gold benefits. Also: precision/IoU informative at last
   (sentence-128 peaks @B200: P=0.193, IoU=0.177); sentence-vs-fixed at 256
   is +0.083 @B400, ~4× the SQuAD gap (cutting mid-sentence now cuts gold).

### Process notes

- Later configs in a multi-config invocation record `+dirty` because earlier
  configs' untracked result files land in the working tree — same benign
  artifact as the day-4/6 committed files (verified: dev-v1.1 seed-2 and
  dense files carry it too). The actual code state for every chroma run was
  clean 3955e85. Improvement for a slack day: scope run_metadata's dirty
  check to computation inputs (`git status --porcelain -- src experiments
  pyproject.toml uv.lock requirements.txt`) so result-file accumulation
  stops tripping it. Don't rewrite existing raw files over this.
- `--per-doc-cap 150` reads oddly in chroma config ids (`cap150`); it's a
  no-op cap, documented in the summary header ("no sampling").

### Next steps (Day 8, in order)

1. **Overlap ablation on chroma** (one invocation: fixed {64,128,256} ×
   {12.5/25/50%}, sentence × {1,2} — runner supports it). Hypothesis worth
   testing: overlap's boundary-repair value should be LARGER on chroma
   (long golds straddle boundaries more often); summarize_ablations already
   parameterized by --dataset. Check whether the 25% recommendation from
   finding 6 changes.
2. **Truncate-rule robustness on chroma** (12 configs, `--budget-rule
   truncate`): does the crossover survive full budget utilization? (The
   stop-rule artifact only affects B=200, which findings 13–15 exclude, so
   expect yes — but the check is cheap and closes the loop like finding 7
   did.)
3. If both land: extend `summarize_seeds`-style robustness thinking to
   chroma is NOT applicable (no sampling); instead consider a
   per-corpus-jackknife of finding 13 (drop each corpus, recompute pooled
   delta) as the analogous stability check — decide whether it merits code
   or an ad hoc NOTES table.
4. Backlog (phase 3): BPE tokenizer robustness check (tiktoken reachable
   since day 2); 512-window encoder ablation; semantic chunker (embedding
   breakpoints) now that the dense stack exists.

### Open questions (carried + new)

- Cross-family overlap table (sentence-o0 vs fixed+25%) still ad hoc —
  chroma overlap run (next step 1) is the moment to generalize it.
- Truncate × overlap cross unrun.
- Scoped dirty-check improvement (process note above).
- chatlogs is the corpus where fixed-64 does WORST at B=1600 relative to
  its own median gold length (59 tokens, longest of the five) — consistent
  with finding 14, but a per-corpus × tercile interaction table would nail
  it if a reviewer asks.

---

## 2026-07-09 — Day 8: chroma overlap + truncate ablations, corpus jackknife — findings 16–18

### Built (all committed BEFORE the grids ran, per house rule)

- **Scoped dirty check** (day-7 process note, closed): `run_metadata` now
  runs `git status --porcelain -- src experiments pyproject.toml uv.lock
  requirements.txt`, so result files accumulating mid-invocation no longer
  stamp later configs `+dirty`. Verified live: all 27 of today's raw files
  record clean `31eac72`. Tests cover clean / output-only / input-modified.
- **Cross-family control table generated** (carried since day 4):
  `summarize_ablations` renders sentence-o0 vs fixed+(size//4) paired deltas
  wherever both runs exist. Regenerated dev-v1.1 summary reproduces the
  day-4 ad hoc number exactly (+0.007 [−0.004, +0.019] @64/B200).
- **Corpus jackknife section** in `summarize_chroma`: pooled challenger −
  baseline delta recomputed with each corpus dropped (the corpus-level
  analogue of the seed check; chroma has no sampling seed to vary).
- Figure-title neutralization, round 2 (same class as day 7): overlap and
  budget-rule suptitles asserted SQuAD verdicts ("for sentence packing it
  is mostly cost" — FALSE on chroma). Now dataset-neutral; captions carry
  interpretation. Only those two dev-v1.1 PNGs changed; all other PNGs
  reproduced bit-identically. Tests 221 → **225** (full suite run with the
  dense group installed).

### Ran (27 configs, ~70 s: overlap 15 + truncate 12, chroma, BM25, stop-rule controls already on disk)

### Findings (README §16–18)

1. **Finding 16 — overlap gains persist across budgets on long golds and
   extend to sentence packing.** fixed-64/o32: +0.048 @B400 → +0.024
   @B1600, still significant (SQuAD gains faded to ~0/negative by B1600);
   fixed-256/o64 +0.059 @B400 ≈ 3× SQuAD; sentence-128/o2 +0.044 @B400 /
   +0.035 @B800 where SQuAD had significant NEGATIVES (−0.010/−0.011) —
   clean sign flip. hit@5 flips the other way: no chroma overlap cell
   improves it (SQuAD fixed-64/o16 gained +0.033). Overlap on long golds
   buys more of each gold once found, not better rankings.
2. **Finding 17 — the boundary-repair reading of overlap has a regime
   boundary.** Cross-family control on chroma: sentence-o0 loses to
   fixed+25% at sizes 64–128 (point-negative 7/8 cells, significant 4,
   worst −0.053 @128/B400); parity-or-better returns at 256 (+0.030 @B200,
   +0.022 @B1600). Interpretation: when gold + context can't fit any single
   window, staggered windows STITCH evidence around the lexical match —
   packing can't replicate that. Caveat recorded: sentence realized mean 51
   vs nominal 64 (finding-4 confound; part of the deficit is operating
   smaller where smaller is penalized).
3. **Finding 18 — crossover robust; tight-budget edge was the stop rule.**
   Under truncate: fixed-64 − fixed-256 = +0.171 @B200 (was +0.590 under
   stop), +0.030 n.s. @B400 (was +0.133*), **−0.041* @B800, −0.047*
   @B1600** — inversion survives, slightly larger. Contrast SQuAD finding 7
   (+0.218 @B200 under truncate): short golds → small chunks genuinely
   better; long golds → the tight-budget advantage was mostly protocol.
   sentence-256 best at B≥800 under both rules (+0.068 over fixed-64 @B800
   truncate). Jackknife: B=1600 inversion significant under ALL five
   drop-one estimates (−0.030…−0.049); B=800 negative in all five but
   grazes zero in four (upper bounds +0.000…+0.007; n drops to ~330–420).
   Cite the B=1600 cell.

### Process notes

- `uv sync --group dense` (torch CPU) needed for the full 225-test count;
  CI still installs default groups only — fine, dense tests importorskip.
- The ablation summarizer's `render_rule_section` signature line is the one
  pre-existing E501 in my files; left untouched (not my diff's).

### Next steps (Day 9)

**Day 9 should be a SIDE-REPO day** — days 1–8 were all flagship; the
weekly rhythm owes 1–2 improvement days to the other repos. Best candidate:
**financial-rag-chatbot**, because the flagship now has directly applicable
results. Check `git log --oneline -15` there first, then pick ONE focused
change, e.g.: (a) inspect its chunking defaults and align them with
findings 6/14/16 (sentence-aware packing, size matched to expected evidence
length, ~25% overlap only if fixed windows), citing rag-chunking-bench in
the README; or (b) add a small retrieval-quality eval script to that repo
(its own corpus, hit@k + a budget-matched recall metric) so future chunking
changes there are measurable. Keep the diff focused; do not port the whole
bench.

Day 10 (flagship) queue, in order:
1. **BPE tokenizer robustness check** (phase-3 backlog, cheap and
   well-scoped): tiktoken vocab is reachable (day-2 probe). Add a
   `TiktokenTokenizer` behind the existing `Tokenizer` protocol, rerun the
   12-config BM25 baseline on dev-v1.1 with budgets in BPE tokens
   (config_id must encode the tokenizer — check GridConfig; add a field
   with a default that keeps old ids stable, like budget_rule), verify the
   size ordering and sentence>fixed claims are unit-invariant.
2. Semantic chunker (embedding-breakpoint) using the dense stack — the last
   chunker family the related work compares.
3. Per-corpus × tercile interaction table (day-7 open question) if a slack
   hour remains.

### Open questions (carried + new)

- Truncate × overlap cross still unrun (one invocation away, both datasets).
- Does the chroma overlap gain concentrate in the long-gold tercile the way
  the size effect does (finding 14)? A tercile split of the OVERLAP delta
  would close the finding-16 mechanism the same way finding 14 closed
  finding 13's. Candidate for a phase-3 slack slot.
- 512-window encoder ablation (day-6 backlog) still open.
- chatlogs per-corpus × tercile table (day-7) still open.

---

## 2026-07-10 — Day 9: SIDE-REPO DAY — findings applied to financial-rag-chatbot

First transfer of bench results into production code, per the day-8 plan
(option a). Commit `edbb087` on financial-rag-chatbot.

### What shipped there

- `chunking.py`: sentence-aware chunker with a HARD token budget counted in
  cl100k_base (the tokenizer of text-embedding-3-small) — replacing
  RecursiveCharacterTextSplitter(1000 chars, 200 overlap). Ported the bench's
  boundary regex (incl. the day-1 closing-quote fix) with one PDF-specific
  adaptation: single `\n` is NOT a boundary (PyMuPDF hard-wraps lines
  mid-sentence; the bench's `\n+` rule would have put chunk boundaries inside
  sentences — the exact failure sentence packing exists to prevent). Blank
  lines remain boundaries. Budget checked on the actual joined text, not
  summed per-sentence counts (BPE merges across joins), and the token-window
  fallback re-checks decoded windows (decode→re-encode does not always
  round-trip counts).
- Defaults from the bench, each traceable to a finding: 256-token sentence
  packing (chroma: sentence-256 top config at B=800/1600 — findings 13/18;
  app budget is k=5 × ≤256 ≈ 1,000–1,300 tokens = that regime), overlap 0
  (sentence-256 overlap deltas n.s. −0.014…+0.010, index +21–44% — day-8
  ablation table), sentence-vs-fixed +0.083 @256/B400 (finding 15).
- 15 invariant tests (budget, coverage, boundary alignment, overlap
  advancement guarantee, metadata, determinism), runnable standalone or via
  pytest — they run in that repo's CI. Verified end-to-end with a real
  PyMuPDF-generated 3-page PDF: 9 chunks, max 251 tokens, every chunk ends
  at a sentence boundary despite hard line wraps.
- README section "Chunking configuration (and why)" citing the bench with
  honest caveats: bench budgets are regex word tokens not BPE; retrievers
  were BM25/TF-IDF/LSA/MiniLM, not text-embedding-3-small (8K window =
  full-chunk-reading regime, where the findings held); retrieval metrics,
  not end-to-end answer quality. CHANGELOG [Unreleased] entry added.

### Repo state note (financial-rag-chatbot — matters for future sessions)

Main had moved since the 07-03 clone: Prakash merged a security migration to
the **LangChain 0.3 line** (langchain_chroma, langchain_core.documents,
pinned CVE-driven versions), **uv + pyproject.toml + committed uv.lock**,
CI (uv sync --locked, ruff --exit-zero, pytest on 3.10–3.12), CHANGELOG,
MIT license, and an HF Spaces live demo. Consequences: dependency changes
must keep uv.lock in sync; tests run in CI so they must not need API keys;
`tiktoken==0.13.0` was already pinned (no packaging change needed for the
chunker). My commit was rebased onto that; pre-existing unused imports in
app.py (os, Path, tempfile, PyMuPDFLoader) left untouched — not my diff.

### What this does NOT claim (and the honest gap)

The bench measured retrieval span-recall with its own retrievers; I could
not re-run the grid with OpenAI embeddings (no OPENAI_API_KEY in this
environment), so the app change is benchmark-motivated, not end-to-end
verified. That repo's eval_harness.py (hit@k/MRR + LLM-judge) is the tool
to close the loop the day an API key is available — its fixture chunks are
pre-defined, so it currently measures the retriever+generator, not the new
chunker; extending it to chunk a fixture document would make the chunking
change measurable there.

### Next steps (Day 10 — FLAGSHIP, queue unchanged from day 8)

1. **BPE tokenizer robustness check**: `TiktokenTokenizer` behind the
   `Tokenizer` protocol; rerun the 12-config BM25 dev-v1.1 baseline with
   budgets in BPE tokens; verify size ordering + sentence>fixed are
   unit-invariant. GridConfig needs a tokenizer field whose default keeps
   old config_ids stable (same pattern as budget_rule on day 4). Extra
   motivation now: today's README caveat in financial-rag-chatbot
   ("regex word tokens vs cl100k BPE") cites this as unverified — the
   check closes it.
2. Semantic chunker (embedding-breakpoint) using the dense stack.
3. Per-corpus × tercile interaction table if slack remains.

---

## 2026-07-11 — Day 10: BPE tokenizer robustness — finding 19, and two chunker bugs the new unit exposed

### Built (committed d57fd44 BEFORE the grid ran, per house rule)

- `TiktokenTokenizer` (cl100k_base) behind the `Tokenizer` protocol.
  tiktoken tokenizes UTF-8 bytes, so spans are reconstructed by mapping each
  token's byte range back to character offsets: a character straddling a
  token boundary belongs to the token where its byte sequence *ends*; a
  token lying entirely inside one multi-byte character gets an empty span
  (still counts toward budgets — the generator pays for it). Special-token
  text (`<|endoftext|>` in a document) is encoded as data, never as a
  control token. ASCII fast path skips the byte→char table.
- Tokenizer axis through the stack, all following the day-4 budget_rule
  pattern: `GridConfig.tokenizer` defaults to `"regex"` and is omitted from
  config_id (old filenames stay valid; non-default appends `_cl100k`),
  `make_tokenizer`, chunker factory threading, `--tokenizer` flag, tiktoken
  version recorded via `_optional_versions`. One deliberate asymmetry:
  `load_raw`'s tokenizer filter defaults CLOSED (`"regex"`), unlike every
  other filter — cross-unit runs share question ids with the primary grid,
  so `check_aligned` cannot catch accidental mixing; the scores would align
  mechanically and mean nothing. Summarizers needed zero changes.
- **The day's real engineering lesson: BPE tokens straddle exactly the
  boundaries that boundary-respecting chunkers cut at** (a token carries
  its leading whitespace). This exposed two containment-vs-overlap bugs in
  `RecursiveCharacterChunker`: a single-word piece cut after a separator
  contains no *complete* BPE token, so `_split`/`_merge` silently dropped
  the whole word, and `_trimmed` cut merged pieces past their first word.
  All three sites now use `tokens_overlapping` — provably identical under
  the regex tokenizer (separator cuts are whitespace cuts and regex tokens
  never contain whitespace, so every merge range is token-aligned).
  Verified empirically before running anything: three regex-unit configs
  (incl. both recursive sizes' neighbors) re-ran bit-for-bit against the
  committed raw files, and all eight existing PNGs regenerate
  bit-identically. SentenceChunker needed no fix — its coverage guarantee
  is restated (overlap, not containment, for the chunk-leading token) and
  documented in the class docstring.
- `experiments/summarize_tokenizers.py`: unit-conversion table recovered
  from stored chunk_stats (no re-tokenization needed at summary time),
  side-by-side levels, adjacent-size paired steps, headline deltas per
  unit, same-size tables, hit@5. First draft had a strict mean-monotonicity
  check; replaced it after it flagged the n.s. fixed-64-vs-128 wobble at
  B≥800 as "NO" in both units — the statistically right statement is
  "no *significant* inversion", so the section now renders paired step CIs.
- `fig_tokenizer_robustness` (BPE-unit budget curves + headline-delta bars
  by unit). Tests 225 → **243** (span tiling, byte-slice equality, emoji
  ZWJ/flag empty-span ordering, chunker contract under BPE, config_id/
  loader/renderer coverage; BPE tests skip without network). tiktoken
  0.13.0 pinned via `uv add` (lockfile in sync, `uv lock --check` clean).

### Ran (12 configs: BM25 baseline × cl100k unit, ~28 s)

### Findings (README §19)

1. **Finding 19 — every headline claim is unit-invariant under real BPE
   accounting.** One regex token costs 1.083–1.105 cl100k tokens on this
   corpus. All 16 headline paired cells keep their significance status and
   every significant cell keeps its sign: fixed-64 − fixed-256 @B400
   +0.134 regex vs **+0.120 [+0.103, +0.138]** cl100k; sentence-64 −
   fixed-64 +0.041 vs **+0.051**; hit@5 still rises with size while
   SpanRecall falls. Even the non-significant 64-vs-128 tie at B≥800
   replicates in both units. Across all 60 generated cells the only
   significance flips are five cells within ±0.011 of zero (two lose, three
   gain); everything ≥0.02 agrees in sign and significance. Sentence
   packing looks marginally *better* under BPE. The day-3 worry — and the
   README limitation entry — that conclusions might be "word-token-unit
   artifacts" is closed for the SQuAD/BM25 grid.

### Process notes

- The cl100k grid runs in ~28 s total (tokenization is not the bottleneck;
  the byte→char table only builds on non-ASCII documents).
- financial-rag-chatbot's README caveat from day 9 ("bench budgets are
  regex word tokens, not BPE") can now cite finding 19 — a one-line update
  for the next side-repo day, not worth a cross-repo commit today.
- Corpus totals in the unit-conversion table come from n_chunks ×
  tokens_mean of the zero-overlap configs — no data/ dependency at summary
  time, unlike summarize_chroma.

### Next steps (Day 11, in order)

1. **Semantic chunker** (last chunker family the related work compares;
   Chroma's ClusterSemanticChunker is the precedent): embedding-breakpoint
   segmentation over sentences using the existing MiniLM stack — cosine
   distance between adjacent sentence embeddings, percentile-threshold
   breakpoints, greedy packing under the token budget with the existing
   window fallback so the budget stays hard. Offsets must stay exact
   (reuse `split_sentences` ranges; never re-join text). Determinism is
   per-environment like all dense results — record model/torch versions in
   payloads (already automatic). Run dev-v1.1 BM25 first (semantic-64/128/
   256/512), compare against sentence packing at matched size — the
   interesting question is whether embedding breakpoints beat regex
   sentence boundaries once budget-matched.
2. Per-corpus × tercile interaction table for chroma (day-7 open item;
   `chatlogs` is the suggestive case).
3. Truncate × overlap cross (still one invocation away, both datasets).

### Open questions (carried)

- Overlap-delta tercile split on chroma (does the finding-16 gain
  concentrate in the long-gold tercile?).
- 512-window encoder ablation (day-6 backlog).
- Whether the semantic chunker's realized-size distribution (finding 4)
  needs a matched-realized-size comparison protocol, not just matched
  nominal size — decide after seeing its chunk stats.

---

## 2026-07-12 — Day 11: semantic chunker — findings 20–21: breakpoints buy size drift, not boundary quality

### Built (committed 4676495 BEFORE the grids ran, per house rule)

- `SemanticChunker` (percentile-breakpoint variant — Kamradt level 4, the
  LangChain default): adjacent-sentence MiniLM cosine distances, per-document
  95th-percentile threshold, strict `>` so an all-equal document degenerates
  to plain sentence packing; greedy packing never crosses a breakpoint;
  oversized sentences reuse the token-window fallback so the budget stays
  hard and offsets exact. Encoder is injectable (deterministic fakes in
  tests: hash-based for the contract battery, orthogonal-topic for exact
  breakpoint placement, constant for the degeneracy identity); the default
  resolves lazily to the process-wide MiniLM encoder, so the core stays
  importable without torch and embeddings are shared with the dense
  retriever (per-invocation memoization made sizes 128–512 cost ~2 s each
  after size 64 paid the ~60–100 s embedding bill).
- `chunker_stats` persisted per config (encoder identity, breakpoint rate,
  prefix-embedded sentence count) — the analogue of `retriever_stats`,
  there to distinguish a degenerate run (threshold never fired) from a
  semantic one without rerunning. `RunResult.chunker_stats` roundtrips.
- `summarize_semantic` (matched-size paired tables vs sentence AND fixed,
  realized-size stats front and center, breakpoint table) +
  `fig_semantic_comparison` (delta bars + realized-vs-nominal panel). All
  eight pre-existing dev-v1.1 PNGs and all seven chroma PNGs regenerate
  bit-identically with the new family present (CHUNKER_ORDER-driven figures
  ignore it by construction); the two main summaries gain semantic rows,
  purely additive diffs. Tests 243 → **327**. One pre-existing test used
  "semantic" as the unknown-chunker name; now "agentic".

### Ran (8 configs: dev-v1.1 + chroma, BM25, sizes {64,128,256,512}, ~3.5 min total)

Breakpoint rates: 5.2% of gaps (SQuAD), 5.0% (chroma) — p95 + strict >
fires on ~5% by construction; not degenerate. Chroma: 181/9,046 sentences
(2%) exceed the encoder window → prefix-embedded before breakpoints are
placed (recorded in stats; chatlogs/finance have paragraph-length
"sentences").

### Findings (README §20–21)

1. **Finding 20 — matched-nominal wins are realized-size drift.**
   Breakpoints only shorten; the shortfall grows with nominal size (SQuAD
   realized 46/100/190/314 vs sentence 48/109/234/475). Deltas track the
   gap, not the boundaries: B=400 Δ(sem−sent) = −0.001 / +0.005 / +0.014* /
   +0.194* across 64→512; the huge 512 cell is finding-5 stop-rule regime
   (realized 475 barely fits B=400, realized 314 slips under). Control:
   nominal 64, realized sizes 2.5% apart → null at every budget on both
   datasets (SQuAD B400: −0.001 [−0.004, +0.001]). Same structure as the
   recursive chunker in finding 4.
2. **Finding 21 — the size-drift account survives falsification.**
   Prediction: where larger effective chunks win (chroma long golds,
   generous budgets — finding 13), semantic should LOSE. It does:
   chroma-256 Δ(sem−sent) = **−0.026 [−0.044, −0.009]** @B1600 while the
   same config "wins" +0.083 @B200. hit@k: 5 significant negatives (incl.
   chroma-256 hit@1 −0.036, hit@5 −0.025) vs 1 grazing positive (SQuAD-128
   hit@1 +0.012). Verdict: budget-matched + size-accounted, the percentile
   chunker shows no boundary-quality effect beyond regex sentence packing
   — it is an expensive way to buy a smaller chunk size. Corroborates Qu,
   Tu & Bao (arXiv:2410.13070, verified today + added to references, with
   the Kamradt notebook); our contribution is the mechanism and the
   sign-flip test.

### Process notes

- The comparison design (same sentences, same budget, only the no-cross
  constraint differs) is what makes the isolation clean — worth reusing for
  any future chunker built on `split_sentences`.
- `summarize.py` main tables now include semantic rows on regeneration;
  `pairwise_same_size_rows` deliberately still covers only
  sentence/recursive-vs-fixed (semantic gets its own summarizer).
- Kamradt notebook URL and Qu et al. both verified reachable today before
  citing (curl 403s on github.com web; verified via search + arXiv abs).

### Next steps (Day 12, in order)

1. **Matched-realized-size protocol** (the open question finding 20 makes
   urgent, and the day-10 carry): compare semantic-512 (realized 314) not
   against sentence-512 but against the sentence config whose realized mean
   matches (interpolate: sentence packing at max_tokens ≈ 340–360 should
   realize ~314; one calibration run finds it). If the deltas collapse to
   ~0, finding 20's story is fully closed; if a residual remains, THAT is
   the boundary-quality effect. Cheap: 2–4 configs, BM25, both datasets.
2. Per-corpus × gold-tercile interaction table for chroma (day-7 carry;
   `chatlogs` still the suggestive case).
3. Truncate × overlap cross (day-8 carry, one invocation, both datasets).
4. Consider `--percentile` as a GridConfig field (default-omitted from
   config_id like budget_rule) if a threshold ablation is worth a day.

### Open questions (carried + new)

- Does semantic's breakpoint placement at least reduce *variance* of
  per-question recall at matched realized size? (Check alongside next-step
  1 — the paired per-question diffs are already on disk.)
- Overlap-delta tercile split on chroma (day-8 carry).
- 512-window encoder ablation (day-6 carry).

---

## 2026-07-13 — Day 12: matched-realized-size protocol — findings 22–23: the semantic wins vanish, and matched means prove insufficient

### Repo state note

Local clone was on a detached HEAD equal to origin/main with a stale local
`main`; reattached and fast-forwarded before starting (no content change).

### Built (committed e07218a BEFORE any grid ran, per house rule)

- `experiments/calibrate_matched.py` — for each semantic run on disk, binary
  search for the sentence `max_tokens` whose corpus-wide realized mean
  matches the semantic run's. The search is *exact*, not heuristic: under
  zero overlap chunks cover every token once, so realized mean = total
  tokens / n_chunks, and greedy packing never produces more chunks at a
  larger budget — realized mean is a nondecreasing step function of
  max_tokens (argument in the module docstring). Calibration is
  chunking-only (no retrieval), ~seconds per target with a per-size cache.
- `experiments/summarize_matched.py` — pairs every semantic run with (a) its
  same-nominal sentence partner and (b) the sentence run nearest in realized
  mean (`match_by_realized_size`; ties to smaller nominal; pairs with >5%
  relative gap are rendered but flagged). Renders both pairings side by
  side, plus hit@k and a per-question dispersion section.
- `metrics.paired_bootstrap_std` — jointly-resampled percentile CI for
  std(A) − std(B), for the day-11 open question ("do breakpoints at least
  reduce per-question variance?"). Same index resampling on both sides so
  shared question difficulty cancels.
- **`load_raw` grew `sizes` and `chunkers` filters** — off-grid calibrated
  sizes must not leak into canonical tables, and figures index a full
  (chunker, size) grid that off-grid files would KeyError. Every
  canonical-grid entry point now pins `BASELINE_SIZES`; the cross-grid
  summarizers (retrievers / seeds / tokenizers) additionally pin
  `STRUCTURAL_CHUNKERS`, which ALSO fixed a latent day-11 regression: the
  semantic grid exists only for BM25/seed-0/regex, so those summarizers'
  same-grid checks had refused to regenerate since it landed (nobody had
  rerun them). All 17 committed summaries and all 15 locally-generated PNGs
  verified bit-identical after the refactor.
- `fig_matched_realized` — 2×4 panels (dataset × nominal size), three series
  per panel: Δ vs nominal partner (stop), Δ vs realized partner (stop), Δ vs
  realized partner (truncate). Only renders well-matched drifted pairs.
- Discovery while verifying: `assets/hero_spanrecall_dev-v1.1_bm25.png` is
  NOT locally reproducible byte-for-byte — it was rendered in the Tier 2 PR's
  environment. Local renders are deterministic (two runs hash-identical) and
  visually identical to the committed file; left the committed file
  untouched. Every figure generated in THIS environment reproduces exactly.
- Tests 327 → **347** (std bootstrap incl. level-shift null; calibration on
  a hand-computable uniform-sentence corpus + monotonicity property; pairing
  incl. tie-break and flag; renderer; loader filters).

### Ran (32 configs total, all BM25: 8 calibrated-sentence stop + 8 semantic truncate + 8 calibrated-sentence truncate + summaries under both rules)

Calibrated sizes — SQuAD: 63/118/211/339 for semantic nominal 64/128/256/512
(realized gaps 0.85/0.37/0.00/0.10%); chroma: 62/118/210/353 (0.15/0.21/
0.18/0.10%). sentence-211 hits semantic-256's realized 190.34 *exactly*.

### Findings (README §22–23)

1. **Finding 22 — at matched realized size the semantic chunker gains
   nothing anywhere, and its long-gold penalty survives full size
   control.** The artifact-free cells (budget-free hit@k; truncate-rule
   recall at B ≫ chunk size) are uniformly null: 24/24 realized-matched
   hit@k cells (one graze +0.007), all SQuAD truncate B=1600 cells |Δ| ≤
   0.007, and the dispersion bootstrap shows no consistency benefit (day-11
   open question closed, answer no). What survives is finding 21's
   NEGATIVE: chroma semantic-512 vs sentence-353 at B=1600 is −0.042*
   (stop) / **−0.038 [−0.065, −0.011]** (truncate) — breakpoints measurably
   fragment long evidence. Verdict upgraded from "no detectable gain" to
   "no gain anywhere, real penalty in the regime semantic chunking is
   marketed for."
2. **Finding 23 — matched mean ≠ matched distribution; the stop rule
   converts residual dispersion into ±0.5 deltas.** Calibration equalizes
   means exactly (SQuAD: 314.3 vs 314.0, 961 vs 962 chunks) but semantic
   mixes breakpoint-shortened segments with budget-filled 512-token chunks
   (max 512 vs sentence's cap 339). Under stop, whichever side's top chunk
   exceeds B retrieves nothing: at B=400 semantic-512 spends 77 tokens
   mean (0.34 chunks — 2/3 of questions retrieve NOTHING) vs sentence-339's
   321 (exactly 1.00) → Δ = −0.533 [−0.554, −0.511]; at B=200 the artifact
   flips to +0.068 because the narrow distribution is now the one that
   never fits. Truncate deletes the boundary artifact; what remains is a
   real tight-budget dispersion penalty — same spend, fewer distinct
   regions (1.34 vs 2.00 chunks at B=400, −0.103) — fading to null by
   B=1600. Methodological rule added to finding 4: compare realized
   *distributions* or evaluate under truncate at B ≫ max chunk;
   matched-mean comparisons can manufacture arbitrarily large effects in
   either direction.

### Process notes

- The stop-rule matched table looked catastrophic at first read (−0.533 in
  a "controlled" comparison); the `chunks`/`tokens` utilization fields
  stored since day 3 resolved it in minutes without rerunning anything —
  third time the instrument-the-bottleneck rule has paid off (LSA ranks
  day 5, encoder exposure day 6).
- hit@k is identical under both budget rules (ranking never consults the
  budget), which makes it the cleanest boundary-quality metric in the
  matched protocol — worth remembering for any future chunker comparison.
- 3-series figure needed the light/dark purple split (stop vs truncate at
  matched realized) — reusing the semantic family color for both rules with
  different markers read as one series at small sizes.

### Next steps (Day 13, in order)

1. **Per-corpus error analysis** (last phase-3 item): per-corpus × gold-
   tercile interaction table for chroma (day-7 carry; `chatlogs` the
   suggestive case) + a look at the questions where fixed-64 loses hardest
   at B=1600 — is it always multi-reference long-gold questions?
2. Consider whether the truncate × overlap cross (day-8 carry) is still
   worth a run before the writeup phase, or gets recorded as
   deliberately-unrun in limitations.
3. Then phase 4: the full-writeup pass (README is already section-complete;
   the writeup phase is coherence editing, a results-navigation table up
   top, and a proper limitations sweep).

### Open questions (carried)

- Overlap-delta tercile split on chroma (day-8 carry).
- 512-window encoder ablation (day-6 carry) — likely limitations material
  rather than a run, decide during writeup.
- Cluster-variant semantic chunker and non-95 percentiles: recorded as
  untested in README §21; a `--percentile` ablation is one GridConfig field
  away if ever needed (day-11 note).

---

## 2026-07-14 — Day 13: per-question error analysis — findings 24–26: composition explains the corpora, the loss tail splits by mechanism, overlap decomposes exactly

### Repo state note

Same detached-HEAD situation as day 12 (local `main` stale behind
origin/main); reattached and fast-forwarded before starting. Fresh
container, so `data/` was re-fetched (`python -m src.data`, all SHA256 pins
matched) before the gold-length analyses could run.

### Built (committed 8ffa7c3 BEFORE any analysis output, per house rule)

- `experiments/summarize_errors.py` — the phase-3 closer. Three instruments,
  all over runs already on disk (zero new retrieval): (1) corpus ×
  gold-length tercile tables plus a **composition test** — observed corpus
  delta vs the delta predicted by reweighting *leave-one-corpus-out* tercile
  means with the corpus's own tercile mix, residual CI from a stratified
  bootstrap (corpus sample and each leave-out stratum resampled
  independently); (2) a **loss taxonomy** at the analysis budget splitting
  Δ ≤ −0.25 questions into complete misses (challenger recall exactly 0 — a
  ranking failure) vs partial-coverage losses, with per-stratum gold
  length / multi-ref share / hit@5 / corpus counts and a worst-questions
  table; (3) an **exact decomposition of every overlap gain** by the
  zero-overlap control's state on the same question — new region (ctrl = 0),
  extension (0 < ctrl < 1), redundancy tax (ctrl = 1) — the three masked
  contributions sum to the total by construction, each bootstrapped over the
  full question set with membership attached to the question. Plus the
  tercile × reference-count moderation view of the same four pairs.
- `fig_error_analysis` in make_figures (three panels: per-question Δ vs gold
  length with ringed complete misses; observed-vs-predicted composition
  forest; signed stacked decomposition bars with net dots). All fifteen
  pre-existing figures regenerate bit-identically; render is deterministic.
- Tests 347 → **357**. The composition-residual cases use deltas constant
  within strata so obs/pred/residual are hand-computable *exactly* (pure
  composition → residual exactly 0; injected corpus shift → residual exactly
  the shift; corpus owning a whole tercile → None).

### Ran (summarizer + figure only; ~2 min, dominated by 10k-resample bootstraps)

### Findings (README §24–26)

1. **Finding 24 — corpus identity adds nothing beyond gold-length mix.**
   Long-tercile B=1600 delta negative in all five corpora (pubmed −0.123*,
   wikitexts −0.098*, chatlogs −0.061*); short tercile null in 4/5. The
   composition test leaves no significant residual anywhere (extremes:
   state_of_the_union +0.042 [−0.009, +0.098], pubmed −0.040 [−0.097,
   +0.014]). The day-7 "chatlogs is suggestive" thread closes: chatlogs
   merely has 50% of its questions in the long tercile vs 14% for
   state_of_the_union. Sharpen of finding 14: the moderator IS gold length.
2. **Finding 25 — the loss tail is two mechanisms, the small one a ranking
   failure on SHORT golds.** 55/472 hard losses at B=1600: 49
   partial-coverage (median 75 gold tokens, 63% multi-ref, challenger hit@5
   0.80 — region found, window too small: finding 14 per-question) and 6
   complete misses on short single-ref golds (median 17 tokens; hit@5 0.00
   vs 1.00 — ~25 retrieved chunks never included the answer). The day-12
   hypothesis ("always multi-reference long-gold?") is 89% right and the
   remaining 11% is the interesting part: a 64-token window around a short
   gold carries too little question context to outrank confusable text —
   the lexical-ranker twin of finding 12's encoder-window mechanism. Wins
   mirror it: fixed-256 hit@5 drops to 0.41 in the win stratum.
3. **Finding 26 — overlap = placement + extension − redundancy tax;
   stitching is budget-limited.** The tax (questions the control already
   answered perfectly) is significant in EVERY non-degenerate cell (−0.015
   to −0.058) — the protocol's redundancy accounting made visible
   per-question. Large windows at tight budgets gain by placement
   (fixed-256/o64 @B400: +0.084 of +0.059 net is new-region), small windows
   persist by extension (fixed-64/o32 @B1600: +0.032 of +0.024). Finding
   17's stitching: real at B=400 on long golds (3/4 pairs significant),
   gone by B=800 everywhere; what persists sits on within-window
   single-reference golds (fixed-64 mid tercile +0.052*; fixed-128 short
   +0.027*). Practical rule amended with a budget clause (README §26).

### Process notes

- The day-8 carry (overlap tercile split) delivered the opposite of the
  expected headline — the naive tercile table looked incoherent until the
  control-state decomposition reframed it. Lesson repeated from days 5/6/12:
  when a moderation table confuses, decompose by mechanism before
  interpreting cells.
- The exact-decomposition trick (masked per-question deltas, membership
  travels with the question through the bootstrap) is reusable for ANY
  paired comparison here — candidate for the writeup's methods section.
- Composition-test design point worth keeping: predict from leave-one-out
  strata, never pooled ones, or the corpus predicts itself.

### Next steps (Day 14, in order)

1. **Phase 4 writeup begins.** README is section-complete with 26 findings;
   do the coherence pass: a findings-navigation table up top (finding →
   one-line claim → section anchor), reconcile early-finding phrasings with
   later refinements (finding 1 vs 13; 6/16/17 vs 26; 20 vs 22–23), and a
   full limitations sweep.
2. During that sweep, record the deliberately-unrun items with reasons:
   truncate × overlap cross (day-8 carry — finding 26 now explains the
   overlap mechanism; the cross would only re-price the tax under the other
   stop rule), 512-window encoder ablation (day-6 carry), cluster-variant
   semantic chunker / percentile sweep (day-11 note).
3. Hero figure / abstract numbers: check both still match the final tables
   after the writeup edits (they should — no numbers changed today).

### Open questions (carried)

- None new. Remaining carries all fold into the writeup-phase decisions
  listed above.

---

## 2026-07-15 — Day 14: phase 4 opens — the coherence pass: navigation, reconciliation, limitations

Writeup day, no new runs: the README got its findings-at-a-glance
navigation table (26 one-line claims, each anchor-linked to its section),
the early findings were reconciled with their later refinements by
explicit cross-references instead of leaving the reader to collate them,
and the limitations section absorbed the deliberately-unrun ablations
with the reasons they stay unrun. Every headline number cited in the
README was re-checked against the committed summary tables before
editing, and the full test suite passes both without the dense stack
(339 + 3 skipped) and with it (355 + 2 = 357, matching the README's
claim).

### Repo state note

Same detached-HEAD situation as days 12–13; reattached to `main`,
fast-forwarded to origin/fe4c853 before starting. No data fetch needed —
today touched only prose.

### Done (committed 97b1d5c)

- **Navigation table** (`## The 26 findings at a glance`, right after the
  abstract): one row per finding, claim compressed to a line, section
  links generated with the same slug rule as `scripts/sync_latest.py`
  and verified programmatically — all 26 references resolve against the
  file's actual headings. Added a three-finding shortlist for skimmers
  (2, 14, 23). The stale `### Findings so far` heading became
  `### Baseline findings: the size effect and the fixed-k confound`
  (nothing else linked to the old anchor; grepped the repo first).
- **Abstract**: "results so far" framing replaced with
  program-complete framing, and the error-analysis findings 24–26 added
  as the closing clause (composition, loss taxonomy, exact overlap
  decomposition).
- **Reconciliation cross-refs** where an early finding's practical rule
  was later amended and the amendment lived 400 lines away: finding 6 →
  26 (the overlap price now itemized as placement + extension − tax),
  finding 17 → 26 (the budget clause on the longer-than-chunk rule),
  finding 14 → 24 (the moderator is gold-length composition, corpus
  identity adds nothing). Finding 1 → 13–15 and 20/21 → 22–23 already
  carried their pointers from the days they landed; verified, left
  alone.
- **Limitations sweep**: the semantic bullet was stale — it still said a
  fair boundary-quality test "requires matching realized size
  distributions" as if that test were future work, when day 12 ran it;
  now states findings 22–23 as the executed test and narrows the open
  residue to cluster variant / other percentiles / larger encoders. Two
  new bullets: the error-analysis scope (BM25/Chroma anatomy only,
  B=1600 taxonomy budget and ≥0.25 threshold stated), and the two
  deliberately-unrun ablations with reasons — overlap × truncate cross
  (finding 26 identifies the mechanisms; truncate would only re-price
  the tax, and rule-robustness is already established by findings 7/18)
  and the larger-window dense encoder (window mechanism established by
  exposure instrumentation; an encoder swap confounds window with model
  quality and exceeds the CPU budget).
- **Status section**: phase-4 coherence pass checked off; the remaining
  phase-4 item is now concrete — a clean-environment reproduction audit
  regenerating every summary and figure from committed raw results and
  verifying against the committed files, then release polish.

### Verified (day-13 carry item 3)

- Abstract and hero caption carry no numbers that could go stale
  (qualitative by design); the specific deltas cited in findings 1, 13,
  18, 22 re-checked against `summary_dev-v1.1_bm25.md`,
  `summary_chroma_bm25_moderation.md`, `summary_chroma_bm25_ablations.md`,
  and both matched summaries — all match. Finding 18's −0.041/−0.047
  correspond to the truncate-table means (0.794−0.834, 0.874−0.921)
  with paired CIs in the moderation summary.

### Process notes

- The composition-test bullet nearly went into limitations as "single
  corpus set" — but that duplicates the existing per-corpus-n bullet.
  Rule of thumb for limitation sweeps: each bullet should name a
  *distinct* threat, or reviewers read padding.
- Writing the one-line claims was itself a useful audit: every finding
  compressed to a line without losing its qualifier except 18, which
  needed both halves (rule-robust crossover AND the artifact-share of
  the tight-budget edge) — a hint that 18 is really two findings, kept
  as one to preserve the numbering that the summaries and notes already
  cite.

### Next steps (Day 15, in order)

1. **Reproduction audit**: in a clean checkout, refetch data
   (`python -m src.data`), regenerate all 22 summaries and all figures
   from committed raw results, and diff against the committed files.
   Expect bit-identity for everything generated in this environment
   (the known exception: the day-12 note about the Tier 2-rendered hero
   PNG — re-verify visually, document in NOTES if it still differs).
   Fix anything that drifts or document why it can't.
2. **Release polish**: read the README top-to-bottom once as a stranger
   (fresh-eyes pass for typos, tense, and any remaining "so far"
   phrasing), confirm the Reproducing section's commands match the
   actual entry points, and decide whether the project is DONE by the
   demanding-reviewer bar. If yes, close it out in ROADMAP.md and pick
   the next flagship from the backlog next session.

### Open questions (carried)

- None. All former carries are either executed, folded into
  limitations with reasons, or listed in README §21's open residue.

---

## 2026-07-16 — Day 15: the reproduction audit catches two stale tables; the flagship closes

The plan was a verification formality; the audit earned its keep instead.
In a fresh clone with a fresh interpreter and refetched data, the new
`experiments/reproduce.py` manifest audit regenerated all 41 committed
tables and figures from the committed raw results: 38 reproduced
byte-identically, two ablation tables turned out to be stale (day 13's
semantic truncate runs had leaked into the structural budget-rule
section — summarizer scoped, regression-tested, re-verified
bit-identical), and the hero PNG was re-rendered so every committed
artifact now originates from this reproducible environment. With that
and the release polish below, `rag-chunking-bench` closes complete:
26 findings, 365 tests, and a replayable byte-level reproduction audit.

### The audit harness (new, committed)

`experiments/reproduce.py`: a manifest mapping **every** committed
artifact — 22 summary tables, 18 figures, the hero PNG — to the exact
generator invocation that produces it (25 steps across the nine
summarizers and two figure scripts). Default mode replays all 25 steps
with `--out-dir` redirected to a scratch directory *inside* the repo
(the generators print `relative_to(ROOT)` paths, so an outside temp dir
crashes them — learned by hitting it) and byte-compares against the
committed files, reporting OK/DRIFT/MISSING/UNEXPECTED and exiting
nonzero on any failure. `--write` regenerates in place; `--tables-only`
skips figures for cross-OS use, where PNG bytes legitimately depend on
the font stack. 7 manifest tests pin the output set to the committed
files on disk (globbed, so a new committed artifact without a manifest
entry fails the suite), plus uniqueness and runnability invariants.

### The clean-environment run

Fresh clone, fresh venv from pinned requirements (no dense stack —
regeneration only reads raw), `python -m src.data` refetched both
datasets (SHA256s passed), suite green, then the full audit (~50 min,
bootstrap-heavy summarizers dominate). Verdict: **38/41 byte-identical,
3 DRIFT** —

1. `summary_dev-v1.1_bm25_ablations.md` and the chroma twin: the
   regenerated tables had four extra `semantic-*/truncate` rows. Root
   cause: day 13's matched-realized-size grids dropped semantic truncate
   runs at canonical sizes into `results/raw/`, and `render_ablations`
   loaded truncate runs **without** the `chunkers=STRUCTURAL_CHUNKERS`
   filter the day-12 refactor gave the other cross-grid summarizers. The
   committed tables predate day 13 and nobody re-ran the summarizer
   after it. No committed number was wrong (the diff was purely added
   rows) — the failure mode was scope leak plus staleness, exactly what
   the audit exists to catch. Fix: both loads in `render_ablations` now
   pass the structural filter (the ablation summary's scope is the
   structural grid; semantic truncate analysis lives in the matched
   summaries); regression test added
   (`test_render_ablations_excludes_semantic_runs`, via the
   `_rewrite_config_field` fixture). Fixed summarizer re-verified
   bit-identical against the committed tables in BOTH environments.
2. The known day-12 hero exception: `assets/hero_spanrecall_*.png` was
   rendered in the Tier 2 PR's environment and never byte-reproduced
   here. Re-rendered locally (deterministic: two uv-env renders and one
   clean-clone pip-env render all hash `24dc9dc4…`, and the clean-clone
   render matches the newly committed file), compared visually against
   the old file (identical modulo font rasterization), committed the
   local render. Every committed artifact now originates from this
   reproducible environment.

Composite result recorded in the README: all 41 artifacts verified
byte-identical in the clean environment. Tests 357 → **365** (7 manifest
+ 1 regression); 347+3skip without the dense stack.

### Release polish (same commit set)

- Fresh-eyes pass over the full README: one real catch — the baseline
  budget-curves caption said "see finding 4" where the stop-rule
  retrieve-nothing artifact is finding **5**. No broken relative links,
  no stale "so far" framing (the two remaining are positional, fine).
- Both command blocks (Experiments intro, Reproducing) had drifted from
  the entry points: neither listed `summarize_matched`,
  `summarize_tokenizers`, `summarize_semantic`, or `calibrate_matched`.
  Replaced the hand-maintained summarizer lists with
  `experiments.reproduce` (+ `--write`) and pointed at the manifest as
  the authoritative map; added `calibrate_matched` to Reproducing.
  Updated the determinism paragraph with the audit result and the
  tables-vs-figures byte-identity expectations.
- Status section: phase 4 checked off with the audit story; added the
  "program complete, open residue in Limitations" closing line.
- Lab-level: ROADMAP restructured (flagship moved to a Completed
  section with the outcome paragraph; "Current flagship: none —
  selection due"); CHANGELOG 0.2.0 entry summarizing days 5–15 at
  release altitude.

### DONE decision

Called it complete against the demanding-reviewer bar: 26 findings, each
with paired CIs and at least one robustness axis; five-axis robustness
program; a mechanism-level error analysis; limitations that name every
deliberate gap with reasons; literature grounding with verified
citations; and now a replayable byte-level reproduction audit. The open
residue (cluster semantic variant, percentile sweep, larger encoders,
multi-hop gold, cross-document retrieval) is recorded in README §
Limitations — it is future work, not unfinished work.

### Next steps (Day 16)

1. **Pick the next flagship.** Fresh scan (arXiv, Papers with Code,
   model release notes, engineering blogs) against the ROADMAP backlog;
   the two strongest inheritances from this project are the
   hallucination/faithfulness bench (reuses the span-metric machinery)
   and the LLM-as-judge reliability audit (needs hosted-API access —
   verify HF Inference API reachability with HF_TOKEN before
   committing to it). Record the rationale in ROADMAP.md, scaffold the
   project directory, and write day 1 of its NOTES.md.
2. If the scan stalls, the side-repo rotation is overdue a
   `data-analysis-agent` day (check its `git log -15` first).

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

# rag-chunking-bench

**How much does chunking actually matter for RAG retrieval — once you control
for the retrieved-token budget?**

## Abstract

Chunking is the highest-leverage, least-principled design decision in
retrieval-augmented generation: every production stack picks a chunk size and
overlap, usually by folklore. Published comparisons of chunking strategies
almost always vary chunk size while holding top-*k* fixed — which confounds
chunking quality with the sheer number of tokens handed to the generator
(500-token chunks at k=5 retrieve five times the text of 100-token chunks at
k=5, and the generator's context budget pays for it). This project benchmarks
structural and semantic chunking strategies under a **budget-matched
protocol**: retrievers are compared at equal retrieved-token budgets using
span-level (token-overlap) metrics against gold evidence spans, with paired
bootstrap confidence intervals over questions. The goal is a defensible
answer to "which chunking decisions survive budget control, and by how much,"
at a scale (CPU, open data) that anyone can reproduce.

## Motivation

Three observations from the literature motivate the design:

1. **Rank-based metrics hide the cost of chunk size.** Recall@k improves with
   larger chunks partly because each retrieved unit simply contains more
   text. Smith & Troynikov (2024) introduced token-level precision/recall/IoU
   against gold excerpts, which this project adopts — and extends by making
   the *retrieved-token budget* (not k) the controlled variable.
2. **Chunking papers rarely quantify uncertainty.** Recent comparisons
   (Merola & Singh, 2025; Duarte et al., 2024) report point estimates on a
   single configuration. Here every comparison is paired per-question with
   bootstrap confidence intervals, so "strategy A beats strategy B" comes
   with an interval, not just a mean.
3. **The generator's context is the scarce resource.** "Lost in the Middle"
   (Liu et al., 2023) showed long, padded contexts actively hurt; a chunking
   strategy that wins only by retrieving more tokens is not a win. Budget
   matching makes that failure mode visible.

## Method

### Evaluation protocol

For a question *q* over a document collection, the gold evidence is a set of
character spans *G* in the source documents. A chunker segments each document
into chunks with exact character offsets; a retriever ranks chunks for *q*;
retrieved chunks are accumulated in rank order until the token budget *B* is
exhausted (the first chunk that would exceed *B* stops accumulation). With
*C* = the set of retrieved tokens and *G* = gold-span tokens:

- **SpanRecall@B**  = |C ∩ G| / |G|
- **SpanPrecision@B** = |C ∩ G| / |C|
- **SpanIoU@B**   = |C ∩ G| / |C ∪ G|

Two accounting rules make the comparison fair. The budget charges *prompt*
tokens — each retrieved chunk costs its own token count, duplicates included,
because that is what a generator would be handed — while scoring uses the
*union* of retrieved tokens, so redundant overlap spends budget without
earning recall. And when a dataset annotates several alternative gold spans
for one question (SQuAD's multiple annotations), each metric takes the max
over alternatives, the standard max-over-answers convention; corpora whose
references are jointly required score against their union through the same
code path.

Metrics are computed per question and aggregated with means and 95% paired
bootstrap confidence intervals (fixed seed). Budgets sweep
B ∈ {200, 400, 800, 1600} tokens; classic hit@k is also reported for
comparability with prior work.

Token counting uses a deterministic regex word/punctuation tokenizer
(`src/tokenization.py`) shared by chunkers, budget accounting, and metrics —
the unit only needs to be consistent, not identical to any model's BPE. The
`Tokenizer` protocol allows a BPE tokenizer to be slotted in as a robustness
check.

### Chunking strategies (`src/chunkers.py`)

All chunkers emit chunks with exact document offsets
(`document[start:end] == chunk.text`), the invariant that makes span-level
scoring exact rather than fuzzy-matched.

| Strategy | Description | Knobs |
|---|---|---|
| `FixedTokenChunker` | sliding token window | size, overlap |
| `SentenceChunker` | greedy packing of whole sentences under a token budget | max_tokens, sentence overlap |
| `RecursiveCharacterChunker` | paragraph > line > space separator hierarchy with greedy merge (LangChain semantics, offset-preserving) | max_tokens |
| semantic chunkers | embedding-similarity breakpoints (phase 2) | — |

### Retrievers

BM25 (lexical, implemented from scratch in `src/retrievers.py` and tested
against a hand-computed example), TF-IDF cosine (sparse), and LSA (low-rank
dense, TruncatedSVD over TF-IDF). BEIR (Thakur et al., 2021) established BM25
as a robust zero-shot baseline; the chunking effect is measured holding each
retriever fixed, and retriever × chunker interaction is itself a studied
variable. A neural dense retriever (a small sentence-transformer running on
CPU) is planned for phase 2 to cover the lexical-vs-dense axis.

### Datasets

- **SQuAD-derived long documents** (`src/data.py`): Wikipedia articles
  reconstructed by concatenating each article's paragraphs, with answer spans
  remapped into article coordinates and verified verbatim at load time.
  dev-v1.1 yields 48 documents (2.7k–16.8k tokens, median 6.2k) and 10,533
  answerable questions after deduplication. Raw JSON is fetched from pinned
  URLs with SHA256 checks (`python -m src.data`).
- **Chroma chunking-evaluation corpora** (Smith & Troynikov, 2024): five
  heterogeneous corpora (state-of-the-union, Wikitexts, chat logs, finance,
  PubMed) with question/gold-excerpt pairs; queries are LLM-generated, which
  is recorded as a provenance caveat.

## Experiments

Phase 2+ will populate this section with real tables. Planned grid: chunker ×
chunk size {64, 128, 256, 512} × overlap {0%, 12.5%, 25%} × retriever ×
budget, plus ablations (sentence-boundary alignment, separator hierarchy,
tokenizer robustness) and per-dataset error analysis.

**No results are reported yet; every number that appears here will come from
runs checked into `experiments/` and `results/`.**

## Status

- [x] Phase 1 (harness): offset-preserving chunkers + tokenization
- [x] Phase 1 (harness): SQuAD loader with verified spans, span metrics,
      budget-matched accumulation, paired bootstrap, BM25 (134 tests)
- [ ] Phase 1 (harness): Chroma corpora loader, TF-IDF/LSA retrievers,
      experiment runner
- [ ] Phase 2: baseline grid runs
- [ ] Phase 3: ablations, error analysis, significance testing
- [ ] Phase 4: full writeup

Day-by-day research log: [`research/NOTES.md`](research/NOTES.md).

## Limitations

- CPU-only environment: neural retrieval is limited to small
  sentence-transformer models; findings may not transfer to large dense
  retrievers or cross-encoder rerankers.
- The regex tokenizer approximates BPE token counts; budget conclusions are
  in "word-token" units (a tiktoken BPE robustness check is planned).
- Chroma corpora queries are LLM-generated (dataset provenance, not ours);
  SQuAD questions are human-written but crowd-sourced over single paragraphs.

## References

- Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna
  Gurevych. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
  Information Retrieval Models.* NeurIPS Datasets & Benchmarks 2021.
  [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)
- Brandon Smith, Anton Troynikov. *Evaluating Chunking Strategies for
  Retrieval.* Chroma Technical Report, July 2024.
  [research.trychroma.com/evaluating-chunking](https://research.trychroma.com/evaluating-chunking)
- Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele
  Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language
  Models Use Long Contexts.* TACL 2024.
  [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- André V. Duarte, João Marques, Miguel Graça, Miguel Freire, Lei Li, Arlindo
  L. Oliveira. *LumberChunker: Long-Form Narrative Document Segmentation.*
  Findings of EMNLP 2024.
  [arXiv:2406.17526](https://arxiv.org/abs/2406.17526)
- Carlo Merola, Jaspinder Singh. *Reconstructing Context: Evaluating Advanced
  Chunking Strategies for Retrieval-Augmented Generation.* 2nd Workshop on
  Knowledge-Enhanced Information Retrieval, ECIR 2025.
  [arXiv:2504.19754](https://arxiv.org/abs/2504.19754)
- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang. *SQuAD:
  100,000+ Questions for Machine Comprehension of Text.* EMNLP 2016.
  [arXiv:1606.05250](https://arxiv.org/abs/1606.05250)

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python 3.11. Experiment run scripts land in `experiments/` in phase 2.

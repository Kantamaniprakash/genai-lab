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

Metrics are computed per question and aggregated with means and 95% paired
bootstrap confidence intervals (fixed seed). Budgets sweep
B ∈ {200, 400, 800, 1600} tokens; classic Recall@k is also reported for
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

BM25 (lexical), TF-IDF cosine (sparse), and LSA (low-rank dense, TruncatedSVD
over TF-IDF). BEIR (Thakur et al., 2021) established BM25 as a robust
zero-shot baseline; the chunking effect is measured holding each retriever
fixed, and retriever × chunker interaction is itself a studied variable.
Neural dense retrievers are excluded by environment constraints (see
Limitations).

### Datasets

- **SQuAD-derived long documents**: Wikipedia articles reconstructed by
  concatenating each article's paragraphs, with answer spans mapped into
  article coordinates — natural questions with exact gold spans in ~3–6k-token
  documents.
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

- [x] Phase 1 (harness): offset-preserving chunkers + tokenization, 80 tests
- [ ] Phase 1 (harness): dataset loaders, span metrics, budget-matched retrieval loop
- [ ] Phase 2: baseline grid runs
- [ ] Phase 3: ablations, error analysis, significance testing
- [ ] Phase 4: full writeup

Day-by-day research log: [`research/NOTES.md`](research/NOTES.md).

## Limitations

- CPU-only environment without access to model hosts: retrievers are
  lexical/sparse/low-rank-dense; findings about chunking × *neural* retriever
  interaction are out of scope and flagged as future work.
- The regex tokenizer approximates BPE token counts; budget conclusions are
  in "word-token" units (a BPE robustness check is planned).
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

"""Invariant and edge-case tests for the chunking strategies.

The core contract every chunker must uphold:
  1. Offsets are exact: document[start:end] == text.
  2. Budget is a hard guarantee: no chunk exceeds the configured token limit.
  3. Coverage: every token of the document lands in at least one chunk
     (exactly one when overlap is zero for the fixed chunker).
Span-level evaluation is meaningless if any of these break, so they are
tested across all chunkers on a shared battery of documents.
"""

import random

import pytest

from src.chunkers import (
    Chunk,
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
    split_sentences,
)
from src.tokenization import RegexWordTokenizer, TokenIndex

TOKENIZER = RegexWordTokenizer()

PROSE = (
    "Retrieval-augmented generation grounds a language model in external "
    "documents. The chunking step decides what a retrievable unit is. "
    "Small chunks are precise but fragment context; large chunks preserve "
    "context but dilute the embedding. This trade-off is rarely measured "
    "under a controlled token budget.\n\n"
    "BM25 remains a robust baseline for zero-shot retrieval. Dense models "
    "often underperform it out of domain! Does chunking interact with the "
    "retriever family? That question motivates this benchmark.\n\n"
    "A final short paragraph."
)

MESSY = "  \n\nword\tword2 -- word3?? (nested. clause) end\n\n\n x  "

UNICODE = "Ceci n'est pas une pipe. Türkçe metin — çok iyi. 数字 123 mixed."

NO_SEPARATORS = "x" * 30 + " " + "supercalifragilistic" * 40

DOCUMENTS = [PROSE, MESSY, UNICODE, NO_SEPARATORS, "one", "", "   \n\n  "]


def random_document(seed: int) -> str:
    rng = random.Random(seed)
    words = ["alpha", "beta", "Gamma.", "delta,", "epsilon!", "Zeta"]
    parts = []
    for _ in range(rng.randint(0, 400)):
        parts.append(rng.choice(words))
        parts.append(rng.choice([" ", " ", " ", "\n", "\n\n"]))
    return "".join(parts)


ALL_DOCUMENTS = DOCUMENTS + [random_document(seed) for seed in range(10)]


def make_chunkers(limit: int):
    return [
        FixedTokenChunker(chunk_size=limit),
        FixedTokenChunker(chunk_size=limit, overlap=limit // 3),
        SentenceChunker(max_tokens=limit),
        SentenceChunker(max_tokens=limit, overlap_sentences=1),
        RecursiveCharacterChunker(max_tokens=limit),
    ]


@pytest.mark.parametrize("document", ALL_DOCUMENTS)
@pytest.mark.parametrize("limit", [1, 4, 16, 64])
def test_offsets_budget_and_coverage(document, limit):
    index = TokenIndex(document, TOKENIZER)
    for chunker in make_chunkers(limit):
        chunks = chunker.chunk(document)
        covered = set()
        for c in chunks:
            assert document[c.start : c.end] == c.text
            assert index.count_in(c.start, c.end) <= limit
            assert index.count_in(c.start, c.end) >= 1
            covered.update(index.tokens_in(c.start, c.end))
        assert covered == set(range(len(index))), type(chunker).__name__
        starts = [c.start for c in chunks]
        assert starts == sorted(starts)


def test_empty_and_whitespace_documents_yield_no_chunks():
    for document in ["", "   \n\t  "]:
        for chunker in make_chunkers(8):
            assert chunker.chunk(document) == []


def test_fixed_chunker_exact_sizes_and_overlap():
    document = " ".join(f"tok{i}" for i in range(23))
    index = TokenIndex(document, TOKENIZER)
    chunker = FixedTokenChunker(chunk_size=10, overlap=4)
    chunks = chunker.chunk(document)
    sizes = [index.count_in(c.start, c.end) for c in chunks]
    assert sizes[:-1] == [10] * (len(sizes) - 1) and sizes[-1] <= 10
    for a, b in zip(chunks, chunks[1:]):
        shared = set(index.tokens_in(a.start, a.end)) & set(
            index.tokens_in(b.start, b.end)
        )
        assert len(shared) == 4


def test_fixed_chunker_no_overlap_partitions_tokens():
    document = PROSE
    index = TokenIndex(document, TOKENIZER)
    chunks = FixedTokenChunker(chunk_size=17).chunk(document)
    seen = []
    for c in chunks:
        seen.extend(index.tokens_in(c.start, c.end))
    assert seen == list(range(len(index)))


def test_fixed_chunker_rejects_bad_params():
    with pytest.raises(ValueError):
        FixedTokenChunker(chunk_size=0)
    with pytest.raises(ValueError):
        FixedTokenChunker(chunk_size=5, overlap=5)
    with pytest.raises(ValueError):
        FixedTokenChunker(chunk_size=5, overlap=-1)


def test_split_sentences_offsets_and_boundaries():
    document = 'First sentence. Second one! A third? "Quoted start." Then\nnewline-split.'
    ranges = split_sentences(document)
    texts = [document[a:b] for a, b in ranges]
    assert texts == [
        "First sentence.",
        "Second one!",
        "A third?",
        '"Quoted start."',
        "Then",
        "newline-split.",
    ]
    for a, b in ranges:
        assert document[a:b] == document[a:b].strip()


def test_sentence_chunker_keeps_sentences_whole_when_they_fit():
    document = "Aaa bbb ccc. Ddd eee. Fff ggg hhh iii. Jjj."
    chunks = SentenceChunker(max_tokens=8).chunk(document)
    for c in chunks:
        assert c.text[0].isupper() and c.text[-1] == "."


def test_sentence_chunker_splits_oversized_sentence():
    document = "word " * 50  # one 50-token "sentence", no terminal punctuation
    chunks = SentenceChunker(max_tokens=8).chunk(document.strip())
    assert len(chunks) == 7  # ceil(50 / 8)


def test_sentence_overlap_repeats_boundary_sentence():
    document = "One two three. Four five six. Seven eight nine. Ten eleven."
    chunks = SentenceChunker(max_tokens=8, overlap_sentences=1).chunk(document)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # consecutive chunks share a sentence


def test_recursive_chunker_prefers_paragraph_boundaries():
    para = "Alpha beta gamma delta. Epsilon zeta eta theta."
    document = "\n\n".join([para] * 4)
    chunks = RecursiveCharacterChunker(max_tokens=12).chunk(document)
    # Each paragraph is 10 tokens; the merge step cannot join two (20 > 12),
    # so chunks must align exactly with paragraphs.
    assert [c.text for c in chunks] == [para] * 4


def test_recursive_chunker_merges_small_pieces():
    document = "a b\n\nc d\n\ne f\n\ng h"
    chunks = RecursiveCharacterChunker(max_tokens=6).chunk(document)
    assert len(chunks) == 2  # three 2-token paragraphs merge into six tokens...
    index = TokenIndex(document, TOKENIZER)
    assert [index.count_in(c.start, c.end) for c in chunks] == [6, 2]


def test_recursive_chunker_handles_separator_free_text():
    document = "abcdefghij" * 40  # single 400-char token, far over any budget
    chunks = RecursiveCharacterChunker(max_tokens=4).chunk(document)
    assert len(chunks) == 1 and chunks[0].text == document


def test_chunk_validates_range():
    with pytest.raises(ValueError):
        Chunk(text="x", start=5, end=3)

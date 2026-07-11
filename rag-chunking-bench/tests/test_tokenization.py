"""Invariant tests for BPE token-span reconstruction.

``TiktokenTokenizer`` maps byte-level BPE tokens back to character offsets.
Everything downstream (TokenIndex range queries, chunk boundaries, budget
accounting) relies on the invariants tested here: spans are ordered and
non-overlapping, they tile the entire document, and there are exactly as
many spans as tiktoken reports tokens — including when token boundaries
fall inside multi-byte characters.

Constructing the encoding downloads its vocabulary, so the whole module
skips gracefully when tiktoken or the vocabulary host is unavailable.
"""

import pytest

from src.chunkers import (
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from src.tokenization import TokenIndex

ASCII_PROSE = (
    "Retrieval-augmented generation grounds a language model in external "
    "documents. Chunking decides what a retrievable unit is; budgets decide "
    "how much of it a generator reads. Numbers like 1974 and names like "
    "O'Neill tokenize differently under BPE than under word rules."
)

UNICODE_PROSE = "Ceci n'est pas une pipe. Türkçe metin — çok iyi. 数字 123 mixed."

# ZWJ sequences and flags force tokens whose bytes sit inside one character
# or split a character across tokens — the hard case for offset recovery.
EMOJI = "start 👩‍👩‍👧‍👦 middle 🇺🇳🙂🙃 end"

SPECIAL = "plain text with <|endoftext|> written inside it"

DOCUMENTS = [ASCII_PROSE, UNICODE_PROSE, EMOJI, SPECIAL, "one", ""]


@pytest.fixture(scope="module")
def bpe():
    pytest.importorskip("tiktoken")
    from src.tokenization import TiktokenTokenizer

    try:
        return TiktokenTokenizer("cl100k_base")
    except Exception as exc:  # vocabulary download needs network access
        pytest.skip(f"cl100k_base vocabulary unavailable: {exc}")


@pytest.mark.parametrize("document", DOCUMENTS)
def test_spans_tile_the_document(bpe, document):
    spans = bpe.spans(document)
    if not document:
        assert spans == []
        return
    assert spans[0].start == 0
    assert spans[-1].end == len(document)
    for a, b in zip(spans, spans[1:], strict=False):
        assert a.end == b.start  # BPE tokens cover every byte: no gaps
    for s in spans:
        assert 0 <= s.start <= s.end <= len(document)


@pytest.mark.parametrize("document", DOCUMENTS)
def test_span_count_matches_token_count(bpe, document):
    assert len(bpe.spans(document)) == bpe.count(document)


def test_ascii_spans_slice_to_token_bytes(bpe):
    ids = bpe._enc.encode(ASCII_PROSE, disallowed_special=())
    token_bytes = bpe._enc.decode_tokens_bytes(ids)
    spans = bpe.spans(ASCII_PROSE)
    assert len(spans) == len(token_bytes)
    for s, raw in zip(spans, token_bytes, strict=True):
        assert ASCII_PROSE[s.start : s.end].encode("utf-8") == raw


def test_multibyte_boundary_spans_stay_ordered(bpe):
    spans = bpe.spans(EMOJI)
    # Some tokens sit entirely inside one character (empty spans); every
    # span must still be ordered and the non-empty ones must not overlap.
    assert any(s.start == s.end for s in spans)
    starts = [s.start for s in spans]
    ends = [s.end for s in spans]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    assert "".join(EMOJI[s.start : s.end] for s in spans) == EMOJI


def test_special_token_text_is_treated_as_text(bpe):
    # Documents are data: <|endoftext|> must tokenize as characters, not as
    # a control token (which would raise or collapse the span structure).
    spans = bpe.spans(SPECIAL)
    assert spans[-1].end == len(SPECIAL)
    assert len(spans) > 1


def test_deterministic(bpe):
    assert bpe.spans(UNICODE_PROSE) == bpe.spans(UNICODE_PROSE)


def test_token_index_range_queries(bpe):
    index = TokenIndex(ASCII_PROSE, bpe)
    assert len(index) == bpe.count(ASCII_PROSE)
    assert index.count_in(0, len(ASCII_PROSE)) == len(index)
    # A mid-document range claims a contiguous, non-empty token run.
    mid = index.tokens_overlapping(50, 120)
    assert len(mid) > 0
    assert mid.stop <= len(index)


@pytest.mark.parametrize("limit", [4, 16, 64])
@pytest.mark.parametrize("document", [ASCII_PROSE, UNICODE_PROSE, EMOJI])
def test_chunkers_uphold_contract_under_bpe(bpe, document, limit):
    # The contract test_chunkers.py enforces under the regex tokenizer,
    # restated for a whitespace-carrying tokenizer: exact offsets and ordered
    # starts always; window chunks additionally contain every token exactly
    # as under regex, while boundary-respecting chunkers cut at sentence and
    # separator edges that BPE tokens straddle, so their coverage guarantee
    # is by token *overlap* (the scoring convention) and holds for every
    # token that carries any non-whitespace text. The budget is hard except
    # for documents with empty spans (tokens inside one multi-byte
    # character), which the containment query can attribute to either
    # neighboring chunk; the emoji case covers that regime.
    index = TokenIndex(document, bpe)
    has_empty_spans = any(s.start == s.end for s in index.spans)
    content_tokens = {
        i for i, s in enumerate(index.spans) if document[s.start : s.end].strip()
    }
    chunkers = (
        FixedTokenChunker(chunk_size=limit, tokenizer=bpe),
        SentenceChunker(max_tokens=limit, tokenizer=bpe),
        RecursiveCharacterChunker(max_tokens=limit, tokenizer=bpe),
    )
    for chunker in chunkers:
        chunks = chunker.chunk(document)
        contained, overlapped = set(), set()
        for c in chunks:
            assert document[c.start : c.end] == c.text
            if not has_empty_spans:
                assert index.count_in(c.start, c.end) <= limit
            contained.update(index.tokens_in(c.start, c.end))
            overlapped.update(index.tokens_overlapping(c.start, c.end))
        if isinstance(chunker, FixedTokenChunker):
            assert contained == set(range(len(index)))
        assert content_tokens <= overlapped, type(chunker).__name__
        starts = [c.start for c in chunks]
        assert starts == sorted(starts)

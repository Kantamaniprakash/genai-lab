"""Chunking strategies that preserve exact character offsets.

Every chunker returns `Chunk` objects whose (start, end) range satisfies
``document[start:end] == chunk.text``. This invariant is what makes span-level
evaluation possible later: gold answer spans live in document coordinates, so
retrieval quality can be scored as token overlap between retrieved chunks and
gold spans without any fuzzy matching.

Implemented strategies (structural family):

- `FixedTokenChunker`  — sliding window of N tokens with configurable overlap;
  the de-facto default in production RAG stacks and the baseline in Chroma's
  chunking evaluation (Smith & Troynikov, 2024).
- `SentenceChunker`    — greedy packing of whole sentences up to a token
  budget, with optional sentence-level overlap.
- `RecursiveCharacterChunker` — splits on a separator hierarchy
  (paragraph > line > space), then greedily merges adjacent pieces under the
  budget; mirrors the semantics of LangChain's RecursiveCharacterTextSplitter.

Semantic (embedding-driven) chunkers are added in a later phase; they plug in
via the same `Chunker` protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .tokenization import RegexWordTokenizer, TokenIndex, Tokenizer


@dataclass(frozen=True)
class Chunk:
    """A contiguous piece of a document, addressed in document coordinates."""

    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not (0 <= self.start <= self.end):
            raise ValueError(f"invalid chunk range [{self.start}, {self.end})")


class Chunker(Protocol):
    def chunk(self, document: str) -> list[Chunk]:
        ...


def _make_chunk(document: str, start: int, end: int) -> Chunk:
    return Chunk(text=document[start:end], start=start, end=end)


class FixedTokenChunker:
    """Sliding window of `chunk_size` tokens advancing by `chunk_size - overlap`.

    Chunk boundaries are token-aligned: each chunk spans from the first
    character of its first token to the last character of its last token, so
    leading/trailing whitespace never pads a chunk. The final window may hold
    fewer than `chunk_size` tokens.
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int = 0,
        tokenizer: Tokenizer | None = None,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tokenizer or RegexWordTokenizer()

    def chunk(self, document: str) -> list[Chunk]:
        spans = self.tokenizer.spans(document)
        if not spans:
            return []
        stride = self.chunk_size - self.overlap
        chunks = []
        for i in range(0, len(spans), stride):
            window = spans[i : i + self.chunk_size]
            chunks.append(_make_chunk(document, window[0].start, window[-1].end))
            if i + self.chunk_size >= len(spans):
                break
        return chunks


# Sentence boundaries: terminal punctuation (optionally followed by closing
# quotes/brackets), then whitespace, then an upper-case letter, digit, or
# opening quote/bracket — or any newline run. Deliberately simple and
# deterministic; known failure modes (abbreviations like "Dr.", initials)
# are documented in the project README.
_SENT_BOUNDARY_RE = re.compile(r'[.!?]["\')\]]*\s+(?=[A-Z0-9"\'(\[])|\n+')


def split_sentences(document: str) -> list[tuple[int, int]]:
    """Character ranges of sentences, whitespace-trimmed, in document order."""
    boundaries = [0]
    for m in _SENT_BOUNDARY_RE.finditer(document):
        boundaries.append(m.end())
    boundaries.append(len(document))
    ranges = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        piece = document[lo:hi]
        stripped = piece.strip()
        if not stripped:
            continue
        left = lo + (len(piece) - len(piece.lstrip()))
        ranges.append((left, left + len(stripped)))
    return ranges


class SentenceChunker:
    """Greedily pack whole sentences into chunks of at most `max_tokens`.

    A single sentence longer than the budget is split by a token window so the
    budget is a hard guarantee. `overlap_sentences` makes each chunk restart
    that many sentences before the previous chunk ended.
    """

    def __init__(
        self,
        max_tokens: int,
        overlap_sentences: int = 0,
        tokenizer: Tokenizer | None = None,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.tokenizer = tokenizer or RegexWordTokenizer()
        self._window = FixedTokenChunker(max_tokens, tokenizer=self.tokenizer)

    def chunk(self, document: str) -> list[Chunk]:
        sentences = split_sentences(document)
        if not sentences:
            return []
        index = TokenIndex(document, self.tokenizer)
        counts = [index.count_in(lo, hi) for lo, hi in sentences]
        chunks: list[Chunk] = []
        i = 0
        while i < len(sentences):
            if counts[i] > self.max_tokens:
                lo, hi = sentences[i]
                for sub in self._window.chunk(document[lo:hi]):
                    chunks.append(_make_chunk(document, lo + sub.start, lo + sub.end))
                i += 1
                continue
            total, j = 0, i
            while j < len(sentences) and total + counts[j] <= self.max_tokens:
                total += counts[j]
                j += 1
            chunks.append(
                _make_chunk(document, sentences[i][0], sentences[j - 1][1])
            )
            # Overlap rewinds the start, but never so far that we stop advancing.
            i = max(j - self.overlap_sentences, i + 1)
        return chunks


class RecursiveCharacterChunker:
    """Split on a separator hierarchy, then merge adjacent pieces greedily.

    A piece over budget is re-split with the next separator in the hierarchy;
    the empty-string separator is a token-window fallback, so `max_tokens` is
    a hard guarantee. Adjacent leaf pieces are then merged left-to-right while
    the merged character range stays within `max_tokens` (token count of a
    merged range is measured on the actual document substring, separators
    included). This mirrors LangChain's RecursiveCharacterTextSplitter but
    keeps exact document offsets.
    """

    DEFAULT_SEPARATORS = ("\n\n", "\n", " ")

    def __init__(
        self,
        max_tokens: int,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
        tokenizer: Tokenizer | None = None,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if any(not s for s in separators):
            raise ValueError("separators must be non-empty strings")
        self.max_tokens = max_tokens
        self.separators = separators
        self.tokenizer = tokenizer or RegexWordTokenizer()
        self._window = FixedTokenChunker(max_tokens, tokenizer=self.tokenizer)

    def chunk(self, document: str) -> list[Chunk]:
        index = TokenIndex(document, self.tokenizer)
        if len(index) == 0:
            return []
        leaves = self._split(document, 0, len(document), index, 0)
        return self._merge(document, leaves, index)

    def _split(
        self, document: str, lo: int, hi: int, index: TokenIndex, level: int
    ) -> list[tuple[int, int]]:
        if index.count_in(lo, hi) <= self.max_tokens:
            return [(lo, hi)] if index.count_in(lo, hi) > 0 else []
        if level >= len(self.separators):
            # Token-window fallback for a piece with no usable separators.
            return [
                (lo + c.start, lo + c.end)
                for c in self._window.chunk(document[lo:hi])
            ]
        sep = self.separators[level]
        cuts = [lo]
        pos = document.find(sep, lo)
        # A separator straddling the piece boundary counts as not found.
        while pos != -1 and pos + len(sep) <= hi:
            cuts.append(pos + len(sep))
            pos = document.find(sep, pos + len(sep))
        cuts.append(hi)
        pieces = [(a, b) for a, b in zip(cuts, cuts[1:]) if a < b]
        if len(pieces) == 1:
            return self._split(document, lo, hi, index, level + 1)
        leaves = []
        for a, b in pieces:
            leaves.extend(self._split(document, a, b, index, level + 1))
        return leaves

    def _merge(
        self, document: str, leaves: list[tuple[int, int]], index: TokenIndex
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        cur = None
        for lo, hi in leaves:
            if index.count_in(lo, hi) == 0:
                continue
            if cur is None:
                cur = (lo, hi)
            elif index.count_in(cur[0], hi) <= self.max_tokens:
                cur = (cur[0], hi)
            else:
                chunks.append(self._trimmed(document, index, *cur))
                cur = (lo, hi)
        if cur is not None:
            chunks.append(self._trimmed(document, index, *cur))
        return chunks

    @staticmethod
    def _trimmed(document: str, index: TokenIndex, lo: int, hi: int) -> Chunk:
        """Shrink a range to token-aligned edges so chunks never carry
        leading/trailing separator characters."""
        tokens = index.tokens_in(lo, hi)
        first, last = index.spans[tokens[0]], index.spans[tokens[-1]]
        return _make_chunk(document, first.start, last.end)

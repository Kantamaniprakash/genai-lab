"""Deterministic tokenization with character offsets.

Every downstream component (chunkers, span-level metrics, budget-matched
retrieval) counts tokens with the same tokenizer, so the choice only needs to
be consistent and reproducible, not identical to any particular model's BPE.
The default is a regex word/punctuation tokenizer: it requires no downloaded
vocabulary, is stable across platforms and library versions, and its counts
correlate strongly with BPE counts on English prose. The `Tokenizer` protocol
lets a BPE tokenizer slot in as a robustness check without touching the
chunkers or metrics — `TiktokenTokenizer` below is that check.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TokenSpan:
    """A token's half-open character range [start, end) in the source text."""

    start: int
    end: int


class Tokenizer(Protocol):
    def spans(self, text: str) -> list[TokenSpan]:
        """Return token spans in order; spans must be non-overlapping."""
        ...

    def count(self, text: str) -> int:
        ...


class RegexWordTokenizer:
    """Tokens are maximal word runs (``\\w+``) or single punctuation marks.

    Whitespace is never part of a token. Deterministic, dependency-free.
    """

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def spans(self, text: str) -> list[TokenSpan]:
        return [TokenSpan(m.start(), m.end()) for m in self._TOKEN_RE.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in self._TOKEN_RE.finditer(text))


class TiktokenTokenizer:
    """BPE token spans from a tiktoken encoding (default ``cl100k_base``).

    Slots into the ``Tokenizer`` protocol so chunk sizes, budgets, and span
    metrics can all be counted in real model-tokenizer units — the robustness
    check the regex default was designed to allow. Two representational
    differences from ``RegexWordTokenizer`` are inherent to BPE and worth
    knowing when reading results:

    - BPE tokens carry their leading whitespace (`` the`` is one token), so
      token spans tile the entire document instead of skipping whitespace,
      and chunk boundaries sit just before the space that precedes a word.
    - tiktoken operates on UTF-8 *bytes*, so a token boundary can fall inside
      a multi-byte character. Spans are recovered by mapping each token's
      byte range back to character offsets; a character straddling a token
      boundary belongs to the token in which its byte sequence *ends*. This
      keeps spans non-overlapping and ordered. A token lying entirely inside
      one character (e.g. a continuation byte of an emoji) gets an empty
      span at that position — it still counts toward budgets, since a
      generator is charged for it, but covers no characters.

    Special-token text (e.g. ``<|endoftext|>``) is tokenized as ordinary
    text, never as a control token — documents are data, not prompts.

    Requires the ``tiktoken`` package; constructing an encoding downloads
    and caches its vocabulary on first use.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken  # deferred so the core stays importable without it

        self._enc = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    def spans(self, text: str) -> list[TokenSpan]:
        ids = self._enc.encode(text, disallowed_special=())
        token_bytes = self._enc.decode_tokens_bytes(ids)
        if text.isascii():  # byte offsets are character offsets
            spans, pos = [], 0
            for tok in token_bytes:
                spans.append(TokenSpan(pos, pos + len(tok)))
                pos += len(tok)
            return spans
        # Cumulative byte offset after each character; a byte boundary maps
        # to the number of characters that end at or before it.
        char_byte_ends: list[int] = []
        pos = 0
        for ch in text:
            pos += len(ch.encode("utf-8"))
            char_byte_ends.append(pos)
        spans = []
        byte_pos = 0
        char_pos = 0
        for tok in token_bytes:
            byte_pos += len(tok)
            char_end = bisect.bisect_right(char_byte_ends, byte_pos)
            spans.append(TokenSpan(char_pos, char_end))
            char_pos = char_end
        return spans

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


class TokenIndex:
    """Precomputed token spans for one document, with fast range queries.

    Chunkers and metrics repeatedly ask "how many tokens fall inside this
    character range?"; recomputing the regex per query would make greedy
    merging quadratic in document length. This index answers in O(log n).
    """

    def __init__(self, text: str, tokenizer: Tokenizer):
        self.text = text
        self.spans = tokenizer.spans(text)
        self._starts = [s.start for s in self.spans]
        self._ends = [s.end for s in self.spans]

    def __len__(self) -> int:
        return len(self.spans)

    def tokens_in(self, start: int, end: int) -> range:
        """Indices of tokens fully contained in the character range [start, end)."""
        lo = bisect.bisect_left(self._starts, start)
        hi = bisect.bisect_right(self._ends, end)
        return range(lo, max(lo, hi))

    def count_in(self, start: int, end: int) -> int:
        return len(self.tokens_in(start, end))

    def tokens_overlapping(self, start: int, end: int) -> range:
        """Indices of tokens overlapping [start, end) by at least one character.

        Chunk boundaries are token-aligned, but gold answer spans need not be;
        scoring uses overlap so a span that starts or ends mid-token still
        claims that token.
        """
        lo = bisect.bisect_right(self._ends, start)
        hi = bisect.bisect_left(self._starts, end)
        return range(lo, max(lo, hi))

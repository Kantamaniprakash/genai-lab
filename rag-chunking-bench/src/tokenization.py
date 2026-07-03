"""Deterministic tokenization with character offsets.

Every downstream component (chunkers, span-level metrics, budget-matched
retrieval) counts tokens with the same tokenizer, so the choice only needs to
be consistent and reproducible, not identical to any particular model's BPE.
The default is a regex word/punctuation tokenizer: it requires no downloaded
vocabulary (this environment cannot reach BPE vocab hosts), is stable across
platforms, and its counts correlate strongly with BPE counts on English prose.
The `Tokenizer` protocol lets a BPE tokenizer slot in later without touching
the chunkers or metrics.
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

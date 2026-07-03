"""Retrievers that rank chunks for a query.

The benchmark measures chunking effects *holding the retriever fixed*, so
retrievers only need to be strong, deterministic baselines. BM25 is
implemented from scratch (Okapi variant with Lucene's non-negative IDF) both
to avoid a dependency and so the scoring function is exactly the one written
in the README; TF-IDF/LSA follow in a later session via scikit-learn.

Ranking is fully deterministic: ties break by ascending chunk index, so runs
are reproducible bit-for-bit across machines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol


class Retriever(Protocol):
    def fit(self, texts: list[str]) -> "Retriever":
        ...

    def rank(self, query: str) -> list[int]:
        """Indices into the fitted texts, best match first."""
        ...


# Retrieval tokenization is intentionally separate from budget tokenization:
# lowercased word runs only, no punctuation tokens (punctuation carries no
# lexical matching signal and would dilute document lengths).
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def query_terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 (Robertson et al.) with Lucene-style smoothed IDF.

    score(q, d) = sum over unique query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    The +1 inside the log keeps IDF non-negative for terms appearing in more
    than half the corpus (Lucene's fix to the classic Robertson IDF).
    Repeated query terms are counted once, matching Lucene/rank_bm25.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def fit(self, texts: list[str]) -> "BM25Retriever":
        if not texts:
            raise ValueError("cannot fit on an empty corpus")
        self._tfs = [Counter(query_terms(t)) for t in texts]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avgdl = sum(self._lengths) / len(self._lengths) or 1.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(texts)
        self._idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }
        return self

    def scores(self, query: str) -> list[float]:
        if not self._tfs:
            raise ValueError("fit() must be called before scoring")
        terms = set(query_terms(query))
        out = []
        for tf, length in zip(self._tfs, self._lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self._avgdl)
            score = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    score += self._idf[term] * f * (self.k1 + 1) / (f + norm)
            out.append(score)
        return out

    def rank(self, query: str) -> list[int]:
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

"""Retrievers that rank chunks for a query.

The benchmark measures chunking effects *holding the retriever fixed*, so
retrievers only need to be strong, deterministic baselines. BM25 is
implemented from scratch (Okapi variant with Lucene's non-negative IDF) both
to avoid a dependency and so the scoring function is exactly the one written
in the README; TF-IDF cosine and LSA use scikit-learn (pinned) and share the
same retrieval tokenizer, so the three retrievers differ only in scoring.

Ranking is fully deterministic: ties break by ascending chunk index, and LSA
uses the ARPACK solver with a fixed seed, so runs are reproducible
bit-for-bit across machines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class Retriever(Protocol):
    def fit(self, texts: list[str]) -> Retriever:
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

    def fit(self, texts: list[str]) -> BM25Retriever:
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


def _fit_tfidf(texts: list[str]) -> tuple[TfidfVectorizer | None, np.ndarray | None]:
    """Fit scikit-learn TF-IDF on a corpus; ``(None, None)`` for empty vocabulary.

    Conventions (scikit-learn defaults, stated here because the numbers in the
    hand-computed tests depend on them): raw term counts, smoothed IDF
    ``ln((1 + N) / (1 + df)) + 1``, L2-normalized rows. Tokenization is
    ``query_terms``, shared with BM25. A corpus whose documents contain no
    word tokens at all (pure punctuation) has an empty vocabulary; callers
    treat that as "every score is zero" rather than an error, matching BM25.
    """
    vectorizer = TfidfVectorizer(analyzer=query_terms)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # "empty vocabulary" — no word token in any document
        return None, None
    return vectorizer, matrix


class TfidfRetriever:
    """TF-IDF cosine similarity over chunk term vectors.

    Both the corpus rows and the query vector are L2-normalized by the
    vectorizer, so scores are cosine similarities in [0, 1]; a query with no
    in-vocabulary term scores zero everywhere and ``rank`` falls back to the
    index order (same tie convention as BM25).
    """

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._n_docs = 0

    def fit(self, texts: list[str]) -> TfidfRetriever:
        if not texts:
            raise ValueError("cannot fit on an empty corpus")
        self._n_docs = len(texts)
        self._vectorizer, self._matrix = _fit_tfidf(texts)
        return self

    def scores(self, query: str) -> list[float]:
        if not self._n_docs:
            raise ValueError("fit() must be called before scoring")
        if self._vectorizer is None:
            return [0.0] * self._n_docs
        q = self._vectorizer.transform([query])
        return (self._matrix @ q.T).toarray().ravel().tolist()

    def rank(self, query: str) -> list[int]:
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


class LSARetriever:
    """Latent semantic analysis: cosine similarity in a TruncatedSVD space.

    The TF-IDF matrix (same conventions as ``TfidfRetriever``) is decomposed
    with ARPACK (``random_state`` fixed; the solver requires the rank to be
    strictly below both matrix dimensions), and the query is folded into the
    latent space with the same projection. The realized rank is
    ``min(n_components, n_docs - 1, n_terms - 1)`` — on per-document chunk
    collections the ``n_docs - 1`` bound frequently binds, in which case the
    decomposition is (nearly) full-rank and LSA degenerates toward plain
    TF-IDF; the experiment summaries report where the bottleneck actually
    binds. Latent cosines can be negative; only the ranking is consumed.
    Degenerate corpora (a single chunk, or a single distinct term) have no
    usable latent space and score zero everywhere, preserving index order.
    """

    def __init__(self, n_components: int = 64) -> None:
        if n_components < 1:
            raise ValueError("require n_components >= 1")
        self.n_components = n_components
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._latent: np.ndarray | None = None
        self._n_docs = 0

    @property
    def realized_rank(self) -> int:
        """Rank actually used after the data-size caps (0 = degenerate fit)."""
        return self._svd.n_components if self._svd is not None else 0

    def fit(self, texts: list[str]) -> LSARetriever:
        if not texts:
            raise ValueError("cannot fit on an empty corpus")
        self._n_docs = len(texts)
        self._vectorizer, matrix = _fit_tfidf(texts)
        self._svd = None
        self._latent = None
        if self._vectorizer is None:
            return self
        rank = min(self.n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if rank < 1:
            self._vectorizer = None
            return self
        self._svd = TruncatedSVD(n_components=rank, algorithm="arpack", random_state=0)
        latent = self._svd.fit_transform(matrix)
        norms = np.linalg.norm(latent, axis=1, keepdims=True)
        self._latent = latent / np.where(norms > 0, norms, 1.0)
        return self

    def scores(self, query: str) -> list[float]:
        if not self._n_docs:
            raise ValueError("fit() must be called before scoring")
        if self._vectorizer is None:
            return [0.0] * self._n_docs
        q = self._svd.transform(self._vectorizer.transform([query]))[0]
        norm = np.linalg.norm(q)
        if norm == 0:
            return [0.0] * self._n_docs
        return (self._latent @ (q / norm)).tolist()

    def rank(self, query: str) -> list[int]:
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

"""Dense retrieval with a small sentence-transformer encoder.

This module is the only place the benchmark touches torch, and it is
deliberately optional: install the ``dense`` dependency group to use it
(``uv sync --group dense``). Everything else in the benchmark stays
dependency-light and bit-reproducible without it.

Two reproducibility caveats the lexical retrievers do not have, stated up
front because the results README leans on them:

- **Determinism is per-environment, not portable.** Inference on a fixed
  machine, torch build, and thread count reproduces scores exactly (verified
  in tests by refitting), but floating-point reductions may differ across
  BLAS builds, so dense numbers are documented together with the recorded
  torch / sentence-transformers versions in each result file's metadata.
- **The encoder truncates.** ``all-MiniLM-L6-v2`` reads at most
  ``max_seq_length`` wordpieces (256, specials included); longer chunks are
  scored by their prefix only. That is exactly what happens in production
  when large chunks meet a small embedding model, so instead of hiding it
  the retriever counts affected chunks (``n_truncated``) and the runner
  persists the exposure per configuration.
"""

from __future__ import annotations

import os

import numpy as np

# Silence the HF fast-tokenizer fork warning; the benchmark is single-process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEncoder:
    """Process-wide memoizing wrapper around a SentenceTransformer model.

    The grid runner builds a fresh retriever per (configuration, document),
    but model loading (~seconds) and query encoding must not be repeated
    thousands of times, so instances share nothing while ``default_encoder``
    returns one process-wide encoder whose memo makes every distinct text —
    chunk or query — cost exactly one forward pass per process. Embeddings
    are L2-normalized float32, so dot products are cosine similarities.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 64):
        if batch_size < 1:
            raise ValueError("require batch_size >= 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._vectors: dict[str, np.ndarray] = {}
        self._token_counts: dict[str, int] = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    @property
    def max_seq_length(self) -> int:
        """Wordpiece window (specials included) beyond which input is cut."""
        return int(self._load().max_seq_length)

    def token_count(self, text: str) -> int:
        """Untruncated wordpiece count of ``text``, specials included."""
        cached = self._token_counts.get(text)
        if cached is None:
            ids = self._load().tokenizer(text, add_special_tokens=True)["input_ids"]
            cached = self._token_counts[text] = len(ids)
        return cached

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embeddings for ``texts`` (rows aligned with input, memoized)."""
        misses: list[str] = []
        seen: set[str] = set()
        for text in texts:
            if text not in self._vectors and text not in seen:
                misses.append(text)
                seen.add(text)
        if misses:
            model = self._load()
            vectors = model.encode(
                misses,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            for text, vector in zip(misses, vectors, strict=True):
                self._vectors[text] = vector
        return np.stack([self._vectors[text] for text in texts])


_DEFAULT_ENCODER: SentenceTransformerEncoder | None = None


def default_encoder() -> SentenceTransformerEncoder:
    """The shared process-wide encoder (created on first use)."""
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        _DEFAULT_ENCODER = SentenceTransformerEncoder()
    return _DEFAULT_ENCODER


class DenseRetriever:
    """Cosine similarity between sentence-transformer embeddings.

    Same contract and tie convention as the lexical retrievers: ``rank``
    orders chunk indices by descending score with ties broken by ascending
    index. ``n_truncated`` counts fitted chunks longer than the encoder
    window — those are scored by prefix, and the runner reports the exposure
    so no reader mistakes prefix retrieval for whole-chunk retrieval.
    """

    def __init__(self, encoder: SentenceTransformerEncoder | None = None):
        self._encoder = encoder if encoder is not None else default_encoder()
        self._embeddings: np.ndarray | None = None
        self.n_texts = 0
        self.n_truncated = 0

    @property
    def model_name(self) -> str:
        return self._encoder.model_name

    @property
    def max_seq_length(self) -> int:
        return self._encoder.max_seq_length

    def fit(self, texts: list[str]) -> DenseRetriever:
        if not texts:
            raise ValueError("cannot fit on an empty corpus")
        self._embeddings = self._encoder.encode(texts)
        self.n_texts = len(texts)
        window = self._encoder.max_seq_length
        self.n_truncated = sum(
            1 for t in texts if self._encoder.token_count(t) > window
        )
        return self

    def scores(self, query: str) -> list[float]:
        if self._embeddings is None:
            raise ValueError("fit() must be called before scoring")
        q = self._encoder.encode([query])[0]
        return (self._embeddings @ q).tolist()

    def rank(self, query: str) -> list[int]:
        scores = self.scores(query)
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))

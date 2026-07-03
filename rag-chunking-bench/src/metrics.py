"""Span-level retrieval metrics under a token budget.

The protocol (README, "Evaluation protocol"): ranked chunks are accumulated
until the token budget B is exhausted, then scored as token-set overlap
against the gold spans of the question's best-matching alternative.

Two accounting decisions worth making explicit:

- **Budget charges prompt tokens, not unique tokens.** Retrieved chunks are
  what a generator would receive concatenated, duplicates included — so each
  chunk costs its own token count even when overlap-configured chunkers
  retrieve overlapping text. Scoring, by contrast, uses the *union* of
  retrieved token indices: reading the same gold token twice does not double
  recall. This is exactly the mechanism by which budget matching penalizes
  redundant overlap.
- **Gold tokens are counted by overlap, not containment.** Chunk boundaries
  are token-aligned but gold answer spans need not be; a span starting
  mid-token still claims that token (``TokenIndex.tokens_overlapping``).

All metrics are computed per question; aggregation is a mean plus a paired
bootstrap confidence interval over questions (fixed seed), so every "A beats
B" claim ships with an interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunkers import Chunk
from .data import GoldSpan
from .tokenization import TokenIndex


@dataclass(frozen=True)
class SpanScores:
    recall: float
    precision: float
    iou: float


def take_until_budget(
    ranked: list[Chunk], index: TokenIndex, budget: int
) -> list[Chunk]:
    """Prefix of `ranked` whose summed token counts fit within `budget`.

    Accumulation stops at the first chunk that would exceed the budget
    (stop-before-exceed); later, smaller chunks are not pulled forward, since
    that would change the retriever's ranking.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    taken: list[Chunk] = []
    used = 0
    for chunk in ranked:
        cost = index.count_in(chunk.start, chunk.end)
        if used + cost > budget:
            break
        taken.append(chunk)
        used += cost
    return taken


def retrieved_token_set(chunks: list[Chunk], index: TokenIndex) -> set[int]:
    """Union of document token indices covered by the chunks."""
    tokens: set[int] = set()
    for chunk in chunks:
        tokens.update(index.tokens_overlapping(chunk.start, chunk.end))
    return tokens


def gold_token_set(spans: tuple[GoldSpan, ...], index: TokenIndex) -> set[int]:
    tokens: set[int] = set()
    for span in spans:
        tokens.update(index.tokens_overlapping(span.start, span.end))
    return tokens


def span_scores(
    chunks: list[Chunk],
    gold_alternatives: tuple[tuple[GoldSpan, ...], ...],
    index: TokenIndex,
) -> SpanScores:
    """Token-overlap recall/precision/IoU against the best gold alternative.

    Each metric independently takes the max over alternatives (the SQuAD
    max-over-answers convention). With zero retrieved tokens all metrics are
    0 by convention.
    """
    retrieved = retrieved_token_set(chunks, index)
    if not retrieved:
        return SpanScores(recall=0.0, precision=0.0, iou=0.0)
    best = SpanScores(recall=0.0, precision=0.0, iou=0.0)
    for alternative in gold_alternatives:
        gold = gold_token_set(alternative, index)
        if not gold:
            raise ValueError("gold alternative covers no tokens")
        hit = len(retrieved & gold)
        best = SpanScores(
            recall=max(best.recall, hit / len(gold)),
            precision=max(best.precision, hit / len(retrieved)),
            iou=max(best.iou, hit / len(retrieved | gold)),
        )
    return best


def hit_at_k(
    ranked: list[Chunk],
    gold_alternatives: tuple[tuple[GoldSpan, ...], ...],
    index: TokenIndex,
    k: int,
) -> bool:
    """Whether any top-k chunk overlaps any gold span (classic-style hit rate).

    Reported for comparability with fixed-k evaluations in prior work; the
    budget-matched span metrics above are the primary measurements.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    for chunk in ranked[:k]:
        ctoks = index.tokens_overlapping(chunk.start, chunk.end)
        for alternative in gold_alternatives:
            for span in alternative:
                gtoks = index.tokens_overlapping(span.start, span.end)
                # Both are contiguous index ranges; overlap is an
                # interval intersection.
                if max(ctoks.start, gtoks.start) < min(ctoks.stop, gtoks.stop):
                    return True
    return False


@dataclass(frozen=True)
class BootstrapResult:
    """Mean of per-question paired differences with a percentile 95% CI."""

    mean_diff: float
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Percentile bootstrap CI for mean(A - B) over paired per-question scores.

    Pairing matters: both systems are scored on the same questions, so
    resampling question indices (not independent score sets) removes
    between-question variance from the comparison.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("paired score lists must have equal length")
    if not scores_a:
        raise ValueError("cannot bootstrap zero questions")
    diffs = np.asarray(scores_a, dtype=np.float64) - np.asarray(
        scores_b, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    resampled_means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(resampled_means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        mean_diff=float(diffs.mean()), ci_low=float(lo), ci_high=float(hi)
    )

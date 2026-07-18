"""Trivial judge floors: always-A, longer-response, random.

Every judge accuracy in this audit is read against these floors. They are
defined at the same granularity as the judge analyses — per item, per
presentation order — so they drop into the same accuracy computations and the
same paired bootstrap as any model. All are deterministic functions of the
item and order; "random" needs no simulation because its expected accuracy is
exactly 1/2 on a benchmark whose gold pairs are one-chosen-one-rejected.

Correctness values are in {0.0, 0.5, 1.0}: ties (equal lengths for the
longer-response heuristic) score half, which equals the expected accuracy of
breaking the tie by coin flip without adding a noise axis.
"""

from __future__ import annotations

from .data import PairItem
from .prompts import ORDERS, Order, build_judge_prompt


def always_a_correct(item: PairItem, order: Order) -> float:
    """Accuracy of the judge that answers "A" regardless of content.

    Exactly the position-bias floor: 1.0 when the gold-chosen response sits
    in position A, else 0.0; averages to 1/2 over the exhaustive order pair.
    """
    return 1.0 if build_judge_prompt(item, order).expected_verdict == "A" else 0.0


def longer_response_correct(item: PairItem, unit: str = "chars") -> float:
    """Accuracy of "pick the longer response" — order-invariant by construction.

    ``unit`` is "chars" (canonical) or "words" (whitespace tokens); both are
    tokenizer-free so the floor does not depend on any judge's vocabulary.
    """
    if unit == "chars":
        len_chosen, len_rejected = len(item.chosen), len(item.rejected)
    elif unit == "words":
        len_chosen, len_rejected = len(item.chosen.split()), len(item.rejected.split())
    else:
        raise ValueError(f"unknown length unit {unit!r}")
    if len_chosen == len_rejected:
        return 0.5
    return 1.0 if len_chosen > len_rejected else 0.0


RANDOM_ACCURACY = 0.5


def summarize_baselines(items: tuple[PairItem, ...]) -> dict:
    """Floor accuracies over a sample, overall and per category."""
    def block(subset: tuple[PairItem, ...]) -> dict:
        n = len(subset)
        return {
            "n_items": n,
            "always_a_raw": sum(
                always_a_correct(it, order) for it in subset for order in ORDERS
            ) / (2 * n),
            "longer_chars": sum(longer_response_correct(it, "chars") for it in subset) / n,
            "longer_words": sum(longer_response_correct(it, "words") for it in subset) / n,
            "random": RANDOM_ACCURACY,
        }

    categories = sorted({it.category for it in items})
    return {
        "overall": block(items),
        "by_category": {
            cat: block(tuple(it for it in items if it.category == cat))
            for cat in categories
        },
    }

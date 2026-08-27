"""Tests for the paired rubric-sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import SwapPair
from src.rubric_pair import (
    RubricPair,
    match_rubric_pairs,
    rubric_pair_view,
)


def sp(item_id: str, z_cf: float, z_rf: float, *, compliant: bool = True,
       mass: float = 0.95) -> SwapPair:
    return SwapPair(item_id=item_id, z_cf=z_cf, z_rf=z_rf,
                    compliant_both=compliant, mass_min=mass)


def test_rubric_pair_deltas_and_agreement():
    # minimal: s=1, b=2; detailed: s=-1, b=0.5 -> verdict flips, bias shrinks.
    pair = RubricPair(item_id="i/1", a=sp("i/1", 3.0, 1.0), b=sp("i/1", -0.5, 1.5))
    assert pair.delta_s == pytest.approx(-2.0)
    assert pair.delta_b == pytest.approx(-1.5)
    assert pair.delta_abs_b == pytest.approx(-1.5)
    assert pair.delta_abs_s == pytest.approx(0.0)
    assert pair.sym_verdict_agrees == 0.0

    same = RubricPair(item_id="i/2", a=sp("i/2", 3.0, 1.0), b=sp("i/2", 5.0, 1.0))
    assert same.sym_verdict_agrees == 1.0

    tie = RubricPair(item_id="i/3", a=sp("i/3", 1.0, 1.0), b=sp("i/3", 2.0, 0.0))
    assert tie.a.s == 0.0
    assert tie.sym_verdict_agrees == 0.5


def test_rubric_pair_rejects_mismatched_items():
    with pytest.raises(ValueError, match="mismatched item_ids"):
        RubricPair(item_id="i/1", a=sp("i/1", 1.0, 0.0), b=sp("i/2", 1.0, 0.0))


def test_match_rubric_pairs_inner_join_counts():
    pairs_a = [sp("i/1", 1.0, 0.0), sp("i/2", 1.0, 0.0), sp("i/3", 1.0, 0.0)]
    pairs_b = [sp("i/2", 2.0, 0.0), sp("i/3", 2.0, 0.0), sp("i/4", 2.0, 0.0)]
    matched, only_a, only_b = match_rubric_pairs(pairs_a, pairs_b)
    assert [m.item_id for m in matched] == ["i/2", "i/3"]
    assert (only_a, only_b) == (1, 1)
    # Sides are oriented: a from pairs_a, b from pairs_b.
    assert matched[0].a.z_cf == 1.0 and matched[0].b.z_cf == 2.0


def test_match_rubric_pairs_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate item_ids"):
        match_rubric_pairs([sp("i/1", 1.0, 0.0), sp("i/1", 2.0, 0.0)], [])


def make_field(n: int = 40, seed: int = 7) -> tuple[list[SwapPair], list[SwapPair]]:
    """Synthetic judge: rubric B halves the bias and adds +0.5 preference."""
    rng = np.random.default_rng(seed)
    pairs_a, pairs_b = [], []
    for i in range(n):
        s = float(rng.normal(0.8, 1.0))
        b = float(rng.normal(2.0, 0.5))
        item = f"sub{i % 2}/{i}"
        pairs_a.append(sp(item, b + s, b - s, compliant=i % 4 != 0))
        s2, b2 = s + 0.5, b / 2
        pairs_b.append(sp(item, b2 + s2, b2 - s2, compliant=True))
    return pairs_a, pairs_b


def test_rubric_pair_view_recovers_construction():
    pairs_a, pairs_b = make_field()
    view = rubric_pair_view(pairs_a, pairs_b, "minimal", "detailed",
                            n_boot=500, seed=0)
    assert view["n_matched_items"] == 40
    assert view["n_only_a"] == 0 and view["n_only_b"] == 0
    # Constructed effects: delta_s = +0.5 exactly, delta_b = -b/2 (mean ~ -1).
    assert view["deltas"]["s"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert view["deltas"]["b"]["mean"] == pytest.approx(
        -np.mean([p.b for p in pairs_a]) / 2, abs=1e-9
    )
    lo, hi = view["deltas"]["abs_b"]["ci95"]
    assert hi < 0  # bias magnitude significantly shrinks by construction
    # Compliance: A compliant on 3/4 of items, B always -> delta +0.25.
    assert view["deltas"]["compliance"]["mean"] == pytest.approx(0.25)
    # s only shifts by a constant -> high cross-rubric correlation.
    assert view["consistency"]["s_pearson"]["r"] > 0.99
    assert view["consistency"]["s_spearman"] > 0.99
    r_lo, r_hi = view["consistency"]["s_pearson"]["ci95"]
    assert r_lo <= view["consistency"]["s_pearson"]["r"] <= r_hi


def test_rubric_pair_view_flip_rate_and_categories():
    pairs_a, pairs_b = make_field()
    flips = sum(
        (a.s > 0) != (b.s > 0) for a, b in zip(pairs_a, pairs_b, strict=True)
    )
    view = rubric_pair_view(
        pairs_a, pairs_b, "minimal", "detailed",
        category_of=lambda item_id: item_id.split("/", 1)[0],
        n_boot=200, seed=0,
    )
    assert view["consistency"]["rubric_flip_rate"]["mean"] == pytest.approx(flips / 40)
    assert set(view["by_category"]) == {"sub0", "sub1"}
    cat = view["by_category"]["sub0"]
    assert cat["n_items"] == 20
    assert cat["sym_acc_a"] == pytest.approx(
        np.mean([p.sym_correct for p in pairs_a[::2]])
    )
    # Category flip rates aggregate back to the overall rate.
    total = sum(
        v["rubric_flip_rate"] * v["n_items"] for v in view["by_category"].values()
    )
    assert total / 40 == pytest.approx(view["consistency"]["rubric_flip_rate"]["mean"])


def test_rubric_pair_view_deterministic_and_minimum_n():
    pairs_a, pairs_b = make_field(n=6)
    v1 = rubric_pair_view(pairs_a, pairs_b, "m", "d", n_boot=300, seed=1)
    v2 = rubric_pair_view(pairs_a, pairs_b, "m", "d", n_boot=300, seed=1)
    assert v1 == v2
    with pytest.raises(ValueError, match="matched items"):
        rubric_pair_view(pairs_a[:2], pairs_b[:2], "m", "d", n_boot=100)

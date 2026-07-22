"""Tests for the calibration view: folding, equal-mass bins, ECE, bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis import SwapPair
from src.calibration import (
    calibration_view,
    ece,
    equal_mass_bins,
    fold_raw,
    fold_sym,
    reliability_curve,
)


def make_pair(item_id: str, z_cf: float, z_rf: float) -> SwapPair:
    return SwapPair(item_id=item_id, z_cf=z_cf, z_rf=z_rf,
                    compliant_both=True, mass_min=0.95)


def test_fold_sym_confidence_and_ties():
    pairs = [make_pair("i/1", 3.0, 1.0), make_pair("i/2", 1.0, 1.0)]
    conf, correct = fold_sym(pairs)
    # s = (z_cf - z_rf)/2 = 1.0 -> confidence sigmoid(1), correct.
    assert conf[0] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert correct[0] == 1.0
    # s = 0: indifferent -> confidence exactly 1/2, half credit.
    assert conf[1] == pytest.approx(0.5)
    assert correct[1] == 0.5


def test_fold_raw_two_points_per_item():
    pairs = [make_pair("i/1", 2.0, -1.0)]
    conf, correct = fold_raw(pairs)
    assert len(conf) == 2
    assert conf[0] == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert correct.tolist() == [1.0, 1.0]  # z_cf >= 0 correct; z_rf < 0 correct


def test_equal_mass_bins_sizes_and_order():
    conf = np.linspace(0.5, 1.0, 25)
    bins = equal_mass_bins(conf, 10)
    assert sum(len(b) for b in bins) == 25
    assert max(len(b) for b in bins) - min(len(b) for b in bins) <= 1
    # Bins are ordered by confidence.
    maxes = [conf[b].max() for b in bins]
    assert maxes == sorted(maxes)


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=20_000)
    correct = (rng.random(20_000) < conf).astype(float)
    assert ece(conf, correct) < 0.02


def test_ece_overconfident_judge():
    # All confidences tied at 0.95: ties are never split, so this is a single
    # bin and ECE = |acc - conf| = 0.45 regardless of how correctness is
    # ordered (stable-order splitting would have manufactured 0.5 here).
    conf = np.full(1000, 0.95)
    correct = np.repeat([1.0, 0.0], 500)  # accuracy 0.5
    assert len(equal_mass_bins(conf)) == 1
    assert ece(conf, correct) == pytest.approx(0.45, abs=1e-9)


def test_equal_mass_bins_tied_run_at_boundary_stays_whole():
    conf = np.array([0.6, 0.7, 0.8, 0.8, 0.8, 0.9])
    bins = equal_mass_bins(conf, 3)
    assert sum(len(b) for b in bins) == 6
    for b in bins:  # no tied value spans two bins
        others = set(np.delete(np.arange(6), b))
        assert not any(conf[i] in conf[list(others)] for i in b if len(others))


def test_reliability_curve_matches_bins():
    conf = np.array([0.55, 0.65, 0.75, 0.85])
    correct = np.array([1.0, 0.0, 1.0, 1.0])
    curve = reliability_curve(conf, correct, n_bins=2)
    assert [b["n"] for b in curve] == [2, 2]
    assert curve[0]["conf"] == pytest.approx(0.60)
    assert curve[0]["acc"] == pytest.approx(0.5)
    assert curve[1]["acc"] == pytest.approx(1.0)


def test_calibration_view_structure_and_ci():
    rng = np.random.default_rng(3)
    pairs = [
        make_pair(f"i/{k}", float(rng.normal(1.0, 2.0)), float(rng.normal(-1.0, 2.0)))
        for k in range(200)
    ]
    view = calibration_view(pairs, n_boot=300, seed=0)
    assert view["n_items"] == 200
    for name in ("sym", "raw"):
        block = view[name]
        assert block["n_points"] == (200 if name == "sym" else 400)
        lo, hi = block["ece"]["ci95"]
        assert 0.0 <= lo <= hi <= 1.0
        gap = block["confidence_minus_accuracy"]
        assert gap["mean"] == pytest.approx(
            block["mean_confidence"] - block["accuracy"], abs=1e-12
        )
        assert sum(b["n"] for b in block["curve"]) == block["n_points"]


def test_calibration_view_rejects_empty():
    with pytest.raises(ValueError):
        calibration_view([])

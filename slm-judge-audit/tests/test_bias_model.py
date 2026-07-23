"""Tests for the additive-shift decomposition and single-order correction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis import SwapPair
from src.bias_model import (
    BiasRow,
    bias_structure,
    build_bias_rows,
    correction_ladder,
    corrected_scores,
    loo_corrections,
    variance_decomposition,
    _design,
    _loo_group_means,
    _loo_regression,
)
from src.data import PairItem

CATEGORY_OF = {"alpha": "Chat", "beta": "Chat", "gamma": "Safety", "delta": "Safety"}


def make_rows(
    n: int,
    seed: int,
    subset_bias: dict[str, float] | None = None,
    noise_sd: float = 1.0,
    s_scale: float = 1.0,
) -> list[BiasRow]:
    """Synthetic swap pairs: b = subset constant + noise, s = signal.

    The gold-chosen response is preferred when s > 0, so an accurate judge
    is one whose s is mostly positive; ``s_scale`` sets how strongly.
    """
    rng = np.random.default_rng(seed)
    subsets = sorted(CATEGORY_OF)
    subset_bias = subset_bias or {sub: 0.0 for sub in subsets}
    rows = []
    for i in range(n):
        subset = subsets[i % len(subsets)]
        b = subset_bias[subset] + rng.normal(scale=noise_sd)
        s = s_scale * (0.6 + rng.normal())
        rows.append(
            BiasRow(
                item_id=f"{subset}/{i}",
                category=CATEGORY_OF[subset],
                subset=subset,
                z_cf=b + s,
                z_rf=b - s,
                log_total_chars=float(rng.uniform(5, 9)),
                abs_dlog_chars=float(abs(rng.normal(scale=0.5))),
                log_prompt_chars=float(rng.uniform(3, 7)),
            )
        )
    return rows


def test_bias_row_decomposition_identity():
    row = make_rows(1, seed=0)[0]
    assert row.z_cf == pytest.approx(row.b + row.s)
    assert row.z_rf == pytest.approx(row.b - row.s)


def test_build_bias_rows_covariates_and_join():
    item = PairItem(
        item_id="alpacaeval-easy/1",
        subset="alpacaeval-easy",
        category="Chat",
        prompt="ppp",
        chosen="aaaa bb",
        rejected="cc",
        chosen_model="m1",
        rejected_model="m2",
    )
    pairs = [SwapPair(item_id="alpacaeval-easy/1", z_cf=1.0, z_rf=-0.5,
                      compliant_both=True, mass_min=0.9)]
    (row,) = build_bias_rows(pairs, {item.item_id: item})
    assert row.subset == "alpacaeval-easy"
    assert row.log_total_chars == pytest.approx(math.log(9))
    assert row.abs_dlog_chars == pytest.approx(math.log(7 / 2))
    assert row.log_prompt_chars == pytest.approx(math.log(3))
    with pytest.raises(KeyError):
        build_bias_rows([SwapPair(item_id="missing/9", z_cf=0.0, z_rf=0.0,
                                  compliant_both=True, mass_min=0.5)], {})


def test_variance_decomposition_recovers_group_structure():
    # Strong subset-level bias, small noise: subset R^2 near 1; category R^2
    # in between (categories mix two subsets with different constants).
    rows = make_rows(
        800, seed=1,
        subset_bias={"alpha": 4.0, "beta": -4.0, "gamma": 2.0, "delta": 2.0},
        noise_sd=0.3,
    )
    out = variance_decomposition(rows, n_boot=200, seed=0)
    assert out["specs"]["subset"]["r2"] > 0.95
    assert out["specs"]["category"]["r2"] < out["specs"]["subset"]["r2"]
    # The regression nests the subset means: R^2 can only match or exceed.
    assert out["specs"]["subset_plus_length"]["r2"] >= out["specs"]["subset"]["r2"] - 1e-9
    for spec in out["specs"].values():
        lo, hi = spec["r2_ci95"]
        assert lo <= spec["r2"] <= hi or abs(spec["r2"] - lo) < 0.05


def test_variance_decomposition_null_bias_structure():
    rows = make_rows(600, seed=2, noise_sd=1.0)
    out = variance_decomposition(rows, n_boot=200, seed=0)
    # No group structure: R^2 stays near zero for the group specs.
    assert out["specs"]["subset"]["r2"] < 0.03
    assert out["specs"]["category"]["r2"] < 0.02
    assert out["b_sd"] == pytest.approx(1.0, abs=0.15)


def test_category_bias_means_recover_truth():
    rows = make_rows(
        1000, seed=3,
        subset_bias={"alpha": 3.0, "beta": 3.0, "gamma": -2.0, "delta": -2.0},
        noise_sd=0.5,
    )
    out = variance_decomposition(rows, n_boot=200, seed=0)
    assert out["category_bias_means"]["Chat"]["mean"] == pytest.approx(3.0, abs=0.15)
    assert out["category_bias_means"]["Safety"]["mean"] == pytest.approx(-2.0, abs=0.15)


def test_loo_group_means_exact_and_singleton_fallback():
    y = np.array([1.0, 3.0, 10.0])
    codes = np.array([0, 0, 1])
    loo, singletons = _loo_group_means(y, codes, 2)
    # Two-member group: LOO mean is the other member; singleton falls back to
    # the LOO grand mean.
    assert loo[0] == pytest.approx(3.0)
    assert loo[1] == pytest.approx(1.0)
    assert loo[2] == pytest.approx((1.0 + 3.0) / 2)
    assert singletons == 1


def test_loo_regression_matches_brute_force():
    rows = make_rows(60, seed=4, subset_bias={"alpha": 1.0, "beta": -1.0,
                                              "gamma": 0.5, "delta": 0.0})
    y = np.array([row.b for row in rows])
    X = _design(rows)
    loo = _loo_regression(y, X)
    from src.bias_model import RIDGE

    for i in [0, 7, 31, 59]:
        mask = np.ones(len(rows), dtype=bool)
        mask[i] = False
        A = X[mask].T @ X[mask] + RIDGE * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X[mask].T @ y[mask])
        assert loo[i] == pytest.approx(float(X[i] @ beta), abs=1e-6)


def test_oracle_correction_equals_symmetrized_verdict():
    rows = make_rows(300, seed=5, subset_bias={"alpha": 5.0, "beta": -3.0,
                                               "gamma": 0.0, "delta": 1.0})
    scores = corrected_scores(rows, np.array([row.b for row in rows]))
    pairs = [SwapPair(item_id=row.item_id, z_cf=row.z_cf, z_rf=row.z_rf,
                      compliant_both=True, mass_min=1.0) for row in rows]
    assert np.allclose(scores, [p.sym_correct for p in pairs])


def test_corrected_scores_none_equals_raw_accuracy():
    rows = make_rows(200, seed=6)
    scores = corrected_scores(rows, np.zeros(len(rows)))
    pairs = [SwapPair(item_id=row.item_id, z_cf=row.z_cf, z_rf=row.z_rf,
                      compliant_both=True, mass_min=1.0) for row in rows]
    # Convention gap is exact zeros only: SwapPair.raw_correct_cf counts
    # z == 0 as an A verdict, corrected_scores as half. No exact zeros here.
    assert np.allclose(scores, [p.raw_correct_mean for p in pairs])


def test_ladder_recovers_symmetrization_when_bias_is_subset_constant():
    # Bias is exactly a subset constant with tiny noise: the subset-corrected
    # single-order judge must recover nearly all of the oracle gain, and the
    # uncorrected judge must sit well below (bias >> signal).
    rows = make_rows(
        600, seed=7,
        subset_bias={"alpha": 6.0, "beta": -6.0, "gamma": 4.0, "delta": -4.0},
        noise_sd=0.1,
        s_scale=0.8,
    )
    out = correction_ladder(rows, n_boot=200, seed=0)
    acc = {name: out["estimators"][name]["acc"]["mean"] for name in out["estimators"]}
    assert acc["none"] < 0.6
    assert acc["oracle"] > 0.65
    assert acc["subset"] > acc["oracle"] - 0.02
    assert out["estimators"]["subset"]["recovered_fraction"] > 0.9
    # Global constant cannot fix sign-opposed subset biases.
    assert acc["global"] < acc["subset"] - 0.05


def test_ladder_no_recovered_fraction_when_oracle_gain_null():
    # No position bias at all: correcting changes nothing; the oracle gain CI
    # includes zero so recovered fractions are (correctly) absent.
    rows = make_rows(300, seed=8, noise_sd=0.0, s_scale=1.0)
    out = correction_ladder(rows, n_boot=200, seed=0)
    assert "recovered_fraction" not in out["estimators"]["subset"]
    lo, hi = out["oracle_gain"]["ci95"]
    assert lo <= 0.0 <= hi


def test_ladder_structure_and_diagnostics():
    rows = make_rows(120, seed=9)
    out = correction_ladder(rows, n_boot=100, seed=0)
    assert out["n_items"] == 120
    assert set(out["estimators"]) == {"none", "global", "category", "subset",
                                      "regression", "oracle"}
    assert out["diagnostics"]["min_group_size"]["subset"] == 30
    assert out["diagnostics"]["singleton_fallbacks"]["subset"] == 0
    for name, block in out["estimators"].items():
        lo, hi = block["acc"]["ci95"]
        assert 0.0 <= lo <= block["acc"]["mean"] <= hi <= 1.0
        assert set(block["by_category"]) == {"Chat", "Safety"}
    assert "delta_vs_none" not in out["estimators"]["none"]
    assert "delta_vs_oracle" not in out["estimators"]["oracle"]


def test_bias_structure_smoke():
    rows = make_rows(200, seed=10, subset_bias={"alpha": 2.0, "beta": -2.0,
                                                "gamma": 1.0, "delta": 0.0})
    out = bias_structure(rows, n_boot=100, seed=0)
    assert set(out) == {"decomposition", "ladder"}


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        variance_decomposition(make_rows(2, seed=0), n_boot=10)
    with pytest.raises(ValueError):
        loo_corrections(make_rows(2, seed=0))
    constant = [
        BiasRow(item_id=f"alpha/{i}", category="Chat", subset="alpha",
                z_cf=1.0, z_rf=1.0, log_total_chars=6.0, abs_dlog_chars=0.1,
                log_prompt_chars=4.0)
        for i in range(5)
    ]
    with pytest.raises(ValueError):
        variance_decomposition(constant, n_boot=10)

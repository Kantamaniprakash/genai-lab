"""Tests for the value-over-length conditional-logit probe."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis import SwapPair
from src.data import PairItem
from src.length_probe import (
    ProbeRow,
    build_rows,
    feature_matrix,
    fit_conditional_logit,
    scale_columns,
    value_over_length,
    _fit_batched,
)


def simulate_rows(n: int, beta_s: float, beta_len: float, seed: int) -> list[ProbeRow]:
    """Bradley–Terry data with known coefficients.

    Draw latent pair differences d ~ N(0, I); the first response wins with
    probability sigmoid(beta . d); orient features chosen − rejected by
    flipping the sign when the second response wins.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 2))
    p_win = 1.0 / (1.0 + np.exp(-(beta_s * d[:, 0] + beta_len * d[:, 1])))
    flip = np.where(rng.random(n) < p_win, 1.0, -1.0)
    oriented = d * flip[:, None]
    return [
        ProbeRow(
            item_id=f"sim/{i}",
            category="Chat" if i % 2 == 0 else "Safety",
            s=float(oriented[i, 0]),
            dlog_chars=float(oriented[i, 1]),
            dlog_words=float(oriented[i, 1]),
            compliant_both=True,
        )
        for i in range(n)
    ]


def test_fit_recovers_known_coefficients():
    rows = simulate_rows(4000, beta_s=0.8, beta_len=-0.4, seed=7)
    X = feature_matrix(rows, ("s", "dlog_chars"))
    beta = fit_conditional_logit(X)  # unit-variance features: no rescaling needed
    assert beta[0] == pytest.approx(0.8, abs=0.15)
    assert beta[1] == pytest.approx(-0.4, abs=0.15)


def test_fit_antisymmetry():
    rows = simulate_rows(500, beta_s=0.6, beta_len=0.2, seed=1)
    X = feature_matrix(rows, ("s", "dlog_chars"))
    assert np.allclose(fit_conditional_logit(-X), -fit_conditional_logit(X), atol=1e-6)


def test_fit_null_data_gives_zero_coefficients():
    rows = simulate_rows(3000, beta_s=0.0, beta_len=0.0, seed=3)
    X = feature_matrix(rows, ("s", "dlog_chars"))
    assert np.abs(fit_conditional_logit(X)).max() < 0.1


def test_separation_stays_finite():
    # All-positive single feature: the MLE diverges; the ridge must cap it.
    X = np.abs(np.random.default_rng(0).normal(size=(80, 1))) + 0.1
    beta = fit_conditional_logit(X)
    assert np.isfinite(beta[0])
    assert 0 < beta[0] < 50


def test_batched_fit_matches_single_fit():
    rows = simulate_rows(300, beta_s=0.5, beta_len=-0.3, seed=11)
    X = feature_matrix(rows, ("s", "dlog_chars"))
    stacked = np.stack([X, -X, X[::-1]])
    batched = _fit_batched(stacked)
    for b, design in zip(batched, stacked):
        assert np.allclose(b, fit_conditional_logit(design), atol=1e-5)


def test_scale_columns_preserves_origin_and_handles_constant():
    X = np.array([[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]])
    scaled, sd = scale_columns(X)
    assert sd[0] > 0 and sd[1] == 0.0
    assert np.allclose(scaled[:, 1], X[:, 1])  # constant column left unscaled
    assert np.allclose(scaled[:, 0] * sd[0], X[:, 0])  # scaling only, no shift


def test_feature_matrix_sign_feature():
    rows = simulate_rows(10, beta_s=0.0, beta_len=0.0, seed=5)
    X = feature_matrix(rows, ("sign_s", "dlog_chars"))
    assert set(np.unique(X[:, 0])) <= {-1.0, 0.0, 1.0}
    assert np.allclose(X[:, 0], np.sign([row.s for row in rows]))


def make_item(item_id: str, chosen: str, rejected: str) -> PairItem:
    return PairItem(
        item_id=item_id,
        subset="alpacaeval-easy",
        category="Chat",
        prompt="p",
        chosen=chosen,
        rejected=rejected,
        chosen_model="m1",
        rejected_model="m2",
    )


def test_build_rows_lengths_and_join():
    items = {
        "alpacaeval-easy/1": make_item("alpacaeval-easy/1", "aaaa bb", "cc"),
    }
    pairs = [SwapPair(item_id="alpacaeval-easy/1", z_cf=1.0, z_rf=-0.5,
                      compliant_both=True, mass_min=0.9)]
    (row,) = build_rows(pairs, items)
    assert row.s == pytest.approx(0.75)
    assert row.dlog_chars == pytest.approx(math.log(7 / 2))
    assert row.dlog_words == pytest.approx(math.log(2 / 1))
    with pytest.raises(KeyError):
        build_rows([SwapPair(item_id="missing/9", z_cf=0.0, z_rf=0.0,
                             compliant_both=True, mass_min=0.5)], items)


def test_value_over_length_structure_and_nesting():
    rows = simulate_rows(400, beta_s=0.7, beta_len=-0.3, seed=13)
    result = value_over_length(rows, n_boot=200, seed=0)
    overall = result["overall"]
    assert overall["n_items"] == 400
    assert set(result["by_category"]) == {"Chat", "Safety"}
    for spec in ("length", "judge", "joint", "joint_sign"):
        block = overall["specs"][spec]
        assert 0.0 <= block["acc"] <= 1.0
        for feat in block["features"]:
            lo, hi = block["coef_ci95"][feat]
            assert lo <= block["coef"][feat] <= hi
    # The joint spec nests both single specs: train log-loss cannot be worse
    # (up to the ridge's negligible perturbation).
    assert overall["specs"]["joint"]["logloss"] <= overall["specs"]["length"]["logloss"] + 1e-4
    assert overall["specs"]["joint"]["logloss"] <= overall["specs"]["judge"]["logloss"] + 1e-4
    # With a strong simulated judge signal, beta_s in the joint spec is
    # significantly positive and the joint spec beats length-only.
    lo, _ = overall["specs"]["joint"]["coef_ci95"]["s"]
    assert lo > 0
    assert overall["acc_joint_minus_length"]["mean"] > 0


def test_value_over_length_rejects_empty():
    with pytest.raises(ValueError):
        value_over_length([])

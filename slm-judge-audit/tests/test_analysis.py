"""Tests for swap-pair assembly, the decomposition, and the paired bootstrap."""

from __future__ import annotations

import pytest

from src.analysis import (
    SwapPair,
    assemble_pairs,
    bootstrap_mean_ci,
    paired_bootstrap_delta_ci,
    summarize_pairs,
)
from src.judge import JudgmentRecord


def make_record(item_id: str, order: str, z: float, *, model: str = "fake",
                rubric: str = "minimal", compliant: bool = True) -> JudgmentRecord:
    return JudgmentRecord(
        model=model,
        rubric=rubric,
        order=order,
        item_id=item_id,
        expected_verdict="A" if order == "chosen_first" else "B",
        z=z,
        logp_a=-0.5,
        logp_b=-0.5 - z,
        mass_ab=0.98,
        argmax_token="A" if compliant else "The",
        compliant=compliant,
        n_prompt_tokens=100,
        prefill_seconds=0.5,
    )


def test_assemble_pairs_and_decomposition():
    records = [
        make_record("i/1", "chosen_first", 3.0),
        make_record("i/1", "rejected_first", 1.0),
    ]
    pairs, incomplete = assemble_pairs(records)
    assert incomplete == 0
    (pair,) = pairs
    assert pair.s == pytest.approx(1.0)   # prefers chosen either way
    assert pair.b == pytest.approx(2.0)   # shifted toward A
    # Reconstruction identity
    assert pair.z_cf == pytest.approx(pair.b + pair.s)
    assert pair.z_rf == pytest.approx(pair.b - pair.s)
    # Raw: cf correct (z>=0 -> A, expected A), rf wrong (z>=0 -> A, expected B)
    assert pair.raw_correct_cf == 1.0
    assert pair.raw_correct_rf == 0.0
    assert pair.raw_correct_mean == 0.5
    # Symmetrized rescues it; and the swap flipped the positional verdict? No:
    # both orders said A, so no flip — the classic bias signature.
    assert pair.sym_correct == 1.0
    assert not pair.positional_flip


def test_positional_flip_and_sym_tie():
    flip = SwapPair(item_id="i/2", z_cf=0.5, z_rf=-0.5, compliant_both=True, mass_min=0.9)
    assert flip.positional_flip
    assert flip.sym_correct == 1.0
    tie = SwapPair(item_id="i/3", z_cf=1.0, z_rf=1.0, compliant_both=True, mass_min=0.9)
    assert tie.s == 0.0
    assert tie.sym_correct == 0.5


def test_assemble_rejects_mixed_and_duplicate_sets():
    with pytest.raises(ValueError, match="mixed"):
        assemble_pairs([
            make_record("i/1", "chosen_first", 1.0, model="a"),
            make_record("i/1", "rejected_first", 1.0, model="b"),
        ])
    with pytest.raises(ValueError, match="duplicate"):
        assemble_pairs([
            make_record("i/1", "chosen_first", 1.0),
            make_record("i/1", "chosen_first", 2.0),
        ])


def test_assemble_counts_incomplete_items():
    records = [
        make_record("i/1", "chosen_first", 1.0),
        make_record("i/1", "rejected_first", -1.0),
        make_record("i/2", "chosen_first", 1.0),  # missing its swap
    ]
    pairs, incomplete = assemble_pairs(records)
    assert len(pairs) == 1 and incomplete == 1


def test_bootstrap_ci_is_deterministic_and_sane():
    values = [0.0, 1.0] * 50
    mean1 = bootstrap_mean_ci(values, n_boot=2000, seed=7)
    mean2 = bootstrap_mean_ci(values, n_boot=2000, seed=7)
    assert mean1 == mean2
    mean, lo, hi = mean1
    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi
    assert 0.35 < lo and hi < 0.65
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])


def test_paired_delta_ci_detects_systematic_gain():
    # Symmetrization rescues every item: sym=1, raw_mean=0.5 on all pairs.
    pairs = [
        SwapPair(item_id=f"i/{k}", z_cf=2.0, z_rf=1.0, compliant_both=True, mass_min=0.9)
        for k in range(40)
    ]
    delta, lo, hi = paired_bootstrap_delta_ci(
        pairs, lambda p: p.sym_correct, lambda p: p.raw_correct_mean, n_boot=500
    )
    assert delta == pytest.approx(0.5)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)  # zero variance


def test_summarize_pairs_block():
    pairs = [
        SwapPair(item_id="i/1", z_cf=3.0, z_rf=1.0, compliant_both=True, mass_min=0.99),
        SwapPair(item_id="i/2", z_cf=0.5, z_rf=-0.5, compliant_both=True, mass_min=0.95),
        SwapPair(item_id="i/3", z_cf=-1.0, z_rf=2.0, compliant_both=False, mass_min=0.60),
    ]
    summary = summarize_pairs(pairs, n_boot=200)
    assert summary["n_items"] == 3
    assert summary["compliance_rate"] == pytest.approx(2 / 3)
    # b values: 2.0, 0.0, 0.5; s values: 1.0, 0.5, -1.5
    assert summary["bias_b"]["mean"] == pytest.approx((2.0 + 0.0 + 0.5) / 3)
    assert summary["preference_s"]["median_abs"] == pytest.approx(1.0)
    assert summary["raw_acc_chosen_first"] == pytest.approx(2 / 3)
    assert summary["raw_acc_rejected_first"] == pytest.approx(1 / 3)
    # sym correct: 1.0, 1.0, 0.0
    assert summary["sym_acc"]["mean"] == pytest.approx(2 / 3)
    assert summary["frac_bias_dominates"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        summarize_pairs([])

"""Calibration of verdict probabilities: reliability curves and ECE.

Finding 12 (2026-07-22) showed that at 1.5B the *magnitude* of the judge's
preference carries length-controlled signal its *sign* does not — the judge is
right where it is confident. Calibration is the formal version of that
observation: does the verdict probability the readout assigns match the
empirical probability of being correct?

Two views, both in the folded (confidence, correctness) form standard for
selective prediction:

- **symmetrized** — one point per swap pair: confidence ``sigmoid(|s|)`` (the
  probability the swap-averaged verdict assigns to its own choice), correct
  when ``sign(s)`` picks the gold-chosen response (ties count half);
- **raw** — one point per judgment: confidence ``sigmoid(|z|)`` (the
  renormalized probability on the winning verdict token), correct when the
  greedy verdict matches the expected letter for that presentation. Position
  bias lives inside ``z``, so this is the calibration a single-order
  deployment actually experiences.

Bins are equal-mass (quantile) rather than equal-width: saturated judges pile
confidence near 1.0 and would leave most equal-width bins empty. ECE is the
bin-mass-weighted mean |accuracy − confidence|; the signed gap
mean(confidence) − accuracy gives the direction (positive = overconfident).
Uncertainty is a percentile bootstrap over items (both orders of an item
resampled together, as everywhere in this audit) with binning recomputed
inside every replicate.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .analysis import SwapPair

N_BINS = 10


def fold_sym(pairs: Sequence[SwapPair]) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correctness) of the symmetrized verdict, one row per item."""
    conf = np.array([1.0 / (1.0 + math.exp(-abs(p.s))) for p in pairs])
    correct = np.array([p.sym_correct for p in pairs])
    return conf, correct


def fold_raw(pairs: Sequence[SwapPair]) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correctness) of raw verdicts, two rows per item (cf, rf)."""
    conf, correct = [], []
    for p in pairs:
        conf.append(1.0 / (1.0 + math.exp(-abs(p.z_cf))))
        correct.append(p.raw_correct_cf)
        conf.append(1.0 / (1.0 + math.exp(-abs(p.z_rf))))
        correct.append(p.raw_correct_rf)
    return np.array(conf), np.array(correct)


def equal_mass_bins(conf: np.ndarray, n_bins: int = N_BINS) -> list[np.ndarray]:
    """Index arrays for quantile bins, ordered by confidence, never splitting ties.

    Tied confidences matter here: a saturated judge piles probability at
    exactly 1.0 (float sigmoid of a large |z|), and splitting one tied run
    across bins with different accuracies manufactures ECE out of the split
    (the bins differ only in accuracy, never in confidence). Each boundary is
    therefore pushed to the end of any tied run, so a tied mass lands in one
    bin; with heavy saturation this yields fewer, larger bins — the honest
    resolution the data supports.
    """
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    sorted_conf = conf[order]
    k = min(n_bins, n)
    bins: list[np.ndarray] = []
    start = 0
    for i in range(1, k + 1):
        end = n if i == k else round(i * n / k)
        while 0 < end < n and sorted_conf[end] == sorted_conf[end - 1]:
            end += 1
        if end > start:
            bins.append(order[start:end])
            start = end
    return bins


def reliability_curve(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS
) -> list[dict]:
    """Per-bin (n, mean confidence, accuracy), the reliability-diagram data."""
    return [
        {
            "n": int(len(idx)),
            "conf": float(conf[idx].mean()),
            "acc": float(correct[idx].mean()),
        }
        for idx in equal_mass_bins(conf, n_bins)
    ]


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Bin-mass-weighted mean |accuracy − confidence| over equal-mass bins."""
    total = len(conf)
    return float(
        sum(
            b["n"] / total * abs(b["acc"] - b["conf"])
            for b in reliability_curve(conf, correct, n_bins)
        )
    )


def calibration_view(
    pairs: Sequence[SwapPair],
    n_bins: int = N_BINS,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Both folded views with bootstrap CIs; binning recomputed per replicate."""
    if not pairs:
        raise ValueError("no complete swap pairs")
    rng = np.random.default_rng(seed)
    n = len(pairs)
    idx = rng.integers(0, n, size=(n_boot, n))

    out: dict = {"n_items": n, "n_bins": n_bins, "n_boot": n_boot, "bootstrap_seed": seed}
    # Fold once into per-item arrays; replicates then only gather rows, so
    # item-level resampling keeps both orders of an item together in the raw
    # view (its arrays are (n, 2): one column per order).
    sym_conf, sym_correct = fold_sym(pairs)
    raw_conf_flat, raw_correct_flat = fold_raw(pairs)
    views = {
        "sym": (sym_conf, sym_correct, sym_conf.reshape(n, 1), sym_correct.reshape(n, 1)),
        "raw": (raw_conf_flat, raw_correct_flat,
                raw_conf_flat.reshape(n, 2), raw_correct_flat.reshape(n, 2)),
    }
    for name, (conf, correct, conf_by_item, correct_by_item) in views.items():
        boots_ece = np.empty(n_boot)
        boots_gap = np.empty(n_boot)
        for b in range(n_boot):
            c = conf_by_item[idx[b]].ravel()
            k = correct_by_item[idx[b]].ravel()
            boots_ece[b] = ece(c, k, n_bins)
            boots_gap[b] = float(c.mean() - k.mean())
        ece_lo, ece_hi = np.quantile(boots_ece, [0.025, 0.975])
        gap_lo, gap_hi = np.quantile(boots_gap, [0.025, 0.975])
        out[name] = {
            "n_points": int(len(conf)),
            "ece": {"mean": ece(conf, correct, n_bins), "ci95": [float(ece_lo), float(ece_hi)]},
            "confidence_minus_accuracy": {
                "mean": float(conf.mean() - correct.mean()),
                "ci95": [float(gap_lo), float(gap_hi)],
            },
            "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean()),
            "curve": reliability_curve(conf, correct, n_bins),
        }
    return out

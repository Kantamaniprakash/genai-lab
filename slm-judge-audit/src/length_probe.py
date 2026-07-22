"""Value-over-length probe: does the judge add signal beyond response length?

Motivated by finding 10 (2026-07-19): the preference signal that emerges with
scale inside the Qwen family tracks response length, and RewardBench Reasoning
punishes exactly that. The question this module answers precisely: does the
judge's order-invariant preference ``s`` carry information about the gold
label *beyond* what a trivial length comparison already provides — overall,
per category, per model?

The model is a Bradley–Terry / conditional-logit fit on oriented pair
differences. For item ``i``, features ``x_i`` are oriented chosen − rejected
(the judge preference log-odds ``s_i``; the log length ratio
``log(len_chosen / len_rejected)``), and

    P(gold-chosen wins) = sigmoid(beta . x_i)

with likelihood ``prod_i sigmoid(beta . x_i)``. There is deliberately no
intercept: relabeling chosen/rejected flips the sign of every feature, so a
constant term is not identified — with the outcome constant by construction
its ML estimate diverges. Equivalently, this is logistic regression on the
antisymmetric doubled data, where the intercept is exactly zero. Nested specs
(length-only, judge-only, joint) turn "does the judge beat a length
heuristic?" into a coefficient question: ``beta_s != 0`` in the joint spec
means the judge adds signal length cannot explain.

Features are scaled by their sample SD but NOT centered — the origin "both
responses equal / judge indifferent" is meaningful and must map to P = 1/2.
Coefficients therefore read as log-odds per SD, comparable across models and
strata; the SDs are reported so raw-scale coefficients are recoverable.

Uncertainty is a percentile bootstrap over items with the full pipeline
(rescaling + refit) inside every replicate. Refits are batched Newton
iterations. A weak ridge (RIDGE) keeps replicates with complete separation
(possible in small one-signed strata) finite without measurably biasing
coefficients at these sample sizes; accuracies are in-sample (2 parameters on
n >= 70 items — optimism is negligible, and the paired spec deltas share it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .analysis import SwapPair
from .data import PairItem

RIDGE = 1e-3
MAX_ITER = 60
STEP_TOL = 1e-9
MAX_STEP_NORM = 5.0  # per-iteration Newton damping; harmless when converging

# Nested specifications. "sign_s" is sign(s) — exactly the symmetrized binary
# verdict a deployment would use, and immune to heavy tails in s.
SPECS: dict[str, tuple[str, ...]] = {
    "length": ("dlog_chars",),
    "judge": ("s",),
    "joint": ("s", "dlog_chars"),
    "joint_sign": ("sign_s", "dlog_chars"),
}


@dataclass(frozen=True)
class ProbeRow:
    """One item's oriented (chosen − rejected) features for the regression."""

    item_id: str
    category: str
    s: float            # judge order-invariant preference log-odds
    dlog_chars: float   # log(len_chars(chosen) / len_chars(rejected))
    dlog_words: float   # log(len_words(chosen) / len_words(rejected))
    compliant_both: bool


def build_rows(
    pairs: Sequence[SwapPair], items_by_id: dict[str, PairItem]
) -> list[ProbeRow]:
    """Join swap pairs to item text statistics. Fails on unknown item ids."""
    rows = []
    for pair in pairs:
        item = items_by_id.get(pair.item_id)
        if item is None:
            raise KeyError(f"no dataset item for pair {pair.item_id}")
        rows.append(
            ProbeRow(
                item_id=pair.item_id,
                category=item.category,
                s=pair.s,
                dlog_chars=math.log(len(item.chosen) / len(item.rejected)),
                dlog_words=math.log(
                    max(len(item.chosen.split()), 1) / max(len(item.rejected.split()), 1)
                ),
                compliant_both=pair.compliant_both,
            )
        )
    return rows


def feature_matrix(rows: Sequence[ProbeRow], features: tuple[str, ...]) -> np.ndarray:
    """(n_items, n_features) oriented feature matrix; "sign_s" derives from s."""
    columns = []
    for name in features:
        if name == "sign_s":
            columns.append([float(np.sign(row.s)) for row in rows])
        else:
            columns.append([float(getattr(row, name)) for row in rows])
    return np.asarray(columns, dtype=np.float64).T


def scale_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X_scaled, sd): divide each column by its SD, never centering.

    A zero-variance column is left unscaled (sd reported as 0.0) — its
    coefficient is then intercept-like and only the ridge keeps it finite,
    which is the honest behavior for a degenerate stratum.
    """
    sd = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
    safe = np.where(sd > 0, sd, 1.0)
    return X / safe, sd


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    ex = np.exp(eta[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def fit_conditional_logit(X: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """MLE of the no-intercept conditional logit; damped Newton iterations."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if n == 0:
        raise ValueError("empty design matrix")
    beta = np.zeros(p)
    for _ in range(MAX_ITER):
        sig = _sigmoid(X @ beta)
        grad = X.T @ (1.0 - sig) - ridge * beta
        w = np.clip(sig * (1.0 - sig), 1e-12, None)
        hess = (X * w[:, None]).T @ X + ridge * np.eye(p)
        step = np.linalg.solve(hess, grad)
        norm = float(np.linalg.norm(step))
        if norm > MAX_STEP_NORM:
            step *= MAX_STEP_NORM / norm
        beta = beta + step
        if float(np.abs(step).max()) < STEP_TOL:
            break
    return beta


def _fit_batched(Xb: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Vectorized Newton over a (B, n, p) stack of bootstrap designs."""
    B, _, p = Xb.shape
    beta = np.zeros((B, p))
    eye = ridge * np.eye(p)
    for _ in range(MAX_ITER):
        sig = _sigmoid(np.einsum("bnp,bp->bn", Xb, beta))
        grad = np.einsum("bnp,bn->bp", Xb, 1.0 - sig) - ridge * beta
        w = np.clip(sig * (1.0 - sig), 1e-12, None)
        hess = np.einsum("bnp,bn,bnq->bpq", Xb, w, Xb, optimize=True) + eye
        step = np.linalg.solve(hess, grad[:, :, None])[:, :, 0]
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        shrink = np.where(norms > MAX_STEP_NORM, MAX_STEP_NORM / np.maximum(norms, 1e-30), 1.0)
        step = step * shrink
        beta = beta + step
        if float(np.abs(step).max()) < STEP_TOL:
            break
    return beta


def _accuracy(eta: np.ndarray) -> float:
    """Fraction of items the fitted score orders correctly; ties score half."""
    return float(np.mean((eta > 0) + 0.5 * (eta == 0)))


def _logloss(eta: np.ndarray) -> float:
    # -log sigmoid(eta), computed stably: log(1 + exp(-eta)).
    return float(np.mean(np.logaddexp(0.0, -eta)))


def _stratum_probe(
    rows: Sequence[ProbeRow], n_boot: int, seed: int, boot_chunk: int = 2000
) -> dict:
    """All specs, full-sample fits + shared-resample bootstrap, one stratum."""
    n = len(rows)
    rng = np.random.default_rng(seed)
    raw = {name: feature_matrix(rows, feats) for name, feats in SPECS.items()}

    out: dict = {"n_items": n}
    s_arr = np.array([row.s for row in rows])
    out["compliance_rate"] = float(np.mean([row.compliant_both for row in rows]))
    for unit in ("chars", "words"):
        d = np.array([getattr(row, f"dlog_{unit}") for row in rows])
        out[f"sign_agree_{unit}"] = float(np.mean(np.sign(s_arr) == np.sign(d)))

    # Full-sample point estimates.
    point: dict[str, dict] = {}
    for name, feats in SPECS.items():
        X, sd = scale_columns(raw[name])
        beta = fit_conditional_logit(X)
        eta = X @ beta
        point[name] = {
            "features": list(feats),
            "feature_sd": [float(v) for v in sd],
            "coef": {f: float(b) for f, b in zip(feats, beta)},
            "acc": _accuracy(eta),
            "logloss": _logloss(eta),
        }

    # Shared item resamples across specs so spec deltas are paired.
    boot_coef = {name: [] for name in SPECS}
    boot_acc = {name: [] for name in SPECS}
    remaining = n_boot
    while remaining > 0:
        chunk = min(boot_chunk, remaining)
        idx = rng.integers(0, n, size=(chunk, n))
        for name in SPECS:
            Xb = raw[name][idx]
            sd = Xb.std(axis=1, ddof=1, keepdims=True)
            Xb = Xb / np.where(sd > 0, sd, 1.0)
            beta = _fit_batched(Xb)
            eta = np.einsum("bnp,bp->bn", Xb, beta)
            boot_coef[name].append(beta)
            boot_acc[name].append(np.mean((eta > 0) + 0.5 * (eta == 0), axis=1))
        remaining -= chunk

    def ci(samples: np.ndarray) -> list[float]:
        lo, hi = np.quantile(samples, [0.025, 0.975])
        return [float(lo), float(hi)]

    for name, feats in SPECS.items():
        coef = np.concatenate(boot_coef[name], axis=0)
        point[name]["coef_ci95"] = {f: ci(coef[:, j]) for j, f in enumerate(feats)}
        point[name]["acc_ci95"] = ci(np.concatenate(boot_acc[name]))
    out["specs"] = point

    acc_len = np.concatenate(boot_acc["length"])
    for name in ("judge", "joint", "joint_sign"):
        delta = np.concatenate(boot_acc[name]) - acc_len
        out[f"acc_{name}_minus_length"] = {
            "mean": point[name]["acc"] - point["length"]["acc"],
            "ci95": ci(delta),
        }
    return out


def value_over_length(
    rows: Sequence[ProbeRow], n_boot: int = 10_000, seed: int = 0
) -> dict:
    """The probe over one model's pairs: overall plus per-category strata."""
    if not rows:
        raise ValueError("no probe rows")
    result = {
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "ridge": RIDGE,
        "overall": _stratum_probe(rows, n_boot, seed),
        "by_category": {},
    }
    by_cat: dict[str, list[ProbeRow]] = {}
    for row in rows:
        by_cat.setdefault(row.category, []).append(row)
    for cat in sorted(by_cat):
        result["by_category"][cat] = _stratum_probe(by_cat[cat], n_boot, seed)
    return result

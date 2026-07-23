"""Position-bias structure: the additive-shift test and single-order correction.

Phase-3 item 2. Swap-averaging removes position bias *exactly* because it
computes ``s_i`` per item — it never needs the additive-shift assumption. But
every cheaper debiasing scheme does: correcting a single-order verdict by a
fitted constant ``sign(z - b_hat)`` is only as good as the assumption that
``b_i`` is predictable from what the deployment knows (nothing, the category,
the subset, or item length statistics). This module makes that assumption
testable and prices it in accuracy points.

Two connected analyses over one model's swap pairs:

1. **Variance decomposition of b.** Because verdict readout is deterministic
   at temperature 0, ``b_i`` carries no sampling noise: all of ``Var(b)`` is
   real item-level bias structure. Nested predictors — grand constant, category
   means, subset means, subset + length covariates — partition it into the
   share a correction scheme could exploit (R^2, refit inside every bootstrap
   replicate) and an irreducible residual. The additive-shift hypothesis is
   the claim R^2 ~ 0 with a small residual SD; its rejection is quantified,
   not asserted.

2. **The correction ladder.** The oracle single-order correction
   ``sign(z - b_i)`` IS the symmetrized verdict (``z_cf - b_i = s_i`` and
   ``z_rf - b_i = -s_i``), so fitted corrections interpolate between the raw
   single-order judge and full symmetrization — at half the inference cost.
   Each estimator is evaluated with exact leave-one-out cross-fitting (closed
   forms: LOO group means; the ridge hat-matrix identity for the regression),
   so no item is corrected using its own bias. The ladder reports how much of
   the symmetrization gain each level of structure recovers.

Uncertainty: R^2 and group means are bootstrapped with a full refit per
replicate. Ladder accuracies bootstrap the fixed per-item LOO scores (the
lab's standard item-resampling), which ignores correction-refit variance —
negligible here, since every estimator is a group mean or a 26-parameter OLS
on n >= 600 items, but recorded so the caveat is explicit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .analysis import SwapPair, bootstrap_mean_ci
from .data import PairItem

# Tiny ridge for the length regression: keeps replicate designs that lost a
# subset column solvable without measurably shrinking coefficients.
RIDGE = 1e-8

LENGTH_FEATURES: tuple[str, ...] = ("log_total_chars", "abs_dlog_chars", "log_prompt_chars")

DECOMP_SPECS: tuple[str, ...] = ("category", "subset", "subset_plus_length")

# Ladder order matters for reading: each row may use strictly more structure.
CORRECTIONS: tuple[str, ...] = ("none", "global", "category", "subset", "regression", "oracle")


@dataclass(frozen=True)
class BiasRow:
    """One item's swap readouts plus the covariates a correction could use.

    Length covariates are *symmetric* in the pair (total, absolute ratio):
    ``b_i`` is order-invariant by construction, so any covariate that flips
    sign under chosen/rejected relabeling cannot linearly predict it.
    """

    item_id: str
    category: str
    subset: str
    z_cf: float
    z_rf: float
    log_total_chars: float   # log(chars(chosen) + chars(rejected))
    abs_dlog_chars: float    # |log(chars(chosen) / chars(rejected))|
    log_prompt_chars: float

    @property
    def b(self) -> float:
        return (self.z_cf + self.z_rf) / 2

    @property
    def s(self) -> float:
        return (self.z_cf - self.z_rf) / 2


def build_bias_rows(
    pairs: Sequence[SwapPair], items_by_id: dict[str, PairItem]
) -> list[BiasRow]:
    """Join swap pairs to item covariates. Fails on unknown item ids."""
    rows = []
    for pair in pairs:
        item = items_by_id.get(pair.item_id)
        if item is None:
            raise KeyError(f"no dataset item for pair {pair.item_id}")
        rows.append(
            BiasRow(
                item_id=pair.item_id,
                category=item.category,
                subset=item.subset,
                z_cf=pair.z_cf,
                z_rf=pair.z_rf,
                log_total_chars=math.log(len(item.chosen) + len(item.rejected)),
                abs_dlog_chars=abs(math.log(len(item.chosen) / len(item.rejected))),
                log_prompt_chars=math.log(max(len(item.prompt), 1)),
            )
        )
    return rows


def _codes(rows: Sequence[BiasRow], attr: str) -> tuple[np.ndarray, list[str]]:
    """Integer-encode a grouping attribute; labels sorted for determinism."""
    labels = sorted({getattr(row, attr) for row in rows})
    index = {label: i for i, label in enumerate(labels)}
    return np.array([index[getattr(row, attr)] for row in rows]), labels


def _design(rows: Sequence[BiasRow]) -> np.ndarray:
    """Subset one-hot + SD-scaled length covariates for the regression spec.

    Scaling is for conditioning only — it changes no fitted value (the one-hot
    block spans constants, and column scaling never changes the column span).
    """
    codes, labels = _codes(rows, "subset")
    onehot = np.zeros((len(rows), len(labels)))
    onehot[np.arange(len(rows)), codes] = 1.0
    lengths = np.array(
        [[getattr(row, f) for f in LENGTH_FEATURES] for row in rows], dtype=np.float64
    )
    sd = lengths.std(axis=0, ddof=1)
    lengths = lengths / np.where(sd > 0, sd, 1.0)
    return np.concatenate([onehot, lengths], axis=1)


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    if sst == 0.0:
        raise ValueError("b is constant; R^2 undefined")
    return 1.0 - float(((y - fitted) ** 2).sum()) / sst


def _fit_group_means(y: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.bincount(codes, weights=y, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return means[codes]


def _fit_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = X.T @ X + RIDGE * np.eye(X.shape[1])
    return X @ np.linalg.solve(A, X.T @ y)


def _bootstrap_r2(
    y: np.ndarray,
    fitted_of_resample,
    n: int,
    n_boot: int,
    seed: int,
    chunk: int = 500,
) -> np.ndarray:
    """Percentile-bootstrap R^2 with a full refit inside every replicate.

    ``fitted_of_resample(idx)`` maps a (B, n) index array to (B, n) fitted
    values, refit within each resampled row set.
    """
    rng = np.random.default_rng(seed)
    out = []
    remaining = n_boot
    while remaining > 0:
        b = min(chunk, remaining)
        idx = rng.integers(0, n, size=(b, n))
        yb = y[idx]
        fitted = fitted_of_resample(idx)
        sse = ((yb - fitted) ** 2).sum(axis=1)
        sst = ((yb - yb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        out.append(1.0 - sse / np.where(sst > 0, sst, np.nan))
        remaining -= b
    return np.concatenate(out)


def _group_fitted_batched(y: np.ndarray, codes: np.ndarray, n_groups: int):
    """Batched group-mean refit: groups absent from a replicate are never
    gathered, so their 0/0 means are harmless."""

    def fit(idx: np.ndarray) -> np.ndarray:
        b = idx.shape[0]
        rows = np.repeat(np.arange(b), idx.shape[1])
        cb = codes[idx]
        sums = np.zeros((b, n_groups))
        counts = np.zeros((b, n_groups))
        np.add.at(sums, (rows, cb.ravel()), y[idx].ravel())
        np.add.at(counts, (rows, cb.ravel()), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / counts
        return means[np.arange(b)[:, None], cb]

    return fit

def _ols_fitted_batched(y: np.ndarray, X: np.ndarray):
    def fit(idx: np.ndarray) -> np.ndarray:
        Xb = X[idx]
        yb = y[idx]
        A = np.einsum("bnp,bnq->bpq", Xb, Xb, optimize=True)
        A += RIDGE * np.eye(X.shape[1])
        c = np.einsum("bnp,bn->bp", Xb, yb)
        beta = np.linalg.solve(A, c[:, :, None])[:, :, 0]
        return np.einsum("bnp,bp->bn", Xb, beta)

    return fit


def variance_decomposition(
    rows: Sequence[BiasRow], n_boot: int = 10_000, seed: int = 0
) -> dict:
    """R^2 of nested bias predictors, refit per bootstrap replicate."""
    if len(rows) < 3:
        raise ValueError("too few rows for a variance decomposition")
    y = np.array([row.b for row in rows])
    if float(y.var()) == 0.0:
        raise ValueError("b is constant; R^2 undefined")
    n = len(rows)

    cat_codes, cat_labels = _codes(rows, "category")
    sub_codes, sub_labels = _codes(rows, "subset")
    X = _design(rows)

    specs = {
        "category": (
            _fit_group_means(y, cat_codes, len(cat_labels)),
            _group_fitted_batched(y, cat_codes, len(cat_labels)),
        ),
        "subset": (
            _fit_group_means(y, sub_codes, len(sub_labels)),
            _group_fitted_batched(y, sub_codes, len(sub_labels)),
        ),
        "subset_plus_length": (_fit_ols(y, X), _ols_fitted_batched(y, X)),
    }

    out: dict = {
        "n_items": n,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "b_sd": float(y.std(ddof=1)),
        "median_abs_s": float(np.median(np.abs([row.s for row in rows]))),
        "specs": {},
    }
    for name in DECOMP_SPECS:
        fitted, refit = specs[name]
        r2_boot = _bootstrap_r2(y, refit, n, n_boot, seed)
        lo, hi = np.nanquantile(r2_boot, [0.025, 0.975])
        out["specs"][name] = {
            "r2": _r2(y, fitted),
            "r2_ci95": [float(lo), float(hi)],
            "resid_sd": float((y - fitted).std(ddof=1)),
        }

    out["category_bias_means"] = {}
    for label in cat_labels:
        values = y[cat_codes == cat_labels.index(label)]
        mean, lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
        out["category_bias_means"][label] = {
            "n_items": int(values.size),
            "mean": mean,
            "ci95": [lo, hi],
        }
    return out


# ---------------------------------------------------------------------------
# Single-order correction ladder (exact leave-one-out)
# ---------------------------------------------------------------------------

def _loo_group_means(y: np.ndarray, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, int]:
    """Exact LOO group mean per item; singleton groups fall back to the LOO
    grand mean (counted, so a sample where that matters is visible)."""
    sums = np.bincount(codes, weights=y, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups)
    grand_loo = (y.sum() - y) / (y.size - 1)
    singleton = counts[codes] == 1
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (sums[codes] - y) / (counts[codes] - 1)
    return np.where(singleton, grand_loo, loo), int(singleton.sum())


def _loo_regression(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Exact ridge LOO predictions via the hat-matrix identity
    ``pred_loo_i = y_i - e_i / (1 - h_ii)``."""
    A = X.T @ X + RIDGE * np.eye(X.shape[1])
    A_inv = np.linalg.solve(A, np.eye(X.shape[1]))
    h = np.einsum("np,pq,nq->n", X, A_inv, X)
    e = y - X @ (A_inv @ (X.T @ y))
    return y - e / np.clip(1.0 - h, 1e-8, None)


def loo_corrections(rows: Sequence[BiasRow]) -> tuple[dict[str, np.ndarray], dict]:
    """Per-item bias estimates ``b_hat`` for every ladder estimator."""
    if len(rows) < 3:
        raise ValueError("too few rows for LOO corrections")
    y = np.array([row.b for row in rows])
    cat_codes, cat_labels = _codes(rows, "category")
    sub_codes, sub_labels = _codes(rows, "subset")

    cat_hat, cat_singletons = _loo_group_means(y, cat_codes, len(cat_labels))
    sub_hat, sub_singletons = _loo_group_means(y, sub_codes, len(sub_labels))
    estimates = {
        "none": np.zeros_like(y),
        "global": (y.sum() - y) / (y.size - 1),
        "category": cat_hat,
        "subset": sub_hat,
        "regression": _loo_regression(y, _design(rows)),
        "oracle": y.copy(),
    }
    diagnostics = {
        "singleton_fallbacks": {"category": cat_singletons, "subset": sub_singletons},
        "min_group_size": {
            "category": int(np.bincount(cat_codes).min()),
            "subset": int(np.bincount(sub_codes).min()),
        },
    }
    return estimates, diagnostics


def corrected_scores(rows: Sequence[BiasRow], b_hat: np.ndarray) -> np.ndarray:
    """Expected accuracy per item of ``sign(z - b_hat)`` under a uniformly
    random presentation order; exact zeros score half (measure-zero in
    practice, but the oracle-equals-symmetrization identity needs them)."""
    z_cf = np.array([row.z_cf for row in rows]) - b_hat
    z_rf = np.array([row.z_rf for row in rows]) - b_hat
    acc_cf = (z_cf > 0) + 0.5 * (z_cf == 0)
    acc_rf = (z_rf < 0) + 0.5 * (z_rf == 0)
    return (acc_cf + acc_rf) / 2.0


def correction_ladder(
    rows: Sequence[BiasRow], n_boot: int = 10_000, seed: int = 0
) -> dict:
    """Accuracy of every corrected single-order judge, overall and per
    category, with paired deltas against the raw judge and the oracle."""
    estimates, diagnostics = loo_corrections(rows)
    scores = {name: corrected_scores(rows, b_hat) for name, b_hat in estimates.items()}
    categories = sorted({row.category for row in rows})
    cat_masks = {
        cat: np.array([row.category == cat for row in rows]) for cat in categories
    }

    out: dict = {
        "n_items": len(rows),
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "diagnostics": diagnostics,
        "estimators": {},
    }
    gain_span = scores["oracle"] - scores["none"]
    for name in CORRECTIONS:
        mean, lo, hi = bootstrap_mean_ci(scores[name], n_boot=n_boot, seed=seed)
        block = {"acc": {"mean": mean, "ci95": [lo, hi]}}
        if name != "none":
            d, d_lo, d_hi = bootstrap_mean_ci(
                scores[name] - scores["none"], n_boot=n_boot, seed=seed
            )
            block["delta_vs_none"] = {"mean": d, "ci95": [d_lo, d_hi]}
        if name != "oracle":
            d, d_lo, d_hi = bootstrap_mean_ci(
                scores[name] - scores["oracle"], n_boot=n_boot, seed=seed
            )
            block["delta_vs_oracle"] = {"mean": d, "ci95": [d_lo, d_hi]}
        block["by_category"] = {
            cat: float(scores[name][mask].mean()) for cat, mask in cat_masks.items()
        }
        out["estimators"][name] = block

    # Recovered fraction of the symmetrization gain, only where that gain is
    # itself significant (a ratio over a null denominator is noise).
    gain, gain_lo, gain_hi = bootstrap_mean_ci(gain_span, n_boot=n_boot, seed=seed)
    out["oracle_gain"] = {"mean": gain, "ci95": [gain_lo, gain_hi]}
    if gain_lo > 0 or gain_hi < 0:
        for name in ("global", "category", "subset", "regression"):
            out["estimators"][name]["recovered_fraction"] = float(
                (scores[name].mean() - scores["none"].mean()) / gain
            )
    return out


def bias_structure(rows: Sequence[BiasRow], n_boot: int = 10_000, seed: int = 0) -> dict:
    """The full phase-3 additive-shift block for one model."""
    return {
        "decomposition": variance_decomposition(rows, n_boot=n_boot, seed=seed),
        "ladder": correction_ladder(rows, n_boot=n_boot, seed=seed),
    }

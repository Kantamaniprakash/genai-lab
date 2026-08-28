"""Paired rubric-sensitivity analysis: the same judge under two rubrics.

The rubric axis asks: how much of a judge's behavior is a property of the
judge, and how much is a property of the prompt it was handed? Because every
store in this audit is keyed by (model, rubric) over the *same* stratified
sample, two rubrics can be compared item-by-item in log-odds — the same
pairing trick the swap-pair machinery uses for order, applied one level up.

The unit here is the :class:`RubricPair`: one item's complete swap pair under
rubric A matched with its complete swap pair under rubric B. Every headline
statistic is a paired per-item delta with a bootstrap CI over items:

- ``delta_s``  = s_B - s_A : did the rubric move the order-invariant
  preference toward the gold-chosen response?
- ``delta_b``  = b_B - b_A : did the rubric move the position bias?
- ``delta_abs_b`` = |b_B| - |b_A| : did it shrink the bias magnitude
  (the interesting direction for a debiasing-by-prompting claim)?
- paired accuracy deltas (symmetrized and per-order raw), and the paired
  compliance delta (the rubric changes the *instruction*, so format
  discipline is allowed to move too).

Beyond deltas, the view reports cross-rubric *consistency*: the rubric flip
rate (fraction of items whose symmetrized verdict changes when only the
rubric text changes — the prompt-level analogue of the positional flip rate)
and correlations of the decomposition components across rubrics. A judge
whose s correlates highly across rubrics is measuring something about the
responses; one whose s decorrelates is measuring the prompt.

Strict-sign conventions match :class:`src.analysis.SwapPair.sym_correct`:
an exactly-zero s counts half, so ties (unobserved in practice with float
log-odds) cannot bias either side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .analysis import SwapPair, bootstrap_mean_ci

N_BOOT = 10_000
SEED = 0


@dataclass(frozen=True)
class RubricPair:
    """One item's complete swap pairs under both rubrics."""

    item_id: str
    a: SwapPair  # rubric A (the reference, canonically "minimal")
    b: SwapPair  # rubric B (the treatment, e.g. "detailed")

    def __post_init__(self) -> None:
        if self.a.item_id != self.b.item_id or self.a.item_id != self.item_id:
            raise ValueError(
                f"mismatched item_ids: {self.item_id!r} / "
                f"{self.a.item_id!r} / {self.b.item_id!r}"
            )

    @property
    def delta_s(self) -> float:
        return self.b.s - self.a.s

    @property
    def delta_b(self) -> float:
        return self.b.b - self.a.b

    @property
    def delta_abs_b(self) -> float:
        return abs(self.b.b) - abs(self.a.b)

    @property
    def delta_abs_s(self) -> float:
        return abs(self.b.s) - abs(self.a.s)

    @property
    def sym_verdict_agrees(self) -> float:
        """1.0 if the symmetrized verdict is the same letter under both
        rubrics, 0.5 if either is exactly indifferent, else 0.0."""
        if self.a.s == 0 or self.b.s == 0:
            return 0.5
        return 1.0 if (self.a.s > 0) == (self.b.s > 0) else 0.0


def match_rubric_pairs(
    pairs_a: Sequence[SwapPair], pairs_b: Sequence[SwapPair]
) -> tuple[list[RubricPair], int, int]:
    """Inner-join two rubric stores on item_id.

    Returns (matched, n_only_a, n_only_b). Unmatched items are counted, not
    silently dropped: a paired analysis over mismatched item sets would carry
    exactly the composition confound finding 26 documented for prefixes, so
    callers must surface the counts whenever they are nonzero.
    """
    by_id_a = {p.item_id: p for p in pairs_a}
    by_id_b = {p.item_id: p for p in pairs_b}
    if len(by_id_a) != len(pairs_a) or len(by_id_b) != len(pairs_b):
        raise ValueError("duplicate item_ids within one rubric's pair list")
    shared = sorted(by_id_a.keys() & by_id_b.keys())
    matched = [RubricPair(item_id=i, a=by_id_a[i], b=by_id_b[i]) for i in shared]
    return matched, len(by_id_a) - len(shared), len(by_id_b) - len(shared)


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Pearson r with a percentile bootstrap CI over items.

    Degenerate replicates (either margin constant, so r is undefined) are
    recorded and excluded from the percentiles rather than coerced to 0.
    """
    if x.size != y.size or x.size < 3:
        raise ValueError("need >= 3 paired observations")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    xb, yb = x[idx], y[idx]
    xc = xb - xb.mean(axis=1, keepdims=True)
    yc = yb - yb.mean(axis=1, keepdims=True)
    denom = np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
    valid = denom > 0
    rs = (xc * yc).sum(axis=1)[valid] / denom[valid]
    lo, hi = np.quantile(rs, [alpha / 2, 1 - alpha / 2])
    point = float(np.corrcoef(x, y)[0, 1])
    return {
        "r": point,
        "ci95": [float(lo), float(hi)],
        "n_degenerate_replicates": int(n_boot - int(valid.sum())),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(v.size, dtype=np.float64)
        return r

    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def _mean_ci(values: Sequence[float], n_boot: int, seed: int) -> dict:
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
    return {"mean": mean, "ci95": [lo, hi]}


def _rubric_stats(side: Sequence[SwapPair], n_boot: int, seed: int) -> dict:
    """Per-rubric block on the matched items (so both sides are comparable)."""
    b_vals = np.array([p.b for p in side])
    s_vals = np.array([p.s for p in side])
    return {
        "sym_acc": _mean_ci([p.sym_correct for p in side], n_boot, seed),
        "raw_acc": float(np.mean([p.raw_correct_mean for p in side])),
        "raw_acc_cf": float(np.mean([p.raw_correct_cf for p in side])),
        "raw_acc_rf": float(np.mean([p.raw_correct_rf for p in side])),
        "positional_flip_rate": float(np.mean([p.positional_flip for p in side])),
        "median_b": float(np.median(b_vals)),
        "median_abs_b": float(np.median(np.abs(b_vals))),
        "median_abs_s": float(np.median(np.abs(s_vals))),
        "frac_bias_dominates": float(np.mean(np.abs(b_vals) > np.abs(s_vals))),
        "compliance_rate": float(np.mean([p.compliant_both for p in side])),
    }


def _paired_deltas(matched: Sequence[RubricPair], n_boot: int, seed: int) -> dict:
    return {
        "sym_acc": _mean_ci(
            [p.b.sym_correct - p.a.sym_correct for p in matched], n_boot, seed
        ),
        "raw_acc": _mean_ci(
            [p.b.raw_correct_mean - p.a.raw_correct_mean for p in matched],
            n_boot,
            seed,
        ),
        "positional_flip_rate": _mean_ci(
            [float(p.b.positional_flip) - float(p.a.positional_flip) for p in matched],
            n_boot,
            seed,
        ),
        "s": _mean_ci([p.delta_s for p in matched], n_boot, seed),
        "b": _mean_ci([p.delta_b for p in matched], n_boot, seed),
        "abs_b": _mean_ci([p.delta_abs_b for p in matched], n_boot, seed),
        "abs_s": _mean_ci([p.delta_abs_s for p in matched], n_boot, seed),
        "compliance": _mean_ci(
            [float(p.b.compliant_both) - float(p.a.compliant_both) for p in matched],
            n_boot,
            seed,
        ),
    }


def rubric_pair_view(
    pairs_a: Sequence[SwapPair],
    pairs_b: Sequence[SwapPair],
    rubric_a: str,
    rubric_b: str,
    category_of: Callable[[str], str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """The full paired view of one judge under two rubrics.

    ``pairs_a``/``pairs_b`` are complete swap pairs from the two stores of the
    *same* model; deltas read B - A throughout, so with A = minimal a positive
    ``deltas.sym_acc`` means the treatment rubric helped.
    """
    matched, only_a, only_b = match_rubric_pairs(pairs_a, pairs_b)
    if len(matched) < 3:
        raise ValueError(f"only {len(matched)} matched items; need >= 3")

    s_a = np.array([p.a.s for p in matched])
    s_b = np.array([p.b.s for p in matched])
    b_a = np.array([p.a.b for p in matched])
    b_b = np.array([p.b.b for p in matched])

    view: dict = {
        "rubric_a": rubric_a,
        "rubric_b": rubric_b,
        "n_matched_items": len(matched),
        "n_only_a": only_a,
        "n_only_b": only_b,
        "per_rubric": {
            rubric_a: _rubric_stats([p.a for p in matched], n_boot, seed),
            rubric_b: _rubric_stats([p.b for p in matched], n_boot, seed),
        },
        "deltas": _paired_deltas(matched, n_boot, seed),
        "consistency": {
            "rubric_flip_rate": _mean_ci(
                [1.0 - p.sym_verdict_agrees for p in matched], n_boot, seed
            ),
            "s_pearson": _bootstrap_corr_ci(s_a, s_b, n_boot, seed),
            "s_spearman": _spearman(s_a, s_b),
            "b_pearson": _bootstrap_corr_ci(b_a, b_b, n_boot, seed),
        },
        "n_boot": n_boot,
        "bootstrap_seed": seed,
    }

    if category_of is not None:
        by_cat: dict[str, list[RubricPair]] = {}
        for pair in matched:
            by_cat.setdefault(category_of(pair.item_id), []).append(pair)
        view["by_category"] = {
            cat: {
                "n_items": len(group),
                "sym_acc_a": float(np.mean([p.a.sym_correct for p in group])),
                "sym_acc_b": float(np.mean([p.b.sym_correct for p in group])),
                "delta_sym_acc": _mean_ci(
                    [p.b.sym_correct - p.a.sym_correct for p in group], n_boot, seed
                ),
                "mean_delta_s": float(np.mean([p.delta_s for p in group])),
                "mean_delta_b": float(np.mean([p.delta_b for p in group])),
                "mean_delta_abs_b": float(np.mean([p.delta_abs_b for p in group])),
                "rubric_flip_rate": float(
                    np.mean([1.0 - p.sym_verdict_agrees for p in group])
                ),
                "compliance_a": float(np.mean([p.a.compliant_both for p in group])),
                "compliance_b": float(np.mean([p.b.compliant_both for p in group])),
            }
            for cat, group in sorted(by_cat.items())
        }

    return view


def fragility_fit(
    matched: Sequence[RubricPair],
    n_quartiles: int = 4,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Fit the one-parameter perturbation account of rubric flips.

    The white-box reading of the rubric flip rate is that rewording the
    prompt perturbs the order-invariant preference, and the sign
    re-randomizes exactly where |s| is small relative to the perturbation.
    The simplest model with that content treats rubric B's preference as a
    contracted copy of rubric A's plus homoskedastic noise,

        s_B = lam * s_A + eps,   eps ~ N(0, sigma^2),

    under which an item with reference preference s flips sign with
    probability Phi(-lam * |s| / sigma). ``lam`` is the least-squares
    through-origin slope and ``sigma`` the residual sd; the return value
    reports both plus, per quartile of |s_A|, the observed flip rate (with
    a bootstrap CI) against the model's prediction — so the fit is
    falsifiable bin by bin, not just on the pooled rate.

    Strict-sign convention: an exactly-zero s on either side counts as half
    a flip, matching ``RubricPair.sym_verdict_agrees``.
    """
    if len(matched) < 3:
        raise ValueError(f"only {len(matched)} matched items; need >= 3")
    s_a = np.array([p.a.s for p in matched])
    s_b = np.array([p.b.s for p in matched])
    flips = np.array([1.0 - p.sym_verdict_agrees for p in matched])

    denom = float(np.sum(s_a * s_a))
    if denom == 0.0:
        raise ValueError("all reference preferences are exactly zero")
    lam = float(np.sum(s_a * s_b) / denom)
    resid = s_b - lam * s_a
    sigma = float(np.std(resid, ddof=1))

    def predicted_flip(abs_s: np.ndarray) -> np.ndarray:
        from math import erf, sqrt

        z = -lam * abs_s / sigma
        return np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in z])

    abs_s = np.abs(s_a)
    edges = np.quantile(abs_s, np.linspace(0, 1, n_quartiles + 1)[1:-1])
    bin_of = np.digitize(abs_s, edges)
    bins = []
    for k in range(n_quartiles):
        mask = bin_of == k
        bins.append(
            {
                "n_items": int(mask.sum()),
                "median_abs_s": float(np.median(abs_s[mask])),
                "observed_flip": _mean_ci(list(flips[mask]), n_boot, seed),
                "predicted_flip": float(np.mean(predicted_flip(abs_s[mask]))),
            }
        )
    return {
        "n_items": len(matched),
        "lam": lam,
        "sigma": sigma,
        "median_abs_s": float(np.median(abs_s)),
        "overall_flip": _mean_ci(list(flips), n_boot, seed),
        "overall_predicted_flip": float(np.mean(predicted_flip(abs_s))),
        "quartiles": bins,
    }

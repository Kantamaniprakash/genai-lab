"""Swap-pair assembly, the preference/bias decomposition, and paired bootstrap.

The unit of every analysis in this audit is the *swap pair*: the two verdict
log-odds a judge produced for one item under both presentation orders. The
exact decomposition (an identity, not a model) is

    s_i = (z_cf - z_rf) / 2   order-invariant preference for the gold-chosen
    b_i = (z_cf + z_rf) / 2   position bias toward whatever sits at A

so ``z_cf = b_i + s_i`` and ``z_rf = b_i - s_i`` reconstruct the raw readouts.

Uncertainty is quantified by a paired bootstrap over items (both orders of an
item always resampled together), matching the machinery this lab used in
``rag-chunking-bench``: percentile intervals, fixed seed, resample counts
recorded in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .judge import JudgmentRecord


@dataclass(frozen=True)
class SwapPair:
    """Both readouts for one item, plus the diagnostics that qualify them."""

    item_id: str
    z_cf: float           # verdict log-odds, chosen shown first (expected "A")
    z_rf: float           # verdict log-odds, rejected shown first (expected "B")
    compliant_both: bool  # unconstrained argmax was a verdict letter in both orders
    mass_min: float       # min over orders of probability mass on {A, B}

    @property
    def s(self) -> float:
        """Order-invariant preference log-odds for the gold-chosen response."""
        return (self.z_cf - self.z_rf) / 2

    @property
    def b(self) -> float:
        """Position-bias log-odds toward position A."""
        return (self.z_cf + self.z_rf) / 2

    @property
    def raw_correct_cf(self) -> float:
        return 1.0 if self.z_cf >= 0 else 0.0

    @property
    def raw_correct_rf(self) -> float:
        return 1.0 if self.z_rf < 0 else 0.0

    @property
    def raw_correct_mean(self) -> float:
        """Expected accuracy of the unswapped judge when the presentation
        order of this item is assigned uniformly at random."""
        return (self.raw_correct_cf + self.raw_correct_rf) / 2

    @property
    def sym_correct(self) -> float:
        """Accuracy of the symmetrized (swap-averaged) verdict, sign(s)."""
        if self.s > 0:
            return 1.0
        if self.s < 0:
            return 0.0
        return 0.5

    @property
    def positional_flip(self) -> bool:
        """True when the positional verdict changed under the swap — the only
        event a black-box flip-rate audit can observe."""
        return (self.z_cf >= 0) != (self.z_rf >= 0)


def assemble_pairs(records: Sequence[JudgmentRecord]) -> tuple[list[SwapPair], int]:
    """Group records into swap pairs; returns (pairs, n_incomplete_items).

    Requires a homogeneous (model, rubric) record set — mixing audits in one
    assembly would silently average across conditions. Items present in only
    one order (an interrupted run) are counted, not dropped silently.
    """
    if not records:
        return [], 0
    signatures = {(r.model, r.rubric) for r in records}
    if len(signatures) > 1:
        raise ValueError(f"mixed (model, rubric) sets in one assembly: {sorted(signatures)}")

    by_item: dict[str, dict[str, JudgmentRecord]] = {}
    for record in records:
        slot = by_item.setdefault(record.item_id, {})
        if record.order in slot:
            raise ValueError(f"duplicate record for {record.item_id} / {record.order}")
        slot[record.order] = record

    pairs: list[SwapPair] = []
    incomplete = 0
    for item_id in sorted(by_item):
        slot = by_item[item_id]
        if len(slot) != 2:
            incomplete += 1
            continue
        cf, rf = slot["chosen_first"], slot["rejected_first"]
        pairs.append(
            SwapPair(
                item_id=item_id,
                z_cf=cf.z,
                z_rf=rf.z,
                compliant_both=cf.compliant and rf.compliant,
                mass_min=min(cf.mass_ab, rf.mass_ab),
            )
        )
    return pairs, incomplete


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap_delta_ci(
    pairs: Sequence[SwapPair],
    metric_a: Callable[[SwapPair], float],
    metric_b: Callable[[SwapPair], float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(delta, lo, hi) for mean(metric_a) - mean(metric_b), resampling items.

    Both metrics are evaluated on the same resampled items, so the interval
    accounts for the pairing (the whole point of running both orders on every
    item rather than randomizing order across items).
    """
    deltas = np.asarray([metric_a(p) - metric_b(p) for p in pairs], dtype=np.float64)
    return bootstrap_mean_ci(deltas, n_boot=n_boot, seed=seed, alpha=alpha)


def two_sample_bootstrap_delta_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(delta, lo, hi) for mean(a) - mean(b) over two *disjoint* item sets.

    Unlike :func:`paired_bootstrap_delta_ci` this resamples the two groups
    independently — the right tool when comparing strata (e.g. compliant vs.
    non-compliant items), where no pairing exists by construction.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("empty stratum")
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
    deltas = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


# Bins over mass_min (the min over both orders of probability mass on {A, B}).
# Edges chosen from the Llama-3.2-1B quartiles observed 2026-07-18
# ([0.10, 0.67, 0.94]) so the extremes — readout nearly meaningless vs. nearly
# the whole distribution — get their own bins. Upper edge is inclusive.
MASS_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 0.9),
    (0.9, 1.0),
)


def _stratum_stats(pairs: Sequence[SwapPair], n_boot: int, seed: int) -> dict:
    """The compact per-stratum block: accuracies with CIs plus decomposition
    medians. Smaller than :func:`summarize_pairs` because strata can be tiny."""
    b_values = np.array([p.b for p in pairs])
    s_values = np.array([p.s for p in pairs])
    sym_mean, sym_lo, sym_hi = bootstrap_mean_ci(
        [p.sym_correct for p in pairs], n_boot=n_boot, seed=seed
    )
    return {
        "n_items": len(pairs),
        "sym_acc": {"mean": sym_mean, "ci95": [sym_lo, sym_hi]},
        "raw_acc": float(np.mean([p.raw_correct_mean for p in pairs])),
        "positional_flip_rate": float(np.mean([p.positional_flip for p in pairs])),
        "median_b": float(np.median(b_values)),
        "median_abs_b": float(np.median(np.abs(b_values))),
        "median_abs_s": float(np.median(np.abs(s_values))),
        "frac_bias_dominates": float(np.mean(np.abs(b_values) > np.abs(s_values))),
        "mass_ab_min_p50": float(np.median([p.mass_min for p in pairs])),
    }


def compliance_view(
    pairs: Sequence[SwapPair],
    category_of: Callable[[str], str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Condition every headline number on the validity of the readout itself.

    Motivated by finding 5 (2026-07-18): at 1B, only ~half the items have a
    verdict-letter argmax in both orders, and the mass on {A, B} spans the
    whole unit interval — so ``z`` sometimes measures a renormalized
    sub-distribution rather than a verdict. This view answers: do the audit's
    conclusions change when restricted to items where the single-token
    contract actually held?

    Three cuts, all descriptive (compliance is judge behavior, not a
    randomized treatment — strata can differ in item difficulty, and the
    per-category composition block is provided precisely so composition
    confounds are visible rather than hidden):

    - strata by argmax compliance (all / compliant in both orders / rest),
      with an unpaired bootstrap CI on the symmetrized-accuracy gap;
    - a validity curve over ``mass_min`` bins (:data:`MASS_BINS`);
    - per-category compliance rates and within-stratum accuracies.
    """
    if not pairs:
        raise ValueError("no complete swap pairs")
    compliant = [p for p in pairs if p.compliant_both]
    rest = [p for p in pairs if not p.compliant_both]

    strata = {"all": _stratum_stats(pairs, n_boot, seed)}
    if compliant:
        strata["compliant_both"] = _stratum_stats(compliant, n_boot, seed)
    if rest:
        strata["non_compliant"] = _stratum_stats(rest, n_boot, seed)

    delta = None
    if compliant and rest:
        d, lo, hi = two_sample_bootstrap_delta_ci(
            [p.sym_correct for p in compliant],
            [p.sym_correct for p in rest],
            n_boot=n_boot,
            seed=seed,
        )
        delta = {"mean": d, "ci95": [lo, hi]}

    mass_bins = []
    for lo_edge, hi_edge in MASS_BINS:
        # Half-open bins; the last is unbounded above so every pair lands
        # somewhere even if float wobble pushes mass_min a hair past 1.0.
        last = (lo_edge, hi_edge) == MASS_BINS[-1]
        members = [
            p for p in pairs
            if lo_edge <= p.mass_min and (last or p.mass_min < hi_edge)
        ]
        block = {"lo": lo_edge, "hi": hi_edge, "n_items": len(members)}
        if members:
            block.update(_stratum_stats(members, n_boot, seed))
        mass_bins.append(block)

    view = {
        "n_items": len(pairs),
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "compliance_rate": len(compliant) / len(pairs),
        "strata": strata,
        "sym_acc_compliant_minus_non": delta,
        "mass_bins": mass_bins,
    }

    if category_of is not None:
        by_cat: dict[str, list[SwapPair]] = {}
        for pair in pairs:
            by_cat.setdefault(category_of(pair.item_id), []).append(pair)
        view["by_category"] = {
            cat: {
                "n_items": len(group),
                "compliance_rate": float(np.mean([p.compliant_both for p in group])),
                "sym_acc_compliant": (
                    float(np.mean([p.sym_correct for p in group if p.compliant_both]))
                    if any(p.compliant_both for p in group) else None
                ),
                "sym_acc_non_compliant": (
                    float(np.mean([p.sym_correct for p in group if not p.compliant_both]))
                    if any(not p.compliant_both for p in group) else None
                ),
            }
            for cat, group in sorted(by_cat.items())
        }
    return view


def summarize_pairs(pairs: Sequence[SwapPair], n_boot: int = 10_000, seed: int = 0) -> dict:
    """The standard quick-look block for one (model, rubric) store."""
    if not pairs:
        raise ValueError("no complete swap pairs")
    b_values = np.array([p.b for p in pairs])
    s_values = np.array([p.s for p in pairs])

    raw_mean, raw_lo, raw_hi = bootstrap_mean_ci(
        [p.raw_correct_mean for p in pairs], n_boot=n_boot, seed=seed
    )
    sym_mean, sym_lo, sym_hi = bootstrap_mean_ci(
        [p.sym_correct for p in pairs], n_boot=n_boot, seed=seed
    )
    delta, delta_lo, delta_hi = paired_bootstrap_delta_ci(
        pairs, lambda p: p.sym_correct, lambda p: p.raw_correct_mean,
        n_boot=n_boot, seed=seed,
    )
    return {
        "n_items": len(pairs),
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "compliance_rate": float(np.mean([p.compliant_both for p in pairs])),
        "mass_ab_min_p50": float(np.median([p.mass_min for p in pairs])),
        "raw_acc": {"mean": raw_mean, "ci95": [raw_lo, raw_hi]},
        "raw_acc_chosen_first": float(np.mean([p.raw_correct_cf for p in pairs])),
        "raw_acc_rejected_first": float(np.mean([p.raw_correct_rf for p in pairs])),
        "sym_acc": {"mean": sym_mean, "ci95": [sym_lo, sym_hi]},
        "sym_minus_raw": {"mean": delta, "ci95": [delta_lo, delta_hi]},
        "positional_flip_rate": float(np.mean([p.positional_flip for p in pairs])),
        "bias_b": {
            "mean": float(b_values.mean()),
            "median": float(np.median(b_values)),
            "sd": float(b_values.std(ddof=1)),
            "iqr": [float(np.quantile(b_values, 0.25)), float(np.quantile(b_values, 0.75))],
            "frac_positive": float(np.mean(b_values > 0)),
        },
        "preference_s": {
            "mean": float(s_values.mean()),
            "median_abs": float(np.median(np.abs(s_values))),
            "sd": float(s_values.std(ddof=1)),
        },
        "frac_bias_dominates": float(np.mean(np.abs(b_values) > np.abs(s_values))),
    }

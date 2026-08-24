"""How representative a partial grid is, under each execution order.

Finding 26 established that a grid stopped part-way was an alphabetical prefix
of the subsets, so an interim read described an unrepresentative corner of the
benchmark rather than the benchmark. :mod:`src.schedule` fixes the cause by
choosing the next item to close the largest proportional deficit. This renders
what that buys, as the distance between the finished part of a grid and the
composition it is meant to be sampling.

The y-axis is total-variation distance between the realized and target stratum
distributions: 0 is a perfectly proportional prefix, and (for the category
panel) 0.746 is where the inherited Qwen2.5-7B store actually sat. Three
trajectories are drawn:

- **sorted (legacy)** — the order this replaced. It stays near 1 for a long
  time because whole subsets are finished before the next is touched, and the
  sawtooth is each subset in turn being completed.
- **balanced** — a grid started under the new scheduler. The first handful of
  items cannot be proportional at all (one finished item is 100% of one
  subset), but the deficit rule holds every subset within one item of
  proportional at every prefix — a bound, not an average — so the distance
  collapses as fast as integrality permits and stays under 0.05 from item 55
  on, against item 561 for the legacy order.
- **balanced, resuming the legacy prefix** — the actual Qwen2.5-7B store,
  which inherited 67 sorted-order items before the scheduler existed. Those
  items cannot be un-judged, so the distance can only dilute as the store
  grows; the curve is how fast it does.

The figure runs no judgments and reads no result store, so it is a stable
artifact of (sample, schedule) and does not change as the grid advances.

Usage:
    python -m experiments.schedule_coverage [--n 600] [--seed 0] [--switched-at 67]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_rewardbench, stratified_sample  # noqa: E402
from src.judge import RESULTS_DIR  # noqa: E402
from src.schedule import balanced_order, by_category, by_subset, coverage  # noqa: E402

from experiments.make_figures import AXIS, GRID, INK, INK_MUTED, INK_SECONDARY  # noqa: E402

FIGURES_DIR = RESULTS_DIR.parent / "figures"

SORTED_COLOR = "#c2483a"
BALANCED_COLOR = "#2b6cb0"
RESUME_COLOR = "#eda100"


def trajectory(items, order, *, stratum, seed_finished=()) -> tuple[list[int], list[float]]:
    """Total-variation distance from target after each scheduled item."""
    finished = list(seed_finished)
    xs, ys = [], []
    for item in order:
        finished.append(item.item_id)
        xs.append(len(finished))
        ys.append(coverage(items, finished, stratum=stratum)["total_variation"])
    return xs, ys


def panel(ax, items, *, stratum, label: str, switched_at: int) -> None:
    legacy = sorted(items, key=lambda it: it.item_id)
    inherited = [i.item_id for i in legacy[:switched_at]]

    ax.plot(*trajectory(items, legacy, stratum=stratum),
            color=SORTED_COLOR, linewidth=1.4, label="sorted (legacy)", zorder=3)
    ax.plot(*trajectory(items, balanced_order(items), stratum=stratum),
            color=BALANCED_COLOR, linewidth=1.6, label="balanced", zorder=5)

    resumed = balanced_order(items, {i: 2 for i in inherited})
    xs, ys = trajectory(items, resumed, stratum=stratum, seed_finished=inherited)
    ax.plot(xs, ys, color=RESUME_COLOR, linewidth=1.6,
            label=f"balanced, resuming {switched_at} sorted items", zorder=4)
    ax.plot([switched_at], [coverage(items, inherited, stratum=stratum)["total_variation"]],
            marker="o", markersize=4.5, color=RESUME_COLOR, zorder=6)

    ax.set_xlim(0, len(items))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("items finished in the store")
    ax.set_title(f"stratum: {label}", pad=6)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--switched-at", type=int, default=67,
                        help="store size in items when balanced scheduling was "
                             "introduced (2026-08-24); the Qwen2.5-7B store's "
                             "inherited sorted-order prefix")
    args = parser.parse_args()

    items = stratified_sample(load_rewardbench(), n=args.n, seed=args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    panel(axes[0], items, stratum=by_subset, label="subset (23)", switched_at=args.switched_at)
    panel(axes[1], items, stratum=by_category, label="category (4)", switched_at=args.switched_at)
    axes[0].set_ylabel("total-variation distance from\ntarget composition")

    axes[0].annotate(
        "half the grid done and still\n0.50 from the benchmark",
        xy=(300, 0.497), xytext=(316, 0.73), fontsize=8, color=SORTED_COLOR,
        arrowprops={"arrowstyle": "->", "color": SORTED_COLOR, "linewidth": 0.8},
    )
    axes[0].annotate(
        "under 0.05 from item 55 on",
        xy=(120, 0.030), xytext=(150, 0.22), fontsize=8, color=BALANCED_COLOR,
        arrowprops={"arrowstyle": "->", "color": BALANCED_COLOR, "linewidth": 0.8},
    )
    axes[1].annotate(
        f"inherited {args.switched_at}-item\nsorted prefix",
        xy=(args.switched_at, 0.746), xytext=(150, 0.88), fontsize=8, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "->", "color": INK_MUTED, "linewidth": 0.8},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.28, top=0.82, wspace=0.08)
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "A partial grid is only a sample of the benchmark if the schedule makes it one",
        fontsize=10.5, color=INK, y=0.97,
    )
    for ax in axes:
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ax.tick_params(color=GRID)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "schedule_coverage.png"
    fig.savefig(out)
    print(f"wrote {out}")

    for stratum, name in ((by_subset, "subset"), (by_category, "category")):
        legacy = sorted(items, key=lambda it: it.item_id)
        inherited = [i.item_id for i in legacy[:args.switched_at]]
        _, sorted_tv = trajectory(items, legacy, stratum=stratum)
        _, bal_tv = trajectory(items, balanced_order(items), stratum=stratum)
        xs, res_tv = trajectory(items, balanced_order(items, {i: 2 for i in inherited}),
                                stratum=stratum, seed_finished=inherited)
        half = args.n // 2
        print(f"[{name}] worst TV  sorted {max(sorted_tv):.3f}  balanced {max(bal_tv):.3f}")
        print(f"[{name}] TV at {half:3d} items  sorted {sorted_tv[half - 1]:.3f}  "
              f"balanced {bal_tv[half - 1]:.3f}  "
              f"resumed {res_tv[xs.index(half)]:.3f}")


if __name__ == "__main__":
    main()

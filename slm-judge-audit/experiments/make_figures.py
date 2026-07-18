"""Render figures from raw judgment stores. Nothing is hand-entered.

    python -m experiments.make_figures                  # all stores
    python -m experiments.make_figures --store qwen2.5-0.5b__minimal

Per store:
  {store}_decomposition.png  — per-item scatter of position bias b_i vs
                               preference s_i, the audit's core picture
  {store}_accuracy.png       — raw / per-order / symmetrized accuracy with
                               95% paired-bootstrap CIs against the floors

Error bars are 95% percentile bootstrap CIs over items (10,000 resamples,
fixed seed), identical to the summary tables. Style follows the lab
conventions from rag-chunking-bench: CVD-safe hues, identity never carried by
color alone, direct labels where possible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import assemble_pairs, bootstrap_mean_ci  # noqa: E402
from src.baselines import longer_response_correct  # noqa: E402
from src.data import fetch, load_rewardbench  # noqa: E402
from src.judge import RESULTS_DIR, load_records  # noqa: E402

FIGURES_DIR = RESULTS_DIR.parent / "figures"

CATEGORY_STYLES = {
    "Chat": ("#2a78d6", "o"),
    "Chat Hard": ("#1baf7a", "s"),
    "Safety": ("#eda100", "^"),
    "Reasoning": ("#9a5bd2", "D"),
}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def category_of(item_id: str) -> str:
    from src.data import SUBSET_TO_CATEGORY

    return SUBSET_TO_CATEGORY[item_id.split("/", 1)[0]]


def decomposition_figure(store: str, pairs, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    # The additive-shift reading: a biased-but-useful judge concentrates in a
    # vertical band away from b=0; points below s=0 are debiased *errors*.
    for cat, (color, marker) in CATEGORY_STYLES.items():
        xs = [p.b for p in pairs if category_of(p.item_id) == cat]
        ys = [p.s for p in pairs if category_of(p.item_id) == cat]
        if not xs:
            continue
        ax.scatter(xs, ys, s=14, alpha=0.55, linewidths=0, color=color,
                   marker=marker, label=f"{cat} (n={len(xs)})", zorder=2)
    mean_b = sum(p.b for p in pairs) / len(pairs)
    ax.axvline(mean_b, color=INK_SECONDARY, linewidth=1.0, linestyle="--", zorder=3)
    ax.annotate(f"mean bias b = {mean_b:+.2f}", xy=(mean_b, ax.get_ylim()[1]),
                xytext=(4, -2), textcoords="offset points", fontsize=8,
                color=INK_SECONDARY, va="top")
    ax.set_xlabel("position bias  $b_i = (z_{cf} + z_{rf})/2$  (log-odds toward position A)")
    ax.set_ylabel("preference  $s_i = (z_{cf} - z_{rf})/2$\n(log-odds toward gold-chosen)")
    ax.set_title(f"{store}: swap-pair decomposition, n={len(pairs)} items")
    ax.grid(True, axis="both", alpha=0.6)
    ax.legend(loc="lower right", markerscale=1.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def accuracy_figure(store: str, pairs, items_by_id: dict, out: Path) -> None:
    raw = bootstrap_mean_ci([p.raw_correct_mean for p in pairs])
    sym = bootstrap_mean_ci([p.sym_correct for p in pairs])
    cf = bootstrap_mean_ci([p.raw_correct_cf for p in pairs])
    rf = bootstrap_mean_ci([p.raw_correct_rf for p in pairs])
    longer = bootstrap_mean_ci(
        [longer_response_correct(items_by_id[p.item_id]) for p in pairs]
    )

    labels = ["chosen\nshown first", "rejected\nshown first",
              "raw\n(random order)", "symmetrized\n(swap-averaged)"]
    stats = [cf, rf, raw, sym]
    colors = ["#86b6ef", "#86b6ef", "#2a78d6", "#184f95"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = range(len(stats))
    means = [s[0] for s in stats]
    errs = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
    bars = ax.bar(xs, means, width=0.62, color=colors, zorder=2)
    ax.errorbar(xs, means, yerr=errs, fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1.1, capsize=3, zorder=3)
    for bar, mean in zip(bars, means):
        ax.annotate(f"{mean:.3f}", xy=(bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    # Floor annotations live in the empty column above the rejected-first bar
    # (always near zero for a biased judge), where nothing occludes them.
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=4)
    ax.annotate("random / always-A floor (0.5)", xy=(1.0, 0.5),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8, color=INK_MUTED, zorder=5)
    ax.axhline(longer[0], color="#c2483f", linewidth=1.0, linestyle="--", zorder=4)
    ax.annotate(f"longer-response floor ({longer[0]:.3f})",
                xy=(1.0, longer[0]), xytext=(0, -11),
                textcoords="offset points", ha="center", fontsize=8,
                color="#c2483f", zorder=5)
    ax.set_xticks(list(xs), labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("accuracy vs. gold label")
    ax.set_title(f"{store}: accuracy by presentation order (n={len(pairs)}, 95% CI)")
    ax.grid(True, axis="y", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=None)
    args = parser.parse_args()

    fetch()
    items_by_id = {item.item_id: item for item in load_rewardbench()}
    paths = ([RESULTS_DIR / f"{args.store}.jsonl"] if args.store
             else sorted(RESULTS_DIR.glob("*__*.jsonl")))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[figures] {path.stem}: skipping {incomplete} incomplete items")
        decomposition_figure(path.stem, pairs, FIGURES_DIR / f"{path.stem}_decomposition.png")
        accuracy_figure(path.stem, pairs, items_by_id,
                        FIGURES_DIR / f"{path.stem}_accuracy.png")
        print(f"[figures] wrote 2 figures for {path.stem} -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()

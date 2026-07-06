"""Render the single headline figure embedded at the top of the READMEs.

    python -m experiments.make_hero_figure --dataset dev-v1.1 --retriever bm25

One compact strategy-comparison chart: SpanRecall at a practical budget
(B=400) for every baseline configuration, grouped by chunker family, with
95% bootstrap CIs. Regenerated from ``results/raw/`` only — nothing is
hand-entered — and written to ``assets/``. Style matches
``experiments.make_figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.aggregate import check_aligned, load_raw, mean_ci

ROOT = Path(__file__).resolve().parent.parent

CHUNKER_ORDER = ("fixed", "sentence", "recursive")
SIZE_RAMP = {64: "#86b6ef", 128: "#5598e7", 256: "#2a78d6", 512: "#184f95"}

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


def fig_hero(results, out: Path, budget: int = 400) -> None:
    """Grouped bars: SpanRecall@budget per chunker family x chunk size."""
    grid = {(rr.config["chunker"], rr.config["chunk_size"]): rr for rr in results}
    sizes = sorted({rr.config["chunk_size"] for rr in results})
    n_questions = len(results[0].records)

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    group_width = 0.78
    bar_width = group_width / len(sizes)
    tick_pos: list[float] = []
    tick_labels: list[str] = []
    for gi, chunker in enumerate(CHUNKER_ORDER):
        for si, size in enumerate(sizes):
            ci = mean_ci(grid[(chunker, size)].metric("recall", budget))
            x = gi + (si - (len(sizes) - 1) / 2) * bar_width
            tick_pos.append(x)
            tick_labels.append(str(size))
            ax.bar(
                x,
                ci.mean_diff,
                width=bar_width * 0.92,
                color=SIZE_RAMP[size],
                edgecolor=SURFACE,
                linewidth=0.6,
            )
            ax.errorbar(
                x,
                ci.mean_diff,
                yerr=[[ci.mean_diff - ci.ci_low], [ci.ci_high - ci.mean_diff]],
                color=INK,
                linewidth=1.0,
                capsize=2.5,
            )
        # Group label below the per-bar size ticks.
        ax.text(
            gi,
            -0.15,
            f"{chunker} chunker",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("chunk size (tokens)", labelpad=26)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"SpanRecall@{budget} (mean, {n_questions:,} questions)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title(
        f"Budget-matched retrieval at B={budget} tokens: smaller chunks win, "
        "and oversized chunks collapse\n"
        "(SQuAD dev-v1.1 articles, BM25, zero overlap; error bars: 95% bootstrap CI)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the README hero figure from raw results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--budget", type=int, default=400)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline = load_raw(
        args.raw_dir,
        dataset=args.dataset,
        retriever=args.retriever,
        budget_rule="stop",
        overlap=0,
    )
    if not baseline:
        raise SystemExit(f"no results for {args.dataset}/{args.retriever} in {args.raw_dir}")
    check_aligned(baseline)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"hero_spanrecall_{args.dataset}_{args.retriever}.png"
    fig_hero(baseline, out, budget=args.budget)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

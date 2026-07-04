"""Render the README figures from raw grid results.

    python -m experiments.make_figures --dataset dev-v1.1 --retriever bm25

Figures are regenerated from ``results/raw/`` only — nothing is hand-entered —
and written to ``results/figures/``. Error bands are 95% bootstrap CIs of the
per-question mean (10,000 resamples, fixed seed), matching the tables.

Style notes: one consistent look across the project — categorical series use a
fixed CVD-validated hue order (blue, aqua, yellow for fixed / sentence /
recursive); the chunk-size dimension uses a single-hue ordinal blue ramp
(small→light, large→dark); identity is never carried by color alone (legends +
direct labels).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from experiments.aggregate import RunResult, check_aligned, load_raw, mean_ci

ROOT = Path(__file__).resolve().parent.parent

# Reference palette (validated for CVD separation and lightness band).
CHUNKER_COLORS = {"fixed": "#2a78d6", "sentence": "#1baf7a", "recursive": "#eda100"}
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


def _by_config(results: list[RunResult]) -> dict[tuple[str, int], RunResult]:
    return {(rr.config["chunker"], rr.config["chunk_size"]): rr for rr in results}


def _log2_axis(ax, values: list[int], label: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(label)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def fig_budget_curves(results: list[RunResult], out: Path) -> None:
    """SpanRecall vs. budget, one panel per chunker, one line per chunk size."""
    grid = _by_config(results)
    budgets = [int(b) for b in results[0].config["budgets"]]
    sizes = sorted({rr.config["chunk_size"] for rr in results})
    fig, axes = plt.subplots(1, len(CHUNKER_ORDER), figsize=(9.6, 3.2), sharey=True)
    for ax, chunker in zip(axes, CHUNKER_ORDER):
        for size in sizes:
            rr = grid[(chunker, size)]
            cis = [mean_ci(rr.metric("recall", b)) for b in budgets]
            means = [c.mean_diff for c in cis]
            ax.fill_between(
                budgets,
                [c.ci_low for c in cis],
                [c.ci_high for c in cis],
                color=SIZE_RAMP[size],
                alpha=0.22,
                linewidth=0,
            )
            ax.plot(
                budgets,
                means,
                color=SIZE_RAMP[size],
                linewidth=2,
                marker="o",
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=f"{size}",
            )
        ax.set_title(f"{chunker} chunker")
        ax.set_ylim(0, 1.0)
        _log2_axis(ax, budgets, "token budget B (log scale)")
    axes[0].set_ylabel("SpanRecall@B (mean, 2,400 questions)")
    axes[-1].legend(
        title="chunk size (tokens)",
        loc="lower right",
        ncols=2,
        title_fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "Budget-matched retrieval: smaller chunks dominate at every budget "
        "(bands: 95% bootstrap CI)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_metric_reversal(results: list[RunResult], out: Path) -> None:
    """hit@5 vs. SpanRecall@400 across chunk sizes: the ranking reverses."""
    grid = _by_config(results)
    sizes = sorted({rr.config["chunk_size"] for rr in results})
    fig, (ax_hit, ax_span) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for chunker in CHUNKER_ORDER:
        color = CHUNKER_COLORS[chunker]
        hit_cis = [mean_ci(grid[(chunker, s)].hits(5)) for s in sizes]
        span_cis = [mean_ci(grid[(chunker, s)].metric("recall", 400)) for s in sizes]
        for ax, cis in ((ax_hit, hit_cis), (ax_span, span_cis)):
            ax.fill_between(
                sizes,
                [c.ci_low for c in cis],
                [c.ci_high for c in cis],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                sizes,
                [c.mean_diff for c in cis],
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=chunker,
            )
    ax_hit.set_title("Fixed-k metric: hit@5\n(rises with chunk size)")
    ax_hit.set_ylabel("hit@5")
    ax_hit.set_ylim(0.55, 1.0)
    ax_hit.legend(title="chunker", loc="lower right", title_fontsize=8, labelcolor=INK_SECONDARY)
    ax_span.set_title("Budget-matched: SpanRecall@400\n(falls with chunk size)")
    ax_span.set_ylabel("SpanRecall@400")
    ax_span.set_ylim(0, 1.0)
    for ax in (ax_hit, ax_span):
        _log2_axis(ax, sizes, "chunk size (tokens, log scale)")
    fig.suptitle(
        "The same grid, two verdicts: fixed-k rewards large chunks; "
        "budget matching reverses it (bands: 95% CI)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render README figures from raw results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = load_raw(args.raw_dir, dataset=args.dataset, retriever=args.retriever)
    if not results:
        raise SystemExit(f"no results for {args.dataset}/{args.retriever} in {args.raw_dir}")
    check_aligned(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves = args.out_dir / f"recall_budget_curves_{args.dataset}_{args.retriever}.png"
    reversal = args.out_dir / f"metric_reversal_{args.dataset}_{args.retriever}.png"
    fig_budget_curves(results, curves)
    fig_metric_reversal(results, reversal)
    for path in (curves, reversal):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

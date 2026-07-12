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

from experiments.aggregate import RunResult, check_aligned, diff_ci, load_raw, mean_ci

ROOT = Path(__file__).resolve().parent.parent

# Reference palette (validated for CVD separation and lightness band).
CHUNKER_COLORS = {"fixed": "#2a78d6", "sentence": "#1baf7a", "recursive": "#eda100"}
CHUNKER_ORDER = ("fixed", "sentence", "recursive")
SIZE_RAMP = {64: "#86b6ef", 128: "#5598e7", 256: "#2a78d6", 512: "#184f95"}
# Retrievers reuse the same validated hue order (identity also carried by
# marker shape, so the two categorical dimensions cannot be confused across
# figures: chunker families are panel titles here, never colors).
RETRIEVER_STYLES = {
    "bm25": ("#2a78d6", "o"),
    "tfidf": ("#1baf7a", "s"),
    "lsa": ("#eda100", "^"),
    "dense": ("#9a5bd2", "D"),
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
    n_questions = len(results[0].records)
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
    axes[0].set_ylabel(f"SpanRecall@B (mean, {n_questions:,} questions)")
    axes[-1].legend(
        title="chunk size (tokens)",
        loc="lower right",
        ncols=2,
        title_fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    # Titles state what is drawn, not a verdict: the size ordering is
    # dataset-dependent (SQuAD: small dominates; Chroma: crossover — see
    # fig_gold_length_crossover), and the same code renders both.
    fig.suptitle(
        f"Budget-matched SpanRecall by chunker family and chunk size, "
        f"{results[0].config['dataset']} (bands: 95% bootstrap CI)",
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


def fig_overlap_ablation(
    stop_runs: list[RunResult], out: Path, budgets: tuple[int, int] = (400, 1600)
) -> None:
    """ΔSpanRecall vs. overlap, paired against the same config at zero overlap.

    One panel per chunker family that has an overlap knob; error bars are 95%
    paired bootstrap CIs, so a bar clear of the zero line is a significant
    effect of overlap alone (size, question set, and rule held fixed).
    """
    runs = {
        (rr.config["chunker"], rr.config["chunk_size"], rr.config["overlap"]): rr
        for rr in stop_runs
    }
    families = (
        ("fixed", "overlap (fraction of chunk size)", lambda size, o: o / size),
        ("sentence", "overlap (sentences)", lambda size, o: o),
    )
    sizes = sorted({size for c, size, o in runs if o > 0 and (c, size, 0) in runs})
    # Error bars of different sizes share x positions; a small deterministic
    # jitter keeps them legible without lying about the x value.
    jitter = {size: (i - (len(sizes) - 1) / 2) for i, size in enumerate(sizes)}
    fig, axes = plt.subplots(
        len(budgets), len(families), figsize=(8.6, 5.6), sharex="col", sharey=True
    )
    for row, budget in enumerate(budgets):
        for col, (chunker, xlabel, xval) in enumerate(families):
            ax = axes[row][col]
            overlaps = sorted({o for c, s, o in runs if c == chunker and o > 0})
            step = (max(overlaps) - min(overlaps)) / 60 if chunker == "sentence" else 0.008
            for size in sizes:
                base = runs[(chunker, size, 0)]
                # Zero overlap is the paired control: Δ = 0 by definition.
                xs, ys, lo, hi = [0.0], [0.0], [0.0], [0.0]
                for o in overlaps:
                    if (chunker, size, o) not in runs:
                        continue
                    ci = diff_ci(
                        runs[(chunker, size, o)].metric("recall", budget),
                        base.metric("recall", budget),
                    )
                    xs.append(xval(size, o) + jitter[size] * step)
                    ys.append(ci.mean_diff)
                    lo.append(ci.ci_low)
                    hi.append(ci.ci_high)
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[
                        [y - l for y, l in zip(ys, lo)],
                        [h - y for y, h in zip(ys, hi)],
                    ],
                    color=SIZE_RAMP[size],
                    linewidth=1.8,
                    marker="o",
                    markersize=4.5,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.9,
                    capsize=2.5,
                    elinewidth=1.0,
                    label=f"{size}",
                )
            ax.axhline(0, color=INK_MUTED, linewidth=0.9)
            ax.set_title(f"{chunker} chunker, B={budget}")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            if row == len(budgets) - 1:
                ax.set_xlabel(xlabel)
                if chunker == "fixed":
                    ax.set_xticks([0, 0.125, 0.25, 0.5], ["0", "12.5%", "25%", "50%"])
                else:
                    ax.set_xticks([0, *overlaps])
        axes[row][0].set_ylabel(f"ΔSpanRecall@{budget}\nvs. zero overlap (paired)")
    axes[0][0].legend(
        title="chunk size (tokens)", loc="upper left", ncols=3, title_fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "Effect of overlap, paired against zero overlap at the same chunk size "
        "(bars: 95% paired CI)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_budget_rule(
    stop_runs: list[RunResult], trunc_runs: list[RunResult], out: Path, budget: int = 200
) -> None:
    """SpanRecall@budget vs. chunk size under both budget rules.

    Same rankings on both sides; only the handling of the budget-straddling
    chunk differs. The figure shows the stop rule's retrieve-nothing collapse
    disappearing under truncation while the size ordering persists.
    """
    stop = _by_config(stop_runs)
    trunc = _by_config(trunc_runs)
    sizes = sorted({rr.config["chunk_size"] for rr in stop_runs})
    fig, axes = plt.subplots(1, len(CHUNKER_ORDER), figsize=(9.6, 3.2), sharey=True)
    for ax, chunker in zip(axes, CHUNKER_ORDER):
        color = CHUNKER_COLORS[chunker]
        for grid, style, marker, label in (
            (trunc, "-", "o", "truncate-final-chunk"),
            (stop, (0, (4, 2)), "s", "stop-before-exceed"),
        ):
            cis = [mean_ci(grid[(chunker, s)].metric("recall", budget)) for s in sizes]
            ax.fill_between(
                sizes,
                [c.ci_low for c in cis],
                [c.ci_high for c in cis],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                sizes,
                [c.mean_diff for c in cis],
                color=color,
                linestyle=style,
                linewidth=2,
                marker=marker,
                markersize=5,
                markerfacecolor=color if marker == "o" else SURFACE,
                markeredgecolor=SURFACE if marker == "o" else color,
                markeredgewidth=1.0,
                label=label,
            )
        ax.set_title(f"{chunker} chunker")
        ax.set_ylim(0, 1.0)
        _log2_axis(ax, sizes, "chunk size (tokens, log scale)")
    axes[0].set_ylabel(
        f"SpanRecall@{budget} (mean, {len(stop_runs[0].records):,} questions)"
    )
    axes[0].legend(title="budget rule", loc="lower left", title_fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle(
        f"Truncating the final chunk removes the stop rule's retrieve-nothing "
        f"collapse at B={budget} (bands: 95% CI)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_retriever_comparison(
    by_retriever: dict[str, list[RunResult]], out: Path, budget: int = 400
) -> None:
    """SpanRecall@budget vs. chunk size, one panel per chunker, one line per
    retriever.

    The question the figure answers: is "small chunks win under budget
    matching" a BM25 artifact, or a property of chunking that holds across
    retrieval models? Chunk lists are identical across retrievers at each
    grid point, so vertical gaps between lines are pure retriever effects.
    """
    grids = {name: _by_config(runs) for name, runs in by_retriever.items()}
    any_runs = next(iter(by_retriever.values()))
    sizes = sorted({rr.config["chunk_size"] for rr in any_runs})
    n_questions = len(any_runs[0].records)
    fig, axes = plt.subplots(1, len(CHUNKER_ORDER), figsize=(9.6, 3.2), sharey=True)
    for ax, chunker in zip(axes, CHUNKER_ORDER, strict=True):
        for name in RETRIEVER_STYLES:
            if name not in grids:
                continue
            color, marker = RETRIEVER_STYLES[name]
            cis = [mean_ci(grids[name][(chunker, s)].metric("recall", budget)) for s in sizes]
            ax.fill_between(
                sizes,
                [c.ci_low for c in cis],
                [c.ci_high for c in cis],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                sizes,
                [c.mean_diff for c in cis],
                color=color,
                linewidth=2,
                marker=marker,
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=name,
            )
        ax.set_title(f"{chunker} chunker")
        ax.set_ylim(0, 1.0)
        _log2_axis(ax, sizes, "chunk size (tokens, log scale)")
    axes[0].set_ylabel(f"SpanRecall@{budget} (mean, {n_questions:,} questions)")
    axes[0].legend(title="retriever", loc="lower left", title_fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle(
        f"SpanRecall@{budget} by chunk size under each retriever, "
        f"{any_runs[0].config['dataset']} (bands: 95% CI)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_dense_window(
    bm25_runs: list[RunResult],
    dense_runs: list[RunResult],
    out: Path,
    budget: int = 800,
    hit_k: int = 5,
) -> None:
    """The encoder-window mechanism behind the dense-vs-BM25 gap.

    Left: paired ΔSpanRecall (dense − BM25) at ``budget`` vs. chunk size, one
    line per chunker family, each point annotated with the fraction of chunks
    the encoder truncated. B=800 is used because it is the smallest budget at
    which every size fits at least one chunk, so no cell is a protocol
    artifact. Right: hit@5 vs. size for the fixed chunker — BM25's fixed-k
    curve rises monotonically with size while dense turns down past the
    window, the signature that separates truncation from generic retriever
    weakness.
    """
    bm25 = _by_config(bm25_runs)
    dense = _by_config(dense_runs)
    sizes = sorted({rr.config["chunk_size"] for rr in dense_runs})
    n_questions = len(dense_runs[0].records)
    window = next(
        rr.retriever_stats["max_seq_length"]
        for rr in dense_runs
        if rr.retriever_stats is not None
    )
    fig, (ax_delta, ax_hit) = plt.subplots(1, 2, figsize=(9.6, 3.4))

    ax_delta.axhline(0, color=AXIS, linewidth=1)
    for chunker in CHUNKER_ORDER:
        color = CHUNKER_COLORS[chunker]
        cis = [
            diff_ci(
                dense[(chunker, s)].metric("recall", budget),
                bm25[(chunker, s)].metric("recall", budget),
            )
            for s in sizes
        ]
        ax_delta.fill_between(
            sizes,
            [c.ci_low for c in cis],
            [c.ci_high for c in cis],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax_delta.plot(
            sizes,
            [c.mean_diff for c in cis],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=chunker,
        )
        # Non-zero truncation shares only: sizes 64/128 are all 0% and the
        # three coincident labels would just overprint each other there.
        # Offsets stagger horizontally because fixed and sentence deltas
        # nearly coincide at the truncated sizes.
        label_offset = {"fixed": (-16, -5), "sentence": (15, 5), "recursive": (0, 10)}
        for size, ci in zip(sizes, cis, strict=True):
            stats = dense[(chunker, size)].retriever_stats
            share = stats["n_chunks_truncated"] / stats["n_chunks"]
            if share == 0:
                continue
            ax_delta.annotate(
                f"{share:.0%}",
                (size, ci.mean_diff),
                textcoords="offset points",
                xytext=label_offset[chunker],
                ha="center",
                fontsize=7,
                color=color,
            )
    ax_delta.set_title(
        f"Paired Δ vs BM25 at B={budget}\n(labels: share of chunks truncated)"
    )
    ax_delta.set_ylabel(f"ΔSpanRecall@{budget} (dense − BM25)")
    ax_delta.legend(title="chunker", loc="lower left", title_fontsize=8,
                    labelcolor=INK_SECONDARY)
    _log2_axis(ax_delta, sizes, "chunk size (tokens, log scale)")

    for name in ("bm25", "dense"):
        color, marker = RETRIEVER_STYLES[name]
        grid = bm25 if name == "bm25" else dense
        ax_hit.plot(
            sizes,
            [sum(grid[("fixed", s)].hits(hit_k)) / n_questions for s in sizes],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=name,
        )
    ax_hit.set_title(f"hit@{hit_k}, fixed chunker")
    ax_hit.set_ylabel(f"hit@{hit_k} (mean, {n_questions:,} questions)")
    ax_hit.legend(title="retriever", loc="lower center", title_fontsize=8,
                  labelcolor=INK_SECONDARY)
    _log2_axis(ax_hit, sizes, "chunk size (tokens, log scale)")

    fig.suptitle(
        f"Past the {window}-wordpiece encoder window, dense retrieval is "
        "prefix retrieval: the BM25 gap widens where truncation sets in, and "
        "even fixed-k hit@5 turns down",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_gold_length_crossover(
    runs_by_line: dict[str, list[RunResult]],
    gold_stats: dict[str, tuple[int, int]],
    out: Path,
    challenger: tuple[str, int] = ("fixed", 64),
    baseline: tuple[str, int] = ("fixed", 256),
    min_budget: int = 400,
) -> None:
    """Where "small chunks win" ends: gold-span length moderates the size effect.

    Left: the paired small-vs-large-chunk delta against budget for three
    (dataset, retriever) settings. On SQuAD's ~3-token golds the advantage
    never reverses; on Chroma's sentence-scale golds it crosses zero as the
    budget grows — except under the window-limited dense encoder, which
    cannot read a large chunk past its prefix, so large chunks never get to
    use their length. Right: the Chroma/BM25 delta split by gold-length
    tercile — the crossover is driven by the questions with the longest gold
    evidence, the direct mechanism check.

    Budgets below ``min_budget`` are excluded: at B=200 the 256-token
    baseline retrieves nothing under the stop rule, so the delta there
    measures the protocol artifact (finding 5), not chunking, and its ~0.6
    magnitude would compress the crossover region the figure exists to show.
    """
    from experiments.summarize_chroma import gold_terciles

    line_styles = {
        "SQuAD dev-v1.1 / BM25": ("#2a78d6", "o"),
        "Chroma / BM25": ("#1baf7a", "s"),
        "Chroma / dense": ("#9a5bd2", "D"),
    }
    tercile_ramp = ("#86b6ef", "#2a78d6", "#184f95")
    ch_label = f"{challenger[0]}-{challenger[1]}"
    bl_label = f"{baseline[0]}-{baseline[1]}"

    def deltas(runs: list[RunResult], indices: list[int] | None = None):
        grid = _by_config(runs)
        a, b = grid[challenger], grid[baseline]
        budgets = [int(x) for x in a.config["budgets"] if int(x) >= min_budget]
        cis = []
        for budget in budgets:
            sa, sb = a.metric("recall", budget), b.metric("recall", budget)
            if indices is not None:
                sa = [sa[i] for i in indices]
                sb = [sb[i] for i in indices]
            cis.append(diff_ci(sa, sb))
        return budgets, cis

    fig, (ax_lines, ax_terc) = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)

    def draw(ax, budgets, cis, color, marker, label):
        ax.fill_between(
            budgets,
            [c.ci_low for c in cis],
            [c.ci_high for c in cis],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(
            budgets,
            [c.mean_diff for c in cis],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=label,
        )

    for label, (color, marker) in line_styles.items():
        if label not in runs_by_line:
            continue
        budgets, cis = deltas(runs_by_line[label])
        draw(ax_lines, budgets, cis, color, marker, label)
    ax_lines.set_title(
        "Same comparison, three settings:\nthe reversal needs long golds AND a full-chunk retriever"
    )
    ax_lines.set_ylabel(f"ΔSpanRecall@B ({ch_label} − {bl_label}, paired)")
    ax_lines.legend(loc="upper right", labelcolor=INK_SECONDARY)

    chroma_bm25 = runs_by_line["Chroma / BM25"]
    qids = _by_config(chroma_bm25)[challenger].qids()
    for (label, indices), color in zip(
        gold_terciles(qids, gold_stats), tercile_ramp, strict=True
    ):
        budgets, cis = deltas(chroma_bm25, indices)
        draw(ax_terc, budgets, cis, color, "o", f"{label} (n={len(indices)})")
    ax_terc.set_title(
        "Chroma / BM25 by gold-evidence length:\nlong-gold questions drive the crossover"
    )
    ax_terc.legend(title="gold-length tercile", loc="upper right", title_fontsize=8,
                   labelcolor=INK_SECONDARY)

    for ax in (ax_lines, ax_terc):
        ax.axhline(0, color=INK_MUTED, linewidth=0.9)
        _log2_axis(ax, budgets, "token budget B (log scale)")
    fig.suptitle(
        "The small-chunk advantage is a property of short gold spans: with "
        "sentence-scale evidence it shrinks, then reverses (bands: 95% paired CI)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tokenizer_robustness(
    regex_runs: list[RunResult], bpe_runs: list[RunResult], out: Path
) -> None:
    """Budget curves under the BPE unit + headline deltas under both units.

    Left: the fixed-family SpanRecall curves with everything (chunk sizes,
    budgets, metric token sets) counted in cl100k_base BPE tokens — the
    ordering to compare against the regex-unit curves figure. Right: the
    headline paired deltas at B=400 computed within each unit; unit-robust
    claims keep their sign and their CI clear of zero in both bars.
    """
    grids = {"regex": _by_config(regex_runs), "cl100k": _by_config(bpe_runs)}
    budgets = [int(b) for b in regex_runs[0].config["budgets"]]
    sizes = sorted({rr.config["chunk_size"] for rr in regex_runs})
    n_questions = len(regex_runs[0].records)
    fig, (ax_curves, ax_delta) = plt.subplots(
        1, 2, figsize=(9.0, 3.4), width_ratios=(1.0, 1.15)
    )

    for size in sizes:
        rr = grids["cl100k"][("fixed", size)]
        cis = [mean_ci(rr.metric("recall", b)) for b in budgets]
        ax_curves.fill_between(
            budgets,
            [c.ci_low for c in cis],
            [c.ci_high for c in cis],
            color=SIZE_RAMP[size],
            alpha=0.22,
            linewidth=0,
        )
        ax_curves.plot(
            budgets,
            [c.mean_diff for c in cis],
            color=SIZE_RAMP[size],
            linewidth=2,
            marker="o",
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=f"{size}",
        )
    ax_curves.set_title("fixed chunker, all counts in cl100k_base BPE tokens")
    ax_curves.set_ylim(0, 1.0)
    _log2_axis(ax_curves, budgets, "token budget B (BPE tokens, log scale)")
    ax_curves.set_ylabel(f"SpanRecall@B (mean, {n_questions:,} questions)")
    ax_curves.legend(title="chunk size (BPE tokens)", loc="lower right", ncols=2, title_fontsize=8)

    pairs = (
        (("fixed", 64), ("fixed", 256)),
        (("fixed", 64), ("fixed", 512)),
        (("sentence", 64), ("fixed", 64)),
        (("sentence", 128), ("fixed", 128)),
    )
    unit_styles = {
        "regex": ("#2a78d6", "regex word unit"),
        "cl100k": ("#1baf7a", "cl100k BPE unit"),
    }
    budget = 400
    width = 0.38
    offsets = (-width / 2, width / 2)
    for offset, (unit, (color, label)) in zip(offsets, unit_styles.items(), strict=True):
        cis = [
            diff_ci(
                grids[unit][challenger].metric("recall", budget),
                grids[unit][baseline].metric("recall", budget),
            )
            for challenger, baseline in pairs
        ]
        xs = [i + offset for i in range(len(pairs))]
        ax_delta.bar(
            xs,
            [c.mean_diff for c in cis],
            width=width,
            color=color,
            label=label,
            zorder=3,
        )
        ax_delta.errorbar(
            xs,
            [c.mean_diff for c in cis],
            yerr=[
                [c.mean_diff - c.ci_low for c in cis],
                [c.ci_high - c.mean_diff for c in cis],
            ],
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
    ax_delta.axhline(0, color=AXIS, linewidth=0.8)
    ax_delta.set_xticks(range(len(pairs)))
    ax_delta.set_xticklabels(
        [f"{c[0]}-{c[1]}\n− {b[0]}-{b[1]}" for c, b in pairs], fontsize=7.5
    )
    ax_delta.set_ylabel(f"ΔSpanRecall@{budget} (paired)")
    ax_delta.set_title(f"headline paired deltas at B={budget}, by token unit")
    ax_delta.grid(axis="y")
    ax_delta.set_axisbelow(True)
    ax_delta.legend(loc="upper right")

    fig.suptitle(
        f"Token-unit robustness check, {regex_runs[0].config['dataset']} / "
        f"{regex_runs[0].config['retriever']} "
        "(bands and error bars: 95% bootstrap CIs)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_semantic_comparison(results: list[RunResult], out: Path) -> None:
    """What embedding breakpoints buy over the sentence boundaries they refine.

    Left: paired ΔSpanRecall (semantic − sentence) at matched nominal size,
    grouped by size with one bar per budget (single-hue ordinal ramp — budget
    is a magnitude, not an identity). Both sides pack identical sentences
    under identical budgets; the semantic side only adds "never pack across a
    breakpoint", so these deltas isolate the breakpoints themselves. Right:
    realized mean chunk size against nominal — breakpoints can only shorten
    chunks, and this panel shows how far below its nominal budget (grey
    identity line) each family actually operates, the confound to keep in
    view when reading the left panel.
    """
    grid = _by_config(results)
    budgets = [int(b) for b in results[0].config["budgets"]]
    sizes = sorted(
        {
            size
            for family, size in grid
            if family == "semantic" and ("sentence", size) in grid
        }
    )
    n_questions = len(results[0].records)
    ramp = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95")
    budget_ramp = {b: ramp[min(i, len(ramp) - 1)] for i, b in enumerate(budgets)}
    fig, (ax_delta, ax_size) = plt.subplots(
        1, 2, figsize=(9.0, 3.4), width_ratios=(1.35, 1.0)
    )

    width = 0.8 / len(budgets)
    for bi, budget in enumerate(budgets):
        cis = [
            diff_ci(
                grid[("semantic", size)].metric("recall", budget),
                grid[("sentence", size)].metric("recall", budget),
            )
            for size in sizes
        ]
        xs = [i + (bi - (len(budgets) - 1) / 2) * width for i in range(len(sizes))]
        ax_delta.bar(
            xs,
            [c.mean_diff for c in cis],
            width=width * 0.92,
            color=budget_ramp[budget],
            label=f"B={budget}",
            zorder=3,
        )
        ax_delta.errorbar(
            xs,
            [c.mean_diff for c in cis],
            yerr=[
                [c.mean_diff - c.ci_low for c in cis],
                [c.ci_high - c.mean_diff for c in cis],
            ],
            fmt="none",
            ecolor=INK,
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
    ax_delta.axhline(0, color=AXIS, linewidth=0.8)
    ax_delta.set_xticks(range(len(sizes)))
    ax_delta.set_xticklabels([str(s) for s in sizes])
    ax_delta.set_xlabel("nominal chunk size (tokens)")
    ax_delta.set_ylabel("ΔSpanRecall, semantic − sentence (paired)")
    ax_delta.set_title("Do embedding breakpoints beat\nregex sentence boundaries?")
    ax_delta.grid(axis="y")
    ax_delta.set_axisbelow(True)
    ax_delta.legend(title="budget", loc="best", title_fontsize=8, fontsize=7.5)

    family_styles = {
        "sentence": (CHUNKER_COLORS["sentence"], "o"),
        "semantic": ("#9a5bd2", "D"),
    }
    all_sizes = sorted({size for _, size in grid})
    ax_size.plot(
        all_sizes,
        all_sizes,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle="--",
        label="nominal (identity)",
        zorder=2,
    )
    for family, (color, marker) in family_styles.items():
        family_sizes = [s for s in all_sizes if (family, s) in grid]
        ax_size.plot(
            family_sizes,
            [grid[(family, s)].chunk_stats["tokens_mean"] for s in family_sizes],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=family,
            zorder=3,
        )
    _log2_axis(ax_size, all_sizes, "nominal chunk size (tokens, log scale)")
    ax_size.set_yscale("log", base=2)
    ax_size.set_yticks(all_sizes)
    ax_size.yaxis.set_major_formatter(ScalarFormatter())
    ax_size.set_ylabel("realized mean tokens/chunk")
    ax_size.set_title("Breakpoints only shorten:\nrealized vs. nominal size")
    ax_size.legend(loc="upper left", fontsize=7.5)

    fig.suptitle(
        f"Semantic (embedding-breakpoint) chunker vs. sentence packing, "
        f"{results[0].config['dataset']} / {results[0].config['retriever']} "
        f"({n_questions:,} questions; error bars: 95% bootstrap CIs)",
        fontsize=10.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render README figures from raw results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline = load_raw(
        args.raw_dir,
        dataset=args.dataset,
        retriever=args.retriever,
        budget_rule="stop",
        overlap=0,
        seed=args.seed,
    )
    if not baseline:
        raise SystemExit(f"no results for {args.dataset}/{args.retriever} in {args.raw_dir}")
    check_aligned(baseline)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    curves = args.out_dir / f"recall_budget_curves_{args.dataset}_{args.retriever}.png"
    reversal = args.out_dir / f"metric_reversal_{args.dataset}_{args.retriever}.png"
    fig_budget_curves(baseline, curves)
    fig_metric_reversal(baseline, reversal)
    written += [curves, reversal]

    stop_all = load_raw(
        args.raw_dir,
        dataset=args.dataset,
        retriever=args.retriever,
        budget_rule="stop",
        seed=args.seed,
    )
    if any(rr.config["overlap"] > 0 for rr in stop_all):
        check_aligned(stop_all)
        overlap = args.out_dir / f"overlap_ablation_{args.dataset}_{args.retriever}.png"
        fig_overlap_ablation(stop_all, overlap)
        written.append(overlap)

    trunc = load_raw(
        args.raw_dir,
        dataset=args.dataset,
        retriever=args.retriever,
        budget_rule="truncate",
        overlap=0,
        seed=args.seed,
    )
    if trunc:
        check_aligned(baseline + trunc)
        rule = args.out_dir / f"budget_rule_{args.dataset}_{args.retriever}.png"
        fig_budget_rule(baseline, trunc, rule)
        written.append(rule)

    bpe = load_raw(
        args.raw_dir,
        dataset=args.dataset,
        retriever=args.retriever,
        budget_rule="stop",
        overlap=0,
        seed=args.seed,
        tokenizer="cl100k",
    )
    if bpe:
        # Units differ, questions must not: the two grids score the same
        # sample, so alignment across them is a real assertion, not a no-op.
        check_aligned(baseline + bpe)
        tok = args.out_dir / f"tokenizer_robustness_{args.dataset}_{args.retriever}.png"
        fig_tokenizer_robustness(baseline, bpe, tok)
        written.append(tok)

    if any(rr.config["chunker"] == "semantic" for rr in baseline):
        semantic = args.out_dir / f"semantic_comparison_{args.dataset}_{args.retriever}.png"
        fig_semantic_comparison(baseline, semantic)
        written.append(semantic)

    by_retriever = {}
    for name in RETRIEVER_STYLES:
        runs = load_raw(
            args.raw_dir, dataset=args.dataset, retriever=name,
            budget_rule="stop", overlap=0, seed=args.seed,
        )
        if runs:
            by_retriever[name] = runs
    if len(by_retriever) >= 2:
        check_aligned([rr for runs in by_retriever.values() for rr in runs])
        comparison = args.out_dir / f"retriever_comparison_{args.dataset}.png"
        fig_retriever_comparison(by_retriever, comparison)
        written.append(comparison)

    if "bm25" in by_retriever and "dense" in by_retriever and all(
        rr.retriever_stats is not None for rr in by_retriever["dense"]
    ):
        window = args.out_dir / f"dense_window_{args.dataset}.png"
        fig_dense_window(by_retriever["bm25"], by_retriever["dense"], window)
        written.append(window)

    # The gold-length crossover spans datasets (SQuAD vs Chroma) and needs
    # gold lengths recomputed from the corpus text, so it renders only when
    # all of its inputs exist — independently of --dataset.
    crossover_inputs = {
        "SQuAD dev-v1.1 / BM25": ("dev-v1.1", "bm25"),
        "Chroma / BM25": ("chroma", "bm25"),
        "Chroma / dense": ("chroma", "dense"),
    }
    runs_by_line = {}
    for label, (dataset, retriever) in crossover_inputs.items():
        runs = load_raw(
            args.raw_dir, dataset=dataset, retriever=retriever,
            budget_rule="stop", overlap=0, seed=args.seed,
        )
        if runs:
            check_aligned(runs)
            runs_by_line[label] = runs
    if (
        set(runs_by_line) == set(crossover_inputs)
        and (args.data_dir / "chroma" / "questions_df.csv").exists()
    ):
        from experiments.summarize_chroma import gold_stats

        crossover = args.out_dir / "gold_length_crossover.png"
        fig_gold_length_crossover(runs_by_line, gold_stats(args.data_dir), crossover)
        written.append(crossover)

    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

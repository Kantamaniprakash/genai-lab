"""Rubric-sensitivity view: the same judge under two rubrics, paired per item.

For every model that has complete swap pairs under both rubrics, reads the two
stores, inner-joins them on item_id, and computes the paired view from
:mod:`src.rubric_pair`: per-rubric stats on the matched items, paired deltas
in accuracy / preference / bias / compliance with bootstrap CIs, cross-rubric
consistency (rubric flip rate, correlations of s and b), and the per-category
breakdown. Writes one combined JSON keyed by model plus a markdown table.

Usage:
    python -m experiments.rubric_view                       # all matched models
    python -m experiments.rubric_view --model qwen2.5-0.5b  # one model
    python -m experiments.rubric_view --rubric-a minimal --rubric-b detailed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import assemble_pairs  # noqa: E402
from src.data import SUBSET_TO_CATEGORY  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402
from src.rubric_pair import match_rubric_pairs, rubric_pair_view  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"


def category_of(item_id: str) -> str:
    return SUBSET_TO_CATEGORY[item_id.split("/", 1)[0]]


def load_pairs(model_key: str, rubric: str) -> list:
    path = RESULTS_DIR / f"{model_key}__{rubric}.jsonl"
    if not path.exists():
        return []
    pairs, incomplete = assemble_pairs(load_records([path]))
    if incomplete:
        print(f"[rubric_view] note: {path.stem} has {incomplete} incomplete items "
              f"(only complete swap pairs enter the join)", flush=True)
    return pairs


def format_table(views: dict[str, dict], rubric_a: str, rubric_b: str) -> str:
    header = (
        f"| model | n | sym {rubric_a} | sym {rubric_b} | Δ sym [95% CI] "
        f"| Δ mean b | Δ |b| [95% CI] | rubric flip | r(s) |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for model_key in sorted(views, key=lambda k: (MODELS[k].params_b, k)):
        v = views[model_key]
        d = v["deltas"]
        cons = v["consistency"]
        rows.append(
            f"| {model_key} | {v['n_matched_items']} "
            f"| {v['per_rubric'][rubric_a]['sym_acc']['mean']:.3f} "
            f"| {v['per_rubric'][rubric_b]['sym_acc']['mean']:.3f} "
            f"| {d['sym_acc']['mean']:+.3f} "
            f"[{d['sym_acc']['ci95'][0]:+.3f}, {d['sym_acc']['ci95'][1]:+.3f}] "
            f"| {d['b']['mean']:+.2f} "
            f"| {d['abs_b']['mean']:+.2f} "
            f"[{d['abs_b']['ci95'][0]:+.2f}, {d['abs_b']['ci95'][1]:+.2f}] "
            f"| {cons['rubric_flip_rate']['mean']:.3f} "
            f"| {cons['s_pearson']['r']:.3f} |"
        )
    caption = (
        f"\nPaired per item over the matched sample; deltas read "
        f"{rubric_b} - {rubric_a}. 'rubric flip' is the fraction of items "
        f"whose symmetrized verdict changes with the rubric text alone; "
        f"r(s) is the cross-rubric Pearson correlation of the "
        f"order-invariant preference.\n"
    )
    return header + "\n".join(rows) + "\n" + caption


def rubric_figure(
    matched_by_model: dict[str, list], rubric_a: str, rubric_b: str, out: Path
) -> None:
    """One row per model: s vs s and b vs b across rubrics, item-paired.

    Points off the identity line are what the rubric changed; points in the
    off-diagonal quadrants of the left panel are rubric flips — the
    symmetrized verdict changed with the prompt text alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.make_figures import (
        AXIS,
        CATEGORY_STYLES,
        INK,
        INK_MUTED,
        INK_SECONDARY,
    )

    models = sorted(matched_by_model, key=lambda k: (MODELS[k].params_b, k))
    fig, axes = plt.subplots(
        len(models), 2, figsize=(9.2, 4.3 * len(models)), squeeze=False
    )
    for row, model_key in enumerate(models):
        matched = matched_by_model[model_key]
        for col, component in enumerate(("s", "b")):
            ax = axes[row][col]
            xs_all = [getattr(p.a, component) for p in matched]
            ys_all = [getattr(p.b, component) for p in matched]
            lim = 1.06 * max(
                (max(map(abs, xs_all + ys_all))), 1e-6
            )
            ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
            ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
            ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, linewidth=0.9,
                    linestyle="--", zorder=1)
            for cat, (color, marker) in CATEGORY_STYLES.items():
                pts = [p for p in matched if category_of(p.item_id) == cat]
                if not pts:
                    continue
                ax.scatter(
                    [getattr(p.a, component) for p in pts],
                    [getattr(p.b, component) for p in pts],
                    s=13, alpha=0.55, linewidths=0, color=color, marker=marker,
                    label=f"{cat} (n={len(pts)})" if col == 0 else None, zorder=2,
                )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.6)
            name = ("preference s" if component == "s" else "position bias b")
            ax.set_xlabel(f"{name}, rubric `{rubric_a}` (log-odds)")
            ax.set_ylabel(f"{name}, rubric `{rubric_b}` (log-odds)")
            if component == "s":
                flips = sum(1.0 - p.sym_verdict_agrees for p in matched)
                ax.set_title(
                    f"{model_key}: {name} across rubrics — "
                    f"rubric flips {flips / len(matched):.1%}",
                    color=INK,
                )
                ax.legend(loc="lower right", markerscale=1.3)
            else:
                ax.set_title(f"{model_key}: {name} across rubrics", color=INK)
            ax.annotate(
                "identity = rubric-invariant", xy=(0.03, 0.97),
                xycoords="axes fraction", fontsize=8, color=INK_SECONDARY,
                va="top",
            )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, choices=sorted(MODELS))
    parser.add_argument("--rubric-a", default="minimal")
    parser.add_argument("--rubric-b", default="detailed")
    args = parser.parse_args()

    model_keys = [args.model] if args.model else sorted(MODELS)
    views: dict[str, dict] = {}
    matched_by_model: dict[str, list] = {}
    for model_key in model_keys:
        pairs_a = load_pairs(model_key, args.rubric_a)
        pairs_b = load_pairs(model_key, args.rubric_b)
        if not pairs_a or not pairs_b:
            if args.model:
                raise SystemExit(
                    f"{model_key}: need complete pairs under both rubrics "
                    f"({args.rubric_a}: {len(pairs_a)}, {args.rubric_b}: {len(pairs_b)})"
                )
            continue
        view = rubric_pair_view(
            pairs_a, pairs_b, args.rubric_a, args.rubric_b, category_of=category_of
        )
        if view["n_only_a"] or view["n_only_b"]:
            print(f"[rubric_view] {model_key}: {view['n_only_a']} items only in "
                  f"{args.rubric_a}, {view['n_only_b']} only in {args.rubric_b}; "
                  f"matched {view['n_matched_items']}", flush=True)
        views[model_key] = view
        matched_by_model[model_key] = match_rubric_pairs(pairs_a, pairs_b)[0]

    if not views:
        raise SystemExit("no model has complete swap pairs under both rubrics")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"rubric_pair__{args.rubric_a}_vs_{args.rubric_b}"
    out_json = SUMMARY_DIR / f"{stem}.json"
    # Single-model runs update that model's entry in place, preserving others.
    existing = json.loads(out_json.read_text()) if out_json.exists() else {}
    existing.update(views)
    views = dict(sorted(existing.items()))
    out_json.write_text(json.dumps(views, indent=2, sort_keys=True) + "\n")

    table = format_table(views, args.rubric_a, args.rubric_b)
    (SUMMARY_DIR / f"{stem}.md").write_text(table)
    print(table)
    print(f"[rubric_view] wrote {out_json}")

    # The committed figure spans every matched model, so it is only rendered
    # by the sweep (no --model): a single-model invocation must not overwrite
    # the canonical multi-model panel with a one-row version.
    if matched_by_model and args.model is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"{stem}.png"
        rubric_figure(matched_by_model, args.rubric_a, args.rubric_b, fig_path)
        print(f"[rubric_view] wrote {fig_path}")


if __name__ == "__main__":
    main()

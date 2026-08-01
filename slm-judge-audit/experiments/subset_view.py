"""Per-subset heterogeneity across judges: sym accuracy with CIs at the
resolution the category tables average away.

Reads every completed ``{model}__{rubric}.jsonl`` store (identical item-set
guard as the other cross-model scripts), computes the per-subset view from raw
records — symmetrized accuracy with 95% bootstrap CIs, decomposition medians,
and each subset's longer-response floor — writes one JSON per rubric to
results/summary/subset_view__{rubric}.json, and renders a forest figure:
subsets on the y axis grouped by category, one marker per judge, the subset's
length floor as a grey tick.

Usage:
    python -m experiments.subset_view [--rubric minimal] [--models ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import assemble_pairs, subset_view  # noqa: E402
from src.baselines import longer_response_correct  # noqa: E402
from src.data import SUBSET_TO_CATEGORY, fetch, load_rewardbench  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402

from experiments.length_probe import MODEL_STYLES  # noqa: E402
from experiments.make_figures import INK, INK_MUTED, INK_SECONDARY  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

CATEGORY_ORDER = ("Chat", "Chat Hard", "Safety", "Reasoning")


def subset_of(item_id: str) -> str:
    return item_id.split("/", 1)[0]


def category_of(item_id: str) -> str:
    return SUBSET_TO_CATEGORY[subset_of(item_id)]


def forest_figure(results: dict[str, dict], rubric: str, out: Path) -> None:
    models = [k for k in MODEL_STYLES if k in results]
    # Subsets ordered by category, then by the cross-model mean sym accuracy
    # so each category block reads easiest-to-hardest at a glance.
    any_view = results[models[0]]["subsets"]
    def cross_model_mean(name: str) -> float:
        return sum(results[m]["subsets"][name]["sym_acc"]["mean"] for m in models) / len(models)
    names = sorted(
        any_view,
        key=lambda s: (CATEGORY_ORDER.index(any_view[s]["category"]),
                       -cross_model_mean(s)),
    )

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(names) + 1.6))
    group_gap = 1.0
    ys, last_cat, offset = [], None, 0.0
    for name in names:
        cat = any_view[name]["category"]
        if last_cat is not None and cat != last_cat:
            offset -= group_gap
        ys.append(offset)
        offset -= 1.0
        last_cat = cat

    jitter = 0.30
    for j, model in enumerate(models):
        color, marker = MODEL_STYLES[model]
        dy = (j - (len(models) - 1) / 2) / max(len(models) - 1, 1) * jitter * 2
        xs, yy, lo_err, hi_err = [], [], [], []
        for y, name in zip(ys, names):
            block = results[model]["subsets"][name]
            mean = block["sym_acc"]["mean"]
            lo, hi = block["sym_acc"]["ci95"]
            xs.append(mean)
            yy.append(y + dy)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        ax.errorbar(xs, yy, xerr=[lo_err, hi_err], fmt=marker, color=color,
                    markersize=3.6, elinewidth=0.7, capsize=0, linestyle="none",
                    alpha=0.9, label=model, zorder=3)

    for y, name in zip(ys, names):
        floor = any_view[name]["longer_floor"]
        ax.plot([floor], [y], marker="|", color=INK_SECONDARY, markersize=11,
                markeredgewidth=1.4, linestyle="none", zorder=2)

    ax.axvline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    labels = [f"{n}  (n={any_view[n]['n_items']})" for n in names]
    ax.set_yticks(ys, labels, fontsize=7)
    # Category separators and labels on the right margin.
    cat_bounds: dict[str, list[float]] = {}
    for y, name in zip(ys, names):
        cat_bounds.setdefault(any_view[name]["category"], []).append(y)
    for cat, bounds in cat_bounds.items():
        ax.annotate(cat, xy=(1.005, (max(bounds) + min(bounds)) / 2),
                    xycoords=("axes fraction", "data"), fontsize=8,
                    color=INK_SECONDARY, rotation=270, va="center", ha="left")
        if min(bounds) != min(ys):
            ax.axhline(min(bounds) - (1.0 + group_gap) / 2, color=INK_MUTED,
                       linewidth=0.6, alpha=0.5, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_xlabel("symmetrized accuracy (95% bootstrap CI); | = subset longer-response floor")
    ax.set_title(f"per-subset judge accuracy, rubric={rubric}", color=INK, pad=30)
    ax.grid(True, axis="x", alpha=0.6)
    # Legend above the axes: inside, any corner covers some subset's marker
    # or floor tick (the lower-left spot hid math-prm's 0.078 floor).
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.002), ncols=3,
              fontsize=7, frameon=False, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model keys to include (default: every completed "
                             "store; use this to exclude an in-flight run)")
    args = parser.parse_args()

    fetch()
    items_by_id = {item.item_id: item for item in load_rewardbench()}

    results: dict[str, dict] = {}
    item_sets = set()
    for key in sorted(MODELS):
        if args.models is not None and key not in args.models:
            continue
        path = RESULTS_DIR / f"{key}__{args.rubric}.jsonl"
        if not path.exists():
            continue
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[subset] {path.stem}: {incomplete} incomplete items — "
                  f"skipping store (partial subsets would bias the view)")
            continue
        item_sets.add(tuple(sorted(p.item_id for p in pairs)))
        results[key] = subset_view(
            pairs,
            subset_of=subset_of,
            category_of=category_of,
            longer_correct_of=lambda i: longer_response_correct(items_by_id[i]),
        )
        print(f"[subset] {key}: {results[key]['n_subsets']} subsets")

    if len(results) < 2:
        raise SystemExit(f"need >=2 complete stores for rubric {args.rubric}, "
                         f"found {len(results)}")
    if len(item_sets) > 1:
        raise SystemExit("stores cover different item sets; refusing to compare")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUMMARY_DIR / f"subset_view__{args.rubric}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[subset] wrote {out_json}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURES_DIR / f"subset_view__{args.rubric}.png"
    forest_figure(results, args.rubric, out_fig)
    print(f"[subset] wrote {out_fig}")


if __name__ == "__main__":
    main()

"""What a partial grid's item prefix does to every judge's numbers (finding 26).

`stratified_sample` sorts by ``item_id`` and `run_grid` walks that order, so a
grid caught mid-run has finished an alphabetical prefix of the subsets rather
than a random subsample. This renders the consequence directly: each judge's
symmetrized accuracy on the full sample beside the same judge's accuracy on
the items the in-flight grid has finished, with the pick-the-longer-response
floor drawn on both sides.

The figure's job is to make three things obvious at a glance — the restriction
moves judges by different amounts (so it reorders the field rather than
shifting it), the in-flight judge only *has* a right-hand point, and the
trivial floor swings further than any judge, from far below chance to near
ceiling. Read together, that is why a mid-run peek compared against another
judge's overall number is a comparison between two different benchmarks.

Reads the two summaries `experiments.master_table` writes (full and
``--restrict-to``); runs no judgments of its own. Writes
results/figures/prefix_skew__{rubric}.png.

Usage:
    python -m experiments.prefix_skew --interim-for qwen2.5-7b [--rubric minimal]
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

from src.judge import RESULTS_DIR  # noqa: E402

from experiments.length_probe import MODEL_STYLES  # noqa: E402
from experiments.make_figures import INK, INK_MUTED, INK_SECONDARY  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

FLOOR_COLOR = "#8a6d3b"


def declutter(labels: list[tuple[float, str, str]], gap: float
              ) -> list[tuple[float, float, str, str]]:
    """Nudge overlapping labels apart, returning (anchor_y, label_y, text, color).

    Judges genuinely land on the same accuracy — both 3B judges hit 0.911 on
    the restricted set — so drawing every label at its own y silently stacks
    them into an unreadable smear. Labels are placed in value order and each
    is pushed up to ``gap`` above the previous one; the anchor stays at the
    true value so the text is offset from the point, never the point from the
    data.
    """
    placed: list[tuple[float, float, str, str]] = []
    last = float("-inf")
    for y, text, color in sorted(labels):
        label_y = max(y, last + gap)
        placed.append((y, label_y, text, color))
        last = label_y
    return placed


def slope_figure(full: dict, interim: dict, interim_for: str, out: Path) -> None:
    n_full, n_part = full["n_items"], interim["n_items"]
    models = [k for k in MODEL_STYLES if k in interim["judges"]]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    x_full, x_part = 0.0, 1.0

    left_labels: list[tuple[float, str, str]] = []
    right_labels: list[tuple[float, str, str]] = []

    for key in models:
        color, marker = MODEL_STYLES[key]
        right = interim["judges"][key]["sym_acc"]["mean"]
        left_block = full["judges"].get(key)
        if left_block is None:
            # The in-flight judge has no full-sample number yet — drawn as a
            # right-hand point only, so the gap in the figure is the honest
            # statement that its row does not exist at full coverage.
            ax.plot([x_part], [right], marker=marker, color=color, markersize=7,
                    linestyle="none", zorder=4)
            right_labels.append((right, f"{key} (in flight)", color))
            continue
        left = left_block["sym_acc"]["mean"]
        ax.plot([x_full, x_part], [left, right], color=color, linewidth=1.4,
                marker=marker, markersize=6, alpha=0.9, zorder=3)
        left_labels.append((left, key, color))
        right_labels.append((right, f"{key}  {right - left:+.3f}", color))

    floor_full = full["floors"]["longer_chars"]
    floor_part = interim["floors"]["longer_chars"]
    ax.plot([x_full, x_part], [floor_full, floor_part],
            color=FLOOR_COLOR, linewidth=1.6, linestyle="--", marker="|",
            markersize=10, zorder=2)
    left_labels.append((floor_full, "longer-response floor", FLOOR_COLOR))
    right_labels.append((floor_part,
                         f"longer-response floor  {floor_part - floor_full:+.3f}",
                         FLOOR_COLOR))

    # Judges collide on both axes (both 3B judges land on 0.911 restricted),
    # so labels are placed with a minimum vertical separation and connected
    # back to their true value by a hairline where they had to move.
    gap = 0.042
    for anchor, label_y, text, color in declutter(left_labels, gap):
        ax.annotate(text, xy=(x_full, label_y), xytext=(-8, 0),
                    textcoords="offset points", fontsize=8, color=color,
                    va="center", ha="right")
        if abs(label_y - anchor) > 1e-9:
            ax.plot([x_full - 0.06, x_full], [label_y, anchor], color=color,
                    linewidth=0.6, alpha=0.55, zorder=2)
    for anchor, label_y, text, color in declutter(right_labels, gap):
        ax.annotate(text, xy=(x_part, label_y), xytext=(8, 0),
                    textcoords="offset points", fontsize=8, color=color,
                    va="center", ha="left")
        if abs(label_y - anchor) > 1e-9:
            ax.plot([x_part, x_part + 0.06], [anchor, label_y], color=color,
                    linewidth=0.6, alpha=0.55, zorder=2)

    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax.annotate("chance", xy=(0.5, 0.5), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=7,
                color=INK_MUTED, ha="center")

    ax.set_xticks([x_full, x_part],
                  [f"full sample\n(n={n_full})",
                   f"items {interim_for} has finished\n(n={n_part})"])
    ax.set_xlim(-0.72, 1.78)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("symmetrized accuracy")
    ax.set_title("A partial grid is an item prefix, not a subsample\n"
                 "(finding 26; numbers are the shift each judge takes)",
                 color=INK, pad=10)
    ax.grid(True, axis="y", alpha=0.6)
    ax.annotate(interim.get("composition_note", ""),
                xy=(0.5, -0.16), xycoords="axes fraction", fontsize=7,
                color=INK_SECONDARY, ha="center", va="top", wrap=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--interim-for", required=True, metavar="MODEL",
                        help="the in-flight model whose interim summary to read")
    args = parser.parse_args()

    full_path = SUMMARY_DIR / f"master_table__{args.rubric}.json"
    interim_path = (SUMMARY_DIR /
                    f"master_table__{args.rubric}__interim_{args.interim_for}.json")
    for path in (full_path, interim_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run experiments.master_table first")

    with open(full_path) as f:
        full = json.load(f)
    with open(interim_path) as f:
        interim = json.load(f)
    if interim["interim_for"] != args.interim_for:
        raise SystemExit(f"{interim_path} is an interim view of "
                         f"{interim['interim_for']!r}, not {args.interim_for!r}")
    if interim["n_items"] >= full["n_items"]:
        raise SystemExit("the interim set is not smaller than the full sample; "
                         "the grid has closed and this figure has nothing to show")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"prefix_skew__{args.rubric}.png"
    slope_figure(full, interim, args.interim_for, out)
    print(f"[prefix] wrote {out} "
          f"({full['n_items']} -> {interim['n_items']} items)")


if __name__ == "__main__":
    main()

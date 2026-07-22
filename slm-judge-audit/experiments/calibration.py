"""Calibration of verdict probabilities across completed stores.

For every selected ``{model}__{rubric}`` store: fold verdict log-odds into
(confidence, correctness) points — symmetrized (one per item, sigmoid(|s|))
and raw (one per judgment, sigmoid(|z|)) — and compute tie-safe equal-mass
reliability curves, ECE, and the signed confidence−accuracy gap with
item-level bootstrap CIs (src.calibration). Writes one JSON plus a figure
with one reliability panel per model, point area proportional to bin mass.

Usage:
    python -m experiments.calibration [--rubric minimal] [--models k1 k2 ...]
        [--n-boot 10000]
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

from src.analysis import assemble_pairs  # noqa: E402
from src.calibration import calibration_view  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402

from experiments.make_figures import AXIS, INK, INK_MUTED  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

SYM_COLOR = "#184f95"
RAW_COLOR = "#c2483f"


def calibration_figure(views: dict[str, dict], rubric: str, out: Path) -> None:
    n_models = len(views)
    fig, axes = plt.subplots(1, n_models, figsize=(3.3 * n_models, 3.7),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (model, view) in zip(axes[0], views.items()):
        ax.plot([0.5, 1.0], [0.5, 1.0], color=AXIS,
                linewidth=0.9, linestyle="--", zorder=1)
        for name, color, marker in (("raw", RAW_COLOR, "s"), ("sym", SYM_COLOR, "o")):
            block = view[name]
            total = block["n_points"]
            xs = [b["conf"] for b in block["curve"]]
            ys = [b["acc"] for b in block["curve"]]
            sizes = [900 * b["n"] / total for b in block["curve"]]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.8, zorder=2)
            ax.scatter(xs, ys, s=sizes, color=color, marker=marker, alpha=0.55,
                       linewidths=0, zorder=3,
                       label=f"{name}  ECE {block['ece']['mean']:.3f} "
                             f"[{block['ece']['ci95'][0]:.3f}, {block['ece']['ci95'][1]:.3f}]")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("mean confidence in bin")
        ax.set_xlim(0.48, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.6)
        ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.4)
    axes[0][0].set_ylabel("accuracy in bin")
    fig.suptitle(
        f"reliability diagrams, tie-safe equal-mass bins (rubric={rubric}; "
        "point area = bin mass; dashed = perfect calibration)",
        fontsize=10, color=INK,
    )
    fig.text(0.5, 0.01,
             "sym: swap-averaged verdict, confidence sigmoid(|s|), one point per item — "
             "raw: single-order verdict, confidence sigmoid(|z|), one point per judgment",
             ha="center", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model keys to include (default: every completed "
                             "store; use this to exclude an in-flight run)")
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    views: dict[str, dict] = {}
    for key, model in sorted(MODELS.items(), key=lambda kv: kv[1].params_b):
        path = RESULTS_DIR / f"{key}__{args.rubric}.jsonl"
        if not path.exists() or (args.models is not None and key not in args.models):
            continue
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[calibration] {path.stem}: {incomplete} incomplete items skipped")
        views[key] = calibration_view(pairs, n_boot=args.n_boot)

    if not views:
        raise SystemExit(f"no completed stores for rubric {args.rubric}")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUMMARY_DIR / f"calibration__{args.rubric}.json"
    with open(out_json, "w") as f:
        json.dump({"rubric": args.rubric, "models": views}, f, indent=2, sort_keys=True)
        f.write("\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURES_DIR / f"calibration__{args.rubric}.png"
    calibration_figure(views, args.rubric, out_fig)

    for key, view in views.items():
        for name in ("raw", "sym"):
            b = view[name]
            print(f"[calibration] {key:14s} {name:3s}  acc {b['accuracy']:.3f}  "
                  f"mean conf {b['mean_confidence']:.3f}  "
                  f"ECE {b['ece']['mean']:.3f} [{b['ece']['ci95'][0]:.3f}, "
                  f"{b['ece']['ci95'][1]:.3f}]  gap {b['confidence_minus_accuracy']['mean']:+.3f}")
    print(f"[calibration] wrote {out_json} and {out_fig}")


if __name__ == "__main__":
    main()

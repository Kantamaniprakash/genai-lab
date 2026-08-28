"""Rubric fragility vs preference magnitude: test the perturbation account.

Finding 36 located rubric flips where |s| is small; the falsifiable version
of that claim is a model. For every judge with complete stores under both
rubrics, fit ``s_B = lam * s_A + eps`` (:func:`src.rubric_pair.fragility_fit`)
and compare the observed flip rate, per quartile of the reference |s|,
against the fitted ``Phi(-lam * |s| / sigma)``. If flips are just sign
re-randomization under a rubric-sized perturbation, the model should track
the quartile profile with two parameters per judge; where the observed rate
falls *below* the prediction, the rubric change is more structured than
noise (the judge moves coherently, so fewer signs flip than a Gaussian
perturbation of the same size would produce).

Writes ``results/summary/rubric_fragility__{a}_vs_{b}.{json,md}`` and the
figure ``results/figures/rubric_fragility__{a}_vs_{b}.png`` (observed
per-quartile flip rates with 95% bootstrap CIs as points, fitted curves as
lines, one color per judge).

Usage:
    python -m experiments.rubric_fragility
    python -m experiments.rubric_fragility --rubric-a minimal --rubric-b detailed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judge import MODELS, RESULTS_DIR  # noqa: E402
from src.rubric_pair import fragility_fit, match_rubric_pairs  # noqa: E402

from experiments.make_figures import INK, INK_MUTED, INK_SECONDARY  # noqa: E402
from experiments.rubric_view import load_pairs  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

MODEL_STYLES = {
    "qwen2.5-0.5b": ("#7fb3e6", "o"),
    "qwen2.5-1.5b": ("#2a78d6", "s"),
    "qwen2.5-3b": ("#174a87", "^"),
    "qwen2.5-7b": ("#0d2c52", "D"),
    "llama-3.2-1b": ("#e08f88", "v"),
    "llama-3.2-3b": ("#c2483f", "P"),
    "llama-3.1-8b": ("#7c2a24", "X"),
}


def format_table(fits: dict[str, dict]) -> str:
    header = (
        "| model | median |s| | flip rate [95% CI] | lam | sigma "
        "| Q1 obs/model | Q2 | Q3 | Q4 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for model_key in sorted(fits, key=lambda k: (MODELS[k].params_b, k)):
        f = fits[model_key]
        ci = f["overall_flip"]["ci95"]
        cells = " | ".join(
            f"{q['observed_flip']['mean']:.3f}/{q['predicted_flip']:.3f}"
            for q in f["quartiles"]
        )
        rows.append(
            f"| {model_key} | {f['median_abs_s']:.3f} "
            f"| {f['overall_flip']['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
            f"| {f['lam']:.3f} | {f['sigma']:.3f} | {cells} |"
        )
    return header + "\n".join(rows) + "\n"


def render_figure(fits: dict[str, dict], rubric_a: str, rubric_b: str,
                  out: Path) -> None:
    from math import erf, sqrt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.grid(True, axis="y", zorder=0)
    xmax = max(
        q["median_abs_s"] for f in fits.values() for q in f["quartiles"]
    ) * 1.25
    grid = np.linspace(0.0, xmax, 200)
    for model_key in sorted(fits, key=lambda k: (MODELS[k].params_b, k)):
        f = fits[model_key]
        color, marker = MODEL_STYLES[model_key]
        lam, sigma = f["lam"], f["sigma"]
        curve = [0.5 * (1.0 + erf(-lam * x / sigma / sqrt(2.0))) for x in grid]
        ax.plot(grid, curve, color=color, linewidth=1.2, alpha=0.7, zorder=2)
        xs = [q["median_abs_s"] for q in f["quartiles"]]
        ys = [q["observed_flip"]["mean"] for q in f["quartiles"]]
        los = [y - q["observed_flip"]["ci95"][0] for y, q in zip(ys, f["quartiles"])]
        his = [q["observed_flip"]["ci95"][1] - y for y, q in zip(ys, f["quartiles"])]
        ax.errorbar(
            xs, ys, yerr=[los, his], fmt=marker, color=color, markersize=5,
            linewidth=0, elinewidth=1.0, capsize=2,
            label=f"{model_key} (λ={lam:.2f}, σ={sigma:.2f})", zorder=3,
        )
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax.annotate("coin flip", xy=(xmax, 0.5), xytext=(-2, 3),
                textcoords="offset points", ha="right", fontsize=8,
                color=INK_MUTED)
    ax.set_xlabel(f"median |s| of quartile under the {rubric_a} rubric (log-odds)")
    ax.set_ylabel("rubric flip rate")
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.set_title(
        f"Rubric flips concentrate where the preference is weak "
        f"({rubric_a} vs {rubric_b})", color=INK,
    )
    ax.legend(loc="upper right", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric-a", default="minimal")
    parser.add_argument("--rubric-b", default="detailed")
    args = parser.parse_args()

    fits: dict[str, dict] = {}
    for model_key in sorted(MODELS):
        pairs_a = load_pairs(model_key, args.rubric_a)
        pairs_b = load_pairs(model_key, args.rubric_b)
        if not pairs_a or not pairs_b:
            continue
        matched, only_a, only_b = match_rubric_pairs(pairs_a, pairs_b)
        if only_a or only_b:
            print(f"[rubric_fragility] {model_key}: {only_a}/{only_b} unmatched "
                  f"items excluded from the join", flush=True)
        fits[model_key] = fragility_fit(matched)
    if not fits:
        raise SystemExit("no model has complete stores under both rubrics")

    stem = f"rubric_fragility__{args.rubric_a}_vs_{args.rubric_b}"
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUMMARY_DIR / f"{stem}.json"
    out_json.write_text(json.dumps(fits, indent=2, sort_keys=True) + "\n")
    table = format_table(fits)
    (SUMMARY_DIR / f"{stem}.md").write_text(table)
    print(table)
    out_fig = FIGURES_DIR / f"{stem}.png"
    render_figure(fits, args.rubric_a, args.rubric_b, out_fig)
    print(f"[rubric_fragility] wrote {out_json}")
    print(f"[rubric_fragility] wrote {out_fig}")


if __name__ == "__main__":
    main()

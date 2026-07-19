"""The judge scaling picture: accuracy and position bias vs. parameter count.

Reads every completed ``{model}__{rubric}.jsonl`` store for one rubric,
recomputes the headline quantities from raw records (never from summary
files, so the figure is regenerable from the same provenance as everything
else), and renders one two-panel figure:

- left: symmetrized and raw (random-order) accuracy vs. nominal parameter
  count, 95% paired-bootstrap CIs, one line per model family, against the
  random and longer-response floors;
- right: median |b| (position-bias magnitude, log-odds) and median |s|
  (content-signal magnitude) vs. parameter count — the bias-vs-signal race
  that raw accuracy hides.

Writes results/figures/scaling__{rubric}.png.

Usage:
    python -m experiments.scaling_curve [--rubric minimal]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import assemble_pairs, bootstrap_mean_ci  # noqa: E402
from src.baselines import longer_response_correct  # noqa: E402
from src.data import fetch, load_rewardbench  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402

from experiments.make_figures import INK, INK_MUTED  # noqa: E402

FIGURES_DIR = RESULTS_DIR.parent / "figures"

FAMILY_STYLES = {
    "Qwen2.5": ("#2a78d6", "o"),
    "Llama-3.2": ("#c2483f", "s"),
}


def family_of(model_key: str) -> str:
    if model_key.startswith("qwen2.5"):
        return "Qwen2.5"
    if model_key.startswith("llama-3.2"):
        return "Llama-3.2"
    raise ValueError(f"unknown family for {model_key}")


def collect(rubric: str, only: list[str] | None = None) -> list[dict]:
    """One row per completed store: point stats + CIs, keyed for plotting."""
    rows = []
    for key, model in sorted(MODELS.items(), key=lambda kv: kv[1].params_b):
        path = RESULTS_DIR / f"{key}__{rubric}.jsonl"
        if not path.exists() or (only is not None and key not in only):
            continue
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[scaling] {path.stem}: {incomplete} incomplete items skipped")
        rows.append(
            {
                "model": key,
                "family": family_of(key),
                "params_b": model.params_b,
                "n_items": len(pairs),
                "sym": bootstrap_mean_ci([p.sym_correct for p in pairs]),
                "raw": bootstrap_mean_ci([p.raw_correct_mean for p in pairs]),
                "median_abs_b": float(np.median([abs(p.b) for p in pairs])),
                "median_abs_s": float(np.median([abs(p.s) for p in pairs])),
                "item_ids": [p.item_id for p in pairs],
            }
        )
    return rows


def _param_axis(ax, rows: list[dict]) -> None:
    """Log parameter axis with ticks only at the audited sizes."""
    from matplotlib.ticker import NullFormatter, NullLocator

    ax.set_xscale("log")
    ax.set_xticks(sorted({r["params_b"] for r in rows}),
                  [f"{p:g}B" for p in sorted({r["params_b"] for r in rows})])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def scaling_figure(rows: list[dict], rubric: str, longer_floor: float, out: Path) -> None:
    fig, (ax_acc, ax_dec) = plt.subplots(1, 2, figsize=(9.6, 4.4))

    for family, (color, marker) in FAMILY_STYLES.items():
        group = [r for r in rows if r["family"] == family]
        if not group:
            continue
        xs = [r["params_b"] for r in group]
        sym = [r["sym"][0] for r in group]
        yerr = [[r["sym"][0] - r["sym"][1] for r in group],
                [r["sym"][2] - r["sym"][0] for r in group]]
        ax_acc.errorbar(xs, sym, yerr=yerr, fmt=f"{marker}-", color=color,
                        elinewidth=1.1, capsize=3, markersize=5, linewidth=1.4,
                        label=f"{family} symmetrized", zorder=3)
        raw = [r["raw"][0] for r in group]
        ax_acc.plot(xs, raw, linestyle="--", marker=marker, markersize=4,
                    color=color, alpha=0.45, linewidth=1.1,
                    label=f"{family} raw (random order)", zorder=2)

        ax_dec.plot(xs, [r["median_abs_b"] for r in group], f"{marker}-",
                    color=color, markersize=5, linewidth=1.4,
                    label=f"{family} median |b| (bias)", zorder=3)
        ax_dec.plot(xs, [r["median_abs_s"] for r in group], f"{marker}--",
                    color=color, markersize=4, alpha=0.45, linewidth=1.1,
                    label=f"{family} median |s| (signal)", zorder=2)

    ax_acc.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=1)
    ax_acc.annotate("random / always-A floor", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=8, color=INK_MUTED)
    ax_acc.axhline(longer_floor, color="#8a6d3b", linewidth=1.0, linestyle="--", zorder=1)
    ax_acc.annotate(f"longer-response floor ({longer_floor:.3f})",
                    xy=(0.02, longer_floor), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", fontsize=8, color="#8a6d3b")
    _param_axis(ax_acc, rows)
    ax_acc.set_xlabel("nominal parameters")
    ax_acc.set_ylabel("accuracy vs. gold label (95% CI)")
    ax_acc.set_ylim(0.3, 1.0)
    ax_acc.set_title("accuracy scaling")
    ax_acc.grid(True, axis="y", alpha=0.6)
    ax_acc.legend(loc="upper left", fontsize=7)

    _param_axis(ax_dec, rows)
    ax_dec.set_xlabel("nominal parameters")
    ax_dec.set_ylabel("median magnitude (log-odds)")
    ax_dec.set_title("position bias |b| vs. content signal |s|")
    ax_dec.grid(True, axis="y", alpha=0.6)
    ax_dec.legend(loc="upper right", fontsize=7)

    n_items = {r["n_items"] for r in rows}
    n_txt = f"n={n_items.pop()}" if len(n_items) == 1 else "varying n"
    fig.suptitle(f"judge scaling on RewardBench stratified sample "
                 f"({n_txt} items, rubric={rubric})", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model keys to include (default: every completed "
                             "store; use this to exclude an in-flight run)")
    args = parser.parse_args()

    rows = collect(args.rubric, only=args.models)
    if len(rows) < 2:
        raise SystemExit(f"need >=2 completed stores for rubric {args.rubric}, "
                         f"found {len(rows)}")

    # All stores must cover the same items or the curve compares samples,
    # not models. Hard failure beats a silently confounded figure.
    id_sets = {tuple(sorted(r["item_ids"])) for r in rows}
    if len(id_sets) > 1:
        raise SystemExit("stores cover different item sets; refusing to plot")

    fetch()
    items_by_id = {item.item_id: item for item in load_rewardbench()}
    longer_floor = float(np.mean(
        [longer_response_correct(items_by_id[i]) for i in rows[0]["item_ids"]]
    ))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"scaling__{args.rubric}.png"
    scaling_figure(rows, args.rubric, longer_floor, out)
    for r in rows:
        print(f"[scaling] {r['model']}: sym {r['sym'][0]:.3f} "
              f"[{r['sym'][1]:.3f}, {r['sym'][2]:.3f}], raw {r['raw'][0]:.3f}, "
              f"med|b| {r['median_abs_b']:.2f}, med|s| {r['median_abs_s']:.2f}")
    print(f"[scaling] wrote {out}")


if __name__ == "__main__":
    main()

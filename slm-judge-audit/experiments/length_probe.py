"""Run the value-over-length probe over completed judgment stores.

For every selected ``{model}__{rubric}`` store: assemble swap pairs, join item
length statistics, fit the nested conditional-logit specs (length-only /
judge-only / joint / joint-sign) overall and per category, with bootstrap CIs
from src.length_probe. Writes one JSON with every model's blocks plus a
two-panel forest figure of the headline quantities:

- left: standardized beta_s in the joint spec — the judge's signal about the
  gold label after controlling for the log length ratio;
- right: the paired in-sample accuracy delta of joint over length-only — what
  adding the judge to a length heuristic is worth in points.

All selected stores must cover the same item set (the guard from
scaling_curve): otherwise strata would compare samples, not models.

Usage:
    python -m experiments.length_probe [--rubric minimal] [--models k1 k2 ...]
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
from src.data import fetch, load_rewardbench  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402
from src.length_probe import build_rows, value_over_length  # noqa: E402

from experiments.make_figures import INK, INK_MUTED, INK_SECONDARY  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

MODEL_STYLES = {
    "qwen2.5-0.5b": ("#9dc3ee", "o"),
    "qwen2.5-1.5b": ("#4f92dd", "o"),
    "qwen2.5-3b": ("#184f95", "o"),
    "llama-3.2-1b": ("#c2483f", "s"),
}

STRATA = ("Overall", "Chat", "Chat Hard", "Reasoning", "Safety")


def stratum_block(result: dict, stratum: str) -> dict:
    return result["overall"] if stratum == "Overall" else result["by_category"][stratum]


def forest_figure(results: dict[str, dict], rubric: str, out: Path) -> None:
    models = [k for k in MODEL_STYLES if k in results]
    fig, (ax_coef, ax_acc) = plt.subplots(1, 2, figsize=(9.6, 5.2), sharey=True)

    group_gap = len(models) + 1.6
    yticks, ylabels = [], []
    for g, stratum in enumerate(STRATA):
        base = -g * group_gap
        yticks.append(base - (len(models) - 1) / 2)
        ylabels.append(stratum)
        for j, model in enumerate(models):
            block = stratum_block(results[model], stratum)
            color, marker = MODEL_STYLES[model]
            y = base - j

            spec = block["specs"]["joint"]
            mean = spec["coef"]["s"]
            lo, hi = spec["coef_ci95"]["s"]
            ax_coef.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]],
                             fmt=marker, color=color, markersize=4.5,
                             elinewidth=1.2, capsize=2.5, zorder=3,
                             label=model if g == 0 else None)

            d = block["acc_joint_minus_length"]
            lo_d, hi_d = d["ci95"]
            ax_acc.errorbar([d["mean"]], [y], xerr=[[d["mean"] - lo_d], [hi_d - d["mean"]]],
                            fmt=marker, color=color, markersize=4.5,
                            elinewidth=1.2, capsize=2.5, zorder=3)

    for ax in (ax_coef, ax_acc):
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.9, zorder=1)
        ax.grid(True, axis="x", alpha=0.6)
        for g in range(len(STRATA) - 1):
            ax.axhline(-g * group_gap - len(models) + 0.35, color="#eceae2",
                       linewidth=0.8, zorder=0)
    ax_coef.set_yticks(yticks, ylabels)
    ax_coef.set_xlabel(r"$\beta_s$ in joint spec (log-odds per SD of $s$, 95% CI)")
    ax_coef.set_title("judge signal beyond length")
    ax_acc.set_xlabel("accuracy delta, joint − length-only (95% CI)")
    ax_acc.set_title("what the judge adds to a length heuristic")
    ax_coef.legend(loc="lower left", fontsize=7)

    n = results[models[0]]["overall"]["n_items"]
    fig.suptitle(
        f"value over length: conditional-logit probe on gold labels "
        f"(n={n} items, rubric={rubric})", fontsize=10, color=INK,
    )
    fig.text(0.5, 0.015,
             "positive = the judge's order-invariant preference s predicts the gold label "
             "after controlling for log length ratio",
             ha="center", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
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

    fetch()
    items_by_id = {item.item_id: item for item in load_rewardbench()}

    results: dict[str, dict] = {}
    item_sets: dict[str, tuple] = {}
    for key in sorted(MODELS):
        path = RESULTS_DIR / f"{key}__{args.rubric}.jsonl"
        if not path.exists() or (args.models is not None and key not in args.models):
            continue
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[probe] {path.stem}: {incomplete} incomplete items skipped")
        rows = build_rows(pairs, items_by_id)
        print(f"[probe] {key}: fitting {len(rows)} items ...", flush=True)
        results[key] = value_over_length(rows, n_boot=args.n_boot)
        item_sets[key] = tuple(sorted(row.item_id for row in rows))

    if len(results) < 1:
        raise SystemExit(f"no completed stores for rubric {args.rubric}")
    if len(set(item_sets.values())) > 1:
        raise SystemExit("stores cover different item sets; refusing to compare "
                         "(use --models to select same-sample stores)")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUMMARY_DIR / f"length_probe__{args.rubric}.json"
    with open(out_json, "w") as f:
        json.dump({"rubric": args.rubric, "models": results}, f, indent=2, sort_keys=True)
        f.write("\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURES_DIR / f"length_probe__{args.rubric}.png"
    forest_figure(results, args.rubric, out_fig)

    for key, result in results.items():
        for stratum in STRATA:
            block = stratum_block(result, stratum)
            spec = block["specs"]["joint"]
            lo, hi = spec["coef_ci95"]["s"]
            d = block["acc_joint_minus_length"]
            print(f"[probe] {key:14s} {stratum:9s} n={block['n_items']:3d}  "
                  f"beta_s {spec['coef']['s']:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"beta_len {spec['coef']['dlog_chars']:+.3f}  "
                  f"acc joint-len {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print(f"[probe] wrote {out_json} and {out_fig}")


if __name__ == "__main__":
    main()

"""Run the additive-shift test and correction ladder over completed stores.

For every selected ``{model}__{rubric}`` store: assemble swap pairs, join item
covariates, and compute (1) the variance decomposition of position bias b —
how much of it category / subset / subset+length structure explains, with a
refit inside every bootstrap replicate — and (2) the single-order correction
ladder: accuracy of ``sign(z - b_hat)`` under exact leave-one-out corrections,
from no correction up to the oracle (which is identically the symmetrized
verdict). Writes one JSON with every model's blocks plus a two-panel figure:

- left: R^2 of each bias predictor per model — the additive-shift test
  (a pure additive shift would put every spec at ~0);
- right: the accuracy ladder per model — what fraction of the two-call
  symmetrization gain each one-call correction recovers.

All selected stores must cover the same item set (the scaling_curve guard).

Usage:
    python -m experiments.bias_model [--rubric minimal] [--models k1 k2 ...]
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
from src.bias_model import CORRECTIONS, DECOMP_SPECS, bias_structure, build_bias_rows  # noqa: E402
from src.data import fetch, load_rewardbench  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402

from experiments.make_figures import INK, INK_MUTED, INK_SECONDARY  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

# Row order: ascending parameter count, families interleaved.
MODEL_ORDER = ("qwen2.5-0.5b", "llama-3.2-1b", "qwen2.5-1.5b", "qwen2.5-3b",
               "llama-3.2-3b")

SPEC_STYLES = {
    "category": ("#2a78d6", "o", "category means"),
    "subset": ("#1baf7a", "s", "subset means"),
    "subset_plus_length": ("#9a5bd2", "D", "subset + length"),
}

ESTIMATOR_STYLES = {
    "none": ("#b3b1a9", "o", "raw single-order"),
    "global": ("#86b6ef", "v", "global constant"),
    "category": ("#2a78d6", "o", "per-category"),
    "subset": ("#1baf7a", "s", "per-subset"),
    "regression": ("#9a5bd2", "D", "subset + length"),
    "oracle": ("#0b0b0b", "*", "oracle = symmetrized (2 calls)"),
}


def figure(results: dict[str, dict], rubric: str, out: Path) -> None:
    models = [k for k in MODEL_ORDER if k in results]
    fig, (ax_r2, ax_acc) = plt.subplots(1, 2, figsize=(9.8, 4.8), sharey=True)
    ys = {model: -i for i, model in enumerate(models)}

    for model in models:
        y = ys[model]
        decomp = results[model]["decomposition"]["specs"]
        for j, spec in enumerate(DECOMP_SPECS):
            color, marker, _ = SPEC_STYLES[spec]
            block = decomp[spec]
            lo, hi = block["r2_ci95"]
            ax_r2.errorbar([block["r2"]], [y + 0.22 - 0.22 * j],
                           xerr=[[block["r2"] - lo], [hi - block["r2"]]],
                           fmt=marker, color=color, markersize=4.5,
                           elinewidth=1.1, capsize=2.5, zorder=3)

        ladder = results[model]["ladder"]["estimators"]
        accs = [ladder[name]["acc"]["mean"] for name in CORRECTIONS]
        ax_acc.plot(accs, [y] * len(accs), color="#dddbd2", linewidth=1.2, zorder=1)
        for name in CORRECTIONS:
            color, marker, _ = ESTIMATOR_STYLES[name]
            block = ladder[name]["acc"]
            lo, hi = block["ci95"]
            ax_acc.errorbar([block["mean"]], [y],
                            xerr=[[block["mean"] - lo], [hi - block["mean"]]],
                            fmt=marker, color=color,
                            markersize=8 if name == "oracle" else 4.5,
                            elinewidth=1.0, capsize=2.0, zorder=3)

    for spec, (color, marker, label) in SPEC_STYLES.items():
        ax_r2.errorbar([], [], fmt=marker, color=color, label=label)
    ladder_handles = [
        ax_acc.errorbar([], [], fmt=marker, color=color, label=label)
        for color, marker, label in ESTIMATOR_STYLES.values()
    ]

    ax_r2.set_yticks([ys[m] for m in models], models)
    ax_r2.axvline(0, color=INK_SECONDARY, linewidth=0.9, zorder=1)
    ax_r2.set_xlabel(r"$R^2$ for position bias $b_i$ (95% CI, refit per replicate)")
    ax_r2.set_title("is the bias an additive constant? (R$^2$ ~ 0 if so)")
    ax_r2.grid(True, axis="x", alpha=0.6)
    ax_r2.legend(loc="lower right", fontsize=7)

    ax_acc.axvline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=":", zorder=1)
    ax_acc.set_xlabel("accuracy of corrected single-order verdict (95% CI)")
    ax_acc.set_title("one-call bias correction vs. two-call symmetrization")
    ax_acc.grid(True, axis="x", alpha=0.6)
    fig.legend(handles=ladder_handles, loc="lower center", ncols=3, fontsize=7.5,
               columnspacing=1.4, handletextpad=0.4)

    n = results[models[0]]["ladder"]["n_items"]
    fig.suptitle(
        f"position-bias structure and single-order correction "
        f"(n={n} items, rubric={rubric}, exact leave-one-out)",
        fontsize=10, color=INK,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
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
            print(f"[bias] {path.stem}: {incomplete} incomplete items skipped")
        rows = build_bias_rows(pairs, items_by_id)
        print(f"[bias] {key}: decomposing {len(rows)} items ...", flush=True)
        results[key] = bias_structure(rows, n_boot=args.n_boot)
        item_sets[key] = tuple(sorted(row.item_id for row in rows))

    if len(results) < 1:
        raise SystemExit(f"no completed stores for rubric {args.rubric}")
    if len(set(item_sets.values())) > 1:
        raise SystemExit("stores cover different item sets; refusing to compare "
                         "(use --models to select same-sample stores)")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUMMARY_DIR / f"bias_model__{args.rubric}.json"
    with open(out_json, "w") as f:
        json.dump({"rubric": args.rubric, "models": results}, f, indent=2, sort_keys=True)
        f.write("\n")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_fig = FIGURES_DIR / f"bias_model__{args.rubric}.png"
    figure(results, args.rubric, out_fig)

    for key in (k for k in MODEL_ORDER if k in results):
        decomp = results[key]["decomposition"]
        ladder = results[key]["ladder"]["estimators"]
        r2 = {s: decomp["specs"][s]["r2"] for s in DECOMP_SPECS}
        accs = "  ".join(f"{name} {ladder[name]['acc']['mean']:.3f}"
                         for name in CORRECTIONS)
        rec = ladder["subset"].get("recovered_fraction")
        rec_str = f"  subset recovers {rec:.0%}" if rec is not None else ""
        print(f"[bias] {key:14s} b_sd {decomp['b_sd']:.2f}  "
              f"R2 cat {r2['category']:.3f} sub {r2['subset']:.3f} "
              f"sub+len {r2['subset_plus_length']:.3f}{rec_str}")
        print(f"[bias] {key:14s} ladder: {accs}")
    print(f"[bias] wrote {out_json} and {out_fig}")


if __name__ == "__main__":
    main()

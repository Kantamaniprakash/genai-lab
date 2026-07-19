"""Compliance-conditioned view of a judge store: does the audit survive the
readout's own validity check?

Finding 5 (2026-07-18) showed verdict-format compliance is a per-family
property: Qwen2.5-0.5B's unconstrained argmax is a verdict letter on 100% of
judgments, Llama-3.2-1B's on barely half, with mass on {A, B} spanning the
whole unit interval. Every Llama number therefore carries the qualification
"z may measure a renormalized sub-distribution". This script makes that
qualification quantitative per store:

- symmetrized accuracy and decomposition stats on the argmax-compliant
  stratum vs. the rest, with an unpaired bootstrap CI on the gap;
- a validity curve over bins of mass_min (min over orders of the probability
  mass on the verdict letters);
- per-category compliance composition, so stratum differences can be read
  against category mix rather than mistaken for pure readout effects.

Writes results/summary/{store}__compliance.json and
results/figures/{store}_compliance.png.

Usage:
    python -m experiments.compliance_view                     # all stores
    python -m experiments.compliance_view --store llama-3.2-1b__minimal
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

from src.analysis import assemble_pairs, compliance_view  # noqa: E402
from src.data import SUBSET_TO_CATEGORY  # noqa: E402
from src.judge import RESULTS_DIR, load_records  # noqa: E402

from experiments.make_figures import (  # noqa: E402
    AXIS,
    INK,
    INK_MUTED,
    INK_SECONDARY,
)

SUMMARY_DIR = RESULTS_DIR.parent / "summary"
FIGURES_DIR = RESULTS_DIR.parent / "figures"

STRATUM_ORDER = ("all", "compliant_both", "non_compliant")
STRATUM_LABELS = {
    "all": "all items",
    "compliant_both": "argmax compliant\n(both orders)",
    "non_compliant": "non-compliant\n(either order)",
}
STRATUM_COLORS = {
    "all": "#2a78d6",
    "compliant_both": "#1baf7a",
    "non_compliant": "#c2483f",
}


def category_of(item_id: str) -> str:
    return SUBSET_TO_CATEGORY[item_id.split("/", 1)[0]]


def compliance_figure(store: str, view: dict, out: Path) -> None:
    fig, (ax_strata, ax_mass) = plt.subplots(
        1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    # Left: symmetrized accuracy by compliance stratum.
    present = [s for s in STRATUM_ORDER if s in view["strata"]]
    xs = range(len(present))
    means = [view["strata"][s]["sym_acc"]["mean"] for s in present]
    los = [view["strata"][s]["sym_acc"]["ci95"][0] for s in present]
    his = [view["strata"][s]["sym_acc"]["ci95"][1] for s in present]
    bars = ax_strata.bar(
        xs, means, width=0.6, color=[STRATUM_COLORS[s] for s in present], zorder=2
    )
    ax_strata.errorbar(
        xs, means,
        yerr=[[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]],
        fmt="none", ecolor=INK_SECONDARY, elinewidth=1.1, capsize=3, zorder=3,
    )
    for x, bar, stratum, mean in zip(xs, bars, present, means):
        ax_strata.annotate(
            f"{mean:.3f}", xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 5), textcoords="offset points", ha="center",
            fontsize=8, color=INK,
        )
        ax_strata.annotate(
            f"n={view['strata'][stratum]['n_items']}",
            xy=(x, 0.02), ha="center", va="bottom", fontsize=8, color="#ffffff",
        )
    ax_strata.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=4)
    ax_strata.set_xticks(list(xs), [STRATUM_LABELS[s] for s in present])
    ax_strata.set_ylim(0, 1.0)
    ax_strata.set_ylabel("symmetrized accuracy (95% CI)")
    delta = view["sym_acc_compliant_minus_non"]
    subtitle = ""
    if delta is not None:
        subtitle = (f"\ncompliant − non-compliant: {delta['mean']:+.3f} "
                    f"[{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]")
    ax_strata.set_title("by argmax-compliance stratum" + subtitle)
    ax_strata.grid(True, axis="y", alpha=0.6)

    # Right: validity curve over mass_min bins (only bins with members).
    bins = [b for b in view["mass_bins"] if b["n_items"] > 0]
    centers = [(b["lo"] + min(b["hi"], 1.0)) / 2 for b in bins]
    widths = [min(b["hi"], 1.0) - b["lo"] for b in bins]
    b_means = [b["sym_acc"]["mean"] for b in bins]
    b_los = [b["sym_acc"]["ci95"][0] for b in bins]
    b_his = [b["sym_acc"]["ci95"][1] for b in bins]
    ax_mass.errorbar(
        centers, b_means,
        yerr=[[m - lo for m, lo in zip(b_means, b_los)],
              [hi - m for m, hi in zip(b_means, b_his)]],
        fmt="o-", color="#2a78d6", ecolor=INK_SECONDARY,
        elinewidth=1.1, capsize=3, markersize=5, linewidth=1.2, zorder=3,
    )
    for center, width, block in zip(centers, widths, bins):
        ax_mass.annotate(
            f"n={block['n_items']}", xy=(center, 0.04), ha="center",
            va="bottom", fontsize=8, color=INK_MUTED,
        )
        ax_mass.axvline(center - width / 2, color=AXIS, linewidth=0.5, zorder=1)
    ax_mass.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
    ax_mass.set_xlim(0, 1.0)
    ax_mass.set_ylim(0, 1.0)
    ax_mass.set_xlabel("mass_min: min over orders of probability mass on {A, B}")
    ax_mass.set_ylabel("symmetrized accuracy (95% CI)")
    ax_mass.set_title("validity curve: accuracy vs. readout mass")
    ax_mass.grid(True, axis="both", alpha=0.6)

    fig.suptitle(f"{store}: readout-validity conditioning "
                 f"(compliance rate {view['compliance_rate']:.3f})",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=None,
                        help="single store stem, e.g. llama-3.2-1b__minimal")
    args = parser.parse_args()

    paths = ([RESULTS_DIR / f"{args.store}.jsonl"] if args.store
             else sorted(RESULTS_DIR.glob("*__*.jsonl")))
    if not paths:
        raise SystemExit("no stores found in results/raw")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[compliance] {path.stem}: skipping {incomplete} incomplete items")
        view = compliance_view(pairs, category_of=category_of)
        out = SUMMARY_DIR / f"{path.stem}__compliance.json"
        with open(out, "w") as f:
            json.dump(view, f, indent=2, sort_keys=True)
            f.write("\n")
        compliance_figure(path.stem, view, FIGURES_DIR / f"{path.stem}_compliance.png")
        print(f"[compliance] wrote {out.name} + figure "
              f"(compliance rate {view['compliance_rate']:.3f})")

        strata = view["strata"]
        print(f"| {path.stem} | stratum | n | sym acc (95% CI) | med b | med |s| | flips |")
        print("|" + "---|" * 7)
        for name in STRATUM_ORDER:
            if name not in strata:
                continue
            s = strata[name]
            acc = s["sym_acc"]
            print(f"| | {name} | {s['n_items']} "
                  f"| {acc['mean']:.3f} [{acc['ci95'][0]:.3f}, {acc['ci95'][1]:.3f}] "
                  f"| {s['median_b']:+.2f} | {s['median_abs_s']:.2f} "
                  f"| {s['positional_flip_rate']:.3f} |")


if __name__ == "__main__":
    main()

"""Quick-look summary of one or more judge stores, against the trivial floors.

Reads results/raw/{model}__{rubric}.jsonl, assembles swap pairs, computes the
standard summary block (accuracies with paired-bootstrap CIs, bias/preference
decomposition stats, compliance diagnostics), attaches the baseline floors
computed on exactly the same items, writes results/summary/{model}__{rubric}.json,
and prints a compact markdown table across all summarized stores.

Usage:
    python -m experiments.summarize                 # all stores in results/raw
    python -m experiments.summarize --store qwen2.5-0.5b__minimal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import assemble_pairs, summarize_pairs  # noqa: E402
from src.baselines import summarize_baselines  # noqa: E402
from src.data import fetch, load_rewardbench  # noqa: E402
from src.judge import RESULTS_DIR, load_records  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"


def summarize_store(path: Path, items_by_id: dict) -> dict:
    records = load_records([path])
    pairs, incomplete = assemble_pairs(records)
    summary = summarize_pairs(pairs)
    summary["store"] = path.stem
    summary["n_incomplete_items"] = incomplete
    pair_items = tuple(items_by_id[p.item_id] for p in pairs)
    summary["baselines"] = summarize_baselines(pair_items)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=None,
                        help="single store stem, e.g. qwen2.5-0.5b__minimal")
    args = parser.parse_args()

    fetch()
    items_by_id = {item.item_id: item for item in load_rewardbench()}

    if args.store:
        paths = [RESULTS_DIR / f"{args.store}.jsonl"]
    else:
        paths = sorted(RESULTS_DIR.glob("*__*.jsonl"))
    if not paths:
        raise SystemExit("no stores found in results/raw")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        summary = summarize_store(path, items_by_id)
        out = SUMMARY_DIR / f"{path.stem}.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        rows.append(summary)
        print(f"[summarize] wrote {out}")

    header = ("| store | n | raw acc (95% CI) | sym acc (95% CI) | Δ sym−raw | "
              "flip rate | mean b | sd b | med |s| | bias>signal | longer floor |")
    print()
    print(header)
    print("|" + "---|" * 11)
    for r in rows:
        raw, sym, delta = r["raw_acc"], r["sym_acc"], r["sym_minus_raw"]
        print(
            f"| {r['store']} | {r['n_items']} "
            f"| {raw['mean']:.3f} [{raw['ci95'][0]:.3f}, {raw['ci95'][1]:.3f}] "
            f"| {sym['mean']:.3f} [{sym['ci95'][0]:.3f}, {sym['ci95'][1]:.3f}] "
            f"| {delta['mean']:+.3f} [{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}] "
            f"| {r['positional_flip_rate']:.3f} "
            f"| {r['bias_b']['mean']:+.2f} | {r['bias_b']['sd']:.2f} "
            f"| {r['preference_s']['median_abs']:.2f} "
            f"| {r['frac_bias_dominates']:.3f} "
            f"| {r['baselines']['overall']['longer_chars']:.3f} |"
        )


if __name__ == "__main__":
    main()

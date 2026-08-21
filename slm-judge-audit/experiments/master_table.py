"""The cross-judge headline table: every completed grid, one row, same items.

The per-grid sections of the README grew in the order the grids finished, so
each one compares the new judge against whichever judges existed at the time.
This script produces the missing spine: a single table over *every* completed
store for a rubric, recomputed from raw records, with the trivial floors
measured on exactly the same items underneath.

Two things live here that the per-store block in ``experiments.summarize``
cannot express, because both are comparisons against a baseline evaluated on
the same items:

- **per-order accuracy** (chosen-first / rejected-first). A judge whose two
  order accuracies are 0.99 and 0.02 is an always-A machine no matter what its
  symmetrized accuracy says; one at 0.60/0.58 is reading content. The pair is
  the fastest visual test for a saturated position bias in the audit.
- **sym − longer**, the paired delta of the symmetrized verdict against the
  pick-the-longer-response floor. This is the "is the judge worth its cost?"
  column: a judge below it is a length heuristic with extra steps.

Coverage discipline: the reference item set is the widest one any store
covers. A store over a strict subset of it is a grid still in flight — it is
dropped with its coverage printed, never averaged in at partial n. A store
carrying items *outside* the reference set is a different sample and aborts
the comparison.

Writes results/summary/master_table__{rubric}.json and the rendered
results/summary/master_table__{rubric}.md, which the README embeds verbatim.

Usage:
    python -m experiments.master_table [--rubric minimal] [--models ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import assemble_pairs, judge_row  # noqa: E402
from src.baselines import RANDOM_ACCURACY, longer_response_correct  # noqa: E402
from src.data import fetch, load_rewardbench  # noqa: E402
from src.judge import MODELS, RESULTS_DIR, load_records  # noqa: E402

from experiments.length_probe import MODEL_STYLES  # noqa: E402

SUMMARY_DIR = RESULTS_DIR.parent / "summary"

COLUMNS = (
    ("judge", "{key}"),
    ("params", "{params_b:g}B"),
    ("compliant", "{compliance_rate:.3f}"),
    ("acc A-first", "{raw_acc_chosen_first:.3f}"),
    ("acc B-first", "{raw_acc_rejected_first:.3f}"),
    ("flip rate", "{positional_flip_rate:.3f}"),
    ("median b", "{median_b:+.2f}"),
    ("b > 0", "{frac_b_positive:.3f}"),
    # The pipes in the |s| notation are escaped: an unescaped one would split
    # the cell and silently shift every column after it by one.
    (r"median \|s\|", "{median_abs_s:.2f}"),
    ("bias > signal", "{frac_bias_dominates:.3f}"),
    ("raw acc", "{raw_acc_mean:.3f}"),
    ("sym acc (95% CI)", "{sym_acc_mean:.3f} [{sym_acc_lo:.3f}, {sym_acc_hi:.3f}]"),
    ("Δ sym−raw", "{rescue:+.3f} [{rescue_lo:+.3f}, {rescue_hi:+.3f}]"),
    ("Δ sym−longer", "{over_len:+.3f} [{over_len_lo:+.3f}, {over_len_hi:+.3f}]"),
)


def flatten(key: str, params_b: float, row: dict) -> dict:
    """Row dict flattened to the scalar names the column templates use."""
    return {
        "key": key,
        "params_b": params_b,
        "compliance_rate": row["compliance_rate"],
        "raw_acc_chosen_first": row["raw_acc_chosen_first"],
        "raw_acc_rejected_first": row["raw_acc_rejected_first"],
        "positional_flip_rate": row["positional_flip_rate"],
        "median_b": row["median_b"],
        "frac_b_positive": row["frac_b_positive"],
        "median_abs_s": row["median_abs_s"],
        "frac_bias_dominates": row["frac_bias_dominates"],
        "raw_acc_mean": row["raw_acc"]["mean"],
        "sym_acc_mean": row["sym_acc"]["mean"],
        "sym_acc_lo": row["sym_acc"]["ci95"][0],
        "sym_acc_hi": row["sym_acc"]["ci95"][1],
        "rescue": row["sym_minus_raw"]["mean"],
        "rescue_lo": row["sym_minus_raw"]["ci95"][0],
        "rescue_hi": row["sym_minus_raw"]["ci95"][1],
        "over_len": row["sym_minus_longer"]["mean"],
        "over_len_lo": row["sym_minus_longer"]["ci95"][0],
        "over_len_hi": row["sym_minus_longer"]["ci95"][1],
    }


def fitted_length_note(rubric: str = "minimal") -> str:
    """The fitted length baseline's accuracy, read from the length-probe
    summary rather than copied by hand. Empty when the probe has not been run,
    so the caption degrades to the qualitative statement instead of carrying a
    number nothing in the repo backs."""
    path = SUMMARY_DIR / f"length_probe__{rubric}.json"
    if not path.exists():
        return ""
    with open(path) as f:
        models = json.load(f)["models"]
    accs = {round(m["overall"]["specs"]["length"]["acc"], 4) for m in models.values()}
    if len(accs) != 1:
        # One fit on one item set: differing values would mean the stores no
        # longer share items, which the caller already refuses to compare.
        return ""
    return f" and scores {accs.pop():.3f} on these items"


def render_markdown(rows: dict[str, dict], floors: dict, rubric: str, n_items: int) -> str:
    """The table as markdown, judges in family × scale order, floors below.

    Floors occupy the same columns where they are defined and are left blank
    where they are not: "always-A" has no symmetrized accuracy, because
    symmetrizing a content-blind rule is the rule itself.
    """
    ordered = [k for k in MODEL_STYLES if k in rows]
    header = "| " + " | ".join(name for name, _ in COLUMNS) + " |"
    sep = "|" + "---|" * len(COLUMNS)
    lines = [header, sep]
    for key in ordered:
        flat = flatten(key, MODELS[key].params_b, rows[key])
        lines.append("| " + " | ".join(tpl.format(**flat) for _, tpl in COLUMNS) + " |")

    names = [name for name, _ in COLUMNS]

    def floor_line(label: str, column: str, value: float) -> str:
        """A floor occupies its one meaningful column and leaves the rest empty,
        so the row stays exactly as wide as a judge row."""
        cells = [""] * len(names)
        cells[0] = f"*{label}*"
        cells[names.index(column)] = f"{value:.3f}"
        return "| " + " | ".join(cells) + " |"

    # The two content-blind floors are defined on the raw (per-order) scale;
    # symmetrizing a rule that ignores content returns the rule itself, so
    # neither has a symmetrized column entry. The length heuristic is
    # order-invariant by construction, so its value belongs in both — it is
    # written in the symmetrized column, where judges are compared to it.
    lines.append(floor_line("random floor", "raw acc", RANDOM_ACCURACY))
    lines.append(floor_line("always-A floor", "raw acc", RANDOM_ACCURACY))
    lines.append(floor_line("longer-response floor", "sym acc (95% CI)",
                            floors["longer_chars"]))

    caption = (
        f"Every completed grid for rubric `{rubric}`, same {n_items} stratified "
        f"RewardBench items, both presentation orders. `b` is position-bias "
        f"log-odds toward whatever sits in position A; `s` is the "
        f"order-invariant preference log-odds for the gold-chosen response. "
        f"Raw accuracy assigns each item's presentation order uniformly at "
        f"random; symmetrized accuracy is `sign(s)`. Intervals are 95% paired "
        f"bootstrap over items ({rows[ordered[0]]['n_boot']:,} resamples, "
        f"seed {rows[ordered[0]]['bootstrap_seed']}). The always-A floor sits "
        f"at exactly 0.5 over the exhaustive order pair by construction.\n\n"
        f"`Δ sym−longer` compares each judge against the *fixed* "
        f"pick-the-longer-response rule, which scores {floors['longer_chars']:.3f} "
        f"here — below chance, because RewardBench's composition punishes "
        f"verbosity. Clearing a below-chance floor is a weak test, and this "
        f"column is not the length-baseline verdict: the real opponent is the "
        f"*fitted* one-parameter length model, which is free to learn the "
        f"anti-verbosity direction{fitted_length_note(rubric)}. Only the two 3B "
        f"judges beat that one (findings 13–14, 18, 22)."
    )
    return "\n".join(lines) + "\n\n" + caption + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model keys to include (default: every completed "
                             "store; use this to exclude an in-flight run)")
    args = parser.parse_args()

    fetch()
    items_by_id = {item.item_id: item for item in load_rewardbench()}

    def longer_correct_of(item_id: str) -> float:
        return longer_response_correct(items_by_id[item_id])

    # An in-flight grid is not detectable from incomplete swap pairs alone:
    # run_grid executes both orders of an item consecutively, so a partial
    # store is a *complete* store over fewer items. The reference item set is
    # therefore the widest one observed; anything strictly inside it is a
    # partial run and is dropped with a printed reason, never averaged in.
    covered: dict[str, tuple[list, frozenset[str]]] = {}
    for key in sorted(MODELS):
        if args.models is not None and key not in args.models:
            continue
        path = RESULTS_DIR / f"{key}__{args.rubric}.jsonl"
        if not path.exists():
            continue
        pairs, incomplete = assemble_pairs(load_records([path]))
        if incomplete:
            print(f"[master] {key}: {incomplete} item(s) missing a swap order — "
                  f"excluded")
            continue
        covered[key] = (pairs, frozenset(p.item_id for p in pairs))

    if not covered:
        raise SystemExit(f"no complete stores for rubric {args.rubric}")
    reference = max((ids for _, ids in covered.values()), key=len)

    rows: dict[str, dict] = {}
    for key, (pairs, ids) in covered.items():
        if ids != reference:
            if ids < reference:
                print(f"[master] {key}: {len(ids)}/{len(reference)} items — "
                      f"excluded, grid still in flight")
                continue
            raise SystemExit(
                f"{key} covers items outside the reference set "
                f"({len(ids - reference)} extra); refusing to compare"
            )
        rows[key] = judge_row(pairs, longer_correct_of)
        print(f"[master] {key}: sym {rows[key]['sym_acc']['mean']:.3f}")

    shared = tuple(sorted(reference))
    floors = {
        "longer_chars": sum(longer_correct_of(i) for i in shared) / len(shared),
        "random": RANDOM_ACCURACY,
        "always_a_raw": RANDOM_ACCURACY,
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SUMMARY_DIR / f"master_table__{args.rubric}.json"
    with open(out_json, "w") as f:
        json.dump({"rubric": args.rubric, "n_items": len(shared),
                   "floors": floors, "judges": rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[master] wrote {out_json}")

    markdown = render_markdown(rows, floors, args.rubric, len(shared))
    out_md = SUMMARY_DIR / f"master_table__{args.rubric}.md"
    with open(out_md, "w") as f:
        f.write(markdown)
    print(f"[master] wrote {out_md}")
    print()
    print(markdown)


if __name__ == "__main__":
    main()

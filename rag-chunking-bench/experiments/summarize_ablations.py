"""Render the overlap and budget-rule ablations as a markdown summary.

    python -m experiments.summarize_ablations --dataset dev-v1.1 --retriever bm25

writes ``results/summary_<dataset>_<retriever>_ablations.md`` and prints it.

Three controlled comparisons, each paired per-question (never against a
different question set):

- **Overlap ablation** — each overlap configuration vs. the zero-overlap run
  of the same (chunker, size), stop rule on both sides. Overlap changes the
  ranked chunk list, so hit@k is reported alongside the budget-matched span
  metrics: folklore says overlap helps, and the interesting question is
  whether that survives paying for the duplicated tokens.
- **Cross-family control** — sentence packing at zero overlap vs. fixed
  windows at 25% overlap, matched nominal size: does boundary-aware packing
  collect overlap's boundary-repair benefit for free?
- **Budget rule** — each zero-overlap configuration run under
  truncate-final-chunk vs. the same configuration under stop-before-exceed.
  The ranked list is identical on both sides; only the boundary handling
  differs, so this isolates the protocol artifact documented in the README
  (stop zeroes configs whose chunks exceed the budget).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.aggregate import (
    BASELINE_SIZES,
    STRUCTURAL_CHUNKERS,
    RunResult,
    diff_ci,
    load_raw,
    mean,
)
from experiments.summarize import _table, fmt_diff

ROOT = Path(__file__).resolve().parent.parent


def _key(rr: RunResult) -> tuple[str, int, int]:
    return (rr.config["chunker"], rr.config["chunk_size"], rr.config["overlap"])


def _check_paired(a: RunResult, b: RunResult) -> None:
    if a.qids() != b.qids():
        raise ValueError(f"question sets differ between {a.label} and {b.label}")


def render_overlap_section(
    overlap_runs: list[RunResult], baselines: dict[tuple[str, int], RunResult], budgets: list[int]
) -> list[str]:
    hit_ks = [int(k) for k in overlap_runs[0].config["hit_ks"]]
    hit_k = 5 if 5 in hit_ks else max(hit_ks)
    lines = [
        "## Overlap ablation (stop rule; paired vs. the same config at zero overlap)",
        "",
        "`fixed` overlap is in tokens, `sentence` overlap in sentences. The "
        "budget charges every retrieved chunk its full token count (duplicated "
        "text included) while scoring uses the union of retrieved tokens, so "
        "overlap must earn back its redundancy to break even.",
        "",
        "### Chunk statistics",
        "",
    ]
    stat_rows = []
    for rr in overlap_runs:
        frac = (
            f"{rr.config['overlap'] / rr.config['chunk_size']:.0%}"
            if rr.config["chunker"] == "fixed"
            else f"{rr.config['overlap']} sent"
        )
        stat_rows.append(
            [
                rr.label,
                frac,
                str(rr.chunk_stats["n_chunks"]),
                str(rr.chunk_stats["tokens_median"]),
                f"{rr.chunk_stats['tokens_mean']:.1f}",
            ]
        )
    lines += _table(
        ["config", "overlap", "chunks", "tokens/chunk median", "mean"], stat_rows
    )

    budget_cols = [f"B={b}" for b in budgets]
    lines += ["### SpanRecall@B (mean)", ""]
    lines += _table(
        ["config", *budget_cols],
        [
            [rr.label, *(f"{mean(rr.metric('recall', b)):.3f}" for b in budgets)]
            for rr in overlap_runs
        ],
    )

    lines += [
        "### ΔSpanRecall vs zero overlap (mean [95% CI]; bold = CI excludes 0)",
        "",
    ]
    diff_rows = []
    hit_rows = []
    for rr in overlap_runs:
        chunker, size, _ = _key(rr)
        base = baselines[(chunker, size)]
        _check_paired(rr, base)
        diff_rows.append(
            [
                rr.label,
                *(
                    fmt_diff(diff_ci(rr.metric("recall", b), base.metric("recall", b)))
                    for b in budgets
                ),
            ]
        )
        hit_rows.append(
            [
                rr.label,
                f"{mean(rr.hits(hit_k)):.3f}",
                f"{mean(base.hits(hit_k)):.3f}",
                fmt_diff(diff_ci(rr.hits(hit_k), base.hits(hit_k))),
            ]
        )
    lines += _table(["config", *budget_cols], diff_rows)

    lines += [
        f"### hit@{hit_k} vs zero overlap (fixed-k view of the same runs)",
        "",
    ]
    lines += _table(
        ["config", f"hit@{hit_k}", f"hit@{hit_k} (o0)", f"Δhit@{hit_k} [95% CI]"],
        hit_rows,
    )
    return lines


def render_cross_family_section(
    stop_all: list[RunResult], budgets: list[int]
) -> list[str]:
    """Sentence packing at zero overlap vs. fixed windows at 25% overlap.

    The mechanism claim behind the overlap findings: overlap earns its keep
    by repairing arbitrary window boundaries. If that is the whole story, a
    boundary-aware chunker collects the same benefit without paying for
    duplicated tokens, and the paired delta between sentence-o0 and
    fixed+25% at matched nominal size should sit at or above zero.
    """
    by_key = {_key(rr): rr for rr in stop_all}
    rows = []
    for chunker, size, overlap in sorted(by_key):
        if chunker != "sentence" or overlap != 0:
            continue
        fixed = by_key.get(("fixed", size, size // 4))
        if fixed is None:
            continue
        sent = by_key[(chunker, size, overlap)]
        _check_paired(sent, fixed)
        rows.append(
            [
                f"{sent.label} − {fixed.label}",
                *(
                    fmt_diff(diff_ci(sent.metric("recall", b), fixed.metric("recall", b)))
                    for b in budgets
                ),
            ]
        )
    if not rows:
        return []
    return [
        "## Cross-family control: sentence packing (o0) vs. fixed windows (+25% overlap)",
        "",
        "Positive Δ favors sentence packing. Zero or better means boundary-aware "
        "packing matches or beats the best overlap setting while retrieving no "
        "duplicated text (mean [95% CI]; bold = CI excludes 0).",
        "",
        *_table(["pair", *(f"B={b}" for b in budgets)], rows),
    ]


def render_rule_section(
    trunc_runs: list[RunResult], stop_runs: dict[tuple[str, int, int], RunResult], budgets: list[int]
) -> list[str]:
    budget_cols = [f"B={b}" for b in budgets]
    lines = [
        "## Budget rule: truncate-final-chunk vs. stop-before-exceed",
        "",
        "Identical rankings on both sides; only the handling of the chunk "
        "that straddles the budget differs. Truncation always spends the "
        "full budget when the ranking offers enough text, so the "
        "retrieve-nothing cells of the stop rule become meaningful "
        "measurements here.",
        "",
        "### SpanRecall@B under truncate (mean)",
        "",
    ]
    lines += _table(
        ["config", *budget_cols],
        [
            [rr.label, *(f"{mean(rr.metric('recall', b)):.3f}" for b in budgets)]
            for rr in trunc_runs
        ],
    )
    lines += [
        "### ΔSpanRecall, truncate − stop (mean [95% CI]; bold = CI excludes 0)",
        "",
    ]
    diff_rows = []
    for rr in trunc_runs:
        base = stop_runs[_key(rr)]
        _check_paired(rr, base)
        diff_rows.append(
            [
                base.label,
                *(
                    fmt_diff(diff_ci(rr.metric("recall", b), base.metric("recall", b)))
                    for b in budgets
                ),
            ]
        )
    lines += _table(["config", *budget_cols], diff_rows)

    lines += [
        "### Budget utilization under truncate (mean prompt tokens spent / budget)",
        "",
    ]
    lines += _table(
        ["config", *budget_cols],
        [
            [
                rr.label,
                *(
                    f"{mean([float(t) for t in rr.tokens_used(b)]) / b:.2f}"
                    for b in budgets
                ),
            ]
            for rr in trunc_runs
        ],
    )
    return lines


def render_ablations(
    dataset: str,
    retriever: str,
    raw_dir: Path,
    seed: int = 0,
    sizes: tuple[int, ...] | None = None,
) -> str:
    # This summary ablates the structural baseline grid; the semantic
    # chunker's stop and truncate runs (canonical sizes, from the semantic
    # and matched-realized-size grids) share this raw directory and must
    # stay in their own summaries.
    stop_all = load_raw(
        raw_dir,
        dataset=dataset,
        retriever=retriever,
        budget_rule="stop",
        seed=seed,
        sizes=sizes,
        chunkers=STRUCTURAL_CHUNKERS,
    )
    overlap_runs = [rr for rr in stop_all if rr.config["overlap"] > 0]
    baselines = {
        (rr.config["chunker"], rr.config["chunk_size"]): rr
        for rr in stop_all
        if rr.config["overlap"] == 0
    }
    trunc_runs = load_raw(
        raw_dir,
        dataset=dataset,
        retriever=retriever,
        budget_rule="truncate",
        overlap=0,
        seed=seed,
        sizes=sizes,
        chunkers=STRUCTURAL_CHUNKERS,
    )
    stop_runs = {_key(rr): rr for rr in stop_all if rr.config["overlap"] == 0}
    if not overlap_runs and not trunc_runs:
        raise SystemExit(f"no ablation results for {dataset}/{retriever} in {raw_dir}")

    cfg = (overlap_runs or trunc_runs)[0].config
    budgets = [int(b) for b in cfg["budgets"]]
    n_questions = len((overlap_runs or trunc_runs)[0].records)
    lines = [
        f"# Ablation summary — {dataset}, {retriever}",
        "",
        f"{n_questions} questions ({cfg['per_doc_cap']}/document cap, seed "
        f"{cfg['seed']}), budgets in regex word tokens. Paired comparisons use "
        f"10,000 bootstrap resamples over questions; every ablation run is "
        f"compared against its own control (same chunker, size, and question "
        f"set).",
        "",
        "Generated by `python -m experiments.summarize_ablations` from "
        "`results/raw/` — do not edit by hand.",
        "",
    ]
    if overlap_runs:
        lines += render_overlap_section(overlap_runs, baselines, budgets)
        lines += render_cross_family_section(stop_all, budgets)
    if trunc_runs:
        lines += render_rule_section(trunc_runs, stop_runs, budgets)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize overlap and budget-rule ablation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    text = render_ablations(
        args.dataset, args.retriever, args.raw_dir, seed=args.seed, sizes=BASELINE_SIZES
    )
    out = args.out_dir / f"summary_{args.dataset}_{args.retriever}_ablations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written to {out.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()

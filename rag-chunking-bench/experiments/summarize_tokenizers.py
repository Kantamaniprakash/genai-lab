"""Render the BPE tokenizer robustness check as a markdown summary.

    python -m experiments.summarize_tokenizers --dataset dev-v1.1 --retriever bm25

writes ``results/summary_<dataset>_<retriever>_tokenizers.md`` and prints it.

Every quantity in this benchmark that is denominated in tokens — chunk
sizes, retrieval budgets, and the token sets behind SpanRecall/Precision/IoU
— is counted by one tokenizer. The primary grids use the deterministic regex
word tokenizer; this check reruns the same nominal grid with everything
counted in cl100k_base BPE tokens instead. The question: are the headline
claims properties of chunking, or of the unit they were measured in?

Statistical care: the two grids sample identical question sets (sampling
does not depend on the tokenizer), so scores align mechanically across
units — but a cross-unit paired delta would compare numbers whose budgets
and denominators mean different things. All paired comparisons here are
therefore *within* one unit; what the reader should check is whether each
claim's sign and significance agree *across* the two unit columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.aggregate import (
    BASELINE_SIZES,
    STRUCTURAL_CHUNKERS,
    RunResult,
    check_aligned,
    diff_ci,
    load_raw,
    mean,
)
from experiments.summarize import _table, fmt_diff, pairwise_same_size_rows

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_NAMES = {"regex": "regex word", "cl100k": "cl100k_base BPE"}

# The claims worth defending across units, as ((chunker, size), (chunker,
# size)) pairs. Ordering matters for sign: positive delta = challenger wins.
HEADLINE_PAIRS = (
    (("fixed", 64), ("fixed", 256)),
    (("fixed", 64), ("fixed", 512)),
    (("sentence", 64), ("fixed", 64)),
    (("sentence", 128), ("fixed", 128)),
)


def _load_unit_grids(
    raw_dir: Path,
    dataset: str,
    retriever: str,
    seed: int,
    tokenizers: list[str],
    sizes: tuple[int, ...] | None = None,
    chunkers: tuple[str, ...] | None = None,
) -> dict[str, dict[tuple[str, int], RunResult]]:
    """One baseline grid per token unit, keyed by (chunker, chunk_size).

    Labels differ across units (non-default tokenizers are suffixed), so the
    grid point itself is the join key.
    """
    grids: dict[str, dict[tuple[str, int], RunResult]] = {}
    for tok in tokenizers:
        runs = load_raw(
            raw_dir,
            dataset=dataset,
            retriever=retriever,
            budget_rule="stop",
            overlap=0,
            seed=seed,
            tokenizer=tok,
            sizes=sizes,
            chunkers=chunkers,
        )
        if not runs:
            raise SystemExit(
                f"no baseline-grid results for tokenizer {tok!r} on "
                f"{dataset}/{retriever} in {raw_dir}"
            )
        check_aligned(runs)
        grids[tok] = {
            (rr.config["chunker"], rr.config["chunk_size"]): rr for rr in runs
        }
    points = [tuple(sorted(g)) for g in grids.values()]
    if len(set(points)) != 1:
        raise SystemExit(f"tokenizers cover different grid points: {points}")
    # The units differ but the questions must not, or the check compares
    # different samples rather than different rulers.
    check_aligned([next(iter(g.values())) for g in grids.values()])
    return grids


def _corpus_totals(grid: dict[tuple[str, int], RunResult]) -> dict[tuple[str, int], int]:
    """Total tokens covered per config, from stored chunk statistics.

    At zero overlap every chunker covers each document token exactly once,
    so n_chunks x mean chunk tokens recovers the corpus size in that
    config's unit without re-tokenizing anything.
    """
    return {
        point: round(rr.chunk_stats["n_chunks"] * rr.chunk_stats["tokens_mean"])
        for point, rr in grid.items()
    }


def _point_label(point: tuple[str, int]) -> str:
    return f"{point[0]}-{point[1]}"


def render_tokenizers(
    dataset: str,
    retriever: str,
    seed: int,
    raw_dir: Path,
    tokenizers: list[str] | None = None,
    hit_k: int = 5,
    sizes: tuple[int, ...] | None = None,
    chunkers: tuple[str, ...] | None = None,
) -> str:
    tokenizers = tokenizers or ["regex", "cl100k"]
    grids = _load_unit_grids(
        raw_dir, dataset, retriever, seed, tokenizers, sizes=sizes, chunkers=chunkers
    )
    points = sorted(grids[tokenizers[0]], key=lambda p: (p[0] != "fixed", p))
    any_run = next(iter(grids[tokenizers[0]].values()))
    budgets = [int(b) for b in any_run.config["budgets"]]
    hit_ks = [int(k) for k in any_run.config["hit_ks"]]
    if hit_k not in hit_ks:
        raise SystemExit(f"hit_k {hit_k} not among run hit_ks {hit_ks}")
    n_questions = len(any_run.records)
    unit_names = [TOKENIZER_NAMES.get(t, t) for t in tokenizers]

    lines = [
        f"# Tokenizer robustness — {dataset}, {retriever}",
        "",
        f"The same nominal grid (chunk sizes, budgets B ∈ {budgets}) measured "
        f"in two token units: {', '.join(unit_names)}. Chunk sizes, budgets, "
        "and metric token sets all switch units together, and both grids "
        f"score the identical {n_questions:,}-question sample. Paired "
        "bootstrap deltas (10,000 resamples) are computed within each unit; "
        "a claim is unit-robust when its sign and significance agree across "
        "the unit columns.",
        "",
        "Generated by `python -m experiments.summarize_tokenizers` from "
        "`results/raw/` — do not edit by hand.",
        "",
        "## Unit conversion and realized chunk sizes",
        "",
        "Corpus size per unit is recovered from stored chunk statistics "
        "(zero-overlap chunkers cover every token exactly once). BPE tokens "
        "include leading whitespace and split rarer words, so the same text "
        "costs more of them — nominal size N means less text under BPE than "
        "under word tokens.",
        "",
    ]
    totals = {tok: _corpus_totals(grids[tok]) for tok in tokenizers}
    rows = []
    for point in points:
        cells = [_point_label(point)]
        for tok in tokenizers:
            rr = grids[tok][point]
            cells.append(
                f"{rr.chunk_stats['n_chunks']:,} / {rr.chunk_stats['tokens_mean']:.0f}"
            )
        ratio = totals[tokenizers[1]][point] / totals[tokenizers[0]][point]
        cells.append(f"{ratio:.3f}")
        rows.append(cells)
    lines += _table(
        [
            "config",
            *(f"{TOKENIZER_NAMES.get(t, t)}: chunks / mean tokens" for t in tokenizers),
            f"corpus tokens, {tokenizers[1]}/{tokenizers[0]}",
        ],
        rows,
    )

    lines += [
        "## SpanRecall@B (mean) by unit",
        "",
        "Levels are not comparable across units at the same nominal B (the "
        "budget buys different amounts of text); the within-column *ordering* "
        "across configs is the object under test.",
        "",
    ]
    rows = []
    for point in points:
        cells = [_point_label(point)]
        for b in budgets:
            for tok in tokenizers:
                cells.append(f"{mean(grids[tok][point].metric('recall', b)):.3f}")
        rows.append(cells)
    lines += _table(
        ["config", *(f"B={b} {t}" for b in budgets for t in tokenizers)], rows
    )

    fixed_sizes = sorted(size for chunker, size in points if chunker == "fixed")
    lines += [
        "## Size ordering (fixed family): adjacent-size paired steps",
        "",
        "Positive = the smaller size wins that step. The ordering claim is "
        "unit-robust when no step flips to a *significant* negative in either "
        "unit; non-significant wobbles between near-tied sizes are expected "
        "at generous budgets and should agree across units too.",
        "",
    ]
    rows = []
    for small, large in zip(fixed_sizes, fixed_sizes[1:], strict=False):
        for b in budgets:
            cells = [f"fixed-{small} − fixed-{large}", f"B={b}"]
            for tok in tokenizers:
                cells.append(
                    fmt_diff(
                        diff_ci(
                            grids[tok][("fixed", small)].metric("recall", b),
                            grids[tok][("fixed", large)].metric("recall", b),
                        )
                    )
                )
            rows.append(cells)
    lines += _table(["step", "budget", *unit_names], rows)

    lines += [
        "## Headline paired deltas by unit (bold = 95% CI excludes 0)",
        "",
    ]
    rows = []
    for challenger, baseline in HEADLINE_PAIRS:
        if challenger not in grids[tokenizers[0]] or baseline not in grids[tokenizers[0]]:
            continue
        for b in budgets:
            cells = [f"{_point_label(challenger)} − {_point_label(baseline)}", f"B={b}"]
            for tok in tokenizers:
                cells.append(
                    fmt_diff(
                        diff_ci(
                            grids[tok][challenger].metric("recall", b),
                            grids[tok][baseline].metric("recall", b),
                        )
                    )
                )
            rows.append(cells)
    lines += _table(["comparison", "budget", *unit_names], rows)

    for tok in tokenizers:
        lines += [
            f"## Same-size ΔSpanRecall vs fixed — {TOKENIZER_NAMES.get(tok, tok)} unit",
            "",
        ]
        lines += _table(
            ["comparison", *(f"B={b}" for b in budgets)],
            pairwise_same_size_rows(list(grids[tok].values()), budgets),
        )

    lines += [
        f"## hit@{hit_k} by unit (fixed-k metric, reversal check)",
        "",
    ]
    rows = []
    for point in points:
        cells = [_point_label(point)]
        for tok in tokenizers:
            cells.append(f"{mean(grids[tok][point].hits(hit_k)):.3f}")
        rows.append(cells)
    lines += _table(["config", *unit_names], rows)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the baseline grid across token units.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizers", nargs="+", default=["regex", "cl100k"])
    parser.add_argument("--hit-k", type=int, default=5)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    text = render_tokenizers(
        args.dataset,
        args.retriever,
        args.seed,
        args.raw_dir,
        args.tokenizers,
        args.hit_k,
        sizes=BASELINE_SIZES,
        chunkers=STRUCTURAL_CHUNKERS,
    )
    out = args.out_dir / f"summary_{args.dataset}_{args.retriever}_tokenizers.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written to {out.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()

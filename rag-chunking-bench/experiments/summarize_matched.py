"""Render the matched-realized-size semantic comparison as a markdown summary.

    python -m experiments.summarize_matched --dataset dev-v1.1 --retriever bm25

writes ``results/summary_<dataset>_<retriever>_matched.md`` and prints it.

The question this summary answers: does anything remain of the semantic
chunker's matched-nominal-size deltas once its realized-size drift is
controlled away? Breakpoints only shorten chunks, so semantic-N operates at
a smaller realized mean chunk size than sentence-N (finding 20); this
summary re-pairs each semantic run with the sentence run whose *realized*
mean is closest — the calibrated sizes come from
``experiments.calibrate_matched`` — and reports the same paired deltas under
both pairings side by side. Under the size-drift account every
realized-matched delta collapses to zero; any residual is a genuine
boundary-placement effect.

A dispersion section asks the finer question: even at equal mean recall,
do embedding breakpoints make per-question recall more *consistent*?
Reported as std(semantic) − std(sentence) with a jointly-resampled
bootstrap CI (``metrics.paired_bootstrap_std``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.aggregate import RunResult, check_aligned, diff_ci, load_raw
from experiments.summarize import _table, fmt_diff
from src.metrics import paired_bootstrap_std

ROOT = Path(__file__).resolve().parent.parent

# A realized-size pairing only isolates boundary placement if the sizes
# actually coincide; pairs further apart than this relative gap are rendered
# but flagged, because their delta still mixes in a size effect.
MAX_REL_GAP = 0.05


@dataclass(frozen=True)
class MatchedPair:
    """One semantic run with its nominal-size and realized-size partners."""

    semantic: RunResult
    nominal: RunResult
    matched: RunResult

    @property
    def rel_gap(self) -> float:
        """Relative realized-mean gap of the realized-size pairing."""
        target = self.semantic.chunk_stats["tokens_mean"]
        return abs(self.matched.chunk_stats["tokens_mean"] - target) / target

    @property
    def well_matched(self) -> bool:
        return self.rel_gap <= MAX_REL_GAP


def match_by_realized_size(results: list[RunResult]) -> list[MatchedPair]:
    """Pair every semantic run with sentence runs by nominal and realized size.

    The realized partner is the sentence run whose realized mean chunk size
    is closest to the semantic run's (ties to the smaller nominal size, for
    determinism). Requires a same-nominal sentence partner for every
    semantic run — without it the drift the summary controls for could not
    be shown in the first place.
    """
    semantic = [rr for rr in results if rr.config["chunker"] == "semantic"]
    sentence = [rr for rr in results if rr.config["chunker"] == "sentence"]
    if not semantic:
        raise ValueError("no semantic-chunker runs among the results")
    if not sentence:
        raise ValueError("no sentence-chunker runs among the results")
    by_nominal = {rr.config["chunk_size"]: rr for rr in sentence}
    pairs = []
    for rr in sorted(semantic, key=lambda r: r.config["chunk_size"]):
        nominal_size = rr.config["chunk_size"]
        if nominal_size not in by_nominal:
            raise ValueError(
                f"semantic-{nominal_size} has no sentence run at the same "
                "nominal size; run the baseline grid first"
            )
        target = rr.chunk_stats["tokens_mean"]
        matched = min(
            sentence,
            key=lambda s: (
                abs(s.chunk_stats["tokens_mean"] - target),
                s.config["chunk_size"],
            ),
        )
        pairs.append(
            MatchedPair(semantic=rr, nominal=by_nominal[nominal_size], matched=matched)
        )
    return pairs


def _delta_row(pair: MatchedPair, partner: RunResult, budgets: list[int]) -> list[str]:
    return [
        fmt_diff(
            diff_ci(
                pair.semantic.metric("recall", bud), partner.metric("recall", bud)
            )
        )
        for bud in budgets
    ]


def render_matched(results: list[RunResult]) -> str:
    """Markdown for the matched-realized-size comparison.

    ``results`` must be aligned zero-overlap sentence and semantic runs
    under one budget rule — the canonical grid plus the calibrated sentence
    sizes.
    """
    check_aligned(results)
    pairs = match_by_realized_size(results)
    cfg = results[0].config
    budgets = [int(b) for b in cfg["budgets"]]
    hit_ks = [int(k) for k in cfg["hit_ks"]]
    budget_cols = [f"B={b}" for b in budgets]
    n_questions = len(results[0].records)
    rule = {
        "stop": "stop-before-exceed",
        "truncate": "truncate-final-chunk",
    }.get(cfg["budget_rule"], cfg["budget_rule"])

    lines = [
        f"# Matched-realized-size comparison — {cfg['dataset']}, {cfg['retriever']}"
        + ("" if cfg["budget_rule"] == "stop" else f", {rule} rule"),
        "",
        f"{n_questions} questions, budgets in regex word tokens, budget rule "
        f"{rule}, zero overlap everywhere. Paired comparisons use "
        "10,000 bootstrap resamples over questions; bold = 95% CI excludes 0. "
        "Each semantic run appears twice: against the sentence run at the "
        "same *nominal* size (the finding-20 comparison, which mixes boundary "
        "placement with realized-size drift) and against the sentence run "
        "whose *realized* mean chunk size is closest (calibrated via "
        "`experiments.calibrate_matched`, isolating boundary placement).",
        "",
        "Generated by `python -m experiments.summarize_matched` from "
        "`results/raw/` — do not edit by hand.",
        "",
        "## Pairings (realized mean chunk size, regex tokens)",
        "",
    ]
    rows = []
    for pair in pairs:
        flag = "" if pair.well_matched else " ⚠ gap > 5%"
        rows.append(
            [
                pair.semantic.label,
                f"{pair.semantic.chunk_stats['tokens_mean']:.1f}",
                f"{pair.nominal.label} @ {pair.nominal.chunk_stats['tokens_mean']:.1f}",
                f"{pair.matched.label} @ {pair.matched.chunk_stats['tokens_mean']:.1f}",
                f"{pair.rel_gap:.1%}{flag}",
            ]
        )
    lines += _table(
        [
            "semantic run",
            "realized",
            "nominal partner @ realized",
            "matched partner @ realized",
            "matched gap",
        ],
        rows,
    )

    lines += [
        "## ΔSpanRecall: semantic − sentence at matched *nominal* size",
        "",
        "(the comparison finding 20 diagnosed — drift included)",
        "",
    ]
    lines += _table(
        ["nominal", *budget_cols],
        [
            [str(pair.semantic.config["chunk_size"]), *_delta_row(pair, pair.nominal, budgets)]
            for pair in pairs
        ],
    )

    lines += [
        "## ΔSpanRecall: semantic − sentence at matched *realized* size",
        "",
        "(drift controlled — residuals are boundary-placement effects)",
        "",
    ]
    lines += _table(
        ["nominal", "partner", *budget_cols],
        [
            [
                str(pair.semantic.config["chunk_size"]),
                pair.matched.label,
                *_delta_row(pair, pair.matched, budgets),
            ]
            for pair in pairs
        ],
    )

    lines += [
        "## Δhit@k: semantic − sentence at matched *realized* size",
        "",
    ]
    lines += _table(
        ["nominal", "partner", *(f"k={k}" for k in hit_ks)],
        [
            [
                str(pair.semantic.config["chunk_size"]),
                pair.matched.label,
                *(
                    fmt_diff(diff_ci(pair.semantic.hits(k), pair.matched.hits(k)))
                    for k in hit_ks
                ),
            ]
            for pair in pairs
        ],
    )

    lines += [
        "## Per-question dispersion at matched *realized* size",
        "",
        "std(semantic) − std(sentence) of per-question SpanRecall, jointly "
        "resampled 95% CI. Negative and significant would mean breakpoints "
        "make retrieval more consistent question-to-question even where the "
        "mean is unchanged.",
        "",
    ]
    rows = []
    for pair in pairs:
        cells = []
        for bud in budgets:
            res = paired_bootstrap_std(
                pair.semantic.metric("recall", bud), pair.matched.metric("recall", bud)
            )
            cells.append(fmt_diff(res))
        rows.append(
            [str(pair.semantic.config["chunk_size"]), pair.matched.label, *cells]
        )
    lines += _table(["nominal", "partner", *budget_cols], rows)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the matched-realized-size semantic comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--budget-rule",
        default="stop",
        choices=("stop", "truncate"),
        help="stop shows the raw protocol; truncate removes the budget-"
        "boundary artifact, which realized-mean matching alone cannot "
        "(the realized-size *distributions* still differ)",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Deliberately no size filter: the calibrated off-grid sentence sizes are
    # exactly what this summary exists to use.
    results = [
        rr
        for rr in load_raw(
            args.raw_dir,
            dataset=args.dataset,
            retriever=args.retriever,
            budget_rule=args.budget_rule,
            overlap=0,
            seed=args.seed,
        )
        if rr.config["chunker"] in ("sentence", "semantic")
    ]
    if not results:
        raise SystemExit(f"no results for {args.dataset}/{args.retriever} in {args.raw_dir}")
    text = render_matched(results)
    suffix = "" if args.budget_rule == "stop" else f"_{args.budget_rule}"
    out = args.out_dir / f"summary_{args.dataset}_{args.retriever}_matched{suffix}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written to {out.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()

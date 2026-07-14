"""Per-question error analysis on the Chroma corpora: where the deltas live.

    python -m experiments.summarize_errors --retriever bm25

writes ``results/summary_chroma_<retriever>_errors.md`` and prints it.

The moderation summary (``experiments.summarize_chroma``) shows *that* the
size crossover is moderated by gold length and varies by corpus. This module
asks *where* the pooled deltas actually come from, question by question:

- **Corpus × gold-length interaction + composition test** — the headline
  size comparison split simultaneously by corpus and by (global) gold-length
  tercile, then, per corpus, its observed delta against the delta predicted
  from its gold-length composition alone. Tercile means for the prediction
  are estimated leave-one-corpus-out, and the residual gets a stratified
  bootstrap CI (corpus sample and each leave-out stratum resampled
  independently, mirroring how the point estimates are formed). A
  significant residual means corpus identity carries an effect beyond its
  gold-length mix.
- **Anatomy of the hard losses** — the per-question deltas at the analysis
  budget, split into complete misses (the challenger retrieved zero gold
  despite spending its whole budget — a ranking failure) vs partial-coverage
  losses (the challenger found the region but covered less of it), with the
  question characteristics of each stratum and the worst single questions.
- **Overlap decomposition by control state** — each overlap configuration's
  gain over its zero-overlap control decomposed by what the control did on
  that question: control recall 0 (overlap surfaced a region the control
  never retrieved — a placement effect), partial control recall (overlap
  extended coverage of a found region), and control recall 1 (a perfect
  control, where duplicated tokens can only crowd useful ones out of the
  budget — the redundancy tax). The three contributions sum exactly to the
  total delta, so this attributes every point of it to a mechanism.
- **Overlap gains by gold tercile and reference count** — the moderation
  view of the same pairs: the direct test of whether overlap's gains sit on
  the long or multi-reference golds the evidence-stitching account predicts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.aggregate import (
    BASELINE_SIZES,
    RunResult,
    check_aligned,
    load_raw,
)
from experiments.summarize import _table, find_baseline, fmt_diff
from experiments.summarize_chroma import (
    _subset_diff_rows,
    gold_stats,
    gold_terciles,
)
from src.data import CHROMA_CORPORA
from src.metrics import BootstrapResult, paired_bootstrap

ROOT = Path(__file__).resolve().parent.parent

# One pair per regime finding 16 flagged: the persistent small-window cell
# (fixed-64 at 50%), the canonical 25% at mid and large windows, and the
# sentence-packing cell that flipped sign relative to SQuAD.
DEFAULT_OVERLAP_PAIRS = "fixed-64/o32,fixed-128/o32,fixed-256/o64,sentence-128/o2"

# Control-state strata of the overlap decomposition, in rendering order.
DECOMP_STRATA = (
    ("new region (ctrl = 0)", lambda c: c == 0.0),
    ("extension (0 < ctrl < 1)", lambda c: (c > 0.0) & (c < 1.0)),
    ("redundancy tax (ctrl = 1)", lambda c: c == 1.0),
)


def corpus_indices(qids: tuple[str, ...]) -> dict[str, list[int]]:
    """Record indices per corpus (qid prefix), pinned corpora first."""
    per: dict[str, list[int]] = {}
    for i, qid in enumerate(qids):
        per.setdefault(qid.split(":")[0], []).append(i)
    ordered = [c for c in CHROMA_CORPORA if c in per] + sorted(set(per) - set(CHROMA_CORPORA))
    return {c: per[c] for c in ordered}


def composition_residual(
    deltas: np.ndarray,
    corpus: list[int],
    terciles: list[list[int]],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, BootstrapResult] | None:
    """Observed corpus delta, composition-predicted delta, and residual CI.

    The prediction reweights leave-one-corpus-out tercile means by the
    corpus's own tercile composition — what this corpus's delta would be if
    its gold-length mix were the only thing distinguishing its questions
    from the rest of the dataset. Returns ``None`` when some tercile the
    corpus occupies has no outside questions to estimate from (possible only
    at fixture scale; the real dataset always has all five corpora).
    """
    inside = set(corpus)
    weights: list[float] = []
    strata: list[np.ndarray] = []
    for indices in terciles:
        own = sum(1 for i in indices if i in inside)
        if own == 0:
            continue  # zero weight, contributes nothing to the prediction
        outside = [i for i in indices if i not in inside]
        if not outside:
            return None
        weights.append(own / len(corpus))
        strata.append(np.asarray(outside))
    own_deltas = deltas[np.asarray(corpus)]
    observed = float(own_deltas.mean())
    predicted = float(sum(w * deltas[s].mean() for w, s in zip(weights, strata, strict=True)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(own_deltas), size=(n_resamples, len(own_deltas)))
    boots = own_deltas[idx].mean(axis=1)
    for w, s in zip(weights, strata, strict=True):
        stratum = deltas[s]
        idx = rng.integers(0, len(stratum), size=(n_resamples, len(stratum)))
        boots = boots - w * stratum[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return observed, predicted, BootstrapResult(
        mean_diff=observed - predicted, ci_low=float(lo), ci_high=float(hi)
    )


def loss_strata(
    deltas: np.ndarray, challenger_recall: np.ndarray, threshold: float
) -> list[tuple[str, list[int]]]:
    """Partition question indices by loss/win severity and miss type.

    ``complete miss`` requires the challenger to have retrieved *no* gold at
    all — with a generous budget that is a ranking failure (the right region
    never surfaced), a different mechanism from covering a found region
    only partially.
    """
    loss = deltas <= -threshold
    win = deltas >= threshold
    return [
        (
            f"complete miss (Δ ≤ −{threshold:g}, challenger recall 0)",
            list(np.flatnonzero(loss & (challenger_recall == 0.0))),
        ),
        (
            f"partial-coverage loss (Δ ≤ −{threshold:g}, challenger recall > 0)",
            list(np.flatnonzero(loss & (challenger_recall > 0.0))),
        ),
        (f"within ±{threshold:g}", list(np.flatnonzero(~loss & ~win))),
        (f"win (Δ ≥ +{threshold:g})", list(np.flatnonzero(win))),
    ]


def _characteristics_row(
    label: str,
    indices: list[int],
    qids: tuple[str, ...],
    stats: dict[str, tuple[int, int]],
    hits_a: np.ndarray,
    hits_b: np.ndarray,
) -> list[str]:
    if not indices:
        return [label, "0", "—", "—", "—", "—", "—"]
    lens = sorted(stats[qids[i]][0] for i in indices)
    multi = sum(1 for i in indices if stats[qids[i]][1] >= 2)
    per: dict[str, int] = {}
    for i in indices:
        per[qids[i].split(":")[0]] = per.get(qids[i].split(":")[0], 0) + 1
    corpora = ", ".join(f"{c} {n}" for c, n in sorted(per.items(), key=lambda kv: -kv[1]))
    return [
        label,
        str(len(indices)),
        str(lens[len(lens) // 2]),
        f"{multi / len(indices):.2f}",
        f"{hits_a[indices].mean():.2f}",
        f"{hits_b[indices].mean():.2f}",
        corpora,
    ]


def overlap_pairs(
    runs: list[RunResult], labels: list[str]
) -> list[tuple[RunResult, RunResult]]:
    """Resolve each requested overlap label to (overlap run, zero-overlap control)."""
    by_label = {rr.label: rr for rr in runs}
    pairs = []
    for label in labels:
        if label not in by_label:
            raise SystemExit(f"overlap run {label!r} not on disk")
        rr = by_label[label]
        control_label = f"{rr.config['chunker']}-{rr.config['chunk_size']}"
        if control_label not in by_label:
            raise SystemExit(f"zero-overlap control {control_label!r} not on disk")
        pairs.append((rr, by_label[control_label]))
    return pairs


def decomposition_rows(
    pairs: list[tuple[RunResult, RunResult]], budgets: list[int]
) -> list[list[str]]:
    """One row per (pair, budget): total delta and its three exact parts.

    A stratum's contribution is the mean over *all* questions of the masked
    per-question delta (zero outside the stratum), so the three contributions
    sum to the total by construction; each is bootstrapped over the full
    question set, keeping the (data-dependent) stratum membership attached to
    the question it describes.
    """
    rows = []
    for run, control in pairs:
        for budget in budgets:
            d = np.asarray(run.metric("recall", budget)) - np.asarray(
                control.metric("recall", budget)
            )
            ctrl = np.asarray(control.metric("recall", budget))
            zeros = [0.0] * len(d)
            row = [
                f"{run.label} vs {control.label}",
                str(budget),
                fmt_diff(paired_bootstrap(list(d), zeros)),
            ]
            for _, predicate in DECOMP_STRATA:
                mask = predicate(ctrl)
                row.append(
                    f"{fmt_diff(paired_bootstrap(list(d * mask), zeros))} (n={int(mask.sum())})"
                )
            rows.append(row)
    return rows


def render_errors(
    results: list[RunResult],
    overlap_runs: list[RunResult],
    stats: dict[str, tuple[int, int]],
    challenger_label: str,
    baseline_label: str,
    pair_labels: list[str],
    analysis_budget: int = 1600,
    threshold: float = 0.25,
) -> str:
    check_aligned(results + overlap_runs)
    challenger = find_baseline(results, challenger_label)
    baseline = find_baseline(results, baseline_label)
    cfg = results[0].config
    budgets = [int(b) for b in cfg["budgets"]]
    if analysis_budget not in budgets:
        raise SystemExit(f"analysis budget {analysis_budget} not in grid budgets {budgets}")
    budget_cols = [f"B={b}" for b in budgets]
    qids = challenger.qids()
    missing = [qid for qid in qids if qid not in stats]
    if missing:
        raise SystemExit(
            f"{len(missing)} run questions absent from the loaded dataset "
            f"(first: {missing[0]}); results and data/ are out of sync"
        )

    per_corpus = corpus_indices(qids)
    terciles = gold_terciles(qids, stats)
    tercile_only = [indices for _, indices in terciles]
    deltas = {
        b: np.asarray(challenger.metric("recall", b))
        - np.asarray(baseline.metric("recall", b))
        for b in budgets
    }

    lines = [
        f"# Chroma error analysis — where the deltas live ({cfg['retriever']})",
        "",
        f"{len(qids)} questions, stop rule, seed {cfg['seed']}. Size comparison: "
        f"**{challenger_label} − {baseline_label}** (paired ΔSpanRecall, 10,000 "
        "bootstrap resamples, bold = 95% CI excludes 0). Gold-length terciles are "
        "global, so cells are comparable across corpora.",
        "",
        "Generated by `python -m experiments.summarize_errors` from "
        "`results/raw/` and `data/chroma/` — do not edit by hand.",
        "",
        "## Corpus × gold-length composition",
        "",
    ]
    comp_rows = []
    for corpus, indices in per_corpus.items():
        inside = set(indices)
        cells = [str(sum(1 for i in t if i in inside)) for t in tercile_only]
        comp_rows.append([corpus, *cells, str(len(indices))])
    lines += _table(["corpus", *(label for label, _ in terciles), "total"], comp_rows)

    for label, tercile in terciles:
        lines += [f"## Per-corpus ΔSpanRecall within {label}", ""]
        tset = set(tercile)
        groups = [
            (corpus, [i for i in indices if i in tset])
            for corpus, indices in per_corpus.items()
        ]
        lines += _table(
            ["corpus", *budget_cols],
            _subset_diff_rows(challenger, baseline, budgets, groups),
        )

    lines += [
        f"## Composition test at B={analysis_budget}: is corpus identity more "
        "than its gold-length mix?",
        "",
        "Predicted = leave-one-corpus-out tercile means reweighted by the "
        "corpus's own tercile composition. Residual CI: stratified bootstrap "
        "(10,000 resamples). A significant residual would mean the corpus "
        "behaves differently from composition-matched questions elsewhere.",
        "",
    ]
    comp_test_rows = []
    for corpus, indices in per_corpus.items():
        est = composition_residual(deltas[analysis_budget], indices, tercile_only)
        if est is None:
            comp_test_rows.append([f"{corpus} (n={len(indices)})", "—", "—", "—"])
            continue
        observed, predicted, residual = est
        comp_test_rows.append(
            [
                f"{corpus} (n={len(indices)})",
                f"{observed:+.3f}",
                f"{predicted:+.3f}",
                fmt_diff(residual),
            ]
        )
    lines += _table(["corpus", "observed Δ", "predicted Δ", "residual [95% CI]"], comp_test_rows)

    hit_ks = [int(k) for k in cfg["hit_ks"]]
    hit_k = 5 if 5 in hit_ks else max(hit_ks)
    lines += [
        f"## Anatomy of the deltas at B={analysis_budget}",
        "",
        f"hit@{hit_k} is the ranking view of the same questions: a stratum "
        f"where the challenger's hit@{hit_k} is 0 never surfaced any gold "
        f"region in its top {hit_k} chunks — those are ranking failures, not "
        "coverage failures.",
        "",
    ]
    d = deltas[analysis_budget]
    recall_a = np.asarray(challenger.metric("recall", analysis_budget))
    recall_b = np.asarray(baseline.metric("recall", analysis_budget))
    hits_a = np.asarray(challenger.hits(hit_k))
    hits_b = np.asarray(baseline.hits(hit_k))
    strata = loss_strata(d, recall_a, threshold)
    lines += _table(
        [
            "stratum",
            "n",
            "gold tokens median",
            "share multi-ref",
            f"{challenger_label} hit@{hit_k}",
            f"{baseline_label} hit@{hit_k}",
            "corpora",
        ],
        [
            _characteristics_row(label, indices, qids, stats, hits_a, hits_b)
            for label, indices in strata
        ],
    )

    lines += ["### Worst questions (most negative Δ)", ""]
    worst = np.argsort(d, kind="stable")[:8]
    lines += _table(
        [
            "qid",
            "Δ",
            "gold tokens",
            "refs",
            f"{challenger_label} recall",
            f"{baseline_label} recall",
        ],
        [
            [
                qids[i],
                f"{d[i]:+.3f}",
                str(stats[qids[i]][0]),
                str(stats[qids[i]][1]),
                f"{recall_a[i]:.3f}",
                f"{recall_b[i]:.3f}",
            ]
            for i in worst
        ],
    )

    pairs = overlap_pairs(overlap_runs, pair_labels)
    lines += [
        "## Overlap gains decomposed by control state",
        "",
        "Each overlap config vs its zero-overlap control; contributions are "
        "means of the per-question delta masked to the stratum, so the three "
        "parts sum exactly to the total. Membership uses the control's recall "
        "at the same budget.",
        "",
    ]
    lines += _table(
        ["pair", "B", "total Δ", *(label for label, _ in DECOMP_STRATA)],
        decomposition_rows(pairs, budgets),
    )

    refs_groups = [
        ("1 reference", [i for i, qid in enumerate(qids) if stats[qid][1] == 1]),
        ("2+ references", [i for i, qid in enumerate(qids) if stats[qid][1] >= 2]),
    ]
    lines += [
        "## Overlap gains by gold tercile and reference count",
        "",
        "The moderation view of the same pairs — the direct test of the "
        "evidence-stitching account (which predicts gains concentrated on "
        "long-gold and multi-reference questions).",
        "",
    ]
    for run, control in pairs:
        lines += [f"### {run.label} vs {control.label}", ""]
        lines += _table(
            ["questions", *budget_cols],
            _subset_diff_rows(run, control, budgets, terciles + refs_groups),
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-question error analysis on the Chroma corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--challenger", default="fixed-64")
    parser.add_argument("--baseline", default="fixed-256")
    parser.add_argument("--analysis-budget", type=int, default=1600)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--overlap-pairs", default=DEFAULT_OVERLAP_PAIRS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not (args.data_dir / "chroma" / "questions_df.csv").exists():
        raise SystemExit(
            "data/chroma not found — run `python -m src.data` first (gold "
            "lengths are recomputed from the corpus text)"
        )
    results = load_raw(
        args.raw_dir,
        dataset="chroma",
        retriever=args.retriever,
        budget_rule="stop",
        overlap=0,
        seed=args.seed,
        sizes=BASELINE_SIZES,
    )
    if not results:
        raise SystemExit(f"no results for chroma/{args.retriever} in {args.raw_dir}")
    overlap_runs = load_raw(
        args.raw_dir,
        dataset="chroma",
        retriever=args.retriever,
        budget_rule="stop",
        seed=args.seed,
        sizes=BASELINE_SIZES,
    )
    text = render_errors(
        results,
        overlap_runs,
        gold_stats(args.data_dir),
        args.challenger,
        args.baseline,
        [label.strip() for label in args.overlap_pairs.split(",") if label.strip()],
        analysis_budget=args.analysis_budget,
        threshold=args.threshold,
    )
    out = args.out_dir / f"summary_chroma_{args.retriever}_errors.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written to {out.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()

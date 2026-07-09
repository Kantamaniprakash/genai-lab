"""Render the Chroma-corpora analyses: per-corpus effects and gold-length moderation.

    python -m experiments.summarize_chroma --retriever bm25

writes ``results/summary_chroma_<retriever>_moderation.md`` and prints it.

The pooled summary (``experiments.summarize``) hides two things the Chroma
dataset exists to show. First, its five corpora are heterogeneous (chat logs
to SEC filings), so any effect could be riding on one of them — the
per-corpus table splits the headline size comparison by corpus, paired
within corpus. Second, Chroma gold references are sentence-scale and often
multi-span, unlike SQuAD's ~3-token answers — the moderation tables split
the same comparison by the question's total gold-evidence length and by its
reference count, which is the direct test of whether optimal chunk size
tracks the length of the evidence being retrieved. A drop-one-corpus
jackknife of the pooled delta sits between the two: it is the corpus-level
analogue of the SQuAD multi-seed check (chroma runs every question, so
there is no sampling seed to vary).

Gold lengths are recomputed from the corpus text, so this summarizer needs
``data/chroma`` downloaded (``python -m src.data``); it refuses to run
otherwise rather than silently skipping the moderation sections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.aggregate import RunResult, check_aligned, diff_ci, load_raw
from experiments.summarize import _table, find_baseline, fmt_diff
from src.data import CHROMA_CORPORA, load_chroma
from src.tokenization import RegexWordTokenizer, TokenIndex

ROOT = Path(__file__).resolve().parent.parent


def gold_stats(data_dir: Path) -> dict[str, tuple[int, int]]:
    """Per question: (total gold tokens, reference count), from the corpus text.

    Token counts use the same tokenizer and overlap convention as scoring
    (``tokens_overlapping``), so a question's gold length here is exactly the
    denominator its SpanRecall was computed against.
    """
    dataset = load_chroma(data_dir)
    indexes = {
        doc_id: TokenIndex(doc.text, RegexWordTokenizer())
        for doc_id, doc in dataset.documents.items()
    }
    stats: dict[str, tuple[int, int]] = {}
    for q in dataset.questions:
        spans = q.gold_alternatives[0]
        tokens: set[int] = set()
        for span in spans:
            tokens.update(indexes[q.doc_id].tokens_overlapping(span.start, span.end))
        stats[q.qid] = (len(tokens), len(spans))
    return stats


def gold_terciles(
    qids: tuple[str, ...], stats: dict[str, tuple[int, int]]
) -> list[tuple[str, list[int]]]:
    """Split record indices into terciles of total gold-evidence length.

    Shared by the moderation tables and the crossover figure so both slice
    the questions identically.
    """
    lengths = sorted(stats[qid][0] for qid in qids)
    t1 = lengths[len(lengths) // 3]
    t2 = lengths[2 * len(lengths) // 3]
    return [
        (
            f"gold ≤ {t1} tokens",
            [i for i, qid in enumerate(qids) if stats[qid][0] <= t1],
        ),
        (
            f"gold {t1 + 1}–{t2} tokens",
            [i for i, qid in enumerate(qids) if t1 < stats[qid][0] <= t2],
        ),
        (
            f"gold > {t2} tokens",
            [i for i, qid in enumerate(qids) if stats[qid][0] > t2],
        ),
    ]


def _subset_diff_rows(
    challenger: RunResult,
    baseline: RunResult,
    budgets: list[int],
    groups: list[tuple[str, list[int]]],
) -> list[list[str]]:
    """Paired ΔSpanRecall rows, one per (group label, record indices) subset."""
    rows = []
    for label, indices in groups:
        row = [f"{label} (n={len(indices)})"]
        if not indices:
            # Degenerate split (tiny fixtures; can't happen with the real
            # 472-question dataset) — keep the row, show no estimate.
            rows.append(row + ["—"] * len(budgets))
            continue
        for budget in budgets:
            row.append(
                fmt_diff(
                    diff_ci(
                        [challenger.records[i]["budgets"][str(budget)]["recall"] for i in indices],
                        [baseline.records[i]["budgets"][str(budget)]["recall"] for i in indices],
                    )
                )
            )
        rows.append(row)
    return rows


def render_moderation(
    results: list[RunResult],
    stats: dict[str, tuple[int, int]],
    challenger_label: str,
    baseline_label: str,
) -> str:
    check_aligned(results)
    challenger = find_baseline(results, challenger_label)
    baseline = find_baseline(results, baseline_label)
    cfg = results[0].config
    budgets = [int(b) for b in cfg["budgets"]]
    budget_cols = [f"B={b}" for b in budgets]
    qids = challenger.qids()
    missing = [qid for qid in qids if qid not in stats]
    if missing:
        raise SystemExit(
            f"{len(missing)} run questions absent from the loaded dataset "
            f"(first: {missing[0]}); results and data/ are out of sync"
        )

    per_corpus: dict[str, list[int]] = {}
    for i, qid in enumerate(qids):
        per_corpus.setdefault(qid.split(":")[0], []).append(i)
    # Real runs cover exactly the five pinned corpora; keep their canonical
    # order and append anything else (synthetic fixtures) alphabetically.
    corpora = [c for c in CHROMA_CORPORA if c in per_corpus] + sorted(
        set(per_corpus) - set(CHROMA_CORPORA)
    )

    lines = [
        f"# Chroma corpora — per-corpus and gold-length views ({cfg['retriever']})",
        "",
        f"{len(qids)} questions across {len(corpora)} corpora (no "
        f"sampling: the per-document cap of {cfg['per_doc_cap']} exceeds every "
        f"corpus's question count). All cells are paired ΔSpanRecall, "
        f"**{challenger_label} − {baseline_label}**, 10,000 bootstrap "
        "resamples, bold = 95% CI excludes 0.",
        "",
        "Generated by `python -m experiments.summarize_chroma` from "
        "`results/raw/` and `data/chroma/` — do not edit by hand.",
        "",
        "## Gold evidence by corpus",
        "",
    ]
    rows = []
    for corpus in corpora:
        lens = sorted(stats[qids[i]][0] for i in per_corpus[corpus])
        multi = sum(1 for i in per_corpus[corpus] if stats[qids[i]][1] > 1)
        rows.append(
            [
                corpus,
                str(len(lens)),
                str(lens[len(lens) // 2]),
                str(lens[-1]),
                f"{multi / len(lens):.2f}",
            ]
        )
    lines += _table(
        ["corpus", "questions", "gold tokens median", "max", "share multi-ref"], rows
    )

    lines += [f"## Per-corpus paired ΔSpanRecall ({challenger_label} − {baseline_label})", ""]
    lines += _table(
        ["corpus", *budget_cols],
        _subset_diff_rows(
            challenger,
            baseline,
            budgets,
            [(corpus, per_corpus[corpus]) for corpus in corpora],
        ),
    )

    # The chroma dataset has no question sampling to vary (every question of
    # every corpus runs), so the stability check analogous to the SQuAD
    # multi-seed grids is corpus-level: recompute the pooled delta with each
    # corpus removed. A finding that flips sign or loses significance under
    # some drop-one estimate is riding on that corpus.
    all_indices = list(range(len(qids)))
    lines += [
        "## Corpus jackknife: pooled ΔSpanRecall with each corpus dropped",
        "",
    ]
    lines += _table(
        ["questions used", *budget_cols],
        _subset_diff_rows(
            challenger,
            baseline,
            budgets,
            [("all corpora", all_indices)]
            + [
                (
                    f"without {corpus}",
                    [i for i in all_indices if i not in set(per_corpus[corpus])],
                )
                for corpus in corpora
            ],
        ),
    )

    lines += [
        "## Moderation by total gold-evidence length (terciles over questions)",
        "",
    ]
    lines += _table(
        ["gold-length tercile", *budget_cols],
        _subset_diff_rows(challenger, baseline, budgets, gold_terciles(qids, stats)),
    )

    ref_groups = [
        ("1 reference", [i for i, qid in enumerate(qids) if stats[qid][1] == 1]),
        ("2+ references", [i for i, qid in enumerate(qids) if stats[qid][1] >= 2]),
    ]
    lines += ["## Moderation by reference count", ""]
    lines += _table(
        ["references per question", *budget_cols],
        _subset_diff_rows(challenger, baseline, budgets, ref_groups),
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-corpus and gold-length moderation on Chroma.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--challenger", default="fixed-64")
    parser.add_argument("--baseline", default="fixed-256")
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
    )
    if not results:
        raise SystemExit(f"no results for chroma/{args.retriever} in {args.raw_dir}")
    text = render_moderation(
        results, gold_stats(args.data_dir), args.challenger, args.baseline
    )
    out = args.out_dir / f"summary_chroma_{args.retriever}_moderation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written to {out.relative_to(ROOT)}]")


if __name__ == "__main__":
    main()

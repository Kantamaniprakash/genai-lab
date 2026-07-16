"""Reproduction audit: regenerate every committed summary table and figure
from ``results/raw/`` and verify them byte-for-byte against the committed
files.

    python -m experiments.reproduce                 # audit; writes nothing
    python -m experiments.reproduce --tables-only   # skip figure rendering
    python -m experiments.reproduce --write         # refresh committed files

The manifest below is the authoritative map from committed artifact to the
invocation that produced it: each step replays one ``python -m`` command
with ``--out-dir`` redirected to a temporary directory inside the repo,
then every generated file is byte-compared against its committed
counterpart. A passing audit proves the committed tables and figures follow
from the committed raw results alone — nothing hand-entered, nothing stale.

Prerequisite: ``python -m src.data`` first — the moderation and
error-analysis summaries and the gold-length figures recompute
gold-evidence lengths from the corpus text; everything else needs only
``results/raw/``.

What byte-identity means here: the tables are deterministic end-to-end
(fixed bootstrap seeds, deterministic retrievers), so they must match on
any machine. The figures are deterministic given the pinned matplotlib,
but PNG bytes also depend on the font stack below it; on a machine with
different fonts, figure drift means "re-render and compare visually", not
"the numbers changed". ``--tables-only`` exists for exactly that case.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Step:
    """One generator invocation and the repo-relative files it writes."""

    module: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]


SUMMARY_STEPS: tuple[Step, ...] = (
    # Baseline grids: one summary per (dataset, retriever).
    *(
        Step(
            "experiments.summarize",
            ("--dataset", dataset, "--retriever", retriever),
            (f"results/summary_{dataset}_{retriever}.md",),
        )
        for dataset in ("dev-v1.1", "chroma")
        for retriever in ("bm25", "tfidf", "lsa", "dense")
    ),
    # Overlap ablation + truncate-rule check (findings 6-7, 16-17).
    *(
        Step(
            "experiments.summarize_ablations",
            ("--dataset", dataset),
            (f"results/summary_{dataset}_bm25_ablations.md",),
        )
        for dataset in ("dev-v1.1", "chroma")
    ),
    # Cross-retriever comparison (findings 8-9).
    *(
        Step(
            "experiments.summarize_retrievers",
            ("--dataset", dataset),
            (f"results/summary_{dataset}_retrievers.md",),
        )
        for dataset in ("dev-v1.1", "chroma")
    ),
    # Multi-seed robustness (finding 10).
    Step(
        "experiments.summarize_seeds",
        (),
        ("results/summary_dev-v1.1_bm25_seeds.md",),
    ),
    # Gold-length moderation on the Chroma corpora (findings 13-15).
    Step(
        "experiments.summarize_chroma",
        (),
        ("results/summary_chroma_bm25_moderation.md",),
    ),
    # BPE tokenizer unit (finding 19).
    Step(
        "experiments.summarize_tokenizers",
        (),
        ("results/summary_dev-v1.1_bm25_tokenizers.md",),
    ),
    # Semantic chunker at matched nominal size (findings 20-21).
    *(
        Step(
            "experiments.summarize_semantic",
            ("--dataset", dataset),
            (f"results/summary_{dataset}_bm25_semantic.md",),
        )
        for dataset in ("dev-v1.1", "chroma")
    ),
    # Matched realized size under both budget rules (findings 22-23).
    *(
        Step(
            "experiments.summarize_matched",
            ("--dataset", dataset, "--budget-rule", rule),
            (
                "results/summary_{}_bm25_matched{}.md".format(
                    dataset, "" if rule == "stop" else "_truncate"
                ),
            ),
        )
        for dataset in ("dev-v1.1", "chroma")
        for rule in ("stop", "truncate")
    ),
    # Per-question error analysis (findings 24-26).
    Step(
        "experiments.summarize_errors",
        (),
        ("results/summary_chroma_bm25_errors.md",),
    ),
)

# make_figures renders the per-dataset figures for --dataset plus every
# cross-dataset figure whose inputs are on disk, so the cross-dataset
# figures appear in both steps; they are deterministic, so the second
# render must byte-match the first (the audit would flag it otherwise).
CROSS_DATASET_FIGURES = (
    "results/figures/gold_length_crossover.png",
    "results/figures/error_analysis_chroma_bm25.png",
    "results/figures/matched_realized_bm25.png",
)

FIGURE_STEPS: tuple[Step, ...] = (
    Step(
        "experiments.make_figures",
        ("--dataset", "dev-v1.1"),
        (
            "results/figures/recall_budget_curves_dev-v1.1_bm25.png",
            "results/figures/metric_reversal_dev-v1.1_bm25.png",
            "results/figures/overlap_ablation_dev-v1.1_bm25.png",
            "results/figures/budget_rule_dev-v1.1_bm25.png",
            "results/figures/tokenizer_robustness_dev-v1.1_bm25.png",
            "results/figures/semantic_comparison_dev-v1.1_bm25.png",
            "results/figures/retriever_comparison_dev-v1.1.png",
            "results/figures/dense_window_dev-v1.1.png",
            *CROSS_DATASET_FIGURES,
        ),
    ),
    Step(
        "experiments.make_figures",
        ("--dataset", "chroma"),
        (
            "results/figures/recall_budget_curves_chroma_bm25.png",
            "results/figures/metric_reversal_chroma_bm25.png",
            "results/figures/overlap_ablation_chroma_bm25.png",
            "results/figures/budget_rule_chroma_bm25.png",
            "results/figures/semantic_comparison_chroma_bm25.png",
            "results/figures/retriever_comparison_chroma.png",
            "results/figures/dense_window_chroma.png",
            *CROSS_DATASET_FIGURES,
        ),
    ),
    Step(
        "experiments.make_hero_figure",
        (),
        ("assets/hero_spanrecall_dev-v1.1_bm25.png",),
    ),
)

MANIFEST: tuple[Step, ...] = SUMMARY_STEPS + FIGURE_STEPS


def run_step(step: Step, out_dir: Path) -> None:
    """Replay one generator with its output redirected to ``out_dir``."""
    module = importlib.import_module(step.module)
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            module.main([*step.argv, "--out-dir", str(out_dir)])
    except Exception:
        print(captured.getvalue())
        raise


def audit(steps: tuple[Step, ...]) -> int:
    """Regenerate into a temp dir, byte-compare, report; 0 iff clean."""
    expected = {Path(rel).name: rel for step in steps for rel in step.outputs}
    failures = []
    # The generators print paths relative to ROOT, so the scratch directory
    # must live inside the repo.
    with tempfile.TemporaryDirectory(prefix=".reproduce-", dir=ROOT) as tmp:
        out_dir = Path(tmp)
        for i, step in enumerate(steps, 1):
            print(f"[{i:2}/{len(steps)}] {step.module} {' '.join(step.argv)}")
            run_step(step, out_dir)
        generated = {path.name for path in out_dir.iterdir()}
        for name in sorted(generated - set(expected)):
            failures.append(f"UNEXPECTED  {name} (generated, not in manifest)")
        for name, rel in sorted(expected.items()):
            committed = ROOT / rel
            fresh = out_dir / name
            if not fresh.exists():
                failures.append(f"MISSING     {rel} (manifest step wrote nothing)")
            elif not committed.exists():
                failures.append(f"UNCOMMITTED {rel} (regenerated, no committed file)")
            elif committed.read_bytes() != fresh.read_bytes():
                failures.append(
                    f"DRIFT       {rel} "
                    f"(committed {committed.stat().st_size} B, "
                    f"regenerated {fresh.stat().st_size} B)"
                )
    print()
    if failures:
        print("\n".join(failures))
        print(f"\nFAIL: {len(failures)} of {len(expected)} artifacts do not reproduce")
        return 1
    print(f"OK: all {len(expected)} artifacts reproduce byte-for-byte")
    return 0


def write(steps: tuple[Step, ...]) -> int:
    """Regenerate every artifact in place (each into its committed dir)."""
    for i, step in enumerate(steps, 1):
        out_dirs = {str((ROOT / rel).parent) for rel in step.outputs}
        assert len(out_dirs) == 1, f"step {step.module} writes to multiple dirs"
        print(f"[{i:2}/{len(steps)}] {step.module} {' '.join(step.argv)}")
        run_step(step, Path(out_dirs.pop()))
    print(f"\nregenerated {sum(len(s.outputs) for s in steps)} artifact paths in place")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all committed summaries and figures and "
        "verify them against the committed files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the committed files in place instead of auditing",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="skip figure steps (PNG bytes depend on the font stack; "
        "tables must reproduce on any machine)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    steps = SUMMARY_STEPS if args.tables_only else MANIFEST
    code = write(steps) if args.write else audit(steps)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

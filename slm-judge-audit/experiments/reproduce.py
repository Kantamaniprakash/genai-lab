"""Reproduction audit: regenerate every committed summary and figure from the
committed raw stores in a clean copy of the tree, and verify byte-identity.

    python -m experiments.reproduce                 # audit; writes nothing
    python -m experiments.reproduce --tables-only   # skip figure comparison
    python -m experiments.reproduce --keep-scratch  # keep the scratch tree

The audit extracts ``git archive HEAD`` into a scratch directory, deletes
every regenerable artifact under ``results/summary/`` and
``results/figures/`` there, seeds the pinned parquet (every generator's
``fetch()`` re-verifies its SHA256, so seeding is integrity-equivalent to
downloading), replays the manifest below as subprocesses inside the copy,
and byte-compares everything against the committed bytes. A passing audit
proves the committed tables and figures follow from the committed raw
stores and the committed code alone — nothing hand-entered, nothing stale.

The manifest is the authoritative map from committed artifact to the
invocation that produced it, and the audit enforces coverage in both
directions (the day-15 lesson: the README copy and the generated copy of a
number can drift independently, so *generated artifacts* are what get
audited): a committed artifact no step claims fails the audit as UNCOVERED,
and a generated file no manifest entry claims fails it as UNEXPECTED.

Two committed files are pinned history rather than regenerable outputs:
``master_table__minimal__interim_qwen2.5-7b.{json,md}`` are the day-9
matched interim read taken while the 7B grid stood at 134/1200 judgments.
The completed store cannot re-produce that restriction set, so the audit
verifies the pair survives untouched instead of regenerating it;
``prefix_skew`` consumes the JSON as its pinned input.

Step order is dependency order, not README order: ``length_probe`` runs
before ``master_table`` because the master table's caption reads
``length_probe__minimal.json`` — regenerating them the other way around
rebuilds the exact staleness the day-15 coherence pass caught.

What byte-identity means here: the tables are deterministic end-to-end
(fixed bootstrap seeds, deterministic readout), so they must match on any
machine running the locked dependency set. The figures are deterministic
given the pinned matplotlib, but PNG bytes also depend on the font stack
below it; on a machine with different fonts, figure drift means "re-render
and compare visually", not "the numbers changed". ``--tables-only`` exists
for exactly that case.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = (
    "llama-3.1-8b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2.5-3b",
    "qwen2.5-7b",
)
RUBRICS = ("detailed", "minimal")

# Committed history, verified untouched rather than regenerated (see module
# docstring).
PINNED_HISTORICAL = (
    "results/summary/master_table__minimal__interim_qwen2.5-7b.json",
    "results/summary/master_table__minimal__interim_qwen2.5-7b.md",
)


@dataclass(frozen=True)
class Step:
    """One generator invocation and the repo-relative files it writes."""

    module: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...]


MANIFEST: tuple[Step, ...] = (
    # Per-store summaries, all 14 stores (finding provenance: every per-grid
    # section of the README).
    Step(
        "experiments.summarize",
        (),
        tuple(
            f"results/summary/{model}__{rubric}.json"
            for model in MODELS
            for rubric in RUBRICS
        ),
    ),
    # Readout-validity conditioning (finding 8 and successors). Only the
    # minimal-rubric views are committed, so the sweep is per-store: a bare
    # sweep would also emit the 7 uncommitted detailed-rubric views.
    *(
        Step(
            "experiments.compliance_view",
            ("--store", f"{model}__minimal"),
            (
                f"results/summary/{model}__minimal__compliance.json",
                f"results/figures/{model}__minimal_compliance.png",
            ),
        )
        for model in MODELS
    ),
    # Per-store decomposition + accuracy panels (minimal-rubric only, as
    # committed; same reason as compliance_view for the per-store sweep).
    *(
        Step(
            "experiments.make_figures",
            ("--store", f"{model}__minimal"),
            (
                f"results/figures/{model}__minimal_accuracy.png",
                f"results/figures/{model}__minimal_decomposition.png",
            ),
        )
        for model in MODELS
    ),
    # Value-over-length probe (findings 12-14). Must precede master_table,
    # whose caption reads length_probe__minimal.json.
    Step(
        "experiments.length_probe",
        (),
        (
            "results/summary/length_probe__minimal.json",
            "results/figures/length_probe__minimal.png",
        ),
    ),
    # Cross-judge headline table (findings 28-35 spine).
    Step(
        "experiments.master_table",
        (),
        (
            "results/summary/master_table__minimal.json",
            "results/summary/master_table__minimal.md",
        ),
    ),
    # Composition skew of the day-9 interim prefix (finding 26); reads the
    # pinned interim JSON plus the just-regenerated master table.
    Step(
        "experiments.prefix_skew",
        ("--interim-for", "qwen2.5-7b"),
        ("results/figures/prefix_skew__minimal.png",),
    ),
    # Representativeness of a partial grid per execution order (finding 27).
    Step(
        "experiments.schedule_coverage",
        (),
        ("results/figures/schedule_coverage.png",),
    ),
    # Judge scaling curve across both families (findings 28-32).
    Step(
        "experiments.scaling_curve",
        (),
        ("results/figures/scaling__minimal.png",),
    ),
    # Reliability diagrams + ECE (finding 15 and successors).
    Step(
        "experiments.calibration",
        (),
        (
            "results/summary/calibration__minimal.json",
            "results/figures/calibration__minimal.png",
        ),
    ),
    # Additive-shift test + correction ladder (findings 19-20).
    Step(
        "experiments.bias_model",
        (),
        (
            "results/summary/bias_model__minimal.json",
            "results/figures/bias_model__minimal.png",
        ),
    ),
    # Per-subset heterogeneity forest (finding 25).
    Step(
        "experiments.subset_view",
        (),
        (
            "results/summary/subset_view__minimal.json",
            "results/figures/subset_view__minimal.png",
        ),
    ),
    # Paired rubric-sensitivity view, full sweep (findings 36-49).
    Step(
        "experiments.rubric_view",
        (),
        (
            "results/summary/rubric_pair__minimal_vs_detailed.json",
            "results/summary/rubric_pair__minimal_vs_detailed.md",
            "results/figures/rubric_pair__minimal_vs_detailed.png",
        ),
    ),
    # Perturbation-model fit (findings 39, 44-48).
    Step(
        "experiments.rubric_fragility",
        (),
        (
            "results/summary/rubric_fragility__minimal_vs_detailed.json",
            "results/summary/rubric_fragility__minimal_vs_detailed.md",
            "results/figures/rubric_fragility__minimal_vs_detailed.png",
        ),
    ),
)

AUDITED_DIRS = ("results/summary", "results/figures")
PARQUET = "data/rewardbench_filtered.parquet"


def manifest_outputs(steps: tuple[Step, ...]) -> tuple[str, ...]:
    return tuple(rel for step in steps for rel in step.outputs)


def extract_head(scratch: Path) -> Path:
    """Extract ``git archive HEAD`` into scratch/tree and return that root."""
    tree = scratch / "tree"
    tree.mkdir()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree, filter="data")
    return tree


def run_step(step: Step, tree: Path) -> None:
    """Replay one generator inside the clean tree; fail loudly on error."""
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, "-m", step.module, *step.argv],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(
            f"step failed ({proc.returncode}): {step.module} {' '.join(step.argv)}"
        )


def audit(steps: tuple[Step, ...], tables_only: bool, keep_scratch: bool) -> int:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "experiments"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_dirty = [line for line in dirty.splitlines() if not line.startswith("??")]
    if tracked_dirty:
        print("WARNING: src/ or experiments/ differs from HEAD — the audit runs")
        print("the committed code, not the working tree:")
        print("\n".join(f"  {line}" for line in tracked_dirty))

    def compared(rel: str) -> bool:
        return not (tables_only and rel.endswith(".png"))

    if tables_only:
        steps = tuple(
            step for step in steps if any(compared(rel) for rel in step.outputs)
        )
    claimed = manifest_outputs(steps)
    failures: list[str] = []
    scratch = Path(tempfile.mkdtemp(prefix=".reproduce-", dir=ROOT))
    try:
        tree = extract_head(scratch)

        # Snapshot the committed artifacts, then clear them from the tree so
        # a step that silently writes nothing surfaces as MISSING, not as a
        # stale pass.
        committed: dict[str, bytes] = {}
        for adir in AUDITED_DIRS:
            for path in sorted((tree / adir).iterdir()):
                rel = f"{adir}/{path.name}"
                committed[rel] = path.read_bytes()
                if rel not in PINNED_HISTORICAL:
                    path.unlink()

        for rel in sorted(set(committed) - set(claimed) - set(PINNED_HISTORICAL)):
            if compared(rel):
                failures.append(f"UNCOVERED   {rel} (committed, no manifest step)")
        for rel in sorted(set(claimed) - set(committed)):
            failures.append(f"PHANTOM     {rel} (in manifest, not committed)")

        if (ROOT / PARQUET).exists():
            (tree / PARQUET).parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT / PARQUET, tree / PARQUET)

        for i, step in enumerate(steps, 1):
            print(f"[{i:2}/{len(steps)}] {step.module} {' '.join(step.argv)}",
                  flush=True)
            run_step(step, tree)

        for rel in sorted(set(claimed)):
            if not compared(rel):
                continue
            fresh = tree / rel
            if not fresh.exists():
                failures.append(f"MISSING     {rel} (manifest step wrote nothing)")
            elif rel in committed and committed[rel] != fresh.read_bytes():
                # A stale artifact is fixed by regenerating it in place and
                # committing; flag the half-done state where the working
                # tree already carries the corrected bytes.
                worktree = ROOT / rel
                fixed = (
                    " — working tree already matches the regeneration; commit it"
                    if worktree.exists() and worktree.read_bytes() == fresh.read_bytes()
                    else ""
                )
                failures.append(
                    f"DRIFT       {rel} "
                    f"(committed {len(committed[rel])} B, "
                    f"regenerated {fresh.stat().st_size} B{fixed})"
                )
        for rel in PINNED_HISTORICAL:
            if committed[rel] != (tree / rel).read_bytes():
                failures.append(f"TOUCHED     {rel} (pinned history was overwritten)")
        for adir in AUDITED_DIRS:
            for path in sorted((tree / adir).iterdir()):
                rel = f"{adir}/{path.name}"
                if rel not in claimed and rel not in PINNED_HISTORICAL and compared(rel):
                    failures.append(f"UNEXPECTED  {rel} (generated, not in manifest)")
    finally:
        if keep_scratch:
            print(f"scratch tree kept at {scratch}")
        else:
            shutil.rmtree(scratch)

    n_audited = sum(1 for rel in set(claimed) | set(PINNED_HISTORICAL) if compared(rel))
    print()
    if failures:
        print("\n".join(sorted(failures)))
        print(f"\nFAIL: {len(failures)} problem(s) across {n_audited} artifacts")
        return 1
    print(f"OK: all {n_audited} artifacts reproduce byte-for-byte from HEAD")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all committed summaries and figures from the "
        "committed raw stores in a clean copy of HEAD and verify byte-identity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="compare only non-PNG artifacts (PNG bytes depend on the font "
        "stack; tables must reproduce on any machine)",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="keep the scratch tree for inspection instead of deleting it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(audit(MANIFEST, args.tables_only, args.keep_scratch))


if __name__ == "__main__":
    main()

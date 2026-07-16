"""Tests for the reproduction-audit manifest (experiments/reproduce.py).

These pin the manifest to the committed artifact set: every committed
summary and figure must be claimed by exactly one output path, and every
manifest step must resolve to a runnable generator. The audit itself
(regenerate + byte-compare) is an integration run, exercised by
``python -m experiments.reproduce``, not by the unit suite.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from experiments.reproduce import (
    CROSS_DATASET_FIGURES,
    FIGURE_STEPS,
    MANIFEST,
    ROOT,
    SUMMARY_STEPS,
)


def manifest_outputs() -> list[str]:
    return [rel for step in MANIFEST for rel in step.outputs]


class TestManifestCoversCommittedArtifacts:
    def test_summary_outputs_match_committed_tables(self):
        committed = {
            f"results/{path.name}" for path in (ROOT / "results").glob("summary_*.md")
        }
        claimed = {rel for step in SUMMARY_STEPS for rel in step.outputs}
        assert claimed == committed

    def test_figure_outputs_match_committed_figures(self):
        committed = {
            f"results/figures/{path.name}"
            for path in (ROOT / "results" / "figures").glob("*.png")
        }
        committed |= {f"assets/{path.name}" for path in (ROOT / "assets").glob("*.png")}
        claimed = {rel for step in FIGURE_STEPS for rel in step.outputs}
        assert claimed == committed

    def test_every_output_is_committed(self):
        missing = [rel for rel in manifest_outputs() if not (ROOT / rel).is_file()]
        assert missing == []

    def test_outputs_unique_up_to_cross_dataset_rerenders(self):
        # Cross-dataset figures are rendered by both make_figures steps
        # (identically — the audit byte-checks that); everything else must
        # be claimed exactly once.
        outputs = manifest_outputs()
        for rel in set(outputs):
            expected = 2 if rel in CROSS_DATASET_FIGURES else 1
            assert outputs.count(rel) == expected, rel

    def test_basenames_unique(self):
        # The audit compares by basename inside one scratch directory, so
        # two different artifacts must never share a filename.
        names = {Path(rel).name: rel for rel in manifest_outputs()}
        assert len(names) == len(set(manifest_outputs()))


class TestManifestStepsAreRunnable:
    def test_modules_import_and_expose_main_with_out_dir(self):
        for step in MANIFEST:
            module = importlib.import_module(step.module)
            assert callable(module.main), step.module
            # Every generator must accept the --out-dir redirect the audit
            # relies on; parse_args also validates the step's stored argv.
            args = module.parse_args([*step.argv, "--out-dir", "ignored"])
            assert str(args.out_dir) == "ignored", step.module

    def test_each_step_writes_a_single_directory(self):
        for step in MANIFEST:
            parents = {str(Path(rel).parent) for rel in step.outputs}
            assert len(parents) == 1, step.module

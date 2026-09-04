"""Tests pinning the reproduction-audit manifest (experiments/reproduce.py).

The audit's value rests on two-way coverage between the manifest and the
committed artifact set — the day-15 lesson (a generated file can go stale
independently of the README) made that coverage the point. These tests keep
the manifest honest as artifacts are added: every committed summary and
figure is claimed by exactly one step or pinned as history, and every step
resolves to a runnable generator. The audit itself (clean-tree regenerate +
byte-compare) is an integration run, exercised by
``python -m experiments.reproduce``, not by the unit suite.
"""

from __future__ import annotations

import importlib

from experiments.reproduce import (
    AUDITED_DIRS,
    MANIFEST,
    PINNED_HISTORICAL,
    ROOT,
    manifest_outputs,
)


def committed_artifacts() -> set[str]:
    return {
        f"{adir}/{path.name}"
        for adir in AUDITED_DIRS
        for path in (ROOT / adir).iterdir()
        if path.is_file()
    }


class TestManifestCoversCommittedArtifacts:
    def test_claimed_plus_pinned_is_exactly_the_committed_set(self):
        claimed = set(manifest_outputs(MANIFEST))
        assert claimed | set(PINNED_HISTORICAL) == committed_artifacts()

    def test_pinned_files_are_not_also_claimed(self):
        assert not set(manifest_outputs(MANIFEST)) & set(PINNED_HISTORICAL)

    def test_each_artifact_claimed_by_exactly_one_step(self):
        outputs = manifest_outputs(MANIFEST)
        duplicates = {rel for rel in outputs if outputs.count(rel) > 1}
        assert duplicates == set()

    def test_pinned_history_exists(self):
        for rel in PINNED_HISTORICAL:
            assert (ROOT / rel).is_file(), rel


class TestManifestStepsAreRunnable:
    def test_modules_import_and_expose_main(self):
        for step in MANIFEST:
            module = importlib.import_module(step.module)
            assert callable(module.main), step.module

    def test_outputs_live_in_audited_dirs(self):
        for rel in manifest_outputs(MANIFEST):
            assert rel.rsplit("/", 1)[0] in AUDITED_DIRS, rel

    def test_length_probe_precedes_master_table(self):
        # The master table's caption reads length_probe__minimal.json; the
        # reverse order rebuilds the day-15 stale caption.
        modules = [step.module for step in MANIFEST]
        assert modules.index("experiments.length_probe") < modules.index(
            "experiments.master_table"
        )

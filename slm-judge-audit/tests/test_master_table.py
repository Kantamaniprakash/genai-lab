"""The generated caption's fitted-length verdict must track the probe summary.

The first version of the caption hardcoded "only the two 3B judges beat that
one", which went stale the day the 7B grid completed. The verdict is now
computed from the length-probe summary; these tests pin the three behaviours
that matter: winners are read off the CIs, no winners degrades gracefully,
and a missing probe summary produces no sentence at all.
"""

import json

import pytest

from experiments import master_table


def probe_summary(deltas: dict[str, tuple[float, float, float]]) -> dict:
    """Minimal length-probe JSON: model -> (mean, lo, hi) for the overall
    joint − length-only accuracy delta."""
    return {
        "models": {
            key: {
                "overall": {
                    "acc_joint_minus_length": {"mean": mean, "ci95": [lo, hi]},
                    "specs": {"length": {"acc": 0.575}},
                }
            }
            for key, (mean, lo, hi) in deltas.items()
        }
    }


@pytest.fixture
def summary_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(master_table, "SUMMARY_DIR", tmp_path)
    return tmp_path


def write_probe(summary_dir, deltas, rubric="minimal"):
    path = summary_dir / f"length_probe__{rubric}.json"
    path.write_text(json.dumps(probe_summary(deltas)))


def test_winners_are_models_whose_ci_excludes_zero(summary_dir):
    write_probe(summary_dir, {
        "qwen2.5-3b": (0.205, 0.156, 0.261),
        "qwen2.5-7b": (0.272, 0.230, 0.327),
        "qwen2.5-0.5b": (-0.007, -0.047, 0.048),
        "llama-3.2-1b": (-0.007, -0.042, 0.047),
    })
    verdict = master_table.fitted_length_verdict()
    assert "`qwen2.5-3b`" in verdict
    assert "`qwen2.5-7b`" in verdict
    assert "qwen2.5-0.5b" not in verdict
    assert "llama-3.2-1b" not in verdict
    assert "4 judges measured" in verdict


def test_positive_point_estimate_alone_is_not_a_win(summary_dir):
    # CI touching zero from below: not significant, not a winner.
    write_probe(summary_dir, {"qwen2.5-1.5b": (0.030, -0.049, 0.082)})
    assert master_table.fitted_length_verdict() == (
        " No judge measured so far beats that one."
    )


def test_missing_probe_summary_yields_no_sentence(summary_dir):
    assert master_table.fitted_length_verdict() == ""


def test_winners_sorted_for_stable_output(summary_dir):
    write_probe(summary_dir, {
        "qwen2.5-7b": (0.272, 0.230, 0.327),
        "llama-3.2-3b": (0.125, 0.075, 0.181),
        "qwen2.5-3b": (0.205, 0.156, 0.261),
    })
    verdict = master_table.fitted_length_verdict()
    assert verdict.index("llama-3.2-3b") < verdict.index("qwen2.5-3b")
    assert verdict.index("qwen2.5-3b") < verdict.index("qwen2.5-7b")

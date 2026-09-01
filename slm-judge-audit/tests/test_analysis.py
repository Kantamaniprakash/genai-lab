"""Tests for swap-pair assembly, the decomposition, and the paired bootstrap."""

from __future__ import annotations

import pytest

from src.analysis import (
    SwapPair,
    assemble_pairs,
    bootstrap_mean_ci,
    paired_bootstrap_delta_ci,
    summarize_pairs,
)
from src.judge import JudgmentRecord


def make_record(item_id: str, order: str, z: float, *, model: str = "fake",
                rubric: str = "minimal", compliant: bool = True) -> JudgmentRecord:
    return JudgmentRecord(
        model=model,
        rubric=rubric,
        order=order,
        item_id=item_id,
        expected_verdict="A" if order == "chosen_first" else "B",
        z=z,
        logp_a=-0.5,
        logp_b=-0.5 - z,
        mass_ab=0.98,
        argmax_token="A" if compliant else "The",
        compliant=compliant,
        n_prompt_tokens=100,
        prefill_seconds=0.5,
    )


def test_assemble_pairs_and_decomposition():
    records = [
        make_record("i/1", "chosen_first", 3.0),
        make_record("i/1", "rejected_first", 1.0),
    ]
    pairs, incomplete = assemble_pairs(records)
    assert incomplete == 0
    (pair,) = pairs
    assert pair.s == pytest.approx(1.0)   # prefers chosen either way
    assert pair.b == pytest.approx(2.0)   # shifted toward A
    # Reconstruction identity
    assert pair.z_cf == pytest.approx(pair.b + pair.s)
    assert pair.z_rf == pytest.approx(pair.b - pair.s)
    # Raw: cf correct (z>=0 -> A, expected A), rf wrong (z>=0 -> A, expected B)
    assert pair.raw_correct_cf == 1.0
    assert pair.raw_correct_rf == 0.0
    assert pair.raw_correct_mean == 0.5
    # Symmetrized rescues it; and the swap flipped the positional verdict? No:
    # both orders said A, so no flip — the classic bias signature.
    assert pair.sym_correct == 1.0
    assert not pair.positional_flip


def test_positional_flip_and_sym_tie():
    flip = SwapPair(item_id="i/2", z_cf=0.5, z_rf=-0.5, compliant_both=True, mass_min=0.9)
    assert flip.positional_flip
    assert flip.sym_correct == 1.0
    tie = SwapPair(item_id="i/3", z_cf=1.0, z_rf=1.0, compliant_both=True, mass_min=0.9)
    assert tie.s == 0.0
    assert tie.sym_correct == 0.5


def test_assemble_rejects_mixed_and_duplicate_sets():
    with pytest.raises(ValueError, match="mixed"):
        assemble_pairs([
            make_record("i/1", "chosen_first", 1.0, model="a"),
            make_record("i/1", "rejected_first", 1.0, model="b"),
        ])
    with pytest.raises(ValueError, match="duplicate"):
        assemble_pairs([
            make_record("i/1", "chosen_first", 1.0),
            make_record("i/1", "chosen_first", 2.0),
        ])


def test_assemble_counts_incomplete_items():
    records = [
        make_record("i/1", "chosen_first", 1.0),
        make_record("i/1", "rejected_first", -1.0),
        make_record("i/2", "chosen_first", 1.0),  # missing its swap
    ]
    pairs, incomplete = assemble_pairs(records)
    assert len(pairs) == 1 and incomplete == 1


def test_bootstrap_ci_is_deterministic_and_sane():
    values = [0.0, 1.0] * 50
    mean1 = bootstrap_mean_ci(values, n_boot=2000, seed=7)
    mean2 = bootstrap_mean_ci(values, n_boot=2000, seed=7)
    assert mean1 == mean2
    mean, lo, hi = mean1
    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi
    assert 0.35 < lo and hi < 0.65
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])


def test_paired_delta_ci_detects_systematic_gain():
    # Symmetrization rescues every item: sym=1, raw_mean=0.5 on all pairs.
    pairs = [
        SwapPair(item_id=f"i/{k}", z_cf=2.0, z_rf=1.0, compliant_both=True, mass_min=0.9)
        for k in range(40)
    ]
    delta, lo, hi = paired_bootstrap_delta_ci(
        pairs, lambda p: p.sym_correct, lambda p: p.raw_correct_mean, n_boot=500
    )
    assert delta == pytest.approx(0.5)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)  # zero variance


def test_summarize_pairs_block():
    pairs = [
        SwapPair(item_id="i/1", z_cf=3.0, z_rf=1.0, compliant_both=True, mass_min=0.99),
        SwapPair(item_id="i/2", z_cf=0.5, z_rf=-0.5, compliant_both=True, mass_min=0.95),
        SwapPair(item_id="i/3", z_cf=-1.0, z_rf=2.0, compliant_both=False, mass_min=0.60),
    ]
    summary = summarize_pairs(pairs, n_boot=200)
    assert summary["n_items"] == 3
    assert summary["compliance_rate"] == pytest.approx(2 / 3)
    # b values: 2.0, 0.0, 0.5; s values: 1.0, 0.5, -1.5
    assert summary["bias_b"]["mean"] == pytest.approx((2.0 + 0.0 + 0.5) / 3)
    assert summary["bias_b"]["median_abs"] == pytest.approx(0.5)
    assert summary["preference_s"]["median_abs"] == pytest.approx(1.0)
    assert summary["raw_acc_chosen_first"] == pytest.approx(2 / 3)
    assert summary["raw_acc_rejected_first"] == pytest.approx(1 / 3)
    # sym correct: 1.0, 1.0, 0.0
    assert summary["sym_acc"]["mean"] == pytest.approx(2 / 3)
    assert summary["frac_bias_dominates"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        summarize_pairs([])


def test_sign_length_agreement_excludes_both_tie_kinds():
    from src.analysis import sign_length_agreement

    pairs = [
        SwapPair(item_id="i/1", z_cf=2.0, z_rf=0.0, compliant_both=True, mass_min=0.9),   # s=+1
        SwapPair(item_id="i/2", z_cf=0.0, z_rf=2.0, compliant_both=True, mass_min=0.9),   # s=-1
        SwapPair(item_id="i/3", z_cf=1.0, z_rf=-1.0, compliant_both=True, mass_min=0.9),  # s=+1
        SwapPair(item_id="i/4", z_cf=1.0, z_rf=1.0, compliant_both=True, mass_min=0.9),   # s=0: excluded
        SwapPair(item_id="i/5", z_cf=3.0, z_rf=-3.0, compliant_both=True, mass_min=0.9),  # dlen=0: excluded
    ]
    dlens = {"i/1": 40, "i/2": 25, "i/3": -10, "i/4": 5, "i/5": 0}
    view = sign_length_agreement(pairs, dlens.__getitem__)
    # Used items: i/1 prefers chosen and chosen is longer (agree); i/2
    # prefers rejected while chosen is longer (disagree); i/3 prefers chosen
    # while chosen is shorter (disagree).
    assert view["n_used"] == 3
    assert view["n_excluded"] == 2
    assert view["agree"] == pytest.approx(1 / 3)
    # All-tie input reports no agreement number rather than a fabricated one.
    empty = sign_length_agreement(
        [SwapPair(item_id="i/6", z_cf=1.0, z_rf=1.0, compliant_both=True, mass_min=0.9)],
        {"i/6": 12}.__getitem__,
    )
    assert empty["agree"] is None and empty["n_used"] == 0


def test_two_sample_delta_ci_detects_group_gap():
    from src.analysis import two_sample_bootstrap_delta_ci

    high = [1.0] * 30 + [0.0] * 10   # mean 0.75
    low = [1.0] * 10 + [0.0] * 30    # mean 0.25
    delta, lo, hi = two_sample_bootstrap_delta_ci(high, low, n_boot=500, seed=1)
    assert delta == pytest.approx(0.5)
    assert lo <= delta <= hi
    assert lo > 0.0  # gap this large should be resolved at n=40 per group
    # Deterministic given the seed
    again = two_sample_bootstrap_delta_ci(high, low, n_boot=500, seed=1)
    assert again == (delta, lo, hi)
    with pytest.raises(ValueError):
        two_sample_bootstrap_delta_ci([], [1.0], n_boot=10)


def make_pair(item_id: str, *, s: float, b: float = 0.0, compliant: bool = True,
              mass: float = 0.95) -> SwapPair:
    return SwapPair(item_id=item_id, z_cf=b + s, z_rf=b - s,
                    compliant_both=compliant, mass_min=mass)


def test_compliance_view_strata_and_delta():
    from src.analysis import compliance_view

    # Compliant items all correct, non-compliant all wrong.
    pairs = (
        [make_pair(f"x/{i}", s=1.0, compliant=True, mass=0.95) for i in range(8)]
        + [make_pair(f"y/{i}", s=-1.0, compliant=False, mass=0.1) for i in range(4)]
    )
    view = compliance_view(pairs, n_boot=200)
    assert view["n_items"] == 12
    assert view["compliance_rate"] == pytest.approx(8 / 12)
    assert view["strata"]["all"]["n_items"] == 12
    assert view["strata"]["compliant_both"]["n_items"] == 8
    assert view["strata"]["compliant_both"]["sym_acc"]["mean"] == pytest.approx(1.0)
    assert view["strata"]["non_compliant"]["sym_acc"]["mean"] == pytest.approx(0.0)
    assert view["sym_acc_compliant_minus_non"]["mean"] == pytest.approx(1.0)
    # Mass bins partition the pairs: totals must add back up to n.
    assert sum(b["n_items"] for b in view["mass_bins"]) == 12
    lowest = view["mass_bins"][0]
    assert (lowest["lo"], lowest["n_items"]) == (0.0, 4)
    with pytest.raises(ValueError):
        compliance_view([])


def test_compliance_view_single_stratum_and_bin_edges():
    from src.analysis import MASS_BINS, compliance_view

    # All compliant: no non_compliant stratum, no delta block.
    pairs = [
        make_pair("x/1", s=1.0, mass=0.25),   # left edge of bin 2 -> bin 2
        make_pair("x/2", s=1.0, mass=0.9),    # left edge of last bin -> last
        make_pair("x/3", s=-1.0, mass=1.0),   # top edge stays in last bin
    ]
    view = compliance_view(pairs, n_boot=100)
    assert "non_compliant" not in view["strata"]
    assert view["sym_acc_compliant_minus_non"] is None
    by_edge = {(b["lo"], b["hi"]): b["n_items"] for b in view["mass_bins"]}
    assert by_edge[MASS_BINS[1]] == 1
    assert by_edge[MASS_BINS[-1]] == 2
    assert sum(by_edge.values()) == 3


def test_compliance_view_category_composition():
    from src.analysis import compliance_view

    pairs = [
        make_pair("chat/1", s=1.0, compliant=True),
        make_pair("chat/2", s=-1.0, compliant=False),
        make_pair("math/1", s=1.0, compliant=True),
    ]
    cats = {"chat": "Chat", "math": "Reasoning"}
    view = compliance_view(
        pairs, category_of=lambda item_id: cats[item_id.split("/", 1)[0]], n_boot=100
    )
    chat = view["by_category"]["Chat"]
    assert chat["n_items"] == 2
    assert chat["compliance_rate"] == pytest.approx(0.5)
    assert chat["sym_acc_compliant"] == pytest.approx(1.0)
    assert chat["sym_acc_non_compliant"] == pytest.approx(0.0)
    reasoning = view["by_category"]["Reasoning"]
    assert reasoning["sym_acc_non_compliant"] is None


def test_compliance_view_category_gap_ci():
    from src.analysis import compliance_view

    # Chat: compliant items all correct, non-compliant all wrong -> gap +1
    # with a degenerate (point-mass) bootstrap distribution.
    pairs = (
        [make_pair(f"chat/c{i}", s=1.0, compliant=True) for i in range(6)]
        + [make_pair(f"chat/n{i}", s=-1.0, compliant=False) for i in range(6)]
        + [make_pair("math/1", s=1.0, compliant=True)]
    )
    cats = {"chat": "Chat", "math": "Reasoning"}
    view = compliance_view(
        pairs, category_of=lambda item_id: cats[item_id.split("/", 1)[0]], n_boot=200
    )
    gap = view["by_category"]["Chat"]["sym_acc_compliant_minus_non"]
    assert gap["mean"] == pytest.approx(1.0)
    assert gap["ci95"][0] == pytest.approx(1.0)
    # Single-stratum category carries no gap block.
    assert "sym_acc_compliant_minus_non" not in view["by_category"]["Reasoning"]


def test_subset_view_groups_and_floor():
    from src.analysis import subset_view

    pairs = (
        [make_pair(f"alpha/{i}", s=1.0, b=2.0) for i in range(4)]      # all correct
        + [make_pair(f"beta/{i}", s=-1.0, b=0.5) for i in range(3)]    # all wrong
        + [make_pair("gamma/0", s=0.0)]                                # tie -> 0.5
    )
    cats = {"alpha": "Chat", "beta": "Reasoning", "gamma": "Safety"}
    floors = {"alpha": 1.0, "beta": 0.0, "gamma": 0.5}
    view = subset_view(
        pairs,
        subset_of=lambda i: i.split("/", 1)[0],
        category_of=lambda i: cats[i.split("/", 1)[0]],
        longer_correct_of=lambda i: floors[i.split("/", 1)[0]],
        n_boot=100,
    )
    assert view["n_subsets"] == 3
    assert sum(b["n_items"] for b in view["subsets"].values()) == view["n_items"] == 8
    alpha = view["subsets"]["alpha"]
    assert (alpha["category"], alpha["longer_floor"]) == ("Chat", 1.0)
    assert alpha["sym_acc"]["mean"] == pytest.approx(1.0)
    assert alpha["median_b"] == pytest.approx(2.0)
    beta = view["subsets"]["beta"]
    assert beta["sym_acc"]["mean"] == pytest.approx(0.0)
    assert view["subsets"]["gamma"]["sym_acc"]["mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        subset_view([], subset_of=lambda i: i)


def test_judge_row_per_order_accuracy_and_length_delta():
    from src.analysis import judge_row

    # A saturated always-A judge: b swamps s on every item, so the positional
    # verdict is "A" in both orders (right when chosen is first, wrong when
    # it is second) while the symmetrized verdict still reads the content.
    pairs = [make_pair(f"x/{i}", s=0.2, b=5.0) for i in range(9)]
    pairs.append(make_pair("x/9", s=-0.2, b=5.0))
    # The length heuristic is right on the first two items only.
    floors = {f"x/{i}": (1.0 if i < 2 else 0.0) for i in range(10)}
    row = judge_row(pairs, lambda i: floors[i], n_boot=200)

    assert row["n_items"] == 10
    assert row["raw_acc_chosen_first"] == pytest.approx(1.0)
    assert row["raw_acc_rejected_first"] == pytest.approx(0.0)
    assert row["raw_acc"]["mean"] == pytest.approx(0.5)
    assert row["positional_flip_rate"] == pytest.approx(0.0)  # never flips
    assert row["frac_b_positive"] == pytest.approx(1.0)
    assert row["frac_bias_dominates"] == pytest.approx(1.0)
    assert row["sym_acc"]["mean"] == pytest.approx(0.9)
    assert row["sym_minus_raw"]["mean"] == pytest.approx(0.4)
    assert row["longer_floor"] == pytest.approx(0.2)
    assert row["sym_minus_longer"]["mean"] == pytest.approx(0.7)
    with pytest.raises(ValueError):
        judge_row([], lambda i: 0.0)


def test_master_table_markdown_renders_every_judge_and_floors():
    from experiments.master_table import COLUMNS, render_markdown
    from src.analysis import judge_row

    rows = {
        key: judge_row([make_pair(f"x/{i}", s=1.0, b=0.5) for i in range(4)],
                       lambda i: 0.5, n_boot=100)
        for key in ("qwen2.5-0.5b", "llama-3.2-1b")
    }
    md = render_markdown(rows, {"longer_chars": 0.42}, "minimal", 4)
    lines = md.splitlines()

    def cells(line: str) -> int:
        """Column separators, ignoring the escaped pipes inside |s|."""
        return line.replace(r"\|", "").count("|")

    # Header, separator, one row per judge, three floor rows — all the same
    # width, which is the property an unescaped pipe in a header would break.
    assert cells(lines[0]) == len(COLUMNS) + 1
    assert all(cells(line) == len(COLUMNS) + 1 for line in lines[1:7])
    assert r"median \|s\|" in lines[0]
    # Family x scale order comes from MODEL_STYLES, not the dict insertion order.
    assert lines[2].startswith("| qwen2.5-0.5b |")
    assert lines[3].startswith("| llama-3.2-1b |")
    assert "0.500" in lines[4] and "*random floor*" in lines[4]
    assert "0.420" in lines[6] and "*longer-response floor*" in lines[6]
    assert "0.5B" in lines[2] and "1B" in lines[3]
    assert "seed 0" in md and "4 stratified" in md


def test_composition_note_quantifies_prefix_skew():
    from experiments.master_table import composition_note

    full = [f"chat/{i}" for i in range(10)] + [f"safety/{i}" for i in range(10)]
    restricted = [f"chat/{i}" for i in range(4)]   # an alphabetical prefix
    note = composition_note(restricted, full,
                            lambda i: i.split("/", 1)[0].capitalize())
    assert "Chat 100% (vs 50% in the full sample)" in note
    assert "Safety 0% (vs 50% in the full sample)" in note
    assert "Safety is not represented at all." in note


def test_interim_markdown_is_labelled_and_drops_full_sample_verdict():
    from experiments.master_table import render_markdown
    from src.analysis import judge_row

    rows = {"qwen2.5-7b": judge_row(
        [make_pair(f"x/{i}", s=1.0, b=0.5) for i in range(4)],
        lambda i: 0.5, n_boot=100)}
    skew = "Category composition of the restricted set: Chat 100%."
    md = render_markdown(rows, {"longer_chars": 0.9}, "minimal", 4,
                         interim="qwen2.5-7b", skew=skew)

    assert md.startswith("**INTERIM — not a result.**")
    assert skew in md
    assert "alphabetical prefix" in md
    # The length-baseline verdict is a full-sample claim and must not ride
    # along on a restricted table, where the floor itself is a different number.
    assert "the real opponent is the *fitted*" not in md
    full = render_markdown(rows, {"longer_chars": 0.9}, "minimal", 4)
    assert "the real opponent is the *fitted*" in full
    assert not full.startswith("**INTERIM")


def test_declutter_separates_collided_labels_without_moving_anchors():
    from experiments.prefix_skew import declutter

    # Two judges landing on the same accuracy, plus one well clear of them.
    placed = declutter([(0.911, "a", "#000"), (0.911, "b", "#111"),
                        (0.500, "c", "#222")], gap=0.04)
    anchors = [p[0] for p in placed]
    label_ys = [p[1] for p in placed]

    assert anchors == [0.500, 0.911, 0.911]        # true values, untouched
    assert label_ys[0] == pytest.approx(0.500)     # lowest label never moves
    assert label_ys[1] == pytest.approx(0.911)     # clear of the one below it
    assert label_ys[2] == pytest.approx(0.951)     # pushed up by exactly gap
    assert all(b - a >= 0.04 - 1e-9 for a, b in zip(label_ys, label_ys[1:]))

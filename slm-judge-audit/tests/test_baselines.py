"""Tests for the trivial judge floors."""

from __future__ import annotations

import pytest

from src.baselines import (
    RANDOM_ACCURACY,
    always_a_correct,
    longer_response_correct,
    summarize_baselines,
)
from src.data import PairItem


def make_item(item_id: str, chosen: str, rejected: str, subset: str = "mt-bench-easy") -> PairItem:
    categories = {"mt-bench-easy": "Chat", "math-prm": "Reasoning"}
    return PairItem(
        item_id=item_id,
        subset=subset,
        category=categories[subset],
        prompt="p",
        chosen=chosen,
        rejected=rejected,
        chosen_model="m1",
        rejected_model="m2",
    )


def test_always_a_is_the_position_floor():
    item = make_item("mt-bench-easy/1", "good answer", "bad")
    assert always_a_correct(item, "chosen_first") == 1.0
    assert always_a_correct(item, "rejected_first") == 0.0


def test_longer_response_units_and_ties():
    longer_chosen = make_item("mt-bench-easy/1", "a much longer chosen answer", "short")
    longer_rejected = make_item("mt-bench-easy/2", "short", "a much longer rejected one")
    tie = make_item("mt-bench-easy/3", "abcd", "wxyz")
    assert longer_response_correct(longer_chosen) == 1.0
    assert longer_response_correct(longer_rejected) == 0.0
    assert longer_response_correct(tie) == 0.5
    # word unit can disagree with chars: two long words vs three short letters
    item = make_item("mt-bench-easy/4", "aaaaaaaaaa bbbbbbbbbb", "a b c")
    assert longer_response_correct(item, "chars") == 1.0
    assert longer_response_correct(item, "words") == 0.0
    with pytest.raises(ValueError):
        longer_response_correct(item, "tokens")


def test_summarize_baselines_overall_and_by_category():
    items = (
        make_item("mt-bench-easy/1", "looooooooong chosen", "short"),
        make_item("math-prm/1", "x", "loooooooonger rejected", subset="math-prm"),
    )
    summary = summarize_baselines(items)
    overall = summary["overall"]
    assert overall["n_items"] == 2
    # always-A over the exhaustive order pair is exactly 1/2 by construction.
    assert overall["always_a_raw"] == 0.5
    assert overall["longer_chars"] == 0.5  # one hit, one miss
    assert overall["random"] == RANDOM_ACCURACY
    assert summary["by_category"]["Chat"]["longer_chars"] == 1.0
    assert summary["by_category"]["Reasoning"]["longer_chars"] == 0.0

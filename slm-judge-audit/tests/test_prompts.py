import pytest

from src.data import SUBSET_TO_CATEGORY, PairItem
from src.prompts import (
    ORDERS,
    RUBRICS,
    VERDICT_TOKENS,
    build_both_orders,
    build_judge_prompt,
)


def make_item() -> PairItem:
    return PairItem(
        item_id="mt-bench-easy/1",
        subset="mt-bench-easy",
        category=SUBSET_TO_CATEGORY["mt-bench-easy"],
        prompt="Explain photosynthesis.",
        chosen="GOLD-BETTER-RESPONSE",
        rejected="GOLD-WORSE-RESPONSE",
        chosen_model="model-x",
        rejected_model="model-y",
    )


class TestBuildJudgePrompt:
    def test_chosen_first_puts_chosen_as_a(self):
        p = build_judge_prompt(make_item(), "chosen_first")
        assert p.user.index("GOLD-BETTER-RESPONSE") < p.user.index("GOLD-WORSE-RESPONSE")
        assert p.expected_verdict == "A"

    def test_rejected_first_puts_chosen_as_b(self):
        p = build_judge_prompt(make_item(), "rejected_first")
        assert p.user.index("GOLD-WORSE-RESPONSE") < p.user.index("GOLD-BETTER-RESPONSE")
        assert p.expected_verdict == "B"

    def test_prompt_contains_instruction_and_both_responses(self):
        for order in ORDERS:
            p = build_judge_prompt(make_item(), order)
            assert "Explain photosynthesis." in p.user
            assert "GOLD-BETTER-RESPONSE" in p.user
            assert "GOLD-WORSE-RESPONSE" in p.user

    def test_prompt_never_reveals_gold(self):
        for order in ORDERS:
            p = build_judge_prompt(make_item(), order)
            text = (p.system + p.user).lower()
            for token in ("chosen", "rejected", "gold label"):
                assert token not in text

    def test_all_rubrics_render_and_request_single_letter(self):
        for name in RUBRICS:
            p = build_judge_prompt(make_item(), "chosen_first", name)
            assert p.rubric_name == name
            assert "exactly one letter: A or B" in p.user

    def test_unknown_order_rejected(self):
        with pytest.raises(ValueError, match="unknown order"):
            build_judge_prompt(make_item(), "gold_first")

    def test_unknown_rubric_rejected(self):
        with pytest.raises(ValueError, match="unknown rubric"):
            build_judge_prompt(make_item(), "chosen_first", "nonexistent")


class TestSwapPair:
    def test_both_orders_cover_both_verdicts(self):
        first, second = build_both_orders(make_item())
        assert (first.order, second.order) == ("chosen_first", "rejected_first")
        assert {first.expected_verdict, second.expected_verdict} == set(VERDICT_TOKENS)

    def test_swap_only_swaps_response_blocks(self):
        first, second = build_both_orders(make_item())
        assert first.system == second.system
        assert first.user != second.user
        # Swapping the two response bodies in one prompt yields the other.
        swapped = (
            first.user.replace("GOLD-BETTER-RESPONSE", "\x00")
            .replace("GOLD-WORSE-RESPONSE", "GOLD-BETTER-RESPONSE")
            .replace("\x00", "GOLD-WORSE-RESPONSE")
        )
        assert swapped == second.user

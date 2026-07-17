"""Judge prompt construction: rubric templates, order swap, verdict readout.

Every judgment in this audit is a *single-token* readout: the prompt is built
so that the model's very next token after the assistant turn opens should be
the verdict letter, and the runner reads the next-token distribution over
{"A", "B"} directly (renormalized), rather than sampling free-form text. That
makes each judgment one prefill-only forward pass (the whole audit is feasible
on CPU), makes greedy verdicts deterministic, and — the point of the project —
yields a verdict *log-odds* per judgment instead of a binary flip, so position
bias, calibration, and debiasing can be measured at the probability level.

Order is explicit and always audited both ways: ``chosen_first`` presents the
gold-preferred response as A, ``rejected_first`` presents it as B. The prompt
text never reveals which is which; ``expected_verdict`` records the letter a
perfect judge would produce for that presentation.

Rubrics are named templates. ``minimal`` is the canonical audit rubric;
``detailed`` adds explicit criteria and is the first axis of the
prompt-sensitivity analysis. Both instruct the model to answer with exactly
one letter so the single-token readout is aligned with what the model was
asked to do (the runner separately records whether the unconstrained argmax
token actually complies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .data import PairItem

Order = Literal["chosen_first", "rejected_first"]
ORDERS: tuple[Order, Order] = ("chosen_first", "rejected_first")

# The two admissible verdict tokens. The runner must verify that each judge's
# tokenizer encodes the assistant-prefixed letter as a single token before any
# run is accepted (recorded per-model in results).
VERDICT_TOKENS: tuple[str, str] = ("A", "B")


@dataclass(frozen=True)
class Rubric:
    name: str
    system: str
    # User template with {instruction}, {response_a}, {response_b} slots.
    user_template: str


RUBRICS: dict[str, Rubric] = {
    "minimal": Rubric(
        name="minimal",
        system=(
            "You are an impartial judge comparing two responses to the same "
            "user instruction. Decide which response answers the instruction "
            "better. Do not let the order in which the responses appear, "
            "their length, or their style influence your decision; judge "
            "only quality and adherence to the instruction."
        ),
        user_template=(
            "Instruction:\n{instruction}\n\n"
            "Response A:\n{response_a}\n\n"
            "Response B:\n{response_b}\n\n"
            "Which response is better? Reply with exactly one letter: A or B."
        ),
    ),
    "detailed": Rubric(
        name="detailed",
        system=(
            "You are an impartial judge comparing two responses to the same "
            "user instruction. Evaluate each response for: (1) adherence to "
            "the instruction and all of its constraints, (2) factual "
            "accuracy, (3) helpfulness and completeness, and (4) safety and "
            "appropriateness of the response given the instruction. Do not "
            "let the order in which the responses appear, their length, or "
            "their style influence your decision."
        ),
        user_template=(
            "Instruction:\n{instruction}\n\n"
            "Response A:\n{response_a}\n\n"
            "Response B:\n{response_b}\n\n"
            "Considering adherence, accuracy, helpfulness, and safety: which "
            "response is better? Reply with exactly one letter: A or B."
        ),
    ),
}


@dataclass(frozen=True)
class JudgePrompt:
    """A fully rendered judgment request for one (item, order, rubric)."""

    item_id: str
    rubric_name: str
    order: Order
    system: str
    user: str
    expected_verdict: str  # the letter a perfect judge produces ("A" or "B")

    def __post_init__(self) -> None:
        if self.expected_verdict not in VERDICT_TOKENS:
            raise ValueError(f"bad expected_verdict {self.expected_verdict!r}")


def build_judge_prompt(item: PairItem, order: Order, rubric_name: str = "minimal") -> JudgePrompt:
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    rubric = RUBRICS.get(rubric_name)
    if rubric is None:
        raise ValueError(f"unknown rubric {rubric_name!r}")

    if order == "chosen_first":
        response_a, response_b = item.chosen, item.rejected
        expected = "A"
    else:
        response_a, response_b = item.rejected, item.chosen
        expected = "B"

    return JudgePrompt(
        item_id=item.item_id,
        rubric_name=rubric.name,
        order=order,
        system=rubric.system,
        user=rubric.user_template.format(
            instruction=item.prompt, response_a=response_a, response_b=response_b
        ),
        expected_verdict=expected,
    )


def build_both_orders(item: PairItem, rubric_name: str = "minimal") -> tuple[JudgePrompt, JudgePrompt]:
    """The swap pair for one item — the unit of every position-bias analysis."""
    return (
        build_judge_prompt(item, "chosen_first", rubric_name),
        build_judge_prompt(item, "rejected_first", rubric_name),
    )

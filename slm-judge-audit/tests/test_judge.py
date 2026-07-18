"""Tests for the judge runner: templates, readout arithmetic, result store.

Everything here runs without llama.cpp except the final smoke test, which is
skipped unless the pinned Qwen2.5-0.5B GGUF is present in models/.
"""

from __future__ import annotations

import json
import math

import pytest

from src.data import PairItem
from src.judge import (
    CHAT_TEMPLATES,
    MODELS,
    JudgmentRecord,
    ResultStore,
    load_records,
    logits_to_record,
)
from src.prompts import build_judge_prompt


def make_item(item_id: str = "mt-bench-easy/7") -> PairItem:
    return PairItem(
        item_id=item_id,
        subset="mt-bench-easy",
        category="Chat",
        prompt="What is 2+2?",
        chosen="2+2 equals 4.",
        rejected="2+2 equals 5.",
        chosen_model="m1",
        rejected_model="m2",
    )


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

def test_chatml_render_is_exact():
    rendered = CHAT_TEMPLATES["chatml"].render("SYS", "USER")
    assert rendered == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nUSER<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_llama3_render_ends_at_assistant_header():
    rendered = CHAT_TEMPLATES["llama3"].render("SYS", "USER")
    assert rendered.startswith("<|begin_of_text|>")
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "SYS" in rendered and "USER" in rendered


def test_templates_survive_braces_in_content():
    # Judge prompts routinely contain code with { } — rendering must not
    # treat content braces as format slots.
    rendered = CHAT_TEMPLATES["chatml"].render("s", "int main() { return {0}; }")
    assert "int main() { return {0}; }" in rendered


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def test_registry_is_consistent():
    for key, model in MODELS.items():
        assert model.key == key
        assert model.template in CHAT_TEMPLATES
        assert len(model.sha256) == 64 and int(model.sha256, 16) >= 0
        assert len(model.revision) == 40
        assert model.url.startswith("https://huggingface.co/")
        assert model.revision in model.url and model.filename in model.url
        assert model.params_b > 0


# ---------------------------------------------------------------------------
# Readout arithmetic
# ---------------------------------------------------------------------------

def fake_logits(n_vocab: int, values: dict[int, float]) -> list[float]:
    logits = [-10.0] * n_vocab
    for token_id, value in values.items():
        logits[token_id] = value
    return logits


def test_logits_to_record_arithmetic():
    prompt = build_judge_prompt(make_item(), "chosen_first")
    logits = fake_logits(16, {3: 2.0, 5: -1.0})
    record = logits_to_record(
        logits,
        token_ids=(3, 5),
        argmax_token="A",
        prompt=prompt,
        model_key="fake",
        n_prompt_tokens=42,
        prefill_seconds=0.1,
    )
    assert record.z == pytest.approx(3.0)
    assert record.logp_a - record.logp_b == pytest.approx(3.0)
    # log-softmax sanity: probabilities from logp must renormalize below 1.
    assert 0 < math.exp(record.logp_a) < 1
    assert record.mass_ab == pytest.approx(
        math.exp(record.logp_a) + math.exp(record.logp_b)
    )
    assert record.compliant
    assert record.greedy_verdict == "A"
    assert record.expected_verdict == "A"
    assert record.raw_correct


def test_greedy_verdict_and_correctness_flip_with_sign():
    prompt = build_judge_prompt(make_item(), "rejected_first")
    logits = fake_logits(16, {3: -4.0, 5: 1.0})
    record = logits_to_record(
        logits,
        token_ids=(3, 5),
        argmax_token="B",
        prompt=prompt,
        model_key="fake",
        n_prompt_tokens=7,
        prefill_seconds=0.0,
    )
    assert record.z == pytest.approx(-5.0)
    assert record.greedy_verdict == "B"
    assert record.expected_verdict == "B"  # rejected_first: gold-chosen is B
    assert record.raw_correct


def test_noncompliant_argmax_is_flagged_but_readout_still_recorded():
    prompt = build_judge_prompt(make_item(), "chosen_first")
    logits = fake_logits(16, {3: 1.0, 5: 0.5, 9: 6.0})
    record = logits_to_record(
        logits,
        token_ids=(3, 5),
        argmax_token="The",
        prompt=prompt,
        model_key="fake",
        n_prompt_tokens=7,
        prefill_seconds=0.0,
    )
    assert not record.compliant
    assert record.z == pytest.approx(0.5)
    assert record.mass_ab < 0.05  # verdict mass collapses when argmax is elsewhere


def test_swap_pair_decomposition_identity():
    # s_i and b_i from two fake readouts reconstruct the raw z values exactly.
    item = make_item()
    z = {}
    for order, (la, lb) in {
        "chosen_first": (2.0, -1.0),
        "rejected_first": (1.5, 0.5),
    }.items():
        prompt = build_judge_prompt(item, order)
        record = logits_to_record(
            fake_logits(8, {0: la, 1: lb}),
            token_ids=(0, 1),
            argmax_token="A",
            prompt=prompt,
            model_key="fake",
            n_prompt_tokens=1,
            prefill_seconds=0.0,
        )
        z[order] = record.z
    s_i = (z["chosen_first"] - z["rejected_first"]) / 2
    b_i = (z["chosen_first"] + z["rejected_first"]) / 2
    assert z["chosen_first"] == pytest.approx(s_i + b_i)
    assert z["rejected_first"] == pytest.approx(b_i - s_i)


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------

def make_record(order: str = "chosen_first", item_id: str = "mt-bench-easy/7") -> JudgmentRecord:
    return JudgmentRecord(
        model="fake",
        rubric="minimal",
        order=order,
        item_id=item_id,
        expected_verdict="A" if order == "chosen_first" else "B",
        z=1.25,
        logp_a=-0.25,
        logp_b=-1.5,
        mass_ab=0.99,
        argmax_token="A",
        compliant=True,
        n_prompt_tokens=100,
        prefill_seconds=0.5,
    )


def test_store_roundtrip_and_resume_keys(tmp_path):
    store = ResultStore("fake", "minimal", results_dir=tmp_path)
    assert store.existing_keys() == set()
    r1 = make_record("chosen_first")
    r2 = make_record("rejected_first")
    store.append(r1)
    store.append(r2)
    assert store.existing_keys() == {r1.key, r2.key}
    loaded = store.load()
    assert loaded == [r1, r2]


def test_store_meta_written_as_json(tmp_path):
    store = ResultStore("fake", "minimal", results_dir=tmp_path)
    store.write_meta({"n_ctx": 2784, "model_key": "fake"})
    meta = json.loads(store.meta_path.read_text())
    assert meta["n_ctx"] == 2784


def test_load_records_concatenates_stores(tmp_path):
    s1 = ResultStore("fake", "minimal", results_dir=tmp_path)
    s2 = ResultStore("fake2", "minimal", results_dir=tmp_path)
    s1.append(make_record(item_id="mt-bench-easy/1"))
    s2.append(make_record(item_id="mt-bench-easy/2"))
    records = load_records(sorted(tmp_path.glob("*.jsonl")))
    assert {r.item_id for r in records} == {"mt-bench-easy/1", "mt-bench-easy/2"}


# ---------------------------------------------------------------------------
# Smoke test against the real model (skipped when the GGUF is absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not MODELS["qwen2.5-0.5b"].path.exists(),
    reason="pinned Qwen2.5-0.5B GGUF not downloaded",
)
def test_real_model_smoke():
    from src.judge import LlamaJudge

    judge = LlamaJudge(
        MODELS["qwen2.5-0.5b"], n_ctx=512, n_threads=2, verify_sha256=False
    )
    assert judge.verdict_token_ids == (32, 33)  # "A", "B" per the 2026-07-17 pilot
    record = judge.judge(build_judge_prompt(make_item(), "chosen_first"))
    assert record.n_prompt_tokens > 50
    assert 0 < record.mass_ab <= 1.0
    assert math.isfinite(record.z)

"""Tests for the experiment runner and aggregation (experiments/)."""

from __future__ import annotations

import gzip
import json

import pytest

from experiments.aggregate import (
    RunResult,
    check_aligned,
    diff_ci,
    load_raw,
    mean_ci,
    sort_key,
)
from experiments.run_grid import (
    GridConfig,
    make_chunker,
    make_retriever,
    run_and_save,
    run_config,
)
from experiments.summarize import find_baseline, render_summary
from src.data import Document, GoldSpan, QADataset, Question

PARA_ZEBRA = (
    "The zebra sanctuary opened in 1974 near the river delta. "
    "Its striped residents graze on imported savanna grass every morning."
)
PARA_VOLCANO = (
    "Volcanic eruptions shaped the northern caldera over millennia. "
    "Basalt columns line the crater rim where tourists gather at sunrise."
)
PARA_MARKET = (
    "The fish market auction begins before dawn with bluefin tuna. "
    "Wholesale prices are settled by hand signals among licensed brokers."
)


def _question(qid: str, doc: Document, text: str, answer: str) -> Question:
    start = doc.text.index(answer)
    return Question(
        qid=qid,
        text=text,
        doc_id=doc.doc_id,
        gold_alternatives=((GoldSpan(start, start + len(answer)),),),
    )


@pytest.fixture()
def tiny_dataset() -> tuple[QADataset, tuple[Question, ...]]:
    doc = Document(
        doc_id="tiny",
        title="tiny",
        text="\n\n".join([PARA_ZEBRA, PARA_VOLCANO, PARA_MARKET]),
    )
    questions = (
        _question("q1", doc, "When did the zebra sanctuary open?", "1974"),
        _question("q2", doc, "What lines the crater rim?", "Basalt columns"),
        _question("q3", doc, "What fish opens the market auction?", "bluefin tuna"),
    )
    dataset = QADataset(name="tiny", documents={doc.doc_id: doc}, questions=questions)
    return dataset, questions


def _config(**overrides) -> GridConfig:
    defaults = dict(
        dataset="tiny",
        chunker="fixed",
        chunk_size=32,
        overlap=0,
        retriever="bm25",
        budgets=(40, 80),
        hit_ks=(1, 3),
        per_doc_cap=50,
        seed=0,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestFactories:
    def test_recursive_rejects_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            make_chunker("recursive", 128, overlap=8)

    def test_unknown_chunker(self):
        with pytest.raises(ValueError, match="unknown chunker"):
            make_chunker("semantic", 128, 0)

    def test_unknown_retriever(self):
        with pytest.raises(ValueError, match="unknown retriever"):
            make_retriever("dense")

    def test_config_id_encodes_grid_point(self):
        cfg = _config(dataset="dev-v1.1", chunker="sentence", chunk_size=128)
        assert cfg.config_id == "dev-v1.1_sentence128_o0_bm25_cap50_seed0"


class TestRunConfig:
    def test_record_shape(self, tiny_dataset):
        dataset, questions = tiny_dataset
        result = run_config(_config(), dataset, questions)
        assert result["n_questions"] == len(questions)
        assert [r["qid"] for r in result["records"]] == ["q1", "q2", "q3"]
        for record in result["records"]:
            assert set(record["budgets"]) == {"40", "80"}
            assert set(record["hits"]) == {"1", "3"}
            for budget, cell in record["budgets"].items():
                assert 0.0 <= cell["recall"] <= 1.0
                assert 0.0 <= cell["precision"] <= 1.0
                assert cell["iou"] <= min(cell["recall"], cell["precision"]) + 1e-9
                assert cell["tokens"] <= int(budget)

    def test_retrieval_finds_distinctive_answers(self, tiny_dataset):
        # Each question's vocabulary uniquely matches its paragraph, so a
        # 32-token chunking with an 80-token budget must recover every answer.
        dataset, questions = tiny_dataset
        result = run_config(_config(), dataset, questions)
        for record in result["records"]:
            assert record["budgets"]["80"]["recall"] == 1.0
            assert record["hits"]["3"] is True

    def test_deterministic(self, tiny_dataset):
        dataset, questions = tiny_dataset
        a = run_config(_config(), dataset, questions)
        b = run_config(_config(), dataset, questions)
        assert a["records"] == b["records"]
        assert a["chunk_stats"] == b["chunk_stats"]

    def test_budget_smaller_than_chunk_yields_zero_tokens(self, tiny_dataset):
        # stop-before-exceed: if no chunk fits the budget, nothing is
        # retrieved and the record says so explicitly (tokens == 0).
        dataset, questions = tiny_dataset
        cfg = _config(chunk_size=60, budgets=(20,))
        result = run_config(cfg, dataset, questions)
        for record in result["records"]:
            cell = record["budgets"]["20"]
            assert cell["tokens"] == 0
            assert cell["chunks"] == 0
            assert cell["recall"] == 0.0


class TestRunAndSave:
    def test_write_skip_force(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        cfg = _config()
        path, ran = run_and_save(cfg, dataset, questions, tmp_path)
        assert ran and path.exists()
        _, ran_again = run_and_save(cfg, dataset, questions, tmp_path)
        assert not ran_again
        _, forced = run_and_save(cfg, dataset, questions, tmp_path, force=True)
        assert forced
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["config"]["chunker"] == "fixed"
        assert payload["meta"]["git_commit"]
        assert len(payload["records"]) == 3


class TestAggregate:
    def _saved_results(self, tiny_dataset, tmp_path) -> list[RunResult]:
        dataset, questions = tiny_dataset
        for chunker in ("fixed", "sentence"):
            run_and_save(_config(chunker=chunker), dataset, questions, tmp_path)
        return load_raw(tmp_path)

    def test_load_raw_roundtrip_and_order(self, tiny_dataset, tmp_path):
        results = self._saved_results(tiny_dataset, tmp_path)
        assert [rr.label for rr in results] == ["fixed-32", "sentence-32"]
        assert results == sorted(results, key=sort_key)
        assert results[0].qids() == ("q1", "q2", "q3")
        assert len(results[0].metric("recall", 80)) == 3

    def test_load_raw_filters(self, tiny_dataset, tmp_path):
        self._saved_results(tiny_dataset, tmp_path)
        assert load_raw(tmp_path, dataset="other") == []
        assert len(load_raw(tmp_path, retriever="bm25")) == 2

    def test_check_aligned_rejects_mismatch(self, tiny_dataset, tmp_path):
        results = self._saved_results(tiny_dataset, tmp_path)
        truncated = RunResult(
            config=results[1].config,
            meta=results[1].meta,
            chunk_stats=results[1].chunk_stats,
            records=results[1].records[:2],
        )
        with pytest.raises(ValueError, match="question sets differ"):
            check_aligned([results[0], truncated])

    def test_mean_ci_of_constant_is_degenerate(self):
        res = mean_ci([0.5] * 20)
        assert res.mean_diff == res.ci_low == res.ci_high == 0.5

    def test_diff_ci_pairing(self):
        # A constant +0.1 per-question advantage has a zero-width CI at +0.1;
        # an unpaired comparison of the same scores would not.
        base = [0.1 * i for i in range(10)]
        better = [s + 0.1 for s in base]
        res = diff_ci(better, base)
        assert res.mean_diff == pytest.approx(0.1)
        assert res.ci_low == pytest.approx(0.1)
        assert res.ci_high == pytest.approx(0.1)
        assert res.significant


class TestSummarize:
    def test_render_summary(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        for chunker in ("fixed", "sentence"):
            run_and_save(_config(chunker=chunker), dataset, questions, tmp_path)
        results = load_raw(tmp_path)
        text = render_summary(results, baseline_label="fixed-32")
        assert "## SpanRecall@B (mean)" in text
        assert "ΔSpanRecall vs fixed-32" in text
        assert "| fixed-32 | — | — |" in text  # baseline row has no self-diff
        assert "## Budget utilization" in text

    def test_missing_baseline(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        with pytest.raises(ValueError, match="baseline"):
            find_baseline(load_raw(tmp_path), "fixed-256")

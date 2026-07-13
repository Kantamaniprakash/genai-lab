"""Tests for the experiment runner and aggregation (experiments/)."""

from __future__ import annotations

import gzip
import json
import subprocess

import pytest

from experiments import run_grid
from experiments.aggregate import (
    RunResult,
    check_aligned,
    diff_ci,
    load_raw,
    mean_ci,
    sort_key,
)
from experiments.calibrate_matched import calibrate
from experiments.run_grid import (
    GridConfig,
    make_chunker,
    make_retriever,
    make_tokenizer,
    run_and_save,
    run_config,
)
from experiments.summarize import (
    find_baseline,
    pairwise_same_size_rows,
    render_summary,
)
from experiments.summarize_ablations import render_ablations
from experiments.summarize_chroma import render_moderation
from experiments.summarize_matched import match_by_realized_size, render_matched
from experiments.summarize_retrievers import render_retrievers
from experiments.summarize_seeds import render_seeds
from experiments.summarize_semantic import render_semantic
from experiments.summarize_tokenizers import render_tokenizers
from src.chunkers import SemanticChunker
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


def _rewrite_config_field(path, key: str, value) -> None:
    """Edit one config field of a saved result payload in place.

    Lets loader-level tests cover non-default fields (e.g. a BPE tokenizer)
    without needing the machinery that produces them at run time.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    payload["config"][key] = value
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


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

    def test_semantic_factory_builds_without_loading_an_encoder(self):
        # Construction must stay torch-free: the default encoder loads
        # lazily on first use, so the CI environment (no dense group) can
        # still expand grids and reject bad configs.
        chunker = make_chunker("semantic", 128, 0)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.stats["encoder"] is None

    def test_semantic_rejects_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            make_chunker("semantic", 128, overlap=8)

    def test_unknown_chunker(self):
        with pytest.raises(ValueError, match="unknown chunker"):
            make_chunker("agentic", 128, 0)

    def test_unknown_retriever(self):
        with pytest.raises(ValueError, match="unknown retriever"):
            make_retriever("splade")

    def test_config_id_encodes_grid_point(self):
        cfg = _config(dataset="dev-v1.1", chunker="sentence", chunk_size=128)
        assert cfg.config_id == "dev-v1.1_sentence128_o0_bm25_cap50_seed0"

    def test_config_id_encodes_nondefault_budget_rule(self):
        # The stop rule is omitted so pre-existing result filenames stay
        # valid; any other rule must be encoded or files would clobber.
        stop = _config(dataset="dev-v1.1")
        trunc = _config(dataset="dev-v1.1", budget_rule="truncate")
        assert stop.config_id == "dev-v1.1_fixed32_o0_bm25_cap50_seed0"
        assert trunc.config_id == "dev-v1.1_fixed32_o0_bm25_cap50_seed0_truncate"

    def test_config_id_encodes_nondefault_tokenizer(self):
        # Same backward-compatibility pattern as budget_rule: the regex
        # default is omitted, any other unit must be encoded.
        bpe = _config(dataset="dev-v1.1", tokenizer="cl100k")
        both = _config(dataset="dev-v1.1", tokenizer="cl100k", budget_rule="truncate")
        assert bpe.config_id == "dev-v1.1_fixed32_o0_bm25_cap50_seed0_cl100k"
        assert both.config_id == "dev-v1.1_fixed32_o0_bm25_cap50_seed0_truncate_cl100k"

    def test_make_tokenizer(self):
        from src.tokenization import RegexWordTokenizer

        assert isinstance(make_tokenizer("regex"), RegexWordTokenizer)
        with pytest.raises(ValueError, match="unknown tokenizer"):
            make_tokenizer("cl9000")


class TestRunMetadata:
    @pytest.fixture()
    def git_repo(self, tmp_path, monkeypatch):
        def git(*args: str) -> None:
            subprocess.run(
                ["git", *args], cwd=tmp_path, check=True, capture_output=True
            )

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text("x = 1\n")
        git("init")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "test")
        git("add", "-A")
        git("commit", "-m", "init")
        monkeypatch.setattr(run_grid, "ROOT", tmp_path)
        return tmp_path

    def test_clean_repo_records_plain_commit(self, git_repo):
        commit = run_grid.run_metadata()["git_commit"]
        assert len(commit) == 40 and not commit.endswith("+dirty")

    def test_result_files_do_not_mark_dirty(self, git_repo):
        # Earlier configs of a multi-config invocation leave result files in
        # the tree; they are outputs, not inputs, so the commit stays clean.
        raw = git_repo / "results" / "raw"
        raw.mkdir(parents=True)
        (raw / "some_config.json.gz").write_bytes(b"")
        assert not run_grid.run_metadata()["git_commit"].endswith("+dirty")

    def test_input_changes_mark_dirty(self, git_repo):
        (git_repo / "src" / "module.py").write_text("x = 2\n")
        assert run_grid.run_metadata()["git_commit"].endswith("+dirty")


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

    def test_truncate_rule_spends_full_budget(self, tiny_dataset):
        # Same oversized-chunk setup as above: under truncate the first
        # chunk is cut to exactly the budget instead of dropped.
        dataset, questions = tiny_dataset
        cfg = _config(chunk_size=60, budgets=(20,), budget_rule="truncate")
        result = run_config(cfg, dataset, questions)
        for record in result["records"]:
            cell = record["budgets"]["20"]
            assert cell["tokens"] == 20
            assert cell["chunks"] == 1


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

    def test_load_raw_sizes_filter(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(chunk_size=16), dataset, questions, tmp_path)
        run_and_save(_config(chunk_size=32), dataset, questions, tmp_path)
        run_and_save(_config(chunk_size=48), dataset, questions, tmp_path)
        # Default open: off-grid sizes load unless the caller pins the grid.
        assert len(load_raw(tmp_path)) == 3
        pinned = load_raw(tmp_path, sizes=(16, 32))
        assert [rr.label for rr in pinned] == ["fixed-16", "fixed-32"]

    def test_load_raw_budget_rule_and_overlap_filters(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        run_and_save(_config(budget_rule="truncate"), dataset, questions, tmp_path)
        run_and_save(_config(overlap=8), dataset, questions, tmp_path)
        assert len(load_raw(tmp_path)) == 3
        stop = load_raw(tmp_path, budget_rule="stop", overlap=0)
        assert [rr.label for rr in stop] == ["fixed-32"]
        trunc = load_raw(tmp_path, budget_rule="truncate")
        assert [rr.label for rr in trunc] == ["fixed-32/truncate"]
        assert [rr.label for rr in load_raw(tmp_path, overlap=8)] == ["fixed-32/o8"]

    def test_load_raw_seed_filter(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        run_and_save(_config(seed=1), dataset, questions, tmp_path)
        assert len(load_raw(tmp_path)) == 2
        assert [rr.config["seed"] for rr in load_raw(tmp_path, seed=1)] == [1]
        assert load_raw(tmp_path, seed=2) == []

    def test_load_raw_fills_missing_budget_rule(self, tiny_dataset, tmp_path):
        # Result files written before the budget_rule field existed are
        # stop-rule runs by construction; loading fills the key in.
        dataset, questions = tiny_dataset
        path, _ = run_and_save(_config(), dataset, questions, tmp_path)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        del payload["config"]["budget_rule"]
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        (rr,) = load_raw(tmp_path)
        assert rr.config["budget_rule"] == "stop"
        assert rr.label == "fixed-32"

    def test_load_raw_tokenizer_filter_defaults_closed(self, tiny_dataset, tmp_path):
        # Cross-unit runs share question ids with the primary grid, so
        # check_aligned cannot catch an accidental mix — the loader must
        # exclude them unless a caller asks explicitly.
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        path, _ = run_and_save(
            _config(chunker="sentence"), dataset, questions, tmp_path
        )
        _rewrite_config_field(path, "tokenizer", "cl100k")
        assert [rr.label for rr in load_raw(tmp_path)] == ["fixed-32"]
        bpe = load_raw(tmp_path, tokenizer="cl100k")
        assert [rr.label for rr in bpe] == ["sentence-32/cl100k"]
        assert len(load_raw(tmp_path, tokenizer=None)) == 2

    def test_load_raw_fills_missing_tokenizer(self, tiny_dataset, tmp_path):
        # Files written before the tokenizer field existed are regex-unit
        # runs by construction.
        dataset, questions = tiny_dataset
        path, _ = run_and_save(_config(), dataset, questions, tmp_path)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        del payload["config"]["tokenizer"]
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        (rr,) = load_raw(tmp_path)
        assert rr.config["tokenizer"] == "regex"
        assert rr.label == "fixed-32"

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

    def test_pairwise_same_size_rows(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        for chunker in ("fixed", "sentence"):
            run_and_save(_config(chunker=chunker), dataset, questions, tmp_path)
        run_and_save(_config(overlap=8), dataset, questions, tmp_path)
        rows = pairwise_same_size_rows(load_raw(tmp_path), budgets=[40, 80])
        # Only zero-overlap stop-rule runs pair up: one comparison, one cell
        # per budget.
        assert len(rows) == 1
        assert rows[0][0] == "sentence-32 vs fixed-32"
        assert len(rows[0]) == 3


class TestSummarizeChroma:
    """Chroma-style fixture: qids carry a `corpus:` prefix, gold lengths vary."""

    def _chroma_like(self) -> tuple[QADataset, tuple[Question, ...]]:
        doc = Document(
            doc_id="tiny",
            title="tiny",
            text="\n\n".join([PARA_ZEBRA, PARA_VOLCANO, PARA_MARKET]),
        )
        long_answer = "Wholesale prices are settled by hand signals"
        questions = (
            _question("finance:001", doc, "When did the sanctuary open?", "1974"),
            _question("finance:002", doc, "What lines the rim?", "Basalt columns"),
            _question("chatlogs:001", doc, "How are prices settled?", long_answer),
        )
        dataset = QADataset(name="chroma", documents={doc.doc_id: doc}, questions=questions)
        return dataset, questions

    def _stats(self, dataset) -> dict[str, tuple[int, int]]:
        from src.tokenization import RegexWordTokenizer, TokenIndex

        index = TokenIndex(dataset.documents["tiny"].text, RegexWordTokenizer())
        return {
            q.qid: (
                len(
                    {
                        t
                        for s in q.gold_alternatives[0]
                        for t in index.tokens_overlapping(s.start, s.end)
                    }
                ),
                len(q.gold_alternatives[0]),
            )
            for q in dataset.questions
        }

    def test_render_moderation_sections_and_groups(self, tmp_path):
        dataset, questions = self._chroma_like()
        for size in (16, 32):
            run_and_save(
                _config(dataset="chroma", chunk_size=size), dataset, questions, tmp_path
            )
        results = load_raw(tmp_path)
        text = render_moderation(results, self._stats(dataset), "fixed-16", "fixed-32")
        assert "## Gold evidence by corpus" in text
        # Unknown-to-Chroma corpus names from the fixture still render, and
        # per-corpus question counts come from the qid prefixes.
        assert "| chatlogs | 1 |" in text
        assert "| finance | 2 |" in text
        assert "chatlogs (n=1)" in text and "finance (n=2)" in text
        assert "## Moderation by total gold-evidence length" in text
        assert "## Moderation by reference count" in text
        assert "1 reference (n=3)" in text
        # Jackknife: one pooled row plus one drop-one row per corpus, and
        # dropping a corpus removes exactly its questions from the pool.
        assert "## Corpus jackknife" in text
        assert "all corpora (n=3)" in text
        assert "without chatlogs (n=2)" in text
        assert "without finance (n=1)" in text

    def test_render_moderation_rejects_unknown_qids(self, tmp_path):
        dataset, questions = self._chroma_like()
        for size in (16, 32):
            run_and_save(
                _config(dataset="chroma", chunk_size=size), dataset, questions, tmp_path
            )
        results = load_raw(tmp_path)
        stats = self._stats(dataset)
        stats.pop("chatlogs:001")
        with pytest.raises(SystemExit, match="out of sync"):
            render_moderation(results, stats, "fixed-16", "fixed-32")


class TestSummarizeAblations:
    def test_render_ablations(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        run_and_save(_config(overlap=8), dataset, questions, tmp_path)
        run_and_save(_config(budget_rule="truncate"), dataset, questions, tmp_path)
        text = render_ablations("tiny", "bm25", tmp_path)
        assert "## Overlap ablation" in text
        assert "fixed-32/o8" in text
        assert "## Budget rule" in text
        assert "ΔSpanRecall, truncate − stop" in text
        # No sentence-o0 run at a size with a fixed+25% partner: the
        # cross-family control has nothing to pair and must stay silent.
        assert "## Cross-family control" not in text

    def test_render_ablations_cross_family_pairing(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        run_and_save(_config(overlap=8), dataset, questions, tmp_path)
        run_and_save(_config(chunker="sentence"), dataset, questions, tmp_path)
        text = render_ablations("tiny", "bm25", tmp_path)
        assert "## Cross-family control" in text
        assert "sentence-32 − fixed-32/o8" in text

    def test_render_ablations_requires_ablation_runs(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(), dataset, questions, tmp_path)
        with pytest.raises(SystemExit, match="no ablation results"):
            render_ablations("tiny", "bm25", tmp_path)


class TestRetrieverGrid:
    def test_all_retrievers_run_and_score(self, tiny_dataset):
        # Each question's vocabulary uniquely matches its paragraph, so
        # every retriever must recover every answer under the loose budget.
        dataset, questions = tiny_dataset
        for retriever in ("bm25", "tfidf", "lsa"):
            result = run_config(_config(retriever=retriever), dataset, questions)
            for record in result["records"]:
                assert record["budgets"]["80"]["recall"] == 1.0, retriever

    def test_lsa_records_realized_ranks(self, tiny_dataset):
        dataset, questions = tiny_dataset
        result = run_config(_config(retriever="lsa"), dataset, questions)
        stats = result["retriever_stats"]
        assert stats["n_docs"] == 1
        # The tiny document chunks into few pieces, so the rank must have
        # been capped by the data, not by n_components.
        assert 1 <= stats["realized_rank_max"] < stats["n_components"]
        assert stats["n_docs_data_bounded"] == 1

    def test_non_lsa_runs_have_no_retriever_stats(self, tiny_dataset):
        dataset, questions = tiny_dataset
        assert "retriever_stats" not in run_config(_config(), dataset, questions)

    def test_retriever_stats_roundtrip_through_load_raw(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(retriever="lsa"), dataset, questions, tmp_path)
        run_and_save(_config(), dataset, questions, tmp_path)
        by = {rr.config["retriever"]: rr for rr in load_raw(tmp_path)}
        assert by["lsa"].retriever_stats is not None
        assert by["bm25"].retriever_stats is None


class TestSummarizeRetrievers:
    def _save_grid(self, tiny_dataset, tmp_path, retrievers=("bm25", "tfidf", "lsa")):
        dataset, questions = tiny_dataset
        for retriever in retrievers:
            for chunker in ("fixed", "sentence"):
                run_and_save(
                    _config(retriever=retriever, chunker=chunker),
                    dataset,
                    questions,
                    tmp_path,
                )

    def test_render_retrievers(self, tiny_dataset, tmp_path):
        self._save_grid(tiny_dataset, tmp_path)
        text = render_retrievers("tiny", ["bm25", "tfidf", "lsa"], tmp_path)
        assert "## SpanRecall@80 (mean) by retriever" in text
        assert "ΔSpanRecall, tfidf − bm25" in text
        assert "ΔSpanRecall, lsa − bm25" in text
        assert "## LSA realized latent rank" in text
        assert "fixed-32 (lsa)" in text

    def test_missing_retriever_exits(self, tiny_dataset, tmp_path):
        self._save_grid(tiny_dataset, tmp_path, retrievers=("bm25",))
        with pytest.raises(SystemExit, match="no baseline-grid results"):
            render_retrievers("tiny", ["bm25", "tfidf"], tmp_path)

    def test_mismatched_grids_exit(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        self._save_grid(tiny_dataset, tmp_path, retrievers=("bm25",))
        run_and_save(_config(retriever="tfidf"), dataset, questions, tmp_path)
        with pytest.raises(SystemExit, match="different grids"):
            render_retrievers("tiny", ["bm25", "tfidf"], tmp_path)

    def test_unknown_reference_exits(self, tiny_dataset, tmp_path):
        self._save_grid(tiny_dataset, tmp_path)
        with pytest.raises(SystemExit, match="reference retriever"):
            render_retrievers("tiny", ["bm25", "tfidf"], tmp_path, reference="lsa")


class TestSummarizeSeeds:
    def _save_seed_grids(self, tiny_dataset, tmp_path, seeds=(0, 1)):
        # Sizes 64/256 so the fixed-64 - fixed-256 headline pair exists.
        dataset, questions = tiny_dataset
        for seed in seeds:
            for size in (64, 256):
                run_and_save(
                    _config(chunk_size=size, seed=seed), dataset, questions, tmp_path
                )

    def test_render_seeds(self, tiny_dataset, tmp_path):
        self._save_seed_grids(tiny_dataset, tmp_path)
        text = render_seeds("tiny", "bm25", [0, 1], tmp_path, budget=40)
        assert "## SpanRecall@40 (mean) by seed" in text
        assert "fixed-64 − fixed-256" in text
        assert "seed 0" in text and "seed 1" in text
        assert "max−min" in text

    def test_missing_seed_exits(self, tiny_dataset, tmp_path):
        self._save_seed_grids(tiny_dataset, tmp_path, seeds=(0,))
        with pytest.raises(SystemExit, match="no baseline-grid results for seed 1"):
            render_seeds("tiny", "bm25", [0, 1], tmp_path, budget=40)

    def test_mismatched_seed_grids_exit(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        self._save_seed_grids(tiny_dataset, tmp_path, seeds=(0,))
        run_and_save(_config(chunk_size=64, seed=1), dataset, questions, tmp_path)
        with pytest.raises(SystemExit, match="different grids"):
            render_seeds("tiny", "bm25", [0, 1], tmp_path, budget=40)

    def test_budget_must_be_in_grid(self, tiny_dataset, tmp_path):
        self._save_seed_grids(tiny_dataset, tmp_path)
        with pytest.raises(SystemExit, match="budget 999"):
            render_seeds("tiny", "bm25", [0, 1], tmp_path, budget=999)


class TestRunConfigCl100k:
    """A real BPE-unit run end to end (skips without tiktoken/its vocab)."""

    @pytest.fixture()
    def bpe_available(self):
        pytest.importorskip("tiktoken")
        try:
            make_tokenizer("cl100k")
        except Exception as exc:  # vocabulary download needs network access
            pytest.skip(f"cl100k_base vocabulary unavailable: {exc}")

    def test_budgets_counted_in_bpe_tokens(self, bpe_available, tiny_dataset):
        dataset, questions = tiny_dataset
        result = run_config(_config(tokenizer="cl100k"), dataset, questions)
        assert result["config"]["tokenizer"] == "cl100k"
        # The lazily imported unit's version is part of reproducibility.
        assert "tiktoken" in result["meta"]
        for record in result["records"]:
            for budget, cell in record["budgets"].items():
                assert cell["tokens"] <= int(budget)
        # Distinctive per-paragraph vocabulary: the zebra question must
        # recover its answer within the 80-BPE-token budget, same as the
        # regex-unit expectation.
        assert result["records"][0]["budgets"]["80"]["recall"] == 1.0

    def test_deterministic(self, bpe_available, tiny_dataset):
        dataset, questions = tiny_dataset
        cfg = _config(tokenizer="cl100k")
        a = run_config(cfg, dataset, questions)
        b = run_config(cfg, dataset, questions)
        assert a["records"] == b["records"]
        assert a["chunk_stats"] == b["chunk_stats"]


class TestSummarizeTokenizers:
    def _save_unit_grids(self, tiny_dataset, tmp_path, bpe_points=("fixed", "sentence")):
        # Regex-unit files come from real runs; the cl100k-unit files are
        # byte-copies relabeled at the config level, which exercises every
        # loader/renderer path without needing the BPE vocabulary.
        dataset, questions = tiny_dataset
        for chunker in ("fixed", "sentence"):
            path, _ = run_and_save(_config(chunker=chunker), dataset, questions, tmp_path)
            if chunker in bpe_points:
                clone = path.with_name(path.name.replace(".json.gz", "_cl100k.json.gz"))
                clone.write_bytes(path.read_bytes())
                _rewrite_config_field(clone, "tokenizer", "cl100k")

    def test_render_tokenizers(self, tiny_dataset, tmp_path):
        self._save_unit_grids(tiny_dataset, tmp_path)
        text = render_tokenizers("tiny", "bm25", 0, tmp_path, hit_k=3)
        assert "# Tokenizer robustness — tiny, bm25" in text
        assert "## Unit conversion and realized chunk sizes" in text
        assert "## SpanRecall@B (mean) by unit" in text
        assert "## Size ordering (fixed family): adjacent-size paired steps" in text
        assert "cl100k_base BPE" in text
        assert "fixed-32" in text
        assert "## hit@3 by unit" in text

    def test_missing_unit_exits(self, tiny_dataset, tmp_path):
        self._save_unit_grids(tiny_dataset, tmp_path, bpe_points=())
        with pytest.raises(SystemExit, match="no baseline-grid results for tokenizer"):
            render_tokenizers("tiny", "bm25", 0, tmp_path)

    def test_mismatched_grid_points_exit(self, tiny_dataset, tmp_path):
        self._save_unit_grids(tiny_dataset, tmp_path, bpe_points=("fixed",))
        with pytest.raises(SystemExit, match="different grid points"):
            render_tokenizers("tiny", "bm25", 0, tmp_path)


class _StubEncoder:
    """Constant-embedding encoder for plumbing tests: no breakpoints ever
    fire (semantic degenerates to sentence packing, which the chunker tests
    verify directly), so these tests exercise persistence and rendering
    without torch. ``model_name`` checks that stats prefer it over the
    class name, matching the real SentenceTransformerEncoder."""

    model_name = "stub-encoder"

    def encode(self, texts):
        import numpy as np

        return np.full((len(texts), 4), 0.5, dtype=np.float32)


class TestSemanticGridAndSummary:
    @pytest.fixture(autouse=True)
    def _stub_default_encoder(self, monkeypatch):
        # make_chunker builds SemanticChunker without an encoder; the lazy
        # default resolves through src.dense.default_encoder at first use.
        monkeypatch.setattr("src.dense.default_encoder", lambda: _StubEncoder())

    def test_semantic_run_records_chunker_stats(self, tiny_dataset):
        dataset, questions = tiny_dataset
        result = run_config(_config(chunker="semantic"), dataset, questions)
        stats = result["chunker_stats"]
        assert stats["encoder"] == "stub-encoder"
        assert stats["percentile"] == 95.0
        assert stats["n_documents"] == 1
        assert stats["n_gaps"] == stats["n_sentences"] - 1
        assert stats["n_breakpoints"] == 0  # constant embeddings never break

    def test_structural_runs_have_no_chunker_stats(self, tiny_dataset):
        dataset, questions = tiny_dataset
        assert "chunker_stats" not in run_config(_config(), dataset, questions)

    def test_chunker_stats_roundtrip_through_load_raw(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(chunker="semantic"), dataset, questions, tmp_path)
        run_and_save(_config(chunker="fixed"), dataset, questions, tmp_path)
        by_chunker = {rr.config["chunker"]: rr for rr in load_raw(tmp_path)}
        assert by_chunker["semantic"].chunker_stats["encoder"] == "stub-encoder"
        assert by_chunker["fixed"].chunker_stats is None

    def test_render_semantic(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        for chunker in ("fixed", "sentence", "semantic"):
            run_and_save(_config(chunker=chunker), dataset, questions, tmp_path)
        results = load_raw(tmp_path)
        text = render_semantic(results)
        assert "# Semantic chunker comparison — tiny, bm25" in text
        assert "## ΔSpanRecall: semantic − sentence at matched nominal size" in text
        assert "## ΔSpanRecall: semantic − fixed at matched nominal size" in text
        assert "stub-encoder @ p95" in text
        assert "semantic-32" in text

    def test_render_semantic_requires_semantic_runs(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(chunker="fixed"), dataset, questions, tmp_path)
        with pytest.raises(ValueError, match="no semantic-chunker runs"):
            render_semantic(load_raw(tmp_path))


class TestCalibrateMatched:
    # Twelve sentences of exactly 6 regex tokens each (5 words + period), so
    # zero-overlap sentence packing at max_tokens m puts floor(m/6) sentences
    # in every chunk and the realized mean is a step function computable by
    # hand: m in [6, 11] -> 6.0, [12, 17] -> 12.0, [18, 23] -> 18.0, ...
    UNIFORM_DOC = " ".join(
        f"Alpha{i} beta{i} gamma{i} delta{i} epsilon{i}." for i in range(12)
    )

    def test_finds_exact_target(self):
        best, achieved = calibrate([self.UNIFORM_DOC], target=18.0, hi=24)
        assert achieved == pytest.approx(18.0)
        assert 18 <= best <= 23

    def test_picks_closer_neighbor_below_target(self):
        # Target 12.5 sits between achievable means 12.0 and 18.0; the
        # predecessor is closer.
        best, achieved = calibrate([self.UNIFORM_DOC], target=12.5, hi=24)
        assert achieved == pytest.approx(12.0)
        assert 12 <= best <= 17

    def test_unreachable_target_clamps_to_hi(self):
        best, achieved = calibrate([self.UNIFORM_DOC], target=100.0, hi=24)
        assert best == 24
        assert achieved == pytest.approx(24.0)

    def test_realized_mean_monotone_over_range(self):
        # The binary search's precondition, checked directly on a messy
        # multi-document corpus with non-uniform sentence lengths.
        docs = [PARA_ZEBRA + " " + PARA_VOLCANO, PARA_MARKET]
        means = [calibrate(docs, target=0.1, hi=m)[1] for m in range(8, 64, 5)]
        assert all(a <= b + 1e-9 for a, b in zip(means, means[1:], strict=False))


def _fake_run(chunker: str, size: int, tokens_mean: float) -> RunResult:
    """Minimal aligned RunResult for pairing tests (2 questions, 1 budget)."""
    records = tuple(
        {
            "qid": f"q{i}",
            "doc_id": "d",
            "budgets": {"40": {"recall": 0.5, "precision": 0.1, "iou": 0.1,
                               "chunks": 1, "tokens": 30}},
            "hits": {"1": True, "3": True},
        }
        for i in range(2)
    )
    return RunResult(
        config={
            "dataset": "tiny", "chunker": chunker, "chunk_size": size,
            "overlap": 0, "retriever": "bm25", "budgets": (40,),
            "hit_ks": (1, 3), "per_doc_cap": 50, "seed": 0,
            "budget_rule": "stop", "tokenizer": "regex",
        },
        meta={},
        chunk_stats={"n_chunks": 10, "tokens_min": 1, "tokens_median": int(tokens_mean),
                     "tokens_mean": tokens_mean, "tokens_max": size},
        records=records,
    )


class TestMatchByRealizedSize:
    def test_pairs_drifted_run_with_calibrated_partner(self):
        semantic = _fake_run("semantic", 512, 314.0)
        nominal = _fake_run("sentence", 512, 475.0)
        calibrated = _fake_run("sentence", 341, 315.0)
        pairs = match_by_realized_size([semantic, nominal, calibrated])
        assert len(pairs) == 1
        assert pairs[0].nominal is nominal
        assert pairs[0].matched is calibrated
        assert pairs[0].well_matched

    def test_already_matched_run_pairs_with_its_nominal_partner(self):
        semantic = _fake_run("semantic", 64, 46.0)
        nominal = _fake_run("sentence", 64, 47.6)
        pairs = match_by_realized_size([semantic, nominal])
        assert pairs[0].matched is nominal
        assert pairs[0].well_matched

    def test_distant_nearest_partner_is_flagged(self):
        semantic = _fake_run("semantic", 512, 314.0)
        nominal = _fake_run("sentence", 512, 475.0)
        pairs = match_by_realized_size([semantic, nominal])
        assert pairs[0].matched is nominal
        assert not pairs[0].well_matched

    def test_ties_break_to_smaller_nominal_size(self):
        semantic = _fake_run("semantic", 128, 100.0)
        lo = _fake_run("sentence", 110, 98.0)
        hi = _fake_run("sentence", 118, 102.0)
        nominal = _fake_run("sentence", 128, 109.0)
        pairs = match_by_realized_size([semantic, nominal, lo, hi])
        assert pairs[0].matched is lo

    def test_requires_semantic_and_sentence_runs(self):
        with pytest.raises(ValueError, match="no semantic-chunker runs"):
            match_by_realized_size([_fake_run("sentence", 64, 50.0)])
        with pytest.raises(ValueError, match="no sentence-chunker runs"):
            match_by_realized_size([_fake_run("semantic", 64, 50.0)])

    def test_requires_same_nominal_partner(self):
        semantic = _fake_run("semantic", 512, 314.0)
        other = _fake_run("sentence", 341, 315.0)
        with pytest.raises(ValueError, match="no sentence run at the same"):
            match_by_realized_size([semantic, other])


class TestRenderMatched:
    @pytest.fixture(autouse=True)
    def _stub_default_encoder(self, monkeypatch):
        monkeypatch.setattr("src.dense.default_encoder", lambda: _StubEncoder())

    def test_render_matched(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        for chunker, size in (("sentence", 24), ("sentence", 32), ("semantic", 32)):
            run_and_save(
                _config(chunker=chunker, chunk_size=size), dataset, questions, tmp_path
            )
        results = load_raw(tmp_path)
        text = render_matched(results)
        assert "# Matched-realized-size comparison — tiny, bm25" in text
        assert "## Pairings (realized mean chunk size, regex tokens)" in text
        assert "matched *realized* size" in text
        assert "## Per-question dispersion at matched *realized* size" in text
        # The stub encoder never fires a breakpoint, so semantic-32 IS
        # sentence packing: its realized mean equals sentence-32's and the
        # realized-size partner is the nominal partner.
        assert "sentence-32 @" in text

    def test_render_matched_requires_semantic(self, tiny_dataset, tmp_path):
        dataset, questions = tiny_dataset
        run_and_save(_config(chunker="sentence"), dataset, questions, tmp_path)
        with pytest.raises(ValueError, match="no semantic-chunker runs"):
            render_matched(load_raw(tmp_path))

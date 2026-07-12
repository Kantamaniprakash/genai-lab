"""Tests for the dense retriever (src/dense.py).

The whole module is skipped when the optional dense stack is absent — CI
installs only the default dependency groups, so these run locally where
``uv sync --group dense`` has been executed and the encoder weights are
available. Everything here uses the real all-MiniLM-L6-v2 model: the
properties under test (normalization, determinism, truncation exposure)
are properties of the actual inference path, not of a mock.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from experiments.run_grid import GridConfig, make_retriever, run_and_save, run_config
from experiments.summarize_retrievers import render_retrievers
from src.data import Document, GoldSpan, QADataset, Question
from src.dense import DenseRetriever, SentenceTransformerEncoder, default_encoder

CAT = "The tabby cat dozed on the warm windowsill all afternoon."
VOLCANO = "Molten basalt poured from the volcano's fissure during the eruption."
FINANCE = "The central bank raised interest rates to curb rising inflation."


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformerEncoder:
    return default_encoder()


class TestEncoder:
    def test_embeddings_normalized_and_aligned(self, encoder):
        vectors = encoder.encode([CAT, VOLCANO, FINANCE])
        assert vectors.shape == (3, 384)
        assert vectors.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
        # Rows align with input order: re-encoding a permutation permutes rows.
        permuted = encoder.encode([FINANCE, CAT, VOLCANO])
        np.testing.assert_array_equal(permuted[1], vectors[0])

    def test_memoization_is_bitwise(self, encoder):
        first = encoder.encode([CAT])[0]
        second = encoder.encode([CAT])[0]
        np.testing.assert_array_equal(first, second)

    def test_duplicates_share_one_vector(self, encoder):
        vectors = encoder.encode([CAT, CAT])
        np.testing.assert_array_equal(vectors[0], vectors[1])

    def test_token_count_ignores_model_window(self, encoder):
        long_text = " ".join(["curious"] * 400)
        assert encoder.token_count(long_text) > encoder.max_seq_length
        assert encoder.token_count(CAT) < encoder.max_seq_length

    def test_rejects_bad_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            SentenceTransformerEncoder(batch_size=0)


class TestDenseRetriever:
    def test_semantic_ranking(self, encoder):
        retriever = DenseRetriever(encoder).fit([VOLCANO, FINANCE, CAT])
        assert retriever.rank("Where did the cat sleep?")[0] == 2
        assert retriever.rank("What did the volcano erupt?")[0] == 0

    def test_scores_are_cosines_in_range(self, encoder):
        retriever = DenseRetriever(encoder).fit([CAT, VOLCANO])
        scores = retriever.scores(CAT)
        assert scores[0] == pytest.approx(1.0, abs=1e-4)
        assert all(-1.0001 <= s <= 1.0001 for s in scores)

    def test_refit_is_deterministic(self, encoder):
        # Same environment, same batching -> identical scores across fits.
        corpus = [CAT, VOLCANO, FINANCE]
        query = "How were interest rates changed?"
        a = DenseRetriever(encoder).fit(corpus).scores(query)
        b = DenseRetriever(encoder).fit(corpus).scores(query)
        assert a == b

    def test_identical_texts_tie_break_by_index(self, encoder):
        retriever = DenseRetriever(encoder).fit([CAT, CAT, VOLCANO])
        assert retriever.rank("Where did the cat sleep?")[:2] == [0, 1]

    def test_truncation_exposure_counted(self, encoder):
        long_text = " ".join(["curious"] * 400)
        retriever = DenseRetriever(encoder).fit([CAT, long_text])
        assert retriever.n_texts == 2
        assert retriever.n_truncated == 1

    def test_empty_corpus_rejected(self, encoder):
        with pytest.raises(ValueError, match="empty corpus"):
            DenseRetriever(encoder).fit([])

    def test_scores_require_fit(self, encoder):
        with pytest.raises(ValueError, match="fit"):
            DenseRetriever(encoder).scores("anything")

    def test_factory_shares_the_process_encoder(self):
        first = make_retriever("dense")
        second = make_retriever("dense")
        assert isinstance(first, DenseRetriever)
        assert first._encoder is second._encoder is default_encoder()


def _tiny_dataset() -> tuple[QADataset, tuple[Question, ...]]:
    text = "\n\n".join([CAT, VOLCANO, FINANCE])
    doc = Document(doc_id="tiny", title="tiny", text=text)
    answer = "windowsill"
    start = text.index(answer)
    questions = (
        Question(
            qid="q1",
            text="Where did the cat sleep?",
            doc_id="tiny",
            gold_alternatives=((GoldSpan(start, start + len(answer)),),),
        ),
    )
    dataset = QADataset(name="tiny", documents={"tiny": doc}, questions=questions)
    return dataset, questions


def _config(**overrides) -> GridConfig:
    defaults = dict(
        dataset="tiny",
        chunker="fixed",
        chunk_size=16,
        overlap=0,
        retriever="dense",
        budgets=(20,),
        hit_ks=(1,),
        per_doc_cap=50,
        seed=0,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestGridIntegration:
    def test_run_config_records_encoder_stats_and_versions(self):
        dataset, questions = _tiny_dataset()
        result = run_config(_config(), dataset, questions)
        stats = result["retriever_stats"]
        assert stats["model"].endswith("all-MiniLM-L6-v2")
        assert stats["max_seq_length"] == 256
        assert stats["n_chunks"] > 0
        assert stats["n_chunks_truncated"] == 0  # tiny corpus, short chunks
        assert "torch" in result["meta"]
        assert "sentence_transformers" in result["meta"]

    def test_cross_retriever_summary_renders_truncation_section(self, tmp_path):
        dataset, questions = _tiny_dataset()
        for retriever in ("bm25", "dense"):
            run_and_save(_config(retriever=retriever), dataset, questions, tmp_path)
        text = render_retrievers("tiny", ["bm25", "dense"], tmp_path)
        assert "Dense encoder truncation exposure" in text
        assert "all-MiniLM-L6-v2" in text

    def test_semantic_chunker_run_records_real_encoder_stats(self):
        # The semantic chunker resolves its default encoder through the same
        # process-wide MiniLM instance as the dense retriever; a grid run
        # must persist the encoder identity and segmentation exposure.
        dataset, questions = _tiny_dataset()
        result = run_config(
            _config(chunker="semantic", retriever="bm25"), dataset, questions
        )
        stats = result["chunker_stats"]
        assert stats["encoder"].endswith("all-MiniLM-L6-v2")
        assert stats["n_sentences"] == 3  # CAT / VOLCANO / FINANCE
        assert stats["n_gaps"] == 2
        assert 0 <= stats["n_breakpoints"] <= stats["n_gaps"]
        assert stats["sentences_over_window"] == 0
        assert "torch" in result["meta"]

"""Tests for SQuAD loading: reconstruction, span remapping, dedup, sampling.

The loader's contract is exactness — every gold span must match the document
text verbatim — so most tests build a small synthetic SQuAD file and check
coordinates by hand. An integration test runs against the real dev sets when
they have been downloaded (`python -m src.data`); it is skipped otherwise so
the suite passes on a fresh clone.
"""

import json
from pathlib import Path

import pytest

from src.data import (
    GoldSpan,
    PARAGRAPH_JOINER,
    Question,
    load_squad,
    sample_questions,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PARA_A1 = "The quick brown fox jumps over the lazy dog."
PARA_A2 = "Foxes are omnivorous mammals belonging to the dog family."
PARA_B1 = "Mount Everest is Earth's highest mountain above sea level."


def _qa(qid, question, answers, impossible=False):
    qa = {"id": qid, "question": question, "answers": answers}
    if impossible:
        qa["is_impossible"] = True
    return qa


def _answer(context, text, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = context.index(text, start + 1)
    return {"answer_start": start, "text": text}


def make_squad_file(tmp_path, articles):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps({"version": "test", "data": articles}))
    return path


@pytest.fixture
def dataset(tmp_path):
    articles = [
        {
            "title": "Fox",
            "paragraphs": [
                {
                    "context": PARA_A1,
                    "qas": [
                        # Three identical annotations collapse to one span.
                        _qa("q1", "What jumps?", [_answer(PARA_A1, "fox")] * 3),
                    ],
                },
                {
                    "context": PARA_A2,
                    "qas": [
                        _qa("q2", "What family?", [_answer(PARA_A2, "dog family")]),
                        # Duplicate question text within the article: dropped.
                        _qa("q3", "What jumps?", [_answer(PARA_A2, "Foxes")]),
                        # Distinct annotated spans: two alternatives.
                        _qa(
                            "q4",
                            "What kind of mammal?",
                            [
                                _answer(PARA_A2, "omnivorous"),
                                _answer(PARA_A2, "omnivorous mammals"),
                            ],
                        ),
                    ],
                },
            ],
        },
        {
            "title": "Everest",
            "paragraphs": [
                {
                    "context": PARA_B1,
                    "qas": [
                        _qa("q5", "Highest what?", [_answer(PARA_B1, "mountain")]),
                        _qa("q6", "Unanswerable?", [], impossible=True),
                    ],
                },
            ],
        },
    ]
    return load_squad(make_squad_file(tmp_path, articles))


class TestLoadSquad:
    def test_documents_join_paragraphs(self, dataset):
        assert dataset.documents["Fox"].text == PARA_A1 + PARAGRAPH_JOINER + PARA_A2
        assert dataset.documents["Everest"].text == PARA_B1

    def test_kept_question_ids(self, dataset):
        # q3 is a duplicate question text, q6 is unanswerable: both dropped.
        assert [q.qid for q in dataset.questions] == ["q1", "q2", "q4", "q5"]

    def test_every_span_matches_document_text(self, dataset):
        for q in dataset.questions:
            text = dataset.documents[q.doc_id].text
            for alternative in q.gold_alternatives:
                for span in alternative:
                    assert text[span.start : span.end]

    def test_first_paragraph_span_kept_verbatim(self, dataset):
        (q1,) = [q for q in dataset.questions if q.qid == "q1"]
        ((span,),) = q1.gold_alternatives
        assert dataset.documents["Fox"].text[span.start : span.end] == "fox"

    def test_second_paragraph_span_remapped(self, dataset):
        (q2,) = [q for q in dataset.questions if q.qid == "q2"]
        ((span,),) = q2.gold_alternatives
        expected_start = len(PARA_A1) + len(PARAGRAPH_JOINER) + PARA_A2.index(
            "dog family"
        )
        assert (span.start, span.end) == (
            expected_start,
            expected_start + len("dog family"),
        )
        assert dataset.documents["Fox"].text[span.start : span.end] == "dog family"

    def test_identical_annotations_collapse(self, dataset):
        (q1,) = [q for q in dataset.questions if q.qid == "q1"]
        assert len(q1.gold_alternatives) == 1

    def test_distinct_annotations_become_alternatives(self, dataset):
        (q4,) = [q for q in dataset.questions if q.qid == "q4"]
        assert len(q4.gold_alternatives) == 2
        text = dataset.documents["Fox"].text
        alt_texts = {
            text[alt[0].start : alt[0].end] for alt in q4.gold_alternatives
        }
        assert alt_texts == {"omnivorous", "omnivorous mammals"}

    def test_corrupt_offset_raises(self, tmp_path):
        articles = [
            {
                "title": "Bad",
                "paragraphs": [
                    {
                        "context": PARA_A1,
                        "qas": [
                            _qa("q1", "?", [{"answer_start": 0, "text": "quick"}])
                        ],
                    }
                ],
            }
        ]
        with pytest.raises(ValueError, match="span mismatch"):
            load_squad(make_squad_file(tmp_path, articles))


class TestGoldSpanValidation:
    def test_empty_span_rejected(self):
        with pytest.raises(ValueError):
            GoldSpan(5, 5)

    def test_question_requires_gold(self):
        with pytest.raises(ValueError):
            Question(qid="q", text="?", doc_id="d", gold_alternatives=())


class TestSampleQuestions:
    def _dataset_with(self, tmp_path, counts):
        articles = []
        for doc_i, n in enumerate(counts):
            context = PARA_A1
            articles.append(
                {
                    "title": f"Doc{doc_i}",
                    "paragraphs": [
                        {
                            "context": context,
                            "qas": [
                                _qa(
                                    f"d{doc_i}q{i}",
                                    f"Question {i}?",
                                    [_answer(context, "fox")],
                                )
                                for i in range(n)
                            ],
                        }
                    ],
                }
            )
        return load_squad(make_squad_file(tmp_path, articles))

    def test_cap_applies_per_document(self, tmp_path):
        ds = self._dataset_with(tmp_path, [10, 3])
        sampled = sample_questions(ds, per_doc_cap=5, seed=0)
        by_doc = {}
        for q in sampled:
            by_doc[q.doc_id] = by_doc.get(q.doc_id, 0) + 1
        assert by_doc == {"Doc0": 5, "Doc1": 3}

    def test_deterministic_across_calls(self, tmp_path):
        ds = self._dataset_with(tmp_path, [10, 10])
        a = sample_questions(ds, per_doc_cap=4, seed=7)
        b = sample_questions(ds, per_doc_cap=4, seed=7)
        assert [q.qid for q in a] == [q.qid for q in b]

    def test_seed_changes_selection(self, tmp_path):
        ds = self._dataset_with(tmp_path, [30])
        a = [q.qid for q in sample_questions(ds, per_doc_cap=5, seed=1)]
        b = [q.qid for q in sample_questions(ds, per_doc_cap=5, seed=2)]
        assert a != b

    def test_selection_independent_of_other_documents(self, tmp_path):
        # The same article must yield the same sample whether or not other
        # articles are present, so document-subset grids stay comparable.
        both = self._dataset_with(tmp_path, [12, 12])
        alone = self._dataset_with(tmp_path, [12])
        from_both = [
            q.qid for q in sample_questions(both, per_doc_cap=4, seed=3)
            if q.doc_id == "Doc0"
        ]
        from_alone = [
            q.qid for q in sample_questions(alone, per_doc_cap=4, seed=3)
        ]
        assert from_both == from_alone

    def test_preserves_document_order(self, tmp_path):
        ds = self._dataset_with(tmp_path, [20])
        sampled = sample_questions(ds, per_doc_cap=6, seed=11)
        indices = [int(q.qid.split("q")[1]) for q in sampled]
        assert indices == sorted(indices)


@pytest.mark.skipif(
    not (DATA_DIR / "dev-v1.1.json").exists(),
    reason="run `python -m src.data` to download SQuAD",
)
class TestRealSquad:
    def test_dev_v11_loads_with_verified_spans(self):
        # load_squad raises on any span mismatch, so loading IS the check.
        ds = load_squad(DATA_DIR / "dev-v1.1.json")
        assert len(ds.documents) == 48
        assert len(ds.questions) == 10533  # 10570 minus 37 duplicate texts
        assert all(q.gold_alternatives for q in ds.questions)

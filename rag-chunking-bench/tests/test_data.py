"""Tests for dataset loading: reconstruction, span remapping, dedup, sampling.

The loaders' shared contract is exactness — every gold span must match the
document text verbatim — so most tests build small synthetic dataset files
and check coordinates by hand. Integration tests run against the real files
when they have been downloaded (`python -m src.data`); they are skipped
otherwise so the suite passes on a fresh clone.
"""

import csv
import json
from pathlib import Path

import pytest

from src.data import (
    CHROMA_CORPORA,
    PARAGRAPH_JOINER,
    GoldSpan,
    Question,
    load_chroma,
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


def _ref(text, content):
    start = text.index(content)
    return {"content": content, "start_index": start, "end_index": start + len(content)}


def make_chroma_dir(tmp_path, corpus_texts, question_rows):
    """Build a synthetic data/chroma layout.

    `corpus_texts` maps corpus name -> text (unnamed corpora get filler
    text, since the loader reads all five pinned corpus files).
    `question_rows` is a list of (corpus_id, question, references) tuples.
    """
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    for corpus in CHROMA_CORPORA:
        text = corpus_texts.get(corpus, f"Filler text for the {corpus} corpus.")
        (chroma_dir / f"{corpus}.md").write_text(text, encoding="utf-8")
    with open(chroma_dir / "questions_df.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "references", "corpus_id"])
        for corpus_id, question, references in question_rows:
            writer.writerow([question, json.dumps(references), corpus_id])
    return tmp_path


class TestLoadChroma:
    def test_multi_reference_question_is_one_joint_alternative(self, tmp_path):
        text = "Alpha said hello. Beta answered politely. Gamma left early."
        data_dir = make_chroma_dir(
            tmp_path,
            {"chatlogs": text},
            [
                (
                    "chatlogs",
                    "Who spoke?",
                    [_ref(text, "Alpha said hello."), _ref(text, "Gamma left early.")],
                )
            ],
        )
        ds = load_chroma(data_dir)
        (q,) = ds.questions
        # Jointly-required references: ONE alternative holding both spans,
        # not two alternatives (which would mean either-suffices).
        assert len(q.gold_alternatives) == 1
        spans = q.gold_alternatives[0]
        assert [text[s.start : s.end] for s in spans] == [
            "Alpha said hello.",
            "Gamma left early.",
        ]

    def test_all_five_corpora_load_as_documents(self, tmp_path):
        data_dir = make_chroma_dir(tmp_path, {}, [])
        ds = load_chroma(data_dir)
        assert set(ds.documents) == set(CHROMA_CORPORA)
        assert ds.questions == ()

    def test_qids_count_per_corpus_in_csv_order(self, tmp_path):
        fin = "Revenue rose sharply. Costs fell notably."
        chat = "User one waved. User two nodded."
        data_dir = make_chroma_dir(
            tmp_path,
            {"finance": fin, "chatlogs": chat},
            [
                ("finance", "What rose?", [_ref(fin, "Revenue rose sharply.")]),
                ("chatlogs", "Who waved?", [_ref(chat, "User one waved.")]),
                ("finance", "What fell?", [_ref(fin, "Costs fell notably.")]),
            ],
        )
        ds = load_chroma(data_dir)
        assert [q.qid for q in ds.questions] == [
            "finance:001",
            "chatlogs:001",
            "finance:002",
        ]

    def test_reference_mismatch_raises(self, tmp_path):
        text = "Exact offsets are the whole contract."
        data_dir = make_chroma_dir(
            tmp_path,
            {"pubmed": text},
            [
                (
                    "pubmed",
                    "?",
                    [{"content": "Exact offsets", "start_index": 1, "end_index": 14}],
                )
            ],
        )
        with pytest.raises(ValueError, match="reference mismatch"):
            load_chroma(data_dir)

    def test_unknown_corpus_raises(self, tmp_path):
        data_dir = make_chroma_dir(
            tmp_path,
            {},
            [("mystery", "?", [{"content": "x", "start_index": 0, "end_index": 1}])],
        )
        with pytest.raises(ValueError, match="unknown corpus"):
            load_chroma(data_dir)


@pytest.mark.skipif(
    not (DATA_DIR / "chroma" / "questions_df.csv").exists(),
    reason="run `python -m src.data` to download the Chroma corpora",
)
class TestRealChroma:
    def test_chroma_loads_with_verified_references(self):
        # load_chroma raises on any offset mismatch, so loading IS the check.
        ds = load_chroma(DATA_DIR)
        assert len(ds.documents) == 5
        assert len(ds.questions) == 472
        by_doc = {}
        for q in ds.questions:
            by_doc[q.doc_id] = by_doc.get(q.doc_id, 0) + 1
        assert by_doc == {
            "wikitexts": 144,
            "pubmed": 99,
            "finance": 97,
            "state_of_the_union": 76,
            "chatlogs": 56,
        }
        # Long-reference regime: at least one question carries several
        # jointly-required spans through a single alternative.
        assert max(len(q.gold_alternatives[0]) for q in ds.questions) == 5
        assert all(len(q.gold_alternatives) == 1 for q in ds.questions)


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

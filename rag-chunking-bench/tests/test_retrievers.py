"""Tests for BM25, checked against an independently hand-computed example.

The expected scores below were derived by evaluating the BM25 formula
(k1 = 1.5, b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5))) by hand on
the three-document corpus — not by running the implementation — so they
catch sign, smoothing, and normalization mistakes.
"""

import pytest

from src.retrievers import BM25Retriever, query_terms

CORPUS = ["the cat sat", "the dog", "cat cat cat"]
# N = 3, lengths = (3, 2, 3), avgdl = 8/3
# df: the = 2, cat = 2, sat = 1, dog = 1
# idf(cat) = idf(the) = ln(1 + 1.5/2.5) = 0.470004
# idf(sat) = idf(dog) = ln(1 + 2.5/1.5) = 0.980829


@pytest.fixture
def bm25():
    return BM25Retriever().fit(CORPUS)


class TestQueryTerms:
    def test_lowercases_and_drops_punctuation(self):
        assert query_terms("The CAT, sat!") == ["the", "cat", "sat"]

    def test_unicode_words(self):
        assert query_terms("café Zürich") == ["café", "zürich"]


class TestBM25HandComputed:
    def test_single_term_query(self, bm25):
        # score("cat", d0): 0.470004 * 1 * 2.5 / (1 + 1.5*(0.25 + 0.75*3/(8/3)))
        # score("cat", d2): tf = 3, same length norm as d0.
        assert bm25.scores("cat") == pytest.approx(
            [0.4449738501734775, 0.0, 0.7596018250436132]
        )

    def test_multi_term_query_sums_terms(self, bm25):
        assert bm25.scores("the dog") == pytest.approx(
            [0.4449738501734775, 1.634741275783056, 0.0]
        )

    def test_rare_term_dominates(self, bm25):
        assert bm25.scores("cat sat") == pytest.approx(
            [1.3735695926697864, 0.0, 0.7596018250436132]
        )
        assert bm25.rank("cat sat") == [0, 2, 1]

    def test_repeated_query_terms_count_once(self, bm25):
        assert bm25.scores("cat cat cat") == pytest.approx(bm25.scores("cat"))


class TestBM25Properties:
    def test_idf_nonnegative_for_ubiquitous_terms(self):
        bm25 = BM25Retriever().fit(["common word a", "common word b", "common word c"])
        assert all(s >= 0 for s in bm25.scores("common"))

    def test_duplicate_documents_score_equally(self):
        bm25 = BM25Retriever().fit(["alpha beta", "gamma delta", "alpha beta"])
        s = bm25.scores("alpha")
        assert s[0] == s[2] > s[1] == 0.0

    def test_tie_breaks_by_document_index(self):
        bm25 = BM25Retriever().fit(["alpha beta", "gamma delta", "alpha beta"])
        assert bm25.rank("alpha") == [0, 2, 1]

    def test_unseen_terms_score_zero_everywhere(self, bm25):
        assert bm25.scores("zebra quark") == [0.0, 0.0, 0.0]
        assert bm25.rank("zebra quark") == [0, 1, 2]

    def test_length_normalization_prefers_shorter_at_equal_tf(self):
        bm25 = BM25Retriever(b=0.75).fit(
            ["match one two three four five six seven", "match one"]
        )
        s = bm25.scores("match")
        assert s[1] > s[0]

    def test_b_zero_disables_length_normalization(self):
        bm25 = BM25Retriever(b=0.0).fit(
            ["match one two three four five six seven", "match one"]
        )
        s = bm25.scores("match")
        assert s[0] == pytest.approx(s[1])

    def test_scores_before_fit_raises(self):
        with pytest.raises(ValueError):
            BM25Retriever().scores("query")

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError):
            BM25Retriever().fit([])

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            BM25Retriever(k1=-0.1)
        with pytest.raises(ValueError):
            BM25Retriever(b=1.5)

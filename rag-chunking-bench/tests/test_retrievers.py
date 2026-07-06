"""Tests for the retrievers, checked against independently hand-computed examples.

The expected BM25 scores were derived by evaluating the formula (k1 = 1.5,
b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5))) by hand on the
three-document corpus; the expected TF-IDF cosines by evaluating
scikit-learn's conventions (idf = ln((1 + N) / (1 + df)) + 1, raw tf,
L2-normalized rows) on the same corpus — neither by running the
implementations — so they catch sign, smoothing, and normalization mistakes.
LSA has no hand-computable closed form at useful sizes; it is tested through
behavioral invariants instead (row-space equivalence with TF-IDF, latent
term bridging, determinism, rank capping).
"""

import pytest

from src.retrievers import BM25Retriever, LSARetriever, TfidfRetriever, query_terms

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


class TestTfidfHandComputed:
    # Same corpus as BM25. N = 3; df: the = 2, cat = 2, sat = 1, dog = 1.
    # idf(the) = idf(cat) = ln(4/3) + 1 = 1.287682, idf(sat) = idf(dog) =
    # ln(4/2) + 1 = 1.693147. Document vectors are tf * idf, L2-normalized;
    # the query is weighted and normalized the same way, so the score is an
    # exact cosine: e.g. score("cat", d0) = idf_cat / ||d0|| / 1 with
    # ||d0|| = sqrt(2 * 1.287682^2 + 1.693147^2).

    @pytest.fixture
    def tfidf(self):
        return TfidfRetriever().fit(CORPUS)

    def test_single_term_query(self, tfidf):
        assert tfidf.scores("cat") == pytest.approx(
            [0.5178561161676974, 0.0, 1.0]
        )

    def test_multi_term_query(self, tfidf):
        assert tfidf.scores("the dog") == pytest.approx(
            [0.31348342733583406, 1.0, 0.0]
        )

    def test_rare_term_dominates(self, tfidf):
        assert tfidf.scores("cat sat") == pytest.approx(
            [0.8554677334345859, 0.0, 0.6053485081062916]
        )
        assert tfidf.rank("cat sat") == [0, 2, 1]

    def test_identical_direction_scores_one(self, tfidf):
        # d2 is pure "cat", so the query "cat" is an exact cosine match.
        assert tfidf.scores("cat")[2] == pytest.approx(1.0)

    def test_repeated_single_term_query_invariant(self, tfidf):
        # L2 normalization makes a one-term query's direction independent of tf.
        assert tfidf.scores("cat cat cat") == pytest.approx(tfidf.scores("cat"))


class TestTfidfProperties:
    def test_duplicate_documents_tie_and_break_by_index(self):
        t = TfidfRetriever().fit(["alpha beta", "gamma delta", "alpha beta"])
        s = t.scores("alpha")
        assert s[0] == pytest.approx(s[2])
        assert s[0] > s[1] == 0.0
        assert t.rank("alpha") == [0, 2, 1]

    def test_unseen_terms_score_zero_everywhere(self):
        t = TfidfRetriever().fit(CORPUS)
        assert t.scores("zebra quark") == [0.0, 0.0, 0.0]
        assert t.rank("zebra quark") == [0, 1, 2]

    def test_scores_are_cosines_in_unit_interval(self):
        t = TfidfRetriever().fit(CORPUS)
        for q in ("cat", "the cat sat", "dog sat unknown"):
            assert all(0.0 <= s <= 1.0 + 1e-12 for s in t.scores(q))

    def test_punctuation_only_corpus_scores_zero(self):
        # query_terms drops punctuation, so the vocabulary is empty; this
        # must degrade to all-zero scores (like BM25), not raise.
        t = TfidfRetriever().fit(["...", "!!!"])
        assert t.scores("anything") == [0.0, 0.0]
        assert t.rank("anything") == [0, 1]

    def test_scores_before_fit_raises(self):
        with pytest.raises(ValueError):
            TfidfRetriever().scores("query")

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError):
            TfidfRetriever().fit([])


class TestLSA:
    def test_full_row_space_matches_tfidf_ranking(self):
        # A duplicated document makes the corpus rank 2, so k = 2 captures
        # the whole row space and LSA must reproduce TF-IDF's ranking (the
        # scores differ by the query's out-of-row-space component only).
        docs = ["alpha beta gamma", "delta epsilon", "alpha beta gamma"]
        lsa = LSARetriever(n_components=64).fit(docs)
        tfidf = TfidfRetriever().fit(docs)
        assert lsa.realized_rank == 2  # capped at n_docs - 1
        for q in ("alpha", "epsilon", "alpha delta", "gamma epsilon"):
            assert lsa.rank(q) == tfidf.rank(q)

    def test_duplicate_documents_tie(self):
        docs = ["alpha beta gamma", "delta epsilon", "alpha beta gamma"]
        s = LSARetriever(n_components=64).fit(docs).scores("alpha")
        assert s[0] == pytest.approx(s[2])

    def test_latent_bridging_scores_cooccurring_document(self):
        # "wheel" never occurs in docs[0], so TF-IDF gives it exactly 0 —
        # but at k = 2 the car documents share a latent dimension, so LSA
        # ranks docs[0] far above the fruit documents. This is the latent
        # generalization that distinguishes LSA from sparse TF-IDF.
        docs = [
            "car automobile engine",
            "car automobile wheel",
            "banana fruit yellow",
            "banana fruit sweet",
        ]
        assert TfidfRetriever().fit(docs).scores("wheel")[0] == 0.0
        s = LSARetriever(n_components=2).fit(docs).scores("wheel")
        assert s[0] > 0.9
        assert abs(s[2]) < 0.1 and abs(s[3]) < 0.1

    def test_deterministic_across_fits(self):
        docs = [
            "car automobile engine",
            "car automobile wheel",
            "banana fruit yellow",
            "banana fruit sweet",
        ]
        a = LSARetriever().fit(docs).scores("car banana engine")
        b = LSARetriever().fit(docs).scores("car banana engine")
        assert a == b

    def test_rank_capped_by_corpus_size(self):
        lsa = LSARetriever(n_components=128).fit(CORPUS)
        assert lsa.realized_rank == 2  # min(128, 3 docs - 1, 4 terms - 1)

    def test_single_document_corpus_degrades_gracefully(self):
        lsa = LSARetriever().fit(["only one chunk here"])
        assert lsa.realized_rank == 0
        assert lsa.scores("chunk") == [0.0]
        assert lsa.rank("chunk") == [0]

    def test_punctuation_only_corpus_scores_zero(self):
        lsa = LSARetriever().fit(["...", "!!!"])
        assert lsa.scores("anything") == [0.0, 0.0]
        assert lsa.rank("anything") == [0, 1]

    def test_all_unseen_query_scores_zero(self):
        docs = ["alpha beta gamma", "delta epsilon", "alpha beta gamma"]
        lsa = LSARetriever().fit(docs)
        assert lsa.scores("zebra quark") == [0.0, 0.0, 0.0]
        assert lsa.rank("zebra quark") == [0, 1, 2]

    def test_scores_before_fit_raises(self):
        with pytest.raises(ValueError):
            LSARetriever().scores("query")

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError):
            LSARetriever().fit([])

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            LSARetriever(n_components=0)

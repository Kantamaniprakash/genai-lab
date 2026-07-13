"""Tests for span-level metrics, budget accounting, and the paired bootstrap.

The scoring examples use a synthetic document of ten 2-character tokens
("aa bb cc ... jj"): token i occupies characters [3i, 3i+2), so every token
set below can be verified by eye.
"""

import pytest

from src.chunkers import Chunk
from src.data import GoldSpan
from src.metrics import (
    gold_token_set,
    hit_at_k,
    paired_bootstrap,
    paired_bootstrap_std,
    retrieved_token_set,
    span_scores,
    take_until_budget,
)
from src.tokenization import RegexWordTokenizer, TokenIndex

DOC = "aa bb cc dd ee ff gg hh ii jj"  # token i at chars [3i, 3i+2)


@pytest.fixture
def index():
    return TokenIndex(DOC, RegexWordTokenizer())


def chunk_of_tokens(first, last):
    """Chunk covering tokens [first, last] inclusive."""
    return Chunk(text=DOC[3 * first : 3 * last + 2], start=3 * first, end=3 * last + 2)


class TestTakeUntilBudget:
    def test_stops_before_exceeding(self, index):
        ranked = [chunk_of_tokens(0, 2), chunk_of_tokens(3, 5), chunk_of_tokens(6, 9)]
        taken = take_until_budget(ranked, index, budget=7)
        # 3 + 3 fits in 7; the 4-token third chunk would make it 10.
        assert taken == ranked[:2]

    def test_does_not_pull_smaller_chunks_forward(self, index):
        ranked = [chunk_of_tokens(0, 4), chunk_of_tokens(5, 9), chunk_of_tokens(0, 0)]
        # Budget 6: first chunk (5 tokens) fits, second (5) would exceed;
        # accumulation stops even though the third chunk (1 token) would fit.
        assert take_until_budget(ranked, index, budget=6) == ranked[:1]

    def test_first_chunk_over_budget_yields_empty(self, index):
        assert take_until_budget([chunk_of_tokens(0, 9)], index, budget=5) == []

    def test_exact_fit_is_taken(self, index):
        ranked = [chunk_of_tokens(0, 4), chunk_of_tokens(5, 9)]
        assert take_until_budget(ranked, index, budget=10) == ranked

    def test_rejects_nonpositive_budget(self, index):
        with pytest.raises(ValueError):
            take_until_budget([], index, budget=0)

    def test_rejects_unknown_rule(self, index):
        with pytest.raises(ValueError, match="budget rule"):
            take_until_budget([], index, budget=5, rule="round-up")


class TestTruncateRule:
    def test_final_chunk_truncated_token_aligned(self, index):
        ranked = [chunk_of_tokens(0, 2), chunk_of_tokens(3, 9)]
        taken = take_until_budget(ranked, index, budget=5, rule="truncate")
        # 3 tokens fit whole; the 7-token chunk is cut to its first 2 tokens
        # ("dd ee"), ending exactly at a token boundary.
        assert taken[0] == ranked[0]
        assert (taken[1].start, taken[1].end) == (9, 14)
        assert taken[1].text == DOC[9:14] == "dd ee"
        assert sum(index.count_in(c.start, c.end) for c in taken) == 5

    def test_first_chunk_over_budget_is_truncated_not_dropped(self, index):
        taken = take_until_budget(
            [chunk_of_tokens(0, 9)], index, budget=4, rule="truncate"
        )
        assert len(taken) == 1
        assert index.count_in(taken[0].start, taken[0].end) == 4
        assert taken[0].text == "aa bb cc dd"

    def test_no_partial_chunk_when_budget_exactly_spent(self, index):
        ranked = [chunk_of_tokens(0, 4), chunk_of_tokens(5, 9)]
        # First chunk spends the whole budget; nothing remains to truncate.
        taken = take_until_budget(ranked, index, budget=5, rule="truncate")
        assert taken == ranked[:1]

    def test_exact_fit_needs_no_truncation(self, index):
        ranked = [chunk_of_tokens(0, 4), chunk_of_tokens(5, 9)]
        assert take_until_budget(ranked, index, budget=10, rule="truncate") == ranked

    def test_stop_is_the_default_rule(self, index):
        ranked = [chunk_of_tokens(0, 2), chunk_of_tokens(3, 9)]
        assert take_until_budget(ranked, index, budget=5) == ranked[:1]


class TestTokenSets:
    def test_retrieved_union_deduplicates_overlap(self, index):
        chunks = [chunk_of_tokens(0, 3), chunk_of_tokens(2, 5)]
        assert retrieved_token_set(chunks, index) == {0, 1, 2, 3, 4, 5}

    def test_gold_span_mid_token_claims_token(self, index):
        # Characters [4, 7) cover the second half of "bb" and all of "cc".
        assert gold_token_set((GoldSpan(4, 7),), index) == {1, 2}

    def test_gold_multiple_spans_union(self, index):
        spans = (GoldSpan(0, 2), GoldSpan(27, 29))  # "aa" and "jj"
        assert gold_token_set(spans, index) == {0, 9}


class TestSpanScores:
    def test_hand_computed_overlap(self, index):
        # Retrieved tokens {0..4}, gold tokens {3..6}: hit = {3, 4}.
        chunks = [chunk_of_tokens(0, 4)]
        gold = ((GoldSpan(9, 20),),)  # tokens 3-6
        s = span_scores(chunks, gold, index)
        assert s.recall == pytest.approx(2 / 4)
        assert s.precision == pytest.approx(2 / 5)
        assert s.iou == pytest.approx(2 / 7)

    def test_perfect_retrieval(self, index):
        chunks = [chunk_of_tokens(3, 6)]
        gold = ((GoldSpan(9, 20),),)
        s = span_scores(chunks, gold, index)
        assert (s.recall, s.precision, s.iou) == (1.0, 1.0, 1.0)

    def test_disjoint_retrieval_scores_zero(self, index):
        s = span_scores([chunk_of_tokens(7, 9)], ((GoldSpan(0, 2),),), index)
        assert (s.recall, s.precision, s.iou) == (0.0, 0.0, 0.0)

    def test_no_retrieved_tokens_scores_zero(self, index):
        s = span_scores([], ((GoldSpan(0, 2),),), index)
        assert (s.recall, s.precision, s.iou) == (0.0, 0.0, 0.0)

    def test_max_over_alternatives_per_metric(self, index):
        # Retrieved tokens {0, 1}. Alternative A = token 0 (recall 1, prec
        # 1/2); alternative B = tokens 0-3 (recall 1/2, prec 1). Each metric
        # takes its own max, so recall comes from A and precision from B.
        chunks = [chunk_of_tokens(0, 1)]
        gold = ((GoldSpan(0, 2),), (GoldSpan(0, 11),))
        s = span_scores(chunks, gold, index)
        assert s.recall == pytest.approx(1.0)
        assert s.precision == pytest.approx(1.0)
        assert s.iou == pytest.approx(1 / 2)

    def test_alternative_with_required_span_pair(self, index):
        # One alternative holding two required spans (Chroma-style): both
        # spans count toward a single gold set.
        chunks = [chunk_of_tokens(0, 0)]
        gold = ((GoldSpan(0, 2), GoldSpan(27, 29)),)
        s = span_scores(chunks, gold, index)
        assert s.recall == pytest.approx(1 / 2)


class TestHitAtK:
    def test_hit_within_k(self, index):
        ranked = [chunk_of_tokens(7, 9), chunk_of_tokens(0, 2)]
        gold = ((GoldSpan(0, 2),),)
        assert hit_at_k(ranked, gold, index, k=2)
        assert not hit_at_k(ranked, gold, index, k=1)

    def test_partial_character_overlap_counts(self, index):
        ranked = [chunk_of_tokens(1, 1)]
        gold = ((GoldSpan(4, 7),),)  # straddles tokens 1 and 2
        assert hit_at_k(ranked, gold, index, k=1)

    def test_rejects_nonpositive_k(self, index):
        with pytest.raises(ValueError):
            hit_at_k([], ((GoldSpan(0, 2),),), index, k=0)


class TestPairedBootstrap:
    def test_identical_scores_give_zero_interval(self):
        scores = [0.2, 0.5, 0.9, 0.4]
        r = paired_bootstrap(scores, scores, n_resamples=200, seed=0)
        assert r.mean_diff == 0.0
        assert (r.ci_low, r.ci_high) == (0.0, 0.0)
        assert not r.significant

    def test_constant_shift_gives_degenerate_interval(self):
        a = [0.3, 0.6, 0.9]
        b = [x - 0.1 for x in a]
        r = paired_bootstrap(a, b, n_resamples=200, seed=0)
        assert r.mean_diff == pytest.approx(0.1)
        assert r.ci_low == pytest.approx(0.1)
        assert r.ci_high == pytest.approx(0.1)
        assert r.significant

    def test_deterministic_given_seed(self):
        a = [0.1, 0.9, 0.4, 0.7, 0.2, 0.6]
        b = [0.2, 0.5, 0.5, 0.6, 0.1, 0.9]
        r1 = paired_bootstrap(a, b, n_resamples=500, seed=42)
        r2 = paired_bootstrap(a, b, n_resamples=500, seed=42)
        assert (r1.mean_diff, r1.ci_low, r1.ci_high) == (
            r2.mean_diff,
            r2.ci_low,
            r2.ci_high,
        )

    def test_interval_contains_mean_for_noisy_diffs(self):
        a = [0.5, 0.7, 0.2, 0.9, 0.4, 0.6, 0.8, 0.3]
        b = [0.4, 0.8, 0.1, 0.7, 0.5, 0.4, 0.6, 0.2]
        r = paired_bootstrap(a, b, n_resamples=2000, seed=1)
        assert r.ci_low <= r.mean_diff <= r.ci_high
        assert r.ci_low < r.ci_high

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            paired_bootstrap([0.1], [0.1, 0.2])

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            paired_bootstrap([], [])


class TestPairedBootstrapStd:
    def test_identical_scores_give_zero_interval(self):
        scores = [0.2, 0.5, 0.9, 0.4]
        r = paired_bootstrap_std(scores, scores, n_resamples=200, seed=0)
        assert r.mean_diff == 0.0
        assert (r.ci_low, r.ci_high) == (0.0, 0.0)
        assert not r.significant

    def test_constant_shift_has_zero_std_difference(self):
        # A level shift changes the mean but not the dispersion: the mean
        # bootstrap flags it, the std bootstrap must not.
        a = [0.3, 0.6, 0.9, 0.1, 0.7]
        b = [x - 0.2 for x in a]
        r = paired_bootstrap_std(a, b, n_resamples=500, seed=0)
        assert r.mean_diff == pytest.approx(0.0)
        assert not r.significant

    def test_point_estimate_matches_sample_stds(self):
        a = [0.0, 1.0, 0.0, 1.0]  # sum of squared deviations 1.0, ddof=1
        b = [0.5, 0.5, 0.5, 0.5]  # sample std = 0
        r = paired_bootstrap_std(a, b, n_resamples=200, seed=0)
        assert r.mean_diff == pytest.approx((1 / 3) ** 0.5, abs=1e-9)

    def test_wider_scores_give_positive_significant_difference(self):
        a = [0.0, 1.0] * 10
        b = [0.5, 0.5] * 10
        r = paired_bootstrap_std(a, b, n_resamples=2000, seed=1)
        assert r.mean_diff > 0
        assert r.significant

    def test_deterministic_given_seed(self):
        a = [0.1, 0.9, 0.4, 0.7, 0.2, 0.6]
        b = [0.2, 0.5, 0.5, 0.6, 0.1, 0.9]
        r1 = paired_bootstrap_std(a, b, n_resamples=500, seed=42)
        r2 = paired_bootstrap_std(a, b, n_resamples=500, seed=42)
        assert (r1.mean_diff, r1.ci_low, r1.ci_high) == (
            r2.mean_diff,
            r2.ci_low,
            r2.ci_high,
        )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            paired_bootstrap_std([0.1], [0.1, 0.2])

    def test_rejects_fewer_than_two_questions(self):
        with pytest.raises(ValueError):
            paired_bootstrap_std([0.1], [0.2])

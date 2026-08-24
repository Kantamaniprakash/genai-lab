import math

import pytest

from src.data import (
    REWARDBENCH_PATH,
    SUBSET_TO_CATEGORY,
    PairItem,
    load_rewardbench,
    stratified_sample,
)
from src.schedule import (
    balanced_order,
    by_category,
    by_subset,
    coverage,
    format_coverage,
)


def make_item(subset: str, idx: int) -> PairItem:
    return PairItem(
        item_id=f"{subset}/{idx:03d}",
        subset=subset,
        category=SUBSET_TO_CATEGORY[subset],
        prompt=f"prompt {idx}",
        chosen=f"chosen {idx}",
        rejected=f"rejected {idx}",
        chosen_model="model-x",
        rejected_model="model-y",
    )


def make_sample(counts: dict[str, int]) -> list[PairItem]:
    return [make_item(subset, i) for subset, n in counts.items() for i in range(n)]


def max_subsets_per_category(items) -> int:
    """Drift bound a category inherits from subset-level balancing."""
    per_category: dict[str, set[str]] = {}
    for item in items:
        per_category.setdefault(item.category, set()).add(item.subset)
    return max(len(s) for s in per_category.values())


def max_prefix_drift(items, order, *, stratum=by_subset) -> float:
    """Worst proportional drift over every prefix of an execution order."""
    worst = 0.0
    finished: list[str] = []
    for item in order:
        finished.append(item.item_id)
        worst = max(worst, coverage(items, finished, stratum=stratum)["max_abs_drift_items"])
    return worst


# A composition with the same awkward shape as the real sample: a few large
# subsets and several small ones, so proportional allocation is not integral.
COUNTS = {
    "math-prm": 90,
    "hep-python": 33,
    "alpacaeval-easy": 20,
    "xstest-should-respond": 50,
    "donotanswer": 27,
    "mt-bench-easy": 6,
    "llmbar-adver-manual": 9,
}


class TestScheduleIsAPermutation:
    def test_fresh_grid_schedules_every_item_exactly_once(self):
        items = make_sample(COUNTS)
        order = balanced_order(items)
        assert len(order) == len(items)
        assert {i.item_id for i in order} == {i.item_id for i in items}

    def test_is_deterministic(self):
        items = make_sample(COUNTS)
        assert [i.item_id for i in balanced_order(items)] == [
            i.item_id for i in balanced_order(items)
        ]

    def test_is_independent_of_input_order(self):
        items = make_sample(COUNTS)
        shuffled = list(reversed(items))
        assert [i.item_id for i in balanced_order(items)] == [
            i.item_id for i in balanced_order(shuffled)
        ]

    def test_items_within_a_subset_keep_id_order(self):
        items = make_sample(COUNTS)
        order = balanced_order(items)
        seen = [i.item_id for i in order if i.subset == "math-prm"]
        assert seen == sorted(seen)


class TestProportionalityBound:
    def test_every_prefix_of_a_fresh_grid_stays_within_one_item(self):
        items = make_sample(COUNTS)
        assert max_prefix_drift(items, balanced_order(items)) < 1.0

    def test_bound_also_holds_at_the_category_level(self):
        items = make_sample(COUNTS)
        order = balanced_order(items, stratum=by_category)
        assert max_prefix_drift(items, order, stratum=by_category) < 1.0

    def test_subset_balancing_keeps_categories_nearly_balanced_too(self):
        # Categories are unions of subsets, so holding every subset within one
        # item of proportional holds each category within its subset count.
        items = make_sample(COUNTS)
        order = balanced_order(items, stratum=by_subset)
        assert max_prefix_drift(items, order, stratum=by_category) < max_subsets_per_category(items)

    def test_the_alphabetical_order_it_replaces_violates_the_bound_badly(self):
        # Finding 26, reproduced as a test: the sorted order this scheduler
        # replaced puts an entire subset ahead of every other.
        items = make_sample(COUNTS)
        sorted_order = sorted(items, key=lambda it: it.item_id)
        assert max_prefix_drift(items, sorted_order) > 10.0

    def test_two_subsets_alternate(self):
        items = make_sample({"alpacaeval-easy": 3, "mt-bench-easy": 3})
        subsets = [i.subset for i in balanced_order(items)]
        assert subsets[0] != subsets[1]
        assert subsets.count("alpacaeval-easy") == 3

    def test_a_rare_subset_is_not_starved_to_the_end(self):
        items = make_sample({"math-prm": 90, "mt-bench-easy": 6})
        order = balanced_order(items)
        positions = [i for i, item in enumerate(order) if item.subset == "mt-bench-easy"]
        # Proportionally the 6 rare items belong at roughly every 16th slot;
        # the last one must not be pushed anywhere near the tail.
        assert positions[0] < 16
        assert max(positions) < len(order) - 5


class TestResumingASkewedStore:
    def test_finished_items_are_not_rescheduled(self):
        items = make_sample(COUNTS)
        done = {items[0].item_id: 2, items[1].item_id: 2}
        order = balanced_order(items, done)
        assert len(order) == len(items) - 2
        assert {items[0].item_id, items[1].item_id}.isdisjoint({i.item_id for i in order})

    def test_half_finished_items_are_scheduled_first(self):
        items = make_sample(COUNTS)
        orphan = items[40]
        order = balanced_order(items, {orphan.item_id: 1})
        assert order[0].item_id == orphan.item_id

    def test_several_orphans_come_first_in_id_order(self):
        items = make_sample(COUNTS)
        orphans = {items[5].item_id: 1, items[70].item_id: 1, items[12].item_id: 1}
        order = balanced_order(items, orphans)
        assert [i.item_id for i in order[:3]] == sorted(orphans)

    def test_the_inherited_excess_is_never_rejudged(self):
        # The real situation this scheduler inherited: one subset finished, the
        # rest untouched.
        items = make_sample(COUNTS)
        done = {i.item_id: 2 for i in items if i.subset == "alpacaeval-easy"}
        order = balanced_order(items, done)
        assert "alpacaeval-easy" not in {i.subset for i in order}

    def test_an_over_served_subset_waits_until_it_is_proportional_again(self):
        # 5 of 6 rare items finished and nothing else: the subset holds 100% of
        # the store against a 6.25% target, so its last item must wait until
        # the store is large enough for 5 to be a proportional count.
        items = make_sample({"math-prm": 90, "mt-bench-easy": 6})
        rare = [i for i in items if i.subset == "mt-bench-easy"]
        order = balanced_order(items, {i.item_id: 2 for i in rare[:5]})
        position = next(k for k, i in enumerate(order, start=1) if i.subset == "mt-bench-easy")
        # The rare subset is served only once its deficit overtakes the equally
        # starved majority: (6/96)(D+1) - 5 >= (90/96)(D+1) - (D - 5) gives
        # D >= 88, and 83 of those 88 are math-prm, so it lands 84th.
        assert position == 84

    def test_total_variation_contracts_to_zero(self):
        items = make_sample(COUNTS)
        done_ids = [i.item_id for i in items if i.subset == "alpacaeval-easy"]
        order = balanced_order(items, {i: 2 for i in done_ids})
        tvs = []
        finished = list(done_ids)
        for item in order:
            finished.append(item.item_id)
            tvs.append(coverage(items, finished)["total_variation"])
        assert tvs[0] > 0.85, "the inherited skew starts far from the target"
        assert tvs[-1] == pytest.approx(0.0), "a completed grid is the target sample exactly"
        # Contraction is not strictly monotone: in the tail, absorbing an
        # integer item into a rounded share can add back a fraction of one
        # item's worth of distance. No step ever adds back a whole item's.
        assert max(b - a for a, b in zip(tvs, tvs[1:])) < 1 / len(items)
        assert tvs[len(tvs) // 4] < 0.25, "most of the distance closes early"

    def test_max_drift_recovers_as_early_as_arithmetic_allows(self):
        # An over-represented stratum cannot be un-judged, so its drift can only
        # dilute as the store grows: with d finished items at share p, drift
        # falls below one item only at D > (d - 1) / p. The scheduler is optimal
        # if it recovers exactly there and not one item later.
        items = make_sample(COUNTS)
        done_ids = [i.item_id for i in items if i.subset == "alpacaeval-easy"]
        order = balanced_order(items, {i: 2 for i in done_ids})
        finished = list(done_ids)
        recovered_at = None
        for k, item in enumerate(order, start=1):
            finished.append(item.item_id)
            if coverage(items, finished)["max_abs_drift_items"] < 1.0:
                recovered_at = k
                break
        d, p = len(done_ids), COUNTS["alpacaeval-easy"] / len(items)
        earliest = math.floor((d - 1) / p) + 1 - d
        assert recovered_at == earliest == 204


class TestValidation:
    def test_empty_sample_is_rejected(self):
        with pytest.raises(ValueError, match="empty item set"):
            balanced_order([])

    def test_duplicate_ids_are_rejected(self):
        item = make_item("math-prm", 0)
        with pytest.raises(ValueError, match="duplicate item ids"):
            balanced_order([item, item])

    def test_finished_ids_outside_the_sample_are_rejected(self):
        items = make_sample({"math-prm": 4})
        with pytest.raises(ValueError, match="outside the sample"):
            balanced_order(items, {"math-prm/999": 2})

    def test_impossible_finished_counts_are_rejected(self):
        items = make_sample({"math-prm": 4})
        with pytest.raises(ValueError, match="outside 0..2"):
            balanced_order(items, {items[0].item_id: 3})

    def test_coverage_rejects_unknown_finished_ids(self):
        items = make_sample({"math-prm": 4})
        with pytest.raises(ValueError, match="outside the sample"):
            coverage(items, {"hep-go/001"})


class TestCoverageReport:
    def test_counts_and_shares_on_a_hand_checked_case(self):
        items = make_sample({"math-prm": 3, "mt-bench-easy": 1})
        report = coverage(items, ["math-prm/000", "math-prm/001"])
        assert report["n_target"] == 4 and report["n_done"] == 2
        assert report["strata"]["math-prm"] == {
            "target": 3,
            "done": 2,
            "target_share": 0.75,
            "done_share": 1.0,
            "drift_items": pytest.approx(0.5),
        }
        assert report["strata"]["mt-bench-easy"]["drift_items"] == pytest.approx(-0.5)
        assert report["total_variation"] == pytest.approx(0.25)

    def test_a_complete_grid_is_exactly_the_target(self):
        items = make_sample(COUNTS)
        report = coverage(items, [i.item_id for i in items])
        assert report["total_variation"] == pytest.approx(0.0)
        assert report["max_abs_drift_items"] == pytest.approx(0.0)

    def test_nothing_finished_has_no_defined_distribution(self):
        report = coverage(make_sample({"math-prm": 3}), [])
        assert report["n_done"] == 0
        assert report["total_variation"] is None
        assert "n/a" in format_coverage(report, label="subset")

    def test_format_line_carries_the_numbers(self):
        items = make_sample({"math-prm": 3, "mt-bench-easy": 1})
        line = format_coverage(coverage(items, ["math-prm/000"]), label="subset")
        assert "1/4 items finished" in line
        assert "total-variation from target 0.250" in line


needs_data = pytest.mark.skipif(
    not REWARDBENCH_PATH.exists(), reason="pinned parquet not downloaded (python -m src.data)"
)


@needs_data
class TestOnTheAuditSample:
    def test_the_audit_sample_stays_representative_at_every_checkpoint(self):
        items = stratified_sample(load_rewardbench(), n=600, seed=0)
        order = balanced_order(items)
        finished: list[str] = []
        worst_subset = worst_category = 0.0
        for item in order:
            finished.append(item.item_id)
            worst_subset = max(worst_subset, coverage(items, finished)["max_abs_drift_items"])
            worst_category = max(
                worst_category,
                coverage(items, finished, stratum=by_category)["max_abs_drift_items"],
            )
        assert worst_subset < 1.0
        assert worst_category < max_subsets_per_category(items)

    def test_resuming_the_real_qwen_7b_prefix_recovers_representativeness(self):
        # The store this scheduler was written for: an alphabetical prefix of
        # 67 items, all Chat plus part of one Safety subset.
        items = stratified_sample(load_rewardbench(), n=600, seed=0)
        prefix = [i.item_id for i in sorted(items, key=lambda it: it.item_id)[:67]]
        assert coverage(items, prefix, stratum=by_category)["total_variation"] > 0.74
        order = balanced_order(items, {i: 2 for i in prefix})
        finished = list(prefix)
        seen = {}
        for k, item in enumerate(order, start=1):
            finished.append(item.item_id)
            if k in (100, 300, len(order)):
                seen[k] = coverage(items, finished, stratum=by_category)["total_variation"]
        # The inherited 67 cannot be un-judged, so the distance falls only as
        # they dilute — but it falls fast, and a completed grid is exact.
        assert seen[100] < 0.24
        assert seen[300] < 0.06
        assert seen[len(order)] == pytest.approx(0.0)

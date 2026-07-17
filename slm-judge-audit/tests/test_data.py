import pytest

from src.data import (
    CATEGORIES,
    EXPECTED_SUBSET_COUNTS,
    EXPECTED_TOTAL,
    REWARDBENCH_PATH,
    SUBSET_TO_CATEGORY,
    PairItem,
    load_rewardbench,
    stratified_sample,
)


def make_item(subset: str, idx: int, **overrides) -> PairItem:
    fields = dict(
        item_id=f"{subset}/{idx}",
        subset=subset,
        category=SUBSET_TO_CATEGORY[subset],
        prompt=f"prompt {idx}",
        chosen=f"chosen {idx}",
        rejected=f"rejected {idx}",
        chosen_model="model-x",
        rejected_model="model-y",
    )
    fields.update(overrides)
    return PairItem(**fields)


class TestCategoryMapping:
    def test_mapping_is_a_partition(self):
        all_subsets = [s for subsets in CATEGORIES.values() for s in subsets]
        assert len(all_subsets) == len(set(all_subsets)) == 23

    def test_expected_counts_cover_exactly_the_mapped_subsets(self):
        assert set(EXPECTED_SUBSET_COUNTS) == set(SUBSET_TO_CATEGORY)
        assert sum(EXPECTED_SUBSET_COUNTS.values()) == EXPECTED_TOTAL

    def test_llmbar_is_fully_embedded(self):
        # LLMBar is 419 instances (Zeng et al., ICLR 2024); RewardBench carries
        # all of them as its llmbar-* subsets. Guard the no-double-count claim.
        llmbar = [s for s in SUBSET_TO_CATEGORY if s.startswith("llmbar-")]
        assert sum(EXPECTED_SUBSET_COUNTS[s] for s in llmbar) == 419


class TestPairItemValidation:
    def test_valid_item_constructs(self):
        make_item("mt-bench-easy", 1)

    def test_empty_field_rejected(self):
        with pytest.raises(ValueError, match="empty field"):
            make_item("mt-bench-easy", 1, chosen="")

    def test_identical_responses_rejected(self):
        with pytest.raises(ValueError, match="identical responses"):
            make_item("mt-bench-easy", 1, chosen="same", rejected="same")

    def test_unknown_subset_rejected(self):
        with pytest.raises(ValueError, match="unknown subset"):
            PairItem(
                item_id="not-a-subset/1",
                subset="not-a-subset",
                category="Chat",
                prompt="p",
                chosen="c",
                rejected="r",
                chosen_model="x",
                rejected_model="y",
            )

    def test_wrong_category_rejected(self):
        with pytest.raises(ValueError, match="wrong category"):
            make_item("mt-bench-easy", 1, category="Reasoning")


class TestStratifiedSample:
    def build_pool(self) -> tuple[PairItem, ...]:
        pool = []
        for subset, count in (("math-prm", 40), ("mt-bench-easy", 10), ("donotanswer", 30)):
            pool.extend(make_item(subset, i) for i in range(count))
        return tuple(pool)

    def test_exact_size_and_proportions(self):
        sample = stratified_sample(self.build_pool(), 40, seed=0)
        assert len(sample) == 40
        by_subset = {}
        for item in sample:
            by_subset[item.subset] = by_subset.get(item.subset, 0) + 1
        assert by_subset == {"math-prm": 20, "mt-bench-easy": 5, "donotanswer": 15}

    def test_deterministic_and_order_independent(self):
        pool = self.build_pool()
        shuffled = tuple(reversed(pool))
        a = stratified_sample(pool, 17, seed=7)
        b = stratified_sample(shuffled, 17, seed=7)
        assert a == b

    def test_different_seeds_differ(self):
        pool = self.build_pool()
        assert stratified_sample(pool, 17, seed=1) != stratified_sample(pool, 17, seed=2)

    def test_no_duplicates(self):
        sample = stratified_sample(self.build_pool(), 40, seed=3)
        assert len({item.item_id for item in sample}) == 40

    def test_full_sample_is_whole_pool(self):
        pool = self.build_pool()
        assert set(stratified_sample(pool, len(pool), seed=0)) == set(pool)

    def test_oversized_request_rejected(self):
        with pytest.raises(ValueError, match="cannot sample"):
            stratified_sample(self.build_pool(), 81, seed=0)


needs_data = pytest.mark.skipif(
    not REWARDBENCH_PATH.exists(), reason="pinned parquet not downloaded (python -m src.data)"
)


@needs_data
class TestLoadRewardbench:
    def test_load_matches_pinned_composition(self):
        items = load_rewardbench()
        assert len(items) == EXPECTED_TOTAL
        counts = {}
        for item in items:
            counts[item.subset] = counts.get(item.subset, 0) + 1
        assert counts == EXPECTED_SUBSET_COUNTS

    def test_ids_unique_and_namespaced(self):
        items = load_rewardbench()
        ids = [item.item_id for item in items]
        assert len(set(ids)) == len(ids)
        assert all(item.item_id.startswith(item.subset + "/") for item in items)

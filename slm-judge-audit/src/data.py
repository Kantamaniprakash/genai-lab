"""RewardBench filtered set: pinned download, validation, stratified sampling.

The audit's data contract is a single pinned artifact: the RewardBench
"filtered" evaluation split (Lambert et al., 2024, arXiv:2403.13787) at a fixed
repository revision, verified by SHA256 at fetch time and by pinned per-subset
counts at load time. Every item is a human-verified (prompt, chosen, rejected)
pair, so judge accuracy is measurable against gold labels without any further
annotation. The llmbar-* subsets are the complete LLMBar meta-evaluation set
(Zeng et al., ICLR 2024, arXiv:2310.07641) — 419 instances with objective
gold preferences — so the adversarial instruction-following axis ships inside
the same artifact and is NOT loaded separately (doing so would double-count).

The raw ``id`` column is not unique across subsets (verified 2026-07-17), so
items are keyed by ``subset/id``, which is checked unique at load.

Sampling is stratified by subset with largest-remainder allocation, so a
600-item sample preserves the benchmark's composition to within one item per
subset, and is deterministic given (n, seed) regardless of input order.

``python -m src.data`` fetches the parquet into ``data/`` (gitignored) and
prints per-category statistics.
"""

from __future__ import annotations

import hashlib
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

REWARDBENCH_REVISION = "168d848cdbbea9764fae4a544dc9ca1e6cca4931"
REWARDBENCH_URL = (
    "https://huggingface.co/datasets/allenai/reward-bench/resolve/"
    f"{REWARDBENCH_REVISION}/data/filtered-00000-of-00001.parquet"
)
REWARDBENCH_SHA256 = "65473c20ed0627e02503557ae102c25c6b0e66b5ed69ee327e224bb95e70ca29"
REWARDBENCH_PATH = DATA_DIR / "rewardbench_filtered.parquet"

# Official subset -> category mapping from the RewardBench paper (Table 1) and
# the allenai/reward-bench scoring code. The loader hard-fails on any subset
# outside this map, so upstream drift cannot silently change the audit's scope.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "Chat": (
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ),
    "Chat Hard": (
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ),
    "Safety": (
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ),
    "Reasoning": (
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ),
}

SUBSET_TO_CATEGORY: dict[str, str] = {
    subset: category for category, subsets in CATEGORIES.items() for subset in subsets
}

# Per-subset counts of the pinned revision, recorded 2026-07-17. The SHA256 pin
# already guarantees the bytes; these make the expected composition explicit
# and catch a bad local parse as loudly as a bad download.
EXPECTED_SUBSET_COUNTS: dict[str, int] = {
    "alpacaeval-easy": 100,
    "alpacaeval-hard": 95,
    "alpacaeval-length": 95,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "llmbar-adver-neighbor": 134,
    "llmbar-natural": 100,
    "math-prm": 447,
    "mt-bench-easy": 28,
    "mt-bench-hard": 37,
    "mt-bench-med": 40,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
}
EXPECTED_TOTAL = 2985


@dataclass(frozen=True)
class PairItem:
    """One human-verified preference pair with a gold label.

    ``chosen`` is the response the benchmark's annotation process verified as
    better; ``rejected`` is the worse one. Presentation order (A/B) is decided
    later, by the prompt builder — a PairItem itself carries no ordering.
    """

    item_id: str
    subset: str
    category: str
    prompt: str
    chosen: str
    rejected: str
    chosen_model: str
    rejected_model: str

    def __post_init__(self) -> None:
        if not self.prompt or not self.chosen or not self.rejected:
            raise ValueError(f"item {self.item_id} has an empty field")
        if self.chosen == self.rejected:
            raise ValueError(f"item {self.item_id} has identical responses")
        if self.subset not in SUBSET_TO_CATEGORY:
            raise ValueError(f"item {self.item_id} has unknown subset {self.subset!r}")
        if self.category != SUBSET_TO_CATEGORY[self.subset]:
            raise ValueError(f"item {self.item_id} has wrong category {self.category!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(force: bool = False) -> Path:
    """Download the pinned parquet if absent; verify SHA256 either way."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not REWARDBENCH_PATH.exists():
        tmp = REWARDBENCH_PATH.with_suffix(".tmp")
        urllib.request.urlretrieve(REWARDBENCH_URL, tmp)
        tmp.replace(REWARDBENCH_PATH)
    actual = _sha256(REWARDBENCH_PATH)
    if actual != REWARDBENCH_SHA256:
        raise RuntimeError(
            f"SHA256 mismatch for {REWARDBENCH_PATH.name}: "
            f"expected {REWARDBENCH_SHA256}, got {actual}"
        )
    return REWARDBENCH_PATH


def load_rewardbench(path: Path | None = None) -> tuple[PairItem, ...]:
    """Load and validate the pinned filtered split.

    Hard-fails on: unknown subsets, per-subset count drift, duplicate
    ``subset/id`` keys, empty or degenerate pairs.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(path or REWARDBENCH_PATH).to_pylist()

    counts: dict[str, int] = {}
    items: list[PairItem] = []
    for row in rows:
        subset = row["subset"]
        if subset not in SUBSET_TO_CATEGORY:
            raise ValueError(f"unexpected subset {subset!r} in data file")
        counts[subset] = counts.get(subset, 0) + 1
        items.append(
            PairItem(
                item_id=f"{subset}/{row['id']}",
                subset=subset,
                category=SUBSET_TO_CATEGORY[subset],
                prompt=row["prompt"],
                chosen=row["chosen"],
                rejected=row["rejected"],
                chosen_model=row["chosen_model"],
                rejected_model=row["rejected_model"],
            )
        )

    if counts != EXPECTED_SUBSET_COUNTS:
        drift = {
            s: (EXPECTED_SUBSET_COUNTS.get(s), counts.get(s))
            for s in set(EXPECTED_SUBSET_COUNTS) | set(counts)
            if EXPECTED_SUBSET_COUNTS.get(s) != counts.get(s)
        }
        raise ValueError(f"per-subset count drift (expected, got): {drift}")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate item ids: {dupes[:5]}")
    return tuple(items)


def stratified_sample(
    items: tuple[PairItem, ...], n: int, seed: int
) -> tuple[PairItem, ...]:
    """Deterministic stratified sample of ``n`` items, proportional by subset.

    Allocation uses largest-remainder rounding so subset totals sum to exactly
    ``n`` and every subset with nonzero share keeps its proportion to within
    one item. Within a subset, items are drawn without replacement by a
    ``random.Random(seed)`` shuffle of the id-sorted items, so the result is
    independent of input order. Output is sorted by item_id.
    """
    if not 0 < n <= len(items):
        raise ValueError(f"cannot sample {n} of {len(items)} items")

    by_subset: dict[str, list[PairItem]] = {}
    for item in sorted(items, key=lambda it: it.item_id):
        by_subset.setdefault(item.subset, []).append(item)

    total = len(items)
    quotas = {s: n * len(group) / total for s, group in by_subset.items()}
    alloc = {s: math.floor(q) for s, q in quotas.items()}
    shortfall = n - sum(alloc.values())
    # Break remainder ties by subset name so allocation is fully deterministic.
    for s in sorted(quotas, key=lambda s: (alloc[s] - quotas[s], s))[:shortfall]:
        alloc[s] += 1

    rng = random.Random(seed)
    sampled: list[PairItem] = []
    for subset in sorted(by_subset):
        group = list(by_subset[subset])
        rng.shuffle(group)
        take = alloc[subset]
        if take > len(group):
            raise ValueError(f"allocation {take} exceeds subset {subset} size {len(group)}")
        sampled.extend(group[:take])
    return tuple(sorted(sampled, key=lambda it: it.item_id))


def main() -> None:
    fetch()
    items = load_rewardbench()
    print(f"rewardbench filtered @ {REWARDBENCH_REVISION[:12]}: {len(items)} items")
    for category, subsets in CATEGORIES.items():
        cat_items = [it for it in items if it.category == category]
        lengths = sorted(
            len(it.prompt) + len(it.chosen) + len(it.rejected) for it in cat_items
        )
        p50 = lengths[len(lengths) // 2]
        p90 = lengths[int(len(lengths) * 0.9)]
        print(
            f"  {category:10s} n={len(cat_items):4d} subsets={len(subsets):2d} "
            f"chars p50={p50:5d} p90={p90:5d} max={lengths[-1]:5d}"
        )


if __name__ == "__main__":
    main()

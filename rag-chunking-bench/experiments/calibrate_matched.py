"""Find sentence-packing sizes that match semantic runs' *realized* sizes.

    python -m experiments.calibrate_matched --dataset dev-v1.1

Embedding breakpoints only shorten chunks, so a semantic configuration
operates below its nominal budget (finding 20): semantic-512 realizes a mean
chunk size around 314 regex tokens on dev-v1.1 while sentence-512 realizes
about 475. Comparing the two at matched *nominal* size therefore mixes two
treatments — boundary placement and effective chunk size. This script
computes, for each semantic configuration on disk, the sentence-packing
``max_tokens`` whose realized mean chunk size lands closest to the semantic
run's realized mean. Running the grid at those calibrated sizes yields the
matched-realized-size comparison, where any residual delta is attributable
to boundary placement alone.

Calibration searches over the *chunking* only — no retrieval runs — so it is
fast and fully deterministic. Under zero overlap the chunks cover every
token exactly once, so the realized mean equals (total corpus tokens) /
(number of chunks); greedy packing produces no more chunks at a larger
budget than at a smaller one, so the realized mean is nondecreasing in
``max_tokens`` and a binary search is exact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.aggregate import BASELINE_SIZES, load_raw
from experiments.run_grid import load_dataset
from src.chunkers import SentenceChunker
from src.data import QADataset
from src.tokenization import RegexWordTokenizer, TokenIndex, Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def calibrate(
    documents: list[str], target: float, hi: int, tokenizer: Tokenizer | None = None
) -> tuple[int, float]:
    """Smallest-gap sentence ``max_tokens`` whose realized mean is closest to ``target``.

    ``hi`` is the semantic run's nominal size: the semantic realized mean can
    never exceed what sentence packing realizes at the same nominal (the
    semantic chunker is the sentence chunker plus extra boundaries), so the
    calibrated size lies in [ceil(target), hi]. Realized mean is a
    nondecreasing step function of ``max_tokens`` (see module docstring);
    binary search finds the first size at or above the target, and the
    closer of that size and its predecessor wins (ties to the smaller size).
    """
    tokenizer = tokenizer or RegexWordTokenizer()
    indexes = [TokenIndex(doc, tokenizer) for doc in documents]
    cache: dict[int, float] = {}

    def mean_at(m: int) -> float:
        if m not in cache:
            total = 0
            n_chunks = 0
            for doc, index in zip(documents, indexes, strict=True):
                chunks = SentenceChunker(max_tokens=m, tokenizer=tokenizer).chunk(doc)
                total += sum(index.count_in(c.start, c.end) for c in chunks)
                n_chunks += len(chunks)
            cache[m] = total / n_chunks
        return cache[m]

    lo = max(1, int(target) + (target > int(target)))
    if lo >= hi or mean_at(hi) <= target:
        return hi, mean_at(hi)
    # Binary search for the first max_tokens whose realized mean >= target;
    # its predecessor is the largest size still below the target, so one of
    # the two is the closest achievable realized mean.
    low, high = lo, hi
    while low < high:
        mid = (low + high) // 2
        if mean_at(mid) >= target:
            high = mid
        else:
            low = mid + 1
    candidates = [low] if low == lo else [low - 1, low]
    best = min(candidates, key=lambda m: (abs(mean_at(m) - target), m))
    return best, mean_at(best)


def calibration_rows(
    dataset: QADataset, raw_dir: Path, retriever: str, seed: int
) -> list[dict]:
    """One calibration record per semantic configuration on disk."""
    runs = load_raw(
        raw_dir,
        dataset=dataset.name,
        retriever=retriever,
        budget_rule="stop",
        overlap=0,
        seed=seed,
        sizes=BASELINE_SIZES,
    )
    semantic = [rr for rr in runs if rr.config["chunker"] == "semantic"]
    if not semantic:
        raise SystemExit(
            f"no semantic runs for {dataset.name}/{retriever} in {raw_dir} — "
            "run the semantic grid before calibrating against it"
        )
    sentence = {
        rr.config["chunk_size"]: rr for rr in runs if rr.config["chunker"] == "sentence"
    }
    documents = [doc.text for doc in dataset.documents.values()]
    rows = []
    for rr in semantic:
        nominal = rr.config["chunk_size"]
        target = rr.chunk_stats["tokens_mean"]
        calibrated, achieved = calibrate(documents, target, hi=nominal)
        row = {
            "nominal": nominal,
            "semantic_realized": target,
            "calibrated": calibrated,
            "calibrated_realized": round(achieved, 2),
            "rel_gap": round(abs(achieved - target) / target, 4),
        }
        if nominal in sentence:
            row["sentence_realized"] = sentence[nominal].chunk_stats["tokens_mean"]
        rows.append(row)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate sentence sizes matching semantic realized sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset, args.data_dir)
    rows = calibration_rows(dataset, args.raw_dir, args.retriever, args.seed)
    header = (
        "nominal | semantic realized | sentence realized | "
        "calibrated max_tokens | calibrated realized | rel gap"
    )
    print(header)
    print("-" * len(header))
    run_sizes = []
    for row in rows:
        print(
            f"{row['nominal']:>7} | {row['semantic_realized']:>17.2f} | "
            f"{row.get('sentence_realized', float('nan')):>17.2f} | "
            f"{row['calibrated']:>21} | {row['calibrated_realized']:>19.2f} | "
            f"{row['rel_gap']:.2%}"
        )
        if row["calibrated"] not in BASELINE_SIZES:
            run_sizes.append(row["calibrated"])
    if run_sizes:
        sizes = " ".join(str(s) for s in sorted(set(run_sizes)))
        print(
            f"\nnext: python -m experiments.run_grid --dataset {args.dataset} "
            f"--chunkers sentence --sizes {sizes} --retrievers {args.retriever}"
        )


if __name__ == "__main__":
    main()

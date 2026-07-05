"""Grid experiment runner for the budget-matched chunking benchmark.

Expands a grid of (chunker, chunk size, overlap, retriever) configurations,
evaluates each on a QA dataset under the budget-matched protocol (README,
"Evaluation protocol"), and writes one gzipped JSON of per-question scores
per configuration into ``results/raw/``.

Design points:

- **Deterministic.** Question sampling is seeded per document, retrievers
  break ties by chunk index, and nothing depends on wall clock or hardware,
  so a rerun reproduces every score bit-for-bit.
- **Resumable.** A configuration whose result file already exists is skipped
  (``--force`` reruns it), so interrupted grids continue where they stopped.
- **Traceable.** Result files embed the full configuration, the git commit
  of the code that produced them, and library versions.

The default arguments reproduce the phase-2 baseline grid:

    python -m experiments.run_grid
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.chunkers import (
    Chunker,
    FixedTokenChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from src.data import QADataset, Question, download_squad, load_squad, sample_questions
from src.metrics import hit_at_k, span_scores, take_until_budget
from src.retrievers import BM25Retriever, Retriever
from src.tokenization import RegexWordTokenizer, TokenIndex

ROOT = Path(__file__).resolve().parent.parent

CHUNKERS = ("fixed", "sentence", "recursive")
RETRIEVERS = ("bm25",)


@dataclass(frozen=True)
class GridConfig:
    """One point of the experiment grid.

    ``overlap`` is interpreted per chunker family: tokens for ``fixed``,
    sentences for ``sentence``. ``recursive`` has no overlap knob in v1 and
    accepts only 0. ``budget_rule`` selects how the budget boundary is
    handled (see ``metrics.take_until_budget``); the default ``"stop"`` is
    the primary protocol, ``"truncate"`` the robustness variant.
    """

    dataset: str
    chunker: str
    chunk_size: int
    overlap: int
    retriever: str
    budgets: tuple[int, ...]
    hit_ks: tuple[int, ...]
    per_doc_cap: int
    seed: int
    budget_rule: str = "stop"

    @property
    def config_id(self) -> str:
        # The default rule is omitted so ids (and result filenames) from
        # grids run before the rule existed remain valid.
        rule = "" if self.budget_rule == "stop" else f"_{self.budget_rule}"
        return (
            f"{self.dataset}_{self.chunker}{self.chunk_size}_o{self.overlap}"
            f"_{self.retriever}_cap{self.per_doc_cap}_seed{self.seed}{rule}"
        )


def make_chunker(name: str, chunk_size: int, overlap: int) -> Chunker:
    if name == "fixed":
        return FixedTokenChunker(chunk_size=chunk_size, overlap=overlap)
    if name == "sentence":
        return SentenceChunker(max_tokens=chunk_size, overlap_sentences=overlap)
    if name == "recursive":
        if overlap != 0:
            raise ValueError("recursive chunker has no overlap knob in v1")
        return RecursiveCharacterChunker(max_tokens=chunk_size)
    raise ValueError(f"unknown chunker {name!r}")


def make_retriever(name: str) -> Retriever:
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"unknown retriever {name!r}")


def run_metadata() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        if git("status", "--porcelain"):
            commit += "+dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_config(
    cfg: GridConfig, dataset: QADataset, questions: tuple[Question, ...]
) -> dict:
    """Evaluate one configuration; returns the JSON-serializable result.

    Per question the record stores, for every budget: span recall / precision
    / IoU, the number of chunks retrieved, and the prompt tokens actually
    spent — the last two make budget-utilization analysis possible without
    rerunning (a large-chunk config that cannot fit a single chunk under a
    small budget shows up as tokens=0, not as a mysterious zero score).
    """
    tokenizer = RegexWordTokenizer()
    chunker = make_chunker(cfg.chunker, cfg.chunk_size, cfg.overlap)
    by_doc: dict[str, list[Question]] = {}
    for q in questions:
        by_doc.setdefault(q.doc_id, []).append(q)
    meta = run_metadata()
    t0 = time.perf_counter()
    records: list[dict] = []
    chunk_tokens: list[int] = []
    for doc_id in sorted(by_doc):
        doc = dataset.documents[doc_id]
        index = TokenIndex(doc.text, tokenizer)
        chunks = chunker.chunk(doc.text)
        chunk_tokens.extend(index.count_in(c.start, c.end) for c in chunks)
        retriever = make_retriever(cfg.retriever).fit([c.text for c in chunks])
        for question in by_doc[doc_id]:
            ranked = [chunks[i] for i in retriever.rank(question.text)]
            budgets: dict[str, dict] = {}
            for budget in cfg.budgets:
                taken = take_until_budget(ranked, index, budget, rule=cfg.budget_rule)
                scores = span_scores(taken, question.gold_alternatives, index)
                budgets[str(budget)] = {
                    "recall": round(scores.recall, 6),
                    "precision": round(scores.precision, 6),
                    "iou": round(scores.iou, 6),
                    "chunks": len(taken),
                    "tokens": sum(index.count_in(c.start, c.end) for c in taken),
                }
            hits = {
                str(k): hit_at_k(ranked, question.gold_alternatives, index, k)
                for k in cfg.hit_ks
            }
            records.append(
                {"qid": question.qid, "doc_id": doc_id, "budgets": budgets, "hits": hits}
            )
    counts = sorted(chunk_tokens)
    return {
        "config": dataclasses.asdict(cfg),
        "meta": meta | {"runtime_s": round(time.perf_counter() - t0, 3)},
        "chunk_stats": {
            "n_chunks": len(counts),
            "tokens_min": counts[0],
            "tokens_median": counts[len(counts) // 2],
            "tokens_mean": round(sum(counts) / len(counts), 2),
            "tokens_max": counts[-1],
        },
        "n_questions": len(records),
        "records": records,
    }


def result_path(raw_dir: Path, cfg: GridConfig) -> Path:
    return raw_dir / f"{cfg.config_id}.json.gz"


def run_and_save(
    cfg: GridConfig,
    dataset: QADataset,
    questions: tuple[Question, ...],
    raw_dir: Path,
    force: bool = False,
) -> tuple[Path, bool]:
    """Run one config and persist it; returns (path, ran).

    Existing result files are trusted and skipped unless ``force`` — config
    ids encode every grid parameter, so a stale file can only mean the same
    config was already run.
    """
    path = result_path(raw_dir, cfg)
    if path.exists() and not force:
        return path, False
    result = run_config(cfg, dataset, questions)
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    tmp.replace(path)
    return path, True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the chunking benchmark grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="dev-v1.1", choices=("dev-v1.1", "dev-v2.0"))
    parser.add_argument("--chunkers", nargs="+", default=list(CHUNKERS), choices=CHUNKERS)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--overlaps", nargs="+", type=int, default=[0])
    parser.add_argument("--retrievers", nargs="+", default=["bm25"], choices=RETRIEVERS)
    parser.add_argument("--budgets", nargs="+", type=int, default=[200, 400, 800, 1600])
    parser.add_argument("--hit-ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--per-doc-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--budget-rule",
        default="stop",
        choices=("stop", "truncate"),
        help="budget boundary handling (see metrics.take_until_budget)",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument(
        "--force", action="store_true", help="rerun configs whose result files exist"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_path = ROOT / "data" / f"{args.dataset}.json"
    if not data_path.exists():
        download_squad(ROOT / "data")
    dataset = load_squad(data_path, name=args.dataset)
    questions = sample_questions(dataset, args.per_doc_cap, args.seed)
    configs = [
        GridConfig(
            dataset=args.dataset,
            chunker=chunker,
            chunk_size=size,
            overlap=overlap,
            retriever=retriever,
            budgets=tuple(args.budgets),
            hit_ks=tuple(args.hit_ks),
            per_doc_cap=args.per_doc_cap,
            seed=args.seed,
            budget_rule=args.budget_rule,
        )
        for chunker in args.chunkers
        for size in args.sizes
        for overlap in args.overlaps
        for retriever in args.retrievers
    ]
    print(
        f"{args.dataset}: {len(dataset.documents)} documents, "
        f"{len(questions)} sampled questions, {len(configs)} configs"
    )
    for cfg in configs:
        t0 = time.perf_counter()
        path, ran = run_and_save(cfg, dataset, questions, args.raw_dir, force=args.force)
        status = f"{time.perf_counter() - t0:6.1f}s" if ran else "cached"
        print(f"  {cfg.config_id}: {status} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

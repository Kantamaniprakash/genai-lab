"""Loading and aggregating raw grid results.

Raw result files (one gzipped JSON per configuration, written by
``experiments.run_grid``) hold per-question scores. This module turns them
into comparable score vectors and bootstrap intervals; rendering lives in
``experiments.summarize`` and ``experiments.make_figures``.

Paired comparisons are only meaningful between runs scored on identical
question sets in identical order — ``check_aligned`` enforces that instead
of trusting filenames.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from src.metrics import BootstrapResult, paired_bootstrap

_CHUNKER_ORDER = {"fixed": 0, "sentence": 1, "recursive": 2}


@dataclass(frozen=True)
class RunResult:
    """One configuration's raw result file, parsed."""

    config: dict
    meta: dict
    chunk_stats: dict
    records: tuple[dict, ...]
    # Only some retrievers report fit statistics (LSA: realized latent ranks);
    # absent for every other retriever and for files written before it existed.
    retriever_stats: dict | None = None

    @property
    def label(self) -> str:
        base = f"{self.config['chunker']}-{self.config['chunk_size']}"
        if self.config["overlap"]:
            base = f"{base}/o{self.config['overlap']}"
        if self.config["budget_rule"] != "stop":
            base = f"{base}/{self.config['budget_rule']}"
        if self.config.get("tokenizer", "regex") != "regex":
            base = f"{base}/{self.config['tokenizer']}"
        return base

    def qids(self) -> tuple[str, ...]:
        return tuple(r["qid"] for r in self.records)

    def metric(self, name: str, budget: int) -> list[float]:
        """Per-question scores for one span metric at one budget."""
        return [r["budgets"][str(budget)][name] for r in self.records]

    def hits(self, k: int) -> list[float]:
        return [float(r["hits"][str(k)]) for r in self.records]

    def tokens_used(self, budget: int) -> list[int]:
        return [r["budgets"][str(budget)]["tokens"] for r in self.records]


def sort_key(rr: RunResult) -> tuple:
    cfg = rr.config
    return (
        _CHUNKER_ORDER.get(cfg["chunker"], 99),
        cfg["chunk_size"],
        cfg["overlap"],
        cfg["budget_rule"],
        cfg.get("tokenizer", "regex"),
    )


def load_raw(
    raw_dir: Path,
    dataset: str | None = None,
    retriever: str | None = None,
    budget_rule: str | None = None,
    overlap: int | None = None,
    seed: int | None = None,
    tokenizer: str | None = "regex",
) -> list[RunResult]:
    """Parse all raw result files, optionally filtered, in presentation order.

    ``None`` filters match everything. Files written before ``budget_rule``
    or ``tokenizer`` existed are stop-rule regex-unit runs by construction;
    the keys are filled in on load so downstream code never special-cases
    them. Different seeds sample different question sets, so any caller doing
    paired comparisons must pin a single seed or ``check_aligned`` will
    (correctly) refuse to proceed.

    ``tokenizer`` is the one filter that defaults closed (``"regex"``)
    rather than open: runs under a different token unit share question ids
    with the primary grid, so ``check_aligned`` cannot catch the mistake of
    pairing them — the scores would align and mean nothing. Callers that
    want cross-unit files ask for them explicitly.
    """
    results = []
    for path in sorted(raw_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        payload["config"].setdefault("budget_rule", "stop")
        payload["config"].setdefault("tokenizer", "regex")
        rr = RunResult(
            config=payload["config"],
            meta=payload["meta"],
            chunk_stats=payload["chunk_stats"],
            records=tuple(payload["records"]),
            retriever_stats=payload.get("retriever_stats"),
        )
        if dataset is not None and rr.config["dataset"] != dataset:
            continue
        if retriever is not None and rr.config["retriever"] != retriever:
            continue
        if budget_rule is not None and rr.config["budget_rule"] != budget_rule:
            continue
        if overlap is not None and rr.config["overlap"] != overlap:
            continue
        if seed is not None and rr.config["seed"] != seed:
            continue
        if tokenizer is not None and rr.config["tokenizer"] != tokenizer:
            continue
        results.append(rr)
    results.sort(key=sort_key)
    return results


def check_aligned(results: list[RunResult]) -> None:
    """Fail loudly if the runs were not scored on the same question sequence."""
    if not results:
        raise ValueError("no results to align")
    reference = results[0].qids()
    for rr in results[1:]:
        if rr.qids() != reference:
            raise ValueError(
                f"question sets differ between {results[0].label} and {rr.label}; "
                "paired comparison would be invalid"
            )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def mean_ci(values: list[float], seed: int = 0) -> BootstrapResult:
    """95% bootstrap CI of a mean (degenerate paired bootstrap against zero)."""
    return paired_bootstrap(values, [0.0] * len(values), seed=seed)


def diff_ci(
    scores_a: list[float], scores_b: list[float], seed: int = 0
) -> BootstrapResult:
    """95% paired bootstrap CI for mean(A − B) over shared questions."""
    return paired_bootstrap(scores_a, scores_b, seed=seed)

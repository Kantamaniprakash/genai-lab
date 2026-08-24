"""Execution order for a judge grid: coverage-balanced item scheduling.

Why this module exists. :func:`~src.data.stratified_sample` returns its items
sorted by ``item_id`` and the grid runner used to walk exactly that order, so a
grid stopped part-way had finished an *alphabetical prefix of the subsets*
rather than a subsample of the benchmark. Finding 26 measured what that costs:
at 45/600 the Qwen2.5-7B store was 100% Chat, and on those items the
pick-the-longer-response floor scores 0.978 against 0.425 on the full sample,
which reordered the entire field of judges. Every interim read had to be
restricted to matched items, and even then could say nothing about the
benchmark — only about an unrepresentative corner of it.

The fix is to choose *which* item to judge next so that the finished set is a
stratified sample of the target sample at every point, not only at the end. At
each step the scheduler takes the stratum with the largest proportional deficit

    deficit_s = p_s * (D + 1) - d_s

where ``p_s`` is the stratum's share of the target sample, ``d_s`` the items
finished in it, and ``D`` the total finished so far. This is largest-remainder
apportionment run incrementally. The deficits sum to exactly 1 at every step
(``sum_s p_s == 1``), so from a balanced start the picked stratum's deficit is
at most 1 and drops to at most 0 once it is served: no stratum ever drifts more
than one item from its proportional share in either direction. From an
*unbalanced* start — the 67-item Chat-heavy prefix this scheduler inherited —
the same rule drains the starved strata first and the drift contracts
monotonically until the bound holds again.

Ordering is deterministic and RNG-free: stratum ties break by name, items
inside a stratum keep their ``item_id`` order. A schedule is reproducible from
(sample, finished set) alone. Randomizing the order would serve the same
statistical purpose, but a seeded shuffle only makes prefixes representative
*in expectation*; the deficit rule makes them representative in every
realization, which is what an interim read of a single store actually needs.

Reordering execution is safe for the stores already collected.
:class:`~src.judge.ResultStore` resumes on the *set* of finished
``(model, rubric, order, item_id)`` keys, and :func:`~src.analysis.assemble_pairs`
groups records in ``item_id`` order, so the order judgments are written in is
not observable by any analysis in this project. (The 2026-08-21 log recorded
the opposite — that reordering would break resume-compatibility with the six
collected stores, and kept the alphabetical order for that reason. That was
wrong; checking it is what made this a one-day change.)

The scheduling unit is the *item*, never the single judgment: both presentation
orders of an item are judged consecutively, because the swap pair is the unit
every analysis in this audit consumes. An item left half-finished by an
interrupted run is scheduled first on the next run, so a store carries at most
one orphan record and only while a run is in flight.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .data import PairItem

Stratum = Callable[[PairItem], str]

N_ORDERS = 2


def by_subset(item: PairItem) -> str:
    """Default stratum: the benchmark subset the sample is stratified on."""
    return item.subset


def by_category(item: PairItem) -> str:
    """Coarser stratum, for reporting drift at the level readers reason about."""
    return item.category


def _check_items(items: Sequence[PairItem]) -> None:
    if not items:
        raise ValueError("cannot schedule an empty item set")
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate item ids in schedule input: {dupes[:5]}")


def balanced_order(
    items: Sequence[PairItem],
    finished_orders: Mapping[str, int] | None = None,
    *,
    n_orders: int = N_ORDERS,
    stratum: Stratum = by_subset,
) -> list[PairItem]:
    """Pending items in coverage-balanced execution order.

    ``finished_orders`` maps ``item_id`` to how many of its ``n_orders``
    presentations are already in the store; absent ids count as zero. Items
    with all orders finished are excluded and counted as finished coverage;
    items with some but not all are emitted first (in ``item_id`` order) so an
    interrupted run closes its orphan pair before anything new is started.

    The returned list is exactly the set of items with unfinished work, so
    ``balanced_order(items) == list(items)`` for a fresh grid, up to order.
    """
    _check_items(items)
    finished = dict(finished_orders or {})
    unknown = set(finished) - {item.item_id for item in items}
    if unknown:
        raise ValueError(f"finished_orders has ids outside the sample: {sorted(unknown)[:5]}")
    bad = {k: v for k, v in finished.items() if not 0 <= v <= n_orders}
    if bad:
        raise ValueError(f"finished_orders outside 0..{n_orders}: {sorted(bad.items())[:5]}")

    targets: dict[str, int] = {}
    for item in items:
        targets[stratum(item)] = targets.get(stratum(item), 0) + 1
    total = len(items)
    share = {s: t / total for s, t in targets.items()}

    done_counts: dict[str, int] = {s: 0 for s in targets}
    partial: list[PairItem] = []
    pending: dict[str, list[PairItem]] = {s: [] for s in targets}
    for item in sorted(items, key=lambda it: it.item_id):
        n_done = finished.get(item.item_id, 0)
        if n_done == n_orders:
            done_counts[stratum(item)] += 1
        elif n_done > 0:
            # Counted as coverage: by the time the greedy items run, the
            # orphan pairs ahead of them in the schedule are closed.
            partial.append(item)
            done_counts[stratum(item)] += 1
        else:
            pending[stratum(item)].append(item)

    order: list[PairItem] = list(partial)
    n_done_total = sum(done_counts.values())
    remaining = sum(len(group) for group in pending.values())
    for _ in range(remaining):
        candidates = [s for s, group in pending.items() if group]
        # Largest proportional deficit; ties by stratum name for determinism.
        pick = min(
            candidates,
            key=lambda s: (-(share[s] * (n_done_total + 1) - done_counts[s]), s),
        )
        order.append(pending[pick].pop(0))
        done_counts[pick] += 1
        n_done_total += 1
    return order


def coverage(
    items: Sequence[PairItem],
    finished_item_ids: Sequence[str] | set[str] = (),
    *,
    stratum: Stratum = by_subset,
) -> dict:
    """How representative the finished part of a grid currently is.

    ``drift_items`` is ``d_s - p_s * D``: how many items a stratum is over
    (positive) or under (negative) its proportional share of what has been
    finished so far. ``total_variation`` is the total-variation distance
    between the realized and target stratum distributions — 0 for a perfectly
    proportional prefix, and the single number worth quoting when deciding
    whether an interim read means anything. It is ``None`` before any item is
    finished.
    """
    _check_items(items)
    finished = set(finished_item_ids)
    unknown = finished - {item.item_id for item in items}
    if unknown:
        raise ValueError(f"finished ids outside the sample: {sorted(unknown)[:5]}")

    targets: dict[str, int] = {}
    done: dict[str, int] = {}
    for item in items:
        s = stratum(item)
        targets[s] = targets.get(s, 0) + 1
        done[s] = done.get(s, 0) + (item.item_id in finished)

    total = len(items)
    n_done = len(finished)
    strata = {}
    tv = 0.0
    for s in sorted(targets):
        target_share = targets[s] / total
        done_share = done[s] / n_done if n_done else 0.0
        tv += abs(done_share - target_share)
        strata[s] = {
            "target": targets[s],
            "done": done[s],
            "target_share": target_share,
            "done_share": done_share,
            "drift_items": done[s] - target_share * n_done,
        }
    return {
        "n_target": total,
        "n_done": n_done,
        "strata": strata,
        "max_abs_drift_items": max((abs(v["drift_items"]) for v in strata.values()), default=0.0),
        "total_variation": (tv / 2) if n_done else None,
    }


def format_coverage(report: dict, *, label: str) -> str:
    """One-line summary of a coverage report, for run logs."""
    tv = report["total_variation"]
    tv_text = "n/a" if tv is None else f"{tv:.3f}"
    return (
        f"[coverage:{label}] {report['n_done']}/{report['n_target']} items finished; "
        f"total-variation from target {tv_text}, "
        f"max drift {report['max_abs_drift_items']:.2f} items"
    )

"""Run one judge over a stratified sample in both presentation orders.

Config-driven, append-only, idempotent: rerunning the same (model, rubric,
n, seed) skips every judgment already in the store, so an interrupted grid
resumes from where it stopped. Both orders of an item are judged
consecutively, so a partial file still contains complete swap pairs for all
finished items (the unit every analysis needs).

*Which* item comes next is chosen by :mod:`src.schedule`, which serves the
subset with the largest proportional deficit, so the finished part of a grid
is a stratified sample of the target sample at every point rather than only at
the end. Before this, execution followed the sample's ``item_id`` order and a
partial grid was an alphabetical prefix of the subsets — see finding 26, where
that made the pick-the-longer floor read 0.978 instead of 0.425 and reordered
the whole field of judges. ``--order sorted`` restores the old behaviour for
reproducing a historical run; it changes only the order judgments are written
in, never the set, since the store resumes on keys and analyses group by
``item_id``.

Usage:
    python -m experiments.run_grid --model qwen2.5-0.5b --rubric minimal \
        --n 600 --seed 0 [--threads 4] [--limit N] [--order balanced|sorted]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fetch, load_rewardbench, stratified_sample  # noqa: E402
from src.judge import MODELS, LlamaJudge, ResultStore  # noqa: E402
from src.prompts import ORDERS, build_both_orders  # noqa: E402
from src.schedule import balanced_order, by_category, coverage, format_coverage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N judgments this invocation (for smoke runs)")
    parser.add_argument("--order", default="balanced", choices=("balanced", "sorted"),
                        help="item execution order: coverage-balanced (default) or the "
                             "legacy item_id order, for reproducing a historical run")
    args = parser.parse_args()

    model = MODELS[args.model]
    fetch()
    items = stratified_sample(load_rewardbench(), n=args.n, seed=args.seed)
    all_prompts = [p for item in items for p in build_both_orders(item, args.rubric)]

    # Size the context window from the whole sample, never from the scheduled
    # subset, so n_ctx is a property of (n, seed, rubric) alone and cannot
    # drift between the sessions a long grid spans. Refuse to truncate, ever.
    sizing = LlamaJudge(model, n_ctx=512, n_threads=args.threads, verify_sha256=False)
    max_tokens = sizing.max_prompt_tokens(all_prompts)
    del sizing
    n_ctx = max_tokens + 16
    print(f"[grid] {len(items)} items x 2 orders = {len(all_prompts)} judgments; "
          f"max prompt {max_tokens} tok -> n_ctx {n_ctx}", flush=True)

    store = ResultStore(model.key, args.rubric)
    done = store.existing_keys()
    finished_orders = {
        item.item_id: sum(
            (model.key, args.rubric, order, item.item_id) in done for order in ORDERS
        )
        for item in items
    }
    complete = [item_id for item_id, n in finished_orders.items() if n == len(ORDERS)]
    if args.order == "balanced":
        scheduled = balanced_order(items, finished_orders)
    else:
        scheduled = [i for i in items if finished_orders[i.item_id] < len(ORDERS)]
    todo = [p for item in scheduled for p in build_both_orders(item, args.rubric)
            if (model.key, args.rubric, p.order, p.item_id) not in done]

    print(f"[grid] {len(done)} judgments already stored ({len(complete)} complete items), "
          f"{len(todo)} to run in {args.order} order", flush=True)
    print(format_coverage(coverage(items, complete), label="subset"), flush=True)
    print(format_coverage(coverage(items, complete, stratum=by_category),
                          label="category"), flush=True)

    judge = LlamaJudge(model, n_ctx=n_ctx, n_threads=args.threads)
    store.write_meta({
        **judge.meta(),
        "sample_n": args.n,
        "sample_seed": args.seed,
        "rubric": args.rubric,
        "n_prompts": len(all_prompts),
        "max_prompt_tokens": max_tokens,
        "execution_order": args.order,
    })

    start = time.perf_counter()
    ran = 0
    pending_orders = dict(finished_orders)
    for prompt in todo:
        record = judge.judge(prompt)
        store.append(record)
        ran += 1
        pending_orders[prompt.item_id] += 1
        if pending_orders[prompt.item_id] == len(ORDERS):
            complete.append(prompt.item_id)
        if ran % 20 == 0 or ran == len(todo):
            elapsed = time.perf_counter() - start
            rate = ran / elapsed
            eta_min = (len(todo) - ran) / rate / 60
            print(f"[grid] {ran}/{len(todo)} ({rate:.2f} judg/s, eta {eta_min:.0f} min)",
                  flush=True)
            print("       " + format_coverage(coverage(items, complete, stratum=by_category),
                                              label="category"), flush=True)
        if args.limit is not None and ran >= args.limit:
            print(f"[grid] stopping at --limit {args.limit}", flush=True)
            break

    print(f"[grid] done: {ran} new judgments in {(time.perf_counter() - start) / 60:.1f} min "
          f"-> {store.path}", flush=True)
    print(format_coverage(coverage(items, complete), label="subset"), flush=True)


if __name__ == "__main__":
    main()

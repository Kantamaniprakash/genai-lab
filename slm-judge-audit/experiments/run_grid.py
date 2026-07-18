"""Run one judge over a stratified sample in both presentation orders.

Config-driven, append-only, idempotent: rerunning the same (model, rubric,
n, seed) skips every judgment already in the store, so an interrupted grid
resumes from where it stopped. The execution order is by item, both orders
consecutively, so a partial file still contains complete swap pairs for all
finished items (the unit every analysis needs).

Usage:
    python -m experiments.run_grid --model qwen2.5-0.5b --rubric minimal \
        --n 600 --seed 0 [--threads 4] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fetch, load_rewardbench, stratified_sample  # noqa: E402
from src.judge import MODELS, LlamaJudge, ResultStore  # noqa: E402
from src.prompts import build_both_orders  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--rubric", default="minimal")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N judgments this invocation (for smoke runs)")
    args = parser.parse_args()

    model = MODELS[args.model]
    fetch()
    items = stratified_sample(load_rewardbench(), n=args.n, seed=args.seed)
    prompts = [p for item in items for p in build_both_orders(item, args.rubric)]

    # Size the context window from the actual sample before paying for the
    # full model load; refuse to truncate anything, ever.
    sizing = LlamaJudge(model, n_ctx=512, n_threads=args.threads, verify_sha256=False)
    max_tokens = sizing.max_prompt_tokens(prompts)
    del sizing
    n_ctx = max_tokens + 16
    print(f"[grid] {len(items)} items x 2 orders = {len(prompts)} judgments; "
          f"max prompt {max_tokens} tok -> n_ctx {n_ctx}", flush=True)

    judge = LlamaJudge(model, n_ctx=n_ctx, n_threads=args.threads)
    store = ResultStore(model.key, args.rubric)
    done = store.existing_keys()
    todo = [p for p in prompts
            if (model.key, args.rubric, p.order, p.item_id) not in done]
    print(f"[grid] {len(done)} judgments already stored, {len(todo)} to run", flush=True)

    store.write_meta({
        **judge.meta(),
        "sample_n": args.n,
        "sample_seed": args.seed,
        "rubric": args.rubric,
        "n_prompts": len(prompts),
        "max_prompt_tokens": max_tokens,
    })

    start = time.perf_counter()
    ran = 0
    for prompt in todo:
        record = judge.judge(prompt)
        store.append(record)
        ran += 1
        if ran % 20 == 0 or ran == len(todo):
            elapsed = time.perf_counter() - start
            rate = ran / elapsed
            eta_min = (len(todo) - ran) / rate / 60
            print(f"[grid] {ran}/{len(todo)} ({rate:.2f} judg/s, eta {eta_min:.0f} min)",
                  flush=True)
        if args.limit is not None and ran >= args.limit:
            print(f"[grid] stopping at --limit {args.limit}", flush=True)
            break

    print(f"[grid] done: {ran} new judgments in {(time.perf_counter() - start) / 60:.1f} min "
          f"-> {store.path}", flush=True)


if __name__ == "__main__":
    main()

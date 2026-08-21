**INTERIM — not a result.** Every judge below is restricted to the 45 items the in-flight `qwen2.5-7b` grid has finished, so the rows are matched (same items, same orders) but the sample is small and *not* representative: `run_grid` walks the sample in `item_id` order, so a partial grid covers an alphabetical prefix of the subsets rather than a random draw. Category composition of the restricted set: Chat 100% (vs 12% in the full sample); Chat Hard 0% (vs 15% in the full sample); Reasoning 0% (vs 48% in the full sample); Safety 0% (vs 25% in the full sample). Chat Hard, Reasoning, Safety are not represented at all. Nothing here is a finding about `qwen2.5-7b`, and none of it belongs in a cross-judge claim until the grid closes over the full sample.

| judge | params | compliant | acc A-first | acc B-first | flip rate | median b | b > 0 | median \|s\| | bias > signal | raw acc | sym acc (95% CI) | Δ sym−raw | Δ sym−longer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.5B | 1.000 | 1.000 | 0.000 | 0.000 | +3.24 | 1.000 | 0.24 | 1.000 | 0.500 | 0.556 [0.422, 0.689] | +0.056 [-0.078, +0.189] | -0.422 [-0.556, -0.289] |
| qwen2.5-1.5b | 1.5B | 1.000 | 0.978 | 0.356 | 0.333 | +1.18 | 0.911 | 0.61 | 0.667 | 0.667 | 0.644 [0.489, 0.778] | -0.022 [-0.144, +0.100] | -0.333 [-0.467, -0.200] |
| qwen2.5-3b | 3B | 1.000 | 0.933 | 0.822 | 0.800 | +0.76 | 0.600 | 8.68 | 0.200 | 0.878 | 0.911 [0.822, 0.978] | +0.033 [-0.033, +0.100] | -0.067 [-0.156, +0.022] |
| qwen2.5-7b | 7B | 1.000 | 1.000 | 0.844 | 0.844 | +1.06 | 0.600 | 11.04 | 0.156 | 0.922 | 0.956 [0.889, 1.000] | +0.033 [-0.022, +0.089] | -0.022 [-0.067, +0.000] |
| llama-3.2-1b | 1B | 0.600 | 0.333 | 0.822 | 0.200 | -0.51 | 0.200 | 0.13 | 0.800 | 0.578 | 0.756 [0.622, 0.867] | +0.178 [+0.056, +0.289] | -0.222 [-0.356, -0.111] |
| llama-3.2-3b | 3B | 0.978 | 1.000 | 0.044 | 0.044 | +2.04 | 1.000 | 0.97 | 0.956 | 0.522 | 0.911 [0.822, 0.978] | +0.389 [+0.300, +0.467] | -0.067 [-0.156, +0.000] |
| *random floor* |  |  |  |  |  |  |  |  |  | 0.500 |  |  |  |
| *always-A floor* |  |  |  |  |  |  |  |  |  | 0.500 |  |  |  |
| *longer-response floor* |  |  |  |  |  |  |  |  |  |  | 0.978 |  |  |

Rubric `minimal`, all judges over the same 45 items, both presentation orders. `b` is position-bias log-odds toward whatever sits in position A; `s` is the order-invariant preference log-odds for the gold-chosen response. Raw accuracy assigns each item's presentation order uniformly at random; symmetrized accuracy is `sign(s)`. Intervals are 95% paired bootstrap over items (10,000 resamples, seed 0). The always-A floor sits at exactly 0.5 over the exhaustive order pair by construction.

| judge | params | compliant | acc A-first | acc B-first | flip rate | median b | b > 0 | median \|s\| | bias > signal | raw acc | sym acc (95% CI) | Δ sym−raw | Δ sym−longer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.5B | 1.000 | 1.000 | 0.002 | 0.002 | +3.65 | 0.998 | 0.24 | 0.998 | 0.501 | 0.568 [0.528, 0.608] | +0.068 [+0.027, +0.107] | +0.143 [+0.088, +0.198] |
| qwen2.5-1.5b | 1.5B | 1.000 | 0.805 | 0.293 | 0.298 | +0.83 | 0.745 | 0.50 | 0.702 | 0.549 | 0.502 [0.462, 0.542] | -0.048 [-0.081, -0.013] | +0.077 [+0.026, +0.129] |
| qwen2.5-3b | 3B | 1.000 | 0.368 | 0.865 | 0.380 | -5.55 | 0.192 | 3.64 | 0.620 | 0.617 | 0.742 [0.707, 0.777] | +0.125 [+0.095, +0.154] | +0.317 [+0.269, +0.361] |
| llama-3.2-1b | 1B | 0.512 | 0.312 | 0.728 | 0.183 | -0.34 | 0.275 | 0.14 | 0.817 | 0.520 | 0.555 [0.517, 0.595] | +0.035 [-0.001, +0.072] | +0.130 [+0.084, +0.177] |
| llama-3.2-3b | 3B | 0.863 | 0.990 | 0.023 | 0.033 | +2.34 | 0.998 | 0.44 | 0.967 | 0.507 | 0.652 [0.613, 0.690] | +0.145 [+0.107, +0.182] | +0.227 [+0.181, +0.273] |
| *random floor* |  |  |  |  |  |  |  |  |  | 0.500 |  |  |  |
| *always-A floor* |  |  |  |  |  |  |  |  |  | 0.500 |  |  |  |
| *longer-response floor* |  |  |  |  |  |  |  |  |  |  | 0.425 |  |  |

Every completed grid for rubric `minimal`, same 600 stratified RewardBench items, both presentation orders. `b` is position-bias log-odds toward whatever sits in position A; `s` is the order-invariant preference log-odds for the gold-chosen response. Raw accuracy assigns each item's presentation order uniformly at random; symmetrized accuracy is `sign(s)`. Intervals are 95% paired bootstrap over items (10,000 resamples, seed 0). The always-A floor sits at exactly 0.5 over the exhaustive order pair by construction.

`Δ sym−longer` compares each judge against the *fixed* pick-the-longer-response rule, which scores 0.425 here — below chance, because RewardBench's composition punishes verbosity. Clearing a below-chance floor is a weak test, and this column is not the length-baseline verdict: the real opponent is the *fitted* one-parameter length model, which is free to learn the anti-verbosity direction and scores 0.575 on these items. Only the two 3B judges beat that one (findings 13–14, 18, 22).

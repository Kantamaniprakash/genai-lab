# slm-judge-audit

**A white-box reliability audit of small open-weight LLMs as pairwise judges:
position bias measured in log-odds, calibration, and value over trivial
baselines, with paired bootstrap confidence intervals.**

*Status: phase 2 (baselines & main grid) — harness complete (runner, analysis
core, floors; 47 tests). Two full grids done on the same 600-item stratified
sample × both orders: Qwen2.5-0.5B and Llama-3.2-1B, findings 1–7 below. The
scaling grid continues with Qwen2.5-1.5B/3B next.*

## Abstract

Small open-weight models (0.5B–8B) are widely used as cheap judges: filtering
synthetic data, scoring RLAIF candidates, running large eval sweeps where a
frontier-judge call per comparison is unaffordable. Existing reliability
audits treat judges as API black boxes — they sample a verdict and count flips
under order swap. Local open-weight judges permit strictly more: the full
next-token distribution. This project audits small judges *white-box*: each
pairwise judgment is read out as the renormalized probability over the verdict
tokens {A, B} at a single position, giving a verdict **log-odds** per
(item, order). For every item the swap pair (z_chosen-first, z_rejected-first)
then decomposes *exactly* into an order-invariant preference component and a
position-bias component. On top of this decomposition the audit measures:
(1) how large position bias is relative to the preference signal across model
scale; (2) whether position bias behaves as an additive shift (a hypothesis
prior work assumes implicitly when it "debiases by swapping" — here it is
tested); (3) how much accuracy symmetrization actually recovers;
(4) whether verdict probabilities are calibrated; and (5) whether small
judges add signal beyond a pick-the-longer-response heuristic. All
comparisons use gold human-verified labels and paired bootstrap confidence
intervals.

## Motivation

The judge-reliability literature is active but almost entirely black-box:
verdicts are sampled, and reliability is quantified by agreement and flip
rates. That conflates two different failure modes — a judge that is *noisy*
(unstable near 50/50) and a judge that is *biased* (systematically shifted
toward a position) — which have different remedies and different scaling
behavior. Reading probabilities instead of samples separates them, at zero
extra compute cost. And because a single-token readout is a prefill-only
forward pass, an audit of exactly the model class people deploy for cheap
judging (≤8B, quantized, CPU-servable) is feasible end-to-end on commodity
hardware — which this repo demonstrates by running everything on 4 CPU cores.

## Method

For item *i* with gold pair (chosen, rejected) and judge *j*:

- Build the identical judge prompt in both presentation orders:
  `chosen_first` (gold-preferred response shown as A) and `rejected_first`
  (shown as B). Prompts never reveal the gold label.
- At the first assistant token, read full-vocabulary logits and take
  `z = logit("A") − logit("B")` — the verdict log-odds toward position A.
  Greedy verdicts, format compliance of the unconstrained argmax, and the
  probability mass on {A, B} are recorded alongside.
- Exact per-item decomposition of the swap pair:
  - **preference** `s_i = (z_cf − z_rf) / 2` — order-invariant log-odds in
    favor of the gold-chosen response; `sign(s_i)` is the symmetrized
    (debiased) verdict.
  - **position bias** `b_i = (z_cf + z_rf) / 2` — log-odds pushed toward
    whatever occupies position A, regardless of content.

  The decomposition is an identity, not a model. The *additive-shift
  hypothesis* — `b_i ≈ b` constant across items — is what swap-averaging
  debiasing implicitly assumes, and phase 3 tests it (dispersion of `b_i`,
  dependence on length gap, preference magnitude, category).

## Data

Single pinned artifact: the **RewardBench** filtered evaluation set (2,985
human-verified chosen/rejected pairs, 23 subsets, 4 categories) at repository
revision `168d848`, SHA256-verified at fetch, per-subset composition verified
at load. The `llmbar-*` subsets are the complete **LLMBar** meta-evaluation
benchmark (419 instances with objective gold preferences), so the adversarial
instruction-following axis is embedded in the same artifact — LLMBar is
deliberately *not* loaded separately, which would double-count it. Stratified
subsampling (largest-remainder by subset, seeded) preserves composition for
budget-limited grids.

## First results — Qwen2.5-0.5B, minimal rubric, n=600, both orders

600 stratified RewardBench items (seed 0, composition-preserving), both
presentation orders = 1,200 judgments; 56.5 min on 4 CPU threads. All
intervals are 95% percentile bootstrap over items (10,000 resamples, paired
where the comparison is paired). Regenerate with the commands below; nothing
is hand-entered.

| metric | value |
|---|---|
| argmax compliance / median mass on {A, B} | 1.000 / ≈1.00 |
| raw accuracy, chosen shown first | 1.000 |
| raw accuracy, rejected shown first | 0.002 |
| raw accuracy, random order assignment | 0.501 [0.500, 0.502] |
| symmetrized (swap-averaged) accuracy | 0.568 [0.528, 0.608] |
| paired gain, symmetrized − raw | +0.068 [+0.027, +0.107] |
| positional flip rate under swap | 0.002 |
| position bias b: mean (sd), share > 0 | +3.68 (1.08), 99.8% |
| preference signal: median \|s\| | 0.24 |
| items where \|b\| > \|s\| | 99.8% |
| floors: random / always-A / longer-response | 0.500 / 0.500 / 0.425 |

**The 0.5B judge is functionally an always-A machine** (finding 2): position
bias exceeds the content signal on 99.8% of items (median ratio ~15x), so
raw accuracy is pure position-assignment noise. **A black-box flip-rate audit
cannot see this** (finding 3): the flip rate is 0.002, which reads as
near-perfect consistency — the bias saturates both orders. The white-box
decomposition shows that "consistency" *is* the bias:

![Swap-pair decomposition](results/figures/qwen2.5-0.5b__minimal_decomposition.png)

*Every item's swap pair, decomposed: position bias b_i (x) vs order-invariant
preference s_i (y). The cloud sits ~3.7 log-odds right of zero — bias toward
position A on 99.8% of items — while content preference hugs zero (median
|s| = 0.24). A judge with no position bias and real discrimination would
concentrate around b = 0 with |s| large.*

**Symmetrization rescues a real but weak signal** (finding 4): swap-averaged
accuracy is 0.568 [0.528, 0.608], significantly above random, always-A, and
the longer-response floor — which is itself *below chance* (0.425) on
RewardBench, whose adversarial subsets punish verbosity-picking. Per
category, the debiased judge is best on Safety (0.608) and has no signal at
all on easy Chat (0.500).

![Accuracy by presentation order](results/figures/qwen2.5-0.5b__minimal_accuracy.png)

*Accuracy by presentation order against the floors. The 1.000 / 0.002 split
between orders collapses to chance under random order assignment; only the
swap-averaged verdict carries signal.*

## Cross-family contrast — Llama-3.2-1B on the identical sample

Same 600 items, same orders, same rubric:

| metric | Qwen2.5-0.5B | Llama-3.2-1B |
|---|---|---|
| argmax compliance (both orders) | 1.000 | 0.512 |
| per-judgment mass on {A, B}, quartiles | ≈1.0 | 0.10 / 0.67 / 0.94 |
| position bias b: median (share > 0) | +3.65 (99.8%) | −0.34 (27.5%) |
| raw accuracy cf / rf | 1.000 / 0.002 | 0.312 / 0.728 |
| raw accuracy, random order | 0.501 [0.500, 0.502] | 0.520 [0.502, 0.537] |
| positional flip rate | 0.002 | 0.183 |
| items with \|b\| > \|s\| | 99.8% | 81.7% |
| symmetrized accuracy | 0.568 [0.528, 0.608] | 0.555 [0.517, 0.595] |
| paired gain from symmetrization | +0.068 [+0.027, +0.107] | +0.035 [−0.001, +0.072] |

Three results (findings 5–7, `research/NOTES.md`):

- **Verdict-format compliance is a family property.** Llama-3.2-1B's
  unconstrained argmax is a verdict letter in only 56% of judgments (it
  prefers to open with "Response…"), so its single-token readout measures a
  renormalized sub-distribution — every Llama number here carries that
  qualification, and the audit records exactly how much (mass quartiles
  above).
- **The flip-rate ranking inverts the true bias ranking.** Llama-1B flips
  under swap 90x more often than Qwen-0.5B (0.183 vs 0.002) — a black-box
  consistency audit would call it far less reliable — while its positional
  bias is ~10x *smaller* (median |b| 0.34 vs 3.65). Flip rate measures bias
  saturation, not bias.
- **Bias direction is family- and category-dependent.** Llama leans toward
  B overall, but its Reasoning items pull toward A (+0.25 mean b) while
  Chat/Safety sit at −0.3 to −0.5 — early evidence against the
  additive-shift assumption implicit in swap-debiasing, ahead of the formal
  phase-3 test.

![Llama swap-pair decomposition](results/figures/llama-3.2-1b__minimal_decomposition.png)

*Llama-3.2-1B's decomposition on the same axes: the cloud centers near
b ≈ −0.3 (mild B-lean) instead of +3.7, with a category-structured right
tail — Reasoning items are biased in the opposite direction from the rest.*

## Planned experiments

1. **Scaling grid** — Qwen2.5-Instruct 0.5B/1.5B/3B/7B, Llama-3.2-Instruct
   1B/3B, and peers (Q4_K_M GGUF), on a stratified sample in both orders;
   trivial floors (always-A, longer-response, random) alongside.
   *(Qwen2.5-0.5B and Llama-3.2-1B done above.)*
2. **Bias anatomy** — dispersion and covariates of `b_i`; test of the
   additive-shift hypothesis; accuracy recovered by symmetrization.
3. **Calibration** — reliability diagrams and ECE of `P(correct)` from
   verdict probabilities, raw vs. symmetrized.
4. **Value over length** — logistic regression of gold on judge log-odds vs.
   length delta; does a small judge beat "pick the longer answer"?
5. **Prompt sensitivity** — minimal vs. detailed rubric as a paired
   comparison in log-odds space.

## Feasibility pilot (2026-07-17, real measurements, anecdote scale)

Qwen2.5-0.5B-Instruct (Q4_K_M) on 4 CPU threads, three real RewardBench
items: "A"/"B" tokenize as single tokens; the unconstrained argmax at the
verdict position was the letter itself in all cases; prefill throughput
153–207 tok/s (197-token prompt: 1.0 s; 2,768-token worst-case: 18 s). On
three items spanning the length distribution, the swap-pair decomposition
gave position bias `b_i` of **+4.1 to +4.8** log-odds toward position A
against preference magnitudes `|s_i|` of **0.02–0.37** — the 0.5B judge's
position bias exceeded its content signal by an order of magnitude on every
item tried. This is a pilot observation on n=3 with no confidence intervals;
the phase-2 grid will measure it properly.

## Repository layout

```
src/data.py       pinned RewardBench download, validation, stratified sampling
src/prompts.py    rubric templates, order swap, single-token verdict design
src/judge.py      llama.cpp runner: chat templates, pinned GGUFs, logit
                  readout, resumable JSONL result stores
src/analysis.py   swap-pair assembly, s/b decomposition, paired bootstrap
src/baselines.py  always-A / longer-response / random floors
experiments/      run_grid, summarize, make_figures
results/raw/      one JSONL store per (model, rubric) + provenance sidecar
results/summary/  quick-look JSON per store
results/figures/  committed PNGs, regenerable from raw stores
tests/            47 tests (schema, templates, readout arithmetic, store
                  resume, decomposition, bootstrap, floors, model smoke)
research/NOTES.md living research log
```

## Reproducing (current state)

```bash
uv sync                      # analysis deps (numpy, pyarrow, matplotlib)
uv run python -m src.data    # fetch pinned parquet, print composition
uv run --group dev pytest    # 47 tests
uv sync --group judge        # llama-cpp-python (compiles ~5 min on 4 cores)
# download the pinned GGUF named in src/judge.py MODELS into models/, then:
uv run python -m experiments.run_grid --model qwen2.5-0.5b --rubric minimal --n 600 --seed 0
uv run python -m experiments.summarize   # tables in results/summary/
uv run python -m experiments.make_figures
```

## References

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
  Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
  Joseph E. Gonzalez, Ion Stoica. *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena.* NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen
  Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah
  A. Smith, Hannaneh Hajishirzi. *RewardBench: Evaluating Reward Models for
  Language Modeling.* [arXiv:2403.13787](https://arxiv.org/abs/2403.13787)
- Zhiyuan Zeng, Jiatong Yu, Tianyu Gao, Yu Meng, Tanya Goyal, Danqi Chen.
  *Evaluating Large Language Models at Evaluating Instruction Following.*
  ICLR 2024. [arXiv:2310.07641](https://arxiv.org/abs/2310.07641)
- Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, Soroush
  Vosoughi. *Judging the Judges: A Systematic Study of Position Bias in
  LLM-as-a-Judge.* AACL-IJCNLP 2025.
  [arXiv:2406.07791](https://arxiv.org/abs/2406.07791)
- Justin D. Norman, Michael U. Rivera, D. Alex Hughes. *Reliability without
  Validity: A Systematic, Large-Scale Evaluation of LLM-as-a-Judge Models
  Across Agreement, Consistency, and Bias.*
  [arXiv:2606.19544](https://arxiv.org/abs/2606.19544) — closest neighbor:
  21 judges, black-box; this project is the white-box, small-model
  counterpart.
- *Self-Preference Bias in LLM-as-a-Judge.*
  [arXiv:2410.21819](https://arxiv.org/abs/2410.21819)
- *JudgeBench: A Benchmark for Evaluating LLM-based Judges.* ICLR 2025.
  [arXiv:2410.12784](https://arxiv.org/abs/2410.12784)
- *Thinking Small Models are Efficient LLM Judges.*
  [arXiv:2509.13332](https://arxiv.org/abs/2509.13332)
- *JudgeBoard: Benchmarking and Enhancing Small Language Models for Reasoning
  Evaluation.* [arXiv:2511.15958](https://arxiv.org/abs/2511.15958)
- *SLMJury: Can Small Language Models Judge as Well as Large Ones?*
  [arXiv:2606.07810](https://arxiv.org/abs/2606.07810)

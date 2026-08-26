# genai-lab

[![CI](https://img.shields.io/github/actions/workflow/status/Kantamaniprakash/genai-lab/ci.yml?branch=main&label=CI)](https://github.com/Kantamaniprakash/genai-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

Hands-on experiments with current Gen AI techniques — RAG, agents, evals, fine-tuning and whatever's moving the field this week

<!-- latest-start -->
## Latest from the lab

<!-- auto-generated from research/NOTES.md by scripts/sync_latest.py; do not hand-edit -->

**2026-08-26 — Day 9, second act: the 8B grid runs end-to-end; the scaling axis closes (findings 32–35)**

- Finding 32 — the two family arcs never cross: at the top tier, family beats scale.
- Finding 33 — the adversarial hole is a family property all the way up.
- Finding 34 — the audit's strongest length-controlled signal rides its strongest length lean, and pays for it on math-prm.
- Finding 35 — the one-call ceiling holds at ~25% at 8B, finer corrections actively hurt, and the Safety compliance migration replicates.

[Full entry →](slm-judge-audit/research/NOTES.md#2026-08-26--day-9-second-act-the-8b-grid-runs-end-to-end-the-scaling-axis-closes-findings-3235)
<!-- latest-end -->

![Symmetrized and raw judge accuracy vs parameter count, with the bias-versus-signal race](slm-judge-audit/results/figures/scaling__minimal.png)

*Flagship result from [`slm-judge-audit`](slm-judge-audit/): debiased accuracy
of small open-weight judges is not monotone in scale, and the raw accuracy most
audits report hides it — regenerated from committed per-judgment results.*

## Current flagship

**[`slm-judge-audit`](slm-judge-audit/)** — a white-box reliability audit of
small open-weight LLMs as pairwise judges. Every judgment is read out as verdict
log-odds at a single token position, so each item's swap pair decomposes exactly
into an order-invariant preference and a position-bias term — which turns
position bias, debiasing gains, and calibration into things you can measure
rather than assume. See [ROADMAP.md](ROADMAP.md) for the rationale, phases, and
project backlog, and
[`slm-judge-audit/research/NOTES.md`](slm-judge-audit/research/NOTES.md) for the
day-by-day research log.

### Results at a glance

The scaling grid is complete: seven grids (Qwen2.5 0.5B/1.5B/3B/7B and
Llama-3 1B/3B/8B, all Q4_K_M) on the same 600 stratified
[RewardBench](https://arxiv.org/abs/2403.13787) items in both presentation
orders, everything on 4 CPU cores. Findings 1–35 and the full cross-judge
table live in the
[project README](slm-judge-audit/README.md#results-at-a-glance):

- **Debiased judge quality is not monotone in scale — and at the top tier,
  family beats scale.** Within Qwen2.5 it runs **0.568 → 0.502 → 0.742 →
  0.837** from 0.5B to 7B — a valley at 1.5B, where the preference that
  emerges with scale is a *verbosity* preference that RewardBench punishes,
  then a monotone climb. Llama-3 runs 0.555 → 0.652 → 0.723 with no valley,
  and its 8B is statistically indistinguishable from Qwen's 3B — a judge
  2.7x smaller from the other family.
- **Flip-rate "consistency" is anti-informative at both ends.** The two most
  saturated judges post the *lowest* flip rates in the audit (0.002 and
  0.033): a bias large enough never to be overturned never produces a flip to
  count. And the audit's two *best* judges post the two *highest* (0.732 and
  0.665), because a content-following verdict changes letter whenever the
  responses swap seats.
- **Position bias beats the content signal** on 62.0% to 99.8% of items for
  five of the seven judges — only Qwen2.5-7B (26.8%) and Llama-3.1-8B
  (33.5%) are *signal-dominant* — and the two families reverse bias
  *direction* with scale in opposite senses.
- **The assumption behind cheap debiasing is false.** Position bias is never
  an additive constant, and the share of the two-call symmetrization gain a
  fitted one-call correction recovers *falls* as judges improve: 68% at 0.5B,
  47% at 3B, ~25% at 7B and 8B.
- **Most of these judges do not beat a length heuristic, and adversarial
  robustness is a family property.** Against a fitted one-parameter length
  baseline (0.575), only both 3Bs, the 7B and the 8B come out ahead — the 7B
  by the largest margin and in every category, while Llama's below-chance
  hole on adversarial pairs persists to 8B (Chat Hard 0.522 vs Qwen-7B's
  0.696 on identical items).

## Completed flagship — `rag-chunking-bench`

**[`rag-chunking-bench`](rag-chunking-bench/)** — a token-budget-controlled
benchmark of chunking strategies for RAG retrieval, with span-level metrics and
paired bootstrap confidence intervals. Closed 2026-07-16 at 26 findings, 365
tests, and a byte-level reproduction audit; day-by-day log in
[`rag-chunking-bench/research/NOTES.md`](rag-chunking-bench/research/NOTES.md).

![Budget-matched SpanRecall@400 by chunking strategy and chunk size, with 95% bootstrap CIs](rag-chunking-bench/assets/hero_spanrecall_dev-v1.1_bm25.png)

*Once the retrieved-token budget is held constant, smaller chunks win in every
chunker family — regenerated from committed per-question results.*

### Results at a glance

Measured on SQuAD dev-v1.1 reconstructed articles (48 documents, 2,400
sampled questions) with BM25, TF-IDF, and LSA retrieval, against a classic
fixed-k evaluation as the baseline protocol. Every number comes from
per-question score files checked into
[`rag-chunking-bench/results/raw/`](rag-chunking-bench/results/raw/) and
carries a 95% paired bootstrap confidence interval:

- **Budget-matched, smaller chunks win.** At a 400-token budget, 64-token
  fixed chunks beat 256-token ones by **+0.134 [+0.117, +0.152]**
  SpanRecall.
- **Fixed-k evaluation reverses the ranking.** hit@5 rises with chunk size
  (0.873 → 0.969) while budget-matched SpanRecall@400 falls (0.879 → 0.023)
  — the token-budget confound in standard chunking comparisons is real and
  large.
- **Sentence alignment adds a small significant edge** (+0.041 [+0.029,
  +0.052] at size 64, B=400); ~25% overlap pays off for fixed windows at
  tight budgets but is pure cost for sentence packing.
- **None of it is a BM25 artifact.** The size ordering, the reversal, and
  the sentence edge all hold under TF-IDF and LSA — and the chunking effect
  (+0.13–0.19 SpanRecall at B=400) outweighs the retriever effect at small
  chunk sizes (≤ 0.053) several times over.

Full tables, figures, all 26 findings (spanning four retriever families, two
datasets, three sampling seeds, both budget-boundary rules, and two token
units), and an honest
[Limitations](rag-chunking-bench/README.md#limitations) section (small
CPU-sized dense encoder only, contiguous gold evidence only, CPU-only
scale) live in the [project README](rag-chunking-bench/README.md).

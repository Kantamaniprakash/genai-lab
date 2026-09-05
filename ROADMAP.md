# genai-lab Roadmap

This lab runs one flagship research project at a time, worked daily until it would
survive review by a demanding referee. Everything here is real: every number in a
writeup comes from an experiment actually run in this repo.

## Current flagship: `slm-judge-audit` — COMPLETE (2026-07-17 → 2026-09-04); next-flagship selection is the next session's first task

The audit closed 2026-09-04: 49 findings, seven judges, two rubrics, 16,800
judgments, and a release reproduction audit that regenerates all 63 committed
artifacts byte-for-byte from a clean copy of HEAD. Full report:
`slm-judge-audit/README.md`; day-by-day log:
`slm-judge-audit/research/NOTES.md`; complete record below under Completed
flagships. The next session runs the same selection scan that opened this
flagship (arXiv / Papers with Code / release notes, rationale recorded here)
against the backlog; the standing front-runner is hallucination measurement
in RAG answers.

## Completed flagships

### `slm-judge-audit` — 2026-07-17 to 2026-09-04, COMPLETE

**Question.** How reliable are small open-weight LLMs (0.5B–8B, the sizes people
actually deploy for cheap large-scale evaluation) as zero-shot pairwise judges —
when you measure them white-box, at the level of verdict token probabilities
rather than sampled outputs?

**Why this, why now.** LLM-as-judge reliability is one of the most active eval
topics right now (a 2026 wave of position-bias, self-preference, and
bias-mitigation papers). The selection scan (2026-07-17) found the closest
neighbors: "Reliability without Validity" (Norman et al., arXiv:2606.19544)
audits 21 judges across agreement/consistency/bias but treats every judge as an
API black box; JudgeBoard (arXiv:2511.15958), SLMJury (arXiv:2606.07810), and
"Thinking Small" (arXiv:2509.13332) study small judges but accuracy-first. The
gap this lab can own: local open-weight judges expose the **full next-token
distribution**, so verdict preference can be measured as log-odds, position bias
as a per-item shift in log-odds under order swap (a structural model one can
*test*, not assume), debiasing-by-symmetrization can be quantified exactly,
calibration (ECE, reliability diagrams) becomes measurable, and "does the judge
add signal beyond a length heuristic?" becomes a regression question — all with
the paired-bootstrap machinery this lab built in `rag-chunking-bench`. White-box
+ small-scale + statistically careful is a genuine unclaimed corner, and it is
the only corner honestly executable on this hardware (CPU-only; single-token
verdict readout makes every judgment a prefill-only forward pass). Free-tier
hosted API limits were verified prohibitive for the alternative (judge audits
need thousands of calls), which also ruled out the backlog's API-dependent
framing of this project.

**Data.** RewardBench filtered set (Lambert et al., arXiv:2403.13787): 2,985
human-verified chosen/rejected pairs across 23 subsets in 4 categories — which
embeds the complete LLMBar meta-evaluation set (Zeng et al., ICLR 2024,
arXiv:2310.07641) as its llmbar-* subsets, giving an adversarial
instruction-following axis for free. Pinned revision + SHA256, verified at load.

**Outcome.** 49 findings across the seven-judge scaling grid, five analysis
axes, and the rubric-sensitivity axis — 16,800 judgments on one pinned
600-item stratified sample. Headlines: debiased judge quality is
non-monotone in scale (a 1.5B valley where the emergent preference is a
verbosity preference the benchmark punishes) and the two family arcs never
cross — family beats scale at the top tier; black-box flip-rate
"consistency" is anti-correlated with the truth at both of its extremes;
position bias is never an additive constant, and the share of the
symmetrization gain a fitted one-call correction recovers falls as judges
improve (68% → ~25%); a fitted one-parameter length baseline outscores
every sub-3B judge; post-debiasing calibration is a family property; and
verdict fragility under rubric change follows a fitted ratio λ·med|s|/σ
that strictly orders all seven judges — confirmed out of sample on a
pre-registered grid — while prompt-side debiasing never buys symmetrized
quality and turns harmful at 8B. 140 tests; all 63 committed artifacts
(33 tables and summaries, 30 figures) reproduce byte-for-byte from the committed
raw stores in a clean copy of HEAD (`experiments/reproduce.py`, release
audit green 2026-09-04). Full report: `slm-judge-audit/README.md`;
day-by-day log: `slm-judge-audit/research/NOTES.md`.

**Phases.**
1. **Harness** — pinned data layer with category mapping and stratified
   sampling; judge prompt builder with order swap and single-token verdict
   readout; llama.cpp-based judge runner with logit extraction; result store.
   *(done 2026-07-18: runner + analysis core + floors, 47 tests)*
2. **Baselines & main grid** — judge scaling curve (Qwen2.5 0.5B/1.5B/3B/7B,
   Llama-3.2 1B/3B, + peers) on a stratified sample, both orderings; trivial
   baselines (always-A, longer-response, random) as floors.
   *(started 2026-07-18: Qwen2.5-0.5B and Llama-3.2-1B grids done on the
   same 600-item sample — findings 1–7: the always-A machine that
   flip-rate audits would call consistent, and the cross-family inversion
   of the flip-rate vs. true-bias ranking. 2026-07-19: readout-validity
   conditioning — the logit readout survives non-compliance, finding 8 —
   and the Qwen2.5-1.5B grid: inverse scaling of debiased accuracy,
   symmetrization backfires on bias-saturated Reasoning items where the
   emergent preference tracks length, findings 9–11. 2026-07-22: the
   value-over-length probe — every judge has signal beyond length but
   below 3B none beats a fitted one-parameter length baseline, and both
   standing mysteries are length-mediated, findings 12–14; calibration —
   symmetrization repairs it at 0.5B/1B only, finding 15; and the
   Qwen2.5-3B grid: the valley closes, sym 0.742, bias flips to B at the
   largest magnitude yet, verbosity un-learns, first judge to beat the
   length floor, findings 16–18. 2026-07-23: the additive-shift formal
   test + exact-LOO single-order correction ladder — position bias is
   never an additive constant and a fitted one-call correction substitutes
   for symmetrization at 0.5B but caps at ~half the gain at 3B, findings
   19–20; and the Llama-3.2-3B grid: both families reverse bias direction
   with scale in opposite senses, Llama-3B is a new always-A machine that
   falls below chance on adversarial Chat Hard while hitting 0.889 on
   Chat, and post-debiasing calibration is a family property, findings
   21–23. 2026-08-21: the cross-judge headline table — the spine the
   arrival-ordered README never had, with per-order accuracy and the
   paired delta against the length floor, every number reproducing the
   per-grid sections — plus a findings index; and finding 26, which came
   out of building its coverage guard: a partial grid in this harness is
   an alphabetical prefix of the subsets, not a subsample, so the day-6
   7B peek was composition (matched on the same 45 items, Qwen-3B's
   median |s| is 8.68, not the 3.64 it was compared against, and the
   length floor on that prefix is 0.978 against 0.425 overall). Interim
   reads now go through `master_table --restrict-to`, which matches items
   across judges and prints the skew. The 7B grid is slow on this host
   (~0.04 judg/s, ~7 h) and spans sessions; store at 134/1200.
   2026-08-24: finding 27 — the root cause behind finding 26 got fixed
   rather than guarded. `src/schedule.py` serves the subset with the
   largest proportional deficit, so a partial grid is a stratified sample
   at every prefix (under 0.05 total-variation distance from item 55,
   against item 569 for the legacy `item_id` order) and no subset ever
   drifts more than one item from proportional. Day 7's reason for not
   doing this — that reordering would break resume-compatibility — was
   false and had never been checked: resume is keyed on the *set* of
   finished judgments and analyses group by `item_id`, so execution order
   is unobservable. Matching item sets recovers comparability between
   judges; only the schedule recovers representativeness of the benchmark.
   The 7B store's inherited 67 sorted-order items cannot be un-judged, so
   its composition converges by dilution only and the 7B row stays a
   non-claim until the grid completes.
   2026-08-26: the 7B grid closed at 1200/1200 (four sessions, compliance
   1.000 throughout) — findings 28–31: the Qwen arc becomes
   0.568 → 0.502 → 0.742 → 0.837 and 7B is the audit's first
   signal-dominant judge (|b| > |s| on only 26.8% of items); the flip-rate
   inversion completes (the best judge posts the highest flip rate, 0.732);
   7B beats the fitted length floor in every category, adversarial Chat
   Hard included, by the audit's largest margin; the one-call debiasing
   ceiling falls again (68% → 47% → 25% of the symmetrization gain within
   Qwen) and Qwen stays overconfident at its top size.
   Later the same day the Llama-3.1-8B grid ran end-to-end in one
   190-minute session — the first grid collected entirely under the
   scheduler — closing the scaling axis at seven judges: findings 32–35.
   The family arcs never cross (Llama-8B ties Qwen-3B at 2.7x the
   parameters and trails Qwen-7B by −0.113); Llama's adversarial Chat-Hard
   hole persists to 8B while its Chat hits 0.958, the audit's best
   category score; the audit's largest length-controlled coefficient rides
   its largest length lean and pays for it on math-prm (0.389, below
   chance); the one-call debiasing ceiling holds at ~25% at 8B with finer
   corrections actively hurting; and the Safety compliance migration
   replicates. **Phase 2 is complete.**
   Next: phase-3 remainder — the rubric-sensitivity axis (minimal vs
   detailed, paired in log-odds) — and the results-narrative restructure
   around the completed scaling arc.)*
3. **Analysis axes** — position bias as additive log-odds shift (test the
   structural model); symmetrization debiasing gains; calibration; value over
   length baseline; rubric-prompt sensitivity; category/subset heterogeneity.
   *(All but the rubric axis were completed alongside phase 2 — findings
   12–15, 19–20, 24–25. Rubric axis started 2026-08-27: paired
   minimal-vs-detailed machinery (`src/rubric_pair.py`) plus detailed-rubric
   grids for the two smallest judges — findings 36–38: 30–43% of symmetrized
   verdicts flip with the rubric text alone, concentrated where |s| is
   small; at 0.5B the detailed rubric contracts the whole log-odds
   distribution without touching the always-A pathology; at 1B it reverses
   both bias direction and length orientation, and the significant accuracy
   gain decomposes as a re-aimed length lean, with compliance collapsing
   0.512 → 0.275. The results-narrative restructure landed the same day.
   2026-08-28: the 1.5B detailed grid plus the perturbation model
   (`fragility_fit`: s_det = λ·s_min + ε, flip probability Φ(−λ|s|/σ)) —
   findings 39–41: the flip-rate arc 0.303/0.432/0.190 is ordered by
   median |s|, not size, and the model reproduces each judge's quartile
   profile; at 1.5B the rubric contracts both components and halves the
   order asymmetry without moving the symmetrized verdict, and the 1B's
   bias-direction reversal does not replicate where |b| is large; the
   valley is rubric-invariant. Later the same day the 3B detailed grid
   closed the fourth point — findings 42–43: the flip-rate arc
   0.303/0.432/0.190/0.102 stays ordered by |s| with the coherent-movement
   deviation deepening as r(s) rises, and prompt-side debiasing posts its
   largest reduction (Δ|b| −1.80 on the audit's largest bias) while buying
   only raw accuracy and order balance — never symmetrized quality — and
   un-saturating the positional flip rate.
   2026-08-29: the 7B detailed grid closed the Qwen line in one 514-minute
   session — findings 44–45: the five-point arc 0.303/0.432/0.190/0.102/0.068
   stays ordered by |s| and lands on the day-11 registered prediction, λ
   plateaus (0.767 → 0.777) while the coherent-movement deviation proves
   non-monotone (peaks at 3B, narrows at 7B where the Gaussian model nearly
   suffices); and the prompt-side lever stops paying at the top — it pushes
   the 7B's balanced bias through zero into a B-lean (median b
   +0.23 → −1.25) while raw, symmetrized, positional-flip, length and
   compliance metrics all sit still, with the only significant category
   movements (Safety +0.047, Chat Hard −0.065) canceling.
   2026-08-30: the Llama-3.2-3B detailed grid, pre-registered in the
   morning commit before results existed — findings 46–47: the raw
   |s|-ordering law bends at its first cross-family test (flip rate 0.172
   at |s| 0.445, below the 1.5B's 0.190 at 0.503) and the surviving law is
   the model's own ratio λ·|s|/σ, strictly monotone across all six judges;
   the 1B's compliance collapse and large σ both fail to replicate at 3B
   (0.863 → 0.818 vs 0.512 → 0.275; σ 0.376 under the |s|-matched Qwen's
   0.418), the always-A bias shrinks without re-signing exactly on the
   findings-40/45 boundary prediction, and the only borderline purchase is
   a narrowed (still below-chance) adversarial Chat Hard hole.
   2026-08-31: the Llama-3.1-8B detailed grid closed the axis at seven
   judges (1200/1200 in 573.7 min, pre-registered in the morning commit) —
   findings 48–49: the λ|s|/σ ordering survives its out-of-sample test
   (fitted ratio 1.370 slots the observed 0.082 flip rate exactly between
   Qwen-3B and Qwen-7B; seven of seven strictly monotone) while the raw-|s|
   law breaks a second time and both the λ-plateau and |s|-scaled-σ
   regularities die cross-family (λ 0.583, σ 0.497 — Llama's σ is
   family-stable at 0.38–0.53 while Qwen's grows 11-fold); and at 8B the
   detailed rubric turns harmful for the first time — signal contracts 42%
   against bias's 12%, chosen-first raw accuracy pays the full −0.045
   [−0.065, −0.025] while the symmetrized verdict is exactly null, the
   re-signing boundary tightens into (0.34, 0.59), and compliance follows
   the Llama-3B Safety-concentrated pattern, not the 1B collapse.
   **Phase 3 is complete.**)*
4. **Writeup** — README as a research report with real tables, figures, and
   limitations; reproduction audit in the `rag-chunking-bench` style.
   *(Started 2026-09-01: full coherence pass — every inline number verified
   against the committed summaries or recomputed from the raw stores, the
   uncommitted sign(s)-vs-length store-join promoted into the per-store
   summary (`sign_length_agreement`, with `bias_b.median_abs` alongside),
   the stale master-table caption regenerated, and the abstract extended to
   the completed-audit story. 2026-09-03: the reproduction audit landed —
   `experiments/reproduce.py` extracts `git archive HEAD` into a scratch
   tree, wipes and regenerates all 61 regenerable artifacts from the
   committed raw stores, and byte-compares with two-way manifest coverage
   (the two day-9 interim master-table files verified as pinned history);
   its first run reproduced 62/63 and caught the stale prefix-skew panel,
   rendered day 12 while the 7B grid was in flight and never re-rendered
   after the grid closed — re-rendered from the same pinned inputs, with
   the README caption corrected to match. 140 tests. 2026-09-04: release —
   the post-commit audit gate passed on the committed tree (`OK: all 63
   artifacts reproduce byte-for-byte from HEAD`, replaying all 25 manifest
   steps in a fresh container with refetched data), final end-to-end read
   of the report, status header rewritten to the completed-audit story,
   CHANGELOG 0.3.0 entry, landing-page sync. **Phase 4 complete — flagship
   closed.**)*

### `rag-chunking-bench` — 2026-07-03 to 2026-07-16, COMPLETE

**Question.** How much does the chunking strategy actually matter for RAG retrieval
quality — once you control for the retrieved-token budget?

**Why this, why now.** Chunking is the highest-leverage, least-principled knob in
every production RAG stack. The recent literature is active but methodologically
loose: most comparisons vary chunk size while holding top-*k* fixed, which confounds
chunking quality with the sheer number of tokens handed to the generator (500-token
chunks at k=5 retrieve 5x the text of 100-token chunks at k=5). Chroma's technical
report (Smith & Troynikov, 2024) introduced token-level metrics that partially
address this; recent arXiv work (Merola & Singh, 2025, arXiv:2504.19754; Duarte et
al., EMNLP 2024 Findings, arXiv:2406.17526) compares advanced chunkers but still
mostly at fixed *k*. A budget-matched, statistically careful comparison across
chunkers, chunk sizes, and retrievers — with paired bootstrap confidence intervals
on span-level metrics — is a genuine gap at a scale one person can execute
rigorously. It also feeds directly back into my `financial-rag-chatbot`.

**Outcome.** 26 findings across baselines, five robustness axes (retriever
family, sampling seed, budget rule, tokenizer unit, corpus jackknife), the
semantic-chunker verdict under a matched-realized-size protocol, and a
closing per-question error analysis. Headlines: fixed-k and budget-matched
evaluation rank chunk sizes in opposite orders; under budget matching the
winning chunk size is set by gold-evidence length; the percentile semantic
chunker shows no boundary-quality gain at matched realized size and a real
long-gold penalty; and matched *mean* size is not a controlled comparison —
realized-size dispersion × the stop rule manufactures ±0.5 recall deltas.
365 tests; every committed table and figure (22 + 19) regenerates
byte-identically from the committed raw results in a clean environment
(`experiments/reproduce.py`, audited 2026-07-16). Full report:
`rag-chunking-bench/README.md`; day-by-day log:
`rag-chunking-bench/research/NOTES.md`.

**Phase history.** Baselines (first grid 2026-07-04:
fixed-k vs budget-matched ranking reversal; overlap ablation + truncate-rule
robustness check 2026-07-05: overlap = boundary repair, size ordering
survives the rule change — findings 6–7; cross-retriever grid 2026-07-06:
all chunking effects transfer to TF-IDF/LSA, chunking effect > retriever
effect, retriever gap grows with chunk size — findings 8–9; multi-seed check
+ dense MiniLM grid 2026-07-07: headline claims replicate under three
independent question samples, chunking effects transfer to dense retrieval,
and past the encoder window dense retrieval degrades to prefix retrieval —
findings 10–12; Chroma long-reference grid 2026-07-08, all four retrievers:
the small-chunk advantage INVERTS at generous budgets on sentence-scale
golds, the inversion is gold-length-driven and requires a full-chunk-reading
retriever, and precision/IoU are finally informative — findings 13–15;
chroma overlap + truncate ablations and corpus jackknife 2026-07-09: overlap
gains persist across budgets on long golds and the cross-family
boundary-repair control breaks at small sizes, while the crossover survives
the budget rule and every drop-one corpus and the tight-budget small-chunk
edge turns out to be mostly a stop-rule artifact — findings 16–18; cl100k
BPE tokenizer unit 2026-07-11: every headline claim is unit-invariant, and
wiring the unit in exposed and fixed two containment-vs-overlap chunker
bugs — finding 19; semantic chunker 2026-07-12: the percentile
embedding-breakpoint chunker's matched-nominal-size wins are realized-size
drift — null where realized sizes coincide, sign-flipped on long golds at
generous budgets, no systematic ranking gains — findings 20–21;
matched-realized-size protocol 2026-07-13: at matched realized size the
semantic chunker gains nothing anywhere and its long-gold penalty
survives, while matched *means* prove insufficient — realized-size
dispersion × the stop rule manufactures ±0.5 deltas, truncate at
B ≫ chunk size is the honest regime — findings 22–23; per-question error
analysis 2026-07-14, no new runs: the per-corpus heterogeneity is
gold-length composition (leave-one-corpus-out composition test, no
significant residual anywhere), the hard-loss tail splits into
partial-coverage losses on long multi-ref golds plus a small set of
complete ranking misses on short golds, and every overlap gain decomposes
exactly into new-region placement + extension − a redundancy tax, with
stitching real only at tight budgets — findings 24–26).

1. **Harness** — offset-preserving chunkers, tokenization, span-level metrics,
   budget-matched retrieval protocol, dataset loaders. *(done — SQuAD +
   Chroma loaders, both with verbatim-verified gold spans)*
2. **Baselines** — BM25 / TF-IDF / LSA / dense retrievers over all chunker x size x
   overlap configs on SQuAD-derived long documents + Chroma eval corpora.
   *(done: all four retriever grids on both datasets; overlap, budget-rule,
   and multi-seed checks on SQuAD)*
3. **Ablations & analysis** — overlap ablation *(done: SQuAD + chroma)*,
   budget-rule check *(done: SQuAD + chroma)*, multi-seed sampling
   *(done, BM25)*, gold-length moderation *(done — finding 14)*, corpus
   jackknife *(done — finding 18)*, BPE tokenizer robustness *(done —
   finding 19)*, semantic vs. structural chunking *(done — findings
   20–21)*, matched-realized-size protocol *(done — findings 22–23)*,
   per-corpus error analysis *(done — findings 24–26; phase complete)*.
4. **Writeup** — README as a full research report with real tables and
   limitations. *(done: coherence pass 2026-07-15; clean-environment
   reproduction audit, `experiments/reproduce.py` manifest + audit tooling,
   and release polish 2026-07-16 — flagship closed)*

**Environment constraints (recorded so results are honest).** CPU-only (4 cores,
16 GB RAM). Network access widened on 2026-07-03: HuggingFace and the tiktoken
vocab host are now reachable (both were blocked on day 1), so phase 2 adds a
small CPU-sized sentence-transformer dense retriever (e.g. all-MiniLM-L6-v2)
alongside BM25 / TF-IDF / LSA, and the BPE tokenizer robustness check becomes
real rather than hypothetical. Large dense retrievers and cross-encoder
rerankers remain out of scope on this hardware and are listed as limitations.

## Backlog (next flagships, roughly prioritized)

- **Hallucination measurement in RAG answers** — span-attribution based
  faithfulness scoring; natural sequel to the chunking bench.
- **Agent tool-call reliability harness** — inject tool failures/latency and
  measure recovery behavior of agent loops; ties into `data-analysis-agent`.
- **Retriever robustness to query noise** — typos, paraphrase, and entity-swap
  perturbations vs. retrieval degradation curves.
- **Time-series foundation models vs. classical baselines** — evaluate on the
  `Bitcoin-Price-Forecasting` data with proper backtesting protocol.

## Weekly rhythm

Most days advance the flagship. One or two days a week ship a focused improvement
to `financial-rag-chatbot`, `data-analysis-agent`, or `Bitcoin-Price-Forecasting`
(evals, tests, robustness) — check `git log` first, never repeat recent work.

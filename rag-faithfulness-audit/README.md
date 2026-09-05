# rag-faithfulness-audit

**A statistically careful audit of zero-training hallucination detectors for
RAG answers, measured span-level against human annotation — with trivial
floors, a small-judge scaling curve read white-box, and a study of how much
of any detector ranking survives the choice of scoring protocol.**

*Status: **DESIGN** — opened 2026-09-05 (day 1). This README is the project
proposal: question, related work, data reconnaissance, and planned method.
No experimental results exist yet; every number below is either a verified
citation or a statistic computed directly from the pinned corpus during the
day-1 reconnaissance (`research/NOTES.md`). Tables and figures will appear
here only once the experiments that produce them have run in this repo.*

## Question

RAG is deployed on the promise that grounding reduces hallucination, and an
ecosystem of detectors now scores RAG answers for faithfulness: lexical
heuristics, NLI entailment models, LLM judges, and encoders trained
specifically for the task. Three questions this lab can answer rigorously on
its hardware:

1. **How far do zero-training detectors get?** Sentence-level lexical
   support, small NLI cross-encoders, and small open-weight LLM judges
   (0.5B–8B) read white-box as verdict log-odds — all evaluated against
   human-annotated hallucination spans, with trained-on-task encoders as the
   reference ceiling and trivial baselines as floors. Nobody reports the
   floors; this audit's predecessor found a fitted one-parameter length
   baseline beating every sub-3B judge, and the analogous floors here
   (flag-nothing, flag-everything, flag-unsupported-numbers) are cheap and
   currently absent from the literature's tables.
2. **Does the judge-scaling picture transfer from preference to grounding?**
   The predecessor audit (`../slm-judge-audit`) mapped how pairwise-judge
   quality scales 0.5B→8B — non-monotone, family-dominated, length-mediated.
   Grounding verification ("is this sentence supported by this document?")
   is a different task shape: reference-based, asymmetric, no position bias
   by construction. Whether the same models show the same valleys,
   family gaps, and calibration signatures is an open, checkable question.
3. **How protocol-dependent are published rankings?** Papers report
   "hallucination detection on RAGTruth" as example-level F1, span-level F1,
   token-level F1, or char-overlap variants, at different span-matching
   thresholds, sometimes after filtering response quality — and the numbers
   are routinely compared across papers. With several detectors under one
   harness, the ranking's sensitivity to the scoring protocol becomes
   measurable: the same detectors, every protocol, paired bootstrap on the
   differences.

## Why now, and the gap

Faithfulness measurement is one of the most active evaluation topics of
2025–2026. The closest published work:

- **RAGTruth** (Niu et al., ACL 2024, [arXiv:2401.00396](https://arxiv.org/abs/2401.00396))
  contributed the corpus this audit stands on: ~18k real RAG responses from
  six generators with human-annotated hallucination spans at character
  level, typed (evident/subtle × conflict/baseless). It remains the only
  public RAG benchmark with human span annotation fine enough for
  sub-sentence scoring, and it ships prompt-based and fine-tuned detection
  baselines measured example- and span-level.
- **LettuceDetect** (Kovács & Recski, [arXiv:2502.17125](https://arxiv.org/abs/2502.17125))
  is the current encoder reference: ModernBERT token classification trained
  on RAGTruth's train split, example-level F1 79.2% on the test split. It is
  small enough to run on this project's CPUs and will be the trained
  ceiling, re-evaluated inside this harness rather than quoted.
- **FaithJudge** (Tamber et al., [arXiv:2505.04847](https://arxiv.org/abs/2505.04847))
  benchmarks frontier LLMs as faithfulness judges anchored by human
  annotations and runs Vectara's hallucination leaderboard on it — judge
  quality measured, but API-scale and example-level.
- **TRIVIA+** (Chen et al., 2026, [arXiv:2605.11330](https://arxiv.org/abs/2605.11330))
  argues current hallucination-detection *evaluation* itself lacks basic
  desiderata (context length realism, label noise handling) — evidence the
  methodology axis is live.
- **Retromorphic testing** (Yu et al., 2026, [arXiv:2603.27752](https://arxiv.org/abs/2603.27752))
  decomposes answers into claims verified hierarchically against context —
  the compositional-verifier end of the design space, evaluated on a 408-sample
  relabeled RAGTruth++.
- **RAGAS** (Es et al., [arXiv:2309.15217](https://arxiv.org/abs/2309.15217))
  is what practitioners actually run: LLM-decomposed claims scored by an LLM
  for support. Its faithfulness score is an API-judge pipeline — exactly the
  design whose small-model reliability the predecessor audit showed to be
  fragile in preference judging, and untested white-box for grounding.

The unclaimed corner, again, is **white-box + small-scale + statistically
careful**: every published detector comparison treats detectors as
black-box classifiers and reports point-estimate F1 with no confidence
intervals, no trivial floors, no calibration analysis, and — because each
paper picks one protocol — no evidence about how rankings move across
protocols. Local small models expose verdict distributions, so detection
confidence is a log-odds this lab can calibrate, threshold on the train
split honestly, and bootstrap pairwise. The machinery exists in this lab:
span algebra and paired cluster bootstrap from `../rag-chunking-bench`,
single-token verdict readout and calibration tooling from
`../slm-judge-audit`.

## Data

`RAGTruth` pinned at commit
[`c103204`](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe)
(MIT license), fetched and SHA256-verified by
`experiments/fetch_data.py`; the corpus itself is not committed. Day-1
reconnaissance of the official test split (2,700 responses; full numbers
and code in `research/NOTES.md`):

- 900 responses per task type (QA / Summarization / Data-to-text); six
  generators × 450 each; 943/2,700 responses carry at least one
  hallucination span (1,533 spans).
- Hallucination base rates are wildly heterogeneous — by task: Data2txt
  64.3%, Summary 22.7%, QA 17.8%; by generator: GPT-4 9.3% up to
  Mistral-7B-Instruct 55.8%. Any headline number is a composition claim;
  everything will be reported stratified.
- Median annotated span is **35 characters** (p90 = 114) against a median
  response sentence count of 6 — hallucination spans are typically
  *sub-sentence*, so sentence-level detectors have an intrinsic granularity
  ceiling that char-level protocols will price and example-level protocols
  hide. This single reconnaissance fact motivates the protocol axis.
- Median context is ~2.2k chars (max ~10k) — within small-NLI windows only
  via windowing, and within small-LLM contexts directly.

## Planned method

- **Unit and readout.** Responses segmented into sentences with character
  offsets (offset-preserving, as in `rag-chunking-bench`). Every detector
  emits a *support score per sentence*; span-level verdicts derive from
  scored sentences intersected with annotated spans under an explicit
  matching rule — the rule itself being a studied axis, not a footnote.
- **Floors.** Flag-nothing, flag-everything, random at matched flag rate,
  and lexical floors: sentences containing numbers/named strings absent
  from the context; max n-gram support of a sentence in any context window.
- **Zero-training detectors.** (a) Small NLI cross-encoders (~100–200M,
  CPU-sized) scoring sentence-vs-best-context-window entailment;
  (b) small open-weight LLM judges (Qwen2.5 0.5B→7B, Llama-3.2 1B/3B,
  Llama-3.1 8B — the predecessor's grid) prompted per sentence with the
  context, verdict read as single-token log-odds. Thresholds fitted on the
  train split only.
- **Ceiling.** LettuceDetect (trained on RAGTruth train) re-run locally
  under the identical protocol.
- **Statistics.** Paired cluster bootstrap resampling *responses* (spans
  and sentences are not independent), BCa intervals where the lab's prior
  tooling applies; stratified reporting by task, generator, and
  hallucination type; calibration (ECE, reliability diagrams) on the
  log-odds detectors.
- **Honesty rails.** Pinned revisions and SHA256 at load; fixed seeds;
  every table/figure regenerable from committed raw stores
  (`experiments/reproduce.py` in the house pattern); no number in this
  README that did not come from a run in this repo.

## Planned phases

1. **Harness** — pinned data layer; offset-preserving sentence
   segmentation; span↔sentence alignment with explicit matching rules;
   scoring protocols (example/span/char-level); result store; bootstrap.
2. **Floors + lexical baselines** — the full floor table with CIs;
   first protocol-sensitivity read on the floors themselves.
3. **NLI + small-judge grids** — the zero-training detector grid, scaling
   curve, calibration; train-split thresholding.
4. **Protocol-sensitivity axis** — all detectors × all protocols; ranking
   stability; the sub-sentence granularity ceiling quantified.
5. **Ceiling + error analysis** — LettuceDetect under the same protocol;
   stratified error anatomy (task, generator, hallucination type,
   evident vs subtle); where zero-training detectors actually fail.
6. **Writeup + reproduction audit** — README as research report; clean-tree
   byte-identical regeneration audit in the house style.

## Feasibility on this hardware

CPU-only, 4 cores, 16 GB RAM. The test split is 18,930 sentences; NLI
cross-encoders at this scale are hours, not days. The judge grid reuses the
predecessor's prefill-only single-token readout (measured 0.04–1.4
judgments/s across 0.5B–8B on this host); full-split × seven judges is
infeasible, so judges run on a pinned stratified response sample with the
train-split threshold fit on a disjoint sample — sizes to be fixed in phase
1 with a pilot, recorded before any grid runs. Frontier-API detectors
(RAGAS-style pipelines with hosted judges) are out of scope beyond what
free tiers allow and are a recorded limitation, not an aspiration.

## References

- Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., Zhang, T.
  *RAGTruth: A Hallucination Corpus for Developing Trustworthy
  Retrieval-Augmented Language Models.* ACL 2024.
  [arXiv:2401.00396](https://arxiv.org/abs/2401.00396)
- Kovács, Á., Recski, G. *LettuceDetect: A Hallucination Detection Framework
  for RAG Applications.* 2025.
  [arXiv:2502.17125](https://arxiv.org/abs/2502.17125)
- Tamber, M. S., Bao, F. S., Xu, C., Luo, G., Kazi, S., Bae, M., Li, M.,
  Mendelevitch, O., Qu, R., Lin, J. *Benchmarking LLM Faithfulness in RAG
  with Evolving Leaderboards.* EMNLP 2025 (industry).
  [arXiv:2505.04847](https://arxiv.org/abs/2505.04847)
- Chen, W., Padmanabhan, V., Giyahchi, T., Wong, E., Akoglu, L. *Rethinking
  Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based
  Benchmark, New Insights.* 2026.
  [arXiv:2605.11330](https://arxiv.org/abs/2605.11330)
- Yu, B., Zhang, Y., Lin, L., Briand, L., Muñoz, E. *Retromorphic Testing
  with Hierarchical Verification for Hallucination Detection in RAG.* 2026.
  [arXiv:2603.27752](https://arxiv.org/abs/2603.27752)
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. *RAGAS: Automated
  Evaluation of Retrieval Augmented Generation.* EACL 2024 (demo).
  [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- Lambert, N., et al. *RewardBench.* [arXiv:2403.13787](https://arxiv.org/abs/2403.13787)
  — the predecessor audit's data; cited here for the judge-grid lineage.

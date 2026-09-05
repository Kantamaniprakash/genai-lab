# rag-faithfulness-audit — research log

One entry per working day, newest last. Every claim in the README traces to an
entry here; every entry ends with exact next steps.

## 2026-09-05 — Day 1: flagship selection, data reconnaissance, project skeleton

Selection day: the landscape scan picked the new flagship — a statistically
careful, span-level audit of zero-training hallucination detectors for RAG
answers on RAGTruth — over the backlog's other candidates (agent-reliability
dropped: its gap closed in the 2026 benchmark wave). The corpus is pinned
and verified, the reconnaissance numbers below fix the design's shape (spans
are sub-sentence; base rates vary 6× by task and generator), and the
skeleton — proposal README with verified citations, pinned fetch script,
this log — is committed.

### Selection scan

`slm-judge-audit` closed yesterday; this session ran the landscape scan the
ROADMAP prescribes before a new flagship opens. Four candidates, each tested
against (a) what the field demonstrably cares about right now and (b) what
this CPU-only host can execute honestly.

**Hallucination measurement in RAG answers — selected.** The 2026 scan shows
this is now one of the densest corners of eval research: FaithJudge and the
Vectara leaderboard (arXiv:2505.04847), TRIVIA+ arguing detection evaluation
itself is broken (arXiv:2605.11330, May 2026), retromorphic claim
verification (arXiv:2603.27752, March 2026), token-level detection via
internal representations (CORTEX, arXiv:2606.31033), SAE-based faithfulness
probes (arXiv:2512.08892, ICLR 2026) — plus the established anchor corpus
RAGTruth (ACL 2024) and its trained-detector line (Luna, LettuceDetect).
Three facts make it *this lab's* project rather than a crowded me-too:
(1) every comparison in that literature is black-box point-estimate F1 — no
floors, no CIs, no calibration; the lab's whole toolkit is exactly the
missing part; (2) RAGTruth gives human span-level ground truth, so unlike
the judge audit no LLM is needed to define truth; (3) the published numbers
are measured under visibly incompatible protocols (example vs span vs token
level), which makes protocol sensitivity itself a finding-generating axis —
the same move the chunking bench made with fixed-k vs budget-matched.
Verified papers exist as cited (arXiv abstract pages fetched today; full
author lists in the README references).

**Agent tool-call reliability — rejected, gap closed.** The scan found the
2026 wave arrived: ReliabilityBench (arXiv:2601.06112) does
chaos-engineering-style fault injection for agent loops; ToolFailBench
(arXiv:2607.04686) diagnoses tool-use failures across five domains; ToolGym
injects failures in a 5,571-tool environment. What made this candidate
attractive in July (nobody measures recovery under injected failure) is now
three benchmarks deep, and executing agent loops at scale here would need
API models anyway. Dropped from the backlog with this note as the record.

**TSFM vs classical on Bitcoin data — kept in backlog, demoted.** Still
feasible (Chronos-family small checkpoints run on CPU) and the
benchmark-methodology angle is live (arXiv:2510.13654 documents selection
bias making 46% of models look SOTA on cherry-picked test sets). But the
community pull is a fraction of the faithfulness topic's, and a single-asset
financial series is a weak substrate for methodology claims. Revisit when
the flagship closes.

**Retriever robustness to query noise — kept, unchanged.** No new evidence
either way in the scan; still the backlog's tail.

### Data reconnaissance (real numbers, code run today)

Cloned RAGTruth at `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (MIT
license). Two files: `response.jsonl` (17,790 responses;
SHA256 `e4c2e4ac…5e18073`) and `source_info.jsonl` (2,965 source prompts;
SHA256 `0dffc26e…12d578b`). Split: 15,090 train / 2,700 test. Test is
exactly 900 per task type (QA / Summary / Data2txt), six generators × 450.
Findings from the reconnaissance pass (stdlib, ~20 lines, will become the
data-layer tests):

- **943/2,700 test responses hallucinate** (≥1 span); 1,533 spans total.
  Labels carry char offsets, text, `label_type`
  (Evident/Subtle × Conflict/Baseless plus a few legacy variants written
  inconsistently in `meta` — parse `label_type`, not `meta`), and
  `implicit_true`/`due_to_null` flags whose semantics need a doc pass
  (day-2 item).
- **Base-rate heterogeneity is extreme**: Data2txt 64.3% of responses
  hallucinate vs QA 17.8%; Mistral-7B 55.8% vs GPT-4 9.3%. Pooled headline
  numbers would be composition artifacts — everything gets stratified
  reporting, and any sampled judge grid must stratify on task × generator ×
  has-hallucination.
- **Spans are sub-sentence**: median 35 chars, p90 114, against median 6
  sentences/response (18,930 test sentences, naive segmentation). A
  sentence-level detector cannot express a 35-char flag; the protocol axis
  (example vs span vs char level, matching rules) is therefore structural,
  not cosmetic. This is the day-1 fact the whole protocol phase hangs on.
- **Scale check**: 18,930 sentences × small NLI ≈ hours on 4 cores — full
  test split feasible for NLI and lexical floors. Judge grid at
  slm-judge-audit rates (0.04–1.4 judg/s by size) needs sampling: e.g.
  300 responses ≈ 2,100 sentence judgments ≈ 40 min at 1B, ~15 h at 7B per
  rubric — same shape as the last audit's grids. Pilot before fixing sizes.

### Skeleton laid

- `README.md` — full proposal: question (three-part), related work with
  verified citations, data recon, planned method and phases, feasibility.
  Status header says DESIGN and promises no numbers that weren't produced
  here; the only numbers in it today are citations or today's recon stats.
- `experiments/fetch_data.py` — pinned fetch: downloads the two jsonl files
  from the pinned commit via raw.githubusercontent, verifies SHA256,
  idempotent re-run (verify-only when present). Run today end-to-end; data
  lands in `data/ragtruth/` (gitignored — MIT would allow committing, but
  house style is fetch+verify, and 36 MB doesn't belong in git).
- ROADMAP updated: this project is the current flagship; agent-reliability
  candidate dropped with rationale; landing page synced via
  `scripts/sync_latest.py`.

### Next steps (Day 2)

1. **Data layer** (`src/data.py`): typed loaders over the two jsonl files,
   task/generator/label-type accessors, split handling, load-time SHA256
   verification against the pins in `fetch_data.py` (shared constants
   module). Resolve `implicit_true` and `due_to_null` semantics from the
   RAGTruth repo docs/paper and record them here; decide whether
   `quality != 'good'` responses (144 incorrect_refusal, 29 truncated
   corpus-wide) are excluded — check what the paper and LettuceDetect do,
   record the choice as a protocol parameter, not a silent filter.
2. **Sentence segmentation with offsets** (`src/segment.py`): the
   chunking-bench offset-preserving pattern; property tests that segments
   reassemble the response verbatim and offsets index correctly.
3. **Span↔sentence alignment** (`src/spans.py`): char-overlap and IoU
   between annotated spans and sentence windows; the matching-rule
   parameterization (any-overlap / IoU≥τ / char-level) as first-class
   protocol objects, since phase 4 audits exactly this.
4. Tests throughout in the house style (the two prior flagships opened with
   the harness fully tested before any experiment ran).

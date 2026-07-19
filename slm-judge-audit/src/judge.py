"""llama.cpp judge runner: chat templating, single-token logit readout, result store.

The runner turns a :class:`~src.prompts.JudgePrompt` into one prefill-only
forward pass and reads the full-vocabulary logits at the first assistant
position. Everything the analysis needs is recorded per judgment: the verdict
log-odds ``z = logit("A") - logit("B")``, the renormalization diagnostics
(probability mass on {A, B}, unconstrained argmax token, compliance flag), the
gold expected verdict for the presentation, and timing.

Design constraints, learned the hard way in the 2026-07-17 pilot:

- ``Llama.scores`` stays zeroed when ``logits_all=False``; the last-position
  logits must be read through the low-level ``llama_cpp.llama_get_logits``
  accessor. This module owns that detail so nothing else touches ctypes.
- Chat templates are rendered *explicitly* from a registry rather than through
  the GGUF's embedded template, so the exact prompt string is a testable,
  versioned artifact of this repo. Templates end with the assistant header:
  the next token the model would emit is the verdict.
- The verdict letters must each encode as a single token (verified per model
  at load; hard failure otherwise, because a multi-token letter would make
  the one-position readout meaningless).

Results are stored append-only as JSONL, one self-contained record per line,
keyed by (model, rubric, order, item_id) for idempotent resume. A sidecar
``.meta.json`` records provenance: model file SHA256, pinned source revision,
llama-cpp-python version, context size, thread count.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .prompts import VERDICT_TOKENS, JudgePrompt

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "raw"


# ---------------------------------------------------------------------------
# Chat templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChatTemplate:
    """A chat template rendered to a raw prompt string ending at the position
    where the model's next token is the verdict letter."""

    name: str
    # Format string with {system} and {user} slots. Special tokens are written
    # literally; the tokenizer is called with special=True so they map to ids.
    layout: str

    def render(self, system: str, user: str) -> str:
        return self.layout.format(system=system, user=user)


CHAT_TEMPLATES: dict[str, ChatTemplate] = {
    # Qwen2.5 family (verified against the official ChatML spec, 2026-07-17).
    "chatml": ChatTemplate(
        name="chatml",
        layout=(
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    ),
    # Llama 3 family. BOS is written literally and mapped by special=True.
    "llama3": ChatTemplate(
        name="llama3",
        layout=(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeModel:
    """One auditable judge: a pinned GGUF artifact plus its chat template."""

    key: str            # short id used in result records and file names
    repo: str           # HF repo the GGUF comes from
    revision: str       # pinned repo revision (full commit hash)
    filename: str       # GGUF file name inside models/
    sha256: str         # pinned artifact digest, verified before every run
    template: str       # key into CHAT_TEMPLATES
    params_b: float     # nominal parameter count in billions (for plots)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


MODELS: dict[str, JudgeModel] = {
    "qwen2.5-0.5b": JudgeModel(
        key="qwen2.5-0.5b",
        repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        revision="9217f5db79a29953eb74d5343926648285ec7e67",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        sha256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        template="chatml",
        params_b=0.5,
    ),
    "qwen2.5-1.5b": JudgeModel(
        key="qwen2.5-1.5b",
        repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        revision="91cad51170dc346986eccefdc2dd33a9da36ead9",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        sha256="6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        template="chatml",
        params_b=1.5,
    ),
    "llama-3.2-1b": JudgeModel(
        key="llama-3.2-1b",
        repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        revision="067b946cf014b7c697f3654f621d577a3e3afd1c",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        sha256="6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83",
        template="llama3",
        params_b=1.0,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Judgment records and the result store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgmentRecord:
    """Everything one forward pass tells us, self-contained for analysis."""

    model: str
    rubric: str
    order: str
    item_id: str
    expected_verdict: str   # letter a perfect judge produces for this order
    z: float                # logit("A") - logit("B") at the verdict position
    logp_a: float           # log-softmax over the full vocabulary
    logp_b: float
    mass_ab: float          # p_a + p_b: how much of the distribution is a verdict
    argmax_token: str       # unconstrained argmax at the verdict position
    compliant: bool         # argmax, stripped, is exactly "A" or "B"
    n_prompt_tokens: int
    prefill_seconds: float

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.model, self.rubric, self.order, self.item_id)

    @property
    def greedy_verdict(self) -> str:
        """The verdict under the constrained readout (argmax over {A, B})."""
        return "A" if self.z >= 0 else "B"

    @property
    def raw_correct(self) -> bool:
        return self.greedy_verdict == self.expected_verdict


def logits_to_record(
    logits,
    *,
    token_ids: tuple[int, int],
    argmax_token: str,
    prompt: JudgePrompt,
    model_key: str,
    n_prompt_tokens: int,
    prefill_seconds: float,
) -> JudgmentRecord:
    """Pure readout: full-vocab logits -> one JudgmentRecord.

    Kept free of llama.cpp so the readout arithmetic is unit-testable.
    ``token_ids`` are the single-token ids of ("A", "B") for this model.
    """
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    id_a, id_b = token_ids
    m = float(logits.max())
    lse = m + math.log(float(np.exp(logits - m).sum()))
    logp_a = float(logits[id_a]) - lse
    logp_b = float(logits[id_b]) - lse
    return JudgmentRecord(
        model=model_key,
        rubric=prompt.rubric_name,
        order=prompt.order,
        item_id=prompt.item_id,
        expected_verdict=prompt.expected_verdict,
        z=float(logits[id_a] - logits[id_b]),
        logp_a=logp_a,
        logp_b=logp_b,
        mass_ab=float(math.exp(logp_a) + math.exp(logp_b)),
        argmax_token=argmax_token,
        compliant=argmax_token.strip() in VERDICT_TOKENS,
        n_prompt_tokens=n_prompt_tokens,
        prefill_seconds=prefill_seconds,
    )


class ResultStore:
    """Append-only JSONL store with idempotent-resume semantics.

    One file per (model, rubric); each line is a full JudgmentRecord dict.
    ``existing_keys`` is loaded once so a rerun skips finished judgments.
    """

    def __init__(self, model_key: str, rubric: str, results_dir: Path = RESULTS_DIR):
        self.path = results_dir / f"{model_key}__{rubric}.jsonl"
        self.meta_path = self.path.with_suffix(".meta.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def existing_keys(self) -> set[tuple[str, str, str, str]]:
        if not self.path.exists():
            return set()
        keys = set()
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    keys.add((d["model"], d["rubric"], d["order"], d["item_id"]))
        return keys

    def append(self, record: JudgmentRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            f.flush()

    def load(self) -> list[JudgmentRecord]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    records.append(JudgmentRecord(**json.loads(line)))
        return records

    def write_meta(self, meta: dict) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")


def load_records(paths: Iterable[Path]) -> list[JudgmentRecord]:
    """Load and concatenate several stores (for cross-model analysis)."""
    records: list[JudgmentRecord] = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(JudgmentRecord(**json.loads(line)))
    return records


# ---------------------------------------------------------------------------
# The llama.cpp runner
# ---------------------------------------------------------------------------

class LlamaJudge:
    """Thin wrapper owning the llama.cpp lifecycle for one judge model.

    ``n_ctx`` must be large enough for the longest prompt; to size it from
    the actual sample first, construct a throwaway instance with a small
    ``n_ctx`` (the weights are mmapped, so a second load is cheap), call
    ``max_prompt_tokens``, and rebuild. Prompts that exceed ``n_ctx`` at
    judgment time raise — silent truncation would corrupt the audit.
    (``vocab_only`` loading is not used: llama-cpp-python 0.3.x cannot create
    a context without weights, verified 2026-07-18.)
    """

    def __init__(
        self,
        model: JudgeModel,
        *,
        n_ctx: int = 4096,
        n_threads: int = 4,
        verify_sha256: bool = True,
    ):
        import llama_cpp

        if not model.path.exists():
            raise FileNotFoundError(
                f"model file {model.path} missing; download from {model.url}"
            )
        if verify_sha256:
            actual = sha256_file(model.path)
            if actual != model.sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {model.filename}: "
                    f"expected {model.sha256}, got {actual}"
                )

        self._llama_cpp = llama_cpp
        self.model = model
        self.template = CHAT_TEMPLATES[model.template]
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = llama_cpp.Llama(
            model_path=str(model.path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            logits_all=False,
            verbose=False,
            seed=0,
        )
        self.verdict_token_ids = self._resolve_verdict_tokens()

    def _resolve_verdict_tokens(self) -> tuple[int, int]:
        ids = []
        for letter in VERDICT_TOKENS:
            toks = self.llm.tokenize(letter.encode(), add_bos=False, special=False)
            if len(toks) != 1:
                raise RuntimeError(
                    f"verdict letter {letter!r} tokenizes to {len(toks)} tokens "
                    f"for {self.model.key}; single-token readout invalid"
                )
            ids.append(toks[0])
        return (ids[0], ids[1])

    def render(self, prompt: JudgePrompt) -> str:
        return self.template.render(prompt.system, prompt.user)

    def tokenize(self, prompt: JudgePrompt) -> list[int]:
        return self.llm.tokenize(self.render(prompt).encode(), add_bos=False, special=True)

    def max_prompt_tokens(self, prompts: Iterable[JudgePrompt]) -> int:
        return max(len(self.tokenize(p)) for p in prompts)

    def judge(self, prompt: JudgePrompt) -> JudgmentRecord:
        import numpy as np

        tokens = self.tokenize(prompt)
        if len(tokens) + 1 > self.n_ctx:
            raise RuntimeError(
                f"prompt for {prompt.item_id} is {len(tokens)} tokens; "
                f"exceeds n_ctx={self.n_ctx} (refusing to truncate)"
            )
        start = time.perf_counter()
        self.llm.reset()
        self.llm.eval(tokens)
        elapsed = time.perf_counter() - start

        n_vocab = self.llm.n_vocab()
        ptr = self._llama_cpp.llama_get_logits(self.llm._ctx.ctx)
        logits = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).astype(np.float64, copy=True)
        argmax_id = int(logits.argmax())
        argmax_token = self.llm.detokenize([argmax_id]).decode("utf-8", errors="replace")
        return logits_to_record(
            logits,
            token_ids=self.verdict_token_ids,
            argmax_token=argmax_token,
            prompt=prompt,
            model_key=self.model.key,
            n_prompt_tokens=len(tokens),
            prefill_seconds=elapsed,
        )

    def meta(self) -> dict:
        import llama_cpp

        return {
            "model_key": self.model.key,
            "model_file": self.model.filename,
            "model_sha256": self.model.sha256,
            "model_repo": self.model.repo,
            "model_revision": self.model.revision,
            "chat_template": self.template.name,
            "verdict_token_ids": list(self.verdict_token_ids),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "llama_cpp_python": llama_cpp.__version__,
        }

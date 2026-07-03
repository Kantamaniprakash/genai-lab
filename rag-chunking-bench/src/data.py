"""Dataset loading with exact gold spans in document coordinates.

The benchmark's central invariant is that gold evidence is addressed as
character spans in the *same* document string that chunkers segment. This
module enforces it at load time: every gold span is verified against the
document text (``document.text[span.start:span.end] == answer_text``) and a
mismatch is a hard error, never a silent skip.

SQuAD ships answers as spans inside individual paragraphs. Paragraph-sized
"documents" would make chunking trivial, so each Wikipedia article is
reconstructed by joining its paragraphs with a blank line, and every answer
span is remapped into article coordinates. This yields ~3-6k-token documents
with human-written questions and exact gold spans — the shape the benchmark
needs.

Gold-span semantics: a question carries a tuple of *alternatives*, each of
which is a tuple of required spans. SQuAD's multiple annotations are
alternative locations of the same answer (any one suffices — the standard
max-over-answers convention), so each distinct annotated span becomes a
singleton alternative. Corpora whose references are jointly required (e.g.
the Chroma evaluation corpora, loaded in a later phase) use a single
alternative holding all reference spans. Metrics take the max over
alternatives, so both semantics score correctly through one code path.

Raw JSON payloads are downloaded to ``data/`` (gitignored) with pinned URLs
and SHA256 checksums; ``python -m src.data`` fetches everything and prints
corpus statistics.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldSpan:
    """Half-open character range [start, end) of gold evidence in a document."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid gold span [{self.start}, {self.end})")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question with gold evidence in one document.

    `gold_alternatives` is a tuple of alternatives; each alternative is a
    tuple of spans that are jointly required. A retrieval is scored against
    the best-matching alternative.
    """

    qid: str
    text: str
    doc_id: str
    gold_alternatives: tuple[tuple[GoldSpan, ...], ...]

    def __post_init__(self) -> None:
        if not self.gold_alternatives or any(not alt for alt in self.gold_alternatives):
            raise ValueError(f"question {self.qid} has empty gold evidence")


@dataclass(frozen=True)
class QADataset:
    name: str
    documents: dict[str, Document]
    questions: tuple[Question, ...]


# Pinned sources. SHA256 computed from the files fetched on 2026-07-03;
# a checksum mismatch means upstream changed and results are not comparable.
_SQUAD_BASE = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset"
SQUAD_FILES = {
    "dev-v1.1.json": (
        f"{_SQUAD_BASE}/dev-v1.1.json",
        "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    ),
    "dev-v2.0.json": (
        f"{_SQUAD_BASE}/dev-v2.0.json",
        "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    ),
}

PARAGRAPH_JOINER = "\n\n"


def download_file(url: str, dest: Path, sha256: str) -> Path:
    """Download `url` to `dest` unless a file with the pinned hash exists."""
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
    got = _sha256(tmp)
    if got != sha256:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {got}")
    tmp.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def download_squad(data_dir: Path) -> dict[str, Path]:
    return {
        name: download_file(url, data_dir / name, sha256)
        for name, (url, sha256) in SQUAD_FILES.items()
    }


def load_squad(path: Path, name: str | None = None) -> QADataset:
    """Load a SQuAD v1.1/v2.0 JSON file as article-level documents.

    - Articles are rebuilt by joining paragraph contexts with a blank line;
      answer spans are remapped to article coordinates and verified verbatim.
    - Unanswerable v2.0 questions are dropped (no gold span to retrieve).
    - Duplicate annotated spans collapse to one alternative; duplicate
      (article, question-text) pairs keep the first occurrence only.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: dict[str, Document] = {}
    questions: list[Question] = []
    seen_qtext: set[tuple[str, str]] = set()
    for article in raw["data"]:
        doc_id = article["title"]
        offsets: list[int] = []
        pieces: list[str] = []
        cursor = 0
        for paragraph in article["paragraphs"]:
            offsets.append(cursor)
            pieces.append(paragraph["context"])
            cursor += len(paragraph["context"]) + len(PARAGRAPH_JOINER)
        text = PARAGRAPH_JOINER.join(pieces)
        documents[doc_id] = Document(doc_id=doc_id, title=doc_id, text=text)
        for offset, paragraph in zip(offsets, article["paragraphs"]):
            for qa in paragraph["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                key = (doc_id, qa["question"].strip())
                if key in seen_qtext:
                    continue
                seen_qtext.add(key)
                spans: list[GoldSpan] = []
                for answer in qa["answers"]:
                    start = offset + answer["answer_start"]
                    end = start + len(answer["text"])
                    if text[start:end] != answer["text"]:
                        raise ValueError(
                            f"span mismatch in {doc_id} qid={qa['id']}: "
                            f"{text[start:end]!r} != {answer['text']!r}"
                        )
                    span = GoldSpan(start, end)
                    if span not in spans:
                        spans.append(span)
                questions.append(
                    Question(
                        qid=qa["id"],
                        text=qa["question"].strip(),
                        doc_id=doc_id,
                        gold_alternatives=tuple((s,) for s in spans),
                    )
                )
    return QADataset(
        name=name or Path(path).stem,
        documents=documents,
        questions=tuple(questions),
    )


def sample_questions(
    dataset: QADataset, per_doc_cap: int, seed: int
) -> tuple[Question, ...]:
    """Deterministically sample at most `per_doc_cap` questions per document.

    Sampling is per-document with a seed derived from (seed, doc_id), so the
    selection for one article does not depend on which other articles are in
    the run — grids over document subsets stay comparable.
    """
    if per_doc_cap < 1:
        raise ValueError("per_doc_cap must be >= 1")
    by_doc: dict[str, list[Question]] = {}
    for q in dataset.questions:
        by_doc.setdefault(q.doc_id, []).append(q)
    sampled: list[Question] = []
    for doc_id in sorted(by_doc):
        pool = by_doc[doc_id]
        if len(pool) > per_doc_cap:
            rng = random.Random(f"{seed}:{doc_id}")
            pool = sorted(rng.sample(pool, per_doc_cap), key=pool.index)
        sampled.extend(pool)
    return tuple(sampled)


if __name__ == "__main__":
    from .tokenization import RegexWordTokenizer

    data_dir = Path(__file__).resolve().parent.parent / "data"
    paths = download_squad(data_dir)
    tokenizer = RegexWordTokenizer()
    for filename, path in paths.items():
        ds = load_squad(path)
        lengths = sorted(tokenizer.count(d.text) for d in ds.documents.values())
        n = len(lengths)
        print(
            f"{filename}: {len(ds.documents)} documents, {len(ds.questions)} questions | "
            f"doc tokens min/median/max = "
            f"{lengths[0]}/{lengths[n // 2]}/{lengths[-1]}"
        )

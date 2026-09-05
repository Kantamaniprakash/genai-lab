#!/usr/bin/env python3
"""Fetch the pinned RAGTruth corpus and verify it byte-for-byte.

Downloads the two dataset files from the pinned upstream commit and checks
their SHA256 digests against the pins below. Idempotent: files already
present and verified are left untouched; a present-but-mismatched file is an
error (never silently re-downloaded — a changed digest means the pin and the
analysis no longer describe the same data).

Data provenance: RAGTruth (Niu et al., ACL 2024, arXiv:2401.00396),
https://github.com/ParticleMedia/RAGTruth, MIT license, pinned at commit
c103204b9ce28d6bbad859304bf30de72b8ed8fe. The corpus is fetched, not
committed; `data/` is gitignored.

Usage: python -m experiments.fetch_data  (from the project root)
Stdlib only.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ragtruth"

PINNED_COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
BASE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    f"{PINNED_COMMIT}/dataset"
)

# filename -> (SHA256, expected line count)
PINS = {
    "response.jsonl": (
        "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
        17790,
    ),
    "source_info.jsonl": (
        "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
        2965,
    ),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def fetch(name: str) -> None:
    expected_sha, expected_lines = PINS[name]
    target = DATA_DIR / name

    if target.exists():
        actual = sha256_of(target)
        if actual != expected_sha:
            sys.exit(
                f"fetch_data: {target} exists but SHA256 {actual[:16]}… does "
                f"not match the pin {expected_sha[:16]}… — refusing to "
                "overwrite; move the file aside and re-run."
            )
        print(f"fetch_data: {name} already present and verified")
        return

    url = f"{BASE_URL}/{name}"
    print(f"fetch_data: downloading {url}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)

    actual = sha256_of(tmp)
    if actual != expected_sha:
        tmp.unlink()
        sys.exit(
            f"fetch_data: downloaded {name} has SHA256 {actual[:16]}… , "
            f"expected {expected_sha[:16]}… — upstream content changed; "
            "do not proceed."
        )
    lines = line_count(tmp)
    if lines != expected_lines:
        tmp.unlink()
        sys.exit(
            f"fetch_data: {name} has {lines} lines, expected {expected_lines}"
        )
    tmp.rename(target)
    print(f"fetch_data: {name} verified ({expected_lines} lines, sha256 ok)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in PINS:
        fetch(name)
    print(f"fetch_data: corpus ready under {DATA_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sync the README "Latest from the lab" section from the newest NOTES.md entry.

Run by .github/workflows/update-readme.yml on every push that touches the
research log; can also be run locally. Stdlib only.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"
START, END = "<!-- latest-start -->", "<!-- latest-end -->"


def current_flagship() -> str:
    """The project directory ROADMAP.md names as the current flagship.

    Hardcoding the project here is what let the landing page freeze on
    `rag-chunking-bench` for the five weeks after that flagship closed. The
    roadmap already carries the answer and is updated the day a flagship
    changes, so it is the single source of truth.
    """
    match = re.search(r"^## Current flagship: `([^`]+)`", ROADMAP.read_text(
        encoding="utf-8"), flags=re.M)
    if not match:
        sys.exit("sync_latest: no '## Current flagship: `name`' line in ROADMAP.md")
    name = match.group(1)
    if not (ROOT / name / "research" / "NOTES.md").exists():
        sys.exit(f"sync_latest: flagship {name!r} has no research/NOTES.md")
    return name


def gh_anchor(heading: str) -> str:
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def day_entries(notes_text: str):
    """All (heading, body) day sections, oldest first."""
    days = list(re.finditer(r"^## (.+)$", notes_text, flags=re.M))
    if not days:
        sys.exit("sync_latest: no '## ' day entries found in NOTES.md")
    entries = []
    for match, nxt in zip(days, days[1:] + [None]):
        end = nxt.start() if nxt else len(notes_text)
        entries.append((match.group(1), notes_text[match.end():end]))
    return entries


# A day's findings have been written three ways across the lab's projects: as
# a numbered list under a "### Findings" heading (rag-chunking-bench), as
# free-standing bold paragraphs anywhere in the entry, and as their own "###"
# headings (slm-judge-audit uses both of the latter). All three are matched,
# results are taken in document order, and a finding claimed twice in one entry
# is kept once — the entry body often restates a heading finding inline.
FINDING_PATTERNS = (
    r"^\d+\.\s+\*\*(Finding \d+.+?)\*\*",
    r"^\*\*(Finding \d+ .+?)\*\*",
    r"^###\s+(Finding \d+ .+?)\s*$",
)


def finding_leads(body: str):
    scoped = re.search(r"^### Findings.*?$(.*?)(?=^### |\Z)", body,
                       flags=re.M | re.S)
    haystacks = [scoped.group(1)] if scoped else []
    # The scoped section is preferred, but headings and stray bold paragraphs
    # live outside it, so the whole entry is scanned too.
    haystacks.append(body)

    found: dict[str, str] = {}
    for haystack in haystacks:
        for pattern in FINDING_PATTERNS:
            for match in re.finditer(pattern, haystack, flags=re.M | re.S):
                lead = re.sub(r"\s+", " ", match.group(1)).rstrip(".") + "."
                found.setdefault(re.match(r"Finding (\d+)", lead).group(1), lead)
    return [found[n] for n in sorted(found, key=int)]


def entry_link(heading: str, project: str) -> str:
    return f"{project}/research/NOTES.md#{gh_anchor(heading)}"


def digest(heading: str, body: str, project: str) -> str:
    lines = [f"**{heading}**", ""]
    leads = finding_leads(body)
    if leads:
        lines += ["- " + lead for lead in leads]
    else:
        # ponytail: no Findings section (side-repo days) -> first prose paragraph
        para = body.strip().split("\n\n")[0]
        if not para.startswith("#"):
            lines.append(re.sub(r"\s+", " ", para))
    lines += ["", f"[Full entry →]({entry_link(heading, project)})"]
    return "\n".join(lines)


def render(notes_text: str, project: str = "rag-chunking-bench") -> str:
    entries = day_entries(notes_text)
    heading, body = entries[-1]
    block = digest(heading, body, project)
    if not finding_leads(body):
        # keep hard numbers on the landing page even on side-repo days:
        # append the most recent entry that carries findings
        for prev_heading, prev_body in reversed(entries[:-1]):
            leads = finding_leads(prev_body)
            if leads:
                block += (
                    f"\n\n**Most recent findings** ([{prev_heading}]"
                    f"({entry_link(prev_heading, project)})):\n\n"
                    + "\n".join("- " + lead for lead in leads)
                )
                break
    return (
        f"{START}\n"
        "## Latest from the lab\n\n"
        "<!-- auto-generated from research/NOTES.md by scripts/sync_latest.py; do not hand-edit -->\n\n"
        f"{block}\n"
        f"{END}"
    )


def main() -> None:
    readme = README.read_text(encoding="utf-8")
    if START not in readme or END not in readme:
        sys.exit("sync_latest: latest-start/latest-end markers missing from README.md")
    project = current_flagship()
    notes = ROOT / project / "research" / "NOTES.md"
    block = render(notes.read_text(encoding="utf-8"), project)
    updated = re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        lambda _: block,
        readme,
        flags=re.S,
    )
    if updated != readme:
        README.write_text(updated, encoding="utf-8")
        print("sync_latest: README.md updated")
    else:
        print("sync_latest: README.md already current")


def selftest() -> None:
    with_findings = (
        "# Log\n\n## 2026-01-01 — Day 1: old\n\nold text\n\n"
        "## 2026-01-02 — Day 2: new stuff — findings 1–2\n\n"
        "### Findings (README §1)\n\n"
        "1. **Finding 1 — spans\n   two lines.** Detail prose.\n"
        "2. **Finding 2 — short.** More detail.\n\n"
        "### Next steps\n\n- whatever\n"
    )
    out = render(with_findings)
    assert "**2026-01-02 — Day 2: new stuff — findings 1–2**" in out
    assert "- Finding 1 — spans two lines." in out
    assert "- Finding 2 — short." in out
    assert "Detail prose" not in out
    assert "#2026-01-02--day-2-new-stuff--findings-12" in out

    no_findings = with_findings + (
        "\n## 2026-01-03 — Day 3: side-repo day\n\n"
        "First transfer of results into\nproduction code.\n\n"
        "### What shipped there\n\n- a thing\n"
    )
    out = render(no_findings)
    assert "**2026-01-03 — Day 3: side-repo day**" in out
    assert "First transfer of results into production code." in out
    assert "a thing" not in out
    # side-repo day still carries the newest hard numbers, from day 2
    assert "**Most recent findings**" in out
    assert "- Finding 2 — short." in out

    # The current flagship writes findings as bold paragraphs and as their own
    # headings rather than as a numbered list, and restates them inline.
    other_shapes = (
        "# Log\n\n## 2026-08-21 — Day 7: mixed shapes\n\n"
        "### Experiment: something\n\n"
        "**Finding 12 — a bold paragraph finding\nwrapped over lines.** Detail.\n\n"
        "### Finding 13 — a heading finding\n\n"
        "Body prose that mentions **Finding 13 — a heading finding** again.\n\n"
        "### Next steps\n\n- tomorrow\n"
    )
    out = render(other_shapes, "slm-judge-audit")
    assert "- Finding 12 — a bold paragraph finding wrapped over lines." in out
    assert "- Finding 13 — a heading finding." in out
    assert out.count("Finding 13 — a heading finding") == 1   # deduped
    assert "Detail." not in out and "tomorrow" not in out
    # Links point at the flagship the caller named, not a hardcoded project.
    assert "slm-judge-audit/research/NOTES.md#2026-08-21--day-7-mixed-shapes" in out

    # Findings are ordered by number, not by which pattern matched first.
    assert out.index("Finding 12") < out.index("Finding 13")
    print("sync_latest: selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()

#!/usr/bin/env python3
"""Sync the README "Latest from the lab" section from the newest NOTES.md entry.

Run by .github/workflows/update-readme.yml on every push that touches the
research log; can also be run locally. Stdlib only.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "rag-chunking-bench" / "research" / "NOTES.md"
README = ROOT / "README.md"
START, END = "<!-- latest-start -->", "<!-- latest-end -->"


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


def finding_leads(body: str):
    findings = re.search(
        r"^### Findings.*?$(.*?)(?=^### |\Z)", body, flags=re.M | re.S
    )
    if not findings:
        return []
    leads = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*", findings.group(1), flags=re.M | re.S)
    return [re.sub(r"\s+", " ", lead) for lead in leads]


def entry_link(heading: str) -> str:
    return f"rag-chunking-bench/research/NOTES.md#{gh_anchor(heading)}"


def digest(heading: str, body: str) -> str:
    lines = [f"**{heading}**", ""]
    leads = finding_leads(body)
    if leads:
        lines += ["- " + lead for lead in leads]
    else:
        # ponytail: no Findings section (side-repo days) -> first prose paragraph
        para = body.strip().split("\n\n")[0]
        if not para.startswith("#"):
            lines.append(re.sub(r"\s+", " ", para))
    lines += ["", f"[Full entry →]({entry_link(heading)})"]
    return "\n".join(lines)


def render(notes_text: str) -> str:
    entries = day_entries(notes_text)
    heading, body = entries[-1]
    block = digest(heading, body)
    if not finding_leads(body):
        # keep hard numbers on the landing page even on side-repo days:
        # append the most recent entry that carries findings
        for prev_heading, prev_body in reversed(entries[:-1]):
            leads = finding_leads(prev_body)
            if leads:
                block += (
                    f"\n\n**Most recent findings** ([{prev_heading}]"
                    f"({entry_link(prev_heading)})):\n\n"
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
    block = render(NOTES.read_text(encoding="utf-8"))
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
    print("sync_latest: selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()

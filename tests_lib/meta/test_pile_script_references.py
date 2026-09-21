"""Every script the pile's guides name must exist, or be listed as retired (#4060).

The VG retirement (#4038) deleted 38 files and did a careful citation pass over
thirteen *published reports* -- but touched `scripts/experiments/pile/README.md`
in two lines.  The live operating guide went on naming **sixteen** deleted
scripts, ten of them inside runnable ``python foo.py ...`` recipes, and nothing
said so: a reader following the "Auditing a class's VG names" section would have
typed four commands in a row that no longer exist.

A dangling script name in a guide is not the same defect as a dangling link in a
report.  A report is attribution -- it says which instrument produced a published
number, and deleting the name would leave the figure unattributed, which is
worse.  A guide is an instruction.  So this gate does not ban the *name*; it
requires the guide to say the name is retired, and it bans the *recipe*.

The allowlist is the README's own "Retired with Visual Genome" table rather than
a list in here, so the document is the single source of truth and cannot drift
from its gate.  The check runs in both directions: a name in that table that
comes back to disk is as much a defect as a name in the prose that left it.
"""

from __future__ import annotations

import re
import subprocess
from functools import cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PILE = REPO_ROOT / "scripts" / "experiments" / "pile"
README = PILE / "README.md"

#: The guides this gate holds.  Both are read by a human about to run something.
GUIDES = (README, PILE / "ANNOTATION-GUIDE.md")

#: A bare or backticked script basename.  Deliberately loose: it matches inside
#: fenced recipes (``python coco_folds.py``), inside prose backticks and inside
#: markdown link targets alike, because all three read as "this exists".
SCRIPT_RE = re.compile(r"\b([A-Za-z0-9_]+\.(?:py|sh))\b")

#: The heading whose table is the allowlist, and the row shape it is read with.
RETIRED_HEADING = "## Retired with Visual Genome (#4038)"
RETIRED_ROW_RE = re.compile(r"^\|\s*`([A-Za-z0-9_]+\.(?:py|sh))`\s*\|")

#: A fenced line that invokes something.  A retired script may be *named* in
#: prose and may not appear here -- a recipe is an instruction, not a citation.
INVOCATION_RE = re.compile(r"^\s*(?:[A-Z][A-Z0-9_]*=\S*\s+)*(?:python|bash|sh)\s")


@cache
def _tracked_basenames() -> frozenset[str]:
    """Every tracked file's basename.  `git ls-files` rather than a walk, so an
    untracked build artefact can never make a dangling reference look resolved."""
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return frozenset(line.rsplit("/", 1)[-1] for line in out.splitlines() if line)


@cache
def _retired() -> frozenset[str]:
    """The names the README declares retired, read out of its own table."""
    lines = README.read_text().splitlines()
    try:
        start = lines.index(RETIRED_HEADING)
    except ValueError:  # pragma: no cover - the assertion below reports it
        return frozenset()
    names = set()
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if match := RETIRED_ROW_RE.match(line):
            names.add(match.group(1))
    return frozenset(names)


def _fenced_lines(text: str) -> list[str]:
    """Lines inside ``` fences -- the ones a reader copies and runs."""
    inside, out = False, []
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        if inside:
            out.append(line)
    return out


def test_the_retired_table_is_present_and_populated():
    """The allowlist is a document section, so its absence must fail loudly
    rather than silently allowing nothing (which would fail every other test
    here with a confusing message)."""
    assert RETIRED_HEADING in README.read_text(), f"{README.name} lost {RETIRED_HEADING!r}"
    assert len(_retired()) >= 16, (
        f"the retired table lists {len(_retired())} scripts; the VG retirement deleted 16 "
        "that these guides name. Did a row lose its backticks?"
    )


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_every_script_named_in_a_guide_exists_or_is_declared_retired(guide: Path):
    """The gate #4060 asked for: no guide may name a script that is not there
    without saying it is gone."""
    named = set(SCRIPT_RE.findall(guide.read_text()))
    dangling = sorted(named - _tracked_basenames() - _retired())
    assert not dangling, (
        f"{guide.relative_to(REPO_ROOT)} names {len(dangling)} script(s) that do not exist "
        f"and are not listed under {RETIRED_HEADING!r} in README.md: {', '.join(dangling)}.\n"
        "Either the script moved (fix the reference), or it was deleted -- in which case add "
        "a row to that table and make sure no recipe still invokes it."
    )


@pytest.mark.parametrize("guide", GUIDES, ids=lambda p: p.name)
def test_no_recipe_invokes_a_retired_script(guide: Path):
    """Marking a name retired is enough for prose and never enough for a fence.
    This is the half that actually protects the reader: #4060's complaint was
    ten runnable commands, not sixteen mentions."""
    offenders = [
        line.strip()
        for line in _fenced_lines(guide.read_text())
        if INVOCATION_RE.match(line) and (set(SCRIPT_RE.findall(line)) & _retired())
    ]
    assert not offenders, (
        f"{guide.relative_to(REPO_ROOT)} still tells a reader to run a retired script:\n  "
        + "\n  ".join(offenders)
        + "\nA recipe is an instruction and cannot be annotated into truth -- remove the line "
        "and keep the finding in prose."
    )


def test_every_retired_script_is_really_gone():
    """The other direction.  A name that comes back to disk while the table still
    calls it retired would let the gate above wave through a live reference, and
    would tell a reader the wrong thing besides."""
    resurrected = sorted(_retired() & _tracked_basenames())
    assert not resurrected, (
        f"README.md lists {', '.join(resurrected)} as retired, but the file(s) exist again. "
        "Drop the row(s) from the retired table."
    )

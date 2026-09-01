#!/usr/bin/env python3
"""Gate: every script in ``scripts/experiments/calibration/`` is in its README index.

The directory is deliberately flat -- 120-odd files, ~20 concluded studies, one
namespace -- because these are archival cluster scripts whose paths are quoted
by the reports they produced, and whose sibling imports (``import common``) work
only because the launcher ``cd``s here.  Issue #3409 proposed splitting them into
per-study subdirectories; that trade was declined (ungated 80-file churn, ~110
stale doc references) in favour of the thing the subdirectories were actually
wanted for: **knowing which files belong to which study.**

That answer is the README's index, and an index nobody checks is an index that
decays back into 120 unclassified files -- which is how the directory got here.
So this makes the mapping total and un-rottable in both directions:

* every ``.py`` / ``.sh`` in the directory appears in the index exactly once, so
  a new study's files cannot land unclassified;
* every file the index names exists, so a deletion cannot leave a phantom row.

Pure stdlib and ~20ms, so it runs in ``run-tests.sh``'s cheap serial stage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "scripts" / "experiments" / "calibration"
README = CALIB / "README.md"

BEGIN = "<!-- BEGIN INDEX -->"
END = "<!-- END INDEX -->"

#: Backticked ``name.py`` / ``name.sh`` tokens, bare (no slash): the index names
#: files relative to the directory it documents.
TOKEN_RE = re.compile(r"`([A-Za-z0-9_]+\.(?:py|sh))`")

#: A **parenthesised** mention is a cross-reference, not a classification -- the
#: #2808 row noting that its analysis reuses ``analyze_spikes.py`` must not make
#: that file look like it is filed under two studies.  Stripping parentheses
#: before tokenizing fails safe: a filename accidentally left inside them reads
#: as unclassified (a loud failure), never as classified twice.
PARENS_RE = re.compile(r"\([^()]*\)")


def indexed_names(text: str) -> tuple[list[str], str | None]:
    """Every filename the index region *classifies*, in order; or an error string."""
    if BEGIN not in text or END not in text:
        return [], f"README.md is missing its {BEGIN} / {END} sentinels"
    region = text.split(BEGIN, 1)[1].split(END, 1)[0]
    # Markdown links are ``[text](target)``; the targets are paths, not entries.
    region = PARENS_RE.sub(" ", region)
    return TOKEN_RE.findall(region), None


def main() -> int:
    if not README.is_file():
        print(f"FAIL: {README} does not exist", file=sys.stderr)
        return 1

    named, err = indexed_names(README.read_text(encoding="utf-8"))
    if err:
        print(f"FAIL: {err}", file=sys.stderr)
        return 1

    on_disk = {p.name for p in CALIB.iterdir() if p.suffix in (".py", ".sh")}
    listed = set(named)

    problems: list[str] = []

    dupes = sorted({n for n in named if named.count(n) > 1})
    if dupes:
        problems.append(
            "listed more than once (a file belongs to exactly one study, or to the shared layer): " + ", ".join(dupes)
        )

    missing = sorted(on_disk - listed)
    if missing:
        problems.append(
            "in the directory but not in the README index -- add a row for the "
            "study they belong to, or list them under the shared layer: " + ", ".join(missing)
        )

    phantom = sorted(listed - on_disk)
    if phantom:
        problems.append("named by the README index but not on disk -- drop the entry: " + ", ".join(phantom))

    if problems:
        print("TESTS BLOCKED: scripts/experiments/calibration/README.md index is out of date", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"calibration index OK: {len(on_disk)} scripts, all classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

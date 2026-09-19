"""Fail clearly when a study's output directory is gone (#4001).

Finished study output dirs under ``/expscratch/sgreenberg`` were deleted on
2026-09-18 under the ruling that re-running an old study is not a reason to keep
its bytes.  The analysis scripts that read them still carry their old defaults,
so without this they die on whatever file they happened to open first -- a
``FileNotFoundError`` on ``results/grid_shape.json`` names neither the study nor
the reason.

The check belongs to **readers only**.  A launcher creates its study dir, so
guarding one would break the re-run that is the whole recovery path.
"""

from __future__ import annotations

import sys
from pathlib import Path

RECORD = "/expscratch/sgreenberg/keep/deleted-20260918.md"
_GONE = "was deleted 2026-09-18 (see #4001 and {record})."


def require_study_file(path: str | Path, flag: str, produced_by: str = "") -> Path:
    """Return *path*, or exit naming it, the deletion, and what rebuilds it.

    The file variant of :func:`require_study_dir`, for the derived artefacts the
    `pile/` readers take as input -- ``annotation_queue.jsonl``, a pass's
    ``controls.json``.  Those are *regenerable*, and a reader that dies on a
    missing one should say by what, because re-running the producer is the whole
    recovery path rather than a consolation.
    """
    p = Path(path)
    if p.is_file():
        return p
    how = f"Re-run {produced_by} to rebuild it" if produced_by else "Re-run the study"
    print(
        f"{p} does not exist: this study's output {_GONE.format(record=RECORD)}\n{how}, or point {flag} at a copy.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def require_study_dir(path: str | Path, flag: str = "--exp") -> Path:
    """Return *path*, or exit naming it and where the deletion is recorded."""
    p = Path(path)
    if p.is_dir():
        return p
    print(
        f"{p} does not exist: this study's output was deleted 2026-09-18 "
        f"(see #4001 and {RECORD}).\nRe-run the study, or point {flag} at a directory that has its results.",
        file=sys.stderr,
    )
    raise SystemExit(2)

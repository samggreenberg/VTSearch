"""Shared setup for the inclusion-knob experiment (issue #2693).

Every stage script imports this and calls :func:`setup_env` **before**
importing anything under ``vtscore`` (the data-dir env vars must be set
first).  Unlike the MLP-vs-SVM grid study this experiment is sized to run
on a single CPU box: the embedding cache and scratch data live under an
experiment directory outside the repo (``INCKNOB_EXP``, defaulting to a
temp-style path), while the committed outputs (CSVs, figures, report) go
to ``docs/experiments/2026-07-27-inclusion-knob/`` in the worktree.

The env/dir setup lives in ``scripts/experiments/_expcommon.py``, shared with
the other studies (#3411).  Because this one runs from the repo checkout
itself, there is no second checkout for the venv's editable install to resolve
to, so it does **not** neutralise that finder the way the grid studies do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import _expcommon
from _expcommon import log, timed  # re-exported: stages call ``common.timed`` / ``common.log``

__all__ = ["CACHE", "DATADIR", "EXP", "REPO", "RESULTS", "log", "setup_env", "timed"]

REPO = Path(__file__).resolve().parents[3]
EXP = Path(os.environ.get("INCKNOB_EXP", str(Path.home() / ".cache" / "incknob-exp")))
DATADIR = EXP / "datadir"
CACHE = EXP / "cache"
#: The study directory every stage reads its committed inputs from and writes
#: its outputs to.  Dated, matching the rest of ``docs/experiments/`` since
#: #3328 — this constant kept the pre-#3328 undated spelling until #3408, so
#: every stage's default ``--csv``/``--out`` pointed at a directory that did not
#: exist and ``setup_env`` silently created it empty.
RESULTS = REPO / "docs" / "experiments" / "2026-07-27-inclusion-knob"


def setup_env() -> None:
    """Point vtscore + HF caches at the experiment dirs; make the repo importable."""
    _expcommon.setup_env(
        repo=REPO,
        datadir=DATADIR,
        models_dir=EXP / "models",
        results=RESULTS,
        extra_dirs=(CACHE,),
        neutralise=False,
    )

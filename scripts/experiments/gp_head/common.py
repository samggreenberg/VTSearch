"""Shared setup for the Gaussian-process head pilot (issue #3954).

Every stage script imports this and calls :func:`setup_env` **before**
importing anything under ``vtscore`` (the data-dir env vars must be set
first).  Like the inclusion-knob study this one is sized for a single CPU box:
the embedding cache and scratch data live under an experiment directory
outside the repo (``GPHEAD_EXP``, default ``~/gp-head``), the per-cell CSVs
under ``GPHEAD_RESULTS`` (default ``$GPHEAD_EXP/results``), and the committed
outputs (tables, figures, viewer, report) go to
``docs/experiments/2026-09-17-gp-head-3954/`` in the checkout.

The env/dir setup lives in ``scripts/experiments/_expcommon.py``, shared with
the other studies (#3411).  This study runs from the repo checkout itself, so
there is no second checkout for the venv's editable install to resolve to and
it does **not** neutralise that finder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import _expcommon
from _expcommon import log, timed  # re-exported: stages call ``common.timed`` / ``common.log``

__all__ = ["DATADIR", "EXP", "REPO", "RESULTS", "STUDY", "log", "setup_env", "timed"]

REPO = Path(__file__).resolve().parents[3]
EXP = Path(os.environ.get("GPHEAD_EXP", str(Path.home() / "gp-head")))
DATADIR = EXP / "datadir"
RESULTS = Path(os.environ.get("GPHEAD_RESULTS", str(EXP / "results")))
#: The study directory the committed outputs land in.
STUDY = REPO / "docs" / "experiments" / "2026-09-17-gp-head-3954"


def setup_env() -> None:
    """Point vtscore + HF caches at the experiment dirs; make the repo importable."""
    _expcommon.setup_env(
        repo=REPO,
        datadir=DATADIR,
        models_dir=EXP / "models",
        results=RESULTS,
        hf_home=EXP / "hf",
        neutralise=False,
    )

"""Shared setup for the MLP-vs-SVM cluster experiment.

Every stage script imports this and calls :func:`setup_env` **before** importing
anything under ``vtscore`` (the data-dir env vars have to be set first).  The
experiment keeps its embeddings, models, and results under ``/exp/$USER/mlp-svm``
so the demo pickle cache is shared across all SLURM array tasks (embed once,
reuse everywhere).

The worktree is put on ``sys.path`` and the main venv's editable-install
meta-path finder is neutralised so ``import vtscore`` resolves to *this* branch
rather than the main checkout (the grid's ``__editable__.vtsearch`` ``.pth``
otherwise wins over ``sys.path``).  Launching via ``gridenv.sh`` does the same
thing through ``PYTHONPATH``; doing it here too makes the scripts robust to being
run either way.  That machinery lives in
``scripts/experiments/_expcommon.py``, shared with the other grid studies so the
#2846 fix has one home (#3411); only this study's constants are here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import _expcommon
from _expcommon import log, timed  # re-exported: stages call ``common.timed`` / ``common.log``

__all__ = ["DATADIR", "EXP", "REPO", "RESULTS", "USER", "WARM_HF", "log", "setup_env", "timed"]

USER = os.environ.get("USER", "sgreenberg")
REPO = Path(os.environ.get("VTS_REPO", f"/exp/{USER}/projects/vts-mlpsvm"))
EXP = Path(os.environ.get("MLPSVM_EXP", f"/exp/{USER}/mlp-svm"))
DATADIR = EXP / "datadir"
RESULTS = Path(os.environ.get("MLPSVM_RESULTS", str(EXP / "results")))
WARM_HF = f"/exp/{USER}/.cache/huggingface"


def setup_env() -> None:
    """Point vtscore + HF at the experiment dirs and make the worktree importable."""
    _expcommon.setup_env(
        repo=REPO,
        datadir=DATADIR,
        models_dir=EXP / "models",
        results=RESULTS,
        hf_home=WARM_HF if Path(WARM_HF).exists() else EXP / "hf",
    )

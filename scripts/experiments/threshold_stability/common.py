"""Shared setup for the threshold-stability cluster experiment (issue #2790).

Every stage imports this and calls :func:`setup_env` before importing anything
under ``vtscore``. Mirrors the calibration / max_patch runners: point vtscore + HF
at the experiment dirs, put the worktree on ``sys.path``, and neutralise the main
venv's editable-install finder so ``import vtscore`` resolves to *this* branch.

CPU-only: no models are downloaded (the SigLIP 2 embeddings are read from the
reused #2790 cache), so this study needs no GPU partition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

USER = os.environ.get("USER", "sgreenberg")
REPO = Path(os.environ.get("VTS_REPO", f"/exp/{USER}/projects/vts-evalfw"))
EXP = Path(os.environ.get("THRSTAB_EXP", f"/exp/{USER}/threshold-stability"))
RESULTS = Path(os.environ.get("THRSTAB_RESULTS", str(EXP / "results")))
#: The reused #2790 SigLIP 2 / whole embedding cache (see the plan). Defaults to the
#: sweep's own cache dir; override with THRSTAB_CACHE_DIR to point at the real run.
CACHE_DIR = Path(os.environ.get("THRSTAB_CACHE_DIR", str(EXP / "cache")))
WARM_HF = f"/exp/{USER}/.cache/huggingface"


def setup_env() -> None:
    os.environ.setdefault("VTSEARCH_DATA_DIR", str(EXP / "datadir"))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(EXP / "models"))
    os.environ.setdefault("HF_HOME", WARM_HF if Path(WARM_HF).exists() else str(EXP / "hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR", "HF_HOME"):
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    _neutralise_editable_finder()


def _neutralise_editable_finder() -> None:
    """Drop the main venv's ``__editable__.vtsearch`` meta-path finder, if present."""
    keep = []
    for finder in sys.meta_path:
        mod = type(finder).__module__ or ""
        name = f"{mod}.{type(finder).__name__}".lower()
        if "editable" in name and ("vtsearch" in name or "vtscore" in name):
            continue
        keep.append(finder)
    sys.meta_path[:] = keep


def log(msg: str) -> None:
    print(msg, flush=True)

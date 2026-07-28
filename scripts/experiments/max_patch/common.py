"""Shared setup for the Max-Patch cluster experiment.

Every stage script imports this and calls :func:`setup_env` **before** importing
anything under ``vtscore`` (the data-dir env vars have to be set first).  The
experiment keeps its embeddings, models, and results under
``/exp/$USER/max-patch`` so the per-(dataset, embedder) pickles are shared
across all SLURM array tasks (embed once, reuse everywhere).

The worktree is put on ``sys.path`` and the main venv's editable-install
meta-path finder is neutralised so ``import vtscore`` resolves to *this* branch
rather than the main checkout (the grid's ``__editable__.vtsearch`` ``.pth``
otherwise wins over ``sys.path``).  Launching via ``gridenv.sh`` does the same
thing through ``PYTHONPATH``; doing it here too makes the scripts robust to
being run either way.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

USER = os.environ.get("USER", "sgreenberg")
REPO = Path(os.environ.get("VTS_REPO", f"/exp/{USER}/projects/vts-maxpatch"))
EXP = Path(os.environ.get("MAXPATCH_EXP", f"/exp/{USER}/max-patch"))
DATADIR = EXP / "datadir"
RESULTS = Path(os.environ.get("MAXPATCH_RESULTS", str(EXP / "results")))
WARM_HF = f"/exp/{USER}/.cache/huggingface"


def setup_env() -> None:
    """Point vtscore + HF at the experiment dirs and make the worktree importable."""
    os.environ.setdefault("VTSEARCH_DATA_DIR", str(DATADIR))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(EXP / "models"))
    os.environ.setdefault("HF_HOME", WARM_HF if Path(WARM_HF).exists() else str(EXP / "hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR", "HF_HOME"):
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Make the worktree win over the main venv's editable install.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    _neutralise_editable_finder()


def _neutralise_editable_finder() -> None:
    """Drop the main venv's ``__editable__.vtsearch`` meta-path finder, if present.

    That finder maps ``vtscore``/``vtsearch`` to the main checkout regardless of
    ``sys.path``; removing it lets the worktree (now first on ``sys.path``) win.
    """
    keep = []
    for finder in sys.meta_path:
        mod = type(finder).__module__ or ""
        name = f"{mod}.{type(finder).__name__}".lower()
        if "editable" in name and ("vtsearch" in name or "vtscore" in name):
            continue
        keep.append(finder)
    sys.meta_path[:] = keep


@contextmanager
def timed(label: str, sink: dict):
    """Record wall-clock seconds for *label* into *sink* and print it."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    sink[label] = round(dt, 2)
    print(f"[timing] {label}: {dt:.1f}s", flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)

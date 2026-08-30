"""Shared setup for the inclusion-knob experiment (issue #2693).

Every stage script imports this and calls :func:`setup_env` **before**
importing anything under ``vtscore`` (the data-dir env vars must be set
first).  Unlike the MLP-vs-SVM grid study this experiment is sized to run
on a single CPU box: the embedding cache and scratch data live under an
experiment directory outside the repo (``INCKNOB_EXP``, defaulting to a
temp-style path), while the committed outputs (CSVs, figures, report) go
to ``docs/experiments/2026-07-27-inclusion-knob/`` in the worktree.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = Path(os.environ.get("INCKNOB_EXP", str(Path.home() / ".cache" / "incknob-exp")))
DATADIR = EXP / "datadir"
CACHE = EXP / "cache"
RESULTS = REPO / "docs" / "experiments" / "inclusion-knob"


def setup_env() -> None:
    """Point vtscore + HF caches at the experiment dirs; make the repo importable."""
    os.environ.setdefault("VTSEARCH_DATA_DIR", str(DATADIR))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(EXP / "models"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for d in (DATADIR, CACHE, EXP / "models", RESULTS):
        d.mkdir(parents=True, exist_ok=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


@contextmanager
def timed(label: str, sink: dict | None = None):
    """Record wall-clock seconds for *label* (optionally into *sink*) and print it."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    if sink is not None:
        sink[label] = round(dt, 2)
    print(f"[timing] {label}: {dt:.1f}s", flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)

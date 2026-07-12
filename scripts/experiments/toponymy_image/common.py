"""Shared paths + environment for the Toponymy image-signpost experiments.

These experiments run on the HLTCOE grid (GPU node), driving the *library
tier* of VTSearch (``vtscore``) directly — no Flask app. All heavy state
(datasets, HF models, venv) lives on node-local scratch; only small durable
artifacts (embeddings npy, texts, topic trees, metrics) go to RESULTS.

Import ``common`` and call :func:`setup_env` **before** importing anything
from ``vtscore`` — ``vtscore.config`` reads ``VTSEARCH_DATA_DIR`` /
``VTSEARCH_MODELS_DIR`` at import time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(os.environ.get("VTS_REPO", "/exp/sgreenberg/projects/VTSearch"))
# Node-local scratch: on HLTCOE grid nodes only /scratch/jobs/$USER is
# user-writable (SLURM-managed; may be cleaned when the user's jobs end).
WORK = Path(os.environ.get("TOPO_WORK", f"/scratch/jobs/{os.environ.get('USER', 'sgreenberg')}/topo-image"))
RESULTS = Path(os.environ.get("TOPO_RESULTS", "/exp/sgreenberg/experiments/toponymy-image/results"))


def setup_env() -> None:
    """Point vtscore + HF at scratch, put the repo on sys.path."""
    os.environ.setdefault("VTSEARCH_DATA_DIR", str(WORK / "vts-data"))
    os.environ.setdefault("VTSEARCH_MODELS_DIR", str(WORK / "vts-models"))
    os.environ.setdefault("HF_HOME", str(WORK / "hf"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for var in ("VTSEARCH_DATA_DIR", "VTSEARCH_MODELS_DIR", "HF_HOME"):
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def ds_dir(dataset: str) -> Path:
    d = RESULTS / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
    print(f"[saved] {path}")


def load_json(path: Path):
    return json.loads(path.read_text())


@contextmanager
def timed(label: str, timings: dict):
    t0 = time.time()
    yield
    dt = time.time() - t0
    timings[label] = round(dt, 2)
    print(f"[timing] {label}: {dt:.1f}s", flush=True)

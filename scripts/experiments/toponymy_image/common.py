"""Shared paths + environment for the Toponymy image-signpost experiments.

These experiments run on the HLTCOE grid (GPU node), driving the *library
tier* of VTSearch (``vtscore``) directly — no Flask app. All heavy state
(datasets, HF models, venv) lives on node-local scratch; only small durable
artifacts (embeddings npy, texts, topic trees, metrics) go to RESULTS.

Import ``common`` and call :func:`setup_env` **before** importing anything
from ``vtscore`` — ``vtscore.config`` reads ``VTSEARCH_DATA_DIR`` /
``VTSEARCH_MODELS_DIR`` at import time.

The env/dir setup itself lives in ``scripts/experiments/_expcommon.py``, shared
with the other studies (#3411).  These runs launch from the repo checkout the
grid job ``cd``s into rather than a dedicated worktree, so they do **not**
neutralise the venv's editable-install finder the way the ``calibration`` /
``max_patch`` / ``mlp_vs_svm`` grids do.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import _expcommon
from _expcommon import timed  # re-exported: stages call ``common.timed``

__all__ = ["REPO", "RESULTS", "WORK", "ds_dir", "load_json", "save_json", "setup_env", "timed"]

REPO = Path(os.environ.get("VTS_REPO", "/exp/sgreenberg/projects/VTSearch"))
# Node-local scratch: on HLTCOE grid nodes only /scratch/jobs/$USER is
# user-writable (SLURM-managed; may be cleaned when the user's jobs end).
WORK = Path(os.environ.get("TOPO_WORK", f"/scratch/jobs/{os.environ.get('USER', 'sgreenberg')}/topo-image"))
RESULTS = Path(os.environ.get("TOPO_RESULTS", "/exp/sgreenberg/experiments/toponymy-image/results"))


def setup_env() -> None:
    """Point vtscore + HF at scratch, put the repo on sys.path."""
    _expcommon.setup_env(
        repo=REPO,
        datadir=WORK / "vts-data",
        models_dir=WORK / "vts-models",
        results=RESULTS,
        hf_home=WORK / "hf",
        neutralise=False,
    )


def ds_dir(dataset: str) -> Path:
    d = RESULTS / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
    print(f"[saved] {path}")


def load_json(path: Path):
    return json.loads(path.read_text())

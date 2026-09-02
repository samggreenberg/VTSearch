"""Filesystem roots for every VTSearch runtime artefact.

Resolved once at import time and anchored to the repository root, never to the
current working directory.  Nothing else in :mod:`vtscore.config` depends on
this module; everything under ``vtscore/`` that needs a directory derives it
from :data:`DATA_DIR` (or from a :attr:`~vtscore.config.CoreConfig.data_dir`
when one is in scope).
"""

from __future__ import annotations

import os
from pathlib import Path

# Data paths are anchored to the repository root, NOT to the current working
# directory.  Without this, starting the app from a different CWD (systemd,
# cron, dev shell) would create a fresh empty `data/` and silently lose the
# user's existing datasets, settings, and embeddings.  Override with the
# ``VTSEARCH_DATA_DIR`` env var if you need to relocate state outside the repo.
#: ``vtscore/config/paths.py`` -> ``vtscore/config`` -> ``vtscore`` -> the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ["VTSEARCH_DATA_DIR"]) if "VTSEARCH_DATA_DIR" in os.environ else _REPO_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_CACHE_DIR = (
    Path(os.environ["VTSEARCH_MODELS_DIR"]) if "VTSEARCH_MODELS_DIR" in os.environ else DATA_DIR / "models"
)

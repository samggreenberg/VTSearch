"""Low-level file I/O helpers for trainable model JSON files.

Provides path resolution, read, and write utilities used by the route
layer (``vtsearch.routes.trainable_models``) and by model-layer modules
that need to persist or inspect trainable-model data without importing
the full route blueprint.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from vtsearch.config import DATA_DIR


def get_trainable_models_dir() -> Path:
    """Return the configured trainable-models directory from settings."""
    from vtsearch.settings import get_trainable_models_dir as _get

    return _get()


#: Backward-compat alias -- prefer :func:`get_trainable_models_dir` for live value.
TRAINABLE_MODELS_DIR = DATA_DIR / "trainable_models"


def _slug(name: str) -> str:
    """Turn a human-readable name into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "model"


def _model_path(name: str) -> Path:
    return get_trainable_models_dir() / f"{_slug(name)}.json"


def _read_model(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_model(path: Path, data: dict) -> None:
    get_trainable_models_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

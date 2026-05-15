"""Low-level file I/O helpers for detector JSON files.

Provides path resolution, read, and write utilities used by the route layer
(``vtsearch.routes.detectors``) and by model-layer modules that need to
persist or inspect detector data without importing the full route blueprint.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from vtsearch.config import DATA_DIR


def get_detectors_dir() -> Path:
    """Return the configured detectors directory from settings."""
    from vtsearch.settings import get_detectors_dir as _get

    return _get()


#: Default location used by tests that bypass settings.
DETECTORS_DIR = DATA_DIR / "detectors"


def _slug(name: str) -> str:
    """Turn a human-readable name into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "detector"


def _detector_path(name: str) -> Path:
    return get_detectors_dir() / f"{_slug(name)}.json"


def _read_detector(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_detector(path: Path, data: dict) -> None:
    get_detectors_dir().mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

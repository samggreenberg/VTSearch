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

from vtscore.config import DATA_DIR


def get_detectors_dir() -> Path:
    """Return the configured detectors directory.

    Reads from ``CoreConfig.from_settings()`` rather than ``vtsearch.settings``
    directly so this module stays library-clean (see Phase 2 of
    ``docs/plans/extract-library.md``).  The classmethod still consults the
    app's settings layer today; after Phase 8 it moves to an app-side shim
    and library callers pass a ``CoreConfig`` explicitly.
    """
    from vtscore.config import CoreConfig  # noqa: PLC0415

    return CoreConfig.from_settings().detectors_dir


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

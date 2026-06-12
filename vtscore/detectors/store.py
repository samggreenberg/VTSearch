"""Low-level file I/O helpers for detector JSON files.

Provides path resolution, read, and write utilities used by the route layer
(``vtsearch.routes.detectors``) and by model-layer modules that need to
persist or inspect detector data without importing the full route blueprint.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import uuid
from pathlib import Path

from vtscore.config import DATA_DIR


def get_detectors_dir() -> Path:
    """Return the configured detectors directory.

    Reads from ``CoreConfig.from_settings()`` rather than ``vtsearch.settings``
    directly so this module stays library-clean (see Phase 2 of
    ``../docs/architecture.md``).  The classmethod still consults the
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
    # Per-writer unique tmp suffix so two threads (or two processes) racing to
    # overwrite the same detector file can't truncate each other's in-flight
    # tmp file or chase one that was already renamed away (which surfaced as
    # ``FileNotFoundError: '<name>.json.tmp' -> '<name>.json'`` from
    # ``os.replace``).  Mirrors ``vtsearch.settings._atomic_write`` and
    # ``vtscore.io.atomic_write_text``.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best-effort tmp cleanup so a failed write doesn't leak a
        # half-written tmp file next to the destination.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise

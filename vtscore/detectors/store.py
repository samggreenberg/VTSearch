"""Low-level file I/O helpers for detector JSON files.

Provides path resolution, read, and write utilities used by the route layer
(``vtsearch.routes.detectors``) and by model-layer modules that need to
persist or inspect detector data without importing the full route blueprint.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from vtscore.config import DATA_DIR

#: Cap on the slug length so ``<slug>.json`` and its longer atomic-write
#: sibling ``<slug>.json.<pid>.<uuid>.tmp`` (worst case ~50 extra chars) stay
#: under the common filesystem ``NAME_MAX`` of 255.  The schema layer already
#: caps user-supplied names well below this (see
#: ``vtsearch.schemas.detectors.MAX_NAME_LENGTH``); this is the last-line
#: backstop for names that reach the store by other paths (combine, CLI,
#: direct library use) so a write can never raise ``[Errno 36] File name too
#: long`` — an uncaught ``OSError`` whose message leaks the absolute path.
_MAX_SLUG_LENGTH = 190


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
    """Turn a human-readable name into a filesystem-safe slug.

    Truncated to ``_MAX_SLUG_LENGTH`` so the derived filename stays under the
    filesystem ``NAME_MAX``.  When truncation drops characters, an 8-hex
    content hash of the full name is appended so two long names sharing a
    prefix don't collide onto the same file.
    """
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "detector"
    if len(slug) > _MAX_SLUG_LENGTH:
        digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).hexdigest()
        slug = f"{slug[: _MAX_SLUG_LENGTH - len(digest) - 1]}_{digest}"
    return slug


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
    # ``os.replace``).  Mirrors ``vtsearch.settings_store._atomic_write`` and
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


def save_detector(
    name: str,
    labelset: Any,
    *,
    media_type: str = "",
    embedder_type: str = "",
    extra: dict | None = None,
) -> Path:
    """Write *labelset* to ``<detectors_dir>/<slug>.json`` and return the path.

    The library-facing entry point for detector persistence: a caller that
    has built a :class:`~vtscore.datasets.labelset.LabelSet` (from votes, from
    an external label store, from its own pipeline) hands it here and gets a
    file the app, the CLI, and :func:`load_detector` all read.

    Only origins and labels are written — never embeddings and never model
    weights (see the "No Persisted Vectors or MLPs" invariant).  An existing
    file for the same slug is replaced wholesale; use :func:`load_detector`
    plus ``LabelSet.merge`` first when you mean to add to one.

    Args:
        name: Human-readable detector name.  Slugified for the filename.
        labelset: The :class:`~vtscore.datasets.labelset.LabelSet` to persist.
        media_type: Media type the detector scores (``"audio"``, ``"image"``, …).
        embedder_type: Locked embedder type (``"semantic"`` /
            ``"patch_semantic"`` / ``"structural"``); ``""`` leaves the
            detector typeless, resolved from its labels on load.
        extra: Additional top-level keys to merge into the file (e.g.
            ``{"text_query": "dog barking"}``).  Never overrides the keys
            above.

    Returns:
        The path written.
    """
    data: dict = dict(extra or {})
    data.update(
        {
            "name": name,
            "media_type": media_type,
            "labelset": labelset.to_dict(),
        }
    )
    if embedder_type:
        data["embedder_type"] = embedder_type
    path = _detector_path(name)
    _write_detector(path, data)
    return path


def load_detector(name: str) -> dict | None:
    """Return the parsed detector JSON for *name*, or ``None`` if absent.

    The counterpart of :func:`save_detector`.  Rebuild the labelset with
    ``LabelSet.from_dict(data["labelset"])``; the head itself is not stored
    and is re-derived from the labelset's origins (see
    :func:`vtscore.detectors.training.train_detector_from_origins`).
    """
    return _read_detector(_detector_path(name))

"""Persistent detector registry.

Maintains a JSON manifest at ``data/detector_registry.json`` that tracks every
detector the user has created.  Each entry stores enough metadata to display
the detector in the dashboard grid.

Every entry is backed by a labelset file at ``data/detectors/<name>.json``.
The MLP that scores the detector is trained on demand from the labelset and
lives only in RAM (see :class:`~vtsearch.state.DetectorContext`).

Multiple detectors can be *loaded* into memory simultaneously.  Which detector
the UI interacts with is determined per-request via the ``X-Detector-Id``
header.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from vtscore.config import DATA_DIR

logger = logging.getLogger(__name__)

REGISTRY_PATH = DATA_DIR / "detector_registry.json"

_lock = threading.RLock()

_entries: list[dict[str, Any]] | None = None

# Set of detector IDs currently loaded in memory (each has a DetectorContext).
_loaded_ids: set[str] = set()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _load() -> list[dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            text = REGISTRY_PATH.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to read detector registry: %s", exc)
    return []


def _save(entries: list[dict[str, Any]]) -> None:
    import os

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(REGISTRY_PATH))


def _ensure_loaded() -> list[dict[str, Any]]:
    global _entries
    if _entries is None:
        _entries = _load()
    return _entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_detectors() -> list[dict[str, Any]]:
    """Return summary info for all registered detectors."""
    with _lock:
        return [dict(e) for e in _ensure_loaded()]


def get_detector(detector_id: str) -> dict[str, Any] | None:
    """Return a single registry entry by *detector_id*, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == detector_id:
                return dict(entry)
    return None


def register_detector(
    *,
    name: str,
    media_type: str,
    num_training: int = 0,
    text_query: str = "",
    media_example: str = "",
    created_by: str = "default",
    embedder: str = "",
) -> dict[str, Any]:
    """Add a new detector to the registry and persist.

    Args:
        name: Display name for the dashboard.  Also the slug used to look
            up the on-disk labelset file at ``data/detectors/<name>.json``.
        media_type: ``"audio"``, ``"image"``, ``"video"``, ``"text"``, etc.
        num_training: Number of training examples (label count).
        text_query: Text-sort query associated with the detector.
        media_example: Optional path to an example media file.
        created_by: Username of the user who created this detector.
        embedder: Name of the embedder used for this detector's labels.
            Defaults to ``""`` for newly created detectors that haven't been
            trained yet; stamped automatically the first time training runs
            (see :func:`record_detector_embedder`).  Read by the smart
            preload predictor so the right model is warmed at startup
            instead of the media type's default.

    Returns:
        The newly created entry dict.
    """
    import uuid

    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "name": name,
        "media_type": media_type,
        "num_training": num_training,
        "text_query": text_query,
        "media_example": media_example,
        "created_by": created_by,
        "created_at": time.time(),
        "embedder": embedder,
    }
    with _lock:
        entries = _ensure_loaded()
        entries.append(entry)
        _save(entries)
    return entry


def unregister_detector(detector_id: str) -> bool:
    """Remove a detector from the registry. Returns ``True`` if found."""
    with _lock:
        entries = _ensure_loaded()
        for i, entry in enumerate(entries):
            if entry["id"] == detector_id:
                entries.pop(i)
                _loaded_ids.discard(detector_id)
                _save(entries)
                return True
    return False


def rename_detector(detector_id: str, new_name: str) -> bool:
    """Rename a registered detector. Returns ``True`` on success."""
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == detector_id:
                entry["name"] = new_name
                _save(entries)
                return True
    return False


def update_detector(detector_id: str, **fields: Any) -> bool:
    """Update arbitrary fields on a registered detector."""
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == detector_id:
                entry.update(fields)
                _save(entries)
                return True
    return False


def record_detector_embedder(detector_id: str, embedder_name: str) -> None:
    """Persist the embedder a detector's labels are currently embedded with.

    Called from the training paths that stamp ``DetectorContext.embedder``
    so the smart preload predictor knows which model to warm on the next
    process start.  No-ops on empty inputs or unknown detector ids; swallows
    registry write failures because losing the optimization is preferable
    to crashing a training cycle.
    """
    if not detector_id or not embedder_name:
        return
    try:
        with _lock:
            entries = _ensure_loaded()
            for entry in entries:
                if entry["id"] == detector_id:
                    if entry.get("embedder") == embedder_name:
                        return
                    entry["embedder"] = embedder_name
                    _save(entries)
                    return
    except Exception as exc:
        logger.warning("Failed to persist embedder for detector %s: %s", detector_id, exc)


def find_by_name(name: str) -> dict[str, Any] | None:
    """Return the entry whose ``name`` matches, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry.get("name") == name:
                return dict(entry)
    return None


def add_loaded_detector_id(detector_id: str) -> None:
    """Add *detector_id* to the set of loaded detectors (without changing active)."""
    with _lock:
        _loaded_ids.add(detector_id)


def remove_loaded_detector_id(detector_id: str) -> None:
    """Remove *detector_id* from the loaded set."""
    with _lock:
        _loaded_ids.discard(detector_id)


def is_detector_loaded(detector_id: str) -> bool:
    """Return ``True`` if *detector_id* is in the loaded set."""
    with _lock:
        return detector_id in _loaded_ids


def get_loaded_detector_ids() -> set[str]:
    """Return a copy of all loaded detector IDs."""
    with _lock:
        return set(_loaded_ids)


def is_find_mode() -> bool:
    """Return ``True`` if the active detector's votes are find/scoring output.

    Find mode is per-detector state (``DetectorContext.find_mode``), not a
    process global: a scoring pass on one detector must never block vote
    syncing on another, and switching to a different detector must not inherit
    the previous detector's find state.
    """
    from vtscore.state.core import _state_lock, get_active_detector_context

    with _state_lock:
        return get_active_detector_context().find_mode


def set_find_mode(enabled: bool = True) -> None:
    """Set or clear find mode on the active detector context.

    No-op when no real detector is active (the empty / request-missing
    sentinel contexts have no labelset to protect).
    """
    from vtscore.state.core import _state_lock, get_active_detector_context

    with _state_lock:
        det_ctx = get_active_detector_context()
        if det_ctx.detector_id:
            det_ctx.find_mode = enabled


def reset_for_tests() -> None:
    """Reset the in-memory cache (for test isolation)."""
    global _entries
    with _lock:
        _entries = None
        _loaded_ids.clear()

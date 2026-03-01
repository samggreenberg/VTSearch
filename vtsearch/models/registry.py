"""Persistent model registry.

Maintains a JSON manifest at ``data/model_registry.json`` that tracks every
model (detector / processor) the user has created.  Each entry stores enough
metadata to display the model in the dashboard grid.

Models come in two flavours:

* **Trainable** — backed by a labelset (``data/trainable_models/<slug>.json``).
  Shows a training-count in the "# Training" column.  Can be loaded for
  labeling and then used for Find.
* **Non-trainable** (pregen) — backed by weights (in-memory autorun detector
  or a detector JSON file on disk).  Shows "-" in the "# Training" column.

At most one model is *loaded* into memory at a time.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from vtsearch.config import DATA_DIR

logger = logging.getLogger(__name__)

REGISTRY_PATH = DATA_DIR / "model_registry.json"

_lock = threading.RLock()

_entries: list[dict[str, Any]] | None = None

# The ``id`` of the currently loaded model, or ``None``.
_loaded_id: str | None = None


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
            logger.warning("Failed to read model registry: %s", exc)
    return []


def _save(entries: list[dict[str, Any]]) -> None:
    import os

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(REGISTRY_PATH))


def _ensure_loaded() -> list[dict[str, Any]]:
    global _entries
    if _entries is None:
        _entries = _load()
    return _entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[dict[str, Any]]:
    """Return summary info for all registered models."""
    with _lock:
        return list(_ensure_loaded())


def get_model(model_id: str) -> dict[str, Any] | None:
    """Return a single registry entry by *model_id*, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == model_id:
                return dict(entry)
    return None


def register_model(
    *,
    name: str,
    media_type: str,
    trainable: bool,
    num_training: int = 0,
    detector_name: str = "",
    trainable_model_name: str = "",
    text_query: str = "",
) -> dict[str, Any]:
    """Add a new model to the registry and persist.

    Args:
        name: Display name for the dashboard.
        media_type: ``"audio"``, ``"image"``, ``"video"``, ``"paragraph"``, etc.
        trainable: Whether the model has a labelset that can be extended.
        num_training: Number of training examples (label count or "-" marker).
        detector_name: The key used in ``autorun_detectors`` (for non-trainable).
        trainable_model_name: The key used in ``data/trainable_models/`` (for trainable).
        text_query: Text-sort query associated with the model.

    Returns:
        The newly created entry dict.
    """
    import uuid

    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "name": name,
        "media_type": media_type,
        "trainable": trainable,
        "num_training": num_training,
        "detector_name": detector_name,
        "trainable_model_name": trainable_model_name,
        "text_query": text_query,
        "created_at": time.time(),
    }
    with _lock:
        entries = _ensure_loaded()
        entries.append(entry)
        _save(entries)
    return entry


def unregister_model(model_id: str) -> bool:
    """Remove a model from the registry. Returns ``True`` if found."""
    global _loaded_id
    with _lock:
        entries = _ensure_loaded()
        for i, entry in enumerate(entries):
            if entry["id"] == model_id:
                entries.pop(i)
                if _loaded_id == model_id:
                    _loaded_id = None
                _save(entries)
                return True
    return False


def rename_model(model_id: str, new_name: str) -> bool:
    """Rename a registered model. Returns ``True`` on success."""
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == model_id:
                entry["name"] = new_name
                _save(entries)
                return True
    return False


def update_model(model_id: str, **fields: Any) -> bool:
    """Update arbitrary fields on a registered model."""
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == model_id:
                entry.update(fields)
                _save(entries)
                return True
    return False


def find_by_detector_name(det_name: str) -> dict[str, Any] | None:
    """Return the entry whose ``detector_name`` matches, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry.get("detector_name") == det_name:
                return dict(entry)
    return None


def find_by_trainable_model_name(tm_name: str) -> dict[str, Any] | None:
    """Return the entry whose ``trainable_model_name`` matches, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry.get("trainable_model_name") == tm_name:
                return dict(entry)
    return None


def get_loaded_id() -> str | None:
    """Return the ID of the currently loaded model, or ``None``."""
    with _lock:
        return _loaded_id


def set_loaded_id(model_id: str | None) -> None:
    """Mark *model_id* as the currently loaded model (or ``None``)."""
    global _loaded_id
    with _lock:
        _loaded_id = model_id


def reset_for_tests() -> None:
    """Reset the in-memory cache (for test isolation)."""
    global _entries, _loaded_id
    with _lock:
        _entries = None
        _loaded_id = None

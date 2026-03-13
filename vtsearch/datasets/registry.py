"""Persistent dataset registry.

Maintains a JSON manifest at ``data/dataset_registry.json`` that tracks every
dataset the user has loaded.  Each entry stores enough metadata to display the
dataset in the dashboard grid and to re-load it from its saved ``.pkl`` file.

At most one dataset is *loaded* into memory at a time, but the registry
remembers all datasets across app restarts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from vtsearch.config import DATA_DIR

logger = logging.getLogger(__name__)

REGISTRY_PATH = DATA_DIR / "dataset_registry.json"


def get_saved_datasets_dir() -> Path:
    """Return the saved-datasets directory from settings."""
    from vtsearch.settings import get_saved_datasets_dir as _get

    return _get()


# Backward-compat alias — prefer :func:`get_saved_datasets_dir` for live value.
SAVED_DATASETS_DIR = DATA_DIR / "saved_datasets"

_lock = threading.RLock()

# In-memory cache — loaded once from disk, written back on every mutation.
_entries: list[dict[str, Any]] | None = None

# The ``id`` of the currently loaded dataset entry, or ``None``.
_loaded_id: str | None = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _load() -> list[dict[str, Any]]:
    """Read the registry from disk, returning an empty list on failure."""
    if REGISTRY_PATH.exists():
        try:
            text = REGISTRY_PATH.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to read dataset registry: %s", exc)
    return []


def _save(entries: list[dict[str, Any]]) -> None:
    """Write *entries* to disk atomically."""
    import os

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(REGISTRY_PATH))


def _ensure_loaded() -> list[dict[str, Any]]:
    """Return the in-memory cache, loading from disk on first call."""
    global _entries
    if _entries is None:
        _entries = _load()
    return _entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_datasets() -> list[dict[str, Any]]:
    """Return summary info for all registered datasets."""
    with _lock:
        return list(_ensure_loaded())


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    """Return a single registry entry by *dataset_id*, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == dataset_id:
                return dict(entry)
    return None


def register_dataset(
    *,
    name: str,
    media_type: str,
    num_items: int,
    pkl_path: str,
    origin: str = "unknown",
    source: dict[str, Any] | None = None,
    num_dupes: int = 0,
    clipper: str = "",
    embedder: str = "",
    created_by: str = "default",
    readers: list[str] | None = None,
) -> dict[str, Any]:
    """Add a new dataset to the registry and persist.

    Args:
        created_by: Username of the user who created this dataset.
        readers: List of usernames granted read access.  An empty list
            (the default) means only the creator can see the dataset.
            Include ``"*"`` to make it visible to all users.

    Returns the newly created entry (with a generated ``id``).
    """
    import uuid

    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex,
        "name": name,
        "media_type": media_type,
        "num_items": num_items,
        "num_dupes": num_dupes,
        "pkl_path": pkl_path,
        "origin": origin,
        "source": source,
        "clipper": clipper,
        "embedder": embedder,
        "created_by": created_by,
        "created_at": time.time(),
        "readers": readers or [],
    }
    with _lock:
        entries = _ensure_loaded()
        entries.append(entry)
        _save(entries)
    return entry


def unregister_dataset(dataset_id: str) -> bool:
    """Remove a dataset from the registry and delete its pkl file.

    Returns ``True`` if the dataset was found and removed.
    """
    global _loaded_id
    with _lock:
        entries = _ensure_loaded()
        for i, entry in enumerate(entries):
            if entry["id"] == dataset_id:
                pkl = Path(entry.get("pkl_path", ""))
                if pkl.is_file():
                    pkl.unlink(missing_ok=True)
                entries.pop(i)
                if _loaded_id == dataset_id:
                    _loaded_id = None
                _save(entries)
                return True
    return False


def rename_dataset(dataset_id: str, new_name: str) -> bool:
    """Rename a registered dataset. Returns ``True`` on success."""
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == dataset_id:
                entry["name"] = new_name
                _save(entries)
                return True
    return False


def update_dataset(dataset_id: str, **fields: Any) -> bool:
    """Update arbitrary fields on a registered dataset."""
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == dataset_id:
                entry.update(fields)
                _save(entries)
                return True
    return False


def get_loaded_id() -> str | None:
    """Return the ID of the currently loaded dataset, or ``None``."""
    with _lock:
        return _loaded_id


def set_loaded_id(dataset_id: str | None) -> None:
    """Mark *dataset_id* as the currently loaded dataset (or ``None``)."""
    global _loaded_id
    with _lock:
        _loaded_id = dataset_id


def find_by_pkl_path(pkl_path: str) -> dict[str, Any] | None:
    """Return the entry whose ``pkl_path`` matches, or ``None``."""
    with _lock:
        for entry in _ensure_loaded():
            if entry.get("pkl_path") == pkl_path:
                return dict(entry)
    return None


def can_user_access(dataset_id: str, username: str) -> bool:
    """Return ``True`` if *username* may view/load the dataset.

    Access is granted when any of the following hold:

    * The user is the dataset creator (``created_by``).
    * The username appears in the ``readers`` list.
    * The ``readers`` list contains the wildcard ``"*"``.
    """
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == dataset_id:
                if entry.get("created_by", "default") == username:
                    return True
                readers = entry.get("readers", [])
                return username in readers or "*" in readers
    return False


def is_owner(dataset_id: str, username: str) -> bool:
    """Return ``True`` if *username* is the creator of the dataset."""
    with _lock:
        for entry in _ensure_loaded():
            if entry["id"] == dataset_id:
                return entry.get("created_by", "default") == username
    return False


def list_datasets_for_user(username: str) -> list[dict[str, Any]]:
    """Return only datasets that *username* is allowed to see.

    A dataset is visible when the user is its creator, is listed in
    ``readers``, or ``"*"`` is in ``readers``.
    """
    with _lock:
        result = []
        for entry in _ensure_loaded():
            creator = entry.get("created_by", "default")
            readers = entry.get("readers", [])
            if creator == username or username in readers or "*" in readers:
                result.append(dict(entry))
        return result


def set_readers(dataset_id: str, readers: list[str], requesting_user: str) -> tuple[bool, str]:
    """Update the ``readers`` list.  Only the creator may call this.

    Returns ``(success, error_message)``.
    """
    with _lock:
        entries = _ensure_loaded()
        for entry in entries:
            if entry["id"] == dataset_id:
                if entry.get("created_by", "default") != requesting_user:
                    return False, "Only the dataset creator can modify readers"
                entry["readers"] = readers
                _save(entries)
                return True, ""
        return False, "Dataset not found"


def reset_for_tests() -> None:
    """Reset the in-memory cache (for test isolation)."""
    global _entries, _loaded_id
    with _lock:
        _entries = None
        _loaded_id = None

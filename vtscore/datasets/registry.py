"""Persistent dataset registry.

Maintains a JSON manifest at ``data/dataset_registry.json`` that tracks every
dataset the user has loaded.  Each entry stores enough metadata to display the
dataset in the dashboard grid and to re-load it from its saved ``.pkl`` file.

Multiple datasets may be *loaded* into memory simultaneously (each in its own
``DatasetContext``).  The registry tracks which datasets are loaded via
``_loaded_ids``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from vtscore.config import DATA_DIR
from vtscore.io import atomic_write_json, file_lock

logger = logging.getLogger(__name__)

REGISTRY_PATH = DATA_DIR / "dataset_registry.json"


def get_saved_datasets_dir() -> Path:
    """Return the saved-datasets directory.

    Routed through ``CoreConfig.from_settings()`` rather than reading
    ``vtsearch.settings`` directly so this module stays library-clean
    (see Phase 2 of ``../docs/architecture.md``).
    """
    from vtscore.config import CoreConfig  # noqa: PLC0415

    return CoreConfig.from_settings().saved_datasets_dir


# Backward-compat alias - prefer :func:`get_saved_datasets_dir` for live value.
SAVED_DATASETS_DIR = DATA_DIR / "saved_datasets"

_T = TypeVar("_T")

# Guards the process-local state below: the ``_entries`` cache and the
# ``_loaded_ids`` / ``_loading_ids`` sets.  Cross-process durability of the
# on-disk manifest is handled separately by :func:`_read_modify_write` under a
# ``file_lock``; this lock only serialises threads within one process.
_lock = threading.RLock()

# In-memory cache - loaded once from disk, refreshed from disk on every
# mutation (see :func:`_read_modify_write`).
_entries: list[dict[str, Any]] | None = None

# The set of dataset IDs that are currently loaded in memory.
_loaded_ids: set[str] = set()

# The set of dataset IDs whose background load is currently in flight.  Used
# to gate the ``.../load`` handler's check-then-act: without it two concurrent
# requests both pass the ``is_loaded`` check (the flag is only set at the *end*
# of the loader) and spawn twin loaders against the same id.
_loading_ids: set[str] = set()


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
    """Write *entries* to disk atomically (per-writer unique temp name)."""
    atomic_write_json(REGISTRY_PATH, entries)


def _ensure_loaded() -> list[dict[str, Any]]:
    """Return the in-memory cache, loading from disk on first call."""
    global _entries
    if _entries is None:
        _entries = _load()
    return _entries


def _read_modify_write(mutator: Callable[[list[dict[str, Any]]], _T]) -> _T:
    """Run *mutator* over a fresh-from-disk registry under a cross-process lock.

    Holding the ``file_lock`` while re-reading the manifest closes the
    multi-process read-modify-write race: a concurrent process (e.g. a CLI
    autodetect run against the same data dir) can't commit between our read and
    write, so a mutation always merges into the *current* on-disk state instead
    of clobbering entries a sibling registered.  ``_load`` is deliberately used
    (not the possibly-stale ``_entries`` cache) so we start from disk truth.

    *mutator* receives the fresh list, mutates it in place, and returns this
    call's result.  The list is always persisted and swapped into the in-memory
    cache, so the cache converges to disk truth on every mutation.
    """
    global _entries
    with file_lock(REGISTRY_PATH):
        entries = _load()
        result = mutator(entries)
        _save(entries)
        with _lock:
            _entries = entries
    return result


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
    embedder_types: list[str] | None = None,
    created_by: str = "default",
    readers: list[str] | None = None,
    file_type_counts: dict[str, int] | None = None,
    ingest_started_at: float | None = None,
    expires_at: float | None = None,
) -> dict[str, Any]:
    """Add a new dataset to the registry and persist.

    Args:
        created_by: Username of the user who created this dataset.
        readers: List of usernames granted read access.  An empty list
            (the default) means only the creator can see the dataset.
            Include ``"*"`` to make it visible to all users.
        file_type_counts: Mapping of file extension to count.
        ingest_started_at: Unix timestamp when ingest began.
        expires_at: Unix timestamp when the dataset expires.  ``None``
            means the dataset never expires.

    Returns the newly created entry (with a generated ``id``).
    """
    import uuid

    # The embedder *types* this dataset supplies (drives detector/dataset
    # compatibility gating without loading the dataset).  Callers that know the
    # full bound set pass it; otherwise classify the single primary embedder.
    if embedder_types is None:
        from vtscore.embedding.binding import embedder_type

        t = embedder_type(embedder)
        embedder_types = [t] if t else []

    now = time.time()
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
        "embedder_types": embedder_types,
        "created_by": created_by,
        "created_at": now,
        "readers": readers or [],
        "file_type_counts": file_type_counts or {},
        "ingest_started_at": ingest_started_at,
        "ingest_finished_at": now,
        "expires_at": expires_at,
    }
    _read_modify_write(lambda entries: entries.append(entry))

    # New entry expands the predicted-embedder set; warm anything new in the
    # background so a subsequent load is instant. Idempotent: already-loaded
    # embedders are skipped inside the worker.
    try:
        from vtscore.embedding.loader import smart_preload_in_background

        smart_preload_in_background()
    except Exception:
        pass

    return entry


def unregister_dataset(dataset_id: str) -> bool:
    """Remove a dataset from the registry and delete its pkl file plus any sidecars.

    Every file in the pkl's directory sharing its stem (e.g. an mmap embedding
    sidecar) is deleted alongside it: the stem is a random uuid
    (``ds_<uuid>``), so this can never collide with another dataset's files,
    and any current or future sidecar convention is swept without needing a
    separate registry field to track it.

    Returns ``True`` if the dataset was found and removed.
    """

    def mutate(entries: list[dict[str, Any]]) -> bool:
        for i, entry in enumerate(entries):
            if entry["id"] == dataset_id:
                pkl = Path(entry.get("pkl_path", ""))
                if pkl.is_file():
                    for sibling in pkl.parent.glob(f"{pkl.stem}.*"):
                        sibling.unlink(missing_ok=True)
                entries.pop(i)
                return True
        return False

    removed = _read_modify_write(mutate)
    with _lock:
        _loaded_ids.discard(dataset_id)
    return removed


def rename_dataset(dataset_id: str, new_name: str) -> bool:
    """Rename a registered dataset. Returns ``True`` on success."""

    def mutate(entries: list[dict[str, Any]]) -> bool:
        for entry in entries:
            if entry["id"] == dataset_id:
                entry["name"] = new_name
                return True
        return False

    return _read_modify_write(mutate)


def update_dataset(dataset_id: str, **fields: Any) -> bool:
    """Update arbitrary fields on a registered dataset."""

    def mutate(entries: list[dict[str, Any]]) -> bool:
        for entry in entries:
            if entry["id"] == dataset_id:
                entry.update(fields)
                return True
        return False

    return _read_modify_write(mutate)


def get_loaded_ids() -> set[str]:
    """Return the set of all currently loaded (in-memory) dataset IDs."""
    with _lock:
        return set(_loaded_ids)


def add_loaded_id(dataset_id: str) -> None:
    """Mark *dataset_id* as loaded in memory (without making it active)."""
    with _lock:
        _loaded_ids.add(dataset_id)


def remove_loaded_id(dataset_id: str) -> None:
    """Remove *dataset_id* from the set of loaded datasets."""
    with _lock:
        _loaded_ids.discard(dataset_id)


def is_loaded(dataset_id: str) -> bool:
    """Return ``True`` if the dataset is currently loaded in memory."""
    with _lock:
        return dataset_id in _loaded_ids


def begin_load(dataset_id: str) -> str:
    """Atomically claim the right to load *dataset_id*.

    This closes the check-then-act race in the ``.../load`` handler: the
    decision to load and the reservation happen under a single lock, so two
    concurrent requests can't both start a loader.

    Returns:
        ``"loaded"`` – already resident in memory; the caller should no-op.
        ``"in_progress"`` – another loader is already running; the caller
            should attach to the existing task instead of spawning a twin.
        ``"reserved"`` – the caller won the race and must run the load, then
            call :func:`end_load` once it settles (success or failure).
    """
    with _lock:
        if dataset_id in _loaded_ids:
            return "loaded"
        if dataset_id in _loading_ids:
            return "in_progress"
        _loading_ids.add(dataset_id)
        return "reserved"


def end_load(dataset_id: str) -> None:
    """Release the load reservation taken by :func:`begin_load`."""
    with _lock:
        _loading_ids.discard(dataset_id)


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

    def mutate(entries: list[dict[str, Any]]) -> tuple[bool, str]:
        for entry in entries:
            if entry["id"] == dataset_id:
                if entry.get("created_by", "default") != requesting_user:
                    return False, "Only the dataset creator can modify readers"
                entry["readers"] = readers
                return True, ""
        return False, "Dataset not found"

    return _read_modify_write(mutate)


def reset_for_tests() -> None:
    """Reset the in-memory cache (for test isolation)."""
    global _entries
    with _lock:
        _entries = None
        _loaded_ids.clear()
        _loading_ids.clear()

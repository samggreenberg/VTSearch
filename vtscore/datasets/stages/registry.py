"""Registry stage: persist the built dataset and migrate its context id.

Saves a context's ``medias`` to a pkl, registers it in the dataset registry,
and re-keys the in-flight context from its temporary task id to the registry
entry id (rolling the entry back if the migration fails).
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from vtscore.concurrency.progress import loading_tasks
from vtscore.datasets.loader import export_dataset_to_file
from vtscore.datasets.registry import (
    add_loaded_id as _reg_add_loaded,
    get_saved_datasets_dir,
    register_dataset as _reg_register,
    unregister_dataset as _reg_unregister,
)

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS, _origin_to_str

if TYPE_CHECKING:
    from vtscore.state import DatasetContext


def _auto_register_dataset(
    media_dict: dict,
    name: str = "",
    origin_str: str = "unknown",
    source: dict | None = None,
    clipper: str = "",
    embedder: str = "",
    created_by: str = "",
    display_name: str | None = None,
    ingest_started_at: float | None = None,
    on_stage: Callable[[str], None] | None = None,
    extra_pickle_keys: dict | None = None,
) -> dict | None:
    """Save *media_dict* as a pkl and register in the dataset registry.

    Unlike the old version, this accepts an explicit *media_dict* instead
    of reading from the global ``medias`` proxy, enabling parallel loads.

    Returns the registry entry dict on success, or ``None`` on failure/skip.
    """
    if not media_dict:
        return None

    _stage = on_stage or (lambda _msg: None)

    first = next(iter(media_dict.values()))
    media_type = first.get("media_type", "audio")
    num_items = len(media_dict)

    if not embedder:
        embedder = first.get("embedder", "")

    # The full set of embedder types this dataset binds (a v3 trio dataset
    # carries several), classified from the first media's bound embedders.
    from vtscore.embedding.binding import dataset_supplied_types
    from vtscore.embedding.media_vectors import media_embedder_names

    embedder_types = sorted(dataset_supplied_types(media_embedder_names(first)))

    if not name:
        name = display_name or origin_str or "Untitled"
        if ":" in name:
            name = name.split(":", 1)[1] or name

    # Count dupes
    num_dupes = sum(
        1
        for m in media_dict.values()
        if isinstance(m.get("origin"), dict) and m["origin"].get("importer") == "dupe_set"
    )

    # Count file types by extension
    from collections import Counter

    ext_counter: Counter[str] = Counter()
    for m in media_dict.values():
        fn = m.get("filename", "")
        if fn and "." in fn:
            ext_counter[fn.rsplit(".", 1)[-1].lower()] += 1
        else:
            ext_counter["(no extension)"] += 1
    file_type_counts = dict(ext_counter.most_common())

    ds_dir = get_saved_datasets_dir()
    ds_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = str(ds_dir / f"ds_{uuid4().hex}.pkl")
    import time as _time

    from vtscore.config import CoreConfig

    now = _time.time()
    try:
        config = CoreConfig.from_settings()
        max_age = config.dataset_max_age_days
    except RuntimeError:
        max_age = None
    expires_at = now + max_age * 86400 if max_age is not None else None

    try:
        data_bytes = export_dataset_to_file(
            media_dict,
            embedder=embedder,
            clipper=clipper,
            media_type=media_type,
            name=name,
            created_at=now,
            expires_at=expires_at,
            on_stage=on_stage,
            extra_pickle_keys=extra_pickle_keys,
        )
        _stage("Writing to disk…")
        Path(pkl_path).write_bytes(data_bytes)
        del data_bytes
    except Exception:
        traceback.print_exc()
        return None

    _stage("Registering dataset…")
    try:
        entry = _reg_register(
            name=name,
            media_type=media_type,
            num_items=num_items,
            num_dupes=num_dupes,
            pkl_path=pkl_path,
            origin=origin_str,
            source=source,
            clipper=clipper,
            embedder=embedder,
            embedder_types=embedder_types,
            created_by=created_by,
            file_type_counts=file_type_counts,
            ingest_started_at=ingest_started_at,
            expires_at=expires_at,
        )
    except Exception:
        # Registry write failed; clean up the orphaned pkl so we don't
        # leave a stale file behind with nothing pointing at it.
        traceback.print_exc()
        Path(pkl_path).unlink(missing_ok=True)
        return None
    _reg_add_loaded(entry["id"])
    return entry


def _register_and_migrate(
    ctx: DatasetContext,
    tracker,
    task_id: str,
    origin: dict,
    name: str,
    clipper: str,
    embedder: str,
    created_by: str,
    ingest_started_at: float,
) -> tuple[str, str | None]:
    """Save to registry, migrate the context from task_id to its real id.

    Returns ``(context_id, registry_entry_id)``; *context_id* is the
    (possibly migrated) context id, and *registry_entry_id* is the id of
    the newly created registry entry (or ``None`` if registration was
    skipped).  Callers should retain *registry_entry_id* so a later
    failure in the surrounding pipeline can roll the entry back.

    If the registry entry is created successfully but the subsequent
    in-memory migration steps raise, the entry is rolled back before the
    exception propagates so we never leave an orphan on disk.
    """

    # The registry save fires _on_stage roughly three times ("Saving to
    # registry…", "Serializing dataset…", "Packaging dataset…"). Report an
    # incrementing count so the finalize bar advances across that window
    # instead of sitting frozen at the slot's floor through the long pickle +
    # zip write. ``_STAGE_TOTAL`` is a rough denominator; the count is clamped
    # so an extra message can't push past it.
    _STAGE_TOTAL = 3
    _stage_n = [0]

    def _on_stage(message: str) -> None:
        # Keep the dashboard row alive through the otherwise-silent
        # serialize → write → register window (the old "frozen bar" gap).
        _stage_n[0] += 1
        tracker.update(
            "loading",
            message,
            current=min(_stage_n[0], _STAGE_TOTAL),
            total=_STAGE_TOTAL,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    # Cache the freshly-built diversity tree into the pickle so reloads skip
    # the expensive hierarchical k-means rebuild (it is re-derived only when
    # absent or stale).  Embeddings/MLPs are still never persisted - this is
    # cluster *topology* keyed by media id, re-resolved against the medias on
    # load and discarded if they no longer match.
    extra_pickle_keys: dict | None = None
    if ctx.diversity_tree is not None:
        extra_pickle_keys = {"diversity_tree": ctx.diversity_tree.to_serializable()}

    _on_stage("Saving to registry…")
    entry = _auto_register_dataset(
        ctx.medias,
        name=name,
        origin_str=_origin_to_str(origin),
        source=origin,
        clipper=clipper,
        embedder=embedder,
        created_by=created_by,
        display_name=name,
        ingest_started_at=ingest_started_at,
        on_stage=_on_stage,
        extra_pickle_keys=extra_pickle_keys,
    )
    if entry is None:
        return task_id, None
    entry_id = entry["id"]
    try:
        _migrate_context_id(task_id, entry_id)
        ctx.dataset_display_name = entry.get("name", name)
        # Associate the loading task with the real dataset ID so the
        # finished-task tick is attributed to the right dashboard row.
        loading_tasks.set_dataset_id(task_id, entry_id)
    except Exception:
        # Migration failed after the registry entry was written; roll
        # it back so the dashboard doesn't show a half-built dataset.
        try:
            _reg_unregister(entry_id)
        except Exception:
            traceback.print_exc()
        raise
    return entry_id, entry_id


def _migrate_context_id(old_id: str, new_id: str) -> None:
    """Re-key a context from *old_id* to *new_id* in the store."""
    from vtscore.state.core import _contexts, _state_lock

    with _state_lock:
        ctx = _contexts.get(old_id)
        if ctx is None:
            return
        ctx.dataset_id = new_id
        # Insert under the new key before removing the old, so the context
        # is never briefly invisible to concurrent lookups.
        _contexts[new_id] = ctx
        if old_id != new_id:
            _contexts.pop(old_id, None)

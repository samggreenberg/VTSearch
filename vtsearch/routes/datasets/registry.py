"""Blueprint for dataset-registry routes (the on-disk dataset catalog).

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

Schema-level validation failures (missing required ``name`` on rename,
missing or wrong-typed ``readers`` on the readers endpoint) surface as
422 with the standard ``errors`` envelope; handler-level rejects (not
loaded, not the creator) keep their HTTP codes (400 / 403 / 404 / 500)
with the standard ``message`` envelope. 404s are intercepted by the
app-level ``NotFound`` errorhandler in ``app.py`` and keep the legacy
``{"error": "Not Found", "request_id": ...}`` shape regardless of the
``message=`` kwarg passed to ``abort()``.

Endpoints
---------
GET    /api/datasets/registry                       List datasets visible to the user.
POST   /api/datasets/registry/<id>/load             Load a registered dataset from its pkl.
POST   /api/datasets/registry/<id>/unload           Unload a dataset from memory.
DELETE /api/datasets/registry/<id>                  Remove a dataset from the registry.
PUT    /api/datasets/registry/<id>/rename           Rename a registered dataset.
PUT    /api/datasets/registry/<id>/readers          Update the readers ACL.
GET    /api/datasets/registry/<id>/stats            Return ingest statistics.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

from flask_smorest import Blueprint, abort

from vtscore.datasets.load_pipeline import _warmup_embedder_async
from vtscore.datasets.loader import load_dataset_from_pickle
from vtscore.datasets.registry import (
    add_loaded_id as _reg_add_loaded,
    can_user_access as _reg_can_access,
    get_dataset as _reg_get,
    is_loaded as _reg_is_loaded,
    is_owner as _reg_is_owner,
    list_datasets_for_user as _reg_list_for_user,
    get_loaded_ids as _reg_loaded_ids,
    remove_loaded_id as _reg_remove_loaded,
    rename_dataset as _reg_rename,
    set_readers as _reg_set_readers,
    unregister_dataset as _reg_unregister,
    update_dataset as _reg_update,
)
from vtsearch.schemas.datasets import (
    DatasetRegistryLoadResponseSchema,
    DatasetRegistryPreloadEmbedderResponseSchema,
    DatasetRegistryReadersRequestSchema,
    DatasetRegistryReadersResponseSchema,
    DatasetRegistryRenameRequestSchema,
    DatasetRegistryRenameResponseSchema,
    DatasetRegistryStatsResponseSchema,
    DatasetsRegistryListResponseSchema,
    DatasetRegistryOkResponseSchema,
)
from vtsearch.state import (
    DatasetContext,
    collapse_duplicates,
    register_context,
    unregister_context,
)
from vtscore.concurrency.progress import CancelledError
from vtscore.concurrency.progress import loading_tasks as _loading_tasks
from vtscore.embedding.matrix import invalidate_embedding_matrix

datasets_registry_bp = Blueprint(
    "datasets_registry",
    __name__,
    description="CRUD over the registered (on-disk) dataset catalog.",
)


@datasets_registry_bp.route("/api/datasets/registry")
@datasets_registry_bp.response(200, DatasetsRegistryListResponseSchema)
def list_registered_datasets():
    """Return registered datasets visible to the current user.

    Each entry includes:
    - ``loaded``: whether the dataset is currently in memory
    """
    from vtsearch.auth import get_current_user

    entries = _reg_list_for_user(get_current_user())
    loaded_ids = _reg_loaded_ids()
    from vtscore.media import get_clipper

    for entry in entries:
        ds_id = entry["id"]
        entry["loaded"] = ds_id in loaded_ids
        entry.setdefault("num_dupes", 0)
        entry.setdefault("embedder", "")
        entry.setdefault("readers", [])
        # Resolve clipper name to display name; default clippers show as "-"
        raw_clipper = entry.get("clipper", "")
        if raw_clipper:
            if raw_clipper.endswith("_default"):
                entry["clipper"] = "-"
            else:
                try:
                    entry["clipper"] = get_clipper(raw_clipper).display_name
                except KeyError:
                    pass  # keep raw name if clipper not found
    return {"datasets": entries}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/load", methods=["POST"])
@datasets_registry_bp.response(200, DatasetRegistryLoadResponseSchema)
@datasets_registry_bp.alt_response(403, description="Access denied for the current user.")
@datasets_registry_bp.alt_response(404, description="Dataset not found, or saved pkl file is missing.")
def load_registered_dataset(dataset_id: str):  # noqa: C901
    """Load a registered dataset from its saved pkl file.

    If the dataset is already loaded in memory, it is simply activated
    (made the current UI-facing dataset) without re-reading the pkl.
    """
    from vtsearch.auth import get_current_user
    from vtscore.concurrency.progress import clear_thread_progress, set_thread_progress
    from vtsearch.state import build_diversity_tree_for_context

    entry = _reg_get(dataset_id)
    if entry is None:
        abort(404, message="Dataset not found in registry")

    if not _reg_can_access(dataset_id, get_current_user()):
        abort(403, message="Access denied")

    # If already loaded in memory, nothing to do.
    if _reg_is_loaded(dataset_id):
        return {"ok": True, "message": "Dataset already loaded", "task_id": ""}

    pkl_path = entry.get("pkl_path", "")
    if not pkl_path or not Path(pkl_path).is_file():
        abort(404, message=f"Saved dataset file not found: {pkl_path}")

    _LOAD_STEPS = 2  # read pickle + process items, build diversity index

    # Create a per-task tracker for this load operation.
    task_id = f"_regload_{dataset_id[:8]}"
    tracker = _loading_tasks.create_task(
        task_id,
        entry.get("name", dataset_id),
        dataset_id=dataset_id,
        media_type=entry.get("media_type", ""),
        embedder=entry.get("embedder", ""),
    )
    tracker.update("loading", "Loading dataset from file...", step=1, total_steps=_LOAD_STEPS)

    def _pickle_progress(status, message, current, total):
        tracker.check_cancelled()
        tracker.update(status, message, current, total, step=1, total_steps=_LOAD_STEPS)

    # Snapshot the user that triggered this load so background settings
    # writes (e.g. autopilot toggles) and settings_source syncs resolve
    # to the right per-user file.
    _request_user = get_current_user()

    def load_task():
        from vtsearch.auth import thread_user

        with thread_user(_request_user):
            try:
                tracker.update("loading", "Preparing…", 0, 0, step=1, total_steps=_LOAD_STEPS)
                # Create a fresh context for this dataset (don't activate yet).
                ctx = DatasetContext(dataset_id)
                register_context(ctx)
                gc.collect()

                # Set thread-local progress for the pickle loader.
                set_thread_progress(
                    lambda status, msg="", cur=0, tot=0: tracker.update(
                        status, msg, cur, tot, step=1, total_steps=_LOAD_STEPS
                    )
                )
                try:
                    load_dataset_from_pickle(Path(pkl_path), ctx.medias, on_progress=_pickle_progress)
                finally:
                    clear_thread_progress()

                tracker.check_cancelled()

                def _dedup_progress(current: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update(
                        "loading",
                        "Removing duplicates…",
                        current=current,
                        total=total,
                        step=2,
                        total_steps=_LOAD_STEPS,
                    )

                _dedup_progress(0, 0)
                collapse_duplicates(ctx.medias, on_progress=_dedup_progress)
                invalidate_embedding_matrix(ctx)

                def _diversity_progress(current: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update(
                        "loading",
                        "Building diversity index…",
                        current=current,
                        total=total,
                        step=2,
                        total_steps=_LOAD_STEPS,
                    )

                _diversity_progress(0, 0)
                build_diversity_tree_for_context(ctx, on_progress=_diversity_progress)
                _reg_add_loaded(dataset_id)
                # Update item count and dupe count in case they changed
                num_dupes = sum(
                    1
                    for m in ctx.medias.values()
                    if isinstance(m.get("origin"), dict) and m["origin"].get("importer") == "dupe_set"
                )
                _reg_update(dataset_id, num_items=len(ctx.medias), num_dupes=num_dupes)
                ctx.dataset_display_name = entry.get("name", "")

                # Embedder warm-up runs fire-and-forget so the dashboard row
                # goes green immediately; text sort waits behind its own
                # progress bar on first use if the model isn't ready yet.
                _warmup_embedder_async(ctx.medias)
            except CancelledError:
                unregister_context(dataset_id)
                _reg_remove_loaded(dataset_id)
                gc.collect()
                tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
            except MemoryError:
                unregister_context(dataset_id)
                _reg_remove_loaded(dataset_id)
                gc.collect()
                tracker.update(
                    "idle", "", 0, 0, error="Out of memory — dataset too large.", step=None, total_steps=None
                )
            except Exception as e:
                import traceback as _tb

                _tb.print_exc()
                unregister_context(dataset_id)
                _reg_remove_loaded(dataset_id)
                error_msg = str(e) or repr(e) or "Unknown error during dataset loading"
                tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
            finally:
                clear_thread_progress()
                _loading_tasks.mark_finished(task_id)

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/unload", methods=["POST"])
@datasets_registry_bp.response(200, DatasetRegistryOkResponseSchema)
@datasets_registry_bp.alt_response(400, description="Dataset is not currently loaded.")
@datasets_registry_bp.alt_response(403, description="Only the dataset creator can unload it.")
def unload_registered_dataset(dataset_id: str):
    """Unload a specific dataset from memory.

    The dataset's context is removed, freeing its RAM.  If it was the
    active dataset, the active pointer is cleared.
    """
    from vtsearch.auth import get_current_user

    if not _reg_is_owner(dataset_id, get_current_user()):
        abort(403, message="Only the dataset creator can unload it")
    if not _reg_is_loaded(dataset_id):
        abort(400, message="This dataset is not currently loaded")
    unregister_context(dataset_id)
    _reg_remove_loaded(dataset_id)
    return {"ok": True}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/preload-embedder", methods=["POST"])
@datasets_registry_bp.response(200, DatasetRegistryPreloadEmbedderResponseSchema)
@datasets_registry_bp.alt_response(403, description="Access denied for the current user.")
@datasets_registry_bp.alt_response(404, description="Dataset not found.")
def preload_dataset_embedder(dataset_id: str):
    """Warm this dataset's embedder in a background daemon thread.

    Called by the dashboard when the user selects a dataset row so that
    the embedder is ready by the time they click Train. Idempotent: the
    background worker exits immediately if the embedder is already in
    memory. Returns ``embedder`` set to the resolved embedder name (or
    ``""`` if no embedder could be resolved for this dataset's media
    type).
    """
    from vtsearch.auth import get_current_user
    from vtscore.embedding.loader import preload_embedder_for_dataset

    if _reg_get(dataset_id) is None:
        abort(404, message="Dataset not found")
    if not _reg_can_access(dataset_id, get_current_user()):
        abort(403, message="Access denied")

    emb_name = preload_embedder_for_dataset(dataset_id)
    return {"ok": True, "embedder": emb_name}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>", methods=["DELETE"])
@datasets_registry_bp.response(200, DatasetRegistryOkResponseSchema)
@datasets_registry_bp.alt_response(403, description="Only the dataset creator can delete it.")
@datasets_registry_bp.alt_response(404, description="Dataset not found.")
def delete_registered_dataset(dataset_id: str):
    """Remove a dataset from the registry and delete its pkl file."""
    from vtsearch.auth import get_current_user

    if not _reg_is_owner(dataset_id, get_current_user()):
        abort(403, message="Only the dataset creator can delete it")

    # If loaded in memory, unload its context.
    if _reg_is_loaded(dataset_id):
        unregister_context(dataset_id)
        _reg_remove_loaded(dataset_id)
    ok = _reg_unregister(dataset_id)
    if not ok:
        abort(404, message="Dataset not found")
    return {"ok": True}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/rename", methods=["PUT"])
@datasets_registry_bp.arguments(DatasetRegistryRenameRequestSchema)
@datasets_registry_bp.response(200, DatasetRegistryRenameResponseSchema)
@datasets_registry_bp.alt_response(403, description="Only the dataset creator can rename it.")
@datasets_registry_bp.alt_response(404, description="Dataset not found.")
def rename_registered_dataset(body: dict, dataset_id: str):
    """Rename a registered dataset."""
    from vtsearch.auth import get_current_user

    if not _reg_is_owner(dataset_id, get_current_user()):
        abort(403, message="Only the dataset creator can rename it")

    new_name = body["name"].strip()
    if not new_name:
        abort(400, message="name is required")
    ok = _reg_rename(dataset_id, new_name)
    if not ok:
        abort(404, message="Dataset not found")
    # Also update display name if this dataset is loaded
    if _reg_is_loaded(dataset_id):
        from vtsearch.state import get_context

        ctx = get_context(dataset_id)
        if ctx is not None:
            ctx.dataset_display_name = new_name
    return {"ok": True, "name": new_name}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/readers", methods=["PUT"])
@datasets_registry_bp.arguments(DatasetRegistryReadersRequestSchema)
@datasets_registry_bp.response(200, DatasetRegistryReadersResponseSchema)
@datasets_registry_bp.alt_response(403, description="Only the dataset creator can update readers.")
@datasets_registry_bp.alt_response(404, description="Dataset not found.")
def update_dataset_readers(body: dict, dataset_id: str):
    """Update the readers list for a dataset.  Only the creator may call this.

    Body: ``{"readers": ["alice", "bob"]}``
    Use ``["*"]`` to make the dataset public to all users.
    """
    from vtsearch.auth import get_current_user

    readers = body["readers"]
    ok, err = _reg_set_readers(dataset_id, readers, get_current_user())
    if not ok:
        status = 403 if "creator" in err else 404
        abort(status, message=err)
    return {"ok": True, "readers": readers}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/stats")
@datasets_registry_bp.response(200, DatasetRegistryStatsResponseSchema)
@datasets_registry_bp.alt_response(404, description="Dataset not found.")
def get_dataset_stats(dataset_id: str):
    """Return ingest statistics for a registered dataset."""
    from vtscore.media import get_clipper

    entry = _reg_get(dataset_id)
    if entry is None:
        abort(404, message="Dataset not found")

    raw_clipper = entry.get("clipper", "") or ""
    if not raw_clipper or raw_clipper.endswith("_default"):
        clipper_display = ""
    else:
        try:
            clipper_display = get_clipper(raw_clipper).display_name
        except KeyError:
            clipper_display = raw_clipper

    return {
        "num_items": entry.get("num_items", 0),
        "num_dupes": entry.get("num_dupes", 0),
        "file_type_counts": entry.get("file_type_counts", {}),
        "ingest_started_at": entry.get("ingest_started_at"),
        "ingest_finished_at": entry.get("ingest_finished_at"),
        "origin": entry.get("origin", ""),
        "source": entry.get("source") or {},
        "clipper": clipper_display,
        "embedder": entry.get("embedder", ""),
    }

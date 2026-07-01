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
POST   /api/datasets/registry/<id>/diversity-tree   Build the diversity index on demand (large datasets).
POST   /api/datasets/registry/<id>/unload           Unload a dataset from memory.
DELETE /api/datasets/registry/<id>                  Remove a dataset from the registry.
PUT    /api/datasets/registry/<id>/rename           Rename a registered dataset.
PUT    /api/datasets/registry/<id>/readers          Update the readers ACL.
GET    /api/datasets/registry/<id>/stats            Return ingest statistics.
"""

from __future__ import annotations

import gc
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

    import time

    entries = _reg_list_for_user(get_current_user())

    now = time.time()
    expired_ids = [e["id"] for e in entries if e.get("expires_at") is not None and now > e["expires_at"]]
    for eid in expired_ids:
        _reg_unregister(eid)
    entries = [e for e in entries if e["id"] not in expired_ids]

    loaded_ids = _reg_loaded_ids()
    from vtscore.media import get_clipper

    for entry in entries:
        ds_id = entry["id"]
        entry["loaded"] = ds_id in loaded_ids
        entry.setdefault("num_dupes", 0)
        entry.setdefault("embedder", "")
        entry.setdefault("readers", [])
        # The embedder types this dataset supplies, for the detector/dataset
        # compatibility gate.  Fall back to classifying the single primary
        # embedder for legacy entries registered before the field existed.
        if not entry.get("embedder_types"):
            from vtscore.embedding.binding import embedder_type

            t = embedder_type(entry.get("embedder", ""))
            entry["embedder_types"] = [t] if t else []
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
    from vtsearch.state import (
        build_diversity_tree_for_context,
        restore_diversity_tree_from_cache,
        should_auto_build_diversity_tree,
    )

    entry = _reg_get(dataset_id)
    if entry is None:
        abort(404, message="Dataset not found in registry")

    expires_at = entry.get("expires_at")
    if expires_at is not None:
        import time

        if time.time() > expires_at:
            _reg_unregister(dataset_id)
            abort(410, message="Dataset has expired and has been removed.")

    if not _reg_can_access(dataset_id, get_current_user()):
        abort(403, message="Access denied")

    # If already loaded in memory, nothing to do.
    if _reg_is_loaded(dataset_id):
        return {"ok": True, "message": "Dataset already loaded", "task_id": ""}

    pkl_path = entry.get("pkl_path", "")
    if not pkl_path or not Path(pkl_path).is_file():
        abort(404, message=f"Saved dataset file not found: {pkl_path}")

    _LOAD_STEPS = 2  # step 1: load items (read + dedup); step 2: diversity index

    # Create a per-task tracker for this load operation.
    task_id = f"_regload_{dataset_id[:8]}"
    tracker = _loading_tasks.create_task(
        task_id,
        entry.get("name", dataset_id),
        dataset_id=dataset_id,
        media_type=entry.get("media_type", ""),
        embedder=entry.get("embedder", ""),
    )
    # Pace the unified bar against the real phase split. Step 1 (pickle read +
    # convert + the near-instant exact-dedup) is seconds at most; step 2 is the
    # diversity index, which is instant when the cached tree restores but a
    # minutes-long hierarchical-k-means rebuild when it doesn't. Weighting step 2
    # as the dominant slice keeps a rebuild advancing the bar across its whole
    # span instead of the old equal split, where the instant dedup drove step 2
    # to ~100% and the bar then sat frozen there through the entire rebuild.
    tracker.set_step_weights([0.15, 0.85])
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
                    cached_diversity_tree = load_dataset_from_pickle(
                        Path(pkl_path), ctx.medias, on_progress=_pickle_progress
                    )
                finally:
                    clear_thread_progress()

                tracker.check_cancelled()

                # Exact-MD5 dedup is part of loading (step 1): the pickle was
                # already deduped at import, so on reload this is a near-instant
                # no-op — keeping it out of step 2 stops it from pre-filling the
                # diversity slice and freezing the bar (see set_step_weights).
                def _dedup_progress(current: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update(
                        "loading",
                        "Removing duplicates…",
                        current=current,
                        total=total,
                        step=1,
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

                # Reuse the diversity tree cached in the pickle when its vector
                # set still matches the (post-dedup) medias; only rebuild the
                # hierarchical k-means from scratch when there is no usable
                # cache (older pickles, or a media set that shifted on load).
                # Past the auto-build threshold a missing cache is left absent -
                # the build would cost minutes/GBs and the user can trigger it
                # on demand via the diversity-tree endpoint.
                if not restore_diversity_tree_from_cache(ctx, cached_diversity_tree):
                    if should_auto_build_diversity_tree(len(ctx.medias)):
                        _diversity_progress(0, 0)
                        build_diversity_tree_for_context(ctx, on_progress=_diversity_progress)
                # Fill the diversity slice to completion in every branch (cache
                # restore, rebuild, or deferred-above-threshold) so the unified
                # bar reaches 100% cleanly instead of stalling at the step-1
                # boundary when the tree restored without emitting any progress.
                tracker.update("loading", "Finalizing…", current=1, total=1, step=2, total_steps=_LOAD_STEPS)
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
                tracker.update("idle", "", 0, 0, error="Out of memory: dataset too large.", step=None, total_steps=None)
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

    from vtsearch.threading import spawn

    spawn(load_task, name=f"ds-load-{dataset_id[:8]}")
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_registry_bp.route("/api/datasets/registry/<dataset_id>/diversity-tree", methods=["POST"])
@datasets_registry_bp.response(200, DatasetRegistryLoadResponseSchema)
@datasets_registry_bp.alt_response(400, description="Dataset is not currently loaded.")
@datasets_registry_bp.alt_response(403, description="Access denied for the current user.")
@datasets_registry_bp.alt_response(404, description="Dataset not found.")
def build_dataset_diversity_tree(dataset_id: str):
    """Build the diversity index for a loaded dataset in a background thread.

    Datasets past ``should_auto_build_diversity_tree`` skip the automatic build
    at load time so they load fast; this endpoint lets the user trigger that
    build on demand.  The dataset must already be loaded in memory.  Progress
    is reported via ``/api/progress`` under the returned ``task_id``; the build
    is cancellable.  Calling it on a dataset that already has a tree rebuilds
    it from scratch.
    """
    from vtsearch.auth import get_current_user
    from vtsearch.state import (
        build_diversity_tree_for_context,
        get_active_detector_context,
        get_context,
        resync_diversity_tree_to_detector,
    )
    from vtscore.state.core import _state_lock

    entry = _reg_get(dataset_id)
    if entry is None:
        abort(404, message="Dataset not found in registry")
    if not _reg_can_access(dataset_id, get_current_user()):
        abort(403, message="Access denied")
    if not _reg_is_loaded(dataset_id):
        abort(400, message="Load the dataset before building its diversity index")

    ctx = get_context(dataset_id)
    if ctx is None:
        abort(400, message="This dataset is not currently loaded")

    task_id = f"_divtree_{dataset_id[:8]}"
    tracker = _loading_tasks.create_task(
        task_id,
        entry.get("name", dataset_id),
        dataset_id=dataset_id,
        media_type=entry.get("media_type", ""),
        embedder=entry.get("embedder", ""),
    )
    tracker.update("loading", "Building diversity index…", 0, 0, step=1, total_steps=1)

    _request_user = get_current_user()

    def build_task():
        from vtsearch.auth import thread_user

        with thread_user(_request_user):
            try:

                def _progress(current: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update("loading", "Building diversity index…", current, total, step=1, total_steps=1)

                _progress(0, 0)
                build_diversity_tree_for_context(ctx, on_progress=_progress)

                # Replay the active detector's votes into the fresh tree when
                # they belong to this dataset, so seen-state is correct right
                # away (mirrors the resync the detector-sync path performs).
                det_ctx = get_active_detector_context()
                if det_ctx is not None and getattr(det_ctx, "votes_dataset_id", None) == dataset_id:
                    with _state_lock:
                        resync_diversity_tree_to_detector(ctx, det_ctx)
            except CancelledError:
                ctx.diversity_tree = None
                tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
            except Exception as e:
                import traceback as _tb

                _tb.print_exc()
                error_msg = str(e) or repr(e) or "Unknown error building diversity index"
                tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
            finally:
                _loading_tasks.mark_finished(task_id)

    from vtsearch.threading import spawn

    spawn(build_task, name=f"divtree-{dataset_id[:8]}")
    return {"ok": True, "message": "Diversity index build started", "task_id": str(task_id)}


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

"""Detector-registry routes that load a detector into memory and take it out again.

Endpoints
---------
POST   /api/detectors/registry/load                   Load a detector into memory.
POST   /api/detectors/registry/<id>/unload            Unload a detector from memory.
DELETE /api/detectors/registry/<id>                   Remove a detector from the registry.
POST   /api/detectors/cancel/<task_id>                Cancel a load task.
"""

from __future__ import annotations

import logging

from flask_smorest import abort

from vtsearch.auth import get_current_user
from vtscore.state.core import DetectorNotLoadedError
from vtscore.detectors.embedder_type import detector_embedder_type_from_data
from vtscore.detectors.store import (
    _detector_path,
    _read_detector,
)
from vtscore.detectors.labelset_ops import (
    restore_labels_from_detector as _restore_labels_from_detector,
    sync_labels_to_loaded_detector,
)
from vtscore.detectors.media_seeding import seed_good_votes_from_examples as _seed_good_votes_from_examples
from vtsearch.routes.detectors.registry._common import detectors_registry_bp
from vtsearch.schemas.detectors import (
    DetectorCancelResponseSchema,
    DetectorRegistryDeleteResponseSchema,
    DetectorRegistryLoadRequestSchema,
    DetectorRegistryLoadResponseSchema,
    DetectorRegistryUnloadResponseSchema,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/load
# ---------------------------------------------------------------------------


_LOAD_STEPS = 3  # restore labels, seed examples, train head
#: Timing task name; its step names and shipped weights live in
#: :data:`vtscore.timing.tasks.TASKS`.
_DETECTOR_LOAD_TASK = "detector_load"


def _run_detector_load_task(
    *,
    detector_id: str,
    det_ctx,
    thread_ds_ctx,
    det_name: str,
    tracker,
    task_id: str,
) -> None:
    """Background worker for :func:`load_detector_route`.

    Restores labels, seeds example votes, embeds + trains the head for the
    detector, then marks it loaded. Cancellation and errors tear the
    just-built context back down and surface on the task tracker. Lifted out
    of the route's closure so the route itself stays simple; the captured
    values are passed explicitly.
    """
    from vtscore.concurrency.progress import CancelledError, detector_loading_tasks
    from vtscore.detectors.registry import (
        add_loaded_detector_id,
        end_detector_load,
        remove_loaded_detector_id,
    )
    from vtscore.state.core import (
        register_detector_context,
        thread_dataset_context,
        thread_detector_context,
    )

    try:
        with thread_dataset_context(thread_ds_ctx), thread_detector_context(det_ctx):
            try:
                if det_name:
                    tracker.check_cancelled()
                    tracker.update(
                        "loading",
                        "Restoring labels…",
                        0,
                        0,
                        step=1,
                        total_steps=_LOAD_STEPS,
                    )
                    # Level the detector with its labelset source first, so
                    # labels kept there (the detector's backup, when this
                    # server lost its own copy) are restored before they are
                    # read below.  The file wins a disagreement; a source that
                    # is behind gets the file's labels pushed to it (#4913).
                    from vtscore.labels.sync import reconcile_labelset_source

                    reconcile_labelset_source(det_name, detector_id=detector_id)
                    det_data = _read_detector(_detector_path(det_name))
                    if det_data:
                        # The detector's locked embedder type (immutable;
                        # set at create time): load it so the label-embed /
                        # score / invalidation paths resolve the concrete
                        # embedder of that type the active dataset supplies.
                        # Empty for a legacy detector with neither type nor
                        # primary → routing falls back to the precedence.
                        det_ctx.embedder_type = detector_embedder_type_from_data(det_data)
                        _restore_labels_from_detector(det_data)

                        tracker.check_cancelled()
                        tracker.update(
                            "loading",
                            "Seeding examples…",
                            0,
                            0,
                            step=2,
                            total_steps=_LOAD_STEPS,
                        )
                        _seed_good_votes_from_examples(det_data.get("examples", []))

                        tracker.check_cancelled()
                        tracker.update(
                            "loading",
                            "Embedding labels…",
                            0,
                            0,
                            step=3,
                            total_steps=_LOAD_STEPS,
                        )

                        from vtscore.datasets.labelset import LabelSet
                        from vtscore.detectors.labelset_ops import train_from_labelset
                        from vtsearch.state import snapshot_medias as _snap_medias

                        labelset = LabelSet.from_dict(det_data.get("labelset") or {})
                        media_type = det_data.get("media_type", "") or ""
                        snap = _snap_medias()

                        det_ctx.labelset_good_count = sum(1 for el in labelset.elements if el.label == "good")
                        det_ctx.labelset_bad_count = sum(1 for el in labelset.elements if el.label == "bad")
                        # Cache the parsed labelset so before_request's rehydrate
                        # hook and learned_sort don't re-read the JSON file.
                        det_ctx.cached_labelset = labelset
                        det_ctx.cached_labelset_media_type = media_type
                        try:
                            det_ctx.cached_labelset_mtime = _detector_path(det_name).stat().st_mtime
                        except OSError:
                            det_ctx.cached_labelset_mtime = 0.0

                        def _embed_progress(name: str, done: int, total: int) -> None:
                            tracker.check_cancelled()
                            tracker.update(
                                "loading",
                                "Embedding labels…",
                                done,
                                total,
                                step=3,
                                total_steps=_LOAD_STEPS,
                            )

                        from vtscore.detectors.query_head import detector_text_query

                        train_from_labelset(
                            det_ctx,
                            labelset,
                            media_type=media_type,
                            snap=snap,
                            on_progress=_embed_progress,
                            # Under the quota a typed-query detector is its query's set (#4758).
                            text_query=detector_text_query(det_data),
                        )

                # Stamp the dataset whose medias the cid-keyed vote dicts were
                # derived against, so before_request's rehydrate hook can detect
                # subsequent dataset switches and re-derive against the new
                # dataset's medias.
                det_ctx.votes_dataset_id = thread_ds_ctx.dataset_id
                # Publish the fully-populated context, THEN flip the loaded flag,
                # so there is never a window where the detector is marked loaded
                # but its context is absent or half-filled in the global store.
                register_detector_context(det_ctx)
                add_loaded_detector_id(detector_id)
                tracker.update("idle", "", 0, 0, step=None, total_steps=None)
            except CancelledError:
                from vtsearch.state import unregister_detector_context as _unreg

                _unreg(detector_id)
                remove_loaded_detector_id(detector_id)
                tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
            except Exception as e:
                logger.exception("Detector load for %s failed", detector_id)
                from vtsearch.state import unregister_detector_context as _unreg

                _unreg(detector_id)
                remove_loaded_detector_id(detector_id)
                error_msg = str(e) or repr(e) or "Unknown error during detector loading"
                tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
    finally:
        end_detector_load(detector_id)
        detector_loading_tasks.mark_finished(task_id)


def _unload_active_detector() -> dict:
    """Tear down the active detector context (the ``detector_id=None`` path).

    Used by :func:`load_detector_route` when the caller passes a null/omitted
    ``detector_id`` to unload without loading another detector. Returns the
    route's success payload.
    """
    from vtscore.detectors.registry import remove_loaded_detector_id
    from vtsearch.state import get_active_detector_context

    try:
        det_ctx = get_active_detector_context()
    except DetectorNotLoadedError:
        # The request named a detector that isn't in memory (or no longer
        # exists): there is nothing to tear down (issue #4086).
        return {"ok": True, "labels_restored": 0, "examples_seeded": 0}
    prev_id = det_ctx.detector_id if det_ctx.detector_id else None
    if prev_id:
        from vtsearch.state import unregister_detector_context

        # Find mode lived on this detector's context (per-detector), so
        # tearing the context down clears it; no separate global to reset.
        unregister_detector_context(prev_id)
        remove_loaded_detector_id(prev_id)
    return {"ok": True, "labels_restored": 0, "examples_seeded": 0}


def _reembed_or_ack_loaded_detector(detector_id: str, entry: dict) -> dict:
    """Handle a load request for a detector already resident in memory.

    Its cached label embeddings may have been built against a different embedder
    than the currently-active dataset uses (e.g. the user switched from a SigLIP
    image dataset to a CLIP one). Re-embed the labels in that case so head
    training mixes only same-space vectors; the invalidation itself happens
    inside ``populate_label_embeddings``, this just surfaces the work via a
    progress task instead of letting it run lazily inside the next request.
    Otherwise it's a no-op acknowledgement.
    """
    from vtscore.detectors.embedder_sync import maybe_start_label_reembed
    from vtsearch.state import get_active_detector_context
    from vtsearch.threading import spawn

    det_ctx_existing = get_active_detector_context()
    if det_ctx_existing.detector_id == detector_id:
        task_id = maybe_start_label_reembed(det_ctx_existing, entry, spawn=spawn)
        if task_id is not None:
            return {"ok": True, "message": "Re-embedding labels", "task_id": task_id}
    return {"ok": True, "labels_restored": 0, "examples_seeded": 0}


def _start_detector_load(detector_id: str, entry: dict, thread_ds_ctx) -> dict:
    """Spawn the background load for a freshly-reserved detector.

    Called by :func:`load_detector_route` after ``begin_detector_load``
    returned ``"reserved"``: the caller owns the load reservation, and this
    helper hands it off to the worker thread.  Until ``spawn`` succeeds, only
    this request can release the reservation and retire the task row, so any
    failure here must do that cleanup itself — skipping it left the detector
    stuck "in progress" forever: every retry attached to a worker that didn't
    exist, Cancel had nothing to cancel, and only an app restart recovered
    (issue #3139).
    """
    from vtscore import timing
    from vtscore.concurrency.progress import detector_loading_tasks
    from vtscore.detectors.registry import end_detector_load
    from vtsearch.state import DetectorContext

    task_id = f"_detload_{detector_id}"
    try:
        # Build the context but do NOT register it yet: the worker publishes it
        # into the global store only after labels/embeddings/head are populated,
        # so no concurrent request can observe a torn, empty-then-filling
        # detector.
        det_ctx = DetectorContext(
            detector_id,
            name=entry.get("name", ""),
            media_type=entry.get("media_type", ""),
        )

        det_media_type = entry.get("media_type", "")

        tracker = detector_loading_tasks.create_task(
            task_id,
            entry.get("name", detector_id),
            detector_id=detector_id,
            media_type=det_media_type,
            step_weights=timing.step_weights(_DETECTOR_LOAD_TASK, media_type=det_media_type),
        )
        tracker.update("loading", "Preparing…", 0, 0, step=1, total_steps=_LOAD_STEPS)

        det_name = entry.get("name", "")

        def load_task():
            _run_detector_load_task(
                detector_id=detector_id,
                det_ctx=det_ctx,
                thread_ds_ctx=thread_ds_ctx,
                det_name=det_name,
                tracker=tracker,
                task_id=task_id,
            )

        from vtsearch.threading import spawn

        spawn(load_task, name=f"det-load-{detector_id[:8]}")
    except Exception:
        logger.exception("Detector load for %s failed before the worker started", detector_id)
        end_detector_load(detector_id)
        leaked_tracker = detector_loading_tasks.get_tracker(task_id)
        if leaked_tracker is not None:
            leaked_tracker.update("idle", "", 0, 0, error="Detector load failed to start", step=None, total_steps=None)
            detector_loading_tasks.mark_finished(task_id)
        raise
    return {
        "ok": True,
        "message": "Loading started",
        "task_id": str(task_id),
    }


@detectors_registry_bp.route("/api/detectors/registry/load", methods=["POST"])
@detectors_registry_bp.arguments(DetectorRegistryLoadRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryLoadResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def load_detector_route(body: dict):
    """Load a detector into memory and make it active.

    Pass ``detector_id=null`` (or omit the field) to unload the active
    detector without loading another one.
    """
    from vtscore.detectors.registry import (
        begin_detector_load,
        can_user_access_detector,
        get_detector,
    )
    from vtsearch.state import (
        bad_votes,
        good_votes,
    )

    detector_id = body.get("detector_id")

    if detector_id is not None:
        entry = get_detector(detector_id)
        if entry is None:
            abort(404, message="Detector not found")
        if not can_user_access_detector(detector_id, get_current_user()):
            abort(403, message="You do not have access to this detector")

    # Flush the outgoing detector's votes -- when there is one.  A tab still
    # open on a deleted detector sends its id as ``X-Detector-Id``, and the
    # proxies raise ``DetectorNotLoadedError`` for it; letting that escape
    # refused the very load that would move the tab off the dead detector
    # (issue #4086).  A detector that isn't in memory has no votes to flush.
    try:
        has_votes = bool(good_votes or bad_votes)
    except DetectorNotLoadedError:
        has_votes = False
    if has_votes:
        sync_labels_to_loaded_detector()

    if detector_id is None:
        return _unload_active_detector()

    from vtsearch.state import get_active_context

    # Snapshot the dataset context BEFORE reserving the load.  When the
    # request's ``X-Dataset-Id`` names a registered-but-unloaded dataset
    # (typical right after an app restart, while the dataset is still being
    # re-loaded from its pickle), ``get_active_context`` raises
    # ``DatasetNotLoadedError`` → a clean 409.  Resolving it after
    # ``begin_detector_load`` leaked the reservation and a forever-"Preparing"
    # task row that only an app restart could clear (issue #3139).
    thread_ds_ctx = get_active_context()

    # Atomically decide our role, closing the check-then-act race where two
    # concurrent loads both saw the detector unloaded (the loaded flag is only
    # set at the end of the loader) and spawned twin loaders.
    load_role = begin_detector_load(detector_id)
    if load_role == "in_progress":
        return {
            "ok": True,
            "message": "Detector load already in progress",
            "task_id": f"_detload_{detector_id}",
        }

    if load_role == "loaded":
        return _reembed_or_ack_loaded_detector(detector_id, entry)

    return _start_detector_load(detector_id, entry, thread_ds_ctx)


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/unload
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/unload", methods=["POST"])
@detectors_registry_bp.response(200, DetectorRegistryUnloadResponseSchema)
@detectors_registry_bp.alt_response(400, description="Detector is not loaded.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def unload_detector_route(detector_id: str):
    """Unload a detector from memory (frees its DetectorContext)."""
    from vtscore.detectors.registry import get_detector, is_detector_loaded, remove_loaded_detector_id
    from vtsearch.state import (
        bad_votes,
        get_active_detector_context,
        good_votes,
        unregister_detector_context,
    )

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not is_detector_loaded(detector_id):
        abort(400, message="Detector is not loaded")

    det_ctx = get_active_detector_context()
    if det_ctx.detector_id == detector_id and (good_votes or bad_votes):
        sync_labels_to_loaded_detector()
        # The sync queues its write (issue #3853); an unloaded detector is
        # something another process may read next, so land it now.
        from vtscore.detectors.store import flush_detector_writes

        flush_detector_writes(_detector_path(entry.get("name", "")))
    # The Evidence baseline a learned sort left waiting (#5120), for the same reason.
    from vtscore.detectors.evidence_baseline import flush_baseline_draw

    flush_baseline_draw(detector_id)

    unregister_detector_context(detector_id)
    remove_loaded_detector_id(detector_id)

    return {"ok": True, "message": "Detector unloaded"}


# ---------------------------------------------------------------------------
# DELETE /api/detectors/registry/<detector_id>
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>", methods=["DELETE"])
@detectors_registry_bp.response(200, DetectorRegistryDeleteResponseSchema)
@detectors_registry_bp.alt_response(403, description="Only the detector creator can delete it.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def delete_registered_detector(detector_id: str):
    """Remove a detector from the registry, including its labelset file."""
    from vtscore.detectors.registry import get_detector, is_detector_owner, unregister_detector

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not is_detector_owner(detector_id, get_current_user()):
        abort(403, message="Only the detector creator can delete it")

    try:
        det_name = entry.get("name", "")
        if det_name:
            det_path = _detector_path(det_name)
            # A queued labelset write landing after the unlink would
            # resurrect the file (issue #3853); drop it first.
            from vtscore.detectors.store import discard_pending_detector_write

            discard_pending_detector_write(det_path)
            if det_path.exists():
                det_path.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to delete detector file for %s", detector_id)

    try:
        from vtscore.detectors.registry import is_detector_loaded, remove_loaded_detector_id
        from vtsearch.state import unregister_detector_context

        if is_detector_loaded(detector_id):
            unregister_detector_context(detector_id)
            remove_loaded_detector_id(detector_id)
    except Exception:
        logger.exception("Failed to unregister detector context for %s", detector_id)

    # Drop AutoFind flag if set.
    try:
        from vtsearch.settings import remove_autofind_detector

        remove_autofind_detector(entry.get("name", ""))
    except Exception:
        logger.exception("Failed to drop AutoFind flag for %s", detector_id)

    unregister_detector(detector_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# POST /api/detectors/cancel/<task_id>
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/cancel/<task_id>", methods=["POST"])
@detectors_registry_bp.response(200, DetectorCancelResponseSchema)
@detectors_registry_bp.alt_response(404, description="Task not found.")
def cancel_detector_loading_task(task_id: str):
    """Cancel a specific detector loading task."""
    from vtscore.concurrency.progress import detector_loading_tasks

    ok = detector_loading_tasks.cancel_task(task_id)
    if not ok:
        abort(404, message="Task not found")
    return {"ok": True}

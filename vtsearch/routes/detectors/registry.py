"""Blueprint for detector-registry routes (the in-memory detector catalog).

Every registered detector is backed by a labelset file on disk and an MLP
that lives only in :class:`~vtsearch.state.DetectorContext` once the user
loads the detector.

Endpoints
---------
GET    /api/detectors/registry                        List registered detectors.
POST   /api/detectors/registry                        Register a new detector.
POST   /api/detectors/registry/from-labelset/<imp>    Seed a new detector from a label importer.
POST   /api/detectors/registry/load                   Load a detector into memory.
POST   /api/detectors/registry/<id>/unload            Unload a detector from memory.
DELETE /api/detectors/registry/<id>                   Remove a detector from the registry.
PUT    /api/detectors/registry/<id>/rename            Rename a registered detector.
POST   /api/detectors/registry/<id>/labelset-source/move-file
                                                      Move an orphaned labelset file after a rename.
PUT    /api/detectors/registry/<id>/autofind           Toggle the detector's Auto-Find flag.
POST   /api/detectors/cancel/<task_id>                Cancel a load task.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``, except for ``POST /from-labelset/<importer>``, which
takes plugin-dependent fields and stays on plain Flask (see
``docs/plans/openapi-schema.md`` *Open questions / Plugin field endpoints*).
"""

from __future__ import annotations

import logging
import time

from flask import jsonify
from flask_smorest import Blueprint, abort

from vtsearch.auth import get_current_user
from vtscore.detectors.store import (
    _detector_path,
    _read_detector,
    _write_detector,
)
from vtscore.detectors.label_restoration import (
    restore_labels_from_detector as _restore_labels_from_detector,
)
from vtscore.detectors.label_sync import sync_labels_to_loaded_detector
from vtscore.detectors.media_seeding import seed_good_votes_from_examples as _seed_good_votes_from_examples
from vtsearch.schemas.detectors import (
    DetectorBrowsePositivesReleaseResponseSchema,
    DetectorBrowsePositivesResponseSchema,
    DetectorCancelResponseSchema,
    DetectorLabelsetMoveRequestSchema,
    DetectorLabelsetMoveResponseSchema,
    DetectorRegistryAutofindRequestSchema,
    DetectorRegistryAutofindResponseSchema,
    DetectorRegistryCreateRequestSchema,
    DetectorRegistryCreateResponseSchema,
    DetectorRegistryDeleteResponseSchema,
    DetectorRegistryListResponseSchema,
    DetectorRegistryLoadRequestSchema,
    DetectorRegistryLoadResponseSchema,
    DetectorRegistryReadersRequestSchema,
    DetectorRegistryReadersResponseSchema,
    DetectorRegistryRenameRequestSchema,
    DetectorRegistryRenameResponseSchema,
    DetectorRegistryStatsResponseSchema,
    DetectorRegistryUnloadResponseSchema,
)

logger = logging.getLogger(__name__)

detectors_registry_bp = Blueprint(
    "detectors_registry",
    __name__,
    description="Register, load, unload, rename, and toggle Auto-Find on detectors.",
)


# ---------------------------------------------------------------------------
# GET /api/detectors/registry
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry")
@detectors_registry_bp.response(200, DetectorRegistryListResponseSchema)
def list_registered_detectors():
    """Return detectors visible to the current user, with loaded/Auto-Find flags.

    Detectors are user-shared like datasets: each entry is the creator's plus
    anyone the creator added to ``readers`` (or everyone via ``"*"``). The
    response also carries ``created_by``, ``readers``, and ``is_owner`` so the
    dashboard can render the access column and gate the security button.
    """
    from vtscore.detectors.registry import get_loaded_detector_ids, list_detectors_for_user
    from vtsearch.settings import get_autofind_detectors

    from vtscore.state.core import get_detector_context

    current_user = get_current_user()
    entries = list_detectors_for_user(current_user)
    loaded_ids = get_loaded_detector_ids()
    autofind_names = set(get_autofind_detectors())
    for entry in entries:
        did = entry["id"]
        entry["loaded"] = did in loaded_ids
        entry["autofind"] = entry.get("name", "") in autofind_names
        entry.setdefault("last_trained_at", None)
        entry.setdefault("created_by", "default")
        entry.setdefault("readers", [])
        entry["is_owner"] = entry.get("created_by", "default") == current_user
        entry["detector_loaded"] = did in loaded_ids
        # Expose the loaded detector's recorded embedder so the frontend
        # can detect a cross-embedder switch and trigger a label re-embed
        # via /api/detectors/registry/load.  Loaded contexts always win
        # because they reflect the live cache state; for unloaded
        # detectors fall back to the registry's persisted embedder (which
        # is stamped the first time training runs against a dataset).
        if entry["detector_loaded"]:
            ctx = get_detector_context(did)
            ctx_emb = ctx.embedder if ctx is not None else ""
            entry["embedder"] = ctx_emb or entry.get("embedder", "") or ""
        else:
            entry["embedder"] = entry.get("embedder", "") or ""
    return {"detectors": entries}


# ---------------------------------------------------------------------------
# POST /api/detectors/registry
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry", methods=["POST"])
@detectors_registry_bp.arguments(DetectorRegistryCreateRequestSchema)
@detectors_registry_bp.response(201, DetectorRegistryCreateResponseSchema)
@detectors_registry_bp.alt_response(400, description="Empty name after stripping, or media_type is 'any'.")
def register_detector_route(body: dict):
    """Register a new detector in the detector registry."""
    from vtscore.detectors.registry import register_detector

    name = body["name"].strip()
    media_type = body["media_type"].strip()
    text_query = body["text_query"]
    media_example = body["media_example"]

    if not name:
        abort(400, message="name is required")
    if not media_type or media_type == "any":
        abort(400, message="media_type is required (must be a specific type, not 'any')")

    det_path = _detector_path(name)
    if not det_path.exists():
        examples = body.get("examples") or []
        if not examples and text_query:
            examples = [{"type": "text", "value": text_query}]
        if not examples and media_example:
            examples = [{"type": "media", "value": media_example}]
        detector_data = {
            "name": name,
            "text_query": text_query,
            "media_example": media_example,
            "media_type": media_type,
            "examples": examples,
            "created_at": time.time(),
            "labelset": {"labels": []},
        }
        _write_detector(det_path, detector_data)

    entry = register_detector(
        name=name,
        media_type=media_type,
        text_query=text_query,
        media_example=media_example,
        created_by=get_current_user(),
    )
    return {"ok": True, "detector": entry}


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/from-labelset/<importer_name>
#
# Plugin-field route: body shape depends on the importer plugin and isn't
# described in the OpenAPI spec.  Runtime validation goes through
# :func:`validate_plugin_args` (per-plugin schema built from the importer's
# :attr:`fields`), so missing required fields / invalid select values
# raise 422.  See ``docs/plans/openapi-schema.md`` (Resolved questions /
# Plugin field endpoints).
# ---------------------------------------------------------------------------


@detectors_registry_bp.route(
    "/api/detectors/registry/from-labelset/<importer_name>",
    methods=["POST"],
)
def register_detector_from_labelset(importer_name: str):  # noqa: C901
    """Create a detector seeded with labels from a label importer.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    from vtscore.datasets.ingest import _media_type_from_origin
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtscore.labels.importers import get_label_importer, list_label_importers
    from vtscore.detectors.registry import register_detector, update_detector
    from vtsearch.routes._shared import (
        get_plugin_or_404,
        run_plugin_or_error,
        validate_plugin_args,
    )

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(importer, extra_keys=("name",))

    # ``name`` is a pass-through key (not a declared plugin field) but is
    # required by this route.  ``validate_plugin_args`` only keeps the
    # keys we list in ``extra_keys``, so the route is in charge of
    # enforcing presence.
    name = str(field_values.pop("name", "") or "").strip()
    if not name:
        abort(422, message="Validation error", errors={"json": {"name": ["Missing data for required field."]}})

    det_path = _detector_path(name)
    if det_path.exists():
        abort(409, message=f"A detector named '{name}' already exists")

    label_entries, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err
    if not isinstance(label_entries, list):
        return jsonify({"error": "Importer did not return a list of label dicts."}), 500

    elements: list[LabeledElement] = []
    detected_types: set[str] = set()
    applied = 0
    skipped = 0
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        origin = entry.get("origin")
        if isinstance(origin, dict):
            mt = _media_type_from_origin(origin)
            if mt:
                detected_types.add(mt)
        elements.append(LabeledElement.from_dict(entry))
        applied += 1

    if not detected_types:
        return jsonify(
            {
                "error": (
                    "Could not infer media type from the imported labels; none of "
                    "the entries carry origin information with a detectable type. "
                    "Re-export the labels with origin metadata, or use a different "
                    "importer."
                ),
            }
        ), 400
    if len(detected_types) > 1:
        return jsonify(
            {
                "error": (
                    f"Imported labels span multiple media types: {sorted(detected_types)}. "
                    "A detector must be for a single media type."
                ),
            }
        ), 400

    media_type = next(iter(detected_types))
    labelset = LabelSet(elements)

    detector_data = {
        "name": name,
        "text_query": "",
        "media_example": "",
        "media_type": media_type,
        "examples": [],
        "created_at": time.time(),
        "labelset": labelset.to_dict(),
    }
    _write_detector(det_path, detector_data)

    entry = register_detector(
        name=name,
        media_type=media_type,
        num_training=len(labelset),
        created_by=get_current_user(),
    )
    update_detector(entry["id"], last_trained_at=time.time())
    entry["num_training"] = len(labelset)
    entry["last_trained_at"] = time.time()

    return jsonify(
        {
            "ok": True,
            "detector": entry,
            "applied": applied,
            "skipped": skipped,
            "num_labels": len(labelset),
        }
    ), 201


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/load
# ---------------------------------------------------------------------------


def _active_dataset_embedder_name() -> str:
    """Return the embedder name recorded on the active dataset's medias, or ``""``."""
    from vtsearch.state import snapshot_medias

    snap = snapshot_medias()
    if not snap:
        return ""
    first = next(iter(snap.values()), {})
    return first.get("embedder", "") or ""


def _embedder_display_name(embedder_name: str) -> str:
    """Return a human-friendly name for *embedder_name*, falling back to the id."""
    if not embedder_name:
        return ""
    from vtscore.media import get_embedder

    try:
        return get_embedder(embedder_name).display_name or embedder_name
    except KeyError:
        return embedder_name


def _maybe_start_label_reembed(det_ctx, entry: dict) -> str | None:
    """Fire a re-embed task when the active dataset uses a different embedder.

    Returns the task id when work was started, or ``None`` when the cache is
    already aligned (same embedder, empty dataset, no labelset cached, etc.)
    so the caller can return the synchronous fast-path.
    """
    new_embedder = _active_dataset_embedder_name()
    if not new_embedder or not det_ctx.embedder or new_embedder == det_ctx.embedder:
        return None

    labelset = det_ctx.cached_labelset
    if labelset is None or not labelset.elements:
        # Nothing to re-embed; just update the stamp so we don't re-enter
        # this branch on every subsequent switch.
        det_ctx.embedder = new_embedder
        return None

    from vtscore.concurrency.progress import CancelledError, detector_loading_tasks
    from vtsearch.state import get_active_context
    from vtscore.state.core import thread_dataset_context, thread_detector_context

    _thread_ds_ctx = get_active_context()
    media_type = det_ctx.cached_labelset_media_type or entry.get("media_type", "") or ""
    display = _embedder_display_name(new_embedder)
    base_msg = f"Re-resolving labels for {display}…" if display else "Re-resolving labels…"

    task_id = f"_detreembed_{det_ctx.detector_id[:8]}"
    tracker = detector_loading_tasks.create_task(
        task_id,
        entry.get("name", det_ctx.detector_id),
        detector_id=det_ctx.detector_id,
        media_type=media_type,
        embedder=new_embedder,
    )
    tracker.update("loading", base_msg, 0, 0, step=1, total_steps=1)

    def reembed_task():
        from vtscore.detectors.labelset_training import train_from_labelset

        try:
            with thread_dataset_context(_thread_ds_ctx), thread_detector_context(det_ctx):

                def _embed_progress(name: str, done: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update("loading", base_msg, done, total, step=1, total_steps=1)

                # ``populate_label_embeddings`` clears the cache when it detects
                # the embedder change, so this rebuilds against the new embedder
                # from scratch. The stamp on ``det_ctx.embedder`` is updated
                # inside ``populate_label_embeddings``.
                from vtsearch.state import snapshot_medias

                try:
                    train_from_labelset(
                        det_ctx,
                        labelset,
                        media_type=media_type,
                        snap=snapshot_medias(),
                        on_progress=_embed_progress,
                    )
                    tracker.update("idle", "", 0, 0, step=None, total_steps=None)
                except CancelledError:
                    tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
                except Exception as e:
                    import traceback as _tb

                    _tb.print_exc()
                    error_msg = str(e) or repr(e) or "Unknown error during label re-embedding"
                    tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            detector_loading_tasks.mark_finished(task_id)

    from vtsearch.threading import spawn

    spawn(reembed_task, name=f"det-reembed-{det_ctx.detector_id[:8]}")
    return task_id


@detectors_registry_bp.route("/api/detectors/registry/load", methods=["POST"])
@detectors_registry_bp.arguments(DetectorRegistryLoadRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryLoadResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def load_detector_route(body: dict):  # noqa: C901
    """Load a detector into memory and make it active.

    Pass ``detector_id=null`` (or omit the field) to unload the active
    detector without loading another one.
    """
    from vtscore.detectors.registry import (
        can_user_access_detector,
        get_detector,
        is_detector_loaded,
    )
    from vtsearch.state import (
        DetectorContext,
        bad_votes,
        get_active_detector_context,
        good_votes,
        register_detector_context,
    )

    detector_id = body.get("detector_id")

    if detector_id is not None:
        entry = get_detector(detector_id)
        if entry is None:
            abort(404, message="Detector not found")
        if not can_user_access_detector(detector_id, get_current_user()):
            abort(403, message="You do not have access to this detector")

    if good_votes or bad_votes:
        sync_labels_to_loaded_detector()

    if detector_id is None:
        from vtscore.detectors.registry import remove_loaded_detector_id

        det_ctx = get_active_detector_context()
        prev_id = det_ctx.detector_id if det_ctx.detector_id else None
        if prev_id:
            from vtsearch.state import unregister_detector_context

            # Find mode lived on this detector's context (per-detector), so
            # tearing the context down clears it; no separate global to reset.
            unregister_detector_context(prev_id)
            remove_loaded_detector_id(prev_id)
        return {"ok": True, "labels_restored": 0, "examples_seeded": 0}

    if is_detector_loaded(detector_id):
        # Detector is already in memory, but its cached label embeddings may
        # have been built against a different embedder than the one the
        # currently-active dataset uses (e.g. user switched from an image
        # dataset embedded with SigLIP to one embedded with CLIP). Re-embed
        # the labels in that case so MLP training mixes only same-space
        # vectors. The cache invalidation itself happens inside
        # ``populate_label_embeddings``; this branch just makes the work
        # visible via a progress task instead of letting it run lazily
        # inside the next vote or learned-sort request.
        det_ctx_existing = get_active_detector_context()
        if det_ctx_existing.detector_id == detector_id:
            task_id = _maybe_start_label_reembed(det_ctx_existing, entry)
            if task_id is not None:
                return {
                    "ok": True,
                    "message": "Re-embedding labels",
                    "task_id": task_id,
                }
        return {"ok": True, "labels_restored": 0, "examples_seeded": 0}

    from vtscore.concurrency.progress import CancelledError, detector_loading_tasks

    det_ctx = DetectorContext(
        detector_id,
        name=entry.get("name", ""),
        media_type=entry.get("media_type", ""),
    )
    register_detector_context(det_ctx)

    _LOAD_STEPS = 3  # restore labels, seed examples, train MLP
    task_id = f"_detload_{detector_id[:8]}"
    tracker = detector_loading_tasks.create_task(
        task_id,
        entry.get("name", detector_id),
        detector_id=detector_id,
        media_type=entry.get("media_type", ""),
    )
    tracker.update("loading", "Preparing…", 0, 0, step=1, total_steps=_LOAD_STEPS)

    det_name = entry.get("name", "")

    from vtsearch.state import get_active_context

    _thread_ds_ctx = get_active_context()

    def load_task():
        from vtscore.detectors.registry import add_loaded_detector_id, remove_loaded_detector_id
        from vtscore.state.core import thread_dataset_context, thread_detector_context

        try:
            with thread_dataset_context(_thread_ds_ctx), thread_detector_context(det_ctx):
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
                        det_data = _read_detector(_detector_path(det_name))
                        if det_data:
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
                            from vtscore.detectors.labelset_training import train_from_labelset
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

                            train_from_labelset(
                                det_ctx,
                                labelset,
                                media_type=media_type,
                                snap=snap,
                                on_progress=_embed_progress,
                            )

                    # Stamp the dataset whose medias the cid-keyed vote dicts were
                    # derived against, so before_request's rehydrate hook can detect
                    # subsequent dataset switches and re-derive against the new
                    # dataset's medias.
                    det_ctx.votes_dataset_id = _thread_ds_ctx.dataset_id
                    add_loaded_detector_id(detector_id)
                    tracker.update("idle", "", 0, 0, step=None, total_steps=None)
                except CancelledError:
                    from vtsearch.state import unregister_detector_context as _unreg

                    _unreg(detector_id)
                    remove_loaded_detector_id(detector_id)
                    tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
                except Exception as e:
                    import traceback as _tb

                    _tb.print_exc()
                    from vtsearch.state import unregister_detector_context as _unreg

                    _unreg(detector_id)
                    remove_loaded_detector_id(detector_id)
                    error_msg = str(e) or repr(e) or "Unknown error during detector loading"
                    tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            detector_loading_tasks.mark_finished(task_id)

    from vtsearch.threading import spawn

    spawn(load_task, name=f"det-load-{detector_id[:8]}")
    return {
        "ok": True,
        "message": "Loading started",
        "task_id": str(task_id),
    }


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

    # Drop Auto-Find flag if set.
    try:
        from vtsearch.settings import remove_autofind_detector

        remove_autofind_detector(entry.get("name", ""))
    except Exception:
        logger.exception("Failed to drop Auto-Find flag for %s", detector_id)

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


# ---------------------------------------------------------------------------
# PUT /api/detectors/registry/<detector_id>/rename
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/rename", methods=["PUT"])
@detectors_registry_bp.arguments(DetectorRegistryRenameRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryRenameResponseSchema)
@detectors_registry_bp.alt_response(400, description="Empty name after stripping.")
@detectors_registry_bp.alt_response(403, description="Only the detector creator can rename it.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def rename_registered_detector(body: dict, detector_id: str):
    """Rename a registered detector and its on-disk labelset file."""
    from vtscore.detectors.labelset_rename import detect_pending_labelset_move
    from vtscore.detectors.registry import get_detector, is_detector_owner, rename_detector
    from vtscore.state.core import get_detector_context

    new_name = body["name"].strip()
    if not new_name:
        abort(400, message="name is required")

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not is_detector_owner(detector_id, get_current_user()):
        abort(403, message="Only the detector creator can rename it")

    pending_move: dict[str, str] | None = None
    old_name = entry.get("name", "")
    if old_name and old_name != new_name:
        old_path = _detector_path(old_name)
        det_data = _read_detector(old_path)
        if det_data:
            new_path = _detector_path(new_name)
            det_data["name"] = new_name
            _write_detector(new_path, det_data)
            if new_path != old_path:
                old_path.unlink(missing_ok=True)

        # Rename Auto-Find entry if present.
        try:
            from vtsearch.settings import get_autofind_detectors, set_autofind_detectors

            current = get_autofind_detectors()
            if old_name in current:
                current = [new_name if n == old_name else n for n in current]
                set_autofind_detectors(current)
        except Exception:
            logger.exception("Failed to rename Auto-Find entry for %s", detector_id)

        # Update the loaded in-memory context so future syncs use the new
        # name in {detector_name} template substitution, and detect any
        # orphaned labelset file the rename leaves behind.
        ctx = get_detector_context(detector_id)
        if ctx is not None:
            pending_move = detect_pending_labelset_move(
                ctx.labelset_source,
                detector_id=detector_id,
                old_name=old_name,
                new_name=new_name,
            )
            ctx.name = new_name

    rename_detector(detector_id, new_name)
    return {"ok": True, "name": new_name, "pending_labelset_move": pending_move}


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/labelset-source/move-file
# ---------------------------------------------------------------------------


@detectors_registry_bp.route(
    "/api/detectors/registry/<detector_id>/labelset-source/move-file",
    methods=["POST"],
)
@detectors_registry_bp.arguments(DetectorLabelsetMoveRequestSchema)
@detectors_registry_bp.response(200, DetectorLabelsetMoveResponseSchema)
@detectors_registry_bp.alt_response(400, description="Invalid path (e.g. traversal outside allowed base).")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
@detectors_registry_bp.alt_response(409, description="Destination already exists.")
def move_labelset_source_file(body: dict, detector_id: str):
    """Move an orphaned labelset file after a detector rename.

    Called by the frontend when the user confirms the *Move existing
    labelset file?* prompt that surfaces after a rename leaves the file
    at the OLD template-resolved path on disk.
    """
    from vtscore.detectors.labelset_rename import move_labelset_file
    from vtscore.detectors.registry import get_detector

    if get_detector(detector_id) is None:
        abort(404, message="Detector not found")

    old_path = body["old_path"]
    new_path = body["new_path"]
    try:
        moved = move_labelset_file(old_path, new_path)
    except FileExistsError as exc:
        abort(409, message=str(exc))
    except ValueError as exc:
        abort(400, message=str(exc))

    return {
        "ok": True,
        "moved": moved,
        "old_path": old_path,
        "new_path": new_path,
    }


# ---------------------------------------------------------------------------
# PUT /api/detectors/registry/<detector_id>/autofind
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/autofind", methods=["PUT"])
@detectors_registry_bp.arguments(DetectorRegistryAutofindRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryAutofindResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
@detectors_registry_bp.alt_response(500, description="Detector has no associated name.")
def set_detector_autofind(body: dict, detector_id: str):
    """Toggle the calling user's Auto-Find flag for a registered detector.

    The flag is stored per-user under ``autofind_detectors`` (see
    :func:`vtsearch.settings.add_autofind_detector`), so each user curates their
    own Auto-Find list. The CLI's ``--autodetect`` flow reads the running
    user's list (the built-in ``default`` user falls back to the server
    settings file). The user must be able to access the detector to flag it.
    """
    from vtscore.detectors.registry import can_user_access_detector, get_detector
    from vtsearch.settings import (
        add_autofind_detector,
        remove_autofind_detector,
    )

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not can_user_access_detector(detector_id, get_current_user()):
        abort(403, message="You do not have access to this detector")

    flag = body["autofind"]

    name = entry.get("name", "")
    if not name:
        abort(500, message="Detector has no name")

    if flag:
        add_autofind_detector(name)
    else:
        remove_autofind_detector(name)
    return {"ok": True, "autofind": flag}


# ---------------------------------------------------------------------------
# PUT /api/detectors/registry/<detector_id>/readers
# ---------------------------------------------------------------------------


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/readers", methods=["PUT"])
@detectors_registry_bp.arguments(DetectorRegistryReadersRequestSchema)
@detectors_registry_bp.response(200, DetectorRegistryReadersResponseSchema)
@detectors_registry_bp.alt_response(403, description="Only the detector creator can update readers.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def update_detector_readers(body: dict, detector_id: str):
    """Update a detector's access list. Only the creator may call this.

    Body: ``{"readers": ["alice", "bob"]}``. Use ``["*"]`` to make the
    detector visible to all users. Mirrors the dataset readers endpoint.
    """
    from vtscore.detectors.registry import set_detector_readers

    readers = body["readers"]
    ok, err = set_detector_readers(detector_id, readers, get_current_user())
    if not ok:
        status = 403 if "creator" in err else 404
        abort(status, message=err)
    return {"ok": True, "readers": readers}


# ---------------------------------------------------------------------------
# GET /api/detectors/registry/<detector_id>/stats
# ---------------------------------------------------------------------------


def _resolved_positive_count(elements) -> int:
    """How many of *elements* (positives) resolve into the active dataset.

    Builds the media lookup once and resolves every element against it, so
    this is one pass rather than one rebuild per element. Returns 0 when no
    dataset is loaded.
    """
    from vtsearch.state import build_media_lookup, resolve_media_ids, snapshot_medias

    snap = snapshot_medias()
    if not snap:
        return 0
    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)
    count = 0
    for el in elements:
        if resolve_media_ids(el.to_dict(), origin_lookup, md5_lookup, name_lookup):
            count += 1
    return count


def _active_dataset_name() -> str:
    """Display name of the active dataset, or ``""`` when none is loaded."""
    from vtsearch.state import get_active_context

    ds_id = get_active_context().dataset_id
    if not ds_id:
        return ""
    from vtscore.datasets.registry import get_dataset

    entry = get_dataset(ds_id)
    return (entry or {}).get("name", "") if entry else ""


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/stats")
@detectors_registry_bp.response(200, DetectorRegistryStatsResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def get_detector_stats(detector_id: str):
    """Return labelset composition and provenance for a registered detector.

    Counts/metadata only — no embeddings or MLP weights are read or
    returned (the labelset file is the canonical persisted form). The
    ``num_positive_resolved`` / ``active_dataset_name`` pair reports how
    much of the positive set the Browse button could currently project.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.registry import (
        can_user_access_detector,
        get_detector,
        get_loaded_detector_ids,
    )
    from vtscore.media import get_clipper
    from vtscore.state.core import get_detector_context
    from vtsearch.settings import get_autofind_detectors

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not can_user_access_detector(detector_id, get_current_user()):
        abort(403, message="You do not have access to this detector")

    name = entry.get("name", "") or ""
    data = _read_detector(_detector_path(name)) or {}

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    positives = [el for el in labelset.elements if el.label == "good"]
    negatives = [el for el in labelset.elements if el.label == "bad"]

    # Embedder: the loaded context's live value wins (it reflects the space
    # the labels are currently resolved against); otherwise the registry's
    # persisted stamp.
    if detector_id in get_loaded_detector_ids():
        ctx = get_detector_context(detector_id)
        embedder = (ctx.embedder if ctx is not None else "") or entry.get("embedder", "") or ""
    else:
        embedder = entry.get("embedder", "") or ""

    input_spec = data.get("input_spec") or {}
    raw_clipper = (input_spec.get("clipper", "") if isinstance(input_spec, dict) else "") or ""
    if not raw_clipper or raw_clipper.endswith("_default"):
        clipper_display = ""
    else:
        try:
            clipper_display = get_clipper(raw_clipper).display_name
        except KeyError:
            clipper_display = raw_clipper

    return {
        "name": name,
        "media_type": entry.get("media_type", "") or "",
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_total": len(labelset.elements),
        "num_positive_resolved": _resolved_positive_count(positives),
        "active_dataset_name": _active_dataset_name(),
        "embedder": embedder,
        "text_query": data.get("text_query", "") or "",
        "media_example": data.get("media_example", "") or "",
        "clipper": clipper_display,
        "created_at": entry.get("created_at"),
        "last_trained_at": entry.get("last_trained_at"),
        "created_by": entry.get("created_by", "default") or "default",
        "readers": entry.get("readers", []) or [],
        "autofind": name in set(get_autofind_detectors()),
    }


# ---------------------------------------------------------------------------
# POST /api/detectors/registry/<detector_id>/browse-positives
# POST /api/detectors/registry/<detector_id>/browse-positives/release
# ---------------------------------------------------------------------------


def _detector_browse_embedder(entry: dict, media_type: str) -> str:
    """The embedder to project a detector's positives with.

    The detector's own recorded embedder wins — browsing a detector must not
    depend on whatever dataset is incidentally selected on the dashboard.
    Falls back to the media type's default embedder when the detector has
    never recorded one (e.g. never trained against a dataset).
    """
    recorded = entry.get("embedder", "") or ""
    if recorded:
        return recorded
    from vtscore.media import embedders_for_type

    avail = embedders_for_type(media_type)
    return avail[0].name if avail else ""


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/browse-positives", methods=["POST"])
@detectors_registry_bp.response(200, DetectorBrowsePositivesResponseSchema)
@detectors_registry_bp.alt_response(403, description="Access denied for the current user.")
@detectors_registry_bp.alt_response(404, description="Detector not found.")
def browse_detector_positives(detector_id: str):
    """Prepare an in-memory VTSBrowse map of just this detector's positives.

    Resolves every positive label's origin to its file and embeds it with the
    **detector's** embedder (reusing the loaded detector's cached vectors when
    it's already in that space), assembles a throwaway ``DatasetContext`` whose
    media carry those embeddings + preview bytes, projects it, and registers it
    under a synthetic ``dataset_id`` the browse view can open. Nothing is
    persisted — the context (vectors and bytes) lives only in memory until
    released. Mixed-source detectors work: a positive needn't be in any loaded
    dataset, only origin-resolvable.

    Returns immediately with the ``dataset_id`` and the ``task_id`` whose
    progress the detector's dashboard row renders while the build runs.
    """
    from vtscore.concurrency.progress import CancelledError, detector_loading_tasks
    from vtscore.detectors.positives_browse import build_positives_browse_context, detpos_dataset_id
    from vtscore.detectors.registry import (
        can_user_access_detector,
        get_detector,
        get_loaded_detector_ids,
    )
    from vtscore.state.core import get_detector_context, register_context, unregister_context
    from vtsearch.threading import spawn

    entry = get_detector(detector_id)
    if entry is None:
        abort(404, message="Detector not found")
    if not can_user_access_detector(detector_id, get_current_user()):
        abort(403, message="You do not have access to this detector")

    name = entry.get("name", "") or ""
    data = _read_detector(_detector_path(name)) or {}
    media_type = entry.get("media_type", "") or data.get("media_type", "") or ""

    from vtscore.datasets.labelset import LabelSet

    labelset = LabelSet.from_dict(data.get("labelset") or {})
    if not any(el.label == "good" for el in labelset.elements):
        abort(409, message="This detector has no positive labels to browse.")

    embedder_name = _detector_browse_embedder(entry, media_type)

    # Reuse the loaded detector's cached label vectors only when they were built
    # in the same space we're about to project in; otherwise re-embed fresh.
    cached_embeddings = None
    if detector_id in get_loaded_detector_ids():
        det_ctx = get_detector_context(detector_id)
        if det_ctx is not None and det_ctx.embedder and det_ctx.embedder == embedder_name:
            cached_embeddings = dict(det_ctx.label_embeddings)

    dataset_id = detpos_dataset_id(detector_id)
    # Drop any stale context from a previous browse of this detector.
    unregister_context(dataset_id)

    task_id = f"_detbrowse_{detector_id[:8]}"
    tracker = detector_loading_tasks.create_task(
        task_id,
        name or detector_id,
        detector_id=detector_id,
        media_type=media_type,
        embedder=embedder_name,
    )
    tracker.update("loading", "Preparing browse…", 0, 0, step=1, total_steps=1)

    def build_task():
        try:
            def _on_progress(current: int, total: int, message: str) -> None:
                tracker.check_cancelled()
                tracker.update("loading", message, current, total, step=1, total_steps=1)

            try:
                ctx = build_positives_browse_context(
                    data,
                    dataset_id,
                    embedder_name=embedder_name,
                    cached_embeddings=cached_embeddings,
                    display_name=f"{name} — positives" if name else "Detector positives",
                    on_progress=_on_progress,
                )
                register_context(ctx)
                tracker.update("idle", "", 0, 0, step=None, total_steps=None)
            except CancelledError:
                tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
            except ValueError as e:
                tracker.update("idle", "", 0, 0, error=str(e), step=None, total_steps=None)
            except Exception as e:
                import traceback as _tb

                _tb.print_exc()
                error_msg = str(e) or repr(e) or "Unknown error preparing browse"
                tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            detector_loading_tasks.mark_finished(task_id)

    spawn(build_task, name=f"det-browse-{detector_id[:8]}")
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "task_id": str(task_id),
        "media_type": media_type,
    }


@detectors_registry_bp.route(
    "/api/detectors/registry/<detector_id>/browse-positives/release",
    methods=["POST"],
)
@detectors_registry_bp.response(200, DetectorBrowsePositivesReleaseResponseSchema)
def release_detector_positives_browse(detector_id: str):
    """Free the ephemeral positives-browse context for *detector_id*.

    Called when the user leaves the browse view. Idempotent: ``released`` is
    ``False`` when there was nothing to drop.
    """
    from vtscore.detectors.positives_browse import detpos_dataset_id
    from vtscore.state.core import unregister_context

    dropped = unregister_context(detpos_dataset_id(detector_id))
    return {"ok": True, "released": dropped is not None}


# ---------------------------------------------------------------------------
# Per-plugin typed routes for /api/detectors/registry/from-labelset/<importer>.
# Registered at module-import time by iterating the label-importer
# registry, so each known importer gets a static URL whose body schema
# is described in /api/openapi.json with real per-field types.  Unknown
# importer names fall through to the parameterized route above
# (preserving the legacy 404 message).
# ---------------------------------------------------------------------------

from vtscore.labels.importers import list_label_importers as _list_label_importers  # noqa: E402
from vtsearch.routes._shared import register_plugin_typed_routes as _register_plugin_typed_routes  # noqa: E402

_register_plugin_typed_routes(
    detectors_registry_bp,
    list_plugins=_list_label_importers,
    path_template="/api/detectors/registry/from-labelset/{plugin_name}",
    endpoint_prefix="register_detector_from_labelset",
    delegate=register_detector_from_labelset,
    extra_keys=("name",),
)

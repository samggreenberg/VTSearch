"""Blueprint for detector-registry routes (the in-memory detector catalog).

Every registered detector is backed by a labelset file on disk and an MLP
that lives only in :class:`~vtsearch.utils.DetectorContext` once the user
loads the detector.

Endpoints
---------
GET    /api/detectors/registry                        List registered detectors.
POST   /api/detectors/registry                        Register a new detector.
POST   /api/detectors/registry/load                   Load a detector into memory.
POST   /api/detectors/registry/<id>/unload            Unload a detector from memory.
DELETE /api/detectors/registry/<id>                   Remove a detector from the registry.
PUT    /api/detectors/registry/<id>/rename            Rename a registered detector.
PUT    /api/detectors/registry/<id>/autorun           Toggle the detector's autorun flag.
GET    /api/detectors/loading-tasks                   Active detector-load progress.
POST   /api/detectors/cancel/<task_id>                Cancel a load task.
"""

from __future__ import annotations

import logging
import threading
import time

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.models.detector_store import (
    _detector_path,
    _read_detector,
    _write_detector,
)
from vtsearch.models.label_restoration import (
    restore_labels_from_detector as _restore_labels_from_detector,
)
from vtsearch.models.label_sync import sync_labels_to_loaded_detector
from vtsearch.models.media_seeding import seed_good_votes_from_examples as _seed_good_votes_from_examples

logger = logging.getLogger(__name__)

detectors_registry_bp = Blueprint("detectors_registry", __name__)


@detectors_registry_bp.route("/api/detectors/registry")
def list_registered_detectors():
    """Return all registered detectors with their loaded state and autorun flag."""
    from vtsearch.models.detector_registry import get_loaded_detector_ids, list_detectors
    from vtsearch.settings import get_autorun_detectors

    entries = list_detectors()
    loaded_ids = get_loaded_detector_ids()
    autorun_names = set(get_autorun_detectors())
    for entry in entries:
        did = entry["id"]
        entry["loaded"] = did in loaded_ids
        entry["autorun"] = entry.get("name", "") in autorun_names
        entry.setdefault("last_trained_at", None)
        entry["detector_loaded"] = did in loaded_ids
    return jsonify({"detectors": entries})


@detectors_registry_bp.route("/api/detectors/registry", methods=["POST"])
def register_detector_route():
    """Register a new detector in the detector registry.

    Expects JSON::

        {
            "name": "Dog Barks",
            "media_type": "audio",
            "text_query": "dog barking sounds"
        }
    """
    from vtsearch.models.detector_registry import register_detector

    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    media_type = data.get("media_type", "").strip()
    text_query = data.get("text_query", "")
    media_example = data.get("media_example", "")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not media_type or media_type == "any":
        return jsonify({"error": "media_type is required (must be a specific type, not 'any')"}), 400

    det_path = _detector_path(name)
    if not det_path.exists():
        examples = data.get("examples", [])
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
    return jsonify({"ok": True, "detector": entry}), 201


@detectors_registry_bp.route(
    "/api/detectors/registry/from-labelset/<importer_name>",
    methods=["POST"],
)
def register_detector_from_labelset(importer_name: str):
    """Create a detector seeded with labels from a label importer."""
    from vtsearch.datasets.ingest import _media_type_from_origin
    from vtsearch.datasets.labelset import LabeledElement, LabelSet
    from vtsearch.labels.importers import get_label_importer, list_label_importers
    from vtsearch.models.detector_registry import register_detector, update_detector
    from vtsearch.routes.helpers import (
        extract_plugin_fields,
        get_plugin_or_404,
        get_request_field,
        run_plugin_or_error,
        validate_filepath_field,
        validate_required_fields,
    )

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err

    has_file_fields = any(f.field_type == "file" for f in importer.fields)
    name = get_request_field("name", has_file_fields).strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    det_path = _detector_path(name)
    if det_path.exists():
        return jsonify({"error": f"A detector named '{name}' already exists"}), 409

    field_values = extract_plugin_fields(importer)
    err = validate_required_fields(importer, field_values)
    if err:
        return err
    err = validate_filepath_field(field_values)
    if err:
        return err

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
                    "Could not infer media type from the imported labels — none of "
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


@detectors_registry_bp.route("/api/detectors/registry/load", methods=["POST"])
def load_detector_route():
    """Load a detector into memory and make it active."""
    from vtsearch.models.detector_registry import (
        get_detector,
        is_detector_loaded,
    )
    from vtsearch.utils import (
        DetectorContext,
        bad_votes,
        get_active_detector_context,
        good_votes,
        register_detector_context,
    )

    data = request.get_json(force=True, silent=True) or {}
    detector_id = data.get("detector_id")

    if detector_id is not None:
        entry = get_detector(detector_id)
        if entry is None:
            return jsonify({"error": "Detector not found"}), 404

    if good_votes or bad_votes:
        sync_labels_to_loaded_detector()

    if detector_id is None:
        from vtsearch.models.detector_registry import remove_loaded_detector_id, set_find_mode

        det_ctx = get_active_detector_context()
        prev_id = det_ctx.detector_id if det_ctx.detector_id else None
        if prev_id:
            from vtsearch.utils import unregister_detector_context

            unregister_detector_context(prev_id)
            remove_loaded_detector_id(prev_id)
        set_find_mode(False)
        return jsonify({"ok": True, "labels_restored": 0, "examples_seeded": 0})

    if is_detector_loaded(detector_id):
        return jsonify({"ok": True, "labels_restored": 0, "examples_seeded": 0})

    from vtsearch.utils.progress import CancelledError, detector_loading_tasks

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

    from vtsearch.utils import get_active_context

    _thread_ds_ctx = get_active_context()

    def load_task():
        from vtsearch.models.detector_registry import add_loaded_detector_id, remove_loaded_detector_id
        from vtsearch.utils.state_core import set_thread_dataset_context, set_thread_detector_context

        set_thread_dataset_context(_thread_ds_ctx)
        set_thread_detector_context(det_ctx)

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

                    from vtsearch.datasets.labelset import LabelSet
                    from vtsearch.models.labelset_training import train_from_labelset
                    from vtsearch.utils import snapshot_medias as _snap_medias

                    labelset = LabelSet.from_dict(det_data.get("labelset") or {})
                    media_type = det_data.get("media_type", "") or ""
                    snap = _snap_medias()

                    det_ctx.labelset_good_count = sum(1 for el in labelset.elements if el.label == "good")
                    det_ctx.labelset_bad_count = sum(1 for el in labelset.elements if el.label == "bad")

                    def _embed_progress(name: str, done: int, total: int) -> None:
                        tracker.check_cancelled()
                        tracker.update(
                            "loading",
                            f"Embedding labels… ({done}/{total})",
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

            add_loaded_detector_id(detector_id)
            tracker.update("idle", "", 0, 0, step=None, total_steps=None)
        except CancelledError:
            from vtsearch.utils import unregister_detector_context as _unreg

            _unreg(detector_id)
            remove_loaded_detector_id(detector_id)
            tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
        except Exception as e:
            import traceback as _tb

            _tb.print_exc()
            from vtsearch.utils import unregister_detector_context as _unreg

            _unreg(detector_id)
            remove_loaded_detector_id(detector_id)
            error_msg = str(e) or repr(e) or "Unknown error during detector loading"
            tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            detector_loading_tasks.mark_finished(task_id)

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()
    return jsonify(
        {
            "ok": True,
            "message": "Loading started",
            "task_id": str(task_id),
        }
    )


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/unload", methods=["POST"])
def unload_detector_route(detector_id: str):
    """Unload a detector from memory (frees its DetectorContext)."""
    from vtsearch.models.detector_registry import get_detector, is_detector_loaded, remove_loaded_detector_id
    from vtsearch.utils import (
        bad_votes,
        get_active_detector_context,
        good_votes,
        unregister_detector_context,
    )

    entry = get_detector(detector_id)
    if entry is None:
        return jsonify({"error": "Detector not found"}), 404
    if not is_detector_loaded(detector_id):
        return jsonify({"error": "Detector is not loaded"}), 400

    det_ctx = get_active_detector_context()
    if det_ctx.detector_id == detector_id and (good_votes or bad_votes):
        sync_labels_to_loaded_detector()

    unregister_detector_context(detector_id)
    remove_loaded_detector_id(detector_id)

    return jsonify({"ok": True, "message": "Detector unloaded"})


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>", methods=["DELETE"])
def delete_registered_detector(detector_id: str):
    """Remove a detector from the registry, including its labelset file."""
    from vtsearch.models.detector_registry import get_detector, unregister_detector

    entry = get_detector(detector_id)
    if entry is None:
        return jsonify({"error": "Detector not found"}), 404

    try:
        det_name = entry.get("name", "")
        if det_name:
            det_path = _detector_path(det_name)
            if det_path.exists():
                det_path.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to delete detector file for %s", detector_id)

    try:
        from vtsearch.models.detector_registry import is_detector_loaded, remove_loaded_detector_id
        from vtsearch.utils import unregister_detector_context

        if is_detector_loaded(detector_id):
            unregister_detector_context(detector_id)
            remove_loaded_detector_id(detector_id)
    except Exception:
        logger.exception("Failed to unregister detector context for %s", detector_id)

    # Drop autorun flag if set.
    try:
        from vtsearch.settings import remove_autorun_detector

        remove_autorun_detector(entry.get("name", ""))
    except Exception:
        logger.exception("Failed to drop autorun flag for %s", detector_id)

    unregister_detector(detector_id)
    return jsonify({"ok": True})


@detectors_registry_bp.route("/api/detectors/loading-tasks")
def detector_loading_tasks_endpoint():
    """Return all active detector loading tasks with their progress."""
    from vtsearch.utils.progress import detector_loading_tasks

    return jsonify({"tasks": detector_loading_tasks.list_tasks()})


@detectors_registry_bp.route("/api/detectors/cancel/<task_id>", methods=["POST"])
def cancel_detector_loading_task(task_id: str):
    """Cancel a specific detector loading task."""
    from vtsearch.utils.progress import detector_loading_tasks

    ok = detector_loading_tasks.cancel_task(task_id)
    if not ok:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"ok": True})


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/rename", methods=["PUT"])
def rename_registered_detector(detector_id: str):
    """Rename a registered detector and its on-disk labelset file."""
    from vtsearch.models.detector_registry import get_detector, rename_detector

    data = request.get_json(force=True, silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400

    entry = get_detector(detector_id)
    if entry is None:
        return jsonify({"error": "Detector not found"}), 404

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

        # Rename autorun setting if present.
        try:
            from vtsearch.settings import get_autorun_detectors, set_autorun_detectors

            current = get_autorun_detectors()
            if old_name in current:
                current = [new_name if n == old_name else n for n in current]
                set_autorun_detectors(current)
        except Exception:
            logger.exception("Failed to rename autorun entry for %s", detector_id)

    rename_detector(detector_id, new_name)
    return jsonify({"ok": True, "name": new_name})


@detectors_registry_bp.route("/api/detectors/registry/<detector_id>/autorun", methods=["PUT"])
def set_detector_autorun(detector_id: str):
    """Toggle the autorun flag for a registered detector.

    Body: ``{"autorun": true}``.

    The flag is stored in ``settings.json`` under ``autorun_detectors`` so the
    CLI's ``--autodetect`` flow and the active-dataset ``/api/auto-detect``
    route both see it.
    """
    from vtsearch.models.detector_registry import get_detector
    from vtsearch.settings import (
        add_autorun_detector,
        remove_autorun_detector,
    )

    entry = get_detector(detector_id)
    if entry is None:
        return jsonify({"error": "Detector not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if "autorun" not in data:
        return jsonify({"error": "autorun is required"}), 400
    flag = bool(data["autorun"])

    name = entry.get("name", "")
    if not name:
        return jsonify({"error": "Detector has no name"}), 500

    if flag:
        add_autorun_detector(name)
    else:
        remove_autorun_detector(name)
    return jsonify({"ok": True, "autorun": flag})

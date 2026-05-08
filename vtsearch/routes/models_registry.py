"""Blueprint for model-registry routes (the in-memory model catalog).

Every registered model is a *trainable model* — backed by a labelset file
on disk and an MLP that lives only in :class:`~vtsearch.utils.DetectorContext`
once the user activates the model.

Endpoints
---------
GET    /api/models/registry                        List registered models.
POST   /api/models/registry                        Register a new model.
POST   /api/models/registry/load                   Load a model into memory.
POST   /api/models/registry/<id>/unload            Unload a model from memory.
DELETE /api/models/registry/<id>                   Remove a model from the registry.
PUT    /api/models/registry/<id>/rename            Rename a registered model.
PUT    /api/models/registry/<id>/autorun           Toggle the model's autorun flag.
GET    /api/models/loading-tasks                   Active model-load progress.
POST   /api/models/cancel/<task_id>                Cancel a load task.
"""

from __future__ import annotations

import logging
import threading
import time

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.models.label_restoration import (
    restore_labels_from_trainable_model as _restore_labels_from_trainable_model,
)
from vtsearch.models.label_sync import sync_labels_to_loaded_model
from vtsearch.models.media_seeding import seed_good_votes_from_examples as _seed_good_votes_from_examples
from vtsearch.models.trainable_model_store import (
    _model_path,
    _read_model,
    _write_model,
)

logger = logging.getLogger(__name__)

models_registry_bp = Blueprint("models_registry", __name__)


@models_registry_bp.route("/api/models/registry")
def list_registered_models():
    """Return all registered models with their loaded state and autorun flag."""
    from vtsearch.models.registry import get_loaded_model_ids, list_models
    from vtsearch.settings import get_autorun_trainable_models

    entries = list_models()
    loaded_ids = get_loaded_model_ids()
    autorun_names = set(get_autorun_trainable_models())
    for entry in entries:
        mid = entry["id"]
        entry["loaded"] = mid in loaded_ids
        entry["autorun"] = entry.get("name", "") in autorun_names
        # Backwards-compat alias for any frontend still reading "autodetect".
        entry["autodetect"] = entry["autorun"]
        entry.setdefault("last_trained_at", None)
        entry["detector_loaded"] = mid in loaded_ids
    return jsonify({"models": entries})


@models_registry_bp.route("/api/models/registry", methods=["POST"])
def register_model_route():
    """Register a new model in the model registry.

    Expects JSON::

        {
            "name": "Dog Barks",
            "media_type": "audio",
            "text_query": "dog barking sounds"
        }
    """
    from vtsearch.models.registry import register_model

    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    media_type = data.get("media_type", "").strip()
    text_query = data.get("text_query", "")
    media_example = data.get("media_example", "")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not media_type or media_type == "any":
        return jsonify({"error": "media_type is required (must be a specific type, not 'any')"}), 400

    tm_path = _model_path(name)
    if not tm_path.exists():
        examples = data.get("examples", [])
        if not examples and text_query:
            examples = [{"type": "text", "value": text_query}]
        if not examples and media_example:
            examples = [{"type": "media", "value": media_example}]
        model_data = {
            "name": name,
            "text_query": text_query,
            "media_example": media_example,
            "media_type": media_type,
            "examples": examples,
            "created_at": time.time(),
            "labelset": {"labels": []},
        }
        _write_model(tm_path, model_data)

    entry = register_model(
        name=name,
        media_type=media_type,
        text_query=text_query,
        media_example=media_example,
        created_by=get_current_user(),
    )
    return jsonify({"ok": True, "model": entry}), 201


@models_registry_bp.route(
    "/api/models/registry/from-labelset/<importer_name>",
    methods=["POST"],
)
def register_model_from_labelset(importer_name: str):
    """Create a trainable model seeded with labels from a label importer."""
    from vtsearch.datasets.ingest import _media_type_from_origin
    from vtsearch.datasets.labelset import LabeledElement, LabelSet
    from vtsearch.labels.importers import get_label_importer, list_label_importers
    from vtsearch.models.registry import register_model, update_model
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

    tm_path = _model_path(name)
    if tm_path.exists():
        return jsonify({"error": f"A trainable model named '{name}' already exists"}), 409

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
                    "A model must be for a single media type."
                ),
            }
        ), 400

    media_type = next(iter(detected_types))
    labelset = LabelSet(elements)

    model_data = {
        "name": name,
        "text_query": "",
        "media_example": "",
        "media_type": media_type,
        "examples": [],
        "created_at": time.time(),
        "labelset": labelset.to_dict(),
    }
    _write_model(tm_path, model_data)

    entry = register_model(
        name=name,
        media_type=media_type,
        num_training=len(labelset),
        created_by=get_current_user(),
    )
    update_model(entry["id"], last_trained_at=time.time())
    entry["num_training"] = len(labelset)
    entry["last_trained_at"] = time.time()

    return jsonify(
        {
            "ok": True,
            "model": entry,
            "applied": applied,
            "skipped": skipped,
            "num_labels": len(labelset),
        }
    ), 201


@models_registry_bp.route("/api/models/registry/load", methods=["POST"])
def load_model_route():
    """Load a model into memory and make it active."""
    from vtsearch.models.registry import (
        get_model,
        is_model_loaded,
    )
    from vtsearch.utils import (
        DetectorContext,
        bad_votes,
        get_active_detector_context,
        good_votes,
        register_detector_context,
    )

    data = request.get_json(force=True, silent=True) or {}
    model_id = data.get("model_id")

    if model_id is not None:
        entry = get_model(model_id)
        if entry is None:
            return jsonify({"error": "Model not found"}), 404

    if good_votes or bad_votes:
        sync_labels_to_loaded_model()

    if model_id is None:
        from vtsearch.models.registry import remove_loaded_model_id, set_find_mode

        det_ctx = get_active_detector_context()
        prev_id = det_ctx.detector_id if det_ctx.detector_id else None
        if prev_id:
            from vtsearch.utils import unregister_detector_context

            unregister_detector_context(prev_id)
            remove_loaded_model_id(prev_id)
        set_find_mode(False)
        return jsonify({"ok": True, "labels_restored": 0, "examples_seeded": 0})

    if is_model_loaded(model_id):
        return jsonify({"ok": True, "labels_restored": 0, "examples_seeded": 0})

    from vtsearch.utils.progress import CancelledError, model_loading_tasks

    det_ctx = DetectorContext(
        model_id,
        name=entry.get("name", ""),
        media_type=entry.get("media_type", ""),
    )
    register_detector_context(det_ctx)

    _LOAD_STEPS = 3  # restore labels, seed examples, train MLP
    task_id = f"_modload_{model_id[:8]}"
    tracker = model_loading_tasks.create_task(
        task_id,
        entry.get("name", model_id),
        model_id=model_id,
        media_type=entry.get("media_type", ""),
    )
    tracker.update("loading", "Preparing…", 0, 0, step=1, total_steps=_LOAD_STEPS)

    tm_name = entry.get("name", "")

    from vtsearch.utils import get_active_context

    _thread_ds_ctx = get_active_context()

    def load_task():
        from vtsearch.models.registry import add_loaded_model_id, remove_loaded_model_id
        from vtsearch.utils.state_core import set_thread_dataset_context, set_thread_detector_context

        set_thread_dataset_context(_thread_ds_ctx)
        set_thread_detector_context(det_ctx)

        try:
            if tm_name:
                tracker.check_cancelled()
                tracker.update(
                    "loading",
                    "Restoring labels…",
                    0,
                    0,
                    step=1,
                    total_steps=_LOAD_STEPS,
                )
                tm_data = _read_model(_model_path(tm_name))
                if tm_data:
                    _restore_labels_from_trainable_model(tm_data)

                    tracker.check_cancelled()
                    tracker.update(
                        "loading",
                        "Seeding examples…",
                        0,
                        0,
                        step=2,
                        total_steps=_LOAD_STEPS,
                    )
                    _seed_good_votes_from_examples(tm_data.get("examples", []))

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

                    labelset = LabelSet.from_dict(tm_data.get("labelset") or {})
                    media_type = tm_data.get("media_type", "") or ""
                    snap = _snap_medias()

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

            add_loaded_model_id(model_id)
            tracker.update("idle", "", 0, 0, step=None, total_steps=None)
        except CancelledError:
            from vtsearch.utils import unregister_detector_context as _unreg

            _unreg(model_id)
            remove_loaded_model_id(model_id)
            tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
        except Exception as e:
            import traceback as _tb

            _tb.print_exc()
            from vtsearch.utils import unregister_detector_context as _unreg

            _unreg(model_id)
            remove_loaded_model_id(model_id)
            error_msg = str(e) or repr(e) or "Unknown error during model loading"
            tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            model_loading_tasks.mark_finished(task_id)

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()
    return jsonify(
        {
            "ok": True,
            "message": "Loading started",
            "task_id": str(task_id),
        }
    )


@models_registry_bp.route("/api/models/registry/<model_id>/unload", methods=["POST"])
def unload_model_route(model_id: str):
    """Unload a model from memory (frees its DetectorContext)."""
    from vtsearch.models.registry import get_model, is_model_loaded, remove_loaded_model_id
    from vtsearch.utils import (
        bad_votes,
        get_active_detector_context,
        good_votes,
        unregister_detector_context,
    )

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404
    if not is_model_loaded(model_id):
        return jsonify({"error": "Model is not loaded"}), 400

    det_ctx = get_active_detector_context()
    if det_ctx.detector_id == model_id and (good_votes or bad_votes):
        sync_labels_to_loaded_model()

    unregister_detector_context(model_id)
    remove_loaded_model_id(model_id)

    return jsonify({"ok": True, "message": "Model unloaded"})


@models_registry_bp.route("/api/models/registry/<model_id>", methods=["DELETE"])
def delete_registered_model(model_id: str):
    """Remove a model from the registry, including its labelset file."""
    from vtsearch.models.registry import get_model, unregister_model

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404

    try:
        tm_name = entry.get("name", "")
        if tm_name:
            tm_path = _model_path(tm_name)
            if tm_path.exists():
                tm_path.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to delete trainable-model file for %s", model_id)

    try:
        from vtsearch.models.registry import is_model_loaded, remove_loaded_model_id
        from vtsearch.utils import unregister_detector_context

        if is_model_loaded(model_id):
            unregister_detector_context(model_id)
            remove_loaded_model_id(model_id)
    except Exception:
        logger.exception("Failed to unregister detector context for %s", model_id)

    # Drop autorun flag if set.
    try:
        from vtsearch.settings import remove_autorun_trainable_model

        remove_autorun_trainable_model(entry.get("name", ""))
    except Exception:
        logger.exception("Failed to drop autorun flag for %s", model_id)

    unregister_model(model_id)
    return jsonify({"ok": True})


@models_registry_bp.route("/api/models/loading-tasks")
def model_loading_tasks_endpoint():
    """Return all active model loading tasks with their progress."""
    from vtsearch.utils.progress import model_loading_tasks

    return jsonify({"tasks": model_loading_tasks.list_tasks()})


@models_registry_bp.route("/api/models/cancel/<task_id>", methods=["POST"])
def cancel_model_loading_task(task_id: str):
    """Cancel a specific model loading task."""
    from vtsearch.utils.progress import model_loading_tasks

    ok = model_loading_tasks.cancel_task(task_id)
    if not ok:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"ok": True})


@models_registry_bp.route("/api/models/registry/<model_id>/rename", methods=["PUT"])
def rename_registered_model(model_id: str):
    """Rename a registered model and its on-disk labelset file."""
    from vtsearch.models.registry import get_model, rename_model

    data = request.get_json(force=True, silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404

    old_name = entry.get("name", "")
    if old_name and old_name != new_name:
        old_path = _model_path(old_name)
        tm_data = _read_model(old_path)
        if tm_data:
            new_path = _model_path(new_name)
            tm_data["name"] = new_name
            _write_model(new_path, tm_data)
            if new_path != old_path:
                old_path.unlink(missing_ok=True)

        # Rename autorun setting if present.
        try:
            from vtsearch.settings import get_autorun_trainable_models, set_autorun_trainable_models

            current = get_autorun_trainable_models()
            if old_name in current:
                current = [new_name if n == old_name else n for n in current]
                set_autorun_trainable_models(current)
        except Exception:
            logger.exception("Failed to rename autorun entry for %s", model_id)

    rename_model(model_id, new_name)
    return jsonify({"ok": True, "name": new_name})


@models_registry_bp.route("/api/models/registry/<model_id>/autorun", methods=["PUT"])
def set_model_autorun(model_id: str):
    """Toggle the autorun flag for a registered model.

    Body: ``{"autorun": true}`` (or ``"autodetect"`` for backwards-compat).

    The flag is stored in ``settings.json`` under ``autorun_trainable_models``
    so the CLI's ``--autodetect`` flow and the active-dataset
    ``/api/auto-detect`` route both see it.
    """
    from vtsearch.models.registry import get_model
    from vtsearch.settings import (
        add_autorun_trainable_model,
        remove_autorun_trainable_model,
    )

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if "autorun" in data:
        flag = bool(data["autorun"])
    elif "autodetect" in data:
        flag = bool(data["autodetect"])
    else:
        return jsonify({"error": "autorun is required"}), 400

    name = entry.get("name", "")
    if not name:
        return jsonify({"error": "Model has no name"}), 500

    if flag:
        add_autorun_trainable_model(name)
    else:
        remove_autorun_trainable_model(name)
    return jsonify({"ok": True, "autorun": flag})

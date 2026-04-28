"""Blueprint for model-registry routes (the in-memory model catalog).

A *registered model* is an entry in the in-memory model registry — distinct
from a *trainable model*, which is the on-disk labelset+query store handled
by :mod:`vtsearch.routes.trainable_models`.  A registered model can be
loaded into a :class:`~vtsearch.utils.DetectorContext` so the user can vote
or run Find against it.

Endpoints
---------
GET    /api/models/registry                        List registered models.
POST   /api/models/registry                        Register a new model.
POST   /api/models/registry/load                   Load a model into memory.
POST   /api/models/registry/<id>/unload            Unload a model from memory.
DELETE /api/models/registry/<id>                   Remove a model from the registry.
PUT    /api/models/registry/<id>/rename            Rename a registered model.
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
    """Return all registered models with their loaded state and autodetect flag."""
    from vtsearch.models.registry import get_loaded_model_ids, list_models
    from vtsearch.settings import get_autorun_detector_names
    from vtsearch.utils import get_autorun_detectors

    entries = list_models()
    loaded_ids = get_loaded_model_ids()
    detectors = get_autorun_detectors()
    autorun_names = set(get_autorun_detector_names())
    for entry in entries:
        mid = entry["id"]
        entry["loaded"] = mid in loaded_ids
        det_name = entry.get("detector_name", "")
        det = detectors.get(det_name) if det_name else None
        if det:
            entry["autodetect"] = bool(det.get("autodetect"))
        else:
            # Fall back to the persisted settings list
            entry["autodetect"] = det_name in autorun_names if det_name else False
        entry.setdefault("last_trained_at", None)
        # detector_loaded: True when the model has inference data in RAM.
        # Either via a DetectorContext (multi-loaded) or via autorun_detectors weights.
        if mid in loaded_ids:
            entry["detector_loaded"] = True
        elif det_name and det:
            entry["detector_loaded"] = det.get("weights") is not None
        else:
            entry["detector_loaded"] = False
    return jsonify({"models": entries})


@models_registry_bp.route("/api/models/registry", methods=["POST"])
def register_model_route():
    """Register a new model in the model registry.

    Expects JSON::

        {
            "name": "Dog Barks",
            "media_type": "audio",
            "trainable": true,
            "text_query": "dog barking sounds"
        }
    """
    from vtsearch.models.registry import register_model

    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    media_type = data.get("media_type", "").strip()
    trainable = data.get("trainable", True)
    text_query = data.get("text_query", "")
    media_example = data.get("media_example", "")
    detector_name = data.get("detector_name", "")
    trainable_model_name = data.get("trainable_model_name", "")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not media_type or media_type == "any":
        return jsonify({"error": "media_type is required (must be a specific type, not 'any')"}), 400

    # If trainable, also create the trainable model file if needed
    if trainable and not trainable_model_name:
        trainable_model_name = name
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
        trainable=trainable,
        text_query=text_query,
        media_example=media_example,
        detector_name=detector_name,
        trainable_model_name=trainable_model_name,
        created_by=get_current_user(),
    )
    return jsonify({"ok": True, "model": entry}), 201


@models_registry_bp.route("/api/models/registry/load", methods=["POST"])
def load_model_route():
    """Load a model into memory and make it active.

    Expects JSON::

        {"model_id": "abc123"}

    Pass ``model_id: null`` to deactivate (no model active).

    When loading a trainable model:

    1. The current model's labels are saved (auto-sync).
    2. A new DetectorContext is created for the model (or an existing one
       is reused if already loaded).
    3. The model's saved labelset is restored into the DetectorContext.
    4. Media examples are seeded as good votes.
    """
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

    # Save current model's labels before switching — but only if there
    # are votes in the active detector context.
    if good_votes or bad_votes:
        sync_labels_to_loaded_model()

    if model_id is None:
        # No model requested — unload the current model (if any) and clear find mode.
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
        # Already loaded — nothing more to do.
        return jsonify({"ok": True, "labels_restored": 0, "examples_seeded": 0})

    # New load: create a DetectorContext, register it, then load labels
    # asynchronously so the frontend can show a progress bar.
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
        task_id, entry.get("name", model_id), model_id=model_id,
        media_type=entry.get("media_type", ""),
    )
    tracker.update("loading", "Preparing…", 0, 0, step=1, total_steps=_LOAD_STEPS)

    # Capture values needed by the background thread.
    tm_name = entry.get("trainable_model_name", "")
    # Capture the dataset context so the thread can resolve medias.
    from vtsearch.utils import get_active_context
    _thread_ds_ctx = get_active_context()

    def load_task():
        from vtsearch.models.registry import add_loaded_model_id, remove_loaded_model_id
        from vtsearch.utils.state_core import set_thread_dataset_context, set_thread_detector_context

        # Set thread-local contexts so proxy objects (medias, good_votes, etc.)
        # resolve correctly in this background thread.
        set_thread_dataset_context(_thread_ds_ctx)
        set_thread_detector_context(det_ctx)

        try:
            if tm_name:
                tracker.check_cancelled()
                tracker.update(
                    "loading", "Restoring labels…", 0, 0,
                    step=1, total_steps=_LOAD_STEPS,
                )
                tm_data = _read_model(_model_path(tm_name))
                if tm_data:
                    _restore_labels_from_trainable_model(tm_data)

                    tracker.check_cancelled()
                    tracker.update(
                        "loading", "Seeding examples…", 0, 0,
                        step=2, total_steps=_LOAD_STEPS,
                    )
                    _seed_good_votes_from_examples(
                        tm_data.get("examples", [])
                    )

                    # Train the MLP from restored votes so that Find can use
                    # det_ctx.model directly without re-resolving label origins.
                    tracker.check_cancelled()
                    from vtsearch.utils import good_votes as _gv, bad_votes as _bv, snapshot_medias as _snap_medias

                    if _gv and _bv:
                        tracker.update(
                            "loading", "Training model…", 0, 0,
                            step=3, total_steps=_LOAD_STEPS,
                        )
                        from vtsearch.models.detector_training import train_and_threshold

                        snap = _snap_medias()
                        X_list = []
                        y_list: list[float] = []
                        for cid in _gv:
                            if cid in snap:
                                X_list.append(snap[cid]["embedding"])
                                y_list.append(1.0)
                        for cid in _bv:
                            if cid in snap:
                                X_list.append(snap[cid]["embedding"])
                                y_list.append(0.0)

                        if X_list and any(v == 1.0 for v in y_list) and any(v == 0.0 for v in y_list):
                            trained_model, threshold = train_and_threshold(X_list, y_list, snap=snap)
                            det_ctx.model = trained_model
                            det_ctx.threshold = threshold

            # Mark as fully loaded so the registry shows detector_loaded=True.
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
    return jsonify({
        "ok": True,
        "message": "Loading started",
        "task_id": str(task_id),
    })


@models_registry_bp.route("/api/models/registry/<model_id>/unload", methods=["POST"])
def unload_model_route(model_id: str):
    """Unload a model from memory (frees its DetectorContext).

    Saves labels before unloading if the model is active.
    """
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

    # Save labels if this model is currently resolved by the context
    det_ctx = get_active_detector_context()
    if det_ctx.detector_id == model_id and (good_votes or bad_votes):
        sync_labels_to_loaded_model()

    # Remove from memory
    unregister_detector_context(model_id)
    remove_loaded_model_id(model_id)

    return jsonify({"ok": True, "message": "Model unloaded"})


@models_registry_bp.route("/api/models/registry/<model_id>", methods=["DELETE"])
def delete_registered_model(model_id: str):
    """Remove a model from the registry."""
    from vtsearch.models.registry import get_model, unregister_model

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404

    # Clean up associated resources.  Failures here must not prevent the
    # registry entry from being removed — otherwise the user sees a detector
    # that cannot be deleted.
    try:
        tm_name = entry.get("trainable_model_name", "")
        if tm_name:
            tm_path = _model_path(tm_name)
            if tm_path.exists():
                tm_path.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to delete trainable-model file for %s", model_id)

    try:
        det_name = entry.get("detector_name", "")
        if det_name:
            from vtsearch.utils import remove_autorun_detector

            remove_autorun_detector(det_name)
    except Exception:
        logger.exception("Failed to remove autorun detector for %s", model_id)

    try:
        from vtsearch.models.registry import is_model_loaded, remove_loaded_model_id
        from vtsearch.utils import unregister_detector_context

        if is_model_loaded(model_id):
            unregister_detector_context(model_id)
            remove_loaded_model_id(model_id)
    except Exception:
        logger.exception("Failed to unregister detector context for %s", model_id)

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
    """Rename a registered model."""
    from vtsearch.models.registry import get_model, rename_model

    data = request.get_json(force=True, silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404

    # Rename the underlying trainable model file if applicable
    tm_name = entry.get("trainable_model_name", "")
    if tm_name:
        old_path = _model_path(tm_name)
        tm_data = _read_model(old_path)
        if tm_data:
            new_path = _model_path(new_name)
            tm_data["name"] = new_name
            _write_model(new_path, tm_data)
            if new_path != old_path:
                old_path.unlink(missing_ok=True)
        from vtsearch.models.registry import update_model

        update_model(model_id, trainable_model_name=new_name)

    rename_model(model_id, new_name)
    return jsonify({"ok": True, "name": new_name})

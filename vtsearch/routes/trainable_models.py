"""Blueprint for trainable model routes.

Trainable models are a persistent reference to a labelset file plus a text-sort
query.  They live on disk in ``data/trainable_models/<slug>.json`` and can
accumulate labels over repeated training sessions.

Endpoints
---------
GET  /api/trainable-models
    List all trainable models.

POST /api/trainable-models
    Create a new trainable model (requires ``name`` and ``text_query``).

GET  /api/trainable-models/<name>
    Retrieve a single trainable model with its labelset.

DELETE /api/trainable-models/<name>
    Delete a trainable model.

PUT  /api/trainable-models/<name>/rename
    Rename a trainable model.

POST /api/trainable-models/<name>/labels
    Save the current votes as the model's labelset.
"""

from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.models.trainable_model_store import (
    _model_path,
    _read_model,
    _write_model,
    get_trainable_models_dir,
)

logger = logging.getLogger(__name__)

trainable_models_bp = Blueprint("trainable_models", __name__)


def _list_all() -> list[dict]:
    """Return summary info for every trainable model on disk."""
    tm_dir = get_trainable_models_dir()
    if not tm_dir.is_dir():
        return []
    models = []
    for p in sorted(tm_dir.iterdir()):
        if p.suffix != ".json":
            continue
        data = _read_model(p)
        if data is None:
            continue
        labels = data.get("labelset", {}).get("labels", [])
        models.append(
            {
                "name": data["name"],
                "text_query": data.get("text_query", ""),
                "media_example": data.get("media_example", ""),
                "media_type": data.get("media_type", ""),
                "examples": data.get("examples", []),
                "num_labels": len(labels),
                "created_at": data.get("created_at", 0),
            }
        )
    return models


# ---------------------------------------------------------------------------
# GET /api/trainable-models
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models", methods=["GET"])
def list_trainable_models():
    """Return all trainable models (summary only, no full labelset)."""
    return jsonify({"models": _list_all()})


# ---------------------------------------------------------------------------
# POST /api/trainable-models
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models", methods=["POST"])
def create_trainable_model():
    """Create a new trainable model.

    Expects JSON::

        {"name": "Dog Barks", "text_query": "dog barking sounds"}
    """
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("name", "").strip()
    text_query = data.get("text_query", "").strip()
    media_example = data.get("media_example", "").strip()
    media_type = data.get("media_type", "").strip()
    examples = data.get("examples")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not text_query and not media_example and not examples:
        return jsonify({"error": "text_query, media_example, or examples is required"}), 400
    if not media_type or media_type == "any":
        return jsonify({"error": "media_type is required (must be a specific type, not 'any')"}), 400

    path = _model_path(name)
    if path.exists():
        return jsonify({"error": f"A trainable model named '{name}' already exists"}), 409

    # Build examples list; if text_query/media_example provided without
    # explicit examples, create a single example from it for backward compat.
    if examples is None and text_query:
        examples = [{"type": "text", "value": text_query}]
    elif examples is None and media_example:
        examples = [{"type": "media", "value": media_example}]

    model_data = {
        "name": name,
        "text_query": text_query,
        "media_example": media_example,
        "media_type": media_type,
        "examples": examples or [],
        "created_at": time.time(),
        "labelset": {"labels": []},
    }
    _write_model(path, model_data)

    return jsonify(
        {
            "success": True,
            "name": name,
            "text_query": text_query,
            "media_example": media_example,
            "media_type": media_type,
            "examples": examples or [],
            "num_labels": 0,
        }
    ), 201


# ---------------------------------------------------------------------------
# GET /api/trainable-models/<name>
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models/<name>", methods=["GET"])
def get_trainable_model(name: str):
    """Retrieve a single trainable model with its full labelset."""
    path = _model_path(name)
    data = _read_model(path)
    if data is None:
        return jsonify({"error": f"Trainable model '{name}' not found"}), 404
    return jsonify(data)


# ---------------------------------------------------------------------------
# DELETE /api/trainable-models/<name>
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models/<name>", methods=["DELETE"])
def delete_trainable_model(name: str):
    """Delete a trainable model."""
    path = _model_path(name)
    if not path.exists():
        return jsonify({"error": f"Trainable model '{name}' not found"}), 404
    path.unlink()
    return jsonify({"success": True, "name": name})


# ---------------------------------------------------------------------------
# PUT /api/trainable-models/<name>/rename
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models/<name>/rename", methods=["PUT"])
def rename_trainable_model(name: str):
    """Rename a trainable model.

    Expects JSON::

        {"new_name": "Cat Meows"}
    """
    old_path = _model_path(name)
    data = _read_model(old_path)
    if data is None:
        return jsonify({"error": f"Trainable model '{name}' not found"}), 404

    body = request.get_json(force=True, silent=True) or {}
    new_name = body.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    new_path = _model_path(new_name)
    if new_path.exists() and new_path != old_path:
        return jsonify({"error": f"A model named '{new_name}' already exists"}), 409

    data["name"] = new_name
    _write_model(new_path, data)
    if new_path != old_path:
        old_path.unlink(missing_ok=True)

    # Update the model registry entry that references this trainable model
    from vtsearch.models.registry import find_by_trainable_model_name, rename_model, update_model

    reg_entry = find_by_trainable_model_name(name)
    if reg_entry:
        update_model(reg_entry["id"], trainable_model_name=new_name)
        rename_model(reg_entry["id"], new_name)

    return jsonify({"success": True, "old_name": name, "new_name": new_name})


# ---------------------------------------------------------------------------
# PUT /api/trainable-models/<name>/examples
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models/<name>/examples", methods=["PUT"])
def set_trainable_model_examples(name: str):
    """Set/replace the examples for a trainable model.

    Expects JSON::

        {"examples": [{"type": "text", "value": "dog barking"}]}
    """
    path = _model_path(name)
    data = _read_model(path)
    if data is None:
        return jsonify({"error": f"Trainable model '{name}' not found"}), 404

    body = request.get_json(force=True, silent=True) or {}
    examples = body.get("examples")
    if examples is None:
        return jsonify({"error": "examples is required"}), 400

    data["examples"] = examples
    # Update text_query from first text example for backward compat
    text_examples = [e for e in examples if e.get("type") == "text" and e.get("value")]
    if text_examples:
        data["text_query"] = text_examples[0]["value"]
    _write_model(path, data)

    return jsonify({"success": True, "name": name, "examples": examples})


# ---------------------------------------------------------------------------
# POST /api/trainable-models/<name>/labels
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/trainable-models/<name>/labels", methods=["POST"])
def save_trainable_model_labels(name: str):
    """Save the current votes as the model's labelset.

    Reads good_votes/bad_votes from global state and the current medias
    to build a fresh LabelSet, then persists it into the model's JSON file.
    """
    path = _model_path(name)
    data = _read_model(path)
    if data is None:
        return jsonify({"error": f"Trainable model '{name}' not found"}), 404

    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.utils import bad_votes, good_votes, snapshot_medias

    labelset = LabelSet.from_clips_and_votes(snapshot_medias(), good_votes, bad_votes, expand_dupes=False)
    data["labelset"] = labelset.to_dict()
    _write_model(path, data)

    # Also update the model registry entry if one exists
    from vtsearch.models.registry import find_by_trainable_model_name, update_model

    import time as _time

    reg_entry = find_by_trainable_model_name(name)
    if reg_entry:
        update_model(reg_entry["id"], num_training=len(labelset), last_trained_at=_time.time())

    return jsonify(
        {
            "success": True,
            "name": name,
            "num_labels": len(labelset),
        }
    )


# ---------------------------------------------------------------------------
# POST /api/trainable-models/<name>/import-labels/<importer_name>
# ---------------------------------------------------------------------------


@trainable_models_bp.route(
    "/api/trainable-models/<name>/import-labels/<importer_name>",
    methods=["POST"],
)
def import_labels_into_model(name: str, importer_name: str):
    """Run a label importer and merge results into this model's labelset.

    Unlike the regular ``/api/label-importers/import/`` route, this does
    **not** require a dataset to be loaded.  The imported label entries are
    merged directly into the trainable model's persisted labelset, and the
    model-registry entry is updated so the dashboard reflects the new count.

    When the model's detector context **is** loaded, the new labels are also
    resolved against the loaded dataset's medias, applied to the detector's
    votes, and a fresh MLP is trained with a cross-validated threshold — all
    inside the loaded detector context.

    Returns JSON with ``applied``, ``skipped``, ``num_labels``, and
    ``message`` keys.
    """
    path = _model_path(name)
    data = _read_model(path)
    if data is None:
        return jsonify({"error": f"Trainable model '{name}' not found"}), 404

    from vtsearch.labels.importers import get_label_importer, list_label_importers
    from vtsearch.routes.helpers import (
        extract_plugin_fields,
        get_plugin_or_404,
        run_plugin_or_error,
        validate_filepath_field,
        validate_required_fields,
    )

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err

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

    # ------------------------------------------------------------------
    # 1) Merge into the persisted labelset (always, whether loaded or not)
    # ------------------------------------------------------------------
    from vtsearch.datasets.labelset import LabeledElement, LabelSet

    existing_ls = LabelSet.from_dict(data.get("labelset") or {})

    # Build a set of existing (md5, label) pairs for dedup
    existing_keys: set[tuple[str, str]] = set()
    for el in existing_ls.elements:
        if el.md5:
            existing_keys.add((el.md5, el.label))

    applied = 0
    skipped = 0
    new_entries: list[dict] = []
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            skipped += 1
            continue
        md5 = entry.get("md5", "")
        if md5 and (md5, label) in existing_keys:
            skipped += 1
            continue
        elem = LabeledElement.from_dict(entry)
        existing_ls.elements.append(elem)
        new_entries.append(entry)
        if md5:
            existing_keys.add((md5, label))
        applied += 1

    data["labelset"] = existing_ls.to_dict()
    _write_model(path, data)

    # Update the model registry entry
    from vtsearch.models.registry import find_by_trainable_model_name, update_model

    reg_entry = find_by_trainable_model_name(name)
    if reg_entry:
        update_model(reg_entry["id"], num_training=len(existing_ls), last_trained_at=time.time())

    # ------------------------------------------------------------------
    # 2) If the detector is loaded, resolve + apply + retrain in context
    # ------------------------------------------------------------------
    resolved = 0
    trained = False
    if applied > 0 and reg_entry:
        from vtsearch.utils.state_core import get_detector_context

        det_ctx = get_detector_context(reg_entry["id"])
        if det_ctx is not None:
            resolved, trained = _apply_and_retrain(
                reg_entry["id"],
                det_ctx,
                new_entries,
                name,
            )

    msg = f"Added {applied} label(s) to model '{name}', skipped {skipped}."
    if resolved > 0:
        msg += f" Resolved {resolved} into the loaded detector."
    if trained:
        msg += " Retrained MLP."
    return jsonify(
        {
            "applied": applied,
            "skipped": skipped,
            "resolved": resolved,
            "trained": trained,
            "num_labels": len(existing_ls),
            "message": msg,
        }
    )


# Canonical location: vtsearch.models.training_workflow
from vtsearch.models.training_workflow import apply_and_retrain as _apply_and_retrain  # noqa: E402


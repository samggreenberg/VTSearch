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

import json
import re
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.config import DATA_DIR

trainable_models_bp = Blueprint("trainable_models", __name__)


def get_trainable_models_dir() -> Path:
    """Return the configured trainable-models directory from settings."""
    from vtsearch.settings import get_trainable_models_dir as _get

    return _get()


#: Backward-compat alias — prefer :func:`get_trainable_models_dir` for live value.
TRAINABLE_MODELS_DIR = DATA_DIR / "trainable_models"


def _slug(name: str) -> str:
    """Turn a human-readable name into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_") or "model"


def _model_path(name: str) -> Path:
    return get_trainable_models_dir() / f"{_slug(name)}.json"


def _read_model(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_model(path: Path, data: dict) -> None:
    get_trainable_models_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sync_labels_to_loaded_model() -> None:
    """Persist the current votes into the loaded model's labelset (if any).

    Called automatically after each vote so the dashboard's "# Training"
    and "Last Trained" columns stay up to date without an explicit save.

    Skipped when the model is in "find mode" (after ``/api/find-label``),
    because the global votes reflect scoring results on a different dataset,
    not the model's original training labels.
    """
    from vtsearch.models.registry import get_loaded_id, get_model, is_find_mode, update_model

    if is_find_mode():
        return

    loaded_id = get_loaded_id()
    if not loaded_id:
        return

    entry = get_model(loaded_id)
    if not entry or not entry.get("trainable") or not entry.get("trainable_model_name"):
        return

    tm_name = entry["trainable_model_name"]
    path = _model_path(tm_name)
    data = _read_model(path)
    if data is None:
        return

    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.utils import bad_votes, good_votes, snapshot_medias

    labelset = LabelSet.from_clips_and_votes(snapshot_medias(), good_votes, bad_votes, expand_dupes=False)
    data["labelset"] = labelset.to_dict()
    _write_model(path, data)

    import time as _time

    update_model(entry["id"], num_training=len(labelset), last_trained_at=_time.time())


def _seed_good_votes_from_examples(examples: list[dict]) -> int:
    """Seed good votes from a model's media examples.

    For each ``type: "media"`` example, reads the file from
    ``data/example_media/`` and adds it to ``good_votes``:

    * **Match by MD5** — if a loaded media has the same content hash,
      that media is voted good (keeping its original dataset origin).
    * **No match** — the example file is embedded using the dataset's
      embedder, inserted into the ``medias`` dict as a new item with
      an ``example_media`` origin, and voted good.  This makes it
      available for training (its embedding is in the medias snapshot)
      and for label export (LabelSet picks it up from medias + votes).

    Returns the number of example entries successfully seeded.
    """
    import hashlib

    from vtsearch.config import DATA_DIR
    from vtsearch.utils import (
        _state_lock,
        apply_label,
        build_media_lookup,
        medias,
        next_media_id,
        snapshot_medias,
    )

    media_examples = [
        ex
        for ex in examples
        if isinstance(ex, dict) and ex.get("type") == "media" and ex.get("value", "").strip()
    ]
    if not media_examples:
        return 0

    snap = snapshot_medias()
    if not snap:
        return 0

    _, md5_lookup = build_media_lookup(snap)
    server_media_dir = DATA_DIR / "example_media"

    # Determine the embedder and media type from the loaded dataset so we
    # can embed example files that aren't already in the dataset.
    first_media = next(iter(snap.values()))
    dataset_media_type = first_media.get("type", "audio")
    dataset_embedder_name = first_media.get("embedder", "")
    embedder = None  # lazily loaded only when needed

    seeded = 0
    for ex in media_examples:
        filename = ex["value"].strip()
        file_path = server_media_dir / filename
        # Prevent directory traversal
        try:
            file_path.resolve().relative_to(server_media_dir.resolve())
        except ValueError:
            continue
        if not file_path.is_file():
            continue

        file_bytes = file_path.read_bytes()
        file_md5 = hashlib.md5(file_bytes).hexdigest()
        cids = md5_lookup.get(file_md5, [])

        if cids:
            # Example matches existing dataset media — just vote good.
            for cid in cids:
                apply_label(cid, "good")
            seeded += 1
        else:
            # Example is NOT in the dataset — embed and insert as new media.
            if embedder is None:
                from vtsearch.media import embedders_for_type, get_embedder

                if dataset_embedder_name:
                    try:
                        embedder = get_embedder(dataset_embedder_name)
                    except KeyError:
                        pass
                if embedder is None:
                    avail = embedders_for_type(dataset_media_type)
                    embedder = avail[0] if avail else None
                if embedder is None:
                    # No embedder available; skip remaining examples.
                    continue

            embedding = embedder.embed_media(file_path)
            if embedding is None:
                continue

            with _state_lock:
                new_id = next_media_id(medias)
                medias[new_id] = {
                    "id": new_id,
                    "type": dataset_media_type,
                    "embedder": dataset_embedder_name,
                    "md5": file_md5,
                    "embedding": embedding,
                    "media_bytes": file_bytes,
                    "filename": filename,
                    "file_size": len(file_bytes),
                    "category": "",
                    "origin": {
                        "importer": "example_media",
                        "params": {"filename": filename},
                    },
                    "origin_name": filename,
                }

            apply_label(new_id, "good")
            seeded += 1

    return seeded


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
        models.append({
            "name": data["name"],
            "text_query": data.get("text_query", ""),
            "media_example": data.get("media_example", ""),
            "media_type": data.get("media_type", ""),
            "examples": data.get("examples", []),
            "num_labels": len(labels),
            "created_at": data.get("created_at", 0),
        })
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

    return jsonify({
        "success": True,
        "name": name,
        "text_query": text_query,
        "media_example": media_example,
        "media_type": media_type,
        "examples": examples or [],
        "num_labels": 0,
    }), 201


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

    return jsonify({
        "success": True,
        "name": name,
        "num_labels": len(labelset),
    })


# ---------------------------------------------------------------------------
# Model registry endpoints
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/models/registry")
def list_registered_models():
    """Return all registered models with their loaded state and autodetect flag."""
    from vtsearch.models.registry import get_loaded_id, list_models
    from vtsearch.settings import get_autorun_detector_names
    from vtsearch.utils import get_autorun_detectors

    entries = list_models()
    loaded_id = get_loaded_id()
    detectors = get_autorun_detectors()
    autorun_names = set(get_autorun_detector_names())
    for entry in entries:
        entry["loaded"] = entry["id"] == loaded_id
        det_name = entry.get("detector_name", "")
        det = detectors.get(det_name) if det_name else None
        if det:
            entry["autodetect"] = bool(det.get("autodetect"))
        else:
            # Fall back to the persisted settings list
            entry["autodetect"] = det_name in autorun_names if det_name else False
        entry.setdefault("last_trained_at", None)
    return jsonify({"models": entries})


@trainable_models_bp.route("/api/models/registry", methods=["POST"])
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


@trainable_models_bp.route("/api/models/registry/load", methods=["POST"])
def load_model_route():
    """Set the currently loaded model by ID.

    Expects JSON::

        {"model_id": "abc123"}

    Pass ``model_id: null`` to unload.

    When loading a model that has media examples, any example files whose
    MD5 matches a loaded media item are automatically added to
    ``good_votes`` so that the labeling session starts with those examples
    pre-labeled as Good.
    """
    from vtsearch.models.registry import get_model, set_loaded_id

    data = request.get_json(force=True, silent=True) or {}
    model_id = data.get("model_id")

    if model_id is not None:
        entry = get_model(model_id)
        if entry is None:
            return jsonify({"error": "Model not found"}), 404

    set_loaded_id(model_id)

    # Seed good votes from media examples on the loaded model.
    examples_seeded = 0
    if model_id is not None and entry is not None:
        tm_name = entry.get("trainable_model_name", "")
        if tm_name:
            tm_data = _read_model(_model_path(tm_name))
            if tm_data:
                examples_seeded = _seed_good_votes_from_examples(
                    tm_data.get("examples", [])
                )

    return jsonify({"ok": True, "examples_seeded": examples_seeded})


@trainable_models_bp.route("/api/models/registry/<model_id>", methods=["DELETE"])
def delete_registered_model(model_id: str):
    """Remove a model from the registry."""
    from vtsearch.models.registry import get_model, unregister_model

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404

    # Also clean up the underlying trainable model file
    tm_name = entry.get("trainable_model_name", "")
    if tm_name:
        tm_path = _model_path(tm_name)
        if tm_path.exists():
            tm_path.unlink(missing_ok=True)

    # Clean up the autorun detector if any
    det_name = entry.get("detector_name", "")
    if det_name:
        from vtsearch.utils import remove_autorun_detector

        remove_autorun_detector(det_name)

    unregister_model(model_id)
    return jsonify({"ok": True})


@trainable_models_bp.route("/api/models/registry/<model_id>/rename", methods=["PUT"])
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

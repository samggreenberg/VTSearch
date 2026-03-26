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
import threading
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
    from vtsearch.models.registry import get_model, is_find_mode, update_model
    from vtsearch.utils import get_active_detector_id

    if is_find_mode():
        return

    loaded_id = get_active_detector_id()
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

    _, md5_lookup, _ = build_media_lookup(snap)
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


def _restore_labels_from_trainable_model(tm_data: dict) -> int:
    """Restore saved labels from a trainable model's labelset into votes.

    Matches labelset entries to loaded medias by origin+origin_name, MD5, and
    origin_name fallback.  For entries that still don't match (cross-dataset
    scenario), resolves the original file from its origin trail, computes its
    MD5, and checks for a match in the loaded dataset.

    Returns the number of labels successfully restored.
    """
    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.utils import (
        apply_label,
        build_media_lookup,
        resolve_media_ids,
        snapshot_medias,
    )

    labelset_dict = tm_data.get("labelset")
    if not labelset_dict:
        return 0

    labelset = LabelSet.from_dict(labelset_dict)
    if not labelset.elements:
        return 0

    snap = snapshot_medias()
    if not snap:
        return 0

    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)

    restored = 0
    unresolved: list[tuple] = []  # (elem, label) pairs needing origin resolution
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup, name_lookup)
        if cids:
            for cid in cids:
                apply_label(cid, elem.label)
            restored += 1
        else:
            unresolved.append(elem)

    # Second pass: resolve unmatched labels from their origin files.
    # When a detector was trained on Dataset A and we're now on Dataset B,
    # the origin+name keys won't match.  But if the same underlying file
    # exists in both datasets, resolving the origin file and computing its
    # MD5 lets us match by content hash.
    if unresolved:
        import hashlib
        import logging

        from vtsearch.models.resolver import resolve_file_from_origin

        _log = logging.getLogger(__name__)

        for elem in unresolved:
            entry = elem.to_dict()
            origin = entry.get("origin")
            origin_name = entry.get("origin_name", "")
            filename = entry.get("filename", "")
            resolved_path = resolve_file_from_origin(origin, origin_name, filename)
            if resolved_path is None:
                continue

            # Compute MD5 of the resolved file and check against loaded medias
            try:
                h = hashlib.md5()
                with open(resolved_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                resolved_md5 = h.hexdigest()
            except OSError:
                _log.debug("restore-labels: could not read resolved file %s", resolved_path)
                continue

            cids = md5_lookup.get(resolved_md5, [])
            if cids:
                for cid in cids:
                    apply_label(cid, elem.label)
                restored += 1
            else:
                _log.debug(
                    "restore-labels: resolved %s but MD5 %s not in loaded dataset",
                    resolved_path, resolved_md5,
                )

    return restored


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
    from vtsearch.routes.helpers import extract_plugin_fields, run_plugin_or_error, validate_filepath_field, validate_required_fields

    importer = get_label_importer(importer_name)
    if importer is None:
        known = [imp.name for imp in list_label_importers()]
        return (
            jsonify({"error": f"Unknown label importer '{importer_name}'. Available: {known}"}),
            404,
        )

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
                reg_entry["id"], det_ctx, new_entries, name,
            )

    msg = f"Added {applied} label(s) to model '{name}', skipped {skipped}."
    if resolved > 0:
        msg += f" Resolved {resolved} into the loaded detector."
    if trained:
        msg += " Retrained MLP."
    return jsonify({
        "applied": applied,
        "skipped": skipped,
        "resolved": resolved,
        "trained": trained,
        "num_labels": len(existing_ls),
        "message": msg,
    })


def _apply_and_retrain(
    model_id: str,
    det_ctx: object,
    new_entries: list[dict],
    tm_name: str,
) -> tuple[int, bool]:
    """Resolve new label entries into a loaded detector and retrain its MLP.

    Temporarily switches the active detector context to *model_id*, resolves
    the entries against the loaded dataset's medias, applies matching labels,
    retrains the MLP with cross-validated threshold, then restores the
    previously active context.

    Returns ``(resolved_count, trained_bool)``.
    """
    from flask import g
    from vtsearch.utils import (
        apply_label,
        build_media_lookup,
        resolve_media_ids,
        snapshot_medias,
    )

    # Override the request-scoped detector context so vote proxies
    # resolve to this model's context for the duration of this call.
    prev_det_ctx = getattr(g, "_detector_context", None)

    try:
        g._detector_context = det_ctx

        snap = snapshot_medias()
        if not snap:
            return 0, False

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)

        resolved = 0
        for entry in new_entries:
            label = entry.get("label", "")
            if label not in ("good", "bad"):
                continue
            cids = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
            for cid in cids:
                apply_label(cid, label)
            if cids:
                resolved += 1

        # Persist the updated votes back to the trainable-model file so the
        # labelset reflects any newly-resolved medias.
        sync_labels_to_loaded_model()

        # Retrain MLP if we have at least one good and one bad vote.
        from vtsearch.utils import bad_votes, good_votes

        trained = False
        if good_votes and bad_votes:
            from vtsearch.models.training import train_and_score
            from vtsearch.utils import (
                get_calibrate_count,
                get_calibration_fraction,
                get_inclusion,
                get_safe_thresholds,
            )

            _, threshold, model = train_and_score(
                snap,
                dict(good_votes),
                dict(bad_votes),
                get_inclusion(),
                safe_thresholds=get_safe_thresholds(),
                calibrate_count=get_calibrate_count(),
                calibration_fraction=get_calibration_fraction(),
            )
            if model is not None:
                det_ctx.model = model
                det_ctx.threshold = threshold
                # Cache voted media items with embeddings.
                training = {}
                for cid in list(good_votes) + list(bad_votes):
                    if cid in snap:
                        training[cid] = snap[cid]
                det_ctx.training_medias = training
                if snap:
                    first = next(iter(snap.values()), {})
                    det_ctx.embedder = first.get("embedder", "")
                    det_ctx.media_type = first.get("type", "")
                trained = True

        return resolved, trained

    finally:
        g._detector_context = prev_det_ctx


# ---------------------------------------------------------------------------
# Model registry endpoints
# ---------------------------------------------------------------------------


@trainable_models_bp.route("/api/models/registry")
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
        # No model requested — nothing to load.
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

    _LOAD_STEPS = 2  # restore labels, seed examples
    task_id = f"_modload_{model_id[:8]}"
    tracker = model_loading_tasks.create_task(
        task_id, entry.get("name", model_id), model_id=model_id,
        media_type=entry.get("media_type", ""),
    )
    tracker.update("loading", "Preparing…", 0, 0, step=1, total_steps=_LOAD_STEPS)

    # Capture values needed by the background thread.
    tm_name = entry.get("trainable_model_name", "")

    def load_task():
        from vtsearch.models.registry import add_loaded_model_id, remove_loaded_model_id

        try:
            labels_restored = 0
            examples_seeded = 0

            if tm_name:
                tracker.check_cancelled()
                tracker.update(
                    "loading", "Restoring labels…", 0, 0,
                    step=1, total_steps=_LOAD_STEPS,
                )
                tm_data = _read_model(_model_path(tm_name))
                if tm_data:
                    labels_restored = _restore_labels_from_trainable_model(tm_data)

                    tracker.check_cancelled()
                    tracker.update(
                        "loading", "Seeding examples…", 0, 0,
                        step=2, total_steps=_LOAD_STEPS,
                    )
                    examples_seeded = _seed_good_votes_from_examples(
                        tm_data.get("examples", [])
                    )

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


@trainable_models_bp.route("/api/models/registry/<model_id>/activate", methods=["POST"])
def activate_model_route(model_id: str):
    """Deprecated — no-op kept for backward compatibility.

    Active state is now request-scoped via ``X-Model-Id`` header.
    This endpoint validates that the model exists and is loaded,
    then returns success without mutating any global state.
    """
    from vtsearch.models.registry import get_model, is_model_loaded

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404
    if not is_model_loaded(model_id):
        return jsonify({"error": "Model is not loaded — load it first"}), 400

    return jsonify({"ok": True, "message": "Model activated"})


@trainable_models_bp.route("/api/models/registry/<model_id>/unload", methods=["POST"])
def unload_model_route(model_id: str):
    """Unload a model from memory (frees its DetectorContext).

    Saves labels before unloading if the model is active.
    """
    from vtsearch.models.registry import get_model, is_model_loaded, remove_loaded_model_id
    from vtsearch.utils import (
        bad_votes,
        get_active_detector_id,
        good_votes,
        unregister_detector_context,
    )

    entry = get_model(model_id)
    if entry is None:
        return jsonify({"error": "Model not found"}), 404
    if not is_model_loaded(model_id):
        return jsonify({"error": "Model is not loaded"}), 400

    # Save labels if this is the active model
    if get_active_detector_id() == model_id and (good_votes or bad_votes):
        sync_labels_to_loaded_model()

    # Remove from memory
    unregister_detector_context(model_id)
    remove_loaded_model_id(model_id)

    return jsonify({"ok": True, "message": "Model unloaded"})


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

    # Clean up the DetectorContext if loaded
    from vtsearch.models.registry import is_model_loaded, remove_loaded_model_id
    from vtsearch.utils import unregister_detector_context

    if is_model_loaded(model_id):
        unregister_detector_context(model_id)
        remove_loaded_model_id(model_id)

    unregister_model(model_id)
    return jsonify({"ok": True})


@trainable_models_bp.route("/api/models/loading-tasks")
def model_loading_tasks_endpoint():
    """Return all active model loading tasks with their progress."""
    from vtsearch.utils.progress import model_loading_tasks

    return jsonify({"tasks": model_loading_tasks.list_tasks()})


@trainable_models_bp.route("/api/models/cancel/<task_id>", methods=["POST"])
def cancel_model_loading_task(task_id: str):
    """Cancel a specific model loading task."""
    from vtsearch.utils.progress import model_loading_tasks

    ok = model_loading_tasks.cancel_task(task_id)
    if not ok:
        return jsonify({"error": "Task not found"}), 404
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

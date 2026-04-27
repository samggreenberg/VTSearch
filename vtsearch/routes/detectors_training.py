"""Detector training routes: vote-based and label-based detector training."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.models.detector_training import serialize_weights, train_and_threshold, validate_good_bad_split
from vtsearch.routes.helpers import (
    extract_plugin_fields,
    get_json_or_400,
    get_json_safe,
    run_plugin_or_error,
    validate_filepath_field,
)
from vtsearch.models import collect_media_origins
from vtsearch.utils import (
    add_autorun_detector,
    get_inclusion,
    snapshot_medias,
)
import vtsearch.utils.paths as _paths

from vtsearch.routes.detectors_crud import get_detectors_dir

detectors_training_bp = Blueprint("detectors_training", __name__)


# ---------------------------------------------------------------------------
# Vote-based training
# ---------------------------------------------------------------------------


@detectors_training_bp.route("/api/detector/export-server", methods=["POST"])
def export_detector_server():
    """Save current votes as a detector file on the server filesystem.

    Stores the origin information of voted medias and the inclusion
    setting, rather than serialised MLP weights.  When the detector is
    later loaded, the weights are re-derived by resolving the original
    media files, embedding them, and training an MLP.

    Expects JSON body with ``name`` (required) and optionally ``overwrite``
    (bool, default false).  The detector is saved as
    ``data/detectors/<name>.json``.  If the file already exists and
    ``overwrite`` is false, returns ``{"exists": true, "path": ...}`` with
    status 409 so the client can ask the user whether to overwrite.
    """
    from vtsearch.utils import bad_votes, good_votes

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    overwrite = data.get("overwrite", False)

    # Sanitise the filename: only allow alphanumeric, hyphen, underscore, space
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ")
    if not safe_name:
        return jsonify({"error": "name contains no valid characters"}), 400

    det_dir = get_detectors_dir()
    det_dir.mkdir(parents=True, exist_ok=True)
    filepath = det_dir / f"{safe_name}.json"

    # Check for existing file before doing expensive training
    if filepath.exists() and not overwrite:
        return jsonify({"exists": True, "name": safe_name}), 409

    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    snap = snapshot_medias()

    # Collect origin info from voted medias
    good_origins = collect_media_origins(good_votes, snap)
    bad_origins = collect_media_origins(bad_votes, snap)

    if not good_origins or not bad_origins:
        return jsonify({"error": "voted medias no longer loaded — reload the dataset"}), 400

    # Determine media type from current medias
    media_type = "audio"
    if snap:
        media_type = next(iter(snap.values())).get("type", "audio")

    inclusion = get_inclusion()

    detector_data = {
        "good_origins": good_origins,
        "bad_origins": bad_origins,
        "inclusion": inclusion,
        "media_type": media_type,
        "name": safe_name,
    }

    filepath.write_text(json.dumps(detector_data, indent=2), encoding="utf-8")

    return jsonify(
        {
            "success": True,
            "name": safe_name,
            "media_type": media_type,
        }
    )


# ---------------------------------------------------------------------------
# Label-based training
# ---------------------------------------------------------------------------


@detectors_training_bp.route("/api/autorun-detectors/import-labels", methods=["POST"])
def import_detector_labels():
    """Import a autorun detector by training on a label file.

    The label file is a JSON object with a ``"labels"`` list. Each entry has
    ``"path"`` (or ``"file"``/``"filename"``) and ``"label"`` (``"good"`` or
    ``"bad"``). Media type is inferred from file extensions; you may also pass
    ``media_type`` as a form field to force a specific type.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    name = request.form.get("name", "").strip()
    if not name:
        name = Path(file.filename).stem

    # Optional explicit media type override
    media_type_hint = request.form.get("media_type", "").strip()

    # Use the media type registry for extension → type lookup and embedding.
    from vtsearch.media import embedders_for_type, get_by_extension

    def _media_type_for_path(p: Path) -> str | None:
        mt = get_by_extension(p.suffix)
        return mt.type_id if mt else None

    def _embed(media_type: str, p: Path):
        avail = embedders_for_type(media_type)
        if not avail:
            return None
        return avail[0].embed_media(p)

    try:
        text = file.read().decode("utf-8")
        try:
            label_data = json.loads(text)
        except Exception:
            return jsonify({"error": "Invalid label file format"}), 400

        labels = label_data.get("labels", [])
        if not labels:
            return jsonify({"error": "No labels found in file"}), 400

        X_list: list = []
        y_list: list = []
        loaded_count = 0
        skipped_count = 0
        detected_media_type: str | None = media_type_hint or None
        _file_base = _paths.get_file_access_base_dir()

        for entry in labels:
            label = entry.get("label")
            if label not in ("good", "bad"):
                skipped_count += 1
                continue

            file_path_str = entry.get("path") or entry.get("file") or entry.get("filename")
            if not file_path_str:
                skipped_count += 1
                continue

            file_path = Path(file_path_str)
            # Ensure the path doesn't escape the allowed directory
            try:
                _paths.validate_server_filepath(file_path_str, base_dir=_file_base)
            except ValueError:
                skipped_count += 1
                continue
            if not file_path.exists():
                skipped_count += 1
                continue

            # Resolve media type for this entry
            mt = media_type_hint or _media_type_for_path(file_path)
            if mt is None:
                skipped_count += 1
                continue

            # Enforce a single media type across all entries
            if detected_media_type is None:
                detected_media_type = mt
            elif detected_media_type != mt:
                skipped_count += 1
                continue

            embedding = _embed(mt, file_path)
            if embedding is None:
                skipped_count += 1
                continue

            X_list.append(embedding)
            y_list.append(1.0 if label == "good" else 0.0)
            loaded_count += 1

        if loaded_count < 2:
            return (
                jsonify(
                    {"error": (f"Need at least 2 valid labeled files (loaded {loaded_count}, skipped {skipped_count})")}
                ),
                400,
            )

        try:
            validate_good_bad_split(y_list)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        snap = snapshot_medias()
        model, threshold = train_and_threshold(X_list, y_list, snap=snap)
        weights = serialize_weights(model)

        final_media_type = detected_media_type or "audio"
        add_autorun_detector(name, final_media_type, weights, threshold, created_by=get_current_user())
        return jsonify(
            {
                "success": True,
                "name": name,
                "media_type": final_media_type,
                "loaded": loaded_count,
                "skipped": skipped_count,
            }
        )

    except Exception:
        import logging

        logging.getLogger(__name__).exception("Detector training from file failed")
        return jsonify({"error": "Detector training from file failed"}), 500


@detectors_training_bp.route("/api/autorun-detectors/from-label-import/<importer_name>", methods=["POST"])
def train_from_label_import(importer_name: str):
    """Train a detector from label importer results without modifying votes.

    Runs the named label importer to get ``[{md5, label}, ...]``, matches the
    md5s to loaded medias, uses the matched embeddings to train a detector, and
    saves it as an autorun detector.  Current votes are *not* modified.
    """
    from vtsearch.labels.importers import get_label_importer, list_label_importers
    from vtsearch.routes.helpers import get_plugin_or_404

    importer, err = get_plugin_or_404(get_label_importer, list_label_importers, importer_name, "label importer")
    if err:
        return err

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded. Load a dataset first."}), 400

    field_values = extract_plugin_fields(importer)

    # name comes from form data or JSON body (not a plugin field)
    has_file_fields = any(f.field_type == "file" for f in importer.fields)
    if has_file_fields:
        name = request.form.get("name", "").strip()
    else:
        name = get_json_safe().get("name", "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    err = validate_filepath_field(field_values)
    if err:
        return err

    label_entries, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err

    if not isinstance(label_entries, list) or not label_entries:
        return jsonify({"error": "Label importer returned no entries."}), 400

    # Match md5s to loaded medias and collect embeddings
    from vtsearch.utils import build_media_lookup, resolve_media_ids

    origin_lookup, md5_lookup, _ = build_media_lookup(snap)

    X_list: list = []
    y_list: list = []
    good_cids: list[int] = []
    bad_cids: list[int] = []
    loaded_count = 0
    skipped_count = 0

    for entry in label_entries:
        label = entry.get("label", "").lower()
        if label not in ("good", "bad"):
            skipped_count += 1
            continue

        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            skipped_count += 1
            continue

        # Use the first matching media's embedding
        cid = cids[0]
        embedding = snap[cid].get("embedding")
        if embedding is None:
            skipped_count += 1
            continue

        X_list.append(embedding)
        y_list.append(1.0 if label == "good" else 0.0)
        if label == "good":
            good_cids.append(cid)
        else:
            bad_cids.append(cid)
        loaded_count += 1

    if loaded_count < 2:
        return (
            jsonify({"error": f"Need at least 2 matched medias (matched {loaded_count}, skipped {skipped_count})"}),
            400,
        )

    try:
        validate_good_bad_split(y_list)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model, threshold = train_and_threshold(X_list, y_list, snap=snap)
    weights = serialize_weights(model)

    media_type = next(iter(snap.values())).get("type", "audio")
    good_origins = collect_media_origins(good_cids, snap)
    bad_origins = collect_media_origins(bad_cids, snap)
    inclusion = get_inclusion()
    add_autorun_detector(
        name,
        media_type,
        weights,
        threshold,
        created_by=get_current_user(),
        good_origins=good_origins,
        bad_origins=bad_origins,
        inclusion=inclusion,
    )

    # Register in the persistent model registry for the dashboard grid.
    from vtsearch.models.registry import find_by_detector_name, register_model

    if not find_by_detector_name(name):
        register_model(
            name=name,
            media_type=media_type,
            trainable=False,
            num_training=loaded_count,
            detector_name=name,
            created_by=get_current_user(),
        )

    return jsonify(
        {
            "success": True,
            "name": name,
            "media_type": media_type,
            "loaded": loaded_count,
            "skipped": skipped_count,
        }
    )

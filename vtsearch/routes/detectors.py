"""Blueprint for detector, extractor, and localizer routes."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.models import (
    build_model_from_weights,
    calculate_cross_calibration_threshold,
    calculate_safe_threshold,
    train_model,
)
from vtsearch.utils import (
    add_favorite_detector,
    add_favorite_extractor,
    add_favorite_localizer,
    medias,
    get_calibrate_count,
    get_calibration_fraction,
    get_favorite_detectors,
    get_favorite_detectors_by_media,
    get_favorite_extractors,
    get_favorite_extractors_by_media,
    get_favorite_localizers,
    get_favorite_localizers_by_media,
    get_inclusion,
    get_safe_thresholds,
    remove_favorite_detector,
    remove_favorite_extractor,
    remove_favorite_localizer,
    rename_favorite_detector,
    rename_favorite_extractor,
    rename_favorite_localizer,
)

detectors_bp = Blueprint("detectors", __name__)


# ---------------------------------------------------------------------------
# Detector routes
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/detector/export", methods=["POST"])
def export_detector():
    """Train MLP on current votes and export the model weights."""
    import torch  # noqa: PLC0415

    from vtsearch.utils import bad_votes, good_votes

    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    # Train the model
    X_list = []
    y_list = []
    for cid in good_votes:
        if cid in medias:
            X_list.append(medias[cid]["embedding"])
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in medias:
            X_list.append(medias[cid]["embedding"])
            y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    input_dim = X.shape[1]

    # Calculate threshold using k-fold calibration with inclusion
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        get_inclusion(),
        calibrate_count=get_calibrate_count(),
        calibration_fraction=get_calibration_fraction(),
    )

    # Train final model on all data with inclusion
    model = train_model(X, y, input_dim, get_inclusion())

    # Apply safe thresholds blending if enabled
    if get_safe_thresholds():
        all_ids = sorted(medias.keys())
        all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)
        with torch.no_grad():
            all_scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()
        threshold = calculate_safe_threshold(threshold, all_scores, len(X_list))

    # Extract model weights
    state_dict = model.state_dict()
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.tolist()

    return jsonify({"weights": weights, "threshold": round(threshold, 4)})


# ---------------------------------------------------------------------------
# Server-side detector file export (ServerFileProcessorExporter)
# ---------------------------------------------------------------------------

#: Default directory for server-side detector files.
SERVER_DETECTOR_DIR = Path("data/detectors")


@detectors_bp.route("/api/detector/export-server", methods=["POST"])
def export_detector_server():
    """Train MLP on current votes and save to a file on the server filesystem.

    Expects JSON body with ``name`` (required) and optionally ``overwrite``
    (bool, default false).  The detector is saved as
    ``data/detectors/<name>.json``.  If the file already exists and
    ``overwrite`` is false, returns ``{"exists": true, "path": ...}`` with
    status 409 so the client can ask the user whether to overwrite.
    """
    import torch  # noqa: PLC0415

    from vtsearch.utils import bad_votes, good_votes

    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    overwrite = data.get("overwrite", False)

    # Sanitise the filename: only allow alphanumeric, hyphen, underscore, space
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ")
    if not safe_name:
        return jsonify({"error": "name contains no valid characters"}), 400

    SERVER_DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SERVER_DETECTOR_DIR / f"{safe_name}.json"

    # Check for existing file before doing expensive training
    if filepath.exists() and not overwrite:
        return jsonify({"exists": True, "path": str(filepath.resolve()), "name": safe_name}), 409

    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    # Train the model (same logic as export_detector)
    X_list = []
    y_list = []
    for cid in good_votes:
        if cid in medias:
            X_list.append(medias[cid]["embedding"])
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in medias:
            X_list.append(medias[cid]["embedding"])
            y_list.append(0.0)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        get_inclusion(),
        calibrate_count=get_calibrate_count(),
        calibration_fraction=get_calibration_fraction(),
    )
    model = train_model(X, y, input_dim, get_inclusion())

    if get_safe_thresholds():
        all_ids = sorted(medias.keys())
        all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)
        with torch.no_grad():
            all_scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()
        threshold = calculate_safe_threshold(threshold, all_scores, len(X_list))

    state_dict = model.state_dict()
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.tolist()

    # Determine media type from current medias
    media_type = "audio"
    if medias:
        media_type = next(iter(medias.values())).get("type", "audio")

    detector_data = {
        "weights": weights,
        "threshold": round(threshold, 4),
        "media_type": media_type,
        "name": safe_name,
    }

    filepath.write_text(json.dumps(detector_data, indent=2), encoding="utf-8")

    return jsonify({
        "success": True,
        "name": safe_name,
        "path": str(filepath.resolve()),
        "threshold": round(threshold, 4),
        "media_type": media_type,
    })


@detectors_bp.route("/api/detector/server-files", methods=["GET"])
def list_server_detector_files():
    """List detector JSON files saved on the server in data/detectors/."""
    if not SERVER_DETECTOR_DIR.is_dir():
        return jsonify({"files": []})

    files = []
    for p in sorted(SERVER_DETECTOR_DIR.glob("*.json")):
        files.append({
            "name": p.stem,
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size,
        })
    return jsonify({"files": files})


@detectors_bp.route("/api/detector-sort", methods=["POST"])
def detector_sort():
    """Score all medias using a loaded detector model."""
    import torch  # noqa: PLC0415

    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    detector = data.get("detector")
    if not detector:
        return jsonify({"error": "detector is required"}), 400

    weights = detector.get("weights")
    threshold = detector.get("threshold", 0.5)

    if not weights:
        return jsonify({"error": "detector weights are required"}), 400

    # Reconstruct the model from weights
    model = build_model_from_weights(weights)

    # Score every media
    all_ids = sorted(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


# ---------------------------------------------------------------------------
# Favorite detectors
# ---------------------------------------------------------------------------


@detectors_bp.route("/api/favorite-detectors")
def get_favorite_detectors_route():
    """Get all favorite detectors."""
    detectors = get_favorite_detectors()
    return jsonify({"detectors": list(detectors.values())})


@detectors_bp.route("/api/favorite-detectors", methods=["POST"])
def add_favorite_detector_route():
    """Add a new favorite detector."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    name = data.get("name", "").strip()
    media_type = data.get("media_type", "").strip()
    weights = data.get("weights")
    threshold = data.get("threshold", 0.5)

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not weights:
        return jsonify({"error": "weights are required"}), 400

    add_favorite_detector(name, media_type, weights, threshold)
    return jsonify({"success": True, "name": name})


@detectors_bp.route("/api/favorite-detectors/<name>", methods=["DELETE"])
def delete_favorite_detector_route(name):
    """Delete a favorite detector."""
    if remove_favorite_detector(name):
        return jsonify({"success": True})
    return jsonify({"error": "Detector not found"}), 404


@detectors_bp.route("/api/favorite-detectors/<name>/rename", methods=["PUT"])
def rename_favorite_detector_route(name):
    """Rename a favorite detector."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_favorite_detector(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Detector not found or new name already exists"}), 400


@detectors_bp.route("/api/favorite-detectors/import-pkl", methods=["POST"])
def import_detector_pkl():
    """Import a favorite detector from a PKL file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    name = request.form.get("name", "").strip()
    if not name:
        # Use filename without extension as default name
        name = Path(file.filename).stem

    try:
        # Read the JSON detector file
        text = file.read().decode("utf-8")
        detector_data = json.loads(text)

        weights = detector_data.get("weights")
        threshold = detector_data.get("threshold", 0.5)

        if not weights:
            return jsonify({"error": "Invalid detector file format"}), 400

        # Prefer media_type stored in the file; fall back to current medias, then "audio"
        media_type = detector_data.get("media_type", "")
        if not media_type:
            if medias:
                media_type = next(iter(medias.values())).get("type", "audio")
            else:
                media_type = "audio"

        add_favorite_detector(name, media_type, weights, threshold)
        return jsonify({"success": True, "name": name, "media_type": media_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@detectors_bp.route("/api/favorite-detectors/import-labels", methods=["POST"])
def import_detector_labels():
    """Import a favorite detector by training on a label file.

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
    from vtsearch.media import get as media_get
    from vtsearch.media import get_by_extension

    def _media_type_for_path(p: Path) -> str | None:
        mt = get_by_extension(p.suffix)
        return mt.type_id if mt else None

    def _embed(media_type: str, p: Path):
        try:
            return media_get(media_type).embed_media(p)
        except KeyError:
            return None

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

        num_good = sum(1 for y in y_list if y == 1.0)
        num_bad = len(y_list) - num_good
        if num_good == 0 or num_bad == 0:
            return (
                jsonify({"error": "Need at least one good and one bad labeled example"}),
                400,
            )

        import torch  # noqa: PLC0415

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        input_dim = X.shape[1]

        threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, get_inclusion())
        model = train_model(X, y, input_dim, get_inclusion())

        # Apply safe thresholds blending if enabled
        if get_safe_thresholds() and medias:
            all_ids = sorted(medias.keys())
            all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
            X_all = torch.tensor(all_embs, dtype=torch.float32)
            with torch.no_grad():
                all_scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()
            threshold = calculate_safe_threshold(threshold, all_scores, len(y_list))

        state_dict = model.state_dict()
        weights = {}
        for key, value in state_dict.items():
            weights[key] = value.tolist()

        final_media_type = detected_media_type or "audio"
        add_favorite_detector(name, final_media_type, weights, threshold)
        return jsonify(
            {
                "success": True,
                "name": name,
                "media_type": final_media_type,
                "loaded": loaded_count,
                "skipped": skipped_count,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@detectors_bp.route("/api/favorite-detectors/from-label-import/<importer_name>", methods=["POST"])
def train_from_label_import(importer_name: str):
    """Train a detector from label importer results without modifying votes.

    Runs the named label importer to get ``[{md5, label}, ...]``, matches the
    md5s to loaded medias, uses the matched embeddings to train a detector, and
    saves it as a favorite.  Current votes are *not* modified.
    """
    from vtsearch.labels.importers import get_label_importer, list_label_importers

    importer = get_label_importer(importer_name)
    if importer is None:
        known = [imp.name for imp in list_label_importers()]
        return (
            jsonify({"error": f"Unknown label importer '{importer_name}'. Available: {known}"}),
            404,
        )

    if not medias:
        return jsonify({"error": "No medias loaded. Load a dataset first."}), 400

    # Build field_values from multipart form data
    has_file_fields = any(f.field_type == "file" for f in importer.fields)
    field_values: dict = {}

    if has_file_fields:
        for f in importer.fields:
            if f.field_type == "file":
                field_values[f.key] = request.files.get(f.key)
            else:
                field_values[f.key] = request.form.get(f.key, f.default if f.default is not None else "")
        name = request.form.get("name", "").strip()
    else:
        body = request.get_json(force=True, silent=True) or {}
        for f in importer.fields:
            field_values[f.key] = body.get(f.key, f.default if f.default is not None else "")
        name = body.get("name", "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    try:
        label_entries = importer.run(field_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Import failed: {exc}"}), 500

    if not isinstance(label_entries, list) or not label_entries:
        return jsonify({"error": "Label importer returned no entries."}), 400

    # Match md5s to loaded medias and collect embeddings
    from vtsearch.utils import build_media_lookup, resolve_media_ids

    origin_lookup, md5_lookup = build_media_lookup(medias)

    X_list: list = []
    y_list: list = []
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
        embedding = medias[cid].get("embedding")
        if embedding is None:
            skipped_count += 1
            continue

        X_list.append(embedding)
        y_list.append(1.0 if label == "good" else 0.0)
        loaded_count += 1

    if loaded_count < 2:
        return (
            jsonify({"error": f"Need at least 2 matched medias (matched {loaded_count}, skipped {skipped_count})"}),
            400,
        )

    num_good = sum(1 for y in y_list if y == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        return jsonify({"error": "Need at least one good and one bad labeled example"}), 400

    import torch  # noqa: PLC0415

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, get_inclusion())
    model = train_model(X, y, input_dim, get_inclusion())

    state_dict = model.state_dict()
    weights = {}
    for key, value in state_dict.items():
        weights[key] = value.tolist()

    media_type = next(iter(medias.values())).get("type", "audio")
    add_favorite_detector(name, media_type, weights, threshold)

    return jsonify(
        {
            "success": True,
            "name": name,
            "media_type": media_type,
            "loaded": loaded_count,
            "skipped": skipped_count,
        }
    )


@detectors_bp.route("/api/auto-detect", methods=["POST"])
def auto_detect():
    """Run all favorite detectors for the current media type and return positive hits."""
    if not medias:
        return jsonify({"error": "No medias loaded"}), 400

    # Import any favorite processors from settings that aren't already loaded
    from vtsearch.settings import ensure_favorite_processors_imported

    newly_imported = ensure_favorite_processors_imported()

    # Determine media type from current medias
    media_type = next(iter(medias.values())).get("type", "audio")

    # Get favorite detectors for this media type
    detectors = get_favorite_detectors_by_media(media_type)

    if not detectors:
        return jsonify({"error": f"No favorite detectors found for media type: {media_type}"}), 400

    # Prepare shared data for all detectors
    import torch  # noqa: PLC0415

    all_ids = sorted(medias.keys())
    all_embs = np.array([medias[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    def _run_single_detector(detector_name, detector_data):
        """Run a single detector and return (name, result_dict)."""
        weights = detector_data["weights"]
        threshold = detector_data["threshold"]

        model = build_model_from_weights(weights)

        with torch.no_grad():
            scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

        positive_hits = []
        negative_hits = []
        for cid, score in zip(all_ids, scores):
            clip_info = medias[cid].copy()
            clip_info.pop("embedding", None)
            clip_info.pop("media_bytes", None)
            clip_info.pop("media_string", None)
            clip_info["score"] = round(score, 4)
            if score >= threshold:
                positive_hits.append(clip_info)
            else:
                negative_hits.append(clip_info)

        positive_hits.sort(key=lambda x: x["score"], reverse=True)
        negative_hits.sort(key=lambda x: x["score"], reverse=True)

        return detector_name, {
            "detector_name": detector_name,
            "threshold": round(threshold, 4),
            "total_hits": len(positive_hits),
            "hits": positive_hits,
            "negative_hits": negative_hits,
        }

    # Run all detectors in parallel (PyTorch releases GIL during tensor ops)
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(detectors), 8)) as pool:
        futures = [pool.submit(_run_single_detector, name, data) for name, data in detectors.items()]
        for future in futures:
            name, result = future.result()
            results[name] = result

    response: dict = {
        "media_type": media_type,
        "detectors_run": len(detectors),
        "results": results,
    }
    if newly_imported:
        response["newly_imported"] = newly_imported

    return jsonify(response)


# ---------------------------------------------------------------------------
# Extractor routes
# ---------------------------------------------------------------------------

# Registry of extractor type constructors.
# Each entry maps an extractor_type string to a callable(name, config) -> Extractor.
_EXTRACTOR_FACTORIES: dict = {}


def _ensure_extractor_factories():
    """Populate the factory registry on first use (lazy to avoid import cycles)."""
    if _EXTRACTOR_FACTORIES:
        return
    from vtsearch.media.image.extractor import ImageClassExtractor

    _EXTRACTOR_FACTORIES["image_class"] = ImageClassExtractor.from_config

    from vtsearch.media.image.ocr_extractor import OCRExtractor

    _EXTRACTOR_FACTORIES["ocr"] = OCRExtractor.from_config

    from vtsearch.media.audio.speech_extractor import SpeechExtractor

    _EXTRACTOR_FACTORIES["speech"] = SpeechExtractor.from_config


def _build_extractor(name: str, extractor_type: str, config: dict):
    """Instantiate an Extractor from its serialised form."""
    _ensure_extractor_factories()
    factory = _EXTRACTOR_FACTORIES.get(extractor_type)
    if factory is None:
        raise ValueError(f"Unknown extractor_type: {extractor_type!r}")
    return factory(name, config)


@detectors_bp.route("/api/favorite-extractors")
def get_favorite_extractors_route():
    """Get all favorite extractors."""
    extractors = get_favorite_extractors()
    return jsonify({"extractors": list(extractors.values())})


@detectors_bp.route("/api/favorite-extractors", methods=["POST"])
def add_favorite_extractor_route():
    """Add a new favorite extractor."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    media_type = data.get("media_type", "").strip()
    config = data.get("config")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    # Validate that the extractor can be built from this config
    try:
        _build_extractor(name, extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    add_favorite_extractor(name, extractor_type, media_type, config)
    return jsonify({"success": True, "name": name})


@detectors_bp.route("/api/favorite-extractors/<name>", methods=["DELETE"])
def delete_favorite_extractor_route(name):
    """Delete a favorite extractor."""
    if remove_favorite_extractor(name):
        return jsonify({"success": True})
    return jsonify({"error": "Extractor not found"}), 404


@detectors_bp.route("/api/favorite-extractors/<name>/rename", methods=["PUT"])
def rename_favorite_extractor_route(name):
    """Rename a favorite extractor."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_favorite_extractor(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Extractor not found or new name already exists"}), 400


@detectors_bp.route("/api/extract", methods=["POST"])
def run_extract():
    """Run a single extractor on all medias and return per-media extraction results."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    extractor_name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    config = data.get("config")

    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    if not medias:
        return jsonify({"error": "No medias loaded"}), 400

    try:
        extractor = _build_extractor(extractor_name or "adhoc", extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    media_type = next(iter(medias.values())).get("type", "")
    if extractor.media_type != media_type:
        return (
            jsonify({"error": f"Extractor media type '{extractor.media_type}' does not match medias '{media_type}'"}),
            400,
        )

    results = []
    for media_id in sorted(medias.keys()):
        media = medias[media_id]
        extractions = extractor.extract(media)
        if extractions:
            clip_info = {
                k: v
                for k, v in media.items()
                if k not in ("embedding", "media_bytes", "media_string")
            }
            clip_info["extractions"] = extractions
            results.append(clip_info)

    return jsonify(
        {
            "extractor_name": extractor.name,
            "media_type": media_type,
            "total_medias_with_hits": len(results),
            "results": results,
        }
    )


@detectors_bp.route("/api/auto-extract", methods=["POST"])
def auto_extract():
    """Run all favorite extractors for the current media type and return extraction results."""
    if not medias:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(medias.values())).get("type", "")
    extractors = get_favorite_extractors_by_media(media_type)

    if not extractors:
        return jsonify({"error": f"No favorite extractors found for media type: {media_type}"}), 400

    sorted_media_ids = sorted(medias.keys())

    def _run_single_extractor(ext_name, ext_data):
        """Run a single extractor on all medias and return (name, result_dict) or None."""
        try:
            extractor = _build_extractor(ext_name, ext_data["extractor_type"], ext_data["config"])
        except Exception:
            return None

        ext_results = []
        for media_id in sorted_media_ids:
            media = medias[media_id]
            extractions = extractor.extract(media)
            if extractions:
                clip_info = {
                    k: v
                    for k, v in media.items()
                    if k not in ("embedding", "media_bytes", "media_string")
                }
                clip_info["extractions"] = extractions
                ext_results.append(clip_info)

        return ext_name, {
            "extractor_name": ext_name,
            "total_medias_with_hits": len(ext_results),
            "results": ext_results,
        }

    # Run all extractors in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(extractors), 8)) as pool:
        futures = [pool.submit(_run_single_extractor, name, data) for name, data in extractors.items()]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result

    return jsonify(
        {
            "media_type": media_type,
            "extractors_run": len(results),
            "results": results,
        }
    )


# ---------------------------------------------------------------------------
# Localizer routes
# ---------------------------------------------------------------------------

# Registry of localizer type constructors.
_LOCALIZER_FACTORIES: dict = {}


def _ensure_localizer_factories():
    """Populate the localizer factory registry on first use."""
    if _LOCALIZER_FACTORIES:
        return
    from vtsearch.media.image.face_localizer import FaceLocalizer

    _LOCALIZER_FACTORIES["face"] = FaceLocalizer.from_config


def _build_localizer(name: str, localizer_type: str, config: dict):
    """Instantiate a Localizer from its serialised form."""
    _ensure_localizer_factories()
    factory = _LOCALIZER_FACTORIES.get(localizer_type)
    if factory is None:
        raise ValueError(f"Unknown localizer_type: {localizer_type!r}")
    return factory(name, config)


@detectors_bp.route("/api/favorite-localizers")
def get_favorite_localizers_route():
    """Get all favorite localizers."""
    localizers = get_favorite_localizers()
    return jsonify({"localizers": list(localizers.values())})


@detectors_bp.route("/api/favorite-localizers", methods=["POST"])
def add_favorite_localizer_route():
    """Add a new favorite localizer."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    name = data.get("name", "").strip()
    localizer_type = data.get("localizer_type", "").strip()
    media_type = data.get("media_type", "").strip()
    config = data.get("config")

    if not name:
        return jsonify({"error": "name is required"}), 400
    if not localizer_type:
        return jsonify({"error": "localizer_type is required"}), 400
    if not media_type:
        return jsonify({"error": "media_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    try:
        _build_localizer(name, localizer_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid localizer config: {e}"}), 400

    add_favorite_localizer(name, localizer_type, media_type, config)
    return jsonify({"success": True, "name": name})


@detectors_bp.route("/api/favorite-localizers/<name>", methods=["DELETE"])
def delete_favorite_localizer_route(name):
    """Delete a favorite localizer."""
    if remove_favorite_localizer(name):
        return jsonify({"success": True})
    return jsonify({"error": "Localizer not found"}), 404


@detectors_bp.route("/api/favorite-localizers/<name>/rename", methods=["PUT"])
def rename_favorite_localizer_route(name):
    """Rename a favorite localizer."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_name = data.get("new_name", "").strip()
    if not new_name:
        return jsonify({"error": "new_name is required"}), 400

    if rename_favorite_localizer(name, new_name):
        return jsonify({"success": True, "new_name": new_name})
    return jsonify({"error": "Localizer not found or new name already exists"}), 400


@detectors_bp.route("/api/localize", methods=["POST"])
def run_localize():
    """Run a single localizer on all clips and return per-clip localization results."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    localizer_name = data.get("name", "").strip()
    localizer_type = data.get("localizer_type", "").strip()
    config = data.get("config")

    if not localizer_type:
        return jsonify({"error": "localizer_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    if not medias:
        return jsonify({"error": "No medias loaded"}), 400

    try:
        localizer = _build_localizer(localizer_name or "adhoc", localizer_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid localizer config: {e}"}), 400

    media_type = next(iter(medias.values())).get("type", "")
    if localizer.media_type != media_type:
        return (
            jsonify({"error": f"Localizer media type '{localizer.media_type}' does not match medias '{media_type}'"}),
            400,
        )

    results = []
    for media_id in sorted(medias.keys()):
        media = medias[media_id]
        localizations = localizer.localize(media)
        if localizations:
            media_info = {k: v for k, v in media.items() if k not in ("embedding", "media_bytes", "media_string")}
            media_info["localizations"] = localizations
            results.append(media_info)

    return jsonify(
        {
            "localizer_name": localizer.name,
            "media_type": media_type,
            "total_medias_with_hits": len(results),
            "results": results,
        }
    )


@detectors_bp.route("/api/auto-localize", methods=["POST"])
def auto_localize():
    """Run all favorite localizers for the current media type."""
    if not medias:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(medias.values())).get("type", "")
    localizers = get_favorite_localizers_by_media(media_type)

    if not localizers:
        return jsonify({"error": f"No favorite localizers found for media type: {media_type}"}), 400

    sorted_media_ids = sorted(medias.keys())

    def _run_single_localizer(loc_name, loc_data):
        try:
            localizer = _build_localizer(loc_name, loc_data["localizer_type"], loc_data["config"])
        except Exception:
            return None

        loc_results = []
        for media_id in sorted_media_ids:
            media = medias[media_id]
            localizations = localizer.localize(media)
            if localizations:
                media_info = {k: v for k, v in media.items() if k not in ("embedding", "media_bytes", "media_string")}
                media_info["localizations"] = localizations
                loc_results.append(media_info)

        return loc_name, {
            "localizer_name": loc_name,
            "total_medias_with_hits": len(loc_results),
            "results": loc_results,
        }

    results = {}
    with ThreadPoolExecutor(max_workers=min(len(localizers), 8)) as pool:
        futures = [pool.submit(_run_single_localizer, name, data) for name, data in localizers.items()]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result

    return jsonify(
        {
            "media_type": media_type,
            "localizers_run": len(results),
            "results": results,
        }
    )


# ---------------------------------------------------------------------------
# Pregen Processors
# ---------------------------------------------------------------------------

# Default pregen processor definitions: (name, processor_type, kind, media_type, config)
# kind is "extractor" or "localizer"
_PREGEN_PROCESSORS = [
    {
        "name": "OCR (PaddleOCR)",
        "kind": "extractor",
        "processor_type": "ocr",
        "media_type": "image",
        "config": {"language": "en", "threshold": 0.5},
    },
    {
        "name": "Speech (Whisper Tiny)",
        "kind": "extractor",
        "processor_type": "speech",
        "media_type": "audio",
        "config": {"model_size": "tiny", "language": None},
    },
    {
        "name": "Face (MediaPipe)",
        "kind": "localizer",
        "processor_type": "face",
        "media_type": "image",
        "config": {"threshold": 0.5, "model_selection": 1},
    },
]


@detectors_bp.route("/api/pregen-processors", methods=["GET"])
def list_pregen_processors():
    """Return the list of available pregen processors."""
    return jsonify({"processors": _PREGEN_PROCESSORS})


@detectors_bp.route("/api/pregen-processors/add", methods=["POST"])
def add_pregen_processors():
    """Add all pregen processors as favorites.

    Registers the OCR extractor, Speech extractor, and Face localizer
    into the favorite extractors and localizers stores.
    """
    added = []
    for proc in _PREGEN_PROCESSORS:
        name = proc["name"]
        kind = proc["kind"]
        media_type = proc["media_type"]
        config = proc["config"]
        processor_type = proc["processor_type"]

        if kind == "extractor":
            add_favorite_extractor(name, processor_type, media_type, config)
        elif kind == "localizer":
            add_favorite_localizer(name, processor_type, media_type, config)
        added.append(name)

    return jsonify({"success": True, "added": added})

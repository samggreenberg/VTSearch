"""Detector training routes: vote-based, label-based, and multi-dataset/model find."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import get_json_or_400
from vtsearch.models import (
    build_model_from_weights,
    calculate_cross_calibration_threshold,
    calculate_safe_threshold,
    train_model,
)
from vtsearch.utils import (
    add_autorun_detector,
    get_autorun_detectors,
    get_calibrate_count,
    get_calibration_fraction,
    get_inclusion,
    get_safe_thresholds,
    medias,
)
from vtsearch.utils.paths import validate_server_filepath

from vtsearch.routes.detectors_crud import get_detectors_dir

detectors_training_bp = Blueprint("detectors_training", __name__)


# ---------------------------------------------------------------------------
# Vote-based training
# ---------------------------------------------------------------------------


@detectors_training_bp.route("/api/detector/export", methods=["POST"])
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

    if not X_list:
        return jsonify({"error": "voted medias no longer loaded — reload the dataset"}), 400

    num_good = sum(1 for y_val in y_list if y_val == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        return jsonify({"error": "need at least one good and one bad vote with loaded medias"}), 400

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


@detectors_training_bp.route("/api/detector/export-server", methods=["POST"])
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

    if not X_list:
        return jsonify({"error": "voted medias no longer loaded — reload the dataset"}), 400

    num_good = sum(1 for y_val in y_list if y_val == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        return jsonify({"error": "need at least one good and one bad vote with loaded medias"}), 400

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

    return jsonify(
        {
            "success": True,
            "name": safe_name,
            "path": str(filepath.resolve()),
            "threshold": round(threshold, 4),
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

        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
        )
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
        add_autorun_detector(name, final_media_type, weights, threshold)
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


@detectors_training_bp.route("/api/autorun-detectors/from-label-import/<importer_name>", methods=["POST"])
def train_from_label_import(importer_name: str):
    """Train a detector from label importer results without modifying votes.

    Runs the named label importer to get ``[{md5, label}, ...]``, matches the
    md5s to loaded medias, uses the matched embeddings to train a detector, and
    saves it as an autorun detector.  Current votes are *not* modified.
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

    # Validate server file paths to prevent path traversal
    if "filepath" in field_values and str(field_values["filepath"]).strip():
        try:
            validate_server_filepath(str(field_values["filepath"]))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

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
    add_autorun_detector(name, media_type, weights, threshold)

    # Register in the persistent model registry for the dashboard grid.
    from vtsearch.models.registry import find_by_detector_name, register_model

    if not find_by_detector_name(name):
        register_model(
            name=name,
            media_type=media_type,
            trainable=False,
            num_training=loaded_count,
            detector_name=name,
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


# ---------------------------------------------------------------------------
# Multi-dataset, multi-model Find
# ---------------------------------------------------------------------------


@detectors_training_bp.route("/api/find", methods=["POST"])
def multi_find():
    """Run selected models on selected datasets and return merged results.

    Expects JSON::

        {
            "dataset_ids": ["abc123", "def456"],
            "model_ids": ["ghi789", "jkl012"]
        }

    For each dataset: loads it from its saved pkl, then for each model runs
    detection.  Returns a merged results table.
    """
    import gc
    import pickle

    import torch

    from vtsearch.datasets.registry import get_dataset as reg_get_ds
    from vtsearch.models.registry import get_model as reg_get_model
    from vtsearch.routes.trainable_models import _model_path, _read_model

    body = request.get_json(force=True, silent=True) or {}
    dataset_ids = body.get("dataset_ids", [])
    model_ids = body.get("model_ids", [])

    if not dataset_ids:
        return jsonify({"error": "No datasets selected"}), 400
    if not model_ids:
        return jsonify({"error": "No models selected"}), 400

    # Resolve datasets and models from registries
    datasets = []
    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            return jsonify({"error": f"Dataset '{ds_id}' not found"}), 404
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            return jsonify({"error": f"Dataset file missing for '{ds.get('name', ds_id)}'"}), 404
        datasets.append(ds)

    models = []
    for m_id in model_ids:
        m = reg_get_model(m_id)
        if m is None:
            return jsonify({"error": f"Model '{m_id}' not found"}), 404
        models.append(m)

    # Build the list of detector configs (weights+threshold) for each model
    model_configs = []
    for m in models:
        det_name = m.get("detector_name", "")
        tm_name = m.get("trainable_model_name", "")

        # Try autorun detector first (has weights)
        if det_name:
            det = get_autorun_detectors().get(det_name)
            if det and det.get("weights"):
                model_configs.append(
                    {
                        "name": m["name"],
                        "model_id": m["id"],
                        "weights": det["weights"],
                        "threshold": det.get("threshold", 0.5),
                    }
                )
                continue

        # For trainable models, we need to train on-the-fly from their labelset
        if tm_name:
            tm_path = _model_path(tm_name)
            tm_data = _read_model(tm_path)
            if tm_data and tm_data.get("labelset", {}).get("labels"):
                model_configs.append(
                    {
                        "name": m["name"],
                        "model_id": m["id"],
                        "trainable_model_data": tm_data,
                    }
                )
                continue

        return jsonify({"error": f"Model '{m['name']}' has no weights or labels for detection"}), 400

    # Run Find across all datasets × models
    all_results = []
    multiple_datasets = len(datasets) > 1
    multiple_models = len(model_configs) > 1
    model_names = [mc["name"] for mc in model_configs]

    for ds in datasets:
        # Load dataset from pkl
        temp_medias: dict = {}
        try:
            pkl_path = ds["pkl_path"]
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            if isinstance(pkl_data, dict) and "medias" in pkl_data:
                raw_medias = pkl_data["medias"]
            else:
                raw_medias = pkl_data

            for cid, mdata in raw_medias.items():
                mid = int(cid) if not isinstance(cid, int) else cid
                emb = mdata.get("embedding")
                if emb is not None:
                    emb = np.array(emb, dtype=np.float32)
                temp_medias[mid] = {**mdata, "id": mid, "embedding": emb}
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset '{ds['name']}': {e}"}), 500

        if not temp_medias:
            continue

        # Build embeddings tensor
        all_ids = sorted(temp_medias.keys())
        all_embs = np.array([temp_medias[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)

        # Per-media result: {media_id -> {info, per_model_scores}}
        media_results: dict[int, dict] = {}
        for cid in all_ids:
            clip = temp_medias[cid]
            media_results[cid] = {
                "id": cid,
                "filename": clip.get("filename", ""),
                "md5": clip.get("md5", ""),
                "origin_name": clip.get("origin_name", clip.get("filename", "")),
                "origin": clip.get("origin"),
                "dataset_name": ds["name"],
                "model_verdicts": {},
            }

        # Run each model on this dataset
        for mc in model_configs:
            if "weights" in mc:
                # Pre-trained detector with weights
                try:
                    model = build_model_from_weights(mc["weights"])
                    with torch.no_grad():
                        scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()
                    threshold = mc.get("threshold", 0.5)
                    for cid, score in zip(all_ids, scores):
                        verdict = "Good" if score >= threshold else "Bad"
                        media_results[cid]["model_verdicts"][mc["name"]] = {
                            "verdict": verdict,
                            "score": round(score, 4),
                        }
                except Exception:
                    for cid in all_ids:
                        media_results[cid]["model_verdicts"][mc["name"]] = {
                            "verdict": "Error",
                            "score": 0,
                        }
            elif "trainable_model_data" in mc:
                # Trainable model — train from labelset, then score
                tm_data = mc["trainable_model_data"]
                labels = tm_data.get("labelset", {}).get("labels", [])

                # Try to match labels to this dataset's medias and train
                try:
                    from vtsearch.utils import build_media_lookup, resolve_media_ids

                    origin_lookup, md5_lookup = build_media_lookup(temp_medias)
                    good_ids, bad_ids = [], []
                    for lbl in labels:
                        matched = resolve_media_ids(lbl, origin_lookup, md5_lookup)
                        label_val = lbl.get("label", "")
                        for mid in matched:
                            if label_val == "good":
                                good_ids.append(mid)
                            elif label_val == "bad":
                                bad_ids.append(mid)

                    if good_ids and bad_ids:
                        good_embs = [temp_medias[i]["embedding"] for i in good_ids if i in temp_medias]
                        bad_embs = [temp_medias[i]["embedding"] for i in bad_ids if i in temp_medias]
                        X_list = good_embs + bad_embs
                        y_list = [1.0] * len(good_embs) + [0.0] * len(bad_embs)

                        from vtsearch.models import calculate_cross_calibration_threshold, train_model

                        input_dim = X_list[0].shape[0]
                        threshold = calculate_cross_calibration_threshold(X_list, y_list, input_dim, get_inclusion())

                        import torch

                        X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
                        y_train = torch.tensor([[v] for v in y_list], dtype=torch.float32)
                        model = train_model(X_train, y_train, input_dim, get_inclusion())

                        with torch.no_grad():
                            scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

                        for cid, score in zip(all_ids, scores):
                            verdict = "Good" if score >= threshold else "Bad"
                            media_results[cid]["model_verdicts"][mc["name"]] = {
                                "verdict": verdict,
                                "score": round(score, 4),
                            }
                    else:
                        # Not enough labels matched this dataset
                        for cid in all_ids:
                            media_results[cid]["model_verdicts"][mc["name"]] = {
                                "verdict": "N/A",
                                "score": 0,
                            }
                except Exception:
                    for cid in all_ids:
                        media_results[cid]["model_verdicts"][mc["name"]] = {
                            "verdict": "Error",
                            "score": 0,
                        }

        # Collect positive hits (Good for at least one model)
        for cid, mr in media_results.items():
            verdicts = mr["model_verdicts"]
            if any(v["verdict"] == "Good" for v in verdicts.values()):
                all_results.append(mr)

        # Free memory
        del temp_medias, X_all
        gc.collect()

    return jsonify(
        {
            "results": all_results,
            "datasets": [ds["name"] for ds in datasets],
            "models": model_names,
            "multiple_datasets": multiple_datasets,
            "multiple_models": multiple_models,
            "total_hits": len(all_results),
        }
    )

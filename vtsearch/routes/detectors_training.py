"""Detector training routes: vote-based, label-based, and multi-dataset/model find."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.routes.detectors_helpers import serialize_weights, train_and_threshold, validate_good_bad_split
from vtsearch.routes.helpers import extract_plugin_fields, get_json_or_400, validate_filepath_field
from vtsearch.models import (
    build_model_from_weights,
    collect_media_origins,
)
from vtsearch.utils import (
    add_autorun_detector,
    get_autorun_detectors,
    get_inclusion,
    snapshot_medias,
)
import vtsearch.utils.paths as _paths

from vtsearch.routes.detectors_crud import get_detectors_dir

from vtsearch.utils.progress import get_find_progress, update_find_progress

detectors_training_bp = Blueprint("detectors_training", __name__)


# ---------------------------------------------------------------------------
# Vote-based training
# ---------------------------------------------------------------------------


@detectors_training_bp.route("/api/detector/export", methods=["POST"])
def export_detector():
    """Train MLP on current votes and export the model weights."""
    from vtsearch.utils import bad_votes, good_votes

    if not good_votes or not bad_votes:
        return jsonify({"error": "need at least one good and one bad vote"}), 400

    snap = snapshot_medias()

    # Train the model
    X_list = []
    y_list = []
    for cid in good_votes:
        if cid in snap:
            X_list.append(snap[cid]["embedding"])
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in snap:
            X_list.append(snap[cid]["embedding"])
            y_list.append(0.0)

    if not X_list:
        return jsonify({"error": "voted medias no longer loaded — reload the dataset"}), 400

    try:
        validate_good_bad_split(y_list)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model, threshold = train_and_threshold(X_list, y_list, snap=snap)
    weights = serialize_weights(model)

    # Collect origin info so the client can forward it when saving
    good_origins = collect_media_origins(good_votes, snap)
    bad_origins = collect_media_origins(bad_votes, snap)

    return jsonify(
        {
            "weights": weights,
            "threshold": round(threshold, 4),
            "good_origins": good_origins,
            "bad_origins": bad_origins,
            "inclusion": get_inclusion(),
        }
    )


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

    importer = get_label_importer(importer_name)
    if importer is None:
        known = [imp.name for imp in list_label_importers()]
        return (
            jsonify({"error": f"Unknown label importer '{importer_name}'. Available: {known}"}),
            404,
        )

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded. Load a dataset first."}), 400

    field_values = extract_plugin_fields(importer)

    # name comes from form data or JSON body (not a plugin field)
    has_file_fields = any(f.field_type == "file" for f in importer.fields)
    if has_file_fields:
        name = request.form.get("name", "").strip()
    else:
        name = (request.get_json(force=True, silent=True) or {}).get("name", "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    err = validate_filepath_field(field_values)
    if err:
        return err

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


# ---------------------------------------------------------------------------
# Multi-dataset, multi-model Find
# ---------------------------------------------------------------------------


@detectors_training_bp.route("/api/find/check-labels", methods=["POST"])
def find_check_labels():
    """Pre-flight check: report how many trainable-model labels can be resolved.

    Takes the same ``model_ids`` / ``dataset_ids`` payload as ``/api/find``
    and returns per-model resolution statistics so the frontend can warn the
    user before starting the (potentially expensive) Find operation.

    Returns JSON::

        {
            "warnings": [
                {
                    "model_name": "Mammals",
                    "total_labels": 82,
                    "resolved_labels": 60,
                    "failed_labels": 22
                }
            ]
        }

    ``warnings`` only contains entries for models that have at least one
    unresolved label.  An empty ``warnings`` list means everything is fine.
    """
    from vtsearch.datasets.loader import safe_pickle_load
    from vtsearch.datasets.registry import get_dataset as reg_get_ds
    from vtsearch.models.registry import get_model as reg_get_model
    from vtsearch.routes.trainable_models import _model_path, _read_model

    body = request.get_json(force=True, silent=True) or {}
    dataset_ids = body.get("dataset_ids", [])
    model_ids = body.get("model_ids", [])

    if not dataset_ids or not model_ids:
        return jsonify({"warnings": []})

    # Collect trainable models that need label resolution
    warnings: list[dict] = []
    for m_id in model_ids:
        m = reg_get_model(m_id)
        if m is None:
            continue
        tm_name = m.get("trainable_model_name", "")
        if not tm_name:
            continue

        tm_path = _model_path(tm_name)
        tm_data = _read_model(tm_path)
        if not tm_data:
            continue
        labels = tm_data.get("labelset", {}).get("labels", [])
        if not labels:
            continue

        # Check if labels match any of the selected datasets directly
        any_direct_match = False
        for ds_id in dataset_ids:
            ds = reg_get_ds(ds_id)
            if ds is None:
                continue
            pkl_path = ds.get("pkl_path", "")
            if not pkl_path or not Path(pkl_path).is_file():
                continue
            try:
                with open(pkl_path, "rb") as f:
                    pkl_data = safe_pickle_load(f)
                raw_medias = pkl_data["medias"] if isinstance(pkl_data, dict) and "medias" in pkl_data else pkl_data
                temp_medias = {}
                for cid, mdata in raw_medias.items():
                    mid = int(cid) if not isinstance(cid, int) else cid
                    temp_medias[mid] = {**mdata, "id": mid}
            except Exception:
                continue

            from vtsearch.utils import build_media_lookup, resolve_media_ids

            origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
            matched = 0
            for lbl in labels:
                if resolve_media_ids(lbl, origin_lookup, md5_lookup):
                    matched += 1
            if matched > 0:
                any_direct_match = True
                break

        if any_direct_match:
            # Labels matched the dataset directly — no resolution needed
            continue

        # Cross-dataset scenario: try the resolver
        from vtsearch.models.resolver import resolve_label_embeddings

        media_type = tm_data.get("media_type", "audio")
        resolved = resolve_label_embeddings(labels, media_type)

        failed = resolved.total_count - resolved.resolved_count
        if failed > 0:
            warnings.append({
                "model_name": m.get("name", tm_name),
                "total_labels": resolved.total_count,
                "resolved_labels": resolved.resolved_count,
                "failed_labels": failed,
            })

    return jsonify({"warnings": warnings})


@detectors_training_bp.route("/api/find/progress")
def find_progress_endpoint():
    """Return the current progress of the Find operation."""
    return jsonify(get_find_progress())


# Number of high-level Find steps: prepare models, load data, score.
_FIND_STEPS = 3


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

    import torch

    from vtsearch.datasets.loader import safe_pickle_load
    from vtsearch.datasets.registry import get_dataset as reg_get_ds
    from vtsearch.models.registry import get_model as reg_get_model
    from vtsearch.routes.trainable_models import _model_path, _read_model

    body = request.get_json(force=True, silent=True) or {}
    dataset_ids = body.get("dataset_ids", [])
    model_ids = body.get("model_ids", [])

    if not dataset_ids:
        update_find_progress("idle", "", step=None, total_steps=None)
        return jsonify({"error": "No datasets selected"}), 400
    if not model_ids:
        update_find_progress("idle", "", step=None, total_steps=None)
        return jsonify({"error": "No models selected"}), 400

    # -- Step 1/3: Preparing models --
    update_find_progress(
        "running", "Preparing models…",
        current=0, total=len(model_ids),
        step=1, total_steps=_FIND_STEPS,
    )

    # Resolve datasets and models from registries
    datasets = []
    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            update_find_progress("idle", "", step=None, total_steps=None)
            return jsonify({"error": f"Dataset '{ds_id}' not found"}), 404
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            update_find_progress("idle", "", step=None, total_steps=None)
            return jsonify({"error": f"Dataset file missing for '{ds.get('name', ds_id)}'"}), 404
        datasets.append(ds)

    models = []
    for m_id in model_ids:
        m = reg_get_model(m_id)
        if m is None:
            update_find_progress("idle", "", step=None, total_steps=None)
            return jsonify({"error": f"Model '{m_id}' not found"}), 404
        models.append(m)

    # Build the list of detector configs (weights+threshold) for each model
    model_configs = []
    for mi, m in enumerate(models):
        det_name = m.get("detector_name", "")
        tm_name = m.get("trainable_model_name", "")

        update_find_progress(
            "running", f"Preparing model \"{m['name']}\"…",
            current=mi + 1, total=len(models),
            step=1, total_steps=_FIND_STEPS,
        )

        # Try loaded DetectorContext first (cached MLP — skip resolve + train)
        from vtsearch.utils.state_core import get_detector_context

        det_ctx = get_detector_context(m["id"])
        if det_ctx is not None and det_ctx.model is not None:
            model_configs.append(
                {
                    "name": m["name"],
                    "model_id": m["id"],
                    "live_model": det_ctx.model,
                    "threshold": det_ctx.threshold,
                }
            )
            continue

        # Try autorun detector next (has serialized weights)
        if det_name:
            det = get_autorun_detectors().get(det_name)
            has_weights = det and det.get("weights")
            if has_weights:
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

        update_find_progress("idle", "", step=None, total_steps=None)
        return jsonify({"error": f"Model '{m['name']}' has no weights or labels for detection"}), 400

    # Run Find across all datasets × models
    all_results = []
    all_negative_results = []
    detected_media_type = ""
    multiple_datasets = len(datasets) > 1
    multiple_models = len(model_configs) > 1
    model_names = [mc["name"] for mc in model_configs]

    # Compute total scoring units for step 3 progress:
    # We'll count items scored across all dataset×model combos.
    total_scoring_units = 0
    scored_units = 0

    for di, ds in enumerate(datasets):
        # -- Step 2/3: Loading dataset --
        ds_label = f"Loading dataset \"{ds['name']}\""
        if len(datasets) > 1:
            ds_label += f" ({di + 1}/{len(datasets)})"
        ds_label += "…"
        update_find_progress(
            "running", ds_label,
            current=di, total=len(datasets),
            step=2, total_steps=_FIND_STEPS,
        )

        # Load dataset from pkl
        temp_medias: dict = {}
        try:
            pkl_path = ds["pkl_path"]
            with open(pkl_path, "rb") as f:
                pkl_data = safe_pickle_load(f)
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
            update_find_progress("idle", "", step=None, total_steps=None)
            return jsonify({"error": f"Failed to load dataset '{ds['name']}': {e}"}), 500

        if not temp_medias:
            continue

        # Detect media type from loaded dataset
        if not detected_media_type:
            first_media = next(iter(temp_medias.values()), {})
            detected_media_type = first_media.get("type", "")

        # Build embeddings tensor
        all_ids = sorted(temp_medias.keys())
        all_embs = np.array([temp_medias[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)

        # Update total scoring units now that we know the dataset size
        total_scoring_units += len(all_ids) * len(model_configs)

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

        # -- Step 3/3: Scoring --
        for mc in model_configs:
            score_label = f"Scoring with \"{mc['name']}\" on \"{ds['name']}\""
            if len(datasets) > 1 or len(model_configs) > 1:
                score_label += f" ({scored_units}/{total_scoring_units} items)"
            score_label += "…"
            update_find_progress(
                "running", score_label,
                current=scored_units, total=total_scoring_units,
                step=3, total_steps=_FIND_STEPS,
            )

            if "live_model" in mc:
                # Loaded detector with cached MLP — use directly, no resolve/train
                try:
                    model = mc["live_model"]
                    with torch.no_grad():
                        raw_logits = model(X_all)
                        scores = torch.sigmoid(raw_logits).squeeze(1).tolist()
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
            elif "weights" in mc:
                # Pre-trained detector with serialized weights
                try:
                    model = build_model_from_weights(mc["weights"])
                    with torch.no_grad():
                        raw_logits = model(X_all)
                        scores = torch.sigmoid(raw_logits).squeeze(1).tolist()
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

                    origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
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
                    else:
                        # Labels didn't match this dataset — resolve from
                        # original sources (cross-dataset scenario).
                        from vtsearch.models.resolver import resolve_label_embeddings

                        media_type = tm_data.get("media_type", "audio")
                        resolved = resolve_label_embeddings(labels, media_type)
                        if resolved.has_good_and_bad:
                            X_list = resolved.embeddings
                            y_list = resolved.labels
                        else:
                            X_list = []
                            y_list = []

                    if X_list and any(v == 1.0 for v in y_list) and any(v == 0.0 for v in y_list):
                        model, threshold = train_and_threshold(X_list, y_list)

                        with torch.no_grad():
                            scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

                        for cid, score in zip(all_ids, scores):
                            verdict = "Good" if score >= threshold else "Bad"
                            media_results[cid]["model_verdicts"][mc["name"]] = {
                                "verdict": verdict,
                                "score": round(score, 4),
                            }
                    else:
                        # Not enough labels resolved
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

            scored_units += len(all_ids)
            update_find_progress(
                "running", f"Scored \"{mc['name']}\" on \"{ds['name']}\"",
                current=scored_units, total=total_scoring_units,
                step=3, total_steps=_FIND_STEPS,
            )

        # Collect positive and negative hits
        for cid, mr in media_results.items():
            verdicts = mr["model_verdicts"]
            if any(v["verdict"] == "Good" for v in verdicts.values()):
                all_results.append(mr)
            elif any(v["verdict"] in ("Bad", "Error", "N/A") for v in verdicts.values()):
                all_negative_results.append(mr)

        # Free memory
        del temp_medias, X_all
        gc.collect()

    # Reset progress to idle
    update_find_progress("idle", "", step=None, total_steps=None)

    return jsonify(
        {
            "results": all_results,
            "negative_results": all_negative_results,
            "datasets": [ds["name"] for ds in datasets],
            "models": model_names,
            "media_type": detected_media_type,
            "multiple_datasets": multiple_datasets,
            "multiple_models": multiple_models,
            "total_hits": len(all_results),
        }
    )

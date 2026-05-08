"""Multi-dataset, multi-model Find routes.

Run selected trainable models against selected datasets and return merged
hit/miss results.  Each model's MLP is sourced from its in-memory
:class:`~vtsearch.utils.DetectorContext` (when loaded) or trained on demand
from its on-disk labelset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify

from vtsearch.models.detector_training import train_and_threshold
from vtsearch.routes.helpers import get_json_safe
from vtsearch.utils.progress import get_find_progress, update_find_progress

model_find_bp = Blueprint("model_find", __name__)


# Number of high-level Find steps: prepare models, load data, score.
_FIND_STEPS = 3


@model_find_bp.route("/api/find/check-labels", methods=["POST"])
def find_check_labels():
    """Pre-flight check: report how many trainable-model labels can be resolved.

    Takes the same ``model_ids`` / ``dataset_ids`` payload as ``/api/find``
    and returns per-model resolution statistics so the frontend can warn the
    user before starting the (potentially expensive) Find operation.
    """
    from vtsearch.datasets.loader import safe_pickle_load
    from vtsearch.datasets.registry import get_dataset as reg_get_ds
    from vtsearch.models.registry import get_model as reg_get_model
    from vtsearch.models.trainable_model_store import _model_path, _read_model

    body = get_json_safe()
    dataset_ids = body.get("dataset_ids", [])
    model_ids = body.get("model_ids", [])

    if not dataset_ids or not model_ids:
        return jsonify({"warnings": []})

    warnings: list[dict] = []
    for m_id in model_ids:
        m = reg_get_model(m_id)
        if m is None:
            continue
        name = m.get("name", "")
        if not name:
            continue

        tm_path = _model_path(name)
        tm_data = _read_model(tm_path)
        if not tm_data:
            continue
        labels = tm_data.get("labelset", {}).get("labels", [])
        if not labels:
            continue

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
            continue

        from vtsearch.models.resolver import resolve_label_embeddings

        media_type = tm_data.get("media_type", "audio")
        resolved = resolve_label_embeddings(labels, media_type)

        failed = resolved.total_count - resolved.resolved_count
        if failed > 0:
            warnings.append(
                {
                    "model_name": m.get("name", name),
                    "total_labels": resolved.total_count,
                    "resolved_labels": resolved.resolved_count,
                    "failed_labels": failed,
                }
            )

    return jsonify({"warnings": warnings})


@model_find_bp.route("/api/find/progress")
def find_progress_endpoint():
    """Return the current progress of the Find operation."""
    return jsonify(get_find_progress())


@model_find_bp.route("/api/find", methods=["POST"])
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
    from vtsearch.models.trainable_model_store import _model_path, _read_model

    body = get_json_safe()
    dataset_ids = body.get("dataset_ids", [])
    model_ids = body.get("model_ids", [])

    if not dataset_ids:
        update_find_progress("idle", "", step=None, total_steps=None)
        return jsonify({"error": "No datasets selected"}), 400
    if not model_ids:
        update_find_progress("idle", "", step=None, total_steps=None)
        return jsonify({"error": "No models selected"}), 400

    update_find_progress(
        "running",
        "Preparing models…",
        current=0,
        total=len(model_ids),
        step=1,
        total_steps=_FIND_STEPS,
    )

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

    model_configs = []
    for mi, m in enumerate(models):
        update_find_progress(
            "running",
            f'Preparing model "{m["name"]}"…',
            current=mi + 1,
            total=len(models),
            step=1,
            total_steps=_FIND_STEPS,
        )

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

        tm_path = _model_path(m["name"])
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
        return jsonify({"error": f"Model '{m['name']}' has no labels for detection"}), 400

    all_results = []
    all_negative_results = []
    detected_media_type = ""
    multiple_datasets = len(datasets) > 1
    multiple_models = len(model_configs) > 1
    model_names = [mc["name"] for mc in model_configs]

    total_scoring_units = 0
    scored_units = 0

    for di, ds in enumerate(datasets):
        ds_label = f'Loading dataset "{ds["name"]}"'
        if len(datasets) > 1:
            ds_label += f" ({di + 1}/{len(datasets)})"
        ds_label += "…"
        update_find_progress(
            "running",
            ds_label,
            current=di,
            total=len(datasets),
            step=2,
            total_steps=_FIND_STEPS,
        )

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

        if not detected_media_type:
            first_media = next(iter(temp_medias.values()), {})
            detected_media_type = first_media.get("type", "")

        all_ids = sorted(temp_medias.keys())
        all_embs = np.array([temp_medias[cid]["embedding"] for cid in all_ids])
        X_all = torch.tensor(all_embs, dtype=torch.float32)

        total_scoring_units += len(all_ids) * len(model_configs)

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

        for mc in model_configs:
            score_label = f'Scoring with "{mc["name"]}" on "{ds["name"]}"'
            if len(datasets) > 1 or len(model_configs) > 1:
                score_label += f" ({scored_units}/{total_scoring_units} items)"
            score_label += "…"
            update_find_progress(
                "running",
                score_label,
                current=scored_units,
                total=total_scoring_units,
                step=3,
                total_steps=_FIND_STEPS,
            )

            if "live_model" in mc:
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
            elif "trainable_model_data" in mc:
                tm_data = mc["trainable_model_data"]
                labels = tm_data.get("labelset", {}).get("labels", [])

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
                "running",
                f'Scored "{mc["name"]}" on "{ds["name"]}"',
                current=scored_units,
                total=total_scoring_units,
                step=3,
                total_steps=_FIND_STEPS,
            )

        for cid, mr in media_results.items():
            verdicts = mr["model_verdicts"]
            if any(v["verdict"] == "Good" for v in verdicts.values()):
                all_results.append(mr)
            elif any(v["verdict"] in ("Bad", "Error", "N/A") for v in verdicts.values()):
                all_negative_results.append(mr)

        del temp_medias, X_all
        gc.collect()

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

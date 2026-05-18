"""Multi-dataset, multi-detector Find routes.

Run selected detectors against selected datasets and return merged hit/miss
results.  Each detector's MLP is sourced from its in-memory
:class:`~vtsearch.state.DetectorContext` (when loaded) or trained on demand
from its on-disk labelset.

Migrated to ``flask_smorest`` so the routes are described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flask_smorest import Blueprint, abort

from vtsearch.concurrency.progress import update_find_progress
from vtsearch.detectors.training import train_and_threshold
from vtsearch.schemas.detectors import (
    FindCheckLabelsRequestSchema,
    FindCheckLabelsResponseSchema,
    FindRequestSchema,
    FindResponseSchema,
)

detector_find_bp = Blueprint(
    "detector_find",
    __name__,
    description="Run detectors against datasets and return merged hits / "
    "negative hits, plus a pre-flight label-resolution check.",
)


# Number of high-level Find steps: prepare detectors, load data, score.
_FIND_STEPS = 3


@detector_find_bp.route("/api/find/check-labels", methods=["POST"])
@detector_find_bp.arguments(FindCheckLabelsRequestSchema)
@detector_find_bp.response(200, FindCheckLabelsResponseSchema)
def find_check_labels(body: dict):  # noqa: C901
    """Pre-flight check: report how many detector labels can be resolved.

    Takes the same ``detector_ids`` / ``dataset_ids`` payload as ``/api/find``
    and returns per-detector resolution statistics so the frontend can warn
    the user before starting the (potentially expensive) Find operation.
    """
    from vtsearch.datasets.loader import safe_pickle_load
    from vtsearch.datasets.registry import get_dataset as reg_get_ds
    from vtsearch.detectors.registry import get_detector as reg_get_detector
    from vtsearch.detectors.store import _detector_path, _read_detector

    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids or not detector_ids:
        return {"warnings": []}

    warnings: list[dict] = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            continue
        name = d.get("name", "")
        if not name:
            continue

        det_path = _detector_path(name)
        det_data = _read_detector(det_path)
        if not det_data:
            continue
        labels = det_data.get("labelset", {}).get("labels", [])
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

            from vtsearch.state import (
                build_media_lookup,
                resolve_media_ids,
            )

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

        from vtsearch.detectors.resolver import resolve_label_embeddings

        media_type = det_data.get("media_type", "audio")
        resolved = resolve_label_embeddings(labels, media_type)

        failed = resolved.total_count - resolved.resolved_count
        if failed > 0:
            warnings.append(
                {
                    "detector_name": d.get("name", name),
                    "total_labels": resolved.total_count,
                    "resolved_labels": resolved.resolved_count,
                    "failed_labels": failed,
                }
            )

    return {"warnings": warnings}


@detector_find_bp.route("/api/find", methods=["POST"])
@detector_find_bp.arguments(FindRequestSchema)
@detector_find_bp.response(200, FindResponseSchema)
@detector_find_bp.alt_response(
    400,
    description="Empty datasets/detectors list, or a selected detector has no labels.",
)
@detector_find_bp.alt_response(404, description="A selected dataset or detector ID is unknown / missing.")
@detector_find_bp.alt_response(500, description="A dataset pkl file could not be loaded.")
def multi_find(body: dict):  # noqa: C901
    """Run selected detectors on selected datasets and return merged results.

    For each dataset: loads it from its saved pkl, then for each detector runs
    detection.  Returns a merged results table.
    """
    import gc

    import torch

    from vtsearch.datasets.loader import safe_pickle_load
    from vtsearch.datasets.registry import get_dataset as reg_get_ds
    from vtsearch.detectors.registry import get_detector as reg_get_detector
    from vtsearch.detectors.store import _detector_path, _read_detector

    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids:
        update_find_progress("idle", "", step=None, total_steps=None)
        abort(400, message="No datasets selected")
    if not detector_ids:
        update_find_progress("idle", "", step=None, total_steps=None)
        abort(400, message="No detectors selected")

    update_find_progress(
        "running",
        "Preparing detectors…",
        current=0,
        total=len(detector_ids),
        step=1,
        total_steps=_FIND_STEPS,
    )

    datasets = []
    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            update_find_progress("idle", "", step=None, total_steps=None)
            abort(404, message=f"Dataset '{ds_id}' not found")
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            update_find_progress("idle", "", step=None, total_steps=None)
            abort(404, message=f"Dataset file missing for '{ds.get('name', ds_id)}'")
        datasets.append(ds)

    detectors = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            update_find_progress("idle", "", step=None, total_steps=None)
            abort(404, message=f"Detector '{d_id}' not found")
        detectors.append(d)

    detector_configs = []
    for di, d in enumerate(detectors):
        update_find_progress(
            "running",
            f'Preparing detector "{d["name"]}"…',
            current=di + 1,
            total=len(detectors),
            step=1,
            total_steps=_FIND_STEPS,
        )

        from vtsearch.state.core import get_detector_context

        det_ctx = get_detector_context(d["id"])
        if det_ctx is not None and det_ctx.model is not None:
            detector_configs.append(
                {
                    "name": d["name"],
                    "detector_id": d["id"],
                    "live_mlp": det_ctx.model,
                    "threshold": det_ctx.threshold,
                }
            )
            continue

        det_path = _detector_path(d["name"])
        det_data = _read_detector(det_path)
        if det_data and det_data.get("labelset", {}).get("labels"):
            detector_configs.append(
                {
                    "name": d["name"],
                    "detector_id": d["id"],
                    "detector_data": det_data,
                }
            )
            continue

        update_find_progress("idle", "", step=None, total_steps=None)
        abort(400, message=f"Detector '{d['name']}' has no labels for detection")

    all_results = []
    all_negative_results = []
    detected_media_type = ""
    multiple_datasets = len(datasets) > 1
    multiple_detectors = len(detector_configs) > 1
    detector_names = [dc["name"] for dc in detector_configs]

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
            abort(500, message=f"Failed to load dataset '{ds['name']}': {e}")

        if not temp_medias:
            continue

        if not detected_media_type:
            first_media = next(iter(temp_medias.values()), {})
            detected_media_type = first_media.get("type", "")

        from vtsearch.embedding.matrix import get_embedding_matrix_for_snap

        all_ids, all_embs = get_embedding_matrix_for_snap(temp_medias)
        X_all = torch.from_numpy(all_embs)

        total_scoring_units += len(all_ids) * len(detector_configs)

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
                "detector_verdicts": {},
            }

        for dc in detector_configs:
            score_label = f'Scoring with "{dc["name"]}" on "{ds["name"]}"'
            if len(datasets) > 1 or len(detector_configs) > 1:
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

            if "live_mlp" in dc:
                try:
                    mlp = dc["live_mlp"]
                    with torch.no_grad():
                        X_in = X_all.to(next(mlp.parameters()).device)
                        raw_logits = mlp(X_in)
                        scores = torch.sigmoid(raw_logits).squeeze(1).cpu().tolist()
                    threshold = dc.get("threshold", 0.5)

                    for cid, score in zip(all_ids, scores):
                        verdict = "Good" if score >= threshold else "Bad"
                        media_results[cid]["detector_verdicts"][dc["name"]] = {
                            "verdict": verdict,
                            "score": round(score, 4),
                        }
                except Exception:
                    for cid in all_ids:
                        media_results[cid]["detector_verdicts"][dc["name"]] = {
                            "verdict": "Error",
                            "score": 0,
                        }
            elif "detector_data" in dc:
                det_data = dc["detector_data"]
                labels = det_data.get("labelset", {}).get("labels", [])

                try:
                    from vtsearch.state import (
                        build_media_lookup,
                        resolve_media_ids,
                    )

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
                        from vtsearch.detectors.resolver import resolve_label_embeddings

                        media_type = det_data.get("media_type", "audio")
                        resolved = resolve_label_embeddings(labels, media_type)
                        if resolved.has_good_and_bad:
                            X_list = resolved.embeddings
                            y_list = resolved.labels
                        else:
                            X_list = []
                            y_list = []

                    if X_list and any(v == 1.0 for v in y_list) and any(v == 0.0 for v in y_list):
                        mlp, threshold = train_and_threshold(X_list, y_list)

                        with torch.no_grad():
                            X_in = X_all.to(next(mlp.parameters()).device)
                            scores = torch.sigmoid(mlp(X_in)).squeeze(1).cpu().tolist()

                        for cid, score in zip(all_ids, scores):
                            verdict = "Good" if score >= threshold else "Bad"
                            media_results[cid]["detector_verdicts"][dc["name"]] = {
                                "verdict": verdict,
                                "score": round(score, 4),
                            }
                    else:
                        for cid in all_ids:
                            media_results[cid]["detector_verdicts"][dc["name"]] = {
                                "verdict": "N/A",
                                "score": 0,
                            }
                except Exception:
                    for cid in all_ids:
                        media_results[cid]["detector_verdicts"][dc["name"]] = {
                            "verdict": "Error",
                            "score": 0,
                        }

            scored_units += len(all_ids)
            update_find_progress(
                "running",
                f'Scored "{dc["name"]}" on "{ds["name"]}"',
                current=scored_units,
                total=total_scoring_units,
                step=3,
                total_steps=_FIND_STEPS,
            )

        for cid, mr in media_results.items():
            verdicts = mr["detector_verdicts"]
            if any(v["verdict"] == "Good" for v in verdicts.values()):
                all_results.append(mr)
            elif any(v["verdict"] in ("Bad", "Error", "N/A") for v in verdicts.values()):
                all_negative_results.append(mr)

        del temp_medias, X_all
        gc.collect()

    update_find_progress("idle", "", step=None, total_steps=None)

    return {
        "results": all_results,
        "negative_results": all_negative_results,
        "datasets": [ds["name"] for ds in datasets],
        "detectors": detector_names,
        "media_type": detected_media_type,
        "multiple_datasets": multiple_datasets,
        "multiple_detectors": multiple_detectors,
        "total_hits": len(all_results),
    }

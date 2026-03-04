"""Detector scoring, extractor execution, and localizer execution routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.models import build_model_from_weights
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import (
    get_autodetect_detectors_by_media,
    get_autorun_extractors_by_media,
    get_autorun_localizers_by_media,
    snapshot_medias,
)
from vtsearch.routes.detectors_crud import _build_extractor, _build_localizer

detectors_scoring_bp = Blueprint("detectors_scoring", __name__)


# ---------------------------------------------------------------------------
# Detector scoring
# ---------------------------------------------------------------------------


@detectors_scoring_bp.route("/api/detector-sort", methods=["POST"])
def detector_sort():
    """Score all medias using a loaded detector model."""
    import torch  # noqa: PLC0415

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    detector = data.get("detector")
    if not detector:
        return jsonify({"error": "detector is required"}), 400

    weights = detector.get("weights")
    threshold = detector.get("threshold", 0.5)

    if not weights:
        return jsonify({"error": "detector weights are required"}), 400

    # Reconstruct the model from weights
    model = build_model_from_weights(weights)

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "no medias loaded"}), 400

    # Score every media
    all_ids = sorted(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)
    with torch.no_grad():
        scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@detectors_scoring_bp.route("/api/auto-detect", methods=["POST"])
def auto_detect():
    """Run autorun detectors for the current media type and return positive hits.

    Accepts an optional JSON body with ``detector_name`` to run a single
    detector instead of all autorun detectors.
    """
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    # Import any autorun processors from settings that aren't already loaded
    from vtsearch.settings import ensure_autorun_processors_imported

    newly_imported = ensure_autorun_processors_imported()

    # Determine media type from current medias
    media_type = next(iter(snap.values())).get("type", "audio")

    # Get autorun detectors for this media type (only those with autodetect enabled)
    detectors = get_autodetect_detectors_by_media(media_type)

    if not detectors:
        return jsonify({"error": f"No autorun detectors found for media type: {media_type}"}), 400

    # Optionally filter to a single detector
    body = request.get_json(silent=True) or {}
    single_name = body.get("detector_name")
    if single_name:
        if single_name not in detectors:
            return jsonify({"error": f"Detector '{single_name}' not found for media type: {media_type}"}), 404
        detectors = {single_name: detectors[single_name]}

    # Prepare shared data for all detectors
    import torch  # noqa: PLC0415

    all_ids = sorted(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    def _run_single_detector(detector_name, detector_data):
        """Run a single detector and return (name, result_dict) or None on failure."""
        try:
            weights = detector_data["weights"]
            threshold = detector_data["threshold"]

            model = build_model_from_weights(weights)

            with torch.no_grad():
                scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

            positive_hits = []
            negative_hits = []
            for cid, score in zip(all_ids, scores):
                clip_info = snap[cid].copy()
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
        except Exception:
            return None

    # Run all detectors in parallel (PyTorch releases GIL during tensor ops)
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(detectors), 8)) as pool:
        futures = [pool.submit(_run_single_detector, name, data) for name, data in detectors.items()]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
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
# Extractor execution
# ---------------------------------------------------------------------------


@detectors_scoring_bp.route("/api/extract", methods=["POST"])
def run_extract():
    """Run a single extractor on all medias and return per-media extraction results."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    extractor_name = data.get("name", "").strip()
    extractor_type = data.get("extractor_type", "").strip()
    config = data.get("config")

    if not extractor_type:
        return jsonify({"error": "extractor_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    try:
        extractor = _build_extractor(extractor_name or "adhoc", extractor_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid extractor config: {e}"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    if extractor.media_type != media_type:
        return (
            jsonify({"error": f"Extractor media type '{extractor.media_type}' does not match medias '{media_type}'"}),
            400,
        )

    results = []
    for media_id in sorted(snap.keys()):
        media = snap[media_id]
        extractions = extractor.extract(media)
        if extractions:
            clip_info = {k: v for k, v in media.items() if k not in ("embedding", "media_bytes", "media_string")}
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


@detectors_scoring_bp.route("/api/auto-extract", methods=["POST"])
def auto_extract():
    """Run all autorun extractors for the current media type and return extraction results."""
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    extractors = get_autorun_extractors_by_media(media_type)

    if not extractors:
        return jsonify({"error": f"No autorun extractors found for media type: {media_type}"}), 400

    sorted_media_ids = sorted(snap.keys())

    def _run_single_extractor(ext_name, ext_data):
        """Run a single extractor on all medias and return (name, result_dict) or None."""
        try:
            extractor = _build_extractor(ext_name, ext_data["extractor_type"], ext_data["config"])
        except Exception:
            return None

        ext_results = []
        for media_id in sorted_media_ids:
            media = snap[media_id]
            extractions = extractor.extract(media)
            if extractions:
                clip_info = {k: v for k, v in media.items() if k not in ("embedding", "media_bytes", "media_string")}
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
# Localizer execution
# ---------------------------------------------------------------------------


@detectors_scoring_bp.route("/api/localize", methods=["POST"])
def run_localize():
    """Run a single localizer on all clips and return per-clip localization results."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    localizer_name = data.get("name", "").strip()
    localizer_type = data.get("localizer_type", "").strip()
    config = data.get("config")

    if not localizer_type:
        return jsonify({"error": "localizer_type is required"}), 400
    if not config or not isinstance(config, dict):
        return jsonify({"error": "config is required"}), 400

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    try:
        localizer = _build_localizer(localizer_name or "adhoc", localizer_type, config)
    except Exception as e:
        return jsonify({"error": f"Invalid localizer config: {e}"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    if localizer.media_type != media_type:
        return (
            jsonify({"error": f"Localizer media type '{localizer.media_type}' does not match medias '{media_type}'"}),
            400,
        )

    results = []
    for media_id in sorted(snap.keys()):
        media = snap[media_id]
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


@detectors_scoring_bp.route("/api/auto-localize", methods=["POST"])
def auto_localize():
    """Run all autorun localizers for the current media type."""
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(snap.values())).get("type", "")
    localizers = get_autorun_localizers_by_media(media_type)

    if not localizers:
        return jsonify({"error": f"No autorun localizers found for media type: {media_type}"}), 400

    sorted_media_ids = sorted(snap.keys())

    def _run_single_localizer(loc_name, loc_data):
        try:
            localizer = _build_localizer(loc_name, loc_data["localizer_type"], loc_data["config"])
        except Exception:
            return None

        loc_results = []
        for media_id in sorted_media_ids:
            media = snap[media_id]
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

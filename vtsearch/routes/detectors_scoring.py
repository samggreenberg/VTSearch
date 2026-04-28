"""Detector scoring, extractor execution, and localizer execution routes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.models import build_model_from_weights
from vtsearch.routes.helpers import get_json_or_400, get_json_safe
from vtsearch.utils import (
    get_autodetect_detectors_by_media,
    get_autorun_extractors_by_media,
    get_autorun_localizers_by_media,
    snapshot_medias,
)
from vtsearch.utils.progress import update_find_progress
from vtsearch.routes.detectors_crud import _build_extractor, _build_localizer

detectors_scoring_bp = Blueprint("detectors_scoring", __name__)

# Keys excluded from API responses (large binary/vector data).
_HEAVYWEIGHT_KEYS = ("embedding", "media_bytes", "media_string", "thumbnail_bytes")


def _media_info_for_response(media: dict) -> dict:
    """Return a copy of *media* without heavyweight fields."""
    return {k: v for k, v in media.items() if k not in _HEAVYWEIGHT_KEYS}


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

    n_total = len(snap)
    update_find_progress(
        "running",
        f"Scoring {n_total} items…",
        current=0,
        total=n_total,
        step=1,
        total_steps=1,
    )

    # Score every media
    all_ids = sorted(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    # Score in batches so the progress endpoint can report percentage
    batch_size = max(1, min(500, n_total // 10))
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_logits = model(X_all[start:end])
            scores.extend(torch.sigmoid(batch_logits).squeeze(1).tolist())
            update_find_progress(
                "running",
                f"Scoring {n_total} items…",
                current=end,
                total=n_total,
                step=1,
                total_steps=1,
            )

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)

    update_find_progress("idle", "Done", current=n_total, total=n_total, step=1, total_steps=1)
    return jsonify({"results": results, "threshold": round(threshold, 4)})


@detectors_scoring_bp.route("/api/find-label", methods=["POST"])
def find_label():
    """Score all loaded medias with a model and apply labels based on threshold.

    Expects JSON::

        {"model_id": "abc123"}

    Resolves the model from the registry, scores every loaded media, and
    applies Good/Bad labels for ALL elements based on the threshold.  Returns
    the sort results so the frontend can display the stripe and scroll order.
    """
    import torch  # noqa: PLC0415

    from vtsearch.models.registry import get_model as reg_get_model
    from vtsearch.models.trainable_model_store import _model_path, _read_model
    from vtsearch.utils import (
        apply_labels_bulk_with_click_time,
        get_autorun_detectors,
        set_find_initial_labels,
    )

    # Total high-level steps: resolve(1) + optional train(2) + score(3) + apply(4)
    _FIND_LABEL_STEPS = 4

    body = get_json_safe()
    model_id = body.get("model_id")
    if not model_id:
        update_find_progress("idle", "")
        return jsonify({"error": "model_id is required"}), 400

    # If the request body specifies a dataset_id, override the request-scoped
    # context so scoring runs against the correct dataset.
    dataset_id = body.get("dataset_id")
    if dataset_id:
        from flask import g
        from vtsearch.utils import get_context

        ctx = get_context(dataset_id)
        if ctx is not None:
            g._dataset_context = ctx

    update_find_progress(
        "running",
        "Resolving model…",
        current=0,
        total=0,
        step=1,
        total_steps=_FIND_LABEL_STEPS,
    )

    m = reg_get_model(model_id)
    if m is None:
        update_find_progress("idle", "")
        return jsonify({"error": f"Model '{model_id}' not found"}), 404

    # Resolve model weights
    det_name = m.get("detector_name", "")
    tm_name = m.get("trainable_model_name", "")
    weights = None
    threshold = 0.5

    if det_name:
        det = get_autorun_detectors().get(det_name)
        if det and det.get("weights"):
            weights = det["weights"]
            threshold = det.get("threshold", 0.5)

    tm_data = None
    if weights is None and tm_name:
        # Trainable model with saved weights in its data
        tm_path = _model_path(tm_name)
        tm_data = _read_model(tm_path)
        if tm_data and tm_data.get("weights"):
            weights = tm_data["weights"]
            threshold = tm_data.get("threshold", 0.5)

    # Check the in-memory DetectorContext for a cached trained model.
    # learned_sort() caches the MLP here after each sort, so if the user
    # trained on Dataset A and then switched to Dataset B, the model is
    # still available without expensive label-origin resolution.
    if weights is None:
        from vtsearch.models.detector_training import serialize_weights as _ser_weights
        from vtsearch.utils.state_core import get_detector_context

        det_ctx = get_detector_context(model_id)
        if det_ctx is not None and det_ctx.model is not None:
            weights = _ser_weights(det_ctx.model)
            threshold = det_ctx.threshold

    # If still no weights, try training on-the-fly from the trainable model's
    # labelset.  This handles the cross-dataset scenario: user trains on
    # Dataset A (labels saved), loads Dataset B, runs Find.  The labelset's
    # origin info lets us resolve the original files, embed them, and train.
    _resolution_diagnostic: dict | None = None
    snap_for_train: dict | None = None
    if weights is None and tm_data:
        label_entries = tm_data.get("labelset", {}).get("labels", [])
        if label_entries:
            import logging as _logging

            _find_log = _logging.getLogger("vtsearch.routes.detectors_scoring")

            from vtsearch.models.detector_training import serialize_weights as _serialize_weights, train_and_threshold

            update_find_progress(
                "running",
                "Training model from labels…",
                current=0,
                total=0,
                step=2,
                total_steps=_FIND_LABEL_STEPS,
            )

            media_type = m.get("media_type", "image")
            X_list: list = []
            y_list: list[float] = []

            # First pass: match labelset MD5s against currently loaded medias.
            # This covers the common case where medias are still loaded (same
            # dataset) or overlap between datasets.
            snap_for_train = snapshot_medias()
            md5_to_emb = {c["md5"]: c["embedding"] for c in snap_for_train.values()}
            unresolved: list[dict] = []
            for entry in label_entries:
                label_val = entry.get("label", "")
                if label_val not in ("good", "bad"):
                    continue
                md5 = entry.get("md5", "")
                if md5 and md5 in md5_to_emb:
                    X_list.append(md5_to_emb[md5])
                    y_list.append(1.0 if label_val == "good" else 0.0)
                else:
                    unresolved.append(entry)

            _md5_matched = len(X_list)
            _find_log.info(
                "find-label: %d of %d labels matched by MD5 in current dataset, %d need origin resolution",
                _md5_matched,
                _md5_matched + len(unresolved),
                len(unresolved),
            )

            # Second pass: resolve remaining entries from their origins (files
            # on disk).  Needed for the cross-dataset case where Dataset A's
            # items are not in Dataset B.
            resolved = None
            if unresolved:
                from vtsearch.models.resolver import resolve_label_embeddings

                _n_unresolved = len(unresolved)
                update_find_progress(
                    "running",
                    f"Resolving {_n_unresolved} label origins…",
                    current=0,
                    total=_n_unresolved,
                    step=2,
                    total_steps=_FIND_LABEL_STEPS,
                )

                def _origin_progress(current: int, total: int) -> None:
                    update_find_progress(
                        "running",
                        f"Resolving {_n_unresolved} label origins…",
                        current=current,
                        total=total,
                        step=2,
                        total_steps=_FIND_LABEL_STEPS,
                    )

                resolved = resolve_label_embeddings(
                    unresolved,
                    media_type,
                    progress_callback=_origin_progress,
                )
                X_list.extend(resolved.embeddings)
                y_list.extend(resolved.labels)

            has_good = any(v == 1.0 for v in y_list)
            has_bad = any(v == 0.0 for v in y_list)
            if has_good and has_bad:
                update_find_progress(
                    "running",
                    "Cross-calibrating threshold…",
                    current=0,
                    total=0,
                    step=2,
                    total_steps=_FIND_LABEL_STEPS,
                )
                trained_model, threshold = train_and_threshold(
                    X_list,
                    y_list,
                    snap=snap_for_train,
                )
                weights = _serialize_weights(trained_model)
            else:
                # Build diagnostic info for the error response
                _resolution_diagnostic = {
                    "total_labels": _md5_matched + len(unresolved),
                    "md5_matched": _md5_matched,
                    "needed_resolution": len(unresolved),
                    "resolved_from_origin": resolved.resolved_count if resolved else 0,
                    "failed_resolution": len(resolved.missing_entries) if resolved else len(unresolved),
                    "has_good": has_good,
                    "has_bad": has_bad,
                    "media_type": media_type,
                }
                if resolved and resolved.missing_entries:
                    # Include first few unresolved for diagnostics
                    samples = resolved.missing_entries[:3]
                    _resolution_diagnostic["sample_failures"] = [
                        {
                            "origin": e.get("origin"),
                            "origin_name": e.get("origin_name", ""),
                            "filename": e.get("filename", ""),
                            "md5": e.get("md5", "")[:12],
                            "label": e.get("label", ""),
                        }
                        for e in samples
                    ]
                elif not unresolved and not has_good:
                    _resolution_diagnostic["hint"] = (
                        "All labels matched by MD5 but all are the same class (need both good and bad)"
                    )
                elif not unresolved and not has_bad:
                    _resolution_diagnostic["hint"] = (
                        "All labels matched by MD5 but all are the same class (need both good and bad)"
                    )

                _find_log.warning(
                    "find-label: cannot train — resolved %d labels total "
                    "(%d MD5, %d origin) but need both good and bad. "
                    "has_good=%s, has_bad=%s. Diagnostic: %r",
                    len(y_list),
                    _md5_matched,
                    resolved.resolved_count if resolved else 0,
                    has_good,
                    has_bad,
                    _resolution_diagnostic,
                )

    if weights is None:
        update_find_progress("idle", "")
        error_msg = f"Model '{m['name']}' has no weights for scoring"
        if _resolution_diagnostic is not None:
            diag = _resolution_diagnostic
            error_msg = (
                f"Model '{m['name']}' could not be trained: "
                f"{diag['total_labels']} training labels found, "
                f"{diag['md5_matched']} matched current dataset by MD5, "
                f"{diag['needed_resolution']} needed origin resolution, "
                f"{diag['resolved_from_origin']} resolved successfully, "
                f"{diag['failed_resolution']} failed to resolve. "
                f"Has good={diag['has_good']}, has bad={diag['has_bad']}."
            )
            if diag.get("sample_failures"):
                first = diag["sample_failures"][0]
                error_msg += (
                    f" First failure: importer={first['origin'].get('importer', '?') if first['origin'] else 'None'}, "
                    f"origin_name={first['origin_name']!r}, "
                    f"params={first['origin'].get('params', {}) if first['origin'] else '{}'}"
                )
            if diag.get("hint"):
                error_msg += f" Hint: {diag['hint']}"
        resp: dict = {"error": error_msg}
        if _resolution_diagnostic is not None:
            resp["resolution_diagnostic"] = _resolution_diagnostic
            # Concise user-facing warning for the frontend status bar
            failed = _resolution_diagnostic["failed_resolution"]
            total = _resolution_diagnostic["total_labels"]
            mt = _resolution_diagnostic.get("media_type", "items")
            # Pluralise: "images", "audios", etc. — fall back to media_type + "s"
            mt_plural = mt + "s" if mt and not mt.endswith("s") else mt
            resp["warning"] = f"{failed} of your {total} {mt_plural} could not be resolved from their original files."
        return jsonify(resp), 400

    snap = snap_for_train if snap_for_train else snapshot_medias()
    if not snap:
        update_find_progress("idle", "")
        return jsonify({"error": "No medias loaded"}), 400

    n_total = len(snap)
    update_find_progress(
        "running",
        f"Scoring {n_total} items…",
        current=0,
        total=n_total,
        step=3,
        total_steps=_FIND_LABEL_STEPS,
    )

    # Score all medias
    model = build_model_from_weights(weights)
    all_ids = sorted(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    # Score in batches so the progress endpoint can report percentage
    batch_size = max(1, min(500, n_total // 10))
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_logits = model(X_all[start:end])
            scores.extend(torch.sigmoid(batch_logits).squeeze(1).tolist())
            update_find_progress(
                "running",
                f"Scoring {n_total} items…",
                current=end,
                total=n_total,
                step=3,
                total_steps=_FIND_LABEL_STEPS,
            )

    results = [{"id": cid, "score": round(s, 4)} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda x: x["score"], reverse=True)

    # Apply labels to ALL elements based on threshold (bulk for performance)
    update_find_progress(
        "running",
        f"Applying labels to {n_total} items…",
        current=0,
        total=n_total,
        step=4,
        total_steps=_FIND_LABEL_STEPS,
    )
    label_pairs = []
    good_count = 0
    bad_count = 0
    for entry in results:
        if entry["score"] >= threshold:
            label_pairs.append((entry["id"], "good"))
            good_count += 1
        else:
            label_pairs.append((entry["id"], "bad"))
            bad_count += 1
    apply_labels_bulk_with_click_time(label_pairs)

    # Snapshot the detector-assigned labels so that corrections
    # (user-changed labels) can be identified later during export.
    set_find_initial_labels({mid: lbl for mid, lbl in label_pairs})

    # Mark the loaded model as being in "find mode" so that
    # sync_labels_to_loaded_model() won't overwrite the model's saved
    # training labels with these scoring results.
    from vtsearch.models.registry import set_find_mode

    set_find_mode(True)

    from vtsearch.labels.sync import sync_to_labelset_source

    sync_to_labelset_source()

    update_find_progress(
        "idle",
        "Done",
        current=n_total,
        total=n_total,
        step=_FIND_LABEL_STEPS,
        total_steps=_FIND_LABEL_STEPS,
    )

    return jsonify(
        {
            "ok": True,
            "results": results,
            "threshold": round(threshold, 4),
            "good_count": good_count,
            "bad_count": bad_count,
            "model_name": m.get("name", ""),
        }
    )


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
            weights = detector_data.get("weights")
            if weights is None:
                return None  # skip untrained detectors
            threshold = detector_data.get("threshold", 0.5)

            model = build_model_from_weights(weights)

            with torch.no_grad():
                scores = torch.sigmoid(model(X_all)).squeeze(1).tolist()

            positive_hits = []
            negative_hits = []
            for cid, score in zip(all_ids, scores):
                clip_info = _media_info_for_response(snap[cid])
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
# Shared helper for extractor / localizer processing
# ---------------------------------------------------------------------------


def _apply_processor_to_medias(processor, snap: dict, method: str, result_key: str) -> list[dict]:
    """Run a processor (extractor or localizer) on all medias, returning hits.

    Args:
        processor: An extractor or localizer instance with *method*.
        snap: A snapshot of the medias dict.
        method: The method name to call (``"extract"`` or ``"localize"``).
        result_key: The key to store results under (``"extractions"`` or ``"localizations"``).
    """
    func = getattr(processor, method)
    results = []
    for media_id in sorted(snap.keys()):
        media = snap[media_id]
        hits = func(media)
        if hits:
            info = _media_info_for_response(media)
            info[result_key] = hits
            results.append(info)
    return results


def _auto_run_processors(
    snap: dict, processors: dict, build_fn, type_key: str, method: str, result_key: str, name_key: str
) -> dict:
    """Run multiple processors in parallel via ThreadPoolExecutor.

    Args:
        snap: A snapshot of the medias dict.
        processors: ``{name: data_dict}`` of registered processors.
        build_fn: Factory ``(name, type_str, config) -> processor``.
        type_key: Key in *data_dict* holding the processor type string.
        method: Method name (``"extract"`` or ``"localize"``).
        result_key: Key for per-media results (``"extractions"`` or ``"localizations"``).
        name_key: Key for the processor name in the result dict.
    """

    def _run_single(proc_name, proc_data):
        try:
            proc = build_fn(proc_name, proc_data[type_key], proc_data["config"])
        except Exception:
            return None

        hits = _apply_processor_to_medias(proc, snap, method, result_key)
        return proc_name, {
            name_key: proc_name,
            "total_medias_with_hits": len(hits),
            "results": hits,
        }

    results = {}
    with ThreadPoolExecutor(max_workers=min(len(processors), 8)) as pool:
        futures = [pool.submit(_run_single, name, data) for name, data in processors.items()]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result
    return results


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

    results = _apply_processor_to_medias(extractor, snap, "extract", "extractions")

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

    results = _auto_run_processors(
        snap, extractors, _build_extractor, "extractor_type", "extract", "extractions", "extractor_name"
    )

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

    results = _apply_processor_to_medias(localizer, snap, "localize", "localizations")

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

    results = _auto_run_processors(
        snap, localizers, _build_localizer, "localizer_type", "localize", "localizations", "localizer_name"
    )

    return jsonify(
        {
            "media_type": media_type,
            "localizers_run": len(results),
            "results": results,
        }
    )

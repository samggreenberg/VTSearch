"""Detector scoring routes.

Implements the active-dataset scoring endpoints — find-label and auto-detect —
on top of the detector concept.  Detectors are loaded into ``DetectorContext``
instances on demand; weights live exclusively in RAM.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import get_json_safe
from vtsearch.utils import snapshot_medias
from vtsearch.utils.progress import update_find_progress

logger = logging.getLogger(__name__)

detector_scoring_bp = Blueprint("detector_scoring", __name__)

# Keys excluded from API responses (large binary/vector data).
_HEAVYWEIGHT_KEYS = ("embedding", "media_bytes", "media_string", "thumbnail_bytes")


def _media_info_for_response(media: dict) -> dict:
    """Return a copy of *media* without heavyweight fields."""
    return {k: v for k, v in media.items() if k not in _HEAVYWEIGHT_KEYS}


def _resolve_or_train_detector(
    detector_id: str,
    det_data: dict | None,
    media_type: str,
    snap: dict | None,
    *,
    progress_step: int = 2,
    progress_total_steps: int = 4,
) -> tuple[object | None, float, dict | None]:
    """Return (mlp, threshold, diagnostic) for *detector_id*.

    Tries the loaded :class:`DetectorContext` first.  Falls back to training
    on demand from the detector's labelset, embedding label media via its
    origin importer.  Returns ``(None, _, diag)`` when training is not
    possible.
    """
    from vtsearch.utils.state_core import get_detector_context

    det_ctx = get_detector_context(detector_id)
    if det_ctx is not None and det_ctx.model is not None:
        return det_ctx.model, det_ctx.threshold, None

    if det_data is None:
        return None, 0.5, None

    label_entries = det_data.get("labelset", {}).get("labels", [])
    if not label_entries:
        return None, 0.5, None

    update_find_progress(
        "running",
        "Training detector from labels…",
        current=0,
        total=0,
        step=progress_step,
        total_steps=progress_total_steps,
    )

    from vtsearch.models.detector_training import train_and_threshold
    from vtsearch.models.resolver import resolve_label_embeddings

    X_list: list = []
    y_list: list[float] = []
    md5_to_emb = {}
    if snap:
        md5_to_emb = {c["md5"]: c["embedding"] for c in snap.values()}

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

    md5_matched = len(X_list)
    resolved = None
    if unresolved:
        n_unresolved = len(unresolved)
        update_find_progress(
            "running",
            f"Resolving {n_unresolved} label origins…",
            current=0,
            total=n_unresolved,
            step=progress_step,
            total_steps=progress_total_steps,
        )

        def _origin_progress(current: int, total: int) -> None:
            update_find_progress(
                "running",
                f"Resolving {n_unresolved} label origins…",
                current=current,
                total=total,
                step=progress_step,
                total_steps=progress_total_steps,
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
            step=progress_step,
            total_steps=progress_total_steps,
        )
        trained_mlp, threshold = train_and_threshold(X_list, y_list, snap=snap)
        return trained_mlp, threshold, None

    diagnostic: dict = {
        "total_labels": md5_matched + len(unresolved),
        "md5_matched": md5_matched,
        "needed_resolution": len(unresolved),
        "resolved_from_origin": resolved.resolved_count if resolved else 0,
        "failed_resolution": len(resolved.missing_entries) if resolved else len(unresolved),
        "has_good": has_good,
        "has_bad": has_bad,
        "media_type": media_type,
    }
    if resolved and resolved.missing_entries:
        samples = resolved.missing_entries[:3]
        diagnostic["sample_failures"] = [
            {
                "origin": e.get("origin"),
                "origin_name": e.get("origin_name", ""),
                "filename": e.get("filename", ""),
                "md5": e.get("md5", "")[:12],
                "label": e.get("label", ""),
            }
            for e in samples
        ]
    elif not unresolved and (not has_good or not has_bad):
        diagnostic["hint"] = "All labels matched by MD5 but all are the same class (need both good and bad)"

    return None, 0.5, diagnostic


@detector_scoring_bp.route("/api/find-label", methods=["POST"])
def find_label():
    """Score all loaded medias with a detector and apply labels based on threshold.

    Expects JSON::

        {"detector_id": "abc123"}

    Resolves the detector from the registry, scores every loaded media, and
    applies Good/Bad labels for ALL elements based on the threshold.  Returns
    the sort results so the frontend can display the stripe and scroll order.
    """
    import torch  # noqa: PLC0415

    from vtsearch.models.detector_registry import get_detector as reg_get_detector
    from vtsearch.models.detector_store import _detector_path, _read_detector
    from vtsearch.utils import (
        apply_labels_bulk_with_click_time,
        set_find_initial_labels,
    )

    # Total high-level steps: resolve(1) + optional train(2) + score(3) + apply(4)
    _FIND_LABEL_STEPS = 4

    body = get_json_safe()
    detector_id = body.get("detector_id")
    if not detector_id:
        update_find_progress("idle", "")
        return jsonify({"error": "detector_id is required"}), 400

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
        "Resolving detector…",
        current=0,
        total=0,
        step=1,
        total_steps=_FIND_LABEL_STEPS,
    )

    d = reg_get_detector(detector_id)
    if d is None:
        update_find_progress("idle", "")
        return jsonify({"error": f"Detector '{detector_id}' not found"}), 404

    snap = snapshot_medias()
    if not snap:
        update_find_progress("idle", "")
        return jsonify({"error": "No medias loaded"}), 400

    media_type = d.get("media_type", "") or next(iter(snap.values())).get("type", "image")
    det_path = _detector_path(d["name"])
    det_data = _read_detector(det_path)

    mlp, threshold, diagnostic = _resolve_or_train_detector(
        detector_id,
        det_data,
        media_type,
        snap,
        progress_step=2,
        progress_total_steps=_FIND_LABEL_STEPS,
    )
    if mlp is None:
        update_find_progress("idle", "")
        if diagnostic is not None:
            error_msg = (
                f"Detector '{d['name']}' could not be trained: "
                f"{diagnostic['total_labels']} training labels found, "
                f"{diagnostic['md5_matched']} matched current dataset by MD5, "
                f"{diagnostic['needed_resolution']} needed origin resolution, "
                f"{diagnostic['resolved_from_origin']} resolved successfully, "
                f"{diagnostic['failed_resolution']} failed to resolve. "
                f"Has good={diagnostic['has_good']}, has bad={diagnostic['has_bad']}."
            )
            if diagnostic.get("sample_failures"):
                first = diagnostic["sample_failures"][0]
                error_msg += (
                    f" First failure: importer={first['origin'].get('importer', '?') if first['origin'] else 'None'}, "
                    f"origin_name={first['origin_name']!r}, "
                    f"params={first['origin'].get('params', {}) if first['origin'] else '{}'}"
                )
            if diagnostic.get("hint"):
                error_msg += f" Hint: {diagnostic['hint']}"
            failed = diagnostic["failed_resolution"]
            total = diagnostic["total_labels"]
            mt = diagnostic.get("media_type", "items")
            mt_plural = mt + "s" if mt and not mt.endswith("s") else mt
            return jsonify(
                {
                    "error": error_msg,
                    "resolution_diagnostic": diagnostic,
                    "warning": (
                        f"{failed} of your {total} {mt_plural} could not be resolved from their original files."
                    ),
                }
            ), 400
        return jsonify({"error": f"Detector '{d['name']}' has no labels for scoring"}), 400

    n_total = len(snap)
    update_find_progress(
        "running",
        f"Scoring {n_total} items…",
        current=0,
        total=n_total,
        step=3,
        total_steps=_FIND_LABEL_STEPS,
    )

    all_ids = sorted(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    batch_size = max(1, min(500, n_total // 10))
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_logits = mlp(X_all[start:end])
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
    apply_labels_bulk_with_click_time(label_pairs, replace_all=True)

    set_find_initial_labels({mid: lbl for mid, lbl in label_pairs})

    from vtsearch.models.detector_registry import set_find_mode

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

    from vtsearch.achievements import record_find

    record_find(n_total)

    return jsonify(
        {
            "ok": True,
            "results": results,
            "threshold": round(threshold, 4),
            "good_count": good_count,
            "bad_count": bad_count,
            "detector_name": d.get("name", ""),
        }
    )


@detector_scoring_bp.route("/api/auto-detect", methods=["POST"])
def auto_detect():
    """Score the active dataset with every detector flagged for autorun.

    Iterates :func:`~vtsearch.settings.get_autorun_detectors` and trains each
    one's MLP on demand from its on-disk labelset.  Returns one result column
    per detector.

    Accepts an optional JSON body with ``detector_name`` to run a single
    autorun detector by name.
    """
    import torch  # noqa: PLC0415

    from vtsearch.models.detector_registry import (
        find_by_name,
        list_detectors,
    )
    from vtsearch.models.detector_store import _detector_path, _read_detector
    from vtsearch.settings import get_autorun_detectors

    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No medias loaded"}), 400

    media_type = next(iter(snap.values())).get("type", "audio")

    autorun_names = get_autorun_detectors()
    body = request.get_json(silent=True) or {}
    single_name = body.get("detector_name")
    if single_name:
        if single_name not in autorun_names:
            return jsonify({"error": f"Detector '{single_name}' not flagged for autorun"}), 404
        autorun_names = [single_name]

    if not autorun_names:
        return jsonify({"error": f"No autorun detectors found for media type: {media_type}"}), 400

    # Build per-name (det_data, registry entry) pairs, filtered by media type.
    detectors_to_run: list[tuple[str, dict, dict | None]] = []
    for name in autorun_names:
        det_data = _read_detector(_detector_path(name))
        if det_data is None:
            continue
        if det_data.get("media_type", "") != media_type:
            continue
        reg_entry = find_by_name(name)
        if reg_entry is None:
            # Fallback: also accept registry entries whose name matches.
            for entry in list_detectors():
                if entry.get("name") == name:
                    reg_entry = entry
                    break
        detectors_to_run.append((name, det_data, reg_entry))

    if not detectors_to_run:
        return jsonify({"error": f"No autorun detectors found for media type: {media_type}"}), 400

    all_ids = sorted(snap.keys())
    all_embs = np.array([snap[cid]["embedding"] for cid in all_ids])
    X_all = torch.tensor(all_embs, dtype=torch.float32)

    def _run_single(name: str, det_data: dict, reg_entry: dict | None):
        try:
            detector_id = reg_entry["id"] if reg_entry else name
            mlp, threshold, _diag = _resolve_or_train_detector(
                detector_id,
                det_data,
                media_type,
                snap,
                progress_step=1,
                progress_total_steps=1,
            )
            if mlp is None:
                return None

            with torch.no_grad():
                scores = torch.sigmoid(mlp(X_all)).squeeze(1).tolist()

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

            return name, {
                "detector_name": name,
                "threshold": round(threshold, 4),
                "total_hits": len(positive_hits),
                "hits": positive_hits,
                "negative_hits": negative_hits,
            }
        except Exception:
            logger.exception("Auto-detect failed for detector %s", name)
            return None

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(detectors_to_run), 8)) as pool:
        futures = [pool.submit(_run_single, name, data, entry) for name, data, entry in detectors_to_run]
        for future in futures:
            outcome = future.result()
            if outcome is not None:
                name, result = outcome
                results[name] = result

    if results:
        from vtsearch.achievements import record_find

        record_find(len(all_ids) * len(results))

    return jsonify(
        {
            "media_type": media_type,
            "detectors_run": len(results),
            "results": results,
        }
    )

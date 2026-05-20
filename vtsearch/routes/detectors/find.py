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
from typing import NoReturn

import numpy as np
from flask_smorest import Blueprint, abort

from vtscore.concurrency.progress import find_progress, update_find_progress
from vtscore.detectors.training import train_and_threshold
from vtsearch.schemas.detectors import (
    FindCancelResponseSchema,
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


def _load_pkl_for_check(pkl_path: str) -> dict | None:
    """Best-effort load of a pkl file for label resolution.

    Returns the medias-ids-stringified-to-int snapshot, or ``None`` on
    any read / parse error (the check-labels endpoint silently skips
    unreadable datasets).
    """
    from vtscore.datasets.loader import safe_pickle_load  # noqa: PLC0415

    try:
        with open(pkl_path, "rb") as f:
            pkl_data = safe_pickle_load(f)
    except Exception:
        return None
    raw_medias = pkl_data["medias"] if isinstance(pkl_data, dict) and "medias" in pkl_data else pkl_data
    temp_medias: dict[int, dict] = {}
    for cid, mdata in raw_medias.items():
        mid = int(cid) if not isinstance(cid, int) else cid
        temp_medias[mid] = {**mdata, "id": mid}
    return temp_medias


def _any_label_resolves_in_pkl(pkl_path: str, labels: list[dict]) -> bool:
    """Return True if at least one label resolves to a media in *pkl_path*."""
    from vtsearch.state import build_media_lookup, resolve_media_ids  # noqa: PLC0415

    temp_medias = _load_pkl_for_check(pkl_path)
    if temp_medias is None:
        return False
    origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
    return any(resolve_media_ids(lbl, origin_lookup, md5_lookup) for lbl in labels)


def _detector_has_direct_match(labels: list[dict], dataset_ids: list[str]) -> bool:
    """True if any selected dataset can resolve at least one label."""
    from vtscore.datasets.registry import get_dataset as reg_get_ds  # noqa: PLC0415

    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            continue
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            continue
        if _any_label_resolves_in_pkl(pkl_path, labels):
            return True
    return False


def _check_detector_warning(d: dict, dataset_ids: list[str]) -> dict | None:
    """Compute the resolver-fallback warning for one detector, or ``None``.

    Returns ``None`` when the detector is unknown, has no name, has no
    on-disk labelset, has no labels, or already resolves directly via one
    of the selected datasets.
    """
    from vtscore.detectors.resolver import resolve_label_embeddings  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415

    name = d.get("name", "")
    if not name:
        return None
    det_data = _read_detector(_detector_path(name))
    if not det_data:
        return None
    labels = det_data.get("labelset", {}).get("labels", [])
    if not labels:
        return None
    if _detector_has_direct_match(labels, dataset_ids):
        return None

    resolved = resolve_label_embeddings(labels, det_data.get("media_type", "audio"))
    failed = resolved.total_count - resolved.resolved_count
    if failed <= 0:
        return None
    return {
        "detector_name": name,
        "total_labels": resolved.total_count,
        "resolved_labels": resolved.resolved_count,
        "failed_labels": failed,
    }


@detector_find_bp.route("/api/find/check-labels", methods=["POST"])
@detector_find_bp.arguments(FindCheckLabelsRequestSchema)
@detector_find_bp.response(200, FindCheckLabelsResponseSchema)
def find_check_labels(body: dict):
    """Pre-flight check: report how many detector labels can be resolved.

    Takes the same ``detector_ids`` / ``dataset_ids`` payload as ``/api/find``
    and returns per-detector resolution statistics so the frontend can warn
    the user before starting the (potentially expensive) Find operation.
    """
    from vtscore.detectors.registry import get_detector as reg_get_detector  # noqa: PLC0415

    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids or not detector_ids:
        return {"warnings": []}

    warnings: list[dict] = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            continue
        warning = _check_detector_warning(d, dataset_ids)
        if warning is not None:
            warnings.append(warning)
    return {"warnings": warnings}


def _abort_find(code: int, message: str) -> NoReturn:
    """Reset find_progress to idle and abort with *code* / *message*."""
    update_find_progress("idle", "", step=None, total_steps=None)
    abort(code, message=message)
    raise RuntimeError("unreachable — abort() raises")  # for type narrowing


def _resolve_find_datasets(dataset_ids: list[str]) -> list[dict]:
    """Resolve dataset IDs to registry entries, asserting each pkl file exists."""
    from vtscore.datasets.registry import get_dataset as reg_get_ds  # noqa: PLC0415

    datasets: list[dict] = []
    for ds_id in dataset_ids:
        ds = reg_get_ds(ds_id)
        if ds is None:
            _abort_find(404, f"Dataset '{ds_id}' not found")
        pkl_path = ds.get("pkl_path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            _abort_find(404, f"Dataset file missing for '{ds.get('name', ds_id)}'")
        datasets.append(ds)
    return datasets


def _resolve_find_detectors(detector_ids: list[str]) -> list[dict]:
    """Resolve detector IDs to registry entries."""
    from vtscore.detectors.registry import get_detector as reg_get_detector  # noqa: PLC0415

    detectors: list[dict] = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            _abort_find(404, f"Detector '{d_id}' not found")
        detectors.append(d)
    return detectors


def _build_detector_config(d: dict) -> dict:
    """Build a per-detector config carrying both the live-MLP and cold paths.

    A Find call can span multiple datasets that use different embedders, so
    we resolve both paths up front and let :func:`_score_dataset` pick per
    dataset based on whether the live MLP's embedder matches that
    dataset's vectors.  When the live MLP doesn't match, the cold path
    retrains on the dataset's own vectors.

    Aborts with 400 when neither path is usable for *d*.
    """
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415
    from vtscore.state.core import get_detector_context  # noqa: PLC0415

    config: dict = {
        "name": d["name"],
        "detector_id": d["id"],
    }

    det_data = _read_detector(_detector_path(d["name"]))
    if det_data and det_data.get("labelset", {}).get("labels"):
        config["detector_data"] = det_data

    det_ctx = get_detector_context(d["id"])
    if det_ctx is not None and det_ctx.model is not None:
        config["live_mlp"] = det_ctx.model
        config["live_embedder"] = det_ctx.embedder
        config["threshold"] = det_ctx.threshold

    if "detector_data" not in config and "live_mlp" not in config:
        _abort_find(400, f"Detector '{d['name']}' has no labels for detection")

    return config


def _build_detector_configs(detectors: list[dict]) -> list[dict]:
    """Walk *detectors* and emit one config per detector, with progress updates."""
    configs: list[dict] = []
    for di, d in enumerate(detectors):
        update_find_progress(
            "running",
            f'Preparing detector "{d["name"]}"…',
            current=di + 1,
            total=len(detectors),
            step=1,
            total_steps=_FIND_STEPS,
        )
        configs.append(_build_detector_config(d))
    return configs


def _load_find_dataset_medias(ds: dict) -> dict[int, dict]:
    """Load *ds*'s pkl into a per-find ``temp_medias`` snapshot.

    Aborts with 500 on any load error.  The snapshot is owned by the
    caller and freed after scoring completes.
    """
    from vtscore.datasets.loader import safe_pickle_load  # noqa: PLC0415

    try:
        with open(ds["pkl_path"], "rb") as f:
            pkl_data = safe_pickle_load(f)
        if isinstance(pkl_data, dict) and "medias" in pkl_data:
            raw_medias = pkl_data["medias"]
        else:
            raw_medias = pkl_data

        temp_medias: dict[int, dict] = {}
        for cid, mdata in raw_medias.items():
            mid = int(cid) if not isinstance(cid, int) else cid
            emb = mdata.get("embedding")
            if emb is not None:
                emb = np.array(emb, dtype=np.float32)
            temp_medias[mid] = {**mdata, "id": mid, "embedding": emb}
        return temp_medias
    except Exception as e:
        _abort_find(500, f"Failed to load dataset '{ds['name']}': {e}")


def _seed_media_results(all_ids: list[int], temp_medias: dict[int, dict], dataset_name: str) -> dict[int, dict]:
    """Build the per-media result skeleton that scorers fill in."""
    media_results: dict[int, dict] = {}
    for cid in all_ids:
        clip = temp_medias[cid]
        media_results[cid] = {
            "id": cid,
            "filename": clip.get("filename", ""),
            "md5": clip.get("md5", ""),
            "origin_name": clip.get("origin_name", clip.get("filename", "")),
            "origin": clip.get("origin"),
            "dataset_name": dataset_name,
            "detector_verdicts": {},
        }
    return media_results


def _record_verdicts(
    media_results: dict[int, dict],
    dc_name: str,
    all_ids: list[int],
    scores: list[float] | None,
    threshold: float,
    fallback_verdict: str | None,
) -> None:
    """Write each detector's verdict for every media id into *media_results*.

    When *scores* is provided, verdicts come from ``score >= threshold``.
    Otherwise *fallback_verdict* (``"Error"`` / ``"N/A"``) is written with
    ``score=0`` for every id.
    """
    if scores is not None:
        for cid, score in zip(all_ids, scores):
            verdict = "Good" if score >= threshold else "Bad"
            media_results[cid]["detector_verdicts"][dc_name] = {
                "verdict": verdict,
                "score": round(score, 4),
            }
        return
    for cid in all_ids:
        media_results[cid]["detector_verdicts"][dc_name] = {
            "verdict": fallback_verdict,
            "score": 0,
        }


def _live_mlp_matches_dataset(dc: dict, ds_embedder: str) -> bool:
    """Return ``False`` when *dc*'s live MLP is bound to a different embedder.

    Returns ``True`` (treat as a match) when either stamp is unknown — the
    invariant only fires on a confident mismatch, mirroring
    :func:`vtscore.detectors.dataset_sync.invalidate_model_on_embedder_switch`.
    """
    live_embedder = dc.get("live_embedder", "") or ""
    if not live_embedder or not ds_embedder:
        return True
    return live_embedder == ds_embedder


def _score_with_live_mlp(dc: dict, X_all, all_ids: list[int], media_results: dict[int, dict]) -> None:
    """Score with an already-trained MLP from a :class:`DetectorContext`."""
    import torch  # noqa: PLC0415

    try:
        mlp = dc["live_mlp"]
        with torch.no_grad():
            X_in = X_all.to(next(mlp.parameters()).device)
            raw_logits = mlp(X_in)
            scores = torch.sigmoid(raw_logits).squeeze(1).cpu().tolist()
        _record_verdicts(media_results, dc["name"], all_ids, scores, dc.get("threshold", 0.5), None)
    except Exception:
        _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "Error")


def _collect_cold_training_data(
    det_data: dict,
    temp_medias: dict[int, dict],
) -> tuple[list, list]:
    """Assemble training X/y for a cold detector.

    Prefers labels that resolve directly into the dataset (good_ids/bad_ids
    via origin/md5 lookup); falls back to :func:`resolve_label_embeddings`
    when the direct path doesn't yield both classes.
    """
    from vtsearch.state import build_media_lookup, resolve_media_ids  # noqa: PLC0415

    labels = det_data.get("labelset", {}).get("labels", [])
    origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
    good_ids: list[int] = []
    bad_ids: list[int] = []
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
        return good_embs + bad_embs, [1.0] * len(good_embs) + [0.0] * len(bad_embs)

    from vtscore.detectors.resolver import resolve_label_embeddings  # noqa: PLC0415

    resolved = resolve_label_embeddings(labels, det_data.get("media_type", "audio"))
    if resolved.has_good_and_bad:
        return resolved.embeddings, resolved.labels
    return [], []


def _score_with_cold_detector(
    dc: dict,
    temp_medias: dict[int, dict],
    X_all,
    all_ids: list[int],
    media_results: dict[int, dict],
) -> None:
    """Train an MLP on-the-fly from the cold detector's labelset and score."""
    import torch  # noqa: PLC0415

    try:
        X_list, y_list = _collect_cold_training_data(dc["detector_data"], temp_medias)
        has_both_classes = X_list and any(v == 1.0 for v in y_list) and any(v == 0.0 for v in y_list)
        if not has_both_classes:
            _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "N/A")
            return

        mlp, threshold = train_and_threshold(X_list, y_list)
        with torch.no_grad():
            X_in = X_all.to(next(mlp.parameters()).device)
            scores = torch.sigmoid(mlp(X_in)).squeeze(1).cpu().tolist()
        _record_verdicts(media_results, dc["name"], all_ids, scores, threshold, None)
    except Exception:
        _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "Error")


def _partition_find_results(media_results: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Split media results into positives (any Good verdict) and negatives."""
    positives: list[dict] = []
    negatives: list[dict] = []
    for mr in media_results.values():
        verdicts = mr["detector_verdicts"]
        if any(v["verdict"] == "Good" for v in verdicts.values()):
            positives.append(mr)
        elif any(v["verdict"] in ("Bad", "Error", "N/A") for v in verdicts.values()):
            negatives.append(mr)
    return positives, negatives


def _score_dataset(
    ds: dict,
    detector_configs: list[dict],
    datasets_total: int,
    scored_units: int,
    total_scoring_units: int,
) -> tuple[list[dict], list[dict], int, int, str]:
    """Score one dataset against every detector config.

    Returns ``(positives, negatives, new_scored_units, added_scoring_units,
    media_type)``.  ``media_type`` is the type id of the first media in the
    dataset (used to populate the response's top-level ``media_type``).
    """
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    temp_medias = _load_find_dataset_medias(ds)
    if not temp_medias:
        return [], [], scored_units, 0, ""

    first_media = next(iter(temp_medias.values()), {})
    detected_media_type = first_media.get("type", "")
    ds_embedder = first_media.get("embedder", "") or ""

    from vtscore.embedding.matrix import get_embedding_matrix_for_snap  # noqa: PLC0415

    all_ids, all_embs = get_embedding_matrix_for_snap(temp_medias)
    X_all = torch.from_numpy(all_embs)

    added_units = len(all_ids) * len(detector_configs)
    new_total = total_scoring_units + added_units
    media_results = _seed_media_results(all_ids, temp_medias, ds["name"])

    for dc in detector_configs:
        score_label = f'Scoring with "{dc["name"]}" on "{ds["name"]}"'
        if datasets_total > 1 or len(detector_configs) > 1:
            score_label += f" ({scored_units}/{new_total} items)"
        score_label += "…"
        update_find_progress(
            "running",
            score_label,
            current=scored_units,
            total=new_total,
            step=3,
            total_steps=_FIND_STEPS,
        )

        if "live_mlp" in dc and _live_mlp_matches_dataset(dc, ds_embedder):
            _score_with_live_mlp(dc, X_all, all_ids, media_results)
        elif "detector_data" in dc:
            _score_with_cold_detector(dc, temp_medias, X_all, all_ids, media_results)
        else:
            # Live MLP exists but is bound to a different embedder, and no
            # cold-path labelset is available — surface as N/A rather than
            # scoring with a mismatched MLP.
            _record_verdicts(media_results, dc["name"], all_ids, None, 0.0, "N/A")

        scored_units += len(all_ids)
        update_find_progress(
            "running",
            f'Scored "{dc["name"]}" on "{ds["name"]}"',
            current=scored_units,
            total=new_total,
            step=3,
            total_steps=_FIND_STEPS,
        )

    positives, negatives = _partition_find_results(media_results)
    del temp_medias, X_all
    gc.collect()
    return positives, negatives, scored_units, added_units, detected_media_type


@detector_find_bp.route("/api/find", methods=["POST"])
@detector_find_bp.arguments(FindRequestSchema)
@detector_find_bp.response(200, FindResponseSchema)
@detector_find_bp.alt_response(
    400,
    description="Empty datasets/detectors list, or a selected detector has no labels.",
)
@detector_find_bp.alt_response(404, description="A selected dataset or detector ID is unknown / missing.")
@detector_find_bp.alt_response(500, description="A dataset pkl file could not be loaded.")
def multi_find(body: dict):
    """Run selected detectors on selected datasets and return merged results.

    For each dataset: loads it from its saved pkl, then for each detector runs
    detection.  Returns a merged results table.
    """
    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids:
        _abort_find(400, "No datasets selected")
    if not detector_ids:
        _abort_find(400, "No detectors selected")

    # Clear a leftover cancel flag from a previously-cancelled run so
    # the new operation doesn't trip on it immediately.
    find_progress.reset_cancel()

    update_find_progress(
        "running",
        "Preparing detectors…",
        current=0,
        total=len(detector_ids),
        step=1,
        total_steps=_FIND_STEPS,
    )

    datasets = _resolve_find_datasets(dataset_ids)
    detectors = _resolve_find_detectors(detector_ids)
    detector_configs = _build_detector_configs(detectors)
    detector_names = [dc["name"] for dc in detector_configs]

    all_results: list[dict] = []
    all_negative_results: list[dict] = []
    detected_media_type = ""
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

        positives, negatives, scored_units, added_units, ds_media_type = _score_dataset(
            ds,
            detector_configs,
            len(datasets),
            scored_units,
            total_scoring_units,
        )
        all_results.extend(positives)
        all_negative_results.extend(negatives)
        total_scoring_units += added_units
        if not detected_media_type and ds_media_type:
            detected_media_type = ds_media_type

    update_find_progress("idle", "", step=None, total_steps=None)

    return {
        "results": all_results,
        "negative_results": all_negative_results,
        "datasets": [ds["name"] for ds in datasets],
        "detectors": detector_names,
        "media_type": detected_media_type,
        "multiple_datasets": len(datasets) > 1,
        "multiple_detectors": len(detector_configs) > 1,
        "total_hits": len(all_results),
    }


@detector_find_bp.route("/api/find/cancel", methods=["POST"])
@detector_find_bp.response(200, FindCancelResponseSchema)
def cancel_find():
    """Cancel any in-flight find-style scoring.

    Sets the cancel flag on the shared ``find_progress`` tracker, which
    every scoring path (``/api/find``, ``/api/find-label``, and
    ``/api/auto-detect``) reports progress through. Long-running loops
    poll the flag between iterations and bail out by raising
    :class:`CancelledError`. Always returns 200 — calling cancel when
    nothing is running is a no-op.
    """
    find_progress.cancel()
    return {"ok": True}

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
from typing import TYPE_CHECKING, Any

import numpy as np
from flask_smorest import Blueprint, abort

if TYPE_CHECKING:
    import torch

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


def _abort_find(status: int, message: str) -> None:
    """Reset the Find progress tracker and abort with *status* / *message*."""
    update_find_progress("idle", "", step=None, total_steps=None)
    abort(status, message=message)


def _resolve_find_datasets(dataset_ids: list[str]) -> list[dict]:
    """Look up *dataset_ids* in the dataset registry; abort on missing / no-pkl."""
    from vtsearch.datasets.registry import get_dataset as reg_get_ds

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


def _resolve_detector_configs(detector_ids: list[str]) -> list[dict]:
    """Look up *detector_ids* and produce a per-detector scoring config.

    Each entry is either:
      - ``{"name", "detector_id", "live_mlp", "threshold"}`` — model already
        trained in the live :class:`DetectorContext`; or
      - ``{"name", "detector_id", "detector_data"}`` — fall back to the
        on-disk labelset (the scorer will train an MLP on demand).
    """
    from vtsearch.detectors.registry import get_detector as reg_get_detector
    from vtsearch.detectors.store import _detector_path, _read_detector
    from vtsearch.state.core import get_detector_context

    detectors: list[dict] = []
    for d_id in detector_ids:
        d = reg_get_detector(d_id)
        if d is None:
            _abort_find(404, f"Detector '{d_id}' not found")
        detectors.append(d)

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

        det_ctx = get_detector_context(d["id"])
        if det_ctx is not None and det_ctx.model is not None:
            configs.append(
                {"name": d["name"], "detector_id": d["id"], "live_mlp": det_ctx.model, "threshold": det_ctx.threshold}
            )
            continue

        det_data = _read_detector(_detector_path(d["name"]))
        if det_data and det_data.get("labelset", {}).get("labels"):
            configs.append({"name": d["name"], "detector_id": d["id"], "detector_data": det_data})
            continue

        _abort_find(400, f"Detector '{d['name']}' has no labels for detection")
    return configs


def _load_find_dataset_medias(ds: dict) -> dict[int, dict]:
    """Load *ds*'s pkl into a temp medias dict with float32 embeddings."""
    from vtsearch.datasets.loader import safe_pickle_load

    try:
        with open(ds["pkl_path"], "rb") as f:
            pkl_data = safe_pickle_load(f)
    except Exception as e:
        _abort_find(500, f"Failed to load dataset '{ds['name']}': {e}")

    raw_medias = pkl_data["medias"] if isinstance(pkl_data, dict) and "medias" in pkl_data else pkl_data
    temp_medias: dict[int, dict] = {}
    for cid, mdata in raw_medias.items():
        mid = int(cid) if not isinstance(cid, int) else cid
        emb = mdata.get("embedding")
        if emb is not None:
            emb = np.array(emb, dtype=np.float32)
        temp_medias[mid] = {**mdata, "id": mid, "embedding": emb}
    return temp_medias


def _init_media_results(ds_name: str, all_ids: list[int], temp_medias: dict[int, dict]) -> dict[int, dict]:
    """Build the per-media result shell with empty ``detector_verdicts``."""
    media_results: dict[int, dict] = {}
    for cid in all_ids:
        clip = temp_medias[cid]
        media_results[cid] = {
            "id": cid,
            "filename": clip.get("filename", ""),
            "md5": clip.get("md5", ""),
            "origin_name": clip.get("origin_name", clip.get("filename", "")),
            "origin": clip.get("origin"),
            "dataset_name": ds_name,
            "detector_verdicts": {},
        }
    return media_results


def _set_uniform_verdict(
    media_results: dict[int, dict], all_ids: list[int], detector_name: str, verdict: str, score: float = 0
) -> None:
    """Stamp the same ``(verdict, score)`` onto every media for *detector_name*."""
    for cid in all_ids:
        media_results[cid]["detector_verdicts"][detector_name] = {"verdict": verdict, "score": score}


def _apply_mlp_scores(
    mlp: Any,
    X_all: "torch.Tensor",
    all_ids: list[int],
    threshold: float,
    media_results: dict[int, dict],
    detector_name: str,
) -> None:
    """Run *mlp* on *X_all* and write Good/Bad verdicts using *threshold*."""
    import torch

    with torch.no_grad():
        X_in = X_all.to(next(mlp.parameters()).device)
        scores = torch.sigmoid(mlp(X_in)).squeeze(1).cpu().tolist()
    for cid, score in zip(all_ids, scores):
        verdict = "Good" if score >= threshold else "Bad"
        media_results[cid]["detector_verdicts"][detector_name] = {"verdict": verdict, "score": round(score, 4)}


def _score_with_live_mlp(dc: dict, X_all: "torch.Tensor", all_ids: list[int], media_results: dict[int, dict]) -> None:
    """Score against an already-loaded MLP from a live :class:`DetectorContext`."""
    try:
        _apply_mlp_scores(dc["live_mlp"], X_all, all_ids, dc.get("threshold", 0.5), media_results, dc["name"])
    except Exception:
        _set_uniform_verdict(media_results, all_ids, dc["name"], "Error")


def _split_label_ids_by_class(labels: list[dict], temp_medias: dict[int, dict]) -> tuple[list[int], list[int]]:
    """Return ``(good_ids, bad_ids)`` for labels that resolve to *temp_medias*."""
    from vtsearch.state import build_media_lookup, resolve_media_ids

    origin_lookup, md5_lookup, _ = build_media_lookup(temp_medias)
    good_ids: list[int] = []
    bad_ids: list[int] = []
    for lbl in labels:
        bucket = good_ids if lbl.get("label", "") == "good" else bad_ids if lbl.get("label", "") == "bad" else None
        if bucket is None:
            continue
        bucket.extend(resolve_media_ids(lbl, origin_lookup, md5_lookup))
    return good_ids, bad_ids


def _gather_training_examples(
    labels: list[dict], temp_medias: dict[int, dict], media_type: str
) -> tuple[list, list[float]]:
    """Build ``(X_list, y_list)`` for on-the-fly training.

    Prefers labels that resolve directly to media in *temp_medias* (so the
    embeddings match the target dataset); falls back to the resolver, which
    re-derives embeddings from each label's origin when no direct match
    exists.
    """
    good_ids, bad_ids = _split_label_ids_by_class(labels, temp_medias)
    if good_ids and bad_ids:
        good_embs = [temp_medias[i]["embedding"] for i in good_ids if i in temp_medias]
        bad_embs = [temp_medias[i]["embedding"] for i in bad_ids if i in temp_medias]
        return good_embs + bad_embs, [1.0] * len(good_embs) + [0.0] * len(bad_embs)

    from vtsearch.detectors.resolver import resolve_label_embeddings

    resolved = resolve_label_embeddings(labels, media_type)
    if resolved.has_good_and_bad:
        return resolved.embeddings, resolved.labels
    return [], []


def _score_with_detector_data(
    dc: dict,
    X_all: "torch.Tensor",
    all_ids: list[int],
    temp_medias: dict[int, dict],
    media_results: dict[int, dict],
) -> None:
    """Train an MLP from *dc*'s on-disk labelset and score *X_all*."""
    det_data = dc["detector_data"]
    labels = det_data.get("labelset", {}).get("labels", [])
    media_type = det_data.get("media_type", "audio")

    try:
        X_list, y_list = _gather_training_examples(labels, temp_medias, media_type)
        if X_list and any(v == 1.0 for v in y_list) and any(v == 0.0 for v in y_list):
            mlp, threshold = train_and_threshold(X_list, y_list)
            _apply_mlp_scores(mlp, X_all, all_ids, threshold, media_results, dc["name"])
        else:
            _set_uniform_verdict(media_results, all_ids, dc["name"], "N/A")
    except Exception:
        _set_uniform_verdict(media_results, all_ids, dc["name"], "Error")


def _bucket_media_results(
    media_results: dict[int, dict], all_results: list[dict], negative_results: list[dict]
) -> None:
    """Append each media to *all_results* (any Good) or *negative_results*."""
    for mr in media_results.values():
        verdicts = mr["detector_verdicts"]
        if any(v["verdict"] == "Good" for v in verdicts.values()):
            all_results.append(mr)
        elif any(v["verdict"] in ("Bad", "Error", "N/A") for v in verdicts.values()):
            negative_results.append(mr)


def _score_one_detector(
    dc: dict,
    ds_name: str,
    X_all: "torch.Tensor",
    all_ids: list[int],
    temp_medias: dict[int, dict],
    media_results: dict[int, dict],
    scored_units: int,
    total_scoring_units: int,
    show_counter: bool,
) -> int:
    """Score one detector against one dataset; returns the new ``scored_units``."""
    score_label = f'Scoring with "{dc["name"]}" on "{ds_name}"'
    if show_counter:
        score_label += f" ({scored_units}/{total_scoring_units} items)"
    score_label += "…"
    update_find_progress(
        "running", score_label, current=scored_units, total=total_scoring_units, step=3, total_steps=_FIND_STEPS
    )

    if "live_mlp" in dc:
        _score_with_live_mlp(dc, X_all, all_ids, media_results)
    elif "detector_data" in dc:
        _score_with_detector_data(dc, X_all, all_ids, temp_medias, media_results)

    scored_units += len(all_ids)
    update_find_progress(
        "running",
        f'Scored "{dc["name"]}" on "{ds_name}"',
        current=scored_units,
        total=total_scoring_units,
        step=3,
        total_steps=_FIND_STEPS,
    )
    return scored_units


def _load_label(ds: dict, di: int, total_datasets: int) -> str:
    """Build the user-facing 'Loading dataset …' progress label."""
    label = f'Loading dataset "{ds["name"]}"'
    if total_datasets > 1:
        label += f" ({di + 1}/{total_datasets})"
    return label + "…"


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
    import gc

    import torch

    from vtsearch.embedding.matrix import get_embedding_matrix_for_snap

    dataset_ids = body["dataset_ids"]
    detector_ids = body["detector_ids"]

    if not dataset_ids:
        _abort_find(400, "No datasets selected")
    if not detector_ids:
        _abort_find(400, "No detectors selected")

    update_find_progress(
        "running", "Preparing detectors…", current=0, total=len(detector_ids), step=1, total_steps=_FIND_STEPS
    )

    datasets = _resolve_find_datasets(dataset_ids)
    detector_configs = _resolve_detector_configs(detector_ids)

    all_results: list[dict] = []
    all_negative_results: list[dict] = []
    detected_media_type = ""
    detector_names = [dc["name"] for dc in detector_configs]

    total_scoring_units = 0
    scored_units = 0

    show_counter = len(datasets) > 1 or len(detector_configs) > 1
    for di, ds in enumerate(datasets):
        update_find_progress(
            "running",
            _load_label(ds, di, len(datasets)),
            current=di,
            total=len(datasets),
            step=2,
            total_steps=_FIND_STEPS,
        )

        temp_medias = _load_find_dataset_medias(ds)
        if not temp_medias:
            continue

        if not detected_media_type:
            detected_media_type = next(iter(temp_medias.values()), {}).get("type", "")

        all_ids, all_embs = get_embedding_matrix_for_snap(temp_medias)
        X_all = torch.from_numpy(all_embs)
        total_scoring_units += len(all_ids) * len(detector_configs)
        media_results = _init_media_results(ds["name"], all_ids, temp_medias)

        for dc in detector_configs:
            scored_units = _score_one_detector(
                dc,
                ds["name"],
                X_all,
                all_ids,
                temp_medias,
                media_results,
                scored_units,
                total_scoring_units,
                show_counter,
            )

        _bucket_media_results(media_results, all_results, all_negative_results)

        del temp_medias, X_all
        gc.collect()

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

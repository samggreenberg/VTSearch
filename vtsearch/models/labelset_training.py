"""Train and score using a detector's full labelset, not just current votes.

The detector's saved labelset on disk is origin-keyed and dataset-agnostic.
At load time we resolve every element to a file via its origin importer,
embed it, and cache the resulting vector on
:attr:`~vtsearch.state.core.DetectorContext.label_embeddings`.  MLP
training (load-time and during interactive learned-sort) then iterates the
labelset directly, so labels from datasets that aren't currently loaded
still contribute.

This module is the single place that knows how to (re-)build the
``label_embeddings`` cache, build ``(X_list, y_list)`` from it, and run
:func:`~vtsearch.models.detector_training.train_and_threshold`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from vtsearch.datasets.labelset import LabeledElement, LabelSet


ProgressCallback = Callable[[str, int, int], None]


def _embedder_for_active_dataset(snap: dict[int, dict[str, Any]] | None) -> str:
    """Return the embedder name to use for fresh resolve+embed work.

    Prefers the active dataset's recorded embedder so that newly embedded
    cross-dataset vectors line up with the existing in-dataset vectors.
    """
    if not snap:
        return ""
    first = next(iter(snap.values()), {})
    return first.get("embedder", "") or ""


def _embed_one(elem: LabeledElement, *, media_type: str, embedder_name: str) -> np.ndarray | None:
    """Resolve *elem*'s origin file and embed it.  Returns ``None`` on failure."""
    from vtsearch.models.resolver import (
        _apply_clip_and_embed,
        embed_file,
        resolve_file_context,
    )

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None

        params = (elem.origin or {}).get("params", {}) if elem.origin else {}
        if isinstance(params, dict) and params.get("clipper"):
            return _apply_clip_and_embed(file_path, media_type, elem.origin, embedder_name)
        return embed_file(file_path, media_type, embedder_name)


def _pool_box_from_media(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Return the region-pooled training vector for *media*, if applicable.

    When *region_box* is set **and** the media has a stored ``patch_grid``,
    pool the box on-the-fly via
    :func:`vtsearch.media.patch_embed.box_to_vote_vector` and return that
    vector.  Otherwise return ``None`` so the caller can fall back to
    ``media["embedding"]`` — i.e. the legacy image-level training vector for
    image-level votes, single-vector embedders, and patch datasets that
    haven't been re-loaded under the v1 storage scheme.  Patch-embedder v2.
    """
    if region_box is None:
        return None
    grid = media.get("patch_grid")
    if grid is None:
        return None
    from vtsearch.media.patch_embed import box_to_vote_vector

    return box_to_vote_vector(np.asarray(grid), region_box)


def populate_label_embeddings(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
    on_progress: ProgressCallback | None = None,
) -> int:
    """Ensure every labelset element has a cached embedding on *det_ctx*.

    Resolution per element (skipping if already cached):

    1. Element resolves to a cid in the active dataset → reuse
       ``snap[cid]["embedding"]`` (no I/O).
    2. Element's origin can be resolved to a file via its importer → embed
       the file with the active dataset's embedder (or the media type's
       default).
    3. Otherwise the element is skipped — it won't contribute to training
       this session.

    Returns the number of elements that have a cached vector after this
    pass.
    """
    from vtsearch.models.labelset_elements import (
        resolve_current_dataset_cid,
        stable_element_id,
    )

    embedder_name = _embedder_for_active_dataset(snap)
    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    total = len(labelset.elements)
    cached = 0

    for idx, elem in enumerate(labelset.elements):
        eid = stable_element_id(elem)
        # Region-voted elements always re-pool from the source patch grid:
        # the cache is keyed by ``stable_element_id`` (origin / md5), which
        # is intentionally stable across region edits.  Re-pooling per
        # training pass is cheap (one patch grid + uniform mean) and is the
        # only way region_box changes propagate without an explicit cache
        # invalidation.  Image-level elements keep the cached embedding so
        # the existing fast path for non-region datasets is unchanged.
        if eid in cache and elem.region_box is None:
            cached += 1
            continue

        cid = resolve_current_dataset_cid(elem) if snap else None
        if cid is not None and snap and cid in snap:
            media = snap[cid]
            # Region-aware path: pool from ``patch_grid`` when the element has
            # a ``region_box`` annotation and the source media has a stored
            # patch grid (i.e. the dataset was loaded with a patch-region
            # embedder).  Otherwise fall back to the full-image embedding.
            pooled = _pool_box_from_media(media, elem.region_box)
            emb = pooled if pooled is not None else media.get("embedding")
            if emb is not None:
                cache[eid] = np.asarray(emb)
                cached += 1
                if on_progress:
                    on_progress(elem.origin_name or elem.filename or eid, idx + 1, total)
                continue

        # Cross-dataset path: we resolve the element via its importer and
        # embed it freshly.  No patch_grid is available here, so a stashed
        # ``region_box`` falls back to the image-level embedding — exactly
        # the design's fallback for legacy / non-patch datasets.
        emb = _embed_one(elem, media_type=media_type, embedder_name=embedder_name)
        if emb is not None:
            cache[eid] = np.asarray(emb)
            cached += 1
        if on_progress:
            on_progress(elem.origin_name or elem.filename or eid, idx + 1, total)

    return cached


def build_xy_from_labelset(
    det_ctx,
    labelset: LabelSet,
) -> tuple[list[np.ndarray], list[float]]:
    """Build ``(X_list, y_list)`` from the cached embeddings on *det_ctx*."""
    from vtsearch.models.labelset_elements import stable_element_id

    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        emb = cache.get(eid)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0 if elem.label == "good" else 0.0)
    return X_list, y_list


def train_from_labelset(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    snap: dict[int, dict[str, Any]] | None,
    on_progress: ProgressCallback | None = None,
) -> bool:
    """Populate the embedding cache, build (X, y), train, and store on *det_ctx*.

    Returns ``True`` when an MLP was trained (need ≥1 good and ≥1 bad cached
    vector); otherwise leaves ``det_ctx.model`` untouched.
    """
    populate_label_embeddings(
        det_ctx,
        labelset,
        media_type=media_type,
        snap=snap,
        on_progress=on_progress,
    )
    X_list, y_list = build_xy_from_labelset(det_ctx, labelset)
    if len(X_list) < 2:
        return False
    if not any(y == 1.0 for y in y_list) or not any(y == 0.0 for y in y_list):
        return False

    from vtsearch.models.detector_training import train_and_threshold

    mlp, threshold = train_and_threshold(X_list, y_list, snap=snap)
    det_ctx.model = mlp
    det_ctx.threshold = threshold
    return True


def labelset_train_and_score(
    det_ctx,
    labelset: LabelSet,
    *,
    media_type: str,
    clips_dict: dict[int, dict[str, Any]],
    inclusion_value: int = 0,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], float, Any | None]:
    """Train an MLP on the full labelset, then score every media in *clips_dict*.

    Replacement for :func:`~vtsearch.models.training.train_and_score` that
    trains on cross-dataset labels.  Scoring is still scoped to the active
    dataset's media, since that is what the user is sorting in the UI.
    """
    import torch

    from vtsearch.models import (
        calculate_safe_threshold,
        train_model,
    )
    from vtsearch.models.training import (
        _auto_hidden_dim,
        cross_calibration_threshold_cached,
    )

    populate_label_embeddings(det_ctx, labelset, media_type=media_type, snap=clips_dict)
    X_list, y_list = build_xy_from_labelset(det_ctx, labelset)

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]
    hidden_dim = _auto_hidden_dim(len(X_list))

    # Skip cross-cal trainings below the ``calculate_safe_threshold`` ramp
    # floor — they're expensive and the result is discarded by the blend
    # (label_weight=0 → pure GMM) or unreliable with so few labels.
    if len(X_list) < 6:
        threshold = 0.5
    else:
        threshold = cross_calibration_threshold_cached(
            X_list,
            y_list,
            input_dim,
            inclusion_value,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            det_ctx=det_ctx,
        )

    model = train_model(X, y, input_dim, inclusion_value, hidden_dim=hidden_dim)

    from vtsearch.models.embedding_matrix import get_embedding_matrix_for_snap

    all_ids, all_embs = get_embedding_matrix_for_snap(clips_dict)
    if not all_ids:
        return [], threshold, model
    X_all = torch.from_numpy(all_embs)
    with torch.no_grad():
        X_all = X_all.to(next(model.parameters()).device)
        scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().tolist()

    if safe_thresholds:
        threshold = calculate_safe_threshold(threshold, scores, len(X_list))

    results = [{"id": cid, "score": s} for cid, s in zip(all_ids, scores)]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results, threshold, model


def update_cache_for_cid(
    det_ctx,
    labelset: LabelSet,
    cid: int,
    snap: dict[int, dict[str, Any]],
) -> None:
    """Refresh the cache entry for whichever labelset element matches *cid*.

    Called after a vote in the active dataset toggles a media item.  Looks
    up which :class:`LabeledElement` has the same origin as ``snap[cid]``
    and copies its embedding into the cache.  No-op if the cid isn't
    represented in the labelset (e.g. the element was just removed).
    """
    from vtsearch.datasets.labelset import element_key, media_element_key
    from vtsearch.models.labelset_elements import stable_element_id

    media = snap.get(cid)
    if not media:
        return
    target_key = media_element_key(media)
    if target_key is None:
        return
    embedding = media.get("embedding")
    if embedding is None:
        return
    for elem in labelset.elements:
        if element_key(elem) == target_key:
            det_ctx.label_embeddings[stable_element_id(elem)] = np.asarray(embedding)
            return

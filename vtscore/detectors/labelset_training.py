"""Train and score using a detector's full labelset, not just current votes.

The detector's saved labelset on disk is origin-keyed and dataset-agnostic.
At load time we resolve every element to a file via its origin importer,
embed it, and cache the resulting vector on
:attr:`~vtscore.state.core.DetectorContext.label_embeddings`.  MLP
training (load-time and during interactive learned-sort) then iterates the
labelset directly, so labels from datasets that aren't currently loaded
still contribute.

This module is the single place that knows how to (re-)build the
``label_embeddings`` cache, build ``(X_list, y_list)`` from it, and run
:func:`~vtscore.detectors.training.train_and_threshold`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet


log = logging.getLogger(__name__)

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


def _patch_pooled_from_file(
    file_path: Path,
    *,
    media_type: str,
    embedder_name: str,
    region_box: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Run ``patch_forward`` on *file_path* and pool *region_box* into one vector.

    Mirrors the in-dataset region path (``_pool_box_from_media`` →
    :func:`box_to_vote_vector`) for the cross-dataset case: resolve the
    origin to a file, rebuild a patch grid via
    :meth:`MediaEmbedder.patch_forward`, and pool the user-drawn box.
    Returns ``None`` when the chosen embedder doesn't support patch
    regions or the forward pass produces no output, so the caller can
    fall back to an image-level embedding.
    """
    from vtscore.media import embedders_for_type, get_embedder
    from vtscore.media.embedder import media_from_path
    from vtscore.media.patch_embed import box_to_vote_vector

    embedder = None
    if embedder_name:
        try:
            embedder = get_embedder(embedder_name)
        except (KeyError, ValueError):
            embedder = None
    if embedder is None:
        avail = embedders_for_type(media_type)
        if not avail:
            return None
        embedder = avail[0]

    if not getattr(embedder, "supports_patch_regions", False):
        return None

    try:
        output = embedder.patch_forward(media_from_path(file_path))
    except Exception:
        log.warning(
            "labelset_training: patch_forward(%s) raised; region vote will fall back to image-level embedding",
            file_path,
            exc_info=True,
        )
        return None
    if output is None:
        return None
    return box_to_vote_vector(np.asarray(output.patch_grid), region_box)


def _embed_one(elem: LabeledElement, *, media_type: str, embedder_name: str) -> np.ndarray | None:
    """Resolve *elem*'s origin file and embed it.  Returns ``None`` on failure.

    When *elem* carries a ``region_box`` and the active embedder supports
    patch regions, the resolved file is patch-forwarded and the box is
    pooled via :func:`box_to_vote_vector` so the user's region-level
    training intent survives a dataset switch.  Logs a warning and falls
    back to a full-file embedding when the patch path is unavailable —
    legacy single-vector embedders, an origin carrying a clipper we'd
    have to replay against an unknown patch grid, or a failed forward
    pass.
    """
    from vtscore.detectors.resolver import (
        _apply_clip_and_embed,
        embed_file,
        resolve_file_context,
    )

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None

        origin = elem.origin or {}
        params = origin.get("params", {}) if isinstance(origin, dict) else {}
        has_clipper = isinstance(params, dict) and bool(params.get("clipper"))

        if elem.region_box is not None and not has_clipper:
            pooled = _patch_pooled_from_file(
                file_path,
                media_type=media_type,
                embedder_name=embedder_name,
                region_box=elem.region_box,
            )
            if pooled is not None:
                return pooled
            log.warning(
                "labelset_training: region_box on %r cannot be honored cross-dataset "
                "(embedder=%r does not support patch regions or patch_forward "
                "produced no output); falling back to image-level embedding",
                elem.origin_name or elem.filename or "<unknown>",
                embedder_name or "<default>",
            )
        elif elem.region_box is not None and has_clipper:
            log.warning(
                "labelset_training: region_box on %r cannot be honored cross-dataset "
                "because the origin carries a clipper; falling back to image-level "
                "embedding",
                elem.origin_name or elem.filename or "<unknown>",
            )

        if has_clipper:
            return _apply_clip_and_embed(file_path, media_type, origin, embedder_name)
        return embed_file(file_path, media_type, embedder_name)


def _pool_box_from_media(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Return the region-pooled training vector for *media*, if applicable.

    When *region_box* is set **and** the media has a stored ``patch_grid``,
    pool the box on-the-fly via
    :func:`vtscore.media.patch_embed.box_to_vote_vector` and return that
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
    from vtscore.media.patch_embed import box_to_vote_vector

    return box_to_vote_vector(np.asarray(grid), region_box)


def _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name: str) -> None:
    """Drop the label-embedding cache when the active dataset's embedder changed.

    Mixing vectors from two embedders into one MLP produces garbage.  When
    ``det_ctx.embedder`` is empty (fresh load or legacy state) we keep the
    cache; otherwise a mismatch with the active embedder forces a rebuild.
    """
    if det_ctx.embedder and embedder_name and det_ctx.embedder != embedder_name:
        det_ctx.label_embeddings.clear()
        det_ctx.label_embedding_regions.clear()


def _resolve_uncached_embedding(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    media_type: str,
    embedder_name: str,
) -> np.ndarray | None:
    """Produce a training vector for *elem*, not consulting the cache.

    Tries the in-dataset path first: when *elem* resolves to a cid in the
    active *snap*, reuse the stored embedding (region-pooling from
    ``patch_grid`` when the element has a ``region_box`` and a patch grid
    is available).  Falls back to the cross-dataset path — resolve via the
    importer and embed freshly.  Returns ``None`` when neither path
    produces a vector.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid

    if snap:
        cid = resolve_current_dataset_cid(elem)
        if cid is not None and cid in snap:
            media = snap[cid]
            pooled = _pool_box_from_media(media, elem.region_box)
            emb = pooled if pooled is not None else media.get("embedding")
            if emb is not None:
                return np.asarray(emb)

    # Cross-dataset path: ``_embed_one`` rebuilds a patch grid on the
    # resolved file when ``elem.region_box`` is set and the embedder
    # supports patch regions, then pools via ``box_to_vote_vector`` so
    # region votes survive a dataset switch.  When the patch path isn't
    # available (legacy single-vector embedder, clipper-bearing origin,
    # failed forward pass) it logs a warning and returns the image-level
    # embedding — the only signal we have left to offer training.
    emb = _embed_one(elem, media_type=media_type, embedder_name=embedder_name)
    return np.asarray(emb) if emb is not None else None


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
    from vtscore.detectors.labelset_elements import stable_element_id

    embedder_name = _embedder_for_active_dataset(snap)
    _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name)
    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    region_cache: dict[str, tuple[float, float, float, float] | None] = det_ctx.label_embedding_regions
    total = len(labelset.elements)
    cached = 0

    for idx, elem in enumerate(labelset.elements):
        eid = stable_element_id(elem)
        # Cache hit only when the cached vector was built against the same
        # ``region_box`` the element currently carries.  Region-voted
        # elements (``region_box is not None``) always fall through so the
        # patch grid is re-pooled with the latest box.  Image-level
        # elements use the cache only when the cached vector was *also*
        # built image-level — otherwise we'd return a stale region-pooled
        # vector after a region→none transition (e.g. good→bad on a
        # previously region-voted media; or un-vote / re-vote without a
        # region).  See ``logical-bug-audit.md`` finding M4.
        if eid in cache and elem.region_box is None and region_cache.get(eid) is None:
            cached += 1
            continue

        emb = _resolve_uncached_embedding(elem, snap, media_type=media_type, embedder_name=embedder_name)
        if emb is not None:
            cache[eid] = emb
            region_cache[eid] = elem.region_box
            cached += 1
        if on_progress:
            on_progress(elem.origin_name or elem.filename or eid, idx + 1, total)

    # Stamp the embedder the cache is now built against so the next call can
    # detect a switch and invalidate.  Also persist to the detector registry
    # so the smart preload predictor warms the right model next session
    # instead of the media type's default.
    if embedder_name:
        det_ctx.embedder = embedder_name
        from vtscore.detectors.registry import record_detector_embedder

        record_detector_embedder(det_ctx.detector_id, embedder_name)
    return cached


def build_xy_from_labelset(
    det_ctx,
    labelset: LabelSet,
) -> tuple[list[np.ndarray], list[float]]:
    """Build ``(X_list, y_list)`` from the cached embeddings on *det_ctx*."""
    from vtscore.detectors.labelset_elements import stable_element_id

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

    from vtscore.detectors.training import train_and_threshold

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

    Replacement for :func:`~vtscore.detectors.training.train_and_score` that
    trains on cross-dataset labels.  Scoring is still scoped to the active
    dataset's media, since that is what the user is sorting in the UI.
    """
    import torch

    from vtscore.training.mlp import _auto_hidden_dim, train_model
    from vtscore.training.thresholds import (
        calculate_safe_threshold,
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

    from vtscore.embedding.matrix import get_embedding_matrix_for_snap

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
    from vtscore.datasets.labelset import element_key, media_element_key
    from vtscore.detectors.labelset_elements import stable_element_id

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
            eid = stable_element_id(elem)
            det_ctx.label_embeddings[eid] = np.asarray(embedding)
            det_ctx.label_embedding_regions[eid] = None
            return

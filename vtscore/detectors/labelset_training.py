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
from vtscore.embedding.media_vectors import media_embedding


log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def _embedder_for_active_dataset(snap: dict[int, dict[str, Any]] | None) -> str:
    """Return the embedder name to use for fresh resolve+embed work.

    Thin alias for the canonical model-keying marker
    :func:`vtscore.embedding.binding.score_marker_embedder_for_snap`: the
    **score** embedder (patch-else-text; the v3 routing table) derived from
    the medias in *snap* so newly embedded cross-dataset vectors line up with
    the in-dataset vectors the MLP is scored against, falling back to the
    recorded primary embedder for a slot-less single-vector dataset and ``""``
    when *snap* is empty.
    """
    from vtscore.embedding.binding import score_marker_embedder_for_snap  # noqa: PLC0415

    return score_marker_embedder_for_snap(snap)


def _detector_embedder(det_ctx, snap: dict[int, dict[str, Any]] | None) -> str:
    """The embedder a *detector* resolves and embeds its labels in.

    The concrete embedder of the detector's locked type (``det_ctx.embedder_type``)
    that the active dataset supplies wins; otherwise the dataset's score
    precedence (the legacy-migration default and the cross-dataset portability
    fallback - re-embed against whatever space the new dataset uses).  This is
    :func:`keying_embedder_for_snap`, so it agrees with the model-invalidation
    and re-embed checks.  Keeping a detector's label cache keyed to its type
    means switching the active dataset under the detector no longer invalidates
    the cache as long as the new dataset binds the same concrete embedder of
    that type.  See ``docs/plans/patch-embedder.md`` → "Per-detector embedder type".
    """
    from vtscore.embedding.binding import keying_embedder_for_snap

    return keying_embedder_for_snap(det_ctx, snap)


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
    back to a full-file embedding when the patch path is unavailable -
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
            result = _apply_clip_and_embed(file_path, media_type, origin, embedder_name)
            if result is None:
                return None
            embedding, _clip_bytes = result
            return embedding
        return embed_file(file_path, media_type, embedder_name)


def _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name: str) -> None:
    """Drop the label-embedding cache when the active dataset's embedder changed.

    Mixing vectors from two embedders into one MLP produces garbage.  When
    ``det_ctx.embedder`` is empty (fresh load or legacy state) we keep the
    cache; otherwise a mismatch with the active embedder forces a rebuild.
    """
    if det_ctx.embedder and embedder_name and det_ctx.embedder != embedder_name:
        det_ctx.label_embeddings.clear()
        det_ctx.label_embedding_regions.clear()
        # Local features are descriptor sets in the old embedder's feature space;
        # a switch invalidates them too (a SIFT template can't verify against a
        # learned-feature candidate, nor vice versa).
        det_ctx.label_local_features.clear()


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
    is available).  Falls back to the cross-dataset path - resolve via the
    importer and embed freshly.  Returns ``None`` when neither path
    produces a vector.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.training import pool_box_from_media

    if snap:
        cid = resolve_current_dataset_cid(elem)
        if cid is not None and cid in snap:
            media = snap[cid]
            pooled = pool_box_from_media(media, elem.region_box)
            # Read the in-dataset vector from the detector's primary space (the
            # same space the cross-dataset path embeds into), not the media's
            # generic primary - they diverge on a multi-embedder dataset.
            emb = pooled if pooled is not None else media_embedding(media, embedder_name or None)
            if emb is not None:
                return np.asarray(emb)

    # Cross-dataset path: ``_embed_one`` rebuilds a patch grid on the
    # resolved file when ``elem.region_box`` is set and the embedder
    # supports patch regions, then pools via ``box_to_vote_vector`` so
    # region votes survive a dataset switch.  When the patch path isn't
    # available (legacy single-vector embedder, clipper-bearing origin,
    # failed forward pass) it logs a warning and returns the image-level
    # embedding - the only signal we have left to offer training.
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
       ``snap[cid]``'s primary vector (no I/O).
    2. Element's origin can be resolved to a file via its importer → embed
       the file with the active dataset's embedder (or the media type's
       default).
    3. Otherwise the element is skipped - it won't contribute to training
       this session.

    Returns the number of elements that have a cached vector after this
    pass.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    embedder_name = _detector_embedder(det_ctx, snap)
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
        # built image-level - otherwise we'd return a stale region-pooled
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


# ---------------------------------------------------------------------------
# Cross-dataset local features (structural / SIFT-VLAD detectors)
# ---------------------------------------------------------------------------


def _resolve_uncached_local_features(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    embedder,
) -> Any | None:
    """Re-derive *elem*'s :class:`StructuralFeatures`, not consulting the cache.

    Tries the in-dataset path first: when *elem* resolves to a cid in the
    active *snap* that already carries ``local_features``, reuse them (no I/O).
    Otherwise resolve the origin to a file via its importer and run the
    embedder's ``local_features_forward`` to detect features freshly.  The
    **full** (unfiltered) feature set is returned; any ``region_box`` is applied
    downstream at template-build time.  Returns ``None`` when neither path
    yields features.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.media.embedder import media_from_path
    from vtscore.media.structural import StructuralFeatures

    if snap:
        cid = resolve_current_dataset_cid(elem)
        if cid is not None and cid in snap:
            feats = snap[cid].get("local_features")
            if isinstance(feats, StructuralFeatures) and feats.count > 0:
                return feats

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None
        try:
            feats = embedder.local_features_forward(media_from_path(file_path))
        except Exception:
            log.warning(
                "labelset_training: local_features_forward(%s) raised; "
                "this label won't contribute a structural template",
                elem.origin_name or elem.filename or "<unknown>",
                exc_info=True,
            )
            return None
    if feats is None or getattr(feats, "count", 0) == 0:
        return None
    return feats


def populate_label_local_features(
    det_ctx,
    labelset: LabelSet,
    *,
    snap: dict[int, dict[str, Any]] | None,
) -> int:
    """Ensure every labelled element has cached local features on *det_ctx*.

    A no-op (returns 0) unless the active dataset's embedder is structural
    (``supports_geometric_verification``) - non-structural detectors never need
    local features.  For a structural detector it re-derives the
    :class:`~vtscore.media.structural.StructuralFeatures` for each good/bad
    element (reusing the active dataset's stored features when the element is
    loaded, resolving the origin file otherwise) and caches them on
    ``det_ctx.label_local_features`` keyed by ``stable_element_id``.  The
    embedder-switch invalidation in :func:`_maybe_clear_cache_on_embedder_switch`
    clears this cache alongside the embedding cache.

    Returns the number of elements that have cached features after this pass.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    embedder_name = _detector_embedder(det_ctx, snap)
    embedder = None
    if embedder_name:
        from vtscore.media import get_embedder

        try:
            embedder = get_embedder(embedder_name)
        except (KeyError, ValueError):
            embedder = None
    if embedder is None or not getattr(embedder, "supports_geometric_verification", False):
        return 0

    cache: dict[str, Any] = det_ctx.label_local_features
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        if eid in cache:
            continue
        feats = _resolve_uncached_local_features(elem, snap, embedder=embedder)
        if feats is not None:
            cache[eid] = feats
    return len(cache)


def _labelset_feature_snapshot(
    det_ctx,
    labelset: LabelSet,
) -> tuple[dict[str, dict[str, Any]], dict[str, None], dict[str, None], dict[str, tuple[float, float, float, float]]]:
    """Project the cached local features into the chokepoint's vote/snap shape.

    Builds a synthetic ``feature_snap`` (``element_id -> {"local_features": ...}``)
    plus ``good_votes`` / ``bad_votes`` / ``region_boxes`` keyed by the same
    ``stable_element_id`` so :func:`maybe_structural_rerank` can build templates
    and train the verification classifier against the cross-dataset labelset
    exactly as it does against in-dataset votes.
    """
    from vtscore.detectors.labelset_elements import stable_element_id

    cache: dict[str, Any] = det_ctx.label_local_features
    feature_snap: dict[str, dict[str, Any]] = {}
    good_votes: dict[str, None] = {}
    bad_votes: dict[str, None] = {}
    region_boxes: dict[str, tuple[float, float, float, float]] = {}
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        eid = stable_element_id(elem)
        feats = cache.get(eid)
        if feats is None:
            continue
        feature_snap[eid] = {"local_features": feats}
        if elem.label == "good":
            good_votes[eid] = None
            if elem.region_box is not None:
                region_boxes[eid] = elem.region_box
        else:
            bad_votes[eid] = None
    return feature_snap, good_votes, bad_votes, region_boxes


def maybe_labelset_structural_rerank(
    det_ctx,
    labelset: LabelSet,
    results: list[dict[str, Any]],
    threshold: float,
    snap: dict[int, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], float]:
    """Stage-2 structural re-rank for the saved-detector (labelset) sort path.

    The counterpart to the vote-driven re-rank wired into
    :func:`~vtscore.detectors.training.train_and_score`: when a saved structural
    detector is sorted against a (possibly different) loaded dataset, this
    re-derives the labelset's local features, builds the RegionYes templates and
    verification classifier from them, and geometrically re-ranks the active
    dataset's Stage-1 shortlist.  A no-op for non-structural datasets (gated on
    the active snapshot carrying ``local_features``) and when no labelled element
    yields a usable template.
    """
    from vtscore.training.structural_similarity import maybe_structural_rerank, snapshot_is_structural

    if not snap or not snapshot_is_structural(snap):
        return results, threshold
    populate_label_local_features(det_ctx, labelset, snap=snap)
    feature_snap, good_votes, bad_votes, region_boxes = _labelset_feature_snapshot(det_ctx, labelset)
    if not good_votes:
        return results, threshold
    return maybe_structural_rerank(
        results,
        threshold,
        snap,
        good_votes,
        bad_votes,
        region_boxes,
        det_ctx,
        feature_snap=feature_snap,
    )


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

    # populate_label_embeddings stamped det_ctx.embedder with the space the
    # labels were embedded in; score the safe-threshold pass in that same space.
    # Pass det_ctx so the fold orderings are cached for a no-retrain Inclusion
    # slide (otherwise the slide can't move the cutoff — see train_and_threshold).
    mlp, threshold = train_and_threshold(
        X_list, y_list, snap=snap, embedder_name=det_ctx.embedder or None, det_ctx=det_ctx
    )
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

    Counterpart to :func:`~vtscore.detectors.training.train_and_score` that
    trains on cross-dataset labels.  It assembles ``(X_list, y_list)`` from
    the resolved labelset (populating the embedding cache on the way) and
    then defers the threshold → train → score → format tail to the shared
    :func:`~vtscore.detectors.training._train_and_score_xy` core, so the two
    pipelines stay in lock-step (region-aware scoring, NaN sanitisation,
    safe-threshold blending).  Scoring is still scoped to the active
    dataset's media, since that is what the user is sorting in the UI.
    """
    from vtscore.detectors.training import _train_and_score_xy

    populate_label_embeddings(det_ctx, labelset, media_type=media_type, snap=clips_dict)
    X_list, y_list = build_xy_from_labelset(det_ctx, labelset)
    results, threshold, model = _train_and_score_xy(
        X_list,
        y_list,
        clips_dict,
        inclusion_value=inclusion_value,
        safe_thresholds=safe_thresholds,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        det_ctx=det_ctx,
    )

    # Stage-2 structural re-rank for a saved structural detector reloaded
    # cross-dataset (the labelset counterpart to the vote-driven re-rank in
    # ``train_and_score``).  A no-op for every non-structural dataset.
    results, threshold = maybe_labelset_structural_rerank(det_ctx, labelset, results, threshold, clips_dict)
    return results, threshold, model

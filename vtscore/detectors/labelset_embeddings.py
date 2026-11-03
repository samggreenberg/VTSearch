"""Resolve a detector's labelset into its label-embedding caches.

The detector's saved labelset on disk is origin-keyed and dataset-agnostic.
:func:`populate_label_embeddings` resolves every element to a vector - from the
active snapshot when the element is loaded, else from its origin file via the
importer - and caches it on
:attr:`~vtscore.state.core.DetectorContext.label_embeddings`, alongside the
region caches a patch detector floods and calibrates from.  The rest of the
labelset tier reads those caches: :mod:`~vtscore.detectors.labelset_matrix`
assembles ``(X, y)`` from them and :mod:`~vtscore.detectors.labelset_training`
trains on it.

This module is the single place that knows how to (re-)build the
``label_embeddings`` cache.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.embedding.binding import keying_embedder_for_snap
from vtscore.embedding.media_vectors import media_embedding


log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def _lookups_for_snap(
    snap: dict[int, dict[str, Any]] | None,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None:
    """Build the ``(origin, md5, name)`` media-lookup triple for *snap*, once.

    Returns ``None`` when *snap* is empty/absent (the in-dataset resolution
    paths are all guarded on ``if snap:`` anyway).  Loop callers build this a
    single time and thread it into :func:`resolve_current_dataset_cid` so the
    per-element cid resolution is O(1) instead of rebuilding the tables (which
    ``json.dumps`` every origin) for each of ``labels`` elements.
    """
    if not snap:
        return None
    from vtscore.state import build_media_lookup

    return build_media_lookup(snap)


def _patch_embedder(*, media_type: str, embedder_name: str):
    """The embedder the cross-dataset patch path forwards through, or ``None``.

    Resolves the detector's embedder, else the media type's default, and
    returns it only when it produces a patch grid.  ``None`` therefore means
    "there is no patch space to pool a ``region_box`` from" - the expected
    state for a whole-image embedder, not a failure.
    """
    from vtscore.media import embedders_for_type, get_embedder

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
    return embedder


def _patch_output_from_file(
    file_path: Path,
    *,
    media_type: str,
    embedder_name: str,
):
    """Re-derive a media's :class:`PatchEmbedOutput` from its file, or ``None``.

    Resolves the embedder via :func:`_patch_embedder` and runs
    :meth:`MediaEmbedder.patch_forward`.  Returns ``None`` when the
    embedder doesn't support patch regions or the forward pass produces no
    output.  Shared by the Good-vote nearest-patch path and the Bad-vote flood
    path so both see the identical grid cross-dataset.
    """
    from vtscore.media.embedder import media_from_path

    embedder = _patch_embedder(media_type=media_type, embedder_name=embedder_name)
    if embedder is None:
        return None

    try:
        output = embedder.patch_forward(media_from_path(file_path))
    except Exception:
        log.warning(
            "labelset_embeddings: patch_forward(%s) raised; region vote will fall back to image-level embedding",
            file_path,
            exc_info=True,
        )
        return None
    return output


def _embedder_supports_patch_regions(embedder_name: str) -> bool:
    """Whether *embedder_name* produces a patch grid at all.

    This is the labelset tier's patch gate - the counterpart of
    :func:`~vtscore.detectors.training._scores_in_patch_space`, phrased against
    the detector's own embedder because the cross-dataset branch has no snap to
    resolve a patch slot from.  It answers "does this detector score in a patch
    space?", and *every* patch behaviour on this path hangs off it: the raw-patch
    pool under a Good element's ``region_box``, a Bad element's flooded
    negatives, and the calibration score-row stacks.

    Two things it buys:

    * On a legacy single-vector detector there is no grid to re-derive, so
      probing every element's origin file would be pure I/O for a guaranteed
      ``None``.
    * On a **multi-embedder** dataset it keeps a detector locked to the text or
      structural space off the patch grid entirely.  That detector is scored
      against its own space's full-image vectors
      (:func:`~vtscore.detectors.training._score_all_media`), so reading the
      media's stored ``patch_grid`` - which lives in the *patch* embedder's
      space - would train it on vectors from an unrelated space.
    """
    if not embedder_name:
        return False
    from vtscore.media import get_embedder

    try:
        embedder = get_embedder(embedder_name)
    except (KeyError, ValueError):
        return False
    return bool(getattr(embedder, "supports_patch_regions", False))


def _patch_pooled_from_file(
    file_path: Path,
    *,
    media_type: str,
    embedder_name: str,
    region_box: tuple[float, float, float, float],
) -> np.ndarray | None:
    """Run ``patch_forward`` on *file_path* and take the patch nearest *region_box*.

    Mirrors the in-dataset path (:func:`~vtscore.detectors.training.pool_box_from_media`
    → :func:`nearest_patch_to_box`) for the cross-dataset case: re-derive the
    patch grid via :func:`_patch_output_from_file` and pick the raw patch under
    the user-drawn box - so a Good region-vote trains on one of the very rows
    the head will max-pool over on the new dataset.  Returns ``None`` when the
    grid can't be rebuilt, so the caller can fall back to an image-level
    embedding.
    """
    from vtscore.media.patch_embed import nearest_patch_to_box

    output = _patch_output_from_file(file_path, media_type=media_type, embedder_name=embedder_name)
    if output is None:
        return None
    return nearest_patch_to_box(np.asarray(output.patch_grid), region_box)


class _LabelVector(NamedTuple):
    """A label's training vector and the ``region_box`` it is final for.

    ``region_box`` is the element's own box whenever resolving that box again
    would produce *vec* again: it was pooled under the box, or the box can't be
    used at all (a whole-image embedder, a clipped or converted origin).  It is
    ``None`` when the element had no box, and when a patch pool failed, so
    :func:`populate_label_embeddings` retries that fallback on its next pass
    rather than keeping it.
    """

    vec: np.ndarray
    region_box: tuple[float, float, float, float] | None


def _as_box(box) -> tuple[float, float, float, float] | None:
    """*box* as a tuple, so a JSON list and a tuple of the same box compare equal."""
    return None if box is None else (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _embed_one(elem: LabeledElement, *, media_type: str, embedder_name: str) -> _LabelVector | None:
    """Resolve *elem*'s origin file and embed it.  Returns ``None`` on failure.

    When *elem* carries a ``region_box`` and the active embedder supports
    patch regions, the resolved file is patch-forwarded and the raw patch under
    the box is taken via :func:`nearest_patch_to_box` so the
    user's region-level training intent survives a dataset switch.  Falls
    back to a full-file embedding when the patch path is unavailable, logged
    at the level its cause deserves:

    * a whole-image embedder has no patch space, so the box stays on the
      label unused - expected, and logged at DEBUG only.  A WARNING here
      fired once per boxed label per retrain (#4192);
    * an origin whose unit was cut or converted out of the resolved file (a
      clipper, a converter): the box is on that unit, not on the file, so the
      patch path can't run on the file - a known limitation, WARNING;
    * a patch-capable embedder whose forward pass produced nothing - a real
      failure, WARNING.
    """
    from vtscore.detectors.resolver import (
        _apply_clip_and_embed,
        _needs_clip_replay,
        embed_file,
        resolve_file_context,
    )

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None

        origin = elem.origin or {}
        params = origin.get("params", {}) if isinstance(origin, dict) else {}
        # A clipped or converted unit: *file_path* is its source (the parent
        # recording, the PDF), so the unit has to be replayed out of it.
        derived = isinstance(params, dict) and _needs_clip_replay(params)

        box = _as_box(elem.region_box)
        if elem.region_box is not None:
            if _patch_embedder(media_type=media_type, embedder_name=embedder_name) is None:
                # The in-dataset path's ``patch_capable`` gate reaches the same
                # image-level vector silently; a box is legitimate data to keep
                # for a later region detector, not a fault.
                log.debug(
                    "labelset_embeddings: region_box on %r unused; embedder=%r is whole-image, "
                    "training on the image-level embedding",
                    elem.origin_name or elem.filename or "<unknown>",
                    embedder_name or "<default>",
                )
            elif not derived:
                pooled = _patch_pooled_from_file(
                    file_path,
                    media_type=media_type,
                    embedder_name=embedder_name,
                    region_box=elem.region_box,
                )
                if pooled is not None:
                    return _LabelVector(np.asarray(pooled), box)
                log.warning(
                    "labelset_embeddings: region_box on %r cannot be honored cross-dataset "
                    "(embedder=%r patch_forward produced no output); falling back to "
                    "image-level embedding",
                    elem.origin_name or elem.filename or "<unknown>",
                    embedder_name or "<default>",
                )
                box = None  # a failed forward pass is retried, not kept
            else:
                log.warning(
                    "labelset_embeddings: region_box on %r cannot be honored cross-dataset "
                    "because the origin is a clipped or converted unit; falling back to "
                    "image-level embedding",
                    elem.origin_name or elem.filename or "<unknown>",
                )

        if derived:
            result = _apply_clip_and_embed(file_path, media_type, origin, embedder_name)
            if result is None:
                return None
            embedding, _clip_bytes = result
        else:
            embedding = embed_file(file_path, media_type, embedder_name)
        return _LabelVector(np.asarray(embedding), box) if embedding is not None else None


def _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name: str) -> None:
    """Drop the label-embedding cache when the active dataset's embedder changed.

    Mixing vectors from two embedders into one head produces garbage.  When
    ``det_ctx.embedder`` is empty (fresh load or legacy state) we keep the
    cache; otherwise a mismatch with the active embedder forces a rebuild.
    """
    if det_ctx.embedder and embedder_name and det_ctx.embedder != embedder_name:
        det_ctx.label_embeddings.clear()
        det_ctx.label_embedding_regions.clear()
        # Flooded patch negatives and the region scoring stacks live in the old
        # embedder's space too.
        det_ctx.label_negative_regions.clear()
        det_ctx.label_score_regions.clear()
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
    patch_capable: bool,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> _LabelVector | None:
    """Produce a training vector for *elem*, not consulting the cache.

    Tries the in-dataset path first: when *elem* resolves to a cid in the
    active *snap*, reuse the stored embedding (taking the raw patch under the
    ``region_box`` when the element carries one).
    Falls back to the cross-dataset path - resolve via the importer and embed
    freshly.  Returns ``None`` when neither path produces a vector, else a
    :class:`_LabelVector` naming the box the vector is final for.

    *patch_capable* (:func:`_embedder_supports_patch_regions`) gates the
    raw-patch pool: a detector that doesn't score in a patch space trains on the
    image-level vector of its own space even when the media carries a grid,
    because the grid's vectors belong to the dataset's *patch* embedder.  The
    cross-dataset branch is already gated the same way inside
    :func:`_patch_output_from_file`.

    *lookups* is the pre-built ``build_media_lookup`` triple for *snap*; loop
    callers pass it so the cid resolution doesn't rebuild the tables per element.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.training import pool_box_from_media

    if snap:
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            media = snap[cid]
            pooled = pool_box_from_media(media, elem.region_box) if patch_capable else None
            if pooled is not None:
                return _LabelVector(np.asarray(pooled), _as_box(elem.region_box))
            # Read the in-dataset vector from the detector's primary space (the
            # same space the cross-dataset path embeds into), not the media's
            # generic primary - they diverge on a multi-embedder dataset.
            emb = media_embedding(media, embedder_name or None)
            if emb is not None:
                # A detector with no patch space can't use a box, so this is its
                # final answer for one; a patch detector whose pool failed retries.
                return _LabelVector(np.asarray(emb), None if patch_capable else _as_box(elem.region_box))

    # Cross-dataset path: ``_embed_one`` re-derives the patch grid on the
    # resolved file when ``elem.region_box`` is set and the embedder
    # supports patch regions, then takes the nearest raw patch so
    # region votes survive a dataset switch.  When the patch path isn't
    # available (whole-image embedder, clipper-bearing origin, failed
    # forward pass) it returns the image-level embedding - the only signal
    # we have left to offer training - warning only for the last two.
    return _embed_one(elem, media_type=media_type, embedder_name=embedder_name)


def _resolve_score_rows(
    elem: LabeledElement,
    snap: dict[int, dict[str, Any]] | None,
    *,
    media_type: str,
    embedder_name: str,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None = None,
) -> list[np.ndarray] | None:
    """*elem*'s MaxPatch score-row stack (image-level vector + every raw patch), or ``None``.

    In-dataset: read the resolved media's stored ``patch_grid`` (no I/O), via
    the same :func:`~vtscore.embedding.matrix.media_score_rows` the scorer
    uses.  Cross-dataset: re-derive the grid from the origin file via
    :func:`_patch_output_from_file`.  Returns ``None`` for non-patch
    datasets/elements (and clipper-bearing origins, whose patch grid we can't
    reconstruct) so the callers fall back to their image-level behaviour - a
    single negative for a Bad element, the training row as the scoring row.

    One resolution answers both questions the caller has, because under
    MaxPatch the rows a Bad vote floods and the rows the scorer max-pools are
    *the same rows*.  (Under the old HAC tree they differed: the flood skipped
    the internal merge nodes - see
    :func:`~vtscore.detectors.training.bad_negative_vecs`.)

    *lookups* is the pre-built ``build_media_lookup`` triple for *snap*; loop
    callers pass it so the cid resolution doesn't rebuild the tables per element.
    """
    from vtscore.detectors.labelset_elements import resolve_current_dataset_cid
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.embedding.matrix import media_score_rows

    if snap:
        cid = resolve_current_dataset_cid(elem, lookups)
        if cid is not None and cid in snap:
            media = snap[cid]
            if media.get("patch_grid") is None:
                return None
            rows = media_score_rows(media, embedder_name or None)
            return list(rows) if rows is not None else None

    origin = elem.origin or {}
    params = origin.get("params", {}) if isinstance(origin, dict) else {}
    if isinstance(params, dict) and params.get("clipper"):
        return None

    with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
        if file_path is None:
            return None
        output = _patch_output_from_file(file_path, media_type=media_type, embedder_name=embedder_name)
    if output is None:
        return None
    grid = np.asarray(output.patch_grid, dtype=np.float32)
    cls_row = np.asarray(output.cls_vec, dtype=np.float32).reshape(1, -1)
    return list(np.concatenate([cls_row, grid.reshape(-1, grid.shape[-1])], axis=0))


def _cache_region_vectors(
    elem: LabeledElement,
    eid: str,
    snap: dict[int, dict[str, Any]] | None,
    *,
    media_type: str,
    embedder_name: str,
    lookups: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]] | None,
    neg_cache: dict[str, list[np.ndarray]],
    score_cache: dict[str, list[np.ndarray]],
    patch_capable: bool,
) -> None:
    """Top up *elem*'s two region caches from a single grid resolution.

    A patch element's score-row stack feeds both: the negatives a Bad element
    floods (the cross-dataset counterpart of
    :func:`~vtscore.detectors.training.bad_negative_vecs`), and the rows the
    *scorer* max-pools that image over - which threshold calibration collapses
    every bag, Good and Bad alike, over.  Under MaxPatch those are the same
    rows, so the two caches are fed from one resolution, cached so re-sorts
    don't re-resolve.

    The resolution is skipped entirely unless the detector scores in a patch
    space (*patch_capable*; see :func:`_embedder_supports_patch_regions`).  On a
    legacy single-vector detector it would return ``None`` anyway, so no origin
    file is probed for a grid that cannot exist.  On a multi-embedder dataset
    the skip is load-bearing: a semantic- or structural-locked detector is
    scored against its own space's full-image vectors, so flooding it with the
    dataset's patch grid would train it on a different embedding space (and
    raise :class:`~vtscore.embedding.precomputed.MismatchedVectorError` outright
    when the two dimensions differ).  Both caches stay empty, and
    :func:`~vtscore.detectors.labelset_matrix.build_xy_from_labelset` falls back
    to the single image-level vector per element.
    """
    if not patch_capable:
        return
    wants_negatives = elem.label == "bad" and eid not in neg_cache
    wants_score_rows = elem.label in ("good", "bad") and eid not in score_cache
    if not (wants_negatives or wants_score_rows):
        return
    rows = _resolve_score_rows(elem, snap, media_type=media_type, embedder_name=embedder_name, lookups=lookups)
    if rows is None:
        return
    if wants_negatives:
        neg_cache[eid] = rows
    if wants_score_rows:
        score_cache[eid] = rows


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

    # Labels are resolved and embedded in the same space the model-invalidation
    # and re-embed checks key on, so the label cache stays valid across an
    # active-dataset switch that keeps binding the same concrete embedder of the
    # detector's type.  See ``docs/ARCHITECTURE.md`` → "Per-detector
    # embedder type".
    embedder_name = keying_embedder_for_snap(det_ctx, snap)
    _maybe_clear_cache_on_embedder_switch(det_ctx, embedder_name)
    cache: dict[str, np.ndarray] = det_ctx.label_embeddings
    region_cache: dict[str, tuple[float, float, float, float] | None] = det_ctx.label_embedding_regions
    total = len(labelset.elements)
    cached = 0

    neg_cache: dict[str, list[np.ndarray]] = det_ctx.label_negative_regions
    score_cache: dict[str, list[np.ndarray]] = det_ctx.label_score_regions
    patch_capable = _embedder_supports_patch_regions(embedder_name)

    # Build the origin/md5/name lookup tables once from *snap* and thread them
    # through every element's cid resolution, so the pass is O(N + labels)
    # instead of O(labels × N).
    lookups = _lookups_for_snap(snap)

    for idx, elem in enumerate(labelset.elements):
        eid = stable_element_id(elem)
        _cache_region_vectors(
            elem,
            eid,
            snap,
            media_type=media_type,
            embedder_name=embedder_name,
            lookups=lookups,
            neg_cache=neg_cache,
            score_cache=score_cache,
            patch_capable=patch_capable,
        )

        # ``region_cache`` holds the box each cached vector is final for (see
        # ``_LabelVector``), so a hit means "resolving again would produce this
        # vector".  A box edit misses and re-pools.  A region→none transition
        # (good→bad on a region-voted media, un-vote / re-vote without a box)
        # misses rather than keeping a stale pooled vector - logical-bug-audit
        # finding M4.  A failed pool was cached against ``None``, so it misses
        # and is retried.  An unchanged box hits, which is what spares a boxed
        # label that doesn't resolve in *snap* a file fetch and embed on every
        # pass (#4192).
        if eid in cache and region_cache.get(eid) == _as_box(elem.region_box):
            cached += 1
            continue

        resolved = _resolve_uncached_embedding(
            elem,
            snap,
            media_type=media_type,
            embedder_name=embedder_name,
            patch_capable=patch_capable,
            lookups=lookups,
        )
        if resolved is not None:
            cache[eid] = resolved.vec
            region_cache[eid] = resolved.region_box
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

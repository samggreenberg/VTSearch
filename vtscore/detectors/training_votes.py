"""Turning a detector's votes into training rows.

A Good vote trains on one row - the raw patch nearest its box, or the
image-level vector - and a Bad vote on every row the scorer will max-pool
(region flooding).  :func:`_build_vote_xy` assembles ``(X_list, y_list,
groups, score_rows)`` from a vote set; :func:`pool_box_from_media` and
:func:`bad_negative_vecs` are the per-vote rules the labelset path and the
eval harness share.  Part of the detector training pipeline
(:mod:`vtscore.detectors.training`); reads only
:mod:`vtscore.detectors.training_snap`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vtscore.detectors.training_snap import (
    _patch_embedder_for_snap,
    _score_embedder_for_snap,
    _scores_in_patch_space,
)


def pool_box_from_media(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Return the region training vector for *media*, or ``None``.

    **MaxPatch Good-vote rule.**  When *region_box* is set and the media
    carries a ``patch_grid``, return the single raw patch vector nearest the
    box (:func:`vtscore.media.patch_embed.nearest_patch_to_box`) - one of the
    very rows :func:`vtscore.embedding.matrix.media_score_rows` will score the
    image over, so the Good vote is a fair representative of what the detector
    actually evaluates.  Returns ``None`` (the caller then uses the image-level
    ``embedding``, which is row 0 of that same stack) for a boxless vote and
    for legacy single-vector embedders that carry no grid.

    This replaced the HAC snap-to-node rule in #2886: over 23 scale-band Visual
    Genome categories the raw patch beat the tree's best-IoU node on both
    halves of the error at every scale band, and by the largest margin exactly
    where the hypothesis said it would - below leaf scale, where the tree's
    smallest pooled candidate already blends object with context while a raw
    patch is a near-pure object sample.  See
    ``docs/experiments/2026-07-29-max-patch/REPORT.md``.

    Note the drawn box's width and height are discarded in essentially every
    case: ``nearest_patch_to_box`` collapses to "the patch nearest the box
    centre" unless the box is thinner than one cell.  That is the shipped
    design and what the study measured, not an oversight - the natural 4-DOF
    alternative (mean of the patches inside the box) is a per-vote amalgam that
    can never be a per-image scored row, so it breaks the train/score
    invariant by construction.

    Shared by the in-dataset vote path (:func:`_training_vec_for_vote`) and
    the cross-dataset labelset path
    (:func:`vtscore.detectors.labelset_embeddings._resolve_uncached_embedding`).
    """
    if region_box is None:
        return None

    grid = media.get("patch_grid")
    if grid is None:
        return None
    from vtscore.media.patch_embed import nearest_patch_to_box  # noqa: PLC0415

    return nearest_patch_to_box(np.asarray(grid), region_box)


def _training_vec_for_vote(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
    embedder_name: str | None = None,
) -> np.ndarray:
    """Return the training vector for one vote on *media*.

    Region-pools via :func:`pool_box_from_media` when the vote designated a
    box and *media* carries a ``patch_grid``; otherwise falls back to the
    image-level vector of *embedder_name* (the score embedder) - or the
    primary vector when ``None``.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    pooled = pool_box_from_media(media, region_box)
    return pooled if pooled is not None else media_embedding(media, embedder_name)


def bad_negative_vecs(
    media: dict[str, Any],
    embedder_name: str | None = None,
) -> list[np.ndarray]:
    """Negative training vectors contributed by one Bad vote on *media*.

    On **patch** media (carrying a ``patch_grid``) a Bad vote floods the
    image-level vector plus **every raw patch** - i.e. exactly
    :func:`vtscore.embedding.matrix.media_score_rows`, the same ~197 rows
    :func:`_score_all_media` max-pools.  This is the multiple-instance-learning
    treatment of a rejected image: since inference scores an image by its
    **best** row, a Bad vote asserts that *no* row of it should score high, so
    every row is trained down.

    The flood and the scoring stack are **the same function call**, which is
    the point.  Under the old HAC tree they were deliberately different - the
    flood covered the CLS node and the leaves but skipped the internal merge
    nodes, a measured exception (#2731) forced by internals being renormalised
    convex-hull points that are not dominated by their own leaves.  MaxPatch
    has no internals, so the gap closes and every scored row is a flooded row.

    All ~197 rows share one bag id in :func:`_build_vote_xy`, so a rejected
    image still counts as **one** vote for weighting, splitting, and the
    threshold's small-count ramp.

    Non-patch media contribute a single image-level vector, exactly as before,
    so every legacy single-vector dataset is byte-for-byte unchanged.
    """
    from vtscore.embedding.matrix import media_score_rows  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    rows = media_score_rows(media, embedder_name)
    if rows is None:
        return [media_embedding(media, embedder_name)]
    return list(rows)


def inference_score_rows(
    media: dict[str, Any],
    embedder_name: str | None = None,
) -> np.ndarray | None:
    """The row stack *media* is max-pooled over at inference, or ``None``.

    A thin alias for :func:`vtscore.embedding.matrix.media_score_rows` - the
    same rows :func:`vtscore.embedding.matrix._build_region_arrays` flattens
    into the matrix :func:`_score_all_media` scores: image-level vector + every
    raw patch on a patch media (~197 rows), a single image-level vector on a
    grid-less one.

    Used to calibrate in inference geometry: a voted image's bag must collapse
    over the same rows the scorer will pool, not over the (Good: 1, Bad: ~197)
    rows the fold model happened to train on.  See
    :func:`_calibration_score_rows`.
    """
    from vtscore.embedding.matrix import media_score_rows  # noqa: PLC0415

    return media_score_rows(media, embedder_name)


def _vote_score_rows(
    media: dict[str, Any],
    embedder_name: str | None,
    row_embedder: str | None,
    *,
    patch_rows: bool,
) -> np.ndarray | None:
    """The rows the scorer max-pools *media* over, in the detector's own space.

    The patch stack (:func:`inference_score_rows`) when the detector scores in
    the patch space, else the single image-level row of *embedder_name* - which
    is exactly what :func:`_score_all_media`'s embedding-matrix fallback scores
    that media by.  See :func:`_scores_in_patch_space`.
    """
    if patch_rows:
        return inference_score_rows(media, row_embedder)
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    vec = media_embedding(media, embedder_name)
    return None if vec is None else np.asarray(vec, dtype=np.float32).reshape(1, -1)


def _vote_negative_vecs(
    media: dict[str, Any],
    embedder_name: str | None,
    row_embedder: str | None,
    *,
    patch_rows: bool,
) -> list[np.ndarray]:
    """The rows one Bad vote on *media* trains down - the rows the scorer pools.

    The flood (:func:`bad_negative_vecs`) when the detector scores in the patch
    space, else the one image-level vector of *embedder_name*.  A non-patch
    detector never max-pools the grid, so flooding it would train the model on
    rows from another embedding space that nothing ever scores.
    """
    if patch_rows:
        return bad_negative_vecs(media, row_embedder)
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    return [media_embedding(media, embedder_name)]


def _build_vote_xy(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    region_boxes: dict[int, tuple[float, float, float, float]],
    embedder_name: str | None = None,
) -> tuple[list[np.ndarray], list[float], list, dict]:
    """Build ``(X_list, y_list, groups, score_rows)`` from filtered votes.

    Good votes that designated a region train on the nearest raw patch via
    :func:`_training_vec_for_vote` (one row each).  Bad votes are expanded by
    :func:`bad_negative_vecs`: one row per image-level vector on a legacy
    dataset, or the image-level vector + every raw patch on a patch dataset
    (region flooding).

    ``groups`` carries one bag id per row - ``("g", cid)`` for a Good vote,
    ``("b", cid)`` shared across all of a Bad vote's ~197 flooded rows - so the
    downstream trainer/calibrator can balance and split by **image**, not by
    row.  On a legacy dataset every bag holds exactly one row, so ``groups`` is
    1:1 with the rows and the whole path collapses to the pre-flood behaviour.
    The caller (:func:`_train_and_score_xy`) enforces the ≥2-samples /
    ≥1-good / ≥1-bad guard.

    ``score_rows`` maps each bag id to the row stack that voted image is
    *scored* over at inference (:func:`inference_score_rows`) - the whole
    ~197-row patch stack on a patch media - so calibration can collapse a Good
    bag and a Bad bag the same way :func:`_score_all_media` collapses any
    image.  Without it a Good bag is a max over its 1 training row against a
    Bad bag's max over ~197, and the calibrated cut lands high.

    *embedder_name* is the detector's primary embedder; when ``None`` the
    dataset score precedence for *clips_dict* is used (the pre-per-detector
    behaviour).  Either way the head trains in the same space
    :func:`_score_all_media` scores against.

    All three patch behaviours above - the Good vote's region pool, the Bad
    vote's flood, and the recorded calibration stack - are gated by
    :func:`_scores_in_patch_space`, the same gate :func:`_score_all_media`
    pools under.  A detector locked to the text or structural space of a
    multi-embedder dataset therefore trains on that space's **full-image**
    vectors, exactly the rows it will be scored over; without the gate its Bad
    votes flooded patch-space rows (and its boxed Good votes pooled one) into a
    model scored in a different space entirely - a
    :class:`~vtscore.embedding.precomputed.MismatchedVectorError` when the two
    dimensions differ, silent garbage negatives when they happen to match.
    """
    patch_rows = _scores_in_patch_space(clips_dict, embedder_name)
    if embedder_name is None:
        embedder_name = _score_embedder_for_snap(clips_dict)
    # The image-level row of a patch stack belongs to the *patch-slot* embedder
    # (the space the grid lives in), which is what the scoring matrix reads -
    # see ``matrix._patch_embedder_for_region_snap``.  On a single-embedder
    # patch dataset this is the score embedder anyway; on a text+patch dataset
    # it keeps the flooded / calibrated rows out of the text space.
    row_embedder = (_patch_embedder_for_snap(clips_dict) or embedder_name) if patch_rows else embedder_name
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows: dict = {}

    def _record_score_rows(group: tuple, cid: int) -> None:
        rows = _vote_score_rows(clips_dict[cid], embedder_name, row_embedder, patch_rows=patch_rows)
        if rows is not None:
            score_rows[group] = rows

    for cid in good_votes:
        if cid in clips_dict:
            box = region_boxes.get(cid) if patch_rows else None
            X_list.append(_training_vec_for_vote(clips_dict[cid], box, embedder_name))
            y_list.append(1.0)
            groups.append(("g", cid))
            _record_score_rows(("g", cid), cid)
    for cid in bad_votes:
        if cid in clips_dict:
            for vec in _vote_negative_vecs(clips_dict[cid], embedder_name, row_embedder, patch_rows=patch_rows):
                X_list.append(vec)
                y_list.append(0.0)
                groups.append(("b", cid))
            _record_score_rows(("b", cid), cid)
    return X_list, y_list, groups, score_rows

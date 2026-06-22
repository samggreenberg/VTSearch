"""Detector training helpers: validate, train, threshold, serialise.

Consolidates the repeated train → calibrate → safe-threshold → serialise
pipeline used by detector route handlers and test helpers.

This module also holds the vote-aware detector training entry points -
:func:`train_and_score` (online, called every time the user toggles a
vote) and :func:`train_detector_from_origins` (load-time, called when
re-deriving an MLP from a saved labelset). Both build on the generic
:mod:`vtscore.training` primitives but layer in the patch-region max-
pool and origin-based file resolution that are detector-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.utils.scores import sigmoid_to_finite_scores

if TYPE_CHECKING:
    import torch.nn as nn


def _active_score_embedder() -> str | None:
    """Resolve the active dataset's score embedder (patch-else-text).

    The detector MLP trains and scores against this embedder's vectors (the v3
    routing table; see :meth:`DatasetContext.routed_embedder`).  Returns
    ``None`` for a slot-less single-vector dataset, where the matrix layer
    falls back to each media's primary vector.  For a single-embedder dataset
    the resolved name equals the primary, which the matrix layer collapses to
    the cached primary path.
    """
    from vtscore.state.core import get_active_context  # noqa: PLC0415

    return get_active_context().routed_embedder("score")


def validate_good_bad_split(y_list: list[float]) -> tuple[int, int]:
    """Check that *y_list* contains at least one good and one bad label.

    Returns ``(num_good, num_bad)``.
    Raises ``ValueError`` when either count is zero.
    """
    num_good = sum(1 for y in y_list if y == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        raise ValueError("Need at least one good and one bad labeled example")
    return num_good, num_bad


def train_and_threshold(
    X_list: list,
    y_list: list[float],
    snap: dict | None = None,
) -> tuple[Any, float]:
    """Train an MLP and compute a calibrated threshold.

    This is the canonical training pipeline used by all detector routes:

    1. Cross-calibration threshold (respects ``calibrate_count`` /
       ``calibration_fraction`` settings).
    2. Full-data model training (respects ``inclusion`` setting).
    3. Optional safe-threshold blending when ``get_safe_thresholds()`` is
       enabled and *snap* is provided.

    ``inclusion`` is read from ``get_inclusion()``, which resolves to the
    *active detector context's* inclusion (seeded from the user's settings
    default the first time it's read for a detector). Both Train and Find
    therefore train at the same per-detector inclusion within a session.

    Args:
        X_list: Embedding vectors (list of numpy arrays).
        y_list: Binary labels (1.0 = good, 0.0 = bad).
        snap: Optional media snapshot for safe-threshold scoring.

    Returns:
        ``(model, threshold)``
    """
    import torch

    from vtscore.state import (
        get_calibrate_count,
        get_calibration_fraction,
        get_inclusion,
        get_safe_thresholds,
    )
    from vtscore.training import (
        calculate_cross_calibration_threshold,
        calculate_safe_threshold,
        train_model,
    )
    from vtscore.training.thresholds import NO_GOOD_THRESHOLD

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    safe = bool(get_safe_thresholds() and snap)
    # Below the safe-threshold ramp floor the cross-cal output is blended
    # away (pure GMM), so don't pay for the fold trainings.
    if safe and len(X_list) < 6:
        threshold = NO_GOOD_THRESHOLD
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
        )

    model = train_model(X, y, input_dim)

    if safe:
        from vtscore.embedding.matrix import get_embedding_matrix_for_snap

        # `safe` is only True when `snap` is truthy (see assignment above),
        # so the narrowing is real even though pyright can't track it.
        assert snap is not None
        _all_ids, all_embs = get_embedding_matrix_for_snap(snap, _active_score_embedder())
        X_all = torch.from_numpy(all_embs)
        with torch.no_grad():
            X_all = X_all.to(next(model.parameters()).device)
            all_scores = sigmoid_to_finite_scores(model(X_all))
        threshold = calculate_safe_threshold(threshold, all_scores, len(y_list))

    return model, threshold


def serialize_weights(model) -> dict[str, list]:
    """Convert a PyTorch model's state dict to JSON-serialisable nested lists."""
    return {key: value.cpu().tolist() for key, value in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Vote-aware detector training (online, called from sort/vote handlers)
# ---------------------------------------------------------------------------


def pool_box_from_media(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Return the region-pooled training vector for *media*, or ``None``.

    When *region_box* is set **and** *media* has a stored ``patch_grid``,
    pool the box on-the-fly via
    :func:`vtscore.media.patch_embed.box_to_vote_vector` and return it.
    Otherwise return ``None`` so the caller can fall back to the media's
    image-level ``embedding`` - the legacy training vector for image-level
    votes, single-vector embedders, and patch datasets that haven't been
    re-loaded under the v1 storage scheme.  Patch-embedder v2.

    Shared by the in-dataset vote path (:func:`_training_vec_for_vote`) and
    the cross-dataset labelset path
    (:func:`vtscore.detectors.labelset_training._resolve_uncached_embedding`).
    """
    if region_box is None:
        return None
    grid = media.get("patch_grid")
    if grid is None:
        return None
    from vtscore.media.patch_embed import box_to_vote_vector  # noqa: PLC0415

    return box_to_vote_vector(np.asarray(grid), region_box)


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


def _build_vote_xy(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    region_boxes: dict[int, tuple[float, float, float, float]],
) -> tuple[list[np.ndarray], list[float]]:
    """Build ``(X_list, y_list)`` from filtered votes.

    Good votes that designated a region are region-pooled via
    :func:`_training_vec_for_vote`; bad votes always use the image-level
    embedding.  Returns the raw lists - the caller
    (:func:`_train_and_score_xy`) enforces the ≥2-samples / ≥1-good /
    ≥1-bad guard.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    # Train against the dataset's score embedder so the MLP shares the space
    # _score_all_media scores against (the v3 routing table).
    embedder_name = _active_score_embedder()
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for cid in good_votes:
        if cid in clips_dict:
            X_list.append(_training_vec_for_vote(clips_dict[cid], region_boxes.get(cid), embedder_name))
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in clips_dict:
            X_list.append(media_embedding(clips_dict[cid], embedder_name))
            y_list.append(0.0)
    return X_list, y_list


def _score_all_media(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
) -> tuple[list[int], list[float], list[int]]:
    """Score every media in *clips_dict* with the trained MLP.

    Region-aware datasets (those whose media expose ``patch_regions``)
    are scored by flattening all (media, region) vectors into one tensor,
    running a single forward pass, then max-pooling per media - so the
    winning region's index can be surfaced for UI overlays.  Plain
    datasets fall back to the cached embedding matrix.

    Returns ``(all_ids, scores_per_media, best_region_index_per_media)``.
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.matrix import (  # noqa: PLC0415
        get_embedding_matrix_for_snap,
        get_region_matrix_for_snap,
    )

    has_regions = any(clips_dict[cid].get("patch_regions") for cid in clips_dict)
    if has_regions:
        # One row per (media, region) pair, built once and cached on the
        # dataset context (the region vectors never change between votes -
        # only the MLP weights do), so online retraining no longer rebuilds
        # a multi-hundred-thousand-row matrix on every vote.
        all_ids, X_np, media_index_per_row, region_index_per_row = get_region_matrix_for_snap(clips_dict)
    else:
        all_ids, X_np = get_embedding_matrix_for_snap(clips_dict, _active_score_embedder())
        n = len(all_ids)
        media_index_per_row = np.arange(n, dtype=np.int64)
        region_index_per_row = np.zeros(n, dtype=np.int64)

    if not all_ids:
        return [], [], []

    with torch.no_grad():
        X_all = torch.from_numpy(X_np).to(next(model.parameters()).device)
        # ``sigmoid_to_finite_scores`` replaces NaN/±Inf with the
        # ``NON_FINITE_SCORE_SENTINEL`` (-1.0) so a destabilised MLP cannot
        # leak non-finite floats into the JSON response. The downstream
        # segmented max-pool then incidentally drops sentinels in favour of
        # any real score (in ``[0, 1]``) for the same media.
        flat_scores = np.asarray(sigmoid_to_finite_scores(model(X_all)), dtype=np.float64)

    scores, best_region = _segmented_max_pool(flat_scores, media_index_per_row, region_index_per_row, len(all_ids))
    return all_ids, scores, best_region


def _segmented_max_pool(
    flat_scores: np.ndarray,
    media_index_per_row: np.ndarray,
    region_index_per_row: np.ndarray,
    n_media: int,
) -> tuple[list[float], list[int]]:
    """Max-pool per-row scores down to one score + winning region per media.

    *media_index_per_row* is non-decreasing and contiguous (every media owns
    a single run of rows), and every media has at least one row, so each
    media's rows form one ``reduceat`` segment.  Returns ``(scores,
    best_region)`` as plain Python lists, where ``best_region[m]`` is the
    region index of the *first* row achieving media ``m``'s max - matching
    the strict-``>`` "first wins" tie-break of the original scalar loop.

    Fully vectorised so the per-vote scoring tail holds the GIL for
    microseconds rather than iterating hundreds of thousands of rows in
    Python (which, in the background training thread, would stall the
    ``gthread`` pool serving the next vote).
    """
    # Start of each media's contiguous run of rows.
    seg_starts = np.searchsorted(media_index_per_row, np.arange(n_media))
    seg_max = np.maximum.reduceat(flat_scores, seg_starts)

    # First row per media that reaches its segment max (region 0 - the
    # CLS/full-image node - is always row 0 of a segment, so an all-sentinel
    # media resolves to region 0, exactly as the old -1.0-seeded loop did).
    is_max = flat_scores >= seg_max[media_index_per_row]
    cand_rows = np.flatnonzero(is_max)
    cand_media = media_index_per_row[cand_rows]
    first_cand = np.searchsorted(cand_media, np.arange(n_media))
    winning_rows = cand_rows[first_cand]
    best_region = region_index_per_row[winning_rows]

    return seg_max.tolist(), best_region.tolist()


def _format_results(
    all_ids: list[int],
    scores: list[float],
    best_region: list[int],
    clips_dict: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort by score (descending) and produce JSON-serialisable result dicts.

    Raw float scores are used for sorting so tiny differences still
    affect ordering; only the response ``score`` field is rounded.
    Region-aware media gain a ``best_region`` key holding the winning
    region's box.
    """
    paired = sorted(
        zip(all_ids, scores, best_region, strict=True),
        key=lambda t: t[1],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for cid, s, bri in paired:
        entry: dict[str, Any] = {"id": cid, "score": round(s, 4)}
        media = clips_dict[cid]
        regions = media.get("patch_regions")
        if regions and 0 <= bri < len(regions):
            entry["best_region"] = list(regions[bri].box)
        results.append(entry)
    return results


def score_media_with_model(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score every media in *clips_dict* with an already-trained *model*.

    Returns sorted (descending by score) result dicts of the same shape the
    vote-driven training path produces: ``{"id", "score"}`` plus a
    ``best_region`` box for patch-region-aware media (the argmax region that
    drove the media's score).  Use this from any route that scores with a
    pre-trained detector - e.g. the Find / detector-scoring path - so the
    best-match highlight is populated regardless of which entry point ran the
    scoring.  Plain single-vector datasets are scored via the cached embedding
    matrix and gain no ``best_region`` field, exactly as before.
    """
    all_ids, scores, best_region = _score_all_media(model, clips_dict)
    return _format_results(all_ids, scores, best_region, clips_dict)


def _train_and_score_xy(
    X_list: list[np.ndarray],
    y_list: list[float],
    clips_dict: dict[int, dict[str, Any]],
    *,
    inclusion_value: int,
    safe_thresholds: bool,
    calibrate_count: int,
    calibration_fraction: float,
    det_ctx: Any,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train an MLP on ``(X_list, y_list)`` and score every media in *clips_dict*.

    Shared core of :func:`train_and_score` (vote-driven) and
    :func:`vtscore.detectors.labelset_training.labelset_train_and_score`
    (labelset-driven): the two pipelines differ only in how they assemble
    ``(X_list, y_list)``, so the guard → threshold → train → score → format
    tail lives here once.

    ``hidden_dim`` is sized from the label count so the cross-calibration
    fold models share the final model's architecture, making fold thresholds
    directly comparable to final-model scores.  Returns ``([], 0.5, None)``
    when the labels don't satisfy ≥2 samples AND ≥1 good AND ≥1 bad.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import _auto_hidden_dim, train_model  # noqa: PLC0415
    from vtscore.training.thresholds import (  # noqa: PLC0415
        calculate_safe_threshold,
        cross_calibration_threshold_cached,
    )

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]
    hidden_dim = _auto_hidden_dim(len(X_list))

    # Skip k-fold calibration when the label count is below the
    # ``calculate_safe_threshold`` ramp floor: the calibration trainings
    # would be expensive (two 200-epoch fits) and the result is either
    # discarded (safe_thresholds=True blends with label_weight=0 → pure
    # GMM) or unreliable.  0.5 is a sensible neutral mid-point.
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

    # Training is image-level in v1 - the MLP only ever sees one vector
    # per labelled media, mirroring the "vote on whole images" rule.
    model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    all_ids, scores, best_region = _score_all_media(model, clips_dict)

    if safe_thresholds:
        threshold = calculate_safe_threshold(threshold, scores, len(X_list))

    results = _format_results(all_ids, scores, best_region, clips_dict)
    return results, threshold, model


def train_and_score(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    inclusion_value: int = 0,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None,
    det_ctx: Any = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train a small MLP on voted media embeddings and score every media.

    Uses k-fold calibration to determine an appropriate decision threshold,
    then trains a final model on all labelled data and scores every media in
    ``clips_dict``.

    Args:
        clips_dict: Mapping of media ID to media data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        good_votes: Dict whose keys are media IDs labelled as good (values are ``None``).
        bad_votes: Dict whose keys are media IDs labelled as bad (values are ``None``).
        inclusion_value: Integer in ``[-10, 10]`` passed to the training and
            threshold-finding functions to control the inclusion/exclusion bias.
        safe_thresholds: When ``True``, blend the cross-calibration threshold with
            a GMM-based threshold for robustness when few labels are available.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for calibration
            in each split (default 0.5).  For example, 0.2 means 80% Train /
            20% Calibrate.
        vote_region_boxes: Optional ``media_id -> (x0, y0, x1, y1)`` map from
            yes-votes that designated a region.  When set and the source
            media has a stored ``patch_grid``, the training vector for that
            vote is pooled on-the-fly via
            :func:`vtscore.media.patch_embed.box_to_vote_vector` instead
            of using ``media["embedding"]``.  Falls back to the full-image
            vector when the media lacks a patch grid (legacy datasets,
            single-vector embedders) or when the box is missing.  Patch-
            embedder v2.

    Returns:
        A tuple ``(results, threshold, model)`` where:

        - ``results`` is a list of ``{"id": int, "score": float}`` dicts, sorted
          by score in descending order (highest confidence first).
        - ``threshold`` is the decision boundary as a float (cross-calibrated,
          or blended with GMM when ``safe_thresholds`` is ``True``).
        - ``model`` is the trained ``nn.Sequential`` model (``None`` when
          training was not possible).
    """
    region_boxes = vote_region_boxes or {}
    X_list, y_list = _build_vote_xy(clips_dict, good_votes, bad_votes, region_boxes)
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

    # Stage-2 structural re-rank: a no-op for every non-structural dataset
    # (gated on media carrying ``local_features``), so existing datasets are
    # untouched.  For a structural (SIFT/VLAD) dataset it geometrically
    # verifies the VLAD shortlist against the RegionYes templates and re-ranks
    # by the match-statistic classifier (or the cold-start inlier gate).  See
    # docs/plans/structural-embedder.md.
    from vtscore.training.structural_similarity import maybe_structural_rerank  # noqa: PLC0415

    results, threshold = maybe_structural_rerank(
        results, threshold, clips_dict, good_votes, bad_votes, region_boxes, det_ctx
    )
    return results, threshold, model


# ---------------------------------------------------------------------------
# Origin-based helpers (for weight-free detector serialisation)
# ---------------------------------------------------------------------------


def collect_media_origins(
    media_ids: dict[int, None] | list[int],
    snap: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect origin info for a set of media IDs from a medias snapshot.

    Each returned dict contains ``origin``, ``origin_name``, ``filename``,
    and ``md5`` - enough to re-resolve the original file later.

    Args:
        media_ids: Media IDs (keys of a votes dict, or a plain list).
        snap: Snapshot of all loaded medias (from :func:`snapshot_medias`).

    Returns:
        A list of origin dicts, one per matched media.
    """
    origins: list[dict[str, Any]] = []
    for cid in media_ids:
        if cid not in snap:
            continue
        media = snap[cid]
        origins.append(
            {
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", ""),
                "filename": media.get("filename", ""),
                "md5": media.get("md5", ""),
            }
        )
    return origins


def train_detector_from_origins(
    good_origins: list[dict[str, Any]],
    bad_origins: list[dict[str, Any]],
    inclusion: int,
    media_type: str,
    embedder_name: str,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
) -> tuple[dict[str, list] | None, float]:
    """Resolve origin entries to files, embed them, and train a detector MLP.

    This is the load-time counterpart of file-based detector export: given
    the origin lists that were saved to disk, it re-derives the MLP weights
    by resolving the original media files, embedding them, and training.

    Args:
        good_origins: Origin dicts for media labelled Good.
        bad_origins: Origin dicts for media labelled Bad.
        inclusion: The inclusion value to use for training.
        media_type: Media type string (e.g. ``"audio"``, ``"image"``).
        embedder_name: Name of the embedder the detector was originally
            trained with. Passed through to :func:`embed_file` so every
            re-embedded media is encoded by the same model that produced
            the saved vectors - otherwise the MLP trains on a mix of
            embedder outputs and learns garbage. Pass ``""`` only when you
            genuinely want the media type's default embedder (e.g. a
            brand-new detector with no recorded embedder yet).
        calibrate_count: Number of k-fold calibration splits.
        calibration_fraction: Fraction reserved for calibration.

    Returns:
        A ``(weights, threshold)`` tuple.  ``weights`` is ``None`` if
        resolution/embedding failed for too many entries (need at least
        one good and one bad).
    """
    import torch  # noqa: PLC0415

    from vtscore.detectors.resolver import embed_file, resolve_file_context
    from vtscore.training.mlp import train_model
    from vtscore.training.thresholds import calculate_cross_calibration_threshold

    X_list: list = []
    y_list: list[float] = []

    for entry in good_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type, embedder_name)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0)

    for entry in bad_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type, embedder_name)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(0.0)

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return None, 0.5

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    if len(X_list) < 6:
        threshold = 0.5
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
        )
    model = train_model(X, y, input_dim)

    state_dict = model.state_dict()
    weights = {k: v.cpu().tolist() for k, v in state_dict.items()}
    return weights, threshold

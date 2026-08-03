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

from vtscore.utils.scores import sigmoid_to_finite_array

if TYPE_CHECKING:
    import torch.nn as nn


def _score_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Resolve the score embedder (structural ▸ patch ▸ text) for the medias in *snap*.

    The detector MLP trains and scores against this embedder's vectors (the v3
    routing table).  Derived from the embedder names *those* medias carry
    rather than the active context, since :func:`_score_all_media` /
    :func:`_build_vote_xy` may be handed an arbitrary snapshot (a test fixture,
    a cross-dataset dict).  Returns ``None`` for a slot-less single-vector
    dataset (e.g. ``dinov2_single``) or medias whose embedder is unregistered,
    so the matrix layer falls back to each media's primary vector.  For a
    single-embedder dataset the resolved name equals the primary, which the
    matrix layer collapses to the cached primary path.
    """
    if not snap:
        return None
    from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

    first = next(iter(snap.values()))
    text, patch, structural = derive_binding_from_names(media_embedder_names(first))
    return structural or patch or text


def detector_score_embedder(det_ctx: Any, snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Embedder a *detector* trains and scores in.

    The concrete embedder of the detector's locked type
    (``det_ctx.embedder_type``) that the snap supplies wins; otherwise the
    dataset score precedence - the legacy-migration default and the
    cross-dataset portability fallback (a detector pointed at a dataset that
    lacks its type re-embeds against that dataset's space).  This is exactly
    :func:`keying_embedder_for_snap`.  Returns a concrete name (collapsed to the
    cached primary path by the matrix layer when it equals the dataset's
    primary), or ``None`` when there is nothing to resolve (empty snap).  See
    ``docs/plans/patch-embedder.md`` → "Per-detector embedder type".
    """
    from vtscore.embedding.binding import keying_embedder_for_snap  # noqa: PLC0415

    return keying_embedder_for_snap(det_ctx, snap) or None


def _patch_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Resolve the patch-slot embedder name for *snap*'s medias, or ``None``."""
    if not snap:
        return None
    from vtscore.embedding.binding import derive_binding_from_names  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedder_names  # noqa: PLC0415

    first = next(iter(snap.values()), {})
    _text, patch, _structural = derive_binding_from_names(media_embedder_names(first))
    return patch


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


def _flood_context(
    X_list: list,
    y_list: list[float],
    groups: list | None,
) -> tuple[int, list | None, Any]:
    """Resolve the bag-aware training context for a possibly-flooded label set.

    Returns ``(n_votes, cal_groups, sample_weights)``:

    * ``n_votes`` - the number of distinct bags (votes/images); the unit the
      hidden-layer width and the safe-threshold ramp should size on, so region
      flooding (many rows per Bad vote) doesn't inflate either.
    * ``cal_groups`` - *groups* when flooding actually occurred (a bag holds
      more than one row), else ``None`` so the calibrator takes its historical
      row-wise path unchanged.
    * ``sample_weights`` - per-bag loss weights when flooding occurred, else
      ``None`` so :func:`~vtscore.training.mlp.train_model` computes its default
      inverse-frequency weights.

    Shared by :func:`train_and_threshold` and :func:`_train_and_score_xy` so the
    vote, labelset, and Find paths flood identically.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.thresholds import _per_bag_fit_weights  # noqa: PLC0415

    n_votes = len(set(groups)) if groups is not None else len(X_list)
    flooded = groups is not None and len(X_list) != n_votes
    cal_groups = groups if flooded else None
    sample_weights = None
    if flooded and groups is not None:
        sample_weights = torch.tensor(
            _per_bag_fit_weights(np.asarray(y_list, dtype=np.float32), groups), dtype=torch.float32
        )
    return n_votes, cal_groups, sample_weights


def _calibration_score_rows(
    groups: list | None,
    cal_groups: list | None,
    score_rows: dict | None,
) -> dict | None:
    """Per-bag **inference** row stacks for the calibrator, or ``None``.

    Calibration collapses each held-out bag with ``max``.  Left to the training
    rows that is an unfair comparison on a patch dataset: a Good vote holds one
    row while a Bad vote holds its ~13 flooded leaves, and ``max`` over 13 draws
    beats ``max`` over 1 with no signal at all - so the cut lands high and the
    detector over-rejects true matches.  Handing the calibrator each voted
    image's *scoring* stack (all ~24 region nodes, exactly what
    :func:`_score_all_media` max-pools) puts both classes in the geometry
    inference actually uses.

    Coverage is **all-or-nothing**: unless every bag has a stack this returns
    ``None`` and the calibrator keeps pooling over the training rows.  Partial
    coverage is worse than none - a Good bag left on its 1 training row while
    the Bad bags widen from 13 rows to 24 would deepen the very bias this
    corrects - so an unresolvable media declines the correction rather than
    skewing it.

    Also returns ``None`` when *cal_groups* is ``None``: nothing flooded, so
    every bag is already a single row and the row-wise path runs unchanged.
    """
    if cal_groups is None or groups is None or not score_rows:
        return None
    bag_ids = set(groups)
    if not bag_ids <= set(score_rows):
        return None
    return {g: np.asarray(score_rows[g], dtype=np.float32) for g in bag_ids}


def train_and_threshold(
    X_list: list,
    y_list: list[float],
    snap: dict | None = None,
    embedder_name: str | None = None,
    det_ctx: Any = None,
    groups: list | None = None,
    score_rows: dict | None = None,
) -> tuple[Any, float]:
    """Train the detector head and compute a calibrated threshold.

    This is the canonical training pipeline used by all detector routes:

    1. Cross-calibration threshold (respects ``calibrate_count`` /
       ``calibration_fraction`` settings).
    2. Full-data model training (respects ``inclusion`` setting).
    3. Optional safe-threshold blending when ``get_safe_thresholds()`` is
       enabled and *snap* is provided.  The GMM half of that blend is fitted
       on the per-media scores :func:`_score_all_media` produces - region
       max-pooled on a patch dataset - so it sees the distribution the
       threshold will actually cut.

    ``inclusion`` is read from ``get_inclusion()``, which resolves to the
    *active detector context's* inclusion (seeded from the user's settings
    default the first time it's read for a detector). Both Train and Find
    therefore train at the same per-detector inclusion within a session.

    Args:
        X_list: Embedding vectors (list of numpy arrays).
        y_list: Binary labels (1.0 = good, 0.0 = bad).
        snap: Optional media snapshot for safe-threshold scoring.
        embedder_name: The detector's primary embedder, used so the
            safe-threshold scoring pass reads vectors from the same space the
            ``X_list`` were built in.  ``None`` falls back to the dataset score
            precedence for *snap* (the pre-per-detector behaviour).
        det_ctx: When provided, the inclusion-independent K fold orderings are
            cached on ``det_ctx.calibration_cache`` (and the fold models are
            sized to match the final model).  This is what lets a later
            Inclusion slide re-derive the threshold over the cached orderings
            instead of being a no-op — see
            :func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`
            and docs/plans/find-verification-workflow.md.  ``None`` keeps the
            legacy (uncached) behaviour for callers that don't own a context.
        groups: Per-row bag ids (one voted image per bag); see
            :func:`_flood_context`.
        score_rows: Per-bag inference row stacks; see
            :func:`_calibration_score_rows`.  Only consulted when *groups*
            reveals flooding.

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
    from vtscore.training.mlp import LINEAR_HEAD
    from vtscore.training.thresholds import NO_GOOD_THRESHOLD, cross_calibration_threshold_cached

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Bag-aware setup (region flooding): size on votes not rows, split/weight
    # per bag.  On a legacy label set every bag is one row, so this collapses
    # to the historical behaviour.
    n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)
    cal_score_rows = _calibration_score_rows(groups, cal_groups, score_rows)

    # The production detector head is linear (logistic regression): stable under
    # sparse positives where a small MLP's per-retrain score wobble drove the
    # threshold-stability #2790 spikes.  The same head sizes both the final model
    # and the cached fold sub-models below, so the cached re-thresholding stays
    # consistent with this run's threshold.
    hidden_dim = LINEAR_HEAD

    safe = bool(get_safe_thresholds() and snap)
    # Below the safe-threshold ramp floor the cross-cal output is blended
    # away (pure GMM), so don't pay for the fold trainings.  Safe-thresholds
    # OFF falls through to real cross-calibration at every label count.
    if safe and n_votes < 6:
        threshold = NO_GOOD_THRESHOLD
        # This branch never recomputes the fold orderings, so drop any stale
        # cache from a previous ≥6-label training - otherwise a later
        # inclusion slide (`recompute_detector_thresholds_for_inclusion`)
        # re-thresholds against orderings for the old label set/model.
        # Mirrors the same guard in :func:`_train_and_score_xy`.
        if det_ctx is not None:
            det_ctx.calibration_cache = None
    elif det_ctx is not None:
        # Cache the K fold orderings on the context so an Inclusion slide can
        # re-derive the cutoff without a no-op (the find-label / detector-load
        # paths land here; without the cache the slide can't move the line).
        threshold = cross_calibration_threshold_cached(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
            hidden_dim=hidden_dim,
            det_ctx=det_ctx,
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )

    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    if safe:
        # `safe` is only True when `snap` is truthy (see assignment above),
        # so the narrowing is real even though pyright can't track it.
        assert snap is not None
        # Fit the GMM on the *inference* score distribution.  `_score_all_media`
        # is the same call scoring makes (`score_media_with_model`), so on a
        # patch dataset the GMM sees the region max-pooled per-media scores the
        # threshold will actually cut.  Scoring the image-level embedding matrix
        # instead fitted the GMM on a systematically lower distribution (the
        # region max is ≥ the single image-level row), biasing the midpoint cut
        # low → over-inclusion on region-voting detectors.  Plain single-vector
        # datasets take `_score_all_media`'s embedding-matrix fallback, so their
        # behaviour is unchanged.  *embedder_name* is forwarded as-is (not
        # pre-resolved) so the region-vs-plain gating matches inference exactly.
        # Mirrors `_train_and_score_xy` and
        # `eval.voting_iterations._blend_safe_threshold`.
        _all_ids, all_scores, _best_region = _score_all_media(model, snap, embedder_name)
        # The GMM blend is applied only on a *fresh* retrain: the fold-ordering
        # cache above stores the raw cross-cal orderings, so a later inclusion
        # slide re-derives the unblended cutoff (intentional - see
        # recompute_detector_thresholds_for_inclusion).
        threshold = calculate_safe_threshold(threshold, all_scores, n_votes)

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
    """Return the region training vector for *media*, or ``None``.

    When *region_box* is set, snap it to the media's nearest ``patch_regions``
    node via :func:`vtscore.media.patch_embed.snap_box_to_region` and return
    that node's vector - i.e. train on the exact sub-image suggestion the MLP
    max-pools over at inference, so the Good vote is a fair representative of
    what the detector will actually score.  Falls back to a uniform pool of
    the raw ``patch_grid`` (:func:`box_to_vote_vector`) only when the media
    carries a grid but no region tree, and to ``None`` (the caller then uses
    the image-level ``embedding``) for legacy single-vector embedders and
    patch datasets that predate region storage.  Patch-embedder v2.

    Shared by the in-dataset vote path (:func:`_training_vec_for_vote`) and
    the cross-dataset labelset path
    (:func:`vtscore.detectors.labelset_training._resolve_uncached_embedding`).
    """
    if region_box is None:
        return None

    regions = media.get("patch_regions")
    if regions:
        from vtscore.media.patch_embed import snap_box_to_region  # noqa: PLC0415

        snapped = snap_box_to_region(regions, region_box)
        if snapped is not None:
            return snapped

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


def bad_negative_vecs(
    media: dict[str, Any],
    embedder_name: str | None = None,
) -> list[np.ndarray]:
    """Negative training vectors contributed by one Bad vote on *media*.

    On **patch** media (carrying a ``patch_regions`` tree) a Bad vote floods
    every region node with no children - the CLS full-image node plus the HAC
    leaves, the disjoint set that tiles the image - as negatives.  This is the
    multiple-instance-learning treatment of a rejected image: since inference
    scores an image by its **best** region (max-pool), a Bad vote asserts that
    *no* region of it should score high, so we train every leaf down.

    Internal HAC nodes (``children`` set) are **not** flooded, even though
    :func:`_score_all_media` max-pools them.  This is a deliberate, measured
    exception, not a redundancy claim: an internal node is *not* dominated by
    its leaves.  :func:`~vtscore.media.patch_embed.build_hac_tree` sets
    ``merged_vec = _l2_normalize(sum_a + sum_b)``, so the merged vector is the
    convex-hull point of its descendants **renormalised back to the unit
    sphere** - a gain of ``1 / ||mean(descendant leaf vecs)||``, ~1.53x on real
    trees.  Under the linear production head that scales the logit by the same
    factor, so an internal node can and does out-score every one of its own
    leaves (~4.7% of node/direction pairs).

    Flooding them anyway was A/B'd over 24 synthetic patch detectors
    (``scripts/probe_hac_internal_flood.py``) and it **hurts**: paired AP
    -0.058 ± 0.036 (95% CI, excludes zero; the leaves-only arm wins 19 of 24
    seeds), while FPR (+0.037 ± 0.045) and FNR (-0.010 ± 0.049) both straddle
    zero.  It does buy back some of the gap - the share of negatives whose
    winning row is an internal node falls 4.6% → 2.7% - but suppressing a
    rejected image's renormalised mean directions also suppresses the geometry
    a *positive* image's concept-blob internal lives in, and that costs more
    than it buys.  So the flood stays leaves-only, and the gap stays pinned by
    ``test_max_hac_floods_every_scored_row_except_hac_internals`` and
    ``tests_lib/detectors/test_hac_internal_flood_gap.py`` rather than papered
    over.  See #2731.

    Non-patch media contribute a single image-level vector, exactly as before,
    so every legacy single-vector dataset is byte-for-byte unchanged.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    regions = media.get("patch_regions")
    if regions:
        leaves = [np.asarray(r.vec, dtype=np.float32) for r in regions if r.children is None]
        if leaves:
            return leaves
    return [media_embedding(media, embedder_name)]


def inference_score_rows(
    media: dict[str, Any],
    embedder_name: str | None = None,
) -> np.ndarray | None:
    """The row stack *media* is max-pooled over at inference, or ``None``.

    Mirrors :func:`vtscore.embedding.matrix._build_region_arrays`, the matrix
    :func:`_score_all_media` scores: a patch media contributes **every**
    ``patch_regions`` node - CLS, HAC internals, and HAC leaves alike, ~24 rows
    - and a region-less media contributes its single image-level vector.

    Used to calibrate in inference geometry: a voted image's bag must collapse
    over the same rows the scorer will pool, not over the (Good: 1, Bad: ~13)
    rows the fold model happened to train on.  See
    :func:`_calibration_score_rows`.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    regions = media.get("patch_regions")
    if regions:
        return np.stack([np.asarray(r.vec, dtype=np.float32) for r in regions])
    emb = media_embedding(media, embedder_name)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32).reshape(1, -1)


def _build_vote_xy(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    region_boxes: dict[int, tuple[float, float, float, float]],
    embedder_name: str | None = None,
) -> tuple[list[np.ndarray], list[float], list, dict]:
    """Build ``(X_list, y_list, groups, score_rows)`` from filtered votes.

    Good votes that designated a region are region-pooled via
    :func:`_training_vec_for_vote` (one row each).  Bad votes are expanded by
    :func:`bad_negative_vecs`: one row per image-level vector on a legacy
    dataset, or one row per region leaf on a patch dataset (region flooding).

    ``groups`` carries one bag id per row - ``("g", cid)`` for a Good vote,
    ``("b", cid)`` shared across all of a Bad vote's flooded leaf rows - so the
    downstream trainer/calibrator can balance and split by **image**, not by
    row.  On a legacy dataset every bag holds exactly one row, so ``groups`` is
    1:1 with the rows and the whole path collapses to the pre-flood behaviour.
    The caller (:func:`_train_and_score_xy`) enforces the ≥2-samples /
    ≥1-good / ≥1-bad guard.

    ``score_rows`` maps each bag id to the row stack that voted image is
    *scored* over at inference (:func:`inference_score_rows`) - the whole
    ~24-node region tree on a patch media - so calibration can collapse a Good
    bag and a Bad bag the same way :func:`_score_all_media` collapses any
    image.  Without it a Good bag is a max over its 1 training row against a
    Bad bag's max over ~13, and the calibrated cut lands high.

    *embedder_name* is the detector's primary embedder; when ``None`` the
    dataset score precedence for *clips_dict* is used (the pre-per-detector
    behaviour).  Either way the MLP trains in the same space
    :func:`_score_all_media` scores against.
    """
    if embedder_name is None:
        embedder_name = _score_embedder_for_snap(clips_dict)
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    groups: list = []
    score_rows: dict = {}

    def _record_score_rows(group: tuple, cid: int) -> None:
        rows = inference_score_rows(clips_dict[cid], embedder_name)
        if rows is not None:
            score_rows[group] = rows

    for cid in good_votes:
        if cid in clips_dict:
            X_list.append(_training_vec_for_vote(clips_dict[cid], region_boxes.get(cid), embedder_name))
            y_list.append(1.0)
            groups.append(("g", cid))
            _record_score_rows(("g", cid), cid)
    for cid in bad_votes:
        if cid in clips_dict:
            for vec in bad_negative_vecs(clips_dict[cid], embedder_name):
                X_list.append(vec)
                y_list.append(0.0)
                groups.append(("b", cid))
            _record_score_rows(("b", cid), cid)
    return X_list, y_list, groups, score_rows


def _score_all_media(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
) -> tuple[list[int], list[float], list[int]]:
    """Score every media in *clips_dict* with the trained detector head.

    Region-aware datasets (those whose media expose ``patch_regions``)
    are scored by flattening all (media, region) vectors into one tensor,
    running a single forward pass, then max-pooling per media - so the
    winning region's index can be surfaced for UI overlays.  Plain
    datasets fall back to the cached embedding matrix.

    *embedder_name* is the detector's primary embedder (the space the MLP was
    trained in).  When it is given, region max-pooling is used **only** if that
    primary is the dataset's patch-slot embedder - a detector scoring in the
    text or structural space of a multi-embedder dataset must score against
    that space's full-image vectors, not the patch tree.  When ``None`` (the
    pre-per-detector behaviour) any media carrying ``patch_regions`` takes the
    region path, matching the dataset-level score precedence.

    Returns ``(all_ids, scores_per_media, best_region_index_per_media)``.
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.matrix import (  # noqa: PLC0415
        get_embedding_matrix_for_snap,
        get_region_matrix_for_snap,
        segmented_max_pool,
    )

    resolved = embedder_name if embedder_name is not None else _score_embedder_for_snap(clips_dict)
    has_regions = any(clips_dict[cid].get("patch_regions") for cid in clips_dict)
    if has_regions and embedder_name is not None:
        # Explicit per-detector primary: region-pool only when scoring in the
        # patch space (the patch tree lives in the patch embedder's vectors).
        patch = _patch_embedder_for_snap(clips_dict)
        has_regions = patch is not None and resolved == patch
    if has_regions:
        # One row per (media, region) pair, built once and cached on the
        # dataset context (the region vectors never change between votes -
        # only the MLP weights do), so online retraining no longer rebuilds
        # a multi-hundred-thousand-row matrix on every vote.
        all_ids, X_np, media_index_per_row, region_index_per_row = get_region_matrix_for_snap(clips_dict)
    else:
        all_ids, X_np = get_embedding_matrix_for_snap(clips_dict, resolved)
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
        flat_scores = sigmoid_to_finite_array(model(X_all)).astype(np.float64, copy=False)

    scores, best_region = segmented_max_pool(flat_scores, media_index_per_row, region_index_per_row, len(all_ids))
    return all_ids, scores, best_region


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
    embedder_name: str | None = None,
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

    *embedder_name* is the detector's primary embedder, so the scoring space
    matches the one the *model* was trained in (the per-detector primary).
    """
    all_ids, scores, best_region = _score_all_media(model, clips_dict, embedder_name)
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
    groups: list | None = None,
    score_rows: dict | None = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train the detector head on ``(X_list, y_list)`` and score every media in *clips_dict*.

    Shared core of :func:`train_and_score` (vote-driven) and
    :func:`vtscore.detectors.labelset_training.labelset_train_and_score`
    (labelset-driven): the two pipelines differ only in how they assemble
    ``(X_list, y_list)``, so the guard → threshold → train → score → format
    tail lives here once.

    The head is the linear (logistic) one at every label count, so region
    flooding can't inflate its capacity; the safe-threshold label count is
    still sized from the **vote** count (distinct *groups*) rather than the row
    count, so flooding - which turns one Bad vote into many leaf rows - doesn't
    shift the small-count threshold ramp.  When *groups*
    reveals at least one multi-row bag (flooding actually happened), the
    calibration split, fold fits, and final fit all run **bag-aware**
    (grouped fold split, per-bag loss weights), and each calibration bag
    collapses over the *scoring* rows *score_rows* supplies rather than the
    rows it trained on (see :func:`_calibration_score_rows`); otherwise every
    row is its own bag and the path is byte-for-byte the pre-flood behaviour.
    Returns ``([], 0.5, None)`` when the labels don't satisfy ≥2 samples AND
    ≥1 good AND ≥1 bad.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import LINEAR_HEAD, train_model  # noqa: PLC0415
    from vtscore.training.thresholds import (  # noqa: PLC0415
        calculate_safe_threshold,
        cross_calibration_threshold_cached,
    )

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    # The detector's primary embedder (the explicit space it scores in), or the
    # dataset score precedence when the detector has no primary yet.  Scoring
    # reads vectors from this same space the X_list were assembled in.
    score_emb = detector_score_embedder(det_ctx, clips_dict)

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Bag-aware setup (region flooding): size on votes not rows, split/weight
    # per bag when a Bad vote flooded its leaf set; a no-op on legacy datasets.
    n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)
    cal_score_rows = _calibration_score_rows(groups, cal_groups, score_rows)
    # Linear (logistic) production head - see train_and_threshold for why.
    hidden_dim = LINEAR_HEAD

    # Skip k-fold calibration *only* when safe-thresholds is on and the label
    # count is below the ``calculate_safe_threshold`` ramp floor: there the
    # calibrated value is entirely discarded by the pure-GMM blend
    # (label_weight=0), so the two 200-epoch fold fits would be pure waste.
    # With safe-thresholds OFF the cross-cal threshold is what the detector
    # actually uses, so it is computed for every label count - this is what
    # keeps the vote/labelset path agreeing with the Find path
    # (:func:`train_and_threshold`), which has always cross-calibrated below 6
    # labels when safe-thresholds is off.
    if safe_thresholds and n_votes < 6:
        threshold = 0.5
        # Drop any fold-ordering cache from a previous ≥6-label training:
        # this branch neither reads nor rewrites it, and a later inclusion
        # slide (`recompute_detector_thresholds_for_inclusion`) trusts any
        # non-None cache - re-thresholding against orderings computed for
        # the *old* label set/model.
        if det_ctx is not None:
            det_ctx.calibration_cache = None
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
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )

    # A Good vote trains on one region (the snapped box); a Bad vote trains on
    # its whole leaf set (region flooding), per-bag weighted so a rejected
    # image counts once.  On a legacy dataset there are no per-bag weights, so
    # the call stays identical to the historical one-vector-per-media fit.
    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    all_ids, scores, best_region = _score_all_media(model, clips_dict, score_emb)

    if safe_thresholds:
        # Blend applies on this fresh retrain only; the cached fold orderings
        # are raw cross-cal, so an inclusion slide re-derives the unblended
        # cutoff (see recompute_detector_thresholds_for_inclusion).  The label
        # count is votes, not flooded rows, so the small-count ramp is unmoved.
        threshold = calculate_safe_threshold(threshold, scores, n_votes)

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
    """Train the detector head on voted media embeddings and score every media.

    Uses k-fold calibration to determine an appropriate decision threshold,
    then trains a final model on all labelled data and scores every media in
    ``clips_dict``.

    Args:
        clips_dict: Mapping of media ID to media data dict. Each value must carry
            a resolvable embedding vector in its per-embedder ``"embeddings"``
            dict store (read via ``media_embedding``).
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
            media carries a ``patch_regions`` tree, the box is snapped to its
            nearest region node (:func:`vtscore.media.patch_embed.snap_box_to_region`)
            and that node's vector trains the vote, instead of
            ``media["embeddings"]``.  Falls back to a uniform patch-grid pool,
            then to the full-image vector, when the media lacks a region tree /
            patch grid (legacy datasets, single-vector embedders) or the box is
            missing.  Patch-embedder v2.

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
    X_list, y_list, groups, score_rows = _build_vote_xy(
        clips_dict, good_votes, bad_votes, region_boxes, detector_score_embedder(det_ctx, clips_dict)
    )
    results, threshold, model = _train_and_score_xy(
        X_list,
        y_list,
        clips_dict,
        inclusion_value=inclusion_value,
        safe_thresholds=safe_thresholds,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        det_ctx=det_ctx,
        groups=groups,
        score_rows=score_rows,
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
    """Resolve origin entries to files, embed them, and train a detector head.

    This is the load-time counterpart of file-based detector export: given
    the origin lists that were saved to disk, it re-derives the head weights
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
    from vtscore.training.mlp import LINEAR_HEAD, train_model
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

    # Real cross-calibration at every label count (the load-time counterpart
    # has no safe-threshold blend to discard the result), matching the
    # vote/labelset and Find paths with safe-thresholds off.  The trainer
    # degrades gracefully below 4 labels / <2-per-class via its own 0.5
    # fallback, so no separate small-label short-circuit is needed here.
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=LINEAR_HEAD,
    )
    model = train_model(X, y, input_dim, hidden_dim=LINEAR_HEAD)

    state_dict = model.state_dict()
    weights = {k: v.cpu().tolist() for k, v in state_dict.items()}
    return weights, threshold

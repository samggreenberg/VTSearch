"""Detector training helpers: validate, train, threshold, serialise.

Consolidates the repeated train → calibrate → fuse → serialise pipeline used
by detector route handlers and test helpers.

This module also holds the vote-aware detector training entry points -
:func:`train_and_score` (online, called every time the user toggles a
vote) and :func:`train_detector_from_origins` (the **library-tier**
re-derivation of a detector head from a saved labelset, for embedders driving
``vtscore`` directly; the app does not call it - see that function's
"Who calls this" note). Both build on the generic
:mod:`vtscore.training` primitives but layer in the patch-region max-
pool and origin-based file resolution that are detector-specific.

The pipeline's stages live in four sibling modules, each reading only the
ones above it:

* :mod:`~vtscore.detectors.training_snap` - which embedding space a
  snapshot's medias train and score in, and the patch gate both sides share.
* :mod:`~vtscore.detectors.training_scoring` - the ``(media, score row)``
  matrix a snapshot is scored over, and scoring it with a head.
* :mod:`~vtscore.detectors.training_votes` - the training rows each vote
  contributes, assembled from a vote set.
* :mod:`~vtscore.detectors.training_calibration` - the bag-aware fold
  context, the Train/Calibrate split, and the line (``_fused_threshold``).

Their public names are re-exported here, so ``vtscore.detectors.training.X``
resolves as it did when this was one module.

**Patch targets** follow the caller: a stub has to be installed where the
name is looked up.  :func:`train_and_threshold` and :func:`_train_and_score_xy`
resolve the stages they call in this module, so stubbing ``_fused_threshold``
or ``scoring_rows_for_snap`` for them means patching this module.  A stage
calling another stage resolves its own module global: stubbing
``_patch_space_for`` for :func:`resolve_calibration_fraction` means patching
:mod:`~vtscore.detectors.training_calibration`, where its other callers read
it too, and stubbing ``scoring_rows_for_snap`` for ``_score_all_media`` means
patching :mod:`~vtscore.detectors.training_scoring`.  A rebind here is
silently ignored by both.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.detectors.training_calibration import (
    _blend_schedule_for_snap,
    _calibration_score_rows,
    _flood_context,
    _fused_threshold,
    calibration_rows_for,
    resolve_calibration_fraction,
    vote_calibrating_groups,
)
from vtscore.detectors.training_scoring import (
    ScoringRows,
    _format_results,
    score_media_with_model,
    score_rows_with_model,
    scoring_progress,
    scoring_rows_for_snap,
)
from vtscore.detectors.training_snap import detector_score_embedder, validate_good_bad_split
from vtscore.detectors.training_votes import (
    _build_vote_xy,
    bad_negative_vecs,
    inference_score_rows,
    pool_box_from_media,
)
from vtscore.embedding.precomputed import stack_vectors

if TYPE_CHECKING:
    import torch.nn as nn

__all__ = [
    "ScoringRows",
    "bad_negative_vecs",
    "calibration_rows_for",
    "collect_media_origins",
    "detector_score_embedder",
    "inference_score_rows",
    "pool_box_from_media",
    "resolve_calibration_fraction",
    "score_media_with_model",
    "score_rows_with_model",
    "scoring_progress",
    "scoring_rows_for_snap",
    "serialize_weights",
    "train_and_score",
    "train_and_threshold",
    "train_detector_from_origins",
    "validate_good_bad_split",
    "vote_calibrating_groups",
]


def train_and_threshold(
    X_list: list,
    y_list: list[float],
    snap: dict | None = None,
    embedder_name: str | None = None,
    det_ctx: Any = None,
    groups: list | None = None,
    score_rows: dict | None = None,
    voted_ids: "set[int] | None" = None,
    haystack: dict | None = None,
    haystack_rows: "ScoringRows | None" = None,
    calibrating_groups: "set | None" = None,
    labels: "Mapping[int, bool] | None" = None,
) -> tuple[Any, float]:
    """Train the detector head and compute a calibrated threshold.

    This is the canonical training pipeline used by all detector routes:

    1. K-fold calibration (respects ``calibrate_count`` /
       ``calibration_fraction`` settings), giving both the cross-calibration
       cut and the fold models.
    2. Full-data model training.  The operating point never reaches the model;
       only the cut in step 3 reads it.
    3. The line whenever *snap* (or *haystack*) gives a haystack to score:
       the labels' line at the balance, with the fold-anchored cut and the
       schedule blend as fallbacks - see :func:`_fused_threshold`.  It is
       fitted on the per-media scores :func:`_score_all_media` produces -
       region max-pooled on a patch dataset - so it sees the distribution the
       threshold will actually cut.  Without a haystack there is nothing to fit
       on and the cross-calibration cut ships alone.

    The balance is *det_ctx*'s own (:func:`vtscore.state.detector_beta`:
    the balance the detector keeps, seeded on first read; #4665), or with no
    *det_ctx* (or one that is not a
    :class:`~vtscore.state.core.DetectorContext`) the active detector
    context's (``line_knobs()``).  A live
    detector's context is the active one, so both read the same value; a
    throwaway context that trains a cold detector (AutoFind, the CLI, Find
    over a detector nobody loaded) is seeded from that detector's JSON and
    cuts at its balance, not at whichever detector happens to be active.
    Both Train and Find therefore cut at the same per-detector balance.  The
    line is the labels' line at that balance (#4413, #4452; see
    :func:`_fused_threshold`).

    Args:
        X_list: Embedding vectors (list of numpy arrays).
        y_list: Binary labels (1.0 = good, 0.0 = bad).
        snap: Optional media snapshot - the haystack the population
            estimator is fitted on, unless *haystack* overrides it.  Without
            either there is nothing to fit on and the threshold is the plain
            cross-calibration cut.
        embedder_name: The detector's primary embedder, used so the
            haystack scoring pass reads vectors from the same space the
            ``X_list`` were built in.  ``None`` falls back to the dataset score
            precedence for *snap* (the pre-per-detector behaviour).
        det_ctx: When provided, the operating-point-independent K fold
            orderings are cached on ``det_ctx.calibration_cache`` (and the fold
            models are sized to match the final model).  This is what lets a
            later re-cut derive the threshold over the cached orderings instead
            of being a no-op — see
            :func:`vtscore.state.core.recut_detector_threshold`.
            ``None`` keeps the
            legacy (uncached) behaviour for callers that don't own a context.
        groups: Per-row bag ids (one voted image per bag); see
            :func:`_flood_context`.
        score_rows: Per-bag inference row stacks; see
            :func:`_calibration_score_rows`.  Only consulted when *groups*
            reveals flooding.
        haystack: The population the cut is *realized* on, when that is not
            *snap*.  The two are the same set for every caller that scores the
            snapshot it calibrated against - the GUI, which scores the loaded
            dataset it was handed - and differ for the CLI, whose scoring pass
            converts, re-clips and re-embeds the loaded medias into the
            detector's own granularity before scoring them
            (:func:`~vtscore.detectors.converter_routing.route_and_embed`).
            The fold-anchored estimator carries its per-fold cuts to the final
            model *in quantile space* and realizes them against this
            distribution, so handing it the loaded medias while inference reads
            the routed ones lands the quantile on the wrong ruler: a cut fitted
            on whole images and applied to the max over their clips sits far
            lower in the scored population than the algorithm intended (issue
            #3647).  *snap* keeps its other job either way - it is the snapshot
            a caller's labels are resolved against, which the routed items,
            with their throwaway ids and recomputed hashes, cannot serve as.
            The blend schedule follows *haystack* too, since it is chosen off
            the scoring geometry.
        voted_ids: Media ids in the haystack whose labels the training set
            carries.
            Excluded from every haystack the fold-anchored estimator fits on -
            their scores under the models trained on them are optimistically
            shifted (issue #3308; see :func:`_fused_threshold`).  ``None`` (the
            callers with no way to name their labels' media) keeps the full
            haystack.
        haystack_rows: the haystack's already-built
            :func:`scoring_rows_for_snap` matrix, when the caller has one in
            hand.  Purely an optimisation - it must be the rows this call would
            have built itself, i.e. ``scoring_rows_for_snap(haystack or snap,
            embedder_name)``.  A caller that scores the same snapshot for its
            own purposes (cross-dataset Find, which scores every media it just
            loaded) passes its copy so the corpus is stacked once for the whole
            operation rather than once here and once there; on a patch dataset
            that stack is ~197 float16 rows per media, so the saving is memory
            as much as time.  ``None`` builds it here.
        calibrating_groups: Deprecated and ignored (#4362).  It named the
            votes the #4220 precision estimate could use as evidence, and a
            retrain no longer builds that estimate.  Leave it unset; anything
            else emits a ``DeprecationWarning``.

    Returns:
        ``(model, threshold)``
    """
    if calibrating_groups is not None:
        from vtscore.datasets.vote_provenance import _retired_calibration_filter  # noqa: PLC0415

        _retired_calibration_filter("train_and_threshold(calibrating_groups=...)")

    import torch

    from vtscore.state import (
        DetectorContext,
        detector_beta,
        get_calibrate_count,
        get_calibration_fraction,
        line_knobs,
    )
    from vtscore.training import (
        calibration_folds,
        calibration_folds_cached,
        threshold_from_folds,
        train_model,
    )
    from vtscore.training.thresholds import PRECISION_FLOOR_FALLBACK_INCLUSION
    from vtscore.training.blend_schedules import BlendContext
    from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD

    X = torch.from_numpy(stack_vectors(X_list, label="training vector"))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Bag-aware setup (region flooding): size on votes not rows, split/weight
    # per bag.  On a legacy label set every bag is one row, so this collapses
    # to the historical behaviour.
    _n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)
    cal_score_rows = _calibration_score_rows(groups, cal_groups, score_rows)

    # The production detector head is a **linear SVM started from the Goods'
    # centroid** (#4785, vtscore.training.svm_prior): one Linear(d, 1) fitted to
    # the maximum-margin boundary (hinge + L2) rather than to balanced BCE, with
    # the Goods' average a cheap direction, so the head fitted at the label quota
    # starts where Test's centroid ranked and leaves it as the labels pay for.
    # Like the logistic head it has no capacity to wobble under sparse positives
    # (the threshold-stability #2790 finding); unlike it, it ranks measurably
    # better.  The same head fits the final model and the calibration fold models
    # on *both* branches below (cached and uncached), so the calibrated threshold
    # is always measured on the head the final model actually has.
    hidden_dim = LINEAR_SVM_PRIOR_HEAD

    # Which preference draws the line (#4413): the detector's own (#4665).  A
    # caller's duck-typed context carries no balance, and takes the active one's.
    beta = detector_beta(det_ctx) if isinstance(det_ctx, DetectorContext) else line_knobs()["beta"]
    # The user's persisted split wins; unset resolves to the per-space
    # production default for this detector's embedder (issue #3287).
    calibration_fraction = resolve_calibration_fraction(get_calibration_fraction(), embedder_name)
    blend_ctx = BlendContext.from_labels(y_list, cal_groups)
    # The calibration folds are computed at *every* label count.  The pre-fusion
    # path skipped them below the blend schedule's floor, where the schedule
    # discarded the cross-cal cut anyway; the fold-anchored estimator that
    # replaced the schedule needs the fold *models* (it anchors on their
    # held-out scores), so there is nothing left to skip.  Two extra linear-head
    # fits at 4-5 votes is the whole cost.
    if det_ctx is not None:
        # Cache the K folds on the context so a re-cut can re-derive the cutoff
        # without a no-op (the find-label / detector-load paths land here;
        # without the cache a balance change can't move the line).
        folds = calibration_folds_cached(
            X_list,
            y_list,
            input_dim,
            calibrate_count=get_calibrate_count(),
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            det_ctx=det_ctx,
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )
    else:
        folds = calibration_folds(
            X_list,
            y_list,
            input_dim,
            calibrate_count=get_calibrate_count(),
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            groups=cal_groups,
            score_rows_by_group=cal_score_rows,
        )
    threshold = threshold_from_folds(folds, PRECISION_FLOOR_FALLBACK_INCLUSION)

    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)

    # Fit the population estimator on the *inference* score distribution.
    # `scoring_rows_for_snap` + `score_rows_with_model` is `_score_all_media`,
    # the same call scoring makes (`score_media_with_model`), split so the rows
    # are built once and reused by the fold passes below: on a patch dataset the
    # mixture sees the region max-pooled per-media scores the threshold will
    # actually cut.  Scoring the image-level embedding matrix instead fitted it
    # on a systematically lower distribution (the region max is ≥ the single
    # image-level row), biasing the cut low → over-inclusion on
    # region-voting detectors.  Plain single-vector datasets take
    # `scoring_rows_for_snap`'s embedding-matrix fallback, so their behaviour is
    # unchanged.  *embedder_name* is forwarded as-is (not pre-resolved) so
    # the region-vs-plain gating matches inference exactly.
    # Mirrors `_train_and_score_xy` and
    # `eval.voting_iterations._safe_threshold_for_step`.
    #
    # The row builder skips media it cannot score, so an empty score list means
    # the haystack contributed nothing to fit on - either there was no haystack
    # at all, or none of its media carry a usable vector in this space.  Both
    # take the no-haystack branch rather than fitting the estimator on an empty
    # distribution.
    #
    # A *partly* embedded haystack has no such branch, and that is why the CLI
    # must embed at load: fitting the cut on the subset that happens to carry a
    # vector is silent, plausible, and wrong - a lower threshold over fewer
    # items (issue #3556).  `vtscore.cli.sources._embed_loaded_medias` is what keeps the
    # CLI's snap embedded before this call, mirroring the GUI's load pipeline.
    #
    # The population the quantile is realized on is almost always *snap* - the
    # caller scores what it calibrated against - but the CLI converts and
    # re-clips before scoring, so it names the routed snapshot instead (#3647).
    hay = haystack if haystack is not None else snap
    rows: ScoringRows | None = None
    all_ids: list[int] = []
    all_scores: list[float] = []
    if hay:
        rows = haystack_rows if haystack_rows is not None else scoring_rows_for_snap(hay, embedder_name)
        all_ids = rows.ids
        all_scores, _best_region = score_rows_with_model(model, rows)
    if rows is not None and all_scores:
        threshold = _fused_threshold(
            threshold,
            folds,
            rows,
            all_scores,
            blend_ctx,
            _blend_schedule_for_snap(hay),
            det_ctx=det_ctx,
            final_ids=all_ids,
            voted_ids=voted_ids,
            labels=labels,
            beta=beta,
        )
    elif det_ctx is not None:
        # No haystack scored: no population estimator or labels' line to
        # re-cut on a balance change, and no ranking for a balance to keep a
        # set of.  A line an earlier retrain parked belongs to a model this
        # one replaced, so it goes too.
        det_ctx.anchored_cut_cache = None
        det_ctx.labels_line = None
        det_ctx.line_ranking = None
        det_ctx.gate_passed = None

    return model, threshold


def serialize_weights(model) -> dict[str, list]:
    """Convert a PyTorch model's state dict to JSON-serialisable nested lists."""
    return {key: value.cpu().tolist() for key, value in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Vote-aware detector training (online, called from sort/vote handlers)
# ---------------------------------------------------------------------------


def _train_and_score_xy(
    X_list: list[np.ndarray],
    y_list: list[float],
    clips_dict: dict[int, dict[str, Any]],
    *,
    calibrate_count: int,
    calibration_fraction: float | None,
    det_ctx: Any,
    groups: list | None = None,
    score_rows: dict | None = None,
    voted_ids: "set[int] | None" = None,
    rows: ScoringRows | None = None,
    labels: "Mapping[int, bool] | None" = None,
    beta: float | None = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train the detector head on ``(X_list, y_list)`` and score every media in *clips_dict*.

    Shared core of :func:`train_and_score` (vote-driven) and
    :func:`vtscore.detectors.labelset_training.labelset_train_and_score`
    (labelset-driven): the two pipelines differ only in how they assemble
    ``(X_list, y_list)``, so the guard → threshold → train → score → format
    tail lives here once.

    The head is the linear SVM at every label count, so region
    flooding can't inflate its capacity; the threshold's label count is still
    sized from the **vote** count (distinct *groups*) rather than the row count,
    so flooding - which turns one Bad vote into many leaf rows - doesn't shift
    the fallback blend's small-count ramp.  When *groups*
    reveals at least one multi-row bag (flooding actually happened), the
    calibration split, fold fits, and final fit all run **bag-aware**
    (grouped fold split, per-bag loss weights), and each calibration bag
    collapses over the *scoring* rows *score_rows* supplies rather than the
    rows it trained on (see :func:`_calibration_score_rows`); otherwise every
    row is its own bag and the path is byte-for-byte the pre-flood behaviour.
    *voted_ids* names the media in *clips_dict* the labels came from, so the
    fold-anchored estimator can drop them from its haystacks (issue #3308; see
    :func:`_fused_threshold`).
    Returns ``([], 0.5, None)`` when the labels don't satisfy ≥2 samples AND
    ≥1 good AND ≥1 bad.

    *rows* lets a caller that has **already** built this snapshot's score rows
    hand them in instead of having them rebuilt.  It is a pure cache
    pass-through - the rows must be
    :func:`scoring_rows_for_snap`'s output for *clips_dict* in the same space
    this function resolves - and exists for cross-dataset Find, which scores
    several detectors over one ``temp_medias`` that
    :func:`~vtscore.embedding.matrix.get_region_matrix_for_snap` will not cache
    (its cache is keyed to the *active* dataset context).  Without it each
    detector would restack the corpus.

    *beta* is the operating point's balance (``None``: the line is the
    Inclusion 0 cut); see :func:`_fused_threshold`.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, train_model  # noqa: PLC0415
    from vtscore.training.blend_schedules import BlendContext  # noqa: PLC0415
    from vtscore.training.thresholds import (  # noqa: PLC0415
        PRECISION_FLOOR_FALLBACK_INCLUSION,
        calibration_folds_cached,
        threshold_from_folds,
    )

    from vtscore.concurrency.stalls import PhaseClock  # noqa: PLC0415

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    # Where a slow retrain spent its time (issue #3853); silent unless slow.
    clock = PhaseClock("train_and_score", rows=len(X_list), corpus=len(clips_dict))

    # The detector's primary embedder (the explicit space it scores in), or the
    # dataset score precedence when the detector has no primary yet.  Scoring
    # reads vectors from this same space the X_list were assembled in.
    score_emb = detector_score_embedder(det_ctx, clips_dict)

    # ``None`` = no explicit user setting: take the per-space production split
    # for the embedder the detector learns in (issue #3287).
    calibration_fraction = resolve_calibration_fraction(calibration_fraction, score_emb)

    X = torch.from_numpy(stack_vectors(X_list, label="training vector"))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Bag-aware setup (region flooding): size on votes not rows, split/weight
    # per bag when a Bad vote flooded its patch stack; a no-op on legacy datasets.
    _n_votes, cal_groups, sample_weights = _flood_context(X_list, y_list, groups)
    cal_score_rows = _calibration_score_rows(groups, cal_groups, score_rows)
    blend_ctx = BlendContext.from_labels(y_list, cal_groups)
    # The production head (the SVM started from the Goods' centroid) - see train_and_threshold.
    hidden_dim = LINEAR_SVM_PRIOR_HEAD

    # K-fold calibration runs at every label count: the fold-anchored estimator
    # anchors on the fold models' held-out scores, so their models are an input
    # rather than something to skip.  (The pre-fusion path skipped the fold fits
    # below the blend schedule's floor, where the schedule multiplied the
    # cross-cal cut by zero; the schedule is no longer what combines the two
    # estimators.)
    folds = calibration_folds_cached(
        X_list,
        y_list,
        input_dim,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        det_ctx=det_ctx,
        groups=cal_groups,
        score_rows_by_group=cal_score_rows,
    )
    threshold = threshold_from_folds(folds, PRECISION_FLOOR_FALLBACK_INCLUSION)
    clock.mark("calibration_folds")

    # A Good vote trains on one row (the raw patch nearest the drawn box); a
    # Bad vote trains on the image's whole score-row stack (region flooding),
    # per-bag weighted so a rejected image counts once.  On a legacy dataset
    # there are no per-bag weights, so the call stays identical to the
    # historical one-vector-per-media fit.
    if sample_weights is not None:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim, sample_weights=sample_weights)
    else:
        model = train_model(X, y, input_dim, hidden_dim=hidden_dim)
    clock.mark("final_fit")

    # One row build for the whole step: the final model's scoring pass and every
    # fold pass inside `_fused_threshold` read the same matrix (only the head
    # changes between them).  A caller that already holds that matrix passes it
    # in (see *rows* above) so a multi-detector pass over one snapshot builds it
    # once rather than once per head.
    if rows is None:
        rows = scoring_rows_for_snap(clips_dict, score_emb)
    clock.mark("score_rows")
    all_ids = rows.ids
    scores, best_region = score_rows_with_model(model, rows)
    clock.mark("score")

    # The label counts feeding the fallback blend are votes, not flooded rows,
    # so its small-count ramp is unmoved by region flooding.
    threshold = _fused_threshold(
        threshold,
        folds,
        rows,
        scores,
        blend_ctx,
        _blend_schedule_for_snap(clips_dict),
        det_ctx=det_ctx,
        final_ids=all_ids,
        voted_ids=voted_ids,
        labels=labels,
        beta=beta,
    )
    clock.mark("fused_threshold")

    results = _format_results(all_ids, scores, best_region, clips_dict)
    clock.finish(votes=_n_votes, score_rows=len(all_ids))
    return results, threshold, model


def train_and_score(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    inclusion_value: int | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float | None = None,
    vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None,
    det_ctx: Any = None,
    beta: float | None = None,
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
        inclusion_value: Deprecated (#4269): Inclusion is no longer a user
            preference.  Leave it unset; ``0`` is accepted with a
            ``DeprecationWarning`` and any other value raises ``ValueError``.
            Set *beta* to choose where the line goes.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for calibration
            in each split.  For example, 0.2 means 80% Train / 20% Calibrate.
            ``None`` (default) resolves to the per-space production split for
            the detector's embedder - 0.3 single-vector, 0.5 patch (see
            :func:`resolve_calibration_fraction`).
        vote_region_boxes: Optional ``media_id -> (x0, y0, x1, y1)`` map from
            yes-votes that designated a region.  When set and the source
            media carries a ``patch_grid``, the raw patch nearest the box
            (:func:`vtscore.media.patch_embed.nearest_patch_to_box`) trains the
            vote instead of ``media["embeddings"]``.  Falls back to the
            full-image vector when the media lacks a patch grid (legacy
            datasets, single-vector embedders) or the box is missing.
        det_ctx: The detector context to cache the calibration folds and the
            fitted estimators on.
        beta: The balance to cut at, F-beta's beta in ``[0.25, 4]``, or
            ``None`` (the default) for no balance: the Inclusion 0 cut.  Under
            a balance the line is the labels' line at it (#4413, #4452; see
            :func:`_fused_threshold`).

    Returns:
        A tuple ``(results, threshold, model)`` where:

        - ``results`` is a list of ``{"id": int, "score": float}`` dicts, sorted
          by score in descending order (highest confidence first).
        - ``threshold`` is the decision boundary as a float, the line drawn on
          ``clips_dict``'s score distribution (see :func:`_fused_threshold`):
          at *beta*, the labels' line; with no balance, or labels that support
          no class model, the fold-anchored cut at Inclusion 0, and the
          schedule blend where no folds form.  On a structural dataset the
          verification gate's boundary replaces it
          (:func:`~vtscore.training.structural_similarity.maybe_structural_rerank`).
        - ``model`` is the trained ``nn.Sequential`` model (``None`` when
          training was not possible).
    """
    from vtscore.config.core_config import _retired_inclusion  # noqa: PLC0415

    _retired_inclusion("train_and_score(inclusion_value=...)", inclusion_value)
    region_boxes = vote_region_boxes or {}
    X_list, y_list, groups, score_rows = _build_vote_xy(
        clips_dict, good_votes, bad_votes, region_boxes, detector_score_embedder(det_ctx, clips_dict)
    )
    results, threshold, model = _train_and_score_xy(
        X_list,
        y_list,
        clips_dict,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        det_ctx=det_ctx,
        groups=groups,
        score_rows=score_rows,
        voted_ids=set(good_votes) | set(bad_votes),
        labels={**dict.fromkeys(good_votes, True), **dict.fromkeys(bad_votes, False)},
        beta=beta,
    )

    # Stage-2 structural re-rank: a no-op for every non-structural dataset
    # (gated on media carrying ``local_features``), so existing datasets are
    # untouched.  For a structural (SIFT/VLAD) dataset it geometrically
    # verifies the VLAD shortlist against the RegionYes templates and re-ranks
    # by the inlier gate.  See docs/plans/structural-embedder.md.
    from vtscore.training.structural_similarity import maybe_structural_rerank  # noqa: PLC0415

    results, threshold = maybe_structural_rerank(
        results, threshold, clips_dict, good_votes, region_boxes, det_ctx, bad_votes=bad_votes, beta=beta
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
    calibration_fraction: float | None = None,
) -> tuple[dict[str, list] | None, float]:
    """Resolve origin entries to files, embed them, and train a detector head.

    This is the load-time counterpart of file-based detector export: given
    the origin lists that were saved to disk, it re-derives the head weights
    by resolving the original media files, embedding them, and training.

    **Who calls this: library consumers, not the app.**  Nothing in
    ``vtsearch`` reaches here.  The app re-derives a saved detector through
    ``POST /api/detectors/registry/load`` ->
    :func:`vtscore.detectors.labelset_training.train_from_labelset` ->
    :func:`train_and_threshold`, which is handed the active dataset's medias
    as a haystack and therefore draws the line a freshly trained detector
    draws - the labels' line at the detector's balance, the fold-anchored cut
    where the labels support no class model (pinned by
    ``tests/detectors/test_load_time_threshold_provenance.py``).  So the plain
    cross-calibration cut below is what a *library* caller with no haystack
    gets - it is not the threshold a resumed VTSearch session starts on
    (issue #3257, evaluated and closed unrun).

    Args:
        good_origins: Origin dicts for media labelled Good.
        bad_origins: Origin dicts for media labelled Bad.
        inclusion: The inclusion value to use for training.
        media_type: Media type string (e.g. ``"audio"``, ``"image"``).
        embedder_name: Name of the embedder the detector was originally
            trained with. Passed through to :func:`embed_file` so every
            re-embedded media is encoded by the same model that produced
            the saved vectors - otherwise the head trains on a mix of
            embedder outputs and learns garbage. Pass ``""`` only when you
            genuinely want the media type's default embedder (e.g. a
            brand-new detector with no recorded embedder yet).
        calibrate_count: Number of k-fold calibration splits.
        calibration_fraction: Fraction reserved for calibration.  ``None``
            (default) resolves to the per-space production split for
            *embedder_name* (see :func:`resolve_calibration_fraction`).

    Returns:
        A ``(weights, threshold)`` tuple.  ``weights`` is ``None`` if
        resolution/embedding failed for too many entries (need at least
        one good and one bad).
    """
    import torch  # noqa: PLC0415

    from vtscore.detectors.resolver import embed_file, resolve_file_context
    from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, train_model
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

    X = torch.from_numpy(stack_vectors(X_list, label="training vector"))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    # Plain cross-calibration: this load-time path re-derives a detector from
    # saved origins and has no haystack to fuse against, so there is no
    # population estimator to fit and the conformal cut ships alone.  The trainer
    # degrades gracefully below 4 labels / <2-per-class via its own 0.5
    # fallback, so no separate small-label short-circuit is needed here.
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=resolve_calibration_fraction(calibration_fraction, embedder_name),
        hidden_dim=LINEAR_SVM_PRIOR_HEAD,
    )
    model = train_model(X, y, input_dim, hidden_dim=LINEAR_SVM_PRIOR_HEAD)

    state_dict = model.state_dict()
    weights = {k: v.cpu().tolist() for k, v in state_dict.items()}
    return weights, threshold

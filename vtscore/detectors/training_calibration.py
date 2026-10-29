"""Calibrating a detector's threshold: the fold context, the split, and the line.

:func:`_fused_threshold` draws the line - the labels' line under a balance,
then the fold-anchored cut, then the schedule blend - over the haystack the
final head scored; :func:`_flood_context` and :func:`_calibration_score_rows`
make the calibration folds bag-aware on a region-flooded label set; and
:func:`resolve_calibration_fraction` picks the Train/Calibrate split for the
detector's embedding space.  Part of the detector training pipeline
(:mod:`vtscore.detectors.training`); reads
:mod:`vtscore.detectors.training_snap` and
:mod:`vtscore.detectors.training_scoring`.

``scripts/gates/check-eval-app-sync.py`` pins four of these against the eval
harness's default arm: ``_fused_threshold``, ``_calibration_score_rows``,
``_blend_schedule_for_snap`` and ``resolve_calibration_fraction``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from vtscore.detectors.training_scoring import ScoringRows, score_rows_with_model
from vtscore.detectors.training_snap import _patch_embedder_for_snap

log = logging.getLogger(__name__)


def _flood_context(
    X_list: list,
    y_list: list[float],
    groups: list | None,
) -> tuple[int, list | None, Any]:
    """Resolve the bag-aware training context for a possibly-flooded label set.

    Returns ``(n_votes, cal_groups, sample_weights)``:

    * ``n_votes`` - the number of distinct bags (votes/images); the unit the
      hidden-layer width and the fallback blend's ramp should size on, so region
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


def _blend_schedule_for_snap(snap: dict | None) -> str:
    """The fallback mix-in schedule these medias should be blended under.

    #2841 measured the two voting modes separately and they want different
    curves, so the schedule is resolved per training call rather than being one
    global constant.  The mode follows the *scoring* geometry, not how the user
    happened to vote: a patch dataset always scores by max-pooling over its raw
    patches (see :func:`_score_all_media`), which is exactly the ``region`` arm of the
    study, while a single-vector dataset is the ``binary`` arm.
    """
    from vtscore.training.blend_schedules import production_schedule_for  # noqa: PLC0415

    if not snap:
        return production_schedule_for(region_voting=None)
    return production_schedule_for(region_voting=_patch_embedder_for_snap(snap) is not None)


def _labels_line(
    folds: Any,
    final_scores: list[float],
    final_ids: list[int] | None,
    labels: "Mapping[int, bool] | None",
    beta: float,
    det_ctx: Any,
) -> float | None:
    """The balance's line from the labels (#4452), parked on *det_ctx*; ``None`` with no class model.

    The class model the calibration folds' held-out scores of the votes imply,
    cut where the expected F-beta peaks at the prevalence that model estimates
    on the corpus this retrain scored - the Train dataset in Train, the
    searched one in Find, AutoFind and the CLI.  The same labels and embedder
    give the same model everywhere; no count is drawn on any corpus, except the
    head's count line while the class model is thin (#4786,
    :func:`~vtscore.training.thresholds.labels_line.head_count_cut`).  ``None``
    (too few votes, one class) leaves the caller's fallbacks, which admit
    nothing when the folds never split.
    """
    from vtscore.training.thresholds import fit_labels_line  # noqa: PLC0415

    line = fit_labels_line(
        folds.orderings if folds.fallback is None else None,
        final_scores,
        final_ids if final_ids is not None else range(len(final_scores)),
        labels,
    )
    if det_ctx is not None:
        det_ctx.labels_line = line
    return None if line is None else float(line.threshold(beta))


def _fused_threshold(
    xcal_threshold: float,
    folds: Any,
    rows: "ScoringRows | None",
    final_scores: list[float],
    blend_ctx: Any,
    schedule: str,
    det_ctx: Any = None,
    final_ids: list[int] | None = None,
    voted_ids: "set[int] | None" = None,
    labels: "Mapping[int, bool] | None" = None,
    beta: float | None = None,
) -> float:
    """The line: the labels' line under a balance, then the fold-anchored cut, then the schedule blend.

    The labels' line (:func:`_labels_line`, #4452) is the line under a
    balance; the fold-anchored cut is the line with no balance and the
    fallback where the labels support no class model, and the schedule blend
    answers where no folds form (see "The line is drawn at an operating point"
    below).

    For the fold-anchored cut it scores the haystack (*rows*, the
    ``(media, score row)`` matrix :func:`scoring_rows_for_snap` built for the
    snapshot) through each calibration fold model, so each fold's anchored
    mixture is fitted on that fold model's own score scale with that fold's
    *held-out* votes as anchors, then carries the per-fold cuts to the final
    model in quantile space
    (:func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold`).  This is
    the estimator the 2026-08-05 deep-regime run picked over both pure
    cross-calibration and the schedule-blended GMM - see
    ``docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md``.

    **Voted media are excluded from every haystack the estimator fits on**
    (issue #3308).  Each model in the chain was trained on the votes, so the
    votes' own scores under it are optimistically shifted - and the calibration
    votes additionally sit in the haystack twice, once as free points and once
    as anchors.  Dropping all of *voted_ids* from the fold haystacks *and* from
    the final model's realization sample keeps every distribution in the
    quantile transfer over the identical population (the unlabeled remainder),
    which is the population the threshold actually decides - the voted items'
    verdicts are already known.  The effect is bounded by the votes' share of
    the (<=50k-sampled) haystack: invisible on a large corpus, a measured win
    on small ones.  The exclusion switches off entirely when it would leave
    fewer than :data:`~vtscore.training.thresholds.EXCLUSION_MIN_REMAINDER`
    scores - a remainder that small is both too coarse to read quantiles from
    and, after deep Autopilot voting, selection-biased toward whatever
    acquisition never wanted; the full contaminated haystack measures better
    there (see the constant's rationale).  The schedule-blend *fallback* below
    deliberately keeps the full distribution: it only fires when there are no
    usable folds (<4 votes, or a single class), where the contamination is at
    most a few points, and its blend weights were measured on the full
    population.  *final_ids* names the media each entry of *final_scores*
    belongs to; the exclusion needs both it and *voted_ids*, and the
    historical include-everything behaviour is kept otherwise.

    The extra scoring passes (one per fold) are the estimator's whole marginal
    cost.  Production trains the *linear* head, so a pass is a matrix multiply
    and the cost is trivial; a heavier head would want measuring first.  Only
    the head changes between passes, never the rows, so *rows* is built once by
    the caller and reused by every fold **and** by the final model's own pass -
    which is what keeps the marginal cost a matmul rather than a restack.  On
    the active dataset the builders cache anyway; on a cross-dataset snapshot
    (Find, the CLI's importer chunks) they do not, and re-deriving per fold
    would have restacked the whole corpus ``calibrate_count + 1`` times - a
    multi-gigabyte float16 rebuild on a patch dataset, where the row stack is
    ~197 rows per media.

    Falls back to the schedule blend
    (:func:`~vtscore.training.thresholds.calculate_safe_threshold`) when there
    are no usable folds - the "label set too small to form calibration folds"
    case.  The x-cal side of that
    blend is then :data:`~vtscore.training.thresholds.NO_GOOD_THRESHOLD`
    ("we never computed a cut, so admit nothing"), which is what the pre-fusion
    pure-GMM branch fed it.

    When *det_ctx* is given, the fitted estimator is parked on
    ``det_ctx.anchored_cut_cache`` so a balance change can re-cut it without
    refitting or re-scoring anything (see
    :func:`vtscore.state.core.recompute_detector_thresholds`).

    **The line is drawn at an operating point.**  Under a balance *beta*
    (the app's case, #4413) the line is the labels' line (#4452,
    :func:`_labels_line`): the class model the calibration folds' held-out
    votes imply, cut where the expected F-beta peaks at the prevalence it
    estimates on the haystack this final model scored, parked on
    ``det_ctx.labels_line`` so a balance change re-cuts it without a retrain.
    The ranking of that haystack, voted items marked, is parked on
    ``det_ctx.line_ranking``: the balance's state counts what the line keeps
    there (#4272), and a spot check draws its bands from it.  With no class
    model (too few votes, one class) or no balance (#4269, a library caller's
    choice) the line is the fold-anchored cut at Inclusion 0 through
    :func:`~vtscore.training.thresholds.reporting_line`, and with no fitted cut
    at all the schedule blend answers as it always has.

    **Unscorable media never reach the fit.**  A media the head cannot score
    (a broken vector, a destabilised model) is recorded at
    :data:`~vtscore.utils.scores.NON_FINITE_SCORE_SENTINEL`, a full unit below
    the sigmoid range; both estimators drop those before fitting, and both do
    it *inside* the shared :mod:`vtscore.training.thresholds` functions, so the
    eval harness's default arm inherits the same population without copying
    anything.  Only the warning below lives here, where the media count is a
    thing the user can act on.
    """
    from vtscore.training.thresholds import (  # noqa: PLC0415
        NO_GOOD_THRESHOLD,
        LineRanking,
        apply_vote_exclusion,
        calculate_safe_threshold,
        drop_voted,
        PRECISION_FLOOR_FALLBACK_INCLUSION,
        fit_fold_anchored_cut,
        reporting_line,
    )
    from vtscore.utils.scores import scored_only  # noqa: PLC0415

    n_unscorable = len(final_scores) - int(scored_only(final_scores).size)
    if n_unscorable:
        log.warning(
            "threshold fit: %d of %d media could not be scored and are excluded from the "
            "population estimate (check for media with broken embeddings)",
            n_unscorable,
            len(final_scores),
        )

    # One decision, taken on the final model's scores and then obeyed by every
    # fold haystack in this fit - the all-or-nothing contract lives in
    # ``apply_vote_exclusion`` rather than being re-derived per haystack here.
    fit_final, excluding = (
        apply_vote_exclusion(final_scores, final_ids, voted_ids)
        if final_ids is not None
        else (np.asarray(final_scores, dtype=np.float64), False)
    )
    # ``excluding`` is only ever True when ``voted_ids`` is a non-empty set, so
    # binding it here narrows the type for the fold loop AND makes the coupling
    # explicit: one name carries both "are we excluding" and "what by".
    exclude: set[int] | None = voted_ids if (excluding and voted_ids) else None

    cut = None
    if folds.fallback is None and rows is not None:
        n_folds = min(len(folds.models), len(folds.orderings))
        fold_haystacks = []
        for model in folds.models[:n_folds]:
            scores, _best = score_rows_with_model(model, rows)
            fold_haystacks.append(drop_voted(scores, rows.ids, exclude) if exclude else np.asarray(scores, np.float64))
        cut = fit_fold_anchored_cut(fold_haystacks, folds.orderings[:n_folds], fit_final)

    # The ranking the line keeps a set of: every scored item, with the voted
    # ones marked so the set is drawn from the unvoted remainder (#4272).  The
    # marking is unconditional - a candidate is unvoted by definition - unlike
    # the #3308 exclusion above, which is about what a *population estimate*
    # is fitted on.
    ranking = LineRanking.from_scores(
        final_ids if final_ids is not None else range(len(final_scores)), final_scores, voted_ids or ()
    )

    if det_ctx is not None:
        det_ctx.anchored_cut_cache = cut
        det_ctx.line_ranking = ranking
        # A new line is drawn on this ranking; a structural re-rank that
        # follows sets the gate's passed set afresh (#4505).
        det_ctx.gate_passed = None

    if cut is not None and cut.n_unconverged:
        # Not a fallback and not an error - the threshold is still this fit's -
        # but a fit that stopped on its iteration cap is not the estimator the
        # calibration studies measured, and until #3825 no surface said so.
        log.warning(
            "threshold fit: %d of %d anchored fold refits stopped on the iteration cap rather than "
            "converging (iterations %s); the threshold is usable but under-converged",
            cut.n_unconverged,
            len(cut.fits),
            ",".join(str(i) for i in cut.fold_iterations),
        )

    # Fitted on every retrain, so a later change of balance re-cuts it without
    # a retrain (``recut_detector_threshold``); drawn only when a balance is set.
    labels_kept = _labels_line(folds, final_scores, final_ids, labels, beta if beta is not None else 1.0, det_ctx)
    if beta is not None and labels_kept is not None:
        return labels_kept
    line = reporting_line(cut, None, inclusion_value=PRECISION_FLOOR_FALLBACK_INCLUSION, min_precision=None)
    if line.threshold is not None:
        return line.threshold
    xcal = NO_GOOD_THRESHOLD if folds.fallback is not None else xcal_threshold
    return calculate_safe_threshold(xcal, final_scores, blend_ctx, schedule=schedule)


def calibration_rows_for(groups: list | None, calibrating_groups: "set | None") -> list[bool] | None:
    """Deprecated (#4362): per training row, whether the vote behind it was in *calibrating_groups*.

    Part of the retired #4245 calibration filter, which chose the held-out
    votes the #4220 precision estimate could use as evidence.  A retrain no
    longer builds that estimate, so nothing reads this answer; it is kept, with
    a ``DeprecationWarning``, so an out-of-tree caller keeps importing it, and
    it answers as it always did.  ``None`` for *calibrating_groups* is
    ``None`` (no filter); with no *groups* no row can be mapped to a vote, so
    the answer is empty.
    """
    from vtscore.datasets.vote_provenance import _retired_calibration_filter  # noqa: PLC0415

    _retired_calibration_filter("calibration_rows_for()")
    if calibrating_groups is None:
        return None
    if groups is None:
        return []
    return [g in calibrating_groups for g in groups]


def vote_calibrating_groups(
    good_votes: "dict[int, None] | set[int]",
    bad_votes: "dict[int, None] | set[int]",
    provenance: "dict[int, dict[str, Any]]",
) -> set:
    """Deprecated (#4362): the ``("g"/"b", media id)`` bags whose vote the learned sort chose.

    Part of the retired #4245 calibration filter (see
    :func:`calibration_rows_for`); nothing reads its answer.  Kept, with a
    ``DeprecationWarning``, answering as it always did through
    :func:`~vtscore.datasets.vote_provenance.calibrates_precision`'s rule.
    """
    from vtscore.datasets.vote_provenance import _calibrates_precision, _retired_calibration_filter  # noqa: PLC0415

    _retired_calibration_filter("vote_calibrating_groups()")
    return {("g", cid) for cid in good_votes if _calibrates_precision(provenance.get(cid))} | {
        ("b", cid) for cid in bad_votes if _calibrates_precision(provenance.get(cid))
    }


def _calibration_score_rows(
    groups: list | None,
    cal_groups: list | None,
    score_rows: dict | None,
) -> dict | None:
    """Per-bag **inference** row stacks for the calibrator, or ``None``.

    Calibration collapses each held-out bag with ``max``.  Left to the training
    rows that is an unfair comparison on a patch dataset: a Good vote holds one
    row while a Bad vote holds its ~197 flooded patches, and ``max`` over 197
    draws beats ``max`` over 1 with no signal at all - so the cut lands high and
    the detector over-rejects true matches.  Handing the calibrator each voted
    image's *scoring* stack (all ~197 rows, exactly what
    :func:`_score_all_media` max-pools) puts both classes in the geometry
    inference actually uses.

    Coverage is **all-or-nothing**: unless every bag has a stack this returns
    ``None`` and the calibrator keeps pooling over the training rows.  Partial
    coverage is worse than none - a Good bag left on its 1 training row while
    the Bad bags stay at their full row count would deepen the very bias this
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


def _patch_space_for(embedder_name: str | None) -> bool | None:
    """Whether *embedder_name* learns in a patch space, or ``None`` for unknown.

    Reads the embedder's **capability** (``supports_patch_regions``), not what
    it is doing in the current configuration: a patch embedder falling back to
    whole-image voting on a boxless dataset still learns in its patch space's
    pickle, and #3287 measured that it wants the patch split there too.  An
    empty name or one the registry doesn't know maps to ``None`` ("unknown"),
    not ``False`` - the split default has a three-state contract and an
    unrecognised embedder takes the mode-agnostic fallback rather than a guess.
    """
    if not embedder_name:
        return None
    from vtscore.media import get_embedder  # noqa: PLC0415

    try:
        embedder = get_embedder(embedder_name)
    except (KeyError, ValueError):
        return None
    return bool(getattr(embedder, "supports_patch_regions", False))


def resolve_calibration_fraction(calibration_fraction: float | None, embedder_name: str | None) -> float:
    """Resolve the Train/Calibrate split for a detector on *embedder_name*.

    An explicit *calibration_fraction* (the user's persisted setting, or a
    caller that pinned one) always wins.  ``None`` means "unset", which takes
    the per-space production default (issue #3287): 0.3 when the detector
    learns in a single-vector space, 0.5 on a patch grid, and 0.5 when the
    space is unknown - see
    :func:`vtscore.training.thresholds.production_split_for`.

    Mirrored by the eval harness: ``_resolve_production_defaults`` resolves
    ``calibration_fraction=None`` through the same
    :func:`~vtscore.training.thresholds.production_split_for` table, keyed on
    whether the dataset carries a ``patch_grid`` (its spelling of "built by a
    patch embedder").  If the predicate here changes, the harness's has to
    move with it - the ``training.split_fraction_default`` mirror in
    ``scripts/gates/check-eval-app-sync.py`` pins this function for that reason.
    """
    if calibration_fraction is not None:
        return float(calibration_fraction)
    from vtscore.training.thresholds import production_split_for  # noqa: PLC0415

    return production_split_for(patch_space=_patch_space_for(embedder_name))

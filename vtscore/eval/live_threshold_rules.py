"""The threshold rules the app no longer ships, as run-level live arms (#4184).

`slides/decks/hold-the-line.deck` tells the calibration story as a ladder of
iterations, each repairing what the last one starved on.  Issue #4184 draws
that ladder as one progression of cost curves on the same benchmark, which
means running every rung *closed-loop*: the rung's cut is what reporting
reads **and** what autopilot's Hard pick aims at, exactly as the shipped cut
is today.  A paired re-cut riding the shipped trajectory (the ``schedule`` and
``gmm_variant`` rows) cannot stand in for that, because the line decides what
gets asked next, and a rung that never chose its own questions is not the rung.

Each rule here *replaces* the live line
:func:`vtscore.eval.voting_iterations._safe_threshold_for_step` drew for the
step (the labels' line under a balance, #4452, else the fold-anchored cut),
reading only what that step already computed - the fold orderings, the final
model's haystack scores, the fitted :class:`FoldAnchoredCut` - so a rung costs
no extra training and cannot move the fold models, the splits or the head.

The rules, in the deck's order:

* ``xcal_mincost`` - **iteration 1, as the "Grade My Own Homework" slide
  draws it.**  Each fold model's cost-minimising cut on its own held-out votes,
  averaged in score space and handed to the final model.  This is the rule of
  commit ``b5033a152`` (2026-02-16) - ``>=`` at an observed score, ties broken
  toward the higher cut - with one deliberate difference: the folds are the
  app's own stratified re-draws rather than two complementary halves, which the
  slide itself calls "polish, not the idea".  No GMM anywhere.
* ``gmm_mid`` - **iteration 2**, "Oops! All Haystack".  A two-component mixture
  fitted to the final model's scores over the whole haystack, cut at the
  midpoint of its means.  Reads no vote.  This never shipped as a detector's
  standalone cut - it is the deck's rung, not a historic one.
* ``blend`` - **iteration 3**, "Cross Examination".  The ``xcal_mincost`` cut
  and the ``gmm_mid`` cut combined under the run's resolved blend schedule,
  which on binary voting is ``corridor20`` (the band clamp the "Weight and See"
  slide draws).  The x-cal side is *this ladder's* rung-1 cut, not the pooled
  conformal quantile, so rung 3 is literally rungs 1 and 2 combined.
* ``anchored_rawmean`` - **iteration 4's strawman**, the ``b`` build of "No
  Mean Feat": each fold's anchored mixture cut at its own midpoint, and the
  per-fold cuts averaged *as raw scores* on the final model.  The slide's point
  is that this is wrong - three models, three scales - and the
  fold-anchored cut's quantile transfer is the fix.

An unset rule (``None``) leaves the step's own line untouched - the labels'
line under a balance, else the fold-anchored cut - so the harness's default arm
cannot drift from the app through this module.

One retired piece of the labels line lives here too, for #4668's build-up of
the F-beta era: :func:`sigma_floor` ``("absolute")`` holds the line's spread
floor at the absolute ``MIN_LOGIT_SIGMA``, the floor before #4492.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vtscore.training.blend_schedules import BlendContext
    from vtscore.training.thresholds import FoldAnchoredCut

#: The rules :func:`live_threshold` knows, in the deck's order.
LIVE_THRESHOLD_RULES: tuple[str, ...] = ("xcal_mincost", "gmm_mid", "blend", "anchored_rawmean")


def check_live_threshold(rule: str | None, *, safe_thresholds: bool, live_cut_rule: str | None) -> None:
    """Refuse a live-threshold request the run cannot honour, before any work.

    Every rule reads what the shipped fused path computed for the step, so it
    needs ``safe_thresholds``; and naming a fold-anchored ``live_cut_rule``
    beside it would configure a cut the rule then throws away.
    """
    if rule is None:
        return
    if rule not in LIVE_THRESHOLD_RULES:
        raise ValueError(f"unknown live_threshold {rule!r}; expected one of {', '.join(LIVE_THRESHOLD_RULES)}")
    if not safe_thresholds:
        raise ValueError(
            f"live_threshold={rule!r} needs safe_thresholds=True: it replaces the fused path's cut "
            "and reads the fold models and haystack scores that path computes"
        )
    if live_cut_rule is not None:
        raise ValueError(
            f"live_threshold={rule!r} and live_cut_rule={live_cut_rule!r} cannot both be set: "
            "the live threshold replaces the fold-anchored cut the cut rule would configure"
        )


def xcal_mincost_threshold(
    fold_orderings: Sequence[tuple[list[float], list[float]]],
    inclusion: float,
) -> float | None:
    """Mean of each fold's cost-minimising held-out cut, or ``None`` with no usable fold.

    Each fold's cut is :func:`~vtscore.eval.calibration_metrics.oracle_cut` at the
    inclusion's cost weights: the observed score that minimises
    ``fpr_weight * FPR + fnr_weight * FNR`` on that fold's held-out votes, under
    ``>=``, ties toward the higher cut.  A fold whose held-out votes lack either
    class cannot say where a line goes and is skipped; unscorable items are
    dropped first, as every production fold rule does.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import oracle_cut  # noqa: PLC0415
    from vtscore.training.thresholds import inclusion_cost_weights, scored_ordering  # noqa: PLC0415

    fpr_weight, fnr_weight = inclusion_cost_weights(inclusion)
    cuts: list[float] = []
    for ordering in fold_orderings:
        scores, labels = scored_ordering(ordering)
        y = np.asarray(labels, dtype=np.float64)
        if not (y == 1.0).any() or not (y == 0.0).any():
            continue
        cut, _cost, _fpr, _fnr = oracle_cut(np.asarray(scores, dtype=np.float64), y, fpr_weight, fnr_weight)
        cuts.append(float(cut))
    return float(np.mean(cuts)) if cuts else None


def gmm_midpoint_threshold(haystack_scores: Sequence[float]) -> float:
    """The label-free cut: a two-component GMM over the haystack, cut at the midpoint of its means."""
    from vtscore.training.thresholds import fit_gmm_threshold  # noqa: PLC0415
    from vtscore.utils.scores import scored_only  # noqa: PLC0415

    population = scored_only(list(haystack_scores))
    cut, _fit = fit_gmm_threshold(population.tolist())
    return float(cut)


def anchored_rawmean_threshold(cut: FoldAnchoredCut) -> float:
    """Each fold's anchored-mixture midpoint, averaged as a raw score (no quantile transfer)."""
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import gmm_cut_from_fit  # noqa: PLC0415

    return float(np.mean([gmm_cut_from_fit(fit, "mid", 1.0, 1.0)[0] for fit in cut.fits]))


def live_threshold(
    rule: str,
    *,
    fold_threshold: float,
    details: dict[str, Any],
    haystack_scores: Sequence[float] | None,
    ctx: BlendContext,
    schedule: str | None,
    fitted_cut: FoldAnchoredCut | None,
    shipped_threshold: float,
    shipped_provenance: str,
    inclusion: float,
) -> tuple[float, str]:
    """The step's live cut under *rule*, and its provenance.

    *fold_threshold* is what the fold computation returned before any fusion -
    a pooled conformal cut, or the fold rule's sentinel when the folds could not
    form - and *details* carries the fold orderings and ``fold_fallback``.
    *haystack_scores* are the final model's scores over the simulation set.
    *fitted_cut* is the fold-anchored estimator's fit (``None`` with no usable
    folds), and *shipped_threshold* / *shipped_provenance* are the line the step
    drew and its provenance: the labels' line under a balance (#4452), else
    that fit's cut, else the blend.

    Where a rule has nothing to cut on it falls back the way the app of its era
    did: ``xcal_mincost`` to the fold rule's sentinel (the too-few-labels
    ``0.5`` / ``NO_GOOD_THRESHOLD`` the fold computation has always returned),
    ``blend`` to blending ``NO_GOOD_THRESHOLD`` in place of the missing x-cal
    cut (production's substitution, see ``_blend_xcal_input``), and
    ``anchored_rawmean`` to the step's own line, which with no fold-anchored
    fit is the labels' line under a balance, else the blend.  The provenance
    names the fallback, so a reader can count how many steps a rung really cut.
    """
    from vtscore.training.thresholds import calculate_safe_threshold  # noqa: PLC0415

    if rule == "anchored_rawmean":
        if fitted_cut is None:
            return shipped_threshold, f"anchored_rawmean:fallback:{shipped_provenance}"
        return anchored_rawmean_threshold(fitted_cut), "anchored_rawmean"
    if rule == "gmm_mid":
        return gmm_midpoint_threshold(haystack_scores or []), "gmm_mid"

    xcal = None
    if details.get("fold_fallback") is None:
        xcal = xcal_mincost_threshold(details.get("fold_orderings") or [], inclusion)
    if rule == "xcal_mincost":
        if xcal is None:
            return fold_threshold, "xcal_mincost:fallback"
        return xcal, "xcal_mincost"
    if rule == "blend":
        if xcal is None:
            # Production's substitution for a cut that was never computed: blend
            # "admit nothing" rather than whichever sentinel came back.
            from vtscore.training.thresholds import NO_GOOD_THRESHOLD  # noqa: PLC0415

            xcal = NO_GOOD_THRESHOLD
        return calculate_safe_threshold(xcal, list(haystack_scores or []), ctx, schedule=schedule), "blend"
    raise ValueError(f"unknown live_threshold {rule!r}")


#: The labels line's spread floors (#4668).  ``relative`` is the app's: a share of
#: the corpus's own robust spread (#4492).  ``absolute`` is the floor before it.
SIGMA_FLOORS: tuple[str, ...] = ("relative", "absolute")


@contextmanager
def sigma_floor(mode: str) -> Iterator[None]:
    """Hold the labels line's spread floor at *mode* while the block runs (#4668).

    ``absolute`` swaps :func:`~vtscore.training.thresholds.labels_line.corpus_sigma_floor`
    for ``MIN_LOGIT_SIGMA``, which is the line exactly as it was before #4492
    (03510401e): the class model held to ``max(raw, MIN_LOGIT_SIGMA)``, and the
    corpus fits floored at ``MIN_LOGIT_SIGMA``.  Every corpus the line is drawn on
    goes through that one function, Train's and Find's alike, so the swap reaches
    both.  ``relative`` changes nothing.  The swap is process-wide, which suits a
    run whose every cell is the same arm, and it is undone on the way out.
    """
    if mode not in SIGMA_FLOORS:
        raise ValueError(f"unknown sigma_floor {mode!r}; expected one of {', '.join(SIGMA_FLOORS)}")
    if mode == "relative":
        yield
        return
    from vtscore.training.thresholds import labels_line  # noqa: PLC0415

    shipped = labels_line.corpus_sigma_floor

    def absolute(scores: Any) -> float:
        del scores  # the absolute floor reads no corpus
        return labels_line.MIN_LOGIT_SIGMA

    labels_line.corpus_sigma_floor = absolute
    try:
        yield
    finally:
        labels_line.corpus_sigma_floor = shipped


@contextmanager
def labels_chances(bad: "str | None" = None, bound_margin: "float | None" = None) -> Iterator[None]:
    """Hold the labels line's chances at an arm's pieces while the block runs (#5040).

    *bad* places the 3-part fit's Bad component
    (:data:`~vtscore.training.thresholds.labels_line.BAD_COMPONENT`): ``"corpus"``
    fits it on the corpus being decided, the held-out Bads counted as labelled
    points of it, ``"corpus_mean"`` its mean alone (the app's since #5040),
    ``"labels"`` pins it at the held-out Bads' mean (the line before).
    *bound_margin* is the factor the 2-part estimate of the positives may
    exceed the counted bound by (``_BOUND_MARGIN``, the app's 4; 2 before
    #5040); ``math.inf`` removes the bound.  ``None``
    leaves a piece as the app ships it.  Every line, Train's and Find's alike,
    reads both constants when it is drawn, so the swap reaches both, and through
    the chances Autopilot's acquisition and its stop too.  Process-wide, as
    :func:`sigma_floor`'s swap is, and undone on the way out.
    """
    from vtscore.training.thresholds import labels_line  # noqa: PLC0415

    if bad is not None and bad not in labels_line.BAD_COMPONENTS:
        raise ValueError(f"labels_chances bad must be one of {labels_line.BAD_COMPONENTS} or None; got {bad!r}")
    if bound_margin is not None and (
        isinstance(bound_margin, bool) or not isinstance(bound_margin, (int, float)) or not bound_margin >= 1.0
    ):
        raise ValueError(f"labels_chances bound_margin must be a number >= 1 or None (the app's); got {bound_margin!r}")
    shipped = labels_line.BAD_COMPONENT, labels_line._BOUND_MARGIN
    if bad is not None:
        labels_line.BAD_COMPONENT = bad
    if bound_margin is not None:
        labels_line._BOUND_MARGIN = float(bound_margin)
    try:
        yield
    finally:
        labels_line.BAD_COMPONENT, labels_line._BOUND_MARGIN = shipped


@contextmanager
def head_count_bads(bads: "int | str | None") -> Iterator[None]:
    """Hold the Bads half of the labels line's head-count gate at *bads* while the block runs (#4786).

    Until the class model holds ``HEAD_COUNT_GOODS`` Goods, and while it holds fewer than
    :data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_BADS` Bads (unbounded in the
    app since #4798), the line keeps the head's count
    (:func:`~vtscore.training.thresholds.labels_line.head_count_cut`).  *bads* bounds that
    half: ``0`` is the line before #4786, ``32`` is #4786's gate, another count prices
    another gate, and ``None`` changes nothing (the app's).  Every line reads the module's constant when it cuts,
    Train's and Find's alike, so the swap reaches both; it is process-wide, as
    :func:`sigma_floor`'s is, and undone on the way out.
    """
    if bads is None:
        yield
        return
    if isinstance(bads, bool) or not isinstance(bads, int) or bads < 0:
        raise ValueError(f"head_count_bads must be an integer >= 0 or None (the app's); got {bads!r}")
    from vtscore.training.thresholds import labels_line  # noqa: PLC0415

    shipped = labels_line.HEAD_COUNT_BADS
    labels_line.HEAD_COUNT_BADS = bads
    try:
        yield
    finally:
        labels_line.HEAD_COUNT_BADS = shipped


@contextmanager
def head_count_recall(exponent: "float | str | None" = None, z: "float | str | None" = None) -> Iterator[None]:
    """Hold the head's count line above a balance of 1 at an arm's pieces while the block runs (#5039).

    *exponent* is how the count grows above beta 1
    (:data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_RECALL_EXPONENT`), *z* where
    the line reaches down the head's excess curve when a held-out Good ranks below the count
    (:data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_RECALL_Z`).  ``"off"`` is that
    piece before #5039 (#4603's ``beta ** 0.708``; no reach), and ``None`` leaves it as the app
    ships it.  Every line reads the module's constants when it cuts, Train's and Find's alike, so
    the swap reaches both; it is process-wide, as :func:`head_count_bads`'s is, and undone on the
    way out.
    """
    for name, v in (("exponent", exponent), ("z", z)):
        if v is None or v == "off":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"head_count_recall {name} must be a finite number, 'off' or None (the app's); got {v!r}")
    from vtscore.training.thresholds import labels_line  # noqa: PLC0415

    shipped = labels_line.HEAD_COUNT_RECALL_EXPONENT, labels_line.HEAD_COUNT_RECALL_Z
    if exponent is not None:
        labels_line.HEAD_COUNT_RECALL_EXPONENT = None if exponent == "off" else float(exponent)
    if z is not None:
        labels_line.HEAD_COUNT_RECALL_Z = None if z == "off" else float(z)
    try:
        yield
    finally:
        labels_line.HEAD_COUNT_RECALL_EXPONENT, labels_line.HEAD_COUNT_RECALL_Z = shipped


@contextmanager
def head_count_precision(scale: "float | str | None" = None, exponent: "float | str | None" = None) -> Iterator[None]:
    """Hold the head's count multiplier at a balance of 1 and below at an arm's while the block runs (#5084).

    The count keeps *scale* ``* beta **`` *exponent* times what stands out
    (:data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_PRECISION_SCALE`,
    :data:`~vtscore.training.thresholds.labels_line.HEAD_COUNT_PRECISION_EXPONENT`).  ``"off"`` is
    that piece before #5084 (#4603's scale 1, exponent 0.708), and ``None`` leaves it as the app
    ships it.  Process-wide, as :func:`head_count_recall`'s swap is, and undone on the way out.
    """
    for name, v in (("scale", scale), ("exponent", exponent)):
        if v is None or v == "off":
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(
                f"head_count_precision {name} must be a positive number, 'off' or None (the app's); got {v!r}"
            )
    from vtscore.training.thresholds import labels_line  # noqa: PLC0415

    shipped = labels_line.HEAD_COUNT_PRECISION_SCALE, labels_line.HEAD_COUNT_PRECISION_EXPONENT
    if scale is not None:
        labels_line.HEAD_COUNT_PRECISION_SCALE = None if scale == "off" else float(scale)
    if exponent is not None:
        labels_line.HEAD_COUNT_PRECISION_EXPONENT = None if exponent == "off" else float(exponent)
    try:
        yield
    finally:
        labels_line.HEAD_COUNT_PRECISION_SCALE, labels_line.HEAD_COUNT_PRECISION_EXPONENT = shipped

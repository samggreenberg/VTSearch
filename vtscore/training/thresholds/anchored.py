"""Population anchoring: fusing the votes into the haystack's mixture, per fold.

The shipped threshold path.  Where :mod:`vtscore.training.thresholds.gmm`
fits one mixture to one score list, this layer runs that fit *per calibration
fold* with the fold's votes anchored into it, transfers each fold's cut into a
common quantile space (:func:`rank_transfer`), and combines them into a single
re-cuttable estimator (:class:`FoldAnchoredCut`,
:func:`fold_anchored_gmm_threshold`).  It also owns the voted-media haystack
exclusion (:func:`apply_vote_exclusion`).
"""

from __future__ import annotations

import math
from collections.abc import Container, Sequence
from dataclasses import dataclass

import numpy as np

from vtscore.training.thresholds.gmm import (
    GmmFit1D,
    fit_anchored_score_gmm,
    fit_score_gmm,
    gmm_cut_from_fit,
    gmm_fit_array,
    scored_ordering,
    snap_cut_to_sample,
)
from vtscore.training.thresholds.knobs import inclusion_cost_weights
from vtscore.utils.scores import scored_only


def rank_transfer(
    cut: float, source_scores: "list[float] | np.ndarray", target_scores: "list[float] | np.ndarray"
) -> float:
    """Carry *cut* from one score distribution to another as a quantile.

    Reads the empirical quantile of *cut* in *source_scores* (the fraction of
    source scores strictly below it) and realizes that quantile on
    *target_scores* via linear interpolation.  A cut is a point on a score
    *scale*; two models scoring the same haystack are related by an
    approximately monotone map, and a quantile is invariant under any monotone
    map - so this is the scale-transfer step that lets a cut measured on a fold
    model's distribution be applied to the final model's (deficit 2 of
    ``docs/plans/population-anchored-calibration.md``).
    """
    src = np.sort(np.asarray(source_scores, dtype=np.float64).ravel())
    tgt = np.asarray(target_scores, dtype=np.float64).ravel()
    if src.size == 0 or tgt.size == 0:
        return float(cut)
    q = float(np.searchsorted(src, cut, side="left")) / float(src.size)
    return float(np.quantile(tgt, min(1.0, max(0.0, q))))


#: Production anchor mass for the fold-anchored threshold: **κ = 0.3**, i.e.
#: each vote counts as three tenths of a haystack point among the ~50k the
#: mixture is fitted on.  The 2026-08-05 deep-regime run swept κ ∈ {1 … 100}
#: and found performance degrading monotonically as κ grew, leaving the optimum
#: on the grid's bottom edge; the 2026-08-06 anchor-mass sweep extended the grid
#: to κ ∈ {0.01 … 3} across six environments and found the optimum **interior**
#: at κ=0.3 (docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md).
FOLD_ANCHOR_WEIGHT = 0.3

#: Production cut rule for the fold-anchored threshold: the **midpoint
#: anchored at inclusion 0, tilted by the rate rule** (issue #2865).
#:
#: The anchor-mass sweep picked the plain midpoint at κ=0.3, and the mechanism
#: is that ``mid`` ignores the mixture weights while ``rate`` reads them: at
#: light anchoring the weights carry the votes' acquisition-biased prevalence
#: out of proportion to their honesty, so the rule that never looks at them
#: wins.  But every arm of both calibration runs was scored at inclusion 0,
#: and a bare midpoint also ignores the *cost* weights the Inclusion knob
#: arrives as - shipping it verbatim made the knob a no-op for every detector
#: with usable folds.
#:
#: ``mid_tilt`` keeps the measured winner exactly where it was measured and
#: restores the knob everywhere else.  In fold-quantile space,
#: ``q(k) = q_mid + (q_rate(k) - q_rate(0))``: the midpoint's combined fold
#: quantile, shifted by however far the rate-optimal cut's own quantile moves
#: from its inclusion-0 position (see :meth:`FoldAnchoredCut._quantile_at`).
#: At inclusion 0 the shift is identically zero, so the threshold is
#: bit-for-bit the measured ``κ=0.3, mid`` arm; away from 0 it inherits
#: ``rate``'s monotone tilt without inheriting ``rate``'s weight-biased
#: *location*.
#:
#: **The tilt is measured** (issue #2865, 336 cells over four environments and
#: thirteen stops of the knob, on the shipped head; see
#: ``docs/experiments/2026-08-21-inclusion-cut-rule/REPORT.md``).  It held: no candidate
#: both delivered more of the knob and stayed within the pre-registered 0.01
#: regret tolerance at every stop.  Two numbers worth keeping here - the ``mid``
#: cut this replaced admitted **one** set for the whole slider in every one of
#: 65,671 measured cell-steps and cost up to +0.18 regret away from inclusion 0;
#: and because ``mid_tilt`` differs from ``rate`` by the *constant*
#: ``q_mid - q_rate(0)`` in fold-quantile space, that sweep re-priced the
#: inclusion-0 choice under thirteen cost weightings and ``mid``'s location
#: survived all of them.
FOLD_ANCHOR_CUT_RULE = "mid_tilt"

#: Production fold-combine rule: mean of the per-fold quantiles.  With the
#: shipped ``calibrate_count=2`` the mean and the median coincide.
FOLD_ANCHOR_COMBINE = "qmean"

#: Minimum unlabeled-remainder size for the voted-media haystack exclusion
#: (issue #3308).  The fold-anchored estimator drops the voted media from its
#: haystacks - their scores under the models trained on them are optimistically
#: shifted - but only while the remainder is still big enough to *be* a
#: population estimate.  Below this many remaining scores the exclusion is
#: switched off entirely and the full (contaminated) haystack is used, because
#: two failure modes take over at once: the empirical quantiles the transfer
#: runs on lose resolution (1/n per order statistic), and after deep Autopilot
#: voting the leftover items are exactly the ones acquisition never found
#: interesting, so the remainder is a *selection-biased* sample of the corpus.
#: Measured on the overlapping-data eval environment (60-item sim set, votes
#: driven to exhaustion, 8 seeds paired step-for-step): exclusion is *neutral*
#: at remainder 50-56 (-0.004 +/- 0.007 direct cost, and only trajectory noise
#: downstream) and harmful below - +0.025 at remainder 40-49, +0.057 at 30-39,
#: +0.18 under 10 (the cut collapses onto a handful of drained leftovers,
#: FNR 0.7-0.9).  A 200-item synthetic single-vector environment is clearly
#: *positive* (-0.03 to -0.06) at every measured remainder >=60, even with 70%
#: of the corpus voted.  60 is therefore the smallest floor with a measured
#: win above it and nothing but measured neutrality or harm below it.  The
#: switch is all-or-nothing, never partial, so the fold haystacks and the
#: final realization sample always cover one identical population.
#:
#: **Both measurements behind this number are synthetic**, which is why issue
#: #3312 puts it on the GRID against real embeddings.  Until that run reports,
#: read 60 as the value that no *measured* cell argues against rather than as a
#: located optimum.
EXCLUSION_MIN_REMAINDER = 60


def resolve_exclusion_floor(min_remainder: float | None = None) -> float:
    """The remainder floor the #3308 vote exclusion should use.

    ``None`` - what every production caller passes - resolves to the shipped
    :data:`EXCLUSION_MIN_REMAINDER`, the same three-state contract
    :func:`production_split_for` and
    :func:`~vtscore.training.blend_schedules.production_schedule_for` follow.
    A float pins it, and ``math.inf`` switches the exclusion off entirely (the
    pre-#3308 behaviour), so one scalar spans the whole axis with no sentinel:
    0 always excludes, ``inf`` never does, and the shipped value sits between.

    The override exists for the #3312 eval arms.  Nothing in the app passes it
    - there is no user setting for the floor - so the app and the harness's
    default arm resolve through this same call and cannot drift apart.
    """
    return float(EXCLUSION_MIN_REMAINDER if min_remainder is None else min_remainder)


def drop_voted(
    scores: "list[float] | np.ndarray",
    score_ids: "Sequence[int]",
    voted_ids: "Container[int]",
) -> np.ndarray:
    """*scores* with the entries whose id is in *voted_ids* removed.

    Filters against **this array's own ids** rather than a mask computed
    elsewhere, because the id order is not guaranteed to be the same for two
    different models: the app's snapshot scorer is row-ordered and stable, but
    the harness's region path returns rows sorted by score, which is
    model-dependent by construction.  A shared positional mask would silently
    drop the wrong media there.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    keep = np.fromiter((i not in voted_ids for i in score_ids), dtype=bool, count=len(score_ids))
    return arr[keep]


def apply_vote_exclusion(
    scores: "list[float] | np.ndarray",
    score_ids: "Sequence[int]",
    voted_ids: "set[int] | None",
    *,
    min_remainder: float | None = None,
) -> tuple[np.ndarray, bool]:
    """``(haystack, applied)`` - the population the threshold should be fitted on.

    **The single decision point for the #3308 exclusion**, called once per
    training step on the *final* model's scores.  When it returns
    ``applied=True`` the caller must put every other haystack in that step -
    each calibration fold's - through :func:`drop_voted` as well; when it
    returns ``False`` every haystack keeps its full population.  Both the app
    (:func:`vtscore.detectors.training._fused_threshold`) and the eval
    harness's default arm route through here, so the floor policy exists in one
    place and the "all-or-nothing, never partial" contract is structural rather
    than a rule each caller has to remember.

    The exclusion is declined - ``applied=False``, *scores* returned whole -
    when nothing is voted, or when it would leave fewer than
    :func:`resolve_exclusion_floor`'s remainder.  A remainder of zero always
    declines, whatever the floor: an empty haystack is not a population
    estimate, it is the absence of one.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if not voted_ids or arr.size == 0:
        return arr, False
    kept = drop_voted(arr, score_ids, voted_ids)
    if kept.size == 0 or kept.size < resolve_exclusion_floor(min_remainder):
        return arr, False
    return kept, True


#: Step size of the **eval-only** ``"q_tilt"`` cut rule (issue #2865's candidate
#: 3), in units of combined-fold quantile per inclusion step.
#:
#: ``q_tilt`` decouples the Inclusion knob from the mixture entirely: it takes
#: the measured midpoint's combined fold quantile and shifts it by a fixed
#: amount per step of the knob, so the admitted fraction moves by construction
#: rather than by whatever the fitted Gaussians happen to imply.  That makes it
#: the simplest rule that cannot be inclusion-blind - and its price is this free
#: parameter, which has no principled value and must be *fitted*.
#:
#: **The step size has now been swept, and the rule lost at every value of it**
#: (issue #2865: {0.005, 0.01, 0.02, 0.04, 0.08} x four environments; see
#: ``docs/experiments/2026-08-21-inclusion-cut-rule/REPORT.md``).  Small steps keep the
#: knob and cannot move far enough at large ``|k|``; large steps run the
#: quantile past 1.0 and admit nothing at the ends.  There is no value at which
#: ``q_tilt`` is not worse than the shipped ``mid_tilt``, so 0.02 remains what
#: it always was - an arbitrary constant on an arm that exists to be beaten -
#: and nothing should ship at it.
FOLD_ANCHOR_QTILT_STEP = 0.02


@dataclass(frozen=True, eq=False)
class FoldAnchoredCut:
    """A fitted fold-anchored ("cross-LabeledGMM") estimator, ready to re-cut.

    Holds everything the threshold depends on *except* inclusion: the per-fold
    anchored mixtures, each fold's sorted haystack sample (to read a cut's
    quantile in the scale it was measured on), and the final model's sorted
    haystack sample (to realize the combined quantile on the scale the
    threshold is applied on).

    Splitting the fit from the cut is what makes the Inclusion knob cheap
    *and* faithful under this estimator: re-cutting at another inclusion is
    arithmetic on the fitted Gaussians plus two array lookups - no EM, no
    scoring pass - so an Inclusion slide reproduces exactly what a fresh
    retrain at that inclusion would have stored (see
    :func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`).

    Under the shipped :data:`FOLD_ANCHOR_CUT_RULE` (``"mid_tilt"``) a re-cut
    answers the knob: inclusion 0 reproduces the measured midpoint cut
    bit-for-bit, and every other inclusion shifts the midpoint's combined
    quantile by the rate rule's own displacement from *its* inclusion-0
    position (issue #2865; see :meth:`_quantile_at`).
    """

    fits: tuple[GmmFit1D, ...]
    fold_haystacks: tuple[np.ndarray, ...]
    final_haystack: np.ndarray
    n_anchored: int
    #: Per-fold count of anchors that actually reached that fold's fit - 0 for a
    #: fold that degenerated and fell back to its unanchored GMM.  ``n_anchored``
    #: counts FOLDS, so it cannot answer "how much mass did the labels carry?";
    #: that needs the vote count per fold, which is otherwise discarded here.
    #: Empty when the cut was built by a caller that predates the field.
    anchor_counts: tuple[int, ...] = ()
    cut_rule: str = FOLD_ANCHOR_CUT_RULE
    combine: str = FOLD_ANCHOR_COMBINE
    #: Only read by the eval-only ``"q_tilt"`` rule; see
    #: :data:`FOLD_ANCHOR_QTILT_STEP`.
    qtilt_step: float = FOLD_ANCHOR_QTILT_STEP

    @property
    def provenance(self) -> str:
        """``"fold_anchored[a/k]"`` - *a* of the *k* used folds fitted anchored."""
        return f"fold_anchored[{self.n_anchored}/{len(self.fits)}]"

    def _combined_fold_quantile(self, rule: str, fpr_weight: float, fnr_weight: float) -> float:
        """Combined fold quantile of *rule*'s per-fold cuts at these cost weights."""
        quantiles = []
        for fit, src in zip(self.fits, self.fold_haystacks, strict=True):
            cut, _kind = gmm_cut_from_fit(fit, rule, fpr_weight, fnr_weight)
            quantiles.append(float(np.searchsorted(src, cut, side="left")) / float(src.size))
        if self.combine == "qmean":
            return float(np.mean(quantiles))
        if self.combine == "qmedian":
            return float(np.median(quantiles))
        raise ValueError(f"unknown fold combine {self.combine!r}; expected 'qmean' or 'qmedian'")

    def _quantile_at(self, fpr_weight: float, fnr_weight: float) -> float:
        """Combined fold quantile at these cost weights, under ``cut_rule``.

        ``"mid"`` and ``"rate"`` are per-fit rules (:func:`gmm_cut_from_fit`)
        read straight through.  ``"mid_tilt"`` is composed here rather than per
        fit because it is defined in fold-quantile space:
        ``q = q_mid + (q_rate(weights) - q_rate(equal weights))``.  At equal
        cost weights - inclusion 0 - the parenthesised shift is *identically*
        zero (both terms are the same computation on the same fits), so the
        rule is bit-for-bit the plain midpoint exactly where the calibration
        runs measured it.  Elsewhere it moves the admitted fraction by however
        much the rate-optimal cut would have moved its own, so it inherits
        ``rate``'s monotonicity in the cost ratio without inheriting ``rate``'s
        weight-biased inclusion-0 location.  A fold too degenerate for a rate
        cut contributes a zero shift (its ``rate`` cut falls back to the
        midpoint at every weight), degrading that fold to plain ``mid`` rather
        than poisoning the tilt.

        ``"q_tilt"`` (**eval-only**, issue #2865's candidate 3) is the same
        shape with the mixture taken out of the tilt: ``q = q_mid - step *
        log2(fnr/fpr)``.  The log-cost ratio *is* the inclusion value for
        weights from :func:`inclusion_cost_weights`, so this reads "shift the
        admitted fraction by :attr:`qtilt_step` per step of the knob" - also
        identically ``q_mid`` at inclusion 0, also monotone, but moving the
        admitted set *by construction* rather than by whatever the fitted
        Gaussians happen to imply.  Its step size is a free parameter with no
        principled value (:data:`FOLD_ANCHOR_QTILT_STEP`), which is the whole of
        what it trades away for that guarantee.
        """
        if self.cut_rule == "mid_tilt":
            q_mid = self._combined_fold_quantile("mid", 1.0, 1.0)
            q_rate = self._combined_fold_quantile("rate", fpr_weight, fnr_weight)
            q_rate_zero = self._combined_fold_quantile("rate", *inclusion_cost_weights(0))
            return q_mid + (q_rate - q_rate_zero)
        if self.cut_rule == "q_tilt":
            q_mid = self._combined_fold_quantile("mid", 1.0, 1.0)
            return q_mid - self.qtilt_step * math.log2(fnr_weight / fpr_weight)
        return self._combined_fold_quantile(self.cut_rule, fpr_weight, fnr_weight)

    def quantile_at(self, inclusion_value: int) -> float:
        """The combined fold quantile this estimator admits at *inclusion_value*.

        :meth:`threshold_at` realizes this quantile on the final model's
        haystack; reading it directly separates "did the rule move the cut" from
        "did the haystack have anything there to move past", which is the
        distinction the #2865 sweep turns on - on a cleanly separated haystack a
        whole band of the knob can move the quantile while realizing the same
        threshold and the same admitted set.
        """
        return self._quantile_at(*inclusion_cost_weights(inclusion_value))

    def threshold_at(self, inclusion_value: float) -> float:
        """The threshold this estimator cuts at *inclusion_value*.

        Inclusion reaches the cut only as the rate weights it optimises
        (:func:`inclusion_cost_weights`).  Under the shipped ``"mid_tilt"``
        rule the midpoint's combined quantile is shifted by the rate rule's
        displacement from its own inclusion-0 position (see
        :meth:`_quantile_at`), so raising inclusion lowers the threshold and
        admits more - the same direction the conformal rule moves - while
        inclusion 0 remains exactly the measured midpoint cut.  ``"rate"``
        tilts the crossing itself; plain ``"mid"`` ignores the weights and is
        constant in inclusion.

        The realized value is finally canonicalised by
        :func:`snap_cut_to_sample`, so a cut landing between two adjacent
        haystack scores reports the midpoint of that empty interval rather than
        wherever ``np.quantile``'s linear interpolation put it.  That is
        decision-exact - the admitted set is identical either way - and it is
        what makes the threshold **reproducible**: without it, interpolating
        across an empty interval multiplies any wobble in *q* by
        ``(n - 1) * gap``, which on a saturated distribution turned a
        sub-part-per-million cross-machine difference into a visible threshold
        move (issue #3166).  The cost is that the threshold now steps rather
        than slides *within* one order-statistic band, which is honest: the
        admitted set was constant across that band all along.

        **Monotone by construction**, so the included sets stay nested
        (everything included at ``k`` stays included at ``k + 1``) - the
        contract that makes "cut off at Inclusion 1, verify up to Inclusion 4"
        well defined.  Every link in the chain is monotone: the per-fold cut in
        the cost weights (:func:`gmm_cut_from_fit`, continued past the
        inter-mean interval at its first-order slope rather than clamped, so
        the exits stay monotone *and strictly moving* - issue #2896), the
        empirical quantile of that cut in its fold's haystack, the mean/median
        across folds, the ``mid_tilt`` composition (a constant plus a
        monotone-in-inclusion shift), realizing a quantile on the final
        haystack, and the snap (which maps each order-statistic band to its own
        midpoint, so it is non-decreasing).  The plateaus in the chain are the
        honest ones - a band of the knob over which the *admitted set* does not
        change, including the boundary where a cut runs off its haystack's
        support and the quantile pins at 0 or 1 - crucially, the acquisition offset
        (:data:`ACQUISITION_INCLUSION_OFFSET`) therefore stays a real gap
        across the slider instead of silently collapsing wherever the tilt
        used to saturate.
        """
        if self.final_haystack.size == 0:
            return 0.5
        q = self._quantile_at(*inclusion_cost_weights(inclusion_value))
        realized = float(np.quantile(self.final_haystack, min(1.0, max(0.0, q))))
        return snap_cut_to_sample(realized, self.final_haystack)


def fit_fold_anchored_cut(
    fold_haystack_scores: "list[np.ndarray]",
    fold_anchor_orderings: list[tuple[list[float], list[float]]],
    final_scores: "list[float] | np.ndarray",
    *,
    anchor_weight: float = FOLD_ANCHOR_WEIGHT,
    cut_rule: str = FOLD_ANCHOR_CUT_RULE,
    combine: str = FOLD_ANCHOR_COMBINE,
) -> FoldAnchoredCut | None:
    """Fit the fold-anchored mixtures; ``None`` when no fold yielded a fit.

    Per calibration fold *k*: fit the anchored mixture on the **fold model's**
    haystack scores (``fold_haystack_scores[k]``) with anchors from that fold's
    *held-out* labelled scores (``fold_anchor_orderings[k]``, the same
    ``(scores, labels)`` orderings the conformal rule pools).  Anchors and
    population then share one scale and the anchors are honest - the labelled
    items were not in that fold model's training set, so their scores carry no
    train-set optimism.

    A fold whose anchored fit degenerates falls back to that fold's
    **unanchored** GMM fit; a fold that fails both is dropped.  ``None`` means
    every fold failed (or there were none), leaving the terminal fallback to
    the caller - :func:`fold_anchored_gmm_threshold` cuts the final model's own
    distribution instead, never 0.5.

    Every score that reaches a fit - haystack and anchor alike - is first put
    through :func:`~vtscore.utils.scores.scored_only`, so an item the head
    could not score (:data:`~vtscore.utils.scores.NON_FINITE_SCORE_SENTINEL`,
    ``-1.0``) is *absent* rather than treated as an observation a full unit
    below the sigmoid range.  Without that, a couple of unreadable media drag
    the fitted cut - and, through the quantile realised on the final haystack,
    the shipped threshold - below zero, where every real score clears it
    (issue #3180).
    """
    final_arr = gmm_fit_array(scored_only(final_scores))
    if final_arr.size == 0:
        return None
    fits: list[GmmFit1D] = []
    haystacks: list[np.ndarray] = []
    anchor_counts: list[int] = []
    n_anchored = 0
    for hay, ordering in zip(fold_haystack_scores, fold_anchor_orderings, strict=True):
        a_scores, a_labels = scored_ordering(ordering)
        arr = gmm_fit_array(scored_only(hay))
        fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=anchor_weight)
        n_anchors = 0
        if fit is None:
            fit = fit_score_gmm(arr)
            if fit is None:
                continue
        elif provenance == "anchored":
            n_anchored += 1
            # The anchors this fold's fit actually used.  A fold that fell back
            # records 0: its fit saw no anchors, so the mass they carried in it
            # is zero, not "the number we hoped to anchor with".
            n_anchors = int(np.size(a_scores))
        fits.append(fit)
        haystacks.append(np.sort(arr))
        anchor_counts.append(n_anchors)
    if not fits:
        return None
    return FoldAnchoredCut(
        fits=tuple(fits),
        fold_haystacks=tuple(haystacks),
        final_haystack=np.sort(final_arr),
        n_anchored=n_anchored,
        anchor_counts=tuple(anchor_counts),
        cut_rule=cut_rule,
        combine=combine,
    )


def fold_anchored_gmm_threshold(
    fold_haystack_scores: "list[np.ndarray]",
    fold_anchor_orderings: list[tuple[list[float], list[float]]],
    final_scores: "list[float] | np.ndarray",
    inclusion_value: int = 0,
    *,
    anchor_weight: float = FOLD_ANCHOR_WEIGHT,
    cut_rule: str = FOLD_ANCHOR_CUT_RULE,
    combine: str = FOLD_ANCHOR_COMBINE,
) -> tuple[float, str]:
    """The fold-anchored ("cross-LabeledGMM") mixture threshold (#2852 comment).

    **This is the shipped threshold path**: the 2026-08-06 anchor-mass sweep
    measured it at the defaults above as the best rule this harness has seen,
    and the best single global setting available - pooled over six environments
    it cuts −0.0437 paired regret vs pure cross-calibration in the deep regime,
    it beats the previously shipped ``κ=1, rate`` head to head in 6 of 6
    environments, and forcing it everywhere leaves each environment within
    0.0067 of its own optimum.  See
    ``docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md``.  The eval
    harness calls this same function for its default arm, so a measured
    baseline cannot drift from the app.

    Two limits of that recommendation are on record rather than fixed here.
    The gain tracks *positive*-anchor count, so on binary-voting detectors with
    few positives this only reaches a dead heat with the ``cap50`` blend it
    replaced (open work in ``docs/plans/population-anchored-calibration.md``).
    And every arm was scored at inclusion 0 - which the shipped ``mid_tilt``
    rule reproduces bit-for-bit while restoring the Inclusion tilt away from
    it; the tilt itself is the part issue #2865's inclusion sweep still owes a
    measurement.

    Fits via :func:`fit_fold_anchored_cut`, then carries each fold's cut to the
    final model as a quantile of that fold's haystack distribution
    (:func:`rank_transfer`'s argument: two models scoring the same haystack are
    related by an approximately monotone map, and quantiles are invariant under
    monotone maps), combines the folds in **quantile** space so no cross-scale
    averaging of raw cuts ever happens, and realizes the result on
    *final_scores*.

    Degeneracy policy per the issue: a fold whose anchored fit degenerates
    falls back to that fold's unanchored GMM fit; if every fold fails both
    fits, the final model's own unanchored GMM midpoint is returned (and its
    median if even that fails) - never 0.5.  Returns ``(threshold,
    provenance)`` with provenance ``"fold_anchored[a/k]"`` (*a* folds anchored
    of *k* used) or ``"fold_fallback_final_unanchored"`` /
    ``"fold_fallback_final_median"`` on the terminal fallbacks.
    """
    cut = fit_fold_anchored_cut(
        fold_haystack_scores,
        fold_anchor_orderings,
        final_scores,
        anchor_weight=anchor_weight,
        cut_rule=cut_rule,
        combine=combine,
    )
    if cut is not None:
        return cut.threshold_at(inclusion_value), cut.provenance

    final_arr = gmm_fit_array(final_scores)
    fallback_fit = fit_score_gmm(final_arr) if final_arr.size >= 2 else None
    if fallback_fit is not None:
        return fallback_fit.midpoint(), "fold_fallback_final_unanchored"
    if final_arr.size:
        return float(np.median(final_arr)), "fold_fallback_final_median"
    return 0.5, "fold_fallback_final_median"

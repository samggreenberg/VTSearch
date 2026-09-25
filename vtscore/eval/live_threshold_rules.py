"""The threshold rules the app no longer ships, as run-level live arms (#4184).

`slides/decks/hold-the-line.deck` tells the calibration story as a ladder of
iterations, each repairing what the last one starved on.  Issue #4184 draws
that ladder as one progression of cost curves on the same benchmark, which
means running every rung *closed-loop*: the rung's cut is what reporting
reads **and** what autopilot's Hard pick aims at, exactly as the shipped cut
is today.  A paired re-cut riding the shipped trajectory (the ``schedule`` and
``gmm_variant`` rows) cannot stand in for that, because the line decides what
gets asked next, and a rung that never chose its own questions is not the rung.

Each rule here *replaces* the shipped live cut after
:func:`vtscore.eval.voting_iterations._safe_threshold_for_step` has fitted it,
reading only what that step already computed - the fold orderings, the final
model's haystack scores, the fitted :class:`FoldAnchoredCut` - so a rung costs
no extra training and cannot move the fold models, the splits or the head.

The rules, in the deck's order:

* ``xcal_mincost`` - **iteration 1, as the "Grading Your Own Homework" slide
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
* ``anchored_rawmean`` - **iteration 4's strawman**, the ``b`` build of "The
  Rank & File": each fold's anchored mixture cut at its own midpoint, and the
  per-fold cuts averaged *as raw scores* on the final model.  The slide's point
  is that this is wrong - three models, three scales - and the shipped
  quantile transfer is the fix.

An unset rule (``None``) is the shipped fold-anchored cut, untouched, so the
harness's default arm cannot drift from the app through this module.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    *fitted_cut* is the shipped estimator's fit (``None`` on its blend fallback)
    and *shipped_threshold* / *shipped_provenance* are what it returned.

    Where a rule has nothing to cut on it falls back the way the app of its era
    did: ``xcal_mincost`` to the fold rule's sentinel (the too-few-labels
    ``0.5`` / ``NO_GOOD_THRESHOLD`` the fold computation has always returned),
    ``blend`` to blending ``NO_GOOD_THRESHOLD`` in place of the missing x-cal
    cut (production's substitution, see ``_blend_xcal_input``), and
    ``anchored_rawmean`` to the shipped estimator's own fallback, which is the
    blend.  The provenance names the fallback, so a reader can count how many
    steps a rung really cut.
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

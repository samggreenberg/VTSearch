"""The typed-query (text / cosine) sort's two lines (issues #3826 and #4136).

A typed-query sort carries a **display** line - what is painted green, drawn
under :data:`TEXT_SORT_CUT_RULE` (the guarded rule, or the count line at a
precision balance) - and an **acquisition** cut, the mixture midpoint
Autopilot's opening samples at whatever the display rule.
:func:`text_sort_cuts` draws both from one mixture fit.

Reads the mixture layers (:mod:`~vtscore.training.thresholds.gmm_core`,
:mod:`~vtscore.training.thresholds.gmm_em`,
:mod:`~vtscore.training.thresholds.gmm_threshold`); nothing in them reads this
module.  :data:`TEXT_SORT_CUT_RULE` and the other ``TEXT_SORT_*`` constants
are read at call time off this module, so this is where a test or a study
patches them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vtscore.training.thresholds.gmm_core import GmmFit1D
from vtscore.training.thresholds.gmm_em import _EM_TOL, _NO_ANCHORS, _anchored_em
from vtscore.training.thresholds.gmm_threshold import _fit_gmm_threshold_with_array, calculate_gmm_threshold

if TYPE_CHECKING:
    from vtscore.training.query_sort import SortLine


#: The two rules a typed-query sort's **display** line can be drawn with.
#:
#: * ``"guarded_tail"`` - :func:`guarded_text_sort_threshold`.  Keep the mixture
#:   (fitted to convergence) only when its two components are separated, and
#:   otherwise cut the upper tail of the one broad mode.  The shipped default
#:   since #4136.
#: * ``"gmm_midpoint"`` - :func:`calculate_gmm_threshold`, the midpoint of a
#:   two-Gaussian fit: the line every cosine sort drew before #3826, and still
#:   the line of an example or label-file sort.
#:
#: Which one :func:`text_sort_threshold` uses is :data:`TEXT_SORT_CUT_RULE`.
#: Neither rule moves the sort's *acquisition* cut
#: (:func:`text_sort_acquisition_threshold`): that is the midpoint under both.
TEXT_SORT_CUT_RULES = ("gmm_midpoint", "guarded_tail")

#: The rule a typed-query sort's display line is drawn with when
#: ``VTSEARCH_TEXT_SORT_CUT`` is unset or unrecognised.
#:
#: ``guarded_tail``, since #4136.  #3826 chose the guarded rule for the line
#: (F1 0.17 -> 0.39 on 1,120 labelled text sorts; the midpoint admitted a
#: median 43% of the haystack), but its trajectory A/B failed the pre-registered
#: ship rule because the line was also where Autopilot's Bad phase sampled:
#: at the guarded line that phase voted the top of the ranking, so the first
#: detectors saw only near-miss negatives (Δcost +0.016 ± 0.005 over 100
#: clicks, +0.071 at 6-20 votes; ``docs/experiments/2026-09-23-text-cut-ab-3826/REPORT.md``).
#: #4136 split the two jobs: the rule draws the display line only, and the
#: opening samples at the midpoint (:func:`text_sort_acquisition_threshold`)
#: whatever the rule, so its picks are the midpoint arm's by construction.  On
#: the F-beta objective the display line wins at beta 1/4 and 1 (+0.29 and
#: +0.22) and ties at beta 4 (``docs/experiments/2026-10-06-cost-fbeta-rescore-4582/REPORT.md``).
TEXT_SORT_CUT_DEFAULT = "guarded_tail"


def resolve_text_sort_cut_rule(value: str | None) -> str:
    """Normalise a ``VTSEARCH_TEXT_SORT_CUT`` value to a rule name; unknown or unset -> the default.

    An unrecognised value falls back to :data:`TEXT_SORT_CUT_DEFAULT` instead of
    raising, so a typo cannot take the sort route down.
    """
    rule = (value or "").strip().lower()
    return rule if rule in TEXT_SORT_CUT_RULES else TEXT_SORT_CUT_DEFAULT


#: The rule every typed-query sort draws its display line with, read once from
#: the ``VTSEARCH_TEXT_SORT_CUT`` environment variable.  ``gmm_midpoint`` is the
#: pre-#4136 line, kept as the A/B's control arm and as the opt-out.
TEXT_SORT_CUT_RULE = resolve_text_sort_cut_rule(os.environ.get("VTSEARCH_TEXT_SORT_CUT"))

#: Ashman's D at or above which the two fitted components count as separated
#: and the mixture's midpoint is kept.  2 is the textbook bimodality criterion
#: for a two-Gaussian mixture.  On 1,120 real text sorts, 9% cleared it, almost
#: all of them single-object images (``caltech101_m``, 60%).  Elsewhere the two
#: components are two halves of one mode (#3826).
TEXT_SORT_SEPARATION_D = 2.0

#: How many bulk sigmas above the median the tail cut sits.  Chosen as the
#: F1-best constant on three environments and tested on the fourth.  It was the
#: pick in 3 of 4 folds (3.5 in the other).  This is a tuned constant, not a
#: p-value: real negative tails are heavier than Gaussian, and at k = 3 a
#: Gaussian bulk under-predicts the negatives above the cut 2.7-6x.
TEXT_SORT_TAIL_K = 3.0

#: The converged fit's stopping rule, for the separated branch: the shipped
#: EM continued to 1e-10 in mean log-likelihood.  At the shipped 1e-3 the fit
#: still moves 5% of the haystack under a random start.  On separated data the
#: continuation is short.
_TEXT_SORT_CONVERGED_TOL = 1e-10
_TEXT_SORT_CONVERGED_MAX_ITER = 5000


def ashman_d(fit: GmmFit1D) -> float:
    """``|mu_hi - mu_lo| * sqrt(2 / (var_lo + var_hi))``: component separation in pooled sigmas."""
    pooled = fit.var_lo + fit.var_hi
    if not pooled > 0.0:
        return math.inf if fit.mu_hi != fit.mu_lo else 0.0
    return abs(fit.mu_hi - fit.mu_lo) * math.sqrt(2.0 / pooled)


def bulk_location_scale(scores: "list[float] | np.ndarray") -> tuple[float, float]:
    """The median of *scores*, and a sigma read from the half **below** it.

    The sigma is 1.4826 x the median distance of the lower half from the median,
    the MAD's Gaussian consistency factor applied to one side.  A typed query's
    matches are a right shoulder on one broad mode, so the lower half is the
    side they do not reach.  A heavy shoulder of positives therefore moves
    neither number much, which a symmetric MAD or a standard deviation would
    not survive.  Non-finite scores are ignored.  Returns ``(nan, nan)`` for an
    empty sample and a zero sigma for a constant one.
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mu = float(np.median(x))
    below = mu - x[x <= mu]
    return mu, 1.4826 * float(np.median(below))


def converge_score_gmm(arr: np.ndarray, start: GmmFit1D) -> GmmFit1D | None:
    """Continue the shipped unanchored EM from *start* until the likelihood stops moving.

    This is the "converged" fit of #3826: the same loop and the same objective as
    :func:`fit_score_gmm`, at :data:`_TEXT_SORT_CONVERGED_TOL`.  The stopping
    point is then no longer a parameter of the answer, and on separated data
    that is what makes the midpoint reproducible.  Returns ``None`` on
    numerical failure.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    fit = _anchored_em(
        x,
        _NO_ANCHORS,
        _NO_ANCHORS,
        start,
        1.0,
        _TEXT_SORT_CONVERGED_MAX_ITER,
        _EM_TOL,
        _TEXT_SORT_CONVERGED_TOL,
    )
    if fit is None:
        return None
    if fit.mu_hi < fit.mu_lo:
        return GmmFit1D(
            w_lo=fit.w_hi, mu_lo=fit.mu_hi, var_lo=fit.var_hi, w_hi=fit.w_lo, mu_hi=fit.mu_lo, var_hi=fit.var_lo
        )
    return fit


def _guarded_line_from_fit(
    scores: list[float], midpoint: float, arr: np.ndarray | None, fit: GmmFit1D | None
) -> tuple[float, str]:
    """The guarded line, given the shipped fit of *scores* and its midpoint.

    The body of :func:`guarded_text_sort_threshold`, split out so that
    :func:`text_sort_cuts` can draw both of a sort's lines from **one** EM fit:
    the acquisition cut is *midpoint* itself, and the guarded line reads the
    same *fit* for its separation test and continues it on the same *arr*.
    """
    if fit is None:
        return midpoint, "fallback"
    if ashman_d(fit) >= TEXT_SORT_SEPARATION_D:
        assert arr is not None
        converged = converge_score_gmm(arr, fit)
        return (fit if converged is None else converged).midpoint(), "gmm"
    mu, sigma = bulk_location_scale(scores)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0.0):
        return midpoint, "gmm"
    return mu + TEXT_SORT_TAIL_K * sigma, "tail"


def guarded_text_sort_threshold(scores: list[float]) -> tuple[float, str]:
    """The guarded line for a typed-query sort, and which branch drew it (issue #3826).

    A cosine sort of a typed query is usually **one** broad mode, with the
    query's matches as a thin right shoulder.  A two-Gaussian fit then splits
    the mode itself, and its midpoint lands in the densest region of the scores.
    On 1,120 real sorts the midpoint admitted a median 43% of the haystack, 12-54x
    the true matches outside single-object images.  That placement is also why
    a re-initialised or re-toleranced fit moved 6% of the verdicts: any
    perturbation of a line drawn through the mode moves thousands of medias.

    So:

    * fit the shipped mixture (:func:`fit_score_gmm`), and if its components are
      **separated** (:func:`ashman_d` >= :data:`TEXT_SORT_SEPARATION_D`),
      converge it (:func:`converge_score_gmm`) and cut at the midpoint.  Branch
      ``"gmm"``.  There the two components are real and the midpoint is well
      posed.
    * otherwise cut the tail: :func:`bulk_location_scale`'s median +
      :data:`TEXT_SORT_TAIL_K` sigmas.  Branch ``"tail"``.  The line admits
      what the bulk cannot explain.  It has no optimiser, so nothing but the
      data can move it.

    Measured against the midpoint on those 1,120 labelled sorts: F1 0.17 -> 0.39,
    and median admitted / true matches 17x -> 1.0x.  A bootstrap resample moves
    0.32% of the haystack instead of 1.2%.  The line is **worse** on the
    Inclusion-0 rate cost (FPR+FNR, +0.053), which prices a missed match at
    1/prevalence false alarms.  That trade was the decision #3826 asked for.
    See ``docs/experiments/2026-09-22-text-cut-3826/REPORT.md``.  Re-scored on
    the F-beta objective the line wins at beta 1/4 and 1 (+0.29 and +0.22) and
    ties at beta 4 (``docs/experiments/2026-10-06-cost-fbeta-rescore-4582/REPORT.md``).

    **This is a display line only.**  Autopilot's opening samples at the
    midpoint (:func:`text_sort_acquisition_threshold`), not here: with the Bad
    phase voting beside *this* line it voted the top of the ranking and the
    first detectors were worse at every beta (#4136).

    **Known failure: a majority-class query.**  When the matches are a large
    share of the haystack ("a person" in COCO is 54%) and the mixture is *not*
    separated, the median sits inside the matches, the tail is only the top of
    them, and the line admits a handful where the midpoint was roughly right.
    On the 28 measured sorts above 20% prevalence, F1 was 0.31 against the
    midpoint's 0.62.  Nothing label-free distinguishes that sort from a
    query the embedder barely separates, so this is documented and pinned by a
    test rather than guessed at.  A *separated* majority class takes the
    ``"gmm"`` branch and is unaffected.

    The tail cut can sit above every score, which paints nothing green.  That
    is "nothing here stands out", not an error.  It happened on 0.3% of the
    measured sorts.

    Fallbacks mirror :func:`calculate_gmm_threshold`: ``0.5`` with fewer than two
    scores (branch ``"fallback"``), and the shipped midpoint when the bulk has no
    spread to measure (a constant or two-valued sample, branch ``"gmm"``).
    """
    midpoint, arr, fit = _fit_gmm_threshold_with_array(scores)
    return _guarded_line_from_fit(scores, midpoint, arr, fit)


@dataclass(frozen=True)
class TextSortCuts:
    """The two lines a typed-query sort carries (issue #4136).

    One number used to do two jobs on a text sort, and the #3826 A/B found
    they want different rules: the line the user sees wants the guarded rule,
    and the position Autopilot's Bad phase samples at wants the midpoint.  So
    a text sort now carries both, the way a learned sort has carried
    ``threshold`` and ``acq_threshold`` since #2876.

    Attributes:
        threshold: The **display** line - what is painted green, what the
            above-threshold count and Find's verdicts read.  Drawn by
            :data:`TEXT_SORT_CUT_RULE`.
        acq_threshold: The **acquisition** cut - the rank position the Hard
            select (Autopilot's Bad phase) samples around.  The shipped
            midpoint, :func:`calculate_gmm_threshold`, under every rule.
        branch: Which branch drew ``threshold``: ``"gmm"``, ``"tail"`` or
            ``"fallback"`` under ``guarded_tail`` (see
            :func:`guarded_text_sort_threshold`), ``"count"`` when a balance
            at or below :data:`TEXT_SORT_COUNT_MAX_BETA` drew it
            (:func:`_count_line`), ``"midpoint"`` under ``gmm_midpoint``.
        line: How to redraw ``threshold`` at another balance without a
            re-score (#4760), when the sort that drew it kept one
            (:class:`~vtscore.training.query_sort.SortLine`); ``None``
            otherwise.  Left out of equality and ``repr``: it carries the
            sort's scores.
    """

    threshold: float
    acq_threshold: float
    branch: str
    line: SortLine | None = field(default=None, compare=False, repr=False)


#: The count line (issue #4603): at a balance of beta 1 or below, a typed-query
#: sort's display line keeps the top ``round(c(beta) * n_hat)`` of the sort, where
#: ``n_hat`` estimates how many of the scored media match.  Above beta 1 the
#: guarded line stays, which the pricing found as good as the count line at beta 4.
TEXT_SORT_COUNT_MAX_BETA = 1.0
#: ``n_hat`` counts the media above ``median + z * sigma`` beyond what a Gaussian
#: bulk puts there, with ``sigma = 1.4826 * MAD`` (the matches are a few percent of
#: a sort at most, so the median and MAD are the non-matches').  At z = 4 it lands
#: on the true count at COCO Better's 0.44% prevalence and errs high below that and
#: low above it, the directions that cost least at beta 1/4 and 1 (#4603).
TEXT_SORT_COUNT_Z = 4.0
#: c(beta) = beta ** TEXT_SORT_COUNT_EXPONENT: the multiple of the true count that
#: scored best at each preset, 3/8 at beta 1/4 and 1 at beta 1, at 0.1%, 0.44% and
#: 2% prevalence alike (#4603), joined by the power curve through the two.
TEXT_SORT_COUNT_EXPONENT = math.log(3 / 8) / math.log(1 / 4)
#: Below this many scores the bulk's spread is not worth estimating; the guarded
#: line stays.
TEXT_SORT_COUNT_MIN_SCORES = 50


def _count_line(scores: np.ndarray, beta: float, *, z: float | None = None, bulk: str = "gauss") -> float | None:
    """The count line at *beta* (#4603), or ``None`` when the bulk has no spread to measure.

    ``n_hat`` is the excess over a Gaussian bulk above ``median + z * sigma``
    (:data:`TEXT_SORT_COUNT_Z`), clamped to ``[1, n]`` (:func:`_count_estimate`); the
    line is the score of the ``round(c(beta) * n_hat)``-th highest, so it keeps that
    many (ties keep more, :func:`_count_keep_line`).  Priced against today's guarded
    line on COCO Better (SigLIP, 144 classes, 10 seeds): +0.13 to +0.30 F-beta at
    beta 1/4 and +0.01 to +0.13 at beta 1, at 0.1%, 0.44% and 2% prevalence.
    """
    n_hat = _count_estimate(scores, z=z, bulk=bulk)
    return None if n_hat is None else _count_keep_line(scores, beta, n_hat)


def _count_estimate(scores: np.ndarray, *, z: float | None = None, bulk: str = "gauss") -> float | None:
    """The count line's ``n_hat`` (#4603), clamped to ``[1, n]``; ``None`` below 50 scores or with no spread.

    *z* and *bulk* draw the other counts #4603 compared, for the centroid's
    candidates (#4759): another bar, and ``bulk="mirror"``, which takes the bulk's
    share above ``median + z * sigma`` to be its count below ``median - z * sigma``.
    """
    n = scores.size
    if n < TEXT_SORT_COUNT_MIN_SCORES:
        return None
    med = float(np.median(scores))
    sigma = 1.4826 * float(np.median(np.abs(scores - med)))
    if not sigma > 0.0:
        return None
    z = TEXT_SORT_COUNT_Z if z is None else float(z)
    above = float(np.count_nonzero(scores > med + z * sigma))
    if bulk == "gauss":
        bulk_above = n * 0.5 * math.erfc(z / math.sqrt(2.0))
    elif bulk == "mirror":
        bulk_above = float(np.count_nonzero(scores < med - z * sigma))
    else:
        raise ValueError(f"unknown bulk {bulk!r}; expected 'gauss' or 'mirror'")
    return min(max(above - bulk_above, 1.0), float(n))


def _count_keep_line(scores: np.ndarray, beta: float, n_hat: float) -> float:
    """The score of the ``round(c(beta) * n_hat)``-th highest of *scores*, at least the top one (#4603)."""
    n = scores.size
    keep = min(max(1, round(beta**TEXT_SORT_COUNT_EXPONENT * n_hat)), n)
    return float(np.partition(scores, n - keep)[n - keep])


def text_sort_cuts(scores: list[float], rule: str | None = None, beta: float | None = None) -> TextSortCuts:
    """Both lines of a typed-query sort, from one mixture fit (issue #4136).

    The single entry point for "where does a typed-query sort's green region
    end, and where does its opening vote".  The app's text route calls it
    through :func:`vtscore.training.query_sort.text_sort_active` and returns
    the pair as ``threshold`` / ``acq_threshold``; the eval harness's opening
    reads the acquisition half through :func:`text_sort_acquisition_threshold`.
    Either way the two cannot disagree about which line was drawn where.

    *rule* overrides :data:`TEXT_SORT_CUT_RULE` for the display line.  The
    acquisition cut is the midpoint whatever the rule - with ``gmm_midpoint``
    the two lines are one and the same number.

    *beta* is the balance the user set (F-beta's beta, #4413).  Under the guarded
    rule, a balance at or below :data:`TEXT_SORT_COUNT_MAX_BETA` draws the
    display line by count instead (:func:`_count_line`, #4603): the guarded line
    keeps about the same set at every balance, about four times the matches at
    0.44% prevalence, which a precision preset pays for.  ``None`` (a library
    caller with no balance) keeps the guarded line.  The acquisition cut never
    moves with *beta*, so a session's clicks are the same at any balance's line.
    """
    chosen = TEXT_SORT_CUT_RULE if rule is None else rule
    if chosen not in TEXT_SORT_CUT_RULES:
        raise ValueError(f"unknown text-sort cut rule {chosen!r}; expected one of {TEXT_SORT_CUT_RULES}")
    midpoint, arr, fit = _fit_gmm_threshold_with_array(scores)
    if chosen == "gmm_midpoint":
        return TextSortCuts(midpoint, midpoint, "midpoint")
    display, branch = _guarded_line_from_fit(scores, midpoint, arr, fit)
    if beta is not None and beta <= TEXT_SORT_COUNT_MAX_BETA:
        counted = _count_line(np.asarray(scores, dtype=np.float64), float(beta))
        if counted is not None:
            display, branch = counted, "count"
    return TextSortCuts(display, midpoint, branch)


def text_sort_threshold(scores: list[float], rule: str | None = None, beta: float | None = None) -> float:
    """The **display** line a typed-query sort draws, under *rule* (default :data:`TEXT_SORT_CUT_RULE`), at *beta*.

    :func:`text_sort_cuts`'s ``threshold`` alone: what the user sees.  Under
    ``gmm_midpoint`` it *is* :func:`calculate_gmm_threshold`.  It is not where
    Autopilot's opening samples - that is :func:`text_sort_acquisition_threshold`,
    and the two differ under the default rule (#4136).
    """
    return text_sort_cuts(scores, rule, beta).threshold


def text_sort_acquisition_threshold(scores: list[float]) -> float:
    """The cut Autopilot's opening samples against on a typed-query sort (issue #4136).

    The shipped midpoint, :func:`calculate_gmm_threshold`, under **every**
    display rule: the Hard select reads a cut as a rank position, and the
    #3826 trajectory A/B found that moving that position to the guarded line
    put the Bad phase inside the matches (median pick at the 2.5th percentile
    of the ranking against the 35th; 1.3 true matches voted per cell while
    hunting negatives, against 0.11), so the first detectors trained on
    near-miss negatives only.  The display line can move; this cannot, short
    of a new trajectory A/B.

    Bit-identical to :func:`text_sort_cuts`'s ``acq_threshold``.  The app's
    text route sends it as the response's ``acq_threshold``, and the eval
    harness's Bad phase and ``@mid`` schedule cut call it directly.
    """
    return calculate_gmm_threshold(scores)

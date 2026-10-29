"""What the balance says about a detector's line: the mixture's counts, the count rule and the line's state (#4413).

:func:`mixture_posterior` is the mixture read every count shares, fitted once
per ranking and memoised on it; :func:`mixture_positives`,
:func:`walk_positives`, :func:`fbeta_count` and :func:`acquisition_count` are
the counts it gives.  :func:`balance_count` and :func:`balance_line` are the
count rule a context with no trained head falls back to, and
:func:`balance_state` / :func:`gate_balance_state` the :class:`BalanceState`
every response that carries a line reports beside it.

:class:`~vtscore.training.thresholds.spot_check.SpotCheck` and
:class:`~vtscore.training.thresholds.spot_check.LineRanking` are read here as
types only: :mod:`~vtscore.training.thresholds.spot_check` re-exports every
name in this module, so importing it back at runtime would be a cycle.

Split out of :mod:`~vtscore.training.thresholds.spot_check` (#5425).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np

from vtscore.training.thresholds.check_schedule import (
    _EPS,
    BALANCE_CHECKED,
    BALANCE_GATE,
    BALANCE_UNCHECKED,
    CHECK_ADVISORY,
    CHECK_TRIM,
    CheckSchedule,
    balance_schedule,
    check_shape,
)
from vtscore.training.thresholds.knobs import ACQUISITION_ARGMAX_FACTOR
from vtscore.training.thresholds.ranges import LikelyRange

if TYPE_CHECKING:
    from vtscore.training.thresholds.spot_check import LineRanking, SpotCheck


# ------------------------------------------------------------ the mixture


#: A mixture component whose spread is under this share of the ranking's
#: score range has collapsed onto a few near-duplicate scores (#4419): its
#: posterior puts no mass on anything else, so the fit is no estimate and the
#: counts that read it keep the schedule's cap instead.
MIXTURE_MIN_STD_SHARE = 1e-2


def _mixture_is_sound(fit: Any, scores: np.ndarray) -> bool:
    """True when both components of *fit* spread over a real share of *scores*' range (#4419)."""
    if fit is None or not (fit.var_hi > 0 and fit.var_lo > 0):
        return False
    spread = float(scores.max() - scores.min()) if scores.size else 0.0
    if not spread > 0:
        return False
    least = MIXTURE_MIN_STD_SHARE * spread
    return math.sqrt(float(fit.var_hi)) >= least and math.sqrt(float(fit.var_lo)) >= least


def mixture_posterior(
    ranking: LineRanking | None,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> np.ndarray | None:
    """The mixture's high-component posterior at each unvoted item of *ranking*, best first; ``None`` when none fits.

    A 2-component mixture is fitted on the ranking's scores
    (:func:`~vtscore.training.thresholds.gmm.anchored_gmm_fit`, which
    subsamples a large ranking), anchored by the scores of the items in
    *labels* (``True`` = Good) that the ranking holds; this is the read every
    mixture-based count shares (#4389, #4413).  **Fitted once per ranking**,
    on the anchors of the first caller - the retrain's training labels, or
    the human votes for a ranking a cold Find built - and memoised on it, so
    the re-cut and every response that reports the line's state read the same
    fit; only the unvoted set the posterior is read over follows the live
    votes, as the ranking's own line does.  A retrain parks a new ranking,
    and with it a new fit on the votes it trained on.  A fit with a collapsed component
    (:data:`MIXTURE_MIN_STD_SHARE`, #4419) is no fit: on a tiny ranking the
    high component can land on the top two scores alone, and its posterior
    then says nothing about the rest.
    """
    if ranking is None or ranking.size == 0:
        return None
    excluded = ranking.voted.union(int(v) for v in also_voted)
    mask = np.fromiter((int(i) not in excluded for i in ranking.ids), dtype=bool, count=ranking.size)
    scores = ranking.scores[mask]  # already best first
    if scores.size == 0:
        return None
    from scipy.stats import norm  # noqa: PLC0415

    if not ranking.mixture:
        from vtscore.training.thresholds.gmm import anchored_gmm_fit  # noqa: PLC0415

        anchor_scores: list[float] = []
        anchor_labels: list[float] = []
        if labels:
            index = {int(cid): i for i, cid in enumerate(ranking.ids.tolist())}
            for cid, is_good in labels.items():
                row = index.get(int(cid))
                if row is not None:
                    anchor_scores.append(float(ranking.scores[row]))
                    anchor_labels.append(1.0 if is_good else 0.0)
        fit, _provenance = anchored_gmm_fit(ranking.scores, anchor_scores, anchor_labels)
        ranking.mixture.append(fit if _mixture_is_sound(fit, ranking.scores) else None)
    fit = ranking.mixture[0]
    if fit is None:
        return None
    hi = fit.w_hi * norm.pdf(scores, fit.mu_hi, math.sqrt(fit.var_hi))
    lo = fit.w_lo * norm.pdf(scores, fit.mu_lo, math.sqrt(fit.var_lo))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(hi / (hi + lo), nan=0.5)


def mixture_positives(
    ranking: LineRanking | None,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> float | None:
    """The mixture's count of positives among the unvoted items: the posterior's total (#4413).

    What a balance walk turns audited shares into recall with.  Over-counts
    on large sparse corpora (10-28x at 0.1%, #3827), which is why the
    unchecked balance line is capped and the walk audits.
    """
    posterior = mixture_posterior(ranking, labels, also_voted)
    return None if posterior is None else float(posterior.sum())


def walk_positives(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> float:
    """The positives a balance walk reads recall against: the mixture's count, else the balance's cap (#4419).

    The mixture (:func:`mixture_positives`) is the estimate when it has one.
    When it does not - nothing fits, or the fit collapsed - the walk still
    has to start, so the denominator is the set the unchecked line would keep
    anyway: the balance's cap (:func:`balance_schedule`), lowered to what is
    unvoted.  Against that denominator the walk goes deeper while the audited
    share right holds up.  A sound fit that counts fewer than one positive (every positive
    already voted Good, say) takes the cap too: the walk's recall is its
    audited positives over this count, so a near-zero count would report
    every check as having found them all.  Always at least one.  The app's
    check route and the harness's end-of-run check both read this, so a
    check starts in the same cases.
    """
    n_pos = mixture_positives(ranking, labels, also_voted)
    if n_pos is not None and n_pos >= 1.0:
        return n_pos
    cap = balance_schedule(beta).candidate
    unvoted = ranking.unvoted_ids(also_voted).size if ranking is not None else 0
    return float(max(1, min(cap, unvoted) if unvoted else cap))


def acquisition_count(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
    factor: float | None = ACQUISITION_ARGMAX_FACTOR,
) -> int | None:
    """How deep Autopilot's acquisition cut sits under a balance: *factor* of the F-beta argmax's depth (#4409).

    The argmax is :func:`fbeta_count`'s, uncapped; the count is at least one.
    ``None`` with no *factor* (the shipped value since the #4427 revert: the
    caller keeps the line - 4 cut), no ranking or no mixture estimate.
    :data:`~vtscore.training.thresholds.knobs.ACQUISITION_ARGMAX_FACTOR` is
    what ships; the harness's ``acq_p_crossing`` arm passes a number.
    """
    if factor is None:
        return None
    k = fbeta_count(ranking, beta, labels, also_voted)
    return None if k is None else max(1, round(float(factor) * k))


def acquisition_threshold(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
    factor: float | None = ACQUISITION_ARGMAX_FACTOR,
) -> float | None:
    """The score the acquisition cut sits at under a balance (#4409): the last unvoted item :func:`acquisition_count` keeps."""
    k = acquisition_count(ranking, beta, labels, also_voted, factor)
    if k is None or ranking is None:
        return None
    cut = ranking.threshold_for(k, also_voted)
    return None if cut is None or not math.isfinite(float(cut)) else float(cut)


def fbeta_count(
    ranking: LineRanking | None,
    beta: float,
    labels: Mapping[int, bool],
    also_voted: Iterable[int] = (),
) -> int | None:
    """How many unvoted items the mixture says maximise F-*beta*: the ``fb-gmm`` rule of #4411.

    The posterior's cumulative sum is the positives in the top *k*, its total
    the corpus's; the count is the argmax of the F-beta those give.  ``None``
    when nothing fits.  Right-sized at 5% and on small corpora, 2-4x too deep
    on large sparse ones, so the unchecked line caps it (:func:`balance_count`).
    """
    posterior = mixture_posterior(ranking, labels, also_voted)
    if posterior is None:
        return None
    tp = np.cumsum(posterior)
    b2 = float(beta) * float(beta)
    est = (1.0 + b2) * tp / (b2 * float(tp[-1]) + np.arange(1, posterior.size + 1))
    return int(np.argmax(est)) + 1


# ------------------------------------------------------------------- the line


def aim_words(state: Mapping[str, Any]) -> str:
    """The balance a line's state was drawn at, in words for a log line (#4413): ``at F1``.

    What a headless run says beside the unchecked set it exports.
    """
    return f"at F{state['beta']:g}"


@dataclass(frozen=True)
class BalanceState:
    """What the balance says about a detector's line: its state, the set's size and what the check estimated (#4413).

    The wire shape every response that carries a line carries beside its
    ``threshold`` (:func:`vtscore.state.core.detector_balance_state`):
    ``status`` is ``unchecked``, ``checked``, or ``gate`` on a structural
    detector's line (:func:`gate_balance_state`); ``precision``
    and ``recall`` are the likely ranges of the set the line kept when the
    check started (``kept_at_check``, #4483), or of the audited set where the
    check recorded no count; ``fbeta`` is the walk's estimate for its own set.
    Under the ``advisory`` shape (the app's, :func:`check_shape`) the audited
    set is the walk's end and the kept ``count`` is the line's - what a drawn
    line keeps (the labels' line, #4452), else the unchecked count rule's;
    under ``trim`` they coincide.
    """

    beta: float
    status: str
    count: int
    precision: LikelyRange | None
    recall: LikelyRange | None
    fbeta: float | None
    stale: bool
    schedule: CheckSchedule
    #: How a check treats the line at this beta (:func:`check_shape`):
    #: ``advisory``, the app's at every preset (the ranges inform, the count
    #: stays the line's), or ``trim``, an eval arm (the walk's end is the count).
    shape: str = CHECK_TRIM
    #: The set the last check audited (the walk's end), which under ``advisory``
    #: is not the set the line keeps; ``None`` while unchecked.
    audited: int | None = None
    #: Whether a check has anything to walk (#4489): an unvoted item in the
    #: ranking.  ``False`` with no ranking - a structural detector, whose line
    #: is the verification gate's boundary rather than a cut on a ranking, or
    #: one not trained on this corpus yet - or with every item in it voted.
    #: A check cannot start on such a line, so a client offers none.
    checkable: bool = True
    #: How many unvoted items the line kept when the last check started: the
    #: set ``precision`` and ``recall`` describe (#4483).  ``None`` while
    #: unchecked, and where the ranges describe the audited set instead.
    kept_at_check: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "status": self.status,
            "count": self.count,
            "precision": None if self.precision is None else {**self.precision.as_dict(), "stale": self.stale},
            "recall": None if self.recall is None else {**self.recall.as_dict(), "stale": self.stale},
            "fbeta": None if self.fbeta is None else round(self.fbeta, 4),
            "schedule": self.schedule.as_dict(),
            "shape": self.shape,
            "audited": self.audited,
            "checkable": self.checkable,
            "kept_at_check": self.kept_at_check,
        }


def applicable_balance(beta: float, result: SpotCheck | None) -> SpotCheck | None:
    """*result* when it is a finished walk at *beta*; ``None`` otherwise.

    A result is about the balance it was run at: a different balance is
    unchecked until it is checked itself (the old result waits, and shows
    again if the balance moves back).
    """
    if result is None or not result.finished:
        return None
    if abs(result.beta - float(beta)) > _EPS:
        return None
    return result


def balance_count(beta: float, result: SpotCheck | None, proposal: int | None = None, shape: str | None = None) -> int:
    """The count the line keeps at *beta*: the finished walk's end where the check's shape lets it move the line, else the unchecked count.

    The unchecked count is the balance's cap (:func:`balance_schedule`),
    lowered to *proposal* when the caller has one: the mixture's F-beta argmax
    (:func:`fbeta_count`).  Under the ``advisory`` shape (:func:`check_shape`,
    every preset; #4452) a finished walk informs the line and never moves it;
    *shape* overrides the preset's (the harness's ``walk_shape`` arms).
    """
    applicable = applicable_balance(beta, result)
    if applicable is not None and (shape or check_shape(beta)) != CHECK_ADVISORY:
        return applicable.k
    count = balance_schedule(beta).candidate
    if proposal is not None:
        count = max(1, min(count, int(proposal)))
    return count


def balance_line(
    ranking: LineRanking | None,
    beta: float,
    result: SpotCheck | None = None,
    also_voted: Iterable[int] = (),
    proposal: int | None = None,
    shape: str | None = None,
) -> float | None:
    """The threshold the line sits at under a balance: the last item of the set it keeps (``None``: no ranking)."""
    if ranking is None:
        return None
    return ranking.threshold_for(balance_count(beta, result, proposal, shape), also_voted)


def balance_state(
    beta: float,
    result: SpotCheck | None = None,
    ranking: LineRanking | None = None,
    also_voted: Iterable[int] = (),
    proposal: int | None = None,
    shape: str | None = None,
    threshold: float | None = None,
) -> BalanceState:
    """The balance's state at *beta*, given the detector's last finished walk, its ranking and the mixture's count.

    With *threshold* (the detector's line: the labels' line, #4452, a
    centroid or typed-query head's line, or a trained head's fallback cut) the
    count is how many unvoted items of *ranking* score at or above it - the set
    the line keeps there, which may be none - rather than the count rule's.

    *shape* overrides the preset's check shape (:func:`check_shape`); the
    harness's full-walk arm passes ``trim`` so the walk's end is the count.
    """
    schedule = balance_schedule(beta)
    shape = shape or check_shape(beta)
    applicable = applicable_balance(beta, result)
    count = balance_count(beta, result, proposal, shape)
    also_voted = frozenset(int(v) for v in also_voted)
    checkable = ranking is not None and ranking.unvoted_ids(also_voted).size > 0
    if threshold is not None and ranking is not None:
        excluded = ranking.voted.union(int(v) for v in also_voted)
        mask = np.fromiter((int(i) not in excluded for i in ranking.ids), dtype=bool, count=ranking.size)
        count = int((ranking.scores[mask] >= float(threshold)).sum())
    elif ranking is not None and (applicable is None or shape == CHECK_ADVISORY):
        # The unchecked rule's count, capped by what is unvoted; a walk's end
        # under ``trim`` is a band edge of the ranking it walked, kept as is.
        count = min(count, len(ranking.candidate(count, also_voted)))
    if applicable is None:
        return BalanceState(
            float(beta), BALANCE_UNCHECKED, count, None, None, None, False, schedule, shape, None, checkable
        )
    stale = ranking is not None and applicable.is_stale(ranking, also_voted)
    # The ranges describe the set the line kept when the check started (#4483),
    # or the walk's own set where the check recorded no count.
    top = applicable.range_count
    return BalanceState(
        float(beta),
        BALANCE_CHECKED,
        count,
        applicable.range_over(top),
        applicable.recall_range_over(top),
        applicable.fbeta_estimate(),
        stale,
        schedule,
        shape,
        applicable.k,
        checkable,
        applicable.kept,
    )


def gate_balance_state(beta: float, passed: Iterable[int], also_voted: Iterable[int] = ()) -> BalanceState:
    """The balance's state on a line the structural verification gate draws (#4505).

    On a structural dataset the line is the gate's boundary, not a cut on a
    ranking (:func:`~vtscore.training.structural_similarity.maybe_structural_rerank`
    drops the ranking), so the count rule's 32 says nothing about it.  The
    count is what the gate keeps: the *passed* items (those at or above its
    threshold) that are not in *also_voted*.  ``status`` is ``gate``: no check
    applies, so it carries no ranges, and ``checkable`` is ``False``.
    """
    voted = frozenset(int(v) for v in also_voted)
    count = sum(1 for i in passed if int(i) not in voted)
    return BalanceState(
        float(beta),
        BALANCE_GATE,
        count,
        None,
        None,
        None,
        False,
        balance_schedule(beta),
        check_shape(beta),
        None,
        False,
    )

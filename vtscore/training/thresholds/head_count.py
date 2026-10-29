"""The head's count line (#4786) and the cues Autopilot reads beside the labels line.

While the class model holds too few Goods to cut on (:func:`head_count_serves`),
the line at every balance keeps what stands out in the head's own scores instead
(:func:`head_count_cut`; #4798, #5039, #5084, #5215).  Beside the line sit the
rules Autopilot and the eval harness's default arm read off it: the stop level
(:func:`stop_level_met`, #4878) and when a spot check is due
(:func:`weak_check_due`, :func:`done_check_due`, #4496).  Reads
:mod:`~vtscore.training.thresholds.class_model`; of the line's modules only
:class:`~vtscore.training.thresholds.labels_line.LabelsLine` reads the count line.
The eval harness swaps the ``HEAD_COUNT_*`` knobs here, where they are read
(:mod:`vtscore.eval.live_threshold_rules`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from vtscore.training.thresholds.class_model import ClassScoreModel, _logit

if TYPE_CHECKING:
    from vtscore.training.thresholds.labels_line import LabelsLine

#: Labels whose Good and Bad scores sit fewer spreads apart than this separate
#: weakly (:attr:`~vtscore.training.thresholds.labels_line.LabelsLine.separation`,
#: #4466, #4496).  Below it a session returns too much: over 200 withheld images
#: in 37-77% of such sessions from click 5 on, against 2-18% above it, and no cut
#: rescues its F.
WEAK_SEPARATION_D = 1.5
#: The fewest votes before weak separation makes a spot check due (#4496): the
#: first few labels separate noisily, and a check needs a line to walk.
WEAK_CHECK_MIN_VOTES = 10
#: Votes after a check ends before the check before Done runs (:func:`done_check_due`, #4496): a check that ended
#: moments ago has already measured this line.  It was also the weak check's repeat until #4909.
WEAK_CHECK_COOLDOWN = 25
#: Votes after a check ends before weak separation makes another due; ``None``: never, so the weak check prompts at
#: most once a session, and only before any other check has ended (#4909, the owner's choice of 2026-10-15).  #4496
#: had chosen again after 25 votes, on an app where a check's picks were worth far more than Autopilot's.  At
#: today's head prompting once matched or beat it: +0.0006 (beta 1) and +0.0013 (beta 4) in mean F-beta over clicks
#: 0-200, level after the closing check, Done at the same click and F-beta, and 24-25 fewer of 200 clicks spent
#: checking (``docs/experiments/2026-10-15-weak-repeat-4909/REPORT.md``).
WEAK_CHECK_REPEAT: int | None = None
#: Media types whose sessions never get a prompted weak check (#4778, #4799).  A
#: face search's opening finds the person's few photos before the check could
#: start (a median of 4 Goods at vote 11), so a check's votes find nothing and
#: only pick which Bads get labelled, worse than acquisition does: without it
#: the face review's sessions gained +0.007 to +0.018 over votes 1-50, with the
#: objective after the end-of-run check and final AP unchanged.  On COCO the
#: check pays most where it starts with fewest Goods, so a Good count cannot
#: stand in for the media type.  Autopilot's check before Done stays (:func:`done_check_due`).
WEAK_CHECK_OFF_MEDIA_TYPES: frozenset[str] = frozenset({"face"})
#: The Bads half of the thin-model gate: while the class model holds fewer than this many Bads
#: (its held-out ones, about 0.6 of the Bads in hand) and fewer than :data:`HEAD_COUNT_GOODS`
#: Goods, the line at every balance keeps the head's count (:func:`head_count_cut`) rather than
#: the class model's cut.  Unbounded since #4798, so the Goods alone close the gate; #4786
#: shipped 32.  ``0`` is the line before #4786.
HEAD_COUNT_BADS: float = math.inf
#: The thin-model gate (#4786, #4798): with this many held-out Goods the class
#: model's Good component stands on its own, and a broad target's labels line keeps what its
#: many positives need where the head's count, a count of what stands out, keeps several
#: times too many.
HEAD_COUNT_GOODS = 6
#: How the head's count grows above a balance of 1 (#5039): ``beta ** HEAD_COUNT_RECALL_EXPONENT``
#: times what stands out, in place of #4603's ``beta ** 0.708`` (:func:`head_count_cut`): 1.3x at beta 4
#: (``log 1.3 / log 4``), what faces and Region Photo's heads want there.  ``None`` keeps #4603's, the
#: count before #5039.
HEAD_COUNT_RECALL_EXPONENT: float | None = 0.18925
#: Where the head's count line reaches down at a balance above 1 (#5039): when a held-out Good ranks
#: below the count of what stands out, the line keeps at least as deep as the head's excess curve's
#: expected F-beta peaks, read from this many bulk spreads up (:func:`_excess_depth`).  Closed loop at
#: beta 4 with the exponent above: Find's mean F-beta +0.0035 over a Binary Photo session's 200 clicks
#: and +0.015 to +0.017 over a face session's 150 votes; 3 spreads gave faces more (+0.022 to +0.023)
#: and Binary half (``docs/experiments/2026-10-18-head-count-5039``).  ``None`` never reaches, the line
#: before #5039.
HEAD_COUNT_RECALL_Z: float | None = 2.75
#: How the head's count scales at a balance of 1 and below (#5084): ``HEAD_COUNT_PRECISION_SCALE * beta **
#: HEAD_COUNT_PRECISION_EXPONENT`` times what stands out, in place of #4603's ``beta ** 0.708``
#: (:func:`head_count_cut`).  The count of what stands out reads the bulk's heavy upper tail as positives on a
#: weak head (Region Photo's keeps 2.6x its positives at beta 1), so a smaller multiplier keeps nearer the
#: positives.  0.71 beta (0.18x the count at beta 1/4 where #4603 kept 0.375x, 0.71x at beta 1): offline on the
#: head's count line Region Photo gains +0.007 to +0.084 F-beta at every prevalence, Binary Photo +0.007 to +0.037
#: at 0.44% and 0.1% and -0.014 / -0.004 at 1%, faces at most 0.008 either way; closed loop, Find's mean F-beta
#: over clicks 0-50 +0.0093 / +0.021 on Region Photo (beta 1 / 1/4) and +0.0030 / +0.0014 on Binary Photo
#: (``docs/experiments/2026-10-24-head-bulk-5084``).  ``None`` is #4603's piece: a scale of 1, an exponent of
#: 0.708.
HEAD_COUNT_PRECISION_SCALE: float | None = 0.71
HEAD_COUNT_PRECISION_EXPONENT: float | None = 1.0
#: Where the head's count keeps #4603's ``beta ** 0.708`` at a balance of 1 and below in place of the multiplier above
#: (#5215): when the corpus's top logit stands fewer than this many bulk spreads (median, ``1.4826`` x the MAD) above
#: the bulk, the positives tend to sit under the bulk's heavy tail, where a smaller count only drops them.  Offline at
#: 6.1 on the head's count line: Binary Photo at 1% +0.012 F-beta (beta 1/4) and +0.005 (beta 1) over 0.71 beta, and
#: -0.005 at 0.1% (beta 1/4); Region Photo within 0.003; faces -0.005 at 0.1% (beta 1)
#: (``docs/experiments/2026-10-25-head-tail-5215``).  ``None`` never gates: the multiplier above on every head.
HEAD_COUNT_PRECISION_TOP_Z: float | None = None
#: The level Autopilot's Done also reads (#4878): Done once every quality light is green, or once
#: the labels line expects at least this F-beta at the session's balance over the corpus it was
#: cut on (:func:`stop_level_met`).  Priced on 2,160 sessions and A/B'd in the phase machine:
#: no change to the session, and the stop comes where the clicks pay
#: (``docs/experiments/2026-10-14-stop-signal-4878``).
STOP_LEVEL = 0.5


def stop_level_reading(line: "LabelsLine | None", beta: float | None, level: float = STOP_LEVEL) -> dict[str, Any]:
    """What Autopilot's stop reads off the labels line (#4878): ``expected_fbeta``, ``level`` and ``met``.

    ``expected_fbeta`` is
    :meth:`~vtscore.training.thresholds.labels_line.LabelsLine.expected_fbeta` at *beta*, the
    line's own estimate of the F-beta of the set it keeps; ``None`` with no
    line, no balance, or while the head's count line serves
    (:func:`head_count_serves`), whose class model is too thin to read.  ``met``
    is whether it reaches *level*.
    """
    if line is None or beta is None or head_count_serves(line.model):
        return {"expected_fbeta": None, "level": float(level), "met": False}
    expected = float(line.expected_fbeta(float(beta)))
    return {"expected_fbeta": expected, "level": float(level), "met": expected >= level}


def stop_level_met(line: "LabelsLine | None", beta: float | None, level: float = STOP_LEVEL) -> bool:
    """Whether the labels line expects at least *level* F-beta at *beta* (:func:`stop_level_reading`)."""
    return bool(stop_level_reading(line, beta, level)["met"])


def head_count_serves(model: ClassScoreModel) -> bool:
    """Whether the line is the head's count line (#4786): a class model with too few Goods to cut on (#4798)."""
    return model.n_neg < HEAD_COUNT_BADS and model.n_pos < HEAD_COUNT_GOODS


def head_count_cut(scores_desc: Any, beta: float, held_out_goods: Any = None) -> float:
    """The cut that keeps what stands out in the head's own scores at *beta* (#4786).

    *scores_desc* are a corpus's unvoted scores, best first.  On their logits the
    rule that draws the Goods' centroid's line
    (:data:`~vtscore.detectors.centroid_head.CENTROID_LINE_RULE`: ``beta ** 0.708``
    times the larger of the Gaussian bulk's excess and the top's widest drop, never
    fewer than that drop) counts what stands out; the cut keeps exactly those items,
    placed between the last kept and the first dropped as
    :func:`~vtscore.training.thresholds.corpus_model.corpus_cut` places its own.  An
    empty corpus keeps nothing.

    At a balance of 1 and below (#5084) the multiplier is
    ``HEAD_COUNT_PRECISION_SCALE * beta ** HEAD_COUNT_PRECISION_EXPONENT`` (0.71 beta) in
    place of ``beta ** 0.708`` (:func:`_scaled_count`); both at ``None`` is the rule's own.  Where the
    top logit stands under :data:`HEAD_COUNT_PRECISION_TOP_Z` bulk spreads (#5215) the rule's own
    ``beta ** 0.708`` keeps instead.

    Above a balance of 1 (#5039) the count grows as ``beta **``
    :data:`HEAD_COUNT_RECALL_EXPONENT` instead, and where one of
    *held_out_goods* (the class model's held-out Goods, logits) ranks below the
    count at a balance of 1 - a positive the count of what stands out misses - the
    line keeps at least as deep as the head's excess curve's expected F-beta
    peaks (:func:`_excess_depth`, from :data:`HEAD_COUNT_RECALL_Z` bulk spreads
    up).  Either constant at ``None`` is that piece before #5039.
    """
    from vtscore.detectors.centroid_head import centroid_cut  # noqa: PLC0415
    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT  # noqa: PLC0415

    s = np.asarray(scores_desc, dtype=np.float64)
    if s.size == 0:
        return 1.0 + 1e-9
    x = _logit(s)
    b = float(beta)
    above = b > 1.0
    # The count's own beta: the one whose #4603 multiplier is beta ** HEAD_COUNT_RECALL_EXPONENT.
    b_count = (
        b ** (HEAD_COUNT_RECALL_EXPONENT / TEXT_SORT_COUNT_EXPONENT)
        if above and HEAD_COUNT_RECALL_EXPONENT is not None
        else b
    )
    if (
        not above
        and (HEAD_COUNT_PRECISION_SCALE is not None or HEAD_COUNT_PRECISION_EXPONENT is not None)
        and not _top_under(x, HEAD_COUNT_PRECISION_TOP_Z)
    ):
        k = _scaled_count(x, b)
    else:
        k = int(np.count_nonzero(x >= centroid_cut(x, beta=b_count)))
    goods = None if held_out_goods is None else np.asarray(held_out_goods, dtype=np.float64)
    if above and HEAD_COUNT_RECALL_Z is not None and goods is not None and goods.size:
        standouts = int(np.count_nonzero(x >= centroid_cut(x, beta=1.0)))
        if int(np.count_nonzero(x >= float(goods.min()))) > standouts:
            k = max(k, _excess_depth(x, b, float(HEAD_COUNT_RECALL_Z)))
    if k <= 0:
        return 1.0 + 1e-9
    return float(s[k - 1]) if k >= s.size else float((s[k - 1] + s[k]) / 2.0)


def _top_under(x: np.ndarray, top_z: float | None) -> bool:
    """Whether the top of the logits *x* stands fewer than *top_z* bulk spreads above their median (#5215).

    The spread is ``1.4826`` x the MAD, as #4603's count reads the bulk; ``None``, or a bulk with no
    spread, never is.
    """
    if top_z is None or x.size == 0:
        return False
    med = float(np.median(x))
    sd = 1.4826 * float(np.median(np.abs(x - med)))
    return sd > 0 and (float(x.max()) - med) / sd < float(top_z)


def _scaled_count(x: np.ndarray, beta: float) -> int:
    """How many of the logits *x* the head's count keeps at *beta* <= 1 under #5084's multiplier.

    :func:`~vtscore.detectors.centroid_head.centroid_cut`'s rule
    (:data:`~vtscore.detectors.centroid_head.CENTROID_LINE_RULE`) with the multiplier
    ``HEAD_COUNT_PRECISION_SCALE * beta ** HEAD_COUNT_PRECISION_EXPONENT`` in place of
    ``beta ** 0.708``: that many times the larger of #4603's count and the top's widest drop
    (from beta 1, the drop below a standout first, #4783), never fewer than that drop.  With
    too few scores to count, the guarded line the rule falls back to.
    """
    from vtscore.detectors.centroid_head import _top_gap_count, centroid_cut  # noqa: PLC0415
    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_EXPONENT, _count_estimate  # noqa: PLC0415

    counted = _count_estimate(x)
    if counted is None:
        return int(np.count_nonzero(x >= centroid_cut(x, beta=beta)))
    scale = 1.0 if HEAD_COUNT_PRECISION_SCALE is None else float(HEAD_COUNT_PRECISION_SCALE)
    exponent = (
        TEXT_SORT_COUNT_EXPONENT if HEAD_COUNT_PRECISION_EXPONENT is None else float(HEAD_COUNT_PRECISION_EXPONENT)
    )
    gap = _top_gap_count(x, drop="skip25" if beta >= 1.0 else "widest")
    keep = min(max(1, round(scale * beta**exponent * max(counted, float(gap)))), x.size)
    srt = -np.sort(-x)
    return int(np.count_nonzero(x >= min(float(srt[keep - 1]), float(srt[gap - 1]))))


def _excess_depth(x_desc: np.ndarray, beta: float, z_min: float) -> int:
    """How many of the best logits *x_desc* keep the head's excess curve at its expected F-*beta* peak (#5039).

    The excess at the *k*-th best is *k* less what a Gaussian bulk puts above its
    score, the bulk's centre and spread the median and ``1.4826`` x the MAD of
    *x_desc*, as #4603's count reads them; the positives' total is the largest
    excess from *z_min* bulk spreads up.  The cut at *k* is expected to score
    ``(1 + beta**2) * excess_k / (beta**2 * total + k)``, the excess held to
    ``[0, total]``; the depth is where that peaks among the cuts at or above
    *z_min*.  ``0`` below #4603's fewest scores, with no spread, or with no score
    that high.
    """
    from scipy.special import ndtr  # noqa: PLC0415

    from vtscore.training.thresholds.gmm import TEXT_SORT_COUNT_MIN_SCORES  # noqa: PLC0415

    n = int(x_desc.size)
    if n < TEXT_SORT_COUNT_MIN_SCORES:
        return 0
    med = float(np.median(x_desc))
    sigma = 1.4826 * float(np.median(np.abs(x_desc - med)))
    if not sigma > 0.0:
        return 0
    z = (x_desc - med) / sigma
    up = z >= z_min
    if not up.any():
        return 0
    k = np.arange(1, n + 1, dtype=np.float64)
    excess = k - n * ndtr(-z)
    total = max(float(excess[up].max()), 1.0)
    b2 = beta * beta
    f = np.where(up, (1.0 + b2) * np.clip(excess, 0.0, total) / (b2 * total + k), -1.0)
    return int(np.argmax(f)) + 1


def weak_check_prompts(media_type: str | None) -> bool:
    """Whether weak separation may prompt a spot check on *media_type*'s detectors (#4799).

    False on :data:`WEAK_CHECK_OFF_MEDIA_TYPES` (face crops); true elsewhere,
    including an unknown type.  The app's :func:`vtscore.state.core.detector_check_prompt`
    and the eval harness's default ``spot_check`` both read it, so the default
    arm stays the app.
    """
    return media_type not in WEAK_CHECK_OFF_MEDIA_TYPES


def weak_check_due(
    separation: float | None,
    n_votes: int,
    votes_at_last_check: int | None,
    *,
    threshold: float = WEAK_SEPARATION_D,
    min_votes: int = WEAK_CHECK_MIN_VOTES,
    cooldown: int | None = WEAK_CHECK_REPEAT,
) -> bool:
    """Whether the labels separate weakly enough that a spot check is due (#4496, the owner's ruling of 2026-10-05).

    Due when the labels line's
    :attr:`~vtscore.training.thresholds.labels_line.LabelsLine.separation` is below
    *threshold*, at least *min_votes* votes have been cast, and either no
    check has ended yet or *cooldown* votes have been cast since the last one
    ended (*votes_at_last_check*, the vote count when it ended; ``None``
    before any).  ``cooldown=None``, the default (:data:`WEAK_CHECK_REPEAT`,
    #4909), makes a check due once at most: never after any check has ended.

    The app's Autopilot runs the check when it is due; the eval harness's
    default arm (``spot_check="weak"``) runs it where this says, so both read
    this rule.
    A weakly separated session's uniform picks within the line's bands are
    better training votes than the acquisition picks they replace: priced at
    equal clicks, +0.008 to +0.033 at click 20 and +0.023 to +0.049 at 50
    over the three presets
    (``docs/experiments/2026-10-05-weak-check-4496/REPORT.md``).
    """
    if separation is None or not separation < threshold or n_votes < min_votes:
        return False
    if votes_at_last_check is None:
        return True
    return cooldown is not None and n_votes - votes_at_last_check >= cooldown


def done_check_due(n_votes: int, votes_at_last_check: int | None, *, cooldown: int = WEAK_CHECK_COOLDOWN) -> bool:
    """Whether Autopilot's spot check before Done is due: no check has ended in the last *cooldown* votes.

    Autopilot checks the line once a run, the first time every quality light is
    green, and only then reports Done.  The check is skipped when one ended
    fewer than *cooldown* votes ago (*votes_at_last_check*, the vote count when
    it ended; ``None`` before any): a weak-separation check just run, or a
    session re-entered right after its own check, has already measured this
    line.  Unlike :func:`weak_check_due` it reads no separation and runs on
    every media type, face crops included.

    The app's Autopilot reads it (``done_check_due`` beside ``check_due``), and
    so does the eval harness's default arm.
    """
    return votes_at_last_check is None or n_votes - votes_at_last_check >= cooldown

"""The Find Stats precision curve: how right the returned set is against how much is returned.

Two readings of one ranking, sampled at a few dozen return counts *k* (the top
*k* of the Find corpus by frozen score):

* **Verified precision** - of the items in the top *k* the user checked by hand,
  the share they kept Good.  Measured, not estimated, but only over what was
  checked: the boundary walk surfaces items near the line, so it is not a
  random sample of the top *k*.
* **Estimated precision** - the lower-bound curve the precision floor reads its
  cut off (:func:`~vtscore.training.thresholds.precision_lower_bound_curve`,
  #4220's estimator), from the detector's calibration folds.  Gated exactly as
  the floor gates it (:data:`~vtscore.training.thresholds.MIN_CALIBRATION_POSITIVES`):
  below the gate the estimate breaks most of its promises, so none is drawn.

The evidence is the precision floor's own (#4245): the
:class:`~vtscore.training.thresholds.PrecisionFloorEstimate` the last training
parked on the detector context (``precision_floor_cache``) - the held-out votes
the learned sort chose, each fold's haystack, and the whole haystack (voted
items included) as the reference pool, the configuration the floor's safety
rests on (#4221).  The chart therefore draws the curve the floor cuts, not a
more optimistic cousin of it.  The corpus is the Find run's frozen scores.
Pure read; nothing is cached or persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from vtscore.state.core import DetectorContext

#: Return counts the curve is sampled at, log-spaced from 1 to the corpus size.
PRECISION_CURVE_POINTS = 40

#: Corpus size above which the estimate is read off a uniform sample.  The
#: estimator is linear in the corpus per bootstrap refit (~0.3 s at this size).
PRECISION_CURVE_MAX_CORPUS = 50_000

#: Seed of that sample, so one Find run always draws one curve.
PRECISION_CURVE_SAMPLE_SEED = 4242

#: ``estimate_status`` values: the curve was drawn; the calibration votes hold
#: too few positives (the floor's own gate); or there is no fold evidence at all
#: (too few votes to hold any out, or a detector trained without the estimator).
ESTIMATE_ESTIMATED = "estimated"
ESTIMATE_INSUFFICIENT = "insufficient_evidence"
ESTIMATE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PrecisionEstimate:
    """The estimated curve at the requested counts, and why it may be missing."""

    status: str
    calibration_positives: int
    values: list[float | None]


def curve_counts(n_scored: int, extra: int | None = None, n_points: int = PRECISION_CURVE_POINTS) -> list[int]:
    """Distinct return counts in ``[1, n_scored]``, log-spaced, plus *extra* when it is in range.

    Log spacing keeps the top of the ranking, where a precision curve moves,
    from collapsing into one point on a large corpus.  *extra* is the current
    cut's count, so the curve has a point exactly at the marker.
    """
    if n_scored <= 0:
        return []
    counts = {int(round(k)) for k in np.geomspace(1, n_scored, num=min(n_scored, n_points))}
    if extra is not None and 1 <= extra <= n_scored:
        counts.add(extra)
    return sorted(counts)


def verified_precision_at(
    ranked_ids: list[int], checked_good: dict[int, bool], counts: list[int]
) -> list[tuple[int, int, float | None]]:
    """``(checked, kept_good, precision)`` of the top *k* of *ranked_ids*, for each *k* in *counts*.

    *checked_good* maps each item the user verified to whether its adopted label
    is Good.  Precision is ``None`` where no item in the top *k* was checked.
    """
    checked = np.cumsum([1 if cid in checked_good else 0 for cid in ranked_ids])
    good = np.cumsum([1 if checked_good.get(cid) else 0 for cid in ranked_ids])
    out: list[tuple[int, int, float | None]] = []
    for k in counts:
        n_checked, n_good = int(checked[k - 1]), int(good[k - 1])
        out.append((n_checked, n_good, n_good / n_checked if n_checked else None))
    return out


def estimated_precision_at(ctx: DetectorContext, corpus_scores: np.ndarray, counts: list[int]) -> PrecisionEstimate:
    """The lower-bound estimated precision of the top *k* of *corpus_scores*, for each *k* in *counts*.

    *corpus_scores* are the Find run's scores, unscorable media included (they
    are dropped here, and a count reaching into them has no estimate); *counts*
    index the full ranking.  Above :data:`PRECISION_CURVE_MAX_CORPUS` the
    estimate is read off a uniform sample, the top *j* of the sample standing
    for the top ``j * n / m`` of the corpus; a count too small to reach one
    sampled item has no estimate.
    """
    from vtscore.training.thresholds import MIN_CALIBRATION_POSITIVES  # noqa: PLC0415
    from vtscore.utils.scores import scored_only  # noqa: PLC0415

    none: list[float | None] = [None] * len(counts)
    estimate = ctx.precision_floor_cache
    if estimate is None:
        return PrecisionEstimate(ESTIMATE_UNAVAILABLE, 0, none)
    n_pos = estimate.calibration_positives
    if n_pos < MIN_CALIBRATION_POSITIVES:
        return PrecisionEstimate(ESTIMATE_INSUFFICIENT, n_pos, none)

    corpus = scored_only(corpus_scores)
    n = corpus.size
    if n == 0:
        return PrecisionEstimate(ESTIMATE_UNAVAILABLE, n_pos, none)
    if n > PRECISION_CURVE_MAX_CORPUS:
        rng = np.random.default_rng(PRECISION_CURVE_SAMPLE_SEED)
        corpus = rng.choice(corpus, size=PRECISION_CURVE_MAX_CORPUS, replace=False)
    curve = estimate.curve_for(corpus)
    if not curve.formed:
        # Enough positives but no contrast to fit (no Bad beside them) - the
        # floor reads this as insufficient evidence too.
        return PrecisionEstimate(ESTIMATE_INSUFFICIENT, n_pos, none)
    bound = curve.bound
    m = bound.size
    values: list[float | None] = []
    for k in counts:
        j = int(round(k * m / n))
        values.append(float(bound[j - 1]) if 1 <= j <= m and k <= n else None)
    return PrecisionEstimate(ESTIMATE_ESTIMATED, n_pos, values)

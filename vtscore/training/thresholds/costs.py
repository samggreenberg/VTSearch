"""What a threshold *costs* on a labelled sample.

:mod:`~vtscore.training.thresholds.knobs` says what an Inclusion value prices
(:func:`~vtscore.training.thresholds.knobs.inclusion_cost_weights` returns the
two rate weights); this module is the other half of that sentence - the one
place that turns those weights, a set of scores and their labels into the
number the knob is defined to minimise::

    cost = fpr_weight * FPR + fnr_weight * FNR

It lives at the bottom of the threshold stack rather than in the eval harness
because *both* tiers need it and neither may import the other: the shipped
Smart indicator (``vtscore.detectors.labeling_progress``) scores its cached
models against the live labelset, and the eval harness
(``vtscore.eval.calibration_metrics``, ``vtscore.eval.step_trainers``) scores
arms against held-out splits.  Those used to be three hand copies of the same
FP/FN counting loop with nothing holding them together (issue #3414); a
delegation cannot drift, which is the only fix that does not need a
``Mirror(...)`` entry in ``scripts/check-eval-app-sync.py`` to stay honest.

Numpy only - no torch - so the eval tier stays unit-testable without the model
stack.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def weighted_error_cost(
    scores: ArrayLike,
    labels: ArrayLike,
    threshold: float,
    fpr_weight: float,
    fnr_weight: float,
) -> tuple[float, float, float]:
    """Return ``(cost, fpr, fnr)`` at *threshold* (predict positive iff ``>=``).

    *labels* are ``1.0`` for positives and anything else (conventionally
    ``0.0``) for negatives.  ``cost = fpr_weight * FPR + fnr_weight * FNR``.

    Empty denominators yield a zero rate - no negatives in the sample means an
    FPR of 0, no positives means an FNR of 0 - so a degenerate sample scores as
    costless rather than raising or returning NaN.  Callers that need to
    distinguish "cheap" from "unmeasurable" must check the class counts
    themselves; the harness's per-step scorers return an empty series in that
    case rather than a zero.

    The ``>=`` convention is the one every threshold in the codebase uses,
    including :data:`~vtscore.training.thresholds.knobs.NO_GOOD_THRESHOLD`,
    which is deliberately above every possible sigmoid score so that no item
    clears it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    predicted = scores >= threshold
    pos = labels == 1.0
    neg = ~pos
    total_pos = float(pos.sum())
    total_neg = float(neg.sum())
    fp = float(np.count_nonzero(predicted & neg))
    fn = float(np.count_nonzero(~predicted & pos))
    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0
    return fpr_weight * fpr + fnr_weight * fnr, fpr, fnr

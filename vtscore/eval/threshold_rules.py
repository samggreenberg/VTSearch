"""Selectable decision-threshold rules for the threshold-stability study (#2790).

The realistic labeling loop's whole / box-pool path calibrates its threshold with
:func:`vtscore.eval.xcal.cross_calibrated_threshold` — a per-fold min-cost
**argmin** averaged over unstratified Train/Calibrate splits. The production app's
Autopilot does **not** do this: it calibrates with the split-conformal quantile +
gap-midpoint rule (issues #2693, #2784), pooling stratified fold orderings and
taking one conformal cut. So the sweep's ``argmin`` behaviour is a *fidelity gap*
vs. what a user actually experiences, and the violent single-step threshold jumps
#2790 reports are exactly the failure the conformal rule was introduced to fix
(an extreme order statistic — the lowest held-out calibration positive — moving
from one vote to the next).

This module makes the rule a **selectable knob** so the sweep can A/B the current
``argmin`` behaviour against the fidelity-correct ``conformal`` rule (and a couple
of diagnostic variants), keeping ``argmin`` byte-identical as the default.

Rules (``rule=`` argument):

* ``"argmin"`` — the historical sod behaviour. Delegates unchanged to
  :func:`vtscore.eval.xcal.cross_calibrated_threshold`.
* ``"conformal"`` — production Autopilot's rule. Stratified per-class folds, pool
  every fold's held-out ``(score, label)`` pairs, and apply :func:`conformal_threshold`
  once (ported verbatim from ``vtscore.training.thresholds`` on ``dev``). This is
  what the real app computes; the arm exists because ``evaluation-framework``
  predates #2784 and has no conformal rule of its own.
* ``"rank-transfer"`` — the ``conformal`` cut converted to a quantile of the pooled
  fold scores, then re-applied at that quantile of the **final** model's own score
  pool. Probes the fold→final scale mismatch (S3): if a cut that is stable in
  *rank* space wanders in *score* space, this arm is stable where ``conformal`` is
  not. Requires the final model's pool scores, so it is applied via
  :func:`rank_transfer_cut` after the final model exists.

:func:`median_smooth` is an orthogonal temporal smoother (the ``med3`` variant):
median of the last ``window`` raw thresholds, the cheapest possible hysteresis fix.

Everything here is head-agnostic (it takes a ``trainer_fn`` exactly like
:mod:`vtscore.eval.xcal`) and free of Flask / app imports, so it lives in the
library tier and is unit-testable on synthetic ``(scores, labels)`` with no models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vtscore.eval.xcal import PredictFn, TrainerFn, cross_calibrated_threshold

#: The inclusion knob's base budget: at ``k = 0`` the false-negative cap and the
#: false-positive guard are the 0.25 / 0.75 quantiles respectively. Kept in sync
#: with ``vtscore.training.thresholds`` on ``dev`` (the value the app ships).
CONFORMAL_BASE_BUDGET = 0.25
#: Positive-score quantile the ``k = -10`` end of the inclusion walk reaches.
CONFORMAL_QPOS_MAX = 0.75

RULES = ("argmin", "conformal", "rank-transfer")


def conformal_threshold(
    scores: Sequence[float],
    labels: Sequence[float],
    inclusion_value: int = 0,
) -> float:
    """Split-conformal quantile threshold over held-out calibration scores.

    Ported verbatim from ``vtscore.training.thresholds.conformal_threshold`` on
    ``dev`` (issue #2784) so the ``conformal`` arm matches production Autopilot
    exactly. Maps ``inclusion_value`` (``k``) to a decision threshold via
    quantiles of the calibration score distributions rather than a min-cost
    search over observed cuts:

    * a **false-negative cap** ``alpha(k) = min(1, BASE * 2^-k)`` — the cut never
      exceeds the ``alpha``-quantile of the calibration *positives*;
    * a **false-positive guard** for ``k <= 0`` at the ``1 - BASE * 2^k`` quantile
      of the *negatives*, with a walk *up* toward the positives as ``k`` drops;
    * the **gap midpoint** at ``k = 0`` — the midpoint of the band between the top
      of the negatives and the lowest positive. Every cut in that band has
      identical empirical error, so its top edge (the single lowest calibration
      positive) is an arbitrary, high-variance choice; the midpoint is the
      max-margin one and is what stops the threshold jumping vote-to-vote.

    Every component is monotone non-increasing in ``k``, so the cut is monotone in
    inclusion by construction. Returns ``0.5`` when the score list is empty or
    single-class (no quantiles to take); never abstains otherwise.
    """
    if len(scores) == 0:
        return 0.5

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    pos = scores_arr[labels_arr == 1.0]
    neg = scores_arr[labels_arr != 1.0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    def _threshold_at(k: int) -> float:
        fn_cap = float(np.quantile(pos, min(1.0, CONFORMAL_BASE_BUDGET * 2.0**-k)))
        if k > 0:
            # The k=0 floor keeps the seam monotone: q_pos(alpha) can sit above
            # the k=0 cut when the budget goes unspent there.
            return min(fn_cap, _threshold_at(0))
        fp_guard = float(np.quantile(neg, 1.0 - CONFORMAL_BASE_BUDGET * 2.0**k))
        # Midpoint of the band the calibration data cannot resolve: the top of the
        # negatives up to the lowest positive. Collapses to ``fp_guard`` under
        # class overlap (no band), so the FPR-controlled regime is untouched.
        gap_mid = (fp_guard + max(fp_guard, float(np.min(pos)))) / 2.0
        # Walk from that midpoint at k=0 up to the QPOS_MAX positive quantile at
        # k=-10, linearly in score space (evenly spaced stops even on few points).
        top = float(np.quantile(pos, CONFORMAL_QPOS_MAX))
        walk = gap_mid + (-k / 10.0) * max(0.0, top - gap_mid)
        return min(fn_cap, max(fp_guard, walk))

    return _threshold_at(inclusion_value)


def stratified_fold_orderings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
) -> list[tuple[list[float], list[float]]]:
    """Held-out ``(scores, labels)`` per fold under **class-stratified** splits.

    The conformal analogue of the fold loop in
    :func:`vtscore.eval.xcal.cross_calibrated_threshold`, differing in two ways
    that matter for the study:

    * splits are **stratified** — each class is permuted and cut at its own
      ``cal_fraction`` boundary, so a fold can never land single-class (the
      unstratified argmin path skips those folds, concentrating variance at low /
      imbalanced vote counts — plan suspect S4);
    * it returns the raw held-out orderings rather than a per-fold threshold, so
      the caller pools them and takes **one** conformal cut over the union
      (production's ``threshold_from_fold_orderings`` behaviour: the knob's
      resolution is bounded by how many calibration scores the quantile sees, so
      pooling beats averaging per-fold quantiles on a handful of votes each).

    Returns ``[]`` when the labels are too few or single-class to split — the
    caller then falls back to ``0.5`` exactly as the argmin path does.
    """
    y = np.asarray(y_train)
    n = int(y.size)
    if n < 4 or len({int(v) for v in y}) < 2:
        return []

    pos_idx = np.flatnonzero(y == 1.0)
    neg_idx = np.flatnonzero(y != 1.0)
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return []

    rng = np.random.default_rng(seed)
    orderings: list[tuple[list[float], list[float]]] = []
    for k in range(max(1, calibrate_count)):
        tr_idx, cal_idx = _stratified_split(pos_idx, neg_idx, cal_fraction, rng)
        if tr_idx is None or cal_idx is None:
            continue
        if len({int(v) for v in y[tr_idx]}) < 2 or len({int(v) for v in y[cal_idx]}) < 2:
            continue
        try:
            predict = trainer_fn(X_train[tr_idx], y_train[tr_idx], seed + k)
        except ValueError:
            continue
        cal_scores = np.asarray(predict(X_train[cal_idx]), dtype=np.float64).tolist()
        cal_labels = [float(v) for v in y[cal_idx]]
        orderings.append((cal_scores, cal_labels))
    return orderings


def _stratified_split(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    cal_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """One class-balanced Train/Calibrate split; ``(None, None)`` if unsplittable.

    Each class contributes ``round(class_n * cal_fraction)`` rows to Calibrate
    (clamped so both Train and Calibrate keep at least one of each class), so
    neither side is ever single-class — the guard the unstratified path lacks.
    """

    def _per_class(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        perm = rng.permutation(idx)
        n_cal = int(round(len(idx) * cal_fraction))
        n_cal = max(1, min(len(idx) - 1, n_cal))
        return perm[n_cal:], perm[:n_cal]  # (train, cal)

    p = _per_class(pos_idx)
    q = _per_class(neg_idx)
    if p is None or q is None:
        return None, None
    tr = np.concatenate([p[0], q[0]])
    cal = np.concatenate([p[1], q[1]])
    return tr, cal


def calibrated_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    rule: str = "argmin",
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
) -> float:
    """Decision threshold under the named *rule*.

    ``"argmin"`` delegates unchanged to
    :func:`vtscore.eval.xcal.cross_calibrated_threshold` (the sweep's current
    behaviour — byte-identical). ``"conformal"`` and ``"rank-transfer"`` build
    stratified fold orderings, pool them, and apply :func:`conformal_threshold`;
    ``"rank-transfer"`` returns the same conformal cut here (the rank re-mapping
    needs the final model's scores and is applied later via
    :func:`rank_transfer_cut`). Falls back to ``0.5`` when the votes cannot form a
    valid split, matching the argmin path's fallback.
    """
    if rule == "argmin":
        return cross_calibrated_threshold(
            X_train,
            y_train,
            trainer_fn,
            seed,
            inclusion_value=inclusion_value,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
        )
    if rule in ("conformal", "rank-transfer"):
        orderings = stratified_fold_orderings(
            X_train,
            y_train,
            trainer_fn,
            seed,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
        )
        if not orderings:
            return 0.5
        pooled_scores = [s for scores, _ in orderings for s in scores]
        pooled_labels = [lb for _, labels in orderings for lb in labels]
        return conformal_threshold(pooled_scores, pooled_labels, inclusion_value)
    raise ValueError(f"unknown threshold rule {rule!r}; expected one of {RULES}")


def rank_transfer_cut(
    conformal_cut: float,
    pooled_fold_scores: Sequence[float],
    final_pool_scores: Sequence[float],
) -> float:
    """Re-express a *conformal_cut* as a rank and read it back on the final scores.

    The conformal cut is chosen on the **fold** models' score scale (they train on
    half the votes and saturate differently), then applied to the **final** model's
    scores — the fold→final scale mismatch (plan suspect S3). This maps the cut to
    its quantile position among the pooled fold scores and returns the score at
    that same quantile of the final model's own pool distribution, so a cut that is
    stable in rank space stays stable in score space.

    Degenerates gracefully: with no fold scores or no final scores it returns
    *conformal_cut* unchanged.
    """
    fold = np.asarray(pooled_fold_scores, dtype=np.float64)
    final = np.asarray(final_pool_scores, dtype=np.float64)
    if fold.size == 0 or final.size == 0:
        return float(conformal_cut)
    # Fraction of fold scores at or below the cut = the cut's quantile position.
    q = float(np.mean(fold <= conformal_cut))
    return float(np.quantile(final, min(1.0, max(0.0, q))))


def median_smooth(threshold_history: Sequence[float], window: int = 3) -> float:
    """Median of the last *window* raw thresholds (the ``med3`` temporal fix).

    ``threshold_history`` is the sequence of raw (pre-smoothing) thresholds up to
    and including the current step; the smoothed threshold is the median of its
    last ``window`` entries (fewer early on). A cheap hysteresis that kills a
    single-step spike without lagging a genuine level shift by more than one step.
    Non-finite entries (a degenerate fallback) are dropped before the median; if
    none remain the last raw value is returned.
    """
    if not threshold_history:
        raise ValueError("threshold_history must be non-empty")
    tail = [float(t) for t in list(threshold_history)[-window:]]
    finite = [t for t in tail if np.isfinite(t)]
    if not finite:
        return float(threshold_history[-1])
    return float(np.median(finite))


# Re-exported for callers that build a trainer_fn and want the type names local.
__all__ = [
    "CONFORMAL_BASE_BUDGET",
    "CONFORMAL_QPOS_MAX",
    "RULES",
    "PredictFn",
    "TrainerFn",
    "calibrated_threshold",
    "conformal_threshold",
    "median_smooth",
    "rank_transfer_cut",
    "stratified_fold_orderings",
]

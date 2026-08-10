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

# The conformal inclusion rule lives in ``vtscore.training.thresholds`` (its dev
# home, #2784) so the whole-image path here and the grouped detector-training path
# there share one implementation. Re-exported for callers/tests that import it
# from this module.
from vtscore.training.thresholds import conformal_threshold

#: The eval's calibration rules. ``argmin`` (the pre-#2784 min-cost cut) is retained
#: only for explicitly reproducing old runs; it is never a default or a study arm.
RULES = ("conformal", "rank-transfer", "argmin")


def _group_index(groups: Sequence, y: np.ndarray) -> tuple[list, list[list[int]], list[float]]:
    """``(group_ids, row_indices_per_group, label_per_group)`` in first-seen order.

    A group is one voted image. Every row in a group shares that image's label, so the
    group's label is its first row's.
    """
    order: list = []
    rows: dict = {}
    for i, g in enumerate(groups):
        if g not in rows:
            rows[g] = []
            order.append(g)
        rows[g].append(i)
    return order, [rows[g] for g in order], [float(y[rows[g][0]]) for g in order]


def stratified_fold_orderings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
    stable_folds: bool = False,
    groups: Sequence | None = None,
    score_rows_by_group: dict | None = None,
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

    *groups* switches on the **bag-aware** path: rows sharing a group id are one voted
    image, they stay together on one side of every split, split counts are taken over
    groups rather than rows, and each calibration group collapses to a single
    max-pooled score, matching how inference scores an image by its best region.
    *score_rows_by_group* (grouped path only) gives the row stack a group should be
    *scored* over, decoupling what a fold trains on from what a calibration bag
    collapses to. Without it a bag collapses over its own training rows, which is
    wrong whenever the two geometries differ: a box-pool HAC negative trains on one
    whole-image CLS row while inference maxes over every tree node, and ``max`` is an
    upward-biased order statistic, so a cut fitted to single rows lands far below every
    inference score and the detector labels everything positive. This mirrors the
    grouped torch path's ``score_rows_by_group``
    (:func:`vtscore.training.thresholds.compute_fold_orderings`). ``groups=None`` keeps
    the historical row-wise behaviour byte-identical.
    """
    y = np.asarray(y_train)
    n = int(y.size)
    if n < 4 or len({int(v) for v in y}) < 2:
        return []

    if groups is not None:
        return _grouped_fold_orderings(
            X_train,
            y_train,
            y,
            trainer_fn,
            seed,
            groups=groups,
            score_rows_by_group=score_rows_by_group,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
            stable_folds=stable_folds,
        )

    pos_idx = np.flatnonzero(y == 1.0)
    neg_idx = np.flatnonzero(y != 1.0)
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return []

    rng = np.random.default_rng(seed)
    orderings: list[tuple[list[float], list[float]]] = []
    for k in range(max(1, calibrate_count)):
        tr_idx, cal_idx = _stratified_split(pos_idx, neg_idx, cal_fraction, rng, stable=stable_folds)
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


def _grouped_fold_orderings(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    groups: Sequence,
    score_rows_by_group: dict | None,
    calibrate_count: int,
    cal_fraction: float,
    stable_folds: bool,
) -> list[tuple[list[float], list[float]]]:
    """Bag-aware :func:`stratified_fold_orderings`; see its *groups* documentation."""
    gids, grows, glabels = _group_index(groups, y)
    gy = np.asarray(glabels)
    if len(gids) < 4 or len({int(v) for v in gy}) < 2:
        return []
    pos_g = np.flatnonzero(gy == 1.0)
    neg_g = np.flatnonzero(gy != 1.0)
    if len(pos_g) < 2 or len(neg_g) < 2:
        return []

    rng = np.random.default_rng(seed)
    orderings: list[tuple[list[float], list[float]]] = []
    for k in range(max(1, calibrate_count)):
        tr_g, cal_g = _stratified_split(pos_g, neg_g, cal_fraction, rng, stable=stable_folds)
        if tr_g is None or cal_g is None:
            continue
        if len({int(gy[i]) for i in tr_g}) < 2 or len({int(gy[i]) for i in cal_g}) < 2:
            continue
        tr_rows = [r for gi in tr_g for r in grows[gi]]
        try:
            predict = trainer_fn(X_train[tr_rows], y_train[tr_rows], seed + k)
        except ValueError:
            continue
        cal_scores, cal_labels = _collapse_groups(predict, cal_g, gids, grows, gy, X_train, score_rows_by_group)
        if len(set(cal_labels)) < 2:
            continue
        orderings.append((cal_scores, cal_labels))
    return orderings


def _collapse_groups(
    predict: PredictFn,
    cal_g: np.ndarray,
    gids: list,
    grows: list[list[int]],
    gy: np.ndarray,
    X_train: np.ndarray,
    score_rows_by_group: dict | None,
) -> tuple[list[float], list[float]]:
    """One max-pooled ``(score, label)`` per calibration group, as inference scores an image."""
    scores: list[float] = []
    labels: list[float] = []
    for gi in cal_g:
        rows = None if score_rows_by_group is None else score_rows_by_group.get(gids[gi])
        mat = X_train[grows[gi]] if rows is None else np.asarray(rows)
        if mat.size == 0:
            continue
        s = np.asarray(predict(mat), dtype=np.float64).reshape(-1)
        finite = s[np.isfinite(s)]
        if finite.size == 0:
            continue
        scores.append(float(finite.max()))
        labels.append(float(gy[gi]))
    return scores, labels


def _stratified_split(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    cal_fraction: float,
    rng: np.random.Generator,
    stable: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """One class-balanced Train/Calibrate split; ``(None, None)`` if unsplittable.

    Each class contributes ``round(class_n * cal_fraction)`` rows to Calibrate
    (clamped so both Train and Calibrate keep at least one of each class), so
    neither side is ever single-class — the guard the unstratified path lacks.

    ``stable=True`` (#2790 causal test) **skips the permutation** — the fold is by
    append position, so an item's Train/Calibrate membership does not reshuffle when
    later votes arrive. Isolates whether the per-step fold reshuffle (which, with ~3
    positives split 1-train/2-cal, jerks ``min(cal-positive)`` and hence the cut) is
    what drives the spikes.
    """

    def _per_class(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        perm = np.asarray(idx) if stable else rng.permutation(idx)
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
    stable_folds: bool = False,
    groups: Sequence | None = None,
    score_rows_by_group: dict | None = None,
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
        if groups is not None:
            raise ValueError("the 'argmin' rule has no bag-aware path; use 'conformal' with groups")
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
            stable_folds=stable_folds,
            groups=groups,
            score_rows_by_group=score_rows_by_group,
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
    "RULES",
    "PredictFn",
    "TrainerFn",
    "calibrated_threshold",
    "conformal_threshold",
    "median_smooth",
    "rank_transfer_cut",
    "stratified_fold_orderings",
]

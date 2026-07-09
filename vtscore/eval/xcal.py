"""Trainer/head-agnostic cross-calibration threshold selection.

This is the generic core of what production does at vote time (and what
``label_curve``'s ``f1_at_xcal`` is measured at): split the labelled examples
k ways into train/cal halves, retrain on each train half, find the
inclusion-weighted optimal threshold on the held-out cal half, and average the
per-fold thresholds. Because it takes a ``trainer_fn`` (``(X, y, seed) ->
predict``) it works for any head — the MLP head trains an MLP; the cosine head
passes a no-op trainer whose ``predict`` is ``X @ query`` — so both the label
curve and the region-curve sweep get the *same* realistic threshold, chosen on
data the model never trained on.

Factored out of ``vtscore.eval.label_curve`` (which now delegates here) so
``vtscore.eval.region_curve`` can reuse it without importing the label-curve
sweep machinery.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]
TrainerFn = Callable[[np.ndarray, np.ndarray, int], PredictFn]


def cross_calibrated_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
) -> float:
    """Average of per-fold inclusion-weighted optimal thresholds on held-out cal splits.

    Splits the labels ``calibrate_count`` ways into train/cal halves, retrains
    on each train half via ``trainer_fn``, and finds the optimal threshold
    (:func:`vtscore.training.thresholds.find_optimal_threshold`) on the cal
    half. Returns ``0.5`` when the label budget is too small to form valid
    splits (mirrors the production fallback), and skips single-class or
    trainer-refused folds.
    """
    from vtscore.training.thresholds import find_optimal_threshold

    n = int(y_train.size)
    if n < 4:
        return 0.5
    n_cal = max(1, round(n * cal_fraction))
    n_tr = n - n_cal
    if n_tr < 2 or n_cal < 1:
        return 0.5

    rng = np.random.default_rng(seed)
    thresholds: list[float] = []
    for k in range(max(1, calibrate_count)):
        order = rng.permutation(n)
        tr_idx = order[:n_tr]
        cal_idx = order[n_tr:]
        # Single-class splits would crash the trainer or short-circuit the
        # threshold finder; just skip this fold.
        if len({int(v) for v in y_train[tr_idx]}) < 2:
            continue
        if len({int(v) for v in y_train[cal_idx]}) < 2:
            continue
        try:
            predict = trainer_fn(X_train[tr_idx], y_train[tr_idx], seed + k)
        except ValueError:
            continue
        cal_scores = np.asarray(predict(X_train[cal_idx]), dtype=np.float64).tolist()
        cal_labels = [float(v) for v in y_train[cal_idx]]
        t = find_optimal_threshold(cal_scores, cal_labels, inclusion_value)
        # find_optimal_threshold can return +/-inf on degenerate splits; only
        # count finite thresholds toward the mean.
        if np.isfinite(t):
            thresholds.append(float(t))

    if not thresholds:
        return 0.5
    return float(np.mean(thresholds))

"""Error-vs-K curve for the small-object-detection sweep.

Dataset-agnostic core: given pre-materialized vectors (positive GT-box exemplar
vectors, a pool of negative whole-image vectors, and per-test-image region
matrices + labels), sweep the few-shot annotation count K and, at each K, report
the **cross-calibrated (realistic)** inclusion-weighted FPR+FNR plus the oracle
min-over-threshold reference. Keeping this in terms of vectors (not images) makes
it unit-testable and lets the ``scripts/sod`` orchestrator own all dataset/zip/
cache concerns.

Per (K, seed):

1. sample K positive exemplars + ``neg_ratio*K`` negative whole vectors as the
   annotation budget;
2. pick the threshold by cross-calibration on held-out folds of that budget
   (:func:`vtscore.eval.xcal.cross_calibrated_threshold`), blended with a GMM
   threshold at low label counts when ``safe_thresholds`` is set
   (:func:`vtscore.training.thresholds.calculate_safe_threshold`);
3. fit the head on the full budget, score every test image by max-pool over its
   regions, and compute the weighted error at that threshold + the oracle.

The head is the MLP for every embedder (primary) or the cosine head for
text-capable embedders (baseline). ``K == 0`` is the zero-shot cosine point
(no annotations to calibrate on → reported at the oracle operating point).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from vtscore.eval.error_metrics import f1_at, min_weighted_cost, weighted_error
from vtscore.eval.scoring_heads import CosineHead, MLPHead, max_pool_with_argmax
from vtscore.eval.xcal import cross_calibrated_threshold


@dataclass
class RegionCurveInputs:
    """Pre-materialized vectors for one (dataset, class, embedder, proposal) cell."""

    pos_exemplars: np.ndarray  # (P, D) annotation-pool positive exemplar vectors
    neg_train_wholes: np.ndarray  # (Nn, D) pool of training-negative whole vectors
    test_region_mats: list[np.ndarray]  # per test image: (R_i, D) region vectors
    test_labels: Sequence[int]  # 0/1 aligned with test_region_mats
    input_dim: int
    meta: dict = field(default_factory=dict)  # carried onto every result row
    # Optional, for localization (IoU): per test image, the region boxes aligned
    # with test_region_mats, and the GT boxes for the class ([] for negatives).
    test_region_boxes: list[np.ndarray] = field(default_factory=list)
    test_gt_boxes: list[list] = field(default_factory=list)


def _calib_mode(k: int, n_neg: int, *, safe_thresholds: bool) -> str:
    """Which threshold regime a (K, n_neg) budget lands in (for plotting)."""
    n = k + n_neg
    if k < 2 or n < 4:  # cross-calibration cannot form ≥2-per-class folds
        return "fallback"
    if safe_thresholds and n < 6:
        return "gmm"
    if safe_thresholds and n < 20:
        return "blend"
    return "xcal"


def _oracle_operating_point(
    scores: Sequence[float] | np.ndarray, labels: Sequence[int], inclusion: int
) -> tuple[float, dict]:
    """Best-case: threshold that minimizes weighted error on the test set itself."""
    from vtscore.training.thresholds import find_optimal_threshold

    thr = find_optimal_threshold(list(scores), [float(v) for v in labels], inclusion)
    if not np.isfinite(thr):
        thr = 0.5
    return float(thr), weighted_error(scores, [float(v) for v in labels], thr, inclusion)


def _iou_metrics(inputs: RegionCurveInputs, argmax: np.ndarray) -> tuple[float, float]:
    """Mean IoU (+ CorLoc@0.5) of the top-scoring region box vs best GT box.

    Averaged over all test positives that have a GT box and a valid winning
    region. Returns (NaN, NaN) when boxes weren't provided or no positive
    qualifies. The ``whole`` proposal (one full-frame region) scores ~0 by
    construction — informative, not a bug.
    """
    if not inputs.test_region_boxes or not inputs.test_gt_boxes:
        return float("nan"), float("nan")
    from vtscore.eval.metrics import box_iou

    ious: list[float] = []
    for i, lbl in enumerate(inputs.test_labels):
        if int(lbl) != 1:
            continue
        gts = inputs.test_gt_boxes[i]
        boxes = inputs.test_region_boxes[i]
        ai = int(argmax[i])
        if not gts or ai < 0 or ai >= len(boxes):
            continue
        pred = tuple(float(v) for v in boxes[ai])
        ious.append(max(box_iou(pred, tuple(float(v) for v in g)) for g in gts))
    if not ious:
        return float("nan"), float("nan")
    return round(sum(ious) / len(ious), 6), round(sum(v >= 0.5 for v in ious) / len(ious), 6)


def _evaluate_one(
    inputs: RegionCurveInputs,
    head_kind: str,
    k: int,
    seed: int,
    *,
    neg_ratio: int,
    inclusion: int,
    safe_thresholds: bool,
    calibrate_count: int,
    cal_fraction: float,
    query_vec: np.ndarray | None,
) -> dict | None:
    """One (K, seed) cell; returns a result row or ``None`` if it can't run."""
    rng = np.random.default_rng(seed)
    test_labels = [int(v) for v in inputs.test_labels]

    # --- K == 0: zero-shot cosine baseline (no annotations to calibrate on) ---
    if k == 0:
        if head_kind != "cosine" or query_vec is None:
            return None  # only the text/cosine head has a K=0 point
        head = CosineHead(query_vec)
        scores, argmax = max_pool_with_argmax(head.score_rows, inputs.test_region_mats)
        thr, err = _oracle_operating_point(scores, test_labels, inclusion)
        f1 = f1_at(scores, test_labels, thr)
        mean_iou, corloc = _iou_metrics(inputs, argmax)
        return _row(inputs, head_kind, 0, 0, seed, thr, err, err["cost"], "none_k0", f1, mean_iou, corloc)

    P = inputs.pos_exemplars.shape[0]
    if k > P:
        return None  # not enough positive exemplars for this K
    n_neg = min(max(1, neg_ratio * k), inputs.neg_train_wholes.shape[0])
    if n_neg < 1:
        return None

    pos_idx = rng.permutation(P)[:k]
    neg_idx = rng.permutation(inputs.neg_train_wholes.shape[0])[:n_neg]
    x_train = np.vstack([inputs.pos_exemplars[pos_idx], inputs.neg_train_wholes[neg_idx]]).astype(np.float32)
    y_train = np.array([1.0] * k + [0.0] * n_neg, dtype=np.float32)

    head = MLPHead(inputs.input_dim) if head_kind == "mlp" else CosineHead(query_vec)  # type: ignore[arg-type]

    xcal_thr = cross_calibrated_threshold(
        x_train,
        y_train,
        head.trainer_fn(),
        seed,
        inclusion_value=inclusion,
        calibrate_count=calibrate_count,
        cal_fraction=cal_fraction,
    )

    # Fit the final head on the full budget, then score the test images.
    try:
        head.fit(x_train, y_train, seed)
    except ValueError:
        return None  # single-class budget (shouldn't happen given k>=1, n_neg>=1)

    threshold = xcal_thr
    n_labels = k + n_neg
    if safe_thresholds and n_labels < 20:
        from vtscore.training.thresholds import calculate_safe_threshold

        train_scores = np.asarray(head.score_rows(x_train), dtype=np.float64).tolist()
        threshold = calculate_safe_threshold(xcal_thr, train_scores, n_labels)

    scores, argmax = max_pool_with_argmax(head.score_rows, inputs.test_region_mats)
    err = weighted_error(scores, [float(v) for v in test_labels], threshold, inclusion)
    oracle = min_weighted_cost(scores, [float(v) for v in test_labels], inclusion)
    f1 = f1_at(scores, test_labels, threshold)
    mean_iou, corloc = _iou_metrics(inputs, argmax)
    return _row(
        inputs,
        head_kind,
        k,
        n_neg,
        seed,
        threshold,
        err,
        oracle,
        _calib_mode(k, n_neg, safe_thresholds=safe_thresholds),
        f1,
        mean_iou,
        corloc,
    )


def _row(inputs, head_kind, k, n_neg, seed, threshold, err, oracle, calib_mode, f1, mean_iou, corloc) -> dict:
    return {
        **inputs.meta,
        "head": head_kind,
        "k": int(k),
        "n_pos": int(k),
        "n_neg_train": int(n_neg),
        "seed": int(seed),
        "cost": err["cost"],
        "fpr": err["fpr"],
        "fnr": err["fnr"],
        "f1": f1,
        "mean_iou": mean_iou,
        "corloc": corloc,
        "oracle_cost": float(oracle),
        "threshold": round(float(threshold), 6),
        "calib_mode": calib_mode,
        "n_test": len(inputs.test_labels),
        "n_test_pos": int(sum(int(v) for v in inputs.test_labels)),
    }


def evaluate_region_curve(
    inputs: RegionCurveInputs,
    head_kind: str,
    *,
    k_values: Sequence[int],
    seeds: Sequence[int],
    neg_ratio: int = 1,
    inclusion: int = 0,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
    query_vec: np.ndarray | None = None,
) -> list[dict]:
    """Run the full K × seed grid for one cell, returning one row per (K, seed).

    ``head_kind`` is ``"mlp"`` or ``"cosine"``; the cosine head needs
    ``query_vec`` (the L2-normalized text-query vector). Rows carry ``inputs.meta``
    (dataset/class/embedder/proposal identifiers) so the orchestrator can
    concatenate cells into one table.
    """
    if head_kind == "cosine" and query_vec is None:
        raise ValueError("cosine head requires query_vec")
    rows: list[dict] = []
    for k in k_values:
        for seed in seeds:
            t0 = time.perf_counter()
            row = _evaluate_one(
                inputs,
                head_kind,
                int(k),
                int(seed),
                neg_ratio=neg_ratio,
                inclusion=inclusion,
                safe_thresholds=safe_thresholds,
                calibrate_count=calibrate_count,
                cal_fraction=cal_fraction,
                query_vec=query_vec,
            )
            if row is not None:
                # Per-cell compute time (calibration + fit + scoring); embeddings are
                # already materialized upstream, so this is the non-cached work that
                # runs on every sweep — reported as the MLP component of "total time".
                row["compute_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                rows.append(row)
    return rows

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

from vtscore.eval.error_metrics import f1_at, max_f1, min_weighted_cost, weighted_error
from vtscore.eval.scoring_heads import CosineHead, MLPHead, max_pool_over_images, max_pool_with_argmax
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
    # Region-voting mode (DINO-patch faithful path). When True, ``pos_exemplars``
    # holds one snapped positive per image (K = good swipes) and negatives come
    # from ``neg_train_bags`` — one (L_i, D) bag of CLS+leaf vectors per negative
    # image — so training floods leaves and balances per image (bag), matching the
    # app detector. ``neg_train_wholes`` is unused in this mode.
    region_voting: bool = False
    neg_train_bags: list[np.ndarray] = field(default_factory=list)
    # Realistic loop only: when True (and not region_voting), a Bad vote contributes ALL of that
    # image's region vectors as one per-image negative bag (via ``train_rv_head``) instead of the
    # single whole-image vector — "No → all windows" for sliding / dino / box-pool proposals. No-op
    # for ``whole`` (one region). ``region_voting`` already does this for hac and takes precedence.
    neg_regions: bool = False
    # Realistic labeling-loop pool (only populated for ``evaluate_realistic_curve``).
    # These describe the *training* images the simulated user can label, keyed by a
    # stable image id, so the loop can rank/select them like the app does. All lists
    # are aligned with ``pool_ids``; ``pool_pos_exemplars`` maps a positive id to its
    # good-vote training rows ((1,D) snapped covering box in region-voting; (M,D)
    # GT-box exemplars otherwise). Empty in the controlled mode.
    pool_ids: list[int] = field(default_factory=list)
    pool_region_mats: list[np.ndarray] = field(default_factory=list)  # per image: (R_i, D)
    pool_region_boxes: list[np.ndarray] = field(default_factory=list)  # per image: (R_i, 4), for the surfacing pred_box
    pool_whole_vecs: np.ndarray | None = None  # (N, D) for the diversity tree
    pool_leaf_masks: list[np.ndarray] = field(default_factory=list)  # per image: (R_i,) bool
    pool_labels: list[int] = field(default_factory=list)  # 1 pos / 0 neg, aligned with pool_ids
    pool_pos_exemplars: dict[int, np.ndarray] = field(default_factory=dict)  # positive id -> (m_i, D)


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


def _oracle_extra(scores: Sequence[float] | np.ndarray, labels: Sequence[int]) -> dict[str, float]:
    """Oracle **ceiling** for F1: the max F1 over all thresholds.

    Companion to ``oracle_cost`` (min cost) for the F1 plot. Unlike cost, F1 has
    its own optimal threshold — at extreme imbalance it differs sharply from the
    min-cost threshold (cost is rate-based; F1 is precision/count-based), so this
    is computed independently as ``max_f1`` rather than "F1 at the min-cost τ".
    By construction ``oracle_f1 >= f1`` always (a true ceiling). fpr/fnr get no
    oracle companion — their per-metric optimum is degenerate (min FPR = 0 by
    predicting all-negative), so only their sum (cost) has a meaningful oracle.
    """
    return {"oracle_f1": max_f1(scores, [float(v) for v in labels])}


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
        b = boxes[ai]
        pred = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        ious.append(max(box_iou(pred, (float(g[0]), float(g[1]), float(g[2]), float(g[3]))) for g in gts))
    if not ious:
        return float("nan"), float("nan")
    return round(sum(ious) / len(ious), 6), round(sum(v >= 0.5 for v in ious) / len(ious), 6)


def _flood_context_np(X_list: list, y_list: list[float], groups: list):
    """Numpy port of ``detectors.training._flood_context``: bag-aware training.

    Returns ``(n_votes, cal_groups, sample_weights)`` where ``n_votes`` is the
    distinct-bag count (the unit the hidden width + safe-threshold ramp size on),
    ``cal_groups`` is ``groups`` only when a bag holds >1 row (so folds split by
    bag), and ``sample_weights`` are per-bag balancing weights (else ``None``, so
    ``train_model`` falls back to inverse-frequency). Mirrors the app detector so
    region-flooded negatives count once per image, not once per leaf.
    """
    from vtscore.training.thresholds import _per_bag_fit_weights

    n_votes = len(set(groups)) if groups else len(X_list)
    flooded = bool(groups) and len(X_list) != n_votes
    cal_groups = groups if flooded else None
    sample_weights = _per_bag_fit_weights(np.asarray(y_list, dtype=np.float32), groups) if flooded else None
    return n_votes, cal_groups, sample_weights


def sample_rv_budget(
    pos_exemplars: np.ndarray, neg_train_bags: list[np.ndarray], k: int, neg_ratio: int, seed: int
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Deterministically draw K positive images + ``neg_ratio*K`` negative bags.

    Shared by the eval and the viz overlay so both simulate the *same* budget for
    a given ``(k, seed)``. Returns ``(pos_rows (k,D), neg_bags)`` or ``None`` when
    the budget can't be met (too few positives/negatives).
    """
    P = pos_exemplars.shape[0]
    n_bag_total = len(neg_train_bags)
    if k < 1 or k > P:
        return None
    n_neg = min(max(1, neg_ratio * k), n_bag_total)
    if n_neg < 1:
        return None
    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(P)[:k]
    neg_idx = rng.permutation(n_bag_total)[:n_neg]
    return pos_exemplars[pos_idx], [neg_train_bags[int(j)] for j in neg_idx]


def train_rv_head(
    pos_rows: np.ndarray,
    neg_bags: list[np.ndarray],
    input_dim: int,
    seed: int,
    *,
    inclusion: int,
    safe_thresholds: bool,
    calibrate_count: int,
    cal_fraction: float,
):
    """Train the region-voting MLP + pick its (pre-GMM) threshold; bag-aware.

    Replicates the app detector's ``_train_and_score_xy`` training/calibration:
    one row per good image (label 1), one row per flooded leaf bagged by negative
    image (label 0), per-bag balancing, cross-calibration split by bag with the
    ``n_votes<6 → 0.5`` skip. Returns ``(predict_fn, raw_threshold, n_votes)``;
    the caller applies the GMM safe-threshold blend against its scored set (the
    ``scores``/``n_votes`` production passes to ``calculate_safe_threshold``). The
    return is ``None`` on a single-class budget.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import _auto_hidden_dim, train_model  # noqa: PLC0415
    from vtscore.training.thresholds import cross_calibration_threshold_cached  # noqa: PLC0415

    from vtscore.eval.scoring_heads import _mlp_predict_factory  # noqa: PLC0415

    k = pos_rows.shape[0]
    X_list: list[np.ndarray] = [pos_rows[i] for i in range(k)]
    y_list: list[float] = [1.0] * k
    groups: list = [("g", i) for i in range(k)]
    for bj, bag in enumerate(neg_bags):
        for row in bag:
            X_list.append(np.asarray(row, dtype=np.float32))
            y_list.append(0.0)
            groups.append(("b", bj))

    n_votes, cal_groups, sample_weights = _flood_context_np(X_list, y_list, groups)
    hidden_dim = _auto_hidden_dim(n_votes)

    if safe_thresholds and n_votes < 6:
        threshold = 0.5
    else:
        threshold = cross_calibration_threshold_cached(
            X_list,
            y_list,
            input_dim,
            inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=cal_fraction,
            hidden_dim=hidden_dim,
            groups=cal_groups,
        )
        if not np.isfinite(threshold):
            threshold = 0.5

    xt = torch.tensor(np.stack(X_list).astype(np.float32, copy=False))
    yt = torch.tensor(np.asarray(y_list, dtype=np.float32)).unsqueeze(1)
    sw = torch.tensor(sample_weights, dtype=torch.float32) if sample_weights is not None else None
    try:
        model = train_model(xt, yt, input_dim, seed=seed, hidden_dim=hidden_dim, sample_weights=sw)
    except ValueError:
        return None  # single-class budget
    return _mlp_predict_factory(model), float(threshold), n_votes


def _evaluate_one_rv(
    inputs: RegionCurveInputs,
    k: int,
    seed: int,
    *,
    neg_ratio: int,
    inclusion: int,
    safe_thresholds: bool,
    calibrate_count: int,
    cal_fraction: float,
) -> dict | None:
    """Region-voting (DINO-patch faithful) cell: replicates the app detector's
    ``_train_and_score_xy`` label construction + bag-aware training/calibration.

    K samples positive *images* (one snapped vector each), ``neg_ratio*K``
    negative *images* (each a bag of CLS+leaf vectors, all flooded as negatives).
    """
    from vtscore.training.thresholds import calculate_safe_threshold  # noqa: PLC0415

    test_labels = [int(v) for v in inputs.test_labels]
    budget = sample_rv_budget(inputs.pos_exemplars, inputs.neg_train_bags, k, neg_ratio, seed)
    if budget is None:
        return None
    pos_rows, neg_bags = budget
    n_neg = len(neg_bags)

    trained = train_rv_head(
        pos_rows,
        neg_bags,
        inputs.input_dim,
        seed,
        inclusion=inclusion,
        safe_thresholds=safe_thresholds,
        calibrate_count=calibrate_count,
        cal_fraction=cal_fraction,
    )
    if trained is None:
        return None
    predict, threshold, n_votes = trained

    scores, argmax = max_pool_with_argmax(predict, inputs.test_region_mats)
    if safe_thresholds:
        # GMM blend over the scored distribution; ramp counts bags (votes), matching
        # the app path's calculate_safe_threshold(threshold, scores, n_votes). This is
        # the *controlled-grid* cell (evaluate_region_curve): its only scored corpus is
        # the test set, so it fits the (unsupervised) GMM over the test scores — the
        # controlled analog of production's all-media fit. The realistic loop
        # (_realistic_one_seed) instead fits over the labeling pool, which is its
        # scored corpus. NB: evaluate_region_curve is not wired to any script today
        # (sweep.py uses evaluate_realistic_curve); this path is kept for controlled runs.
        threshold = calculate_safe_threshold(threshold, list(scores), n_votes)

    err = weighted_error(scores, [float(v) for v in test_labels], threshold, inclusion)
    oracle = min_weighted_cost(scores, [float(v) for v in test_labels], inclusion)
    oracle_extra = _oracle_extra(scores, test_labels)
    f1 = f1_at(scores, test_labels, threshold)
    mean_iou, corloc = _iou_metrics(inputs, argmax)
    calib_mode = _calib_mode(k, n_neg, safe_thresholds=safe_thresholds)
    return _row(inputs, "mlp", k, n_neg, seed, threshold, err, oracle, calib_mode, f1, mean_iou, corloc, oracle_extra)


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
        # K=0 reported at the oracle-cost operating point; oracle_f1 is still the
        # independent max-F1 ceiling (>= the reported f1).
        oracle_extra = _oracle_extra(scores, test_labels)
        return _row(inputs, head_kind, 0, 0, seed, thr, err, err["cost"], "none_k0", f1, mean_iou, corloc, oracle_extra)

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
    oracle_extra = _oracle_extra(scores, test_labels)
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
        oracle_extra,
    )


def _row(
    inputs, head_kind, k, n_neg, seed, threshold, err, oracle, calib_mode, f1, mean_iou, corloc, oracle_extra=None
) -> dict:
    oe = oracle_extra or {}
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
        # Oracle F1 ceiling (max F1 over τ) for the F1 plot; NaN when the caller
        # didn't supply it (e.g. legacy paths). See ``_oracle_extra``. fpr/fnr have
        # no oracle companion (their per-metric optimum is degenerate).
        "oracle_f1": float(oe.get("oracle_f1", float("nan"))),
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
            if inputs.region_voting and head_kind == "mlp":
                row = _evaluate_one_rv(
                    inputs,
                    int(k),
                    int(seed),
                    neg_ratio=neg_ratio,
                    inclusion=inclusion,
                    safe_thresholds=safe_thresholds,
                    calibrate_count=calibrate_count,
                    cal_fraction=cal_fraction,
                )
            else:
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


# ---------------------------------------------------------------------------
# Realistic labeling loop (faithful port of the app's Autopilot active learning)
#
# Instead of a controlled K-positive budget, the simulated user labels one item
# per step in the order the app would surface it, retraining from scratch each
# step. The x-axis is ``t`` = total annotations (good+bad); the pos/neg mix
# EMERGES from prevalence + the selection policy. See the module docstring of the
# controlled path above for the shared training/threshold conventions.
# ---------------------------------------------------------------------------

# Autopilot phase → select-mode map (frontend label-view.component.ts:302-326).
# ``done`` maps to ``new`` so the curve keeps extending past the recommended stop
# (unless ``stop_at_done``); the row still records ``stop_recommended``.
_PHASE_TO_SELECT = {"good": "top", "bad": "hard", "hard": "hard", "new": "new", "done": "new"}

# Span "green" node count — mirrors autopilot_goal_diversity (default 40); a k=3
# tree reaches this at 4 full BFS levels (1+3+9+27). Kept as a constant so this
# library tier does not depend on app settings.
_SPAN_GREEN = 40
_SPAN_YELLOW = 10


class AutopilotPhaseMachine:
    """Faithful port of ``checkPhaseTransition`` (autopilot-state.service.ts:87).

    Stateless in counts: the phase is re-derived from the current good/bad counts
    and the three indicator colors every step, so it advances (or regresses)
    deterministically. ``good_to_start``/``bad_to_start`` default to the app's 3/4.
    """

    def __init__(self, *, good_to_start: int = 3, bad_to_start: int = 4) -> None:
        self.good_to_start = int(good_to_start)
        self.bad_to_start = int(bad_to_start)

    def next_phase(self, good: int, bad: int, smart: str, stable: str, span: str) -> str:
        if good < self.good_to_start:
            return "good"
        if bad < self.bad_to_start:
            return "bad"
        if smart == "green" and stable == "green":
            return "done" if span == "green" else "new"
        return "hard"


def _smart_status(cost_history: list[float], good: int, bad: int) -> str:
    """Error-cost-flatness color (port of ``_compute_smart_status`` shape).

    Faithful to the app's red/yellow/green logic and the ``-0.015`` relative-slope
    flatness threshold, but regresses the **held-out test cost** we already compute
    each step instead of the app's cached-model vote-eval error cost (which would
    require re-implementing the ``_eval_cached_models``/``_live_models`` cache).
    Test-cost flatness is a strictly better "can I stop?" signal, so this is a
    deliberate, documented approximation.
    """
    if good < 5 or bad < 5:
        return "red"
    recent = cost_history[-10:]
    recent = [c for c in recent if np.isfinite(c)]
    if len(recent) < 3:
        return "yellow"
    n = len(recent)
    x_mean = (n - 1) / 2.0
    y_mean = sum(recent) / n
    numer = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denom = sum((i - x_mean) ** 2 for i in range(n))
    slope = numer / denom if denom else 0.0
    relative_slope = slope / y_mean if y_mean > 0 else slope
    return "yellow" if relative_slope < -0.015 else "green"


def _stable_status(flip_history: list[dict], good: int, bad: int) -> str:
    """Prediction-flip-rate color (faithful port of ``_compute_stable_status``).

    ``flip_history`` holds per-step ``{"num_flips", "num_unlabeled"}`` over the
    still-unlabeled pool. Green iff avg flip rate < 0.5% and max < 1% over the
    last 10 steps; red until ≥5 good and ≥5 bad; yellow until ≥5 stability entries.
    """
    if good < 5 or bad < 5:
        return "red"
    if len(flip_history) < 5:
        return "yellow"
    recent = flip_history[-10:]
    rates = [(s["num_flips"] / s["num_unlabeled"]) if s["num_unlabeled"] > 0 else 0.0 for s in recent]
    avg_rate = sum(rates) / len(rates)
    if avg_rate < 0.005 and max(rates) < 0.01:
        return "green"
    return "yellow"


def _span_status(tree) -> str:
    """Diversity-coverage color (faithful port of the span block, :745-780).

    ``green`` on a degenerate/absent tree; otherwise compares the BFS-consecutive
    seen-node count to ``min(_SPAN_GREEN, total_nodes)``.
    """
    if tree is None:
        return "green"
    total = int(tree.total_nodes)
    if total <= 0:
        return "green"
    level = int(tree.coverage_level())
    green_at = min(_SPAN_GREEN, total)
    yellow_at = min(_SPAN_YELLOW, green_at)
    if level >= green_at:
        return "green"
    if level >= yellow_at:
        return "yellow"
    return "red"


def _select_top(pool_ids, pool_scores, labeled, idx):
    """Highest-scoring unlabeled item (port of the ``top`` select mode)."""
    best, best_score = None, float("-inf")
    for i, pid in enumerate(pool_ids):
        if pid in labeled:
            continue
        if pool_scores[i] > best_score:
            best_score, best = pool_scores[i], pid
    return best


def _select_hard(pool_ids, pool_scores, threshold, labeled):
    """Unlabeled item nearest the decision boundary (port of the ``hard`` mode).

    Ranks the full pool descending, finds the first rank whose score ≤ threshold,
    then picks the unlabeled id minimizing ``|rank - thresholdIndex|``. Matches the
    frontend ``autoSelectNext`` boundary pick (label-view.component.ts:1310-1333).
    """
    order = np.argsort(-np.asarray(pool_scores))
    thr_index = len(order)
    for rank, pi in enumerate(order):
        if pool_scores[pi] <= threshold:
            thr_index = rank
            break
    best, best_dist = None, float("inf")
    for rank, pi in enumerate(order):
        pid = pool_ids[int(pi)]
        if pid in labeled:
            continue
        d = abs(rank - thr_index)
        if d < best_dist:
            best_dist, best = d, pid
    return best


def _select_new(tree, pool_ids, pool_scores, threshold, labeled, idx):
    """Diversity pick (``new`` mode) via ``CoverageAtlas.next_sample``; falls back
    to ``hard`` when the tree is absent/exhausted or returns a labeled id."""
    if tree is not None:
        scores_dict = {pid: float(pool_scores[idx[pid]]) for pid in pool_ids}
        cand = tree.next_sample(scores_dict, threshold)
        if cand is not None and cand not in labeled:
            return cand
    return _select_hard(pool_ids, pool_scores, threshold, labeled)


def _select_next(select_mode, tree, pool_ids, pool_scores, threshold, labeled, idx):
    if select_mode == "top":
        return _select_top(pool_ids, pool_scores, labeled, idx)
    if select_mode == "new":
        return _select_new(tree, pool_ids, pool_scores, threshold, labeled, idx)
    return _select_hard(pool_ids, pool_scores, threshold, labeled)  # "hard" (and any fallback)


def _train_pool_head(
    inputs: RegionCurveInputs,
    good_ids: set[int],
    bad_ids: set[int],
    head_kind: str,
    seed: int,
    idx: dict[int, int],
    *,
    inclusion: int,
    safe_thresholds: bool,
    calibrate_count: int,
    cal_fraction: float,
):
    """Train a head on the current good/bad votes; returns
    ``(predict_fn, raw_threshold, n_votes, calib_mode)`` or ``None`` on a
    single-class / empty budget. Region-voting reuses :func:`train_rv_head`
    (snapped positives + leaf-flood bags); otherwise a box-pool/whole MLP."""
    good = sorted(good_ids)
    bad = sorted(bad_ids)
    if not good or not bad:
        return None

    if (inputs.region_voting or inputs.neg_regions) and head_kind == "mlp":
        # Per-image negative bags: region-voting floods childless nodes (leaf_mask); --neg-regions
        # floods all of a bad image's regions (leaf_mask back-fills all-True for sliding/dino/whole).
        pos_rows = np.vstack([inputs.pool_pos_exemplars[i] for i in good]).astype(np.float32)
        neg_bags: list[np.ndarray] = []
        for i in bad:
            mat = inputs.pool_region_mats[idx[i]]
            mask = inputs.pool_leaf_masks[idx[i]]
            bag = mat[mask] if mask is not None and mask.any() else mat
            if bag.shape[0] > 0:
                neg_bags.append(bag)
        if not neg_bags:
            return None
        trained = train_rv_head(
            pos_rows,
            neg_bags,
            inputs.input_dim,
            seed,
            inclusion=inclusion,
            safe_thresholds=safe_thresholds,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
        )
        if trained is None:
            return None
        predict, raw_thr, n_votes = trained
        calib = _calib_mode(len(good), len(neg_bags), safe_thresholds=safe_thresholds)
        return predict, raw_thr, n_votes, calib

    # Box-pool / whole-image MLP path.
    assert inputs.pool_whole_vecs is not None  # populated by build_pool for this path
    pos = np.vstack([inputs.pool_pos_exemplars[i] for i in good]).astype(np.float32)
    neg = np.vstack([inputs.pool_whole_vecs[idx[i]][None, :] for i in bad]).astype(np.float32)
    x = np.vstack([pos, neg]).astype(np.float32)
    y = np.array([1.0] * pos.shape[0] + [0.0] * neg.shape[0], dtype=np.float32)
    head = MLPHead(inputs.input_dim)
    raw_thr = cross_calibrated_threshold(
        x,
        y,
        head.trainer_fn(),
        seed,
        inclusion_value=inclusion,
        calibrate_count=calibrate_count,
        cal_fraction=cal_fraction,
    )
    if not np.isfinite(raw_thr):
        raw_thr = 0.5
    try:
        head.fit(x, y, seed)
    except ValueError:
        return None
    n_votes = len(good) + len(bad)
    calib = _calib_mode(len(good), len(bad), safe_thresholds=safe_thresholds)
    return head.score_rows, float(raw_thr), n_votes, calib


def _cosine_coldstart(inputs: RegionCurveInputs, query_vec: np.ndarray):
    """Single-class cold-start: cosine to the seed exemplar / text query, with a
    GMM threshold over the pool's cosine scores. Returns ``(predict_fn, threshold)``."""
    from vtscore.training.thresholds import calculate_gmm_threshold  # noqa: PLC0415

    head = CosineHead(query_vec)
    pool_scores = max_pool_over_images(head.score_rows, inputs.pool_region_mats)
    finite = [float(s) for s in pool_scores if np.isfinite(s)]
    thr = calculate_gmm_threshold(finite) if finite else 0.5
    if not np.isfinite(thr):
        thr = 0.5
    return head.score_rows, float(thr)


def _row_realistic(
    inputs,
    head_kind,
    t,
    n_good,
    n_bad,
    seed,
    threshold,
    err,
    oracle,
    calib,
    f1,
    mean_iou,
    corloc,
    phase,
    select_mode,
    stop_recommended,
    oracle_extra=None,
) -> dict:
    row = _row(inputs, head_kind, t, n_bad, seed, threshold, err, oracle, calib, f1, mean_iou, corloc, oracle_extra)
    row["n_pos"] = int(n_good)  # override: for the realistic loop n_pos is the good count, not t
    row["t"] = int(t)
    row["n_good"] = int(n_good)
    row["n_bad"] = int(n_bad)
    row["phase"] = phase
    row["select_mode"] = select_mode
    row["stop_recommended"] = bool(stop_recommended)
    return row


def _resolve_step_head(
    inputs,
    good_ids,
    bad_ids,
    head_kind,
    seed,
    idx,
    cold_query,
    cached,
    steps_since_train,
    *,
    retrain_cadence,
    inclusion,
    safe_thresholds,
    calibrate_count,
    cal_fraction,
):
    """Return ``(cached, steps_since_train)`` for one step.

    ``cached`` is ``(predict, raw_thr, n_votes, calib, blend)``. Trains a fresh
    head when both classes exist and a retrain is due (no model yet, cadence
    elapsed, or the current model is a cold-start); otherwise reuses the cache, or
    falls back to the cosine cold-start while only one class is labeled.
    """
    both = bool(good_ids) and bool(bad_ids)
    due = cached is None or steps_since_train >= retrain_cadence or not cached[4]
    if both and due:
        res = _train_pool_head(
            inputs,
            good_ids,
            bad_ids,
            head_kind,
            seed,
            idx,
            inclusion=inclusion,
            safe_thresholds=safe_thresholds,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
        )
        if res is not None:
            predict, raw_thr, n_votes, calib = res
            return (predict, raw_thr, n_votes, calib, True), 0
    if cached is None or not both:
        predict, thr0 = _cosine_coldstart(inputs, cold_query)
        return (predict, thr0, len(good_ids) + len(bad_ids), "cosine_coldstart", False), 0
    return cached, steps_since_train + 1


def _step_indicators(cost_history, flip_history, prev_pred, cur_pred, tree, good, bad):
    """Append this step's prediction-flip entry and return ``(smart, stable, span)``."""
    if prev_pred is not None:
        common = cur_pred.keys() & prev_pred.keys()
        flips = sum(1 for pid in common if cur_pred[pid] != prev_pred[pid])
        flip_history.append({"num_flips": flips, "num_unlabeled": len(common)})
    return (
        _smart_status(cost_history, good, bad),
        _stable_status(flip_history, good, bad),
        _span_status(tree),
    )


def _realistic_setup(inputs: RegionCurveInputs, seed: int, query_vec: np.ndarray | None):
    """Per-seed loop setup: pick the seed positive, the cold-start query vector, and
    build the diversity tree. Returns ``(pool_ids, idx, pool_label, seed_id, cold_query,
    tree)`` or ``None`` when the pool has no positives."""
    pool_ids = list(inputs.pool_ids)
    if not pool_ids:
        return None
    idx = {pid: i for i, pid in enumerate(pool_ids)}
    pool_label = {pid: int(inputs.pool_labels[i]) for i, pid in enumerate(pool_ids)}
    positives = [pid for pid in pool_ids if pool_label[pid] == 1]
    if not positives:
        return None
    seed_id = positives[int(np.random.default_rng(seed).integers(len(positives)))]
    # Cold-start query: text query when available (text embedders), else the seed
    # positive's exemplar (DINO/patch has no text encoder — matches the app's
    # example-seeded cosine cold-start).
    if query_vec is not None:
        cold_query = np.asarray(query_vec, dtype=np.float32)
    else:
        cold_query = np.asarray(inputs.pool_pos_exemplars[seed_id], dtype=np.float32).mean(axis=0)
    # Diversity tree over the pool's whole-image vectors (for the "new" select mode).
    tree = None
    if inputs.pool_whole_vecs is not None and len(pool_ids) >= 2:
        try:
            from vtscore.state.coverage_atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

            vecs = {pid: inputs.pool_whole_vecs[idx[pid]] for pid in pool_ids}
            tree = CoverageAtlas(vecs, k=3, max_depth=auto_max_depth(len(pool_ids), k=3))
        except Exception:
            tree = None
    return pool_ids, idx, pool_label, seed_id, cold_query, tree


def _surface_meta(predict, inputs: RegionCurveInputs, pi: int, select_mode: str, threshold: float) -> dict:
    """Why an item was surfaced: its top region + score (and box) under the current head,
    plus the full per-region score vector and the matched (argmax) region index. These are
    the head-dependent bits the trace carries; the region geometry (boxes, cell masks, HAC
    children, attention) is re-read from the region npz at trace-render time, not stored here."""
    sreg = np.asarray(predict(inputs.pool_region_mats[pi])).reshape(-1)
    bi = int(sreg.argmax())
    box = inputs.pool_region_boxes[pi][bi] if inputs.pool_region_boxes else None
    return {
        "select_mode": select_mode,
        "surface_score": round(float(sreg[bi]), 6),
        "surface_margin": round(float(sreg[bi] - threshold), 6),
        "pred_box": [round(float(v), 6) for v in box] if box is not None else None,
        "region_scores": [round(float(s), 6) for s in sreg],
        "matched_region": bi,
    }


def _realistic_one_seed(
    inputs: RegionCurveInputs,
    head_kind: str,
    seed: int,
    *,
    max_labels: int,
    inclusion: int,
    safe_thresholds: bool,
    calibrate_count: int,
    cal_fraction: float,
    query_vec: np.ndarray | None,
    good_to_start: int,
    bad_to_start: int,
    retrain_cadence: int,
    stop_at_done: bool,
) -> tuple[list[dict], dict | None]:
    """Run one seed's labeling loop. Returns ``(rows, final)`` where ``final`` is a dict
    ``{predict, threshold, n_good, n_bad, t, trace}`` for the last step (head + blended
    threshold the final row measured, plus the per-step trace), or ``None`` when no step ran."""
    from vtscore.training.thresholds import calculate_safe_threshold  # noqa: PLC0415

    setup = _realistic_setup(inputs, seed, query_vec)
    if setup is None:
        return [], None
    pool_ids, idx, pool_label, seed_id, cold_query, tree = setup
    test_labels = [int(v) for v in inputs.test_labels]

    machine = AutopilotPhaseMachine(good_to_start=good_to_start, bad_to_start=bad_to_start)
    good_ids: set[int] = set()
    bad_ids: set[int] = set()
    labeled: set[int] = set()
    cost_history: list[float] = []
    flip_history: list[dict] = []
    prev_pred: dict[int, int] | None = None
    rows: list[dict] = []

    pending = seed_id
    # Metadata for how ``pending`` was surfaced (filled by the previous step's
    # selection under that step's head); the seed is a random cold-start pick, so it
    # carries no head score — only the head-independent region geometry for its trace.
    pending_meta: dict = {
        "select_mode": "seed",
        "surface_score": None,
        "surface_margin": None,
        "pred_box": None,
        "region_scores": None,
        "matched_region": None,
    }
    cached = None  # (predict, raw_thr, n_votes, calib, blend)
    steps_since_train = 0
    trace: list[dict] = []  # per-step labeling record (order, id, phase, head, threshold, …)
    final: dict | None = None  # {predict, threshold, n_good, n_bad, t, trace} of the last step

    for t in range(1, max_labels + 1):
        if pending is None:
            break
        labeled_id = pending
        (good_ids if pool_label[labeled_id] == 1 else bad_ids).add(labeled_id)
        labeled.add(labeled_id)
        if tree is not None:
            tree.label(labeled_id, pool_label[labeled_id] == 1)

        cached, steps_since_train = _resolve_step_head(
            inputs,
            good_ids,
            bad_ids,
            head_kind,
            seed,
            idx,
            cold_query,
            cached,
            steps_since_train,
            retrain_cadence=retrain_cadence,
            inclusion=inclusion,
            safe_thresholds=safe_thresholds,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
        )
        predict, raw_thr, n_votes, calib, blend = cached

        test_scores, test_argmax = max_pool_with_argmax(predict, inputs.test_region_mats)
        pool_scores = max_pool_over_images(predict, inputs.pool_region_mats)
        threshold = (
            calculate_safe_threshold(raw_thr, [float(s) for s in pool_scores], n_votes)
            if (blend and safe_thresholds)
            else raw_thr
        )

        err = weighted_error(test_scores, [float(v) for v in test_labels], threshold, inclusion)
        oracle = min_weighted_cost(test_scores, [float(v) for v in test_labels], inclusion)
        oracle_extra = _oracle_extra(test_scores, test_labels)
        f1 = f1_at(test_scores, test_labels, threshold)
        mean_iou, corloc = _iou_metrics(inputs, test_argmax)
        cost_history.append(err["cost"])

        cur_pred = {pid: int(pool_scores[idx[pid]] >= threshold) for pid in pool_ids if pid not in labeled}
        good, bad = len(good_ids), len(bad_ids)
        smart, stable, span = _step_indicators(cost_history, flip_history, prev_pred, cur_pred, tree, good, bad)
        prev_pred = cur_pred
        phase = machine.next_phase(good, bad, smart, stable, span)
        select_mode = _PHASE_TO_SELECT[phase]
        stop_recommended = phase == "done"

        rows.append(
            _row_realistic(
                inputs,
                head_kind,
                t,
                good,
                bad,
                seed,
                threshold,
                err,
                oracle,
                calib,
                f1,
                mean_iou,
                corloc,
                phase,
                select_mode,
                stop_recommended,
                oracle_extra=oracle_extra,
            )
        )
        # Per-step labeling record: how the item was surfaced (from pending_meta,
        # the previous step's head) + the state after labeling it.
        trace.append(
            {
                "t": t,
                "image_id": int(labeled_id),
                "gt_label": "good" if pool_label[labeled_id] == 1 else "bad",
                "select_mode": pending_meta["select_mode"],
                "phase": phase,
                "head": "cosine" if calib == "cosine_coldstart" else "mlp",
                "calib_mode": calib,
                "threshold": round(float(threshold), 6),
                "surface_score": pending_meta["surface_score"],
                "surface_margin": pending_meta["surface_margin"],
                "pred_box": pending_meta["pred_box"],
                "region_scores": pending_meta.get("region_scores"),
                "matched_region": pending_meta.get("matched_region"),
                "n_good": good,
                "n_bad": bad,
                "n_votes": n_votes,
                "smart": smart,
                "stable": stable,
                "span": span,
                "cost": err["cost"],
                "fpr": err["fpr"],
                "fnr": err["fnr"],
                "f1": f1,
                "stop_recommended": stop_recommended,
            }
        )
        final = {
            "predict": predict,
            "threshold": float(threshold),
            "n_good": good,
            "n_bad": bad,
            "t": t,
            "trace": trace,
        }

        if stop_recommended and stop_at_done:
            break
        pending = _select_next(select_mode, tree, pool_ids, pool_scores, threshold, labeled, idx)
        # Record why the next item was surfaced, to attach to its trace entry next step.
        if pending is not None:
            pending_meta = _surface_meta(predict, inputs, idx[pending], select_mode, threshold)

    return rows, final


def evaluate_realistic_curve(
    inputs: RegionCurveInputs,
    head_kind: str,
    *,
    seeds: Sequence[int],
    max_labels: int = 60,
    inclusion: int = 0,
    safe_thresholds: bool = True,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
    query_vec: np.ndarray | None = None,
    select_strategy: str = "autopilot",
    good_to_start: int = 3,
    bad_to_start: int = 4,
    retrain_cadence: int = 1,
    stop_at_done: bool = False,
    return_finals: bool = False,
) -> list[dict] | tuple[list[dict], dict[int, dict]]:
    """Realistic active-learning labeling curve: cost/fpr/fnr/F1/IoU vs ``t`` (total
    annotations), one row per ``(seed, t)``.

    Simulates the app's Autopilot loop over ``inputs``' training pool: seed one
    positive, cold-start rank (cosine-to-exemplar for text-less DINO/patch, else
    the text query), then per step select the next item by the current phase's mode
    (good→top, bad/hard→boundary, new→diversity), reveal its ground-truth label,
    retrain from scratch, re-score, and record. Only ``select_strategy="autopilot"``
    is wired today (the selection layer is factored so ``top``/``hard``/``new`` can
    be exposed later).

    When ``return_finals`` is set, returns ``(rows, finals)`` where ``finals`` maps each
    seed to a dict ``{predict, threshold, n_good, n_bad, t, trace}`` — the in-process
    final head + blended threshold (e.g. for prediction overlays at the max ``t`` the
    loop reached) plus the full per-step ``trace`` (order, labeled id, gt label, select
    mode, phase, head, calib_mode, threshold, surface score/margin, pred_box, indicators,
    metrics). Default (``False``) returns just ``rows``, preserving the plain-list
    contract every other caller relies on.
    """
    if select_strategy != "autopilot":
        raise ValueError(f"unsupported select_strategy {select_strategy!r} (only 'autopilot' is wired)")
    rows: list[dict] = []
    finals: dict[int, dict] = {}
    for seed in seeds:
        t0 = time.perf_counter()
        seed_rows, final = _realistic_one_seed(
            inputs,
            head_kind,
            int(seed),
            max_labels=max_labels,
            inclusion=inclusion,
            safe_thresholds=safe_thresholds,
            calibrate_count=calibrate_count,
            cal_fraction=cal_fraction,
            query_vec=query_vec,
            good_to_start=good_to_start,
            bad_to_start=bad_to_start,
            retrain_cadence=retrain_cadence,
            stop_at_done=stop_at_done,
        )
        # Amortize the seed's wall-clock over its rows (per-step retrain dominates).
        per = round((time.perf_counter() - t0) * 1000.0 / max(len(seed_rows), 1), 3)
        for r in seed_rows:
            r["compute_ms"] = per
        rows.extend(seed_rows)
        if final is not None:
            finals[int(seed)] = final
    return (rows, finals) if return_finals else rows

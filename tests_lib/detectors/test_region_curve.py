"""Tests for the small-object-detection sweep core (vtscore.eval.*).

Covers the shared error metrics, the scoring heads, and the K-curve. All
synthetic (no models), so these run in the fast CPU suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.error_metrics import f1_at, inclusion_weights, min_weighted_cost, weighted_error
from vtscore.eval.region_curve import (
    AutopilotPhaseMachine,
    RegionCurveInputs,
    _flood_context_np,
    _iou_metrics,
    _select_hard,
    _select_new,
    _span_status,
    evaluate_realistic_curve,
    evaluate_region_curve,
    sample_rv_budget,
    train_rv_head,
)
from vtscore.eval.region_sources import _covering_box
from vtscore.eval.scoring_heads import CosineHead, MLPHead, max_pool_over_images


# --------------------------------------------------------------------------
# error_metrics
# --------------------------------------------------------------------------


def test_inclusion_weights():
    assert inclusion_weights(0) == (1.0, 1.0)
    assert inclusion_weights(2) == (1.0, 4.0)
    assert inclusion_weights(-1) == (2.0, 1.0)


def test_weighted_error_hand_computed():
    # scores/labels; threshold 0.5 -> preds [1,1,0,0]; pos=2 neg=2; FP=1, FN=1.
    scores = [0.9, 0.8, 0.4, 0.1]
    labels = [1.0, 0.0, 1.0, 0.0]
    r = weighted_error(scores, labels, 0.5, 0)
    assert r == {"cost": 1.0, "fpr": 0.5, "fnr": 0.5}
    # inclusion 1 up-weights FNR: cost = 0.5 + 2*0.5 = 1.5
    assert weighted_error(scores, labels, 0.5, 1)["cost"] == 1.5


def test_weighted_error_empty():
    assert np.isnan(weighted_error([], [], 0.5, 0)["cost"])
    assert np.isnan(min_weighted_cost([], [], 0))


def test_min_weighted_cost_is_oracle_lower_bound():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 40
        scores = rng.random(n)
        labels = (rng.random(n) > 0.5).astype(float)
        if labels.sum() == 0 or labels.sum() == n:
            continue
        oracle = min_weighted_cost(scores, labels, 0)
        # No single threshold can beat the oracle min.
        for t in np.unique(scores):
            assert weighted_error(scores, labels, float(t), 0)["cost"] >= oracle - 1e-6


def test_min_weighted_cost_separable_is_zero():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1.0, 1.0, 0.0, 0.0]
    assert min_weighted_cost(scores, labels, 0) == 0.0


def test_f1_at_hand_computed():
    # thr 0.5 -> preds [1,1,0,0]; TP=1, FP=1, FN=1 -> F1 = 2*1/(2*1+1+1) = 0.5
    assert f1_at([0.9, 0.8, 0.4, 0.1], [1.0, 0.0, 1.0, 0.0], 0.5) == 0.5
    assert np.isnan(f1_at([], [], 0.5))
    # perfect separation at 0.5
    assert f1_at([0.9, 0.1], [1.0, 0.0], 0.5) == 1.0


def _boxes_inputs(labels, region_boxes, gt_boxes) -> RegionCurveInputs:
    z = np.zeros((0, 4), np.float32)
    return RegionCurveInputs(z, z, [], labels, 4, test_region_boxes=region_boxes, test_gt_boxes=gt_boxes)


def test_iou_metrics_direct():
    # pos 0: argmax box exactly overlaps GT -> IoU 1.0; pos 1: argmax box disjoint -> 0.0.
    region_boxes = [
        np.array([[0.0, 0.0, 0.5, 0.5], [0.9, 0.9, 1.0, 1.0]], np.float32),
        np.array([[0.9, 0.9, 1.0, 1.0], [0.0, 0.0, 0.5, 0.5]], np.float32),
    ]
    gt = [[(0.0, 0.0, 0.5, 0.5)], [(0.0, 0.0, 0.5, 0.5)]]
    inp = _boxes_inputs([1, 1], region_boxes, gt)
    mean_iou, corloc = _iou_metrics(inp, np.array([0, 0]))  # both pick region 0
    assert mean_iou == 0.5  # (1.0 + 0.0)/2
    assert corloc == 0.5  # one of two hits IoU>=0.5


def test_iou_metrics_nan_without_boxes():
    inp = RegionCurveInputs(np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32), [], [1, 0], 4)
    mean_iou, corloc = _iou_metrics(inp, np.array([0, 0]))
    assert np.isnan(mean_iou) and np.isnan(corloc)


def test_region_curve_row_has_f1_iou_fields():
    inputs, _ = _inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[4], seeds=[0], neg_ratio=1)
    assert rows and {"f1", "mean_iou", "corloc"} <= set(rows[0])
    # no boxes provided in _inputs() -> IoU fields are NaN, F1 is finite
    assert np.isnan(rows[0]["mean_iou"])


def test_weighted_error_matches_legacy_inline():
    """weighted_error must reproduce voting_iterations' old FP/FN/cost block."""
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = 25
        scores = rng.random(n).tolist()
        labels = (rng.random(n) > 0.4).astype(float).tolist()
        thr = float(rng.random())
        incl = int(rng.integers(-2, 3))
        total_pos = sum(1 for x in labels if x == 1.0)
        total_neg = len(labels) - total_pos
        fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= thr and y == 0.0)
        fn = sum(1 for s, y in zip(scores, labels, strict=True) if s < thr and y == 1.0)
        fpr = fp / total_neg if total_neg else 0.0
        fnr = fn / total_pos if total_pos else 0.0
        wfpr, wfnr = inclusion_weights(incl)
        expect = round(wfpr * fpr + wfnr * fnr, 6)
        assert weighted_error(scores, labels, thr, incl)["cost"] == expect


# --------------------------------------------------------------------------
# scoring heads
# --------------------------------------------------------------------------


def _blobs(dim=16, n=10, seed=0):
    rng = np.random.default_rng(seed)
    cpos = np.zeros(dim, np.float32)
    cpos[0] = 1.0
    cneg = np.zeros(dim, np.float32)
    cneg[1] = 1.0

    def blob(c, m):
        v = c + 0.1 * rng.standard_normal((m, dim)).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    return blob(cpos, n), blob(cneg, n), cpos


def test_cosine_head_separates():
    pos, neg, q = _blobs()
    head = CosineHead(q)
    mats = [p[None, :] for p in pos] + [n[None, :] for n in neg]
    scores = max_pool_over_images(head.score_rows, mats)
    assert scores[: len(pos)].mean() > scores[len(pos) :].mean() + 0.5


def test_mlp_head_ranks_separable_data():
    pos, neg, _ = _blobs(n=12)
    x = np.vstack([pos, neg])
    y = np.array([1.0] * 12 + [0.0] * 12, dtype=np.float32)
    head = MLPHead(x.shape[1])
    head.fit(x, y, seed=0)
    mats = [p[None, :] for p in pos] + [n[None, :] for n in neg]
    scores = max_pool_over_images(head.score_rows, mats)
    labels = [1] * 12 + [0] * 12
    # The head learns the right direction; its ranking is well better than chance,
    # even though tiny-data MLP outputs sit compressed near 0.5.
    assert scores[:12].mean() > scores[12:].mean()
    assert min_weighted_cost(scores, labels, 0) < 0.45


def test_max_pool_takes_best_region():
    head = CosineHead(np.array([1.0, 0.0], np.float32))
    # image with a bad region and a good region -> max should pick the good one.
    mats = [np.array([[0.0, 1.0], [1.0, 0.0]], np.float32)]
    assert max_pool_over_images(head.score_rows, mats)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# region_curve
# --------------------------------------------------------------------------


def _inputs(seed=0, dim=24):
    rng = np.random.default_rng(seed)
    cpos = np.zeros(dim, np.float32)
    cpos[0] = 1.0
    cneg = np.zeros(dim, np.float32)
    cneg[1] = 1.0

    def blob(c, m):
        v = c + 0.15 * rng.standard_normal((m, dim)).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    pos_ex = blob(cpos, 40)
    neg_tr = blob(cneg, 100)
    test_mats = [blob(cpos, 2) for _ in range(15)] + [blob(cneg, 2) for _ in range(45)]
    test_labels = [1] * 15 + [0] * 45
    return RegionCurveInputs(pos_ex, neg_tr, test_mats, test_labels, dim, meta={"embedder": "syn"}), cpos


def test_region_curve_mlp_rows_and_fields():
    inputs, _ = _inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[2, 8], seeds=[0, 1], neg_ratio=1)
    assert rows
    assert {r["k"] for r in rows} == {2, 8}
    for r in rows:
        assert set(r) >= {"k", "cost", "fpr", "fnr", "oracle_cost", "threshold", "calib_mode", "head"}
        assert r["embedder"] == "syn"


def test_region_curve_realistic_never_below_oracle():
    inputs, _ = _inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[2, 4, 8, 16], seeds=[0, 1, 2], neg_ratio=1)
    for r in rows:
        if not np.isnan(r["cost"]) and not np.isnan(r["oracle_cost"]):
            assert r["cost"] >= r["oracle_cost"] - 1e-6


def test_region_curve_cosine_has_k0_baseline():
    inputs, q = _inputs()
    rows = evaluate_region_curve(inputs, "cosine", k_values=[0, 4], seeds=[0], neg_ratio=1, query_vec=q)
    k0 = [r for r in rows if r["k"] == 0]
    assert len(k0) == 1
    assert k0[0]["calib_mode"] == "none_k0"


def test_region_curve_mlp_has_no_k0():
    inputs, _ = _inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[0, 4], seeds=[0], neg_ratio=1)
    assert all(r["k"] != 0 for r in rows)  # MLP needs >= 1 positive


def test_region_curve_cosine_requires_query():
    inputs, _ = _inputs()
    with pytest.raises(ValueError, match="query_vec"):
        evaluate_region_curve(inputs, "cosine", k_values=[4], seeds=[0])


def test_region_curve_calib_mode_transitions():
    inputs, _ = _inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[1, 16], seeds=[0], neg_ratio=1)
    modes = {r["k"]: r["calib_mode"] for r in rows}
    assert modes[1] == "fallback"  # K<2 -> cross-calibration can't form folds
    assert modes[16] == "xcal"  # n_labels=32 -> pure cross-calibration


# --------------------------------------------------------------------------
# region-voting (DINO-patch faithful path)
# --------------------------------------------------------------------------


def _rv_inputs(seed=0, dim=24):
    """RegionCurveInputs in region-voting mode: one snapped positive per image,
    negatives as per-image bags of CLS+leaf vectors. neg_train_wholes is empty on
    purpose — the rv path must ignore it and use the bags."""
    rng = np.random.default_rng(seed)
    cpos = np.zeros(dim, np.float32)
    cpos[0] = 1.0
    cneg = np.zeros(dim, np.float32)
    cneg[1] = 1.0

    def blob(c, m):
        v = c + 0.15 * rng.standard_normal((m, dim)).astype(np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    pos_ex = blob(cpos, 40)  # 40 positive images, one snapped vector each
    neg_bags = [blob(cneg, int(rng.integers(3, 7))) for _ in range(100)]  # varied bag sizes
    test_mats = [blob(cpos, 3) for _ in range(15)] + [blob(cneg, 3) for _ in range(45)]
    test_labels = [1] * 15 + [0] * 45
    inputs = RegionCurveInputs(
        pos_ex,
        np.zeros((0, dim), np.float32),
        test_mats,
        test_labels,
        dim,
        meta={"embedder": "syn"},
        region_voting=True,
        neg_train_bags=neg_bags,
    )
    return inputs, cpos


def test_covering_box():
    assert _covering_box([(0.1, 0.2, 0.3, 0.4), (0.5, 0.1, 0.7, 0.9)]) == (0.1, 0.1, 0.7, 0.9)
    assert _covering_box([(0.2, 0.2, 0.6, 0.6)]) == (0.2, 0.2, 0.6, 0.6)


def test_flood_context_bag_aware_weights():
    # 2 good (1 row each) + 1 bad image flooded into 4 rows.
    x = [np.zeros(4, np.float32)] * 6
    y = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    groups = [("g", 0), ("g", 1), ("b", 0), ("b", 0), ("b", 0), ("b", 0)]
    n_votes, cal_groups, sw = _flood_context_np(x, y, groups)
    assert n_votes == 3  # 2 good bags + 1 bad bag, NOT 6 rows
    assert cal_groups is groups  # flooded -> calibrate by bag
    assert sw is not None
    # Each bad row carries 1/4 of one image's negative mass.
    assert np.allclose(np.asarray(sw)[2:], 0.25)


def test_flood_context_not_flooded_is_noop():
    x = [np.zeros(4, np.float32)] * 3
    y = [1.0, 0.0, 0.0]
    groups = [("g", 0), ("b", 1), ("b", 2)]  # one row per bag
    n_votes, cal_groups, sw = _flood_context_np(x, y, groups)
    assert n_votes == 3 and cal_groups is None and sw is None


def test_sample_rv_budget():
    inputs, _ = _rv_inputs()
    b = sample_rv_budget(inputs.pos_exemplars, inputs.neg_train_bags, 4, 2, 0)
    assert b is not None
    pos_rows, bags = b
    assert pos_rows.shape[0] == 4 and len(bags) == 8  # neg_ratio*k negative images
    assert sample_rv_budget(inputs.pos_exemplars, inputs.neg_train_bags, 999, 1, 0) is None  # K > positives


def test_train_rv_head_counts_votes_by_bag():
    inputs, _ = _rv_inputs()
    pos_rows, bags = sample_rv_budget(inputs.pos_exemplars, inputs.neg_train_bags, 8, 1, 0)
    out = train_rv_head(
        pos_rows, bags, inputs.input_dim, 0, inclusion=0, safe_thresholds=True, calibrate_count=2, cal_fraction=0.5
    )
    assert out is not None
    predict, thr, n_votes = out
    assert n_votes == 16  # 8 good images + 8 bad images (NOT the flooded leaf rows)
    s = predict(inputs.test_region_mats[0])
    assert s.shape[0] == inputs.test_region_mats[0].shape[0]


def test_region_curve_region_voting_rows_and_fields():
    inputs, _ = _rv_inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[2, 8], seeds=[0, 1], neg_ratio=1)
    assert rows
    assert {r["k"] for r in rows} == {2, 8}
    for r in rows:
        assert set(r) >= {"k", "cost", "fpr", "fnr", "oracle_cost", "threshold", "calib_mode", "head", "compute_ms"}
        assert r["head"] == "mlp"


def test_region_curve_region_voting_realistic_never_below_oracle():
    inputs, _ = _rv_inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[2, 8, 16], seeds=[0, 1], neg_ratio=2)
    for r in rows:
        if not np.isnan(r["cost"]) and not np.isnan(r["oracle_cost"]):
            assert r["cost"] >= r["oracle_cost"] - 1e-6


def test_region_curve_region_voting_separates():
    inputs, _ = _rv_inputs()
    rows = evaluate_region_curve(inputs, "mlp", k_values=[16], seeds=[0, 1, 2], neg_ratio=2)
    # Separable blobs -> the oracle operating point should be clearly good.
    assert min(r["oracle_cost"] for r in rows) < 0.3


# --------------------------------------------------------------------------
# realistic labeling loop (Autopilot active-learning port)
# --------------------------------------------------------------------------


def _realistic_inputs(seed=0, dim=16, n_pos=30, n_neg=90):
    """RegionCurveInputs with a training pool for the realistic loop: separable
    positive/negative blobs, region-voting mode (one snapped exemplar per positive
    image, per-image leaf bags for negatives)."""
    rng = np.random.default_rng(seed)
    cpos = np.zeros(dim, np.float32)
    cpos[0] = 1.0
    cneg = np.zeros(dim, np.float32)
    cneg[1] = 1.0

    def blob(c, m):
        v = c + 0.15 * rng.standard_normal((m, dim)).astype(np.float32)
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

    pool_ids: list[int] = []
    pool_region_mats: list[np.ndarray] = []
    pool_region_boxes: list[np.ndarray] = []
    pool_whole: list[np.ndarray] = []
    pool_leaf: list[np.ndarray] = []
    pool_labels: list[int] = []
    pool_pos_ex: dict[int, np.ndarray] = {}

    def boxes(m):
        return np.tile(np.array([0.1, 0.1, 0.5, 0.5], np.float32), (m, 1))

    for i in range(n_pos):
        reg = blob(cpos, int(rng.integers(3, 6)))
        pool_ids.append(i)
        pool_region_mats.append(reg)
        pool_region_boxes.append(boxes(reg.shape[0]))
        pool_whole.append(reg.mean(0))
        pool_leaf.append(np.ones(reg.shape[0], dtype=bool))
        pool_labels.append(1)
        pool_pos_ex[i] = blob(cpos, 1)  # snapped covering-box exemplar (1, D)
    for j in range(n_neg):
        iid = n_pos + j
        reg = blob(cneg, int(rng.integers(3, 6)))
        pool_ids.append(iid)
        pool_region_mats.append(reg)
        pool_region_boxes.append(boxes(reg.shape[0]))
        pool_whole.append(reg.mean(0))
        pool_leaf.append(np.ones(reg.shape[0], dtype=bool))
        pool_labels.append(0)

    test_mats = [blob(cpos, 3) for _ in range(15)] + [blob(cneg, 3) for _ in range(45)]
    test_labels = [1] * 15 + [0] * 45
    return RegionCurveInputs(
        pos_exemplars=np.vstack(list(pool_pos_ex.values())),
        neg_train_wholes=np.zeros((0, dim), np.float32),
        test_region_mats=test_mats,
        test_labels=test_labels,
        input_dim=dim,
        meta={"embedder": "syn", "dataset": "syn", "class": "c", "proposal": "hac"},
        region_voting=True,
        pool_ids=pool_ids,
        pool_region_mats=pool_region_mats,
        pool_region_boxes=pool_region_boxes,
        pool_whole_vecs=np.vstack(pool_whole).astype(np.float32),
        pool_leaf_masks=pool_leaf,
        pool_labels=pool_labels,
        pool_pos_exemplars=pool_pos_ex,
    )


def test_phase_machine_table():
    m = AutopilotPhaseMachine(good_to_start=3, bad_to_start=4)
    # Below the good/bad floors -> good then bad, regardless of indicators.
    assert m.next_phase(0, 0, "green", "green", "green") == "good"
    assert m.next_phase(2, 9, "green", "green", "green") == "good"
    assert m.next_phase(3, 0, "green", "green", "green") == "bad"
    assert m.next_phase(5, 3, "green", "green", "green") == "bad"
    # Past both floors: indicators drive hard/new/done.
    assert m.next_phase(5, 5, "yellow", "green", "green") == "hard"
    assert m.next_phase(5, 5, "green", "yellow", "green") == "hard"
    assert m.next_phase(5, 5, "green", "green", "yellow") == "new"
    assert m.next_phase(5, 5, "green", "green", "green") == "done"


def test_realistic_t_monotonic_and_emergent_mix():
    inputs = _realistic_inputs()
    rows = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=14)
    ts = [r["t"] for r in rows]
    assert ts == list(range(1, len(rows) + 1))  # one row per t, strictly 1..N
    for r in rows:
        assert r["n_good"] + r["n_bad"] == r["t"]  # emergent mix, sums to total
        assert r["k"] == r["t"]  # k mirrors t so existing plotting keys work
        assert np.isfinite(r["cost"])
    # The mix is emergent, not a fixed neg_ratio: at least one bad appears and the
    # good:bad ratio is not a constant integer multiple across the curve.
    assert rows[-1]["n_bad"] >= 1 and rows[-1]["n_good"] >= 1


def test_realistic_coldstart_single_class():
    inputs = _realistic_inputs()
    rows = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=6)
    # t=1 seeds a single positive -> cosine cold-start, finite, no crash.
    assert rows[0]["t"] == 1
    assert rows[0]["n_good"] == 1 and rows[0]["n_bad"] == 0
    assert rows[0]["calib_mode"] == "cosine_coldstart"
    assert np.isfinite(rows[0]["cost"])


def test_realistic_region_voting_engages():
    inputs = _realistic_inputs()
    rows = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=16)
    # Once both classes exist the MLP (region-voting) path trains -> a non-cosine
    # calib_mode appears on at least one row.
    assert any(r["calib_mode"] != "cosine_coldstart" for r in rows)


def test_realistic_deterministic():
    inputs = _realistic_inputs()
    r1 = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=8)
    r2 = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=8)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2, strict=True):
        assert (a["t"], a["n_good"], a["n_bad"], a["phase"], a["select_mode"]) == (
            b["t"],
            b["n_good"],
            b["n_bad"],
            b["phase"],
            b["select_mode"],
        )
        assert np.allclose(a["cost"], b["cost"], atol=1e-9)


def test_realistic_oracle_floor():
    inputs = _realistic_inputs()
    rows = evaluate_realistic_curve(inputs, "mlp", seeds=[0, 1], max_labels=12)
    for r in rows:
        if np.isfinite(r["cost"]) and np.isfinite(r["oracle_cost"]):
            assert r["cost"] >= r["oracle_cost"] - 1e-6


def test_realistic_unsupported_strategy_raises():
    inputs = _realistic_inputs()
    with pytest.raises(ValueError, match="select_strategy"):
        evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=4, select_strategy="hard")


def test_select_hard_picks_near_threshold():
    # Descending scores; threshold falls between ranks -> nearest-rank unlabeled wins.
    pool_ids = [10, 11, 12, 13, 14]
    pool_scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    # thresholdIndex = first rank with score <= 0.55 -> rank 2 (id 12). Nearest
    # unlabeled to rank 2 is id 12 itself.
    assert _select_hard(pool_ids, pool_scores, 0.55, labeled=set()) == 12
    # If 12 is already labeled, the next-nearest ranks (11 @1, 13 @3) tie -> the
    # lower rank (11) wins on the argsort order.
    assert _select_hard(pool_ids, pool_scores, 0.55, labeled={12}) == 11


def test_select_new_uses_diversity_tree():
    from vtscore.state.diversity_tree import DiversityTree, auto_max_depth

    inputs = _realistic_inputs(n_pos=40, n_neg=80)
    vecs = {pid: inputs.pool_whole_vecs[i] for i, pid in enumerate(inputs.pool_ids)}
    tree = DiversityTree(vecs, k=3, max_depth=auto_max_depth(len(inputs.pool_ids), k=3))
    idx = {pid: i for i, pid in enumerate(inputs.pool_ids)}
    scores = np.zeros(len(inputs.pool_ids), dtype=np.float64)
    level0 = tree.diversity_level()
    pick = _select_new(tree, inputs.pool_ids, scores, 0.5, labeled=set(), idx=idx)
    assert pick in inputs.pool_ids
    tree.label(pick)
    assert tree.diversity_level() >= level0  # labeling a fresh cluster never lowers coverage


def test_span_status_degenerate_tree_is_green():
    assert _span_status(None) == "green"


def test_realistic_return_finals():
    inputs = _realistic_inputs()
    rows, finals = evaluate_realistic_curve(inputs, "mlp", seeds=[0, 1], max_labels=10, return_finals=True)
    assert isinstance(rows, list) and rows
    assert set(finals) == {0, 1}  # one final per seed
    for seed in (0, 1):
        fin = finals[seed]
        # predict is a callable head that scores a region matrix to one score per row.
        mat = inputs.test_region_mats[0]
        scores = np.asarray(fin["predict"](mat))
        assert scores.shape[0] == mat.shape[0]
        assert np.isfinite(fin["threshold"])
        # final matches that seed's LAST row.
        last = [r for r in rows if r["seed"] == seed][-1]
        assert (fin["t"], fin["n_good"], fin["n_bad"]) == (last["t"], last["n_good"], last["n_bad"])


def test_realistic_trace_contract():
    inputs = _realistic_inputs()
    _rows, finals = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=12, return_finals=True)
    trace = finals[0]["trace"]
    # one entry per step, in order 1..t, aligned with the final t.
    assert [e["t"] for e in trace] == list(range(1, len(trace) + 1))
    assert len(trace) == finals[0]["t"]
    first = trace[0]
    # the seed step is a random cold-start pick: no surfacing score/box, select_mode 'seed'.
    assert first["select_mode"] == "seed"
    assert first["surface_score"] is None and first["pred_box"] is None
    assert first["gt_label"] == "good"  # loop always seeds a positive
    for e in trace:
        assert e["gt_label"] in ("good", "bad")
        assert e["head"] in ("cosine", "mlp")
        assert e["phase"] in ("good", "bad", "hard", "new", "done")
        assert np.isfinite(e["threshold"])
        assert e["n_good"] + e["n_bad"] == e["t"]
    # once both classes exist, later steps surface via a real head → a pred_box appears.
    assert any(e["pred_box"] is not None and e["surface_score"] is not None for e in trace)


def test_realistic_return_finals_default_is_plain_list():
    inputs = _realistic_inputs()
    out = evaluate_realistic_curve(inputs, "mlp", seeds=[0], max_labels=6)
    assert isinstance(out, list)  # default (return_finals=False) preserves the list contract

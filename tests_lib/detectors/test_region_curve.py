"""Tests for the small-object-detection sweep core (vtscore.eval.*).

Covers the shared error metrics, the scoring heads, and the K-curve. All
synthetic (no models), so these run in the fast CPU suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.error_metrics import inclusion_weights, min_weighted_cost, weighted_error
from vtscore.eval.region_curve import RegionCurveInputs, evaluate_region_curve
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

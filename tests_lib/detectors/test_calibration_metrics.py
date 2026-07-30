"""Unit tests for the torch-free calibration metrics + pooling (issue #2781).

These exercise the pure-numpy core of the calibration study: the oracle cut
(checked against an independent brute-force sweep), the operating cost, the
degenerate/percentile helpers, and the three pooling variants.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.calibration_metrics import (
    inclusion_weights,
    is_degenerate,
    negative_block_null,
    operating_cost,
    oracle_cut,
    pool_blocks,
    pool_segment,
    segment_counts,
    segment_max_pool,
    segment_pnorm_pool,
    segment_topk_mean_pool,
    threshold_percentile,
)


def _brute_oracle(scores, labels, wf, wn):
    """Independent O(n^2) oracle: try every score (and predict-nothing) as the cut."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    cands = list(np.unique(scores)) + [float(scores.max()) + 1.0]
    best = None
    for t in cands:
        cost, fpr, fnr = operating_cost(scores, labels, t, wf, wn)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, fpr, fnr, t)
    assert best is not None
    return best


@pytest.mark.parametrize("seed", range(25))
def test_oracle_cut_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    scores = rng.normal(size=n).round(2)  # rounding forces ties
    labels = (rng.random(n) < 0.4).astype(float)
    if labels.sum() == 0 or labels.sum() == n:  # keep both classes present
        labels[0], labels[-1] = 1.0, 0.0
    for wf, wn in [(1.0, 1.0), (1.0, 2.0), (0.5, 1.0)]:
        thr, cost, fpr, fnr = oracle_cut(scores, labels, wf, wn)
        b_cost = _brute_oracle(scores, labels, wf, wn)[0]
        assert cost == pytest.approx(b_cost, abs=1e-9)
        # The returned threshold must actually realise the reported cost.
        c2, fpr2, fnr2 = operating_cost(scores, labels, thr, wf, wn)
        assert c2 == pytest.approx(cost, abs=1e-9)
        assert fpr2 == pytest.approx(fpr, abs=1e-9)
        assert fnr2 == pytest.approx(fnr, abs=1e-9)


def test_oracle_never_worse_than_trained():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (scores + rng.normal(0, 0.3, 200) > 0.5).astype(float)
    wf, wn = 1.0, 1.0
    _, oracle_cost, _, _ = oracle_cut(scores, labels, wf, wn)
    for t in np.linspace(0, 1, 21):
        trained_cost, _, _ = operating_cost(scores, labels, t, wf, wn)
        assert oracle_cost <= trained_cost + 1e-12


def test_perfectly_separable_oracle_is_zero():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    thr, cost, fpr, fnr = oracle_cut(scores, labels, 1.0, 1.0)
    assert cost == pytest.approx(0.0)
    assert 0.3 < thr <= 0.8
    # And the trained cost at that oracle threshold is also zero.
    assert operating_cost(scores, labels, thr, 1.0, 1.0)[0] == pytest.approx(0.0)


def test_operating_cost_and_weights():
    assert inclusion_weights(0) == (1.0, 1.0)
    assert inclusion_weights(1) == (1.0, 2.0)
    assert inclusion_weights(-1) == (2.0, 1.0)
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    # threshold 0.5: predict [1,1,0,0] -> fp=1 (0.8/neg), fn=1 (0.2/pos)
    cost, fpr, fnr = operating_cost(scores, labels, 0.5, 1.0, 1.0)
    assert fpr == pytest.approx(0.5)
    assert fnr == pytest.approx(0.5)
    assert cost == pytest.approx(1.0)


def test_degenerate_and_percentile():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    assert is_degenerate(scores, 1.5) is True  # above max -> all-negative
    assert is_degenerate(scores, 0.05) is True  # below min -> all-positive
    assert is_degenerate(scores, 0.5) is False
    # The #2781 runaway signature: cut above max -> FNR 1, FPR 0.
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    cost, fpr, fnr = operating_cost(scores, labels, 1.5, 1.0, 1.0)
    assert (fpr, fnr, cost) == (0.0, 1.0, 1.0)
    assert threshold_percentile(scores, 1.5) == pytest.approx(1.0)
    assert threshold_percentile(scores, 0.0) == pytest.approx(0.0)
    assert threshold_percentile(scores, 0.5) == pytest.approx(0.5)


def test_segment_helpers_and_counts():
    # three images with 2, 3, 1 nodes
    flat = np.array([0.1, 0.9, 0.5, 0.4, 0.2, 0.7])
    seg = np.array([0, 2, 5])
    assert segment_counts(seg, flat.size).tolist() == [2, 3, 1]
    assert segment_max_pool(flat, seg).tolist() == pytest.approx([0.9, 0.5, 0.7])


def test_topk_mean_pool():
    flat = np.array([0.1, 0.9, 0.5, 0.4, 0.2, 0.8, 0.7])
    seg = np.array([0, 2])  # two images: sizes 2 and 5
    out = segment_topk_mean_pool(flat, seg, k=4)
    # img0 has 2 nodes < k -> mean of both = 0.5
    assert out[0] == pytest.approx(0.5)
    # img1 top-4 of [0.5,0.4,0.2,0.8,0.7] = {0.8,0.7,0.5,0.4} mean = 0.6
    assert out[1] == pytest.approx(0.6)
    # k>=N collapses to the plain mean; k=1 collapses to the max.
    assert segment_topk_mean_pool(flat, seg, k=1).tolist() == pytest.approx(segment_max_pool(flat, seg).tolist())


def test_pnorm_pool_math_and_monotonicity():
    null = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 10 values
    flat = np.array([0.55, 0.55])  # one image, 2 nodes, max 0.55
    seg = np.array([0])
    # F_neg(0.55) = 6/10 (values <= 0.55 are 0.0..0.5); score = 1 - 0.6^2 = 0.64
    out = segment_pnorm_pool(flat, seg, null)
    assert out[0] == pytest.approx(1.0 - 0.6**2)
    # More nodes at the same max -> higher pnorm score (bigger N penalty removed
    # only by a rarer max): 1 - F^N grows with N for F in (0,1).
    flat2 = np.array([0.55, 0.55, 0.55, 0.55])
    assert segment_pnorm_pool(flat2, np.array([0]), null)[0] > out[0]
    # pool_segment dispatch parity
    assert pool_segment(flat, seg, "pnorm", null_sorted=null)[0] == pytest.approx(out[0])
    assert pool_segment(flat, seg, "max").tolist() == segment_max_pool(flat, seg).tolist()


def test_pnorm_requires_null():
    with pytest.raises(ValueError, match="pnorm"):
        pool_segment(np.array([0.5]), np.array([0]), "pnorm")


def test_pool_blocks_matches_segment_pooling():
    # blocks of sizes 2, 3, 1 (unequal, as calibration bags are)
    blocks = [np.array([0.1, 0.9]), np.array([0.5, 0.4, 0.2]), np.array([0.7])]
    flat = np.concatenate(blocks)
    seg = np.array([0, 2, 5])
    assert pool_blocks(blocks, "max") == pytest.approx(segment_max_pool(flat, seg).tolist())
    assert pool_blocks(blocks, "topk", topk=4) == pytest.approx(segment_topk_mean_pool(flat, seg, 4).tolist())
    null = np.sort(np.array([0.0, 0.2, 0.5, 0.8]))
    assert pool_blocks(blocks, "pnorm", null_sorted=null) == pytest.approx(segment_pnorm_pool(flat, seg, null).tolist())


def test_negative_block_null():
    blocks = [np.array([0.1, 0.9]), np.array([0.5, 0.4]), np.array([0.7])]
    labels = [1.0, 0.0, 0.0]
    null = negative_block_null(blocks, labels)
    # only the two negative bags' nodes, sorted
    assert null.tolist() == pytest.approx([0.4, 0.5, 0.7])
    assert negative_block_null(blocks, [1.0, 1.0, 1.0]).size == 0

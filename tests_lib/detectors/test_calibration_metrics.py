"""Unit tests for the torch-free calibration metrics + pooling (issue #2781).

These exercise the pure-numpy core of the calibration study: the oracle cut
(checked against an independent brute-force sweep), the operating cost, the
degenerate/percentile helpers, and the three pooling variants.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.calibration_metrics import (
    detection_metrics,
    fbeta_from_rates,
    fbeta_metrics,
    inclusion_weights,
    is_degenerate,
    negative_block_null,
    operating_cost,
    oracle_cut,
    oracle_fbeta_cut,
    oracle_fbeta_metrics,
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
    # F_neg(0.55) = 6/10 (values <= 0.55 are 0.0..0.5); score = F^N = 0.6^2 = 0.36
    out = segment_pnorm_pool(flat, seg, null)
    assert out[0] == pytest.approx(0.6**2)
    # The N-penalty: more nodes at the same max -> LOWER score (a max over more
    # nodes is less surprising), F^N shrinking with N for F in (0, 1).
    flat2 = np.array([0.55, 0.55, 0.55, 0.55])
    assert segment_pnorm_pool(flat2, np.array([0]), null)[0] < out[0]
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


# --- The objective's columns (#4584) -------------------------------------------


def _brute_fbeta(scores, labels, threshold, beta):
    """F-beta by its textbook definition, from precision and recall."""
    pred = np.asarray(scores) >= threshold
    y = np.asarray(labels) == 1
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return (1 + beta**2) * p * r / (beta**2 * p + r)


@pytest.mark.parametrize("seed", range(5))
def test_fbeta_metrics_is_the_textbook_fbeta_at_the_rows_beta_and_each_preset(seed):
    rng = np.random.default_rng(seed)
    labels = (rng.random(300) < 0.1).astype(float)
    scores = rng.normal(labels, 1.0)
    thr = float(np.quantile(scores, 0.8))
    out = fbeta_metrics(scores, labels, thr, 0.5)
    assert set(out) == {"fbeta", "fbeta_b025", "fbeta_b1", "fbeta_b4"}
    assert out["fbeta"] == pytest.approx(_brute_fbeta(scores, labels, thr, 0.5))
    for col, beta in (("fbeta_b025", 0.25), ("fbeta_b1", 1.0), ("fbeta_b4", 4.0)):
        assert out[col] == pytest.approx(_brute_fbeta(scores, labels, thr, beta))
    # F1 is the beta-1 column, by the definition the f1 column already uses.
    assert out["fbeta_b1"] == pytest.approx(detection_metrics(scores, labels, thr)["f1"])


def test_fbeta_has_no_value_without_a_balance_or_without_positives():
    scores = np.array([0.9, 0.5, 0.1])
    out = fbeta_metrics(scores, np.array([1.0, 0.0, 0.0]), 0.4, None)
    assert np.isnan(out["fbeta"])  # the Inclusion arm drew no balance line
    assert out["fbeta_b1"] == pytest.approx(2 / 3)  # the presets are filled anyway
    none_pos = fbeta_metrics(scores, np.zeros(3), 0.4, 1.0)
    assert all(np.isnan(v) for v in none_pos.values())  # recall is undefined


def test_a_cut_that_returns_nothing_scores_zero_not_nan():
    """It found none of the positives; a NaN would drop the cell and flatter it (#4452)."""
    out = fbeta_metrics(np.array([0.2, 0.1]), np.array([1.0, 0.0]), 0.9, 1.0)
    assert out["fbeta"] == 0.0
    assert np.isnan(detection_metrics(np.array([0.2, 0.1]), np.array([1.0, 0.0]), 0.9)["precision"])


def test_fbeta_from_rates_reads_back_what_the_row_carries():
    """An older frame back-filled from its rates scores each row as a new row would carry it."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(40):
        labels = (rng.random(200) < 0.08).astype(float)
        scores = rng.normal(labels, 1.0)
        thr = float(np.quantile(scores, rng.uniform(0.5, 1.0))) + (10.0 if i == 0 else 0.0)
        beta = (0.25, 1.0, 4.0, 0.5)[i % 4]
        det = {k: round(v, 6) for k, v in detection_metrics(scores, labels, thr).items()}
        rows.append((det["precision"], det["recall"], beta, fbeta_metrics(scores, labels, thr, beta)["fbeta"]))
    p, r, b, want = (np.array(c, dtype=float) for c in zip(*rows, strict=True))
    assert np.isnan(p[0])  # the empty cut is in the sample
    np.testing.assert_allclose(fbeta_from_rates(p, r, b), want, atol=2e-5)


def test_fbeta_from_rates_is_nan_where_the_row_has_no_value():
    out = fbeta_from_rates(np.array([0.5, 0.5, np.nan]), np.array([np.nan, 0.5, 0.0]), np.array([1.0, np.nan, 1.0]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.0


def _brute_best_fbeta(scores, labels, beta):
    """Independent O(n^2) best cut: every observed score as the threshold, highest first on a tie."""
    best = None
    for t in sorted(np.unique(scores), reverse=True):
        f = _brute_fbeta(scores, labels, t, beta)
        if best is None or f > best[0] + 1e-12:
            best = (f, t)
    assert best is not None
    return best


@pytest.mark.parametrize("seed", range(25))
def test_oracle_fbeta_cut_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    labels = (rng.random(n) < 0.2).astype(float)
    labels[0] = 1.0  # at least one positive
    scores = rng.normal(labels, 1.0).round(1)  # rounding forces ties
    for beta in (0.25, 1.0, 4.0):
        thr, best, fpr, fnr = oracle_fbeta_cut(scores, labels, beta)
        b_best, b_thr = _brute_best_fbeta(scores, labels, beta)
        assert best == pytest.approx(b_best, abs=1e-9)
        assert thr == b_thr  # ties in F-beta go to the higher threshold, as oracle_cut's do
        # The cut realises what it reports.
        assert _brute_fbeta(scores, labels, thr, beta) == pytest.approx(best, abs=1e-9)
        _cost, fpr2, fnr2 = operating_cost(scores, labels, thr, 1.0, 1.0)
        assert (fpr, fnr) == pytest.approx((fpr2, fnr2), abs=1e-12)


def test_the_best_cut_bounds_every_cut_and_the_cost_cut_is_no_ceiling_on_it():
    """#4654: on a rare class the cost oracle cuts deep, so its F1 sits under a good line's; the F1 oracle never does."""
    rng = np.random.default_rng(42)
    labels = (rng.random(5000) < 0.01).astype(float)
    scores = rng.normal(2.5 * labels, 1.0)
    best = oracle_fbeta_cut(scores, labels, 1.0)[1]
    for t in np.quantile(scores, np.linspace(0.5, 0.999, 60)):
        assert fbeta_metrics(scores, labels, float(t), 1.0)["fbeta"] <= best + 1e-12
    cost_thr = oracle_cut(scores, labels, 1.0, 1.0)[0]
    cost_cut = detection_metrics(scores, labels, cost_thr)
    assert cost_cut["recall"] > 0.75 and cost_cut["precision"] < 0.2  # the deep cut the viewer used to draw
    assert cost_cut["f1"] < best - 0.1


def test_oracle_fbeta_metrics_carries_each_preset_and_the_rows_own_cut():
    rng = np.random.default_rng(7)
    labels = (rng.random(400) < 0.05).astype(float)
    scores = rng.normal(2.0 * labels, 1.0)
    out = oracle_fbeta_metrics(scores, labels, 4.0)
    assert set(out) == {
        "oracle_fbeta",
        "oracle_fbeta_b025",
        "oracle_fbeta_b1",
        "oracle_fbeta_b4",
        "fbeta_oracle_threshold",
        "fbeta_oracle_fpr",
        "fbeta_oracle_fnr",
    }
    for col, beta in (("oracle_fbeta_b025", 0.25), ("oracle_fbeta_b1", 1.0), ("oracle_fbeta_b4", 4.0)):
        assert out[col] == pytest.approx(oracle_fbeta_cut(scores, labels, beta)[1])
    assert out["oracle_fbeta"] == out["oracle_fbeta_b4"]  # the row's own beta
    thr, _best, fpr, fnr = oracle_fbeta_cut(scores, labels, 4.0)
    assert (out["fbeta_oracle_threshold"], out["fbeta_oracle_fpr"], out["fbeta_oracle_fnr"]) == (thr, fpr, fnr)
    # Each oracle bounds the row's own F-beta at any threshold.
    row = fbeta_metrics(scores, labels, float(np.quantile(scores, 0.97)), 4.0)
    for col in ("fbeta", "fbeta_b025", "fbeta_b1", "fbeta_b4"):
        assert row[col] <= out[f"oracle_{col}"] + 1e-12


def test_oracle_fbeta_has_no_value_without_a_balance_or_without_positives():
    scores = np.array([0.9, 0.5, 0.1])
    out = oracle_fbeta_metrics(scores, np.array([1.0, 0.0, 0.0]), None)
    assert np.isnan(out["oracle_fbeta"]) and np.isnan(out["fbeta_oracle_threshold"])  # no balance drew a line
    assert out["oracle_fbeta_b1"] == pytest.approx(1.0)  # the presets are filled anyway
    assert all(np.isnan(v) for v in oracle_fbeta_metrics(scores, np.zeros(3), 1.0).values())
    assert all(np.isnan(v) for v in oracle_fbeta_cut(np.array([]), np.array([]), 1.0))

"""Unit tests for labelset-kNN evidence coverage (``vtscore.detectors.evidence_coverage``).

Pure geometry over embeddings — no Flask, no dataset context — so these live in
the library tier.  See docs/plans/coverage-atlas.md §6.1 (phase v0).
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.evidence_coverage import (
    _kth_nn_distance,
    _loo_kth_nn_distances,
    evidence_coverage_report,
    predicted_support_pvalues,
    support_pvalues,
    trust_scores,
)


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def _cluster(rng, center: np.ndarray, n: int, spread: float = 0.05) -> np.ndarray:
    """*n* unit vectors tightly around *center* (already unit)."""
    pts = center[None, :] + spread * rng.standard_normal((n, center.shape[0])).astype(np.float32)
    return _unit(pts)


def _basis(d: int, i: int) -> np.ndarray:
    v = np.zeros(d, dtype=np.float32)
    v[i] = 1.0
    return v


class TestKthNnDistance:
    def test_empty_refs_is_maximally_far(self):
        q = _unit(np.ones((3, 8), dtype=np.float32))
        out = _kth_nn_distance(q, np.zeros((0, 8), dtype=np.float32), k=1)
        assert out.shape == (3,)
        assert np.allclose(out, 2.0)

    def test_nearest_is_zero_for_identical_point(self):
        rng = np.random.default_rng(0)
        refs = _unit(rng.standard_normal((5, 8)))
        # Query equals a ref row → nearest cosine distance ~ 0.
        out = _kth_nn_distance(refs[2:3], refs, k=1)
        assert out[0] < 1e-4

    def test_k_clamped_to_ref_count(self):
        rng = np.random.default_rng(1)
        refs = _unit(rng.standard_normal((3, 8)))
        q = _unit(rng.standard_normal((2, 8)))
        # k larger than the ref count uses the farthest ref (max distance).
        big_k = _kth_nn_distance(q, refs, k=99)
        farthest = (1.0 - (q @ refs.T)).max(axis=1)
        assert np.allclose(big_k, farthest, atol=1e-5)


class TestLooDistances:
    def test_single_point_class_has_zero_spread(self):
        one = _unit(np.ones((1, 8), dtype=np.float32))
        assert np.allclose(_loo_kth_nn_distances(one, k=1), 0.0)

    def test_excludes_self(self):
        rng = np.random.default_rng(2)
        refs = _unit(rng.standard_normal((6, 8)))
        loo = _loo_kth_nn_distances(refs, k=1)
        # Every LOO distance is strictly positive (self is excluded, so no
        # zero-distance self-match leaks in).
        assert np.all(loo > 1e-6)


class TestSupportPvalues:
    def test_empty_refs_gives_zero(self):
        q = _unit(np.ones((4, 8), dtype=np.float32))
        p = support_pvalues(q, np.zeros((0, 8), dtype=np.float32), k=1)
        assert np.allclose(p, 0.0)

    def test_in_domain_high_out_of_domain_low(self):
        rng = np.random.default_rng(3)
        d = 16
        refs = _cluster(rng, _basis(d, 0), 40)
        in_domain = _cluster(rng, _basis(d, 0), 10)
        far = _cluster(rng, _basis(d, 7), 10)  # orthogonal direction
        p_in = support_pvalues(in_domain, refs, k=1)
        p_far = support_pvalues(far, refs, k=1)
        # p-values are in (0, 1].
        assert np.all(p_in > 0.0) and np.all(p_in <= 1.0)
        assert np.all(p_far > 0.0) and np.all(p_far <= 1.0)
        # In-domain queries look typical; far queries land in the vacuum.
        assert p_in.mean() > 0.3
        assert p_far.max() < 0.1


class TestTrustScores:
    def test_predicted_class_near_evidence_is_trustworthy(self):
        rng = np.random.default_rng(4)
        d = 16
        pos = _cluster(rng, _basis(d, 0), 30)
        neg = _cluster(rng, _basis(d, 1), 30)
        # A query near the positive cluster, predicted positive → TS > 1.
        q = _cluster(rng, _basis(d, 0), 8)
        ts = trust_scores(q, pos, neg, np.ones(8, dtype=bool), k=1)
        assert np.all(ts > 1.0)

    def test_predicted_class_far_from_evidence_is_suspect(self):
        rng = np.random.default_rng(5)
        d = 16
        pos = _cluster(rng, _basis(d, 0), 30)
        neg = _cluster(rng, _basis(d, 1), 30)
        # A query sitting in the negative cluster but *predicted positive* is
        # closer to the other class's evidence → TS < 1.
        q = _cluster(rng, _basis(d, 1), 8)
        ts = trust_scores(q, pos, neg, np.ones(8, dtype=bool), k=1)
        assert np.all(ts < 1.0)

    def test_no_predicted_class_evidence_collapses_toward_zero(self):
        rng = np.random.default_rng(6)
        d = 8
        neg = _cluster(rng, _basis(d, 1), 20)
        q = _cluster(rng, _basis(d, 1), 5)
        # Predicted positive but no positive evidence at all → denominator is
        # the maximal distance, so TS is small.
        ts = trust_scores(q, np.zeros((0, d), dtype=np.float32), neg, np.ones(5, dtype=bool), k=1)
        assert np.all(ts < 1.0)


class TestPredictedSupportRouting:
    def test_routes_by_predicted_class(self):
        rng = np.random.default_rng(7)
        d = 16
        pos = _cluster(rng, _basis(d, 0), 30)
        neg = _cluster(rng, _basis(d, 1), 30)
        # A batch drawn from the negative cluster.  Predicted as their true
        # (negative) class they are well supported; predicted as positive they
        # sit in the positive class's evidence vacuum.  Routing must therefore
        # score the *same* items far higher under the correct predicted class.
        q = _cluster(rng, _basis(d, 1), 20)
        as_neg = predicted_support_pvalues(q, pos, neg, np.zeros(20, dtype=bool), k=1)
        as_pos = predicted_support_pvalues(q, pos, neg, np.ones(20, dtype=bool), k=1)
        assert as_neg.mean() > 0.3
        assert as_pos.max() < 0.1
        assert as_neg.mean() > as_pos.mean()


class TestEvidenceCoverageReport:
    def test_empty_query(self):
        rep = evidence_coverage_report(
            np.ones((3, 8), dtype=np.float32),
            np.ones((3, 8), dtype=np.float32),
            np.zeros((0, 8), dtype=np.float32),
            np.zeros(0, dtype=bool),
        )
        assert rep["n_items"] == 0
        assert rep["unsupported"] is False
        assert rep["frac_unsupported"] == 0.0

    def test_in_domain_reads_supported(self):
        rng = np.random.default_rng(8)
        d = 16
        pos = _cluster(rng, _basis(d, 0), 40)
        neg = _cluster(rng, _basis(d, 1), 40)
        # Queries drawn from the two clusters, predicted as their own class.
        q_pos = _cluster(rng, _basis(d, 0), 30)
        q_neg = _cluster(rng, _basis(d, 1), 30)
        queries = np.vstack([q_pos, q_neg])
        pred = np.array([True] * 30 + [False] * 30)
        rep = evidence_coverage_report(pos, neg, queries, pred)
        assert rep["n_items"] == 60
        assert rep["n_pos_labels"] == 40 and rep["n_neg_labels"] == 40
        assert rep["frac_unsupported"] < 0.2
        assert rep["unsupported"] is False
        assert rep["median_trust"] > 1.0

    def test_domain_shift_reads_unsupported(self):
        rng = np.random.default_rng(9)
        d = 16
        pos = _cluster(rng, _basis(d, 0), 40)
        neg = _cluster(rng, _basis(d, 1), 40)
        # Every query sits in a *third* region the labelset never covered, all
        # predicted positive → an evidence vacuum for the positive class.
        queries = _cluster(rng, _basis(d, 9), 80)
        pred = np.ones(80, dtype=bool)
        rep = evidence_coverage_report(pos, neg, queries, pred)
        assert rep["frac_unsupported"] > 0.5
        assert rep["z_score"] > 3.0
        assert rep["unsupported"] is True

    def test_deterministic(self):
        rng = np.random.default_rng(10)
        d = 12
        pos = _cluster(rng, _basis(d, 0), 20)
        neg = _cluster(rng, _basis(d, 1), 20)
        queries = _cluster(rng, _basis(d, 0), 15)
        pred = np.ones(15, dtype=bool)
        a = evidence_coverage_report(pos, neg, queries, pred)
        b = evidence_coverage_report(pos, neg, queries, pred)
        assert a == b

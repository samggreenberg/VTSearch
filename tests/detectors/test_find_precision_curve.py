"""Unit tests for the Find Stats precision-curve helpers (#4242).

The route-level behaviour (the response fields, the gate, the Kept rate) is in
``test_find_verification.py``; these pin the sampling of return counts and the
large-corpus path, which a small fixture dataset never reaches.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import planted_fold_anchored_cut
from vtscore.state.core import DetectorContext
from vtscore.utils.scores import NON_FINITE_SCORE_SENTINEL
from vtsearch.routes.detectors import _find_precision
from vtsearch.routes.detectors._find_precision import (
    ESTIMATE_ESTIMATED,
    curve_counts,
    estimated_precision_at,
    verified_precision_at,
)


class TestCurveCounts:
    def test_log_spaced_from_one_to_the_corpus(self):
        counts = curve_counts(100_000)
        assert counts[0] == 1
        assert counts[-1] == 100_000
        assert counts == sorted(set(counts))
        assert len(counts) <= _find_precision.PRECISION_CURVE_POINTS
        # The top of the ranking gets as many points as the long tail.
        assert sum(1 for k in counts if k <= 316) >= len(counts) // 2 - 1

    def test_the_current_cut_gets_its_own_point(self):
        assert 1234 in curve_counts(100_000, extra=1234)

    def test_an_out_of_range_cut_is_ignored(self):
        assert curve_counts(10, extra=0) == curve_counts(10)
        assert curve_counts(10, extra=11) == curve_counts(10)

    def test_a_small_corpus_is_every_count(self):
        assert curve_counts(4) == [1, 2, 3, 4]
        assert curve_counts(0) == []


class TestVerifiedPrecision:
    def test_counts_only_what_was_checked(self):
        ranked = [10, 11, 12, 13]
        checked = {10: True, 12: False}
        assert verified_precision_at(ranked, checked, [1, 2, 3, 4]) == [
            (1, 1, 1.0),
            (1, 1, 1.0),
            (2, 1, 0.5),
            (2, 1, 0.5),
        ]

    def test_nothing_checked_has_no_precision(self):
        assert verified_precision_at([1, 2], {}, [1, 2]) == [(0, 0, None), (0, 0, None)]


class TestLargeCorpus:
    def _ctx(self) -> DetectorContext:
        ctx = DetectorContext()
        ctx.anchored_cut_cache = planted_fold_anchored_cut(n_pos_per_fold=8)
        return ctx

    def test_a_sampled_corpus_is_read_at_the_matching_rank(self, monkeypatch):
        """Above the cap the estimate comes off a uniform sample, mapped back to full-corpus counts."""
        corpus = np.random.default_rng(3).beta(1.0, 4.0, 2000)
        counts = [1, 5, 10, 100, 1000, 2000]
        full = estimated_precision_at(self._ctx(), corpus, counts)
        monkeypatch.setattr(_find_precision, "PRECISION_CURVE_MAX_CORPUS", 200)
        sampled = estimated_precision_at(self._ctx(), corpus, counts)
        assert sampled.status == full.status == ESTIMATE_ESTIMATED
        # A count below one sampled item (2000 / 200 = 10 per item) has no estimate.
        assert sampled.values[:2] == [None, None]
        # The rest track the unsampled curve.
        for got, want in zip(sampled.values[2:], full.values[2:], strict=True):
            assert got == pytest.approx(want, abs=0.1)

    def test_unscorable_items_are_left_out_of_the_estimate(self):
        corpus = np.concatenate([np.random.default_rng(4).beta(1.0, 4.0, 300), [NON_FINITE_SCORE_SENTINEL] * 5])
        est = estimated_precision_at(self._ctx(), corpus, [1, 300, 305])
        assert est.values[0] is not None and est.values[1] is not None
        # The last count reaches into the unscorable tail: no estimate there.
        assert est.values[2] is None

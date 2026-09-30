"""Unit tests for the Find Stats precision-curve helpers (#4242).

The route-level behaviour (the response fields, the Kept rate) is in
``test_find_verification.py``; these pin the sampling of return counts and the
verified-precision reading.
"""

from __future__ import annotations

from vtsearch.routes.detectors import _find_precision
from vtsearch.routes.detectors._find_precision import curve_counts, verified_precision_at


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

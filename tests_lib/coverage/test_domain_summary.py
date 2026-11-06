"""Tests for the training-domain summary's statistics (#5567): anchors plus calibration scores."""

import json

import numpy as np
import pytest

from vtscore.coverage import summarize_reference as exported_summarize
from vtscore.coverage import summary_shift_report as exported_report
from vtscore.coverage.domain_shift import ReferenceTooSmall, domain_shift_report
from vtscore.coverage.domain_summary import (
    SUMMARY_ANCHORS,
    SUMMARY_MAX_CAL,
    summarize_reference,
    summary_pvalues,
    summary_shift_report,
)

_CENTERS = np.random.default_rng(3).standard_normal((40, 64))


def _diffuse(n, seed):
    """A many-cluster space, the shape #3348's tests measured the full report on."""
    rng = np.random.default_rng(seed)
    x = _CENTERS[rng.integers(0, 40, n)] + 1.2 * rng.standard_normal((n, 64))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _foreign(n, seed=9):
    x = np.random.default_rng(seed).standard_normal((n, 64))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestSummarizeReference:
    def test_anchors_are_items_of_the_reference(self):
        """The issue's ask: the centralmost *items*, not averages, so a caller can name them by origin."""
        ref = _diffuse(4000, 1)
        rows, cal = summarize_reference(ref)
        assert rows.dtype.kind == "i"
        assert np.all(np.diff(rows) > 0), "ascending and distinct"
        assert 0 < rows.size <= SUMMARY_ANCHORS
        assert rows.max() < ref.shape[0]
        assert cal.size == SUMMARY_MAX_CAL
        assert np.all(np.diff(cal) >= 0), "sorted"

    def test_deterministic(self):
        ref = _diffuse(3000, 1)
        rows_a, cal_a = summarize_reference(ref)
        rows_b, cal_b = summarize_reference(ref.copy())
        assert np.array_equal(rows_a, rows_b)
        assert np.array_equal(cal_a, cal_b)

    def test_a_small_reference_anchors_on_its_whole_fit_half(self):
        ref = _diffuse(100, 1)
        rows, cal = summarize_reference(ref)
        assert rows.size == 50
        assert cal.size == 50

    def test_too_small_to_calibrate(self):
        with pytest.raises(ReferenceTooSmall):
            summarize_reference(_diffuse(39, 1))

    def test_needs_an_anchor(self):
        with pytest.raises(ValueError, match="n_anchors"):
            summarize_reference(_diffuse(200, 1), n_anchors=0)

    def test_exported_from_the_package(self):
        assert exported_summarize is summarize_reference
        assert exported_report is summary_shift_report


class TestSummaryPvalues:
    def test_in_domain_pvalues_are_uniform(self):
        """The property a radius alone lacked: about alpha of the reference's own fresh items fall below alpha."""
        ref, fresh = _diffuse(4000, 1), _diffuse(2000, 2)
        rows, cal = summarize_reference(ref)
        p = summary_pvalues(ref[rows], cal, fresh)
        # Binomial sd at n=2000, p=0.05 is 0.005; the 1 000 calibration scores add about as much.
        assert abs(float(np.mean(p < 0.05)) - 0.05) < 0.025
        assert abs(float(np.median(p)) - 0.5) < 0.06

    def test_foreign_items_look_atypical(self):
        ref = _diffuse(4000, 1)
        rows, cal = summarize_reference(ref)
        assert float(np.mean(summary_pvalues(ref[rows], cal, _foreign(300)) < 0.05)) > 0.8

    def test_reads_back_from_json_exactly(self):
        """The record stores the scores as JSON floats; ranking against them must not move."""
        ref, fresh = _diffuse(2000, 1), _diffuse(500, 2)
        rows, cal = summarize_reference(ref)
        p = summary_pvalues(ref[rows], cal, fresh)
        assert np.array_equal(summary_pvalues(ref[rows], json.loads(json.dumps(cal.tolist())), fresh), p)

    def test_refuses_too_few_scores(self):
        with pytest.raises(ReferenceTooSmall):
            summary_pvalues(_diffuse(5, 1), [0.5] * 19, _diffuse(10, 2))

    def test_refuses_no_anchors_or_a_width_mismatch(self):
        with pytest.raises(ValueError, match="no anchors"):
            summary_pvalues(np.zeros((0, 64), dtype=np.float32), [0.5] * 40, _diffuse(10, 2))
        with pytest.raises(ValueError, match="width"):
            summary_pvalues(_diffuse(5, 1), [0.5] * 40, np.ones((3, 32), dtype=np.float32))

    def test_no_queries(self):
        assert summary_pvalues(_diffuse(5, 1), [0.5] * 40, np.zeros((0, 64), dtype=np.float32)).shape == (0,)


class TestSummaryShiftReport:
    def test_same_shape_as_the_full_report(self):
        ref, fresh = _diffuse(2000, 1), _diffuse(300, 2)
        rows, cal = summarize_reference(ref)
        assert summary_shift_report(ref[rows], cal, fresh).keys() == domain_shift_report(ref, fresh).keys()

    def test_quiet_in_domain_and_flags_a_shift(self):
        ref = _diffuse(4000, 1)
        rows, cal = summarize_reference(ref)
        quiet = summary_shift_report(ref[rows], cal, _diffuse(1500, 2))
        assert quiet["shifted"] is False
        shifted = summary_shift_report(ref[rows], cal, np.vstack([_diffuse(1000, 2), _foreign(500)]))
        assert shifted["shifted"] is True
        # A third of the dataset is foreign; the summary sees most of it.
        assert 0.25 < shifted["frac_atypical"] < 0.45

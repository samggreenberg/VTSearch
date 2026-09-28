"""Recovering the inclusion behind a cut the knob did not set (#4224).

Under a precision floor the reporting cut is wherever the floor lands, not
``threshold_at(k)`` for a ``k`` the user chose.  Autopilot's acquisition cut
still has to sit :data:`ACQUISITION_INCLUSION_OFFSET` steps *stricter* than the
reporting cut, so the offset needs an origin:
:meth:`FoldAnchoredCut.inclusion_for_threshold` inverts ``threshold_at``, and
:func:`detector_acquisition_threshold` reads it when the caller passes no
inclusion.

The estimator is built by hand, as in ``test_acquisition_threshold.py``: a
trained fit's saturation edge depends on ambient torch RNG state, and every pin
here is about the inverse, not about a fit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.state.core import DetectorContext, detector_acquisition_threshold
from vtscore.training.thresholds import (
    ACQUISITION_INCLUSION_OFFSET,
    INCLUSION_SEARCH_SPAN,
    FoldAnchoredCut,
    GmmFit1D,
)

KNOB = range(-10, 11)


def _cut(n: int = 1801) -> FoldAnchoredCut:
    """Equal-variance fit over a haystack wide enough that no cut in range reaches its edges."""
    fit = GmmFit1D(w_lo=0.7, mu_lo=0.3, var_lo=0.02, w_hi=0.3, mu_hi=0.7, var_hi=0.02)
    hay = np.linspace(-0.2, 1.6, n)
    return FoldAnchoredCut(fits=(fit,), fold_haystacks=(hay,), final_haystack=hay, n_anchored=1)


class TestInclusionForThreshold:
    def test_a_realized_cut_round_trips_exactly(self):
        """The contract a precision floor relies on: the recovered inclusion cuts the same line."""
        cut = _cut()
        for k in KNOB:
            t = cut.threshold_at(k)
            assert cut.threshold_at(cut.inclusion_for_threshold(t)) == t, k

    def test_it_returns_the_strict_end_of_a_plateau(self):
        """On a coarse haystack a band of inclusions shares one cut; the inverse takes the strictest.

        That is what keeps an acquisition offset read from the recovered
        inclusion *at least* as high up the ranking as one read from the
        original: ``k' <= k`` and ``threshold_at`` is non-increasing.
        """
        cut = _cut(n=25)
        for k in KNOB:
            recovered = cut.inclusion_for_threshold(cut.threshold_at(k))
            assert recovered <= k + 1e-3, (k, recovered)
            acq_recovered = cut.threshold_at(recovered + ACQUISITION_INCLUSION_OFFSET)
            acq_original = cut.threshold_at(k + ACQUISITION_INCLUSION_OFFSET)
            assert acq_recovered >= acq_original, k

    def test_an_off_grid_threshold_maps_to_the_first_cut_at_or_below_it(self):
        """A cut set by another rule lands between two realizable ones: take the stricter side that admits it all."""
        cut = _cut(n=25)
        hi_cut, lo_cut = cut.threshold_at(-2), cut.threshold_at(2)
        assert hi_cut > lo_cut, "the fixture must separate the two stops"
        between = 0.5 * (hi_cut + lo_cut)
        recovered = cut.inclusion_for_threshold(between)
        assert cut.threshold_at(recovered) <= between
        # ...and nothing measurably stricter would also admit everything *between* admits.
        assert cut.threshold_at(recovered - 2e-3) > between

    def test_it_is_monotone_in_the_threshold(self):
        """A stricter cut maps to a stricter (lower) inclusion."""
        cut = _cut()
        thresholds = sorted({cut.threshold_at(k) for k in KNOB}, reverse=True)
        recovered = [cut.inclusion_for_threshold(t) for t in thresholds]
        assert all(b >= a for a, b in zip(recovered, recovered[1:], strict=False)), recovered

    def test_it_clamps_to_the_search_bracket(self):
        """Beyond every realizable cut the answer is the bracket's edge, not an error."""
        cut = _cut()
        assert cut.inclusion_for_threshold(10.0) == -INCLUSION_SEARCH_SPAN
        assert cut.inclusion_for_threshold(-10.0) == INCLUSION_SEARCH_SPAN

    @pytest.mark.parametrize("threshold", [math.nan, math.inf])
    def test_a_non_finite_threshold_has_no_inclusion(self, threshold):
        assert _cut().inclusion_for_threshold(threshold) is None

    def test_an_empty_haystack_has_no_inclusion(self):
        """``threshold_at`` is the constant 0.5 there, so there is nothing to invert."""
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.3, var_lo=0.02, w_hi=0.3, mu_hi=0.7, var_hi=0.02)
        hay = np.linspace(-0.2, 1.6, 101)
        cut = FoldAnchoredCut(fits=(fit,), fold_haystacks=(hay,), final_haystack=np.array([]), n_anchored=1)
        assert cut.inclusion_for_threshold(0.5) is None


class TestAcquisitionFromTheReportingCut:
    def test_without_an_inclusion_it_samples_four_steps_stricter_than_the_line(self):
        """``inclusion_value=None`` reads the offset's origin off ``ctx.threshold``."""
        cut = _cut()
        ctx = DetectorContext(detector_id="det-acq-derived", media_type="audio")
        ctx.anchored_cut_cache = cut
        for k in KNOB:
            ctx.threshold = cut.threshold_at(k)
            recovered = cut.inclusion_for_threshold(ctx.threshold)
            acq = detector_acquisition_threshold(ctx)
            assert acq == cut.threshold_at(recovered + ACQUISITION_INCLUSION_OFFSET)
            assert acq >= ctx.threshold

    def test_it_tracks_the_knob_path_to_one_haystack_item(self):
        """Today's behaviour is the special case, up to the plateau the reporting cut sits on.

        The recovered inclusion is the strict end of the band of ``k`` that
        realizes the same line, so four steps below it can land one order
        statistic higher than four steps below the knob's own ``k``: never
        looser, and never more than one item apart.
        """
        cut = _cut()
        hay = cut.final_haystack
        ctx = DetectorContext(detector_id="det-acq-agree", media_type="audio")
        ctx.anchored_cut_cache = cut
        for k in KNOB:
            ctx.threshold = cut.threshold_at(k)
            derived, knob = detector_acquisition_threshold(ctx), detector_acquisition_threshold(ctx, k)
            assert derived >= knob, k
            assert int(np.sum((hay >= knob) & (hay < derived))) <= 1, k

    def test_a_cut_from_another_rule_still_gets_an_acquisition_gap(self):
        """The #4224 case: the line was set by a precision floor, not the knob."""
        cut = _cut()
        ctx = DetectorContext(detector_id="det-acq-floor", media_type="audio")
        ctx.anchored_cut_cache = cut
        ctx.threshold = 0.5 * (cut.threshold_at(-1) + cut.threshold_at(0))
        assert detector_acquisition_threshold(ctx) > ctx.threshold

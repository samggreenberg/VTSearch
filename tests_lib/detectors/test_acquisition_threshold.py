"""The app's acquisition cut is a *second* cut, taken from the same estimator.

``det_ctx.threshold`` is the decision line the user sees.  Autopilot's Hard and
New picks read a threshold as a **rank position** instead, so they take their own
cut :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET` inclusion
steps below it - which raises it and moves it *up* the ranking.  PR #2876
measured that at 4.5x the positives per 100 votes and lower cost; see
``docs/experiments/acquisition-inclusion/REPORT.md``.

The direction is the opposite of the intuition from the cost weights, and the
offset is relative rather than absolute.  Both are easy to get backwards and
neither shows up as a crash, so they are pinned here.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.training import train_and_score
from vtscore.state.core import DetectorContext, detector_acquisition_threshold
from vtscore.training.thresholds import ACQUISITION_INCLUSION_OFFSET, FoldAnchoredCut, GmmFit1D

DIM = 8


def _clips(rng: np.random.Generator, cids: range) -> dict[int, dict]:
    return {
        cid: {
            "embeddings": {"test": rng.standard_normal(DIM).astype(np.float32)},
            "embedder": "test",
            "media_type": "audio",
            "md5": f"m{cid:031d}",
        }
        for cid in cids
    }


#: Haystack size for the trained fixtures below.  A cut is realized as an
#: empirical quantile of this haystack (:meth:`FoldAnchoredCut.threshold_at`),
#: so the haystack's own spacing is the finest gap two cuts can be apart: with
#: 20 items a quantile has to move a full 5% before the threshold moves at all.
#: :data:`ACQUISITION_INCLUSION_OFFSET` is a *few* inclusion steps (three, as
#: shipped), and the tilt of even three of them is routinely smaller than that,
#: so on a 20-item haystack the acquisition cut lands on the *same* haystack
#: element as the reporting cut and a strict ``>`` degenerates into an equality.
#: 100 resolves a single step with room to spare (verified across seeds), which
#: is the harder requirement and therefore still the right size, keeping the
#: direction pin about the offset rather than about discretization.
#:
#: Deliberately sized against ONE step rather than against whatever the constant
#: currently is: the fixture should not need re-sizing every time the shipped
#: offset moves, and it has moved three times (-3, -1, -3).
#:
#: The 20-item version *looked* nondeterministic (issue #3101): it failed only
#: in a full run, from a fully seeded fixture.  What varied with process
#: ordering was the ambient training budget, not the estimator - a stray
#: ``vtscore.config`` reload reverted ``TRAIN_EPOCHS`` to production, which
#: moved the fit, which moved which side of a haystack gap the two cuts landed
#: on.  The offset itself was never at risk: measured on the failing fixture,
#: the per-fold rate cut moved 0.528 -> 0.551 between the two inclusions with
#: an interior stationary point at both, i.e. exactly the tilt this file pins,
#: too small to cross a 20-sample gap.  The leak is fixed at its source, but
#: the fixture stays wide: a pin whose margin is one haystack spacing is a pin
#: that reports unrelated drift as a falsified conclusion.
HAYSTACK = 100


def _trained(seed: int, detector_id: str, inclusion_value: int = 0):
    rng = np.random.default_rng(seed)
    clips = _clips(rng, range(500, 500 + HAYSTACK))
    good = {cid: None for cid in range(500, 504)}
    bad = {cid: None for cid in range(504, 508)}
    det_ctx = DetectorContext(detector_id=detector_id, media_type="audio")
    _results, threshold, model = train_and_score(clips, good, bad, inclusion_value=inclusion_value, det_ctx=det_ctx)
    assert model is not None
    det_ctx.threshold = threshold
    return det_ctx


class TestAcquisitionThresholdIsDecoupled:
    def test_it_sits_above_the_reporting_cut(self):
        """The headline direction, in the smallest form that can fail.

        Higher cut -> further up the descending ranking -> Hard samples nearer
        the top -> more positives.  If this ever flips, the shipped default is
        the falsified ``acq_p2`` arm and nothing would say so.
        """
        det_ctx = _trained(7, "det-acq-direction")
        assert det_ctx.anchored_cut_cache is not None, "safe-on training must park the fitted estimator"
        acq = detector_acquisition_threshold(det_ctx, 0)
        assert acq > det_ctx.threshold

    def test_the_reporting_cut_is_untouched(self):
        """Deriving the acquisition cut must not disturb the decision line.

        Everything the user is shown reads ``det_ctx.threshold``; a derivation
        with a side effect here would move the green/red line as a side effect
        of an Autopilot pick.
        """
        det_ctx = _trained(7, "det-acq-no-side-effect")
        before = det_ctx.threshold
        for k in (-4, 0, 4):
            detector_acquisition_threshold(det_ctx, k)
        assert det_ctx.threshold == before

    def test_the_offset_is_relative_to_inclusion_not_absolute(self):
        """The gap is what was measured, so it must survive an Inclusion slide.

        Read absolutely, ``-3`` would become a no-op at reporting inclusion -3
        and invert below it.  Read as an offset, the selector stays above the
        reporting line wherever the user puts the slider.

        The estimator is built by hand rather than trained (issue #2896): a
        trained fit's saturation edge depends on ambient torch RNG state, so a
        strict ``>`` at deep-negative inclusions held or failed with process
        ordering.  This geometry is exact and order-independent: an
        equal-variance fit whose rate crossing runs off the inter-mean
        interval at ``|k| ~ 6``, so k in {-10, -6} exercise the continuation
        regime where the edge-clamped cut used to collapse the offset to a
        no-op, over a haystack wide enough that no cut in range reaches its
        edges.
        """
        fit = GmmFit1D(w_lo=0.7, mu_lo=0.3, var_lo=0.02, w_hi=0.3, mu_hi=0.7, var_hi=0.02)
        hay = np.linspace(-0.2, 1.6, 1801)
        cut = FoldAnchoredCut(fits=(fit,), fold_haystacks=(hay,), final_haystack=hay, n_anchored=1)
        det_ctx = DetectorContext(detector_id="det-acq-relative", media_type="audio")
        det_ctx.anchored_cut_cache = cut
        for k in (-10, -6, -3, 0, 3, 10):
            det_ctx.threshold = cut.threshold_at(k)
            acq = detector_acquisition_threshold(det_ctx, k)
            assert acq == cut.threshold_at(k + ACQUISITION_INCLUSION_OFFSET)
            # Never *below* the reporting line, wherever the slider sits - the
            # failure an absolute reading would produce.  Not strict, because
            # the cost weights saturate at the ends of the slider: once the
            # quantile has hit its ceiling there is no higher cut left to take,
            # and the offset legitimately lands on the reporting cut itself.
            assert acq >= det_ctx.threshold, f"the offset inverted at reporting inclusion {k}"

        # ...and it is a real gap rather than a no-op wherever the estimator
        # still has room to move, which is what was measured.
        det_ctx.threshold = cut.threshold_at(0)
        assert detector_acquisition_threshold(det_ctx, 0) > det_ctx.threshold

    def test_it_is_monotone_in_inclusion_like_the_reporting_cut(self):
        """Same nesting contract - the acquisition cut is the same estimator."""
        det_ctx = _trained(9, "det-acq-monotone")
        cut = det_ctx.anchored_cut_cache
        assert cut is not None
        seen = []
        for k in range(-10, 11):
            det_ctx.threshold = cut.threshold_at(k)
            seen.append(detector_acquisition_threshold(det_ctx, k))
        assert all(b <= a + 1e-12 for a, b in zip(seen, seen[1:], strict=False)), seen
        assert seen[0] > seen[-1], seen

    def test_without_an_estimator_the_two_jobs_coincide(self):
        """Degenerate fit / schedule blend: nothing inclusion-aware to re-cut.

        Falling back to the reporting threshold is the honest answer; carrying
        some other step's cut would sample this model's scores against a cut
        fitted elsewhere.
        """
        det_ctx = _trained(11, "det-acq-no-estimator")
        det_ctx.anchored_cut_cache = None
        assert detector_acquisition_threshold(det_ctx, 0) == det_ctx.threshold

    def test_an_untrained_context_reports_its_own_default(self):
        det_ctx = DetectorContext(detector_id="det-acq-untrained", media_type="audio")
        assert detector_acquisition_threshold(det_ctx, 0) == det_ctx.threshold

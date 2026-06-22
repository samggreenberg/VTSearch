"""Tests for the whole-job ``overall`` fraction + overall ETA on ProgressTracker.

These cover the consolidated-progress-bar behaviour: when a caller reports a
``step``/``total_steps`` structure, the tracker exposes a single ``overall``
fraction (0..1) that advances once across the *entire* job instead of resetting
at each phase, plus an ``eta_seconds`` derived from the overall rate.
"""

import time

import pytest

from vtscore.concurrency.progress import ProgressTracker, _PROGRESS_COMMON_EXTRAS


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))


class TestOverallFraction:
    def test_none_without_step_structure(self):
        t = _tracker()
        t.update("loading", "no steps", current=5, total=10)
        # No step/total_steps reported -> overall stays None, frontend falls
        # back to current/total.
        assert t.get()["overall"] is None

    def test_equal_weight_per_step(self):
        t = _tracker()
        # Step 1 of 4, half done within the step -> (0 + 0.5) / 4 = 0.125.
        t.update("downloading", "dl", current=50, total=100, step=1, total_steps=4)
        assert t.get()["overall"] == pytest.approx(0.125)
        # Step 3 of 4, half done -> (2 + 0.5) / 4 = 0.625.
        t.update("embedding", "emb", current=5, total=10, step=3, total_steps=4)
        assert t.get()["overall"] == pytest.approx(0.625)

    def test_step_floor_when_within_unknown(self):
        t = _tracker()
        # Indeterminate sub-step (total=0): the bar sits at the step floor, not
        # at a spinner, so the unified bar shows a real whole-job position.
        t.update("loading", "model", current=0, total=0, step=2, total_steps=4)
        assert t.get()["overall"] == pytest.approx(0.25)

    def test_advances_across_phases_without_resetting(self):
        t = _tracker()
        overalls = []
        # Simulate a four-phase job: each phase's within-step fraction climbs
        # 0 -> 1, then the next phase begins. The overall fraction must be
        # monotonic non-decreasing across the whole sequence.
        for step in (1, 2, 3, 4):
            for cur in (0, 5, 10):
                t.update("loading", "x", current=cur, total=10, step=step, total_steps=4)
                overalls.append(t.get()["overall"])
        assert overalls == sorted(overalls)
        assert overalls[0] == pytest.approx(0.0)
        assert overalls[-1] == pytest.approx(1.0)

    def test_monotonic_within_step_ignores_backwards_jitter(self):
        t = _tracker()
        t.update("embedding", "x", current=8, total=10, step=3, total_steps=4)
        high = t.get()["overall"]
        # A backwards within-step report (jitter) must not rewind the bar.
        t.update("embedding", "x", current=2, total=10, step=3, total_steps=4)
        assert t.get()["overall"] == pytest.approx(high)

    def test_new_job_resets_on_step_decrease(self):
        t = _tracker()
        # Finish a job at step 4.
        t.update("loading", "x", current=10, total=10, step=4, total_steps=4)
        assert t.get()["overall"] == pytest.approx(1.0)
        # A fresh job (step drops back to 1) resets the bar instead of clamping
        # to the previous job's 100%.
        t.update("downloading", "y", current=0, total=100, step=1, total_steps=4)
        assert t.get()["overall"] == pytest.approx(0.0)

    def test_clears_overall_when_step_structure_drops(self):
        t = _tracker()
        t.update("loading", "x", current=5, total=10, step=2, total_steps=4)
        assert t.get()["overall"] is not None
        # A later update without a step structure clears overall.
        t.update("idle", "", current=0, total=0, step=None, total_steps=None)
        assert t.get()["overall"] is None


class TestWeightedOverall:
    def test_weights_shape_the_fraction(self):
        t = _tracker()
        t.set_step_weights([0.25, 0.15, 0.50, 0.10])
        # Step 1 fully done -> the download slice (0.25) of the whole job.
        t.update("downloading", "dl", current=100, total=100, step=1, total_steps=4)
        assert t.get()["overall"] == pytest.approx(0.25)
        # Step 3 (embed, weight 0.50) half done -> 0.25 + 0.15 + 0.50*0.5 = 0.65.
        t.update("embedding", "emb", current=5, total=10, step=3, total_steps=4)
        assert t.get()["overall"] == pytest.approx(0.65)
        # Final step fully done -> 1.0.
        t.update("loading", "fin", current=1, total=1, step=4, total_steps=4)
        assert t.get()["overall"] == pytest.approx(1.0)

    def test_heavy_step_advances_bar_more(self):
        # The same within-step progress on a heavier step should move the bar
        # further than on a lighter step.
        light = _tracker()
        light.set_step_weights([0.1, 0.9])
        light.update("a", "x", current=1, total=1, step=1, total_steps=2)
        heavy = _tracker()
        heavy.set_step_weights([0.9, 0.1])
        heavy.update("a", "x", current=1, total=1, step=1, total_steps=2)
        assert heavy.get()["overall"] > light.get()["overall"]

    def test_length_mismatch_falls_back_to_equal(self):
        t = _tracker()
        t.set_step_weights([0.5, 0.5])  # only 2 weights for a 4-step job
        t.update("embedding", "emb", current=0, total=0, step=3, total_steps=4)
        # Falls back to equal weighting: (3 - 1) / 4 = 0.5.
        assert t.get()["overall"] == pytest.approx(0.5)

    def test_clearing_weights_restores_equal(self):
        t = _tracker()
        t.set_step_weights([0.9, 0.1])
        t.set_step_weights(None)
        t.update("a", "x", current=0, total=0, step=2, total_steps=2)
        assert t.get()["overall"] == pytest.approx(0.5)


class TestOverallEta:
    def test_overall_eta_spans_whole_job(self, monkeypatch):
        t = _tracker()
        clock = {"now": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        # Start the job.
        t.update("downloading", "x", current=0, total=100, step=1, total_steps=4)
        assert t.get()["eta_seconds"] is None  # not enough elapsed yet

        # 10s later we are 25% through the whole job -> ~30s remaining.
        clock["now"] = 1010.0
        t.update("loading", "x", current=0, total=0, step=2, total_steps=4)
        eta = t.get()["eta_seconds"]
        assert eta is not None
        # elapsed=10s for 0.25 of the job -> 40s total -> ~30s left.
        assert eta == pytest.approx(30.0, rel=0.2)

    def test_overall_eta_does_not_reset_between_phases(self, monkeypatch):
        t = _tracker()
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        t.update("downloading", "x", current=0, total=100, step=1, total_steps=2)
        clock["now"] = 6.0
        t.update("downloading", "x", current=100, total=100, step=1, total_steps=2)
        eta_phase1 = t.get()["eta_seconds"]
        assert eta_phase1 is not None
        # Crossing into phase 2 must keep an ETA (the global clock keeps
        # running); it must not snap back to None as a per-phase ETA would.
        clock["now"] = 7.0
        t.update("embedding", "x", current=10, total=100, step=2, total_steps=2)
        assert t.get()["eta_seconds"] is not None

"""Tests for the whole-job ``overall`` fraction + overall ETA on ProgressTracker.

These cover the consolidated-progress-bar behaviour: when a caller reports a
``step``/``total_steps`` structure, the tracker exposes a single ``overall``
fraction (0..1) that advances once across the *entire* job instead of resetting
at each phase, plus an ``eta_seconds`` derived from the overall rate.
"""

import time

import pytest

from vtscore.concurrency.progress import _ETA_LADDER, ProgressTracker, _PROGRESS_COMMON_EXTRAS


def _tracker() -> ProgressTracker:
    return ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))


def _shown(tracker: ProgressTracker, raw: float) -> float:
    """The rung ``tracker`` would publish for *raw*, narrowed to non-``None``."""
    rung = tracker._humble_eta(float(raw))
    assert rung is not None, f"expected a rung for {raw}"
    return rung


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

    def test_within_override_beats_current_total(self):
        t = _tracker()
        # The ``within`` kwarg drives the overall math while current/total keep
        # the caller's real counts (download bytes stay visible in the UI); the
        # AdaptiveLoadPacer uses it to composite step-1 sub-phases.
        t.update("downloading", "dl", current=999_999, total=4_000_000, step=1, total_steps=4, within=0.5)
        snap = t.get()
        assert snap["overall"] == pytest.approx(0.125)
        assert snap["current"] == 999_999
        assert snap["total"] == 4_000_000

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


class TestOverallStepEnd:
    """``overall_step_end`` marks where the current step's slice ends, so a
    count-less step (``overall`` parked at the slice floor) still tells the
    frontend "the job is somewhere in [overall, overall_step_end]" — the
    bounded indeterminate zone."""

    def test_none_without_step_structure(self):
        t = _tracker()
        t.update("loading", "no steps", current=5, total=10)
        assert t.get()["overall_step_end"] is None

    def test_countless_step_brackets_its_slice(self):
        # The motivating example: three steps weighted 50/30/20. Step 1 done,
        # step 2 count-less -> the bar parks at 0.5 and the slice ends at 0.8;
        # the pair bounds where the job really is.
        t = _tracker()
        t.set_step_weights([0.5, 0.3, 0.2])
        t.update("a", "T1 done", current=1, total=1, step=1, total_steps=3)
        t.update("b", "T2 (no counts)", current=0, total=0, step=2, total_steps=3)
        snap = t.get()
        assert snap["overall"] == pytest.approx(0.5)
        assert snap["overall_step_end"] == pytest.approx(0.8)

    def test_equal_weight_slice_end(self):
        t = _tracker()
        # Step 3 of 4, half done -> overall 0.625, slice ends at 3/4.
        t.update("embedding", "emb", current=5, total=10, step=3, total_steps=4)
        snap = t.get()
        assert snap["overall"] == pytest.approx(0.625)
        assert snap["overall_step_end"] == pytest.approx(0.75)

    def test_never_below_the_clamped_overall(self):
        t = _tracker()
        # Earn 0.8 within step 3 of 4, then jitter backwards: overall holds at
        # its max, and step_end must still bracket it from above.
        t.update("embedding", "x", current=8, total=10, step=3, total_steps=4)
        t.update("embedding", "x", current=2, total=10, step=3, total_steps=4)
        snap = t.get()
        assert snap["overall_step_end"] >= snap["overall"]

    def test_clears_when_step_structure_drops(self):
        t = _tracker()
        t.update("loading", "x", current=5, total=10, step=2, total_steps=4)
        assert t.get()["overall_step_end"] is not None
        t.update("idle", "", current=0, total=0, step=None, total_steps=None)
        assert t.get()["overall_step_end"] is None


class TestLoadStepMapping:
    """The status→step map and clipper step keep the unified bar monotonic."""

    def test_converting_shares_the_loading_step(self):
        # Source→media conversion (document→image, video→frames) is pre-embed
        # work; it must map to a real step so it does not null `overall` and
        # bounce the bar onto the raw current/total scale.
        from vtscore.datasets.stages._common import _STATUS_TO_STEP

        assert _STATUS_TO_STEP["converting"] == _STATUS_TO_STEP["loading"]

    def test_clipper_reports_embed_step_not_finalize(self):
        # Clipping cuts + embeds clips and runs *before* the embed step. If it
        # reported the finalize step the bar would run to ~100% and then the
        # following embed step (a lower number) would trip the "new job" reset
        # and slam the bar backwards. Pinning clipping to the embed step keeps
        # the whole-job fraction monotonic across a clipped load.
        from vtscore.datasets.stages._common import _STATUS_TO_STEP, _TOTAL_LOAD_STEPS

        clip_step = _STATUS_TO_STEP["embedding"]
        t = _tracker()
        t.set_step_weights([0.25, 0.10, 0.55, 0.10])
        overalls = []

        def record(status, cur, total, step):
            t.update(status, "x", current=cur, total=total, step=step, total_steps=_TOTAL_LOAD_STEPS)
            overalls.append(t.get()["overall"])

        # download → load → clip(+embed clips) → embed-missing(no-op) → finalize.
        record("downloading", 100, 100, 1)
        record("loading", 0, 0, 2)
        for cur in (0, 5, 10):  # clipping/embedding clips, reported on the embed step
            record("loading", cur, 10, clip_step)
        record("loading", 0, 0, clip_step)  # embed-missing finds nothing to do
        record("loading", 1, 1, _TOTAL_LOAD_STEPS)  # finalize

        assert overalls == sorted(overalls)  # never rewinds
        assert overalls[-1] == pytest.approx(1.0)
        # The clip+embed slice really advanced the bar (not pinned at a floor).
        assert max(overalls[2:5]) > overalls[1]


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

    def test_warm_phases_do_not_poison_the_overall_eta(self, monkeypatch):
        # Regression for #2615: a warm re-import completes steps 1-2 almost
        # instantly, banking ~30% of the bar in under a second. The job-global
        # average rate then projected the whole load at that pace ("10 sec
        # left?" at the start of a 2.5-minute embed, creeping upward the whole
        # time). The rate window must rebase at each step boundary so the
        # extrapolation reflects the pace the job sustains *now*.
        t = _tracker()
        t.set_step_weights([0.2, 0.1, 0.3, 0.4])
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

        t.update("loading", "prep", current=0, total=0, step=1, total_steps=4)
        clock["now"] = 0.5
        # Warm archive: straight to embedding, 30% of the bar banked instantly.
        t.update("embedding", "x", current=0, total=1000, step=3, total_steps=4)
        assert t.get()["eta_seconds"] is None  # an instant span is not a rate signal
        # 10s into embedding, 70/1000 done (a ~140s phase).
        clock["now"] = 10.5
        t.update("embedding", "x", current=70, total=1000, step=3, total_steps=4)
        eta = t.get()["eta_seconds"]
        assert eta is not None
        # Phase-local extrapolation: 10s bought 0.021 of the bar, so the
        # remaining 0.679 reads as minutes — not the tens of seconds the
        # banked warm spans used to suggest.
        assert eta > 100.0


class TestHumbleEta:
    """The published ``eta_seconds`` is coarse and sticky.

    A raw estimate reported to the second reads as a system that has no idea:
    "9 min left … 10 min … 9.5 min … 11 min". The tracker therefore publishes the
    nearest rung of a coarse ladder and holds it until the estimate moves
    decisively. These tests pin both halves — the coarseness and the stickiness —
    plus the cases where the display *must* still move.
    """

    def test_published_values_are_always_ladder_rungs(self):
        t = _tracker()
        for raw in (3, 12, 47, 121, 400, 1000, 5000, 40_000, 10**7):
            t._eta_rung = None  # each value judged fresh
            assert t._humble_eta(float(raw)) in _ETA_LADDER

    def test_a_wobbling_estimate_never_moves_the_display(self):
        # The exact complaint this exists to fix: an estimate oscillating around
        # ten minutes must read "about 10 min" the entire time.
        t = _tracker()
        shown = [t._humble_eta(float(raw)) for raw in (540, 600, 570, 660, 620, 700, 520, 480)]
        assert set(shown) == {600.0}

    def test_a_sustained_slowdown_is_still_reported(self):
        # Humility is not silence. A job that really is taking longer must say
        # so; clamping the display to never rise would move the lie from the
        # number to its trend.
        t = _tracker()
        shown = [_shown(t, raw) for raw in (600, 900, 1300, 2000, 3000)]
        assert shown[-1] > shown[0]
        assert shown == sorted(shown)

    def test_a_large_correction_jumps_straight_to_the_new_rung(self):
        # One rung per update would lag badly on a slow-updating job; a decisive
        # move should be believed at once.
        t = _tracker()
        assert t._humble_eta(600.0) == 600.0
        assert t._humble_eta(30.0) == 30.0

    def test_small_moves_toward_a_neighbour_do_not_cross_it(self):
        t = _tracker()
        assert t._humble_eta(600.0) == 600.0
        # 450 and 900 are the neighbours; their boundaries with 600 sit at
        # ~520 and ~735, and the hysteresis margin pushes those further out.
        assert t._humble_eta(530.0) == 600.0
        assert t._humble_eta(720.0) == 600.0

    def test_below_and_above_the_ladder_clamp_to_its_ends(self):
        t = _tracker()
        assert t._humble_eta(0.4) == _ETA_LADDER[0]
        t2 = _tracker()
        assert t2._humble_eta(10**9) == _ETA_LADDER[-1]

    @pytest.mark.parametrize("bad", [None, 0.0, -5.0, float("inf"), float("nan")])
    def test_no_estimate_clears_the_held_rung(self, bad):
        t = _tracker()
        t._humble_eta(600.0)
        assert t._humble_eta(bad) is None
        assert t._eta_rung is None

    def test_a_new_job_does_not_inherit_the_previous_rung(self, monkeypatch):
        t = _tracker()
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
        t.update("downloading", "x", current=0, total=100, step=1, total_steps=2)
        clock["now"] = 20.0
        t.update("downloading", "x", current=10, total=100, step=1, total_steps=2)
        assert t.get()["eta_seconds"] is not None
        # A different step structure reads as a brand-new job.
        t.update("running", "y", current=0, total=10, step=1, total_steps=3)
        assert t.get()["eta_seconds"] is None
        assert t._eta_rung is None

    def test_the_internal_estimate_stays_unquantized(self, monkeypatch):
        # Feeding the snapped value back into the EMA would let the ladder
        # capture the estimate and stop it converging at all, so only the
        # published field is coarsened.
        t = _tracker()
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
        t.update("embedding", "x", current=0, total=1000)
        clock["now"] = 10.0
        t.update("embedding", "x", current=70, total=1000)
        # 10s bought 70 of 1000 items, so 930 remain at 7/s ≈ 132.9s.
        assert t._smoothed_eta == pytest.approx(132.857, rel=1e-3)
        assert t.get()["eta_seconds"] == 120.0  # the nearest rung


class TestFinalizeProgress:
    """The FinalizeProgress proxy maps each finalize sub-stage into its own
    ordered slice of step 4, so no single sub-stage can pin the unified bar at
    100% while later sub-stages are still running (the old "frozen at 100%
    while saving to disk" bug)."""

    def _finalize_tracker(self):
        from vtscore.datasets.stages._common import _LOAD_STEP_WEIGHTS, _TOTAL_LOAD_STEPS

        t = _tracker()
        t.set_step_weights(_LOAD_STEP_WEIGHTS)
        # Pretend steps 1-3 finished so the bar is parked at the start of the
        # finalize slice, exactly where a cache-backed demo load lands.
        t.update("embedding", "emb", current=1, total=1, step=3, total_steps=_TOTAL_LOAD_STEPS)
        return t

    def test_substages_advance_monotonically_within_step_four(self):
        from vtscore.datasets.stages._common import FinalizeProgress

        t = self._finalize_tracker()
        start = t.get()["overall"]
        fin = FinalizeProgress(t)
        overalls = []

        def rec():
            overalls.append(t.get()["overall"])

        fin.begin("dedup")
        fin.update("loading", "Removing duplicates…", current=0, total=10)
        rec()
        fin.update("loading", "Removing duplicates…", current=10, total=10)
        rec()
        after_dedup = t.get()["overall"]
        fin.begin("coverage")
        fin.update("loading", "Building coverage atlas…", current=5, total=10)
        rec()
        fin.begin("registry")
        fin.update("loading", "Saving to registry…", current=3, total=3)
        rec()
        fin.begin("projection")
        fin.update("loading", "Projection ready", current=1, total=1)
        rec()

        # Never rewinds, starts no earlier than the finalize floor, ends full.
        assert overalls == sorted(overalls)
        assert start <= overalls[0]
        assert overalls[-1] == pytest.approx(1.0)
        # The key fix: finishing the first sub-stage (dedup) must NOT slam the
        # whole bar to 100% — later sub-stages still have room to advance.
        assert after_dedup < 1.0

    def test_check_cancelled_forwards_to_real_tracker(self):
        from vtscore.concurrency.progress import CancelledError
        from vtscore.datasets.stages._common import FinalizeProgress

        t = self._finalize_tracker()
        fin = FinalizeProgress(t)
        t.cancel()
        with pytest.raises(CancelledError):
            fin.check_cancelled()


class TestEmbedLoopProgress:
    """The EmbedLoopProgress proxy spreads a multi-embedder ingest loop across
    the shared embed step, so the bar advances cumulatively across the loop
    instead of restarting at 0 for each bound embedder (the v3-trio "static
    stretch" bug)."""

    def _weights(self):
        from vtscore.datasets.stages._common import _LOAD_STEP_WEIGHTS

        return _LOAD_STEP_WEIGHTS

    def _embed_tracker(self):
        from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

        t = _tracker()
        t.set_step_weights(self._weights())
        # Pretend download finished so the bar sits at the start of the embed
        # region, where the embed loop takes over.
        t.update("downloading", "dl", current=1, total=1, step=1, total_steps=_TOTAL_LOAD_STEPS)
        return t

    def test_single_embedder_forwards_unchanged(self):
        # n == 1 must behave exactly as the old per-embedder closure: status maps
        # to its step and current/total pass through verbatim.
        from vtscore.datasets.stages._common import _STATUS_TO_STEP, _TOTAL_LOAD_STEPS
        from vtscore.datasets.stages.embedding import EmbedLoopProgress

        calls = []

        class _RecTracker:
            def check_cancelled(self):
                pass

            def update(self, status, message="", current=0, total=0, **kw):
                calls.append((status, current, total, kw.get("step"), kw.get("total_steps")))

        prog = EmbedLoopProgress(_RecTracker(), 1)
        prog.begin(0)
        prog("loading", "Loading…", 0, 0)
        prog("embedding", "Embedding 5…", 2, 5)
        assert calls == [
            ("loading", 0, 0, _STATUS_TO_STEP["loading"], _TOTAL_LOAD_STEPS),
            ("embedding", 2, 5, _STATUS_TO_STEP["embedding"], _TOTAL_LOAD_STEPS),
        ]

    def test_cumulative_advance_across_embedders(self):
        # Three embedders, each filling its embed pass 0→total. The overall bar
        # must advance monotonically across the whole loop and never rewind when
        # the next embedder restarts its own current at 0.
        from vtscore.datasets.stages.embedding import EmbedLoopProgress

        t = self._embed_tracker()
        prog = EmbedLoopProgress(t, 3)
        overalls = []

        def emb_pass(idx):
            prog.begin(idx)
            prog("embedding", "Embedding 4…", 0, 4)
            overalls.append(t.get()["overall"])
            prog("embedding", "Embedding 4…", 4, 4)
            overalls.append(t.get()["overall"])

        emb_pass(0)
        mid_first = t.get()["overall"]
        emb_pass(1)
        emb_pass(2)

        assert overalls == sorted(overalls)  # never rewinds across the loop
        # Finishing the first embedder must NOT fill the whole embed slice —
        # the 2nd/3rd embedders still have room to advance (the fix).
        from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

        t.update("embedding", "done", current=1, total=1, step=3, total_steps=_TOTAL_LOAD_STEPS)
        full_embed = t.get()["overall"]
        assert mid_first < full_embed

    def test_later_embedder_model_load_does_not_rewind(self):
        # A 2nd/3rd embedder loads its own model mid-loop. Folding that "loading"
        # into the embed step (rather than reporting step 2) keeps the step
        # number from going backwards, which would reset the overall clock and
        # slam the bar to the step-2 floor.
        from vtscore.datasets.stages.embedding import EmbedLoopProgress

        t = self._embed_tracker()
        prog = EmbedLoopProgress(t, 2)

        prog.begin(0)
        prog("loading", "Loading model…", 0, 0)
        prog("embedding", "Embedding 4…", 4, 4)
        after_first = t.get()["overall"]

        prog.begin(1)
        prog("loading", "Loading model…", 0, 0)  # would be step 2 → rewind, if not folded
        after_second_load = t.get()["overall"]

        assert after_second_load >= after_first

    def test_check_cancelled_forwards_to_real_tracker(self):
        from vtscore.concurrency.progress import CancelledError
        from vtscore.datasets.stages.embedding import EmbedLoopProgress

        t = self._embed_tracker()
        prog = EmbedLoopProgress(t, 3)
        t.cancel()
        with pytest.raises(CancelledError):
            prog.check_cancelled()

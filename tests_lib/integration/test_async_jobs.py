"""Tests for the coalescing single-runner JobManager.

The manager must guarantee:

* Only one job runs at a time (no parallel training threads).
* A ``start()`` issued while a job is running enqueues a single pending
  job; further ``start()`` calls update the same pending in place and
  return the same ``AsyncJob`` (latest signature wins).
* When the running job finishes, the pending job is promoted and
  spawned automatically.
* When the running job raises, the pending is still promoted (the error
  path must hand off).
* The signature cache continues to short-circuit identical re-runs.
"""

from __future__ import annotations

import threading
import time

import pytest

from vtscore.concurrency.async_jobs import (
    AsyncJob,
    JobManager,
    bind_job_cancellation,
    check_job_cancelled,
    current_job,
)
from vtscore.concurrency.progress import CancelledError
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    get_thread_dataset_context,
    get_thread_detector_context,
    register_context,
    register_detector_context,
    unregister_context,
    unregister_detector_context,
)


def _make_target(release: threading.Event, started: threading.Event, marker: list):
    """Return a target that records its signature, then waits to be released."""

    def target(job: AsyncJob) -> None:
        started.set()
        marker.append(job.signature)
        # Block until the test releases us so we can interleave more start() calls.
        release.wait(timeout=5)
        job.result = {"signature": job.signature}

    return target


class TestCoalescing:
    def test_single_start_runs_immediately(self):
        mgr = JobManager("test")
        release = threading.Event()
        started = threading.Event()
        ran: list = []
        release.set()  # let it run through immediately

        job = mgr.start("sig-A", _make_target(release, started, ran))
        assert job.status in ("running", "done")
        assert job.done_event.wait(timeout=5)
        assert job.status == "done"
        assert ran == ["sig-A"]

    def test_second_start_while_running_pends(self):
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        first = mgr.start("sig-A", _make_target(first_release, first_started, ran))
        assert first_started.wait(timeout=5)
        assert first.status == "running"

        # Second start while first is running; must enqueue, not spawn.
        second_release = threading.Event()
        second_started = threading.Event()
        second_release.set()
        second = mgr.start("sig-B", _make_target(second_release, second_started, ran))
        assert second.status == "pending"
        assert second.job_id != first.job_id

        # Second's target must NOT have run yet; only first is on a thread.
        assert ran == ["sig-A"]
        assert not second_started.is_set()

        # Release first; second should be promoted automatically.
        first_release.set()
        assert second.done_event.wait(timeout=5)
        assert ran == ["sig-A", "sig-B"]
        assert second.status == "done"
        assert first.status == "done"

    def test_third_start_coalesces_into_pending(self):
        """Rapid #100, #101, #102 → run #100, then run #102 (skip #101)."""
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        mgr.start("sig-A", _make_target(first_release, first_started, ran))
        assert first_started.wait(timeout=5)

        # Two more starts while first is running.
        release_bc = threading.Event()
        release_bc.set()
        started_b = threading.Event()
        started_c = threading.Event()
        second = mgr.start("sig-B", _make_target(release_bc, started_b, ran))
        third = mgr.start("sig-C", _make_target(release_bc, started_c, ran))

        # Same pending object reused; latest signature wins.
        assert second.job_id == third.job_id
        assert third.signature == "sig-C"
        assert second.signature == "sig-C"  # second was mutated in place
        assert third.status == "pending"

        first_release.set()
        assert third.done_event.wait(timeout=5)
        # Only sig-A and sig-C ran; sig-B was coalesced away.
        assert ran == ["sig-A", "sig-C"]
        assert not started_b.is_set(), "sig-B target should never have executed"
        assert started_c.is_set(), "sig-C target should have executed as the coalesced pending"
        # Both callers (#101 and #102) get the same job's result.
        assert second.result == {"signature": "sig-C"}
        assert third.result == {"signature": "sig-C"}

    def test_no_parallel_threads_under_burst(self):
        """N rapid starts must produce at most 2 actually-run targets:
        the one that started first, plus the coalesced latest."""
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        mgr.start("sig-0", _make_target(first_release, first_started, ran))
        assert first_started.wait(timeout=5)

        followups_release = threading.Event()
        followups_release.set()
        followup = None
        for i in range(1, 50):
            followup = mgr.start(
                f"sig-{i}",
                _make_target(followups_release, threading.Event(), ran),
            )
            # Every one of them should land in the same pending slot.
            assert followup.status == "pending"
        assert followup is not None
        assert followup.signature == "sig-49"

        first_release.set()
        assert followup.done_event.wait(timeout=5)
        # Only two targets actually executed: sig-0 and sig-49.
        assert ran == ["sig-0", "sig-49"]

    def test_pending_promoted_after_running_job_raises(self):
        """Even if the running job blows up, the pending must still run."""
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        def boom(job: AsyncJob) -> None:
            first_started.set()
            first_release.wait(timeout=5)
            raise RuntimeError("kaboom")

        first = mgr.start("sig-A", boom)
        assert first_started.wait(timeout=5)

        second_release = threading.Event()
        second_release.set()
        second = mgr.start("sig-B", _make_target(second_release, threading.Event(), ran))
        assert second.status == "pending"

        first_release.set()
        # Wait for first to finish (with error).
        assert first.done_event.wait(timeout=5)
        assert first.status == "error"
        assert "kaboom" in (first.error or "")

        # Pending must have been promoted regardless.
        assert second.done_event.wait(timeout=5)
        assert second.status == "done"
        assert ran == ["sig-B"]

    def test_signature_cache_works_across_coalesced_runs(self):
        """After a coalesced run completes, an identical-signature start()
        short-circuits via cached_for()."""
        mgr = JobManager("test")
        release = threading.Event()
        release.set()
        ran: list = []

        job1 = mgr.start("sig-X", _make_target(release, threading.Event(), ran))
        assert job1.done_event.wait(timeout=5)
        assert job1.status == "done"

        # Same signature → cached_for hits.
        cached = mgr.cached_for("sig-X")
        assert cached is not None
        assert cached.job_id == job1.job_id

        # Different signature misses the cache.
        assert mgr.cached_for("sig-Y") is None

    def test_serial_starts_after_idle_do_not_enqueue(self):
        """When nothing is running, start() should spawn immediately, not
        pend; the pending slot is only used while a job is in flight."""
        mgr = JobManager("test")
        release = threading.Event()
        release.set()
        ran: list = []

        job1 = mgr.start("sig-A", _make_target(release, threading.Event(), ran))
        assert job1.done_event.wait(timeout=5)

        job2 = mgr.start("sig-B", _make_target(release, threading.Event(), ran))
        assert job2.status in ("running", "done")  # not pending
        assert job2.done_event.wait(timeout=5)
        assert ran == ["sig-A", "sig-B"]

    def test_only_one_thread_runs_at_a_time(self):
        """Sanity: even with overlapping starts, no two targets execute
        concurrently.  Track concurrency with a counter."""
        mgr = JobManager("test")
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        gate = threading.Event()

        def target(job: AsyncJob) -> None:
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            gate.wait(timeout=2)
            with active_lock:
                active -= 1
            job.result = job.signature

        first = mgr.start("sig-A", target)
        # Pile on while first is running.
        _ = [mgr.start(f"sig-{i}", target) for i in range(20)]

        gate.set()
        assert first.done_event.wait(timeout=5)
        # Drain any promoted pending.
        for _ in range(10):
            cur = mgr.current()
            if cur is None or cur.status == "done":
                break
            cur.done_event.wait(timeout=2)
        assert max_active == 1


class TestPendingIsolation:
    """The pending slot must not blend requests from different (dataset,
    detector) pairs or users.

    Regression tests: the coalescing branch used to update the parked
    pending in place for *any* new ``start()``, so a poller holding the
    pending's job_id could receive results trained for a different
    dataset/detector (and a different user).  And a pending job that was
    cancelled while parked was still promoted and ran to completion.
    """

    def test_start_for_different_pair_supersedes_pending(self):
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        mgr.start(
            "sig-A",
            _make_target(first_release, first_started, ran),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        assert first_started.wait(timeout=5)

        release_bc = threading.Event()
        release_bc.set()
        second = mgr.start(
            "sig-B",
            _make_target(release_bc, threading.Event(), ran),
            dataset_id="ds-2",
            detector_id="det-2",
        )
        assert second.status == "pending"

        third = mgr.start(
            "sig-C",
            _make_target(release_bc, threading.Event(), ran),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        # Different pair → must NOT coalesce into second's job object.
        assert third.job_id != second.job_id
        # Second's pollers see a terminal state, not sig-C's results.
        assert second.done_event.is_set()
        assert second.status == "cancelled"
        assert second.signature == "sig-B"
        assert second.dataset_id == "ds-2"

        first_release.set()
        assert third.done_event.wait(timeout=5)
        assert third.status == "done"
        assert third.result == {"signature": "sig-C"}
        # sig-B never executed.
        assert ran == ["sig-A", "sig-C"]

    def test_same_pair_still_coalesces(self):
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        mgr.start(
            "sig-A",
            _make_target(first_release, first_started, ran),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        assert first_started.wait(timeout=5)

        release = threading.Event()
        release.set()
        second = mgr.start(
            "sig-B",
            _make_target(release, threading.Event(), ran),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        third = mgr.start(
            "sig-C",
            _make_target(release, threading.Event(), ran),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        assert third.job_id == second.job_id
        assert second.signature == "sig-C"

        first_release.set()
        assert third.done_event.wait(timeout=5)
        assert ran == ["sig-A", "sig-C"]

    def test_cancelled_pending_is_not_promoted(self):
        mgr = JobManager("test")
        first_release = threading.Event()
        first_started = threading.Event()
        ran: list = []

        first = mgr.start("sig-A", _make_target(first_release, first_started, ran))
        assert first_started.wait(timeout=5)

        pending_started = threading.Event()
        pending = mgr.start("sig-B", _make_target(threading.Event(), pending_started, ran))
        assert pending.status == "pending"
        pending.cancel()

        first_release.set()
        assert first.done_event.wait(timeout=5)
        assert pending.done_event.wait(timeout=5)
        assert pending.status == "cancelled"
        assert not pending_started.is_set(), "cancelled pending job must not run"
        assert ran == ["sig-A"]
        # The manager is idle again: a fresh start spawns immediately.
        release = threading.Event()
        release.set()
        job = mgr.start("sig-D", _make_target(release, threading.Event(), ran))
        assert job.status in ("running", "done")
        assert job.done_event.wait(timeout=5)
        assert ran == ["sig-A", "sig-D"]


class TestThreadContextPropagation:
    """``_run()`` must resolve and set the dataset/detector thread-local
    contexts from ``job.dataset_id`` / ``job.detector_id`` so the target's
    ``get_active_context()`` doesn't fall through to the empty fallback.

    Without this, every learned-sort / eval job target that reads votes,
    media, or any other context-scoped state from a background thread sees
    empty containers; silent miscompute. See logical-bug-audit finding C2.
    """

    def test_run_sets_thread_dataset_and_detector_context_from_ids(self):
        mgr = JobManager("ctx-test")
        ds_ctx = DatasetContext("ds-ctx-A")
        det_ctx = DetectorContext("det-ctx-A")
        register_context(ds_ctx)
        register_detector_context(det_ctx)

        captured: dict = {}

        def target(job: AsyncJob) -> None:
            captured["ds"] = get_thread_dataset_context()
            captured["det"] = get_thread_detector_context()
            job.result = "ok"

        try:
            job = mgr.start(
                signature=("sig",),
                target=target,
                dataset_id="ds-ctx-A",
                detector_id="det-ctx-A",
            )
            assert job.done_event.wait(timeout=5)
            assert job.status == "done"
            assert captured["ds"] is ds_ctx
            assert captured["det"] is det_ctx
        finally:
            unregister_context("ds-ctx-A")
            unregister_detector_context("det-ctx-A")

    def test_run_skips_when_ids_resolve_to_no_context(self):
        """Unknown / unloaded ids must not crash and must not set a context."""
        mgr = JobManager("ctx-test")
        captured: dict = {}

        def target(job: AsyncJob) -> None:
            captured["ds"] = get_thread_dataset_context()
            captured["det"] = get_thread_detector_context()
            job.result = "ok"

        job = mgr.start(
            signature=("sig",),
            target=target,
            dataset_id="never-registered",
            detector_id="never-registered",
        )
        assert job.done_event.wait(timeout=5)
        assert job.status == "done"
        assert captured["ds"] is None
        assert captured["det"] is None

    def test_run_clears_thread_context_after_target_returns(self):
        """The worker thread's thread-locals must be reset after the target
        runs, including on the error path, so a follow-up job on a freshly
        spawned thread doesn't inherit stale context (defensive; daemon
        threads are one-shot today, but the contract should hold)."""
        mgr = JobManager("ctx-test")
        ds_ctx = DatasetContext("ds-ctx-B")
        det_ctx = DetectorContext("det-ctx-B")
        register_context(ds_ctx)
        register_detector_context(det_ctx)

        after_done = threading.Event()
        captured: dict = {}

        def target(job: AsyncJob) -> None:
            captured["ds_inside"] = get_thread_dataset_context()
            job.result = "ok"

        def boom(job: AsyncJob) -> None:
            captured["ds_inside_boom"] = get_thread_dataset_context()
            raise RuntimeError("kaboom")

        try:
            ok = mgr.start(("sig-ok",), target, dataset_id="ds-ctx-B", detector_id="det-ctx-B")
            assert ok.done_event.wait(timeout=5)
            assert captured["ds_inside"] is ds_ctx

            err = mgr.start(("sig-err",), boom, dataset_id="ds-ctx-B", detector_id="det-ctx-B")
            assert err.done_event.wait(timeout=5)
            assert err.status == "error"
            assert captured["ds_inside_boom"] is ds_ctx

            # The worker thread itself is daemon and one-shot, but verify the
            # finally block ran by checking that a fresh job with no ids
            # observes a clean slate (no leak from prior runs into the
            # registry-keyed lookup path).
            def verify(job: AsyncJob) -> None:
                captured["ds_clean"] = get_thread_dataset_context()
                captured["det_clean"] = get_thread_detector_context()
                after_done.set()
                job.result = "ok"

            clean = mgr.start(("sig-clean",), verify)
            assert clean.done_event.wait(timeout=5)
            assert after_done.is_set()
            assert captured["ds_clean"] is None
            assert captured["det_clean"] is None
        finally:
            unregister_context("ds-ctx-B")
            unregister_detector_context("det-ctx-B")


def _cancellable_target(started: threading.Event, marker: list):
    """Return a target that polls ``check_job_cancelled`` until cancelled.

    The loop is unbounded, so it exits *only* via ``CancelledError`` (never
    by running out of iterations before the cancel arrives) - the pattern
    CLAUDE.md prescribes for interruptible work.
    """

    def target(job: AsyncJob) -> None:
        started.set()
        marker.append("start")
        while True:  # exits only via CancelledError
            check_job_cancelled()
            time.sleep(0.01)

    return target


class TestCheckJobCancelledPrimitive:
    """Unit coverage for the thread-local cancellation binding used by the
    deep training / eval loops."""

    def test_noop_without_binding(self):
        assert current_job() is None
        check_job_cancelled()  # must not raise when nothing is bound

    def test_bound_uncancelled_job_does_not_raise(self):
        job = AsyncJob(job_id="j")
        with bind_job_cancellation(job):
            assert current_job() is job
            check_job_cancelled()  # not cancelled → no raise
        assert current_job() is None  # binding restored on exit

    def test_bound_cancelled_job_raises(self):
        job = AsyncJob(job_id="j")
        job.cancel()
        with bind_job_cancellation(job):  # noqa: SIM117
            with pytest.raises(CancelledError):
                check_job_cancelled()

    def test_binding_restored_after_exception(self):
        job = AsyncJob(job_id="j")
        try:
            with bind_job_cancellation(job):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert current_job() is None

    def test_nested_binding_restores_outer(self):
        outer = AsyncJob(job_id="outer")
        inner = AsyncJob(job_id="inner")
        with bind_job_cancellation(outer):
            assert current_job() is outer
            with bind_job_cancellation(inner):
                assert current_job() is inner
            assert current_job() is outer


class TestRunningJobCancellation:
    """Cancelling a *running* job must actually stop it - the job's target
    polls ``check_job_cancelled`` (as the real training / eval loops now do),
    transitions to ``cancelled`` (not ``error`` or ``done``), never caches
    its half-built result, and hands off to any parked pending.
    """

    def test_running_job_transitions_to_cancelled(self):
        mgr = JobManager("cancel-test")
        started = threading.Event()
        marker: list = []

        job = mgr.start("sig-A", _cancellable_target(started, marker))
        assert started.wait(timeout=5)
        assert job.status == "running"

        job.cancel()
        assert job.done_event.wait(timeout=5)
        assert job.status == "cancelled"
        # A cancelled job's partial result is never promoted to the cache.
        assert mgr.cached_for("sig-A") is None

    def test_cancelled_running_job_promotes_pending(self):
        mgr = JobManager("cancel-test")
        started = threading.Event()
        marker: list = []

        running = mgr.start(
            "sig-A",
            _cancellable_target(started, marker),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        assert started.wait(timeout=5)

        # Park a pending job for the same requester context.
        pending_release = threading.Event()
        pending_release.set()
        pending = mgr.start(
            "sig-B",
            _make_target(pending_release, threading.Event(), marker),
            dataset_id="ds-1",
            detector_id="det-1",
        )
        assert pending.status == "pending"

        # Cancelling the running job must let the pending one run.
        running.cancel()
        assert running.done_event.wait(timeout=5)
        assert running.status == "cancelled"
        assert pending.done_event.wait(timeout=5)
        assert pending.status == "done"
        assert pending.result == {"signature": "sig-B"}
        assert marker == ["start", "sig-B"]


class TestPhaseProgress:
    """``AsyncJob.set_phase`` — the multi-phase progress structure."""

    def test_phase_defaults_to_single_phase(self):
        job = AsyncJob(job_id="j1")
        assert (job.step, job.total_steps) == (0, 0)

    def test_set_phase_records_position_and_clears_within_phase_counts(self):
        """Entering a phase zeroes the counts left over from the previous one.

        The within-phase ``current``/``total`` describe the phase being *left*;
        carrying them into the next phase would make a poller draw a bar that
        is momentarily complete for work that hasn't started.
        """
        job = AsyncJob(job_id="j1")
        job.update_progress(7, 7, "arranging items")

        job.set_phase(2, 3, "building pyramid")

        assert (job.step, job.total_steps) == (2, 3)
        assert (job.current, job.total) == (0, 0)
        assert job.message == "building pyramid"

    def test_set_phase_keeps_the_previous_message_when_given_none(self):
        job = AsyncJob(job_id="j1")
        job.update_progress(1, 2, "arranging items")
        job.set_phase(2, 3)
        assert job.message == "arranging items"


class TestTrackerBackedProgress:
    """``AsyncJob`` state lives in one :class:`ProgressTracker`, not a copy of it.

    The point of the delegation is that a job stops being a second, poorer
    implementation of the progress model: whatever the tracker computes for a
    dataset load (whole-job ``overall``, a smoothed ETA, push subscriptions)
    a job gets for free, and there is exactly one cancel flag to observe.
    """

    def test_attribute_writes_land_in_the_tracker_snapshot(self):
        """The legacy attribute API is a view of the tracker, not a parallel store."""
        job = AsyncJob(job_id="j1")
        job.update_progress(3, 10, "training")

        snap = job.progress.get()
        assert (snap["current"], snap["total"], snap["message"]) == (3, 10, "training")

        # ...and a bare attribute write (the route's ``job.total = n`` seed)
        # publishes through the tracker too, leaving its neighbours alone.
        job.total = 20
        snap = job.progress.get()
        assert (snap["current"], snap["total"], snap["message"]) == (3, 20, "training")

    def test_phase_structure_yields_a_whole_job_overall_fraction(self):
        """``overall``/``overall_step_end`` come from the tracker, not the poller.

        Every consumer that renders a whole-job bar used to re-derive this pair
        from ``step``/``total_steps``/``current``/``total``; now it is computed
        once, in the place that also owns the step weights.
        """
        job = AsyncJob(job_id="j1")
        job.set_phase(2, 4, "building pyramid")
        job.update_progress(1, 2, "building pyramid")

        snap = job.progress.get()
        # Step 2 of 4, half way through it: (1 + 0.5) / 4.
        assert snap["overall"] == pytest.approx(0.375)
        assert snap["overall_step_end"] == pytest.approx(0.5)

    def test_overall_is_none_for_a_single_phase_job(self):
        job = AsyncJob(job_id="j1")
        job.update_progress(1, 2, "training")
        snap = job.progress.get()
        assert snap["overall"] is None
        assert snap["overall_step_end"] is None

    def test_step_weights_reshape_the_overall_fraction(self):
        """A job can pace its bar by phase cost — a tracker feature, now reachable."""
        job = AsyncJob(job_id="j1")
        job.progress.set_step_weights([1.0, 9.0])
        job.set_phase(2, 2, "the expensive half")

        # Step 1 was a tenth of the work, so entering step 2 banks 0.1 — not
        # the 0.5 an equal-weight split would claim.
        assert job.progress.get()["overall"] == pytest.approx(0.1)

    def test_subscribers_are_pushed_every_job_update(self):
        """``subscribe()`` is what a push channel would attach to."""
        seen: list[tuple[int, int]] = []
        job = AsyncJob(job_id="j1")
        job.progress.subscribe(lambda snap: seen.append((snap["current"], snap["total"])))

        job.update_progress(1, 3)
        job.update_progress(2, 3)

        assert seen == [(1, 3), (2, 3)]

    def test_one_cancel_flag_serves_every_cancellation_entry_point(self):
        """The job's event *is* the tracker's, so no two flags can disagree."""
        job = AsyncJob(job_id="j1")
        assert job.cancel_event is job.progress.cancel_event

        job.cancel()
        assert job.is_cancelled
        assert job.progress.is_cancelled
        with pytest.raises(CancelledError):
            job.progress.check_cancelled()
        with bind_job_cancellation(job), pytest.raises(CancelledError):
            check_job_cancelled()

    def test_cancelling_through_the_tracker_stops_a_running_job(self):
        """Cancel reached via ``job.progress`` unwinds the runner exactly as ``job.cancel()`` does."""
        mgr = JobManager("tracker-cancel")
        started = threading.Event()

        def _target(job):
            started.set()
            while True:
                check_job_cancelled()
                time.sleep(0.01)

        job = mgr.start("sig", _target)
        assert started.wait(timeout=5)
        job.progress.cancel()
        assert job.done_event.wait(timeout=5)
        assert job.status == "cancelled"


class TestSingletonRegistry:
    """``JOB_MANAGERS`` is the single registry of module-level managers.

    It used to hold only the user-visible ones, which forced
    ``reset_all_async_jobs_for_tests`` to re-list the hidden managers by
    hand — and that second list went stale, leaving
    ``archive_thumbnail_jobs`` unreset between tests (issue #3404).  These
    tests fail if a future manager is added to the module but not to the
    registry, which is the drift that caused the miss.
    """

    def test_every_module_level_manager_is_registered(self):
        from vtscore.concurrency import async_jobs

        registered = set(map(id, async_jobs.JOB_MANAGERS.values()))
        missing = sorted(
            name
            for name, obj in vars(async_jobs).items()
            if isinstance(obj, JobManager) and id(obj) not in registered
        )
        assert not missing, (
            f"module-level JobManager singletons absent from JOB_MANAGERS: {missing}. "
            "Add them to the registry (with user_visible=False if they earn no "
            "/api/jobs/active spinner) rather than re-listing them by hand."
        )

    def test_reset_walks_hidden_managers_too(self):
        """A hidden manager's in-flight job is cleared by the autouse reset."""
        from vtscore.concurrency.async_jobs import (
            archive_thumbnail_jobs,
            reset_all_async_jobs_for_tests,
        )

        started = threading.Event()

        def _target(job):
            started.set()
            while True:
                check_job_cancelled()
                time.sleep(0.01)

        job = archive_thumbnail_jobs.start("sig", _target)
        assert started.wait(timeout=5)
        try:
            reset_all_async_jobs_for_tests()
            assert job.done_event.wait(timeout=5)
            assert archive_thumbnail_jobs.active_jobs() == []
        finally:
            job.cancel()
            job.done_event.wait(timeout=5)

    def test_hidden_managers_earn_no_active_pairs_row(self):
        """``/api/jobs/active`` still shows only user-visible work."""
        from vtscore.concurrency.async_jobs import (
            labeling_status_jobs,
            list_active_pairs,
        )

        started = threading.Event()

        def _target(job):
            started.set()
            while True:
                check_job_cancelled()
                time.sleep(0.01)

        job = labeling_status_jobs.start(
            "sig", _target, dataset_id="ds1", detector_id="det1"
        )
        try:
            assert started.wait(timeout=5)
            assert list_active_pairs() == []
        finally:
            job.cancel()
            job.done_event.wait(timeout=5)

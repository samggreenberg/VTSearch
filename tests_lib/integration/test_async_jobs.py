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

from vtscore.concurrency.async_jobs import AsyncJob, JobManager
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


class TestThreadContextPropagation:
    """``_run()`` must resolve and set the dataset/detector thread-local
    contexts from ``job.dataset_id`` / ``job.detector_id`` so the target's
    ``get_active_context()`` doesn't fall through to the empty fallback.

    Without this, every learned-sort / eval job target that reads votes,
    media, or any other context-scoped state from a background thread sees
    empty containers; silent miscompute. See ``docs/plans/logical-bug-audit.md``
    finding C2.
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

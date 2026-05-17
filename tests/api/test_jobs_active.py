"""Tests for ``GET /api/jobs/active``.

The endpoint underpins the top-bar pulldown's spinner glyph — it must
list every ``(dataset_id, detector_id)`` pair with a running or pending
job on any registered :class:`JobManager`, deduplicate across managers,
and exclude jobs that were started without a pair (legacy callers, test
fixtures) so the frontend never gets a spinner with nowhere to attach.
"""

from __future__ import annotations

import threading

import app as app_module  # noqa: F401 — triggers conftest side effects
from vtsearch.concurrency.async_jobs import (
    AsyncJob,
    eval_jobs,
    learned_sort_jobs,
    list_active_pairs,
)


def _blocking_target(release: threading.Event):
    """Return a target that blocks on *release* so the job stays ``running``."""

    def target(job: AsyncJob) -> None:
        release.wait(timeout=5)
        job.result = {"ok": True}

    return target


class TestApiJobsActive:
    def test_empty_when_no_jobs(self, client):
        resp = client.get("/api/jobs/active")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body == {"busy_pairs": []}

    def test_lists_running_pair_with_job_type(self, client):
        release = threading.Event()
        try:
            learned_sort_jobs.start(
                signature=("sig-A",),
                target=_blocking_target(release),
                dataset_id="ds-1",
                detector_id="det-1",
            )

            resp = client.get("/api/jobs/active")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body == {
                "busy_pairs": [
                    {"dataset_id": "ds-1", "detector_id": "det-1", "job_types": ["learned-sort"]},
                ]
            }
        finally:
            release.set()
            # Drain the worker so it doesn't bleed into the next test.
            learned_sort_jobs.reset_for_tests()

    def test_merges_job_types_on_same_pair(self, client):
        """A pair with jobs on multiple managers gets one entry, all types listed."""
        release_a = threading.Event()
        release_b = threading.Event()
        try:
            learned_sort_jobs.start(
                signature=("a",),
                target=_blocking_target(release_a),
                dataset_id="ds-1",
                detector_id="det-1",
            )
            eval_jobs.start(
                signature=("b",),
                target=_blocking_target(release_b),
                dataset_id="ds-1",
                detector_id="det-1",
            )

            resp = client.get("/api/jobs/active")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["busy_pairs"] == [
                {"dataset_id": "ds-1", "detector_id": "det-1", "job_types": ["eval", "learned-sort"]},
            ]
        finally:
            release_a.set()
            release_b.set()
            learned_sort_jobs.reset_for_tests()
            eval_jobs.reset_for_tests()

    def test_drops_jobs_without_pair(self, client):
        """Legacy callers that omit dataset_id/detector_id must not surface."""
        release = threading.Event()
        try:
            learned_sort_jobs.start(
                signature=("a",),
                target=_blocking_target(release),
                # No dataset_id / detector_id — should be excluded.
            )

            resp = client.get("/api/jobs/active")
            assert resp.status_code == 200
            assert resp.get_json() == {"busy_pairs": []}
        finally:
            release.set()
            learned_sort_jobs.reset_for_tests()


class TestListActivePairs:
    """Unit tests for the underlying helper (no Flask client needed)."""

    def test_includes_pending_jobs(self):
        """The pulldown spinner should reflect work that's about to run, not just
        what's executing this instant — pending counts."""
        release = threading.Event()
        try:
            learned_sort_jobs.start(
                signature=("running",),
                target=_blocking_target(release),
                dataset_id="ds-1",
                detector_id="det-1",
            )
            # Second start with the running slot occupied → pending.
            learned_sort_jobs.start(
                signature=("pending",),
                target=_blocking_target(release),
                dataset_id="ds-2",
                detector_id="det-2",
            )

            pairs = list_active_pairs()
            keys = {(p["dataset_id"], p["detector_id"]) for p in pairs}
            assert ("ds-1", "det-1") in keys
            assert ("ds-2", "det-2") in keys
        finally:
            release.set()
            learned_sort_jobs.reset_for_tests()

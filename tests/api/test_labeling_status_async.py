"""Tests for the background-refresh behaviour of ``GET /api/labeling-status``.

Issue #2397 moved the per-step cache advancement off the 2 s poll thread: a
poll that finds the cache behind ``label_history`` returns the last-computed
snapshot immediately with ``stale = true`` and kicks a background worker to
advance the cache, instead of retraining MLPs inline.  These tests exercise
the stale→fresh transition, the snapshot placeholder, and the coalescing of a
rapid poll burst into a single in-flight refresh.
"""

from __future__ import annotations

import time

from vtscore.concurrency.async_jobs import labeling_status_jobs
from vtsearch.state import bad_votes, good_votes, label_history


def _seed_votes_and_history():
    """A handful of good/bad votes plus matching label history — enough for the
    labeling-status cache to train real per-step MLPs."""
    for cid in (1, 2, 3):
        good_votes[cid] = None
        label_history.append((cid, "good", 0.0))
    for cid in (4, 5, 6):
        bad_votes[cid] = None
        label_history.append((cid, "bad", 0.0))


def _wait_for_refresh(timeout: float = 30.0) -> None:
    """Block until no labeling-status refresh job is running or pending."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        active = labeling_status_jobs.active_jobs()
        if not active:
            return
        for job in active:
            job.done_event.wait(timeout=1.0)
    raise AssertionError("labeling-status refresh did not finish in time")


class TestLabelingStatusBackgroundRefresh:
    def test_empty_history_is_fresh_and_not_stale(self, client):
        """With no votes the cache trivially covers the (empty) history, so the
        status is computed inline and flagged ``stale: false``."""
        resp = client.get("/api/labeling-status")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["stale"] is False
        assert body["good_count"] == 0
        assert body["bad_count"] == 0
        for key in ("smart", "stable", "span"):
            assert "status" in body[key]

    def test_first_poll_after_votes_is_stale_then_refreshes(self, client):
        """A cache that is behind ``label_history`` yields an immediate stale
        response plus a background refresh; a later poll returns fresh data."""
        _seed_votes_and_history()

        first = client.get("/api/labeling-status").get_json()
        assert first["stale"] is True
        # Counts + span are always real even in the placeholder.
        assert first["good_count"] == 3
        assert first["bad_count"] == 3
        assert "status" in first["span"]

        _wait_for_refresh()

        second = client.get("/api/labeling-status").get_json()
        assert second["stale"] is False
        assert second["good_count"] == 3
        assert second["bad_count"] == 3
        # Enough votes for the Smart/Stable indicators to emit a real status.
        for key in ("smart", "stable", "span"):
            assert second[key]["status"] in ("red", "yellow", "green")

    def test_stale_poll_returns_previous_snapshot(self, client):
        """After a fresh compute, a genuinely new vote leaves the cache one step
        behind; the next poll serves the prior snapshot (stale) rather than
        retraining inline."""
        _seed_votes_and_history()
        client.get("/api/labeling-status")
        _wait_for_refresh()
        fresh = client.get("/api/labeling-status").get_json()
        assert fresh["stale"] is False

        # A new (non-flip) vote grows the history without truncating the cache.
        good_votes[7] = None
        label_history.append((7, "good", 0.0))

        stale = client.get("/api/labeling-status").get_json()
        assert stale["stale"] is True
        # The snapshot predates the new vote, so its indicator objects match the
        # last fully-computed status (identical smart/stable payloads).
        assert stale["smart"] == fresh["smart"]
        assert stale["stable"] == fresh["stable"]

        _wait_for_refresh()

    def test_rapid_polls_coalesce_into_one_refresh(self, client):
        """A burst of stale polls must not fan out parallel retrains: the job
        manager keeps at most one running + one pending refresh at any time.

        (A poll late in the burst may legitimately come back fresh once the
        background refresh lands, so the coalescing bound - not the ``stale``
        flag - is the invariant under test.)"""
        _seed_votes_and_history()
        for _ in range(6):
            resp = client.get("/api/labeling-status")
            assert resp.status_code == 200
            assert isinstance(resp.get_json()["stale"], bool)
            # Single-runner + one coalescing pending slot, by construction.
            assert len(labeling_status_jobs.active_jobs()) <= 2
        _wait_for_refresh()

"""Unit tests for :func:`vtscore.datasets.ingest_task.start_ingest_task`.

Library tier: no Flask, no routes.  A synchronous ``spawn`` stand-in runs the
worker inline so the task's bookkeeping (tracker registration, terminal
``ingest_result``, cancel / error handling, thread-progress binding) can be
asserted without threading.
"""

from __future__ import annotations

from unittest.mock import patch

from vtscore.concurrency.progress import detector_loading_tasks
from vtscore.datasets import ingest as ingest_module
from vtscore.datasets.ingest_task import start_ingest_task


def _sync_spawn(target, name=None):
    """A ``spawn`` stand-in that runs *target* inline."""
    target()
    return None


ENTRIES = [{"md5": "a", "origin": {"importer": "server_folder", "params": {}}}]


def _snapshot(task_id: str) -> dict:
    """The tracker snapshot for *task_id*, asserted to exist."""
    tracker = detector_loading_tasks.get_tracker(task_id)
    assert tracker is not None, f"task {task_id!r} was never registered"
    return tracker.get()


class TestStartIngestTask:
    def test_publishes_ingested_count_as_ingest_result(self):
        with patch.object(ingest_module, "ingest_missing_medias", lambda *a, **k: 3):
            task_id = start_ingest_task(ENTRIES, {}, task_id="_t_ok", name="Det", spawn=_sync_spawn, detector_id="d1")
        assert task_id == "_t_ok"
        snapshot = _snapshot(task_id)
        assert snapshot["status"] == "idle"
        assert snapshot["error"] in (None, "")
        assert snapshot["ingest_result"] == {"ingested": 3}
        assert detector_loading_tasks.is_finished(task_id)

    def test_after_ingest_contributes_to_the_result(self):
        seen = {}

        def after(ingested: int) -> dict:
            seen["ingested"] = ingested
            return {"applied": 2, "unresolved": 0}

        with patch.object(ingest_module, "ingest_missing_medias", lambda *a, **k: 3):
            task_id = start_ingest_task(
                ENTRIES, {}, task_id="_t_after", name="Det", spawn=_sync_spawn, after_ingest=after
            )
        assert seen == {"ingested": 3}
        snapshot = _snapshot(task_id)
        assert snapshot["ingest_result"] == {"ingested": 3, "applied": 2, "unresolved": 0}

    def test_task_row_carries_its_association_fields(self):
        with patch.object(ingest_module, "ingest_missing_medias", lambda *a, **k: 0):
            task_id = start_ingest_task(
                ENTRIES,
                {},
                task_id="_t_assoc",
                name="My Detector",
                spawn=_sync_spawn,
                detector_id="det-42",
                media_type="audio",
            )
        row = next(t for t in detector_loading_tasks.list_tasks() if t["task_id"] == task_id)
        assert row["name"] == "My Detector"
        assert row["detector_id"] == "det-42"
        assert row["media_type"] == "audio"

    def test_failure_surfaces_on_the_tracker_and_finishes(self):
        def boom(*a, **k):
            raise RuntimeError("origin unreachable")

        with patch.object(ingest_module, "ingest_missing_medias", boom):
            task_id = start_ingest_task(ENTRIES, {}, task_id="_t_err", name="Det", spawn=_sync_spawn)
        snapshot = _snapshot(task_id)
        assert snapshot["error"] == "origin unreachable"
        assert snapshot["ingest_result"] is None
        assert detector_loading_tasks.is_finished(task_id)

    def test_cancel_surfaces_as_cancelled(self):
        def cancel_then_report(entries, medias, on_progress=None):
            detector_loading_tasks.cancel_task("_t_cancel")
            assert on_progress is not None
            on_progress("ingesting", "fetching", 0, 1)
            raise AssertionError("progress callback must raise after a cancel")

        with patch.object(ingest_module, "ingest_missing_medias", cancel_then_report):
            task_id = start_ingest_task(ENTRIES, {}, task_id="_t_cancel", name="Det", spawn=_sync_spawn)
        snapshot = _snapshot(task_id)
        assert snapshot["error"] == "Cancelled"
        assert detector_loading_tasks.is_finished(task_id)

    def test_nested_importer_progress_lands_on_this_task(self):
        """The legacy ingest path reports via ``get_thread_progress``.

        Those frames must reach this task's tracker, not the global dataset
        bar, or an unrelated progress row lights up mid-import.
        """
        from vtscore.concurrency.progress import get_thread_progress

        def report_via_thread_hook(entries, medias, on_progress=None):
            cb = get_thread_progress()
            assert cb is not None, "the task must bind the thread-progress hook"
            cb("ingesting", "Re-ingesting from server_folder", 4, 9)
            return 1

        with patch.object(ingest_module, "ingest_missing_medias", report_via_thread_hook):
            task_id = start_ingest_task(ENTRIES, {}, task_id="_t_hook", name="Det", spawn=_sync_spawn)
        # Terminal update resets the counters, so assert the hook was bound and
        # the run completed cleanly rather than the mid-flight numbers.
        snapshot = _snapshot(task_id)
        assert snapshot["ingest_result"] == {"ingested": 1}
        assert get_thread_progress() is None, "the hook must be cleared afterwards"

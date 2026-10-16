"""A failed dataset task stays listed until dismissed, and only its owner sees it.

A failed import never becomes a dataset, so its task row is the only trace of
it once the failure toast is gone. The dataset tracker used to prune that row
30 seconds after the failure, and the dashboard lost it on the next reload or
the next load to start (#4973). ``LoadingTasksTracker(keep_failures=True)``
keeps it until :meth:`~LoadingTasksTracker.dismiss_task`, and the SSE stream
shows it only to the user who started the task.
"""

import json
import time

from vtscore.concurrency.events import initial_snapshot, stream_progress_events
from vtscore.concurrency.progress import (
    CANCELLED_ERROR,
    LoadingTasksTracker,
    detector_loading_tasks,
    loading_tasks,
)
from vtscore.state.current_user import DEFAULT_USER, thread_user


def _fail(tracker: LoadingTasksTracker, task_id: str, error: str = "Import produced no medias.", **kwargs) -> None:
    tracker.create_task(task_id, f"Dataset {task_id}", **kwargs).update("idle", "", 0, 0, error=error)
    tracker.mark_finished(task_id)


def _age(tracker: LoadingTasksTracker, task_id: str, seconds: float) -> None:
    """Pretend *task_id* finished *seconds* ago."""
    with tracker._lock:
        tracker._tasks[task_id]["finished_at"] = time.time() - seconds


def _ids(tasks: list[dict]) -> list[str]:
    return [t["task_id"] for t in tasks]


class TestKeepFailures:
    def test_failure_outlives_the_old_30s_window(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        _fail(tracker, "imp")
        _age(tracker, "imp", 3600)
        assert _ids(tracker.list_tasks()) == ["imp"]

    def test_dataset_tracker_keeps_failures_and_detector_tracker_does_not(self):
        assert loading_tasks.keep_failures is True
        assert detector_loading_tasks.keep_failures is False

    def test_tracker_without_keep_failures_still_prunes_after_30s(self):
        tracker = LoadingTasksTracker()
        _fail(tracker, "imp")
        _age(tracker, "imp", 35)
        assert tracker.list_tasks() == []

    def test_cancel_is_not_kept(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        _fail(tracker, "stopped", error=CANCELLED_ERROR)
        _age(tracker, "stopped", 35)
        assert tracker.list_tasks() == []

    def test_success_is_not_kept(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        tracker.create_task("ok", "Fine").update("idle", "")
        tracker.mark_finished("ok")
        _age(tracker, "ok", 10)
        assert tracker.list_tasks() == []

    def test_oldest_failures_go_past_the_cap(self):
        tracker = LoadingTasksTracker(keep_failures=True, max_kept_failures=2)
        for i, age in enumerate((300, 200, 100)):
            _fail(tracker, f"f{i}")
            _age(tracker, f"f{i}", age)
        tracker.create_task("running", "Still going")

        assert sorted(_ids(tracker.list_tasks())) == ["f1", "f2", "running"]
        # Pruned for good, not just hidden from one snapshot.
        assert tracker.get_tracker("f0") is None

    def test_snapshot_carries_owner_source_and_finish_time(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        before = time.time()
        _fail(tracker, "imp", created_by="alice", source="server_folder:/data/empty")
        (snap,) = tracker.list_tasks()
        assert snap["created_by"] == "alice"
        assert snap["source"] == "server_folder:/data/empty"
        assert snap["finished_at"] >= before

    def test_owner_defaults_to_the_current_user(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        with thread_user("bob"):
            tracker.create_task("t", "T")
        assert tracker.list_tasks()[0]["created_by"] == "bob"

    def test_running_task_has_no_finish_time(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        tracker.create_task("t", "T")
        assert "finished_at" not in tracker.list_tasks()[0]


class TestDismissTask:
    def test_removes_a_finished_task(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        _fail(tracker, "imp")
        notified: list[list[dict]] = []
        tracker.subscribe(notified.append)

        assert tracker.dismiss_task("imp") is True
        assert tracker.list_tasks() == []
        assert notified and notified[-1] == []

    def test_refuses_a_running_task(self):
        tracker = LoadingTasksTracker(keep_failures=True)
        tracker.create_task("t", "Running")
        assert tracker.dismiss_task("t") is False
        assert _ids(tracker.list_tasks()) == ["t"]

    def test_unknown_task(self):
        assert LoadingTasksTracker(keep_failures=True).dismiss_task("nope") is False


class TestVisibleTo:
    def _tracker(self) -> LoadingTasksTracker:
        tracker = LoadingTasksTracker(keep_failures=True)
        _fail(tracker, "alice_fail", created_by="alice")
        _fail(tracker, "bob_fail", created_by="bob")
        _fail(tracker, "nobody_fail", created_by=DEFAULT_USER)
        _fail(tracker, "bob_cancel", error=CANCELLED_ERROR, created_by="bob")
        tracker.create_task("bob_running", "Bob's load", created_by="bob")
        return tracker

    def test_hides_other_users_failures_only(self):
        tracker = self._tracker()
        visible = _ids(tracker.visible_to(tracker.list_tasks(), "alice"))
        # Bob's running load and his cancel are not failures; Bob's failure is his alone.
        assert sorted(visible) == ["alice_fail", "bob_cancel", "bob_running", "nobody_fail"]

    def test_no_viewer_means_unfiltered(self):
        tracker = self._tracker()
        assert len(tracker.visible_to(tracker.list_tasks(), None)) == 5

    def test_tracker_that_lets_failures_lapse_is_not_filtered(self):
        tracker = LoadingTasksTracker()
        _fail(tracker, "bob_fail", created_by="bob")
        assert _ids(tracker.visible_to(tracker.list_tasks(), "alice")) == ["bob_fail"]


def _loading_tasks_payload(frame: str) -> list[dict]:
    data_line = next(ln for ln in frame.splitlines() if ln.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


class TestStreamFiltersPerViewer:
    def test_initial_snapshot_shows_a_failure_to_its_owner_only(self):
        _fail(loading_tasks, "bob_fail", created_by="bob")

        def visible(viewer):
            frame = next(f for f in initial_snapshot(viewer) if f.startswith("event: loading-tasks\n"))
            return _ids(_loading_tasks_payload(frame))

        assert visible("bob") == ["bob_fail"]
        assert visible("alice") == []
        assert visible(None) == ["bob_fail"]

    def test_live_update_is_filtered_for_the_stream_viewer(self):
        gen = stream_progress_events(heartbeat_seconds=60.0, keepalive_seconds=60.0, viewer="alice")
        try:
            # The connect comment, then every channel's opening snapshot.
            while not next(gen).startswith("event: detector-loading-tasks\n"):
                pass
            _fail(loading_tasks, "bob_fail", created_by="bob")
            _fail(loading_tasks, "alice_fail", created_by="alice")

            frames = []
            while len(frames) < 4:  # create, update and mark_finished per failure; enough to see both
                frame = next(gen)
                if frame.startswith("event: loading-tasks\n"):
                    frames.append(_loading_tasks_payload(frame))
            seen = {tid for payload in frames for tid in _ids(payload)}
            assert "alice_fail" in seen
            # Bob's task shows while it runs, never once it has failed.
            assert all(not (t["task_id"] == "bob_fail" and t.get("error")) for payload in frames for t in payload)
        finally:
            gen.close()

"""A failed import stays listed until dismissed, and only its owner sees it.

A failed import never becomes a dataset, so its task row is the only trace of
it once the failure toast is gone. The dataset tracker used to prune that row
30 seconds after the failure, and the dashboard lost it on the next reload or
the next load to start (#4973). A task created with ``keep_failure=True`` (as
every import is) keeps its failure until
:meth:`~LoadingTasksTracker.dismiss_task`, and the SSE stream shows that
failure only to the user who started the task.
"""

import json
import time

from vtscore.concurrency.events import initial_snapshot, stream_progress_events
from vtscore.concurrency.progress import CANCELLED_ERROR, LoadingTasksTracker, loading_tasks
from vtscore.datasets.load_workers import _handle_load_failure, _start_import_task
from vtscore.state.current_user import DEFAULT_USER, thread_user


def _fail(
    tracker: LoadingTasksTracker,
    task_id: str,
    error: str = "Import produced no medias.",
    keep_failure: bool = True,
    **kwargs,
) -> None:
    task = tracker.create_task(task_id, f"Dataset {task_id}", keep_failure=keep_failure, **kwargs)
    task.update("idle", "", 0, 0, error=error)
    tracker.mark_finished(task_id)


def _age(tracker: LoadingTasksTracker, task_id: str, seconds: float) -> None:
    """Pretend *task_id* finished *seconds* ago."""
    with tracker._lock:
        tracker._tasks[task_id]["finished_at"] = time.time() - seconds


def _ids(tasks: list[dict]) -> list[str]:
    return [t["task_id"] for t in tasks]


class TestKeepFailure:
    def test_kept_failure_outlives_the_30s_window(self):
        tracker = LoadingTasksTracker()
        _fail(tracker, "imp")
        _age(tracker, "imp", 3600)
        assert _ids(tracker.list_tasks()) == ["imp"]

    def test_failure_without_keep_failure_still_lapses_after_30s(self):
        # A failed load of an existing dataset, an atlas build, an AutoFind run:
        # the dataset row is still there to come back to.
        tracker = LoadingTasksTracker()
        _fail(tracker, "load", keep_failure=False)
        _age(tracker, "load", 35)
        assert tracker.list_tasks() == []

    def test_imports_keep_their_failures(self):
        task = _start_import_task(prefix="_loading_", display_name="Mine", total_steps=4, source="here")
        (snap,) = [t for t in loading_tasks.list_tasks() if t["task_id"] == task.task_id]
        assert snap["keep_failure"] is True
        assert snap["source"] == "here"

    def test_failure_after_registering_lets_go_of_the_dataset_id(self):
        # The import failed after its registry entry was written; the entry is
        # rolled back, so the kept row must not attach itself to a later dataset
        # that reuses the id (a demo re-imported after the failure).
        tracker = loading_tasks.create_task("imp", "Demo", keep_failure=True)
        loading_tasks.set_dataset_id("imp", "demo_x")
        _handle_load_failure(ValueError("boom"), "imp", tracker, registry_entry_id="demo_x", task_id="imp")
        (snap,) = loading_tasks.list_tasks()
        assert snap["error"] == "boom"
        assert "dataset_id" not in snap

    def test_cancel_is_not_kept(self):
        tracker = LoadingTasksTracker()
        _fail(tracker, "stopped", error=CANCELLED_ERROR)
        _age(tracker, "stopped", 35)
        assert tracker.list_tasks() == []

    def test_success_is_not_kept(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("ok", "Fine", keep_failure=True).update("idle", "")
        tracker.mark_finished("ok")
        _age(tracker, "ok", 10)
        assert tracker.list_tasks() == []

    def test_oldest_failures_go_past_the_cap(self):
        tracker = LoadingTasksTracker(max_kept_failures=2)
        for i, age in enumerate((300, 200, 100)):
            _fail(tracker, f"f{i}")
            _age(tracker, f"f{i}", age)
        tracker.create_task("running", "Still going")

        assert sorted(_ids(tracker.list_tasks())) == ["f1", "f2", "running"]
        # Pruned for good, not just hidden from one snapshot.
        assert tracker.get_tracker("f0") is None

    def test_snapshot_carries_owner_source_and_finish_time(self):
        tracker = LoadingTasksTracker()
        before = time.time()
        _fail(tracker, "imp", created_by="alice", source="server_folder:/data/empty")
        (snap,) = tracker.list_tasks()
        assert snap["created_by"] == "alice"
        assert snap["source"] == "server_folder:/data/empty"
        assert snap["finished_at"] >= before

    def test_owner_defaults_to_the_current_user(self):
        tracker = LoadingTasksTracker()
        with thread_user("bob"):
            tracker.create_task("t", "T")
        assert tracker.list_tasks()[0]["created_by"] == "bob"

    def test_running_task_has_no_finish_time(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t", "T")
        assert "finished_at" not in tracker.list_tasks()[0]


class TestDismissTask:
    def test_removes_a_finished_task(self):
        tracker = LoadingTasksTracker()
        _fail(tracker, "imp")
        notified: list[list[dict]] = []
        tracker.subscribe(notified.append)

        assert tracker.dismiss_task("imp") is True
        assert tracker.list_tasks() == []
        assert notified and notified[-1] == []

    def test_refuses_a_running_task(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t", "Running")
        assert tracker.dismiss_task("t") is False
        assert _ids(tracker.list_tasks()) == ["t"]

    def test_unknown_task(self):
        assert LoadingTasksTracker().dismiss_task("nope") is False


class TestVisibleTo:
    def _tracker(self) -> LoadingTasksTracker:
        tracker = LoadingTasksTracker()
        _fail(tracker, "alice_fail", created_by="alice")
        _fail(tracker, "bob_fail", created_by="bob")
        _fail(tracker, "nobody_fail", created_by=DEFAULT_USER)
        _fail(tracker, "bob_cancel", error=CANCELLED_ERROR, created_by="bob")
        _fail(tracker, "bob_lapsing", keep_failure=False, created_by="bob")
        tracker.create_task("bob_running", "Bob's load", created_by="bob", keep_failure=True)
        return tracker

    def test_hides_other_users_kept_failures_only(self):
        tracker = self._tracker()
        visible = _ids(tracker.visible_to(tracker.list_tasks(), "alice"))
        # Bob's running load, his cancel and his lapsing failure are not kept
        # failures; his kept failure is his alone.
        assert sorted(visible) == ["alice_fail", "bob_cancel", "bob_lapsing", "bob_running", "nobody_fail"]

    def test_no_viewer_means_unfiltered(self):
        tracker = self._tracker()
        assert len(tracker.visible_to(tracker.list_tasks(), None)) == 6


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

            # create, update and mark_finished each notify: Bob's three, then Alice's first.
            frames = []
            while len(frames) < 4:
                frame = next(gen)
                if frame.startswith("event: loading-tasks\n"):
                    frames.append(_loading_tasks_payload(frame))
            assert "alice_fail" in {tid for payload in frames for tid in _ids(payload)}
            # Bob's task shows while it runs, never once it has failed.
            assert not any(t["task_id"] == "bob_fail" and t.get("error") for payload in frames for t in payload)
        finally:
            gen.close()

"""Tests for parallel dataset loading: per-task progress, concurrent loads."""

import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from vtsearch.utils.progress import (
    CancelledError,
    LoadingTasksTracker,
    ProgressTracker,
    get_progress,
    loading_tasks,
    set_thread_progress,
    get_thread_progress,
    clear_thread_progress,
)


# ---------------------------------------------------------------------------
# LoadingTasksTracker unit tests
# ---------------------------------------------------------------------------


class TestLoadingTasksTracker:
    """Unit tests for the LoadingTasksTracker."""

    def test_create_and_list(self):
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "Dataset A")
        pt.update("loading", "Working...", 50, 100)

        tasks = tracker.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"
        assert tasks[0]["name"] == "Dataset A"
        assert tasks[0]["status"] == "loading"
        assert tasks[0]["current"] == 50
        assert tasks[0]["total"] == 100

    def test_multiple_tasks(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "A").update("loading", "A loading", 10, 100)
        tracker.create_task("t2", "B").update("embedding", "B embedding", 50, 200)

        tasks = tracker.list_tasks()
        assert len(tasks) == 2
        names = {t["task_id"]: t["name"] for t in tasks}
        assert names == {"t1": "A", "t2": "B"}

    def test_cancel_task(self):
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "A")
        assert tracker.cancel_task("t1") is True
        with pytest.raises(CancelledError):
            pt.check_cancelled()

    def test_cancel_nonexistent(self):
        tracker = LoadingTasksTracker()
        assert tracker.cancel_task("nope") is False

    def test_cancel_all(self):
        tracker = LoadingTasksTracker()
        pt1 = tracker.create_task("t1", "A")
        pt2 = tracker.create_task("t2", "B")
        tracker.cancel_all()
        with pytest.raises(CancelledError):
            pt1.check_cancelled()
        with pytest.raises(CancelledError):
            pt2.check_cancelled()

    def test_remove_task(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "A")
        tracker.remove_task("t1")
        assert tracker.list_tasks() == []

    def test_mark_finished_and_auto_cleanup(self):
        """Finished tasks older than 5s are removed from list_tasks."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "A")
        pt.update("idle", "Done")
        # Mark as finished 10 seconds ago
        with tracker._lock:
            tracker._tasks["t1"]["finished_at"] = time.time() - 10
        tasks = tracker.list_tasks()
        assert len(tasks) == 0

    def test_recently_finished_still_visible(self):
        """Tasks finished less than 5s ago should still appear."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "A")
        pt.update("idle", "Done")
        tracker.mark_finished("t1")
        tasks = tracker.list_tasks()
        assert len(tasks) == 1

    def test_has_active_tasks(self):
        tracker = LoadingTasksTracker()
        assert tracker.has_active_tasks() is False
        pt = tracker.create_task("t1", "A")
        pt.update("loading", "Working")
        assert tracker.has_active_tasks() is True
        pt.update("idle", "Done")
        assert tracker.has_active_tasks() is False

    def test_reset_for_tests(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "A")
        tracker.reset_for_tests()
        assert tracker.list_tasks() == []


# ---------------------------------------------------------------------------
# Thread-local progress tests
# ---------------------------------------------------------------------------


class TestThreadLocalProgress:
    """Verify that per-thread progress callbacks work correctly."""

    def test_default_is_none(self):
        clear_thread_progress()
        assert get_thread_progress() is None

    def test_set_and_get(self):
        cb = lambda s, m="", c=0, t=0: None  # noqa: E731
        set_thread_progress(cb)
        assert get_thread_progress() is cb
        clear_thread_progress()
        assert get_thread_progress() is None

    def test_thread_isolation(self):
        """Each thread has its own callback."""
        results = {}
        barrier = threading.Barrier(2, timeout=5)

        def worker(name, cb):
            set_thread_progress(cb)
            barrier.wait()  # sync so both threads are alive
            results[name] = get_thread_progress()
            clear_thread_progress()

        cb_a = lambda s, m="", c=0, t=0: "a"  # noqa: E731
        cb_b = lambda s, m="", c=0, t=0: "b"  # noqa: E731

        t1 = threading.Thread(target=worker, args=("a", cb_a))
        t2 = threading.Thread(target=worker, args=("b", cb_b))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["a"] is cb_a
        assert results["b"] is cb_b


# ---------------------------------------------------------------------------
# get_progress() integration with loading tasks
# ---------------------------------------------------------------------------


class TestGetProgressWithLoadingTasks:
    """Verify that get_progress() checks loading tasks first."""

    def test_returns_active_loading_task(self):
        """get_progress() should return the active loading task, not the global tracker."""
        pt = loading_tasks.create_task("test_gp", "TestDS")
        pt.update("loading", "Embedding files", 42, 100, step=3, total_steps=4)
        try:
            progress = get_progress()
            assert progress["status"] == "loading"
            assert progress["message"] == "Embedding files"
            assert progress["current"] == 42
            assert progress["task_id"] == "test_gp"
        finally:
            loading_tasks.remove_task("test_gp")

    def test_returns_errored_task(self):
        """get_progress() should return an errored task even when idle."""
        pt = loading_tasks.create_task("test_err", "FailDS")
        pt.update("idle", "", error="Something went wrong")
        try:
            progress = get_progress()
            assert progress["error"] == "Something went wrong"
            assert progress["task_id"] == "test_err"
        finally:
            loading_tasks.remove_task("test_err")

    def test_falls_back_to_global(self):
        """With no loading tasks, get_progress() returns the global tracker."""
        from vtsearch.utils.progress import update_progress

        update_progress("idle", "Ready")
        progress = get_progress()
        assert progress["status"] == "idle"
        assert "task_id" not in progress


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestLoadingTasksEndpoint:
    """Test the /api/dataset/loading-tasks endpoint."""

    def test_returns_empty_when_no_tasks(self, client):
        resp = client.get("/api/dataset/loading-tasks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["tasks"] == []

    def test_returns_active_tasks(self, client):
        pt = loading_tasks.create_task("api_test", "API Test DS")
        pt.update("loading", "Processing", 25, 50)
        try:
            resp = client.get("/api/dataset/loading-tasks")
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data["tasks"]) == 1
            task = data["tasks"][0]
            assert task["task_id"] == "api_test"
            assert task["name"] == "API Test DS"
            assert task["status"] == "loading"
            assert task["current"] == 25
        finally:
            loading_tasks.remove_task("api_test")


class TestCancelTaskEndpoint:
    """Test the /api/dataset/cancel/<task_id> endpoint."""

    def test_cancel_existing_task(self, client):
        pt = loading_tasks.create_task("cancel_test", "CancelDS")
        try:
            resp = client.post("/api/dataset/cancel/cancel_test")
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            assert pt.is_cancelled
        finally:
            loading_tasks.remove_task("cancel_test")

    def test_cancel_nonexistent_returns_404(self, client):
        resp = client.post("/api/dataset/cancel/nonexistent_task")
        assert resp.status_code == 404


class TestImportEndpointsReturnTaskId:
    """Verify that import endpoints include task_id in the response."""

    def test_load_demo_returns_task_id(self, client):
        from vtsearch.datasets import DEMO_DATASETS

        demo_name = list(DEMO_DATASETS.keys())[0]

        with mock.patch("vtsearch.routes.datasets._run_importer_in_background", return_value="test_task_123"):
            resp = client.post(
                "/api/dataset/load-demo",
                json={"name": demo_name},
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["task_id"] == "test_task_123"

    def test_load_folder_returns_task_id(self, client, tmp_path):
        folder = tmp_path / "test_folder"
        folder.mkdir()
        (folder / "test.wav").write_bytes(b"fake")

        with mock.patch("vtsearch.routes.datasets._run_importer_in_background", return_value="folder_task"):
            resp = client.post(
                "/api/dataset/load-folder",
                json={"path": str(folder), "media_type": "sounds"},
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["task_id"] == "folder_task"


# ---------------------------------------------------------------------------
# Parallel load integration test
# ---------------------------------------------------------------------------


class TestParallelLoadConcurrency:
    """Test that two loads can run concurrently without interfering."""

    def test_two_concurrent_loads_have_separate_progress(self):
        """Two concurrent loading tasks must each have their own progress."""
        pt_a = loading_tasks.create_task("load_a", "Dataset A")
        pt_b = loading_tasks.create_task("load_b", "Dataset B")

        pt_a.update("loading", "Loading A", 10, 100)
        pt_b.update("embedding", "Embedding B", 50, 200)

        try:
            tasks = loading_tasks.list_tasks()
            by_id = {t["task_id"]: t for t in tasks}

            assert by_id["load_a"]["status"] == "loading"
            assert by_id["load_a"]["current"] == 10
            assert by_id["load_b"]["status"] == "embedding"
            assert by_id["load_b"]["current"] == 50
        finally:
            loading_tasks.remove_task("load_a")
            loading_tasks.remove_task("load_b")

    def test_cancel_one_does_not_affect_other(self):
        """Cancelling one task should not cancel the other."""
        pt_a = loading_tasks.create_task("cancel_a", "A")
        pt_b = loading_tasks.create_task("cancel_b", "B")

        try:
            loading_tasks.cancel_task("cancel_a")
            assert pt_a.is_cancelled
            assert not pt_b.is_cancelled
        finally:
            loading_tasks.remove_task("cancel_a")
            loading_tasks.remove_task("cancel_b")


class TestBuildDiversityTreeForContext:
    """Test the context-specific diversity tree builder."""

    def test_builds_tree_on_context(self):
        from vtsearch.utils.state_core import DatasetContext
        from vtsearch.utils.state_diversity import build_diversity_tree_for_context

        rng = np.random.default_rng(42)
        ctx = DatasetContext("test_diversity")
        for i in range(10):
            ctx.medias[i] = {
                "id": i,
                "embedding": rng.standard_normal(128).astype(np.float32),
            }

        build_diversity_tree_for_context(ctx)
        assert ctx.diversity_tree is not None

    def test_empty_context_sets_none(self):
        from vtsearch.utils.state_core import DatasetContext
        from vtsearch.utils.state_diversity import build_diversity_tree_for_context

        ctx = DatasetContext("test_empty")
        build_diversity_tree_for_context(ctx)
        assert ctx.diversity_tree is None

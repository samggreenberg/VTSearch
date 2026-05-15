"""Tests for parallel dataset loading: per-task progress, concurrent loads."""

import threading
import time
from unittest import mock

import numpy as np
import pytest

from vtsearch.concurrency.progress import (
    CancelledError,
    LoadingTasksTracker,
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

    def test_media_type_in_task(self):
        """Tasks created with media_type expose it in list_tasks output."""
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "Image DS", media_type="image")
        tracker.create_task("t2", "Audio DS", media_type="audio")
        tracker.create_task("t3", "No Type")

        tasks = {t["task_id"]: t for t in tracker.list_tasks()}
        assert tasks["t1"]["media_type"] == "image"
        assert tasks["t2"]["media_type"] == "audio"
        assert "media_type" not in tasks["t3"]

    def test_media_type_in_finished_task(self):
        """media_type is still visible on recently-finished tasks."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "DS", media_type="image")
        pt.update("idle", "Done")
        tracker.mark_finished("t1")

        tasks = tracker.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["media_type"] == "image"

    def test_set_dataset_id(self):
        """set_dataset_id updates the dataset_id on an existing task."""
        tracker = LoadingTasksTracker()
        tracker.create_task("t1", "DS")
        tasks = tracker.list_tasks()
        assert tasks[0].get("dataset_id") is None or tasks[0].get("dataset_id") == ""

        tracker.set_dataset_id("t1", "real-registry-id")
        tasks = tracker.list_tasks()
        assert tasks[0]["dataset_id"] == "real-registry-id"

    def test_set_dataset_id_nonexistent(self):
        """set_dataset_id on a missing task is a no-op."""
        tracker = LoadingTasksTracker()
        tracker.set_dataset_id("nope", "some-id")  # should not raise

    def test_set_dataset_id_visible_after_finish(self):
        """dataset_id is still visible on recently-finished tasks."""
        tracker = LoadingTasksTracker()
        pt = tracker.create_task("t1", "DS")
        tracker.set_dataset_id("t1", "ds-123")
        pt.update("idle", "Done")
        tracker.mark_finished("t1")

        tasks = tracker.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["dataset_id"] == "ds-123"


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
        from vtsearch.concurrency.progress import update_progress

        update_progress("idle", "Ready")
        progress = get_progress()
        assert progress["status"] == "idle"
        assert "task_id" not in progress


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestLoadingTasksTrackerEndpoint:
    """Test the loading_tasks tracker (streamed via the SSE `loading-tasks` channel)."""

    def test_returns_empty_when_no_tasks(self, client):
        assert loading_tasks.list_tasks() == []

    def test_returns_active_tasks(self, client):
        pt = loading_tasks.create_task("api_test", "API Test DS")
        pt.update("loading", "Processing", 25, 50)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            task = tasks[0]
            assert task["task_id"] == "api_test"
            assert task["name"] == "API Test DS"
            assert task["status"] == "loading"
            assert task["current"] == 25
        finally:
            loading_tasks.remove_task("api_test")


class TestLoadingTasksMediaType:
    """Test that the loading_tasks tracker exposes media_type."""

    def test_tasks_include_media_type(self, client):
        pt = loading_tasks.create_task("mt_test", "Image DS", media_type="image")
        pt.update("loading", "Working", 10, 100)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["media_type"] == "image"
        finally:
            loading_tasks.remove_task("mt_test")

    def test_tasks_omit_empty_media_type(self, client):
        pt = loading_tasks.create_task("mt_test2", "Unknown DS")
        pt.update("loading", "Working", 10, 100)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert "media_type" not in tasks[0]
        finally:
            loading_tasks.remove_task("mt_test2")


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

        with mock.patch("vtsearch.routes.datasets.load._run_importer_in_background", return_value="test_task_123"):
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

        with mock.patch("vtsearch.routes.datasets.load._run_importer_in_background", return_value="folder_task"):
            resp = client.post(
                "/api/dataset/load-folder",
                json={"path": str(folder), "media_type": "audio"},
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


class TestErrorVisibility:
    """Verify that loading errors are visible to the polling frontend."""

    def test_error_message_always_non_empty(self):
        """Exception handlers must produce non-empty error strings."""
        pt = loading_tasks.create_task("err_test", "Fail")
        # Simulate the backend error handler with an exception that has
        # an empty str() representation:
        e = Exception()
        error_msg = str(e) or repr(e) or "Unknown error during dataset loading"
        pt.update("idle", "", 0, 0, error=error_msg)
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["error"]  # must be truthy
            assert tasks[0]["error"] != ""
        finally:
            loading_tasks.remove_task("err_test")

    def test_errored_task_visible_after_finish(self):
        """Errored tasks stay visible in list_tasks longer than success tasks."""
        pt = loading_tasks.create_task("err_vis", "ErrorDS")
        pt.update("idle", "", 0, 0, error="Something broke")
        # Mark finished 10 seconds ago — non-error tasks would be cleaned up
        with loading_tasks._lock:
            loading_tasks._tasks["err_vis"]["finished_at"] = time.time() - 10
        try:
            tasks = loading_tasks.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["error"] == "Something broke"
        finally:
            loading_tasks.remove_task("err_vis")

    def test_errored_task_cleaned_after_30s(self):
        """Errored tasks are eventually cleaned up after 30 seconds."""
        pt = loading_tasks.create_task("err_old", "OldError")
        pt.update("idle", "", 0, 0, error="Old error")
        with loading_tasks._lock:
            loading_tasks._tasks["err_old"]["finished_at"] = time.time() - 35
        tasks = loading_tasks.list_tasks()
        assert len(tasks) == 0

    def test_errored_task_listed_after_finish(self, client):
        """list_tasks() returns errored tasks until the stale window elapses."""
        pt = loading_tasks.create_task("api_err", "API Err DS")
        pt.update("idle", "", 0, 0, error="Load failed")
        loading_tasks.mark_finished("api_err")
        try:
            tasks = loading_tasks.list_tasks()
            errored = [t for t in tasks if t.get("error")]
            assert len(errored) == 1
            assert errored[0]["error"] == "Load failed"
        finally:
            loading_tasks.remove_task("api_err")

    def test_concurrent_load_one_fails_other_succeeds(self):
        """When two tasks run and one errors, the error is visible while the other continues."""
        pt_ok = loading_tasks.create_task("ok_task", "Good DS")
        pt_fail = loading_tasks.create_task("fail_task", "Bad DS")

        pt_ok.update("loading", "Still working", 50, 100)
        pt_fail.update("idle", "", 0, 0, error="Download failed")
        loading_tasks.mark_finished("fail_task")

        try:
            tasks = loading_tasks.list_tasks()
            by_id = {t["task_id"]: t for t in tasks}

            assert "ok_task" in by_id
            assert by_id["ok_task"]["status"] == "loading"

            assert "fail_task" in by_id
            assert by_id["fail_task"]["error"] == "Download failed"
        finally:
            loading_tasks.remove_task("ok_task")
            loading_tasks.remove_task("fail_task")


class TestResetCancelSafety:
    """Verify that starting a new load does not interfere with running loads."""

    def test_reset_cancel_skipped_when_tasks_active(self):
        """dataset_progress.reset_cancel() must not fire when loads are in progress."""
        from vtsearch.concurrency.progress import dataset_progress

        # Create an active task
        pt = loading_tasks.create_task("active_task", "Running")
        pt.update("loading", "Downloading", 10, 100)

        # Cancel the global tracker (simulating user cancel of a previous load)
        dataset_progress.cancel()
        assert dataset_progress.is_cancelled

        try:
            # Start a new load — should NOT reset global cancel since a task is active
            from unittest.mock import patch

            from vtsearch.datasets.load_pipeline import _run_origin_load_in_background

            with patch("vtsearch.datasets.load_pipeline.threading.Thread"):
                _run_origin_load_in_background(
                    lambda: None,
                    {"importer": "test", "params": {}},
                )

            # The global cancel should still be set (not reset)
            assert dataset_progress.is_cancelled
        finally:
            loading_tasks.remove_task("active_task")
            dataset_progress.reset_cancel()

    def test_reset_cancel_allowed_when_no_tasks_active(self):
        """dataset_progress.reset_cancel() fires when no loads are in progress."""
        from vtsearch.concurrency.progress import dataset_progress

        dataset_progress.cancel()
        assert dataset_progress.is_cancelled

        from unittest.mock import patch

        from vtsearch.datasets.load_pipeline import _run_origin_load_in_background

        with patch("vtsearch.datasets.load_pipeline.threading.Thread"):
            task_id = _run_origin_load_in_background(
                lambda: None,
                {"importer": "test", "params": {}},
            )

        try:
            # The global cancel should have been reset
            assert not dataset_progress.is_cancelled
        finally:
            loading_tasks.remove_task(task_id)
            dataset_progress.reset_cancel()


class TestConcurrentModelLoading:
    """Verify that concurrent load_models() calls are serialised."""

    def test_concurrent_load_models_only_loads_once(self):
        """Two threads calling load_models() on the same embedder must not
        both execute _load_models_impl() concurrently — the lock should
        serialise them so the second caller sees the model already loaded."""
        from vtsearch.media.embedder import MediaEmbedder

        call_count = 0
        started = threading.Event()
        proceed = threading.Event()

        class FakeEmbedder(MediaEmbedder):
            name = "fake"
            media_type_id = "test"

            def __init__(self):
                super().__init__()
                self._model = None

            def _load_models_impl(self):
                nonlocal call_count
                if self._model is not None:
                    return
                started.set()
                proceed.wait(timeout=5)
                call_count += 1
                self._model = "loaded"

            def _embed_media_impl(self, media):
                return None

            def embed_text(self, text):
                return None

        emb = FakeEmbedder()

        t1 = threading.Thread(target=emb.load_models)
        t2 = threading.Thread(target=emb.load_models)

        t1.start()
        started.wait(timeout=5)
        # t1 is inside _load_models_impl holding the lock.
        # Start t2 — it must block on the lock.
        t2.start()
        # Let t1 finish.
        proceed.set()

        t1.join(timeout=5)
        t2.join(timeout=5)

        assert call_count == 1, f"_load_models_impl ran {call_count} times, expected 1"
        assert emb._model == "loaded"


class TestLoadingGates:
    """Verify the download/embed gates serialise (or pipeline) concurrent loads."""

    def test_second_load_waits_for_first(self):
        """With the default limit of 1, a second load should show 'Waiting…'
        for the download gate and only proceed after the first releases it."""
        from vtsearch.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
        )

        first_started = threading.Event()
        first_proceed = threading.Event()
        second_started = threading.Event()
        load_order = []

        def first_load(medias):
            load_order.append("first_start")
            first_started.set()
            first_proceed.wait(timeout=10)
            load_order.append("first_end")

        def second_load(medias):
            load_order.append("second_start")
            second_started.set()

        task1 = _run_origin_load_in_background(
            first_load,
            {"importer": "test1", "params": {}},
            name="First",
        )

        # Wait for first load to actually start running.
        assert first_started.wait(timeout=10)

        task2 = _run_origin_load_in_background(
            second_load,
            {"importer": "test2", "params": {}},
            name="Second",
        )

        # Second load should be waiting — give it a moment to start its
        # thread and hit the gate wait.
        time.sleep(0.3)
        assert not second_started.is_set(), "Second load should be queued, not running"

        # Check that the second task shows a "Waiting" message.
        task2_info = loading_tasks.get_tracker(task2)
        assert task2_info is not None
        status = task2_info.get()
        assert "Waiting" in status.get("message", "")

        # Let the first load finish.
        first_proceed.set()

        # Now the second should proceed.
        assert second_started.wait(timeout=10), "Second load never started after first finished"
        assert load_order[:2] == ["first_start", "first_end"]
        assert "second_start" in load_order

        # Clean up — wait for tasks to finish.
        deadline = time.time() + 10
        while loading_tasks.has_active_tasks() and time.time() < deadline:
            time.sleep(0.1)
        loading_tasks.remove_task(task1)
        loading_tasks.remove_task(task2)
        # Sanity: gates fully released after both tasks finish.
        assert _download_gate.active == 0

    def test_cancel_while_waiting_does_not_corrupt_gate(self):
        """Cancelling a queued task must not release the gate it never
        acquired, which would let extra loads through."""
        from vtsearch.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
        )

        first_started = threading.Event()
        first_proceed = threading.Event()

        def first_load(medias):
            first_started.set()
            first_proceed.wait(timeout=10)

        task1 = _run_origin_load_in_background(
            first_load,
            {"importer": "test1", "params": {}},
            name="First",
        )
        assert first_started.wait(timeout=10)

        # Start a second load — it will be queued on the download gate.
        task2 = _run_origin_load_in_background(
            lambda medias: None,
            {"importer": "test2", "params": {}},
            name="Second",
        )
        time.sleep(0.3)

        # Cancel the queued task before it acquires the gate.
        loading_tasks.cancel_task(task2)
        time.sleep(0.5)

        # The gate should still show exactly one holder (the first load).
        # If the cancel wrongly released, active would drop to 0.
        assert _download_gate.active == 1, "Cancelled task that never held the gate must not release it"

        # Clean up.
        first_proceed.set()
        deadline = time.time() + 10
        while loading_tasks.has_active_tasks() and time.time() < deadline:
            time.sleep(0.1)
        loading_tasks.remove_task(task1)
        loading_tasks.remove_task(task2)
        assert _download_gate.active == 0

    def test_download_and_embed_can_overlap(self):
        """When the importer signals the embedding phase, the download gate
        is released so a second dataset can start downloading in parallel
        even though the first hasn't finished embedding."""
        from vtsearch.datasets.load_pipeline import (
            _download_gate,
            _embed_gate,
            _run_origin_load_in_background,
        )

        first_in_embed = threading.Event()
        first_proceed = threading.Event()
        second_started = threading.Event()
        second_proceed = threading.Event()

        def first_load(medias):
            cb = get_thread_progress()
            assert cb is not None
            # Signal the importer's per-file embedding phase.  This must
            # cause the orchestrator to swap from the download gate to the
            # embed gate, freeing the download slot for task 2.
            cb("embedding", "Embedding…", 0, 1)
            first_in_embed.set()
            first_proceed.wait(timeout=10)

        def second_load(medias):
            second_started.set()
            second_proceed.wait(timeout=10)

        task1 = _run_origin_load_in_background(
            first_load,
            {"importer": "first", "params": {}},
            name="First",
        )
        assert first_in_embed.wait(timeout=10)

        # Task 1 should now be holding the embed gate, not the download gate.
        assert _embed_gate.active == 1
        assert _download_gate.active == 0

        # Task 2 should be able to acquire the download gate immediately and
        # start running its load_fn in parallel with task 1's embedding.
        task2 = _run_origin_load_in_background(
            second_load,
            {"importer": "second", "params": {}},
            name="Second",
        )
        assert second_started.wait(timeout=10), (
            "Second load never started — download gate was not released after the swap"
        )
        assert _download_gate.active == 1

        # Let both finish.
        first_proceed.set()
        second_proceed.set()
        deadline = time.time() + 10
        while loading_tasks.has_active_tasks() and time.time() < deadline:
            time.sleep(0.1)
        loading_tasks.remove_task(task1)
        loading_tasks.remove_task(task2)
        assert _download_gate.active == 0
        assert _embed_gate.active == 0

    def test_download_limit_is_user_configurable(self):
        """Bumping ``max_concurrent_dataset_downloads`` should let the second
        load start its download phase in parallel with the first."""
        from vtsearch import settings as settings_mod
        from vtsearch.datasets.load_pipeline import (
            _download_gate,
            _run_origin_load_in_background,
        )

        original = settings_mod.get_max_concurrent_dataset_downloads()
        settings_mod.set_max_concurrent_dataset_downloads(2)
        try:
            first_started = threading.Event()
            first_proceed = threading.Event()
            second_started = threading.Event()
            second_proceed = threading.Event()

            def first_load(medias):
                first_started.set()
                first_proceed.wait(timeout=10)

            def second_load(medias):
                second_started.set()
                second_proceed.wait(timeout=10)

            task1 = _run_origin_load_in_background(
                first_load,
                {"importer": "first", "params": {}},
                name="First",
            )
            assert first_started.wait(timeout=10)

            task2 = _run_origin_load_in_background(
                second_load,
                {"importer": "second", "params": {}},
                name="Second",
            )
            assert second_started.wait(timeout=10), (
                "Second load did not start in parallel — limit change did not take effect"
            )
            assert _download_gate.active == 2

            first_proceed.set()
            second_proceed.set()
            deadline = time.time() + 10
            while loading_tasks.has_active_tasks() and time.time() < deadline:
                time.sleep(0.1)
            loading_tasks.remove_task(task1)
            loading_tasks.remove_task(task2)
        finally:
            settings_mod.set_max_concurrent_dataset_downloads(original)


class TestConcurrencyGate:
    """Unit tests for the dynamic-limit ConcurrencyGate."""

    def test_blocking_acquire_when_limit_changes(self):
        """A waiter blocked at limit=1 must wake up when limit grows to 2."""
        from vtsearch.datasets.load_pipeline import ConcurrencyGate

        limit = [1]
        gate = ConcurrencyGate(lambda: limit[0])
        assert gate.acquire(blocking=False)

        # Second acquisition should block.
        acquired = threading.Event()

        def second():
            gate.acquire()
            acquired.set()

        t = threading.Thread(target=second, daemon=True)
        t.start()
        # Confirm it's actually blocked.
        assert not acquired.wait(timeout=0.3)

        # Raise the limit and notify — the waiter should wake up.
        with gate._cv:  # type: ignore[attr-defined]
            limit[0] = 2
            gate._cv.notify_all()  # type: ignore[attr-defined]
        assert acquired.wait(timeout=2)

        gate.release()
        gate.release()
        assert gate.active == 0

    def test_non_blocking_acquire_respects_limit(self):
        from vtsearch.datasets.load_pipeline import ConcurrencyGate

        gate = ConcurrencyGate(lambda: 2)
        assert gate.acquire(blocking=False)
        assert gate.acquire(blocking=False)
        assert not gate.acquire(blocking=False)
        gate.release()
        assert gate.acquire(blocking=False)
        gate.release()
        gate.release()

    def test_zero_limit_is_clamped_to_one(self):
        """A configured limit of 0 should still allow one acquisition."""
        from vtsearch.datasets.load_pipeline import ConcurrencyGate

        gate = ConcurrencyGate(lambda: 0)
        assert gate.acquire(blocking=False)
        assert not gate.acquire(blocking=False)
        gate.release()


class TestBuildDiversityTreeForContext:
    """Test the context-specific diversity tree builder."""

    def test_builds_tree_on_context(self):
        from vtsearch.state.core import DatasetContext
        from vtsearch.state.diversity import build_diversity_tree_for_context

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
        from vtsearch.state.core import DatasetContext
        from vtsearch.state.diversity import build_diversity_tree_for_context

        ctx = DatasetContext("test_empty")
        build_diversity_tree_for_context(ctx)
        assert ctx.diversity_tree is None

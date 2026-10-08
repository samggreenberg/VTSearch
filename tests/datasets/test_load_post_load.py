"""The load pipeline's ``post_load`` (#4252) and ``on_finished`` (#4616) hooks.

``_run_origin_load_in_background(post_load=...)`` is how the app starts a
user's AutoFind detectors on a freshly imported dataset.  What it owes the
caller: the hook runs once, only after a load that succeeded, with the new
dataset pinned as the thread's dataset context, after the load's own task has
parked terminal - and a hook that raises never turns a saved dataset into a
failed import.

``on_finished=...`` is how the app runs the functions an admin names with
``--on-dataset-imported``.  It owes them one ``DatasetImported`` per load that
succeeded or failed, none for a cancel, after ``post_load``, under the same
"a raising hook changes nothing" rule.

These drive a real load on the calling thread, the way
``test_load_terminal_state.py`` does.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from vtscore.concurrency.progress import CancelledError, loading_tasks
from vtscore.datasets.import_event import FAILED, SUCCEEDED, DatasetImported


def _sync_thread_factory():
    """Run the load body inline so the assertions see a finished load."""

    def fake_thread(target, daemon=True, name=None):
        t = mock.MagicMock()
        t.start = lambda: target()
        return t

    return fake_thread


def _fake_load(target_medias):
    """Minimal importer: one already-embedded media, so no model is needed."""
    target_medias[1] = {
        "id": 1,
        "media_type": "audio",
        "duration": 1.0,
        "file_size": 100,
        "md5": "post-load-md5",
        "embedder": "",
        "embedding": np.zeros(8, dtype=np.float32),
        "filename": "fake.wav",
        "category": "unknown",
        "origin": None,
        "origin_name": "fake.wav",
        "media_bytes": None,
        "media_string": None,
        "media_path": None,
    }


def _failing_load(_target_medias):
    raise RuntimeError("importer blew up")


def _cancelled_load(_target_medias):
    raise CancelledError("Operation cancelled by user")


_ORIGIN = {"importer": "test_post_load", "params": {}}


def _run_load(tmp_path, load_fn, post_load=None, on_finished=None) -> str:
    """Run one full load synchronously and return its task id."""
    from vtscore.datasets.load_pipeline import _run_origin_load_in_background
    from vtsearch import settings as settings_mod

    settings_mod.set_saved_datasets_dir(str(tmp_path / "saved"))
    with mock.patch(
        "vtscore.datasets.load_pipeline.threading.Thread",
        side_effect=_sync_thread_factory(),
    ):
        return _run_origin_load_in_background(
            load_fn,
            _ORIGIN,
            name="Post-load test",
            media_type="audio",
            embedder="clap",
            post_load=post_load,
            on_finished=on_finished,
        )


def _snapshot(task_id: str) -> dict:
    tracker = loading_tasks.get_tracker(task_id)
    assert tracker is not None, "the finished load should still be listed"
    return tracker.get()


class TestPostLoadHook:
    def test_runs_once_after_a_successful_load(self, isolated_settings, tmp_path):
        from vtscore.datasets.registry import list_datasets
        from vtscore.state.core import get_active_context

        calls: list[dict] = []

        def hook(ctx):
            calls.append({"ctx": ctx, "active": get_active_context(), "media_ids": sorted(ctx.medias)})

        _run_load(tmp_path, _fake_load, hook)

        assert len(calls) == 1
        (call,) = calls
        assert call["active"] is call["ctx"], "the new dataset must be the hook's active context"
        assert call["media_ids"] == [1]
        registered = [e["id"] for e in list_datasets()]
        assert call["ctx"].dataset_id in registered, "the hook sees the saved dataset, not the in-flight task"

    def test_task_has_parked_before_the_hook_runs(self, isolated_settings, tmp_path):
        """The hook's work is not part of the load: the row is already finished."""
        seen: list[dict] = []

        def hook(_ctx):
            active = [t for t in loading_tasks.list_tasks() if t["task_id"].startswith("_loading_")]
            seen.extend(active)

        _run_load(tmp_path, _fake_load, hook)

        assert seen, "the load's task should still be listed while the hook runs"
        assert all(t["status"] == "idle" and t["error"] is None for t in seen), seen

    def test_not_called_when_the_load_fails(self, isolated_settings, tmp_path):
        hook = mock.Mock()
        task_id = _run_load(tmp_path, _failing_load, hook)

        hook.assert_not_called()
        assert _snapshot(task_id)["error"]

    def test_a_raising_hook_does_not_fail_the_import(self, isolated_settings, tmp_path):
        def hook(_ctx):
            raise RuntimeError("AutoFind could not start")

        task_id = _run_load(tmp_path, _fake_load, hook)

        snapshot = _snapshot(task_id)
        assert snapshot["status"] == "idle"
        assert snapshot["error"] is None


class TestOnFinishedHook:
    def test_reports_a_successful_load(self, isolated_settings, tmp_path):
        from vtscore.datasets.registry import list_datasets

        events: list[DatasetImported] = []
        _run_load(tmp_path, _fake_load, on_finished=events.append)

        assert len(events) == 1
        (event,) = events
        assert event.outcome == SUCCEEDED
        assert event.dataset_id in [e["id"] for e in list_datasets()], "the saved dataset's id, not the task's"
        assert event.name == "Post-load test"
        assert event.user == "default"
        assert event.media_type == "audio"
        assert event.n_media == 1
        assert event.error == ""
        assert event.origin == _ORIGIN
        assert event.origin is not _ORIGIN, "a hook must not be able to edit the dataset's recorded origin"

    def test_reports_a_failed_load(self, isolated_settings, tmp_path):
        events: list[DatasetImported] = []
        task_id = _run_load(tmp_path, _failing_load, on_finished=events.append)

        assert len(events) == 1
        (event,) = events
        assert event.outcome == FAILED
        assert event.error == "importer blew up" == _snapshot(task_id)["error"]
        assert event.dataset_id == ""
        assert event.n_media == 0
        assert event.name == "Post-load test"
        assert event.user == "default"

    def test_silent_for_a_cancelled_load(self, isolated_settings, tmp_path):
        hook = mock.Mock()
        task_id = _run_load(tmp_path, _cancelled_load, on_finished=hook)

        assert _snapshot(task_id)["error"] == "Cancelled"
        hook.assert_not_called()

    def test_runs_after_post_load(self, isolated_settings, tmp_path):
        order: list[str] = []
        _run_load(
            tmp_path,
            _fake_load,
            post_load=lambda _ctx: order.append("post_load"),
            on_finished=lambda _event: order.append("on_finished"),
        )

        assert order == ["post_load", "on_finished"]

    def test_a_raising_hook_does_not_fail_the_import(self, isolated_settings, tmp_path):
        def hook(_event):
            raise RuntimeError("mail server unreachable")

        task_id = _run_load(tmp_path, _fake_load, on_finished=hook)

        snapshot = _snapshot(task_id)
        assert snapshot["status"] == "idle"
        assert snapshot["error"] is None

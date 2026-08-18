"""A finished import must be distinguishable from a wedged one (#3167).

An import that had *succeeded* left every external signal saying "still
loading": its loading task sat on the last message the pipeline happened to
emit, and the global ``dataset_progress`` tracker — the SSE ``dataset``
channel — kept whatever an unscoped model load had written into it.  Nothing
cleared either, because only the failure paths wrote a terminal state.  The
only way to learn the truth was to attach a profiler to the process.

These drive a real load through ``_run_origin_load_in_background`` on the
calling thread and assert both channels come to rest.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from vtscore.concurrency.progress import dataset_progress, loading_tasks


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
        "md5": "terminal-state-md5",
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


def _run_load(tmp_path) -> str:
    """Run one full load synchronously and return its task id."""
    from vtsearch import settings as settings_mod
    from vtscore.datasets.load_pipeline import _run_origin_load_in_background
    from vtscore.datasets.registry import list_datasets, unregister_dataset

    settings_mod.set_saved_datasets_dir(str(tmp_path / "saved"))
    with mock.patch(
        "vtscore.datasets.load_pipeline.threading.Thread",
        side_effect=_sync_thread_factory(),
    ):
        task_id = _run_origin_load_in_background(
            _fake_load,
            {"importer": "test_terminal", "params": {}},
            media_type="audio",
            embedder="clap",
        )
    for entry in list_datasets():
        unregister_dataset(entry["id"])
    return task_id


class TestSuccessfulLoadParksItsChannels:
    def test_task_tracker_leaves_loading(self, isolated_settings, tmp_path):
        task_id = _run_load(tmp_path)
        try:
            tracker = loading_tasks.get_tracker(task_id)
            assert tracker is not None, "the finished task should still be listed"
            snapshot = tracker.get()
            assert snapshot["status"] == "idle", (
                "a load that finished must say so; parking on the last 'loading' "
                f"message is what made a success look like a hang, got {snapshot!r}"
            )
            assert snapshot["error"] is None, f"the load succeeded, got {snapshot!r}"
        finally:
            loading_tasks.remove_task(task_id)

    def test_has_active_tasks_goes_false(self, isolated_settings, tmp_path):
        """The success path must not leave the tracker claiming to be busy."""
        task_id = _run_load(tmp_path)
        try:
            assert not loading_tasks.has_active_tasks(), (
                "nothing is running; a stale 'active' task also blocks the next load's cancel-flag reset"
            )
        finally:
            loading_tasks.remove_task(task_id)

    def test_global_dataset_channel_is_left_idle(self, isolated_settings, tmp_path):
        """The SSE ``dataset`` channel must not keep narrating a finished import.

        Anything that reports progress without a per-thread callback lands on
        this singleton — the observed symptom was it stuck on "Loading SigLIP
        processor…" for forty minutes with no loader thread in the process.
        """
        dataset_progress.update("loading", "Loading SigLIP processor…", 0, 0)

        task_id = _run_load(tmp_path)
        try:
            snapshot = dataset_progress.get()
            assert snapshot["status"] == "idle", (
                f"the last load out of the door parks an orphaned dataset channel; got {snapshot!r}"
            )
        finally:
            loading_tasks.remove_task(task_id)

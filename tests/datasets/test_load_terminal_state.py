"""A finished import must be distinguishable from a wedged one (#3167).

An import that had *succeeded* left every external signal saying "still
loading": its loading task sat on the last message the pipeline happened to
emit, and the global ``dataset_progress`` tracker — the SSE ``dataset``
channel — kept whatever an unscoped model load had written into it.  Nothing
cleared either, because only the failure paths wrote a terminal state.  The
only way to learn the truth was to attach a profiler to the process.

Half of that is now structural: the global tracker is gone, so an unscoped
model load resolves to a no-op instead of parking a channel nobody could
clear.  What is left to test is the half that still has to be got right —
the load's own per-task tracker reaching a terminal state — plus a guard
that no new process-wide sink has appeared to take the old one's place.

These drive a real load through ``_run_origin_load_in_background`` on the
calling thread and assert it comes to rest.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from vtscore.concurrency.progress import loading_tasks


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

    def test_a_finished_load_leaves_no_channel_narrating(self, isolated_settings, tmp_path):
        """No SSE channel may outlive the load it was describing.

        The observed symptom was the ``dataset`` channel stuck on "Loading
        SigLIP processor…" for forty minutes with no loader thread in the
        process, because anything reporting progress without a per-thread
        callback landed on a singleton that had no idea when the work ended.
        There is no such singleton now, so the guarantee is checked at the
        stream: once the load is over, every snapshot frame is terminal.
        """
        import json

        from vtscore.concurrency.events import initial_snapshot

        task_id = _run_load(tmp_path)
        try:
            for frame in initial_snapshot():
                assert not frame.startswith("event: dataset\n"), (
                    "the legacy dataset channel is back; it is a sink nothing terminates"
                )
                lines = frame.splitlines()
                channel = lines[0].removeprefix("event: ")
                payload = json.loads([ln for ln in lines if ln.startswith("data: ")][0].removeprefix("data: "))
                if channel in ("loading-tasks", "detector-loading-tasks"):
                    assert all(t.get("status") == "idle" for t in payload), (
                        f"{channel} still claims work after the load finished: {payload!r}"
                    )
                elif isinstance(payload, dict) and "status" in payload:
                    assert payload["status"] == "idle", f"{channel} is still narrating: {payload!r}"
        finally:
            loading_tasks.remove_task(task_id)

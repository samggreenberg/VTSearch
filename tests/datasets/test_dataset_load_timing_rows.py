"""A dataset import must feed the generic ``VTSEARCH_TIMING_RECORD`` sink.

``dataset_load`` is declared in the task registry like every other long-running
family, but for a while it was the one family with no ``record_task`` call site:
imports fed only the older, load-specific ``VTSEARCH_PROFILE_LOAD`` profiler, so
an admin who armed the documented recorder got rows for detector loads, text
sorts, and Finds while every import silently wrote nothing (#2845).

These tests drive a real load through ``_run_origin_load_in_background`` on the
calling thread and assert the rows land — and that they land in a shape the
fitter accepts, since rows the fitter rejects are as useless as rows never
written.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from vtscore.timing.fit import load_rows, normalize_row
from vtscore.timing.recorder import RECORD_ENV_VAR


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
        "md5": "timing-row-md5",
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


def _run_load(tmp_path, **kwargs) -> str:
    """Run one full load synchronously, returning nothing but the task's rows."""
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
            {"importer": "test_timing", "params": {}},
            media_type="audio",
            embedder="clap",
            **kwargs,
        )
    for entry in list_datasets():
        unregister_dataset(entry["id"])
    return task_id


class TestDatasetLoadRecordsTimingRows:
    def test_import_appends_rows_to_the_timing_sink(self, isolated_settings, tmp_path, monkeypatch):
        """The regression itself: an import with the recorder armed writes rows."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_load(tmp_path)

        assert sink.exists(), "an import with VTSEARCH_TIMING_RECORD armed must write rows (#2845)"
        rows = load_rows([str(sink)])
        assert rows, "the sink was created but empty"
        assert {r["task"] for r in rows} == {"dataset_load"}

    def test_rows_cover_the_declared_steps_and_survive_the_fitter(self, isolated_settings, tmp_path, monkeypatch):
        """Rows the fitter rejects are no better than rows never written, so the
        emitted shape is checked through ``normalize_row`` rather than by eye."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_load(tmp_path)

        rows = load_rows([str(sink)])
        # A completed load emits every declared step, including the ones it
        # skipped (recorded as a real zero, not dropped).
        assert {r["step"] for r in rows} == {"download", "extract", "load", "embed", "finalize"}
        assert all(r["ok"] for r in rows)
        assert all(r["complete"] for r in rows), "a load that reached finalize must be marked complete"
        assert all(r["media_type"] == "audio" and r["embedder"] == "clap" for r in rows)
        assert all(normalize_row(r) is not None for r in rows), "the fitter must accept every emitted row"

    def test_download_size_hint_rides_along_for_byte_scaled_steps(self, isolated_settings, tmp_path, monkeypatch):
        """``download``/``extract`` are fit as a per-MB rate, so a load that knows
        its archive size must record it or those two steps go unfittable."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        _run_load(tmp_path, download_size_mb_hint=42.5)

        rows = load_rows([str(sink)])
        assert rows and all(r["size_mb"] == 42.5 for r in rows)

    def test_failed_import_is_marked_not_ok(self, isolated_settings, tmp_path, monkeypatch):
        """A load that died partway measured an abort, not a cost; the fitter has
        to be able to drop it."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.setenv(RECORD_ENV_VAR, str(sink))

        from vtsearch import settings as settings_mod
        from vtscore.datasets.load_pipeline import _run_origin_load_in_background

        settings_mod.set_saved_datasets_dir(str(tmp_path / "saved"))
        with mock.patch(
            "vtscore.datasets.load_pipeline.threading.Thread",
            side_effect=_sync_thread_factory(),
        ):
            _run_origin_load_in_background(
                lambda target_medias: None,  # produces nothing -> load error
                {"importer": "test_timing_fail", "params": {}},
                media_type="audio",
            )

        rows = load_rows([str(sink)])
        assert rows, "even a failed load should record what it measured"
        assert all(not r["ok"] for r in rows)
        assert all(normalize_row(r) is None for r in rows), "the fitter must reject a failed run's rows"

    def test_disarmed_recorder_writes_nothing(self, isolated_settings, tmp_path, monkeypatch):
        """The off path stays free: no file, no subscription, no behaviour change."""
        sink = tmp_path / "timings.jsonl"
        monkeypatch.delenv(RECORD_ENV_VAR, raising=False)

        _run_load(tmp_path)

        assert not sink.exists()

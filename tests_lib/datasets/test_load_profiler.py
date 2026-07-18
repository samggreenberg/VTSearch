"""Tests for the env-gated ``_load_profiler`` measurement instrument.

``vtscore/datasets/stages/_load_profiler.py`` is the ``VTSEARCH_PROFILE_LOAD``
recorder behind ``docs/plans/progress-weight-calibration.md``: when armed, it
subscribes to the load's :class:`ProgressTracker`, stamps each phase boundary,
and writes one JSONL row per phase (and per finalize sub-slot) on
:meth:`finish`. It has zero behaviour effect when off. These tests exercise the
armed path end-to-end (records stages, emits the expected schema) plus the
off/no-op path, since a silent regression here corrupts calibration data with
nothing else to catch it.
"""

from __future__ import annotations

import json

import pytest

from vtscore.concurrency.progress import _PROGRESS_COMMON_EXTRAS, ProgressTracker
from vtscore.datasets.stages import _load_profiler
from vtscore.datasets.stages._load_profiler import (
    _NULL_PROFILER,
    LoadProfiler,
    note_finalize_slot,
    profiling_enabled,
    start_profiler,
)


@pytest.fixture
def clean_seen_embedders():
    """Reset the process-global cold/warm-model set around a test.

    ``_seen_embedders`` is module-level and drives ``cold_model``; isolate it so
    the flag is deterministic regardless of test order.
    """
    saved = set(_load_profiler._seen_embedders)
    _load_profiler._seen_embedders.clear()
    try:
        yield _load_profiler._seen_embedders
    finally:
        _load_profiler._seen_embedders.clear()
        _load_profiler._seen_embedders.update(saved)


def _make_tracker() -> ProgressTracker:
    """A tracker shaped like the real load tracker (exposes ``step``)."""
    return ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))


def _drive_phases(tracker: ProgressTracker) -> None:
    """Walk the tracker through the four load steps so the profiler stamps each
    phase boundary (download → model_load → embed → finalize)."""
    for step in (1, 2, 3, 4):
        tracker.update("loading", "", step=step, total_steps=4)


def test_extracting_status_records_its_own_phase(monkeypatch, tmp_path, clean_seen_embedders):
    """Step 1's two sub-phases (download vs extract) land as separate rows, so
    the fit can estimate an extraction rate independent of network bandwidth."""
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))
    tracker = _make_tracker()
    prof = start_profiler(tracker, "audio", "clap")
    tracker.update("downloading", "", 0, 100, step=1, total_steps=4)
    tracker.update("extracting", "", 0, 10, step=1, total_steps=4)
    tracker.update("loading", "", step=2, total_steps=4)
    tracker.update("embedding", "", step=3, total_steps=4)
    prof.finish(n=5, dataset_id="demo_x")
    phases = [r["phase"] for r in _read_rows(out)]
    assert phases[:4] == ["download", "extract", "model_load", "embed"]


def _read_rows(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -- off / no-op path -------------------------------------------------------


def test_profiling_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VTSEARCH_PROFILE_LOAD", raising=False)
    assert profiling_enabled() is False


def test_profiling_enabled_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(tmp_path / "prof.jsonl"))
    assert profiling_enabled() is True


def test_start_profiler_returns_noop_when_off(monkeypatch, tmp_path):
    monkeypatch.delenv("VTSEARCH_PROFILE_LOAD", raising=False)
    tracker = _make_tracker()
    prof = start_profiler(tracker, "audio", "clap")
    assert prof is _NULL_PROFILER

    # The no-op must accept the full call sequence and write nothing.
    prof.bind_thread()
    note_finalize_slot("dedup")  # not bound -> short-circuits
    _drive_phases(tracker)
    prof.finish(10, "some_dataset")
    assert not (tmp_path / "prof.jsonl").exists()


# -- armed path -------------------------------------------------------------


def test_records_phases_and_writes_schema(monkeypatch, tmp_path, clean_seen_embedders):
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))
    tracker = _make_tracker()

    prof = start_profiler(tracker, "image", "siglip")
    assert isinstance(prof, LoadProfiler)

    _drive_phases(tracker)
    prof.finish(1240, "caltech101_s")

    rows = _read_rows(out)
    phases = {r["phase"] for r in rows}
    assert phases == {"download", "model_load", "embed", "finalize"}

    expected_keys = {
        "device",
        "media_type",
        "embedder",
        "dataset_id",
        "n",
        "download_size_mb",
        "cold_model",
        "cold_download",
        "phase",
        "seconds",
    }
    for r in rows:
        assert set(r) == expected_keys
        assert r["media_type"] == "image"
        assert r["embedder"] == "siglip"
        assert r["dataset_id"] == "caltech101_s"
        assert r["n"] == 1240
        assert isinstance(r["device"], str) and r["device"]
        assert isinstance(r["seconds"], (int, float)) and r["seconds"] >= 0.0
        # A fast in-test load never spends >1s downloading.
        assert r["cold_download"] is False


def test_finalize_sub_slots_recorded(monkeypatch, tmp_path, clean_seen_embedders):
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))
    tracker = _make_tracker()

    prof = start_profiler(tracker, "audio", "clap")
    prof.bind_thread()  # so note_finalize_slot lands on this profiler
    _drive_phases(tracker)
    note_finalize_slot("dedup")
    note_finalize_slot("coverage_atlas")
    prof.finish(64, "gtzan_s")

    rows = _read_rows(out)
    finalize_slots = {r["phase"] for r in rows if r["phase"].startswith("finalize:")}
    assert finalize_slots == {"finalize:dedup", "finalize:coverage_atlas"}
    # Sub-slot rows carry the same base schema as main-phase rows.
    for r in rows:
        assert r["dataset_id"] == "gtzan_s"
        assert r["seconds"] >= 0.0


def test_cold_model_flips_to_warm_on_second_load(monkeypatch, tmp_path, clean_seen_embedders):
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))

    for _ in range(2):
        tracker = _make_tracker()
        prof = start_profiler(tracker, "audio", "clap")
        _drive_phases(tracker)
        prof.finish(8, "ds")

    rows = _read_rows(out)
    # First load's rows are cold (model not yet resident); second load's warm.
    first_load = rows[:4]
    second_load = rows[4:8]
    assert all(r["cold_model"] is True for r in first_load)
    assert all(r["cold_model"] is False for r in second_load)


def test_download_size_and_dataset_id_read_from_env(monkeypatch, tmp_path, clean_seen_embedders):
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))
    monkeypatch.setenv("VTSEARCH_PROFILE_DATASET_ID", "env_dataset")
    monkeypatch.setenv("VTSEARCH_PROFILE_DOWNLOAD_MB", "42.5")
    tracker = _make_tracker()

    prof = start_profiler(tracker, "text", "e5")
    _drive_phases(tracker)
    prof.finish(100)  # no explicit dataset_id -> falls back to the env var

    rows = _read_rows(out)
    assert rows
    for r in rows:
        assert r["dataset_id"] == "env_dataset"
        assert r["download_size_mb"] == 42.5


def test_explicit_dataset_id_overrides_env(monkeypatch, tmp_path, clean_seen_embedders):
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))
    monkeypatch.setenv("VTSEARCH_PROFILE_DATASET_ID", "env_dataset")
    tracker = _make_tracker()

    prof = start_profiler(tracker, "text", "e5")
    _drive_phases(tracker)
    prof.finish(5, "explicit_dataset")

    rows = _read_rows(out)
    assert all(r["dataset_id"] == "explicit_dataset" for r in rows)


def test_finish_unbinds_thread_local(monkeypatch, tmp_path, clean_seen_embedders):
    """After ``finish``, the thread-local binding is released so a later stray
    ``note_finalize_slot`` cannot land on a completed profiler."""
    out = tmp_path / "prof.jsonl"
    monkeypatch.setenv("VTSEARCH_PROFILE_LOAD", str(out))
    tracker = _make_tracker()

    prof = start_profiler(tracker, "audio", "clap")
    prof.bind_thread()
    _drive_phases(tracker)
    prof.finish(3, "ds")

    assert _load_profiler._active_profiler() is None

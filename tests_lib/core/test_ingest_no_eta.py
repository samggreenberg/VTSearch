"""Tests that ingest progress bars never publish a remaining-time estimate.

An import's rate is set by the network, the source's disks and the files
themselves, so its estimate swung too wildly to be worth showing. It used to be
an operator switch (``CoreConfig.hide_ingest_eta``, issue #4233); since issue
#4667 every ingest path builds its tracker with ``publish_eta=False``, which
keeps ``eta_seconds`` at ``None`` while the bar itself still moves.
"""

import dataclasses
import time

import pytest

from vtscore import config
from vtscore.concurrency.progress import (
    PROGRESS_COMMON_EXTRAS,
    LoadingTasksTracker,
    ProgressTracker,
    detector_loading_tasks,
    ingest_eta_hidden,
    loading_tasks,
)
from vtscore.config import core_config


@pytest.fixture
def clock(monkeypatch):
    """A controllable ``time.monotonic``, so an ETA appears deterministically."""
    now = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: now["t"])
    return now


def _run_job(tracker: ProgressTracker, clock) -> dict:
    """Drive a two-step job far enough that a normal tracker publishes an ETA."""
    tracker.update("downloading", "x", current=0, total=100, step=1, total_steps=2)
    clock["t"] += 10.0
    tracker.update("embedding", "x", current=50, total=100, step=2, total_steps=2)
    return tracker.get()


class TestPublishEta:
    def test_default_tracker_publishes_an_eta(self, clock):
        # Control: the same job does produce an estimate when not suppressed,
        # so the ``None`` below is the suppression, not a job too short to
        # estimate.
        snap = _run_job(ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS)), clock)
        assert snap["eta_seconds"] is not None

    def test_suppressed_tracker_never_publishes_an_eta(self, clock):
        snap = _run_job(ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS), publish_eta=False), clock)
        assert snap["eta_seconds"] is None

    def test_suppressed_tracker_still_moves_the_bar(self, clock):
        snap = _run_job(ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS), publish_eta=False), clock)
        assert snap["overall"] == pytest.approx(0.75)
        assert (snap["current"], snap["total"]) == (50, 100)

    def test_single_phase_eta_is_suppressed_too(self, clock):
        # No step structure: the per-phase ETA path, not the whole-job one.
        tracker = ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS), publish_eta=False)
        tracker.update("loading", "x", current=1, total=100)
        clock["t"] += 10.0
        tracker.update("loading", "x", current=50, total=100)
        assert tracker.get()["eta_seconds"] is None

    def test_subscribers_see_no_eta(self, clock):
        # The SSE feed reads subscriber snapshots, not ``get()``.
        seen: list[dict] = []
        tracker = ProgressTracker(extra_fields=dict(PROGRESS_COMMON_EXTRAS), publish_eta=False)
        tracker.subscribe(seen.append)
        _run_job(tracker, clock)
        assert seen and all(snap["eta_seconds"] is None for snap in seen)

    def test_create_task_passes_publish_eta_through(self, clock):
        bag = LoadingTasksTracker()
        tracker = bag.create_task("ingest_1", "ingest", publish_eta=False)
        _run_job(tracker, clock)
        assert bag.list_tasks()[0]["eta_seconds"] is None


@pytest.fixture
def old_switch_off(monkeypatch):
    """Install a ``CoreConfig`` builder with the retired switch explicitly off.

    Proves the ingest paths no longer consult it: a deployment that still
    carries ``hide_ingest_eta: false`` gets no ETA all the same.
    """
    replaced = dataclasses.replace(config.CoreConfig.from_settings(), hide_ingest_eta=False)
    monkeypatch.setattr(core_config, "_core_config_builder", lambda _path=None: replaced)


class TestDeprecatedSwitch:
    def test_ingest_eta_hidden_is_always_true(self, old_switch_off):
        assert ingest_eta_hidden() is True

    def test_core_config_still_accepts_the_field(self):
        # Kept so an out-of-tree ``CoreConfig(..., hide_ingest_eta=...)`` is not
        # a ``TypeError``.
        assert "hide_ingest_eta" in {f.name for f in dataclasses.fields(config.CoreConfig)}


class TestIngestPaths:
    """Every ingest path builds an ETA-less tracker, whatever the old switch says."""

    def test_import_task_tracker(self, old_switch_off, clock):
        from vtscore.datasets.load_pipeline import _start_import_task

        task = _start_import_task(prefix="_test_", display_name="test", total_steps=2)
        assert loading_tasks.get_tracker(task.task_id) is task.tracker
        snap = _run_job(task.tracker, clock)
        assert snap["eta_seconds"] is None
        assert snap["overall"] == pytest.approx(0.75)

    def test_labelset_ingest_tracker(self, old_switch_off, clock):
        from vtscore.datasets.ingest_task import start_ingest_task

        # A spawn that never runs the worker: only the tracker is under test.
        task_id = start_ingest_task([], {}, task_id="_test_ingest", name="test", spawn=lambda *_a, **_k: None)
        tracker = detector_loading_tasks.get_tracker(task_id)
        assert tracker is not None
        snap = _run_job(tracker, clock)
        assert snap["eta_seconds"] is None

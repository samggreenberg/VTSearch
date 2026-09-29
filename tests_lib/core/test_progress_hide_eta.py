"""Tests for withholding the ETA from ingest progress bars (issue #4233).

On some deployments an import's rate is too erratic for any timing profile to
predict, so an operator can switch the remaining-time estimate off for ingest
bars. The switch reaches the library as ``CoreConfig.hide_ingest_eta``; the
ingest paths turn it into ``publish_eta=False`` on the tracker they create,
which keeps ``eta_seconds`` at ``None`` while the bar itself still moves.
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


@pytest.fixture
def hide_ingest_eta(monkeypatch):
    """Install a ``CoreConfig`` builder with ``hide_ingest_eta`` set."""

    def _apply(value: bool) -> None:
        replaced = dataclasses.replace(config.CoreConfig.from_settings(), hide_ingest_eta=value)
        monkeypatch.setattr(core_config, "_core_config_builder", lambda _path=None: replaced)

    return _apply


def _run_job(tracker: ProgressTracker, clock) -> dict:
    """Drive a two-step job far enough that a normal tracker publishes an ETA."""
    tracker.update("downloading", "x", current=0, total=100, step=1, total_steps=2)
    clock["t"] += 10.0
    tracker.update("embedding", "x", current=50, total=100, step=2, total_steps=2)
    return tracker.get()


class TestPublishEta:
    def test_default_tracker_publishes_an_eta(self, clock):
        # Control: the same job does produce an estimate when not suppressed,
        # so the ``None`` below is the switch, not a job too short to estimate.
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


class TestIngestEtaHidden:
    def test_off_by_default(self):
        assert ingest_eta_hidden() is False

    def test_follows_core_config(self, hide_ingest_eta):
        hide_ingest_eta(True)
        assert ingest_eta_hidden() is True

    def test_no_builder_reads_as_shown(self, monkeypatch):
        # A library-only process with no builder installed keeps its ETAs.
        monkeypatch.setattr(core_config, "_core_config_builder", None)
        assert ingest_eta_hidden() is False


class TestIngestPaths:
    """Every ingest path builds its tracker from the policy."""

    @pytest.mark.parametrize("hidden", [True, False])
    def test_import_task_tracker(self, hide_ingest_eta, clock, hidden):
        from vtscore.datasets.load_pipeline import _start_import_task

        hide_ingest_eta(hidden)
        task = _start_import_task(prefix="_test_", family="dataset_load", display_name="test", total_steps=2)
        assert loading_tasks.get_tracker(task.task_id) is task.tracker
        snap = _run_job(task.tracker, clock)
        assert (snap["eta_seconds"] is None) is hidden

    @pytest.mark.parametrize("hidden", [True, False])
    def test_labelset_ingest_tracker(self, hide_ingest_eta, clock, hidden):
        from vtscore.datasets.ingest_task import start_ingest_task

        hide_ingest_eta(hidden)
        # A spawn that never runs the worker: only the tracker is under test.
        task_id = start_ingest_task([], {}, task_id="_test_ingest", name="test", spawn=lambda *_a, **_k: None)
        tracker = detector_loading_tasks.get_tracker(task_id)
        assert tracker is not None
        snap = _run_job(tracker, clock)
        assert (snap["eta_seconds"] is None) is hidden

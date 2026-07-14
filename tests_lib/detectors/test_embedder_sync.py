"""Unit tests for the switch-embedder → re-embed-labels cold path.

``vtscore.detectors.embedder_sync.maybe_start_label_reembed`` fires a
progress-tracked background re-embed when an already-loaded detector is
re-selected against a dataset whose active embedder differs from the one its
cached label vectors were built with.  It is Flask-free by construction (the
``spawn`` that starts the worker is injected), so it can be pinned down here in
the library tier without a Flask client.

The end-to-end wiring through ``POST /api/detectors/registry/load`` is covered
by ``tests/detectors/test_cross_embedder_switch.py``; these tests exercise the
decision logic directly:

* the trigger conditions (when work is / isn't scheduled),
* the initial progress report and the task metadata it stamps,
* the no-op fast paths (aligned cache, empty dataset, no labelset), and
* the spawned worker's progress / idle / error / cancel transitions.

``active_dataset_embedder_name`` (the marker derivation) is monkeypatched in
the ``maybe_start_label_reembed`` tests so the trigger is controlled directly;
it and ``embedder_display_name`` are also tested against real state / registry.
"""

from __future__ import annotations

import pytest

from vtscore.datasets.labelset import LabelSet
from vtscore.detectors import embedder_sync
from vtscore.state.core import DetectorContext

# A populated labelset the cold path can walk; the two elements make
# ``labelset.elements`` truthy so the empty-labelset no-op is skipped.
_LABELSET_DICT = {
    "labels": [
        {
            "md5": "a1" * 16,
            "label": "good",
            "origin": {"importer": "absent", "params": {}},
            "origin_name": "alpha.wav",
            "filename": "alpha.wav",
        },
        {
            "md5": "b2" * 16,
            "label": "bad",
            "origin": {"importer": "absent", "params": {}},
            "origin_name": "beta.wav",
            "filename": "beta.wav",
        },
    ]
}


class _CaptureSpawn:
    """A ``spawn``-style stand-in that records calls without starting a thread.

    Deterministic by design: the worker target is stored, not run, so tests
    that only care about the scheduling decision never touch a background
    thread, and tests that want to exercise the worker body run the captured
    target synchronously.
    """

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, target, *, name):
        self.calls.append((target, name))
        return None

    @property
    def target(self):
        return self.calls[0][0]

    @property
    def name(self):
        return self.calls[0][1]


def _det_ctx(embedder: str = "clap") -> DetectorContext:
    return DetectorContext("detabc123def456", name="my detector", media_type="audio", embedder=embedder)


def _with_cached_labelset(det_ctx: DetectorContext, media_type: str = "audio") -> DetectorContext:
    det_ctx.cached_labelset = LabelSet.from_dict(_LABELSET_DICT)
    det_ctx.cached_labelset_media_type = media_type
    return det_ctx


@pytest.fixture
def force_active_embedder(monkeypatch):
    """Return a setter that pins ``active_dataset_embedder_name``'s result."""

    def _set(value: str):
        monkeypatch.setattr(embedder_sync, "active_dataset_embedder_name", lambda det_ctx=None: value)

    return _set


class TestNoOpTriggerConditions:
    """The early-return paths that leave the cache untouched and spawn nothing."""

    def test_noop_when_active_embedder_unknown(self, force_active_embedder):
        force_active_embedder("")  # no active dataset / no marker
        det_ctx = _with_cached_labelset(_det_ctx("clap"))
        spawn = _CaptureSpawn()

        result = embedder_sync.maybe_start_label_reembed(det_ctx, {}, spawn=spawn)

        assert result is None
        assert spawn.calls == []
        # Nothing to key against → the stamp is left alone (not blanked).
        assert det_ctx.embedder == "clap"

    def test_noop_when_detector_embedder_empty(self, force_active_embedder):
        force_active_embedder("siglip")
        det_ctx = _with_cached_labelset(_det_ctx(""))
        spawn = _CaptureSpawn()

        result = embedder_sync.maybe_start_label_reembed(det_ctx, {}, spawn=spawn)

        assert result is None
        assert spawn.calls == []
        # An unstamped detector resolves lazily at first use; the cold path
        # must not stamp it here.
        assert det_ctx.embedder == ""

    def test_noop_when_embedders_match(self, force_active_embedder):
        force_active_embedder("clap")
        det_ctx = _with_cached_labelset(_det_ctx("clap"))
        spawn = _CaptureSpawn()

        result = embedder_sync.maybe_start_label_reembed(det_ctx, {}, spawn=spawn)

        assert result is None
        assert spawn.calls == []
        assert det_ctx.embedder == "clap"


class TestNoOpEmptyLabelset:
    """A mismatch with nothing cached to re-embed: re-stamp and return."""

    def test_noop_when_no_cached_labelset_restamps(self, force_active_embedder):
        force_active_embedder("siglip")
        det_ctx = _det_ctx("clap")  # cached_labelset defaults to None
        spawn = _CaptureSpawn()

        result = embedder_sync.maybe_start_label_reembed(det_ctx, {}, spawn=spawn)

        assert result is None
        assert spawn.calls == []
        # The stamp advances so a subsequent switch doesn't re-enter this branch.
        assert det_ctx.embedder == "siglip"

    def test_noop_when_labelset_has_no_elements_restamps(self, force_active_embedder):
        force_active_embedder("siglip")
        det_ctx = _det_ctx("clap")
        det_ctx.cached_labelset = LabelSet.from_dict({"labels": []})
        det_ctx.cached_labelset_media_type = "audio"
        spawn = _CaptureSpawn()

        result = embedder_sync.maybe_start_label_reembed(det_ctx, {}, spawn=spawn)

        assert result is None
        assert spawn.calls == []
        assert det_ctx.embedder == "siglip"


class TestTriggerSchedulesReembed:
    """A genuine mismatch with a cached labelset schedules the background job."""

    def test_returns_task_id_and_spawns_worker(self, force_active_embedder):
        force_active_embedder("siglip")
        det_ctx = _with_cached_labelset(_det_ctx("clap"))
        spawn = _CaptureSpawn()

        task_id = embedder_sync.maybe_start_label_reembed(det_ctx, {"name": "my detector"}, spawn=spawn)

        assert task_id == f"_detreembed_{det_ctx.detector_id[:8]}"
        assert len(spawn.calls) == 1
        target, name = spawn.calls[0]
        assert callable(target)
        assert name == f"det-reembed-{det_ctx.detector_id[:8]}"
        # The stamp is NOT advanced synchronously; ``populate_label_embeddings``
        # re-stamps it inside the worker once the re-embed actually completes.
        assert det_ctx.embedder == "clap"

    def test_task_records_detector_and_target_embedder(self, force_active_embedder):
        from vtscore.concurrency.progress import detector_loading_tasks

        force_active_embedder("siglip")
        det_ctx = _with_cached_labelset(_det_ctx("clap"))
        spawn = _CaptureSpawn()

        task_id = embedder_sync.maybe_start_label_reembed(det_ctx, {"name": "my detector"}, spawn=spawn)

        task = next(t for t in detector_loading_tasks.list_tasks() if t["task_id"] == task_id)
        assert task["detector_id"] == det_ctx.detector_id
        assert task["embedder"] == "siglip"  # the target embedder, not the stale one
        assert task["media_type"] == "audio"

    def test_initial_progress_message_names_target_embedder(self, force_active_embedder):
        from vtscore.concurrency.progress import detector_loading_tasks

        force_active_embedder("siglip")
        det_ctx = _with_cached_labelset(_det_ctx("clap"))
        spawn = _CaptureSpawn()

        task_id = embedder_sync.maybe_start_label_reembed(det_ctx, {}, spawn=spawn)

        tracker = detector_loading_tasks.get_tracker(task_id)
        snap = tracker.get()
        assert snap["status"] == "loading"
        expected_display = embedder_sync.embedder_display_name("siglip")
        assert snap["message"] == f"Re-resolving labels for {expected_display}…"
        assert snap["step"] == 1
        assert snap["total_steps"] == 1

    def test_media_type_prefers_cached_labelset_over_entry(self, force_active_embedder):
        from vtscore.concurrency.progress import detector_loading_tasks

        force_active_embedder("siglip")
        det_ctx = _with_cached_labelset(_det_ctx("clap"), media_type="audio")
        spawn = _CaptureSpawn()

        task_id = embedder_sync.maybe_start_label_reembed(
            det_ctx, {"media_type": "image"}, spawn=spawn
        )

        task = next(t for t in detector_loading_tasks.list_tasks() if t["task_id"] == task_id)
        assert task["media_type"] == "audio"

    def test_media_type_falls_back_to_entry(self, force_active_embedder):
        from vtscore.concurrency.progress import detector_loading_tasks

        force_active_embedder("siglip")
        det_ctx = _det_ctx("clap")
        det_ctx.cached_labelset = LabelSet.from_dict(_LABELSET_DICT)
        det_ctx.cached_labelset_media_type = ""  # no cached media type
        spawn = _CaptureSpawn()

        task_id = embedder_sync.maybe_start_label_reembed(
            det_ctx, {"media_type": "image"}, spawn=spawn
        )

        task = next(t for t in detector_loading_tasks.list_tasks() if t["task_id"] == task_id)
        assert task["media_type"] == "image"


class TestReembedWorkerBody:
    """Run the captured worker synchronously to pin down its progress reporting."""

    def _schedule(self, monkeypatch, train_stub):
        """Schedule a re-embed with ``train_from_labelset`` stubbed; return
        (task_id, worker_target, det_ctx)."""
        monkeypatch.setattr(embedder_sync, "active_dataset_embedder_name", lambda det_ctx=None: "siglip")
        monkeypatch.setattr(
            "vtscore.detectors.labelset_training.train_from_labelset",
            train_stub,
        )
        det_ctx = _with_cached_labelset(_det_ctx("clap"))
        spawn = _CaptureSpawn()
        task_id = embedder_sync.maybe_start_label_reembed(det_ctx, {"name": "my detector"}, spawn=spawn)
        return task_id, spawn.target, det_ctx

    def test_worker_reports_progress_then_settles_idle(self, monkeypatch):
        from vtscore.concurrency.progress import detector_loading_tasks

        seen: dict = {}

        def train_stub(det_ctx, labelset, *, media_type, snap, on_progress):
            seen["det_ctx"] = det_ctx
            seen["labelset"] = labelset
            seen["media_type"] = media_type
            on_progress("embedding", 3, 10)  # mid-flight progress from the embedder

        task_id, worker, det_ctx = self._schedule(monkeypatch, train_stub)

        progress_seen: list[tuple] = []
        detector_loading_tasks.get_tracker(task_id).subscribe(
            lambda s: progress_seen.append((s["status"], s["current"], s["total"]))
        )

        worker()

        # ``train_from_labelset`` received the cached labelset + resolved media type.
        assert seen["det_ctx"] is det_ctx
        assert seen["labelset"] is det_ctx.cached_labelset
        assert seen["media_type"] == "audio"
        # The embedder's mid-flight callback was surfaced as a loading update.
        assert ("loading", 3, 10) in progress_seen
        # Success settles the tracker back to idle with no error.
        final = detector_loading_tasks.get_tracker(task_id).get()
        assert final["status"] == "idle"
        assert not final.get("error")
        # The worker marks the task finished in its ``finally``.
        assert detector_loading_tasks.is_finished(task_id)

    def test_worker_records_training_error(self, monkeypatch):
        from vtscore.concurrency.progress import detector_loading_tasks

        def train_stub(det_ctx, labelset, *, media_type, snap, on_progress):
            raise ValueError("embedder blew up")

        task_id, worker, _ = self._schedule(monkeypatch, train_stub)

        worker()

        final = detector_loading_tasks.get_tracker(task_id).get()
        assert final["status"] == "idle"
        assert final["error"] == "embedder blew up"
        assert detector_loading_tasks.is_finished(task_id)

    def test_worker_records_cancellation(self, monkeypatch):
        from vtscore.concurrency.progress import CancelledError, detector_loading_tasks

        def train_stub(det_ctx, labelset, *, media_type, snap, on_progress):
            raise CancelledError("cancelled by user")

        task_id, worker, _ = self._schedule(monkeypatch, train_stub)

        worker()

        final = detector_loading_tasks.get_tracker(task_id).get()
        assert final["status"] == "idle"
        assert final["error"] == "Cancelled"
        assert detector_loading_tasks.is_finished(task_id)

    def test_worker_reports_repr_for_message_less_exception(self, monkeypatch):
        from vtscore.concurrency.progress import detector_loading_tasks

        class _Blank(Exception):
            def __str__(self):
                return ""

        def train_stub(det_ctx, labelset, *, media_type, snap, on_progress):
            raise _Blank()

        task_id, worker, _ = self._schedule(monkeypatch, train_stub)

        worker()

        final = detector_loading_tasks.get_tracker(task_id).get()
        assert final["status"] == "idle"
        # Falls back to repr()/a default rather than an empty error string.
        assert final["error"]


class TestEmbedderDisplayName:
    def test_empty_name_returns_empty(self):
        assert embedder_sync.embedder_display_name("") == ""

    def test_unknown_name_falls_back_to_id(self):
        assert embedder_sync.embedder_display_name("no_such_embedder_xyz") == "no_such_embedder_xyz"

    def test_known_name_returns_display_name(self):
        from vtscore.media import get_embedder

        # Pick any registered embedder and confirm the helper mirrors its
        # display name (falling back to the id when the embedder has none).
        det_ctx = _det_ctx("clap")
        name = det_ctx.embedder
        emb = get_embedder(name)
        assert embedder_sync.embedder_display_name(name) == (emb.display_name or name)


class TestActiveDatasetEmbedderName:
    def test_empty_dataset_yields_empty_marker(self):
        """No medias in the active context → no embedder marker."""
        from vtscore.state import clear_medias

        clear_medias()
        assert embedder_sync.active_dataset_embedder_name(_det_ctx("clap")) == ""

    def test_delegates_to_keying_embedder_for_snap(self, monkeypatch):
        captured: dict = {}

        def fake_keying(det_ctx, snap):
            captured["det_ctx"] = det_ctx
            captured["snap"] = snap
            return "siglip"

        monkeypatch.setattr("vtscore.embedding.binding.keying_embedder_for_snap", fake_keying)
        det_ctx = _det_ctx("clap")

        result = embedder_sync.active_dataset_embedder_name(det_ctx)

        assert result == "siglip"
        assert captured["det_ctx"] is det_ctx
        assert isinstance(captured["snap"], dict)

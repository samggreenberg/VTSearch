"""Progress channels must reach a terminal state when the work ends (#3167).

A ``server_folder`` import that had *succeeded* was indistinguishable from a
wedged one: the SSE ``dataset`` channel sat on ``"Loading SigLIP processor…"``
indefinitely, with no loader thread anywhere in the process.  Two mechanisms
produced that.

* **An unscoped model load narrated itself on a channel nobody terminates.**
  A thread that installs no ``progress_scope`` reports through the embedder's
  process-wide default sink, which the app used to wire to a global
  ``dataset_progress`` tracker that could not see when the work ended.  That
  is now closed at the root rather than patched at the load: the default sink
  the app installs resolves *per thread*, so an unscoped load on a thread that
  bound no tracker reaches a no-op and there is no channel left parked.
* **The load task's success path wrote no terminal state.**  Only the failure
  paths parked the tracker; a clean finish left it on its last "loading …"
  message until the finished entry aged out.  Still a live requirement, and
  still tested below.
"""

from __future__ import annotations

import numpy as np

from vtscore.datasets.load_pipeline import _park_load_terminal
from vtscore.datasets.stages.embedding import _ensure_model_loaded
from vtscore.media.embedder import MediaEmbedder


class _RecordingSink:
    """A progress callback that keeps every tick it was handed."""

    def __init__(self) -> None:
        self.ticks: list[tuple[str, str]] = []

    def __call__(self, status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        self.ticks.append((status, message))

    @property
    def statuses(self) -> list[str]:
        return [status for status, _msg in self.ticks]


class _NoisyEmbedder(MediaEmbedder):
    """Embedder whose model load announces itself, as the real ones do."""

    @property
    def name(self) -> str:
        return "noisy_test_embedder"

    @property
    def media_type_id(self) -> str:
        return "audio"

    def __init__(self) -> None:
        self._model = None

    def _load_models_impl(self) -> None:
        self._on_progress("loading", "Loading Noisy processor…", 0, 0)
        self._model = True

    def _embed_media_impl(self, media: dict) -> np.ndarray:
        return np.ones(3, dtype=np.float32)


def _fresh_embedder(default_sink) -> _NoisyEmbedder:
    emb = _NoisyEmbedder()
    emb.set_default_progress_callback(default_sink)
    return emb


class TestUnscopedModelLoad:
    """An unscoped load cannot leave a channel parked, because it has none."""

    def test_unscoped_load_reaches_the_default_sink_verbatim(self):
        """No synthetic terminal tick is appended any more.

        ``load_models`` used to send one because the sink it borrowed was a
        process-wide tracker that nothing else would ever clear.  Now the sink
        resolves per thread, so the load has nothing to clean up and reports
        exactly what it did.
        """
        sink = _RecordingSink()
        emb = _fresh_embedder(sink)

        emb.load_models()

        assert sink.ticks == [("loading", "Loading Noisy processor…")], (
            f"the load should narrate itself on the default sink and nothing more, got {sink.ticks!r}"
        )

    def test_app_default_sink_drops_ticks_from_an_unwatched_thread(self):
        """The sink the app installs is ``update_progress``, itself per-thread.

        With no tracker bound this is where an unscoped model load lands, and
        it has to be a no-op: a process-wide destination is exactly the #3167
        phantom, whoever writes to it.
        """
        from vtscore.concurrency.progress import clear_thread_progress, update_progress

        clear_thread_progress()
        emb = _fresh_embedder(update_progress)

        emb.load_models()  # must not raise, and must reach nothing

        assert emb._model is True

    def test_app_default_sink_reaches_a_bound_tracker(self):
        """Bound a tracker, and the same unscoped load lands on it."""
        from vtscore.concurrency.progress import (
            clear_thread_progress,
            set_thread_progress,
            update_progress,
        )

        sink = _RecordingSink()
        emb = _fresh_embedder(update_progress)
        set_thread_progress(sink)
        try:
            emb.load_models()
        finally:
            clear_thread_progress()

        assert sink.ticks == [("loading", "Loading Noisy processor…")], (
            f"a thread that bound a tracker must see the load it started, got {sink.ticks!r}"
        )

    def test_scoped_load_leaves_the_default_sink_alone(self):
        default = _RecordingSink()
        scoped = _RecordingSink()
        emb = _fresh_embedder(default)

        with emb.progress_scope(scoped):
            emb.load_models()

        assert default.ticks == [], (
            f"a caller with its own channel must not touch the default sink, got {default.ticks!r}"
        )
        assert scoped.statuses == ["loading"], (
            f"the caller owns termination of its own channel; load_models must not park it, got {scoped.ticks!r}"
        )

    def test_silent_progress_suppresses_the_load_entirely(self):
        sink = _RecordingSink()
        emb = _fresh_embedder(sink)

        with emb.silent_progress():
            emb.load_models()

        assert sink.ticks == [], f"a silenced warm-up must publish nothing, got {sink.ticks!r}"

    def test_already_loaded_model_publishes_nothing(self):
        sink = _RecordingSink()
        emb = _fresh_embedder(sink)
        emb._model = True

        emb.load_models()

        assert sink.ticks == [], f"a no-op load should not tick at all, got {sink.ticks!r}"


class TestImportModelLoadRouting:
    """The import's model load belongs on the import's own tracker."""

    def test_ensure_model_loaded_uses_the_supplied_callback(self):
        default = _RecordingSink()
        supplied = _RecordingSink()
        emb = _fresh_embedder(default)

        _ensure_model_loaded(emb, supplied)

        assert default.ticks == [], (
            "the import has a tracker of its own; routing its model load through "
            f"the global default is what made it a phantom, got {default.ticks!r}"
        )
        assert ("loading", "Loading Noisy processor…") in supplied.ticks, (
            f"the model's own progress should reach the load's tracker, got {supplied.ticks!r}"
        )


class TestLoadTaskTerminalState:
    """A finished load parks its tracker, success or failure."""

    def test_success_parks_the_tracker_idle(self):
        from vtscore.concurrency.progress import ProgressTracker

        tracker = ProgressTracker(extra_fields={"error": None, "step": None, "total_steps": None})
        tracker.update("loading", "Loading Noisy processor…", 0, 0)

        _park_load_terminal(tracker, 300)

        snapshot = tracker.get()
        assert snapshot["status"] == "idle", f"a completed load must leave 'loading', got {snapshot['status']!r}"
        assert snapshot["error"] is None
        assert "300" in snapshot["message"], f"the terminal message should say what landed, got {snapshot['message']!r}"

    def test_failure_keeps_its_error(self):
        from vtscore.concurrency.progress import ProgressTracker

        tracker = ProgressTracker(extra_fields={"error": None, "step": None, "total_steps": None})
        tracker.update("idle", "", 0, 0, error="Cancelled")

        _park_load_terminal(tracker, 0)

        snapshot = tracker.get()
        assert snapshot["error"] == "Cancelled", (
            "the failure path already parked this tracker; its error is the "
            f"terminal state and must survive, got {snapshot!r}"
        )

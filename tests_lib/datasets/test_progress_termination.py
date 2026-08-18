"""Progress channels must reach a terminal state when the work ends (#3167).

A ``server_folder`` import that had *succeeded* was indistinguishable from a
wedged one: the SSE ``dataset`` channel sat on ``"Loading SigLIP processor…"``
indefinitely, with no loader thread anywhere in the process.  Two mechanisms
produced that, and both are covered here.

* **An unscoped model load narrates itself on a channel nobody terminates.**
  A thread that installs no ``progress_scope`` reports through the embedder's
  process-wide default sink, which the app wires to the global
  ``dataset_progress`` tracker.  The sink cannot see when the work it is
  narrating ends, so ``load_models`` says so itself.
* **The load task's success path wrote no terminal state.**  Only the failure
  paths parked the tracker; a clean finish left it on its last "loading …"
  message until the finished entry aged out.
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
    """``load_models`` terminates the sink it borrowed."""

    def test_unscoped_load_ends_on_idle(self):
        sink = _RecordingSink()
        emb = _fresh_embedder(sink)

        emb.load_models()

        assert sink.ticks[0] == ("loading", "Loading Noisy processor…"), (
            f"the load should still narrate itself on the default sink, got {sink.ticks!r}"
        )
        assert sink.statuses[-1] == "idle", (
            "an unscoped model load must park the sink it borrowed; leaving it on "
            f"'loading' is the #3167 phantom, got {sink.ticks!r}"
        )

    def test_terminal_tick_fires_even_when_the_load_raises(self):
        sink = _RecordingSink()
        emb = _fresh_embedder(sink)

        def _boom() -> None:
            emb._on_progress("loading", "Loading Noisy processor…", 0, 0)
            raise RuntimeError("model download failed")

        emb._load_models_impl = _boom  # type: ignore[method-assign]

        try:
            emb.load_models()
        except RuntimeError:
            pass
        else:  # pragma: no cover - the stub always raises
            raise AssertionError("the failing load should have propagated")

        assert sink.statuses[-1] == "idle", (
            f"a failed load leaves the same phantom as a successful one, got {sink.ticks!r}"
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

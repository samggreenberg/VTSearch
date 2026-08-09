"""Concurrent dataset loads must not trample each other's embed progress.

Embedders are process-wide singletons (``vtscore.media._embedder_registry``),
so the embed stage's ``_on_progress`` save-and-restore used to be shared
mutable state: with two loads inside ``embed_missing`` on the same embedder
(the CPU embed gate allows several, and the staging path bypasses it
entirely), load B's assignment re-routed load A's still-running bulk embed
into B's tracker — mis-drawing B's bar, letting a cancel of B raise inside A,
and finally silencing B when A's ``finally`` restored a stale callback.

``MediaEmbedder._on_progress`` is now per-thread over a process-wide default,
so each load keeps its own tracker for the whole pass.
"""

from __future__ import annotations

import threading

import numpy as np

from vtscore.concurrency.progress import CancelledError
from vtscore.datasets.stages.embedding import embed_missing
from vtscore.media.embedder import MediaEmbedder


class _FakeEmbedder(MediaEmbedder):
    """Minimal single-vector embedder whose bulk pass emits two progress calls."""

    _NAME = "fake_test_embedder"

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def media_type_id(self) -> str:
        return "audio"

    def __init__(self) -> None:
        self._model = True

    def _load_models_impl(self) -> None:  # pragma: no cover - model is pre-set
        self._model = True

    def _embed_media_impl(self, media: dict) -> np.ndarray:
        return np.ones(3, dtype=np.float32)

    def _embed_media_bulk_impl(self, medias: list[dict]) -> list[np.ndarray]:
        total = len(medias)
        self._on_progress("embedding", "start", 0, total)
        self._park()
        self._on_progress("embedding", "end", total, total)
        return [np.ones(3, dtype=np.float32) for _ in medias]

    def _park(self) -> None:
        """Hook: block mid-bulk.  The plain embedder never parks."""


class _Gate:
    """One bulk pass's park: ``entered`` fires on arrival, ``release`` resumes it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()


class _GatedEmbedder(_FakeEmbedder):
    """Embedder that parks mid-bulk on a per-call gate, in call-arrival order.

    Parking lets a second thread enter its own ``embed_missing`` (installing its
    own callback) while the first is still mid-bulk, and lets the test resume
    the two passes one at a time — the exact interleaving that used to cross the
    two loads' progress wires.
    """

    _NAME = "gated_test_embedder"

    def __init__(self, gates: list[_Gate]) -> None:
        super().__init__()
        self._gates = list(gates)
        self._gate_lock = threading.Lock()

    def _park(self) -> None:
        with self._gate_lock:
            gate = self._gates.pop(0)
        gate.entered.set()
        assert gate.release.wait(timeout=10)


def _medias(n: int) -> dict[int, dict]:
    return {i: {"media_type": "audio", "embeddings": {}} for i in range(n)}


class _Recorder:
    """Progress callback that records every message it is handed."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.messages: list[str] = []

    def __call__(self, status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        self.messages.append(message)


class _registered:
    """Context manager registering *emb* in the process embedder registry."""

    def __init__(self, emb: MediaEmbedder) -> None:
        self._emb = emb

    def __enter__(self):
        from vtscore.media import _embedder_registry, register_embedder

        self._registry = _embedder_registry
        self._prev = _embedder_registry.get(self._emb.name)
        register_embedder(self._emb)
        return self._emb

    def __exit__(self, *exc):
        if self._prev is None:
            self._registry.pop(self._emb.name, None)
        else:
            self._registry[self._emb.name] = self._prev
        return False


def _overlapped_embed(emb: _GatedEmbedder, gates: list[_Gate], calls) -> None:
    """Run *calls* (``(medias, on_progress)`` pairs) overlapped inside the bulk pass.

    Every pass parks; the first is then resumed and joined **while the second is
    still parked**, so the second's callback is the most recently installed one
    when the first emits its remaining progress.  That ordering is what a shared
    ``_on_progress`` would mis-route, and it is deterministic — no reliance on
    which released thread happens to run first.
    """
    threads = []
    for medias, on_progress in calls:
        t = threading.Thread(target=embed_missing, args=(medias, emb.name), kwargs={"on_progress": on_progress})
        t.start()
        assert gates[len(threads)].entered.wait(timeout=10), "bulk pass never reached its park"
        threads.append(t)
    for gate, t in zip(gates, threads):
        gate.release.set()
        t.join(timeout=10)
        assert not t.is_alive()


class TestConcurrentEmbedProgressIsolation:
    def test_each_load_keeps_its_own_callback(self):
        """Two overlapping ``embed_missing`` calls report into their own trackers."""
        gates = [_Gate(), _Gate()]
        emb = _GatedEmbedder(gates)
        a, b = _Recorder("a"), _Recorder("b")
        medias_a, medias_b = _medias(2), _medias(3)

        with _registered(emb):
            _overlapped_embed(emb, gates, [(medias_a, a), (medias_b, b)])

        # Each recorder saw its own pass's announce plus both of its bulk
        # updates, and nothing from the other load.  Before the fix, A's "end"
        # landed in B's recorder because B had overwritten the shared slot.
        assert a.messages == ["Embedding 2 item(s)…", "start", "end"]
        assert b.messages == ["Embedding 3 item(s)…", "start", "end"]

    def test_cancel_of_one_load_does_not_abort_the_other(self):
        """A cancelling tracker raises only inside its own load's embed pass.

        Progress callbacks call ``check_cancelled()``, so a mis-routed callback
        does more than mis-draw a bar: it aborts whichever embed pass calls it.
        """
        gates = [_Gate(), _Gate()]
        emb = _GatedEmbedder(gates)
        survivor = _Recorder("survivor")
        medias_a, medias_b = _medias(2), _medias(2)

        def cancelling(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
            if message == "end":
                raise CancelledError("cancelled")

        with _registered(emb):
            _overlapped_embed(emb, gates, [(medias_a, survivor), (medias_b, cancelling)])

        # A ran to completion with its own progress intact…
        assert survivor.messages == ["Embedding 2 item(s)…", "start", "end"]
        assert all(m["embeddings"] for m in medias_a.values())
        # …while the cancel landed in B's own pass, which embedded nothing.
        assert not any(m["embeddings"] for m in medias_b.values())


class TestProgressScope:
    def test_scope_is_thread_local_over_a_shared_default(self):
        emb = _FakeEmbedder()
        default = _Recorder("default")
        emb.set_default_progress_callback(default)

        scoped = _Recorder("scoped")
        seen_in_thread: list[object] = []
        with emb.progress_scope(scoped):
            # Another thread never sees this thread's override.
            t = threading.Thread(target=lambda: seen_in_thread.append(emb._on_progress))
            t.start()
            t.join(timeout=10)
            assert emb._on_progress is scoped

        assert seen_in_thread == [default]
        # Restored to the process-wide default, not pinned to the scoped one.
        assert emb._on_progress is default

    def test_scope_restores_on_exception(self):
        emb = _FakeEmbedder()
        default = _Recorder("default")
        emb.set_default_progress_callback(default)

        try:
            with emb.progress_scope(_Recorder("scoped")):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert emb._on_progress is default

    def test_scopes_nest(self):
        emb = _FakeEmbedder()
        outer, inner = _Recorder("outer"), _Recorder("inner")
        with emb.progress_scope(outer):
            with emb.progress_scope(inner):
                assert emb._on_progress is inner
            assert emb._on_progress is outer

    def test_default_is_noop_until_wired(self):
        emb = _FakeEmbedder()
        # No callback anywhere: reads the no-op default rather than raising.
        emb._on_progress("embedding", "ignored", 0, 1)

    def test_plain_assignment_stays_thread_scoped(self):
        """The legacy ``emb._on_progress = cb`` idiom is a per-thread override."""
        emb = _FakeEmbedder()
        default = _Recorder("default")
        emb.set_default_progress_callback(default)

        assigned = _Recorder("assigned")
        emb._on_progress = assigned
        seen_in_thread: list[object] = []
        t = threading.Thread(target=lambda: seen_in_thread.append(emb._on_progress))
        t.start()
        t.join(timeout=10)

        assert emb._on_progress is assigned
        assert seen_in_thread == [default]

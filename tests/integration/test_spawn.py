"""Tests for ``vtsearch.threading.spawn``: background thread that
replays the calling thread's user / dataset / detector context."""

from __future__ import annotations

import threading

from vtsearch.auth import get_current_user, get_thread_user, set_thread_user
from vtsearch.state import (
    DatasetContext,
    DetectorContext,
    get_active_context,
    get_active_detector_context,
    register_context,
    register_detector_context,
    unregister_context,
    unregister_detector_context,
)
from vtsearch.threading import spawn
from vtscore.state.core import (
    set_thread_dataset_context,
    set_thread_detector_context,
)


def _wait(event: threading.Event) -> None:
    assert event.wait(timeout=5), "spawn body never ran"


class TestSpawnUserContext:
    def teardown_method(self):
        set_thread_user(None)

    def test_replays_thread_user(self):
        # ``set_thread_user`` writes the thread-local that
        # ``get_current_user`` falls back to outside a Flask request
        # context; perfect for verifying spawn's snapshot/replay
        # without standing up a multi-user login provider.
        set_thread_user("alice")
        captured: list[str] = []
        done = threading.Event()

        def body():
            captured.append(get_current_user())
            done.set()

        thread = spawn(body, name="test-spawn-user")
        _wait(done)
        thread.join(timeout=5)
        assert captured == ["alice"]

    def test_clears_user_in_thread_on_exit(self):
        set_thread_user("bob")
        observed_during: list[str | None] = []
        ran = threading.Event()

        def body():
            observed_during.append(get_thread_user())
            ran.set()

        thread = spawn(body, name="test-spawn-cleanup")
        _wait(ran)
        thread.join(timeout=5)
        assert observed_during == ["bob"]

        # Spawn a second job from a thread where no user is set; the
        # cleanup in the first spawn's ``finally`` ran on its own
        # daemon thread (already dead), so this just verifies the
        # snapshot captured from *this* thread is None, proving the
        # caller-side ``set_thread_user(None)`` below propagates
        # correctly.
        set_thread_user(None)
        observed_after: list[str | None] = []
        ran2 = threading.Event()

        def body2():
            observed_after.append(get_thread_user())
            ran2.set()

        thread2 = spawn(body2, name="test-spawn-cleanup-2")
        _wait(ran2)
        thread2.join(timeout=5)
        assert observed_after == [None]


class TestSpawnDatasetContext:
    def test_replays_dataset_context(self):
        ctx = DatasetContext("test-spawn-ds")
        register_context(ctx)
        try:
            set_thread_dataset_context(ctx)
            captured: list[DatasetContext] = []
            done = threading.Event()

            def body():
                captured.append(get_active_context())
                done.set()

            thread = spawn(body, name="test-spawn-ds")
            _wait(done)
            thread.join(timeout=5)
            assert captured[0] is ctx
        finally:
            set_thread_dataset_context(None)
            unregister_context("test-spawn-ds")

    def test_replays_detector_context(self):
        det_ctx = DetectorContext("test-spawn-det")
        register_detector_context(det_ctx)
        try:
            set_thread_detector_context(det_ctx)
            captured: list[DetectorContext | None] = []
            done = threading.Event()

            def body():
                captured.append(get_active_detector_context())
                done.set()

            thread = spawn(body, name="test-spawn-det")
            _wait(done)
            thread.join(timeout=5)
            assert captured[0] is det_ctx
        finally:
            set_thread_detector_context(None)
            unregister_detector_context("test-spawn-det")

    def test_args_and_kwargs_forwarded(self):
        seen: list[tuple] = []
        done = threading.Event()

        def body(a, b, *, kw):
            seen.append((a, b, kw))
            done.set()

        thread = spawn(body, 1, "two", name="test-spawn-args", kw="extra")
        _wait(done)
        thread.join(timeout=5)
        assert seen == [(1, "two", "extra")]

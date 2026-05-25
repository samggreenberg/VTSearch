"""Tests for ``vtsearch.threading.spawn`` — background thread that
replays the calling thread's user / dataset / detector context."""

from __future__ import annotations

import threading

from vtsearch.auth import (
    DefaultLoginProvider,
    LoginProvider,
    get_current_user,
    get_thread_user,
    set_login_provider,
    set_thread_user,
)
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


class _MultiUserProvider(LoginProvider):
    """Login provider whose ``get_current_user`` always returns the
    thread-local user — used so the spawn snapshot has something
    interesting to replay even outside a Flask request context.
    """

    name = "multi-user"

    def get_current_user(self):  # pragma: no cover - exercised indirectly
        return get_thread_user() or "default"

    def list_users(self):
        return ["default"]

    def supports_auth(self):
        return True

    def get_user_data_dir(self, username, base_data_dir):
        return base_data_dir / username


def _wait(event: threading.Event) -> None:
    assert event.wait(timeout=5), "spawn body never ran"


class TestSpawnUserContext:
    def setup_method(self):
        set_login_provider(_MultiUserProvider())

    def teardown_method(self):
        set_login_provider(DefaultLoginProvider())
        set_thread_user(None)

    def test_replays_thread_user(self):
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
        observed_after: list[str | None] = []
        ran = threading.Event()

        def body():
            observed_during.append(get_thread_user())
            ran.set()

        thread = spawn(body, name="test-spawn-cleanup")
        _wait(ran)
        thread.join(timeout=5)
        # The cleanup happens inside the spawn-managed thread, not the
        # caller's thread.  Verify by spawning a second target on a
        # *fresh* thread (Python's threading module reuses Thread objects
        # but not the underlying OS thread for a one-shot daemon).
        observed_after_evt = threading.Event()

        def body2():
            observed_after.append(get_thread_user())
            observed_after_evt.set()

        set_thread_user(None)
        thread2 = spawn(body2, name="test-spawn-cleanup-2")
        _wait(observed_after_evt)
        thread2.join(timeout=5)
        assert observed_during == ["bob"]
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

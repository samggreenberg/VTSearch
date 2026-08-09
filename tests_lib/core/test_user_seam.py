"""The library-tier current-user seam (:mod:`vtscore.user`).

The app installs a request-user resolver (``flask.g.user``); the library
falls back to a thread-local, then to ``"default"``.  Background jobs
capture whoever triggered them and replay that identity on the worker
thread, so per-user settings writes land in the right file.
"""

from __future__ import annotations

import threading

import pytest

from vtscore.concurrency.async_jobs import JobManager
from vtscore.user import (
    DEFAULT_USER,
    _default_request_user_resolver,
    get_current_user,
    get_thread_user,
    register_request_user_resolver,
    set_thread_user,
    thread_user,
)


@pytest.fixture(autouse=True)
def _restore_seam():
    """Leave the module-level resolver / thread-local as we found them."""
    yield
    register_request_user_resolver(_default_request_user_resolver)
    set_thread_user(None)


class TestResolutionOrder:
    def test_defaults_to_default_user(self):
        assert get_current_user() == DEFAULT_USER

    def test_thread_local_wins_over_default(self):
        with thread_user("alice"):
            assert get_current_user() == "alice"
        assert get_current_user() == DEFAULT_USER

    def test_registered_resolver_wins_over_thread_local(self):
        register_request_user_resolver(lambda: "from-request")
        with thread_user("alice"):
            assert get_current_user() == "from-request"

    def test_resolver_returning_none_falls_through(self):
        register_request_user_resolver(lambda: None)
        with thread_user("alice"):
            assert get_current_user() == "alice"
        assert get_current_user() == DEFAULT_USER

    def test_thread_user_restores_the_previous_value(self):
        set_thread_user("outer")
        with thread_user("inner"):
            assert get_thread_user() == "inner"
        assert get_thread_user() == "outer"

    def test_thread_local_does_not_leak_across_threads(self):
        seen: list[str] = []
        done = threading.Event()

        def worker():
            seen.append(get_current_user())
            done.set()

        with thread_user("alice"):
            t = threading.Thread(target=worker)
            t.start()
            done.wait(timeout=5)
            t.join(timeout=5)

        assert seen == [DEFAULT_USER]


class TestJobManagerCapturesTheUser:
    def test_start_captures_the_resolved_user_and_replays_it_on_the_worker(self):
        register_request_user_resolver(lambda: "bob")
        manager = JobManager("test-user-seam")
        seen: list[str] = []

        def target(job):
            seen.append(get_current_user())
            job.result = "ok"

        job = manager.start("sig", target)
        assert job.done_event.wait(timeout=10)

        assert job.status == "done", job.error
        assert job.user == "bob"
        # The worker thread has no request context; the captured user is
        # replayed via the thread-local so per-user work resolves.
        assert seen == ["bob"]

    def test_start_works_without_any_resolver_installed(self):
        """The library default must not require an app (issue #2931)."""
        manager = JobManager("test-user-seam-default")

        def target(job):
            job.result = get_current_user()

        job = manager.start("sig", target)
        assert job.done_event.wait(timeout=10)

        assert job.status == "done", job.error
        assert job.result == DEFAULT_USER
        assert job.user == DEFAULT_USER

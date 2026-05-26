"""Tests for the ``thread_user`` / ``thread_dataset_context`` /
``thread_detector_context`` context-manager scopes.

These helpers snapshot the prior thread-local value on entry and restore
it on exit, so a future pooled / reused worker thread cannot leak
identity or context across jobs.  The bare ``set_thread_*`` setters
remain available for tests, but new production code should use the
``with``-block scopes (see audit item M22).
"""

from __future__ import annotations

import threading

from vtsearch.auth import get_thread_user, set_thread_user, thread_user
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    get_thread_dataset_context,
    get_thread_detector_context,
    set_thread_dataset_context,
    set_thread_detector_context,
    thread_dataset_context,
    thread_detector_context,
)


class TestThreadUserScope:
    def test_sets_and_restores_to_none(self):
        set_thread_user(None)
        assert get_thread_user() is None
        with thread_user("alice"):
            assert get_thread_user() == "alice"
        assert get_thread_user() is None

    def test_restores_prior_value(self):
        set_thread_user("alice")
        try:
            with thread_user("bob"):
                assert get_thread_user() == "bob"
            assert get_thread_user() == "alice"
        finally:
            set_thread_user(None)

    def test_restores_on_exception(self):
        set_thread_user("alice")
        try:
            try:
                with thread_user("bob"):
                    assert get_thread_user() == "bob"
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            assert get_thread_user() == "alice"
        finally:
            set_thread_user(None)

    def test_nested_scopes_restore_outer(self):
        set_thread_user(None)
        with thread_user("alice"):
            assert get_thread_user() == "alice"
            with thread_user("bob"):
                assert get_thread_user() == "bob"
                with thread_user("carol"):
                    assert get_thread_user() == "carol"
                assert get_thread_user() == "bob"
            assert get_thread_user() == "alice"
        assert get_thread_user() is None

    def test_per_thread_isolation(self):
        """A thread spawned with no prior identity sees ``None`` even
        while another thread has a scope active.  Proves the scope is
        thread-local, not process-global."""
        set_thread_user("main_user")
        seen = {}
        ready = threading.Event()
        proceed = threading.Event()

        def worker():
            seen["before"] = get_thread_user()
            with thread_user("worker_user"):
                seen["inside"] = get_thread_user()
                ready.set()
                proceed.wait(timeout=5)
            seen["after"] = get_thread_user()

        t = threading.Thread(target=worker)
        try:
            t.start()
            ready.wait(timeout=5)
            # Main thread is still its own user; worker's scope is
            # invisible here.
            assert get_thread_user() == "main_user"
            proceed.set()
            t.join(timeout=5)
        finally:
            set_thread_user(None)

        assert seen["before"] is None
        assert seen["inside"] == "worker_user"
        assert seen["after"] is None


class TestThreadDatasetContextScope:
    def test_sets_and_restores_to_none(self):
        set_thread_dataset_context(None)
        ctx = DatasetContext("test_ds")
        with thread_dataset_context(ctx):
            assert get_thread_dataset_context() is ctx
        assert get_thread_dataset_context() is None

    def test_restores_prior_value(self):
        prior = DatasetContext("prior_ds")
        inner = DatasetContext("inner_ds")
        set_thread_dataset_context(prior)
        try:
            with thread_dataset_context(inner):
                assert get_thread_dataset_context() is inner
            assert get_thread_dataset_context() is prior
        finally:
            set_thread_dataset_context(None)

    def test_restores_on_exception(self):
        prior = DatasetContext("prior_ds")
        inner = DatasetContext("inner_ds")
        set_thread_dataset_context(prior)
        try:
            try:
                with thread_dataset_context(inner):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            assert get_thread_dataset_context() is prior
        finally:
            set_thread_dataset_context(None)


class TestThreadDetectorContextScope:
    def test_sets_and_restores_to_none(self):
        set_thread_detector_context(None)
        ctx = DetectorContext("test_det")
        with thread_detector_context(ctx):
            assert get_thread_detector_context() is ctx
        assert get_thread_detector_context() is None

    def test_restores_prior_value(self):
        prior = DetectorContext("prior_det")
        inner = DetectorContext("inner_det")
        set_thread_detector_context(prior)
        try:
            with thread_detector_context(inner):
                assert get_thread_detector_context() is inner
            assert get_thread_detector_context() is prior
        finally:
            set_thread_detector_context(None)

    def test_restores_on_exception(self):
        prior = DetectorContext("prior_det")
        inner = DetectorContext("inner_det")
        set_thread_detector_context(prior)
        try:
            try:
                with thread_detector_context(inner):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            assert get_thread_detector_context() is prior
        finally:
            set_thread_detector_context(None)


class TestCombinedScopes:
    def test_stacked_scopes_all_restore(self):
        """All three scopes can be stacked in a single ``with`` and
        each restores independently on exit — the pattern used by
        ``vtscore/labels/sync.py:_push_with_thread_context``."""
        set_thread_user("outer_user")
        outer_ds = DatasetContext("outer_ds")
        outer_det = DetectorContext("outer_det")
        set_thread_dataset_context(outer_ds)
        set_thread_detector_context(outer_det)
        try:
            inner_ds = DatasetContext("inner_ds")
            inner_det = DetectorContext("inner_det")
            with (
                thread_user("inner_user"),
                thread_dataset_context(inner_ds),
                thread_detector_context(inner_det),
            ):
                assert get_thread_user() == "inner_user"
                assert get_thread_dataset_context() is inner_ds
                assert get_thread_detector_context() is inner_det
            assert get_thread_user() == "outer_user"
            assert get_thread_dataset_context() is outer_ds
            assert get_thread_detector_context() is outer_det
        finally:
            set_thread_user(None)
            set_thread_dataset_context(None)
            set_thread_detector_context(None)

    def test_simulated_pool_reuse_does_not_leak(self):
        """Simulate a thread that runs two jobs back-to-back with
        different identities.  With the context-manager scopes, the
        second job sees the empty thread-local on entry — proving that
        a future ``ThreadPoolExecutor`` would not leak identity across
        jobs even though the same OS thread runs both."""
        observed = []
        proceed = threading.Event()

        def worker():
            # Job 1
            with thread_user("user_a"):
                observed.append(("inside_job_1", get_thread_user()))
            observed.append(("between_jobs", get_thread_user()))
            proceed.wait(timeout=5)
            # Job 2 — same OS thread, fresh identity
            with thread_user("user_b"):
                observed.append(("inside_job_2", get_thread_user()))
            observed.append(("after_job_2", get_thread_user()))

        t = threading.Thread(target=worker)
        t.start()
        proceed.set()
        t.join(timeout=5)

        assert observed == [
            ("inside_job_1", "user_a"),
            ("between_jobs", None),
            ("inside_job_2", "user_b"),
            ("after_job_2", None),
        ]

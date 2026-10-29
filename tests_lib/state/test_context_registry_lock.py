"""The context-store lookups must not queue behind ``_state_lock`` (#3869).

``_set_request_context`` resolves the SPA's ``X-Dataset-Id`` /
``X-Detector-Id`` headers on *every* request, so ``get_context`` /
``get_detector_context`` sit in front of routes that are supposed to stay
responsive while a vote or a load holds ``_state_lock`` for seconds. They
take ``_context_registry_lock`` instead, which is only ever held across a
dict operation.

The assertions here deliberately avoid a wall-clock threshold: a lookup that
queued on ``_state_lock`` could only return *after* the holder released it,
so "the holder is still holding" is both exact and flake-free.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    _state_lock,
    get_context,
    get_detector_context,
    list_loaded_dataset_ids,
    list_loaded_detector_ids,
    loaded_detector_contexts,
    register_context,
    register_detector_context,
    rekey_dataset_context,
    unregister_context,
    unregister_detector_context,
)


@contextmanager
def _state_lock_held_elsewhere() -> Iterator[threading.Event]:
    """Hold ``_state_lock`` on another thread for the body of the block.

    Yields an event that is set only once the holder has let go, so a test
    can assert its own call returned while the lock was still taken. The
    holder's own wait is bounded so a regression fails in seconds instead of
    hanging until the per-test timeout.
    """
    acquired = threading.Event()
    released = threading.Event()
    stop = threading.Event()

    def hold() -> None:
        with _state_lock:
            acquired.set()
            stop.wait(timeout=10)
        released.set()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert acquired.wait(timeout=5), "holder thread never acquired _state_lock"
    try:
        yield released
    finally:
        stop.set()
        holder.join(timeout=5)


class TestLookupsAreLockFree:
    def test_dataset_lookups_do_not_wait_on_state_lock(self):
        ctx = DatasetContext("ds-lockfree")
        register_context(ctx)
        try:
            with _state_lock_held_elsewhere() as released:
                assert get_context("ds-lockfree") is ctx
                assert get_context("ds-not-loaded") is None
                assert "ds-lockfree" in list_loaded_dataset_ids()
                assert not released.is_set(), "lookup returned only after _state_lock was freed"
        finally:
            unregister_context("ds-lockfree")

    def test_detector_lookups_do_not_wait_on_state_lock(self):
        det = DetectorContext("det-lockfree")
        register_detector_context(det)
        try:
            with _state_lock_held_elsewhere() as released:
                assert get_detector_context("det-lockfree") is det
                assert get_detector_context("det-not-loaded") is None
                assert "det-lockfree" in list_loaded_detector_ids()
                assert det in loaded_detector_contexts()
                assert not released.is_set(), "lookup returned only after _state_lock was freed"
        finally:
            unregister_detector_context("det-lockfree")


class TestRekeyDatasetContext:
    def test_moves_the_entry_and_updates_the_id(self):
        ctx = DatasetContext("task-123")
        register_context(ctx)
        try:
            assert rekey_dataset_context("task-123", "ds-real") is True
            assert ctx.dataset_id == "ds-real"
            assert get_context("ds-real") is ctx
            assert get_context("task-123") is None
        finally:
            unregister_context("ds-real")

    def test_same_id_keeps_the_entry(self):
        ctx = DatasetContext("ds-same")
        register_context(ctx)
        try:
            assert rekey_dataset_context("ds-same", "ds-same") is True
            assert get_context("ds-same") is ctx
        finally:
            unregister_context("ds-same")

    def test_unknown_id_is_a_no_op(self):
        assert rekey_dataset_context("nothing-here", "ds-new") is False
        assert get_context("ds-new") is None

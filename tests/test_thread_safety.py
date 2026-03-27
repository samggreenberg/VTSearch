"""Tests for thread-safe global state operations.

Validates that the ``_state_lock`` in ``vtsearch.utils.state`` correctly
serialises concurrent access to votes, click-times, label history, and
autorun detectors.  Also validates the ``_settings_lock`` in
``vtsearch.settings`` and the ``_progress_lock`` in
``vtsearch.models.progress``.
"""

import threading

from vtsearch.utils.state_core import (
    get_thread_dataset_context,
    get_thread_detector_context,
    set_thread_dataset_context,
    set_thread_detector_context,
)
from vtsearch.utils import (
    apply_label,
    apply_label_with_click_time,
    assign_click_time,
    bad_votes,
    good_votes,
    label_history,
    toggle_vote,
    vote_click_times,
)
import vtsearch.utils.state as _state
import vtsearch.utils.state_core as _core
import vtsearch.settings as _settings_mod
import vtsearch.models.progress as _progress_mod


class TestStateLock:
    """Verify that _state_lock exists and is an RLock."""

    def test_lock_is_rlock(self):
        assert isinstance(_state._state_lock, type(threading.RLock()))

    def test_lock_is_reentrant(self):
        """RLock should allow the same thread to acquire it multiple times."""
        with _state._state_lock:
            with _state._state_lock:
                pass  # should not deadlock


class TestToggleVote:
    """Test the atomic toggle_vote compound operation."""

    def test_toggle_good_on(self):
        toggle_vote(1, "good")
        assert 1 in good_votes
        assert 1 not in bad_votes

    def test_toggle_good_off(self):
        good_votes[1] = None
        toggle_vote(1, "good")
        assert 1 not in good_votes

    def test_toggle_bad_on(self):
        toggle_vote(1, "bad")
        assert 1 in bad_votes
        assert 1 not in good_votes

    def test_toggle_bad_off(self):
        bad_votes[1] = None
        toggle_vote(1, "bad")
        assert 1 not in bad_votes

    def test_toggle_good_replaces_bad(self):
        bad_votes[1] = None
        toggle_vote(1, "good")
        assert 1 in good_votes
        assert 1 not in bad_votes

    def test_toggle_bad_replaces_good(self):
        good_votes[1] = None
        toggle_vote(1, "bad")
        assert 1 in bad_votes
        assert 1 not in good_votes

    def test_toggle_records_label_history(self):
        toggle_vote(1, "good")
        assert len(label_history) == 1
        assert label_history[0][0] == 1
        assert label_history[0][1] == "good"

    def test_toggle_off_records_unlabel(self):
        good_votes[1] = None
        toggle_vote(1, "good")
        assert len(label_history) == 1
        assert label_history[0][1] == "unlabel"

    def test_toggle_assigns_click_time(self):
        toggle_vote(1, "good")
        assert 1 in vote_click_times

    def test_toggle_off_removes_click_time(self):
        good_votes[1] = None
        vote_click_times[1] = 1
        _core._set_click_counter(1)
        toggle_vote(1, "good")
        assert 1 not in vote_click_times


class TestApplyLabel:
    """Test the atomic apply_label compound operation."""

    def test_apply_good(self):
        apply_label(1, "good")
        assert 1 in good_votes
        assert 1 not in bad_votes

    def test_apply_bad(self):
        apply_label(1, "bad")
        assert 1 in bad_votes
        assert 1 not in good_votes

    def test_apply_replaces_existing(self):
        good_votes[1] = None
        apply_label(1, "bad")
        assert 1 not in good_votes
        assert 1 in bad_votes

    def test_apply_records_history(self):
        apply_label(1, "good")
        assert len(label_history) == 1
        assert label_history[0][1] == "good"

    def test_apply_no_click_time(self):
        """apply_label should NOT assign click times (imports have no click time)."""
        apply_label(1, "good")
        assert 1 not in vote_click_times


class TestApplyLabelWithClickTime:
    """Test the atomic apply_label_with_click_time compound operation."""

    def test_apply_with_click_time(self):
        apply_label_with_click_time(1, "good")
        assert 1 in good_votes
        assert 1 in vote_click_times

    def test_click_times_are_monotonic(self):
        apply_label_with_click_time(1, "good")
        apply_label_with_click_time(2, "bad")
        assert vote_click_times[2] > vote_click_times[1]


class TestConcurrentClickCounter:
    """Verify that _click_counter increments are atomic under contention."""

    def test_concurrent_assign_click_time(self):
        """Run many concurrent assign_click_time calls and verify all values are unique."""
        num_threads = 10
        calls_per_thread = 50
        results: list[int] = []
        lock = threading.Lock()

        def worker(start_id):
            for i in range(calls_per_thread):
                ct = assign_click_time(start_id + i)
                with lock:
                    results.append(ct)

        threads = []
        for t in range(num_threads):
            th = threading.Thread(target=worker, args=(t * calls_per_thread,))
            threads.append(th)

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        total = num_threads * calls_per_thread
        assert len(results) == total
        # All click times must be unique (no two threads got the same counter value)
        assert len(set(results)) == total
        # Counter values should be 1..total (exact set)
        assert set(results) == set(range(1, total + 1))


class TestConcurrentVoteToggle:
    """Verify that concurrent vote toggles don't corrupt state."""

    def test_concurrent_votes_on_different_media(self):
        """Concurrent votes on different media IDs should all succeed."""
        num_threads = 20
        errors = []
        ds_ctx = get_thread_dataset_context()
        det_ctx = get_thread_detector_context()

        def worker(media_id):
            set_thread_dataset_context(ds_ctx)
            set_thread_detector_context(det_ctx)
            try:
                toggle_vote(media_id, "good")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, num_threads + 1)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        for i in range(1, num_threads + 1):
            assert i in good_votes

    def test_concurrent_toggle_same_media(self):
        """Concurrent toggles on the same media should not corrupt state."""
        num_threads = 20
        errors = []

        def worker():
            try:
                toggle_vote(1, "good")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        # After an even number of toggles, media should be unlabeled;
        # after odd, it should be good. Either way, state is consistent.
        assert (1 in good_votes) != (num_threads % 2 == 0)


class TestConcurrentApplyLabel:
    """Verify that concurrent apply_label calls don't corrupt state."""

    def test_concurrent_label_import(self):
        """Simulate concurrent label imports on different media."""
        num_threads = 20
        errors = []
        ds_ctx = get_thread_dataset_context()
        det_ctx = get_thread_detector_context()

        def worker(media_id, label):
            set_thread_dataset_context(ds_ctx)
            set_thread_detector_context(det_ctx)
            try:
                apply_label(media_id, label)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(1, num_threads + 1):
            label = "good" if i % 2 == 0 else "bad"
            threads.append(threading.Thread(target=worker, args=(i, label)))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        for i in range(1, num_threads + 1):
            if i % 2 == 0:
                assert i in good_votes
            else:
                assert i in bad_votes


class TestConcurrentSetInclusion:
    """Verify that concurrent set_inclusion keeps in-memory and on-disk state in sync."""

    def test_concurrent_set_inclusion_memory_disk_sync(self, isolated_settings):
        """After concurrent writes, in-memory inclusion must equal the persisted value."""
        num_threads = 20
        iterations = 30
        errors = []

        def worker(value):
            try:
                for _ in range(iterations):
                    _state.set_inclusion(value)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i % 5,)) for i in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        # The critical invariant: in-memory value must match the on-disk value.
        in_memory = _state.get_inclusion()
        on_disk = _settings_mod.get_inclusion()
        assert in_memory == on_disk


class TestSettingsLock:
    """Verify that _settings_lock exists and is an RLock."""

    def test_lock_is_rlock(self):
        assert isinstance(_settings_mod._settings_lock, type(threading.RLock()))

    def test_lock_is_reentrant(self):
        """RLock should allow the same thread to acquire it multiple times."""
        with _settings_mod._settings_lock:
            with _settings_mod._settings_lock:
                pass  # should not deadlock


class TestConcurrentSettingsAccess:
    """Verify that concurrent settings reads and writes don't corrupt state."""

    def test_concurrent_get_set_volume(self):
        """Concurrent get/set on the same setting should not raise."""
        num_threads = 20
        errors = []

        def reader():
            try:
                for _ in range(50):
                    _settings_mod.get_volume()
            except Exception as e:
                errors.append(e)

        def writer(val):
            try:
                for _ in range(50):
                    _settings_mod.set_volume(val)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            if i % 2 == 0:
                threads.append(threading.Thread(target=reader))
            else:
                threads.append(threading.Thread(target=writer, args=(i / num_threads,)))

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        # Volume should be a valid float in [0, 1]
        vol = _settings_mod.get_volume()
        assert 0.0 <= vol <= 1.0

    def test_concurrent_get_all(self):
        """Concurrent get_all calls should not raise."""
        num_threads = 20
        errors = []

        def worker():
            try:
                for _ in range(50):
                    result = _settings_mod.get_all()
                    assert isinstance(result, dict)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors


class TestProgressLock:
    """Verify that _progress_lock exists and is an RLock."""

    def test_lock_is_rlock(self):
        assert isinstance(_progress_mod._progress_lock, type(threading.RLock()))

    def test_lock_is_reentrant(self):
        """RLock should allow the same thread to acquire it multiple times."""
        with _progress_mod._progress_lock:
            with _progress_mod._progress_lock:
                pass  # should not deadlock


class TestConcurrentProgressCache:
    """Verify that concurrent progress cache operations don't corrupt state."""

    def test_concurrent_clear_progress_cache(self):
        """Concurrent clears should not raise."""
        num_threads = 20
        errors = []

        def worker():
            try:
                for _ in range(50):
                    _progress_mod.clear_progress_cache()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors


class TestPluginRegistryLock:
    """Verify that PluginRegistry._ensure_discovered is thread-safe."""

    def test_registry_has_lock(self):
        from vtsearch.utils.registry import PluginRegistry

        reg = PluginRegistry(package="vtsearch.exporters", sentinel="EXPORTER", label="exporter")
        assert isinstance(reg._lock, type(threading.Lock()))

    def test_concurrent_first_access_discovers_once(self):
        """Concurrent .list() calls should trigger _discover exactly once."""
        from unittest.mock import patch
        from vtsearch.utils.registry import PluginRegistry

        reg = PluginRegistry(package="vtsearch.exporters", sentinel="EXPORTER", label="exporter")
        call_count = 0
        original_discover = reg._discover

        def counting_discover():
            nonlocal call_count
            call_count += 1
            original_discover()

        barrier = threading.Barrier(10)
        errors = []

        def worker():
            try:
                barrier.wait()
                reg.list()
            except Exception as e:
                errors.append(e)

        with patch.object(reg, "_discover", side_effect=counting_discover):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

        assert not errors
        assert call_count == 1, f"_discover called {call_count} times, expected 1"

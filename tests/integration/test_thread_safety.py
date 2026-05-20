"""Tests for thread-safe global state operations.

Validates that the ``_state_lock`` in ``vtsearch.state`` correctly
serialises concurrent access to votes, click-times, label history, and
autorun detectors.  Also validates the ``_settings_lock`` in
``vtsearch.settings`` and the ``_progress_lock`` in
``vtscore.detectors.labeling_progress``.
"""

import threading

from vtscore.state.core import (
    get_thread_dataset_context,
    get_thread_detector_context,
    set_thread_dataset_context,
    set_thread_detector_context,
)
from vtsearch.state import (
    apply_label,
    apply_label_with_click_time,
    assign_click_time,
    bad_votes,
    good_votes,
    label_history,
    toggle_vote,
    vote_click_times,
)
import vtsearch.state as _state
import vtscore.state.core as _core
import vtsearch.settings as _settings_mod
import vtscore.detectors.labeling_progress as _progress_mod


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


class TestSlowSettingsIODoesNotBlockOthers:
    """H29 regression: a slow settings I/O sink must not stall unrelated
    settings reads/writes.

    H28's fix moved ``_sync_to_source`` outside the file lock, and the
    H29 completion follow-up moved ``_atomic_write`` outside
    ``_settings_lock`` (it still runs under the per-file
    cross-process ``_file_lock``). Combined, this means:

    * A hung NFS/webhook ``source.save`` can't stall any settings access.
    * A slow local fsync only stalls writes to the *same* user's file
      (via the per-file lock); other users' writes and any reads
      proceed.
    """

    def test_slow_sync_to_source_does_not_block_reader(self, monkeypatch, isolated_settings):
        """One thread inside _sync_to_source must NOT freeze a reader."""
        # Plant a placeholder source config first so the setter under
        # test actually invokes _sync_to_source — only then install the
        # slow stub so the placeholder write itself stays fast.
        _settings_mod.set_settings_source_config({"source_name": "_h29_unused", "field_values": {}})

        in_sync = threading.Event()
        unblock = threading.Event()

        def slow_sync(username, data):
            in_sync.set()
            # Generous upper bound so the suite never hangs if something
            # unexpected goes wrong; the test releases this in <1s.
            unblock.wait(timeout=30)

        monkeypatch.setattr(_settings_mod, "_sync_to_source", slow_sync)

        errors: list[BaseException] = []
        reader_done = threading.Event()

        def slow_setter():
            try:
                _settings_mod.set_volume(0.42)
            except BaseException as exc:  # pragma: no cover - surfaced via errors
                errors.append(exc)

        def reader():
            try:
                _settings_mod.get_theme()
                _settings_mod.get_volume()
                reader_done.set()
            except BaseException as exc:  # pragma: no cover - surfaced via errors
                errors.append(exc)

        setter_thread = threading.Thread(target=slow_setter)
        setter_thread.start()
        assert in_sync.wait(timeout=5), "_sync_to_source was never called"

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        assert reader_done.wait(timeout=5), "Reader thread was blocked by the slow source — H29 has regressed"

        unblock.set()
        setter_thread.join(timeout=5)
        reader_thread.join(timeout=5)
        assert not setter_thread.is_alive()
        assert not reader_thread.is_alive()
        assert not errors, f"Threads raised: {errors!r}"

    def test_slow_atomic_write_does_not_block_other_users(self, monkeypatch, isolated_settings, tmp_path):
        """While user A's local fsync hangs, user B's set_volume must complete.

        ``_atomic_write`` runs under the per-file cross-process lock only
        (not under ``_settings_lock`` after the H29 follow-up), and each
        user has its own ``.lock`` file, so user B's setter is unaffected.
        """
        from vtsearch.auth import set_thread_user

        # Per-user files under tmp_path/<user>/user_settings.json so the
        # two users are truly isolated on disk.
        _settings_mod.set_user_data_dir_override(tmp_path)

        in_write_for_user_a = threading.Event()
        unblock = threading.Event()
        real_atomic_write = _settings_mod._atomic_write

        def selective_atomic_write(path, data):
            if "user_a" in str(path):
                in_write_for_user_a.set()
                unblock.wait(timeout=30)
            real_atomic_write(path, data)

        monkeypatch.setattr(_settings_mod, "_atomic_write", selective_atomic_write)

        errors: list[BaseException] = []
        user_b_done = threading.Event()

        def user_a_setter():
            try:
                set_thread_user("user_a")
                _settings_mod.set_volume(0.1)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                set_thread_user(None)

        def user_b_setter():
            try:
                set_thread_user("user_b")
                _settings_mod.set_volume(0.9)
                user_b_done.set()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                set_thread_user(None)

        ta = threading.Thread(target=user_a_setter)
        ta.start()
        assert in_write_for_user_a.wait(timeout=5), "user_a's _atomic_write was never reached"

        tb = threading.Thread(target=user_b_setter)
        tb.start()
        assert user_b_done.wait(timeout=5), "user_b's set_volume was blocked by user_a's hung fsync — H29 has regressed"

        unblock.set()
        ta.join(timeout=5)
        tb.join(timeout=5)
        _settings_mod.set_user_data_dir_override(None)
        assert not ta.is_alive()
        assert not tb.is_alive()
        assert not errors, f"Threads raised: {errors!r}"

    def test_slow_atomic_write_does_not_block_settings_reads(self, monkeypatch, isolated_settings):
        """A hung local fsync for the current user must NOT block other
        threads doing settings *reads* — those only need ``_settings_lock``,
        which is no longer held across file I/O.
        """
        in_write = threading.Event()
        unblock = threading.Event()
        real_atomic_write = _settings_mod._atomic_write

        def slow_atomic_write(path, data):
            in_write.set()
            unblock.wait(timeout=30)
            real_atomic_write(path, data)

        monkeypatch.setattr(_settings_mod, "_atomic_write", slow_atomic_write)

        errors: list[BaseException] = []
        reader_done = threading.Event()

        def slow_setter():
            try:
                _settings_mod.set_volume(0.42)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def reader():
            try:
                _settings_mod.get_theme()
                _settings_mod.get_volume()
                reader_done.set()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        setter_thread = threading.Thread(target=slow_setter)
        setter_thread.start()
        assert in_write.wait(timeout=5), "_atomic_write was never reached"

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()
        assert reader_done.wait(timeout=5), "Reader thread was blocked by the slow local fsync — H29 has regressed"

        unblock.set()
        setter_thread.join(timeout=5)
        reader_thread.join(timeout=5)
        assert not setter_thread.is_alive()
        assert not reader_thread.is_alive()
        assert not errors, f"Threads raised: {errors!r}"


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
        from vtscore.plugins import PluginRegistry

        reg = PluginRegistry(package="vtscore.exporters", sentinel="EXPORTER", label="exporter", eager=False)
        assert isinstance(reg._lock, type(threading.Lock()))

    def test_concurrent_first_access_discovers_once(self):
        """Concurrent .list() calls should trigger _discover exactly once.

        Uses ``eager=False`` so we can observe the deferred-discovery path —
        the default eager construction skips :meth:`_ensure_discovered` work
        on subsequent calls entirely, so the lock is uninteresting there.
        """
        from unittest.mock import patch
        from vtscore.plugins import PluginRegistry

        reg = PluginRegistry(package="vtscore.exporters", sentinel="EXPORTER", label="exporter", eager=False)
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


class TestStateProgressLockOrder:
    """Regression for audit M1: never hold ``_state_lock`` while acquiring ``_progress_lock``.

    The canonical lock order is ``_state_lock`` first, ``_progress_lock`` strictly
    outside it.  Every state→progress callsite (``set_vote``, ``toggle_vote``,
    ``clear_votes``, ``clear_medias``, ``set_inclusion``, ``register_detector_context``,
    ``unregister_detector_context``) must release ``_state_lock`` before calling
    into ``vtscore.detectors.labeling_progress``.  Otherwise a contributor adding
    code that takes the locks in the reverse order opens a deadlock window.
    """

    def _patch_capture(self, monkeypatch, attr: str) -> list[bool]:
        """Wrap ``labeling_progress.<attr>`` to record whether ``_state_lock`` was held.

        Returns the list that captures one ``_is_owned()`` reading per call.
        ``RLock._is_owned`` is a stable CPython attribute used by the
        ``threading`` module itself, so depending on it for an invariant test
        is acceptable here.
        """
        captured: list[bool] = []
        original = getattr(_progress_mod, attr)

        def wrapper(*args, **kwargs):
            captured.append(_state._state_lock._is_owned())
            return original(*args, **kwargs)

        monkeypatch.setattr(_progress_mod, attr, wrapper)
        return captured

    def test_set_vote_releases_state_lock_before_progress_invalidate(self, monkeypatch):
        held = self._patch_capture(monkeypatch, "invalidate_progress_cache_from")
        from vtsearch.state import set_vote

        # First vote: none→good, no invalidation (old == "none").
        set_vote(1, "good")
        # Second vote: good→bad, triggers invalidation (old == "good").
        set_vote(1, "bad")
        assert held, "invalidate_progress_cache_from was never called on good→bad"
        assert held == [False] * len(held), f"_state_lock held during progress invalidate: {held}"

    def test_toggle_vote_releases_state_lock_before_progress_invalidate(self, monkeypatch):
        held = self._patch_capture(monkeypatch, "invalidate_progress_cache_from")
        from vtsearch.state import toggle_vote

        # First toggle: none→good (no invalidate).
        toggle_vote(2, "good")
        # Second toggle: good→none (triggers invalidate).
        toggle_vote(2, "good")
        assert held, "invalidate_progress_cache_from was never called on toggle-off"
        assert held == [False] * len(held), f"_state_lock held during progress invalidate: {held}"

    def test_clear_votes_releases_state_lock_before_progress_clear(self, monkeypatch):
        held = self._patch_capture(monkeypatch, "clear_progress_cache")
        from vtsearch.state import clear_votes

        clear_votes()
        assert held, "clear_progress_cache was never called from clear_votes"
        assert held == [False] * len(held), f"_state_lock held during clear_progress_cache: {held}"

    def test_clear_medias_releases_state_lock_before_progress_clear(self, monkeypatch):
        held = self._patch_capture(monkeypatch, "clear_progress_cache")
        from vtsearch.state import clear_medias

        clear_medias()
        assert held, "clear_progress_cache was never called from clear_medias"
        assert held == [False] * len(held), f"_state_lock held during clear_progress_cache: {held}"

    def test_set_inclusion_releases_state_lock_before_progress_clear(self, monkeypatch, isolated_settings):
        held = self._patch_capture(monkeypatch, "clear_progress_cache")
        import vtsearch.state as _vstate

        # Two distinct values to force the change-detection branch that triggers a clear.
        current = _vstate.get_inclusion()
        new_value = (current + 1) % 11  # inclusion is in [-10, 10]; bump within range
        _vstate.set_inclusion(new_value)
        assert held, "clear_progress_cache was never called from set_inclusion"
        assert held == [False] * len(held), f"_state_lock held during clear_progress_cache: {held}"

    def test_vote_mutation_does_not_block_when_progress_lock_held(self):
        """Concurrent ``set_vote`` mutations must not deadlock when ``_progress_lock`` is held.

        With the pre-fix code, ``_set_vote_locked`` acquired ``_progress_lock``
        while still holding ``_state_lock``.  A second thread doing any
        ``_state_lock``-only mutation would then block — even though there's
        no reason it should — because the first thread is parked waiting for
        ``_progress_lock``.  After the fix, the first thread releases
        ``_state_lock`` before attempting ``_progress_lock``, so the second
        ``_state_lock``-only mutation proceeds independently.
        """
        from vtsearch.state import set_vote, good_votes

        # Set media 3's initial vote so the next ``set_vote(3, "bad")`` will trigger
        # progress-cache invalidation (old != "none").
        set_vote(3, "good")
        assert 3 in good_votes

        unblock = threading.Event()

        # Hold _progress_lock from a background thread to simulate a long-running
        # progress-cache reader.
        progress_held = threading.Event()

        def holder():
            with _progress_mod._progress_lock:
                progress_held.set()
                unblock.wait(timeout=10)

        bg = threading.Thread(target=holder)
        bg.start()
        assert progress_held.wait(timeout=5), "background thread never acquired _progress_lock"

        # In a separate thread, call set_vote(3, "bad") which will:
        #   1. acquire _state_lock, mutate, release _state_lock
        #   2. try to acquire _progress_lock (blocked because holder has it)
        invalidating_done = threading.Event()

        def invalidator():
            set_vote(3, "bad")
            invalidating_done.set()

        inv = threading.Thread(target=invalidator)
        inv.start()

        # While the invalidator is parked waiting on _progress_lock, an unrelated
        # _state_lock-only mutation must still complete promptly.
        other_done = threading.Event()

        def other():
            set_vote(4, "good")
            other_done.set()

        other_thread = threading.Thread(target=other)
        other_thread.start()
        assert other_done.wait(timeout=5), (
            "concurrent set_vote was blocked — invalidator is holding _state_lock while waiting "
            "on _progress_lock (M1 regression)"
        )
        other_thread.join(timeout=5)

        # Release _progress_lock; invalidator should finish.
        unblock.set()
        assert invalidating_done.wait(timeout=5), "invalidator did not complete after _progress_lock released"
        inv.join(timeout=5)
        bg.join(timeout=5)

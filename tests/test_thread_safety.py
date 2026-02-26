"""Tests for thread-safe global state operations.

Validates that the ``_state_lock`` in ``vtsearch.utils.state`` correctly
serialises concurrent access to votes, click-times, label history, and
favorite detectors.
"""

import threading

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
        _state._click_counter = 1
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

        def worker(media_id):
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

        def worker(media_id, label):
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

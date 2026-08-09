"""Tests for the intercept_tqdm_progress context manager.

These verify that tqdm progress bars are intercepted and forwarded to a
callback during model loading, and that tqdm behaviour is fully restored
after the context manager exits.
"""

import io
import threading

import pytest
import tqdm.auto
import tqdm.std

from vtscore.media.embedder import intercept_tqdm_progress


class TestInterceptTqdmProgress:
    """Unit tests for intercept_tqdm_progress."""

    def test_captures_determinate_progress(self):
        """A tqdm bar with a known total should forward updates to the callback."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            bar = tqdm.auto.tqdm(total=100, desc="Downloading model.safetensors")
            bar.update(30)
            bar.update(70)
            bar.close()

        # At minimum we expect the initial report and the two updates
        assert len(calls) >= 3
        # First call should be at creation (current=0)
        assert calls[0] == ("loading", "Downloading model.safetensors", 0, 100)
        # Last call should show completion
        assert calls[-1][2] == 100  # current
        assert calls[-1][3] == 100  # total

    def test_ignores_indeterminate_bars(self):
        """Bars with no total (indeterminate spinners) should be silently skipped."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            bar = tqdm.auto.tqdm(desc="Loading config")  # no total
            bar.update(1)
            bar.close()

        assert len(calls) == 0

    @pytest.mark.xfail(reason="partialmethod identity is unstable across attribute accesses")
    def test_restores_original_tqdm_after_exit(self):
        """After the context manager exits, tqdm should behave normally."""
        orig_init = tqdm.std.tqdm.__init__
        orig_update = tqdm.std.tqdm.update
        orig_close = tqdm.std.tqdm.close

        with intercept_tqdm_progress(lambda *a: None):
            # Inside the CM, methods are patched
            assert tqdm.std.tqdm.__init__ is not orig_init
            assert tqdm.std.tqdm.update is not orig_update

        # After the CM, methods are restored
        assert tqdm.std.tqdm.__init__ is orig_init
        assert tqdm.std.tqdm.update is orig_update
        assert tqdm.std.tqdm.close is orig_close

    @pytest.mark.xfail(reason="partialmethod identity is unstable across attribute accesses")
    def test_restores_on_exception(self):
        """If an exception occurs inside the CM, tqdm is still restored."""
        orig_init = tqdm.std.tqdm.__init__
        orig_update = tqdm.std.tqdm.update

        try:
            with intercept_tqdm_progress(lambda *a: None):
                raise ValueError("boom")
        except ValueError:
            pass

        assert tqdm.std.tqdm.__init__ is orig_init
        assert tqdm.std.tqdm.update is orig_update

    def test_prefers_bar_with_largest_total(self):
        """When multiple bars exist, the one with the largest total is reported."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            small_bar = tqdm.auto.tqdm(total=5, desc="Shards")
            big_bar = tqdm.auto.tqdm(total=600_000_000, desc="model.safetensors")
            big_bar.update(100_000_000)
            small_bar.update(1)
            big_bar.close()
            small_bar.close()

        # The big_bar update should have been reported (not the small_bar)
        big_updates = [c for c in calls if c[3] == 600_000_000]
        assert len(big_updates) >= 2  # init + at least one update
        # Verify the 100M update was captured
        assert any(c[2] == 100_000_000 for c in big_updates)

    def test_strips_trailing_colon_from_desc(self):
        """Trailing ': ' in tqdm descriptions should be stripped."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            bar = tqdm.auto.tqdm(total=10, desc="Loading checkpoint shards: ")
            bar.update(5)
            bar.close()

        # Description should have trailing ': ' stripped
        assert calls[0][1] == "Loading checkpoint shards"

    def test_disabled_bar_not_tracked(self):
        """A bar created with disable=True should not be tracked."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            bar = tqdm.auto.tqdm(total=100, desc="Silent", disable=True)
            bar.update(50)
            bar.close()

        assert len(calls) == 0

    def test_suppresses_console_output(self, capsys):
        """Intercepted bars should not write to stdout/stderr."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            bar = tqdm.auto.tqdm(total=100, desc="Loading weights")
            bar.update(50)
            bar.update(50)
            bar.close()

        captured = capsys.readouterr()
        # The native tqdm output (e.g. "Loading weights: 100%|███|")
        # should NOT appear on stdout or stderr
        assert "Loading weights" not in captured.out
        assert "Loading weights" not in captured.err
        # But the callback should still have received the progress
        assert len(calls) >= 3

    def test_suppresses_console_output_with_explicit_stderr(self, capsys):
        """Bars created with file=sys.stderr (like huggingface_hub) should still be suppressed."""
        import sys

        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with intercept_tqdm_progress(cb):
            # huggingface_hub's tqdm wrapper passes file=sys.stderr explicitly
            bar = tqdm.auto.tqdm(total=100, desc="model.safetensors", file=sys.stderr)
            bar.update(50)
            bar.update(50)
            bar.close()

        captured = capsys.readouterr()
        assert "model.safetensors" not in captured.out
        assert "model.safetensors" not in captured.err
        # Callback should still receive progress
        assert len(calls) >= 3

    def test_callback_receives_status_loading(self):
        """All forwarded calls should use status='loading'."""
        calls = []

        def cb(status, message, current, total):
            calls.append(status)

        with intercept_tqdm_progress(cb):
            bar = tqdm.auto.tqdm(total=50, desc="Weights")
            bar.update(25)
            bar.close()

        assert all(s == "loading" for s in calls)


class TestConcurrentTqdmInterception:
    """Two embedders can load at once; interception must survive that."""

    def test_interleaved_sessions_keep_progress_separate_and_restore(self):
        """Overlapping sessions must not corrupt tqdm for the rest of the process.

        Model loading is serialised per embedder *class*, so two embedders can
        be inside ``intercept_tqdm_progress`` simultaneously.  With naive
        save/patch/restore the second entrant saves the *patched* functions as
        its originals, and the ordering below (A enters, B enters, A exits, B
        exits) leaves ``tqdm.std.tqdm.__init__`` permanently bound to A's dead
        closure — so every later bar in the process is forwarded to A's stale
        callback.
        """
        a_calls: list[tuple] = []
        b_calls: list[tuple] = []
        errors: list[BaseException] = []

        a_entered = threading.Event()
        b_entered = threading.Event()
        a_bar_done = threading.Event()
        b_bar_done = threading.Event()
        a_exited = threading.Event()

        def run_bar(desc: str, total: int) -> None:
            bar = tqdm.auto.tqdm(total=total, desc=desc)
            bar.update(total)
            bar.close()

        def thread_a() -> None:
            try:
                with intercept_tqdm_progress(lambda *c: a_calls.append(c)):
                    a_entered.set()
                    assert b_entered.wait(timeout=10)
                    run_bar("A", 100)
                    a_bar_done.set()
                    assert b_bar_done.wait(timeout=10)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                a_entered.set()
                a_bar_done.set()
                a_exited.set()

        def thread_b() -> None:
            try:
                assert a_entered.wait(timeout=10)
                with intercept_tqdm_progress(lambda *c: b_calls.append(c)):
                    b_entered.set()
                    run_bar("B", 200)
                    b_bar_done.set()
                    assert a_exited.wait(timeout=10)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                b_entered.set()
                b_bar_done.set()

        threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()
        assert errors == []

        # Each session saw only its own bar, not the other thread's.
        assert a_calls and all(c[1] == "A" and c[3] == 100 for c in a_calls)
        assert b_calls and all(c[1] == "B" and c[3] == 200 for c in b_calls)

        # Both sessions are gone: a fresh bar must reach neither callback.
        before = (len(a_calls), len(b_calls))
        bar = tqdm.auto.tqdm(total=7, desc="After", file=io.StringIO())
        bar.update(7)
        bar.close()
        assert (len(a_calls), len(b_calls)) == before


class TestProgressTrackerUpdate:
    """Tests for ProgressTracker.update() merge semantics."""

    def test_update_preserves_unspecified_extra_fields(self):
        """Calling update() without an extra field should not reset it."""
        from vtscore.concurrency.progress import ProgressTracker

        tracker = ProgressTracker(extra_fields={"error": None, "staging_result": None})
        tracker.update("loading", error="something broke")
        # A subsequent update that doesn't mention 'error' should preserve it
        tracker.update("loading", message="still going")
        snap = tracker.get()
        assert snap["error"] == "something broke"

    def test_update_can_explicitly_overwrite_extra_field(self):
        """Passing an extra field explicitly should overwrite the previous value."""
        from vtscore.concurrency.progress import ProgressTracker

        tracker = ProgressTracker(extra_fields={"error": None})
        tracker.update("loading", error="first error")
        tracker.update("loading", error="second error")
        assert tracker.get()["error"] == "second error"

    def test_update_can_explicitly_clear_extra_field(self):
        """Passing None for an extra field should clear it."""
        from vtscore.concurrency.progress import ProgressTracker

        tracker = ProgressTracker(extra_fields={"error": None})
        tracker.update("loading", error="oops")
        tracker.update("idle", error=None)
        assert tracker.get()["error"] is None

    def test_update_preserves_multiple_extra_fields_independently(self):
        """Each extra field is preserved independently when not specified."""
        from vtscore.concurrency.progress import ProgressTracker

        tracker = ProgressTracker(extra_fields={"error": None, "staging_result": None})
        tracker.update("idle", staging_result={"path": "/tmp/x"})
        tracker.update("loading")
        snap = tracker.get()
        assert snap["staging_result"] == {"path": "/tmp/x"}
        assert snap["error"] is None  # was never set, stays at default

    def test_free_function_preserves_extra_fields(self):
        """The update_progress() wrapper should also preserve unspecified extras."""
        from vtscore.concurrency.progress import dataset_progress, get_progress, update_progress

        # Set error via the free function
        update_progress("loading", error="whoops")
        # Update without mentioning error
        update_progress("loading", message="progress")
        snap = get_progress()
        assert snap["error"] == "whoops"
        # Clean up for other tests
        dataset_progress.update("idle", error=None, staging_result=None)

"""Tests for the intercept_tqdm_progress context manager.

These verify that tqdm progress bars are intercepted and forwarded to a
callback during model loading, and that tqdm behaviour is fully restored
after the context manager exits.
"""

import tqdm.auto
import tqdm.std

from vtsearch.media.base import intercept_tqdm_progress


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

"""Tests for console progress output during embedding model preloading.

Verifies that ``preload_predicted_embedders`` prints intermediate status
messages and download progress bars to stdout so the user can see what is
happening during the (potentially long) startup phase.

Also tests ``intercept_weight_loading_progress`` which tracks tensor-level
progress during model weight loading via ``set_module_tensor_to_device``
and ``load_state_dict``.

Also tests ``load_pretrained_local_first`` which avoids network hangs by
preferring locally cached model files.
"""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vtsearch.media.embedder import intercept_weight_loading_progress, load_pretrained_local_first, timed_progress
from vtsearch.embedding.loader import _make_console_progress, predict_embedders_to_preload, preload_predicted_embedders


class TestMakeConsoleProgress:
    """Unit tests for the _make_console_progress wrapper."""

    def test_forwards_to_original_callback(self):
        """Every call should be forwarded to the original callback."""
        calls = []

        def original(status, message="", current=0, total=0):
            calls.append((status, message, current, total))

        cb = _make_console_progress(original)

        cb("loading", "Loading model...", 0, 0)
        cb("loading", "weights.bin", 50, 100)

        assert len(calls) == 2
        assert calls[0] == ("loading", "Loading model...", 0, 0)
        assert calls[1] == ("loading", "weights.bin", 50, 100)

    def test_prints_phase_messages(self, capsys):
        """Phase messages (total=0) should be printed on their own lines."""
        cb = _make_console_progress(lambda *a, **kw: None)

        cb("loading", "Loading CLAP model weights...", 0, 0)
        cb("loading", "Loading CLAP processor...", 0, 0)

        captured = capsys.readouterr()
        assert "Loading CLAP model weights..." in captured.out
        assert "Loading CLAP processor..." in captured.out

    def test_deduplicates_repeated_phase_messages(self, capsys):
        """The same phase message repeated should only be printed once."""
        cb = _make_console_progress(lambda *a, **kw: None)

        cb("loading", "Loading model...", 0, 0)
        cb("loading", "Loading model...", 0, 0)
        cb("loading", "Loading model...", 0, 0)

        captured = capsys.readouterr()
        assert captured.out.count("Loading model...") == 1

    def test_prints_progress_bar_for_measurable_progress(self, capsys):
        """Calls with total > 0 should render an ASCII progress bar."""
        cb = _make_console_progress(lambda *a, **kw: None)

        cb("loading", "model.safetensors", 50, 100)

        captured = capsys.readouterr()
        assert "model.safetensors" in captured.out
        assert "50%" in captured.out
        assert "[" in captured.out
        assert "]" in captured.out

    def test_progress_bar_completes_with_newline(self, capsys):
        """When current >= total, a newline should be emitted."""
        cb = _make_console_progress(lambda *a, **kw: None)

        cb("loading", "model.safetensors", 0, 100)
        cb("loading", "model.safetensors", 100, 100)

        captured = capsys.readouterr()
        # The completed bar should end with a newline
        assert "100%" in captured.out
        # Should end in a newline (not just a \r)
        lines = captured.out.split("\n")
        assert len(lines) >= 2  # at least one \n was written after completion

    def test_progress_bar_percentage_capped_at_100(self, capsys):
        """Overshoot (current > total) should cap at 100%."""
        cb = _make_console_progress(lambda *a, **kw: None)

        cb("loading", "weights", 150, 100)

        captured = capsys.readouterr()
        assert "100%" in captured.out

    def test_phase_after_progress_starts_new_line(self, capsys):
        """A phase message arriving mid-progress-bar should start a new line."""
        cb = _make_console_progress(lambda *a, **kw: None)

        # Start a progress bar (mid-way, not completed)
        cb("loading", "model.safetensors", 30, 100)
        # Now a phase message arrives
        cb("loading", "Warming up pipeline...", 0, 0)

        captured = capsys.readouterr()
        # The phase message should appear on its own line
        assert "Warming up pipeline..." in captured.out


class TestPreloadConsoleOutput:
    """Integration-style tests for console output during preload_predicted_embedders."""

    @patch("vtsearch.embedding.loader.predict_embedders_to_preload", return_value=["clap"])
    @patch("vtsearch.media.get_embedder")
    def test_prints_preloading_banner_and_progress(self, mock_get_embedder, _mock_predict, capsys):
        """preload_predicted_embedders should print the banner and forward progress to console."""
        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb._on_progress = lambda *a, **kw: None

        def fake_load_models():
            mock_emb._on_progress("loading", "Loading CLAP model weights...", 0, 0)
            mock_emb._on_progress("loading", "model.safetensors", 50, 100)
            mock_emb._on_progress("loading", "model.safetensors", 100, 100)
            mock_emb._on_progress("loading", "Warming up audio pipeline: importing libraries...", 1, 4)
            mock_emb._on_progress("loading", "Warming up audio pipeline: resampling JIT...", 2, 4)
            mock_emb._on_progress("loading", "Warming up audio pipeline: preprocessing...", 3, 4)
            mock_emb._on_progress("loading", "Warming up audio pipeline: running model...", 4, 4)

        mock_emb.load_models = fake_load_models
        mock_get_embedder.return_value = mock_emb

        result = preload_predicted_embedders()

        captured = capsys.readouterr()
        assert result == ["clap"]
        assert "Preloading clap embedder" in captured.out
        assert "Loading CLAP model weights..." in captured.out
        assert "model.safetensors" in captured.out
        assert "Warming up audio pipeline: importing libraries..." in captured.out

    @patch("vtsearch.embedding.loader.predict_embedders_to_preload", return_value=["clap"])
    @patch("vtsearch.media.get_embedder")
    def test_restores_original_callback_after_load(self, mock_get_embedder, _mock_predict):
        """The original _on_progress callback should be restored after load_models."""
        original_cb = MagicMock()
        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb._on_progress = original_cb
        mock_emb.load_models = MagicMock()
        mock_get_embedder.return_value = mock_emb

        preload_predicted_embedders()

        assert mock_emb._on_progress is original_cb

    @patch("vtsearch.embedding.loader.predict_embedders_to_preload", return_value=["clap"])
    @patch("vtsearch.media.get_embedder")
    def test_restores_callback_on_exception(self, mock_get_embedder, _mock_predict, capsys):
        """The original callback should be restored even when load_models raises."""
        original_cb = MagicMock()
        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb._on_progress = original_cb
        mock_emb.load_models.side_effect = RuntimeError("boom")
        mock_get_embedder.return_value = mock_emb

        result = preload_predicted_embedders()

        assert mock_emb._on_progress is original_cb
        assert result == []
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @patch("vtsearch.embedding.loader.predict_embedders_to_preload", return_value=[])
    def test_no_predicted_embedders_produces_no_output(self, _mock_predict, capsys):
        """When there are no predicted embedders, nothing should be printed."""
        result = preload_predicted_embedders()

        assert result == []
        captured = capsys.readouterr()
        assert captured.out == ""


class TestPredictEmbeddersToPreload:
    """Smart-preload prediction derived from dataset + detector registries."""

    def test_empty_registries_predict_nothing(self):
        with (
            patch("vtsearch.datasets.registry.list_datasets", return_value=[]),
            patch("vtsearch.detectors.registry.list_detectors", return_value=[]),
        ):
            assert predict_embedders_to_preload() == []

    def test_dataset_with_explicit_embedder_is_preloaded(self):
        with (
            patch(
                "vtsearch.datasets.registry.list_datasets", return_value=[{"media_type": "audio", "embedder": "clap"}]
            ),
            patch("vtsearch.detectors.registry.list_detectors", return_value=[]),
        ):
            assert predict_embedders_to_preload() == ["clap"]

    def test_dataset_without_embedder_uses_media_type_default(self):
        fake_default = MagicMock()
        fake_default.name = "siglip"
        with (
            patch("vtsearch.datasets.registry.list_datasets", return_value=[{"media_type": "image", "embedder": ""}]),
            patch("vtsearch.detectors.registry.list_detectors", return_value=[]),
            patch("vtsearch.media.embedders_for_type", return_value=[fake_default]),
        ):
            assert predict_embedders_to_preload() == ["siglip"]

    def test_unique_embedders_only_no_duplicates(self):
        with (
            patch(
                "vtsearch.datasets.registry.list_datasets",
                return_value=[
                    {"media_type": "audio", "embedder": "clap"},
                    {"media_type": "audio", "embedder": "clap"},
                    {"media_type": "image", "embedder": "siglip"},
                ],
            ),
            patch("vtsearch.detectors.registry.list_detectors", return_value=[]),
        ):
            assert predict_embedders_to_preload() == ["clap", "siglip"]

    def test_detectors_contribute_default_embedder(self):
        fake_default = MagicMock()
        fake_default.name = "e5"
        with (
            patch("vtsearch.datasets.registry.list_datasets", return_value=[]),
            patch("vtsearch.detectors.registry.list_detectors", return_value=[{"media_type": "text"}]),
            patch("vtsearch.media.embedders_for_type", return_value=[fake_default]),
        ):
            assert predict_embedders_to_preload() == ["e5"]

    def test_unregistered_embedder_name_filtered_out(self):
        # entry["embedder"] = "ghost" is not in the embedder registry → dropped.
        with (
            patch(
                "vtsearch.datasets.registry.list_datasets", return_value=[{"media_type": "audio", "embedder": "ghost"}]
            ),
            patch("vtsearch.detectors.registry.list_detectors", return_value=[]),
        ):
            assert predict_embedders_to_preload() == []


class TestInterceptWeightLoadingProgress:
    """Unit tests for intercept_weight_loading_progress context manager."""

    def test_tracks_load_state_dict_progress(self):
        """load_state_dict should report per-tensor progress via callback."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        model = nn.Linear(4, 2)
        state_dict = model.state_dict()
        num_params = len(state_dict)

        # Create a fresh model and load the state dict inside the interceptor
        model2 = nn.Linear(4, 2)
        with intercept_weight_loading_progress(cb, "Loading test model…"):
            model2.load_state_dict(state_dict)

        # Should have received progress calls (at least one per unique key access)
        assert len(calls) > 0
        # All calls should use our label
        assert all(c[1] == "Loading test model…" for c in calls)
        # Final call should show current == total
        assert calls[-1][2] == calls[-1][3]
        # Total should match state dict size
        assert calls[-1][3] == num_params

    def test_tracks_safetensors_load_file_for_total(self):
        """load_file interception should set the total tensor count."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        # Simulate safetensors.torch.load_file by temporarily patching it
        import safetensors.torch as st

        original_load_file = st.load_file
        fake_result = {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)}

        def mock_load_file(*a, **kw):
            return dict(fake_result)

        st.load_file = mock_load_file
        try:
            with intercept_weight_loading_progress(cb, "Test weights…"):
                # Call load_file (sets total) then load_state_dict (tracks progress)
                result = st.load_file("dummy.safetensors")
                model = nn.Linear(3, 2)
                model.load_state_dict(result)
        finally:
            st.load_file = original_load_file

        # Should have progress calls with total = 2 (from load_file)
        assert len(calls) > 0
        assert calls[-1][3] == 2

    def test_restores_patches_on_exit(self):
        """All monkey-patches should be removed when the context manager exits."""
        import torch.nn as _nn

        orig_lsd = _nn.Module.load_state_dict

        with intercept_weight_loading_progress(lambda *a: None, "test"):
            pass

        assert _nn.Module.load_state_dict is orig_lsd

    def test_restores_patches_on_exception(self):
        """Patches should be restored even if an exception occurs inside the block."""
        import torch.nn as _nn

        orig_lsd = _nn.Module.load_state_dict

        try:
            with intercept_weight_loading_progress(lambda *a: None, "test"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert _nn.Module.load_state_dict is orig_lsd

    def test_noop_when_no_loading_happens(self):
        """No callbacks should fire if no model loading occurs inside the block."""
        calls = []

        with intercept_weight_loading_progress(lambda *a: calls.append(a), "test"):
            x = torch.zeros(3, 3)  # noqa: F841 — no model loading, just tensor creation

        assert len(calls) == 0

    def test_current_never_exceeds_total(self):
        """Progress current should be capped at total."""
        calls = []

        def cb(status, message, current, total):
            calls.append((current, total))

        model = nn.Linear(4, 2)
        state_dict = model.state_dict()

        model2 = nn.Linear(4, 2)
        with intercept_weight_loading_progress(cb, "test"):
            model2.load_state_dict(state_dict)

        for current, total in calls:
            assert current <= total

    def test_progress_with_nested_module(self):
        """Progress should track all tensors in a multi-layer model."""
        calls = []

        def cb(status, message, current, total):
            calls.append((current, total))

        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        state_dict = model.state_dict()
        num_params = len(state_dict)  # 4: two weight + two bias tensors

        model2 = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        with intercept_weight_loading_progress(cb, "test"):
            model2.load_state_dict(state_dict)

        assert len(calls) > 0
        # Final call should show all params loaded
        assert calls[-1] == (num_params, num_params)

    def test_set_module_tensor_to_device_tracking(self):
        """set_module_tensor_to_device interception should count dispatched tensors."""
        # ``set_module_tensor_to_device`` is re-exported from ``accelerate`` at
        # runtime but isn't in the transformers stubs.
        try:
            import transformers.modeling_utils as tm

            orig = tm.set_module_tensor_to_device  # pyright: ignore[reportAttributeAccessIssue]
        except (ImportError, AttributeError):
            # accelerate not installed — patch it onto transformers.modeling_utils
            # so the interceptor can find it
            import transformers.modeling_utils as tm

            call_log: list[tuple] = []

            def fake_set_module(model, name, device, value=None, **kw):
                call_log.append((name, device))

            tm.set_module_tensor_to_device = fake_set_module  # pyright: ignore[reportAttributeAccessIssue]
            orig = None

        calls: list[tuple] = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        # Simulate: load_file sets total, then set_module_tensor_to_device is called per tensor
        import safetensors.torch as st

        original_lf = st.load_file
        fake_tensors = {f"layer.{i}.weight": torch.zeros(2) for i in range(5)}

        def mock_lf(*a, **kw):
            return dict(fake_tensors)

        st.load_file = mock_lf
        try:
            with intercept_weight_loading_progress(cb, "Loading weights…"):
                # Trigger load_file to set total
                st.load_file("dummy")
                # Simulate 5 tensor dispatches
                model = nn.Linear(2, 2)
                for i in range(5):
                    tm.set_module_tensor_to_device(model, "weight", "cpu", torch.zeros(2, 2))  # pyright: ignore[reportAttributeAccessIssue]
        finally:
            st.load_file = original_lf
            if orig is not None:
                tm.set_module_tensor_to_device = orig  # pyright: ignore[reportAttributeAccessIssue]
            else:
                delattr(tm, "set_module_tensor_to_device")

        # Should have 5 progress reports
        weight_calls = [c for c in calls if c[1] == "Loading weights…"]
        assert len(weight_calls) == 5
        assert weight_calls[-1][2] == 5
        assert weight_calls[-1][3] == 5


class TestLoadPretrainedLocalFirst:
    """Unit tests for load_pretrained_local_first."""

    def test_returns_result_from_local_only_when_available(self):
        """When local_files_only=True succeeds, the result is returned directly."""
        sentinel = object()

        def fake_load(*args, **kwargs):
            assert kwargs.get("local_files_only") is True
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id", cache_dir="/tmp")
        assert result is sentinel

    def test_falls_back_to_network_on_oserror(self):
        """When local_files_only=True raises OSError, retry without it."""
        call_count = [0]
        sentinel = object()

        def fake_load(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("local_files_only"):
                raise OSError("model not cached")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id", cache_dir="/tmp")
        assert result is sentinel
        assert call_count[0] == 2

    def test_falls_back_on_file_not_found_error(self):
        """FileNotFoundError (subclass of OSError) should also trigger fallback."""
        call_count = [0]
        sentinel = object()

        def fake_load(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("No cached files")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id")
        assert result is sentinel
        assert call_count[0] == 2

    def test_passes_through_all_args_and_kwargs(self):
        """Positional and keyword arguments should be forwarded to load_fn."""
        captured = {}

        def fake_load(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = {k: v for k, v in kwargs.items() if k != "local_files_only"}
            return "ok"

        load_pretrained_local_first(fake_load, "model-id", low_cpu_mem_usage=True, token=False)
        assert captured["args"] == ("model-id",)
        assert captured["kwargs"] == {"low_cpu_mem_usage": True, "token": False}

    def test_falls_back_on_type_error(self):
        """TypeError (e.g. sentencepiece gets None vocab_file) should trigger fallback."""
        call_count = [0]
        sentinel = object()

        def fake_load(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("local_files_only"):
                raise TypeError("not a string")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id")
        assert result is sentinel
        assert call_count[0] == 2

    def test_falls_back_on_value_error(self):
        """ValueError from partial cache should trigger fallback."""
        call_count = [0]
        sentinel = object()

        def fake_load(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("local_files_only"):
                raise ValueError("invalid vocab path")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id")
        assert result is sentinel
        assert call_count[0] == 2

    def test_non_oserror_exceptions_propagate(self):
        """Non-OSError exceptions (e.g. ImportError) should not be caught."""

        def fake_load(*args, **kwargs):
            raise ImportError("missing transformers")

        try:
            load_pretrained_local_first(fake_load, "model-id")
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

    def test_network_fallback_error_propagates(self):
        """If both local and network attempts fail, the network error propagates."""

        def fake_load(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            raise ConnectionError("network down")

        try:
            load_pretrained_local_first(fake_load, "model-id")
            assert False, "Should have raised ConnectionError"
        except ConnectionError:
            pass

    @patch("vtsearch.media.embedder.time.sleep")
    def test_retries_on_transient_hf_hub_error(self, mock_sleep):
        """Transient HfHubHTTPError (5xx) should be retried with backoff."""
        network_calls = [0]
        sentinel = object()

        # Simulate HfHubHTTPError by creating a class with the right name.
        class HfHubHTTPError(Exception):
            pass

        def fake_load(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            network_calls[0] += 1
            if network_calls[0] <= 2:
                raise HfHubHTTPError("Server error '504 Gateway Time-out' for url '...'")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id")
        assert result is sentinel
        # 2 failed network + 1 successful network = 3
        assert network_calls[0] == 3
        # Should have slept twice (2s, 4s backoff).
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 2
        assert mock_sleep.call_args_list[1][0][0] == 4

    @patch("vtsearch.media.embedder.time.sleep")
    def test_retries_exhausted_raises_last_error(self, mock_sleep):
        """When all retries are exhausted, the last transient error is raised."""
        errors = []

        class HfHubHTTPError(Exception):
            pass

        def fake_load(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            exc = HfHubHTTPError(f"502 Bad Gateway (attempt {len(errors) + 1})")
            errors.append(exc)
            raise exc

        try:
            load_pretrained_local_first(fake_load, "model-id")
            assert False, "Should have raised HfHubHTTPError"
        except Exception as exc:
            # Should be the last error raised.
            assert exc is errors[-1]
            assert "attempt 3" in str(exc)

    @patch("vtsearch.media.embedder.time.sleep")
    def test_non_transient_error_not_retried(self, mock_sleep):
        """Non-transient errors (e.g. 404) should not be retried."""

        class HfHubHTTPError(Exception):
            pass

        def fake_load(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            raise HfHubHTTPError("404 Not Found for url '...'")

        try:
            load_pretrained_local_first(fake_load, "model-id")
            assert False, "Should have raised HfHubHTTPError"
        except Exception:
            pass
        mock_sleep.assert_not_called()

    @patch("vtsearch.media.embedder.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        """ConnectionError should be retried as transient."""
        network_calls = [0]
        sentinel = object()

        def fake_load(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            network_calls[0] += 1
            if network_calls[0] <= 2:
                raise ConnectionError("connection reset by peer")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id")
        assert result is sentinel
        assert mock_sleep.call_count == 2

    @patch("vtsearch.media.embedder.time.sleep")
    def test_retries_on_timeout_error(self, mock_sleep):
        """TimeoutError should be retried as transient."""
        network_calls = [0]
        sentinel = object()

        def fake_load(*args, **kwargs):
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            network_calls[0] += 1
            if network_calls[0] <= 2:
                raise TimeoutError("request timed out")
            return sentinel

        result = load_pretrained_local_first(fake_load, "model-id")
        assert result is sentinel
        assert mock_sleep.call_count == 2


class TestTimedProgress:
    """Unit tests for the timed_progress context manager."""

    def test_sends_initial_progress_immediately(self):
        """The initial progress message should be sent before the block executes."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with timed_progress(cb, "loading", "Importing torch…", 1, 2):
            pass  # fast — no tick fires

        assert len(calls) >= 1
        assert calls[0] == ("loading", "Importing torch…", 1, 2)

    def test_no_elapsed_suffix_for_fast_operations(self):
        """If the block completes in under 1 second, no elapsed suffix should appear."""
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with timed_progress(cb, "loading", "Importing torch…", 1, 2):
            pass  # completes instantly

        # Only the initial call (no time suffix)
        assert all("(" not in c[1] for c in calls)

    def test_elapsed_time_updates_during_slow_operation(self):
        """A slow block should produce elapsed-time suffixed messages."""
        import time

        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        with timed_progress(cb, "loading", "Importing torch…", 1, 2):
            # Sleep long enough for at least one tick (~1s interval)
            time.sleep(2.5)

        # Should have the initial message plus at least one timed update
        timed_calls = [c for c in calls if "(" in c[1]]
        assert len(timed_calls) >= 1
        # Check format: "Importing torch… (1s)" or "Importing torch… (2s)"
        assert any("(1s)" in c[1] or "(2s)" in c[1] for c in timed_calls)
        # All calls should preserve status, current, total
        for c in calls:
            assert c[0] == "loading"
            assert c[2] == 1
            assert c[3] == 2

    def test_ticker_stops_after_block_exits(self):
        """The background ticker thread should stop once the with block exits."""
        import time

        calls = []

        def cb(status, message, current, total):
            calls.append(time.monotonic())

        with timed_progress(cb, "loading", "test", 0, 1):
            time.sleep(1.5)

        count_during = len(calls)
        time.sleep(1.5)
        count_after = len(calls)

        # No new calls should arrive after the block exits
        assert count_after == count_during

    def test_exception_in_block_still_stops_ticker(self):
        """The ticker should be cleaned up even if the block raises."""
        import time

        calls = []

        def cb(status, message, current, total):
            calls.append(message)

        try:
            with timed_progress(cb, "loading", "test", 0, 0):
                time.sleep(1.5)
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        count_during = len(calls)
        time.sleep(1.5)
        count_after = len(calls)

        assert count_after == count_during

    def test_works_with_console_progress_wrapper(self, capsys):
        """timed_progress should integrate with _make_console_progress."""
        import time

        cb = _make_console_progress(lambda *a, **kw: None)

        with timed_progress(cb, "loading", "Importing torch…", 1, 2):
            time.sleep(1.5)

        captured = capsys.readouterr()
        # Should see the initial message and at least one elapsed update
        assert "Importing torch…" in captured.out

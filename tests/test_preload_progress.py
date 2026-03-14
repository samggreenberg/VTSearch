"""Tests for console progress output during embedding model preloading.

Verifies that ``preload_autoload_media_types`` prints intermediate status
messages and download progress bars to stdout so the user can see what is
happening during the (potentially long) startup phase.

Also tests ``intercept_weight_loading_progress`` which tracks tensor-level
progress during model weight loading via ``set_module_tensor_to_device``
and ``load_state_dict``.
"""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from vtsearch.media.base import intercept_weight_loading_progress
from vtsearch.models.loader import _make_console_progress, preload_autoload_media_types


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
    """Integration-style tests for console output during preload_autoload_media_types."""

    @patch("vtsearch.settings.get_autoload_media_embedders", return_value=[])
    @patch("vtsearch.settings.get_autoload_media_types", return_value=["audio"])
    @patch("vtsearch.media.embedders_for_type")
    def test_prints_preloading_banner_and_progress(self, mock_embedders_for_type, mock_favs, mock_emb_favs, capsys):
        """preload_autoload_media_types should print the banner and forward progress to console."""
        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._on_progress = lambda *a, **kw: None

        def fake_load_models():
            # Simulate what a real load_models does
            mock_emb._on_progress("loading", "Loading CLAP model weights...", 0, 0)
            mock_emb._on_progress("loading", "model.safetensors", 50, 100)
            mock_emb._on_progress("loading", "model.safetensors", 100, 100)
            mock_emb._on_progress("loading", "Warming up audio pipeline: importing libraries...", 1, 4)
            mock_emb._on_progress("loading", "Warming up audio pipeline: resampling JIT...", 2, 4)
            mock_emb._on_progress("loading", "Warming up audio pipeline: preprocessing...", 3, 4)
            mock_emb._on_progress("loading", "Warming up audio pipeline: running model...", 4, 4)

        mock_emb.load_models = fake_load_models
        mock_embedders_for_type.return_value = [mock_emb]

        result = preload_autoload_media_types()

        captured = capsys.readouterr()
        assert result == ["clap"]
        assert "Preloading clap embedder" in captured.out
        assert "Loading CLAP model weights..." in captured.out
        assert "model.safetensors" in captured.out
        assert "Warming up audio pipeline: importing libraries..." in captured.out

    @patch("vtsearch.settings.get_autoload_media_embedders", return_value=[])
    @patch("vtsearch.settings.get_autoload_media_types", return_value=["audio"])
    @patch("vtsearch.media.embedders_for_type")
    def test_restores_original_callback_after_load(self, mock_embedders_for_type, mock_favs, mock_emb_favs):
        """The original _on_progress callback should be restored after load_models."""
        original_cb = MagicMock()
        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb._on_progress = original_cb
        mock_emb.load_models = MagicMock()
        mock_embedders_for_type.return_value = [mock_emb]

        preload_autoload_media_types()

        assert mock_emb._on_progress is original_cb

    @patch("vtsearch.settings.get_autoload_media_embedders", return_value=[])
    @patch("vtsearch.settings.get_autoload_media_types", return_value=["audio"])
    @patch("vtsearch.media.embedders_for_type")
    def test_restores_callback_on_exception(self, mock_embedders_for_type, mock_favs, mock_emb_favs, capsys):
        """The original callback should be restored even when load_models raises."""
        original_cb = MagicMock()
        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb._on_progress = original_cb
        mock_emb.load_models.side_effect = RuntimeError("boom")
        mock_embedders_for_type.return_value = [mock_emb]

        result = preload_autoload_media_types()

        assert mock_emb._on_progress is original_cb
        assert result == []  # failed, so not in preloaded list
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    @patch("vtsearch.settings.get_autoload_media_types", return_value=[])
    def test_no_autoload_types_produces_no_output(self, mock_favs, capsys):
        """When there are no autoload media types, nothing should be printed."""
        result = preload_autoload_media_types()

        assert result == []
        captured = capsys.readouterr()
        assert captured.out == ""


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
        calls = []

        def cb(status, message, current, total):
            calls.append((status, message, current, total))

        try:
            import transformers.modeling_utils as tm
        except ImportError:
            return  # skip if transformers not installed

        orig = tm.set_module_tensor_to_device

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
                    tm.set_module_tensor_to_device(model, "weight", "cpu", torch.zeros(2, 2))
        finally:
            st.load_file = original_lf
            tm.set_module_tensor_to_device = orig

        # Should have 5 progress reports
        weight_calls = [c for c in calls if c[1] == "Loading weights…"]
        assert len(weight_calls) == 5
        assert weight_calls[-1][2] == 5
        assert weight_calls[-1][3] == 5

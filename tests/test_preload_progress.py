"""Tests for console progress output during embedding model preloading.

Verifies that ``preload_autoload_media_types`` prints intermediate status
messages and download progress bars to stdout so the user can see what is
happening during the (potentially long) startup phase.
"""

from unittest.mock import MagicMock, patch

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

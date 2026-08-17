"""Tests for the CLI progress emitter (``--progress-format``).

Covers two layers:

1. The :mod:`vtscore.cli_progress` module itself; format toggle, text /
   JSON output shape, error routing.
2. End-to-end behaviour via :func:`vtscore.cli.autodetect_main`; the
   pipeline emits the documented events when invoked with the JSON format
   selected, and is byte-identical to the pre-flag behaviour in text mode.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

import app as app_module
from tests.helpers import make_dataset_file as _make_dataset_file
from vtscore import cli_progress

from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.settings import get_detectors_dir


# ---------------------------------------------------------------------------
# Unit tests: emitter module
# ---------------------------------------------------------------------------


class TestSetFormat:
    def test_default_is_text(self):
        assert cli_progress.get_format() == "text"

    def test_text_and_json_accepted(self):
        cli_progress.set_format("json")
        assert cli_progress.get_format() == "json"
        cli_progress.set_format("text")
        assert cli_progress.get_format() == "text"

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown progress format"):
            cli_progress.set_format("yaml")


class TestEmitText:
    def test_writes_text_to_stdout(self, capsys):
        cli_progress.emit("foo", text="hello world")
        captured = capsys.readouterr()
        assert captured.out == "hello world\n"
        assert captured.err == ""

    def test_no_text_no_output(self, capsys):
        # An event without a text= analogue is silent in text mode; used
        # for JSON-only events that have no prose representation.
        cli_progress.emit("foo", chunk_num=1)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_custom_stream_respected(self):
        buf = io.StringIO()
        cli_progress.emit("foo", text="line", stream=buf)
        assert buf.getvalue() == "line\n"


class TestEmitJson:
    def test_emits_ndjson_line(self, capsys):
        cli_progress.set_format("json")
        cli_progress.emit("chunk_start", chunk_num=2, chunk_size=500)
        captured = capsys.readouterr()
        assert captured.out.endswith("\n")
        # exactly one line
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "chunk_start"
        assert event["chunk_num"] == 2
        assert event["chunk_size"] == 500
        # Z-terminated ISO 8601 timestamp
        assert event["ts"].endswith("Z")
        assert "T" in event["ts"]

    def test_text_arg_ignored_in_json_mode(self, capsys):
        cli_progress.set_format("json")
        cli_progress.emit("export_complete", text="Done.", message="Done.")
        captured = capsys.readouterr()
        # Only the JSON line; no duplicate prose
        lines = [ln for ln in captured.out.splitlines() if ln.strip()]
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "export_complete"
        assert event["message"] == "Done."

    def test_multiple_events_are_ndjson(self, capsys):
        cli_progress.set_format("json")
        cli_progress.emit("a", x=1)
        cli_progress.emit("b", x=2)
        captured = capsys.readouterr()
        lines = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip()]
        assert [ln["event"] for ln in lines] == ["a", "b"]
        assert [ln["x"] for ln in lines] == [1, 2]


class TestEmitError:
    def test_text_mode_writes_to_stderr_with_prefix(self, capsys):
        cli_progress.emit_error("dataset not found")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "Error: dataset not found\n"

    def test_json_mode_writes_event_to_stdout(self, capsys):
        cli_progress.set_format("json")
        cli_progress.emit_error("boom")
        captured = capsys.readouterr()
        assert captured.err == ""
        event = json.loads(captured.out.strip())
        assert event["event"] == "error"
        assert event["message"] == "boom"


class TestProgressCallback:
    def test_text_mode_is_noop(self, capsys):
        # In text mode tqdm paints its own bars on stderr; our callback
        # must stay silent to avoid double-reporting.
        cli_progress.progress_callback("loading", "model.safetensors", 50, 100)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_json_mode_emits_progress_event(self, capsys):
        cli_progress.set_format("json")
        cli_progress.progress_callback("loading", "model.safetensors", 50, 100)
        captured = capsys.readouterr()
        event = json.loads(captured.out.strip())
        assert event["event"] == "progress"
        assert event["status"] == "loading"
        assert event["message"] == "model.safetensors"
        assert event["current"] == 50
        assert event["total"] == 100
        assert event["pct"] == 50.0

    def test_json_mode_drops_empty_ticks(self, capsys):
        cli_progress.set_format("json")
        # Idle ticks with no message and no total; pure noise; drop them.
        cli_progress.progress_callback("idle", "", 0, 0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_json_mode_omits_pct_when_total_zero(self, capsys):
        cli_progress.set_format("json")
        cli_progress.progress_callback("loading", "starting", 0, 0)
        captured = capsys.readouterr()
        event = json.loads(captured.out.strip())
        assert "pct" not in event
        assert "current" not in event
        assert "total" not in event


class TestNotificationSubscriber:
    """The headless sink for the notifications the GUI renders as toasts."""

    def _note(self, **kwargs):
        from vtscore.concurrency.notifications import Notification

        defaults = {"id": "note_1", "level": "warning", "message": "Skipped 3 files"}
        return Notification(**{**defaults, **kwargs})

    def test_text_mode_writes_one_line_to_stderr(self, capsys):
        cli_progress.notification_subscriber(self._note(detail="a, b, c", source="Server Folder"))
        captured = capsys.readouterr()
        # stdout stays clean: a caller may be piping an exporter's output.
        assert captured.out == ""
        assert captured.err == "Warning: [Server Folder] Skipped 3 files - a, b, c\n"

    def test_text_mode_omits_absent_source_and_detail(self, capsys):
        cli_progress.notification_subscriber(self._note(level="info", message="All good"))
        assert capsys.readouterr().err == "Note: All good\n"

    @pytest.mark.parametrize(
        "level,label",
        [("info", "Note"), ("success", "Done"), ("warning", "Warning"), ("error", "Error")],
    )
    def test_text_mode_labels_each_level(self, capsys, level, label):
        cli_progress.notification_subscriber(self._note(level=level, message="Hi"))
        assert capsys.readouterr().err == f"{label}: Hi\n"

    def test_json_mode_emits_notification_event_on_stdout(self, capsys):
        cli_progress.set_format("json")
        cli_progress.notification_subscriber(self._note(detail="a, b, c", source="Server Folder"))
        captured = capsys.readouterr()
        assert captured.err == ""
        event = json.loads(captured.out.strip())
        assert event["event"] == "notification"
        assert event["level"] == "warning"
        assert event["message"] == "Skipped 3 files"
        assert event["detail"] == "a, b, c"
        assert event["source"] == "Server Folder"

    def test_notify_reaches_the_cli_once_subscribed(self, capsys):
        from vtscore.concurrency.notifications import notifications, notify

        notifications.subscribe(cli_progress.notification_subscriber)
        try:
            notify("Partial results", level="warning", source="Remote API")
        finally:
            notifications.unsubscribe(cli_progress.notification_subscriber)
        assert capsys.readouterr().err == "Warning: [Remote API] Partial results\n"


# ---------------------------------------------------------------------------
# End-to-end tests; autodetect emits the documented events
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_tm_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _write_detector(name: str, labelset: dict) -> Path:
    from vtscore.detectors.store import _detector_path, _write_detector as _w

    path = _detector_path(name)
    _w(
        path,
        {
            "name": name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "labelset": labelset,
        },
    )
    return path


def _stub_resolve(monkeypatch, file_map: dict[str, Path]) -> None:
    from contextlib import contextmanager

    import vtscore.detectors.resolver as resolver_mod

    @contextmanager
    def _fake_ctx(origin, origin_name="", filename=""):
        yield file_map.get(origin_name) or file_map.get(filename)

    monkeypatch.setattr(resolver_mod, "resolve_file_context", _fake_ctx)


def _make_audio_files(tmp_path: Path, names: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for i, name in enumerate(names):
        path = tmp_path / name
        path.write_bytes(generate_wav(220 + 110 * i, 0.1))
        out[name] = path
    return out


def _settings_with_detectors(tmp_path: Path, names: list[str]) -> Path:
    settings = {
        "autofind_detectors": list(names),
        "detectors_dir": str(get_detectors_dir()),
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings))
    return settings_path


def _audio_labelset() -> dict:
    return {
        "labels": [
            {
                "md5": "a" * 32,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "alpha.wav",
            },
            {
                "md5": "b" * 32,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "beta.wav",
            },
            {
                "md5": "c" * 32,
                "label": "bad",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "gamma.wav",
            },
        ]
    }


class TestAutodetectJsonOutput:
    def test_emits_export_complete_event_in_json_mode(self, client, tmp_path, monkeypatch, capsys):
        """A successful autodetect run emits an ``export_complete`` NDJSON event."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)
        _write_detector("json-fmt-det", _audio_labelset())

        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_with_detectors(tmp_path, ["json-fmt-det"])
        out_path = tmp_path / "hits.json"

        cli_progress.set_format("json")

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
        )

        captured = capsys.readouterr()
        # Every stdout line must parse as JSON
        events = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip()]
        assert events, "Expected at least one NDJSON event on stdout"
        export_events = [e for e in events if e["event"] == "export_complete"]
        assert len(export_events) == 1
        assert "message" in export_events[0]
        assert export_events[0]["ts"].endswith("Z")
        # The exporter actually wrote the file
        assert out_path.exists()

    def test_text_mode_output_is_unchanged(self, client, tmp_path, monkeypatch, capsys):
        """In text mode the chunk-summary and export-confirmation prose is preserved."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)
        _write_detector("text-fmt-det", _audio_labelset())

        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_with_detectors(tmp_path, ["text-fmt-det"])
        out_path = tmp_path / "hits.json"

        # text is the default but make it explicit so the test documents intent.
        cli_progress.set_format("text")

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
        )

        captured = capsys.readouterr()
        # No JSON braces on stdout; pure prose
        assert "{" not in captured.out
        # Exporter confirmation message is preserved verbatim
        assert "Saved" in captured.out

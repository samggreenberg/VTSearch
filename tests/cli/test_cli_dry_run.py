"""Tests for the ``--dry-run`` autodetect mode.

The dry-run path must print the plan derived from the command-line
arguments + the settings file and return immediately; without consuming
the media-source iterator, training any detectors, or invoking the
exporter. It must still validate importer/exporter names, the dataset
pickle's existence, and any required CLI fields so misconfiguration
fails fast.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import app as app_module
from helpers import make_dataset_file as _make_dataset_file
from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def _clean_tm_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _write_trainable_model(name: str, labelset: dict, media_type: str = "audio") -> Path:
    from vtscore.detectors.store import _detector_path, _write_detector

    path = _detector_path(name)
    _write_detector(
        path,
        {
            "name": name,
            "text_query": "",
            "media_type": media_type,
            "examples": [],
            "labelset": labelset,
        },
    )
    return path


def _settings_file_with_detectors(tmp_path: Path, tm_names: list[str]) -> Path:
    settings = {
        "autofind_detectors": list(tm_names),
        "detectors_dir": str(get_detectors_dir()),
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings))
    return settings_path


def _make_labelset_with_two_audio_labels() -> dict:
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
                "label": "bad",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": "beta.wav",
            },
        ]
    }


class TestDryRunPickle:
    def test_dry_run_pickle_prints_plan_and_writes_nothing(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])
        out_path = tmp_path / "results.json"

        from vtscore.cli import autodetect_main

        autodetect_main(
            str(dataset_path),
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(out_path)},
            dry_run=True,
        )

        captured = capsys.readouterr()
        out = captured.out
        assert "DRY RUN" in out
        assert "Dataset pickle:" in out
        assert str(dataset_path) in out
        assert "Auto-Find detectors (1)" in out
        assert "dry-tm" in out
        assert "media_type=audio" in out
        assert "labels=2" in out
        assert "Exporter: server_json_file" in out
        assert str(out_path) in out
        # Critical: no exporter ran.
        assert not out_path.exists()

    def test_dry_run_pickle_reports_chunk_size(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            str(dataset_path),
            chunk_size=250,
            settings_path=str(settings_path),
            dry_run=True,
        )

        out = capsys.readouterr().out
        assert "Chunk size: 250" in out

    def test_dry_run_reports_streaming(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            str(dataset_path),
            chunk_size=250,
            settings_path=str(settings_path),
            exporter_name="server_json_file",
            exporter_field_values={"filepath": str(tmp_path / "out.ndjson")},
            dry_run=True,
            stream_results=True,
        )

        out = capsys.readouterr().out
        assert "Streaming: yes" in out
        assert "negatives dropped" in out

    def test_dry_run_streaming_without_stream_results_has_no_streaming_line(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_main_chunked

        autodetect_main_chunked(
            str(dataset_path),
            chunk_size=250,
            settings_path=str(settings_path),
            dry_run=True,
        )

        out = capsys.readouterr().out
        assert "Streaming:" not in out

    def test_dry_run_missing_dataset_file_errors(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])
        bogus = tmp_path / "does-not-exist.pkl"

        from vtscore.cli import autodetect_main

        with pytest.raises(SystemExit):
            autodetect_main(str(bogus), settings_path=str(settings_path), dry_run=True)

        err = capsys.readouterr().err
        assert "does-not-exist.pkl" in err

    def test_dry_run_missing_detector_flagged(self, client, tmp_path, capsys):
        # No detector JSON written for "ghost-tm".
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["ghost-tm"])

        from vtscore.cli import autodetect_main

        autodetect_main(str(dataset_path), settings_path=str(settings_path), dry_run=True)

        out = capsys.readouterr().out
        assert "ghost-tm" in out
        assert "MISSING" in out

    def test_dry_run_empty_autofind_list_calls_it_out(self, client, tmp_path, capsys):
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, [])

        from vtscore.cli import autodetect_main

        autodetect_main(str(dataset_path), settings_path=str(settings_path), dry_run=True)

        out = capsys.readouterr().out
        assert "Auto-Find detectors: (none" in out


class TestDryRunImporter:
    def test_dry_run_importer_prints_params(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_importer_main

        autodetect_importer_main(
            "server_folder",
            {"path": "/data/sounds", "media_type": "audio"},
            settings_path=str(settings_path),
            dry_run=True,
        )

        out = capsys.readouterr().out
        assert "Importer: server_folder" in out
        assert "path: /data/sounds" in out
        assert "media_type: audio" in out

    def test_dry_run_importer_validates_required_fields(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_importer_main

        # server_folder requires "path"; empty value should error during dry-run.
        with pytest.raises(SystemExit):
            autodetect_importer_main(
                "server_folder",
                {"path": "", "media_type": "audio"},
                settings_path=str(settings_path),
                dry_run=True,
            )

        err = capsys.readouterr().err
        assert "--path" in err or "Missing required" in err

    def test_dry_run_unknown_importer_errors(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_importer_main

        with pytest.raises(SystemExit):
            autodetect_importer_main(
                "no_such_importer",
                {},
                settings_path=str(settings_path),
                dry_run=True,
            )

        err = capsys.readouterr().err
        assert "no_such_importer" in err


class TestDryRunExporterValidation:
    def test_dry_run_unknown_exporter_errors(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_main

        with pytest.raises(SystemExit):
            autodetect_main(
                str(dataset_path),
                settings_path=str(settings_path),
                exporter_name="no_such_exporter",
                exporter_field_values={},
                dry_run=True,
            )

        err = capsys.readouterr().err
        assert "no_such_exporter" in err

    def test_dry_run_missing_required_exporter_field_errors(self, client, tmp_path, capsys):
        _write_trainable_model("dry-tm", _make_labelset_with_two_audio_labels())
        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file_with_detectors(tmp_path, ["dry-tm"])

        from vtscore.cli import autodetect_main

        # server_json_file requires "filepath".
        with pytest.raises(SystemExit):
            autodetect_main(
                str(dataset_path),
                settings_path=str(settings_path),
                exporter_name="server_json_file",
                exporter_field_values={},
                dry_run=True,
            )

        err = capsys.readouterr().err
        assert "--filepath" in err or "Missing required" in err

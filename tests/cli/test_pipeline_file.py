"""Tests for the ``--pipeline pipeline.yaml`` CLI path.

Exercises the YAML loader/validator and the dispatch into the shared
``_run_pipeline`` flow. The schema is documented in ``docs/CLI.md``.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

import app as app_module
from helpers import make_dataset_file as _make_dataset_file
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


def _write_detector(name: str, labelset: dict) -> Path:
    from vtscore.detectors.store import _detector_path, _write_detector

    path = _detector_path(name)
    _write_detector(
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


def _settings_file(tmp_path: Path, autorun: list[str]) -> Path:
    settings = {
        "autorun_detectors": list(autorun),
        "detectors_dir": str(get_detectors_dir()),
    }
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return p


def _trained_labelset() -> dict:
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


# ---------------------------------------------------------------------------
# load_pipeline_file - schema validation
# ---------------------------------------------------------------------------


class TestLoadPipelineFile:
    def test_missing_file_raises_filenotfound(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        with pytest.raises(FileNotFoundError):
            load_pipeline_file(tmp_path / "nope.yaml")

    def test_non_mapping_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_pipeline_file(p)

    def test_unknown_top_level_key_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text("dataset: foo.pkl\nbogus_key: 1\n")
        with pytest.raises(ValueError, match="Unknown pipeline key"):
            load_pipeline_file(p)

    def test_neither_dataset_nor_importer_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text("settings: settings.json\n")
        with pytest.raises(ValueError, match="dataset.*importer"):
            load_pipeline_file(p)

    def test_both_dataset_and_importer_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "importer": {"name": "server_folder", "fields": {"path": "/x"}},
                }
            )
        )
        with pytest.raises(ValueError, match="both"):
            load_pipeline_file(p)

    def test_unknown_importer_name_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"importer": {"name": "no_such_importer"}}))
        with pytest.raises(ValueError, match="Unknown importer"):
            load_pipeline_file(p)

    def test_unknown_exporter_name_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "exporter": {"name": "no_such_exporter"},
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown exporter"):
            load_pipeline_file(p)

    def test_chunk_size_must_be_positive_int(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "chunk_size": 0}))
        with pytest.raises(ValueError, match="positive integer"):
            load_pipeline_file(p)

    def test_detectors_must_be_list_of_strings(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "detectors": "not-a-list"}))
        with pytest.raises(ValueError, match="list of detector names"):
            load_pipeline_file(p)

    def test_import_labels_requires_detector_and_file(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "import_labels": {"detector": "d"}}))
        with pytest.raises(ValueError, match="import_labels.file"):
            load_pipeline_file(p)

    def test_unknown_importer_field_key_raises(self, tmp_path):
        """A typo in importer.fields surfaces at load time, like argparse
        rejects unknown CLI flags."""
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "importer": {
                        "name": "server_folder",
                        "fields": {"path": "/x", "media_type": "audio", "paht_typo": "/x"},
                    }
                }
            )
        )
        with pytest.raises(ValueError, match="paht_typo"):
            load_pipeline_file(p)

    def test_unknown_exporter_field_key_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "exporter": {"name": "server_json_file", "fields": {"bogus": "x"}},
                }
            )
        )
        with pytest.raises(ValueError, match="bogus"):
            load_pipeline_file(p)

    def test_minimal_valid_config_round_trips(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "data/sounds.pkl"}))
        cfg = load_pipeline_file(p)
        assert cfg["dataset"] == "data/sounds.pkl"
        assert cfg["importer"] is None
        assert cfg["detectors"] is None
        assert cfg["chunk_size"] is None
        assert cfg["import_labels"] is None
        assert cfg["exporter"] is None
        assert cfg["exporter_fields"] == {}


# ---------------------------------------------------------------------------
# Dispatch - end-to-end with a real dataset + detector
# ---------------------------------------------------------------------------


class TestRunPipelineFile:
    def test_yaml_drives_full_autodetect_run(self, client, tmp_path, monkeypatch):
        """A pipeline file with dataset + settings + exporter writes results
        identical to the equivalent --autodetect flag invocation."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)

        _write_detector("yaml-detector", _trained_labelset())

        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file(tmp_path, ["yaml-detector"])
        out_path = tmp_path / "hits.json"

        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "dataset": str(dataset_path),
                    "settings": str(settings_path),
                    "exporter": {
                        "name": "server_json_file",
                        "fields": {"filepath": str(out_path)},
                    },
                }
            )
        )

        from vtscore.cli_pipeline import run_pipeline_file

        run_pipeline_file(pipeline_path)

        body = json.loads(out_path.read_text())
        assert "yaml-detector" in body.get("results", {})

    def test_detectors_override_ignores_settings_autorun_list(self, client, tmp_path, monkeypatch):
        """`detectors:` in the YAML overrides the settings file's autorun
        list for that run only - the file on disk must NOT be touched."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)

        _write_detector("from-yaml", _trained_labelset())
        # The settings file points at a *different* detector that doesn't
        # exist on disk - if the override isn't honoured the run will fail.

        dataset_path = _make_dataset_file(tmp_path, app_module.medias)
        settings_path = _settings_file(tmp_path, ["nonexistent-detector"])
        out_path = tmp_path / "hits.json"

        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "dataset": str(dataset_path),
                    "settings": str(settings_path),
                    "detectors": ["from-yaml"],
                    "exporter": {
                        "name": "server_json_file",
                        "fields": {"filepath": str(out_path)},
                    },
                }
            )
        )

        from vtscore.cli_pipeline import run_pipeline_file

        run_pipeline_file(pipeline_path)

        body = json.loads(out_path.read_text())
        assert "from-yaml" in body.get("results", {})
        # The settings file itself must not have been rewritten.
        on_disk = json.loads(settings_path.read_text())
        assert on_disk["autorun_detectors"] == ["nonexistent-detector"]

    def test_missing_file_exits_with_nonzero_status(self, tmp_path):
        from vtscore.cli_pipeline import run_pipeline_file

        with pytest.raises(SystemExit) as exc:
            run_pipeline_file(tmp_path / "nope.yaml")
        assert exc.value.code == 1

    def test_bad_schema_exits_with_nonzero_status(self, tmp_path):
        from vtscore.cli_pipeline import run_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text("dataset: 123\n")  # dataset must be a string
        with pytest.raises(SystemExit) as exc:
            run_pipeline_file(p)
        assert exc.value.code == 1

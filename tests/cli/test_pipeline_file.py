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

from tests.helpers import make_dataset_file as _make_dataset_file
from vtscore.media.audio.audio_generator import generate_wav
from vtsearch.settings import get_detectors_dir
from vtsearch.state import medias


@pytest.fixture(autouse=True)
def _clean_detectors_dir():
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)
    yield
    d = get_detectors_dir()
    if d.is_dir():
        shutil.rmtree(d)


def _write_detector(name: str, labelset: dict, media_type: str = "audio") -> Path:
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


class _FilelessLabelImporter:
    """Minimal stand-in for a label importer that reads no file (issue #4174)."""

    name = "fileless_labels"

    def __init__(self):
        from vtscore.plugins import PluginField

        self.fields = [PluginField(key="detectors", label="Detectors", field_type="text")]


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


def _make_image_files(tmp_path: Path, names: list[str]) -> dict[str, Path]:
    import io

    from PIL import Image

    out: dict[str, Path] = {}
    for i, name in enumerate(names):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (40 * i, 200, 90)).save(buf, format="PNG")
        path = tmp_path / name
        path.write_bytes(buf.getvalue())
        out[name] = path
    return out


def _settings_file(tmp_path: Path, autofind: list[str]) -> Path:
    settings = {
        "autofind_detectors": list(autofind),
        "detectors_dir": str(get_detectors_dir()),
    }
    p = tmp_path / "settings.json"
    p.write_text(json.dumps(settings))
    return p


def _trained_labelset(ext: str = "wav") -> dict:
    return {
        "labels": [
            {
                "md5": "a" * 32,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": f"alpha.{ext}",
            },
            {
                "md5": "b" * 32,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": f"beta.{ext}",
            },
            {
                "md5": "c" * 32,
                "label": "bad",
                "origin": {"importer": "ds_a", "params": {}},
                "origin_name": f"gamma.{ext}",
            },
        ]
    }


# ---------------------------------------------------------------------------
# load_pipeline_file: schema validation
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

    def test_stream_results_requires_chunk_size(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "stream_results": True}))
        with pytest.raises(ValueError, match="requires 'chunk_size:'"):
            load_pipeline_file(p)

    def test_keep_negatives_requires_stream_results(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "chunk_size": 10, "keep_negatives": True}))
        with pytest.raises(ValueError, match="only applies with 'stream_results:'"):
            load_pipeline_file(p)

    def test_stream_results_must_be_bool(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "chunk_size": 10, "stream_results": "yes"}))
        with pytest.raises(ValueError, match="must be a boolean"):
            load_pipeline_file(p)

    def test_stream_results_and_keep_negatives_round_trip(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "chunk_size": 10,
                    "stream_results": True,
                    "keep_negatives": True,
                    "tempimport": True,
                }
            )
        )
        cfg = load_pipeline_file(p)
        assert cfg["stream_results"] is True
        assert cfg["keep_negatives"] is True
        assert cfg["tempimport"] is True

    def test_stream_results_requires_tempimport(self, tmp_path):
        """A streamed run never holds the whole dataset, so it cannot be the
        saved one (#4226): the file must say it is temporary."""
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "chunk_size": 10, "stream_results": True}))
        with pytest.raises(ValueError, match="requires 'tempimport: true'"):
            load_pipeline_file(p)

    def test_tempimport_must_be_bool(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "tempimport": "yes"}))
        with pytest.raises(ValueError, match="'tempimport:' must be a boolean"):
            load_pipeline_file(p)

    def test_tempimport_defaults_to_saving(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl"}))
        assert load_pipeline_file(p)["tempimport"] is False

    def test_import_labels_default_importer_requires_filepath(self, tmp_path):
        """The default ``server_json_file`` importer still needs a path, via
        either ``file:`` or ``importer.fields.filepath``."""
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump({"dataset": "foo.pkl", "import_labels": {"detector": "d"}}))
        with pytest.raises(ValueError, match="filepath"):
            load_pipeline_file(p)

    def test_import_labels_flat_form_maps_file_to_filepath(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "import_labels": {"detector": "d", "importer": "server_csv_file", "file": "l.csv"},
                }
            )
        )
        cfg = load_pipeline_file(p)
        assert cfg["import_labels"] == {
            "detector": "d",
            "importer": "server_csv_file",
            "fields": {"filepath": "l.csv"},
            "create": False,
            "media_type": "",
        }

    def test_import_labels_accepts_plugin_mapping(self, tmp_path):
        """``import_labels.importer`` takes the same ``{name, fields}`` shape
        as ``importer:`` / ``exporter:``."""
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "import_labels": {
                        "detector": "d",
                        "importer": {"name": "server_json_file", "fields": {"filepath": "l.json"}},
                    },
                }
            )
        )
        cfg = load_pipeline_file(p)
        assert cfg["import_labels"] == {
            "detector": "d",
            "importer": "server_json_file",
            "fields": {"filepath": "l.json"},
            "create": False,
            "media_type": "",
        }

    def test_import_labels_fileless_importer_needs_no_file(self, tmp_path, monkeypatch):
        """Issue #4174: a label importer whose fields are not a file path must
        not be forced to carry ``file:``."""
        import vtscore.labels.importers as li_mod
        from vtscore.cli_pipeline import load_pipeline_file

        plugin = _FilelessLabelImporter()
        real_get = li_mod.get_label_importer
        monkeypatch.setattr(li_mod, "get_label_importer", lambda n: plugin if n == plugin.name else real_get(n))

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "import_labels": {
                        "detector": "Dogs",
                        "importer": {"name": plugin.name, "fields": {"detectors": "somestring"}},
                    },
                }
            )
        )
        cfg = load_pipeline_file(p)
        assert cfg["import_labels"] == {
            "detector": "Dogs",
            "importer": plugin.name,
            "fields": {"detectors": "somestring"},
            "create": False,
            "media_type": "",
        }

    def test_import_labels_unknown_field_key_raises(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "import_labels": {
                        "detector": "d",
                        "importer": {"name": "server_json_file", "fields": {"filepath": "l.json", "typo": 1}},
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="typo"):
            load_pipeline_file(p)

    def test_import_labels_file_and_filepath_field_conflict(self, tmp_path):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "dataset": "foo.pkl",
                    "import_labels": {
                        "detector": "d",
                        "file": "a.json",
                        "importer": {"name": "server_json_file", "fields": {"filepath": "b.json"}},
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="pick one"):
            load_pipeline_file(p)

    @pytest.mark.parametrize("family", ["exporter", "importer", "label_importer"])
    def test_every_declared_plugin_field_is_accepted(self, tmp_path, family):
        """Every field a registered plugin declares is a legal YAML key for it,
        so no plugin field is unreachable from a pipeline file."""
        from vtscore.cli_pipeline import load_pipeline_file
        from vtscore.datasets.importers import list_importers
        from vtscore.exporters import list_exporters
        from vtscore.labels.importers import list_label_importers

        plugins = {"exporter": list_exporters, "importer": list_importers, "label_importer": list_label_importers}[
            family
        ]()
        assert plugins
        for plugin in plugins:
            fields = {f.key: f.default or "x" for f in plugin.fields}
            if family == "exporter":
                doc = {"dataset": "foo.pkl", "exporter": {"name": plugin.name, "fields": fields}}
            elif family == "importer":
                doc = {"importer": {"name": plugin.name, "fields": fields}}
            else:
                doc = {
                    "dataset": "foo.pkl",
                    "import_labels": {"detector": "d", "importer": {"name": plugin.name, "fields": fields}},
                }
            p = tmp_path / f"{family}-{plugin.name}.yaml"
            p.write_text(yaml.safe_dump(doc))
            cfg = load_pipeline_file(p)
            got = cfg["import_labels"]["fields"] if family == "label_importer" else cfg[f"{family}_fields"]
            assert got == fields, plugin.name

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
        assert cfg["importer_outputs"] == []
        assert cfg["detectors"] is None
        assert cfg["chunk_size"] is None
        assert cfg["import_labels"] is None
        assert cfg["exporter"] is None
        assert cfg["exporter_fields"] == {}


def _outputs_yaml(outputs, fields=None, importer="server_folder") -> dict:
    return {"importer": {"name": importer, "fields": fields or {"path": "/d"}, "outputs": outputs}}


class TestImporterOutputs:
    """``importer.outputs``: several datasets from the one importer run (#4707)."""

    def _load(self, tmp_path, config: dict):
        from vtscore.cli_pipeline import load_pipeline_file

        p = tmp_path / "p.yaml"
        p.write_text(yaml.safe_dump(config))
        return load_pipeline_file(p)

    def test_outputs_round_trip_as_output_specs(self, tmp_path):
        from vtscore.datasets.importers.base import OutputSpec

        cfg = self._load(
            tmp_path,
            _outputs_yaml(
                [
                    {"media_type": "audio", "embedder": "clap"},
                    {
                        "media_type": "image",
                        "category": "document",
                        "source_specs": [{"source_type": "document", "converter": "document2image"}],
                        "dataset_name": "scans",
                        "clipper_params": {"grid": 2},
                    },
                ]
            ),
        )
        assert cfg["importer"] == "server_folder"
        assert cfg["importer_fields"] == {"path": "/d"}, "the list is its own key, not a field"
        audio, image = cfg["importer_outputs"]
        assert isinstance(audio, OutputSpec)
        assert (audio.media_type, audio.category, audio.embedder) == ("audio", "audio", "clap")
        assert (image.media_type, image.category, image.dataset_name) == ("image", "document", "scans")
        assert [(s.source_type, s.converter) for s in image.source_specs] == [("document", "document2image")]
        assert image.clipper_params == {"grid": 2}

    def test_without_outputs_the_config_lists_none(self, tmp_path):
        cfg = self._load(tmp_path, {"importer": {"name": "server_folder", "fields": {"path": "/d"}}})
        assert cfg["importer_outputs"] == []

    @pytest.mark.parametrize("outputs", [[], "audio", {"media_type": "audio"}])
    def test_outputs_must_be_a_non_empty_list(self, tmp_path, outputs):
        with pytest.raises(ValueError, match="'importer.outputs' must be a non-empty list"):
            self._load(tmp_path, _outputs_yaml(outputs))

    def test_every_entry_must_be_a_mapping(self, tmp_path):
        with pytest.raises(ValueError, match=r"'importer.outputs\[1\]' must be a mapping, got str"):
            self._load(tmp_path, _outputs_yaml([{"media_type": "audio"}, "image"]))

    def test_unknown_entry_key_names_the_allowed_ones(self, tmp_path):
        with pytest.raises(
            ValueError, match=r"'importer.outputs\[0\]' has unknown key\(s\): recursive\. Allowed: .*media_type"
        ):
            self._load(tmp_path, _outputs_yaml([{"media_type": "audio", "recursive": True}]))

    def test_every_entry_needs_a_media_type(self, tmp_path):
        with pytest.raises(ValueError, match=r"'importer.outputs\[1\]' needs a 'media_type'"):
            self._load(tmp_path, _outputs_yaml([{"media_type": "audio"}, {"embedder": "siglip"}]))

    def test_conflicts_with_a_top_level_media_type(self, tmp_path):
        with pytest.raises(ValueError, match="'importer.outputs' and 'importer.fields.media_type' both set"):
            self._load(tmp_path, _outputs_yaml([{"media_type": "audio"}], fields={"path": "/d", "media_type": "audio"}))

    def test_refused_for_an_importer_that_makes_one_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="importer 'demo' produces one dataset per run"):
            self._load(tmp_path, _outputs_yaml([{"media_type": "audio"}], fields={"name": "x"}, importer="demo"))

    def test_the_web_validator_s_message_is_prefixed(self, tmp_path):
        with pytest.raises(ValueError, match="'importer.outputs': Unknown media type: 'hologram'"):
            self._load(tmp_path, _outputs_yaml([{"media_type": "hologram"}]))

    def test_a_converter_must_produce_the_output_s_type(self, tmp_path):
        with pytest.raises(ValueError, match="'importer.outputs':"):
            self._load(
                tmp_path,
                _outputs_yaml(
                    [
                        {
                            "media_type": "face",
                            "source_specs": [{"source_type": "document", "converter": "document2image"}],
                        }
                    ]
                ),
            )

    def test_outputs_is_not_an_exporter_key(self, tmp_path):
        with pytest.raises(ValueError, match=r"'exporter' has unknown key\(s\): outputs. Allowed: name, fields."):
            self._load(tmp_path, {"dataset": "x.pkl", "exporter": {"name": "gui", "outputs": []}})

    def test_the_importer_error_names_outputs_among_the_allowed_keys(self, tmp_path):
        with pytest.raises(ValueError, match="Allowed: name, fields, outputs."):
            self._load(tmp_path, {"importer": {"name": "server_folder", "fields": {"path": "/d"}, "chunks": 1}})


# ---------------------------------------------------------------------------
# Dispatch: end-to-end with a real dataset + detector
# ---------------------------------------------------------------------------


class TestRunPipelineFile:
    def test_yaml_drives_full_autodetect_run(self, client, tmp_path, monkeypatch):
        """A pipeline file with dataset + settings + exporter writes results
        identical to the equivalent --autodetect flag invocation."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)

        _write_detector("yaml-detector", _trained_labelset())

        dataset_path = _make_dataset_file(tmp_path, medias)
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

    def test_detectors_override_ignores_settings_autofind_list(self, client, tmp_path, monkeypatch):
        """`detectors:` in the YAML overrides the settings file's AutoFind
        list for that run only; the file on disk must NOT be touched."""
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        _stub_resolve(monkeypatch, files)

        _write_detector("from-yaml", _trained_labelset())
        # The settings file points at a *different* detector that doesn't
        # exist on disk; if the override isn't honoured the run will fail.

        dataset_path = _make_dataset_file(tmp_path, medias)
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
        assert on_disk["autofind_detectors"] == ["nonexistent-detector"]

    @pytest.mark.parametrize("tempimport", [True, False])
    def test_two_output_pipeline_makes_two_datasets_from_one_synthetic_run(
        self, client, tmp_path, monkeypatch, tempimport
    ):
        """``importer.outputs`` (#4707): the synthetic importer runs once for an image and an
        audio dataset; each is scored by the detectors that reach its type and exported to
        its own ``{dataset_name}`` file; a saving run registers both."""
        from vtscore.datasets.registry import list_datasets
        from vtscore.media import all_embedders

        # The session stub covers ``embed_media``; the image embedder's own bulk
        # path would decline a real PNG without its model.
        for emb in all_embedders():
            monkeypatch.setattr(emb, "embed_media_bulk", lambda items, _e=emb: [_e.embed_media(m) for m in items])
        files = _make_audio_files(tmp_path, ["alpha.wav", "beta.wav", "gamma.wav"])
        files.update(_make_image_files(tmp_path, ["alpha.png", "beta.png", "gamma.png"]))
        _stub_resolve(monkeypatch, files)
        _write_detector("ears", _trained_labelset("wav"), media_type="audio")
        _write_detector("eyes", _trained_labelset("png"), media_type="image")
        settings_path = _settings_file(tmp_path, ["ears", "eyes"])

        pipeline_path = tmp_path / "pipeline.yaml"
        pipeline_path.write_text(
            yaml.safe_dump(
                {
                    "importer": {
                        "name": "synthetic",
                        "fields": {"size": 3},
                        "outputs": [
                            {"media_type": "image", "dataset_name": "fake-pics"},
                            {"media_type": "audio", "dataset_name": "fake-sounds"},
                        ],
                    },
                    "settings": str(settings_path),
                    "tempimport": tempimport,
                    "exporter": {
                        "name": "server_json_file",
                        "fields": {"filepath": str(tmp_path / "hits-{dataset_name}.json")},
                    },
                }
            )
        )

        from vtscore.cli_pipeline import run_pipeline_file

        run_pipeline_file(pipeline_path)

        pics = json.loads((tmp_path / "hits-fake-pics.json").read_text())
        sounds = json.loads((tmp_path / "hits-fake-sounds.json").read_text())
        assert pics["media_type"] == "image" and pics["dataset"]["name"] == "fake-pics"
        assert sounds["media_type"] == "audio" and sounds["dataset"]["name"] == "fake-sounds"
        assert set(pics["results"]) == {"eyes"}, "no route from images to the audio detector"
        assert "ears" in sounds["results"]

        entries = {e["name"]: e for e in list_datasets()}
        if tempimport:
            assert entries == {}
            assert pics["dataset"]["id"] is None
        else:
            assert set(entries) == {"fake-pics", "fake-sounds"}
            assert entries["fake-pics"]["media_type"] == "image"
            assert entries["fake-pics"]["num_items"] == 3
            assert entries["fake-sounds"]["media_type"] == "audio"
            assert pics["dataset"]["id"] == entries["fake-pics"]["id"]

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


class TestDispatchImportLabels:
    def test_dispatch_passes_label_importer_fields(self, monkeypatch):
        """_dispatch hands the whole field mapping, not just a file path, to
        the label import (issue #4174)."""
        import vtscore.cli as vtcli
        from vtscore.cli_pipeline import _dispatch

        seen = {}

        def _fake_import(detector, importer, field_values, create_media_type=""):
            seen.update(detector=detector, importer=importer, fields=field_values)
            return (1, 0)

        monkeypatch.setattr(vtcli, "import_labels_into_detector", _fake_import)
        monkeypatch.setattr(vtcli, "_load_pickle_whole", lambda path: [])
        monkeypatch.setattr(vtcli, "_run_pipeline", lambda *a, **k: None)
        _dispatch(
            {
                "dataset": "foo.pkl",
                "importer": None,
                "importer_fields": {},
                "settings": None,
                "detectors": None,
                "chunk_size": None,
                "import_labels": {"detector": "Dogs", "importer": "x", "fields": {"detectors": "somestring"}},
                "exporter": None,
                "exporter_fields": {},
                "tempimport": True,
            }
        )
        assert seen == {"detector": "Dogs", "importer": "x", "fields": {"detectors": "somestring"}}


class TestDispatchDetectorSelection:
    """Issue #4235: ``import_labels:`` names the run's detector unless
    ``detectors:`` is set."""

    def _dispatch_capturing(self, monkeypatch, detectors):
        import vtscore.cli as vtcli
        from vtscore.cli_pipeline import _dispatch

        seen = {}
        monkeypatch.setattr(vtcli, "import_labels_into_detector", lambda *a, **k: (1, 0))
        monkeypatch.setattr(vtcli, "_run_source", lambda spec, **kw: seen.update(kw))
        _dispatch(
            {
                "dataset": "foo.pkl",
                "importer": None,
                "importer_fields": {},
                "settings": None,
                "detectors": detectors,
                "chunk_size": None,
                "import_labels": {"detector": "Dogs", "importer": "server_json_file", "fields": {"filepath": "x"}},
                "exporter": None,
                "exporter_fields": {},
                "tempimport": True,
            }
        )
        return seen

    def test_import_labels_detector_replaces_autofind_list(self, monkeypatch):
        seen = self._dispatch_capturing(monkeypatch, None)
        assert seen["override_detectors"] == ["Dogs"]

    def test_explicit_detectors_list_wins(self, monkeypatch):
        seen = self._dispatch_capturing(monkeypatch, ["Cats", "Dogs"])
        assert seen["override_detectors"] == ["Cats", "Dogs"]


class TestImportLabelsCreate:
    """Issue #4238: ``import_labels.create`` / ``import_labels.media_type``."""

    @staticmethod
    def _parse(block):
        from vtscore.cli_pipeline import _parse_import_labels

        return _parse_import_labels({"detector": "Dogs", "file": "labels.json", **block})

    def test_defaults_to_no_create(self):
        parsed = self._parse({})
        assert parsed["create"] is False
        assert parsed["media_type"] == ""

    def test_create_with_media_type(self):
        parsed = self._parse({"create": True, "media_type": "image"})
        assert parsed["create"] is True
        assert parsed["media_type"] == "image"

    def test_create_must_be_boolean(self):
        with pytest.raises(ValueError, match="'import_labels.create' must be a boolean"):
            self._parse({"create": "yes"})

    def test_media_type_needs_create(self):
        with pytest.raises(ValueError, match="only applies with 'import_labels.create: true'"):
            self._parse({"media_type": "image"})

    @staticmethod
    def _dispatch(monkeypatch, import_labels):
        import vtscore.cli as vtcli
        from vtscore.cli_pipeline import _dispatch

        monkeypatch.setattr(vtcli, "_run_source", lambda spec, **kw: None)
        _dispatch(
            {
                "dataset": None,
                "importer": "server_folder",
                "importer_fields": {"path": "/x", "media_type": "video"},
                "settings": None,
                "detectors": None,
                "chunk_size": None,
                "import_labels": import_labels,
                "exporter": None,
                "exporter_fields": {},
                "tempimport": True,
            }
        )

    def test_create_uses_the_sources_media_type(self, monkeypatch):
        import vtscore.cli as vtcli

        seen = {}
        monkeypatch.setattr(
            vtcli,
            "import_labels_into_detector",
            lambda det, imp, fields, create_media_type="": seen.update(mt=create_media_type) or (1, 0),
        )
        self._dispatch(
            monkeypatch,
            {"detector": "Dogs", "importer": "server_json_file", "fields": {"filepath": "x"}, "create": True},
        )
        assert seen["mt"] == "video"

    def test_missing_detector_without_create_names_the_key(self, monkeypatch, tmp_path):
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps({"labels": [{"md5": "a" * 32, "label": "good"}]}))
        with pytest.raises(ValueError, match="Set 'import_labels.create: true'"):
            self._dispatch(
                monkeypatch,
                {"detector": "No Such Det", "importer": "server_json_file", "fields": {"filepath": str(labels)}},
            )

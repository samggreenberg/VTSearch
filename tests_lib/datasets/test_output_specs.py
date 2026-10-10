"""The multi-dataset import contract on the importer base (#4703).

``OutputSpec`` describes one dataset of a multi-dataset import; the base
hooks ``run_outputs`` / ``run_outputs_chunked`` default to running the
single-dataset ``run`` / ``run_chunked`` once per output; the URL-archive
importer overrides them to download once.  Library tier: no Flask, no app.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests_lib.helpers import make_wav_file
from vtscore.datasets.importers.base import (
    ImporterBase,
    OutputSpec,
    PluginField,
    SourceSpec,
    output_dataset_name,
    parse_output_specs,
)


def _has_module(name: str) -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# parse_output_specs
# ---------------------------------------------------------------------------


class TestParseOutputSpecs:
    @pytest.mark.parametrize("raw", [None, "", [], "[]"])
    def test_absent_means_single_dataset(self, raw):
        assert parse_output_specs(raw) == []

    def test_resolves_folder_name_and_type_id(self):
        outs = parse_output_specs([{"media_type": "image"}, {"media_type": "audio"}])
        assert [o.media_type for o in outs] == ["image", "audio"]

    def test_category_defaults_to_media_type(self):
        (out,) = parse_output_specs([{"media_type": "image"}])
        assert out.category == "image"
        assert out.category_label() == "Image"

    def test_document_pages_are_an_image_dataset_in_the_document_category(self):
        (out,) = parse_output_specs(
            [
                {
                    "media_type": "image",
                    "category": "document",
                    "source_specs": [{"source_type": "document", "converter": "document2image", "params": {}}],
                }
            ]
        )
        assert out.media_type == "image"
        assert out.category == "document"
        assert out.category_label() == "Document"
        assert out.source_specs == [SourceSpec("document", "document2image", {})]

    def test_empty_source_specs_fall_back_to_the_direct_row(self):
        (out,) = parse_output_specs([{"media_type": "audio", "source_specs": []}])
        assert out.source_specs == [SourceSpec("audio", None, {})]

    def test_json_string_body(self):
        raw = json.dumps([{"media_type": "audio", "embedder": "clap", "dataset_name": "Birds"}])
        (out,) = parse_output_specs(raw)
        assert (out.media_type, out.embedder, out.dataset_name) == ("audio", "clap", "Birds")

    def test_per_output_options_are_carried(self):
        (out,) = parse_output_specs(
            [
                {
                    "media_type": "audio",
                    "embedder": "clap",
                    "embedders": '["clap", "clap", "", "other"]',
                    "clipper": "audio_default",
                    "clipper_params": '{"seconds": 5}',
                    "cleaners": [{"name": "trim", "params": {}}],
                }
            ]
        )
        assert out.embedders == ["clap", "other"]
        assert out.clipper == "audio_default"
        assert out.clipper_params == {"seconds": 5}
        assert out.cleaners == [{"name": "trim", "params": {}}]

    def test_unknown_media_type_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown media type"):
            parse_output_specs([{"media_type": "hologram"}])

    def test_missing_media_type_is_rejected(self):
        with pytest.raises(ValueError, match="media_type"):
            parse_output_specs([{"source_specs": []}])

    def test_non_object_entry_is_rejected(self):
        with pytest.raises(ValueError, match="must be an object"):
            parse_output_specs(["image"])

    def test_non_list_is_rejected(self):
        with pytest.raises(ValueError, match="must be a list"):
            parse_output_specs('{"media_type": "image"}')

    def test_bad_json_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid outputs JSON"):
            parse_output_specs("[not json")

    def test_converter_must_produce_the_output_type(self):
        with pytest.raises(ValueError, match="produces"):
            parse_output_specs(
                [
                    {
                        "media_type": "face",
                        "source_specs": [{"source_type": "document", "converter": "document2image", "params": {}}],
                    }
                ]
            )

    def test_bad_clipper_params_json_is_rejected(self):
        with pytest.raises(ValueError, match="clipper_params"):
            parse_output_specs([{"media_type": "audio", "clipper_params": "{nope"}])


# ---------------------------------------------------------------------------
# OutputSpec.narrow / output_dataset_name
# ---------------------------------------------------------------------------


class TestNarrow:
    def test_narrowed_values_look_like_a_single_dataset_import(self):
        out = OutputSpec(
            media_type="image",
            source_specs=[SourceSpec("video", "video2image", {"fps": "1"})],
            embedder="siglip",
            clipper="image_default",
        )
        shared = {"url": "http://x/y.zip", "outputs": ["ignored"], "embedder": "clap", "recursive": True}
        narrowed = out.narrow(shared)

        assert narrowed["media_type"] == "image"  # the type's folder_import_name
        assert narrowed["source_specs"] == [
            {"source_type": "video", "converter": "video2image", "params": {"fps": "1"}}
        ]
        assert narrowed["embedder"] == "siglip"
        assert narrowed["clipper"] == "image_default"
        assert narrowed["url"] == "http://x/y.zip"
        assert narrowed["recursive"] is True
        assert "outputs" not in narrowed
        assert "dataset_name" not in narrowed
        # The shared dict is left alone.
        assert shared["embedder"] == "clap" and "outputs" in shared

    def test_explicit_dataset_name_is_folded_in(self):
        out = OutputSpec(media_type="audio", dataset_name="Birds")
        assert out.narrow({})["dataset_name"] == "Birds"

    def test_round_trips_through_to_dict(self):
        out = OutputSpec(media_type="face", source_specs=[SourceSpec("image", "image2face", {})], category="face")
        d = out.to_dict()
        assert d["media_type"] == "face"
        assert d["source_specs"] == [{"source_type": "image", "converter": "image2face", "params": {}}]
        assert d["embedders"] is None


class TestOutputDatasetName:
    def test_explicit_name_wins(self):
        assert output_dataset_name("holiday", OutputSpec("image", dataset_name="Snaps")) == "Snaps"

    def test_base_plus_category_label(self):
        assert output_dataset_name("holiday", OutputSpec("image")) == "holiday – Image"
        assert output_dataset_name("holiday", OutputSpec("image", category="document")) == "holiday – Document"

    def test_label_alone_without_a_base(self):
        assert output_dataset_name("", OutputSpec("face")) == "Face"


# ---------------------------------------------------------------------------
# Default run_outputs / run_outputs_chunked on the base
# ---------------------------------------------------------------------------


class _Recording(ImporterBase):
    """Records the field values each single-dataset call received."""

    name = "recording_multi"
    display_name = "Recording"
    description = "test double"
    fields = [
        PluginField("path", "Path", "text"),
        PluginField("media_type", "Media type", "select", options=["audio", "image"], default="audio", required=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.run_calls: list[dict] = []
        self.chunk_calls: list[dict] = []

    def run(self, field_values, medias, thin=False):
        self.run_calls.append(dict(field_values))
        medias[1] = {"id": 1, "media_type": field_values["media_type"], "filename": "a"}

    def run_chunked(self, field_values, chunk_size, thin=False):
        self.chunk_calls.append(dict(field_values))
        yield {1: {"id": 1, "media_type": field_values["media_type"], "filename": "a"}}
        yield {1: {"id": 1, "media_type": field_values["media_type"], "filename": "b"}}


class TestDefaultMultiOutputHooks:
    def test_run_outputs_runs_once_per_output_with_narrowed_values(self):
        imp = _Recording()
        outputs = [OutputSpec("audio", embedder="clap"), OutputSpec("image", embedder="siglip")]
        got = list(imp.run_outputs({"path": "/p", "embedder": "shared"}, outputs))

        assert [o for o, _ in got] == outputs, "the very OutputSpec objects come back"
        assert [m[1]["media_type"] for _, m in got] == ["audio", "image"]
        assert [c["media_type"] for c in imp.run_calls] == ["audio", "image"]
        assert [c["embedder"] for c in imp.run_calls] == ["clap", "siglip"]
        assert all(c["path"] == "/p" for c in imp.run_calls)

    def test_run_outputs_chunked_runs_run_chunked_once_per_output(self):
        imp = _Recording()
        outputs = [OutputSpec("audio"), OutputSpec("image")]
        got = list(imp.run_outputs_chunked({"path": "/p"}, outputs, chunk_size=1))

        assert [o.media_type for o, _ in got] == ["audio", "audio", "image", "image"]
        assert [c["media_type"] for c in imp.chunk_calls] == ["audio", "image"]

    def test_to_dict_advertises_multi_output_only_with_a_media_type_field(self):
        assert _Recording().to_dict()["supports_multi_output"] is True

        class NoType(ImporterBase):
            name = "no_type"
            display_name = "No type"
            description = "x"
            fields = [PluginField("path", "Path", "text")]

            def run(self, field_values, medias, thin=False):
                pass

        assert NoType().to_dict()["supports_multi_output"] is False

    def test_an_importer_can_opt_out(self):
        class Fixed(_Recording):
            name = "fixed"
            multi_output = False

        assert Fixed().to_dict()["supports_multi_output"] is False

    def test_shipped_importers_advertise_as_expected(self):
        from vtscore.datasets.importers import get_importer

        assert get_importer("server_folder").to_dict()["supports_multi_output"] is True
        assert get_importer("http_archive").to_dict()["supports_multi_output"] is True
        assert get_importer("demo").to_dict()["supports_multi_output"] is False, "a demo is one named download"
        assert get_importer("pickle").to_dict()["supports_multi_output"] is False, "no media_type field to multiply"


# ---------------------------------------------------------------------------
# http_archive: one download for every output
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_module("PIL"), reason="PIL needed to write the test image")
class TestHttpArchiveDownloadsOnce:
    def _prepare(self, tmp_path: Path, monkeypatch) -> tuple[list[Path], list[Path]]:
        """Patch the download+extract step to lay out one wav and one png; return (calls, dirs)."""
        from tests_lib.helpers import make_png_bytes
        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        source = tmp_path / "source"
        source.mkdir()
        make_wav_file(source, "clip.wav")
        (source / "photo.png").write_bytes(make_png_bytes())

        calls: list[Path] = []
        dirs: list[Path] = []

        def fake_download_and_extract(self, field_values):
            calls.append(Path(field_values["url"]))
            extract_dir = tmp_path / f"extract_{len(calls)}"
            shutil.copytree(source, extract_dir)
            dirs.append(extract_dir)
            return extract_dir

        monkeypatch.setattr(HttpArchiveDatasetImporter, "_download_and_extract", fake_download_and_extract)
        return calls, dirs

    def test_run_outputs_chunked_downloads_once_and_cleans_up(self, tmp_path, monkeypatch):
        from vtscore.datasets.importers.http_archive import IMPORTER

        calls, dirs = self._prepare(tmp_path, monkeypatch)
        outputs = [OutputSpec("audio"), OutputSpec("image")]
        got = list(IMPORTER.run_outputs_chunked({"url": "http://example.com/x.zip"}, outputs, chunk_size=10))

        assert len(calls) == 1, "one download for two datasets"
        by_output = {id(o): [] for o in outputs}
        for out, chunk in got:
            by_output[id(out)].extend(chunk.values())
        assert [m["filename"] for m in by_output[id(outputs[0])]] == ["clip.wav"]
        assert [m["filename"] for m in by_output[id(outputs[1])]] == ["photo.png"]
        assert all(m["media_type"] == "audio" for m in by_output[id(outputs[0])])
        assert all(m["media_type"] == "image" for m in by_output[id(outputs[1])])
        assert not dirs[0].exists(), "the shared extraction is deleted after the last output"

    def test_run_outputs_downloads_once(self, tmp_path, monkeypatch):
        from vtscore.datasets.importers.http_archive import IMPORTER

        calls, dirs = self._prepare(tmp_path, monkeypatch)
        outputs = [OutputSpec("audio"), OutputSpec("image")]
        got = list(IMPORTER.run_outputs({"url": "http://example.com/x.zip"}, outputs))

        assert len(calls) == 1
        assert [(o.media_type, len(m)) for o, m in got] == [("audio", 1), ("image", 1)]
        assert not dirs[0].exists()

    def test_single_dataset_run_still_cleans_up(self, tmp_path, monkeypatch):
        """The refactor must not change the single-dataset path's behaviour."""
        from vtscore.datasets.importers.http_archive import IMPORTER

        calls, dirs = self._prepare(tmp_path, monkeypatch)
        medias: dict = {}
        IMPORTER.run({"url": "http://example.com/x.zip", "media_type": "audio"}, medias)

        assert len(calls) == 1
        assert [m["filename"] for m in medias.values()] == ["clip.wav"]
        assert not dirs[0].exists()

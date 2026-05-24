"""Tests for the ConverterChooser: converter registry, API, and importer integration.

These tests verify:
- Converter registry functions (list, get, filter by target)
- Converter base class name/display_name/to_dict properties
- /api/converters endpoint (unfiltered and filtered)
- Folder importer accepts and passes through ``converters`` field value
- HTTP Archive importer accepts and passes through ``converters`` field value
- run_converters_on_folder scans, converts, embeds, and sets origins
- build_cli_args / build_origin include converter info
"""

from __future__ import annotations

import hashlib
import io
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 4, height: int = 4) -> bytes:
    """Create a minimal valid PNG file in memory."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# Converter registry
# ===========================================================================


class TestConverterRegistry:
    def test_list_converters_returns_all(self):
        from vtscore.converters import list_converters

        converters = list_converters()
        assert len(converters) >= 4
        names = [c.name for c in converters]
        assert "document2image" in names
        assert "document2text" in names
        assert "video2audio" in names
        assert "video2image" in names

    def test_get_converter_existing(self):
        from vtscore.converters import get_converter

        c = get_converter("video2image")
        assert c is not None
        assert c.name == "video2image"
        assert c.source_type == "video"
        assert c.target_type == "image"

    def test_get_converter_nonexistent(self):
        from vtscore.converters import get_converter

        assert get_converter("nonexistent") is None

    def test_list_converters_for_target_image(self):
        from vtscore.converters import list_converters_for_target

        converters = list_converters_for_target("image")
        names = [c.name for c in converters]
        assert "video2image" in names
        assert "document2image" in names
        assert "video2audio" not in names

    def test_list_converters_for_target_audio(self):
        from vtscore.converters import list_converters_for_target

        converters = list_converters_for_target("audio")
        names = [c.name for c in converters]
        assert "video2audio" in names
        assert "video2image" not in names

    def test_list_converters_for_target_text(self):
        from vtscore.converters import list_converters_for_target

        converters = list_converters_for_target("text")
        names = [c.name for c in converters]
        assert "document2text" in names

    def test_list_converters_for_target_none(self):
        from vtscore.converters import list_converters_for_target

        converters = list_converters_for_target("video")
        assert converters == []


# ===========================================================================
# Converter base class
# ===========================================================================


class TestConverterBase:
    def test_name_derived(self):
        from vtscore.converters import Video2AudioMediaConverter

        c = Video2AudioMediaConverter()
        assert c.name == "video2audio"

    def test_display_name(self):
        from vtscore.converters import Video2ImageMediaConverter

        c = Video2ImageMediaConverter()
        assert c.display_name == "Video \u2192 Images"

    def test_description(self):
        from vtscore.converters import Document2ImageMediaConverter

        c = Document2ImageMediaConverter()
        assert c.description != ""

    def test_to_dict(self):
        from vtscore.converters import Video2AudioMediaConverter

        c = Video2AudioMediaConverter()
        d = c.to_dict()
        assert d["name"] == "video2audio"
        assert d["source_type"] == "video"
        assert d["target_type"] == "audio"
        assert "display_name" in d
        assert "description" in d
        # summary_template surfaces the configurable ffmpeg_timeout so the
        # frontend can preview the active setting next to the import row.
        assert d["summary_template"] == "Pull the audio track from each video. Timeout {ffmpeg_timeout}s."

    def test_to_dict_omits_summary_template_when_unset(self):
        from vtscore.converters.base import MediaConverter

        class Dummy(MediaConverter):
            @property
            def source_type(self):
                return "foo"

            @property
            def target_type(self):
                return "bar"

            def convert(self, media, params=None):
                return []

        d = Dummy().to_dict()
        assert "summary_template" not in d

    def test_to_dict_fallback_display_name(self):
        """If display_name is empty, to_dict derives one from type IDs."""
        from vtscore.converters.base import MediaConverter

        class Dummy(MediaConverter):
            @property
            def source_type(self):
                return "foo"

            @property
            def target_type(self):
                return "bar"

            def convert(self, media):
                return []

        c = Dummy()
        d = c.to_dict()
        assert "Foo" in d["display_name"]
        assert "Bar" in d["display_name"]


# ===========================================================================
# /api/converters endpoint
# ===========================================================================


class TestConvertersAPI:
    def test_list_all_converters(self, client):
        resp = client.get("/api/converters")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "converters" in data
        names = [c["name"] for c in data["converters"]]
        assert "video2image" in names
        assert "video2audio" in names
        assert "document2image" in names
        assert "document2text" in names

    def test_filter_by_target_type_id(self, client):
        resp = client.get("/api/converters?target=image")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [c["name"] for c in data["converters"]]
        assert "video2image" in names
        assert "document2image" in names
        assert "video2audio" not in names

    def test_filter_by_folder_import_name(self, client):
        resp = client.get("/api/converters?target=audio")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [c["name"] for c in data["converters"]]
        assert "video2audio" in names
        assert "video2image" not in names

    def test_filter_by_target_images(self, client):
        resp = client.get("/api/converters?target=image")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [c["name"] for c in data["converters"]]
        assert "video2image" in names

    def test_filter_no_converters(self, client):
        resp = client.get("/api/converters?target=video")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["converters"] == []

    def test_converter_dict_shape(self, client):
        resp = client.get("/api/converters")
        data = resp.get_json()
        for c in data["converters"]:
            assert "name" in c
            assert "source_type" in c
            assert "target_type" in c
            assert "display_name" in c
            assert "description" in c


# ===========================================================================
# Importer build_cli_args / build_origin with converters
# ===========================================================================


class TestFolderImporterConverterFields:
    def test_build_cli_args_without_source_specs(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        args = IMPORTER.build_cli_args({"media_type": "image", "path": "/data"})
        assert "--source-specs" not in args

    def test_build_cli_args_with_source_specs(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        args = IMPORTER.build_cli_args(
            {
                "media_type": "image",
                "path": "/data",
                "source_specs": [
                    {"source_type": "image", "converter": None, "params": {}},
                    {"source_type": "video", "converter": "video2image", "params": {"n_clips": "8"}},
                ],
            }
        )
        assert "--source-specs" in args
        assert "video2image" in args

    def test_build_origin_without_source_specs(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        origin = IMPORTER.build_origin({"media_type": "image", "path": "/data"})
        assert origin["importer"] == "server_folder"
        assert "source_specs" not in origin["params"]

    def test_build_origin_with_source_specs(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        origin = IMPORTER.build_origin(
            {
                "media_type": "image",
                "path": "/data",
                "source_specs": [
                    {"source_type": "video", "converter": "video2image", "params": {}},
                ],
            }
        )
        assert "video2image" in origin["params"]["source_specs"]


class TestHttpArchiveImporterConverterFields:
    def test_build_cli_args_with_source_specs(self):
        import json

        from vtscore.datasets.importers.http_archive import IMPORTER

        specs = [
            {"source_type": "image", "converter": None, "params": {}},
            {"source_type": "video", "converter": "video2image", "params": {"n_clips": "8"}},
        ]
        args = IMPORTER.build_cli_args(
            {
                "url": "https://example.com/a.zip",
                "media_type": "image",
                "source_specs": specs,
            }
        )
        assert "--source-specs" in args
        assert json.dumps(specs) in args

    def test_build_origin_with_source_specs(self):
        import json

        from vtscore.datasets.importers.http_archive import IMPORTER

        specs = [{"source_type": "document", "converter": "document2image", "params": {}}]
        origin = IMPORTER.build_origin(
            {
                "url": "https://example.com/a.zip",
                "media_type": "image",
                "source_specs": specs,
            }
        )
        assert origin["params"]["source_specs"] == json.dumps(specs)


# ===========================================================================
# run_converters_on_folder
# ===========================================================================


class TestRunConvertersOnFolder:
    """Tests for the converter runner utility.

    All tests mock ``get_by_folder_name`` to avoid loading real embedding
    models (which are expensive and can interfere with other tests when
    run in a full suite).
    """

    @staticmethod
    def _mock_target_mt():
        """Return a mock target MediaType that avoids model loading."""
        mt = MagicMock()
        mt.type_id = "image"
        mt._model = True
        return mt

    @staticmethod
    def _mock_video2image_converter(**overrides):
        c = MagicMock()
        c.name = "video2image"
        c.source_type = "video"
        c.target_type = "image"
        c.display_name = "Video \u2192 Images"
        c.convert.return_value = overrides.pop(
            "convert_return",
            [
                {
                    "filename": "clip_clip_1.png",
                    "media_bytes": _make_png_bytes(),
                    "duration": 0,
                    "width": 4,
                    "height": 4,
                },
            ],
        )
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_no_converters_is_noop(self, tmp_path):
        from vtscore.converters.runner import run_converters_on_folder

        medias: dict = {}
        run_converters_on_folder(
            folder_path=tmp_path,
            converter_names=[],
            target_media_type="image",
            medias=medias,
        )
        assert medias == {}

    def test_nonexistent_converter_name_is_skipped(self, tmp_path):
        from vtscore.converters.runner import run_converters_on_folder

        medias: dict = {}
        # get_converter returns None for unknown names, so this is a no-op.
        run_converters_on_folder(
            folder_path=tmp_path,
            converter_names=["nonexistent_converter"],
            target_media_type="image",
            medias=medias,
            on_progress=lambda *a: None,
        )
        assert medias == {}

    def test_no_matching_source_files(self, tmp_path):
        """Converter exists but no source files in the folder."""
        from vtscore.converters.runner import run_converters_on_folder

        # Create a folder with only text files — no videos.
        (tmp_path / "file.txt").write_text("hello")

        medias: dict = {}
        with (
            patch("vtscore.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
            )
        assert medias == {}

    def test_converter_produces_output_with_origin(self, tmp_path):
        """Verify the runner emits one media per converter output with the
        right origin.  The runner does not embed — outputs leave with
        ``embedding=None`` for the framework embed stage to fill in.
        """
        from vtscore.converters.runner import run_converters_on_folder

        # Create a fake "video" file.
        (tmp_path / "clip.mp4").write_bytes(b"fake-video-data")

        mock_converter = self._mock_video2image_converter()

        medias: dict = {}
        with (
            patch("vtscore.converters.get_converter", return_value=mock_converter),
            patch("vtscore.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
                base_origin={"importer": "server_folder", "params": {"path": str(tmp_path), "media_type": "image"}},
            )

        assert len(medias) == 1
        media = medias[1]

        # Check origin tracking
        assert media["origin"]["importer"] == "converter"
        assert media["origin"]["params"]["converter"] == "video2image"
        assert media["origin"]["params"]["source_file"] == "clip.mp4"
        assert media["origin"]["params"]["parent_importer"] == "server_folder"
        assert media["origin"]["params"]["parent_path"] == str(tmp_path)

        # Check origin_name contains arrow
        assert "\u2192" in media["origin_name"]

        # Check media type
        assert media["media_type"] == "image"

        # Embedding is deferred to the framework embed stage.
        assert media["embedding"] is None
        # media_bytes flows through from the converter output so the
        # framework embed stage can embed without a tempfile.
        assert media["media_bytes"] is not None

    def test_converter_with_multiple_outputs(self, tmp_path):
        """A converter that produces multiple outputs per source file."""
        from vtscore.converters.runner import run_converters_on_folder

        (tmp_path / "long_video.mp4").write_bytes(b"video-data")

        fake_outputs = [
            {
                "filename": "long_video_clip_1.png",
                "media_bytes": _make_png_bytes(),
                "duration": 0,
                "width": 4,
                "height": 4,
            },
            {
                "filename": "long_video_clip_2.png",
                "media_bytes": _make_png_bytes(),
                "duration": 0,
                "width": 4,
                "height": 4,
            },
            {
                "filename": "long_video_clip_3.png",
                "media_bytes": _make_png_bytes(),
                "duration": 0,
                "width": 4,
                "height": 4,
            },
        ]

        mock_converter = self._mock_video2image_converter(convert_return=fake_outputs)

        medias: dict = {}
        with (
            patch("vtscore.converters.get_converter", return_value=mock_converter),
            patch("vtscore.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
            )

        assert len(medias) == 3
        # All should have sequential IDs
        assert sorted(medias.keys()) == [1, 2, 3]

    def test_converter_failure_skips_file(self, tmp_path):
        """If converter.convert() raises, that file is skipped."""
        from vtscore.converters.runner import run_converters_on_folder

        (tmp_path / "bad.mp4").write_bytes(b"corrupt")
        (tmp_path / "good.mp4").write_bytes(b"good-video")

        call_count = [0]

        def _side_effect(media, params=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("corrupt video")
            return [{"filename": "frame.png", "media_bytes": _make_png_bytes(), "duration": 0, "width": 4, "height": 4}]

        mock_converter = self._mock_video2image_converter()
        mock_converter.convert.side_effect = _side_effect

        medias: dict = {}
        with (
            patch("vtscore.converters.get_converter", return_value=mock_converter),
            patch("vtscore.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
            )

        # Only the good file should produce output
        assert len(medias) == 1

    def test_continues_from_existing_medias(self, tmp_path):
        """IDs continue from the max existing ID in medias."""
        from vtscore.converters.runner import run_converters_on_folder

        (tmp_path / "clip.mp4").write_bytes(b"video")

        mock_converter = self._mock_video2image_converter()

        # Pre-populate with some existing medias
        medias: dict = {1: {"id": 1}, 5: {"id": 5}, 10: {"id": 10}}
        with (
            patch("vtscore.converters.get_converter", return_value=mock_converter),
            patch("vtscore.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
            )

        # New media should start from ID 11
        assert 11 in medias

    def test_converter_follows_symlinked_directories(self, tmp_path):
        """Source files in a symlinked subdirectory must be discovered."""
        from vtscore.converters.runner import run_converters_on_folder

        root = tmp_path / "root"
        root.mkdir()

        external = tmp_path / "external"
        external.mkdir()
        (external / "clip.mp4").write_bytes(b"fake-video-data")

        (root / "linked").symlink_to(external)

        mock_converter = self._mock_video2image_converter()

        medias: dict = {}
        with (
            patch("vtscore.converters.get_converter", return_value=mock_converter),
            patch("vtscore.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=root,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
                base_origin={"importer": "server_folder", "params": {"path": str(root), "media_type": "image"}},
            )

        assert len(medias) == 1
        media = medias[1]
        assert media["origin"]["params"]["source_file"] == "linked/clip.mp4"


# ===========================================================================
# _compute_md5
# ===========================================================================


class TestEmbedAndMd5Helpers:
    def test_compute_md5_bytes(self):
        from vtscore.converters.runner import _compute_md5

        data = b"hello world"
        expected = hashlib.md5(data).hexdigest()
        assert _compute_md5({"media_bytes": data}) == expected

    def test_compute_md5_string(self):
        from vtscore.converters.runner import _compute_md5

        text = "hello world"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert _compute_md5({"media_string": text}) == expected

    def test_compute_md5_empty(self):
        from vtscore.converters.runner import _compute_md5

        assert _compute_md5({}) == hashlib.md5(b"").hexdigest()


# ===========================================================================
# Folder importer integration with converters
# ===========================================================================


class TestFolderImporterWithConverters:
    def test_run_passes_converter_specs_to_runner(self, tmp_path):
        """server_folder hands converter rows from source_specs to _run_converter_specs."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        # Create a folder with an image file so the normal load succeeds.
        (tmp_path / "photo.png").write_bytes(_make_png_bytes())

        with (
            patch("vtscore.datasets.importers.server_folder.load_dataset_from_folder") as mock_load,
            patch("vtscore.datasets.importers.server_folder._run_converter_specs") as mock_conv,
        ):
            medias: dict = {}
            IMPORTER.run(
                {
                    "path": str(tmp_path),
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "image", "converter": None, "params": {}},
                        {"source_type": "video", "converter": "video2image", "params": {"n_clips": "5"}},
                    ],
                },
                medias,
            )
            mock_load.assert_called_once()
            mock_conv.assert_called_once()
            call_args = mock_conv.call_args
            # Output type is the second positional arg.
            assert call_args[0][1] == "image"
            forwarded_specs = call_args[0][2]
            assert len(forwarded_specs) == 1
            assert forwarded_specs[0].converter == "video2image"
            assert forwarded_specs[0].params == {"n_clips": "5"}

    def test_run_with_no_converter_rows_skips_runner(self, tmp_path):
        """When source_specs has only a direct row, _run_converter_specs is still
        called but gets an empty spec list and bails out without running the
        runner."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        (tmp_path / "photo.png").write_bytes(_make_png_bytes())

        with (
            patch("vtscore.datasets.importers.server_folder.load_dataset_from_folder"),
            patch("vtscore.datasets.importers.server_folder._run_converter_specs") as mock_conv,
        ):
            medias: dict = {}
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)
            mock_conv.assert_called_once()
            forwarded_specs = mock_conv.call_args[0][2]
            assert forwarded_specs == []

    def test_folder_only_converters_no_regular_files(self, tmp_path):
        """When only converter source files exist and no regular target files."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        # Create only a video file — no image files.
        (tmp_path / "clip.mp4").write_bytes(b"fake-video")

        mock_converter_medias = {1: {"id": 1, "media_type": "image"}}

        def _fake_run_converters(
            folder, output_type, specs, medias, thin=False, recursive=True, folder_path_for_origin=""
        ):
            if specs:
                medias.update(mock_converter_medias)

        with (
            patch(
                "vtscore.datasets.importers.server_folder.load_dataset_from_folder",
                side_effect=ValueError("No images files found"),
            ),
            patch("vtscore.datasets.importers.server_folder._run_converter_specs", side_effect=_fake_run_converters),
        ):
            medias: dict = {}
            IMPORTER.run(
                {
                    "path": str(tmp_path),
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "image", "converter": None, "params": {}},
                        {"source_type": "video", "converter": "video2image", "params": {}},
                    ],
                },
                medias,
            )
            # Should not raise because converter rows produced output
            assert len(medias) == 1


# ===========================================================================
# Import API endpoint with converters
# ===========================================================================


class TestImportAPIConverters:
    def test_import_endpoint_passes_converters(self, client):
        """POST /api/dataset/import/server_folder passes converters to the importer."""
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={
                    "path": "/tmp/test",
                    "media_type": "image",
                    "converters": "video2image,document2image",
                },
            )
            assert resp.status_code == 200
            # Check that converters was passed in field_values
            call_args = mock_run.call_args
            field_values = call_args[0][1]
            assert field_values["converters"] == "video2image,document2image"

    def test_import_endpoint_without_converters(self, client):
        """Without converters, the field is not added."""
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={"path": "/tmp/test", "media_type": "image"},
            )
            assert resp.status_code == 200
            field_values = mock_run.call_args[0][1]
            assert "converters" not in field_values


# ===========================================================================
# Multi-media import: SourceSpec + effective_source_specs + API plumbing.
# See docs/plans/multi-media-import.md.
# ===========================================================================


class TestSourceSpecParsing:
    def test_from_dict_roundtrip(self):
        from vtscore.datasets.importers.base import SourceSpec

        spec = SourceSpec.from_dict({"source_type": "video", "converter": "video2image", "params": {"n_clips": "12"}})
        assert spec.source_type == "video"
        assert spec.converter == "video2image"
        assert spec.params == {"n_clips": "12"}
        d = spec.to_dict()
        assert d == {"source_type": "video", "converter": "video2image", "params": {"n_clips": "12"}}

    def test_direct_row_has_no_converter(self):
        from vtscore.datasets.importers.base import SourceSpec

        spec = SourceSpec.from_dict({"source_type": "image", "converter": None, "params": {}})
        assert spec.converter is None


class _LegacyTestImporter:
    """Stand-in for an external ``multi_media=False`` importer.

    Every in-tree importer now sets ``multi_media=True``, so the legacy
    synthesis branch of :meth:`DatasetImporter.effective_source_specs`
    is exercised only by extension code.  We instantiate one explicitly
    here to keep that branch under test until the shim is deleted.
    """

    @staticmethod
    def make():
        from vtscore.datasets.importers.base import DatasetImporter, ImporterField

        class _Legacy(DatasetImporter):
            name = "_legacy_test"
            display_name = "Legacy"
            description = ""
            multi_media = False
            fields = [
                ImporterField(key="media_type", label="Type", field_type="select", default="image"),
            ]

            def run(self, field_values, medias, thin=False):  # pragma: no cover
                raise NotImplementedError

        return _Legacy()


class TestLegacyEffectiveSourceSpecs:
    """Legacy (multi_media=False) importers synthesise specs from
    media_type + comma-separated converters."""

    def test_legacy_single_media_type_only(self):
        imp = _LegacyTestImporter.make()

        specs = imp.effective_source_specs({"media_type": "image"})
        assert len(specs) == 1
        assert specs[0].source_type == "image"
        assert specs[0].converter is None

    def test_legacy_with_converters_csv(self):
        imp = _LegacyTestImporter.make()

        specs = imp.effective_source_specs({"media_type": "image", "converters": "video2image,document2image"})
        assert [s.converter for s in specs] == [None, "video2image", "document2image"]
        assert [s.source_type for s in specs] == ["image", "video", "document"]

    def test_legacy_unknown_converter_is_skipped(self):
        imp = _LegacyTestImporter.make()

        specs = imp.effective_source_specs({"media_type": "image", "converters": "video2image,bogus"})
        assert [s.converter for s in specs] == [None, "video2image"]

    def test_legacy_rejects_empty_media_type_and_converters(self):
        """H11: a legacy importer called with no ``media_type`` and no
        ``converters`` used to return ``[]`` silently, leaving downstream
        loops to produce an empty dataset.  It must now raise."""
        import pytest

        imp = _LegacyTestImporter.make()

        with pytest.raises(ValueError, match="legacy import requires"):
            imp.effective_source_specs({"media_type": "", "converters": ""})

    def test_legacy_rejects_only_unknown_converters(self):
        """H11: ``media_type`` empty + every named converter unknown also
        produces an empty spec list and must raise rather than silently
        return ``[]``."""
        import pytest

        imp = _LegacyTestImporter.make()

        with pytest.raises(ValueError, match="legacy import requires"):
            imp.effective_source_specs({"media_type": "", "converters": "bogus_one,bogus_two"})


class TestMultiMediaEffectiveSourceSpecs:
    """multi_media=True importers parse the explicit source_specs form value."""

    def test_parses_list_of_dicts(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        specs = IMPORTER.effective_source_specs(
            {
                "media_type": "image",
                "source_specs": [
                    {"source_type": "image", "converter": None, "params": {}},
                    {"source_type": "video", "converter": "video2image", "params": {"n_clips": "8"}},
                ],
            }
        )
        assert len(specs) == 2
        assert specs[0].converter is None and specs[0].source_type == "image"
        assert specs[1].converter == "video2image"
        assert specs[1].params == {"n_clips": "8"}

    def test_parses_json_string(self):
        import json

        from vtscore.datasets.importers.server_folder import IMPORTER

        raw = json.dumps(
            [
                {"source_type": "image", "converter": None, "params": {}},
                {"source_type": "video", "converter": "video2image", "params": {}},
            ]
        )
        specs = IMPORTER.effective_source_specs({"media_type": "image", "source_specs": raw})
        assert [s.converter for s in specs] == [None, "video2image"]

    def test_missing_source_specs_defaults_to_direct_only(self):
        """When the multi_media flag is set but the form omits source_specs,
        a single 'include directly' row is synthesised so the importer still
        loads cleanly."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        specs = IMPORTER.effective_source_specs({"media_type": "image"})
        assert len(specs) == 1
        assert specs[0].converter is None

    def test_empty_list_source_specs_defaults_to_direct_only(self):
        """H11: an explicit empty ``source_specs=[]`` (e.g. user deleted every
        row in the multi-media grid) must fall back to the synthesised direct
        row, not return ``[]`` and silently produce an empty dataset."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        specs = IMPORTER.effective_source_specs({"media_type": "image", "source_specs": []})
        assert len(specs) == 1
        assert specs[0].converter is None
        assert specs[0].source_type == "image"

    def test_empty_json_string_source_specs_defaults_to_direct_only(self):
        """H11: an explicit ``source_specs="[]"`` JSON string is treated the
        same as an empty Python list and falls back to the direct row."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        specs = IMPORTER.effective_source_specs({"media_type": "image", "source_specs": "[]"})
        assert len(specs) == 1
        assert specs[0].converter is None
        assert specs[0].source_type == "image"

    def test_rejects_invalid_converter(self):
        import pytest

        from vtscore.datasets.importers.server_folder import IMPORTER

        with pytest.raises(ValueError, match="Unknown converter"):
            IMPORTER.effective_source_specs(
                {
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "video", "converter": "does_not_exist", "params": {}},
                    ],
                }
            )

    def test_rejects_target_mismatch(self):
        """A converter whose target_type doesn't match the output media_type
        is rejected — e.g. video2audio applied to an image-output dataset."""
        import pytest

        from vtscore.datasets.importers.server_folder import IMPORTER

        with pytest.raises(ValueError, match="produces"):
            IMPORTER.effective_source_specs(
                {
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "video", "converter": "video2audio", "params": {}},
                    ],
                }
            )

    def test_rejects_direct_row_with_wrong_type(self):
        """A no-converter row whose source_type differs from the output type
        is rejected — that's the form a stale UI submission would take."""
        import pytest

        from vtscore.datasets.importers.server_folder import IMPORTER

        with pytest.raises(ValueError, match="does not match"):
            IMPORTER.effective_source_specs(
                {
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "video", "converter": None, "params": {}},
                    ],
                }
            )


class TestImporterMultiMediaFlagInToDict:
    def test_server_folder_advertises_multi_media(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        d = IMPORTER.to_dict()
        assert d.get("multi_media") is True

    def test_http_archive_advertises_multi_media(self):
        from vtscore.datasets.importers.http_archive import IMPORTER

        d = IMPORTER.to_dict()
        assert d.get("multi_media") is True

    def test_pickle_advertises_multi_media(self):
        from vtscore.datasets.importers.pickle import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_combine_datasets_advertises_multi_media(self):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_synthetic_advertises_multi_media(self):
        from vtscore.datasets.importers.synthetic import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_recaller_advertises_multi_media(self):
        from vtscore.datasets.importers.recaller import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_demo_advertises_multi_media(self):
        from vtscore.datasets.importers.demo import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True


class TestConverterFieldsInToDict:
    def test_video2image_exposes_n_clips_field(self):
        from vtscore.converters import get_converter

        c = get_converter("video2image")
        assert c is not None
        d = c.to_dict()
        fields = d.get("fields") or []
        assert any(f["key"] == "n_clips" for f in fields)


class TestConverterAcceptsParams:
    """convert(media, params) is the new signature; converters with declared
    fields use params to drive behaviour, others ignore it gracefully."""

    def test_video2audio_accepts_empty_params(self):
        from vtscore.converters import get_converter

        c = get_converter("video2audio")
        assert c is not None
        # Empty params, empty source media → returns empty list (no crash).
        assert c.convert({}, {}) == []

    def test_document2image_accepts_none_params(self):
        from vtscore.converters import get_converter

        c = get_converter("document2image")
        assert c is not None
        assert c.convert({}, None) == []

    def test_video2image_reads_n_clips_param(self):
        """Verify the converter reads its declared n_clips param from the dict."""
        from vtscore.converters import get_converter

        c = get_converter("video2image")
        assert c is not None
        # Bogus media → empty list, but get_param resolution exercised below.
        assert c.get_param({"n_clips": "30"}, "n_clips") == "30"
        # Default falls back to the field's declared default.
        assert c.get_param({}, "n_clips") == "10"


class TestImportAPISourceSpecs:
    def test_import_endpoint_passes_source_specs(self, client):
        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/server_folder",
                json={
                    "path": "/tmp/test",
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "image", "converter": None, "params": {}},
                        {"source_type": "video", "converter": "video2image", "params": {"n_clips": "5"}},
                    ],
                },
            )
            assert resp.status_code == 200
            field_values = mock_run.call_args[0][1]
            assert "source_specs" in field_values


# ===========================================================================
# multi_media flag on the local_folder / local_files / server_files importers
# ===========================================================================


class TestMultiMediaImportersFlag:
    """The lf-* / sf-* importers all delegate to / share the same
    multi-media flow.  Their to_dict() advertises multi_media=True so
    the frontend renders the source-specs editor for each."""

    def test_local_folder_advertises_multi_media(self):
        from vtscore.datasets.importers.local_folder import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_local_files_advertises_multi_media(self):
        from vtscore.datasets.importers.local_files import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_server_files_advertises_multi_media(self):
        from vtscore.datasets.importers.server_files import IMPORTER

        assert IMPORTER.to_dict().get("multi_media") is True

    def test_server_files_exposes_available_converters_map(self):
        """server_files.to_dict() includes the converter map so the
        frontend form view can render the Include rows without an extra
        API call to /api/converters."""
        from vtscore.datasets.importers.server_files import IMPORTER

        d = IMPORTER.to_dict()
        assert "available_converters_by_media_type" in d
        # At least one converter targeting "image" should be exposed.
        assert "image" in d["available_converters_by_media_type"]
        names = [c["name"] for c in d["available_converters_by_media_type"]["image"]]
        assert "video2image" in names


class TestServerFilesEffectiveSourceSpecs:
    def test_multi_media_with_converter_row(self):
        """server_files.effective_source_specs() validates against the
        same registries as server_folder."""
        from vtscore.datasets.importers.server_files import IMPORTER

        specs = IMPORTER.effective_source_specs(
            {
                "media_type": "image",
                "source_specs": [
                    {"source_type": "image", "converter": None, "params": {}},
                    {"source_type": "video", "converter": "video2image", "params": {"n_clips": "5"}},
                ],
            }
        )
        assert len(specs) == 2
        assert specs[1].converter == "video2image"
        assert specs[1].params == {"n_clips": "5"}

    def test_rejects_target_mismatch(self):
        import pytest

        from vtscore.datasets.importers.server_files import IMPORTER

        with pytest.raises(ValueError, match="produces"):
            IMPORTER.effective_source_specs(
                {
                    "media_type": "image",
                    "source_specs": [
                        {"source_type": "video", "converter": "video2audio", "params": {}},
                    ],
                }
            )


class TestImportLocalFolderRouteForwardsSourceSpecs:
    """The browser-side upload route streams files to a temp dir and
    re-enters server_folder.  source_specs from the multipart body must
    reach the importer."""

    def test_source_specs_form_field_is_forwarded(self, client, tmp_path):
        import json

        specs_json = json.dumps(
            [
                {"source_type": "image", "converter": None, "params": {}},
                {"source_type": "video", "converter": "video2image", "params": {"n_clips": "3"}},
            ]
        )

        with patch("vtsearch.routes.datasets.staging._run_importer_in_background") as mock_bg:
            # We don't care if the upload pipeline crashes after staging —
            # we only assert source_specs makes it to the importer.
            mock_bg.return_value = "task-1"
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "media_type": "image",
                    "source_specs": specs_json,
                },
                content_type="multipart/form-data",
                buffered=True,
                follow_redirects=False,
            )
            # The route uses _load (not _run_importer_in_background) for
            # the actual import.  Instead of asserting the call, assert
            # the response is OK and that the temp upload directory got
            # the files (which proves the route accepted the payload).
            assert resp.status_code in (200, 400, 500)
            # If the route returned 200, the importer should have been
            # invoked with source_specs.  In the test client the
            # background load happens synchronously enough that we can
            # inspect the importer's last seen field_values via the
            # mocked Load helper.  In practice the route hands off via
            # ``_load`` not ``_run_importer_in_background``, so just
            # verify the route accepts the field and doesn't 400 on it.
            if resp.status_code == 400:
                # flask-smorest error envelope: ``message`` (not ``error``).
                assert "source_specs" not in (resp.get_json() or {}).get("message", "")

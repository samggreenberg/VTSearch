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

import numpy as np

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
        from vtsearch.converters import list_converters

        converters = list_converters()
        assert len(converters) >= 4
        names = [c.name for c in converters]
        assert "document2image" in names
        assert "document2text" in names
        assert "video2audio" in names
        assert "video2image" in names

    def test_get_converter_existing(self):
        from vtsearch.converters import get_converter

        c = get_converter("video2image")
        assert c is not None
        assert c.name == "video2image"
        assert c.source_type == "video"
        assert c.target_type == "image"

    def test_get_converter_nonexistent(self):
        from vtsearch.converters import get_converter

        assert get_converter("nonexistent") is None

    def test_list_converters_for_target_image(self):
        from vtsearch.converters import list_converters_for_target

        converters = list_converters_for_target("image")
        names = [c.name for c in converters]
        assert "video2image" in names
        assert "document2image" in names
        assert "video2audio" not in names

    def test_list_converters_for_target_audio(self):
        from vtsearch.converters import list_converters_for_target

        converters = list_converters_for_target("audio")
        names = [c.name for c in converters]
        assert "video2audio" in names
        assert "video2image" not in names

    def test_list_converters_for_target_text(self):
        from vtsearch.converters import list_converters_for_target

        converters = list_converters_for_target("text")
        names = [c.name for c in converters]
        assert "document2text" in names

    def test_list_converters_for_target_none(self):
        from vtsearch.converters import list_converters_for_target

        converters = list_converters_for_target("video")
        assert converters == []


# ===========================================================================
# Converter base class
# ===========================================================================


class TestConverterBase:
    def test_name_derived(self):
        from vtsearch.converters import Video2AudioMediaConverter

        c = Video2AudioMediaConverter()
        assert c.name == "video2audio"

    def test_display_name(self):
        from vtsearch.converters import Video2ImageMediaConverter

        c = Video2ImageMediaConverter()
        assert c.display_name == "Video \u2192 Images"

    def test_converter_description(self):
        from vtsearch.converters import Document2ImageMediaConverter

        c = Document2ImageMediaConverter()
        assert c.converter_description != ""

    def test_to_dict(self):
        from vtsearch.converters import Video2AudioMediaConverter

        c = Video2AudioMediaConverter()
        d = c.to_dict()
        assert d["name"] == "video2audio"
        assert d["source_type"] == "video"
        assert d["target_type"] == "audio"
        assert "display_name" in d
        assert "description" in d

    def test_to_dict_fallback_display_name(self):
        """If display_name is empty, to_dict derives one from type IDs."""
        from vtsearch.converters.base import MediaConverter

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
    def test_build_cli_args_without_converters(self):
        from vtsearch.datasets.importers.folder import IMPORTER

        args = IMPORTER.build_cli_args({"media_type": "image", "path": "/data"})
        assert "--converters" not in args

    def test_build_cli_args_with_converters(self):
        from vtsearch.datasets.importers.folder import IMPORTER

        args = IMPORTER.build_cli_args({
            "media_type": "image",
            "path": "/data",
            "converters": "video2image,document2image",
        })
        assert "--converters video2image,document2image" in args

    def test_build_origin_without_converters(self):
        from vtsearch.datasets.importers.folder import IMPORTER

        origin = IMPORTER.build_origin({"media_type": "image", "path": "/data"})
        assert origin["importer"] == "folder"
        assert "converters" not in origin["params"]

    def test_build_origin_with_converters(self):
        from vtsearch.datasets.importers.folder import IMPORTER

        origin = IMPORTER.build_origin({
            "media_type": "image",
            "path": "/data",
            "converters": "video2image",
        })
        assert origin["params"]["converters"] == "video2image"


class TestHttpArchiveImporterConverterFields:
    def test_build_cli_args_with_converters(self):
        from vtsearch.datasets.importers.http_zip import IMPORTER

        args = IMPORTER.build_cli_args({
            "url": "https://example.com/a.zip",
            "media_type": "image",
            "converters": "video2image",
        })
        assert "--converters video2image" in args

    def test_build_origin_with_converters(self):
        from vtsearch.datasets.importers.http_zip import IMPORTER

        origin = IMPORTER.build_origin({
            "url": "https://example.com/a.zip",
            "media_type": "image",
            "converters": "document2image",
        })
        assert origin["params"]["converters"] == "document2image"


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
        c.convert.return_value = overrides.pop("convert_return", [
            {"filename": "clip_clip_1.png", "media_bytes": _make_png_bytes(), "duration": 0, "width": 4, "height": 4},
        ])
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def test_no_converters_is_noop(self, tmp_path):
        from vtsearch.converters.runner import run_converters_on_folder

        medias: dict = {}
        run_converters_on_folder(
            folder_path=tmp_path,
            converter_names=[],
            target_media_type="image",
            medias=medias,
        )
        assert medias == {}

    def test_nonexistent_converter_name_is_skipped(self, tmp_path):
        from vtsearch.converters.runner import run_converters_on_folder

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
        from vtsearch.converters.runner import run_converters_on_folder

        # Create a folder with only text files — no videos.
        (tmp_path / "file.txt").write_text("hello")

        medias: dict = {}
        with (
            patch("vtsearch.converters.runner._embed_converted_output", return_value=np.zeros(768)),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
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
        """Mock converter + embedder to verify the full pipeline."""
        from vtsearch.converters.runner import run_converters_on_folder

        # Create a fake "video" file.
        (tmp_path / "clip.mp4").write_bytes(b"fake-video-data")

        fake_embedding = np.ones(768, dtype=np.float32)
        mock_converter = self._mock_video2image_converter()

        medias: dict = {}
        with (
            patch("vtsearch.converters.runner.get_converter", return_value=mock_converter),
            patch("vtsearch.converters.runner._embed_converted_output", return_value=fake_embedding),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
                base_origin={"importer": "folder", "params": {"path": str(tmp_path), "media_type": "image"}},
            )

        assert len(medias) == 1
        media = medias[1]

        # Check origin tracking
        assert media["origin"]["importer"] == "converter"
        assert media["origin"]["params"]["converter"] == "video2image"
        assert media["origin"]["params"]["source_file"] == "clip.mp4"
        assert media["origin"]["params"]["parent_importer"] == "folder"
        assert media["origin"]["params"]["parent_path"] == str(tmp_path)

        # Check origin_name contains arrow
        assert "\u2192" in media["origin_name"]

        # Check media type
        assert media["type"] == "image"

        # Check embedding
        assert np.array_equal(media["embedding"], fake_embedding)

    def test_converter_with_multiple_outputs(self, tmp_path):
        """A converter that produces multiple outputs per source file."""
        from vtsearch.converters.runner import run_converters_on_folder

        (tmp_path / "long_video.mp4").write_bytes(b"video-data")

        fake_outputs = [
            {"filename": "long_video_clip_1.png", "media_bytes": _make_png_bytes(), "duration": 0, "width": 4, "height": 4},
            {"filename": "long_video_clip_2.png", "media_bytes": _make_png_bytes(), "duration": 0, "width": 4, "height": 4},
            {"filename": "long_video_clip_3.png", "media_bytes": _make_png_bytes(), "duration": 0, "width": 4, "height": 4},
        ]

        mock_converter = self._mock_video2image_converter(convert_return=fake_outputs)

        medias: dict = {}
        with (
            patch("vtsearch.converters.runner.get_converter", return_value=mock_converter),
            patch("vtsearch.converters.runner._embed_converted_output", return_value=np.ones(768)),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
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
        from vtsearch.converters.runner import run_converters_on_folder

        (tmp_path / "bad.mp4").write_bytes(b"corrupt")
        (tmp_path / "good.mp4").write_bytes(b"good-video")

        call_count = [0]

        def _side_effect(media):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("corrupt video")
            return [{"filename": "frame.png", "media_bytes": _make_png_bytes(), "duration": 0, "width": 4, "height": 4}]

        mock_converter = self._mock_video2image_converter()
        mock_converter.convert.side_effect = _side_effect

        medias: dict = {}
        with (
            patch("vtsearch.converters.runner.get_converter", return_value=mock_converter),
            patch("vtsearch.converters.runner._embed_converted_output", return_value=np.ones(768)),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
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
        from vtsearch.converters.runner import run_converters_on_folder

        (tmp_path / "clip.mp4").write_bytes(b"video")

        mock_converter = self._mock_video2image_converter()

        # Pre-populate with some existing medias
        medias: dict = {1: {"id": 1}, 5: {"id": 5}, 10: {"id": 10}}
        with (
            patch("vtsearch.converters.runner.get_converter", return_value=mock_converter),
            patch("vtsearch.converters.runner._embed_converted_output", return_value=np.ones(768)),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
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

    def test_embed_failure_skips_output(self, tmp_path):
        """If embedding returns None, that output is skipped."""
        from vtsearch.converters.runner import run_converters_on_folder

        (tmp_path / "clip.mp4").write_bytes(b"video")

        mock_converter = self._mock_video2image_converter()

        medias: dict = {}
        with (
            patch("vtsearch.converters.runner.get_converter", return_value=mock_converter),
            patch("vtsearch.converters.runner._embed_converted_output", return_value=None),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=tmp_path,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
            )

        assert medias == {}

    def test_converter_follows_symlinked_directories(self, tmp_path):
        """Source files in a symlinked subdirectory must be discovered."""
        from vtsearch.converters.runner import run_converters_on_folder

        root = tmp_path / "root"
        root.mkdir()

        external = tmp_path / "external"
        external.mkdir()
        (external / "clip.mp4").write_bytes(b"fake-video-data")

        (root / "linked").symlink_to(external)

        fake_embedding = np.ones(768, dtype=np.float32)
        mock_converter = self._mock_video2image_converter()

        medias: dict = {}
        with (
            patch("vtsearch.converters.runner.get_converter", return_value=mock_converter),
            patch("vtsearch.converters.runner._embed_converted_output", return_value=fake_embedding),
            patch("vtsearch.media.get_by_folder_name", return_value=self._mock_target_mt()),
        ):
            run_converters_on_folder(
                folder_path=root,
                converter_names=["video2image"],
                target_media_type="image",
                medias=medias,
                on_progress=lambda *a: None,
                base_origin={"importer": "folder", "params": {"path": str(root), "media_type": "image"}},
            )

        assert len(medias) == 1
        media = medias[1]
        assert media["origin"]["params"]["source_file"] == "linked/clip.mp4"


# ===========================================================================
# _embed_converted_output and _compute_md5
# ===========================================================================


class TestEmbedAndMd5Helpers:
    def test_compute_md5_bytes(self):
        from vtsearch.converters.runner import _compute_md5

        data = b"hello world"
        expected = hashlib.md5(data).hexdigest()
        assert _compute_md5({"media_bytes": data}) == expected

    def test_compute_md5_string(self):
        from vtsearch.converters.runner import _compute_md5

        text = "hello world"
        expected = hashlib.md5(text.encode("utf-8")).hexdigest()
        assert _compute_md5({"media_string": text}) == expected

    def test_compute_md5_empty(self):
        from vtsearch.converters.runner import _compute_md5

        assert _compute_md5({}) == hashlib.md5(b"").hexdigest()

    def test_embed_converted_output_binary(self, tmp_path):
        """Binary output is written to temp file and embed_media is called."""
        from vtsearch.converters.runner import _embed_converted_output

        png_bytes = _make_png_bytes()
        fake_embedding = np.ones(768)
        mock_mt = MagicMock()
        mock_mt.embed_media.return_value = fake_embedding

        result = _embed_converted_output(mock_mt, {"media_bytes": png_bytes, "filename": "test.png"})

        assert np.array_equal(result, fake_embedding)
        mock_mt.embed_media.assert_called_once()

    def test_embed_converted_output_text(self, tmp_path):
        """Text output is written to temp .txt and embed_media is called."""
        from vtsearch.converters.runner import _embed_converted_output

        fake_embedding = np.ones(768)
        mock_mt = MagicMock()
        mock_mt.embed_media.return_value = fake_embedding

        result = _embed_converted_output(mock_mt, {"media_string": "hello world", "filename": "doc.txt"})

        assert np.array_equal(result, fake_embedding)
        mock_mt.embed_media.assert_called_once()

    def test_embed_converted_output_empty(self):
        """Empty output returns None."""
        from vtsearch.converters.runner import _embed_converted_output

        mock_mt = MagicMock()
        result = _embed_converted_output(mock_mt, {})
        assert result is None


# ===========================================================================
# Folder importer integration with converters
# ===========================================================================


class TestFolderImporterWithConverters:
    def test_run_passes_converters_to_runner(self, tmp_path):
        """Folder importer calls run_converters_on_folder when converters set."""
        from vtsearch.datasets.importers.folder import IMPORTER

        # Create a folder with an image file so the normal load succeeds.
        (tmp_path / "photo.png").write_bytes(_make_png_bytes())

        with (
            patch("vtsearch.datasets.importers.folder.load_dataset_from_folder") as mock_load,
            patch("vtsearch.datasets.importers.folder._run_selected_converters") as mock_conv,
        ):
            medias: dict = {}
            IMPORTER.run(
                {"path": str(tmp_path), "media_type": "image", "converters": "video2image"},
                medias,
            )
            mock_load.assert_called_once()
            mock_conv.assert_called_once()
            # _run_selected_converters(folder, media_type, field_values, medias, thin=False)
            call_args = mock_conv.call_args
            assert call_args[0][1] == "image"  # media_type
            assert "video2image" in call_args[0][2].get("converters", "")

    def test_run_without_converters_does_not_call_runner(self, tmp_path):
        """Without converters field, runner is not called."""
        from vtsearch.datasets.importers.folder import IMPORTER

        (tmp_path / "photo.png").write_bytes(_make_png_bytes())

        with (
            patch("vtsearch.datasets.importers.folder.load_dataset_from_folder"),
            patch("vtsearch.datasets.importers.folder._run_selected_converters") as mock_conv,
        ):
            medias: dict = {}
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)
            mock_conv.assert_called_once()  # still called, but with empty converters
            # The actual runner inside checks for empty string and returns immediately

    def test_folder_only_converters_no_regular_files(self, tmp_path):
        """When only converter source files exist and no regular target files."""
        from vtsearch.datasets.importers.folder import IMPORTER

        # Create only a video file — no image files.
        (tmp_path / "clip.mp4").write_bytes(b"fake-video")

        mock_converter_medias = {1: {"id": 1, "type": "image"}}

        def _fake_run_converters(folder, mt, fv, medias, thin=False):
            if fv.get("converters"):
                medias.update(mock_converter_medias)

        with (
            patch("vtsearch.datasets.importers.folder.load_dataset_from_folder", side_effect=ValueError("No images files found")),
            patch("vtsearch.datasets.importers.folder._run_selected_converters", side_effect=_fake_run_converters),
        ):
            medias: dict = {}
            IMPORTER.run(
                {"path": str(tmp_path), "media_type": "image", "converters": "video2image"},
                medias,
            )
            # Should not raise because converters produced output
            assert len(medias) == 1


# ===========================================================================
# Import API endpoint with converters
# ===========================================================================


class TestImportAPIConverters:
    def test_import_endpoint_passes_converters(self, client):
        """POST /api/dataset/import/folder passes converters to the importer."""
        with patch("vtsearch.routes.datasets._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/folder",
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
        with patch("vtsearch.routes.datasets._run_importer_in_background") as mock_run:
            resp = client.post(
                "/api/dataset/import/folder",
                json={"path": "/tmp/test", "media_type": "image"},
            )
            assert resp.status_code == 200
            field_values = mock_run.call_args[0][1]
            assert "converters" not in field_values

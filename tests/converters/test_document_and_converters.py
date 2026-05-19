"""Tests for the Document media type and MediaConverter framework."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.converters.base import MediaConverter
from vtscore.media.document.media_type import DocumentMediaType


# ---------------------------------------------------------------------------
# Helpers for creating minimal test files
# ---------------------------------------------------------------------------


def _make_minimal_pdf(text: str = "Hello World") -> bytes:
    """Build a tiny valid PDF with the given text on one page."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((50, 100), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_two_page_pdf() -> bytes:
    """Build a tiny valid PDF with two pages, each containing distinct text."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not installed")
    doc = fitz.open()
    p1 = doc.new_page(width=200, height=200)
    p1.insert_text((50, 100), "Page one content")
    p2 = doc.new_page(width=200, height=200)
    p2.insert_text((50, 100), "Page two content")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_minimal_video(frames: int = 30, width: int = 64, height: int = 64) -> bytes:
    """Create a minimal MP4 video using OpenCV.

    Returns the MP4 bytes, or calls pytest.skip() if cv2 is unavailable.
    """
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not installed")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    # cv2 stubs miss VideoWriter_fourcc (runtime-only opencv builtin).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # pyright: ignore[reportAttributeAccessIssue]
    writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (width, height))
    for i in range(frames):
        frame = np.full((height, width, 3), fill_value=(i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    video_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return video_bytes


# ===========================================================================
# DocumentMediaType tests
# ===========================================================================


class TestDocumentMediaType:
    def test_type_id(self):
        mt = DocumentMediaType()
        assert mt.type_id == "document"

    def test_name(self):
        mt = DocumentMediaType()
        assert mt.name == "Document"

    def test_icon(self):
        mt = DocumentMediaType()
        assert mt.icon == "document"

    def test_file_extensions(self):
        mt = DocumentMediaType()
        exts = mt.file_extensions
        assert "*.pdf" in exts
        assert "*.doc" in exts
        assert "*.ppt" in exts

    def test_folder_import_name(self):
        mt = DocumentMediaType()
        assert mt.folder_import_name == "document"

    def test_loops_false(self):
        mt = DocumentMediaType()
        assert mt.loops is False

    def test_no_demo_datasets(self):
        mt = DocumentMediaType()
        assert mt.demo_datasets == []

    def test_embed_text_returns_none(self):
        mt = DocumentMediaType()
        assert mt.embed_text("hello") is None

    def test_load_media_data(self, tmp_path):
        mt = DocumentMediaType()
        pdf_bytes = _make_minimal_pdf()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(pdf_bytes)
        data = mt.load_media_data(pdf_path)
        assert data["media_bytes"] == pdf_bytes
        assert data["duration"] == 0

    def test_media_response_pdf(self):
        mt = DocumentMediaType()
        media = {"id": 1, "filename": "report.pdf", "media_bytes": b"fake-pdf"}
        resp = mt.media_response(media)
        assert resp.mimetype == "application/pdf"
        assert resp.data == b"fake-pdf"

    def test_media_response_doc(self):
        mt = DocumentMediaType()
        media = {"id": 2, "filename": "report.doc", "media_bytes": b"fake-doc"}
        resp = mt.media_response(media)
        assert resp.mimetype == "application/msword"

    def test_media_response_ppt(self):
        mt = DocumentMediaType()
        media = {"id": 3, "filename": "slides.ppt", "media_bytes": b"fake-ppt"}
        resp = mt.media_response(media)
        assert resp.mimetype == "application/vnd.ms-powerpoint"


class TestDocumentRegistration:
    def test_document_in_registry(self):
        from vtscore.media import get

        mt = get("document")
        assert mt.type_id == "document"

    def test_get_by_extension_pdf(self):
        from vtscore.media import get_by_extension

        mt = get_by_extension(".pdf")
        assert mt is not None
        assert mt.type_id == "document"

    def test_get_by_extension_doc(self):
        from vtscore.media import get_by_extension

        mt = get_by_extension(".doc")
        assert mt is not None
        assert mt.type_id == "document"

    def test_get_by_extension_ppt(self):
        from vtscore.media import get_by_extension

        mt = get_by_extension(".ppt")
        assert mt is not None
        assert mt.type_id == "document"

    def test_get_by_folder_name(self):
        from vtscore.media import get_by_folder_name

        mt = get_by_folder_name("document")
        assert mt.type_id == "document"


# ===========================================================================
# MediaConverter base class tests
# ===========================================================================


class TestMediaConverterAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MediaConverter()  # pyright: ignore[reportAbstractUsage]

    def test_concrete_subclass(self):
        class DummyConverter(MediaConverter):
            @property
            def source_type(self):
                return "a"

            @property
            def target_type(self):
                return "b"

            def convert(self, media):
                return [{"filename": "out.txt", "media_string": "hello", "duration": 0}]

        c = DummyConverter()
        assert c.source_type == "a"
        assert c.target_type == "b"
        result = c.convert({"filename": "in.a"})
        assert len(result) == 1
        assert result[0]["filename"] == "out.txt"


# ===========================================================================
# Document2ImageMediaConverter tests
# ===========================================================================


class TestDocument2ImageMediaConverter:
    def test_source_and_target_types(self):
        from vtscore.converters.document2image import Document2ImageMediaConverter

        c = Document2ImageMediaConverter()
        assert c.source_type == "document"
        assert c.target_type == "image"

    def test_convert_single_page_pdf(self):
        from vtscore.converters.document2image import Document2ImageMediaConverter

        pdf_bytes = _make_minimal_pdf("Test page")
        media = {"filename": "test.pdf", "media_bytes": pdf_bytes}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "test_page_1.png"
        assert isinstance(r["media_bytes"], bytes)
        assert len(r["media_bytes"]) > 0
        assert r["duration"] == 0
        assert r["width"] > 0
        assert r["height"] > 0
        # Verify it's valid PNG
        assert r["media_bytes"][:4] == b"\x89PNG"

    def test_convert_two_page_pdf(self):
        from vtscore.converters.document2image import Document2ImageMediaConverter

        pdf_bytes = _make_two_page_pdf()
        media = {"filename": "multi.pdf", "media_bytes": pdf_bytes}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert len(results) == 2
        assert results[0]["filename"] == "multi_page_1.png"
        assert results[1]["filename"] == "multi_page_2.png"

    def test_convert_from_path(self, tmp_path):
        from vtscore.converters.document2image import Document2ImageMediaConverter

        pdf_bytes = _make_minimal_pdf("Path test")
        pdf_path = tmp_path / "from_path.pdf"
        pdf_path.write_bytes(pdf_bytes)
        media = {"filename": "from_path.pdf", "media_bytes": None, "media_path": str(pdf_path)}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert len(results) == 1

    def test_convert_empty_bytes(self):
        from vtscore.converters.document2image import Document2ImageMediaConverter

        media = {"filename": "empty.pdf", "media_bytes": b""}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert results == []

    def test_convert_no_data(self):
        from vtscore.converters.document2image import Document2ImageMediaConverter

        media = {"filename": "none.pdf"}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert results == []


# ===========================================================================
# Document2TextMediaConverter tests
# ===========================================================================


class TestDocument2TextMediaConverter:
    def test_source_and_target_types(self):
        from vtscore.converters.document2text import Document2TextMediaConverter

        c = Document2TextMediaConverter()
        assert c.source_type == "document"
        assert c.target_type == "text"

    def test_extract_text_from_pdf(self):
        from vtscore.converters.document2text import Document2TextMediaConverter

        pdf_bytes = _make_minimal_pdf("Hello Extracted Text")
        media = {"filename": "test.pdf", "media_bytes": pdf_bytes}
        c = Document2TextMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "test.txt"
        assert "Hello Extracted Text" in r["media_string"]
        assert r["duration"] == 0
        assert r["word_count"] >= 3
        assert r["character_count"] > 0

    def test_extract_text_two_pages(self):
        from vtscore.converters.document2text import Document2TextMediaConverter

        pdf_bytes = _make_two_page_pdf()
        media = {"filename": "multi.pdf", "media_bytes": pdf_bytes}
        c = Document2TextMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        text = results[0]["media_string"]
        assert "Page one content" in text
        assert "Page two content" in text

    def test_extract_text_from_path(self, tmp_path):
        from vtscore.converters.document2text import Document2TextMediaConverter

        pdf_bytes = _make_minimal_pdf("Path text test")
        pdf_path = tmp_path / "pathtest.pdf"
        pdf_path.write_bytes(pdf_bytes)
        media = {"filename": "pathtest.pdf", "media_bytes": None, "media_path": str(pdf_path)}
        c = Document2TextMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        assert "Path text test" in results[0]["media_string"]

    def test_empty_bytes(self):
        from vtscore.converters.document2text import Document2TextMediaConverter

        media = {"filename": "empty.pdf", "media_bytes": b""}
        c = Document2TextMediaConverter()
        assert c.convert(media) == []


# ===========================================================================
# Video2ImageMediaConverter tests
# ===========================================================================


class TestVideo2ImageMediaConverter:
    def test_source_and_target_types(self):
        from vtscore.converters.video2image import Video2ImageMediaConverter

        c = Video2ImageMediaConverter()
        assert c.source_type == "video"
        assert c.target_type == "image"

    def test_default_n_clips_is_field_default(self):
        """n_clips is a PluginField, not a constructor arg.  Its declared
        default flows in through get_param() when the caller passes no
        params."""
        from vtscore.converters.video2image import Video2ImageMediaConverter

        c = Video2ImageMediaConverter()
        n_clips_field = next(f for f in c.fields if f.key == "n_clips")
        assert n_clips_field.default == "10"
        assert c.get_param({}, "n_clips") == "10"

    def test_custom_n_clips_via_params(self):
        from vtscore.converters.video2image import Video2ImageMediaConverter

        c = Video2ImageMediaConverter()
        assert c.get_param({"n_clips": "3"}, "n_clips") == "3"

    def test_convert_video_to_images(self):
        from vtscore.converters.video2image import Video2ImageMediaConverter

        video_bytes = _make_minimal_video(frames=30)
        media = {"filename": "test.mp4", "media_bytes": video_bytes}
        c = Video2ImageMediaConverter()
        results = c.convert(media, {"n_clips": "3"})
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["filename"] == f"test_clip_{i + 1}.png"
            assert isinstance(r["media_bytes"], bytes)
            assert r["media_bytes"][:4] == b"\x89PNG"
            assert r["duration"] == 0
            assert r["width"] == 64
            assert r["height"] == 64

    def test_convert_fewer_frames_than_clips(self):
        from vtscore.converters.video2image import Video2ImageMediaConverter

        video_bytes = _make_minimal_video(frames=3)
        media = {"filename": "short.mp4", "media_bytes": video_bytes}
        c = Video2ImageMediaConverter()
        results = c.convert(media, {"n_clips": "10"})
        # Should produce min(10, 3) = 3 clips
        assert len(results) <= 3

    def test_convert_from_path(self, tmp_path):
        from vtscore.converters.video2image import Video2ImageMediaConverter

        video_bytes = _make_minimal_video(frames=20)
        video_path = tmp_path / "from_path.mp4"
        video_path.write_bytes(video_bytes)
        media = {"filename": "from_path.mp4", "media_bytes": None, "media_path": str(video_path)}
        c = Video2ImageMediaConverter()
        results = c.convert(media, {"n_clips": "2"})
        assert len(results) == 2

    def test_convert_no_data(self):
        from vtscore.converters.video2image import Video2ImageMediaConverter

        media = {"filename": "none.mp4"}
        c = Video2ImageMediaConverter()
        assert c.convert(media) == []


# ===========================================================================
# Video2AudioMediaConverter tests
# ===========================================================================


class TestVideo2AudioMediaConverter:
    def test_source_and_target_types(self):
        from vtscore.converters.video2audio import Video2AudioMediaConverter

        c = Video2AudioMediaConverter()
        assert c.source_type == "video"
        assert c.target_type == "audio"

    def test_convert_no_data(self):
        from vtscore.converters.video2audio import Video2AudioMediaConverter

        media = {"filename": "none.mp4"}
        c = Video2AudioMediaConverter()
        assert c.convert(media) == []

    def test_convert_empty_bytes(self):
        from vtscore.converters.video2audio import Video2AudioMediaConverter

        media = {"filename": "empty.mp4", "media_bytes": b""}
        c = Video2AudioMediaConverter()
        assert c.convert(media) == []

    def test_convert_no_ffmpeg(self):
        """When ffmpeg is not found, should return empty list."""
        from vtscore.converters.video2audio import Video2AudioMediaConverter

        video_bytes = _make_minimal_video(frames=10)
        media = {"filename": "test.mp4", "media_bytes": video_bytes}
        c = Video2AudioMediaConverter()
        with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
            results = c.convert(media)
        assert results == []


# ===========================================================================
# Converters package import tests
# ===========================================================================


class TestConvertersPackage:
    def test_all_converters_importable(self):
        from vtscore.converters import (
            Document2ImageMediaConverter,
            Document2TextMediaConverter,
            MediaConverter,
            Video2AudioMediaConverter,
            Video2ImageMediaConverter,
        )

        assert MediaConverter is not None
        assert Document2ImageMediaConverter is not None
        assert Document2TextMediaConverter is not None
        assert Video2AudioMediaConverter is not None
        assert Video2ImageMediaConverter is not None

    def test_converter_inheritance(self):
        from vtscore.converters import (
            Document2ImageMediaConverter,
            Document2TextMediaConverter,
            MediaConverter,
            Video2AudioMediaConverter,
            Video2ImageMediaConverter,
        )

        assert issubclass(Document2ImageMediaConverter, MediaConverter)
        assert issubclass(Document2TextMediaConverter, MediaConverter)
        assert issubclass(Video2AudioMediaConverter, MediaConverter)
        assert issubclass(Video2ImageMediaConverter, MediaConverter)


# ===========================================================================
# Folder importer includes documents option
# ===========================================================================


class TestFolderImporterDocumentsOption:
    def test_documents_in_media_type_options(self):
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        importer = ServerFolderDatasetImporter()
        media_type_field = next(f for f in importer.fields if f.key == "media_type")
        assert "document" in media_type_field.options


# ===========================================================================
# Converter registry: list_converters_for_source
# ===========================================================================


class TestConverterRegistrySourceFilter:
    def test_list_converters_for_source_video(self):
        from vtscore.converters import list_converters_for_source

        results = list_converters_for_source("video")
        names = [c.name for c in results]
        assert "video2image" in names
        assert "video2audio" in names
        assert "document2image" not in names

    def test_list_converters_for_source_document(self):
        from vtscore.converters import list_converters_for_source

        results = list_converters_for_source("document")
        names = [c.name for c in results]
        assert "document2image" in names
        assert "document2text" in names
        assert "video2image" not in names

    def test_list_converters_for_source_nonexistent(self):
        from vtscore.converters import list_converters_for_source

        results = list_converters_for_source("nonexistent_type")
        assert results == []


# ===========================================================================
# /api/converters endpoint — source filter
# ===========================================================================


class TestConvertersAPISourceFilter:
    def test_converters_no_filter(self, client):
        resp = client.get("/api/converters")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "converters" in data
        assert len(data["converters"]) >= 4

    def test_converters_filter_by_target(self, client):
        resp = client.get("/api/converters?target=image")
        data = resp.get_json()
        names = [c["name"] for c in data["converters"]]
        assert "document2image" in names
        assert "video2image" in names
        assert "document2text" not in names

    def test_converters_filter_by_source(self, client):
        resp = client.get("/api/converters?source=video")
        data = resp.get_json()
        names = [c["name"] for c in data["converters"]]
        assert "video2image" in names
        assert "video2audio" in names
        assert "document2image" not in names

    def test_converters_filter_by_source_folder_name(self, client):
        resp = client.get("/api/converters?source=video")
        data = resp.get_json()
        names = [c["name"] for c in data["converters"]]
        assert "video2image" in names

    def test_converters_filter_by_source_no_results(self, client):
        resp = client.get("/api/converters?source=text")
        data = resp.get_json()
        assert data["converters"] == []


# ===========================================================================
# Demo list includes available converters
# ===========================================================================


class TestDemoListConverters:
    def test_demo_list_includes_converters(self, client):
        resp = client.get("/api/dataset/demo-list")
        assert resp.status_code == 200
        data = resp.get_json()
        demos = data["datasets"]
        assert len(demos) > 0
        for demo in demos:
            assert "available_converters" in demo
            assert isinstance(demo["available_converters"], list)

    def test_demo_list_video_demo_has_video_converters(self, client):
        """Video demos should list video2image and video2audio converters."""
        resp = client.get("/api/dataset/demo-list")
        demos = resp.get_json()["datasets"]
        video_demos = [d for d in demos if d["media_type"] == "video"]
        if not video_demos:
            pytest.skip("No video demo datasets registered")
        demo = video_demos[0]
        conv_names = [c["name"] for c in demo["available_converters"]]
        assert "video2image" in conv_names
        assert "video2audio" in conv_names

    def test_demo_list_audio_demo_has_audio2image(self, client):
        """Audio demos should list the audio2image (spectrogram) converter."""
        resp = client.get("/api/dataset/demo-list")
        demos = resp.get_json()["datasets"]
        audio_demos = [d for d in demos if d["media_type"] == "audio"]
        if not audio_demos:
            pytest.skip("No audio demo datasets registered")
        demo = audio_demos[0]
        conv_names = [c["name"] for c in demo["available_converters"]]
        assert "audio2image" in conv_names


# ===========================================================================
# Importer to_dict includes available_converters_by_media_type
# ===========================================================================


class TestImporterConverterMetadata:
    def test_folder_importer_to_dict_has_converters(self):
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        importer = ServerFolderDatasetImporter()
        d = importer.to_dict()
        assert "available_converters_by_media_type" in d
        by_mt = d["available_converters_by_media_type"]
        assert isinstance(by_mt, dict)
        # image type should have document2image and video2image
        if "image" in by_mt:
            names = [c["name"] for c in by_mt["image"]]
            assert "document2image" in names
            assert "video2image" in names

    def test_http_archive_importer_to_dict_has_converters(self):
        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        importer = HttpArchiveDatasetImporter()
        d = importer.to_dict()
        assert "available_converters_by_media_type" in d
        by_mt = d["available_converters_by_media_type"]
        # audio type should have video2audio
        if "audio" in by_mt:
            names = [c["name"] for c in by_mt["audio"]]
            assert "video2audio" in names

    def test_importers_endpoint_includes_converters(self, client):
        resp = client.get("/api/dataset/importers")
        assert resp.status_code == 200
        importers = resp.get_json()["importers"]
        folder_imp = next((i for i in importers if i["name"] == "server_folder"), None)
        if folder_imp:
            assert "available_converters_by_media_type" in folder_imp


# ===========================================================================
# Demo dataset loading with converter
# ===========================================================================


class TestLoadDemoWithConverter:
    def test_load_demo_endpoint_accepts_converter(self, client):
        """The load-demo endpoint should accept a converter parameter."""
        resp = client.post(
            "/api/dataset/load-demo",
            json={"name": "nonexistent_demo", "converter": "video2image"},
        )
        # Should fail with "Invalid dataset name", not with a parameter error.
        # flask-smorest error envelope: ``message`` (not ``error``).
        assert resp.status_code == 400
        assert "Invalid dataset" in resp.get_json()["message"]

    def test_apply_converter_to_demo_unknown_converter(self):
        """_apply_converter_to_demo should raise for unknown converters."""
        from vtscore.datasets.loader import _apply_converter_to_demo

        with pytest.raises(ValueError, match="Unknown converter"):
            _apply_converter_to_demo(
                converter_name="nonexistent_converter",
                dataset_name="test",
                medias={},
            )

    def test_apply_converter_to_demo_empty_medias(self):
        """_apply_converter_to_demo with empty medias should produce empty output."""
        from vtscore.datasets.loader import _apply_converter_to_demo

        medias: dict = {}
        _apply_converter_to_demo(
            converter_name="document2image",
            dataset_name="test",
            medias=medias,
        )
        assert medias == {}

    def test_apply_converter_to_demo_converts_documents(self):
        """_apply_converter_to_demo should convert document medias to images."""
        from vtscore.datasets.loader import _apply_converter_to_demo

        pdf_bytes = _make_minimal_pdf("Convert me")
        medias = {
            1: {
                "id": 1,
                "type": "document",
                "filename": "test.pdf",
                "media_bytes": pdf_bytes,
                "media_path": "",
                "category": "test_cat",
            }
        }
        _apply_converter_to_demo(
            converter_name="document2image",
            dataset_name="test_demo",
            medias=medias,
        )
        # Should have converted the document to image(s)
        assert len(medias) >= 1
        for m in medias.values():
            assert m["type"] == "image"
            assert m["origin"]["importer"] == "converter"
            assert m["origin"]["params"]["converter"] == "document2image"
            assert m["origin"]["params"]["parent_importer"] == "demo"
            assert m["origin"]["params"]["parent_demo"] == "test_demo"
            assert "test.pdf" in m["origin"]["params"]["source_file"]
            # Category should be preserved from source
            assert m["category"] == "test_cat"

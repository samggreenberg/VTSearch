"""Tests for the Document media type and MediaConverter framework."""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vtsearch.converters.base import MediaConverter
from vtsearch.media.document.media_type import DocumentMediaType


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


def _make_minimal_wav(duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Build a minimal WAV file (mono, 16-bit PCM)."""
    n_samples = int(duration_s * sample_rate)
    samples = np.zeros(n_samples, dtype=np.int16)
    buf = io.BytesIO()
    import wave

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
        assert mt.icon == "📑"

    def test_file_extensions(self):
        mt = DocumentMediaType()
        exts = mt.file_extensions
        assert "*.pdf" in exts
        assert "*.doc" in exts
        assert "*.ppt" in exts

    def test_folder_import_name(self):
        mt = DocumentMediaType()
        assert mt.folder_import_name == "documents"

    def test_loops_false(self):
        mt = DocumentMediaType()
        assert mt.loops is False

    def test_no_demo_datasets(self):
        mt = DocumentMediaType()
        assert mt.demo_datasets == []

    def test_embed_media_returns_none(self, tmp_path):
        mt = DocumentMediaType()
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(_make_minimal_pdf())
        assert mt.embed_media(pdf_path) is None

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
        from vtsearch.media import get

        mt = get("document")
        assert mt.type_id == "document"

    def test_get_by_extension_pdf(self):
        from vtsearch.media import get_by_extension

        mt = get_by_extension(".pdf")
        assert mt is not None
        assert mt.type_id == "document"

    def test_get_by_extension_doc(self):
        from vtsearch.media import get_by_extension

        mt = get_by_extension(".doc")
        assert mt is not None
        assert mt.type_id == "document"

    def test_get_by_extension_ppt(self):
        from vtsearch.media import get_by_extension

        mt = get_by_extension(".ppt")
        assert mt is not None
        assert mt.type_id == "document"

    def test_get_by_folder_name(self):
        from vtsearch.media import get_by_folder_name

        mt = get_by_folder_name("documents")
        assert mt.type_id == "document"


# ===========================================================================
# MediaConverter base class tests
# ===========================================================================


class TestMediaConverterAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MediaConverter()

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
        from vtsearch.converters.document2image import Document2ImageMediaConverter

        c = Document2ImageMediaConverter()
        assert c.source_type == "document"
        assert c.target_type == "image"

    def test_convert_single_page_pdf(self):
        from vtsearch.converters.document2image import Document2ImageMediaConverter

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
        from vtsearch.converters.document2image import Document2ImageMediaConverter

        pdf_bytes = _make_two_page_pdf()
        media = {"filename": "multi.pdf", "media_bytes": pdf_bytes}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert len(results) == 2
        assert results[0]["filename"] == "multi_page_1.png"
        assert results[1]["filename"] == "multi_page_2.png"

    def test_convert_from_path(self, tmp_path):
        from vtsearch.converters.document2image import Document2ImageMediaConverter

        pdf_bytes = _make_minimal_pdf("Path test")
        pdf_path = tmp_path / "from_path.pdf"
        pdf_path.write_bytes(pdf_bytes)
        media = {"filename": "from_path.pdf", "media_bytes": None, "media_path": str(pdf_path)}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert len(results) == 1

    def test_convert_empty_bytes(self):
        from vtsearch.converters.document2image import Document2ImageMediaConverter

        media = {"filename": "empty.pdf", "media_bytes": b""}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert results == []

    def test_convert_no_data(self):
        from vtsearch.converters.document2image import Document2ImageMediaConverter

        media = {"filename": "none.pdf"}
        c = Document2ImageMediaConverter()
        results = c.convert(media)
        assert results == []


# ===========================================================================
# Document2TextMediaConverter tests
# ===========================================================================


class TestDocument2TextMediaConverter:
    def test_source_and_target_types(self):
        from vtsearch.converters.document2text import Document2TextMediaConverter

        c = Document2TextMediaConverter()
        assert c.source_type == "document"
        assert c.target_type == "paragraph"

    def test_extract_text_from_pdf(self):
        from vtsearch.converters.document2text import Document2TextMediaConverter

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
        from vtsearch.converters.document2text import Document2TextMediaConverter

        pdf_bytes = _make_two_page_pdf()
        media = {"filename": "multi.pdf", "media_bytes": pdf_bytes}
        c = Document2TextMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        text = results[0]["media_string"]
        assert "Page one content" in text
        assert "Page two content" in text

    def test_extract_text_from_path(self, tmp_path):
        from vtsearch.converters.document2text import Document2TextMediaConverter

        pdf_bytes = _make_minimal_pdf("Path text test")
        pdf_path = tmp_path / "pathtest.pdf"
        pdf_path.write_bytes(pdf_bytes)
        media = {"filename": "pathtest.pdf", "media_bytes": None, "media_path": str(pdf_path)}
        c = Document2TextMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        assert "Path text test" in results[0]["media_string"]

    def test_empty_bytes(self):
        from vtsearch.converters.document2text import Document2TextMediaConverter

        media = {"filename": "empty.pdf", "media_bytes": b""}
        c = Document2TextMediaConverter()
        assert c.convert(media) == []


# ===========================================================================
# Video2ImageMediaConverter tests
# ===========================================================================


class TestVideo2ImageMediaConverter:
    def test_source_and_target_types(self):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        c = Video2ImageMediaConverter(n_clips=5)
        assert c.source_type == "video"
        assert c.target_type == "image"

    def test_default_n_clips(self):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        c = Video2ImageMediaConverter()
        assert c.n_clips == 10

    def test_custom_n_clips(self):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        c = Video2ImageMediaConverter(n_clips=3)
        assert c.n_clips == 3

    def test_convert_video_to_images(self):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        video_bytes = _make_minimal_video(frames=30)
        media = {"filename": "test.mp4", "media_bytes": video_bytes}
        c = Video2ImageMediaConverter(n_clips=3)
        results = c.convert(media)
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["filename"] == f"test_clip_{i + 1}.png"
            assert isinstance(r["media_bytes"], bytes)
            assert r["media_bytes"][:4] == b"\x89PNG"
            assert r["duration"] == 0
            assert r["width"] == 64
            assert r["height"] == 64

    def test_convert_fewer_frames_than_clips(self):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        video_bytes = _make_minimal_video(frames=3)
        media = {"filename": "short.mp4", "media_bytes": video_bytes}
        c = Video2ImageMediaConverter(n_clips=10)
        results = c.convert(media)
        # Should produce min(10, 3) = 3 clips
        assert len(results) <= 3

    def test_convert_from_path(self, tmp_path):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        video_bytes = _make_minimal_video(frames=20)
        video_path = tmp_path / "from_path.mp4"
        video_path.write_bytes(video_bytes)
        media = {"filename": "from_path.mp4", "media_bytes": None, "media_path": str(video_path)}
        c = Video2ImageMediaConverter(n_clips=2)
        results = c.convert(media)
        assert len(results) == 2

    def test_convert_no_data(self):
        from vtsearch.converters.video2image import Video2ImageMediaConverter

        media = {"filename": "none.mp4"}
        c = Video2ImageMediaConverter()
        assert c.convert(media) == []


# ===========================================================================
# Video2AudioMediaConverter tests
# ===========================================================================


class TestVideo2AudioMediaConverter:
    def test_source_and_target_types(self):
        from vtsearch.converters.video2audio import Video2AudioMediaConverter

        c = Video2AudioMediaConverter()
        assert c.source_type == "video"
        assert c.target_type == "audio"

    def test_convert_no_data(self):
        from vtsearch.converters.video2audio import Video2AudioMediaConverter

        media = {"filename": "none.mp4"}
        c = Video2AudioMediaConverter()
        assert c.convert(media) == []

    def test_convert_empty_bytes(self):
        from vtsearch.converters.video2audio import Video2AudioMediaConverter

        media = {"filename": "empty.mp4", "media_bytes": b""}
        c = Video2AudioMediaConverter()
        assert c.convert(media) == []

    def test_convert_no_ffmpeg(self):
        """When ffmpeg is not found, should return empty list."""
        from vtsearch.converters.video2audio import Video2AudioMediaConverter

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
        from vtsearch.converters import (
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
        from vtsearch.converters import (
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
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        importer = FolderDatasetImporter()
        media_type_field = next(f for f in importer.fields if f.key == "media_type")
        assert "documents" in media_type_field.options

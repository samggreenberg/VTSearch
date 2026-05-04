"""Tests for PDF-to-image import support.

These tests verify:
- render_pdf_pages converts PDF pages into PIL Images with correct naming
- Folder importer picks up PDFs when media_type is "image"
- PDF-derived medias have origin {"importer": "pdf", ...}
- PDF-only folders work (no regular image files present)
- PDFs are ignored when media_type is not "image"
- Thin mode stores no media_bytes for PDF pages
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np


def _create_test_pdf(path, num_pages=2):
    """Create a minimal PDF at *path* with *num_pages* coloured pages."""
    import fitz

    doc = fitz.open()
    colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    for i in range(num_pages):
        page = doc.new_page(width=100, height=100)
        colour = colours[i % len(colours)]
        page.draw_rect(fitz.Rect(0, 0, 100, 100), color=colour, fill=colour)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# render_pdf_pages
# ---------------------------------------------------------------------------


class TestRenderPdfPages:
    def test_returns_correct_number_of_pages(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=3)

        from vtsearch.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert len(pages) == 3

    def test_page_names_follow_pattern(self, tmp_path):
        pdf = tmp_path / "mydoc.pdf"
        _create_test_pdf(pdf, num_pages=2)

        from vtsearch.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert pages[0][0] == "mydoc.pdf-1"
        assert pages[1][0] == "mydoc.pdf-2"

    def test_returns_pil_images(self, tmp_path):
        from PIL import Image

        pdf = tmp_path / "test.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtsearch.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert isinstance(pages[0][1], Image.Image)

    def test_images_are_rgb(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtsearch.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert pages[0][1].mode == "RGB"

    def test_custom_dpi(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtsearch.datasets.pdf import render_pdf_pages

        pages_low = render_pdf_pages(pdf, dpi=72)
        pages_high = render_pdf_pages(pdf, dpi=300)
        # Higher DPI should produce larger images
        assert pages_high[0][1].width > pages_low[0][1].width
        assert pages_high[0][1].height > pages_low[0][1].height

    def test_single_page_pdf(self, tmp_path):
        pdf = tmp_path / "single.pdf"
        _create_test_pdf(pdf, num_pages=1)

        from vtsearch.datasets.pdf import render_pdf_pages

        pages = render_pdf_pages(pdf)
        assert len(pages) == 1
        assert pages[0][0] == "single.pdf-1"


# ---------------------------------------------------------------------------
# Folder importer – PDF integration
# ---------------------------------------------------------------------------


class TestFolderImporterPdf:
    def _make_fake_image_media_type(self):
        mt = mock.MagicMock()
        mt.type_id = "image"
        mt.folder_import_name = "image"
        mt.file_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
        mt.load_media_data.return_value = {"media_bytes": b"fake", "duration": 0, "width": 100, "height": 100}
        return mt

    def _make_fake_embedder(self):
        emb = mock.MagicMock()
        emb.name = "clip"
        emb.media_type_id = "image"
        emb._model = True
        emb.embed_media.return_value = np.zeros(768)
        emb.embed_pil_image.return_value = np.zeros(768)
        emb.embed_media_bulk.side_effect = lambda medias: [emb.embed_media(m) for m in medias]
        return emb

    def _patch_media_registry(self, mt, emb):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(mock.patch("vtsearch.media.get_by_folder_name", return_value=mt))
        stack.enter_context(mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]))
        return stack

    def test_pdf_pages_added_to_medias(self, tmp_path):
        """PDFs in an image folder should produce per-page medias."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=2)
        # Also create a regular image so load_dataset_from_folder doesn't raise
        (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        # 1 regular image + 2 PDF pages = 3 total
        assert len(medias) == 3

    def test_pdf_origin_is_pdf_importer(self, tmp_path):
        """PDF-derived medias should have origin importer='pdf'."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "report.pdf"
        _create_test_pdf(pdf, num_pages=1)
        (tmp_path / "photo.png").write_bytes(b"\x89PNG" + b"\x00" * 100)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        pdf_medias = [m for m in medias.values() if m["origin"] and m["origin"]["importer"] == "pdf"]
        assert len(pdf_medias) == 1
        assert pdf_medias[0]["origin"]["params"]["path"] == str(pdf)

    def test_pdf_page_filenames(self, tmp_path):
        """PDF page filenames should follow the name-N pattern."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "slides.pdf"
        _create_test_pdf(pdf, num_pages=3)
        (tmp_path / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        pdf_names = sorted(m["filename"] for m in medias.values() if m["filename"].startswith("slides.pdf-"))
        assert pdf_names == ["slides.pdf-1", "slides.pdf-2", "slides.pdf-3"]

    def test_pdf_only_folder(self, tmp_path):
        """A folder with only PDFs (no regular images) should still work."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "only.pdf"
        _create_test_pdf(pdf, num_pages=2)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        assert len(medias) == 2

    def test_pdf_ignored_for_non_image_types(self, tmp_path):
        """PDFs should not be processed when media_type is not 'images'."""
        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=1)

        # _load_pdf_images is only called when media_type == "image",
        # so verify the folder importer run method doesn't call it for sounds
        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.folder_import_name = "audio"
        mt.file_extensions = ["*.wav"]
        mt.embed_media.return_value = np.zeros(512)
        mt.load_media_data.return_value = {"duration": 1.0}

        # Write a WAV so load_dataset_from_folder succeeds
        import io
        import struct
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<" + "h" * 100, *([0] * 100)))
        (tmp_path / "tone.wav").write_bytes(buf.getvalue())

        from vtsearch.datasets.importers.server_folder import IMPORTER

        medias: dict = {}
        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            IMPORTER.run({"path": str(tmp_path), "media_type": "audio"}, medias)

        # Only the WAV should be loaded, not the PDF
        assert len(medias) == 1
        assert medias[1]["type"] == "audio"

    def test_pdf_thin_mode_no_media_bytes(self, tmp_path):
        """In thin mode, PDF-derived medias should have media_bytes=None."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=1)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias, thin=True)

        pdf_medias = [m for m in medias.values() if m["filename"].endswith(".pdf-1")]
        assert len(pdf_medias) == 1
        assert pdf_medias[0]["media_bytes"] is None

    def test_pdf_media_has_image_type(self, tmp_path):
        """PDF-derived medias should have type='image'."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=1)
        (tmp_path / "img.png").write_bytes(b"\x89PNG" + b"\x00" * 100)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        pdf_medias = [m for m in medias.values() if m["filename"].startswith("doc.pdf-")]
        assert len(pdf_medias) >= 1
        assert all(m["type"] == "image" for m in pdf_medias)

    def test_pdf_media_has_width_height(self, tmp_path):
        """PDF-derived medias should have non-None width and height."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=1)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        pdf_medias = [m for m in medias.values() if m["filename"].startswith("doc.pdf-")]
        assert len(pdf_medias) == 1
        assert pdf_medias[0]["width"] is not None
        assert pdf_medias[0]["height"] is not None
        assert pdf_medias[0]["width"] > 0
        assert pdf_medias[0]["height"] > 0

    def test_pdf_media_has_md5(self, tmp_path):
        """PDF-derived medias should have a valid MD5 hash."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        pdf = tmp_path / "doc.pdf"
        _create_test_pdf(pdf, num_pages=1)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        pdf_medias = [m for m in medias.values() if m["filename"].startswith("doc.pdf-")]
        assert len(pdf_medias) == 1
        assert len(pdf_medias[0]["md5"]) == 32

    def test_multiple_pdfs_in_folder(self, tmp_path):
        """Multiple PDFs should all be expanded."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        _create_test_pdf(tmp_path / "a.pdf", num_pages=2)
        _create_test_pdf(tmp_path / "b.pdf", num_pages=3)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        # 2 + 3 = 5 pages total
        assert len(medias) == 5

    def test_pdf_origin_name_matches_filename(self, tmp_path):
        """origin_name should equal the page filename."""
        from vtsearch.datasets.importers.server_folder import IMPORTER

        _create_test_pdf(tmp_path / "doc.pdf", num_pages=1)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with self._patch_media_registry(mt, emb):
            IMPORTER.run({"path": str(tmp_path), "media_type": "image"}, medias)

        m = list(medias.values())[0]
        assert m["origin_name"] == m["filename"]
        assert m["origin_name"] == "doc.pdf-1"


class TestPdfSymlinkDiscovery:
    """PDF scanning must follow symlinked directories."""

    def _make_fake_image_media_type(self):
        mt = mock.MagicMock()
        mt.type_id = "image"
        mt.folder_import_name = "image"
        mt.file_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
        mt.load_media_data.return_value = {"media_bytes": b"fake", "duration": 0, "width": 100, "height": 100}
        return mt

    def _make_fake_embedder(self):
        emb = mock.MagicMock()
        emb.name = "clip"
        emb.media_type_id = "image"
        emb._model = True
        emb.embed_media.return_value = np.zeros(768)
        emb.embed_pil_image.return_value = np.zeros(768)
        emb.embed_media_bulk.side_effect = lambda medias: [emb.embed_media(m) for m in medias]
        return emb

    def test_pdfs_in_symlinked_subdir_are_discovered(self, tmp_path):
        """PDFs inside a symlinked subdirectory should be found and rendered."""
        from contextlib import ExitStack

        from vtsearch.datasets.importers.server_folder import _load_pdf_images

        root = tmp_path / "root"
        root.mkdir()

        external = tmp_path / "external"
        external.mkdir()
        _create_test_pdf(external / "linked_doc.pdf", num_pages=2)

        (root / "linked").symlink_to(external)

        mt = self._make_fake_image_media_type()
        emb = self._make_fake_embedder()
        medias: dict = {}

        with ExitStack() as stack:
            stack.enter_context(mock.patch("vtsearch.media.get_by_folder_name", return_value=mt))
            stack.enter_context(mock.patch("vtsearch.media.get_embedder", return_value=emb))
            stack.enter_context(mock.patch("vtsearch.media.embedders_for_type", return_value=[emb]))
            _load_pdf_images(root, medias, embedder_name="clip")

        assert len(medias) == 2
        filenames = {m["filename"] for m in medias.values()}
        assert "linked/linked_doc.pdf-1" in filenames
        assert "linked/linked_doc.pdf-2" in filenames

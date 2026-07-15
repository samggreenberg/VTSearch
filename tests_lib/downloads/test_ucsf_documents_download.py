"""Tests for UCSF Industry Documents demo dataset download and load_demo_source integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ucsf_fixture(tmp_path: Path, categories: list[str] | None = None, docs_per_cat: int = 3) -> Path:
    """Pre-populate a ucsf_documents/ directory with tiny PDFs per category."""
    if categories is None:
        categories = ["Tobacco", "Food"]
    extract_dir = tmp_path / "ucsf_documents"
    for cat in categories:
        cat_dir = extract_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for i in range(docs_per_cat):
            # Write minimal non-empty files (real PDFs are not needed;
            # tests mock render_pdf_pages).
            (cat_dir / f"doc{i:04d}.pdf").write_bytes(b"%PDF-1.0 stub")
    return extract_dir


def _fake_solr_response(doc_ids: list[str]) -> dict:
    """Return a dict mimicking a Solr JSON response."""
    return {
        "response": {
            "numFound": len(doc_ids),
            "docs": [{"id": did} for did in doc_ids],
        }
    }


# ---------------------------------------------------------------------------
# download_ucsf_documents
# ---------------------------------------------------------------------------


class TestDownloadUcsfDocuments:
    def test_returns_extract_directory(self, tmp_path):
        """download_ucsf_documents returns the ucsf_documents/ directory."""
        from vtscore.datasets import downloader as dl_module

        solr_resp = _fake_solr_response(["abcd0001", "abcd0002"])
        mock_response = MagicMock()
        mock_response.json.return_value = solr_resp
        mock_response.raise_for_status = MagicMock()

        downloads = []

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"%PDF-1.0 stub")
            downloads.append(dest.name)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
            patch("requests.get", return_value=mock_response),
        ):
            result = dl_module.download_ucsf_documents(
                categories=["Tobacco"],
                docs_per_category=2,
                on_progress=lambda *a: None,
            )

        assert result.name == "ucsf_documents"
        assert result.exists()
        assert (result / "Tobacco").is_dir()
        assert len(list((result / "Tobacco").glob("*.pdf"))) == 2

    def test_caps_docs_per_category_when_solr_overreturns(self, tmp_path):
        """Only docs_per_category PDFs download even if Solr returns more.

        The live UCSF IDL Solr endpoint ignores the ``rows`` limit and returns
        up to 1000 ids per query, so the downloader must cap client-side or it
        would pull ~1000 PDFs per category instead of docs_per_category.
        """
        from vtscore.datasets import downloader as dl_module

        # Solr hands back 10 ids despite a rows=3 request.
        solr_resp = _fake_solr_response([f"abcd{i:04d}" for i in range(10)])
        mock_response = MagicMock()
        mock_response.json.return_value = solr_resp
        mock_response.raise_for_status = MagicMock()

        downloads = []

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"%PDF-1.0 stub")
            downloads.append(dest.name)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
            patch("requests.get", return_value=mock_response),
        ):
            result = dl_module.download_ucsf_documents(
                categories=["Tobacco"],
                docs_per_category=3,
                on_progress=lambda *a: None,
            )

        assert len(list((result / "Tobacco").glob("*.pdf"))) == 3

    def test_constructs_correct_download_url(self, tmp_path):
        """PDF download URLs follow the UCSF split-character scheme."""
        from vtscore.datasets import downloader as dl_module

        solr_resp = _fake_solr_response(["xyzw1234"])
        mock_response = MagicMock()
        mock_response.json.return_value = solr_resp
        mock_response.raise_for_status = MagicMock()

        captured_urls = []

        def fake_download(url, dest, size, cb):
            captured_urls.append(url)
            dest.write_bytes(b"%PDF-1.0 stub")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
            patch("requests.get", return_value=mock_response),
        ):
            dl_module.download_ucsf_documents(
                categories=["Drug"],
                docs_per_category=1,
                on_progress=lambda *a: None,
            )

        assert len(captured_urls) == 1
        assert "/x/y/z/w/xyzw1234/xyzw1234.pdf" in captured_urls[0]

    def test_cached_pdfs_skip_download(self, tmp_path):
        """If category directories already contain PDFs, no download occurs."""
        from vtscore.datasets import downloader as dl_module

        _make_ucsf_fixture(tmp_path, ["Tobacco", "Food"], docs_per_cat=3)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_ucsf_documents(
                categories=["Tobacco", "Food"],
                on_progress=lambda *a: None,
            )

        assert not download_called, "download should be skipped when cache exists"
        assert result.exists()

    def test_quotes_multi_word_industry(self, tmp_path):
        """Multi-word industry names are quoted in the Solr query."""
        from vtscore.datasets import downloader as dl_module

        solr_resp = _fake_solr_response(["ffff0001"])
        mock_response = MagicMock()
        mock_response.json.return_value = solr_resp
        mock_response.raise_for_status = MagicMock()

        captured_params = []

        def fake_get(url, params=None, timeout=None):
            captured_params.append(params)
            return mock_response

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"%PDF-1.0 stub")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
            patch("requests.get", fake_get),
        ):
            dl_module.download_ucsf_documents(
                categories=["Fossil Fuel"],
                docs_per_category=1,
                on_progress=lambda *a: None,
            )

        assert captured_params
        query = captured_params[0]["q"]
        assert '"Fossil Fuel"' in query

    def test_skips_short_document_ids(self, tmp_path):
        """Document IDs shorter than 4 characters are skipped."""
        from vtscore.datasets import downloader as dl_module

        solr_resp = _fake_solr_response(["ab", "abcd0001"])
        mock_response = MagicMock()
        mock_response.json.return_value = solr_resp
        mock_response.raise_for_status = MagicMock()

        downloads = []

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"%PDF-1.0 stub")
            downloads.append(url)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
            patch("requests.get", return_value=mock_response),
        ):
            result = dl_module.download_ucsf_documents(
                categories=["Tobacco"],
                docs_per_category=5,
                on_progress=lambda *a: None,
            )

        # Only the valid 4+ char ID should be downloaded.  The stubbed
        # downloader sees a temp path (the atomic temp+rename wrapper), so
        # assert on the URL and the published file instead of the dest name.
        assert len(downloads) == 1
        assert downloads[0].endswith("/abcd0001.pdf")
        published = list(result.rglob("abcd0001.pdf"))
        assert published, "the downloaded PDF must be published at its final path"

    def test_handles_api_failure_gracefully(self, tmp_path):
        """If the Solr API request fails, the category is skipped."""
        from vtscore.datasets import downloader as dl_module

        def fail_get(*a, **kw):
            raise ConnectionError("network down")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("requests.get", fail_get),
        ):
            result = dl_module.download_ucsf_documents(
                categories=["Tobacco"],
                docs_per_category=5,
                on_progress=lambda *a: None,
            )

        # Should still return the directory (possibly empty).
        assert result.name == "ucsf_documents"


# ---------------------------------------------------------------------------
# load_demo_source: ucsf_documents branch (now on the Document type)
# ---------------------------------------------------------------------------


class TestLoadDemoSourceUcsfDocuments:
    """DocumentMediaType.load_demo_source with source='ucsf_documents'.

    The demo now lives on the Document type and produces **raw-PDF document**
    clips (one per PDF); rendering PDF pages to images is the ``document2image``
    converter's job at load time, not this method's.
    """

    def _make_document_media_type(self):
        from vtscore.media.document.media_type import DocumentMediaType

        return DocumentMediaType()

    def test_ucsf_documents_populates_clips(self, tmp_path):
        """load_demo_source with source='ucsf_documents' fills the clips dict."""
        from vtscore.datasets import downloader as dl_module

        docs_dir = _make_ucsf_fixture(tmp_path, ["Tobacco", "Food"], docs_per_cat=2)

        mt = self._make_document_media_type()
        clips: dict = {}

        with patch.object(dl_module, "download_ucsf_documents", return_value=docs_dir):
            mt.load_demo_source(
                source="ucsf_documents",
                categories=["Tobacco", "Food"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
            )

        assert len(clips) == 4  # 2 docs × 2 categories
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"Tobacco", "Food"}

    def test_clips_have_expected_fields(self, tmp_path):
        """Each clip is a raw-PDF document media, not a rendered image."""
        from vtscore.datasets import downloader as dl_module

        docs_dir = _make_ucsf_fixture(tmp_path, ["Drug"], docs_per_cat=1)

        mt = self._make_document_media_type()
        clips: dict = {}

        with patch.object(dl_module, "download_ucsf_documents", return_value=docs_dir):
            mt.load_demo_source(
                source="ucsf_documents",
                categories=["Drug"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
            )

        assert len(clips) == 1
        clip = clips[1]
        assert clip["media_type"] == "document"
        assert clip["category"] == "Drug"
        assert clip["media_bytes"] == b"%PDF-1.0 stub"
        assert clip["filename"].endswith(".pdf")
        # Documents are not embedded here.
        assert clip["embeddings"] == {}

    def test_slice_is_applied(self, tmp_path):
        """slice_start/slice_end limits documents per category."""
        from vtscore.datasets import downloader as dl_module

        docs_dir = _make_ucsf_fixture(tmp_path, ["Tobacco"], docs_per_cat=10)

        mt = self._make_document_media_type()
        clips: dict = {}

        with patch.object(dl_module, "download_ucsf_documents", return_value=docs_dir):
            mt.load_demo_source(
                source="ucsf_documents",
                categories=["Tobacco"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
            )

        # Only docs[2:5] = 3 documents.
        assert len(clips) == 3

    def test_unknown_source_raises(self, tmp_path):
        """An unrecognised demo source is rejected."""
        import pytest

        mt = self._make_document_media_type()
        with pytest.raises(ValueError):
            mt.load_demo_source(
                source="not_a_source",
                categories=[],
                slice_start=0,
                slice_end=None,
                clips={},
                on_progress=lambda *a: None,
            )

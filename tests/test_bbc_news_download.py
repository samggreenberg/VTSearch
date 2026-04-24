"""Tests for BBC News dataset download and load_demo_source integration."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bbc_zip(tmp_path: Path, top_folder: str = "") -> Path:
    """Create a minimal BBC News zip fixture with 3 articles per category."""
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    zip_path = tmp_path / "bbc-fulltext.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        for cat in categories:
            for i in range(1, 4):
                article_text = f"BBC {cat} article number {i}. This is sample text."
                # Optionally nest inside a top-level folder.
                arc_name = f"{top_folder}/{cat}/{i:03d}.txt" if top_folder else f"{cat}/{i:03d}.txt"
                zf.writestr(arc_name, article_text)

    return zip_path


# ---------------------------------------------------------------------------
# _find_bbc_root
# ---------------------------------------------------------------------------


class TestFindBbcRoot:
    def test_flat_structure(self, tmp_path):
        """Category directories sit directly inside the root."""
        from vtsearch.datasets.downloader import _find_bbc_root

        for cat in ("business", "sport", "tech"):
            (tmp_path / cat).mkdir()

        result = _find_bbc_root(tmp_path)
        assert result == tmp_path

    def test_nested_structure(self, tmp_path):
        """Category directories are one level deeper (zip has a top-level folder)."""
        from vtsearch.datasets.downloader import _find_bbc_root

        inner = tmp_path / "bbc"
        inner.mkdir()
        for cat in ("business", "sport", "tech"):
            (inner / cat).mkdir()

        result = _find_bbc_root(tmp_path)
        assert result == inner

    def test_no_match_returns_none(self, tmp_path):
        from vtsearch.datasets.downloader import _find_bbc_root

        (tmp_path / "unrelated").mkdir()
        assert _find_bbc_root(tmp_path) is None


# ---------------------------------------------------------------------------
# download_bbc_news
# ---------------------------------------------------------------------------


class TestDownloadBbcNews:
    def test_returns_articles_by_category(self, tmp_path):
        """download_bbc_news returns a dict of category -> list[str] from a zip."""
        from vtsearch.datasets import downloader as dl_module

        zip_path = _make_bbc_zip(tmp_path)

        progress_calls = []

        def fake_progress(status, msg, cur, tot):
            progress_calls.append((status, msg))

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core, "download_file_with_progress", lambda url, dest, size, cb: zip_path.rename(dest)
            ),
        ):
            result = dl_module.download_bbc_news(on_progress=fake_progress)

        assert set(result.keys()) == {"business", "entertainment", "politics", "sport", "tech"}
        for articles in result.values():
            assert len(articles) == 3
            assert all(isinstance(a, str) and a for a in articles)

    def test_nested_zip_structure(self, tmp_path):
        """download_bbc_news handles a zip with a top-level 'bbc/' folder."""
        from vtsearch.datasets import downloader as dl_module

        zip_path = _make_bbc_zip(tmp_path, top_folder="bbc")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core, "download_file_with_progress", lambda url, dest, size, cb: zip_path.rename(dest)
            ),
        ):
            result = dl_module.download_bbc_news(on_progress=lambda *a: None)

        assert "business" in result
        assert len(result["business"]) == 3

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract directory already exists, no download is triggered."""
        from vtsearch.datasets import downloader as dl_module

        # Pre-create the extract directory with one category.
        extract_dir = tmp_path / "bbc-fulltext"
        sport_dir = extract_dir / "sport"
        sport_dir.mkdir(parents=True)
        (sport_dir / "001.txt").write_text("A sport article.")

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_bbc_news(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert "sport" in result


# ---------------------------------------------------------------------------
# load_demo_source — bbc_news branch
# ---------------------------------------------------------------------------


class TestLoadDemoSourceBbcNews:
    """TextMediaType.load_demo_source with source='bbc_news'."""

    def _make_text_media_type(self):
        from vtsearch.media.text.media_type import TextMediaType

        mt = TextMediaType()
        stub_model = MagicMock()
        stub_model.encode.return_value = [0.1] * 768
        mt._model = stub_model
        return mt

    def _make_fake_embedder(self):
        import numpy as np

        emb = MagicMock()
        emb.name = "e5"
        emb.media_type_id = "text"
        emb._model = True
        emb._on_progress = lambda *a: None
        emb.embed_text_passage.return_value = np.zeros(768)
        return emb

    def test_bbc_news_source_populates_clips(self, tmp_path):
        """load_demo_source with source='bbc_news' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module

        fake_articles = {
            "business": ["Business article one.", "Business article two."],
            "sport": ["Sport article one.", "Sport article two."],
        }

        mt = self._make_text_media_type()
        emb = self._make_fake_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_bbc_news", return_value=fake_articles):
            mt.load_demo_source(
                source="bbc_news",
                categories=["business", "sport"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 4
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"business", "sport"}

    def test_bbc_news_slice_is_applied(self, tmp_path):
        """slice_start/slice_end limits articles per category."""
        from vtsearch.datasets import downloader as dl_module

        fake_articles = {
            "tech": [f"Tech article {i}." for i in range(10)],
        }

        mt = self._make_text_media_type()
        emb = self._make_fake_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_bbc_news", return_value=fake_articles):
            mt.load_demo_source(
                source="bbc_news",
                categories=["tech"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        # Only articles[2:5] = 3 articles.
        assert len(clips) == 3

    def test_unsupported_source_still_raises(self):
        """Non-existent sources still raise ValueError."""
        from vtsearch.media.text.media_type import TextMediaType

        mt = TextMediaType()
        with pytest.raises(ValueError, match="Unsupported text source"):
            mt.load_demo_source(
                source="unknown_source",
                categories=[],
                slice_start=0,
                slice_end=10,
                clips={},
                on_progress=lambda *a: None,
            )

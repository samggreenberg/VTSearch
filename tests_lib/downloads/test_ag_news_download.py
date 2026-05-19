"""Tests for AG News dataset download and load_demo_source integration."""

from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ag_news_csv(tmp_path: Path) -> Path:
    """Create a minimal AG News CSV fixture with 3 articles per category."""
    csv_path = tmp_path / "ag_news_train.csv"
    # AG News CSV format: "class_index","title","description"
    # 1=World, 2=Sports, 3=Business, 4=Sci/Tech
    lines = []
    for class_idx in range(1, 5):
        for i in range(1, 4):
            title = f"Title {class_idx}-{i}"
            desc = f"Description for class {class_idx} article {i}. This is sample text."
            lines.append(f'"{class_idx}","{title}","{desc}"')
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


# ---------------------------------------------------------------------------
# download_ag_news
# ---------------------------------------------------------------------------


class TestDownloadAgNews:
    def test_returns_articles_by_category(self, tmp_path):
        """download_ag_news returns a dict of category -> list[str] from a CSV."""
        from vtscore.datasets import downloader as dl_module

        csv_path = _make_ag_news_csv(tmp_path)

        progress_calls = []

        def fake_progress(status, msg, cur, tot):
            progress_calls.append((status, msg))

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: csv_path.rename(dest) if dest != csv_path else None,
            ),
        ):
            # Ensure the CSV is at the expected location.
            if not (tmp_path / "ag_news_train.csv").exists():
                _make_ag_news_csv(tmp_path)
            result = dl_module.download_ag_news(on_progress=fake_progress)

        assert set(result.keys()) == {"World", "Sports", "Business", "Sci/Tech"}
        for articles in result.values():
            assert len(articles) == 3
            assert all(isinstance(a, str) and a for a in articles)

    def test_articles_combine_title_and_description(self, tmp_path):
        """Each article text should contain both the title and description."""
        from vtscore.datasets import downloader as dl_module

        _make_ag_news_csv(tmp_path)

        with patch.object(dl_module.core, "DATA_DIR", tmp_path):
            result = dl_module.download_ag_news(on_progress=lambda *a: None)

        # Check that the first World article has both title and description.
        first_article = result["World"][0]
        assert "Title 1-1" in first_article
        assert "Description for class 1 article 1" in first_article

    def test_cached_csv_skips_download(self, tmp_path):
        """If the CSV already exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        _make_ag_news_csv(tmp_path)
        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_ag_news(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert "World" in result

    def test_skips_malformed_rows(self, tmp_path):
        """Rows with fewer than 3 columns are silently skipped."""
        from vtscore.datasets import downloader as dl_module

        csv_path = tmp_path / "ag_news_train.csv"
        csv_path.write_text(
            '"1","Good title","Good description"\n"bad row"\n"2","Another","Article"\n', encoding="utf-8"
        )

        with patch.object(dl_module.core, "DATA_DIR", tmp_path):
            result = dl_module.download_ag_news(on_progress=lambda *a: None)

        total = sum(len(v) for v in result.values())
        assert total == 2


# ---------------------------------------------------------------------------
# load_demo_source — ag_news branch
# ---------------------------------------------------------------------------


class TestLoadDemoSourceAgNews:
    """TextMediaType.load_demo_source with source='ag_news'."""

    def test_ag_news_source_populates_clips(self):
        """load_demo_source with source='ag_news' fills the clips dict."""
        from vtscore.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_articles = {
            "World": ["World article one.", "World article two."],
            "Sports": ["Sports article one.", "Sports article two."],
        }

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_ag_news", return_value=fake_articles):
            mt.load_demo_source(
                source="ag_news",
                categories=["World", "Sports"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 4
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"World", "Sports"}

    def test_ag_news_slice_is_applied(self):
        """slice_start/slice_end limits articles per category."""
        from vtscore.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_articles = {
            "Business": [f"Business article {i}." for i in range(10)],
        }

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_ag_news", return_value=fake_articles):
            mt.load_demo_source(
                source="ag_news",
                categories=["Business"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        # Only articles[2:5] = 3 articles.
        assert len(clips) == 3

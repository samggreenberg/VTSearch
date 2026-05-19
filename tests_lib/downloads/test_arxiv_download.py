"""Tests for arXiv abstracts download and load_demo_source integration."""

import json
from unittest.mock import MagicMock, patch


_ATOM_TEMPLATE = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>
"""

_ENTRY_TEMPLATE = """<entry>
  <title>{title}</title>
  <summary>{summary}</summary>
</entry>"""


def _make_atom_feed(entries: list[tuple[str, str]]) -> bytes:
    body = "\n".join(_ENTRY_TEMPLATE.format(title=t, summary=s) for t, s in entries)
    return _ATOM_TEMPLATE.format(entries=body).encode("utf-8")


# ---------------------------------------------------------------------------
# _parse_arxiv_feed
# ---------------------------------------------------------------------------


class TestParseArxivFeed:
    def test_extracts_title_and_summary(self):
        from vtsearch.datasets.downloader.text import _parse_arxiv_feed

        xml = _make_atom_feed([("My Paper", "A great abstract."), ("Other", "Another abstract.")])
        result = _parse_arxiv_feed(xml)

        assert result == ["My Paper A great abstract.", "Other Another abstract."]

    def test_collapses_whitespace(self):
        from vtsearch.datasets.downloader.text import _parse_arxiv_feed

        xml = _make_atom_feed([("Wrapped\n  Title", "Line one\n  line two\n  line three")])
        [text] = _parse_arxiv_feed(xml)

        assert "  " not in text
        assert "\n" not in text

    def test_empty_feed_returns_empty(self):
        from vtsearch.datasets.downloader.text import _parse_arxiv_feed

        assert _parse_arxiv_feed(b"<feed xmlns='http://www.w3.org/2005/Atom'/>") == []

    def test_malformed_xml_returns_empty(self):
        from vtsearch.datasets.downloader.text import _parse_arxiv_feed

        assert _parse_arxiv_feed(b"not xml at all") == []


# ---------------------------------------------------------------------------
# download_arxiv_abstracts
# ---------------------------------------------------------------------------


class TestDownloadArxivAbstracts:
    def test_loads_from_cache_when_present(self, tmp_path):
        """If the JSON cache covers the requested categories, no requests fire."""
        from vtsearch.datasets import downloader as dl_module

        cache_path = tmp_path / "arxiv_abstracts.json"
        cache_path.write_text(
            json.dumps({"cs.AI": ["Cached AI paper."], "cs.CV": ["Cached CV paper."]}),
            encoding="utf-8",
        )

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("vtsearch.datasets.downloader.text.requests.get") as mock_get,
        ):
            result = dl_module.download_arxiv_abstracts(
                categories=["cs.AI", "cs.CV"],
                on_progress=lambda *a: None,
            )

        mock_get.assert_not_called()
        assert result == {"cs.AI": ["Cached AI paper."], "cs.CV": ["Cached CV paper."]}

    def test_fetches_and_caches(self, tmp_path):
        """First load hits the API and writes a cache file."""
        from vtsearch.datasets import downloader as dl_module

        def fake_get(url, timeout):
            # Return a small Atom feed once, then an empty one to stop paging.
            if "start=0" in url:
                xml = _make_atom_feed([("Paper 1", "Body 1"), ("Paper 2", "Body 2")])
            else:
                xml = _make_atom_feed([])
            resp = MagicMock()
            resp.content = xml
            resp.raise_for_status = lambda: None
            return resp

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch("vtsearch.datasets.downloader.text.requests.get", side_effect=fake_get),
            patch("vtsearch.datasets.downloader.text.time.sleep", lambda *a, **k: None),
        ):
            result = dl_module.download_arxiv_abstracts(
                categories=["cs.AI"],
                max_per_category=5,
                on_progress=lambda *a: None,
            )

        assert result["cs.AI"] == ["Paper 1 Body 1", "Paper 2 Body 2"]
        cache = json.loads((tmp_path / "arxiv_abstracts.json").read_text(encoding="utf-8"))
        assert cache["cs.AI"] == ["Paper 1 Body 1", "Paper 2 Body 2"]

    def test_stops_when_request_fails(self, tmp_path):
        """Network error stops paging for that category but doesn't crash."""
        import requests as _requests

        from vtsearch.datasets import downloader as dl_module

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch(
                "vtsearch.datasets.downloader.text.requests.get",
                side_effect=_requests.ConnectionError("offline"),
            ),
            patch("vtsearch.datasets.downloader.text.time.sleep", lambda *a, **k: None),
        ):
            result = dl_module.download_arxiv_abstracts(
                categories=["cs.AI"],
                max_per_category=200,
                on_progress=lambda *a: None,
            )

        assert result == {"cs.AI": []}


# ---------------------------------------------------------------------------
# load_demo_source — arxiv branch
# ---------------------------------------------------------------------------


class TestLoadDemoSourceArxiv:
    def test_arxiv_source_populates_clips(self):
        from vtsearch.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_papers = {
            "cs.AI": ["AI paper one.", "AI paper two."],
            "math.AG": ["AG paper one.", "AG paper two."],
        }

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_arxiv_abstracts", return_value=fake_papers):
            mt.load_demo_source(
                source="arxiv",
                categories=["cs.AI", "math.AG"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 4
        assert {c["category"] for c in clips.values()} == {"cs.AI", "math.AG"}

    def test_arxiv_slice_is_applied(self):
        from vtsearch.datasets import downloader as dl_module
        from tests_lib.downloads._helpers import make_text_embedder_stub, make_text_media_type_stub

        fake_papers = {"cs.LG": [f"LG paper {i}." for i in range(10)]}

        mt = make_text_media_type_stub()
        emb = make_text_embedder_stub()
        clips: dict = {}

        with patch.object(dl_module, "download_arxiv_abstracts", return_value=fake_papers):
            mt.load_demo_source(
                source="arxiv",
                categories=["cs.LG"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=emb,
            )

        assert len(clips) == 3

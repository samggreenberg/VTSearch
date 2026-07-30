"""Tests for the url_download media source (re-fetch an exemplar from its URL).

Backs origins stamped by the ``url_download`` datasource importer so a
URL-fetched exemplar stays resolvable after the ``example_media/`` cache
file is gone (issue #2774).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vtscore.datasets.sources import get_source_for_origin
from vtscore.datasets.sources.url_download import UrlDownloadSource


URL = "https://media.example.test/sounds/dog%20bark.wav?tok=1"


@pytest.fixture
def fake_download(monkeypatch):
    """Stub the downloader + URL validator; records every fetched URL."""
    calls: list[str] = []

    def _download(url, dest_path, expected_size=0, on_progress=None):
        calls.append(url)
        Path(dest_path).write_bytes(b"RIFFxxxxWAVE")

    import vtscore.datasets.downloader as downloader_mod
    import vtscore.security.url_validation as url_mod

    monkeypatch.setattr(downloader_mod, "download_file_with_progress", _download)
    monkeypatch.setattr(url_mod, "validate_url", lambda u: u)
    return calls


class TestFactory:
    def test_get_source_for_origin_builds_url_source(self):
        source = get_source_for_origin({"importer": "url_download", "params": {"url": URL}})
        assert isinstance(source, UrlDownloadSource)

    def test_missing_url_param_returns_none(self):
        assert get_source_for_origin({"importer": "url_download", "params": {}}) is None


class TestUrlDownloadSource:
    def test_resolve_path_downloads_with_real_extension(self, fake_download):
        source = UrlDownloadSource(URL)
        try:
            item = source.resolve_path(origin_name=URL, filename="abc.wav")
            assert item.path is not None
            assert item.path.read_bytes() == b"RIFFxxxxWAVE"
            # Suffix comes from the URL path (drives downstream decode).
            assert item.path.name == "dog bark.wav"
            assert fake_download == [URL]
        finally:
            source.cleanup()

    def test_repeat_access_downloads_once(self, fake_download):
        source = UrlDownloadSource(URL)
        try:
            first = source.resolve_path().path
            second = source.fetch_item("anything").path
            assert first == second
            assert fake_download == [URL]
        finally:
            source.cleanup()

    def test_cleanup_removes_temp_download(self, fake_download):
        source = UrlDownloadSource(URL)
        path = source.resolve_path().path
        assert path is not None and path.is_file()
        source.cleanup()
        assert not path.exists()

    def test_private_url_is_never_fetched(self, monkeypatch):
        """The stored URL is re-validated with the real SSRF guard before any fetch."""
        calls = []

        import vtscore.datasets.downloader as downloader_mod

        monkeypatch.setattr(
            downloader_mod,
            "download_file_with_progress",
            lambda *a, **k: calls.append(a),
        )
        source = UrlDownloadSource("http://127.0.0.1/x.wav")
        assert source.resolve_path().path is None
        assert calls == []

    def test_list_items_yields_single_item_without_downloading(self, fake_download):
        source = UrlDownloadSource(URL)
        items = list(source.list_items())
        assert [i.filename for i in items] == ["dog bark.wav"]
        assert fake_download == []

    def test_list_items_extension_filter(self, fake_download):
        source = UrlDownloadSource(URL)
        assert list(source.list_items(extensions=[".png"])) == []
        assert len(list(source.list_items(extensions=[".wav"]))) == 1


class TestResolverDispatch:
    def test_resolve_file_from_origin_refetches_from_url(self, fake_download):
        """End-to-end: a url_download origin resolves via source dispatch."""
        from vtscore.detectors.resolver import resolve_file_context

        origin = {"importer": "url_download", "params": {"url": URL}}
        with resolve_file_context(origin, URL, "abc.wav") as path:
            assert path is not None
            assert path.read_bytes() == b"RIFFxxxxWAVE"
        # The ExitStack ran the source's cleanup on exit.
        assert not path.exists()

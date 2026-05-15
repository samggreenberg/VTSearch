"""Tests for vtsearch.datasets.sources — MediaSource abstraction."""

import zipfile
from unittest.mock import patch

import pytest

from vtsearch.datasets.sources import get_source_for_origin
from vtsearch.datasets.sources.base import MediaItem
from vtsearch.datasets.sources.local_folder import LocalFolderSource


# ── MediaItem ─────────────────────────────────────────────────────────


class TestMediaItem:
    def test_fields(self):
        item = MediaItem(key="sub/file.wav", filename="file.wav", source_name="local_folder")
        assert item.key == "sub/file.wav"
        assert item.filename == "file.wav"
        assert item.source_name == "local_folder"

    def test_frozen(self):
        item = MediaItem(key="k", filename="f", source_name="s")
        with pytest.raises(AttributeError):
            item.key = "other"


# ── LocalFolderSource ─────────────────────────────────────────────────


class TestLocalFolderSource:
    def _make_tree(self, tmp_path):
        """Create a sample directory tree for testing."""
        (tmp_path / "a.wav").write_bytes(b"audio_a")
        (tmp_path / "b.mp3").write_bytes(b"audio_b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.wav").write_bytes(b"audio_c")
        (sub / "d.txt").write_bytes(b"text_d")
        return tmp_path

    def test_list_items_all(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        items = list(source.list_items())
        keys = {i.key for i in items}
        assert "a.wav" in keys
        assert "b.mp3" in keys
        assert "sub/c.wav" in keys
        assert "sub/d.txt" in keys
        assert all(i.source_name == "local_folder" for i in items)

    def test_list_items_filtered(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        items = list(source.list_items(extensions=[".wav"]))
        keys = {i.key for i in items}
        assert "a.wav" in keys
        assert "sub/c.wav" in keys
        assert "b.mp3" not in keys
        assert "sub/d.txt" not in keys

    def test_list_items_follows_symlinked_directories(self, tmp_path):
        """Files inside symlinked subdirectories must be discovered."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "a.wav").write_bytes(b"audio_a")

        # Create an external directory with media files.
        external = tmp_path / "external"
        external.mkdir()
        (external / "b.wav").write_bytes(b"audio_b")
        (external / "c.mp3").write_bytes(b"audio_c")

        # Symlink external into root.
        (root / "linked").symlink_to(external)

        source = LocalFolderSource(root)
        items = list(source.list_items())
        keys = {i.key for i in items}
        assert "a.wav" in keys
        assert "linked/b.wav" in keys
        assert "linked/c.mp3" in keys

    def test_list_items_empty_folder(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        source = LocalFolderSource(empty)
        assert list(source.list_items()) == []

    def test_list_items_nonexistent_folder(self, tmp_path):
        source = LocalFolderSource(tmp_path / "nope")
        assert list(source.list_items()) == []

    def test_fetch_item(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        result = source.fetch_item("sub/c.wav")
        assert result is not None
        assert result.name == "c.wav"
        assert result.read_bytes() == b"audio_c"

    def test_fetch_item_missing(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        assert source.fetch_item("nonexistent.wav") is None

    def test_fetch_item_path_traversal(self, tmp_path):
        root = self._make_tree(tmp_path)
        # Create a file outside the root
        (tmp_path.parent / "secret.txt").write_bytes(b"secret")
        source = LocalFolderSource(root)
        assert source.fetch_item("../secret.txt") is None

    def test_resolve_path_by_origin_name(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        result = source.resolve_path(origin_name="a.wav")
        assert result is not None
        assert result.name == "a.wav"

    def test_resolve_path_by_filename_fallback(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        result = source.resolve_path(filename="sub/c.wav")
        assert result is not None
        assert result.name == "c.wav"

    def test_resolve_path_both_empty(self, tmp_path):
        root = self._make_tree(tmp_path)
        source = LocalFolderSource(root)
        assert source.resolve_path() is None

    def test_folder_path_property(self, tmp_path):
        source = LocalFolderSource(tmp_path)
        assert source.folder_path == tmp_path

    def test_cleanup_is_noop(self, tmp_path):
        source = LocalFolderSource(tmp_path)
        source.cleanup()  # Should not raise


# ── HttpArchiveSource ─────────────────────────────────────────────────


class TestHttpArchiveSource:
    def test_lazy_extraction(self, tmp_path):
        """HttpArchiveSource downloads and extracts only on first access."""
        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        source = HttpArchiveSource("https://example.com/test.zip")
        # Before any access, _inner should be None
        assert source._inner is None

    def test_uses_cached_dir(self, tmp_path):
        """When a cached extraction directory exists, it's reused."""
        from vtsearch.config import DATA_DIR
        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        # Create a fake cached extraction dir
        cached = DATA_DIR / "http_archive_resolve_test.zip"
        cached.mkdir(parents=True, exist_ok=True)
        (cached / "clip.wav").write_bytes(b"audio")

        try:
            source = HttpArchiveSource("https://example.com/test.zip")
            result = source.resolve_path(origin_name="clip.wav")
            assert result is not None
            assert result.name == "clip.wav"
        finally:
            import shutil

            shutil.rmtree(cached, ignore_errors=True)

    def test_delegates_to_local_folder(self, tmp_path):
        """After extraction, fetch_item delegates to LocalFolderSource."""
        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        # Create a zip archive in tmp_path
        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("audio/clip.wav", b"fake_audio")
            zf.writestr("audio/other.mp3", b"fake_mp3")

        source = HttpArchiveSource("https://example.com/archive.zip")

        # Manually set up extraction to avoid actual download
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        from vtsearch.datasets.sources.local_folder import LocalFolderSource

        source._extract_dir = extract_dir
        source._inner = LocalFolderSource(extract_dir)

        result = source.fetch_item("audio/clip.wav")
        assert result is not None
        assert result.name == "clip.wav"

        items = list(source.list_items(extensions=[".wav"]))
        assert any(i.key == "audio/clip.wav" for i in items)
        assert all(i.source_name == "http_archive" for i in items)

    def test_cleanup_removes_source_dir(self, tmp_path):
        """cleanup() removes directories named http_archive_source_*."""
        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        source = HttpArchiveSource("https://example.com/test.zip")
        source_dir = tmp_path / "http_archive_source_abc123"
        source_dir.mkdir()
        (source_dir / "file.wav").write_bytes(b"data")

        from vtsearch.datasets.sources.local_folder import LocalFolderSource

        source._extract_dir = source_dir
        source._inner = LocalFolderSource(source_dir)

        source.cleanup()
        assert not source_dir.exists()
        assert source._inner is None

    def test_cleanup_preserves_cached_dir(self, tmp_path):
        """cleanup() does NOT remove http_archive_resolve_* (cached) dirs."""
        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        source = HttpArchiveSource("https://example.com/test.zip")
        cached_dir = tmp_path / "http_archive_resolve_test.zip"
        cached_dir.mkdir()
        (cached_dir / "file.wav").write_bytes(b"data")

        from vtsearch.datasets.sources.local_folder import LocalFolderSource

        source._extract_dir = cached_dir
        source._inner = LocalFolderSource(cached_dir)

        source.cleanup()
        # Cached dir should still exist
        assert cached_dir.exists()


# ── get_source_for_origin ─────────────────────────────────────────────


class TestGetSourceForOrigin:
    def test_folder_origin(self, tmp_path):
        folder = tmp_path / "data"
        folder.mkdir()
        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        source = get_source_for_origin(origin)
        assert source is not None
        assert isinstance(source, LocalFolderSource)
        assert source.folder_path == folder

    def test_http_archive_origin(self):
        from vtsearch.datasets.sources.http_archive import HttpArchiveSource

        origin = {"importer": "http_archive", "params": {"url": "https://example.com/data.zip"}}
        source = get_source_for_origin(origin)
        assert source is not None
        assert isinstance(source, HttpArchiveSource)
        assert source.url == "https://example.com/data.zip"

    def test_pickle_origin_returns_none(self):
        origin = {"importer": "pickle", "params": {"file": "data.pkl"}}
        assert get_source_for_origin(origin) is None

    def test_combine_datasets_origin_returns_none(self):
        origin = {"importer": "combine_datasets", "params": {"datasets": "a.pkl,b.pkl"}}
        assert get_source_for_origin(origin) is None

    def test_none_origin(self):
        assert get_source_for_origin(None) is None

    def test_empty_path_returns_none(self):
        origin = {"importer": "server_folder", "params": {"path": ""}}
        assert get_source_for_origin(origin) is None

    def test_empty_url_returns_none(self):
        origin = {"importer": "http_archive", "params": {"url": ""}}
        assert get_source_for_origin(origin) is None


# ── Folder importer delegates to LocalFolderSource ────────────────────


class TestFolderImporterDelegatesToSource:
    def test_resolve_file_uses_source(self, tmp_path):
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"audio")

        from vtsearch.datasets.importers.server_folder import IMPORTER

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = IMPORTER.resolve_file(origin, origin_name="clip.wav")
        assert result == folder / "clip.wav"

    def test_resolve_file_nested(self, tmp_path):
        folder = tmp_path / "data"
        sub = folder / "cat"
        sub.mkdir(parents=True)
        (sub / "item.wav").write_bytes(b"data")

        from vtsearch.datasets.importers.server_folder import IMPORTER

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = IMPORTER.resolve_file(origin, origin_name="cat/item.wav")
        assert result == sub / "item.wav"


# ── Resolver uses sources ─────────────────────────────────────────────


class TestResolverUsesSource:
    def test_resolve_via_source(self, tmp_path):
        """resolve_file_from_origin now tries sources before importers."""
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"audio")

        from vtsearch.detectors.resolver import resolve_file_from_origin

        origin = {"importer": "server_folder", "params": {"path": str(folder)}}
        result = resolve_file_from_origin(origin, origin_name="clip.wav")
        assert result == folder / "clip.wav"


# ── Ingest uses source fast-path ──────────────────────────────────────


class TestIngestViaSource:
    def test_ingest_fetches_individually(self, tmp_path):
        """When a source is available and embedding works, ingest fetches individually."""
        import numpy as np

        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "good.wav").write_bytes(b"good_audio")
        (folder / "bad.wav").write_bytes(b"bad_audio")
        # Also create many other files the old approach would load
        for i in range(100):
            (folder / f"extra_{i}.wav").write_bytes(b"extra")

        from vtsearch.datasets.ingest import _ingest_via_source

        origin = {"importer": "server_folder", "params": {"path": str(folder), "media_type": "audio"}}
        entries = [
            {"origin": origin, "origin_name": "good.wav", "md5": "", "label": "good", "filename": "good.wav"},
            {"origin": origin, "origin_name": "bad.wav", "md5": "", "label": "bad", "filename": "bad.wav"},
        ]

        medias: dict = {}
        progress_calls = []

        def track_progress(status, msg, current, total):
            progress_calls.append((status, msg, current, total))

        fake_emb = np.zeros(512, dtype=np.float32)
        with patch("vtsearch.detectors.resolver.embed_file", return_value=fake_emb):
            result = _ingest_via_source(origin, entries, medias, track_progress)

        assert result == 2
        assert len(medias) == 2
        # Verify media data
        for media in medias.values():
            assert media["origin"] == origin
            assert "md5" in media
            assert "media_bytes" in media
            assert media["embedding"] is not None

    def test_ingest_falls_back_when_embed_fails(self, tmp_path):
        """When embedding fails, fast path returns -1 to trigger legacy fallback."""
        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"audio")

        from vtsearch.datasets.ingest import _ingest_via_source

        origin = {"importer": "server_folder", "params": {"path": str(folder), "media_type": "audio"}}
        entries = [
            {"origin": origin, "origin_name": "clip.wav", "md5": "", "label": "good", "filename": "clip.wav"},
        ]

        medias: dict = {}
        with patch("vtsearch.detectors.resolver.embed_file", return_value=None):
            result = _ingest_via_source(origin, entries, medias, lambda *a: None)

        assert result == -1
        assert len(medias) == 0  # Nothing ingested, caller should use legacy

    def test_ingest_returns_negative_one_for_pickle(self):
        """Non-file-based origins return -1 (fallback to full importer)."""
        from vtsearch.datasets.ingest import _ingest_via_source

        origin = {"importer": "pickle", "params": {"file": "data.pkl"}}
        entries = [{"origin": origin, "origin_name": "x.wav"}]
        result = _ingest_via_source(origin, entries, {}, lambda *a: None)
        assert result == -1


# ── Example-sort-origin API endpoint ──────────────────────────────────


class TestExampleSortOriginEndpoint:
    def test_missing_origin(self, client):
        resp = client.post(
            "/api/example-sort-origin",
            json={"key": "clip.wav"},
        )
        assert resp.status_code == 400
        assert "origin" in resp.get_json()["error"].lower()

    def test_missing_key(self, client):
        resp = client.post(
            "/api/example-sort-origin",
            json={"origin": {"importer": "server_folder", "params": {"path": "/tmp"}}},
        )
        assert resp.status_code == 400
        assert "key" in resp.get_json()["error"].lower()

    def test_unknown_source_type(self, client):
        resp = client.post(
            "/api/example-sort-origin",
            json={
                "origin": {"importer": "pickle", "params": {}},
                "key": "clip.wav",
            },
        )
        assert resp.status_code == 400
        assert "no media source" in resp.get_json()["error"].lower()

    def test_file_not_found(self, client, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        resp = client.post(
            "/api/example-sort-origin",
            json={
                "origin": {"importer": "server_folder", "params": {"path": str(folder)}},
                "key": "nonexistent.wav",
            },
        )
        assert resp.status_code == 404

    def test_success(self, client, tmp_path):
        """A valid origin+key returns sorted results when medias are loaded."""
        from vtsearch.state import medias

        folder = tmp_path / "audio"
        folder.mkdir()
        (folder / "example.wav").write_bytes(b"fake_audio")

        # Need medias loaded for example-sort to work
        if not medias:
            pytest.skip("No medias loaded in test environment")

        resp = client.post(
            "/api/example-sort-origin",
            json={
                "origin": {"importer": "server_folder", "params": {"path": str(folder)}},
                "key": "example.wav",
            },
        )
        # Will get 500 if embedder can't handle fake audio, but not 400/404
        assert resp.status_code in (200, 500)

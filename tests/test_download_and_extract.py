"""Tests for the _download_and_extract() generic helper."""

import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tar_gz(tmp_path: Path, arcname: str = "inner") -> Path:
    """Create a minimal tar.gz archive containing a single directory."""
    tar_path = tmp_path / "test.tar.gz"
    content_dir = tmp_path / "staging" / arcname
    content_dir.mkdir(parents=True)
    (content_dir / "file.txt").write_text("hello")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(content_dir, arcname=arcname)
    return tar_path


def _make_tar(tmp_path: Path, arcname: str = "inner") -> Path:
    """Create a minimal uncompressed tar archive."""
    tar_path = tmp_path / "test.tar"
    content_dir = tmp_path / "staging" / arcname
    content_dir.mkdir(parents=True)
    (content_dir / "file.txt").write_text("hello")
    with tarfile.open(tar_path, "w:") as tf:
        tf.add(content_dir, arcname=arcname)
    return tar_path


def _make_zip(tmp_path: Path, arcname: str = "inner") -> Path:
    """Create a minimal zip archive."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{arcname}/file.txt", "hello")
    return zip_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDownloadAndExtract:
    def test_tar_gz_extraction(self, tmp_path):
        """Extracts a .tar.gz archive and removes it afterward."""
        from vtsearch.datasets import downloader as dl_module

        tar_path = _make_tar_gz(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda url, dest, size, cb: tar_path.rename(dest),
            ),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.tar.gz",
                archive_name="test.tar.gz",
                extract_to=extract_to,
                check_path=check_path,
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
            )

        assert check_path.exists()
        assert (check_path / "file.txt").read_text() == "hello"
        # Archive should be deleted after extraction.
        assert not (tmp_path / "test.tar.gz").exists()

    def test_tar_extraction(self, tmp_path):
        """Extracts a .tar archive and removes it afterward."""
        from vtsearch.datasets import downloader as dl_module

        tar_path = _make_tar(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda url, dest, size, cb: tar_path.rename(dest),
            ),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.tar",
                archive_name="test.tar",
                extract_to=extract_to,
                check_path=check_path,
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
            )

        assert check_path.exists()
        assert (check_path / "file.txt").read_text() == "hello"
        assert not (tmp_path / "test.tar").exists()

    def test_zip_extraction(self, tmp_path):
        """Extracts a .zip archive and removes it afterward."""
        from vtsearch.datasets import downloader as dl_module

        zip_path = _make_zip(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda url, dest, size, cb: zip_path.rename(dest),
            ),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.zip",
                archive_name="test.zip",
                extract_to=extract_to,
                check_path=check_path,
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
            )

        assert check_path.exists()
        assert (check_path / "file.txt").read_text() == "hello"
        assert not (tmp_path / "test.zip").exists()

    def test_tgz_extension(self, tmp_path):
        """Recognises .tgz as gzip tar."""
        from vtsearch.datasets import downloader as dl_module

        tar_path = _make_tar_gz(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        def fake_download(url, dest, size, cb):
            import shutil

            shutil.copy(str(tar_path), str(dest))

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(dl_module, "download_file_with_progress", fake_download),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.tgz",
                archive_name="test.tgz",
                extract_to=extract_to,
                check_path=check_path,
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
            )

        assert check_path.exists()

    def test_skips_when_check_path_exists(self, tmp_path):
        """No download or extraction when check_path already exists."""
        from vtsearch.datasets import downloader as dl_module

        check_path = tmp_path / "already_here"
        check_path.mkdir()

        download_called = []

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.tar.gz",
                archive_name="test.tar.gz",
                extract_to=tmp_path / "extracted",
                check_path=check_path,
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
            )

        assert not download_called

    def test_unsupported_format_raises(self, tmp_path):
        """Raises ValueError for unsupported archive extensions."""
        from vtsearch.datasets import downloader as dl_module

        # Create a dummy file so the download step succeeds.
        dummy = tmp_path / "test.rar"
        dummy.write_bytes(b"fake")

        with (
            patch.object(dl_module, "DATA_DIR", tmp_path),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda *a, **kw: None,
            ),
        ):
            with pytest.raises(ValueError, match="Unsupported archive format"):
                dl_module._download_and_extract(
                    url="http://example.com/test.rar",
                    archive_name="test.rar",
                    extract_to=tmp_path / "extracted",
                    check_path=tmp_path / "nonexistent",
                    download_size_mb=1,
                    dataset_name="Test",
                    on_progress=lambda *a: None,
                )

    def test_progress_messages(self, tmp_path):
        """Reports download and extraction progress."""
        from vtsearch.datasets import downloader as dl_module

        # Place the archive in a subdirectory so it doesn't collide with
        # DATA_DIR / archive_name (which would cause the download to be skipped).
        staging = tmp_path / "staging_area"
        staging.mkdir()
        tar_path = _make_tar_gz(staging, arcname="data_dir")
        progress_calls = []

        def track_progress(status, msg, cur, tot):
            progress_calls.append((status, msg))

        data_dir = tmp_path / "data"

        with (
            patch.object(dl_module, "DATA_DIR", data_dir),
            patch.object(
                dl_module,
                "download_file_with_progress",
                lambda url, dest, size, cb: tar_path.rename(dest),
            ),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.tar.gz",
                archive_name="test.tar.gz",
                extract_to=data_dir / "extracted",
                check_path=data_dir / "extracted" / "data_dir",
                download_size_mb=1,
                dataset_name="MyDataset",
                on_progress=track_progress,
            )

        messages = [msg for _, msg in progress_calls]
        assert any("Starting MyDataset download" in m for m in messages)
        assert any("Extracting MyDataset" in m for m in messages)

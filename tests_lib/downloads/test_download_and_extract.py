"""Tests for the _download_and_extract() generic helper."""

import tarfile
import threading
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
        from vtscore.datasets import downloader as dl_module

        tar_path = _make_tar_gz(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
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
        from vtscore.datasets import downloader as dl_module

        tar_path = _make_tar(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
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
        from vtscore.datasets import downloader as dl_module

        zip_path = _make_zip(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
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
        from vtscore.datasets import downloader as dl_module

        tar_path = _make_tar_gz(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        def fake_download(url, dest, size, cb):
            import shutil

            shutil.copy(str(tar_path), str(dest))

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
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
        from vtscore.datasets import downloader as dl_module

        check_path = tmp_path / "already_here"
        check_path.mkdir()

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
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
        from vtscore.datasets import downloader as dl_module

        # Create a dummy file so the download step succeeds.
        dummy = tmp_path / "test.rar"
        dummy.write_bytes(b"fake")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
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
        from vtscore.datasets import downloader as dl_module

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
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(
                dl_module.core,
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

    def test_extraction_progress_has_total(self, tmp_path):
        """Extraction progress reports current/total counts (not indeterminate 0/0)."""
        from vtscore.datasets import downloader as dl_module

        staging = tmp_path / "staging_area"
        staging.mkdir()
        tar_path = _make_tar_gz(staging, arcname="data_dir")
        progress_calls = []

        def track_progress(status, msg, cur, tot):
            progress_calls.append((status, msg, cur, tot))

        data_dir = tmp_path / "data"

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(
                dl_module.core,
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

        # Find extraction progress calls (not the initial 0/0 announcement)
        extract_progress = [(msg, cur, tot) for _, msg, cur, tot in progress_calls if "Extracting" in msg and tot > 0]
        assert len(extract_progress) > 0, "Expected determinate extraction progress"
        _last_msg, last_cur, last_tot = extract_progress[-1]
        assert last_cur == last_tot, "Final progress should show completion"


class TestCorruptArchiveValidation:
    """Corrupt or non-archive downloads are detected and cleaned up."""

    def test_corrupt_tar_gz_deleted_and_raises(self, tmp_path):
        """An HTML page saved as .tar.gz is detected, deleted, and raises."""
        from vtscore.datasets import downloader as dl_module

        def fake_download(url, dest, size, cb):
            dest.write_text("<html>404 Not Found</html>")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            with pytest.raises(RuntimeError, match="invalid file"):
                dl_module._download_and_extract(
                    url="http://example.com/test.tar.gz",
                    archive_name="test.tar.gz",
                    extract_to=tmp_path / "extracted",
                    check_path=tmp_path / "nonexistent",
                    download_size_mb=1,
                    dataset_name="Test Dataset",
                    on_progress=lambda *a: None,
                )

        # The corrupt file should have been deleted so retries re-download.
        assert not (tmp_path / "test.tar.gz").exists()

    def test_corrupt_zip_deleted_and_raises(self, tmp_path):
        """Non-zip content saved as .zip is detected and cleaned up."""
        from vtscore.datasets import downloader as dl_module

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"this is not a zip file")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            with pytest.raises(RuntimeError, match="invalid file"):
                dl_module._download_and_extract(
                    url="http://example.com/test.zip",
                    archive_name="test.zip",
                    extract_to=tmp_path / "extracted",
                    check_path=tmp_path / "nonexistent",
                    download_size_mb=1,
                    dataset_name="Test Dataset",
                    on_progress=lambda *a: None,
                )

        assert not (tmp_path / "test.zip").exists()

    def test_valid_tar_gz_passes_validation(self, tmp_path):
        """A valid .tar.gz file passes validation and extracts normally."""
        from vtscore.datasets import downloader as dl_module

        tar_path = _make_tar_gz(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
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

    def test_corrupt_download_cleaned_up(self, tmp_path):
        """A corrupt download is detected and its temp file is cleaned up."""
        from vtscore.datasets import downloader as dl_module

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"this is not a valid archive at all")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            with pytest.raises(RuntimeError, match="invalid file"):
                dl_module._download_and_extract(
                    url="http://example.com/test.tar.gz",
                    archive_name="test.tar.gz",
                    extract_to=tmp_path / "extracted",
                    check_path=tmp_path / "nonexistent",
                    download_size_mb=1,
                    dataset_name="UCF-101 subset",
                    on_progress=lambda *a: None,
                )

        # No temp files should remain after cleanup.
        leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".dl_")]
        assert not leftover, f"Temp archive files should be cleaned up: {leftover}"

    def test_error_message_is_user_friendly(self, tmp_path):
        """The error message mentions the dataset name and suggests retrying."""
        from vtscore.datasets import downloader as dl_module

        def fake_download(url, dest, size, cb):
            dest.write_text("<html>Service Unavailable</html>")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            with pytest.raises(RuntimeError, match="My Dataset") as exc_info:
                dl_module._download_and_extract(
                    url="http://example.com/test.tar.gz",
                    archive_name="test.tar.gz",
                    extract_to=tmp_path / "extracted",
                    check_path=tmp_path / "nonexistent",
                    download_size_mb=1,
                    dataset_name="My Dataset",
                    on_progress=lambda *a: None,
                )

        msg = str(exc_info.value)
        assert "try again" in msg.lower()


class TestCdnDecompressedTarGz:
    """CDN-decompressed .tar.gz files (raw tar) are accepted and extracted."""

    def test_raw_tar_with_tar_gz_name_passes_validation(self, tmp_path):
        """A raw tar served for a .tar.gz URL passes validation."""
        from vtscore.datasets import downloader as dl_module

        # Create an uncompressed tar but name it .tar.gz (mimics CDN behaviour).
        tar_path = _make_tar(tmp_path, arcname="data_dir")
        misnamed = tmp_path / "test.tar.gz"
        tar_path.rename(misnamed)

        # Should not raise; the file is a valid tar even without gzip.
        dl_module._validate_archive(misnamed, "test.tar.gz", "Test")

    def test_raw_tar_with_tar_gz_name_extracts(self, tmp_path):
        """A raw tar named .tar.gz extracts correctly via _download_and_extract."""
        from vtscore.datasets import downloader as dl_module

        tar_path = _make_tar(tmp_path, arcname="data_dir")
        misnamed = tmp_path / "staging" / "test.tar.gz"
        misnamed.parent.mkdir(parents=True, exist_ok=True)
        tar_path.rename(misnamed)

        extract_to = tmp_path / "extracted"
        check_path = extract_to / "data_dir"
        data_dir = tmp_path / "data"

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: misnamed.rename(dest),
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


class TestCaltech101ExtractionProgress:
    """Caltech-101 inner tar extraction reports determinate progress."""

    def test_inner_tar_reports_progress(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        # Build a fake caltech-101.zip containing a nested tar.gz
        inner_staging = tmp_path / "inner_staging" / "101_ObjectCategories"
        inner_staging.mkdir(parents=True)
        for i in range(5):
            (inner_staging / f"img_{i}.jpg").write_bytes(b"fake")

        inner_tar_path = tmp_path / "101_ObjectCategories.tar.gz"
        with tarfile.open(inner_tar_path, "w:gz") as tf:
            tf.add(inner_staging, arcname="101_ObjectCategories")

        outer_zip_path = tmp_path / "caltech-101.zip"
        with zipfile.ZipFile(outer_zip_path, "w") as zf:
            # Real zip extracts to DATA_DIR; inner tar must land under caltech-101/
            zf.write(inner_tar_path, "caltech-101/101_ObjectCategories.tar.gz")

        progress_calls = []

        def track(status, msg, cur, tot):
            progress_calls.append((status, msg, cur, tot))

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(dl_module.core, "IMAGE_DIR", data_dir / "images"),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: __import__("shutil").copy(str(outer_zip_path), str(dest)),
            ),
        ):
            result = dl_module.download_caltech101(on_progress=track)

        assert result.exists()

        # The inner tar extraction should report determinate progress
        inner_progress = [
            (msg, cur, tot)
            for _, msg, cur, tot in progress_calls
            if "Extracting 101_ObjectCategories" in msg and tot > 0
        ]
        assert len(inner_progress) > 0, "Inner tar extraction should report progress with total"
        _last_msg, last_cur, last_tot = inner_progress[-1]
        assert last_cur == last_tot


class TestBbcNewsExtractionProgress:
    """BBC News zip extraction reports determinate progress."""

    def test_zip_reports_progress(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        # Build a fake BBC News zip
        zip_path = tmp_path / "bbc-news.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for cat in ("business", "entertainment", "politics", "sport", "tech"):
                for i in range(3):
                    zf.writestr(f"bbc/{cat}/{i:03d}.txt", f"Article {cat} {i}")

        progress_calls = []

        def track(status, msg, cur, tot):
            progress_calls.append((status, msg, cur, tot))

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(dl_module.core, "IMAGE_DIR", data_dir / "images"),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: __import__("shutil").copy(str(zip_path), str(dest)),
            ),
        ):
            result = dl_module.download_bbc_news(on_progress=track)

        # download_bbc_news returns a dict of category -> articles
        assert isinstance(result, dict)
        assert len(result) > 0

        # The zip extraction should report determinate progress
        extract_progress = [
            (msg, cur, tot) for _, msg, cur, tot in progress_calls if "Extracting BBC News dataset" in msg and tot > 0
        ]
        assert len(extract_progress) > 0, "BBC News extraction should report progress with total"
        _last_msg, last_cur, last_tot = extract_progress[-1]
        assert last_cur == last_tot


class TestConcurrentDownloads:
    """Concurrent downloads of the same archive do not interfere."""

    def test_two_concurrent_downloads_both_succeed(self, tmp_path):
        """Two threads downloading the same archive both complete without error."""
        import shutil

        from vtscore.datasets import downloader as dl_module

        # Create a valid zip archive to serve as the download source.
        source_zip = tmp_path / "source.zip"
        with zipfile.ZipFile(source_zip, "w") as zf:
            zf.writestr("data_dir/file.txt", "hello")

        barrier = threading.Barrier(2, timeout=10)
        errors: list[Exception] = []

        def slow_download(url, dest, size, cb):
            """Simulate a slow download; both threads start before either finishes."""
            shutil.copy(str(source_zip), str(dest))
            barrier.wait()  # ensure both threads are mid-download

        data_dir = tmp_path / "data"
        extract_to = data_dir / "extracted"
        check_path = extract_to / "data_dir"

        def run():
            try:
                dl_module._download_and_extract(
                    url="http://example.com/test.zip",
                    archive_name="test.zip",
                    extract_to=extract_to,
                    check_path=check_path,
                    download_size_mb=1,
                    dataset_name="Test",
                    on_progress=lambda *a: None,
                )
            except Exception as exc:
                errors.append(exc)

        # Patch on the MAIN thread (around start+join) rather than inside each
        # worker. ``patch.object`` mutates a module-global, so applying it per
        # thread is not thread-safe: under load a worker could still be inside
        # its ``with`` block (or hung on the barrier) when the block was meant
        # to exit, leaking ``slow_download`` into later tests. Keeping the patch
        # on the main thread guarantees it is restored only after both joins.
        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(dl_module.core, "download_file_with_progress", slow_download),
        ):
            t1 = threading.Thread(target=run)
            t2 = threading.Thread(target=run)
            t1.start()
            t2.start()
            t1.join(timeout=30)
            t2.join(timeout=30)

        assert not errors, f"Concurrent downloads raised errors: {errors}"
        assert check_path.exists()
        assert (check_path / "file.txt").read_text() == "hello"

        # No temp files should remain.
        leftover = [p for p in data_dir.iterdir() if p.name.startswith(".dl_") or p.name.startswith(".extract_")]
        assert not leftover, f"Temp files should be cleaned up: {leftover}"

    def test_second_download_defers_to_first(self, tmp_path):
        """If the first download finishes before the second starts extracting,
        the second cleans up without errors."""
        import shutil

        from vtscore.datasets import downloader as dl_module

        source_zip = tmp_path / "source.zip"
        with zipfile.ZipFile(source_zip, "w") as zf:
            zf.writestr("data_dir/file.txt", "hello")

        data_dir = tmp_path / "data"
        extract_to = data_dir / "extracted"
        check_path = extract_to / "data_dir"

        call_count = 0

        def download_and_create_check_path(url, dest, size, cb):
            """First call creates the check_path to simulate another thread finishing."""
            nonlocal call_count
            shutil.copy(str(source_zip), str(dest))
            call_count += 1
            if call_count == 1:
                # Simulate the first download having already finished extraction.
                check_path.mkdir(parents=True, exist_ok=True)
                (check_path / "file.txt").write_text("from first download")

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(dl_module.core, "download_file_with_progress", download_and_create_check_path),
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
        # Content should be from the "first download" that finished first.
        assert (check_path / "file.txt").read_text() == "from first download"


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests.Response``.

    Yields the given *chunks* from ``iter_content``; if *fail_after_chunks* is
    set, raises ``ChunkedEncodingError`` once that many chunks have been yielded
    (simulating a mid-stream connection drop / ``IncompleteRead``).
    """

    is_redirect = False
    is_permanent_redirect = False

    def __init__(self, chunks, status_code=200, headers=None, fail_after_chunks=None):
        self._chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}
        self._fail_after_chunks = fail_after_chunks
        self.closed = False

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after_chunks is not None and i >= self._fail_after_chunks:
                import requests

                raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
            yield chunk

    def close(self):
        self.closed = True


class TestDownloadResume:
    """``download_file_with_progress`` recovers from mid-stream connection drops."""

    @staticmethod
    def _install(monkeypatch, responses):
        """Patch ``requests.Session.get`` to hand out *responses* in order and
        record the request headers (so the Range header can be asserted). Also
        neutralizes the retry backoff sleep."""
        import requests

        from vtscore.datasets.downloader import core as dl_core

        calls = []

        def fake_get(self, url, *args, headers=None, **kwargs):
            calls.append({"url": url, "headers": headers})
            return responses.pop(0)

        monkeypatch.setattr(requests.Session, "get", fake_get)
        monkeypatch.setattr(dl_core.time, "sleep", lambda *a, **k: None)
        return calls

    def test_resumes_after_connection_drop(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        payload = bytes(range(256)) * 8  # 2048 deterministic bytes
        chunks = [payload[i : i + 256] for i in range(0, len(payload), 256)]  # 8 chunks
        # Attempt 1: full-size 200 response that dies after 3 chunks (768 bytes).
        first = _FakeResponse(
            chunks, status_code=200, headers={"content-length": str(len(payload))}, fail_after_chunks=3
        )
        # Attempt 2: 206 partial response serving the remaining bytes.
        second = _FakeResponse(
            chunks[3:],
            status_code=206,
            headers={
                "content-length": str(len(payload) - 768),
                "Content-Range": f"bytes 768-{len(payload) - 1}/{len(payload)}",
            },
        )
        calls = self._install(monkeypatch, [first, second])

        dest = tmp_path / "archive.tar"
        reported = []
        dl_core.download_file_with_progress(
            "https://example.com/archive.tar", dest, on_progress=lambda *a: reported.append(a)
        )

        # The file is complete and byte-identical despite the drop.
        assert dest.read_bytes() == payload
        # The resume requested exactly the bytes already on disk.
        assert calls[1]["headers"] == {"Range": "bytes=768-"}
        # Progress always reported the true total, never a truncated one.
        assert {a[3] for a in reported} == {len(payload)}

    def test_restarts_when_server_ignores_range(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        payload = bytes(range(256)) * 4  # 1024 bytes
        chunks = [payload[i : i + 256] for i in range(0, len(payload), 256)]  # 4 chunks
        first = _FakeResponse(
            chunks, status_code=200, headers={"content-length": str(len(payload))}, fail_after_chunks=2
        )
        # Server ignores Range and resends the whole body with 200, not 206.
        full_again = _FakeResponse(chunks, status_code=200, headers={"content-length": str(len(payload))})
        self._install(monkeypatch, [first, full_again])

        dest = tmp_path / "archive.tar"
        dl_core.download_file_with_progress("https://example.com/archive.tar", dest, on_progress=lambda *a: None)

        # Restart-from-scratch must not duplicate the already-written prefix.
        assert dest.read_bytes() == payload

    def test_raises_after_exhausting_attempts(self, tmp_path, monkeypatch):
        import requests

        from vtscore.datasets.downloader import core as dl_core

        chunks = [b"x" * 256]
        dropping = [
            _FakeResponse(chunks, headers={"content-length": "256"}, fail_after_chunks=0)
            for _ in range(dl_core._MAX_DOWNLOAD_ATTEMPTS)
        ]
        self._install(monkeypatch, dropping)

        dest = tmp_path / "archive.tar"
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            dl_core.download_file_with_progress("https://example.com/archive.tar", dest, on_progress=lambda *a: None)

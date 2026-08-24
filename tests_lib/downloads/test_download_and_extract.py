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

    def test_is_complete_overrides_check_path(self, tmp_path):
        """When *is_complete* is given, an existing check_path does NOT skip.

        Guards the partial-state trap: a bare (e.g. empty) check_path dir left
        by a failed run must not masquerade as a finished download. The probe
        decides completion by content, so the archive still downloads.
        """
        from vtscore.datasets import downloader as dl_module

        zip_path = _make_zip(tmp_path, arcname="data_dir")
        extract_to = tmp_path / "extracted"
        # A bare leftover dir whose mere existence would satisfy a check_path gate.
        check_path = extract_to / "partial"
        check_path.mkdir(parents=True)

        download_called = []

        def fake_download(url, dest, size, cb):
            download_called.append(True)
            zip_path.rename(dest)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            dl_module._download_and_extract(
                url="http://example.com/test.zip",
                archive_name="test.zip",
                extract_to=extract_to,
                check_path=check_path,
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
                is_complete=lambda: (extract_to / "data_dir" / "file.txt").exists(),
            )

        assert download_called, "is_complete=False must not be short-circuited by check_path.exists()"
        assert (extract_to / "data_dir" / "file.txt").read_text() == "hello"

    def test_is_complete_true_skips_download(self, tmp_path):
        """When *is_complete* returns True, nothing is downloaded."""
        from vtscore.datasets import downloader as dl_module

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
                url="http://example.com/test.zip",
                archive_name="test.zip",
                extract_to=tmp_path / "extracted",
                check_path=tmp_path / "does_not_exist",
                download_size_mb=1,
                dataset_name="Test",
                on_progress=lambda *a: None,
                is_complete=lambda: True,
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


class TestSpoolOnDestinationFilesystem:
    """Temp archive + temp extraction spool onto the destination filesystem.

    When extract_to (or a parent) is a symlink onto another volume - the shared
    demo-cache setup, or any relocated dataset dir - the multi-GB temp bytes must
    land there, not on DATA_DIR's (possibly small) volume, and the final publish
    must be a same-filesystem rename rather than a cross-device copy.

    A single-filesystem test can't observe the cross-device copy directly, but it
    can pin the observable proxy: the temp files are placed next to / inside the
    *resolved* target, never under DATA_DIR's root.
    """

    def test_temp_files_spool_next_to_resolved_symlink_target(self, tmp_path):
        from vtscore.datasets import downloader as dl_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # A relocated dataset dir: DATA_DIR/gtzan is a symlink onto a populated
        # cache dir living outside DATA_DIR (mimicking link-demo-cache.sh).
        cache_real = tmp_path / "cache" / "gtzan_real"
        cache_real.mkdir(parents=True)
        link = data_dir / "gtzan"
        link.symlink_to(cache_real, target_is_directory=True)

        # Build the download source out-of-tree so it doesn't collide with spool.
        staging = tmp_path / "staging"
        staging.mkdir()
        zip_path = _make_zip(staging, arcname="genres")

        captured = {}

        def fake_download(url, dest, size, cb):
            captured["dest"] = Path(dest)
            zip_path.rename(dest)

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            dl_module._download_and_extract(
                url="http://example.com/genres.zip",
                archive_name="genres.zip",
                extract_to=link,
                check_path=link / "genres",
                download_size_mb=1,
                dataset_name="GTZAN",
                on_progress=lambda *a: None,
            )

        # The temp archive spooled inside the resolved target, not DATA_DIR.
        assert captured["dest"].parent == cache_real
        # No temp litter leaked onto DATA_DIR's root filesystem.
        data_dir_entries = list(data_dir.iterdir())
        assert data_dir_entries == [link], f"unexpected entries under DATA_DIR: {data_dir_entries}"
        # Content published through the symlink into the relocated cache dir.
        assert (cache_real / "genres" / "file.txt").read_text() == "hello"
        assert (link / "genres" / "file.txt").read_text() == "hello"

    def test_nonexistent_target_spools_in_resolved_parent(self, tmp_path):
        """When extract_to doesn't exist yet, temp files spool in its parent and
        the publish is a rename into place (the common, non-symlinked case)."""
        from vtscore.datasets import downloader as dl_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        extract_to = data_dir / "gtzan"  # does not exist yet

        staging = tmp_path / "staging"
        staging.mkdir()
        zip_path = _make_zip(staging, arcname="genres")

        captured = {}

        def fake_download(url, dest, size, cb):
            captured["dest"] = Path(dest)
            zip_path.rename(dest)

        with (
            patch.object(dl_module.core, "DATA_DIR", data_dir),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
        ):
            dl_module._download_and_extract(
                url="http://example.com/genres.zip",
                archive_name="genres.zip",
                extract_to=extract_to,
                check_path=extract_to / "genres",
                download_size_mb=1,
                dataset_name="GTZAN",
                on_progress=lambda *a: None,
            )

        # Spooled next to the (to-be-created) target, i.e. in its parent.
        assert captured["dest"].parent == data_dir
        assert (extract_to / "genres" / "file.txt").read_text() == "hello"
        # Temp files cleaned up; only the published dir remains.
        assert [p.name for p in data_dir.iterdir()] == ["gtzan"]


class TestZipTraversalRejection:
    """_extract_archive's zip branch must reject traversal members with the
    same strict check as archive.py.

    Regression: the previous inline ``startswith`` prefix test lacked a
    trailing separator, so a member resolving to a *sibling* directory whose
    name merely starts with the destination path (``/x/dest-evil`` vs
    ``/x/dest``) passed validation.
    """

    def test_dotdot_member_rejected(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../escape.txt", "gotcha")
        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(ValueError, match="Path traversal"):
            dl_core._extract_archive(zip_path, "evil.zip", dest, "Test", lambda *a: None)
        assert not (tmp_path / "escape.txt").exists()

    def test_prefix_sibling_member_rejected(self, tmp_path):
        """A member escaping to a sibling dir sharing the dest's name prefix."""
        from vtscore.datasets.downloader import core as dl_core

        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../dest-evil/escape.txt", "gotcha")
        dest = tmp_path / "dest"
        dest.mkdir()

        with pytest.raises(ValueError, match="Path traversal"):
            dl_core._extract_archive(zip_path, "evil.zip", dest, "Test", lambda *a: None)
        assert not (tmp_path / "dest-evil").exists()

    def test_benign_zip_still_extracts(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        zip_path = _make_zip(tmp_path, arcname="ok")
        dest = tmp_path / "dest"
        dest.mkdir()

        dl_core._extract_archive(zip_path, "test.zip", dest, "Test", lambda *a: None)
        assert (dest / "ok" / "file.txt").read_text() == "hello"


class TestValidateArchiveHeaderRead:
    """_validate_archive must read only the archive's magic bytes.

    The memory regression (read_bytes() materialising a multi-GB archive to
    inspect 4 bytes) can't be asserted directly, but the behaviour contract
    can: validation decisions are identical and made from the first 4 bytes.
    """

    def test_valid_zip_header_passes(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        zip_path = _make_zip(tmp_path)
        dl_core._validate_archive(zip_path, "test.zip", "Test")
        assert zip_path.exists()

    def test_html_error_page_rejected_and_deleted(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        bad = tmp_path / "fake.zip"
        bad.write_bytes(b"<html>503 Service Unavailable</html>")
        with pytest.raises(RuntimeError, match="invalid file"):
            dl_core._validate_archive(bad, "fake.zip", "Test")
        assert not bad.exists()


class TestDownloadFileAtomic:
    """download_file_atomic must never leave partial bytes at the final path.

    download_file_with_progress deliberately leaves partial bytes at its
    destination on failure (resume support), so exists()-gated callers that
    downloaded straight to the final path cached truncated files forever.
    """

    def test_success_publishes_final_path(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        final = tmp_path / "labels.mat"

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"complete-content")

        with patch.object(dl_core, "download_file_with_progress", fake_download):
            dl_core.download_file_atomic("http://example.com/labels.mat", final, 0, lambda *a: None)

        assert final.read_bytes() == b"complete-content"
        # No temp litter.
        assert [p.name for p in tmp_path.iterdir()] == ["labels.mat"]

    def test_failure_leaves_no_file_at_final_path(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        final = tmp_path / "labels.mat"

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"partial")  # simulate bytes written before the failure
            raise ConnectionError("network died")

        with patch.object(dl_core, "download_file_with_progress", fake_download):
            with pytest.raises(ConnectionError):
                dl_core.download_file_atomic("http://example.com/labels.mat", final, 0, lambda *a: None)

        assert not final.exists(), "a truncated file at the final path poisons the exists() cache gate"
        assert list(tmp_path.iterdir()) == [], "temp file must be cleaned up"

    def test_concurrent_winner_is_preserved(self, tmp_path):
        from vtscore.datasets.downloader import core as dl_core

        final = tmp_path / "labels.mat"

        def fake_download(url, dest, size, cb):
            dest.write_bytes(b"mine")
            final.write_bytes(b"other-download-won")  # a rival completed first

        with patch.object(dl_core, "download_file_with_progress", fake_download):
            dl_core.download_file_atomic("http://example.com/labels.mat", final, 0, lambda *a: None)

        assert final.read_bytes() == b"other-download-won"
        assert [p.name for p in tmp_path.iterdir()] == ["labels.mat"]


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
    (simulating a mid-stream connection drop / ``IncompleteRead``).  *text* is
    the whole-body view the small-payload fetches read instead of streaming.
    """

    is_redirect = False
    is_permanent_redirect = False

    def __init__(self, chunks, status_code=200, headers=None, fail_after_chunks=None, text=""):
        self._chunks = list(chunks)
        self.status_code = status_code
        self.headers = headers or {}
        self._fail_after_chunks = fail_after_chunks
        self.text = text
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
        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            dl_core.download_file_with_progress("https://example.com/archive.tar", dest, on_progress=lambda *a: None)

        # The user-facing message is a sentence naming the host, not a nested
        # urllib3 MaxRetryError dump (issue #3216).
        message = str(excinfo.value)
        assert "example.com" in message
        assert "https://example.com/archive.tar" not in message
        assert excinfo.value.attempts == dl_core._MAX_DOWNLOAD_ATTEMPTS
        # The original exception stays reachable for the server-side traceback.
        assert isinstance(excinfo.value.__cause__, requests.exceptions.ChunkedEncodingError)


class TestConnectFailureHandling:
    """Connection-level failures retry with an escalating connect budget and
    end in one actionable sentence rather than a urllib3 dump (issue #3216)."""

    @staticmethod
    def _install_failing_get(monkeypatch, exc_factory, succeed_after=None):
        """Patch ``requests.Session.get`` to raise *exc_factory()* on every call
        (or until *succeed_after* calls have been made, then hand back a tiny
        200 response).  Records each call's timeout and neutralizes the backoff
        sleep.  Returns the list of recorded timeouts."""
        import requests

        from vtscore.datasets.downloader import core as dl_core

        timeouts = []

        def fake_get(self, url, *args, timeout=None, **kwargs):
            timeouts.append(timeout)
            if succeed_after is not None and len(timeouts) > succeed_after:
                return _FakeResponse([b"payload"], status_code=200, headers={"content-length": "7"})
            raise exc_factory()

        monkeypatch.setattr(requests.Session, "get", fake_get)
        monkeypatch.setattr(dl_core.time, "sleep", lambda *a, **k: None)
        return timeouts

    def test_connect_budget_escalates_across_attempts(self, tmp_path, monkeypatch):
        """A host slow to *accept* must not be written off on the same 10 s
        budget six times over."""
        import requests

        from vtscore.datasets.downloader import core as dl_core

        timeouts = self._install_failing_get(monkeypatch, lambda: requests.exceptions.ConnectTimeout("connect timeout"))

        with pytest.raises(dl_core.RemoteUnreachableError):
            dl_core.download_file_with_progress(
                "https://archive.org/download/x/y.mp3", tmp_path / "y.mp3", on_progress=lambda *a: None
            )

        connect_budgets = [t[0] for t in timeouts]
        assert connect_budgets == [10.0, 15.0, 20.0, 30.0, 30.0, 30.0]
        assert {t[1] for t in timeouts} == {dl_core._READ_TIMEOUT_S}

    def test_connect_timeout_message_names_the_host(self, tmp_path, monkeypatch):
        import requests

        from vtscore.datasets.downloader import core as dl_core

        self._install_failing_get(monkeypatch, lambda: requests.exceptions.ConnectTimeout("connect timeout"))

        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            dl_core.download_file_with_progress(
                "https://archive.org/download/Apollo11Audio/11-03301.mp3",
                tmp_path / "11-03301.mp3",
                on_progress=lambda *a: None,
            )

        message = str(excinfo.value)
        assert message.startswith("Couldn't reach archive.org:")
        assert "timed out" in message
        assert "retry" in message
        assert excinfo.value.url == "https://archive.org/download/Apollo11Audio/11-03301.mp3"

    def test_retry_message_says_retrying_before_any_bytes_land(self, tmp_path, monkeypatch):
        """ "Resuming ... at 0 bytes" read as a stalled transfer when in fact
        nothing had ever connected."""
        import requests

        from vtscore.datasets.downloader import core as dl_core

        self._install_failing_get(
            monkeypatch, lambda: requests.exceptions.ConnectTimeout("connect timeout"), succeed_after=2
        )

        reported = []
        dl_core.download_file_with_progress(
            "https://archive.org/download/x/y.mp3", tmp_path / "y.mp3", on_progress=lambda *a: reported.append(a)
        )

        retry_messages = [a[1] for a in reported if "Connection interrupted" in a[1]]
        assert retry_messages, "the retries were never reported"
        assert all("retrying y.mp3" in m for m in retry_messages)
        assert not any("resuming" in m or "0 bytes" in m for m in retry_messages)

    def test_retry_message_says_resuming_once_bytes_are_on_disk(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        payload = b"z" * 512
        chunks = [payload[:256], payload[256:]]
        first = _FakeResponse(chunks, status_code=200, headers={"content-length": "512"}, fail_after_chunks=1)
        second = _FakeResponse(
            chunks[1:],
            status_code=206,
            headers={"content-length": "256", "Content-Range": "bytes 256-511/512"},
        )
        TestDownloadResume._install(monkeypatch, [first, second])

        reported = []
        dl_core.download_file_with_progress(
            "https://example.com/archive.tar", tmp_path / "archive.tar", on_progress=lambda *a: reported.append(a)
        )

        retry_messages = [a[1] for a in reported if "Connection interrupted" in a[1]]
        assert retry_messages == [
            f"Connection interrupted at 256 bytes - resuming archive.tar (attempt 2/{dl_core._MAX_DOWNLOAD_ATTEMPTS})..."
        ]


class TestRetryableStatusExhaustion:
    """A retryable status the server never stops returning ends in the same
    actionable sentence a connection failure does (issue #3227).

    The Internet Archive answered HTTP 500 for one Apollo track on all six
    attempts; ``raise_for_status`` surfaced that as a raw ``HTTPError`` naming
    the redirect's data-node URL, which says neither which site failed nor that
    the failure was the server's rather than the user's.
    """

    def test_download_retries_a_500_then_names_the_host(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        responses = [_FakeResponse([], status_code=500) for _ in range(dl_core._MAX_DOWNLOAD_ATTEMPTS)]
        calls = TestDownloadResume._install(monkeypatch, responses)

        url = "https://archive.org/download/Apollo11Audio/11-03303.mp3"
        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            dl_core.download_file_with_progress(url, tmp_path / "11-03303.mp3", on_progress=lambda *a: None)

        # Every attempt was spent before giving up.
        assert len(calls) == dl_core._MAX_DOWNLOAD_ATTEMPTS
        message = str(excinfo.value)
        assert message.startswith("archive.org kept returning an internal server error (HTTP 500)")
        assert "server's side, not yours" in message
        # Not the raw HTTPError, and not a URL the user can't act on.
        assert url not in message
        assert excinfo.value.url == url
        assert excinfo.value.attempts == dl_core._MAX_DOWNLOAD_ATTEMPTS

    def test_a_recovering_server_still_downloads(self, tmp_path, monkeypatch):
        """The 500 path must stay a *retry*, not become a fail-fast."""
        from vtscore.datasets.downloader import core as dl_core

        payload = b"ID3" + b"\x00" * 61
        responses = [
            _FakeResponse([], status_code=503),
            _FakeResponse([], status_code=500),
            _FakeResponse([payload], status_code=200, headers={"content-length": str(len(payload))}),
        ]
        TestDownloadResume._install(monkeypatch, responses)

        dest = tmp_path / "11-03303.mp3"
        dl_core.download_file_with_progress(str("https://archive.org/x"), dest, on_progress=lambda *a: None)

        assert dest.read_bytes() == payload

    def test_rate_limit_reads_as_a_rate_limit(self, tmp_path, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        responses = [_FakeResponse([], status_code=429) for _ in range(dl_core._MAX_DOWNLOAD_ATTEMPTS)]
        TestDownloadResume._install(monkeypatch, responses)

        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            dl_core.download_file_with_progress(
                "https://huggingface.co/datasets/x/y.tar", tmp_path / "y.tar", on_progress=lambda *a: None
            )
        assert "a rate-limit refusal (HTTP 429)" in str(excinfo.value)

    def test_a_non_retryable_status_still_raises_http_error(self, tmp_path, monkeypatch):
        """404 is not the server wobbling - it is the URL being wrong, and it
        must not be dressed up as a transient outage."""
        import requests

        from vtscore.datasets.downloader import core as dl_core

        calls = TestDownloadResume._install(monkeypatch, [_FakeResponse([], status_code=404)])

        with pytest.raises(requests.HTTPError):
            dl_core.download_file_with_progress(
                "https://archive.org/download/x/gone.mp3", tmp_path / "gone.mp3", on_progress=lambda *a: None
            )
        assert len(calls) == 1, "a 404 must not burn the retry budget"

    def test_text_fetch_names_the_host_too(self, monkeypatch):
        from vtscore.datasets.downloader import core as dl_core

        responses = [_FakeResponse([], status_code=502) for _ in range(dl_core._MAX_DOWNLOAD_ATTEMPTS)]
        TestDownloadResume._install(monkeypatch, responses)

        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            dl_core.fetch_text_with_retry("https://archive.org/metadata/Apollo11Audio", on_progress=lambda *a: None)
        assert str(excinfo.value).startswith("archive.org kept returning a bad-gateway error (HTTP 502)")


class TestFetchTextWithRetry:
    """Manifest/index fetches share the transfer's retry budget and error shape.

    A one-shot GET for a few KB of JSON used to be strictly more fragile than
    the multi-GB download it precedes.
    """

    def test_returns_the_body_on_success(self, monkeypatch):
        import requests

        from vtscore.datasets.downloader import core as dl_core

        response = _FakeResponse([], status_code=200, text='{"files": []}')
        monkeypatch.setattr(requests.Session, "get", lambda self, url, *a, **k: response)

        assert dl_core.fetch_text_with_retry("https://archive.org/metadata/x") == '{"files": []}'
        assert response.closed

    def test_retries_a_transient_connect_failure(self, monkeypatch):
        import requests

        from vtscore.datasets.downloader import core as dl_core

        response = _FakeResponse([], status_code=200, text="ok")
        calls = []

        def fake_get(self, url, *args, **kwargs):
            calls.append(url)
            if len(calls) < 3:
                raise requests.exceptions.ConnectTimeout("connect timeout")
            return response

        monkeypatch.setattr(requests.Session, "get", fake_get)
        monkeypatch.setattr(dl_core.time, "sleep", lambda *a, **k: None)

        reported = []
        result = dl_core.fetch_text_with_retry(
            "https://archive.org/metadata/x", "the track list", lambda *a: reported.append(a)
        )

        assert result == "ok"
        assert len(calls) == 3
        # The retry notice names the fetch, not a filename it doesn't have.
        assert [a[1] for a in reported] == [
            f"Connection interrupted - retrying the track list (attempt {n}/{dl_core._MAX_DOWNLOAD_ATTEMPTS})..."
            for n in (2, 3)
        ]

    def test_exhausted_attempts_raise_the_named_host_error(self, monkeypatch):
        import requests

        from vtscore.datasets.downloader import core as dl_core

        def fake_get(self, url, *args, **kwargs):
            raise requests.exceptions.ConnectTimeout("connect timeout")

        monkeypatch.setattr(requests.Session, "get", fake_get)
        monkeypatch.setattr(dl_core.time, "sleep", lambda *a, **k: None)

        with pytest.raises(dl_core.RemoteUnreachableError) as excinfo:
            dl_core.fetch_text_with_retry("https://archive.org/metadata/x", on_progress=lambda *a: None)
        assert "archive.org" in str(excinfo.value)

"""Tests for the 20 Newsgroups archive prefetch.

The regression these guard against: ``download_20newsgroups`` used to delegate
the network fetch to ``sklearn.datasets.fetch_20newsgroups``, whose internal
``urlretrieve`` has no timeout and whose retry loop only catches
``URLError``/``TimeoutError``. A silently stalled connection therefore hung
forever with no progress ("Downloading source" stuck on step 1). The fix
pre-downloads the archive through ``download_file_with_progress`` (fail-fast
timeout, retries, byte-level progress) into sklearn's cache dir so the fetch
reuses the checksum-matching tar without touching the network.
"""

from types import SimpleNamespace
from unittest.mock import patch


def _archive_meta():
    from sklearn.datasets._twenty_newsgroups import ARCHIVE, CACHE_NAME

    return ARCHIVE, CACHE_NAME


class TestPrefetch20Newsgroups:
    def test_downloads_archive_via_timeout_guarded_helper(self, tmp_path):
        """A clean cache triggers download_file_with_progress into 20news_home."""
        from vtscore.datasets.downloader import text as text_module

        ARCHIVE, _ = _archive_meta()
        calls = []

        with (
            patch("sklearn.datasets.get_data_home", return_value=str(tmp_path)),
            patch.object(
                text_module._core,
                "download_file_with_progress",
                lambda url, dest, size, cb: calls.append((url, dest, size)),
            ),
        ):
            text_module._prefetch_20newsgroups_archive(on_progress=lambda *a: None)

        assert len(calls) == 1, "expected exactly one timeout-guarded download"
        url, dest, size = calls[0]
        assert url == ARCHIVE.url
        assert dest == tmp_path / "20news_home" / ARCHIVE.filename
        assert size > 0
        # The destination directory must be created before the download runs.
        assert (tmp_path / "20news_home").is_dir()

    def test_noop_when_pickle_cache_exists(self, tmp_path):
        """The parsed .pkz cache short-circuits the whole prefetch."""
        from vtscore.datasets.downloader import text as text_module

        _, cache_name = _archive_meta()
        (tmp_path / cache_name).write_bytes(b"cached")
        called = []

        with (
            patch("sklearn.datasets.get_data_home", return_value=str(tmp_path)),
            patch.object(
                text_module._core,
                "download_file_with_progress",
                lambda *a, **kw: called.append(True),
            ),
        ):
            text_module._prefetch_20newsgroups_archive(on_progress=lambda *a: None)

        assert not called, "download must be skipped when the pickle cache exists"

    def test_noop_when_archive_already_present(self, tmp_path):
        """An already-downloaded tar is left for sklearn to extract."""
        from vtscore.datasets.downloader import text as text_module

        ARCHIVE, _ = _archive_meta()
        twenty_home = tmp_path / "20news_home"
        twenty_home.mkdir()
        (twenty_home / ARCHIVE.filename).write_bytes(b"tar")
        called = []

        with (
            patch("sklearn.datasets.get_data_home", return_value=str(tmp_path)),
            patch.object(
                text_module._core,
                "download_file_with_progress",
                lambda *a, **kw: called.append(True),
            ),
        ):
            text_module._prefetch_20newsgroups_archive(on_progress=lambda *a: None)

        assert not called, "download must be skipped when the archive already exists"


class TestDownload20Newsgroups:
    def test_prefetch_runs_before_sklearn_fetch(self):
        """download_20newsgroups prefetches the archive before fetching."""
        from vtscore.datasets.downloader import text as text_module

        order = []

        def fake_prefetch(on_progress):
            order.append("prefetch")

        def fake_fetch(**kwargs):
            order.append("fetch")
            return SimpleNamespace(
                data=["a sports article", "a science article"],
                target=[0, 1],
                target_names=["rec.sport.baseball", "sci.space"],
            )

        with (
            patch.object(text_module, "_prefetch_20newsgroups_archive", fake_prefetch),
            patch("sklearn.datasets.fetch_20newsgroups", fake_fetch),
        ):
            texts, labels, names = text_module.download_20newsgroups(["sports", "science"], on_progress=lambda *a: None)

        assert order == ["prefetch", "fetch"], "prefetch must run before the sklearn fetch"
        assert texts == ["a sports article", "a science article"]
        assert names == ["sports", "science"]

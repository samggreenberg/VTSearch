"""HTTP-archive media source — access media files inside a remote archive.

Downloads the archive on first access, extracts it to a temporary directory,
and delegates all file operations to a :class:`LocalFolderSource` over the
extracted contents.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from uuid import uuid4

from vtsearch.config import DATA_DIR
from vtsearch.datasets.sources.base import MediaItem, MediaSource

if TYPE_CHECKING:
    from vtsearch.datasets.sources.local_folder import LocalFolderSource

__all__ = ["HttpArchiveSource"]

log = logging.getLogger(__name__)

_extract_lock = threading.Lock()


class HttpArchiveSource(MediaSource):
    """A media source backed by a remote archive (.zip, .tar.gz, etc.).

    The archive is downloaded and extracted lazily on first access to
    :meth:`list_items`, :meth:`fetch_item`, or :meth:`resolve_path`.

    Args:
        url: Public URL to the archive file.
    """

    name = "http_archive"

    def __init__(self, url: str) -> None:
        self._url = url
        self._extract_dir: Path | None = None
        self._inner: LocalFolderSource | None = None

    @property
    def url(self) -> str:
        return self._url

    # ------------------------------------------------------------------
    # Lazy materialisation
    # ------------------------------------------------------------------

    def _ensure_extracted(self) -> LocalFolderSource:
        """Download and extract the archive if not already done."""
        if self._inner is not None:
            return self._inner

        from vtsearch.datasets.sources.local_folder import LocalFolderSource

        url_filename = self._url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "archive"
        cached_dir = DATA_DIR / f"http_archive_resolve_{url_filename}"

        # Fast path: a previous run already published a cached extraction.
        if cached_dir.is_dir():
            self._extract_dir = cached_dir
            self._inner = LocalFolderSource(cached_dir)
            return self._inner

        from vtsearch.datasets.downloader import download_file_with_progress
        from vtsearch.datasets.importers.http_archive import _extract_archive
        from vtsearch.security.url_validation import validate_url

        validate_url(self._url)
        DATA_DIR.mkdir(exist_ok=True)

        # Download and extract to a unique temp directory. We deliberately do
        # this *outside* _extract_lock so concurrent imports of different URLs
        # don't serialise on the slow download.
        run_id = uuid4().hex[:12]
        archive_path = DATA_DIR / f"http_archive_download_{run_id}_{url_filename}"
        extract_dir = DATA_DIR / f"http_archive_source_{run_id}"

        try:
            log.info("Downloading %s for media source...", self._url)
            download_file_with_progress(self._url, archive_path)
            extract_dir.mkdir(exist_ok=True)
            _extract_archive(archive_path, extract_dir)
        finally:
            archive_path.unlink(missing_ok=True)

        # Publish the extraction as the cached dir. Re-check cached_dir under
        # the lock: two concurrent imports of the same URL can both pass the
        # earlier is_dir() check, and without this guard they'd both try to
        # rename onto the same destination (clobbering each other on POSIX,
        # raising on Windows). The loser discards its own extraction.
        with _extract_lock:
            if cached_dir.is_dir():
                shutil.rmtree(extract_dir, ignore_errors=True)
                final_dir = cached_dir
            else:
                try:
                    extract_dir.rename(cached_dir)
                    final_dir = cached_dir
                except OSError:
                    # e.g. cross-device rename; fall back to the unique dir.
                    final_dir = extract_dir

        self._extract_dir = final_dir
        self._inner = LocalFolderSource(final_dir)
        return self._inner

    # ------------------------------------------------------------------
    # MediaSource interface
    # ------------------------------------------------------------------

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        inner = self._ensure_extracted()
        for item in inner.list_items(extensions):
            yield MediaItem(
                key=item.key,
                filename=item.filename,
                source_name=self.name,
            )

    def fetch_item(self, key: str) -> Path | None:
        inner = self._ensure_extracted()
        return inner.fetch_item(key)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> Path | None:
        inner = self._ensure_extracted()
        return inner.resolve_path(origin_name, filename)

    def cleanup(self) -> None:
        """Remove the temporary extraction directory."""
        if self._extract_dir is not None and self._extract_dir.is_dir():
            # Only clean up directories we created (not cached ones).
            if "http_archive_source_" in self._extract_dir.name:
                shutil.rmtree(self._extract_dir, ignore_errors=True)
        self._extract_dir = None
        self._inner = None


class _HttpArchiveSourceFactory:
    """Factory for auto-discovery by :class:`~vtsearch.plugins.PluginRegistry`."""

    name = "http_archive"

    def create_from_origin(self, origin: dict) -> HttpArchiveSource | None:
        params = origin.get("params", {})
        url = params.get("url", "")
        return HttpArchiveSource(url) if url else None


SOURCE = _HttpArchiveSourceFactory()

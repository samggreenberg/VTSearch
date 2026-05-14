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

        with _extract_lock:
            if self._inner is not None:
                return self._inner

            from vtsearch.datasets.downloader import download_file_with_progress
            from vtsearch.datasets.importers.http_archive import _extract_archive
            from vtsearch.utils.url_validation import validate_url

            validate_url(self._url)
            DATA_DIR.mkdir(exist_ok=True)

            url_filename = self._url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "archive"
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

            self._extract_dir = extract_dir
            self._inner = LocalFolderSource(extract_dir)
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
            shutil.rmtree(self._extract_dir, ignore_errors=True)
        self._extract_dir = None
        self._inner = None


class _HttpArchiveSourceFactory:
    """Factory for auto-discovery by :class:`~vtsearch.utils.registry.PluginRegistry`."""

    name = "http_archive"

    def create_from_origin(self, origin: dict) -> HttpArchiveSource | None:
        params = origin.get("params", {})
        url = params.get("url", "")
        return HttpArchiveSource(url) if url else None


SOURCE = _HttpArchiveSourceFactory()

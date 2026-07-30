"""URL-download media source - re-fetch a single file from its public URL.

Backs origins stamped by the ``url_download`` datasource importer
(``{"importer": "url_download", "params": {"url": ...}}``), so an exemplar
fetched from a URL keeps a live origin: cross-dataset label resolution and
example-sort can re-download the file on demand instead of depending on the
local ``example_media/`` byte cache.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

__all__ = ["UrlDownloadSource"]

log = logging.getLogger(__name__)


def _filename_from_url(url: str) -> str:
    """Derive a filename (with its real extension) from *url*'s path.

    The suffix matters: it drives how downstream code (embedders, media
    decode) interprets the downloaded bytes.
    """
    name = Path(unquote(urlparse(url).path)).name
    return name or "download.bin"


class UrlDownloadSource(MediaSource):
    """A media source backed by a single public http(s) URL.

    The download happens lazily on first access, into a temporary directory
    owned by the source; :meth:`cleanup` removes it.  The stored URL was
    originally user-supplied, so it is re-validated with the same SSRF guard
    the ingress ran (:func:`~vtscore.security.url_validation.validate_url`)
    before any fetch; the downloader re-checks every redirect hop.
    """

    name = "url_download"

    def __init__(self, url: str) -> None:
        self._url = url
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        self._path: Path | None = None
        self._failed = False

    def _download(self) -> Path | None:
        """Fetch the URL once; return the local path, or ``None`` on failure."""
        if self._path is not None:
            return self._path
        if self._failed:
            return None

        from vtscore.datasets.downloader import download_file_with_progress
        from vtscore.security.url_validation import validate_url

        try:
            url = validate_url(self._url)
            self._tmp_dir = tempfile.TemporaryDirectory(prefix="vts_url_source_")
            dest = Path(self._tmp_dir.name) / _filename_from_url(url)
            download_file_with_progress(url, dest, on_progress=lambda *a, **k: None)
            if not dest.is_file() or dest.stat().st_size == 0:
                raise ValueError(f"The URL returned no data: {url}")
        except Exception:
            log.warning("UrlDownloadSource: could not fetch %r", self._url, exc_info=True)
            self._failed = True
            self.cleanup()
            return None

        self._path = dest
        return dest

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield the source's single item (without downloading it)."""
        filename = _filename_from_url(self._url)
        if extensions is not None and Path(filename).suffix.lower() not in {e.lower() for e in extensions}:
            return
        yield MediaItem(key=filename, filename=filename, source_name=self.name)

    def fetch_item(self, key: str) -> FetchedItem:  # noqa: ARG002 (single-item source; any key maps to the URL)
        return FetchedItem(path=self._download())

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        return FetchedItem(path=self._download())

    def cleanup(self) -> None:
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
        self._path = None


class _UrlDownloadSourceFactory:
    """Factory for auto-discovery by :class:`~vtscore.plugins.PluginRegistry`.

    Resolves origins emitted by the ``url_download`` datasource importer.
    """

    name = "url_download"

    def create_from_origin(self, origin: dict) -> UrlDownloadSource | None:
        url = origin.get("params", {}).get("url", "")
        return UrlDownloadSource(url) if url else None


SOURCE = _UrlDownloadSourceFactory()

"""HTTP-archive media source - access media files inside a remote archive.

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

from vtscore.config import DATA_DIR
from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

if TYPE_CHECKING:
    from vtscore.datasets.sources.local_folder import LocalFolderSource

__all__ = ["HttpArchiveSource"]

log = logging.getLogger(__name__)

_extract_lock = threading.Lock()


def _signature_path(cached_dir: Path) -> Path:
    """Sidecar path recording the remote signature (ETag/Last-Modified/size)
    observed at extraction time, so a later access can detect a changed
    remote archive and bust the cache instead of serving stale bytes forever.

    Kept as a *sibling* of the extraction dir, never inside it - a file
    inside would otherwise surface as a bogus media item to an
    extensionless ``list_items(None)`` call.
    """
    return cached_dir.parent / f"{cached_dir.name}.sig"


def _read_signature(cached_dir: Path) -> str | None:
    try:
        return _signature_path(cached_dir).read_text(encoding="utf-8")
    except OSError:
        return None


def _write_signature(cached_dir: Path, signature: str) -> None:
    try:
        _signature_path(cached_dir).write_text(signature, encoding="utf-8")
    except OSError:
        pass


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

        import hashlib

        from vtscore.datasets.downloader.core import fetch_remote_signature
        from vtscore.datasets.sources.local_folder import LocalFolderSource

        url_filename = self._url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "archive"
        # Key the cache on the *full* URL, not just its basename: two archives
        # whose URLs share a final segment (siteA/images.zip vs siteB/images.zip)
        # must not silently serve each other's extraction.  The basename is
        # kept in the dir name purely for human readability.
        url_hash = hashlib.sha256(self._url.encode("utf-8")).hexdigest()[:16]
        cached_dir = DATA_DIR / f"http_archive_resolve_{url_hash}_{url_filename}"

        # Fast path: a previous run already published a cached extraction.
        # Only bother probing the remote when the cache carries a recorded
        # signature to compare against - a cache with no sidecar (predates
        # this check, or its own probe failed) is trusted as-is rather than
        # invalidated on missing information. A probe failure (offline,
        # flaky CDN) also fails open onto the existing cache.
        remote_sig: str | None = None
        if cached_dir.is_dir():
            cached_sig = _read_signature(cached_dir)
            remote_sig = fetch_remote_signature(self._url) if cached_sig else None
            if cached_sig is None or remote_sig is None or remote_sig == cached_sig:
                self._extract_dir = cached_dir
                self._inner = LocalFolderSource(cached_dir)
                return self._inner
            log.info(
                "Remote archive changed for %s; cached extraction is stale, re-downloading",
                self._url,
            )

        from vtscore.datasets.archive import extract_archive
        from vtscore.datasets.downloader import download_file_with_progress
        from vtscore.security.url_validation import validate_url

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
            extract_archive(archive_path, extract_dir)
        finally:
            archive_path.unlink(missing_ok=True)

        if remote_sig is None:
            # No signature yet (fresh download, or the pre-download probe
            # failed) - one more attempt so a future access can detect drift.
            remote_sig = fetch_remote_signature(self._url)

        # Publish the extraction as the cached dir. Re-check cached_dir under
        # the lock: two concurrent imports of the same URL can both pass the
        # earlier is_dir() check, and without this guard they'd both try to
        # rename onto the same destination (clobbering each other on POSIX,
        # raising on Windows). The loser discards its own extraction.
        with _extract_lock:
            if cached_dir.is_dir():
                existing_sig = _read_signature(cached_dir)
                if remote_sig is not None and existing_sig is not None and existing_sig != remote_sig:
                    # A concurrent run published a now-stale extraction; evict it
                    # (and its sidecar) in favor of the fresh one we just built.
                    shutil.rmtree(cached_dir, ignore_errors=True)
                    _signature_path(cached_dir).unlink(missing_ok=True)
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
                if remote_sig is not None:
                    _write_signature(final_dir, remote_sig)

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

    def fetch_item(self, key: str) -> FetchedItem:
        inner = self._ensure_extracted()
        return inner.fetch_item(key)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
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
    """Factory for auto-discovery by :class:`~vtscore.plugins.PluginRegistry`."""

    name = "http_archive"

    def create_from_origin(self, origin: dict) -> HttpArchiveSource | None:
        params = origin.get("params", {})
        url = params.get("url", "")
        return HttpArchiveSource(url) if url else None


SOURCE = _HttpArchiveSourceFactory()

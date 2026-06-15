"""Local-archive media source – access media files inside an archive on disk.

Extracts the archive on first access (to a cached directory under
:data:`~vtscore.config.DATA_DIR`) and delegates all file operations to a
:class:`~vtscore.datasets.sources.local_folder.LocalFolderSource` over the
extracted contents.  This is the resolver counterpart of the ``local_archive``
origin emitted by :mod:`vtscore.datasets.archive`, so media imported from a
zip/tar re-derive their files on demand without ever persisting the extracted
bytes in the dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

if TYPE_CHECKING:
    from vtscore.datasets.sources.local_folder import LocalFolderSource

__all__ = ["LocalArchiveSource"]


class LocalArchiveSource(MediaSource):
    """A media source backed by a local archive file (.zip, .tar.gz, etc.).

    The archive is extracted lazily on first access to :meth:`list_items`,
    :meth:`fetch_item`, or :meth:`resolve_path`.  Extraction is cached, so
    repeated resolves of the same archive reuse one directory.

    Args:
        archive_path: Absolute path to the archive file on the server.
    """

    name = "local_archive"

    def __init__(self, archive_path: str | Path) -> None:
        self._archive_path = Path(archive_path)
        self._inner: LocalFolderSource | None = None

    @property
    def archive_path(self) -> Path:
        return self._archive_path

    def _ensure_extracted(self) -> LocalFolderSource:
        if self._inner is not None:
            return self._inner

        from vtscore.datasets.archive import extract_archive_cached  # noqa: PLC0415
        from vtscore.datasets.sources.local_folder import LocalFolderSource  # noqa: PLC0415

        extract_dir = extract_archive_cached(self._archive_path)
        self._inner = LocalFolderSource(extract_dir)
        return self._inner

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        inner = self._ensure_extracted()
        for item in inner.list_items(extensions):
            yield MediaItem(key=item.key, filename=item.filename, source_name=self.name)

    def fetch_item(self, key: str) -> FetchedItem:
        return self._ensure_extracted().fetch_item(key)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        return self._ensure_extracted().resolve_path(origin_name, filename)


class _LocalArchiveSourceFactory:
    """Factory for auto-discovery by :class:`~vtscore.plugins.PluginRegistry`.

    Resolves the ``local_archive`` origins emitted by
    :mod:`vtscore.datasets.archive` (used by the ``server_folder`` and
    ``http_archive`` importers when loading from a local archive).
    """

    name = "local_archive"

    def create_from_origin(self, origin: dict) -> LocalArchiveSource | None:
        params = origin.get("params", {})
        path = params.get("path", "")
        return LocalArchiveSource(path) if path else None


SOURCE = _LocalArchiveSourceFactory()

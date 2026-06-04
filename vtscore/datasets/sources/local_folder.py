"""Local-folder media source - access media files in a directory on disk."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from vtscore.datasets.sources.base import FetchedItem, MediaItem, MediaSource

__all__ = ["LocalFolderSource"]


class LocalFolderSource(MediaSource):
    """A media source backed by a local directory.

    Args:
        folder_path: Absolute path to the directory containing media files.
            May be a string or :class:`Path`.
    """

    name = "local_folder"

    def __init__(self, folder_path: str | Path) -> None:
        self._folder = Path(folder_path)

    @property
    def folder_path(self) -> Path:
        """The root directory of this source."""
        return self._folder

    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield media items for every file in the folder (recursively).

        Args:
            extensions: Optional list of lowercase extensions to include
                (e.g. ``[".wav", ".mp3"]``).  Dotted form required.
        """
        if not self._folder.is_dir():
            return

        ext_set = {e.lower() for e in extensions} if extensions else None

        for dirpath, _dirnames, filenames in os.walk(self._folder, followlinks=True):
            for fname in sorted(filenames):
                file_path = Path(dirpath) / fname
                if ext_set is not None and file_path.suffix.lower() not in ext_set:
                    continue

                key = file_path.relative_to(self._folder).as_posix()
                yield MediaItem(
                    key=key,
                    filename=file_path.name,
                    source_name=self.name,
                )

    def fetch_item(self, key: str) -> FetchedItem:
        """Return a :class:`FetchedItem` for *key* (a relative path within the folder).

        ``item.path`` is ``None`` if the file doesn't exist or escapes the root.
        """
        candidate = (self._folder / key).resolve()
        # Prevent path traversal
        try:
            candidate.relative_to(self._folder.resolve())
        except ValueError:
            return FetchedItem(path=None)
        return FetchedItem(path=candidate if candidate.is_file() else None)

    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        """Resolve by trying *origin_name* then *filename* as relative paths."""
        for name in [origin_name, filename]:
            if name:
                item = self.fetch_item(name)
                if item.path is not None:
                    return item
        return FetchedItem(path=None)


class _LocalFolderSourceFactory:
    """Factory for auto-discovery by :class:`~vtscore.plugins.PluginRegistry`.

    Resolves origins emitted by the :mod:`server_folder` dataset importer.
    """

    name = "server_folder"

    def create_from_origin(self, origin: dict) -> LocalFolderSource | None:
        params = origin.get("params", {})
        path = params.get("path", "")
        return LocalFolderSource(path) if path else None


SOURCE = _LocalFolderSourceFactory()

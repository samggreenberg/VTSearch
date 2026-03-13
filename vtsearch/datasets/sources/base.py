"""Base classes for the media source abstraction.

A :class:`MediaSource` describes how to access media files from a location
(local folder, HTTP archive, S3 bucket, etc.).  It provides three core
operations:

- **list_items** — enumerate available media files, optionally filtered by
  file extension.
- **fetch_item** — retrieve a single file by its key (relative path within
  the source), returning a local ``Path``.
- **resolve_path** — find a file by ``origin_name`` or ``filename``, used by
  the resolver module for cross-dataset label resolution.

Sources are instantiated per-use (not singletons) because they may carry
state such as downloaded archives or temporary extraction directories.
Callers should call :meth:`MediaSource.cleanup` when they're done.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

__all__ = ["MediaItem", "MediaSource"]


@dataclass(frozen=True)
class MediaItem:
    """A single discoverable media file within a source.

    Attributes:
        key: Unique identifier within the source, typically the relative
            path from the source root (e.g. ``"subdir/audio123.wav"``).
        filename: The basename of the file (e.g. ``"audio123.wav"``).
        source_name: The source type that produced this item (e.g.
            ``"local_folder"``, ``"http_archive"``).
    """

    key: str
    filename: str
    source_name: str


class MediaSource(ABC):
    """Abstract base class for media sources.

    Subclass this and implement the three abstract methods to add a new
    source type.  See :class:`~vtsearch.datasets.sources.local_folder.LocalFolderSource`
    for a minimal concrete example.
    """

    #: Short identifier for this source type (e.g. ``"local_folder"``).
    name: str = ""

    @abstractmethod
    def list_items(self, extensions: list[str] | None = None) -> Iterator[MediaItem]:
        """Yield all media items in this source.

        Args:
            extensions: Optional list of lowercase file extensions to filter
                by (e.g. ``[".wav", ".mp3"]``).  When ``None``, all files
                are yielded.

        Yields:
            :class:`MediaItem` instances for each matching file.
        """

    @abstractmethod
    def fetch_item(self, key: str) -> Path | None:
        """Return a local file path for the item identified by *key*.

        May trigger a download or extraction on demand.

        Args:
            key: The :attr:`MediaItem.key` (typically a relative path).

        Returns:
            A :class:`Path` to the file on disk, or ``None`` if the key
            does not exist in this source.
        """

    @abstractmethod
    def resolve_path(self, origin_name: str = "", filename: str = "") -> Path | None:
        """Resolve a media file by origin_name or filename.

        Tries *origin_name* first, then *filename*.  Used by the resolver
        module for cross-dataset label resolution.

        Args:
            origin_name: The ``origin_name`` stored on the media dict.
            filename: The ``filename`` stored on the media dict.

        Returns:
            A :class:`Path` to the resolved file, or ``None``.
        """

    def cleanup(self) -> None:
        """Release any temporary resources (extraction directories, etc.).

        The default implementation is a no-op.  Sources that create
        temporary files should override this.
        """

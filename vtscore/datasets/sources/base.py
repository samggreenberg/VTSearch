"""Base classes for the media source abstraction.

A :class:`MediaSource` describes how to access media files from a location
(local folder, HTTP archive, S3 bucket, etc.).  It provides five core
operations:

- **list_items** - enumerate available media files, optionally filtered by
  file extension.
- **fetch_item** - retrieve a single file by its key (relative path within
  the source), returning a local ``Path``.
- **fetch_items** - retrieve multiple files by key in one call; the default
  loops over :meth:`fetch_item`, but sources can override to parallelise
  (e.g. concurrent network downloads).
- **resolve_path** - find a file by ``origin_name`` or ``filename``, used by
  the resolver module for cross-dataset label resolution.
- **resolve_paths** - bulk form of :meth:`resolve_path`; returns a list
  aligned with the input sequence.  Override to parallelise.

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

    Subclass this and implement the three abstract methods (``list_items``,
    ``fetch_item``, ``resolve_path``) to add a new source type.  The two
    bulk methods (``fetch_items``, ``resolve_paths``) have default
    implementations that loop over their single-item counterparts; override
    them to parallelise I/O (e.g. concurrent network downloads).

    See :class:`~vtscore.datasets.sources.local_folder.LocalFolderSource`
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

    def fetch_items(self, keys: list[str]) -> dict[str, Path | None]:
        """Return local file paths for multiple *keys*.

        The default loops over :meth:`fetch_item`.  Override to parallelise
        (e.g. concurrent network downloads).

        Args:
            keys: The :attr:`MediaItem.key` values to fetch.

        Returns:
            Mapping from each input key to its resolved :class:`Path`, or
            ``None`` when the key does not exist in this source.  Keys not
            present in the source map to ``None``.
        """
        return {key: self.fetch_item(key) for key in keys}

    def resolve_paths(self, entries: list[tuple[str, str]]) -> list[Path | None]:
        """Resolve multiple ``(origin_name, filename)`` pairs.

        Returns a list aligned with *entries*: index *i* in the result
        corresponds to ``entries[i]``.  The default loops over
        :meth:`resolve_path`.  Override to parallelise.

        Args:
            entries: Sequence of ``(origin_name, filename)`` pairs.

        Returns:
            A list of resolved :class:`Path` objects (or ``None`` for
            entries that could not be found).
        """
        return [self.resolve_path(origin_name, filename) for origin_name, filename in entries]

    def cleanup(self) -> None:
        """Release any temporary resources (extraction directories, etc.).

        The default implementation is a no-op.  Sources that create
        temporary files should override this.
        """

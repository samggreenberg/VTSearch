"""Base classes for the media source abstraction.

A :class:`MediaSource` describes how to access media files from a location
(local folder, HTTP archive, S3 bucket, etc.).  It provides five core
operations:

- **list_items** - enumerate available media files, optionally filtered by
  file extension.
- **fetch_item** - retrieve a single file by its key; returns a
  :class:`FetchedItem` so sources can bundle pre-computed embeddings and
  metadata alongside the path.
- **fetch_items** - bulk form of :meth:`fetch_item`; the default loops, but
  sources can override to parallelise (e.g. one batch API call that returns
  files + vectors + metadata for all keys at once).
- **resolve_path** - find a file by ``origin_name`` or ``filename``, used by
  the resolver module for cross-dataset label resolution; returns a
  :class:`FetchedItem`.
- **resolve_paths** - bulk form of :meth:`resolve_path`; returns a list of
  :class:`FetchedItem` aligned with the input.  Override to parallelise.

Sources are instantiated per-use (not singletons) because they may carry
state such as downloaded archives or temporary extraction directories.
Callers should call :meth:`MediaSource.cleanup` when they're done.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import numpy as np

__all__ = ["FetchedItem", "MediaItem", "MediaSource"]


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


@dataclass
class FetchedItem:
    """Result of any fetch or resolve operation on a :class:`MediaSource`.

    All four fetch/resolve methods return ``FetchedItem``, so sources that
    have pre-computed data available (e.g. a remote API that returns the file
    bytes, an embedding, and file metadata in one round-trip) can surface it
    through both the single-item and bulk paths without loss.

    Attributes:
        path: Local file path, or ``None`` when the item could not be
            resolved/downloaded.
        embedding: Pre-computed embedding vector.  When set the ingest path
            skips re-embedding the file, provided the origin carries no clip
            params (clip embeddings must be derived from the clipped segment).
        embedder_name: Name of the embedder that produced *embedding*.  Must
            be set whenever *embedding* is provided so the vector can be
            matched against the dataset's embedding space.
        extra: Arbitrary source-provided metadata keyed by media-record field
            name (e.g. ``"file_size"``, ``"duration"``, ``"created_at"``).
            The ingest path merges these into the media record, letting remote
            sources supply authoritative values without a second file read.
    """

    path: Path | None
    embedding: np.ndarray | None = None
    embedder_name: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class MediaSource(ABC):
    """Abstract base class for media sources.

    Subclass this and implement the three abstract methods (``list_items``,
    ``fetch_item``, ``resolve_path``) to add a new source type.  All four
    fetch/resolve methods return :class:`FetchedItem` so pre-computed
    embeddings and metadata flow through single-item and bulk paths alike.

    The two bulk methods (``fetch_items``, ``resolve_paths``) have default
    implementations that loop over their single-item counterparts; override
    them to parallelise I/O and/or surface pre-computed data from a batch API.

    See :class:`~vtscore.datasets.sources.local_folder.LocalFolderSource`
    for a minimal concrete example and
    :class:`~vtscore.datasets.sources.pullwrest.PullWrestSource`
    for an example of a bulk override that returns rich metadata.
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
    def fetch_item(self, key: str) -> FetchedItem:
        """Return a :class:`FetchedItem` for the item identified by *key*.

        May trigger a download or extraction on demand.  Sources that have
        pre-computed embeddings or metadata available (e.g. from a remote
        API response) should populate the corresponding :class:`FetchedItem`
        fields rather than returning a bare path.

        Args:
            key: The :attr:`MediaItem.key` (typically a relative path).

        Returns:
            A :class:`FetchedItem`; ``item.path`` is ``None`` if the key
            does not exist in this source.
        """

    @abstractmethod
    def resolve_path(self, origin_name: str = "", filename: str = "") -> FetchedItem:
        """Resolve a media file by origin_name or filename.

        Tries *origin_name* first, then *filename*.  Used by the resolver
        module for cross-dataset label resolution.

        Args:
            origin_name: The ``origin_name`` stored on the media dict.
            filename: The ``filename`` stored on the media dict.

        Returns:
            A :class:`FetchedItem`; ``item.path`` is ``None`` when neither
            name resolves to an existing file.
        """

    def fetch_items(self, keys: list[str]) -> dict[str, FetchedItem]:
        """Fetch multiple items, returning a :class:`FetchedItem` per key.

        The default calls :meth:`fetch_item` once per key.  Override to
        parallelise downloads and/or return pre-computed embeddings and
        metadata from a batch API.

        Args:
            keys: The :attr:`MediaItem.key` values to fetch.

        Returns:
            Mapping from each input key to its :class:`FetchedItem`.
            ``item.path`` is ``None`` for keys not found in this source.
        """
        return {key: self.fetch_item(key) for key in keys}

    def resolve_paths(self, entries: list[tuple[str, str]]) -> list[FetchedItem]:
        """Resolve multiple ``(origin_name, filename)`` pairs.

        Returns a list aligned with *entries*: index *i* corresponds to
        ``entries[i]``.  The default calls :meth:`resolve_path` once per
        entry.  Override to parallelise and/or return pre-computed embeddings
        and metadata.

        Args:
            entries: Sequence of ``(origin_name, filename)`` pairs.

        Returns:
            A :class:`FetchedItem` per entry; ``item.path`` is ``None`` when
            the entry could not be resolved.
        """
        return [self.resolve_path(origin_name, filename) for origin_name, filename in entries]

    def cleanup(self) -> None:
        """Release any temporary resources (extraction directories, etc.).

        The default implementation is a no-op.  Sources that create
        temporary files should override this.
        """

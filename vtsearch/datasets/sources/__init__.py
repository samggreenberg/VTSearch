"""Media source abstraction — low-level access to media at a location.

``MediaSource`` is an optional composition layer that sits *below*
``DatasetImporter``.  Importers that access file-like sources (folder,
http_archive, future S3/SFTP) compose a ``MediaSource``.  Importers that
don't deal with individual files (pickle, combine_datasets) continue
working without one.

Use :func:`get_source_for_origin` to instantiate the right source for a
given origin dict.  Sources are **stateful** (e.g. an archive source may
download and extract on first access), so each call returns a fresh
instance — callers should call :meth:`~MediaSource.cleanup` when done.
"""

from __future__ import annotations

from typing import Any

from vtsearch.datasets.sources.base import MediaItem, MediaSource

__all__ = ["MediaItem", "MediaSource", "get_source_for_origin"]


def get_source_for_origin(origin: dict[str, Any] | None) -> MediaSource | None:
    """Instantiate the appropriate :class:`MediaSource` for *origin*.

    Returns ``None`` for origin types that don't map to a file-based source
    (e.g. pickle, combine_datasets, synthetic origins like dupe_set/converter).
    """
    if origin is None:
        return None

    importer = origin.get("importer", "")
    params = origin.get("params", {})

    if importer == "folder":
        path = params.get("path", "")
        if path:
            from vtsearch.datasets.sources.local_folder import LocalFolderSource

            return LocalFolderSource(path)

    if importer == "http_archive":
        url = params.get("url", "")
        if url:
            from vtsearch.datasets.sources.http_archive import HttpArchiveSource

            return HttpArchiveSource(url)

    return None

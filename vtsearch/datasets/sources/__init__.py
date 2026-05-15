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

Source factories are auto-discovered via the ``SOURCE`` sentinel attribute,
just like exporters and importers.
"""

from __future__ import annotations

from typing import Any

from vtsearch.datasets.sources.base import MediaItem, MediaSource
from vtsearch.plugins import PluginRegistry

__all__ = ["MediaItem", "MediaSource", "get_source_for_origin", "list_media_sources"]

_registry: PluginRegistry = PluginRegistry(
    package="vtsearch.datasets.sources",
    sentinel="SOURCE",
    label="media source",
    discover_modules=True,
    entry_point_group="vtsearch.media_sources",
)


def list_media_sources() -> list:
    """Return all registered media source factories."""
    return _registry.list()


def get_source_for_origin(origin: dict[str, Any] | None) -> MediaSource | None:
    """Instantiate the appropriate :class:`MediaSource` for *origin*.

    Returns ``None`` for origin types that don't map to a file-based source
    (e.g. pickle, combine_datasets, synthetic origins like dupe_set/converter).
    """
    if origin is None:
        return None

    importer = origin.get("importer", "")
    factory = _registry.get(importer)
    if factory is not None:
        return factory.create_from_origin(origin)
    return None

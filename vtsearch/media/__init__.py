"""Media type and embedder registries.

All built-in media types and embedders are registered at the bottom of this
module.  Third-party or project-specific types can be added by calling
:func:`register` / :func:`register_embedder` after importing this module::

    from vtsearch.media import register, register_embedder
    from mypackage.media_type import SourceCodeMediaType
    from mypackage.embedder import SourceCodeEmbedder

    register(SourceCodeMediaType())
    register_embedder(SourceCodeEmbedder())

The new type/embedder will then be picked up automatically by model
initialisation, dataset loading, HTTP routing, and the demo-dataset listing.
"""

from __future__ import annotations

from vtsearch.media.base import (
    DemoDataset,
    Detector,
    Extractor,
    Localizer,
    MediaClipper,
    MediaEmbedder,
    MediaResponse,
    MediaType,
    Processor,
    ProgressCallback,
)

# ------------------------------------------------------------------
# Media type registry
# ------------------------------------------------------------------

_registry: dict[str, "MediaType"] = {}


def register(media_type: "MediaType") -> None:
    """Add *media_type* to the registry, keyed by :attr:`~MediaType.type_id`."""
    _registry[media_type.type_id] = media_type


def get(type_id: str) -> "MediaType":
    """Return the :class:`MediaType` registered under *type_id*.

    Raises :class:`KeyError` if *type_id* is not registered.
    """
    if type_id not in _registry:
        raise KeyError(f"Unknown media type: {type_id!r}")
    return _registry[type_id]


def get_by_folder_name(folder_name: str) -> "MediaType":
    """Return the :class:`MediaType` whose :attr:`~MediaType.folder_import_name`
    matches *folder_name*.

    Raises :class:`KeyError` if no registered type has that folder name.
    """
    for mt in _registry.values():
        if mt.folder_import_name == folder_name:
            return mt
    raise KeyError(f"No media type with folder_import_name: {folder_name!r}")


def all_types() -> list["MediaType"]:
    """Return all registered :class:`MediaType` instances."""
    return list(_registry.values())


def get_by_extension(ext: str) -> "MediaType | None":
    """Return the :class:`MediaType` that handles files with extension *ext*.

    *ext* should include the leading dot (e.g. ``".wav"``).
    Returns ``None`` if no registered type handles that extension.
    """
    ext = ext.lower()
    for mt in _registry.values():
        for pattern in mt.file_extensions:
            # pattern looks like "*.wav" — extract the extension part
            if ext == pattern.lstrip("*"):
                return mt
    return None


def all_types_dict() -> list[dict]:
    """Return a list of JSON-serialisable dicts describing all registered types.

    Used by the ``/api/media-types`` endpoint so the frontend can render UI
    elements dynamically.
    """
    return [mt.to_dict() for mt in _registry.values()]


def all_demo_datasets() -> dict:
    """Return a flat ``{dataset_id: info_dict}`` mapping built from every
    registered media type's :attr:`~MediaType.demo_datasets` list.

    Each value is a dict with the keys expected by the datasets route:
    ``label``, ``description``, ``categories``, ``media_type``,
    optionally ``source``, and optionally ``required_folder``.
    """
    result: dict = {}
    for mt in _registry.values():
        for ds in mt.demo_datasets:
            entry: dict = {
                "label": ds.label,
                "description": ds.description,
                "categories": ds.categories,
                "media_type": mt.type_id,
                "slice_start": ds.slice_start,
                "slice_end": ds.slice_end,
                "download_size_mb": ds.download_size_mb,
            }
            if ds.source:
                entry["source"] = ds.source
            if ds.required_folder is not None:
                entry["required_folder"] = ds.required_folder
            result[ds.id] = entry
    return result


# ------------------------------------------------------------------
# Embedder registry
# ------------------------------------------------------------------

_embedder_registry: dict[str, "MediaEmbedder"] = {}


def register_embedder(embedder: "MediaEmbedder") -> None:
    """Add *embedder* to the registry, keyed by :attr:`~MediaEmbedder.name`."""
    _embedder_registry[embedder.name] = embedder


def get_embedder(name: str) -> "MediaEmbedder":
    """Return the :class:`MediaEmbedder` registered under *name*.

    Raises :class:`KeyError` if *name* is not registered.
    """
    if name not in _embedder_registry:
        raise KeyError(f"Unknown embedder: {name!r}")
    return _embedder_registry[name]


def embedders_for_type(type_id: str) -> list["MediaEmbedder"]:
    """Return all embedders registered for a given media type."""
    return [e for e in _embedder_registry.values() if e.media_type_id == type_id]


def all_embedders() -> list["MediaEmbedder"]:
    """Return all registered :class:`MediaEmbedder` instances."""
    return list(_embedder_registry.values())


def all_embedders_dict() -> list[dict]:
    """Return a list of JSON-serialisable dicts describing all registered embedders."""
    return [e.to_dict() for e in _embedder_registry.values()]


# ------------------------------------------------------------------
# Register all built-in media types
# ------------------------------------------------------------------

from vtsearch.media.audio.media_type import AudioMediaType  # noqa: E402
from vtsearch.media.document.media_type import DocumentMediaType  # noqa: E402
from vtsearch.media.image.media_type import ImageMediaType  # noqa: E402
from vtsearch.media.text.media_type import TextMediaType  # noqa: E402
from vtsearch.media.video.media_type import VideoMediaType  # noqa: E402

register(AudioMediaType())
register(VideoMediaType())
register(ImageMediaType())
register(TextMediaType())
register(DocumentMediaType())

# ------------------------------------------------------------------
# Register all built-in embedders
# ------------------------------------------------------------------

from vtsearch.media.audio.embedder import AudioClapEmbedder  # noqa: E402
from vtsearch.media.image.embedder import ImageClipEmbedder  # noqa: E402
from vtsearch.media.text.embedder import TextE5Embedder  # noqa: E402
from vtsearch.media.video.embedder import VideoXClipEmbedder  # noqa: E402

register_embedder(AudioClapEmbedder())
register_embedder(ImageClipEmbedder())
register_embedder(TextE5Embedder())
register_embedder(VideoXClipEmbedder())


def set_progress_callback(callback: "ProgressCallback") -> None:
    """Set the progress callback on all registered media types and embedders.

    Call this once at application startup to wire media types and embedders
    into whatever progress reporting mechanism the host application uses.
    """
    for mt in _registry.values():
        mt._on_progress = callback
    for emb in _embedder_registry.values():
        emb._on_progress = callback


__all__ = [
    "MediaType",
    "MediaEmbedder",
    "MediaClipper",
    "MediaResponse",
    "DemoDataset",
    "Processor",
    "Detector",
    "Localizer",
    "Extractor",
    "ProgressCallback",
    "register",
    "register_embedder",
    "get",
    "get_embedder",
    "get_by_folder_name",
    "get_by_extension",
    "all_types",
    "all_types_dict",
    "all_demo_datasets",
    "all_embedders",
    "all_embedders_dict",
    "embedders_for_type",
    "set_progress_callback",
]

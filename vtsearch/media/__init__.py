"""Media type, embedder, and clipper registries.

Built-in media types, embedders, and clippers are **auto-discovered** at
import time by scanning sub-packages of ``vtsearch.media`` for sentinel
attributes:

- ``MEDIA_TYPE`` — a single :class:`MediaType` instance.
- ``EMBEDDERS`` — a list of :class:`MediaEmbedder` instances (may be empty).
- ``CLIPPERS``  — a list of :class:`MediaClipper` instances (may be empty).

To add a new media type (or embedder / clipper), create a sub-package under
``vtsearch/media/`` with an ``__init__.py`` that exposes the relevant
sentinels.  Symlinked directories are supported.

Third-party or project-specific types can still be registered manually::

    from vtsearch.media import register, register_embedder
    register(MyCustomMediaType())
    register_embedder(MyCustomEmbedder())
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

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

def normalize_type_id(type_id: str) -> str:
    """Validate that *type_id* is a known canonical type name.

    Returns *type_id* unchanged (legacy aliases have been removed).
    """
    return type_id


def register(media_type: "MediaType") -> None:
    """Add *media_type* to the registry, keyed by :attr:`~MediaType.type_id`."""
    _registry[media_type.type_id] = media_type


def get(type_id: str) -> "MediaType":
    """Return the :class:`MediaType` registered under *type_id*.

    Raises :class:`KeyError` if *type_id* is not registered.
    """
    type_id = normalize_type_id(type_id)
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


def all_folder_names() -> list[str]:
    """Return the :attr:`~MediaType.folder_import_name` of every registered type.

    Used by dataset importers to populate their media-type selection fields
    dynamically, so adding a new media type to the registry is all that's
    needed — no importer code changes required.
    """
    return [mt.folder_import_name for mt in _registry.values()]


def all_type_ids() -> list[str]:
    """Return the :attr:`~MediaType.type_id` of every registered type.

    Used by settings validation so the set of valid media types stays in
    sync with the registry automatically.
    """
    return list(_registry.keys())


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
                "slice_frac_start": ds.slice_frac_start,
                "slice_frac_end": ds.slice_frac_end,
                "download_size_mb": ds.download_size_mb,
            }
            if ds.source:
                entry["source"] = ds.source
            if ds.required_folder is not None:
                entry["required_folder"] = ds.required_folder
            result[ds.id] = entry
    return result


# ------------------------------------------------------------------
# Clipper registry
# ------------------------------------------------------------------

_clipper_registry: dict[str, "MediaClipper"] = {}


def register_clipper(clipper: "MediaClipper") -> None:
    """Add *clipper* to the registry, keyed by :attr:`~MediaClipper.name`."""
    _clipper_registry[clipper.name] = clipper


def get_clipper(name: str) -> "MediaClipper":
    """Return the :class:`MediaClipper` registered under *name*.

    Raises :class:`KeyError` if *name* is not registered.
    """
    if name not in _clipper_registry:
        raise KeyError(f"Unknown clipper: {name!r}")
    return _clipper_registry[name]


def clippers_for_type(type_id: str) -> list["MediaClipper"]:
    """Return all clippers registered for a given media type."""
    return [c for c in _clipper_registry.values() if c.media_type == type_id]


def all_clippers() -> list["MediaClipper"]:
    """Return all registered :class:`MediaClipper` instances."""
    return list(_clipper_registry.values())


def all_clippers_dict() -> list[dict]:
    """Return a list of JSON-serialisable dicts describing all registered clippers."""
    return [c.to_dict() for c in _clipper_registry.values()]


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
# Auto-discover media types, embedders, and clippers
# ------------------------------------------------------------------


def _discover_media_plugins() -> None:
    """Scan sub-packages of ``vtsearch.media`` for sentinel attributes.

    Each sub-package (directory with ``__init__.py``) may expose:

    - ``MEDIA_TYPE`` — a single :class:`MediaType` instance.
    - ``EMBEDDERS`` — a list of :class:`MediaEmbedder` instances.
    - ``CLIPPERS``  — a list of :class:`MediaClipper` instances.

    Symlinked directories are followed (``entry.is_dir()`` resolves
    symlinks), so an external media-type package can be symlinked into
    this directory and will be discovered automatically.
    """
    package_dir = Path(__file__).parent
    for entry in sorted(package_dir.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if not entry.is_dir() or "." in entry.name or not (entry / "__init__.py").exists():
            continue
        try:
            mod = importlib.import_module(f"vtsearch.media.{entry.name}")
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load media sub-package '{entry.name}': {exc}",
                stacklevel=2,
            )
            continue

        mt = getattr(mod, "MEDIA_TYPE", None)
        if mt is not None:
            register(mt)
        for emb in getattr(mod, "EMBEDDERS", []):
            register_embedder(emb)
        for clip in getattr(mod, "CLIPPERS", []):
            register_clipper(clip)


_discover_media_plugins()


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
    "register_clipper",
    "get",
    "get_embedder",
    "get_clipper",
    "get_by_folder_name",
    "get_by_extension",
    "all_types",
    "all_folder_names",
    "all_type_ids",
    "all_types_dict",
    "all_demo_datasets",
    "all_embedders",
    "all_embedders_dict",
    "all_clippers",
    "all_clippers_dict",
    "embedders_for_type",
    "clippers_for_type",
    "normalize_type_id",
    "set_progress_callback",
]

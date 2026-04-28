"""Media type, embedder, and clipper registries.

Built-in media types, embedders, and clippers are **auto-discovered** at
import time by scanning sub-packages of ``vtsearch.media``:

- Each media-type sub-package (``audio/``, ``image/``, ...) exposes a
  ``MEDIA_TYPE`` sentinel in its ``__init__.py`` (plus a ``CLIPPERS`` list).
- Each embedder lives under the media-type package as either a flat
  module (e.g. ``vtsearch/media/audio/embedder_clap_music.py``) or a
  sub-package (e.g. ``vtsearch/media/image/embedder_fancy/__init__.py``)
  and exposes a module-level ``EMBEDDER`` sentinel — one embedder per
  module or sub-package.  Any ``embedder*.py`` file or ``embedder*/``
  directory with an ``__init__.py`` found inside a media-type package
  is auto-loaded.

To add a new embedder, drop an ``embedder_<name>.py`` file — or an
``embedder_<name>/`` sub-package containing an ``__init__.py`` — into
the appropriate media-type package with an ``EMBEDDER`` sentinel at the
module/package top level.  Symlinked directories and symlinked embedder
files are both supported, so custom embedders living outside the
VTSearch tree can be wired in without editing any package ``__init__.py``.

Third-party or project-specific types can still be registered manually::

    from vtsearch.media import register, register_embedder
    register(MyCustomMediaType())
    register_embedder(MyCustomEmbedder())
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
)
from vtsearch.media.clipper import MediaClipper
from vtsearch.media.embedder import MediaEmbedder
from vtsearch.media.processors import (
    Detector,
    Extractor,
    Localizer,
    Processor,
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
                "items_per_category": ds.items_per_category,
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


def _discover_embedders_in(media_type_dir: Path, package_name: str) -> None:
    """Scan a media-type sub-package for modules or sub-packages exposing an ``EMBEDDER``.

    Any ``embedder*.py`` file **or** ``embedder*/`` sub-package (directory
    containing an ``__init__.py``) is imported, and its module-level
    ``EMBEDDER`` attribute is registered if present.  Symlinked files and
    symlinked directories are both loaded via
    :func:`importlib.util.spec_from_file_location` so that symlinks
    pointing outside the package are handled reliably (mirrors the same
    approach used by :class:`vtsearch.utils.registry.PluginRegistry`).
    """
    for entry in sorted(media_type_dir.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if not entry.name.startswith("embedder"):
            continue

        # Flat module: embedder_<name>.py
        if entry.is_file() and entry.suffix == ".py":
            module_stem = entry.stem
            load_path = entry
            is_package = False
        # Sub-package: embedder_<name>/__init__.py.  Skip names containing
        # dots — they aren't valid Python identifiers and would be
        # misinterpreted as nested module paths by importlib.
        elif entry.is_dir() and "." not in entry.name and (entry / "__init__.py").exists():
            module_stem = entry.name
            load_path = entry / "__init__.py"
            is_package = True
        else:
            continue

        full_name = f"{package_name}.{module_stem}"
        try:
            if entry.is_symlink():
                resolved = load_path.resolve()
                spec = importlib.util.spec_from_file_location(
                    full_name,
                    str(resolved),
                    submodule_search_locations=[str(resolved.parent)] if is_package else None,
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)
            else:
                mod = importlib.import_module(full_name)
        except Exception as exc:  # pragma: no cover
            sys.modules.pop(full_name, None)
            warnings.warn(
                f"Failed to load embedder module '{full_name}': {exc}",
                stacklevel=2,
            )
            continue
        emb = getattr(mod, "EMBEDDER", None)
        if emb is not None:
            register_embedder(emb)


def _discover_media_plugins() -> None:
    """Scan sub-packages of ``vtsearch.media`` for sentinel attributes.

    Each media-type sub-package (directory with ``__init__.py``) may expose:

    - ``MEDIA_TYPE`` — a single :class:`MediaType` instance.
    - ``CLIPPERS``  — a list of :class:`MediaClipper` instances.

    Embedders are auto-discovered per module: every ``embedder*.py`` file
    inside a media-type sub-package is scanned for an ``EMBEDDER`` sentinel
    (see :func:`_discover_embedders_in`).

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
        package_name = f"vtsearch.media.{entry.name}"
        try:
            mod = importlib.import_module(package_name)
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load media sub-package '{entry.name}': {exc}",
                stacklevel=2,
            )
            continue

        mt = getattr(mod, "MEDIA_TYPE", None)
        if mt is not None:
            register(mt)
        for clip in getattr(mod, "CLIPPERS", []):
            register_clipper(clip)
        _discover_embedders_in(entry, package_name)


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

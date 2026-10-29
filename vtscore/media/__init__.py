"""Media type, embedder, clipper, and cleaner registries.

Built-in media types, embedders, clippers, and cleaners are
**auto-discovered** at import time by scanning sub-packages of
``vtscore.media``:

- Each media-type sub-package (``audio/``, ``image/``, ...) exposes a
  ``MEDIA_TYPE`` sentinel in its ``__init__.py`` (plus optional
  ``CLIPPERS`` and ``CLEANERS`` lists).
- Each embedder lives under the media-type package as either a flat
  module (e.g. ``vtscore/media/audio/embedder_clap_music.py``) or a
  sub-package (e.g. ``vtscore/media/image/embedder_fancy/__init__.py``)
  and exposes a module-level ``EMBEDDER`` sentinel - one embedder per
  module or sub-package.  Any ``embedder*.py`` file or ``embedder*/``
  directory with an ``__init__.py`` found inside a media-type package
  is auto-loaded, as is one found inside that package's ``embedders/``
  container (where ``image/`` keeps its own).

To add a new embedder, drop an ``embedder_<name>.py`` file - or an
``embedder_<name>/`` sub-package containing an ``__init__.py`` - into
the appropriate media-type package with an ``EMBEDDER`` sentinel at the
module/package top level.  Symlinked directories and symlinked embedder
files are both supported, so custom embedders living outside the
VTSearch tree can be wired in without editing any package ``__init__.py``.

Third-party or project-specific types can still be registered manually::

    from vtscore.media import register, register_cleaner, register_clipper, register_embedder
    register(MyCustomMediaType())
    register_embedder(MyCustomEmbedder())
    register_clipper(MyCustomClipper())
    register_cleaner(MyCustomCleaner())
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from pathlib import Path

from vtscore.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
)
from vtscore.media.clipping.cleaner import MediaCleaner
from vtscore.media.clipping.clipper import MediaClipper
from vtscore.media.embedder import MediaEmbedder

# ------------------------------------------------------------------
# Media type registry
# ------------------------------------------------------------------

_registry: dict[str, "MediaType"] = {}


def normalize_type_id(type_id: str) -> str:
    """Return *type_id* unchanged.

    A pass-through: there are no alias spellings to fold, and it does not
    validate (:func:`get` raises :class:`KeyError` for an unknown id).
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
            # pattern looks like "*.wav" - extract the extension part
            if ext == pattern.lstrip("*"):
                return mt
    return None


def all_folder_names() -> list[str]:
    """Return the :attr:`~MediaType.folder_import_name` of every registered type.

    Used by dataset importers to populate their media-type selection fields
    dynamically, so adding a new media type to the registry is all that's
    needed - no importer code changes required.
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
# Cleaner registry
# ------------------------------------------------------------------

#: Cleaners live in their own registry so they never surface in a clipper
#: chooser: a clipper is a radio choice (pick one per import), a cleaner is a
#: checkbox every item of that type passes through.  Registration order is
#: also *run* order (the chain runner appends enabled cleaners in this order);
#: shipped cleaners are expected to be order-insensitive in practice.
_cleaner_registry: dict[str, "MediaCleaner"] = {}


def register_cleaner(cleaner: "MediaCleaner") -> None:
    """Add *cleaner* to the registry, keyed by :attr:`~MediaCleaner.name`."""
    _cleaner_registry[cleaner.name] = cleaner


def get_cleaner(name: str) -> "MediaCleaner":
    """Return the :class:`MediaCleaner` registered under *name*.

    Raises :class:`KeyError` if *name* is not registered.
    """
    if name not in _cleaner_registry:
        raise KeyError(f"Unknown cleaner: {name!r}")
    return _cleaner_registry[name]


def cleaners_for_type(type_id: str) -> list["MediaCleaner"]:
    """Return all cleaners registered for a given media type, in run order."""
    return [c for c in _cleaner_registry.values() if c.media_type == type_id]


def all_cleaners() -> list["MediaCleaner"]:
    """Return all registered :class:`MediaCleaner` instances."""
    return list(_cleaner_registry.values())


def all_cleaners_dict() -> list[dict]:
    """Return a list of JSON-serialisable dicts describing all registered cleaners."""
    return [c.to_dict() for c in _cleaner_registry.values()]


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
    """Return the **user-selectable** embedders registered for a media type.

    The default embedder (``is_default == True``) is sorted first so callers
    using ``embedders_for_type(t)[0]`` receive the default.  Remaining order
    follows registry insertion (which is alphabetical filename order).

    Embedders declaring :attr:`~vtscore.media.embedder.MediaEmbedder.eval_only`
    are withheld.  This function is what every picker, every "available
    embedders" error message and every per-media-type default is built from, so
    filtering here is what keeps a research arm out of the app; resolution by
    name (:func:`get_embedder`) is deliberately *not* filtered, because a
    dataset embedded by an eval arm still has to load.
    """
    matches = [e for e in _embedder_registry.values() if e.media_type_id == type_id and not e.eval_only]
    return sorted(matches, key=lambda e: not e.is_default)


def all_embedders() -> list["MediaEmbedder"]:
    """Return all registered :class:`MediaEmbedder` instances."""
    return list(_embedder_registry.values())


def all_embedders_dict() -> list[dict]:
    """Return JSON-serialisable dicts for every **user-selectable** embedder.

    This is what ``GET /api/embedders`` serialises, so it applies the same
    :attr:`~vtscore.media.embedder.MediaEmbedder.eval_only` filter as
    :func:`embedders_for_type`.  Use :func:`all_embedders` when you want the
    registry itself (docs inventories, name validation, the eval harness).
    """
    return [e.to_dict() for e in _embedder_registry.values() if not e.eval_only]


def embedder_for_medias(media_dict: dict) -> "MediaEmbedder | None":
    """Return the appropriate embedder for the given medias, or ``None``.

    Looks up the ``"embedder"`` name stored on the first media entry; falls
    back to the default embedder for its ``"media_type"``.

    Lives here (rather than in the routes layer that first grew it) so the
    dataset-load pipeline can resolve an embedder without importing
    ``vtsearch.routes`` - which would drag Flask into a library code path.
    """
    if not media_dict:
        return None
    first = next(iter(media_dict.values()))
    embedder_name = first.get("embedder", "")
    media_type = first.get("media_type", "audio")

    if embedder_name:
        try:
            return get_embedder(embedder_name)
        except KeyError:
            pass

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


# ------------------------------------------------------------------
# Old module paths
# ------------------------------------------------------------------

#: Public modules that moved into sub-packages (#5401), old dotted path → new.
#: :class:`_MovedModuleFinder` keeps extension code written against an old path
#: importing, with a :class:`DeprecationWarning`.  The private (``_``-prefixed)
#: modules that moved alongside them have no entry.
_MOVED_MODULES: dict[str, str] = {
    "vtscore.media.clipper": "vtscore.media.clipping.clipper",
    "vtscore.media.cleaner": "vtscore.media.clipping.cleaner",
    "vtscore.media.clip_recipe": "vtscore.media.clipping.clip_recipe",
    "vtscore.media.lazy_clip": "vtscore.media.clipping.lazy_clip",
    "vtscore.media.structural_geometry": "vtscore.media.structural.structural_geometry",
    "vtscore.media.structural_splg": "vtscore.media.structural.structural_splg",
    "vtscore.media.structural_tiles": "vtscore.media.structural.structural_tiles",
    "vtscore.media.torch_ops": "vtscore.media.torch_utils.torch_ops",
    "vtscore.media.torch_setup": "vtscore.media.torch_utils.torch_setup",
    **{
        f"vtscore.media.image.{stem}": f"vtscore.media.image.embedders.{stem}"
        for stem in (
            "embedder_clip",
            "embedder_clip_l",
            "embedder_dinov2_patch",
            "embedder_dinov2_single",
            "embedder_dinov3_patch",
            "embedder_dinov3_single",
            "embedder_embeddinggemma2",
            "embedder_eupe_patch",
            "embedder_eupe_single",
            "embedder_sift_vlad",
            "embedder_sift_vlad_doc",
            "embedder_siglip",
            "embedder_siglip2",
            "embedder_siglip2_l",
            "embedder_siglip_l",
        )
    },
}


class _MovedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve an old :data:`_MOVED_MODULES` path to the module now at the new one.

    Appended to :data:`sys.meta_path`, so it only answers for a name no real
    file claims.  Loading swaps the placeholder out of :data:`sys.modules` for
    the real module, so both names are one module object: nothing (an
    ``EMBEDDER`` sentinel included) executes twice, and a ``mock.patch``
    through the old path patches the module the code actually runs.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _MOVED_MODULES:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        new_name = _MOVED_MODULES[module.__name__]
        warnings.warn(
            f"{module.__name__} has moved to {new_name}; import it from there",
            DeprecationWarning,
            stacklevel=2,
        )
        sys.modules[module.__name__] = importlib.import_module(new_name)


# Installed before discovery runs, so a symlinked-in plugin that imports an
# old path while it is being discovered still resolves.
sys.meta_path.append(_MovedModuleFinder())


# ------------------------------------------------------------------
# Auto-discover media types, embedders, clippers, and cleaners
# ------------------------------------------------------------------


def _load_symlinked_embedder(full_name: str, load_path: Path, is_package: bool):
    """Load a symlinked embedder module or package from the file its link points at."""
    resolved = load_path.resolve()
    spec = importlib.util.spec_from_file_location(
        full_name,
        str(resolved),
        submodule_search_locations=[str(resolved.parent)] if is_package else None,
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _discover_embedders_in(media_type_dir: Path, package_name: str) -> None:
    """Scan a media-type sub-package for modules or sub-packages exposing an ``EMBEDDER``.

    Any ``embedder*.py`` file **or** ``embedder*/`` sub-package (directory
    containing an ``__init__.py``) is imported, and its module-level
    ``EMBEDDER`` attribute is registered if present.  Symlinked files and
    symlinked directories are both loaded via
    :func:`importlib.util.spec_from_file_location` so that symlinks
    pointing outside the package are handled reliably (mirrors the same
    approach used by :class:`vtscore.plugins.PluginRegistry`).

    The one exception is a sub-package named exactly ``embedders/``: it is a
    *container* (``vtscore/media/image/embedders/`` holds the image embedders),
    so it is scanned by these same rules instead of being loaded as one embedder.
    """
    for entry in sorted(media_type_dir.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if not entry.name.startswith("embedder"):
            continue

        if entry.name == "embedders" and entry.is_dir() and (entry / "__init__.py").exists():
            _discover_embedders_in(entry, f"{package_name}.embedders")
            continue

        # Flat module: embedder_<name>.py
        if entry.is_file() and entry.suffix == ".py":
            module_stem = entry.stem
            load_path = entry
            is_package = False
        # Sub-package: embedder_<name>/__init__.py.  Skip names containing
        # dots - they aren't valid Python identifiers and would be
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
                mod = _load_symlinked_embedder(full_name, load_path, is_package)
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
    """Scan sub-packages of ``vtscore.media`` for sentinel attributes.

    Each media-type sub-package (directory with ``__init__.py``) may expose:

    - ``MEDIA_TYPE`` - a single :class:`MediaType` instance.
    - ``CLIPPERS``  - a list of :class:`MediaClipper` instances.
    - ``CLEANERS``  - a list of :class:`MediaCleaner` instances.

    Embedders are auto-discovered per module: every ``embedder*.py`` file
    inside a media-type sub-package (or its ``embedders/`` container) is
    scanned for an ``EMBEDDER`` sentinel (see :func:`_discover_embedders_in`).

    The shared, non-media-type sub-packages (``clipping/``, ``structural/``,
    ``torch_utils/``) are imported by the same scan; they expose no sentinels,
    so nothing registers.

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
        package_name = f"vtscore.media.{entry.name}"
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
        for cleaner in getattr(mod, "CLEANERS", []):
            register_cleaner(cleaner)
        _discover_embedders_in(entry, package_name)


_discover_media_plugins()


def set_progress_callback(callback: "ProgressCallback") -> None:
    """Set the progress callback on all registered media types and embedders.

    Call this once at application startup to wire media types and embedders
    into whatever progress reporting mechanism the host application uses.

    Embedders go through :meth:`~vtscore.media.embedder.MediaEmbedder.set_default_progress_callback`
    rather than a plain ``_on_progress`` assignment: that attribute is
    thread-scoped (so concurrent dataset loads can't trample each other's
    tracker), whereas this startup wiring is a process-wide default every
    thread must see.
    """
    for mt in _registry.values():
        mt._on_progress = callback
    for emb in _embedder_registry.values():
        emb.set_default_progress_callback(callback)


__all__ = [
    "MediaType",
    "MediaEmbedder",
    "MediaClipper",
    "MediaCleaner",
    "MediaResponse",
    "DemoDataset",
    "ProgressCallback",
    "register",
    "register_embedder",
    "register_clipper",
    "register_cleaner",
    "get",
    "get_embedder",
    "get_clipper",
    "get_cleaner",
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
    "all_cleaners",
    "all_cleaners_dict",
    "embedders_for_type",
    "clippers_for_type",
    "cleaners_for_type",
    "normalize_type_id",
    "set_progress_callback",
]

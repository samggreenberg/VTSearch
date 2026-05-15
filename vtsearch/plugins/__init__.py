"""Generic plugin registry with auto-discovery.

Provides :class:`PluginRegistry` — a reusable registry that scans a package
directory for sub-packages exposing a sentinel attribute, and
:class:`PluginField` / :class:`PluginBase` — shared base types that eliminate
the duplicated field-dataclass and CLI / serialisation boilerplate across the
four plugin families (dataset importers, exporters, label importers, processor
importers).

Usage — creating a registry::

    from vtsearch.plugins import PluginRegistry

    _registry: PluginRegistry[MyPlugin] = PluginRegistry(
        package="vtsearch.exporters",
        sentinel="EXPORTER",
        label="exporter",
    )

    get_exporter    = _registry.get
    list_exporters  = _registry.list

Usage — defining a field / base class::

    from vtsearch.plugins import PluginBase, PluginField

    class MyExporter(PluginBase):
        ...
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, TypeVar

FieldType = Literal["file", "folder", "url", "text", "password", "email", "select", "server_path", "checkbox"]

__all__ = [
    "FieldType",
    "PluginBase",
    "PluginField",
    "PluginRegistry",
    "make_plugin_registry",
]


# ---------------------------------------------------------------------------
# PluginField — shared field descriptor
# ---------------------------------------------------------------------------


@dataclass
class PluginField:
    """Describes a single configurable input for a plugin.

    The ``field_type`` value drives how the frontend renders it:

    - ``"file"``     – OS file-picker; value arrives as a Werkzeug
      :class:`~werkzeug.datastructures.FileStorage` object.
    - ``"folder"``   – Path text-input or OS folder-picker.
    - ``"url"``      – Text input pre-validated as a URL.
    - ``"text"``     – Generic single-line text input.
    - ``"password"`` – Text input whose characters are masked.
    - ``"email"``    – Text input pre-validated as an e-mail address.
    - ``"select"``   – Drop-down; ``options`` must be populated (or
      :attr:`dynamic_options` set, in which case options are fetched at
      runtime from the plugin's ``get_field_options`` method).
    - ``"server_path"`` – File-browser picker for server filesystem paths.
    - ``"checkbox"`` – Boolean tick-box.  ``default`` should be ``"true"`` or
      ``"false"``; values arrive at :meth:`run` as plain strings (or already
      coerced bools) and should be parsed via ``str(value).lower() == "true"``.

    Dynamic option fields
    ---------------------
    Set ``dynamic_options=True`` on a ``"select"`` field whose options must
    be computed at runtime — e.g. by querying a remote service.  The plugin
    must implement ``get_field_options(field_key, current_values)`` to return
    the list.  The frontend re-fetches options every time any field listed
    in :attr:`depends_on` changes value.  Currently honoured by dataset
    importers (``POST /api/dataset/import/<name>/options``); other plugin
    families may opt in similarly.
    """

    key: str
    label: str
    field_type: FieldType
    description: str = ""
    #: For ``"file"`` fields: comma-separated extensions, e.g. ``".pkl"``.
    accept: str = ""
    #: For ``"select"`` fields: the list of allowed values.
    options: list[str] = field(default_factory=list)
    #: Pre-filled default value shown in the UI.
    default: str = ""
    required: bool = True
    #: Hint shown as placeholder text inside the input widget.
    placeholder: str = ""
    #: When ``True``, :attr:`options` is computed at runtime by the plugin's
    #: ``get_field_options(field_key, current_values)`` method.  Static
    #: :attr:`options` (if any) are still served as the initial list.
    dynamic_options: bool = False
    #: Field keys whose values this field's options depend on.  When any
    #: listed field changes, the frontend re-fetches options for this field.
    #: Only meaningful when :attr:`dynamic_options` is ``True``.
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "field_type": self.field_type,
            "description": self.description,
            "accept": self.accept,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "placeholder": self.placeholder,
            "dynamic_options": self.dynamic_options,
            "depends_on": list(self.depends_on),
        }


# ---------------------------------------------------------------------------
# PluginBase — shared base class with CLI & serialisation helpers
# ---------------------------------------------------------------------------


class PluginBase:
    """Mixin providing the CLI-argument, validation, and serialisation helpers
    that are identical across all four plugin families."""

    #: Internal snake_case identifier used in API routes.
    name: str
    #: Human-readable label shown in the UI.
    display_name: str
    #: One-sentence description shown as a subtitle in the UI.
    description: str
    #: Emoji or icon string shown next to the display name.
    icon: str = ""
    #: Ordered list of fields the user must fill.
    fields: list[PluginField]

    #: How the frontend should render this plugin's UI.
    #: ``"form"`` — generic form built from :attr:`fields` (default).
    #: ``"file_upload"`` — the frontend should use its native file picker.
    #: ``"custom"`` — the plugin has a dedicated UI section in the frontend.
    #: ``"none"`` — no user-facing UI (e.g. the GUI exporter is handled
    #:   automatically by the frontend results view).
    ui_mode: str = "form"

    #: When ``True``, this plugin is excluded from the generic picker list
    #: in the frontend.  Useful for plugins that are always invoked through
    #: a dedicated code path (e.g. the GUI exporter).
    hidden_from_picker: bool = False

    def resolve_display_name(self, field_values: dict[str, Any]) -> str:
        """Return a human-readable name for a dataset loaded with *field_values*.

        The default returns :attr:`display_name`.  Subclasses (e.g. the demo
        importer) can override this to return a dataset-specific label.
        """
        return self.display_name

    # -- CLI support --------------------------------------------------------

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register this plugin's fields as ``argparse`` arguments.

        The default implementation converts each :class:`PluginField` into a
        CLI flag (e.g. a field with ``key="media_type"`` becomes
        ``--media-type``).  ``"select"`` fields gain a ``choices`` constraint.
        """
        for f in self.fields:
            arg_name = f"--{f.key.replace('_', '-')}"
            kwargs: dict[str, Any] = {
                "dest": f.key,
                "help": f.description or f.label,
            }
            if f.field_type == "checkbox":
                # ``--<key>`` / ``--no-<key>`` boolean flag.
                kwargs["action"] = argparse.BooleanOptionalAction
                kwargs["default"] = str(f.default).lower() == "true"
                parser.add_argument(arg_name, **kwargs)
                continue
            if f.default:
                kwargs["default"] = f.default
            if f.field_type == "select" and f.options:
                kwargs["choices"] = f.options
            parser.add_argument(arg_name, **kwargs)

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required field is missing or empty."""
        for f in self.fields:
            # Booleans are always populated by argparse (default included).
            if f.field_type == "checkbox":
                continue
            if f.required and not field_values.get(f.key):
                cli_flag = f"--{f.key.replace('_', '-')}"
                raise ValueError(f"Missing required argument: {cli_flag}")

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise plugin metadata for API endpoints."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "fields": [f.to_dict() for f in self.fields],
            "ui_mode": self.ui_mode,
            "hidden_from_picker": self.hidden_from_picker,
        }


# ---------------------------------------------------------------------------
# PluginRegistry — generic auto-discovery registry
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Auto-discovering plugin registry.

    Parameters
    ----------
    package:
        Fully-qualified dotted name of the package whose sub-packages will be
        scanned, e.g. ``"vtsearch.exporters"``.
    sentinel:
        Module-level attribute name to look for in each sub-package, e.g.
        ``"EXPORTER"``.
    label:
        Human-readable noun used in warning messages, e.g. ``"exporter"``.
    discover_modules:
        When ``True``, also scan flat ``.py`` files (not just sub-packages)
        for the sentinel.  Useful for plugin families where each plugin is
        a single module rather than a sub-package (e.g. converters, sources).
    """

    def __init__(self, package: str, sentinel: str, label: str, *, discover_modules: bool = False) -> None:
        self._package = package
        self._sentinel = sentinel
        self._label = label
        self._discover_modules = discover_modules
        self._items: dict[str, T] = {}
        self._discovered = False
        self._discovering = False
        self._lock = threading.Lock()

    # -- Discovery ----------------------------------------------------------

    def _discover(self) -> None:
        """Scan sub-packages (and optionally flat modules) for sentinel objects.

        Uses direct filesystem scanning so that symlinked directories are
        reliably discovered.  A symlink to a package directory (containing
        ``__init__.py``) is treated identically to a regular sub-package.

        When :attr:`_discover_modules` is ``True``, also scans ``.py`` files
        (excluding ``__init__.py`` and ``base.py``) for the sentinel.
        """
        parent = importlib.import_module(self._package)
        package_dir = Path(parent.__file__).parent
        for entry in sorted(package_dir.iterdir()):
            if entry.name.startswith((".", "_")):
                continue

            # Sub-packages (directories with __init__.py)
            if entry.is_dir():
                # Skip names containing dots — they aren't valid Python
                # identifiers and would be misinterpreted as nested module
                # paths by importlib (e.g. "foo.symbolic_link" would try to
                # import package "foo" first).  This commonly happens with
                # symlinks whose names include an extension or suffix.
                if "." in entry.name:
                    continue
                init_path = entry / "__init__.py"
                if not init_path.exists():
                    continue
                self._try_load(entry.name, file_path=init_path if entry.is_symlink() else None)
            # Flat modules (.py files)
            elif self._discover_modules and entry.is_file() and entry.suffix == ".py":
                if entry.name in ("__init__.py", "base.py"):
                    continue
                self._try_load(entry.stem, file_path=entry if entry.is_symlink() else None)

    def _try_load(self, module_name: str, *, file_path: Path | None = None) -> None:
        """Import *module_name* under this registry's package and register its sentinel.

        When *file_path* is given (symlinked entries), uses
        :func:`importlib.util.spec_from_file_location` to load the module
        directly from the resolved path.  Python's default ``FileFinder`` can
        miss symlinked packages on some platforms because its directory cache
        may not follow symlinks consistently.
        """
        full_name = f"{self._package}.{module_name}"
        try:
            if file_path is not None:
                resolved = file_path.resolve()
                is_package = resolved.name == "__init__.py"
                spec = importlib.util.spec_from_file_location(
                    full_name,
                    str(resolved),
                    submodule_search_locations=[str(resolved.parent)] if is_package else None,
                )
                if spec is None or spec.loader is None:  # pragma: no cover
                    return
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)
            else:
                mod = importlib.import_module(full_name)
            plugin = getattr(mod, self._sentinel, None)
            if plugin is not None:
                self._items[plugin.name] = plugin
        except Exception as exc:  # pragma: no cover
            # Clean up partially-registered module on failure.
            sys.modules.pop(full_name, None)
            warnings.warn(
                f"Failed to load {self._label} '{module_name}': {exc}",
                stacklevel=2,
            )

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        with self._lock:
            if not self._discovered:
                # Guard against re-entrant discovery.  When discover_modules
                # is True, importing a sibling module may trigger get()/list()
                # on this registry before discovery finishes (e.g. runner.py
                # importing from its own package's __init__).  In that case
                # we return early with a partial registry — the ongoing
                # discovery will complete shortly and fill in the rest.
                if self._discovering:
                    return
                self._discovering = True
                try:
                    self._discover()
                finally:
                    self._discovering = False
                self._discovered = True

    # -- Public API ---------------------------------------------------------

    def get(self, name: str) -> T | None:
        """Return the registered plugin with *name*, or ``None``."""
        self._ensure_discovered()
        return self._items.get(name)

    def list(self) -> list[T]:
        """Return all registered plugins in discovery order."""
        self._ensure_discovered()
        return list(self._items.values())


# ---------------------------------------------------------------------------
# Factory helper — collapses the per-package boilerplate into one call
# ---------------------------------------------------------------------------


def make_plugin_registry(
    package: str,
    sentinel: str,
    label: str,
    *,
    discover_modules: bool = False,
) -> tuple[Callable[[str], Any], Callable[[], list[Any]]]:
    """Create a :class:`PluginRegistry` and return its ``(get, list)`` accessors.

    Shorthand for the boilerplate repeated across every plugin ``__init__.py``::

        from vtsearch.plugins import make_plugin_registry

        get_importer, list_importers = make_plugin_registry(
            package=__name__,
            sentinel="IMPORTER",
            label="dataset importer",
        )

    Parameters are forwarded to :class:`PluginRegistry`.
    """
    registry: PluginRegistry = PluginRegistry(
        package=package,
        sentinel=sentinel,
        label=label,
        discover_modules=discover_modules,
    )
    return registry.get, registry.list

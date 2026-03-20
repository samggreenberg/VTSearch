"""Generic plugin registry with auto-discovery.

Provides :class:`PluginRegistry` — a reusable registry that scans a package
directory for sub-packages exposing a sentinel attribute, and
:class:`PluginField` / :class:`PluginBase` — shared base types that eliminate
the duplicated field-dataclass and CLI / serialisation boilerplate across the
four plugin families (dataset importers, exporters, label importers, processor
importers).

Usage — creating a registry::

    from vtsearch.utils.registry import PluginRegistry

    _registry: PluginRegistry[MyPlugin] = PluginRegistry(
        package="vtsearch.exporters",
        sentinel="EXPORTER",
        label="exporter",
    )

    get_exporter    = _registry.get
    list_exporters  = _registry.list

Usage — defining a field / base class::

    from vtsearch.utils.registry import PluginBase, PluginField

    class MyExporter(PluginBase):
        ...
"""

from __future__ import annotations

import argparse
import importlib
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

FieldType = Literal["file", "folder", "url", "text", "password", "email", "select"]

__all__ = [
    "FieldType",
    "PluginBase",
    "PluginField",
    "PluginRegistry",
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
    - ``"select"``   – Drop-down; ``options`` must be populated.
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
            if f.default:
                kwargs["default"] = f.default
            if f.field_type == "select" and f.options:
                kwargs["choices"] = f.options
            parser.add_argument(arg_name, **kwargs)

    def validate_cli_field_values(self, field_values: dict[str, Any]) -> None:
        """Raise ``ValueError`` if any required field is missing or empty."""
        for f in self.fields:
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
    """

    def __init__(self, package: str, sentinel: str, label: str) -> None:
        self._package = package
        self._sentinel = sentinel
        self._label = label
        self._items: dict[str, T] = {}
        self._discovered = False
        self._lock = threading.Lock()

    # -- Discovery ----------------------------------------------------------

    def _discover(self) -> None:
        """Scan sub-packages for sentinel objects and register them.

        Uses direct filesystem scanning instead of :func:`pkgutil.iter_modules`
        so that symlinked directories are reliably discovered.  A symlink to a
        package directory (containing ``__init__.py``) is treated identically to
        a regular sub-package.
        """
        parent = importlib.import_module(self._package)
        package_dir = Path(parent.__file__).parent
        for entry in sorted(package_dir.iterdir()):
            # Accept both real directories and symlinks that resolve to
            # directories.  entry.is_dir() follows symlinks by default.
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            # Must contain __init__.py to be a proper Python package.
            if not (entry / "__init__.py").exists():
                continue
            module_name = entry.name
            try:
                mod = importlib.import_module(f"{self._package}.{module_name}")
                plugin = getattr(mod, self._sentinel, None)
                if plugin is not None:
                    self._items[plugin.name] = plugin
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"Failed to load {self._label} '{module_name}': {exc}",
                    stacklevel=2,
                )

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        with self._lock:
            if not self._discovered:
                self._discover()
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

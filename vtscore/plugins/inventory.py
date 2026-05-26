"""Centralised plugin inventory.

Gathers every plugin family auto-discovered by VTSearch into a single
data structure for the ``python app.py --list-plugins`` CLI and any other
tooling that wants a cross-family view of what's installed.

The inventory covers both ``PluginRegistry``-backed families (importers,
exporters, label sources, settings I/O, converters, media sources) and
the embedder / clipper / media-type registries that live directly on
:mod:`vtscore.media`.

Families are registered with :func:`register_plugin_family`.  Library
families self-register at module import; app-only families
(``settings_io/*``) are injected by the application layer via
:mod:`vtsearch.shim` so this module stays free of cross-boundary imports
ahead of the ``vtscore`` library split (see
``../docs/architecture.md`` Phase 5).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class PluginEntry:
    """A single plugin's identity for inventory purposes."""

    name: str
    display_name: str
    description: str
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
        }
        if self.extra:
            out.update(self.extra)
        return out


def _entry_from_plugin(plugin: Any, *, extra: dict[str, Any] | None = None) -> PluginEntry:
    return PluginEntry(
        name=getattr(plugin, "name", repr(plugin)),
        display_name=getattr(plugin, "display_name", "") or "",
        description=getattr(plugin, "description", "") or "",
        extra=extra or {},
    )


def _safe_list(loader: Callable[[], Iterable[Any]]) -> list[Any]:
    """Run *loader* and swallow ImportError-class failures.

    A missing optional dependency in one plugin shouldn't block inventory
    of every other family.  Anything more serious is left to bubble up so
    the caller sees the real bug.
    """
    try:
        return list(loader())
    except (ImportError, ModuleNotFoundError):
        return []


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------


@dataclass
class FamilyProvider:
    """How to enumerate one plugin family for the inventory.

    Attributes
    ----------
    key:
        Stable snake_case identifier (e.g. ``"importers"``).  Used as the
        dict key in :func:`gather_plugins`, the CLI ``--plugin-family``
        argument, and the ``--list-<key>`` shortcut.
    label:
        Human-readable family heading used in plain-text output.
    loader:
        Zero-argument callable returning the raw plugins.  Wrapped in
        :func:`_safe_list` so ``ImportError`` from optional deps degrades
        gracefully.
    entry_builder:
        Maps one raw plugin to a :class:`PluginEntry`.  Defaults to
        :func:`_entry_from_plugin` (no extras).
    """

    key: str
    label: str
    loader: Callable[[], Iterable[Any]]
    entry_builder: Callable[[Any], PluginEntry] = field(default=_entry_from_plugin)


_FAMILIES_REGISTRY: dict[str, FamilyProvider] = {}


def register_plugin_family(provider: FamilyProvider) -> None:
    """Register *provider* so :func:`gather_plugins` enumerates it.

    Subsequent calls with the same :attr:`FamilyProvider.key` replace the
    earlier registration (last writer wins).  Insertion order is preserved
    so the plain-text output stays grouped as the registrar intended.
    """
    _FAMILIES_REGISTRY[provider.key] = provider


def _converter_entry(plugin: Any) -> PluginEntry:
    return _entry_from_plugin(
        plugin,
        extra={
            "source_type": getattr(plugin, "source_type", ""),
            "target_type": getattr(plugin, "target_type", ""),
        },
    )


def _media_type_entry(mt: Any) -> PluginEntry:
    return PluginEntry(
        name=getattr(mt, "type_id", ""),
        display_name=getattr(mt, "display_name", "") or getattr(mt, "type_id", ""),
        description=getattr(mt, "description", "") or "",
        extra={"folder_import_name": getattr(mt, "folder_import_name", "")},
    )


def _embedder_entry(emb: Any) -> PluginEntry:
    return PluginEntry(
        name=getattr(emb, "name", ""),
        display_name=getattr(emb, "display_name", "") or getattr(emb, "name", ""),
        description=getattr(emb, "description", "") or "",
        extra={
            "media_type": getattr(emb, "media_type_id", ""),
            "is_default": bool(getattr(emb, "is_default", False)),
        },
    )


def _clipper_entry(clip: Any) -> PluginEntry:
    return PluginEntry(
        name=getattr(clip, "name", ""),
        display_name=getattr(clip, "display_name", "") or getattr(clip, "name", ""),
        description=getattr(clip, "description", "") or "",
        extra={"media_type": getattr(clip, "media_type", "")},
    )


# Lazy loaders for the library-tier families.  Each one imports its
# package on first call so ``python app.py --list-plugins`` doesn't pay
# the full app startup cost - Flask blueprints, model registries, etc.
# only get imported when their family is asked for.


def _load_importers() -> Iterable[Any]:
    from vtscore.datasets.importers import list_importers

    return list_importers()


def _load_exporters() -> Iterable[Any]:
    from vtscore.exporters import list_exporters

    return list_exporters()


def _load_label_importers() -> Iterable[Any]:
    from vtscore.labels.importers import list_label_importers

    return list_label_importers()


def _load_labelset_sources() -> Iterable[Any]:
    from vtscore.labels.sources import list_labelset_sources

    return list_labelset_sources()


def _load_converters() -> Iterable[Any]:
    from vtscore.converters import list_converters

    return list_converters()


def _load_media_sources() -> Iterable[Any]:
    from vtscore.datasets.sources import list_media_sources

    return list_media_sources()


def _load_media_types() -> Iterable[Any]:
    from vtscore.media import all_types

    return all_types()


def _load_embedders() -> Iterable[Any]:
    from vtscore.media import all_embedders

    return all_embedders()


def _load_clippers() -> Iterable[Any]:
    from vtscore.media import all_clippers

    return all_clippers()


# App-only families (settings importers/exporters/sources) are NOT
# registered here; the app installs them at startup via
# :func:`vtsearch.shim.register_app_plugin_families`.
_LIBRARY_FAMILIES: tuple[FamilyProvider, ...] = (
    FamilyProvider("importers", "Dataset importers", _load_importers),
    FamilyProvider("exporters", "Results exporters", _load_exporters),
    FamilyProvider("label_importers", "Label importers", _load_label_importers),
    FamilyProvider("labelset_sources", "Labelset sources", _load_labelset_sources),
    FamilyProvider("converters", "Media converters", _load_converters, _converter_entry),
    FamilyProvider("media_sources", "Media sources", _load_media_sources),
    FamilyProvider("media_types", "Media types", _load_media_types, _media_type_entry),
    FamilyProvider("embedders", "Media embedders", _load_embedders, _embedder_entry),
    FamilyProvider("clippers", "Media clippers", _load_clippers, _clipper_entry),
)


for _provider in _LIBRARY_FAMILIES:
    register_plugin_family(_provider)
del _provider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gather_plugins() -> dict[str, list[PluginEntry]]:
    """Return ``{family: [PluginEntry, ...]}`` for every registered plugin.

    Family keys are stable, snake-cased identifiers suitable for shell
    completion scripts; ordering matches discovery order within a family
    (alphabetical by file name for built-ins, then entry-point order).
    The set of families is whatever :func:`register_plugin_family` has
    populated by the time this is called.
    """
    inventory: dict[str, list[PluginEntry]] = {}
    for key, provider in _FAMILIES_REGISTRY.items():
        inventory[key] = [provider.entry_builder(p) for p in _safe_list(provider.loader)]
    return inventory


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_plain(inventory: dict[str, list[PluginEntry]]) -> str:
    """Render *inventory* as a human-readable plain-text listing."""
    out: list[str] = []
    for family, entries in inventory.items():
        provider = _FAMILIES_REGISTRY.get(family)
        label = provider.label if provider is not None else family
        out.append(f"{label} ({len(entries)}):")
        if not entries:
            out.append("  (none)")
        else:
            width = max(len(e.name) for e in entries)
            for entry in entries:
                line = f"  {entry.name.ljust(width)}"
                if entry.display_name and entry.display_name != entry.name:
                    line += f"  {entry.display_name}"
                out.append(line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def format_names(inventory: dict[str, list[PluginEntry]], family: str | None = None) -> str:
    """Render plugin *names* one per line - designed for shell completion.

    When *family* is given, only that family's names are emitted; otherwise
    every family is emitted as ``<family>:<name>`` so a completion script
    can split on the first colon.
    """
    if family is not None:
        entries = inventory.get(family, [])
        return "\n".join(e.name for e in entries) + ("\n" if entries else "")
    lines: list[str] = []
    for fam, entries in inventory.items():
        for entry in entries:
            lines.append(f"{fam}:{entry.name}")
    return "\n".join(lines) + ("\n" if lines else "")


def format_json(inventory: dict[str, list[PluginEntry]]) -> str:
    """Render *inventory* as a JSON object."""
    import json

    return json.dumps(
        {family: [entry.to_dict() for entry in entries] for family, entries in inventory.items()},
        indent=2,
        sort_keys=False,
    )


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class _ListFamilyAction(argparse.Action):
    """Argparse action backing the ``--list-<family>`` shortcuts.

    Sets ``list_plugins=True`` and ``plugin_family=<family>`` on the
    namespace so the existing ``--list-plugins`` early-exit branch picks
    them up unchanged.
    """

    def __init__(self, option_strings, dest, const=None, default=None, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.list_plugins = True
        namespace.plugin_family = self.const


def family_flag(family: str) -> str:
    """Return the CLI flag name for *family* (e.g. ``label_importers`` → ``--list-label-importers``)."""
    return f"--list-{family.replace('_', '-')}"


def register_family_shortcuts(parser: argparse.ArgumentParser) -> None:
    """Add ``--list-<family>`` shortcut flags to *parser* for every registered family.

    Each shortcut is equivalent to ``--list-plugins --plugin-family
    <family>`` and obeys the same ``--format`` setting.  Reads the current
    family registry, so any family registered via
    :func:`register_plugin_family` automatically gets a shortcut.
    """
    for family in _FAMILIES_REGISTRY:
        parser.add_argument(
            family_flag(family),
            action=_ListFamilyAction,
            const=family,
            dest="list_plugins",
            help=f"Shortcut for --list-plugins --plugin-family {family}.",
        )


def __getattr__(name: str) -> Any:
    """Module-level dynamic attributes.

    ``FAMILIES`` is exposed as a tuple snapshot of the current registry
    keys at access time, so callers that ``from vtscore.plugins.inventory
    import FAMILIES`` see every family that's been registered by then -
    including app-only ones the shim installs at startup.
    """
    if name == "FAMILIES":
        return tuple(_FAMILIES_REGISTRY.keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FAMILIES",  # noqa: F822 - exposed dynamically via module __getattr__
    "FamilyProvider",
    "PluginEntry",
    "family_flag",
    "format_json",
    "format_names",
    "format_plain",
    "gather_plugins",
    "register_family_shortcuts",
    "register_plugin_family",
]

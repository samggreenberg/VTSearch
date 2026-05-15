"""Centralised plugin inventory.

Gathers every plugin family auto-discovered by VTSearch into a single
data structure for the ``python app.py --list-plugins`` CLI and any other
tooling that wants a cross-family view of what's installed.

The inventory covers both ``PluginRegistry``-backed families (importers,
exporters, label sources, settings I/O, converters, media sources) and
the embedder / clipper / media-type registries that live directly on
:mod:`vtsearch.media`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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


def _safe_list(loader: Callable[[], list[Any]]) -> list[Any]:
    """Run *loader* and swallow ImportError-class failures.

    A missing optional dependency in one plugin shouldn't block inventory
    of every other family.  Anything more serious is left to bubble up so
    the caller sees the real bug.
    """
    try:
        return list(loader())
    except (ImportError, ModuleNotFoundError):
        return []


def gather_plugins() -> dict[str, list[PluginEntry]]:
    """Return ``{family: [PluginEntry, ...]}`` for every registered plugin.

    Family keys are stable, snake-cased identifiers suitable for shell
    completion scripts; ordering matches discovery order within a family
    (alphabetical by file name for built-ins, then entry-point order).
    """
    # Importers are looked up lazily so that ``python app.py --list-plugins``
    # doesn't pay the full app startup cost — Flask blueprints, model
    # registries, etc. only get imported when their family is asked for.
    from vtsearch.converters import list_converters
    from vtsearch.datasets.importers import list_importers
    from vtsearch.datasets.sources import list_media_sources
    from vtsearch.exporters import list_exporters
    from vtsearch.labels.importers import list_label_importers
    from vtsearch.labels.sources import list_labelset_sources
    from vtsearch.media import all_clippers, all_embedders, all_types
    from vtsearch.settings_io.exporters import list_settings_exporters
    from vtsearch.settings_io.importers import list_settings_importers
    from vtsearch.settings_io.sources import list_settings_sources

    inventory: dict[str, list[PluginEntry]] = {}

    inventory["importers"] = [_entry_from_plugin(p) for p in _safe_list(list_importers)]
    inventory["exporters"] = [_entry_from_plugin(p) for p in _safe_list(list_exporters)]
    inventory["label_importers"] = [_entry_from_plugin(p) for p in _safe_list(list_label_importers)]
    inventory["labelset_sources"] = [_entry_from_plugin(p) for p in _safe_list(list_labelset_sources)]
    inventory["settings_importers"] = [_entry_from_plugin(p) for p in _safe_list(list_settings_importers)]
    inventory["settings_exporters"] = [_entry_from_plugin(p) for p in _safe_list(list_settings_exporters)]
    inventory["settings_sources"] = [_entry_from_plugin(p) for p in _safe_list(list_settings_sources)]
    inventory["converters"] = [
        _entry_from_plugin(
            c,
            extra={
                "source_type": getattr(c, "source_type", ""),
                "target_type": getattr(c, "target_type", ""),
            },
        )
        for c in _safe_list(list_converters)
    ]
    inventory["media_sources"] = [_entry_from_plugin(p) for p in _safe_list(list_media_sources)]

    # Media-side registries — embedders / clippers / media types don't
    # use PluginBase, so build entries from their attribute surface.
    inventory["media_types"] = [
        PluginEntry(
            name=getattr(mt, "type_id", ""),
            display_name=getattr(mt, "display_name", "") or getattr(mt, "type_id", ""),
            description=getattr(mt, "description", "") or "",
            extra={"folder_import_name": getattr(mt, "folder_import_name", "")},
        )
        for mt in _safe_list(all_types)
    ]
    inventory["embedders"] = [
        PluginEntry(
            name=getattr(emb, "name", ""),
            display_name=getattr(emb, "display_name", "") or getattr(emb, "name", ""),
            description=getattr(emb, "description", "") or "",
            extra={
                "media_type": getattr(emb, "media_type_id", ""),
                "is_default": bool(getattr(emb, "is_default", False)),
            },
        )
        for emb in _safe_list(all_embedders)
    ]
    inventory["clippers"] = [
        PluginEntry(
            name=getattr(clip, "name", ""),
            display_name=getattr(clip, "display_name", "") or getattr(clip, "name", ""),
            description=getattr(clip, "description", "") or "",
            extra={"media_type": getattr(clip, "media_type", "")},
        )
        for clip in _safe_list(all_clippers)
    ]

    return inventory


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


_FAMILY_LABELS: dict[str, str] = {
    "importers": "Dataset importers",
    "exporters": "Results exporters",
    "label_importers": "Label importers",
    "labelset_sources": "Labelset sources",
    "settings_importers": "Settings importers",
    "settings_exporters": "Settings exporters",
    "settings_sources": "Settings sources",
    "converters": "Media converters",
    "media_sources": "Media sources",
    "media_types": "Media types",
    "embedders": "Media embedders",
    "clippers": "Media clippers",
}


def format_plain(inventory: dict[str, list[PluginEntry]]) -> str:
    """Render *inventory* as a human-readable plain-text listing."""
    out: list[str] = []
    for family, entries in inventory.items():
        label = _FAMILY_LABELS.get(family, family)
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
    """Render plugin *names* one per line — designed for shell completion.

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


FAMILIES: tuple[str, ...] = tuple(_FAMILY_LABELS.keys())

__all__ = [
    "FAMILIES",
    "PluginEntry",
    "gather_plugins",
    "format_plain",
    "format_names",
    "format_json",
]

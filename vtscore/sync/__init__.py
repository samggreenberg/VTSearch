"""Shared base for bidirectional sync sources.

A *sync source* is a plugin that can both :meth:`load` data from an
external target (like an importer) and :meth:`save` data back to it
(like an exporter).  Concrete subclasses include
:class:`vtsearch.settings_io.sources.base.SettingsSource` (settings
round-trip) and :class:`vtscore.labels.sources.base.LabelsetSource`
(detector labels round-trip).

The two type parameters allow each kind of source to specify what it
returns from ``load()`` and accepts in ``save()`` independently.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from vtscore.plugins import PluginBase, PluginField

LoadT = TypeVar("LoadT")
SaveT = TypeVar("SaveT")

__all__ = ["SyncSource"]


class SyncSource(PluginBase, Generic[LoadT, SaveT]):
    """Abstract base class for bidirectional sync sources.

    Subclasses set the standard :class:`PluginBase` class attributes
    (``name``, ``display_name``, ``fields``, …) and implement
    :meth:`load` / :meth:`save`.
    """

    icon: str = "\U0001f504"  # counterclockwise arrows (sync)
    fields: list[PluginField]

    def load(self, field_values: dict[str, Any]) -> LoadT:
        """Import data from the source.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.load() is not implemented")

    def save(self, data: SaveT, /, field_values: dict[str, Any]) -> None:
        """Export *data* to the source.

        The first parameter is positional-only so subclasses can name it
        according to what they save (``labelset``, ``settings``, ...) without
        breaking the override contract.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.save() is not implemented")

    def peek_version(self, field_values: dict[str, Any]) -> Any | None:
        """Return an opaque token representing the source's current version.

        Used to cheaply detect whether the source has changed since the
        last successful :meth:`load`.  The caller compares the returned
        token against a previously-stashed value; if they differ, a full
        load is due.

        Subclasses should override with a cheap freshness probe (e.g.
        ``st_mtime_ns`` for a local file, ``ETag`` for an HTTP source).
        The default returns ``None``, which means *"I can't cheaply
        check"* — the caller falls back to explicit manual sync.
        """
        return None

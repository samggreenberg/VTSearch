"""Shared base for bidirectional sync sources.

A *sync source* is a plugin that can both :meth:`load` data from an
external target (like an importer) and :meth:`save` data back to it
(like an exporter).  Concrete subclasses include
:class:`vtsearch.settings_io.sources.base.SettingsSource` (settings
round-trip) and :class:`vtsearch.labels.sources.base.LabelsetSource`
(detector labels round-trip).

The two type parameters allow each kind of source to specify what it
returns from ``load()`` and accepts in ``save()`` independently.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from vtsearch.plugins import PluginBase, PluginField

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

    def save(self, data: SaveT, field_values: dict[str, Any]) -> None:
        """Export *data* to the source.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.save() is not implemented")

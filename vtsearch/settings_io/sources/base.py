"""Base class for Settings Sources.

A settings source is a bidirectional sync target: it can both *load*
settings (like an importer) and *save* them back (like an exporter).
When an active source is configured, settings are auto-imported on
startup and auto-exported whenever they change.

Standalone importers and exporters remain fully functional regardless
of whether a source is active.
"""

from __future__ import annotations

from typing import Any

from vtsearch.utils.registry import PluginBase, PluginField

SettingsSourceField = PluginField

__all__ = ["SettingsSource", "SettingsSourceField"]


class SettingsSource(PluginBase):
    """Abstract base class for settings sources.

    Subclass this, set the class-level attributes, implement :meth:`load`
    and :meth:`save`, and expose a module-level
    ``SETTINGS_SOURCE = YourSource()`` — the registry picks it up
    automatically.
    """

    icon: str = "\U0001f504"  # counterclockwise arrows (sync)
    fields: list[PluginField]

    def load(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Import settings from the source.

        Returns:
            A dict of settings key-value pairs to apply.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.load() is not implemented")

    def save(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> None:
        """Export settings to the source.

        Args:
            settings_data: The full settings dict to persist.
            field_values: Source configuration (e.g. filepath).

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.save() is not implemented")

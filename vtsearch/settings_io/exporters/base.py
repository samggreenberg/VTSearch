"""Base class for Settings Exporters.

To add a new settings exporter, subclass :class:`SettingsExporter`, define its
class attributes and :meth:`~SettingsExporter.export`, then expose a module-level
``SETTINGS_EXPORTER`` instance from a package under this directory.  The
registry will discover it automatically.

The :meth:`export` method receives the current settings dict and user-supplied
field values, and must return a dict with at minimum a ``"message"`` key.
"""

from __future__ import annotations

from typing import Any

from vtsearch.plugins import PluginBase, PluginField

SettingsExporterField = PluginField

__all__ = ["SettingsExporter", "SettingsExporterField"]


class SettingsExporter(PluginBase):
    """Abstract base class for settings exporters.

    Subclass this, set the class-level attributes, implement :meth:`export`,
    and expose a module-level ``SETTINGS_EXPORTER = YourExporter()`` -- the
    registry picks it up automatically.

    The :meth:`export` method receives the full settings dict and must return
    a dict with at minimum a ``"message"`` key describing what happened.
    """

    icon: str = "\U0001f4e4"  # outbox tray
    fields: list[PluginField]

    def export(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Perform the export and return a status dict.

        Args:
            settings_data: The full settings dict from ``settings.get_all()``.
            field_values: Mapping of field key to value string supplied by
                the user.

        Returns:
            A dict that **must** contain a ``"message"`` key with a short
            human-readable confirmation string.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.export() is not implemented")

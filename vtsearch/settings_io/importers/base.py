"""Base class for Settings Importers.

To add a new settings importer, subclass :class:`SettingsImporter`, define its
class attributes and :meth:`~SettingsImporter.run`, then expose a module-level
``SETTINGS_IMPORTER`` instance from a package under this directory.  The
registry will discover it automatically.

The :meth:`run` method must return a dict of settings key-value pairs that
will be applied to the application settings.
"""

from __future__ import annotations

from typing import Any

from vtscore.plugins import PluginBase, PluginField

SettingsImporterField = PluginField

__all__ = ["SettingsImporter", "SettingsImporterField"]


class SettingsImporter(PluginBase):
    """Abstract base class for settings importers.

    Subclass this, set the class-level attributes, implement :meth:`run`,
    and expose a module-level ``SETTINGS_IMPORTER = YourImporter()`` -- the
    registry picks it up automatically.

    The :meth:`run` method must return a settings dict that will be applied
    via the settings module's update mechanism.
    """

    icon: str = "\u2699\ufe0f"  # gear
    fields: list[PluginField]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Perform the import and return a settings dict.

        Args:
            field_values: Mapping of field key to value.  Fields with
                ``field_type="file"`` receive a Werkzeug
                :class:`~werkzeug.datastructures.FileStorage` object; all
                other fields receive plain strings.

        Returns:
            A dict of settings key-value pairs to apply.

        Raises:
            NotImplementedError: If the subclass has not implemented this.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

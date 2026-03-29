"""Local JSON settings exporter -- returns settings for browser download.

The settings JSON is returned in the API response so the frontend can
trigger a browser download.  No file path field is needed.
"""

from __future__ import annotations

from typing import Any

from vtsearch.settings_io.exporters.base import SettingsExporter


class LocalFileSettingsExporter(SettingsExporter):
    """Export settings as a JSON file downloaded via the browser."""

    name = "local_json_file"
    display_name = "Local JSON File"
    description = "Download settings as a JSON file to your computer."
    icon = "\U0001f4c1"  # file folder
    fields = []

    def export(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Return settings data for browser download."""
        return {
            "message": "Settings ready for download.",
            "download": True,
            "data": settings_data,
            "filename": "settings.json",
        }


SETTINGS_EXPORTER = LocalFileSettingsExporter()

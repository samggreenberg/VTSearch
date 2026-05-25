"""Server JSON settings exporter -- saves settings to a JSON file on the server.

Writes the current application settings to a JSON file on the server
filesystem.  The user supplies a destination path via the file browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.io import atomic_write_json
from vtsearch.settings_io.exporters.base import SettingsExporter, SettingsExporterField


class ServerFileSettingsExporter(SettingsExporter):
    """Export settings to a JSON file on the server filesystem."""

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Write settings to a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        SettingsExporterField(
            key="filepath",
            label="Save to (server path)",
            field_type="server_path",
            description="Where on the server to write the settings JSON file.",
            hint="Absolute or relative path; parent directories are created automatically.",
            placeholder="data/settings_backup.json",
            default="data/settings_backup.json",
        ),
    ]

    def export(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write settings to a JSON file on the server."""
        filepath = Path(field_values["filepath"])
        atomic_write_json(filepath, settings_data)
        return {
            "message": f"Settings saved to {filepath.resolve()}.",
            "filepath": str(filepath.resolve()),
        }


SETTINGS_EXPORTER = ServerFileSettingsExporter()

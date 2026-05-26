"""Server JSON settings importer -- loads settings from a JSON file on the server.

The user provides a file path on the server filesystem via the file browser.
The file must be a JSON object whose keys are valid settings names.
"""

from __future__ import annotations

from typing import Any

from vtscore.io import read_server_json
from vtsearch.settings_io.importers.base import SettingsImporter, SettingsImporterField


class ServerFileSettingsImporter(SettingsImporter):
    """Import settings from a JSON file on the server filesystem."""

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Import settings from a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        SettingsImporterField(
            key="filepath",
            label="Path or URL",
            field_type="server_path",
            description="A VTSearch settings JSON file on the server to load settings from.",
            placeholder="data/settings.json",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Read and parse the JSON settings file from the server filesystem."""
        data = read_server_json(field_values["filepath"])
        if not isinstance(data, dict):
            raise ValueError("Settings JSON must be a JSON object (dict).")
        return data


SETTINGS_IMPORTER = ServerFileSettingsImporter()

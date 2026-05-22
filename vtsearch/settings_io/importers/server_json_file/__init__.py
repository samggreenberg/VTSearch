"""Server JSON settings importer -- loads settings from a JSON file on the server.

The user provides a file path on the server filesystem via the file browser.
The file must be a JSON object whose keys are valid settings names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        filepath = (field_values.get("filepath") or "").strip()
        if not filepath:
            raise ValueError("A file path is required.")

        path = Path(filepath)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Settings JSON must be a JSON object (dict).")
        return data


SETTINGS_IMPORTER = ServerFileSettingsImporter()

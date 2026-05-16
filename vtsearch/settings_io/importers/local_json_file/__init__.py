"""Local JSON settings importer -- loads settings from a browser-uploaded JSON file.

The user selects a ``.json`` file via the browser file picker.  The file
must be a JSON object whose keys are valid settings names.
"""

from __future__ import annotations

import json
from typing import Any

from vtsearch.settings_io.importers.base import SettingsImporter, SettingsImporterField


class LocalFileSettingsImporter(SettingsImporter):
    """Import settings from a JSON file uploaded via the browser."""

    name = "local_json_file"
    display_name = "Local JSON File"
    description = "Import settings from a JSON file on your computer."
    icon = "\U0001f4c1"  # file folder
    fields = [
        SettingsImporterField(
            key="file",
            label="Upload a file",
            field_type="file",
            description="A VTSearch settings JSON file.",
            accept=".json",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Parse the uploaded JSON file and return a settings dict."""
        file_storage = field_values.get("file")
        if file_storage is None:
            raise ValueError("No file was uploaded.")

        raw = file_storage.read()
        if not raw:
            raise ValueError("The uploaded file is empty.")

        return _parse_settings_json(raw)


def _parse_settings_json(raw: bytes) -> dict[str, Any]:
    """Decode raw bytes as JSON and validate it is a dict."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Settings JSON must be a JSON object (dict).")
    return data


SETTINGS_IMPORTER = LocalFileSettingsImporter()

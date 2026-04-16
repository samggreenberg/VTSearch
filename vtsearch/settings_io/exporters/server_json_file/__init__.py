"""Server JSON settings exporter -- saves settings to a JSON file on the server.

Writes the current application settings to a JSON file on the server
filesystem.  The user supplies a destination path via the file browser.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
            label="Server File Path",
            field_type="server_path",
            description=(
                "Absolute or relative path on the server where the settings "
                "JSON file will be written.  Parent directories are created "
                "automatically."
            ),
            placeholder="data/settings_backup.json",
            default="data/settings_backup.json",
        ),
    ]

    def export(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write settings to a JSON file on the server."""
        filepath_str = field_values.get("filepath", "").strip()
        if not filepath_str:
            raise ValueError("A file path is required.")

        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: tmp + rename. A direct write_text leaves the file
        # truncated if the process is killed mid-write.
        tmp = filepath.with_name(filepath.name + ".tmp")
        tmp.write_text(json.dumps(settings_data, indent=2), encoding="utf-8")
        os.replace(tmp, filepath)

        return {
            "message": f"Settings saved to {filepath.resolve()}.",
            "filepath": str(filepath.resolve()),
        }


SETTINGS_EXPORTER = ServerFileSettingsExporter()

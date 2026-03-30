"""Server JSON settings source — bidirectional sync with a JSON file on the server.

Loads settings from, and saves settings to, a JSON file on the server
filesystem.  The ``filepath`` field supports a ``{username}`` template
that is resolved at runtime via :func:`~vtsearch.auth.get_current_user`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vtsearch.settings_io.sources.base import SettingsSource, SettingsSourceField


class ServerFileSettingsSource(SettingsSource):
    """Sync settings with a JSON file on the server filesystem."""

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Sync settings with a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        SettingsSourceField(
            key="filepath",
            label="Server File Path",
            field_type="server_path",
            description=(
                "Absolute or relative path to a settings JSON file on the "
                "server.  Supports {username} template."
            ),
            placeholder="data/{username}.settings.json",
        ),
    ]

    def load(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Read settings from a JSON file on the server."""
        path = Path(_resolve_filepath(field_values))
        if not path.exists():
            return {}
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

    def save(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> None:
        """Write settings to a JSON file on the server."""
        filepath = Path(_resolve_filepath(field_values))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings_data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, filepath)


def _resolve_filepath(field_values: dict[str, Any]) -> str:
    """Resolve the filepath, expanding {username} template."""
    filepath = (field_values.get("filepath") or "").strip()
    if not filepath:
        raise ValueError("A file path is required.")

    if "{username}" in filepath:
        from vtsearch.auth import get_current_user

        username = get_current_user() or "default"
        filepath = filepath.replace("{username}", username)

    return filepath


SETTINGS_SOURCE = ServerFileSettingsSource()

"""Server JSON settings source — bidirectional sync with a JSON file on the server.

Loads settings from, and saves settings to, a JSON file on the server
filesystem.  The ``filepath`` field supports a ``{username}`` template
that is resolved at runtime via :func:`~vtsearch.auth.get_current_user`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.io import atomic_write_json, read_server_json
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
            label="Save to (server path)",
            field_type="server_path",
            description="The JSON file on the server to sync your settings with.",
            hint="Absolute or relative server path.  Template variable: {username}.",
            placeholder="data/{username}.settings.json",
            template_vars=("username",),
        ),
    ]

    def _do_load(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Read settings from a JSON file on the server."""
        data = read_server_json(field_values["filepath"], missing_ok=True)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Settings JSON must be a JSON object (dict).")
        return data

    def _do_save(self, settings_data: dict[str, Any], field_values: dict[str, Any]) -> None:
        """Write settings to a JSON file on the server."""
        atomic_write_json(field_values["filepath"], settings_data)

    def _do_peek_version(self, field_values: dict[str, Any]) -> int | None:
        """Return the source file's ``st_mtime_ns`` as a freshness token.

        A change in the returned value signals the file has been
        rewritten since the last sync.  Returns ``None`` if the file
        doesn't exist (or can't be statted) — the settings layer
        interprets that as "skip the auto re-sync until the file
        appears."
        """
        try:
            return Path(field_values["filepath"]).stat().st_mtime_ns
        except OSError:
            return None


SETTINGS_SOURCE = ServerFileSettingsSource()

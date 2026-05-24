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
            label="Save to (server path)",
            field_type="server_path",
            description="The JSON file on the server to sync your settings with.",
            hint="Absolute or relative server path.  Template variable: {username}.",
            placeholder="data/{username}.settings.json",
            template_vars=("username",),
        ),
    ]

    def load(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Read settings from a JSON file on the server."""
        path = Path(_normalized(self, field_values)["filepath"])
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
        filepath = Path(_normalized(self, field_values)["filepath"])
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(settings_data, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, filepath)

    def peek_version(self, field_values: dict[str, Any]) -> int | None:
        """Return the source file's ``st_mtime_ns`` as a freshness token.

        A change in the returned value signals the file has been
        rewritten since the last sync.  Returns ``None`` if the file
        doesn't exist (or the path can't be resolved / statted) — the
        settings layer interprets that as "skip the auto re-sync until
        the file appears."
        """
        try:
            path = Path(_normalized(self, field_values)["filepath"])
        except Exception:
            return None
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None


def _normalized(source: ServerFileSettingsSource, field_values: dict[str, Any]) -> dict[str, Any]:
    """Return *field_values* with the source's declarative knobs applied.

    Settings-source callers (vtsearch/settings.py auto-sync hooks) pass
    field_values straight from `data/settings.json`, bypassing the route
    layer's normalize hook, so we apply it here.  Idempotent: callers
    that already normalized get the same dict back.
    """
    from vtscore.plugins.normalize import normalize_field_values

    return normalize_field_values(source, dict(field_values))


SETTINGS_SOURCE = ServerFileSettingsSource()

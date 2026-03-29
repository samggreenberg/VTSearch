"""Server JSON labelset source — bidirectional sync with a JSON file on the server.

Loads labels from, and saves labels to, a JSON file on the server
filesystem.  The ``filepath`` field supports ``{detector_id}`` and
``{detector_name}`` templates resolved at runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtsearch.labels.sources.base import LabelsetSource, LabelsetSourceField

if TYPE_CHECKING:
    from vtsearch.datasets.labelset import LabelSet


class ServerFileLabelsetSource(LabelsetSource):
    """Sync detector labels with a JSON file on the server filesystem."""

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Sync detector labels with a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        LabelsetSourceField(
            key="filepath",
            label="Server File Path",
            field_type="server_path",
            description=(
                "Absolute or relative path to a labels JSON file on the "
                "server.  Supports {detector_id} and {detector_name} templates."
            ),
            placeholder="data/labels/{detector_name}.labels.json",
        ),
    ]

    def load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Read labels from a JSON file on the server."""
        path = Path(_resolve_filepath(field_values))
        if not path.exists():
            return []
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        labels = data.get("labels")
        if not isinstance(labels, list):
            raise ValueError("JSON must contain a top-level 'labels' list.")
        return [entry for entry in labels if isinstance(entry, dict)]

    def save(self, labelset: LabelSet, field_values: dict[str, Any]) -> None:
        """Write labels to a JSON file on the server."""
        filepath = Path(_resolve_filepath(field_values))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = filepath.with_suffix(".tmp")
        tmp.write_text(json.dumps(labelset.to_dict(), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, filepath)


def _resolve_filepath(field_values: dict[str, Any]) -> str:
    """Resolve the filepath, expanding template variables."""
    filepath = (field_values.get("filepath") or "").strip()
    if not filepath:
        raise ValueError("A file path is required.")

    if "{detector_id}" in filepath or "{detector_name}" in filepath:
        from vtsearch.utils.state_core import get_active_detector_context

        ctx = get_active_detector_context()
        if ctx is not None:
            filepath = filepath.replace("{detector_id}", ctx.detector_id)
            filepath = filepath.replace("{detector_name}", ctx.name)

    return filepath


LABELSET_SOURCE = ServerFileLabelsetSource()

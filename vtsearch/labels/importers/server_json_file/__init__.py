"""Server JSON label importer – loads labels from a ``.json`` file on the server.

This importer reads a VTSearch label-format JSON file from the server
filesystem (the machine where the Python process is running), as opposed
to the local (browser-upload) importer which receives files uploaded from
the user's browser.  The file format is identical::

    {"labels": [{"md5": "...", "label": "good"}, ...]}

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class ServerJsonLabelImporter(LabelImporter):
    """Import labels from a JSON file on the server filesystem.

    The user provides a file path on the server (the machine running the
    Python process).  The file must be a JSON object with a top-level
    ``"labels"`` key whose value is a list of ``{"md5": "...", "label":
    "good"|"bad"}`` dicts.
    """

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Import labels from a VTSearch-format JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        LabelImporterField(
            key="filepath",
            label="Path or URL",
            field_type="server_path",
            description=("Absolute or relative path to a VTSearch labels JSON file on the server filesystem."),
            placeholder="data/labels/my_labels.json",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Read and parse the JSON labels file from the server filesystem."""
        filepath = (field_values.get("filepath") or "").strip()
        if not filepath:
            raise ValueError("A file path is required.")

        path = Path(filepath)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        return _parse_json_bytes(raw)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        """Load labels from a file-path string (CLI usage)."""
        return self.run(field_values)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--filepath",
            dest="filepath",
            help="Path to a VTSearch labels JSON file on the server.",
            required=False,
        )


def _parse_json_bytes(raw: bytes) -> list[dict[str, str]]:
    """Decode *raw* bytes as JSON and extract the labels list."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    labels = data.get("labels")
    if not isinstance(labels, list):
        raise ValueError("JSON must contain a top-level 'labels' list.")
    return [entry for entry in labels if isinstance(entry, dict)]


LABEL_IMPORTER = ServerJsonLabelImporter()

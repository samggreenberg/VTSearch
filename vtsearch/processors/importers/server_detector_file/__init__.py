"""Server-file detector importer -- loads a detector from a ``.json`` file on the server.

This importer reads a VTSearch detector JSON file from the server filesystem
(the machine where the Python process is running), as opposed to the local
(browser-upload) importer which receives files uploaded from the user's
browser.  The file format is identical::

    {
        "weights": {"0.weight": [...], "0.bias": [...], "2.weight": [...], "2.bias": [...]},
        "threshold": 0.5,
        "media_type": "audio",
        "name": "my detector"
    }

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.processors.importers.base import ProcessorImporter, ProcessorImporterField


class ServerFileDetectorImporter(ProcessorImporter):
    """Import a processor (detector) from a JSON file on the server filesystem.

    The user provides a file path on the server (the machine running the
    Python process).  The file must contain ``"weights"`` (serialised MLP
    state dict) and ``"threshold"`` (float).  An optional ``"media_type"``
    key specifies the media type; when absent it defaults to ``"audio"``.
    """

    name = "server_detector_file"
    display_name = "Server Detector File (.json)"
    description = "Import a pre-trained detector from a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        ProcessorImporterField(
            key="filepath",
            label="Server File Path",
            field_type="text",
            description=(
                "Absolute or relative path to a VTSearch detector JSON file "
                "on the server filesystem."
            ),
            placeholder="data/detectors/my_detector.json",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Read and parse the JSON detector file from the server filesystem."""
        filepath = (field_values.get("filepath") or "").strip()
        if not filepath:
            raise ValueError("A file path is required.")

        path = Path(filepath)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        return _parse_detector_json(raw)

    def run_cli(self, field_values: dict[str, Any]) -> dict[str, Any]:
        """Load a detector from a file-path string (CLI usage)."""
        return self.run(field_values)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--filepath",
            dest="filepath",
            help="Path to a VTSearch detector JSON file on the server.",
            required=False,
        )


def _parse_detector_json(raw: bytes) -> dict[str, Any]:
    """Decode *raw* bytes as JSON and extract detector data."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    weights = data.get("weights")
    if not weights:
        raise ValueError("Detector file missing 'weights' field.")

    threshold = data.get("threshold", 0.5)
    media_type = data.get("media_type", "audio")
    suggested_name = data.get("name", "")

    result: dict[str, Any] = {
        "media_type": media_type,
        "weights": weights,
        "threshold": threshold,
    }
    if suggested_name:
        result["name"] = suggested_name
    return result


PROCESSOR_IMPORTER = ServerFileDetectorImporter()

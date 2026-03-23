"""Server CSV label importer – loads labels from a ``.csv`` file on the server.

This importer reads a CSV file from the server filesystem (the machine
where the Python process is running), as opposed to the local
(browser-upload) importer which receives files uploaded from the user's
browser.  The file format is identical: a header row with ``md5`` and
``label`` columns.

No additional pip packages are required; uses only Python's ``csv``,
``io``, and ``pathlib`` stdlib modules.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from vtsearch.labels.importers.base import LabelImporter, LabelImporterField


class ServerCsvLabelImporter(LabelImporter):
    """Import labels from a CSV file on the server filesystem.

    The user provides a file path on the server (the machine running the
    Python process).  The file must have a header row with ``md5`` and
    ``label`` columns.
    """

    name = "server_csv_file"
    display_name = "Server CSV File"
    description = "Import labels from a CSV file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        LabelImporterField(
            key="filepath",
            label="Server File Path",
            field_type="server_path",
            description=(
                "Absolute or relative path to a CSV file with md5 and label columns on the server filesystem."
            ),
            placeholder="data/labels/my_labels.csv",
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, Any]]:
        """Read and parse the CSV labels file from the server filesystem."""
        filepath = (field_values.get("filepath") or "").strip()
        if not filepath:
            raise ValueError("A file path is required.")

        path = Path(filepath)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        raw = path.read_bytes()
        return _parse_csv_bytes(raw)

    def run_cli(self, field_values: dict[str, Any]) -> list[dict[str, Any]]:
        """Load labels from a file-path string (CLI usage)."""
        return self.run(field_values)

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument(
            "--filepath",
            dest="filepath",
            help="Path to a CSV file with md5 and label columns on the server.",
            required=False,
        )


def _parse_csv_bytes(raw: bytes) -> list[dict[str, Any]]:
    """Decode *raw* bytes as CSV and extract ``md5``/``label`` pairs."""
    try:
        text = raw.decode("utf-8-sig")  # strip BOM if present
    except UnicodeDecodeError:
        import logging

        logging.getLogger(__name__).warning(
            "CSV file is not valid UTF-8; falling back to latin-1 encoding. Non-Latin characters may be corrupted."
        )
        text = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV file appears to be empty.")

    # Normalise header names to lower-case and strip whitespace
    normalised = {k.strip().lower(): k for k in reader.fieldnames if k}
    if "md5" not in normalised or "label" not in normalised:
        raise ValueError("CSV must have 'md5' and 'label' column headers.")

    # Optional columns that enrich resolution (origin_name, filename, category, origin)
    optional_cols = ("origin_name", "filename", "category")

    results = []
    for row in reader:
        md5 = row.get(normalised["md5"], "").strip()
        label = row.get(normalised["label"], "").strip().lower()
        if md5 and label:
            entry: dict[str, Any] = {"md5": md5, "label": label}
            for col in optional_cols:
                if col in normalised:
                    val = row.get(normalised[col], "").strip()
                    if val:
                        entry[col] = val
            # Parse origin dict from JSON if present
            if "origin" in normalised:
                origin_raw = row.get(normalised["origin"], "").strip()
                if origin_raw:
                    try:
                        origin = json.loads(origin_raw)
                        if isinstance(origin, dict):
                            entry["origin"] = origin
                    except (json.JSONDecodeError, ValueError):
                        pass
            results.append(entry)
    return results


LABEL_IMPORTER = ServerCsvLabelImporter()

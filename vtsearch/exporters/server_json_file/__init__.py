"""Server JSON exporter – saves auto-detect results to a JSON file on the server.

Writes a JSON file to the server filesystem (the machine where the Python
process is running).  The user supplies a path (absolute or relative) as a
text field rather than a browser file picker.

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vtsearch.exporters._template import resolve_export_filepath
from vtsearch.exporters.base import ExporterField, LabelsetExporter


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (tmp file + rename).

    A direct ``write_text`` call leaves the destination truncated if the
    process is killed mid-write.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ServerJsonLabelsetExporter(LabelsetExporter):
    """Save auto-detect results as a JSON file on the server filesystem.

    The user supplies the destination path (absolute or relative to the
    server's current working directory).  Parent directories are created
    automatically.
    """

    name = "server_json_file"
    display_name = "Server JSON File"
    description = "Write the results to a JSON file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        ExporterField(
            key="filepath",
            label="Server File Path",
            field_type="server_path",
            description=(
                "Absolute or relative path on the server where the JSON "
                "results file will be written.  Parent directories are "
                "created automatically.  Supports {YYYYMMDD-HHMMSS}, "
                "{detector_name} and {username} templates."
            ),
            placeholder="data/autodetect_results_{YYYYMMDD-HHMMSS}.json",
            default="data/autodetect_results_{YYYYMMDD-HHMMSS}.json",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = field_values.get("filepath", "").strip()
        if not filepath_str:
            raise ValueError("A file path is required.")

        filepath = Path(resolve_export_filepath(filepath_str))
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Labels format (from the export modal UI) — filter to selected columns
        if "labels" in results:
            return self._export_labels(results, filepath)

        # Autodetect results format (from CLI / fill-from-sort)
        _atomic_write_text(filepath, json.dumps(results, indent=2))

        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }

    def _export_labels(self, results: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """Export labels, filtering to selected columns when provided."""
        labels = results.get("labels", [])
        selected_columns: list[str] | None = results.get("selected_columns")

        if selected_columns is not None:
            filtered_labels = []
            for entry in labels:
                row: dict[str, Any] = {}
                meta = entry.get("custom_metadata") or {}
                for col in selected_columns:
                    if col in entry:
                        row[col] = entry[col]
                    elif col in meta:
                        row[col] = meta[col]
                filtered_labels.append(row)
            output = {"labels": filtered_labels, "selected_columns": selected_columns}
        else:
            output = {"labels": labels}

        _atomic_write_text(filepath, json.dumps(output, indent=2))
        return {
            "message": f"Saved {len(labels)} label(s) to {filepath.resolve()}.",
            "filepath": str(filepath.resolve()),
        }


EXPORTER = ServerJsonLabelsetExporter()

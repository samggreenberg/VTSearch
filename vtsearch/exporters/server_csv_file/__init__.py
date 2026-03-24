"""Server CSV exporter – saves auto-detect results to a CSV file on the server.

Writes a CSV file to the server filesystem (the machine where the Python
process is running).  The user supplies a path (absolute or relative) as a
text field.

No additional pip packages are required; uses only Python's ``csv`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


class ServerCsvLabelsetExporter(LabelsetExporter):
    """Save auto-detect results as a CSV file on the server filesystem.

    Produces one row per hit across all detectors, with columns for the
    detector name, filename, category, and score.  Opens directly in
    Excel, Google Sheets, or any spreadsheet application.
    """

    name = "server_csv_file"
    display_name = "Server CSV File"
    description = "Write the results to a CSV file on the server filesystem."
    icon = "\U0001f5a5"  # desktop computer
    fields = [
        ExporterField(
            key="filepath",
            label="Server File Path",
            field_type="server_path",
            description=(
                "Absolute or relative path on the server where the CSV "
                "results file will be written.  Parent directories are "
                "created automatically."
            ),
            placeholder="data/autodetect_results.csv",
            default="data/autodetect_results.csv",
        ),
    ]

    #: Base columns for label exports (used when no selected_columns provided).
    _LABEL_BASE_COLUMNS = ["label", "md5", "origin_name", "filename", "category", "origin"]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = field_values.get("filepath", "").strip()
        if not filepath_str:
            raise ValueError("A file path is required.")

        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Labels format (from the export modal UI)
        if "labels" in results:
            return self._export_labels(results, filepath)

        # Autodetect results format (from CLI / fill-from-sort)
        return self._export_autodetect(results, filepath)

    def _export_labels(self, results: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """Export labels with user-selected columns."""
        labels = results.get("labels", [])
        columns: list[str] = results.get("selected_columns") or self._LABEL_BASE_COLUMNS

        total_rows = 0
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

            for entry in labels:
                meta = entry.get("custom_metadata") or {}
                row = []
                for col in columns:
                    if col in entry:
                        val = entry[col]
                        # Serialize dicts (e.g. origin) as JSON so they
                        # survive the CSV round-trip.
                        if isinstance(val, dict):
                            row.append(json.dumps(val, sort_keys=True))
                        else:
                            row.append(str(val if val is not None else ""))
                    elif col in meta:
                        row.append(str(meta[col] if meta[col] is not None else ""))
                    else:
                        row.append("")
                writer.writerow(row)
                total_rows += 1

        return {
            "message": f"Saved {total_rows} label(s) to {filepath.resolve()}.",
            "filepath": str(filepath.resolve()),
        }

    def _export_autodetect(self, results: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """Export autodetect results (detector hits)."""
        total_hits = 0
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["detector", "threshold", "filename", "category", "score", "origin", "origin_name"])

            for det_result in results.get("results", {}).values():
                detector_name = det_result.get("detector_name", "unknown")
                threshold = det_result.get("threshold", "")
                for hit in det_result.get("hits", []):
                    origin = hit.get("origin")
                    origin_str = ""
                    if origin:
                        from vtsearch.datasets.origin import Origin

                        origin_str = Origin.from_dict(origin).display()
                    writer.writerow(
                        [
                            detector_name,
                            threshold,
                            hit.get("filename", ""),
                            hit.get("category", ""),
                            hit.get("score", ""),
                            origin_str,
                            hit.get("origin_name", ""),
                        ]
                    )
                    total_hits += 1

        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = ServerCsvLabelsetExporter()

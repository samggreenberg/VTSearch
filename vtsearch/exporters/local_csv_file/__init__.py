"""Local CSV exporter – returns auto-detect results as CSV for browser download.

In the GUI the CSV data is returned inline so the frontend can trigger
a file download on the user's local machine.  In the CLI the CSV is written
to a local file path.

No additional pip packages are required; uses only Python's ``csv`` and
``io`` stdlib modules.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


class LocalCsvLabelsetExporter(LabelsetExporter):
    """Return auto-detect results as CSV for download to the user's local machine.

    In the GUI this exporter returns the CSV data inline in the API
    response (via ``download_content``).  The frontend is expected to
    create a Blob and trigger a browser download.

    In the CLI (via ``export_cli``) the CSV is written to a file path.
    """

    name = "local_csv_file"
    display_name = "Local CSV File"
    description = "Download the results as a CSV file to your local machine."
    icon = "\U0001f4ca"  # bar chart
    fields = [
        ExporterField(
            key="filepath",
            label="File Path",
            field_type="text",
            description=(
                "Filename for the downloaded CSV file (used by the CLI; "
                "the browser uses its own save-as dialog)."
            ),
            placeholder="autodetect_results.csv",
            default="autodetect_results.csv",
            required=False,
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        csv_content, total_hits = _build_csv_string(results)
        return {
            "message": (
                f"Prepared {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) for download."
            ),
            "download_content": csv_content,
            "download_filename": field_values.get("filepath", "").strip() or "autodetect_results.csv",
            "download_content_type": "text/csv",
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write CSV to a local file path (CLI usage)."""
        filepath_str = (field_values.get("filepath") or "").strip() or "autodetect_results.csv"
        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        csv_content, total_hits = _build_csv_string(results)
        filepath.write_text(csv_content, encoding="utf-8")

        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


def _build_csv_string(results: dict[str, Any]) -> tuple[str, int]:
    """Build a CSV string from results and return ``(csv_text, total_hits)``."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["detector", "threshold", "filename", "category", "score", "origin", "origin_name"])

    total_hits = 0
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

    return buf.getvalue(), total_hits


EXPORTER = LocalCsvLabelsetExporter()

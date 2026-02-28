"""Local JSON exporter – returns auto-detect results as JSON for browser download.

In the GUI the results JSON is returned inline so the frontend can trigger
a file download on the user's local machine.  In the CLI the JSON is written
to a local file path.

No additional pip packages are required; uses only Python's ``json`` stdlib
module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


class LocalJsonLabelsetExporter(LabelsetExporter):
    """Return auto-detect results as JSON for download to the user's local machine.

    In the GUI this exporter returns the JSON data inline in the API
    response (via ``download_content``).  The frontend is expected to
    create a Blob and trigger a browser download.

    In the CLI (via ``export_cli``) the JSON is written to a file path.
    """

    name = "local_json_file"
    display_name = "Local JSON File"
    description = "Download the results as a JSON file to your local machine."
    icon = "\U0001f4be"  # floppy disk
    fields = [
        ExporterField(
            key="filepath",
            label="File Path",
            field_type="text",
            description=(
                "Filename for the downloaded JSON file (used by the CLI; the browser uses its own save-as dialog)."
            ),
            placeholder="autodetect_results.json",
            default="autodetect_results.json",
            required=False,
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (
                f"Prepared {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s) for download."
            ),
            "download_content": json.dumps(results, indent=2),
            "download_filename": field_values.get("filepath", "").strip() or "autodetect_results.json",
            "download_content_type": "application/json",
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write JSON to a local file path (CLI usage)."""
        filepath_str = (field_values.get("filepath") or "").strip() or "autodetect_results.json"
        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(results, indent=2), encoding="utf-8")

        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = LocalJsonLabelsetExporter()

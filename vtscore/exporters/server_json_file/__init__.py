"""Server JSON exporter – saves auto-detect results to a JSON file on the server.

Writes a JSON file to the server filesystem (the machine where the Python
process is running).  The user supplies a path (absolute or relative) as a
text field rather than a browser file picker.

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtscore.config import DATA_DIR
from vtscore.exporters.base import ExporterField, LabelsetExporter
from vtscore.io import atomic_write_json

_DEFAULT_JSON_PATH = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.json"


# Re-export under the historical private name so any third-party exporter
# that imported it directly keeps working.  New code should call
# :func:`vtscore.io.atomic_write_text` or
# :func:`vtscore.io.atomic_write_json` instead.
from vtscore.io import atomic_write_text as _atomic_write_text  # noqa: E402, F401


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
            label="Save to (server path)",
            field_type="server_path",
            description="Where on the server to write the JSON results file.",
            hint=(
                "Absolute or relative path; parent directories are created automatically.\n"
                "Template variables: {YYYYMMDD-HHMMSS}, {detector_name}, {username}."
            ),
            placeholder=_DEFAULT_JSON_PATH,
            default=_DEFAULT_JSON_PATH,
            template_vars=("YYYYMMDD-HHMMSS", "detector_name", "username"),
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath = Path(field_values["filepath"])

        # Labels format (from the export modal UI) — filter to selected columns
        if "labels" in results:
            return self._export_labels(results, filepath)

        # Autodetect results format (from CLI / fill-from-sort)
        atomic_write_json(filepath, results)

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

        atomic_write_json(filepath, output)
        return {
            "message": f"Saved {len(labels)} label(s) to {filepath.resolve()}.",
            "filepath": str(filepath.resolve()),
        }


EXPORTER = ServerJsonLabelsetExporter()

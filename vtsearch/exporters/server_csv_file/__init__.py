"""Server CSV exporter – saves auto-detect results to a CSV file on the server.

Writes a CSV file to the server filesystem (the machine where the Python
process is running).  The user supplies a path (absolute or relative) as a
text field.

No additional pip packages are required; uses only Python's ``csv`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any

from vtsearch.exporters._template import resolve_export_filepath
from vtsearch.exporters.base import ExporterField, LabelsetExporter


def _atomic_write_csv(path: Path, write_rows) -> None:
    """Build CSV content in memory then write atomically.

    *write_rows* is invoked with a ``csv.writer`` and is expected to emit
    every row.  Buffering in memory keeps the destination file untouched
    until the write succeeds, so a process crash mid-write cannot leave
    a half-written CSV behind.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    write_rows(writer)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# Characters that trigger formula execution in spreadsheet applications.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _sanitize_csv_cell(value: str) -> str:
    """Prefix formula-like cell values with a single quote to prevent injection."""
    if value and value[0] in _FORMULA_PREFIXES:
        return "'" + value
    return value


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
            label="Save to (server path)",
            field_type="server_path",
            description=(
                "Absolute or relative path on the server where the CSV "
                "results file will be written.  Parent directories are "
                "created automatically.  Supports {YYYYMMDD-HHMMSS}, "
                "{detector_name} and {username} templates."
            ),
            placeholder="data/autodetect_results_{YYYYMMDD-HHMMSS}.csv",
            default="data/autodetect_results_{YYYYMMDD-HHMMSS}.csv",
        ),
    ]

    #: Base columns for label exports (used when no selected_columns provided).
    #: ``origin`` is always appended as the last column (not listed here).
    _LABEL_BASE_COLUMNS = ["label", "md5", "origin_name", "filename", "category"]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        filepath_str = field_values.get("filepath", "").strip()
        if not filepath_str:
            raise ValueError("A file path is required.")

        filepath = Path(resolve_export_filepath(filepath_str))
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Labels format (from the export modal UI)
        if "labels" in results:
            return self._export_labels(results, filepath)

        # Autodetect results format (from CLI / fill-from-sort)
        return self._export_autodetect(results, filepath)

    def _export_labels(self, results: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """Export labels with user-selected columns.

        The ``origin`` column is always written as the last column so
        that the CSV can be re-imported without data loss.
        """
        labels = results.get("labels", [])
        columns: list[str] = list(results.get("selected_columns") or self._LABEL_BASE_COLUMNS)

        # Ensure origin is always the last column (required for re-import).
        if "origin" in columns:
            columns.remove("origin")
        columns.append("origin")

        total_rows = 0

        def _write(writer: "csv.writer") -> None:  # type: ignore[name-defined]
            nonlocal total_rows
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
                            cell = str(val if val is not None else "")
                            row.append(_sanitize_csv_cell(cell))
                    elif col in meta:
                        cell = str(meta[col] if meta[col] is not None else "")
                        row.append(_sanitize_csv_cell(cell))
                    else:
                        row.append("")
                writer.writerow(row)
                total_rows += 1

        _atomic_write_csv(filepath, _write)

        return {
            "message": f"Saved {total_rows} label(s) to {filepath.resolve()}.",
            "filepath": str(filepath.resolve()),
        }

    def _export_autodetect(self, results: dict[str, Any], filepath: Path) -> dict[str, Any]:  # noqa: C901
        """Export autodetect results (detector hits)."""
        # Scan all hits to determine which clip columns are present.
        all_hits: list[tuple[str, Any, dict]] = []
        for det_result in results.get("results", {}).values():
            detector_name = det_result.get("detector_name", "unknown")
            threshold = det_result.get("threshold", "")
            for hit in det_result.get("hits", []):
                all_hits.append((detector_name, threshold, hit))

        has_clip_start = any(h.get("clip_start") is not None for _, _, h in all_hits)
        has_clip_end = any(h.get("clip_end") is not None for _, _, h in all_hits)
        has_clip_box = any(h.get("clip_box") is not None for _, _, h in all_hits)

        base_cols = ["detector", "threshold", "filename", "category", "score"]
        if has_clip_start:
            base_cols.append("clip_start")
        if has_clip_end:
            base_cols.append("clip_end")
        if has_clip_box:
            base_cols.append("clip_box")
        base_cols.extend(["origin", "origin_name"])

        total_hits = 0

        def _write(writer: "csv.writer") -> None:  # type: ignore[name-defined]
            nonlocal total_hits
            writer.writerow(base_cols)
            for detector_name, threshold, hit in all_hits:
                origin = hit.get("origin")
                origin_str = ""
                if origin:
                    from vtsearch.datasets.origin import Origin

                    origin_str = Origin.from_dict(origin).display()
                row = [
                    _sanitize_csv_cell(str(detector_name)),
                    threshold,
                    _sanitize_csv_cell(hit.get("filename", "")),
                    _sanitize_csv_cell(hit.get("category", "")),
                    hit.get("score", ""),
                ]
                if has_clip_start:
                    row.append(hit.get("clip_start", ""))
                if has_clip_end:
                    row.append(hit.get("clip_end", ""))
                if has_clip_box:
                    cb = hit.get("clip_box")
                    row.append(",".join(str(v) for v in cb) if cb else "")
                row.extend(
                    [
                        _sanitize_csv_cell(origin_str),
                        _sanitize_csv_cell(hit.get("origin_name", "")),
                    ]
                )
                writer.writerow(row)
                total_hits += 1

        _atomic_write_csv(filepath, _write)

        return {
            "message": (
                f"Saved {total_hits} hit(s) across "
                f"{results.get('detectors_run', 0)} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = ServerCsvLabelsetExporter()

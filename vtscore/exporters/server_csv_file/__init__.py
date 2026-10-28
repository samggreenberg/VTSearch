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
from pathlib import Path
from typing import Any, Iterator

from vtscore.config import DATA_DIR
from vtscore.exporters.base import PluginField, ResultsExporter, StreamResume
from vtscore.io import (
    atomic_write_stream,
    atomic_write_text,
    commit_write_stream,
    resumable_write_stream,
    sanitize_csv_cell as _sanitize_csv_cell,
)

_DEFAULT_CSV_PATH = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.csv"


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
    atomic_write_text(path, buf.getvalue())


def _label_cell(val: Any) -> str:
    """One label entry's value as a CSV cell that survives the round-trip.

    Dicts (``origin``) and lists (``region_box``) are written as JSON, which is
    what the CSV label importer parses them back from; anything else is text,
    guarded against formula injection.
    """
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True)
    return _sanitize_csv_cell(str(val if val is not None else ""))


def _stream_thresholds(header: dict[str, Any]) -> dict[str, Any]:
    """Each streamed detector's threshold, by name, for the ``threshold`` column."""
    return {d.get("detector_name", ""): d.get("threshold", "") for d in header.get("detectors", [])}


def _stream_hit_row(detector_name: str, hit: dict[str, Any], thresholds: dict[str, Any]) -> list[Any]:
    """One streamed hit as a row in :attr:`ServerCsvResultsExporter._STREAM_COLUMNS` order."""
    from vtscore.datasets.origin import Origin  # noqa: PLC0415

    origin = hit.get("origin")
    origin_str = Origin.from_dict(origin).display() if origin else ""
    clip_box = hit.get("clip_box")
    return [
        _sanitize_csv_cell(str(detector_name)),
        thresholds.get(detector_name, ""),
        _sanitize_csv_cell(hit.get("filename", "")),
        _sanitize_csv_cell(hit.get("category", "")),
        hit.get("score", ""),
        _sanitize_csv_cell(hit.get("label", "")),
        hit.get("clip_start", ""),
        hit.get("clip_end", ""),
        ",".join(str(v) for v in clip_box) if clip_box else "",
        _sanitize_csv_cell(origin_str),
        _sanitize_csv_cell(hit.get("origin_name", "")),
    ]


class ServerCsvResultsExporter(ResultsExporter):
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
        PluginField(
            key="filepath",
            label="Save to (server path)",
            field_type="server_path",
            description="Where on the server to write the CSV results file.",
            hint=(
                "Absolute or relative path; parent directories are created automatically.\n"
                "Template variables: {YYYYMMDD-HHMMSS}, {YYYYMMDD}, {YYYY}, {MM}, {DD}, "
                "{detector_name}, {username}."
            ),
            placeholder=_DEFAULT_CSV_PATH,
            default=_DEFAULT_CSV_PATH,
            template_vars=("YYYYMMDD-HHMMSS", "YYYYMMDD", "YYYY", "MM", "DD", "detector_name", "username"),
        ),
    ]

    #: Base columns for label exports (used when no selected_columns provided).
    #: ``origin`` is always appended as the last column (not listed here).
    _LABEL_BASE_COLUMNS = ["label", "md5", "origin_name", "filename", "category", "region_box"]

    def export_find_results(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write one CSV row per hit, across all detectors."""
        filepath = Path(field_values["filepath"])
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return self._export_autodetect(results, filepath)

    def export_labelset(self, labelset: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write one CSV row per label, in the user's chosen column order."""
        filepath = Path(field_values["filepath"])
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return self._export_labels(labelset, filepath)

    #: Fixed column order for streamed autodetect exports.  Streaming cannot
    #: pre-scan all hits to detect which optional clip columns are present, so
    #: the full superset is always written (empty cells where a hit lacks the
    #: field).
    _STREAM_COLUMNS = [
        "detector",
        "threshold",
        "filename",
        "category",
        "score",
        "label",
        "clip_start",
        "clip_end",
        "clip_box",
        "origin",
        "origin_name",
    ]

    @property
    def supports_streaming(self) -> bool:
        return True

    def export_cli_streaming(
        self,
        header: dict[str, Any],
        records: Iterator[tuple[str, dict[str, Any]]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Write hits as CSV rows, one per hit, flushed as they stream.

        Uses a fixed column superset (see :attr:`_STREAM_COLUMNS`) because a
        streaming writer cannot look ahead to decide which optional clip
        columns are present.  :func:`~vtscore.io.atomic_write_stream` owns the
        tmp-file/``fsync``/rename ritual, so a run that dies mid-stream leaves
        no half-written export and two concurrent exports to the same path
        can't clobber each other's temp file.
        """
        filepath = Path(field_values["filepath"])
        thresholds = _stream_thresholds(header)

        total_hits = 0
        with atomic_write_stream(filepath) as f:
            writer = csv.writer(f)
            writer.writerow(self._STREAM_COLUMNS)
            for detector_name, hit in records:
                writer.writerow(_stream_hit_row(detector_name, hit, thresholds))
                total_hits += 1

        return {
            "message": (
                f"Streamed {total_hits} hit(s) across "
                f"{len(header.get('detectors', []))} detector(s) to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }

    def resumable_stream_path(self, field_values: dict[str, Any]) -> str:
        return str(field_values["filepath"])

    def export_cli_streaming_resumable(
        self,
        header: dict[str, Any],
        chunks: Iterator[list[tuple[str, dict[str, Any]]]],
        field_values: dict[str, Any],
        resume: StreamResume,
    ) -> dict[str, Any]:
        """Write the same CSV as :meth:`export_cli_streaming`, in place, a chunk at a time.

        The header row is written once, by the run that starts the file; each
        chunk's rows are ``fsync``\\ ed before the chunk is committed, so a
        resumed run cuts the file back to the last whole chunk and appends.
        """
        filepath = Path(field_values["filepath"])
        thresholds = _stream_thresholds(header)

        total_hits = 0
        with resumable_write_stream(filepath, resume.position) as f:
            writer = csv.writer(f)
            if resume.position is None:
                writer.writerow(self._STREAM_COLUMNS)
                resume.commit(commit_write_stream(f))
            for chunk in chunks:
                for detector_name, hit in chunk:
                    writer.writerow(_stream_hit_row(detector_name, hit, thresholds))
                    total_hits += 1
                resume.commit(commit_write_stream(f))

        verb = "Streamed" if resume.position is None else "Appended"
        return {
            "message": (
                f"{verb} {total_hits} hit(s) across "
                f"{len(header.get('detectors', []))} detector(s) to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }

    #: Fixed column order for a streamed labelset: the label export's base
    #: columns, led by the detector that labelled each row and ending in
    #: ``origin`` (JSON) as every label CSV does, so the CSV label importer
    #: reads the file back.  No ``region_box``: a scored run draws no regions.
    _LABEL_STREAM_COLUMNS = ["detector", "label", "md5", "origin_name", "filename", "category", "origin"]

    def export_labelset_streaming(
        self,
        header: dict[str, Any],
        elements: Iterator[dict[str, Any]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Write one CSV row per label entry, flushed as they stream.

        The columns are fixed (:attr:`_LABEL_STREAM_COLUMNS`) because a
        streaming writer cannot look ahead, and they are the ones the
        ``server_csv_file`` label importer reads, which ignores the extra
        ``detector`` column.  Written through
        :func:`~vtscore.io.atomic_write_stream`, as the hit stream is.
        """
        filepath = Path(field_values["filepath"])

        total = 0
        with atomic_write_stream(filepath) as f:
            writer = csv.writer(f)
            writer.writerow(self._LABEL_STREAM_COLUMNS)
            for entry in elements:
                writer.writerow([_label_cell(entry.get(col, "")) for col in self._LABEL_STREAM_COLUMNS])
                total += 1

        return {
            "message": (
                f"Streamed {total} label(s) from {len(header.get('detectors', []))} detector(s) "
                f"to {filepath.resolve()}."
            ),
            "filepath": str(filepath.resolve()),
        }

    def _export_labels(self, results: dict[str, Any], filepath: Path) -> dict[str, Any]:
        """Export labels with user-selected columns.

        The ``origin`` column is always written as the last column so
        that the CSV can be re-imported without data loss.  A Good's
        ``region_box`` is written as a JSON list, and its column is left out
        when no label has a box (every label off a patch dataset).
        """
        labels = results.get("labels", [])
        columns: list[str] = list(results.get("selected_columns") or self._LABEL_BASE_COLUMNS)

        # Ensure origin is always the last column (required for re-import).
        if "origin" in columns:
            columns.remove("origin")
        columns.append("origin")
        if "region_box" in columns and not any(entry.get("region_box") is not None for entry in labels):
            columns.remove("region_box")

        total_rows = 0

        def _write(writer: "csv.writer") -> None:  # type: ignore[name-defined]
            nonlocal total_rows
            writer.writerow(columns)
            for entry in labels:
                meta = entry.get("custom_metadata") or {}
                row = []
                for col in columns:
                    if col in entry:
                        row.append(_label_cell(entry[col]))
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
        # Scan all hits to determine which optional columns are present.
        all_hits: list[tuple[str, Any, dict]] = []
        for det_result in results.get("results", {}).values():
            detector_name = det_result.get("detector_name", "unknown")
            threshold = det_result.get("threshold", "")
            for hit in det_result.get("hits", []):
                all_hits.append((detector_name, threshold, hit))

        # A payload holding both sides of the threshold labels every hit
        # (``export_find_results``); one side alone needs no column for it.
        has_label = any(h.get("label") is not None for _, _, h in all_hits)
        has_clip_start = any(h.get("clip_start") is not None for _, _, h in all_hits)
        has_clip_end = any(h.get("clip_end") is not None for _, _, h in all_hits)
        has_clip_box = any(h.get("clip_box") is not None for _, _, h in all_hits)

        base_cols = ["detector", "threshold", "filename", "category", "score"]
        if has_label:
            base_cols.append("label")
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
                    from vtscore.datasets.origin import Origin

                    origin_str = Origin.from_dict(origin).display()
                row = [
                    _sanitize_csv_cell(str(detector_name)),
                    threshold,
                    _sanitize_csv_cell(hit.get("filename", "")),
                    _sanitize_csv_cell(hit.get("category", "")),
                    hit.get("score", ""),
                ]
                if has_label:
                    row.append(_sanitize_csv_cell(hit.get("label", "")))
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


EXPORTER = ServerCsvResultsExporter()

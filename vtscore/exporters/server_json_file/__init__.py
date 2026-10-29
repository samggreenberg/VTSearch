"""Server JSON exporter – saves auto-detect results to a JSON file on the server.

Writes a JSON file to the server filesystem (the machine where the Python
process is running).  The user supplies a path (absolute or relative) as a
text field rather than a browser file picker.

No additional pip packages are required; uses only Python's ``json`` and
``pathlib`` stdlib modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from vtscore.config import DATA_DIR
from vtscore.exporters.base import PluginField, ResultsExporter, StreamResume
from vtscore.utils.io import atomic_write_json, atomic_write_stream, commit_write_stream, resumable_write_stream

_DEFAULT_JSON_PATH = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.json"


# Re-export under the historical private name so any third-party exporter
# that imported it directly keeps working.  New code should call
# :func:`vtscore.utils.io.atomic_write_text` or
# :func:`vtscore.utils.io.atomic_write_json` instead.
from vtscore.utils.io import atomic_write_text as _atomic_write_text  # noqa: E402, F401


def _ndjson_meta(header: dict[str, Any]) -> dict[str, Any]:
    """The ``_meta`` object an NDJSON hit stream opens with."""
    meta: dict[str, Any] = {
        "format": "vtsearch-hits-ndjson/v1",
        "media_type": header.get("media_type", "unknown"),
        "detectors": header.get("detectors", []),
        "keep_negatives": bool(header.get("keep_negatives", False)),
    }
    if header.get("dataset"):
        # One of several datasets from one run (#4707): say which.
        meta["dataset"] = header["dataset"]
    if header.get("max_results"):
        # --max-results (#5279): the lines are each detector's best, sorted
        # by score, not chunk-ordered.
        meta["max_results"] = header["max_results"]
    return meta


def _ndjson_hit_line(detector_name: str, hit: dict[str, Any]) -> str:
    """One hit as an NDJSON line, its ``detector`` name merged in."""
    return json.dumps({"detector": detector_name, **hit}) + "\n"


class ServerJsonResultsExporter(ResultsExporter):
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
        PluginField(
            key="filepath",
            label="Save to (server path)",
            field_type="server_path",
            description="Where on the server to write the JSON results file.",
            hint=(
                "Absolute or relative path; parent directories are created automatically.\n"
                "Template variables: {YYYYMMDD-HHMMSS}, {YYYYMMDD}, {YYYY}, {MM}, {DD}, "
                "{detector_name}, {username}."
            ),
            placeholder=_DEFAULT_JSON_PATH,
            default=_DEFAULT_JSON_PATH,
            template_vars=("YYYYMMDD-HHMMSS", "YYYYMMDD", "YYYY", "MM", "DD", "detector_name", "username"),
        ),
    ]

    def export_find_results(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write the scored run verbatim as JSON."""
        filepath = Path(field_values["filepath"])
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

    @property
    def supports_streaming(self) -> bool:
        return True

    def export_cli_streaming(
        self,
        header: dict[str, Any],
        records: Iterator[tuple[str, dict[str, Any]]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Write hits as newline-delimited JSON (NDJSON), one hit per line.

        The first line is a metadata object describing the run; every
        subsequent line is a single hit with its ``detector`` name merged in.
        Lines are flushed as they stream, so the full result set is never
        held in memory.  :func:`~vtscore.utils.io.atomic_write_stream` owns the
        tmp-file/``fsync``/rename ritual, so a crash mid-run cannot leave a
        half-written file at the destination and two concurrent exports to the
        same path can't clobber each other's temp file.
        """
        filepath = Path(field_values["filepath"])
        meta = _ndjson_meta(header)

        total_hits = 0
        with atomic_write_stream(filepath) as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for detector_name, hit in records:
                f.write(_ndjson_hit_line(detector_name, hit))
                total_hits += 1

        return {
            "message": (
                f"Streamed {total_hits} hit(s) across "
                f"{len(meta['detectors'])} detector(s) to {filepath.resolve()} (NDJSON)."
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
        """Write the same NDJSON as :meth:`export_cli_streaming`, in place, a chunk at a time.

        Each chunk's lines are ``fsync``\\ ed before the chunk is committed, so
        the byte offset the CLI checkpoints always ends on a whole chunk; a
        resumed run cuts the file back to it and appends, which leaves the same
        file an uninterrupted run would have written.
        """
        filepath = Path(field_values["filepath"])
        meta = _ndjson_meta(header)

        total_hits = 0
        with resumable_write_stream(filepath, resume.position) as f:
            if resume.position is None:
                f.write(json.dumps({"_meta": meta}) + "\n")
                resume.commit(commit_write_stream(f))
            for chunk in chunks:
                for detector_name, hit in chunk:
                    f.write(_ndjson_hit_line(detector_name, hit))
                    total_hits += 1
                resume.commit(commit_write_stream(f))

        verb = "Streamed" if resume.position is None else "Appended"
        return {
            "message": (
                f"{verb} {total_hits} hit(s) across "
                f"{len(meta['detectors'])} detector(s) to {filepath.resolve()} (NDJSON)."
            ),
            "filepath": str(filepath.resolve()),
        }

    def export_labelset(self, labelset: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Write the labelset as JSON, filtering to the selected columns.

        Entries keep their ``origin`` as a real nested dict rather than a
        display string, which is what makes the file re-importable.
        """
        filepath = Path(field_values["filepath"])
        labels = labelset.get("labels", [])
        selected_columns: list[str] | None = labelset.get("selected_columns")

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

    def export_labelset_streaming(
        self,
        header: dict[str, Any],
        elements: Iterator[dict[str, Any]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Write a labels JSON file one entry at a time, re-importable as it stands.

        Not NDJSON, unlike the hit stream: the file is the ``{"labels": [...]}``
        object :meth:`export_labelset` writes and the ``server_json_file`` label
        importer reads, with the run's metadata as sibling keys.  The metadata
        goes first and ``labels`` last, so each entry can be appended to the
        open list as it arrives; ``detector_meta`` carries the media type the
        way a detector's own labelset does.
        """
        filepath = Path(field_values["filepath"])

        meta: dict[str, Any] = {
            "format": "vtsearch-labelset-stream/v1",
            "detector_meta": {"media_type": header.get("media_type", "unknown")},
            "detectors": list(header.get("detectors", [])),
            "keep_negatives": bool(header.get("keep_negatives", False)),
        }
        if header.get("dataset"):
            meta["dataset"] = header["dataset"]
        if header.get("max_results"):
            meta["max_results"] = header["max_results"]

        total = 0
        with atomic_write_stream(filepath) as f:
            # Everything but the closing "}" of the metadata object, then the
            # open labels list.
            f.write(json.dumps(meta)[:-1] + ', "labels": [')
            for entry in elements:
                f.write(("\n" if total == 0 else ",\n") + json.dumps(entry))
                total += 1
            f.write("\n]}\n")

        return {
            "message": (
                f"Streamed {total} label(s) from {len(meta['detectors'])} detector(s) "
                f"to {filepath.resolve()} (labels JSON)."
            ),
            "filepath": str(filepath.resolve()),
        }


EXPORTER = ServerJsonResultsExporter()

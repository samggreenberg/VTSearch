"""Display labelset exporter – shows results in the browser (GUI) or prints to console (CLI).

No additional pip packages are required; in GUI mode this exporter is handled
entirely by the frontend JavaScript, and in CLI mode it prints to stdout.
"""

from __future__ import annotations

from typing import Any, Iterator

from vtscore.exporters.base import LabelsetExporter


def _format_origin(hit: dict[str, Any]) -> str:
    """Return a human-readable origin string for a hit, or ``""``."""
    origin = hit.get("origin")
    if origin is None:
        return ""
    try:
        from vtscore.datasets.origin import Origin

        return Origin.from_dict(origin).display()
    except Exception:
        return str(origin)


class DisplayLabelsetExporter(LabelsetExporter):
    """Display auto-detect results in the browser (GUI) or print to console (CLI).

    This is the default exporter: in GUI mode it performs no server-side work
    and simply passes the results back to the frontend, which renders them in
    the Auto-Detect Results modal.  In CLI mode it prints a summary to stdout.
    No configuration fields are needed.
    """

    name = "gui"
    display_name = "Display Results"
    description = "Display the results in the browser (GUI) or print to console (CLI)."
    icon = "🖥️"
    hidden_from_picker = True
    fields = []  # no questions to ask

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        # If results come from /api/labels/export (LabelSet format), convert
        # to the display format expected by displayAutodetectResults().
        if "labels" in results and "results" not in results:
            results = self._labelset_to_display(results)

        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return {
            "message": (f"Showing {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s)."),
            "display_results": results,
        }

    @staticmethod
    def _labelset_to_display(labelset: dict[str, Any]) -> dict[str, Any]:
        """Convert a LabelSet dict to the autodetect-results display format."""
        labels = labelset.get("labels", [])
        good_hits = [e for e in labels if e.get("label") == "good"]
        bad_hits = [e for e in labels if e.get("label") == "bad"]
        total = len(good_hits) + len(bad_hits)
        return {
            "media_type": f"labels ({len(good_hits)} good, {len(bad_hits)} bad)",
            "detectors_run": 0,
            "results": {
                "labels": {
                    "detector_name": "Labels",
                    "total_hits": total,
                    "hits": good_hits + bad_hits,
                },
            },
        }

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Print origins and names of Good results to stdout (no categories, no scores)."""
        lines: list[str] = []
        total_hits = 0
        for det_result in results.get("results", {}).values():
            hits = det_result.get("hits", [])
            total_hits += len(hits)
            for hit in hits:
                origin_str = _format_origin(hit)
                name = hit.get("origin_name") or hit.get("filename", "")
                if origin_str:
                    lines.append(f"  {origin_str}  {name}")
                else:
                    lines.append(f"  {name}")
        if not lines:
            print("No items predicted as Good.")
        else:
            print(f"Predicted Good ({total_hits} items):\n")
            print("\n".join(lines))
        return {
            "message": (f"Printed {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s) to stdout."),
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
        """Print each hit's origin/name to stdout as it streams in.

        Bounded memory: only the current hit is held, so this works on a
        media source far larger than RAM.  Only above-threshold hits stream
        unless ``--keep-negatives`` was set, so (matching :meth:`export_cli`)
        the printed list is the predicted-Good set.
        """
        detector_count = len(header.get("detectors", []))
        total_hits = 0
        printed_header = False
        for detector_name, hit in records:
            if hit.get("label") == "bad":
                continue
            if not printed_header:
                print("Predicted Good:\n")
                printed_header = True
            origin_str = _format_origin(hit)
            name = hit.get("origin_name") or hit.get("filename", "")
            print(f"  {origin_str}  {name}" if origin_str else f"  {name}")
            total_hits += 1
        if total_hits == 0:
            print("No items predicted as Good.")
        return {
            "message": (f"Printed {total_hits} hit(s) across {detector_count} detector(s) to stdout."),
        }


EXPORTER = DisplayLabelsetExporter()

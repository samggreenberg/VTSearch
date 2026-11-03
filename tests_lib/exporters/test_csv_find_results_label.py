"""The CSV exporter keeps the good/bad split of a find-results payload (#5283).

A payload whose ``hits`` hold both sides of the threshold - the Find Results
dialog's **Both** - labels every hit ``"good"`` or ``"bad"``
(``ResultsExporter.export_find_results``).  The CSV writes that ``label`` in
the column the streamed CSV puts it in, and leaves the column out for a
one-sided payload, whose hits carry none.
"""

from __future__ import annotations

import csv

from vtscore.exporters.server_csv_file import ServerCsvResultsExporter


def _export(hits, tmp_path):
    filepath = tmp_path / "results.csv"
    ServerCsvResultsExporter().export_find_results(
        {
            "media_type": "audio",
            "detectors_run": 1,
            "results": {
                "owl": {
                    "detector_name": "owl",
                    "threshold": 0.5,
                    "total_hits": len(hits),
                    "hits": hits,
                    "negative_hits": [],
                }
            },
        },
        {"filepath": str(filepath)},
    )
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCsvFindResultsLabel:
    def test_both_sides_keep_their_label(self, tmp_path):
        rows = _export(
            [
                {"id": 1, "filename": "a.wav", "category": "x", "score": 0.9, "label": "good"},
                {"id": 2, "filename": "b.wav", "category": "x", "score": 0.2, "label": "bad"},
            ],
            tmp_path,
        )
        assert [(r["filename"], r["label"]) for r in rows] == [("a.wav", "good"), ("b.wav", "bad")]

    def test_label_sits_where_the_streamed_csv_puts_it(self, tmp_path):
        rows = _export([{"id": 1, "filename": "a.wav", "category": "x", "score": 0.9, "label": "good"}], tmp_path)
        assert list(rows[0])[:6] == ServerCsvResultsExporter._STREAM_COLUMNS[:6]

    def test_one_sided_payload_has_no_label_column(self, tmp_path):
        rows = _export([{"id": 1, "filename": "a.wav", "category": "x", "score": 0.9}], tmp_path)
        assert "label" not in rows[0]

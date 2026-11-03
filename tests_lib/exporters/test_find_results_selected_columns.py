"""A scored run's ``selected_columns``: the Find Results dialog's column choice (#5464).

The dialog sends the columns it lists, in order; ``hit_column_value`` resolves
one cell (a hit key, else a ``custom_metadata`` key, with ``detector`` and
``threshold`` read off the detector result), the Server CSV exporter writes
exactly those columns, and the Server JSON exporter cuts each hit to them.
Without the key, both keep their fixed output.
"""

from __future__ import annotations

import csv
import json

from vtscore.exporters.base import hit_column_value
from vtscore.exporters.server_csv_file import ServerCsvResultsExporter
from vtscore.exporters.server_json_file import ServerJsonResultsExporter

_ORIGIN = {"importer": "folder", "params": {"path": "/photos"}}


def _run(selected: list[str] | None) -> dict:
    run = {
        "media_type": "image",
        "detectors_run": 1,
        "results": {
            "cats": {
                "detector_name": "cats",
                "threshold": 0.5,
                "total_hits": 2,
                "hits": [
                    {
                        "id": 1,
                        "filename": "a.jpg",
                        "origin_name": "A",
                        "md5": "m1",
                        "score": 0.9,
                        "origin": _ORIGIN,
                        "custom_metadata": {"Catalogue ID": "=cmd()", "width": 64},
                    },
                    {"id": 2, "filename": "b.jpg", "md5": "m2", "score": 0.7, "clip_box": [1, 2, 3, 4]},
                ],
                "negative_hits": [],
            }
        },
    }
    if selected is not None:
        run["selected_columns"] = selected
    return run


class TestHitColumnValue:
    def test_resolves_hit_keys_metadata_and_the_detector_result(self):
        det = _run(None)["results"]["cats"]
        hit = det["hits"][0]
        assert hit_column_value(det, hit, "filename") == "a.jpg"
        assert hit_column_value(det, hit, "Catalogue ID") == "=cmd()"
        assert hit_column_value(det, hit, "detector") == "cats"
        assert hit_column_value(det, hit, "threshold") == 0.5
        assert hit_column_value(det, hit, "origin") == _ORIGIN

    def test_a_missing_key_is_none(self):
        det = _run(None)["results"]["cats"]
        assert hit_column_value(det, det["hits"][1], "Catalogue ID") is None


class TestServerCsv:
    def _rows(self, tmp_path, selected):
        path = tmp_path / "out.csv"
        ServerCsvResultsExporter().export_find_results(_run(selected), {"filepath": str(path)})
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.reader(f))

    def test_writes_exactly_the_chosen_columns_in_order(self, tmp_path):
        rows = self._rows(tmp_path, ["origin_name", "score", "detector", "Catalogue ID", "origin", "clip_box"])
        assert rows[0] == ["origin_name", "score", "detector", "Catalogue ID", "origin", "clip_box"]
        # A text cell is guarded against formula injection; the origin reads as
        # its display string, a box as comma-joined numbers, a gap as blank.
        assert rows[1][:4] == ["A", "0.9", "cats", "'=cmd()"]
        assert rows[1][4] and rows[1][4] != json.dumps(_ORIGIN)
        assert rows[2] == ["", "0.7", "cats", "", "", "1,2,3,4"]

    def test_without_a_choice_writes_the_fixed_columns(self, tmp_path):
        rows = self._rows(tmp_path, None)
        assert rows[0] == [
            "detector",
            "threshold",
            "filename",
            "category",
            "score",
            "clip_box",
            "origin",
            "origin_name",
        ]


class TestServerJson:
    def _written(self, tmp_path, selected):
        path = tmp_path / "out.json"
        ServerJsonResultsExporter().export_find_results(_run(selected), {"filepath": str(path)})
        return json.loads(path.read_text())

    def test_cuts_each_hit_to_the_chosen_columns(self, tmp_path):
        written = self._written(tmp_path, ["filename", "width", "origin", "detector"])
        hits = written["results"]["cats"]["hits"]
        # The origin stays a dict; a metadata key comes up to the top level;
        # a key the hit has no value for is left out.
        assert hits[0] == {"filename": "a.jpg", "width": 64, "origin": _ORIGIN, "detector": "cats"}
        assert hits[1] == {"filename": "b.jpg", "detector": "cats"}
        assert written["selected_columns"] == ["filename", "width", "origin", "detector"]
        assert written["results"]["cats"]["threshold"] == 0.5

    def test_without_a_choice_writes_the_run_verbatim(self, tmp_path):
        assert self._written(tmp_path, None) == _run(None)

"""Export → import round-trip for the CSV labelset exporter / label importer.

The exporter escapes any cell starting with a spreadsheet formula prefix
(``=``, ``+``, ``-``, ``@``, tab, CR) by prefixing an apostrophe.  Its
docstring promises "the CSV can be re-imported without data loss", which
only holds if the importer undoes that escaping — otherwise a media file
named ``-take2.wav`` comes back as ``'-take2.wav`` and the origin_name /
filename resolution used by the missing-media re-ingest paths fails to
find it.
"""

from __future__ import annotations

import csv

from vtscore.exporters.server_csv_file import ServerCsvResultsExporter
from vtscore.labels.importers.server_csv_file import _parse_csv_bytes

# Names that begin with each formula prefix the exporter escapes.  Tab and
# CR are excluded: leading whitespace does not survive either side's
# ``strip()``, escaped or not.
_TRICKY_NAMES = ["-take2.wav", "@handle_post.txt", "=mc2.wav", "+1_more.wav"]


def _export_then_import(labels, tmp_path, columns=None):
    filepath = tmp_path / "labels.csv"
    ServerCsvResultsExporter().export_labelset(
        {"labels": labels, "selected_columns": columns or ["label", "md5", "origin_name", "filename", "category"]},
        {"filepath": str(filepath)},
    )
    return _parse_csv_bytes(filepath.read_bytes())


class TestCsvLabelRoundTrip:
    def test_formula_prefixed_names_survive_round_trip(self, tmp_path):
        labels = [
            {
                "label": "good",
                "md5": f"{i:032x}",
                "origin_name": name,
                "filename": name,
                "category": name,
            }
            for i, name in enumerate(_TRICKY_NAMES)
        ]

        imported = _export_then_import(labels, tmp_path)

        assert len(imported) == len(labels)
        for original, entry in zip(labels, imported):
            assert entry["origin_name"] == original["origin_name"]
            assert entry["filename"] == original["filename"]
            assert entry["category"] == original["category"]
            assert entry["md5"] == original["md5"]
            assert entry["label"] == "good"

    def test_written_csv_still_escapes_for_spreadsheets(self, tmp_path):
        """De-sanitizing on read must not weaken the on-disk escaping."""
        filepath = tmp_path / "labels.csv"
        ServerCsvResultsExporter().export_labelset(
            {
                "labels": [{"label": "good", "md5": "abc123", "filename": "-take2.wav"}],
                "selected_columns": ["label", "md5", "filename"],
            },
            {"filepath": str(filepath)},
        )
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        # ``origin`` is always appended as the last column (empty here).
        assert rows[1] == ["good", "abc123", "'-take2.wav", ""]

    def test_ordinary_names_are_unchanged(self, tmp_path):
        labels = [{"label": "bad", "md5": "abc123", "origin_name": "clip.wav", "filename": "sub/clip.wav"}]

        imported = _export_then_import(labels, tmp_path, columns=["label", "md5", "origin_name", "filename"])

        assert imported == [{"label": "bad", "md5": "abc123", "origin_name": "clip.wav", "filename": "sub/clip.wav"}]

    def test_origin_dict_survives_round_trip(self, tmp_path):
        origin = {"importer": "server_folder", "params": {"path": "/data/-odd"}}
        labels = [{"label": "good", "md5": "abc123", "origin_name": "-odd.wav", "origin": origin}]

        imported = _export_then_import(labels, tmp_path, columns=["label", "md5", "origin_name", "origin"])

        assert imported[0]["origin"] == origin
        assert imported[0]["origin_name"] == "-odd.wav"

    def test_literal_apostrophe_name_is_not_over_stripped(self, tmp_path):
        """Only an apostrophe followed by a formula prefix is escaping."""
        labels = [{"label": "good", "md5": "abc123", "filename": "'quoted.wav"}]

        imported = _export_then_import(labels, tmp_path, columns=["label", "md5", "filename"])

        assert imported[0]["filename"] == "'quoted.wav"


class TestCsvRegionBoxRoundTrip:
    """A Good's drawn box survives the CSV, written as a JSON list (issue #5130)."""

    _COLUMNS = ["label", "md5", "origin_name", "region_box"]

    def test_box_survives_round_trip(self, tmp_path):
        labels = [
            {"label": "good", "md5": "a", "origin_name": "a.png", "region_box": [0.1, 0.2, 0.7, 0.8]},
            {"label": "good", "md5": "b", "origin_name": "b.png"},
            {"label": "bad", "md5": "c", "origin_name": "c.png"},
        ]

        imported = _export_then_import(labels, tmp_path, columns=self._COLUMNS)

        assert [e.get("region_box") for e in imported] == [[0.1, 0.2, 0.7, 0.8], None, None]

    def test_box_is_written_as_json(self, tmp_path):
        filepath = tmp_path / "labels.csv"
        ServerCsvResultsExporter().export_labelset(
            {
                "labels": [{"label": "good", "md5": "a", "region_box": [0.0, 0.25, 0.5, 1.0]}],
                "selected_columns": ["md5", "region_box"],
            },
            {"filepath": str(filepath)},
        )
        with open(filepath, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows == [["md5", "region_box", "origin"], ["a", "[0.0, 0.25, 0.5, 1.0]", ""]]

    def test_column_is_left_out_when_no_label_has_a_box(self, tmp_path):
        filepath = tmp_path / "labels.csv"
        ServerCsvResultsExporter().export_labelset(
            {"labels": [{"label": "good", "md5": "a"}], "selected_columns": self._COLUMNS},
            {"filepath": str(filepath)},
        )
        with open(filepath, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == ["label", "md5", "origin_name", "origin"]

    def test_default_columns_carry_the_box(self, tmp_path):
        """A labelset export with no ``selected_columns`` keeps the box too."""
        filepath = tmp_path / "labels.csv"
        ServerCsvResultsExporter().export_labelset(
            {"labels": [{"label": "good", "md5": "a", "region_box": [0.1, 0.2, 0.3, 0.4]}]},
            {"filepath": str(filepath)},
        )
        assert _parse_csv_bytes(filepath.read_bytes())[0]["region_box"] == [0.1, 0.2, 0.3, 0.4]

    def test_importer_ignores_a_box_on_a_bad_row(self):
        """No-votes are image-level, so a hand-edited box on one is dropped."""
        csv_text = 'md5,label,region_box\na,bad,"[0.1, 0.2, 0.3, 0.4]"\n'

        assert _parse_csv_bytes(csv_text.encode()) == [{"md5": "a", "label": "bad"}]

    def test_importer_ignores_a_malformed_box(self):
        csv_text = 'md5,label,region_box\na,good,"[0.1, 0.2]"\nb,good,not json\nc,good,"[true, 0, 1, 1]"\n'

        assert _parse_csv_bytes(csv_text.encode()) == [
            {"md5": "a", "label": "good"},
            {"md5": "b", "label": "good"},
            {"md5": "c", "label": "good"},
        ]


class TestCsvImporterRaggedRows:
    def test_short_row_does_not_crash(self):
        """``csv.DictReader`` fills missing trailing cells with ``None``."""
        csv_text = "md5,label,origin_name,filename\nabc123,good\n"

        result = _parse_csv_bytes(csv_text.encode())

        assert result == [{"md5": "abc123", "label": "good"}]

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

from vtscore.exporters.server_csv_file import ServerCsvLabelsetExporter
from vtscore.labels.importers.server_csv_file import _parse_csv_bytes

# Names that begin with each formula prefix the exporter escapes.  Tab and
# CR are excluded: leading whitespace does not survive either side's
# ``strip()``, escaped or not.
_TRICKY_NAMES = ["-take2.wav", "@handle_post.txt", "=mc2.wav", "+1_more.wav"]


def _export_then_import(labels, tmp_path, columns=None):
    filepath = tmp_path / "labels.csv"
    ServerCsvLabelsetExporter().export(
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
        ServerCsvLabelsetExporter().export(
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


class TestCsvImporterRaggedRows:
    def test_short_row_does_not_crash(self):
        """``csv.DictReader`` fills missing trailing cells with ``None``."""
        csv_text = "md5,label,origin_name,filename\nabc123,good\n"

        result = _parse_csv_bytes(csv_text.encode())

        assert result == [{"md5": "abc123", "label": "good"}]

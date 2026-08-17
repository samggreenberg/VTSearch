"""The per-media prediction dump: schema, alignment, and the env gate.

These dumps are what lets a report show the errors behind an error rate, and
the whole point is that a human can check a row against the image. So the two
things worth pinning are that a row's label/score belong to the media named on
it, and that the dump stays off unless a run explicitly asks for it — the
alternative is a 270-cell array writing a file per step.
"""

from __future__ import annotations

import csv

from vtscore.eval.score_dumps import COLUMNS, maybe_dump_predictions, write_prediction_dump

MEDIAS = {
    7: {"filename": "000007.jpg", "categories": ["sky", "cloud"]},
    9: {"origin_name": "vg/9.jpg", "categories": []},
    11: {"filename": "000011.jpg"},
}


def _read(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def test_dump_carries_the_annotations_that_make_a_label_checkable(tmp_path):
    out = tmp_path / "cell.csv"
    write_prediction_dump(out, MEDIAS, [7, 9, 11], [0.9, 0.4, 0.1], [1, 0, 0], 0.5, "sky")

    rows = _read(out)
    assert list(rows[0]) == list(COLUMNS)
    assert [r["media_id"] for r in rows] == ["7", "9", "11"]
    # Score, label and identity travel together: the row is inspectable.
    assert rows[0]["filename"] == "000007.jpg"
    assert rows[0]["all_categories"] == "sky|cloud"
    assert rows[0]["label"] == "1"
    assert float(rows[0]["score"]) == 0.9
    # `origin_name` stands in when there is no filename; no annotations is "".
    assert rows[1]["filename"] == "vg/9.jpg"
    assert rows[1]["all_categories"] == ""
    assert rows[2]["all_categories"] == ""
    assert {r["threshold"] for r in rows} == {"0.5"}
    assert {r["target_category"] for r in rows} == {"sky"}


def test_misaligned_scores_raise_rather_than_writing_a_wrong_row(tmp_path):
    # A short score array silently zipped against the ids would attach one
    # media's score to another's filename, which is worse than no dump at all.
    try:
        write_prediction_dump(tmp_path / "bad.csv", MEDIAS, [7, 9, 11], [0.9, 0.4], [1, 0], 0.5, "sky")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError from the strict zip")


def test_env_gate_is_off_by_default_and_names_the_file_when_on(tmp_path, monkeypatch):
    monkeypatch.delenv("VTS_DUMP_TEST_SCORES", raising=False)
    maybe_dump_predictions(MEDIAS, [7], [0.9], [1], 0.5, "sky")
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setenv("VTS_DUMP_TEST_SCORES", str(tmp_path))
    monkeypatch.setenv("VTS_DUMP_TAG", "vg_siglip_sky")
    maybe_dump_predictions(MEDIAS, [7], [0.9], [1], 0.5, "sky", suffix="__eval")
    assert (tmp_path / "vg_siglip_sky__eval.csv").exists()
    # No temp file left behind: the write is a rename, so a reader never sees
    # a half-written dump.
    assert not list(tmp_path.glob("*.tmp"))

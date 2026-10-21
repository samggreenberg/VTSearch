"""A Good's ``region_box`` survives Export labels → file → import (issue #5130).

On a patch dataset a Good label's drawn box decides which patch the label
pools to, so ``labelset_signature`` counts it.  The documented way to move a
detector (``docs/user/howto/move-a-detector.md``) is **Export labels → Server
JSON File**, then **Trained → Create & Import** at the other end.  The Export
modal used to leave ``region_box`` out of the columns it sends, so the file
lost every box and the rebuilt detector pooled its Goods whole-image.

These tests drive the routes the way the modal and the New Detector form do,
with the column list the modal sends (its Vitest spec pins the same list).
"""

from __future__ import annotations

import json

import pytest

from vtscore.datasets.labelset import LabelSet
from vtscore.detectors.model_loading import labelset_signature
from vtsearch.state import bad_votes, clear_votes, good_votes, medias, vote_region_boxes

#: What ``ExportModalComponent.exportColumns`` sends with every column ticked:
#: ``BASE_COLUMNS`` then ``ALWAYS_EXPORT_KEYS``.
_MODAL_COLUMNS = ["label", "md5", "filename", "category", "origin", "origin_name", "region_box"]

_BOX = [0.1, 0.2, 0.7, 0.8]

_FORMATS = [("server_json_file", "labels.json"), ("server_csv_file", "labels.csv")]


#: The test medias carry no origin, and Create & Import reads the media type
#: from the labels' origins, so the voted medias get one for the test.
_ORIGIN = {"importer": "server_folder", "params": {"path": "/tmp/image_clips", "media_type": "image"}}


@pytest.fixture(autouse=True)
def _voted_medias_have_an_origin(monkeypatch):
    for mid in (1, 2, 3):
        monkeypatch.setitem(medias[mid], "origin", _ORIGIN)


def _vote_boxed_labelset(client) -> list[dict]:
    """Vote a boxed Good, a plain Good and a Bad; return the export route's labels."""
    assert client.post("/api/medias/1/vote", json={"target": "good", "region_box": _BOX}).status_code == 200
    assert client.post("/api/medias/2/vote", json={"target": "good"}).status_code == 200
    assert client.post("/api/medias/3/vote", json={"target": "bad"}).status_code == 200
    res = client.get("/api/labels/export")
    assert res.status_code == 200
    return res.get_json()["labels"]


def _export_like_the_modal(client, exporter: str, labels: list[dict], filepath) -> None:
    res = client.post(
        "/api/exporters/export",
        json={
            "exporter_name": exporter,
            "field_values": {"filepath": str(filepath)},
            "results": {"labels": labels, "selected_columns": _MODAL_COLUMNS},
            "payload_kind": "labelset",
        },
    )
    assert res.status_code == 200, res.get_json()


@pytest.mark.parametrize(("fmt", "filename"), _FORMATS)
def test_create_and_import_keeps_the_box(client, tmp_path, fmt, filename):
    exported = _vote_boxed_labelset(client)
    filepath = tmp_path / filename
    _export_like_the_modal(client, fmt, exported, filepath)

    res = client.post(
        f"/api/detectors/registry/from-labelset/{fmt}",
        json={"name": "Moved", "filepath": str(filepath)},
    )
    assert res.status_code == 201, res.get_json()

    rebuilt = LabelSet.from_dict(client.get("/api/detectors/Moved").get_json()["labelset"])
    boxes = {el.md5: el.region_box for el in rebuilt.elements}
    assert boxes[medias[1]["md5"]] == tuple(_BOX)
    assert boxes[medias[2]["md5"]] is None
    assert boxes[medias[3]["md5"]] is None
    # The rebuilt detector is the exported one, box for box.
    assert labelset_signature(rebuilt) == labelset_signature(LabelSet.from_dict({"labels": exported}))


@pytest.mark.parametrize(("fmt", "filename"), _FORMATS)
def test_label_importer_keeps_the_box(client, tmp_path, fmt, filename):
    """Import labels into an existing detector applies the box to the vote too."""
    exported = _vote_boxed_labelset(client)
    filepath = tmp_path / filename
    _export_like_the_modal(client, fmt, exported, filepath)
    clear_votes()
    assert not vote_region_boxes

    res = client.post(f"/api/label-importers/import/{fmt}", json={"filepath": str(filepath)})
    assert res.status_code == 200, res.get_json()
    assert res.get_json()["applied"] == 3

    assert set(good_votes) == {1, 2}
    assert set(bad_votes) == {3}
    assert dict(vote_region_boxes) == {1: tuple(_BOX)}


def test_json_file_writes_the_box_only_where_drawn(client, tmp_path):
    exported = _vote_boxed_labelset(client)
    filepath = tmp_path / "labels.json"
    _export_like_the_modal(client, "server_json_file", exported, filepath)

    rows = {row["md5"]: row for row in json.loads(filepath.read_text())["labels"]}
    assert rows[medias[1]["md5"]]["region_box"] == _BOX
    assert "region_box" not in rows[medias[2]["md5"]]
    assert "region_box" not in rows[medias[3]["md5"]]

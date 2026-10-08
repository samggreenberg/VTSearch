"""``GET /api/labeling-status`` on a document (tiled structural) dataset (#4488).

A document detector stops on the dry run, so the route answers
``stop_rule: "dry_run"`` with the readout and the three lights ``off``, from
the vote order alone; every other dataset answers ``stop_rule: "lights"``.
"""

from __future__ import annotations

import numpy as np

import vtsearch.routes.eval as eval_routes
from vtsearch.state import bad_votes, good_votes, label_history, medias


def _vote(cid: int, label: str) -> None:
    (good_votes if label == "good" else bad_votes)[cid] = None
    label_history.append((cid, label, 0.0))


def _tiled(monkeypatch) -> None:
    """Make the active dataset a tiled one, as ``sift_vlad_doc`` media are."""
    monkeypatch.setattr(eval_routes, "_TILED_MEMO", {})
    monkeypatch.setitem(medias[1], "tile_vectors", np.zeros((1, 4), dtype=np.float16))


def test_a_photo_dataset_stops_on_the_lights(client, monkeypatch):
    monkeypatch.setattr(eval_routes, "_TILED_MEMO", {})
    body = client.get("/api/labeling-status").get_json()
    assert body["stop_rule"] == "lights"
    assert "dry_run" not in body


def test_a_document_dataset_stops_on_the_dry_run(client, monkeypatch):
    _tiled(monkeypatch)
    _vote(1, "good")
    for cid in (4, 5, 6):
        _vote(cid, "bad")
    body = client.get("/api/labeling-status").get_json()
    assert body["stop_rule"] == "dry_run"
    assert body["stale"] is False
    assert (body["good_count"], body["bad_count"]) == (1, 3)
    assert (body["dry_run"]["run"], body["dry_run"]["target"], body["dry_run"]["status"]) == (3, 16, "red")
    assert {body[k]["status"] for k in ("smart", "stable", "span")} == {"off"}


def test_a_document_dataset_never_schedules_the_lights_refresh(client, monkeypatch):
    """The readout is the vote order alone: nothing trains on the poll thread or behind it."""
    _tiled(monkeypatch)
    _vote(1, "good")
    _vote(4, "bad")

    def _refused(*_a, **_k):
        raise AssertionError("a document dataset scheduled the Smart/Stable refresh")

    monkeypatch.setattr(eval_routes, "_schedule_status_refresh", _refused)
    monkeypatch.setattr(eval_routes, "compute_labeling_status", _refused)
    assert client.get("/api/labeling-status").get_json()["stop_rule"] == "dry_run"


def test_the_tiled_answer_is_kept_per_dataset_and_count(client, monkeypatch):
    """A large photo dataset is scanned for tiles once, not on every 2 s poll."""
    import vtscore.training.structural_stage1 as stage1

    scans = []
    real = stage1.snapshot_has_tiles

    def _counted(snap):
        scans.append(len(snap))
        return real(snap)

    monkeypatch.setattr(eval_routes, "_TILED_MEMO", {})
    monkeypatch.setattr(stage1, "snapshot_has_tiles", _counted)
    for _ in range(3):
        assert client.get("/api/labeling-status").get_json()["stop_rule"] == "lights"
    assert len(scans) == 1

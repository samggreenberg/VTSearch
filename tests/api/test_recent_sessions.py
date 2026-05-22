"""Tests for the Recent Sessions API (``/api/sessions/recent``)."""

from __future__ import annotations

from vtsearch import settings
from vtscore.datasets.registry import register_dataset
from vtscore.detectors.registry import register_detector
from vtsearch.routes.sessions import MAX_RECENT_SESSIONS


def _register_pair(suffix: str = "1") -> tuple[str, str]:
    """Register one dataset + one detector and return their ids."""
    ds = register_dataset(
        name=f"ds-{suffix}",
        media_type="image",
        num_items=1,
        pkl_path=f"/tmp/ds-{suffix}.pkl",
        embedder="siglip",
    )
    det = register_detector(name=f"det-{suffix}", media_type="image")
    return ds["id"], det["id"]


class TestRecentSessionsContract:
    """GET/POST shape and behavior."""

    def test_get_empty_by_default(self, client):
        resp = client.get("/api/sessions/recent")
        assert resp.status_code == 200
        assert resp.get_json() == {"sessions": []}

    def test_post_inserts_and_returns_entry(self, client):
        ds_id, det_id = _register_pair()
        resp = client.post(
            "/api/sessions/recent",
            json={"dataset_id": ds_id, "detector_id": det_id},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["sessions"]) == 1
        entry = data["sessions"][0]
        assert entry["dataset_id"] == ds_id
        assert entry["detector_id"] == det_id
        assert entry["dataset_name"] == "ds-1"
        assert entry["detector_name"] == "det-1"
        assert isinstance(entry["last_activity"], float)
        assert entry["last_activity"] > 0

    def test_post_dedupes_and_moves_to_front(self, client):
        ds1, det1 = _register_pair("a")
        ds2, det2 = _register_pair("b")
        # Insert pair A, then pair B, then pair A again.
        client.post("/api/sessions/recent", json={"dataset_id": ds1, "detector_id": det1})
        client.post("/api/sessions/recent", json={"dataset_id": ds2, "detector_id": det2})
        resp = client.post(
            "/api/sessions/recent",
            json={"dataset_id": ds1, "detector_id": det1},
        )
        sessions = resp.get_json()["sessions"]
        assert len(sessions) == 2
        # Pair A should now be most-recent.
        assert sessions[0]["dataset_id"] == ds1
        assert sessions[0]["detector_id"] == det1
        assert sessions[1]["dataset_id"] == ds2

    def test_caps_at_max(self, client):
        # Insert more than MAX_RECENT_SESSIONS distinct pairs.
        for i in range(MAX_RECENT_SESSIONS + 3):
            ds_id, det_id = _register_pair(f"cap-{i}")
            client.post(
                "/api/sessions/recent",
                json={"dataset_id": ds_id, "detector_id": det_id},
            )
        resp = client.get("/api/sessions/recent")
        assert len(resp.get_json()["sessions"]) == MAX_RECENT_SESSIONS

    def test_filters_deleted_ids(self, client):
        # Insert a valid entry plus a stale one written directly to settings.
        ds_id, det_id = _register_pair("real")
        client.post(
            "/api/sessions/recent",
            json={"dataset_id": ds_id, "detector_id": det_id},
        )
        raw = settings.get_recent_sessions()
        raw.append(
            {
                "dataset_id": "ghost-dataset",
                "detector_id": "ghost-detector",
                "last_activity": 1.0,
            }
        )
        settings.set_recent_sessions(raw)
        resp = client.get("/api/sessions/recent")
        sessions = resp.get_json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["dataset_id"] == ds_id

    def test_post_requires_both_ids(self, client):
        resp = client.post("/api/sessions/recent", json={"dataset_id": "x"})
        # Marshmallow validation failure → 422 (standard error envelope).
        assert resp.status_code == 422

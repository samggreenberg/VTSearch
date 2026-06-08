"""Tests for ``POST /api/dataset/promote``.

The promote endpoint turns a set of media items from the active dataset
(the Find "Goods" pile) into a brand-new saved dataset. The promoted
items keep their origins and embeddings; the new dataset gets a fresh
``created_at`` but inherits the source dataset's ``expires_at``.
"""

from __future__ import annotations


class TestPromoteToDataset:
    def test_creates_registered_dataset_and_roundtrips(self, client, tmp_path):
        from vtscore.datasets import registry as reg
        from vtscore.datasets.loader import load_dataset_from_pickle
        from vtsearch.state import snapshot_medias

        snap = snapshot_medias()
        ids = list(snap.keys())[:3]
        assert len(ids) >= 1

        before = len(reg.list_datasets())
        resp = client.post(
            "/api/dataset/promote",
            json={"name": "My Promoted Set", "media_ids": ids},
        )
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["ok"] is True
        assert body["name"] == "My Promoted Set"
        assert body["num_items"] == len(ids)

        # Registry grew by exactly one, and the new entry is findable.
        assert len(reg.list_datasets()) == before + 1
        entry = reg.get_dataset(body["dataset_id"])
        assert entry is not None
        assert entry["name"] == "My Promoted Set"
        assert entry["num_items"] == len(ids)
        assert entry["origin"] == "promote"

        # The pkl exists and round-trips back into a loadable dataset whose
        # items preserve their original origins.
        roundtrip: dict = {}
        load_dataset_from_pickle(entry["pkl_path"], roundtrip)
        assert len(roundtrip) == len(ids)
        original_origins = {
            (snap[cid].get("origin_name") or snap[cid].get("filename")) for cid in ids
        }
        promoted_origins = {
            (m.get("origin_name") or m.get("filename")) for m in roundtrip.values()
        }
        assert promoted_origins == original_origins

    def test_renumbers_ids_from_one(self, client):
        from vtscore.datasets import registry as reg
        from vtscore.datasets.loader import load_dataset_from_pickle
        from vtsearch.state import snapshot_medias

        ids = list(snapshot_medias().keys())[:2]
        resp = client.post(
            "/api/dataset/promote",
            json={"name": "Renumbered", "media_ids": ids},
        )
        assert resp.status_code == 200
        entry = reg.get_dataset(resp.get_json()["dataset_id"])
        roundtrip: dict = {}
        load_dataset_from_pickle(entry["pkl_path"], roundtrip)
        assert sorted(roundtrip.keys()) == list(range(1, len(ids) + 1))

    def test_inherits_expires_at_with_fresh_created_at(self, client):
        from vtscore.datasets import registry as reg
        from vtscore.state import core as state_core
        from vtsearch.state import snapshot_medias

        snap = snapshot_medias()
        ids = list(snap.keys())[:2]

        # Register a source dataset with an explicit death date and load its
        # medias into a context whose id matches, so the active context (via
        # the X-Dataset-Id header) resolves to a registered dataset.
        src = reg.register_dataset(
            name="Source DS",
            media_type="audio",
            num_items=len(ids),
            pkl_path="/tmp/source-does-not-need-to-exist.pkl",
            expires_at=99_999.0,
        )
        reg.add_loaded_id(src["id"])
        ctx = state_core.DatasetContext(src["id"])
        for cid in ids:
            ctx.medias[cid] = snap[cid]
        state_core.register_context(ctx)
        try:
            resp = client.post(
                "/api/dataset/promote",
                json={"name": "Inherited DS", "media_ids": ids},
                headers={"X-Dataset-Id": src["id"]},
            )
            assert resp.status_code == 200, resp.get_json()
            new_entry = reg.get_dataset(resp.get_json()["dataset_id"])
            # Death date inherited; created date is fresh (not the source's).
            assert new_entry["expires_at"] == 99_999.0
            assert new_entry["created_at"] != src["created_at"]
        finally:
            state_core.unregister_context(src["id"])
            reg.remove_loaded_id(src["id"])

    def test_rejects_unknown_ids(self, client):
        resp = client.post(
            "/api/dataset/promote",
            json={"name": "Nope", "media_ids": [999_999_999]},
        )
        assert resp.status_code == 400
        assert "current dataset" in resp.get_json()["message"]

    def test_requires_nonempty_name(self, client):
        from vtsearch.state import snapshot_medias

        ids = list(snapshot_medias().keys())[:1]
        resp = client.post(
            "/api/dataset/promote",
            json={"name": "", "media_ids": ids},
        )
        assert resp.status_code == 422

    def test_requires_media_ids(self, client):
        resp = client.post(
            "/api/dataset/promote",
            json={"name": "No items", "media_ids": []},
        )
        assert resp.status_code == 422

    def test_400_when_no_dataset_loaded(self, client):
        from vtsearch.state import medias

        saved = dict(medias)
        medias.clear()
        try:
            resp = client.post(
                "/api/dataset/promote",
                json={"name": "Empty", "media_ids": [1, 2, 3]},
            )
            assert resp.status_code == 400
            assert "No dataset loaded" in resp.get_json()["message"]
        finally:
            medias.update(saved)

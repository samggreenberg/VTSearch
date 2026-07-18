"""Tests for ``POST /api/dataset/promote``.

The promote endpoint turns a set of media items from the active dataset
(the Find "Goods" pile) into a brand-new saved dataset. The promoted
items keep their origins and embeddings; the new dataset gets a fresh
``created_at`` but inherits the source dataset's ``expires_at``.

The route follows the task-id/poll contract: the subset snapshot and
metadata derivation happen synchronously (so a bad request still 400s at
request time), while the coverage-atlas build, pickle write, and registry
insert run in a background task. The response carries the ``task_id``;
the finished task's ``dataset_id`` association identifies the new
dataset.
"""

from __future__ import annotations

import time

from vtscore.concurrency.progress import loading_tasks


def _wait_for_task(task_id: str, timeout: float = 30.0) -> dict:
    """Poll until background task *task_id* finishes; return its snapshot."""
    deadline = time.time() + timeout
    while not loading_tasks.is_finished(task_id) and time.time() < deadline:
        time.sleep(0.05)
    assert loading_tasks.is_finished(task_id), "promote task never finished"
    snap = next((t for t in loading_tasks.list_tasks() if t["task_id"] == task_id), None)
    assert snap is not None, "finished promote task pruned before it could be inspected"
    return snap


def _promote(client, name: str, media_ids: list[int], headers: dict | None = None) -> dict:
    """Kick off a promote, wait for the background task, and return its snapshot."""
    resp = client.post(
        "/api/dataset/promote",
        json={"name": name, "media_ids": media_ids},
        headers=headers or {},
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["ok"] is True
    assert body["task_id"]
    snap = _wait_for_task(body["task_id"])
    assert not snap.get("error"), snap
    return snap


class TestPromoteToDataset:
    def test_creates_registered_dataset_and_roundtrips(self, client, tmp_path):
        from vtscore.datasets import registry as reg
        from vtscore.datasets.loader import load_dataset_from_pickle
        from vtsearch.state import snapshot_medias

        snap = snapshot_medias()
        ids = list(snap.keys())[:3]
        assert len(ids) >= 1

        before = len(reg.list_datasets())
        task = _promote(client, "My Promoted Set", ids)

        # Registry grew by exactly one, and the new entry is findable via the
        # finished task's dataset_id association.
        assert len(reg.list_datasets()) == before + 1
        entry = reg.get_dataset(task["dataset_id"])
        assert entry is not None
        assert entry["name"] == "My Promoted Set"
        assert entry["num_items"] == len(ids)
        assert entry["origin"] == "promote"

        # The pkl exists and round-trips back into a loadable dataset whose
        # items preserve their original origins.
        roundtrip: dict = {}
        load_dataset_from_pickle(entry["pkl_path"], roundtrip)
        assert len(roundtrip) == len(ids)
        original_origins = {(snap[cid].get("origin_name") or snap[cid].get("filename")) for cid in ids}
        promoted_origins = {(m.get("origin_name") or m.get("filename")) for m in roundtrip.values()}
        assert promoted_origins == original_origins

    def test_caches_coverage_atlas_in_pickle(self, client):
        """The promoted pickle carries a cached coverage atlas so reopening it
        restores the atlas instead of rebuilding it (hierarchical k-means) on
        every reload. The subset's renumbered IDs must match the cache exactly.
        """
        from vtscore.datasets import registry as reg
        from vtscore.datasets.loader import load_dataset_from_pickle
        from vtscore.state import restore_coverage_atlas_from_cache
        from vtscore.state.core import DatasetContext
        from vtsearch.state import snapshot_medias

        ids = list(snapshot_medias().keys())[:3]
        assert len(ids) >= 1
        task = _promote(client, "Cached Atlas", ids)
        entry = reg.get_dataset(task["dataset_id"])
        assert entry is not None

        roundtrip: dict = {}
        cached = load_dataset_from_pickle(entry["pkl_path"], roundtrip)
        # An atlas payload was written, and it restores cleanly onto the reloaded
        # (renumbered) medias — i.e. no rebuild is needed on reload.
        assert cached is not None, "promote should cache a coverage atlas in the pickle"
        ctx = DatasetContext("reload")
        ctx.medias = roundtrip
        assert restore_coverage_atlas_from_cache(ctx, cached) is True

    def test_renumbers_ids_from_one(self, client):
        from vtscore.datasets import registry as reg
        from vtscore.datasets.loader import load_dataset_from_pickle
        from vtsearch.state import snapshot_medias

        ids = list(snapshot_medias().keys())[:2]
        task = _promote(client, "Renumbered", ids)
        entry = reg.get_dataset(task["dataset_id"])
        assert entry is not None
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
            task = _promote(client, "Inherited DS", ids, headers={"X-Dataset-Id": src["id"]})
            new_entry = reg.get_dataset(task["dataset_id"])
            assert new_entry is not None
            # Death date inherited; created date is fresh (not the source's).
            assert new_entry["expires_at"] == 99_999.0
            assert new_entry["created_at"] != src["created_at"]
        finally:
            state_core.unregister_context(src["id"])
            reg.remove_loaded_id(src["id"])

    def test_background_failure_reports_task_error_and_cleans_up(self, client, monkeypatch):
        """A failure inside the background task (here: serialization) must not
        leave a registry entry or an orphaned pkl; it surfaces as the task's
        ``error`` field instead of an HTTP error.
        """
        from vtscore.datasets import registry as reg
        from vtsearch.routes.datasets import staging as staging_module
        from vtsearch.state import snapshot_medias

        def boom(*args, **kwargs):
            raise RuntimeError("serialization exploded")

        monkeypatch.setattr(staging_module, "export_dataset_to_file", boom)

        ids = list(snapshot_medias().keys())[:2]
        before = len(reg.list_datasets())
        resp = client.post(
            "/api/dataset/promote",
            json={"name": "Doomed", "media_ids": ids},
        )
        assert resp.status_code == 200, resp.get_json()
        snap = _wait_for_task(resp.get_json()["task_id"])
        assert "serialization exploded" in (snap.get("error") or "")
        assert snap.get("dataset_id") is None
        assert len(reg.list_datasets()) == before

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

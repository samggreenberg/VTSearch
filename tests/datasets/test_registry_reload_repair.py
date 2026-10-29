"""Reloading a dataset saved with an incomplete item repairs it, once (issue #3798).

Before the embed stage emptied such items, an import could save an item that
one of its embedders could not read but another had: it kept a vector, so it
was saved, and every Browse, sort or Train request for the missing space raised
``has no embedding for embedder`` - on every load, since nothing re-checked.
``POST /api/datasets/registry/<id>/load`` now removes those items, rewrites the
dataset's file without them, lists them on the registry entry and toasts.
"""

from __future__ import annotations

import pickle
import threading

import numpy as np

from tests.helpers import make_image_media

SECOND = "aux_embedder"


def _saved_medias(n: int, *, missing_second: tuple[int, ...]) -> dict[int, dict]:
    """*n* image media carrying a ``siglip`` and an ``aux_embedder`` vector, bar *missing_second*."""
    medias = {}
    for i in range(1, n + 1):
        media = make_image_media(i, embedding_dim=8)
        media["duration"] = 0
        if i not in missing_second:
            media["embeddings"][SECOND] = np.full(8, float(i), dtype=np.float32)
        medias[i] = media
    return medias


def _load(client, monkeypatch, dataset_id: str) -> None:
    """Load *dataset_id* from the registry and wait for its worker to finish."""
    import vtsearch.threading as vts_threading

    spawned: list[threading.Thread] = []
    real_spawn = vts_threading.spawn

    def _spawn(*args, **kwargs):
        thread = real_spawn(*args, **kwargs)
        spawned.append(thread)
        return thread

    monkeypatch.setattr(vts_threading, "spawn", _spawn)
    resp = client.post(f"/api/datasets/registry/{dataset_id}/load")
    assert resp.status_code == 200
    (worker,) = [t for t in spawned if t.name == f"ds-load-{dataset_id[:8]}"]
    worker.join(timeout=30)
    assert not worker.is_alive()


class TestRegistryReloadRepair:
    def test_an_incomplete_item_is_removed_listed_and_saved_away(self, client, monkeypatch):
        from vtscore.concurrency.notifications import notifications
        from vtscore.datasets.container import write_container
        from vtscore.datasets.loader import load_dataset_from_pickle
        from vtscore.datasets.registry import get_dataset, register_dataset, unregister_dataset
        from vtscore.embedding.matrix import get_embedding_matrix
        from vtscore.state.core import get_context
        from vtsearch.settings import get_saved_datasets_dir

        medias = _saved_medias(5, missing_second=(3,))
        ds_dir = get_saved_datasets_dir()
        ds_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = ds_dir / "test_registry_reload_repair.pkl"
        write_container(pkl_path, pickle.dumps({"medias": medias}), {"format_version": 1})
        entry = register_dataset(
            name="Photos",
            media_type="image",
            num_items=len(medias),
            pkl_path=str(pkl_path),
            embedder="siglip",
            created_by="default",
        )
        dataset_id = entry["id"]
        received = []
        notifications.subscribe(received.append)
        try:
            _load(client, monkeypatch, dataset_id)

            ctx = get_context(dataset_id)
            assert ctx is not None
            assert sorted(ctx.medias) == [1, 2, 4, 5]
            # The space the item was missing builds again.
            assert list(get_embedding_matrix(ctx, embedder_name=SECOND)[0]) == [1, 2, 4, 5]

            repaired = get_dataset(dataset_id)
            assert repaired["num_items"] == 4
            assert repaired["num_dropped"] == 1
            (group,) = repaired["dropped_groups"]
            assert group["items"] == ["image_3.png"]
            assert SECOND in group["reason"]
            notes = [n for n in received if n.source == "Dataset load"]
            assert len(notes) == 1 and notes[0].items == ("image_3.png",)

            stats = client.get(f"/api/datasets/registry/{dataset_id}/stats").get_json()
            assert stats["num_dropped"] == 1
            assert stats["dropped_groups"] == repaired["dropped_groups"]
            listed = client.get("/api/datasets/registry").get_json()["datasets"]
            assert next(d for d in listed if d["id"] == dataset_id)["num_dropped"] == 1

            # The file itself no longer holds the item...
            on_disk: dict[int, dict] = {}
            load_dataset_from_pickle(pkl_path, on_disk)
            assert sorted(on_disk) == [1, 2, 4, 5]

            # ...so the next load has nothing to repair, and lists nothing twice.
            client.post(f"/api/datasets/registry/{dataset_id}/unload")
            received.clear()
            _load(client, monkeypatch, dataset_id)
            assert sorted(get_context(dataset_id).medias) == [1, 2, 4, 5]
            assert get_dataset(dataset_id)["num_dropped"] == 1
            assert [n for n in received if n.source == "Dataset load"] == []
        finally:
            notifications.unsubscribe(received.append)
            client.post(f"/api/datasets/registry/{dataset_id}/unload")
            unregister_dataset(dataset_id)
            pkl_path.unlink(missing_ok=True)

    def test_a_complete_dataset_loads_untouched(self, client, monkeypatch):
        from vtscore.datasets.container import write_container
        from vtscore.datasets.registry import get_dataset, register_dataset, unregister_dataset
        from vtsearch.settings import get_saved_datasets_dir

        medias = _saved_medias(3, missing_second=())
        ds_dir = get_saved_datasets_dir()
        ds_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = ds_dir / "test_registry_reload_complete.pkl"
        write_container(pkl_path, pickle.dumps({"medias": medias}), {"format_version": 1})
        before = pkl_path.stat().st_mtime_ns
        entry = register_dataset(
            name="Complete", media_type="image", num_items=3, pkl_path=str(pkl_path), created_by="default"
        )
        try:
            _load(client, monkeypatch, entry["id"])
            assert get_dataset(entry["id"])["num_dropped"] == 0
            assert pkl_path.stat().st_mtime_ns == before
        finally:
            client.post(f"/api/datasets/registry/{entry['id']}/unload")
            unregister_dataset(entry["id"])
            pkl_path.unlink(missing_ok=True)

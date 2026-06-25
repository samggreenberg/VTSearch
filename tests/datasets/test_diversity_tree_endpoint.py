"""App-tier tests for the on-demand diversity-tree build endpoint.

``POST /api/datasets/registry/<id>/diversity-tree`` lets the user build the
diversity index for a loaded dataset that skipped the automatic build at load
time (see Phase 2.1 Part B in ``docs/plans/scalability-plan.md``).
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from vtscore.state.core import (
    DatasetContext,
    get_context,
    register_context,
    set_thread_dataset_context,
)


def _make_media(media_id: int) -> dict:
    rng = np.random.default_rng(media_id)
    return {
        "id": media_id,
        "media_type": "audio",
        "embedder": "clap",
        "duration": 1.0,
        "file_size": 100,
        "md5": hashlib.md5(f"divtree_{media_id}".encode()).hexdigest(),
        "embedding": rng.standard_normal(64).astype(np.float32),
        "media_bytes": b"fake",
        "filename": f"divtree_{media_id}.wav",
        "category": "test",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"divtree_{media_id}.wav",
    }


def _loaded_dataset(n: int = 30):
    from vtscore.datasets.registry import add_loaded_id, register_dataset

    entry = register_dataset(name="DivTree", media_type="audio", num_items=n, pkl_path="/tmp/divtree.pkl")
    ctx = DatasetContext(entry["id"])
    for i in range(n):
        ctx.medias[i] = _make_media(i)
    register_context(ctx)
    add_loaded_id(entry["id"])
    set_thread_dataset_context(ctx)
    return entry, ctx


def _wait_for_tree(dataset_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ctx = get_context(dataset_id)
        if ctx is not None and ctx.diversity_tree is not None:
            return ctx.diversity_tree
        time.sleep(0.05)
    return None


class TestDiversityTreeEndpoint:
    def test_builds_tree_on_demand(self, client):
        entry, ctx = _loaded_dataset()
        # Simulate a large-dataset load that deferred the build.
        ctx.diversity_tree = None

        resp = client.post(f"/api/datasets/registry/{entry['id']}/diversity-tree")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["task_id"]

        tree = _wait_for_tree(entry["id"])
        assert tree is not None
        # Every media with an embedding lands in a leaf.
        assert set(tree.vector_to_leaf.keys()) == set(ctx.medias.keys())

    def test_not_loaded_returns_400(self, client):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(
            name="NotLoaded", media_type="audio", num_items=1, pkl_path="/tmp/divtree_nl.pkl"
        )
        resp = client.post(f"/api/datasets/registry/{entry['id']}/diversity-tree")
        assert resp.status_code == 400

    def test_unknown_dataset_returns_404(self, client):
        resp = client.post("/api/datasets/registry/does-not-exist/diversity-tree")
        assert resp.status_code == 404

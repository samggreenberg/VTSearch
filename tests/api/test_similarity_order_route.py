"""``POST /api/medias/similarity-order``: the Find Results list's Similarity sort (#5464).

The order is the dataset's 1-D UMAP, fit with its Browse layout.  A dataset
that has one answers ``ready`` with the asked ids in it; one that has none
answers ``building`` while the fit runs, then ``ready``.  Both fits are faked:
what is under test is the route, not the arrangement.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import numpy as np

from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.projection import Projection
from vtscore.projection import similarity_order as so
from vtscore.projection.similarity_order import SimilarityOrder
from vtscore.state.core import DatasetContext, register_context


def _load(dataset_id: str, n: int = 4, *, embedded: bool = True) -> DatasetContext:
    ctx = DatasetContext(dataset_id)
    for i in range(1, n + 1):
        media: dict[str, Any] = {"id": i, "media_type": "image", "md5": f"md5-{i}", "filename": f"{i}.jpg"}
        if embedded:
            media["embedder"] = "clip"
            media["embeddings"] = {"clip": np.random.default_rng(i).standard_normal(8).astype(np.float32)}
        ctx.medias[i] = media
    register_context(ctx)
    return ctx


def _fake_layout_fit(matrix, ids, **kwargs):
    coords = np.random.default_rng(0).standard_normal((len(ids), 2)).astype(np.float32)
    return Projection(uuid.uuid4().hex, list(ids), coords, "fake")


def _reversed(matrix, ids, **kwargs):
    return SimilarityOrder(list(reversed(ids)), "umap")


def _post(client, dataset_id: str, ids: list[int]):
    return client.post("/api/medias/similarity-order", json={"ids": ids}, headers={"X-Dataset-Id": dataset_id})


def _await_jobs() -> None:
    for _ in range(600):
        active = projection_jobs.active_jobs()
        if not active:
            return
        active[0].done_event.wait(timeout=0.05)
    raise TimeoutError("projection job did not finish")


class TestSimilarityOrderRoute:
    def test_a_stored_order_puts_the_asked_ids_in_it(self, client):
        ctx = _load("photos")
        ctx._similarity_order = SimilarityOrder([3, 1, 4, 2], "umap")
        with patch("vtscore.projection.store.pkl_path_for", return_value=None):
            res = _post(client, "photos", [1, 2, 3, 99, 1])
        # Unknown and repeated ids are dropped; the rest come in the order.
        assert res.status_code == 200
        assert res.get_json() == {"status": "ready", "ids": [3, 1, 2], "method": "umap"}

    def test_a_dataset_with_no_order_builds_one_then_answers(self, client):
        _load("fresh")
        with (
            patch("vtscore.projection.store.pkl_path_for", return_value=None),
            patch("vtscore.projection.fit_projection", side_effect=_fake_layout_fit),
            patch("vtscore.projection.service.prep_signposts_best_effort"),
            patch.object(so, "fit_similarity_order", side_effect=_reversed),
        ):
            first = _post(client, "fresh", [1, 2, 3, 4]).get_json()
            assert first["status"] == "building"
            assert first["job_id"]
            _await_jobs()
            ready = _post(client, "fresh", [2, 4]).get_json()
        assert ready == {"status": "ready", "ids": [4, 2], "method": "umap"}

    def test_a_dataset_without_embeddings_is_a_conflict(self, client):
        _load("bare", embedded=False)
        assert _post(client, "bare", [1]).status_code == 409

    def test_requires_a_dataset(self, client):
        res = client.post("/api/medias/similarity-order", json={"ids": [1]}, headers={"X-Dataset-Id": ""})
        assert res.status_code == 400

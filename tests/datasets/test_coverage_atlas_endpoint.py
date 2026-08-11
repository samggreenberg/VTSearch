"""App-tier tests for the on-demand coverage-atlas build and domain-shift endpoints.

``POST /api/datasets/registry/<id>/coverage-atlas`` lets the user build the
coverage atlas for a loaded dataset that skipped the automatic build at load
time (see the S2/S8 auto-defer item in ``docs/plans/scalability.md``).

``GET /api/datasets/registry/<id>/domain-shift`` reports how typical the
active dataset's embeddings look under ``<id>``'s coverage atlas, so a user
can tell whether a detector trained on ``<id>`` should be trusted on the
active dataset without hands-on verification.
"""

from __future__ import annotations

import time

import numpy as np

from vtscore.state.core import (
    DatasetContext,
    get_context,
    register_context,
    set_thread_dataset_context,
)
from vtscore.utils.hashing import content_md5


def _make_media(media_id: int, *, seed_offset: int = 0, shifted: bool = False) -> dict:
    rng = np.random.default_rng(media_id + seed_offset)
    emb = np.zeros(64, dtype=np.float32)
    if shifted:
        # Directions confined to the second half of the coordinates — unlike
        # anything the reference dataset (first half) contains.
        emb[32:] = rng.standard_normal(32).astype(np.float32)
        emb[32] += 4.0
    else:
        # Two well-separated clusters in the first half of the coordinates —
        # the multi-blob structure real embedding spaces have.
        emb[:32] = rng.standard_normal(32).astype(np.float32)
        emb[media_id % 2] += 4.0
    emb /= np.linalg.norm(emb)
    return {
        "id": media_id,
        "media_type": "audio",
        "embedder": "clap",
        "duration": 1.0,
        "file_size": 100,
        "md5": content_md5(f"atlas_{seed_offset}_{media_id}".encode()),
        "embeddings": {"clap": emb},
        "media_bytes": b"fake",
        "filename": f"atlas_{media_id}.wav",
        "category": "test",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"atlas_{media_id}.wav",
    }


def _loaded_dataset(n: int = 30, *, name: str = "Atlas", seed_offset: int = 0, shifted: bool = False, embedder="clap"):
    from vtscore.datasets.registry import add_loaded_id, register_dataset

    entry = register_dataset(
        name=name,
        media_type="audio",
        num_items=n,
        pkl_path=f"/tmp/{name.lower()}.pkl",
        embedder=embedder,
    )
    ctx = DatasetContext(entry["id"])
    for i in range(n):
        ctx.medias[i] = _make_media(i, seed_offset=seed_offset, shifted=shifted)
    register_context(ctx)
    add_loaded_id(entry["id"])
    set_thread_dataset_context(ctx)
    return entry, ctx


def _wait_for_atlas(dataset_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ctx = get_context(dataset_id)
        if ctx is not None and ctx.coverage_atlas is not None:
            return ctx.coverage_atlas
        time.sleep(0.05)
    return None


class TestCoverageAtlasEndpoint:
    def test_builds_atlas_on_demand(self, client):
        entry, ctx = _loaded_dataset()
        # Simulate a large-dataset load that deferred the build.
        ctx.coverage_atlas = None

        resp = client.post(f"/api/datasets/registry/{entry['id']}/coverage-atlas")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ok"] is True
        assert body["task_id"]

        atlas = _wait_for_atlas(entry["id"])
        assert atlas is not None
        # Every media with an embedding lands in a leaf.
        assert set(atlas.vector_to_leaf.keys()) == set(ctx.medias.keys())

    def test_not_loaded_returns_400(self, client):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(name="NotLoaded", media_type="audio", num_items=1, pkl_path="/tmp/atlas_nl.pkl")
        resp = client.post(f"/api/datasets/registry/{entry['id']}/coverage-atlas")
        assert resp.status_code == 400

    def test_unknown_dataset_returns_404(self, client):
        resp = client.post("/api/datasets/registry/does-not-exist/coverage-atlas")
        assert resp.status_code == 404


class TestDomainShiftEndpoint:
    def _reference_with_atlas(self, client, n: int = 120):
        """Register + load a reference dataset and build its atlas synchronously."""
        from vtscore.state.coverage_atlas import CoverageAtlas

        entry, ctx = _loaded_dataset(n, name="Reference", seed_offset=0)
        vectors = {cid: m["embeddings"]["clap"] for cid, m in ctx.medias.items()}
        ctx.coverage_atlas = CoverageAtlas(vectors, k=3)
        return entry, ctx

    def test_same_domain_not_shifted(self, client):
        ref_entry, _ = self._reference_with_atlas(client)
        target_entry, _ = _loaded_dataset(40, name="TargetSame", seed_offset=1000)

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["reference_dataset_id"] == ref_entry["id"]
        assert data["n_items"] == 40
        assert data["shifted"] is False
        # Small-sample noise puts a handful of items under alpha; the point is
        # the rate stays far from the shifted regime (which reads ~1.0 here).
        assert data["frac_atypical"] < 0.2

    def test_shifted_domain_flagged(self, client):
        ref_entry, _ = self._reference_with_atlas(client)
        target_entry, _ = _loaded_dataset(40, name="TargetShifted", seed_offset=2000, shifted=True)

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["shifted"] is True
        assert data["frac_atypical"] > 0.5
        assert data["median_pvalue"] < 0.5

    def test_reference_without_atlas_returns_400(self, client):
        ref_entry, ref_ctx = _loaded_dataset(20, name="NoAtlasRef", seed_offset=3000)
        ref_ctx.coverage_atlas = None
        target_entry, _ = _loaded_dataset(10, name="TargetNoAtlas", seed_offset=4000)

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 400

    def test_reference_not_loaded_returns_400(self, client):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(
            name="UnloadedRef", media_type="audio", num_items=1, pkl_path="/tmp/dsref.pkl", embedder="clap"
        )
        target_entry, _ = _loaded_dataset(10, name="TargetUnloadedRef", seed_offset=5000)
        resp = client.get(
            f"/api/datasets/registry/{entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 400

    def test_unknown_reference_returns_404(self, client):
        resp = client.get("/api/datasets/registry/does-not-exist/domain-shift")
        assert resp.status_code == 404

    def test_same_dataset_returns_400(self, client):
        ref_entry, _ = self._reference_with_atlas(client)
        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": ref_entry["id"]},
        )
        assert resp.status_code == 400

    def test_embedder_mismatch_returns_400(self, client):
        ref_entry, _ = self._reference_with_atlas(client)
        target_entry, _ = _loaded_dataset(10, name="TargetOtherEmb", seed_offset=6000, embedder="siglip")

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 400
        assert "mismatch" in resp.get_json()["message"].lower()

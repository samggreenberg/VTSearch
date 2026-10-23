"""App-tier tests for the on-demand coverage-atlas build and domain-shift endpoints.

``POST /api/datasets/registry/<id>/coverage-atlas`` lets the user build the
coverage atlas for a loaded dataset that skipped the automatic build at load
time (see the S2/S8 auto-defer item in ``docs/plans/scalability.md``).

``GET /api/datasets/registry/<id>/domain-shift`` reports how typical the
active dataset's embeddings look against ``<id>``'s own embeddings (a
split-conformal kNN since #3348; it needs no atlas), so a user
can tell whether a detector trained on ``<id>`` should be trusted on the
active dataset without hands-on verification.
"""

from __future__ import annotations

import threading
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


# A real hierarchical k-means build: seconds on an idle box, but well past 10 s
# on a loaded one (a 10 s wait flaked), so the waits are generous.
_BUILD_TIMEOUT = 60.0


def _wait_until(predicate, timeout: float = _BUILD_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _wait_for_atlas(dataset_id: str, timeout: float = _BUILD_TIMEOUT):
    def _built() -> bool:
        ctx = get_context(dataset_id)
        return ctx is not None and ctx.coverage_atlas is not None

    if not _wait_until(_built, timeout):
        return None
    ctx = get_context(dataset_id)
    return ctx.coverage_atlas if ctx is not None else None


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

    def test_finished_build_reports_idle(self, client):
        """A finished build's row goes ``idle``, so clients see it end at once
        rather than when the tracker prunes it (the Stats window waits on it)."""
        from vtscore.concurrency.progress import loading_tasks

        entry, _ = _loaded_dataset(name="AtlasIdle")
        task_id = client.post(f"/api/datasets/registry/{entry['id']}/coverage-atlas").get_json()["task_id"]
        # Held so the row can't be pruned from under the check once finished.
        tracker = loading_tasks.get_tracker(task_id)
        assert tracker is not None

        assert _wait_until(lambda: loading_tasks.is_finished(task_id))
        row = tracker.get()
        assert row["status"] == "idle"
        assert not row.get("error")

    def test_not_loaded_returns_400(self, client):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(name="NotLoaded", media_type="audio", num_items=1, pkl_path="/tmp/atlas_nl.pkl")
        resp = client.post(f"/api/datasets/registry/{entry['id']}/coverage-atlas")
        assert resp.status_code == 400

    def test_unknown_dataset_returns_404(self, client):
        resp = client.post("/api/datasets/registry/does-not-exist/coverage-atlas")
        assert resp.status_code == 404


class TestStatsCoverageAtlasStatus:
    """The ``coverage_atlas`` block on ``GET .../stats``, read by the Stats window's Build row (#4995)."""

    @staticmethod
    def _status(client, dataset_id: str):
        resp = client.get(f"/api/datasets/registry/{dataset_id}/stats")
        assert resp.status_code == 200
        return resp.get_json()["coverage_atlas"]

    def test_loaded_with_atlas_is_built(self, client):
        from vtscore.coverage.atlas import CoverageAtlas
        from vtscore.state.coverage import COVERAGE_ATLAS_AUTO_THRESHOLD

        entry, ctx = _loaded_dataset(name="StatsBuilt")
        ctx.coverage_atlas = CoverageAtlas({cid: m["embeddings"]["clap"] for cid, m in ctx.medias.items()}, k=3)

        assert self._status(client, entry["id"]) == {
            "built": True,
            "building": False,
            "auto_build_threshold": COVERAGE_ATLAS_AUTO_THRESHOLD,
        }

    def test_loaded_without_atlas_is_not_built(self, client):
        entry, ctx = _loaded_dataset(name="StatsNotBuilt")
        ctx.coverage_atlas = None

        status = self._status(client, entry["id"])
        assert status["built"] is False
        assert status["building"] is False

    def test_unloaded_is_null(self, client):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(name="StatsUnloaded", media_type="audio", num_items=1, pkl_path="/tmp/atlas_su.pkl")
        assert self._status(client, entry["id"]) is None

    def test_running_build_reports_building_and_absorbs_a_repeat_request(self, client, monkeypatch):
        from vtscore.concurrency.progress import loading_tasks

        started = threading.Event()

        def _endless_build(ctx, on_progress):
            started.set()
            while True:  # exits ONLY via CancelledError, which on_progress raises once cancelled
                on_progress(0, 0)
                time.sleep(0.05)

        monkeypatch.setattr("vtsearch.state.build_coverage_atlas_for_context", _endless_build)
        entry, _ = _loaded_dataset(name="StatsBuilding")
        url = f"/api/datasets/registry/{entry['id']}/coverage-atlas"

        task_id = client.post(url).get_json()["task_id"]
        assert started.wait(timeout=_BUILD_TIMEOUT)
        tracker = loading_tasks.get_tracker(task_id)
        status = self._status(client, entry["id"])
        assert status["built"] is False
        assert status["building"] is True

        # A repeat request attaches to the running build rather than starting
        # a twin that would replace its task row.
        repeat = client.post(url)
        assert repeat.status_code == 200
        assert repeat.get_json()["task_id"] == task_id
        assert loading_tasks.get_tracker(task_id) is tracker

        # Cancelled, the build leaves no atlas and nothing running.
        assert loading_tasks.cancel_task(task_id)
        assert _wait_until(lambda: not loading_tasks.worker_alive(task_id))
        status = self._status(client, entry["id"])
        assert status["built"] is False
        assert status["building"] is False


class TestDomainShiftEndpoint:
    def _reference(self, client, n: int = 120):
        """Register + load a reference dataset (no atlas: the report reads its vectors)."""
        entry, ctx = _loaded_dataset(n, name="Reference", seed_offset=0)
        ctx.coverage_atlas = None
        return entry, ctx

    def test_same_domain_not_shifted(self, client):
        ref_entry, _ = self._reference(client)
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
        ref_entry, _ = self._reference(client)
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

    def test_reference_too_small_returns_400(self, client):
        """Under 40 reference vectors no p-value can reach below alpha, so there is no verdict."""
        ref_entry, _ = _loaded_dataset(20, name="TinyRef", seed_offset=3000)
        target_entry, _ = _loaded_dataset(10, name="TargetTinyRef", seed_offset=4000)

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 400
        assert "too small" in resp.get_json()["message"].lower()

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
        ref_entry, _ = self._reference(client)
        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": ref_entry["id"]},
        )
        assert resp.status_code == 400

    def test_embedder_mismatch_returns_400(self, client):
        ref_entry, _ = self._reference(client)
        target_entry, _ = _loaded_dataset(10, name="TargetOtherEmb", seed_offset=6000, embedder="siglip")

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 400
        assert "mismatch" in resp.get_json()["message"].lower()

    def test_patch_embedder_reports(self, client):
        """A patch-embedder reference is answered since #3348.

        The atlas guard it replaced fired on 80 % of a ``dinov3_patch``
        reference's own held-out data and was refused there; the kNN report
        is uniform on its own data on every embedder measured, ``dinov3_patch``
        included (``docs/experiments/2026-10-23-atlas-node-model-3348/``).
        """
        ref_entry, _ = _loaded_dataset(120, name="RefPatch", seed_offset=7000, embedder="dinov3_patch")
        target_entry, _ = _loaded_dataset(
            40, name="TargetPatch", seed_offset=8000, shifted=True, embedder="dinov3_patch"
        )

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 200
        assert resp.get_json()["shifted"] is True

    def test_single_vector_embedder_still_reports(self, client):
        """The patch gate reads the capability, so single-vector siblings pass.

        ``dinov3_single`` is the same model family as the refused
        ``dinov3_patch`` and must not be caught by the gate: what fails is the
        patch *space*, not the architecture.
        """
        from vtscore.coverage.atlas import CoverageAtlas

        ref_entry, ref_ctx = _loaded_dataset(120, name="RefSingle", seed_offset=9000, embedder="dinov3_single")
        ref_ctx.coverage_atlas = CoverageAtlas({cid: m["embeddings"]["clap"] for cid, m in ref_ctx.medias.items()}, k=3)
        target_entry, _ = _loaded_dataset(10, name="TargetSingle", seed_offset=9500, embedder="dinov3_single")

        resp = client.get(
            f"/api/datasets/registry/{ref_entry['id']}/domain-shift",
            headers={"X-Dataset-Id": target_entry["id"]},
        )
        assert resp.status_code == 200
        assert resp.get_json()["n_items"] == 10

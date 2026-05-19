"""Cross-embedder switch tests for the active-context switcher.

When the active dataset changes and the new dataset's embedder differs from
the one the loaded detector's label vectors were built with, mixing the two
into a single MLP produces garbage. These tests guard the two halves of the
fix:

1. ``populate_label_embeddings`` invalidates ``det_ctx.label_embeddings`` and
   re-stamps ``det_ctx.embedder`` when the active dataset's embedder differs
   from the one currently on the detector.
2. ``POST /api/detectors/registry/load`` for an already-loaded detector
   detects the same mismatch and starts a re-embed task with a
   ``"Re-resolving labels for <embedder>…"`` progress message, instead of
   silently returning the synchronous fast-path.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _make_snap(embedder: str) -> dict[int, dict]:
    """Return a minimal ``snap`` with one media item using *embedder*."""
    rng = np.random.default_rng(0)
    return {
        1: {
            "id": 1,
            "type": "audio",
            "embedder": embedder,
            "embedding": rng.standard_normal(8).astype(np.float32),
        }
    }


def _seed_det_ctx_with_cache(embedder: str):
    """Return a fresh ``DetectorContext`` with a label-embedding cache stamped
    against *embedder*."""
    from vtscore.state.core import DetectorContext

    det_ctx = DetectorContext("d1", name="d1", media_type="audio", embedder=embedder)
    rng = np.random.default_rng(1)
    det_ctx.label_embeddings = {
        "eid_a": rng.standard_normal(8).astype(np.float32),
        "eid_b": rng.standard_normal(8).astype(np.float32),
    }
    return det_ctx


class TestPopulateLabelEmbeddingsCrossEmbedder:
    """Unit-level: the cache is dropped when the embedder changes."""

    def test_cache_cleared_on_embedder_change(self):
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_training import populate_label_embeddings

        det_ctx = _seed_det_ctx_with_cache("clap")
        # Empty labelset → no resolve+embed work, but the invalidation pass
        # must still fire. This isolates the cache-clearing logic from the
        # (slow, side-effectful) resolve+embed code path.
        empty_labelset = LabelSet.from_dict({"labels": []})

        populate_label_embeddings(
            det_ctx,
            empty_labelset,
            media_type="audio",
            snap=_make_snap("siglip"),
        )

        assert det_ctx.label_embeddings == {}
        # The new embedder is stamped after the pass so the next call doesn't
        # re-clear an already-aligned (empty) cache.
        assert det_ctx.embedder == "siglip"

    def test_cache_preserved_when_embedder_matches(self):
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_training import populate_label_embeddings

        det_ctx = _seed_det_ctx_with_cache("clap")
        before = dict(det_ctx.label_embeddings)
        empty_labelset = LabelSet.from_dict({"labels": []})

        populate_label_embeddings(
            det_ctx,
            empty_labelset,
            media_type="audio",
            snap=_make_snap("clap"),
        )

        assert set(det_ctx.label_embeddings.keys()) == set(before.keys())

    def test_cache_preserved_when_active_embedder_unknown(self):
        """No active dataset → ``embedder_name`` resolves to ``""``; we keep the
        cache rather than wiping it on insufficient information."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.labelset_training import populate_label_embeddings

        det_ctx = _seed_det_ctx_with_cache("clap")
        before = dict(det_ctx.label_embeddings)
        empty_labelset = LabelSet.from_dict({"labels": []})

        populate_label_embeddings(
            det_ctx,
            empty_labelset,
            media_type="audio",
            snap=None,
        )

        assert set(det_ctx.label_embeddings.keys()) == set(before.keys())


class TestLoadEndpointReembedTask:
    """End-to-end: ``POST /api/detectors/registry/load`` fires a re-embed task
    when the active dataset's embedder differs from the loaded detector's."""

    def _seed_loaded_detector(self, embedder: str = "clap"):
        """Create + register a detector with a small cached labelset and mark
        it loaded. Returns the detector id."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.registry import (
            add_loaded_detector_id,
            register_detector,
            reset_for_tests,
        )
        from vtscore.detectors.store import _detector_path, _write_detector
        from vtscore.state.core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
        )

        reset_for_tests()

        labelset_dict = {
            "labels": [
                {
                    "md5": "a1" * 16,
                    "label": "good",
                    "origin": {"importer": "absent", "params": {}},
                    "origin_name": "alpha.wav",
                    "filename": "alpha.wav",
                },
                {
                    "md5": "b2" * 16,
                    "label": "bad",
                    "origin": {"importer": "absent", "params": {}},
                    "origin_name": "beta.wav",
                    "filename": "beta.wav",
                },
            ]
        }
        name = "xemb-det"
        _write_detector(
            _detector_path(name),
            {
                "name": name,
                "text_query": "",
                "media_type": "audio",
                "examples": [],
                "labelset": labelset_dict,
            },
        )
        entry = register_detector(name=name, media_type="audio")
        detector_id = entry["id"]

        det_ctx = DetectorContext(detector_id, name=name, media_type="audio", embedder=embedder)
        # Pre-stamp a cached labelset so ``_maybe_start_label_reembed`` has
        # something to walk.
        det_ctx.cached_labelset = LabelSet.from_dict(labelset_dict)
        det_ctx.cached_labelset_media_type = "audio"
        register_detector_context(det_ctx)
        set_thread_detector_context(det_ctx)
        add_loaded_detector_id(detector_id)
        return detector_id, det_ctx

    def _set_active_dataset_embedder(self, embedder: str):
        """Replace every media item's ``embedder`` field so the active
        dataset reports the requested embedder."""
        from vtsearch.state import medias

        for cid in list(medias):
            medias[cid]["embedder"] = embedder

    def _drain_detector_tasks(self, timeout: float = 5.0) -> list[dict]:
        """Block until every detector-load task settles, then return them."""
        from vtscore.concurrency.progress import detector_loading_tasks

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            tasks = detector_loading_tasks.list_tasks()
            active = [t for t in tasks if t.get("status") != "idle"]
            if not active:
                return tasks
            time.sleep(0.05)
        return detector_loading_tasks.list_tasks()

    def test_same_embedder_returns_synchronous_fast_path(self, client):
        detector_id, _ = self._seed_loaded_detector(embedder="clap")
        self._set_active_dataset_embedder("clap")

        res = client.post(
            "/api/detectors/registry/load",
            json={"detector_id": detector_id},
        )
        body = res.get_json()
        assert res.status_code == 200
        # Synchronous fast-path: no task id in the response, no new task in
        # the tracker.
        assert "task_id" not in body
        assert body.get("ok") is True

    def test_different_embedder_starts_reembed_task(self, client):
        from vtscore.concurrency.progress import detector_loading_tasks

        detector_id, det_ctx = self._seed_loaded_detector(embedder="clap")
        self._set_active_dataset_embedder("clap-fused")

        # Sanity: snapshot the pre-call tracker.
        before = {t["task_id"] for t in detector_loading_tasks.list_tasks()}

        res = client.post(
            "/api/detectors/registry/load",
            json={"detector_id": detector_id},
        )
        body = res.get_json()
        assert res.status_code == 200
        assert body.get("ok") is True
        assert "task_id" in body, f"expected a re-embed task, got {body!r}"

        # The new task should be present in the tracker. Inspect the
        # ``embedder`` field (stable across the task's lifecycle) rather
        # than the live progress message (which can race with the
        # background thread completing).
        after = detector_loading_tasks.list_tasks()
        added = [t for t in after if t["task_id"] not in before]
        assert added, "re-embed task missing from the tracker"
        task = added[0]
        assert task["detector_id"] == detector_id
        assert task.get("embedder") == "clap-fused"

        # Let the background thread finish so test cleanup is clean.
        self._drain_detector_tasks()


class TestDetectorRegistryEmbedderField:
    """``GET /api/detectors/registry`` exposes the loaded detector's embedder."""

    def test_loaded_entry_exposes_embedder(self, client):
        from vtscore.detectors.registry import (
            add_loaded_detector_id,
            register_detector,
            reset_for_tests,
        )
        from vtscore.state.core import (
            DetectorContext,
            register_detector_context,
        )

        reset_for_tests()
        entry = register_detector(name="emb-probe", media_type="audio")
        detector_id = entry["id"]
        det_ctx = DetectorContext(detector_id, name="emb-probe", media_type="audio", embedder="clap")
        register_detector_context(det_ctx)
        add_loaded_detector_id(detector_id)

        res = client.get("/api/detectors/registry")
        assert res.status_code == 200
        rows = {row["id"]: row for row in res.get_json()["detectors"]}
        assert rows[detector_id]["embedder"] == "clap"

    def test_unloaded_entry_has_empty_embedder(self, client):
        from vtscore.detectors.registry import register_detector, reset_for_tests

        reset_for_tests()
        entry = register_detector(name="emb-unloaded", media_type="audio")
        detector_id = entry["id"]

        res = client.get("/api/detectors/registry")
        assert res.status_code == 200
        rows = {row["id"]: row for row in res.get_json()["detectors"]}
        assert rows[detector_id]["embedder"] == ""

"""Tests for multiple loaded detectors: DetectorContext lifecycle, vote isolation,
model registry multi-loaded support, activate/unload endpoints, and MLP caching."""

import hashlib
import shutil

import numpy as np
import pytest

from tests import load_detector_and_wait as _load_detector_and_wait
from vtsearch.settings import get_detectors_dir


@pytest.fixture(autouse=True)
def clean_detectors_dir():
    """Remove the detectors directory before and after each test."""
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_detectors_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


def _make_test_media(media_id: int, media_type: str = "audio") -> dict:
    rng = np.random.default_rng(media_id)
    return {
        "id": media_id,
        "type": media_type,
        "embedder": "clap",
        "duration": 1.0,
        "file_size": 100,
        "md5": hashlib.md5(f"media_{media_id}".encode()).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "media_bytes": b"fake",
        "filename": f"media_{media_id}.wav",
        "category": "test",
        "origin": {"importer": "test", "params": {}},
        "origin_name": f"media_{media_id}.wav",
    }


# ---------------------------------------------------------------------------
# DetectorContext store tests
# ---------------------------------------------------------------------------


class TestDetectorContextStore:
    """Register, unregister, and list detector contexts."""

    def test_register_and_get(self):
        from vtsearch.utils.state_core import (
            DetectorContext,
            get_detector_context,
            register_detector_context,
        )

        ctx = DetectorContext("det_reg")
        register_detector_context(ctx)
        assert get_detector_context("det_reg") is ctx

    def test_list_loaded(self):
        from vtsearch.utils.state_core import list_loaded_detector_ids

        ids = list_loaded_detector_ids()
        assert "_test_default_det" in ids

    def test_unregister_removes_context(self):
        from vtsearch.utils.state_core import (
            DetectorContext,
            get_detector_context,
            register_detector_context,
            unregister_detector_context,
        )

        ctx = DetectorContext("det_unreg")
        register_detector_context(ctx)
        removed = unregister_detector_context("det_unreg")
        assert removed is ctx
        assert get_detector_context("det_unreg") is None

    def test_unregister_clears_active_if_match(self):
        from vtsearch.utils.state_core import (
            DetectorContext,
            get_thread_detector_context,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )

        ctx = DetectorContext("det_active_unreg")
        register_detector_context(ctx)
        set_thread_detector_context(ctx)
        unregister_detector_context("det_active_unreg")
        assert get_thread_detector_context() is None

    def test_clear_all_detector_contexts(self):
        from vtsearch.utils.state_core import (
            DetectorContext,
            clear_all_detector_contexts,
            list_loaded_detector_ids,
            register_detector_context,
        )

        register_detector_context(DetectorContext("d1"))
        register_detector_context(DetectorContext("d2"))
        clear_all_detector_contexts()
        assert list_loaded_detector_ids() == []

    def test_register_clears_progress_cache(self):
        """Registering a new detector must clear the progress cache so stale
        training indicators from a previous detector don't carry over."""
        from vtsearch.models.progress import _cached_steps, clear_progress_cache
        from vtsearch.utils.state_core import DetectorContext, register_detector_context

        # Seed the cache with a fake entry (simulating a previous detector's training)
        clear_progress_cache()
        _cached_steps.append({"model": None, "threshold": 0.5, "good_ids": set(), "bad_ids": set(), "stability": None})
        assert len(_cached_steps) == 1

        # Registering a new detector should clear the stale cache
        register_detector_context(DetectorContext("det_progress_clear"))
        assert len(_cached_steps) == 0

    def test_unregister_clears_progress_cache(self):
        """Unregistering a detector must clear the progress cache so stale
        training indicators don't leak to the next detector."""
        from vtsearch.models.progress import _cached_steps
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            unregister_detector_context,
        )

        ctx = DetectorContext("det_progress_unreg")
        register_detector_context(ctx)
        # clear after register (since register also clears)
        _cached_steps.append({"model": None, "threshold": 0.5, "good_ids": set(), "bad_ids": set(), "stability": None})
        assert len(_cached_steps) == 1

        unregister_detector_context("det_progress_unreg")
        assert len(_cached_steps) == 0


# ---------------------------------------------------------------------------
# Vote isolation across detectors
# ---------------------------------------------------------------------------


class TestVoteIsolation:
    """Votes are per-detector, not per-dataset."""

    def test_votes_isolated_between_detectors(self):
        from vtsearch.utils import (
            bad_votes,
            good_votes,
        )
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
        )

        det_a = DetectorContext("iso_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        good_votes[1] = None
        good_votes[2] = None

        det_b = DetectorContext("iso_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        bad_votes[3] = None

        # B sees only its own votes
        assert len(good_votes) == 0
        assert 3 in bad_votes

        # A sees only its own votes
        set_thread_detector_context(det_a)
        assert len(good_votes) == 2
        assert len(bad_votes) == 0

    def test_toggle_vote_goes_to_active_detector(self):
        from vtsearch.utils import good_votes, toggle_vote
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
        )

        det = DetectorContext("toggle_det")
        register_detector_context(det)
        set_thread_detector_context(det)

        toggle_vote(1, "good")
        assert 1 in good_votes
        assert 1 in det.good_votes

    def test_clear_votes_only_clears_active_detector(self):
        from vtsearch.utils import clear_votes, good_votes
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
        )

        det_a = DetectorContext("clear_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        good_votes[1] = None

        det_b = DetectorContext("clear_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        good_votes[2] = None
        clear_votes()
        assert len(good_votes) == 0

        # A's votes are untouched
        set_thread_detector_context(det_a)
        assert 1 in good_votes


# ---------------------------------------------------------------------------
# Model registry multi-loaded
# ---------------------------------------------------------------------------


class TestModelRegistryMultiLoaded:
    """Model registry supports multiple loaded models."""

    def test_multiple_models_loaded(self):
        from vtsearch.models.detector_registry import (
            add_loaded_detector_id,
            get_loaded_detector_ids,
            is_detector_loaded,
        )

        add_loaded_detector_id("m1")
        add_loaded_detector_id("m2")
        assert is_detector_loaded("m1")
        assert is_detector_loaded("m2")
        assert len(get_loaded_detector_ids()) == 2

    def test_remove_loaded_removes_from_set(self):
        from vtsearch.models.detector_registry import (
            add_loaded_detector_id,
            is_detector_loaded,
            remove_loaded_detector_id,
        )

        add_loaded_detector_id("m1")
        assert is_detector_loaded("m1")
        remove_loaded_detector_id("m1")
        assert not is_detector_loaded("m1")

    def test_add_loaded_model_id_adds_to_loaded_set(self):
        from vtsearch.models.detector_registry import (
            add_loaded_detector_id,
            is_detector_loaded,
        )

        add_loaded_detector_id("m1")
        assert is_detector_loaded("m1")


# ---------------------------------------------------------------------------
# API endpoints: load, activate, unload
# ---------------------------------------------------------------------------


class TestModelLoadEndpoints:
    """Test the multi-loaded model API endpoints."""

    def _register_trainable_model(self, client, name):
        """Helper: create detector + register in model registry."""
        client.post(
            "/api/detectors",
            json={"name": name, "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": name,
                "media_type": "audio",
                "trainable": True,
                "text_query": "test",
            },
        )
        return res.get_json()["detector"]["id"]

    def test_load_creates_detector_context(self, client):
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "LoadCtx")
        res = _load_detector_and_wait(client, mid)
        assert res.status_code == 200

        det = get_detector_context(mid)
        assert det is not None
        assert det.detector_id == mid

    def test_load_twice_reuses_context(self, client):
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "LoadTwice")
        _load_detector_and_wait(client, mid)
        det1 = get_detector_context(mid)

        # Load again — should reuse, not create new
        _load_detector_and_wait(client, mid)
        det2 = get_detector_context(mid)
        assert det1 is det2

    def test_registry_shows_loaded(self, client):
        mid1 = self._register_trainable_model(client, "Reg1")
        mid2 = self._register_trainable_model(client, "Reg2")

        _load_detector_and_wait(client, mid1)
        _load_detector_and_wait(client, mid2)

        res = client.get("/api/detectors/registry")
        models = {m["id"]: m for m in res.get_json()["detectors"]}

        assert models[mid1]["loaded"] is True
        assert models[mid2]["loaded"] is True

    def test_unload_removes_context(self, client):
        from vtsearch.models.detector_registry import is_detector_loaded
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "Unload")
        _load_detector_and_wait(client, mid)
        assert get_detector_context(mid) is not None

        res = client.post(f"/api/detectors/registry/{mid}/unload")
        assert res.status_code == 200
        assert get_detector_context(mid) is None
        assert not is_detector_loaded(mid)

    def test_unload_not_loaded_returns_400(self, client):
        mid = self._register_trainable_model(client, "UnloadNot")
        res = client.post(f"/api/detectors/registry/{mid}/unload")
        assert res.status_code == 400

    def test_delete_cleans_up_context(self, client):
        from vtsearch.models.detector_registry import is_detector_loaded
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "Delete")
        _load_detector_and_wait(client, mid)
        assert get_detector_context(mid) is not None

        res = client.delete(f"/api/detectors/registry/{mid}")
        assert res.status_code == 200
        assert get_detector_context(mid) is None
        assert not is_detector_loaded(mid)


# ---------------------------------------------------------------------------
# Model loading tasks endpoints
# ---------------------------------------------------------------------------


class TestModelLoadingTasks:
    """Test the model loading tasks progress API."""

    def test_loading_tasks_empty(self, client):
        res = client.get("/api/detectors/loading-tasks")
        assert res.status_code == 200
        assert res.get_json()["tasks"] == []

    def test_cancel_nonexistent_task(self, client):
        res = client.post("/api/detectors/cancel/nonexistent")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# MLP caching in DetectorContext
# ---------------------------------------------------------------------------


class TestMLPCaching:
    """learned_sort caches the MLP in the active DetectorContext."""

    def test_learned_sort_caches_model(self, client):
        from vtsearch.utils import good_votes, bad_votes
        from vtsearch.utils.state_core import get_active_detector_context, _empty_detector_context

        # Vote to enable training
        good_votes[1] = None
        good_votes[2] = None
        good_votes[3] = None
        bad_votes[18] = None
        bad_votes[19] = None
        bad_votes[20] = None

        res = client.post("/api/learned-sort", json={"wait": True})
        assert res.status_code == 200

        det_ctx = get_active_detector_context()
        assert det_ctx is not _empty_detector_context
        assert det_ctx.model is not None
        assert det_ctx.threshold != 0.5  # should have been calibrated
        assert len(det_ctx.training_medias) == 6  # 3 good + 3 bad

    def test_cached_model_updates_on_retrain(self, client):
        from vtsearch.utils import good_votes, bad_votes
        from vtsearch.utils.state_core import get_active_detector_context

        good_votes[1] = None
        good_votes[2] = None
        good_votes[3] = None
        bad_votes[18] = None
        bad_votes[19] = None
        bad_votes[20] = None

        client.post("/api/learned-sort", json={"wait": True})
        det_ctx = get_active_detector_context()
        model1 = det_ctx.model

        # Add more votes and retrain
        good_votes[4] = None
        client.post("/api/learned-sort", json={"wait": True})

        # Model should have been updated
        model2 = det_ctx.model
        assert model2 is not model1
        assert len(det_ctx.training_medias) == 7  # 4 good + 3 bad


# ---------------------------------------------------------------------------
# Labeling-status indicator reset on detector switch
# ---------------------------------------------------------------------------


class TestLabelingStatusResetOnDetectorSwitch:
    """Regression: deleting a detector and loading a new one must not
    inherit stale training indicators from the old detector.

    The progress cache is module-level, not per-detector.  Without
    clearing it on detector switch, _ensure_cache sees
    len(_cached_steps) >= len(new_label_history) and returns stale
    all-green indicators, causing autopilot to skip to 'Done'.
    """

    def test_labeling_status_not_green_after_detector_switch(self, client):
        """After building up cached progress on detector A, switching to
        a fresh detector B must NOT return green smart/stable status."""
        from vtsearch.models.progress import _cached_steps, _progress_lock
        from vtsearch.utils import good_votes, bad_votes, label_history
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            set_thread_detector_context,
            unregister_detector_context,
        )

        # --- Set up detector A with votes and build progress cache ---
        det_a = DetectorContext("det_a_status", name="Detector A", media_type="audio")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)

        # Add votes through apply_label so label_history is also populated
        from vtsearch.utils import apply_label

        for mid in [1, 2, 3, 4, 5]:
            apply_label(mid, "good")
        for mid in [16, 17, 18, 19, 20]:
            apply_label(mid, "bad")

        # Call labeling-status to populate the progress cache
        res = client.get("/api/labeling-status")
        assert res.status_code == 200

        with _progress_lock:
            cached_before = len(_cached_steps)
        assert cached_before > 0, "Progress cache should be populated after labeling-status call"

        # --- Delete detector A and create detector B ---
        unregister_detector_context("det_a_status")

        with _progress_lock:
            cached_after_unreg = len(_cached_steps)
        assert cached_after_unreg == 0, "Progress cache must be cleared on unregister"

        det_b = DetectorContext("det_b_status", name="Detector B", media_type="audio")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)

        # Detector B has NO votes and NO label history
        assert len(good_votes) == 0
        assert len(bad_votes) == 0
        assert len(label_history) == 0

        # --- Check labeling-status ---
        res = client.get("/api/labeling-status")
        assert res.status_code == 200
        data = res.get_json()

        # With 0 good and 0 bad votes the indicators must NOT be green
        assert data["smart"]["status"] == "red", f"smart should be red with 0 votes, got {data['smart']['status']}"
        assert data["stable"]["status"] == "red", f"stable should be red with 0 votes, got {data['stable']['status']}"

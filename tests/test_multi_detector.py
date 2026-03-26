"""Tests for multiple loaded detectors: DetectorContext lifecycle, vote isolation,
model registry multi-loaded support, activate/unload endpoints, and MLP caching."""

import hashlib
import shutil

import numpy as np
import pytest

from tests import load_model_and_wait as _load_model_and_wait
from vtsearch.settings import get_trainable_models_dir


@pytest.fixture(autouse=True)
def clean_trainable_models_dir():
    """Remove the trainable models directory before and after each test."""
    tm_dir = get_trainable_models_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_trainable_models_dir()
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
            get_active_detector_id,
            register_detector_context,
            set_active_detector_id,
            unregister_detector_context,
        )

        ctx = DetectorContext("det_active_unreg")
        register_detector_context(ctx)
        set_active_detector_id("det_active_unreg")
        unregister_detector_context("det_active_unreg")
        assert get_active_detector_id() is None

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
            set_active_detector_id,
        )

        det_a = DetectorContext("iso_a")
        register_detector_context(det_a)
        set_active_detector_id("iso_a")
        good_votes[1] = None
        good_votes[2] = None

        det_b = DetectorContext("iso_b")
        register_detector_context(det_b)
        set_active_detector_id("iso_b")
        bad_votes[3] = None

        # B sees only its own votes
        assert len(good_votes) == 0
        assert 3 in bad_votes

        # A sees only its own votes
        set_active_detector_id("iso_a")
        assert len(good_votes) == 2
        assert len(bad_votes) == 0

    def test_toggle_vote_goes_to_active_detector(self):
        from vtsearch.utils import good_votes, toggle_vote
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            set_active_detector_id,
        )

        det = DetectorContext("toggle_det")
        register_detector_context(det)
        set_active_detector_id("toggle_det")

        toggle_vote(1, "good")
        assert 1 in good_votes
        assert 1 in det.good_votes

    def test_clear_votes_only_clears_active_detector(self):
        from vtsearch.utils import clear_votes, good_votes
        from vtsearch.utils.state_core import (
            DetectorContext,
            register_detector_context,
            set_active_detector_id,
        )

        det_a = DetectorContext("clear_a")
        register_detector_context(det_a)
        set_active_detector_id("clear_a")
        good_votes[1] = None

        det_b = DetectorContext("clear_b")
        register_detector_context(det_b)
        set_active_detector_id("clear_b")
        good_votes[2] = None
        clear_votes()
        assert len(good_votes) == 0

        # A's votes are untouched
        set_active_detector_id("clear_a")
        assert 1 in good_votes


# ---------------------------------------------------------------------------
# Model registry multi-loaded
# ---------------------------------------------------------------------------


class TestModelRegistryMultiLoaded:
    """Model registry supports multiple loaded models."""

    def test_multiple_models_loaded(self):
        from vtsearch.models.registry import (
            add_loaded_model_id,
            get_loaded_model_ids,
            is_model_loaded,
        )

        add_loaded_model_id("m1")
        add_loaded_model_id("m2")
        assert is_model_loaded("m1")
        assert is_model_loaded("m2")
        assert len(get_loaded_model_ids()) == 2

    def test_remove_loaded_removes_from_set(self):
        from vtsearch.models.registry import (
            add_loaded_model_id,
            is_model_loaded,
            remove_loaded_model_id,
        )

        add_loaded_model_id("m1")
        assert is_model_loaded("m1")
        remove_loaded_model_id("m1")
        assert not is_model_loaded("m1")

    def test_add_loaded_model_id_adds_to_loaded_set(self):
        from vtsearch.models.registry import (
            add_loaded_model_id,
            is_model_loaded,
        )

        add_loaded_model_id("m1")
        assert is_model_loaded("m1")


# ---------------------------------------------------------------------------
# API endpoints: load, activate, unload
# ---------------------------------------------------------------------------


class TestModelLoadEndpoints:
    """Test the multi-loaded model API endpoints."""

    def _register_trainable_model(self, client, name):
        """Helper: create trainable model + register in model registry."""
        client.post(
            "/api/trainable-models",
            json={"name": name, "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={
                "name": name,
                "media_type": "audio",
                "trainable": True,
                "text_query": "test",
            },
        )
        return res.get_json()["model"]["id"]

    def test_load_creates_detector_context(self, client):
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "LoadCtx")
        res = _load_model_and_wait(client, mid)
        assert res.status_code == 200

        det = get_detector_context(mid)
        assert det is not None
        assert det.detector_id == mid

    def test_load_twice_reuses_context(self, client):
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "LoadTwice")
        _load_model_and_wait(client, mid)
        det1 = get_detector_context(mid)

        # Load again — should reuse, not create new
        _load_model_and_wait(client, mid)
        det2 = get_detector_context(mid)
        assert det1 is det2

    def test_registry_shows_loaded(self, client):
        mid1 = self._register_trainable_model(client, "Reg1")
        mid2 = self._register_trainable_model(client, "Reg2")

        _load_model_and_wait(client, mid1)
        _load_model_and_wait(client, mid2)

        res = client.get("/api/models/registry")
        models = {m["id"]: m for m in res.get_json()["models"]}

        assert models[mid1]["loaded"] is True
        assert models[mid2]["loaded"] is True

    def test_unload_removes_context(self, client):
        from vtsearch.models.registry import is_model_loaded
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "Unload")
        _load_model_and_wait(client, mid)
        assert get_detector_context(mid) is not None

        res = client.post(f"/api/models/registry/{mid}/unload")
        assert res.status_code == 200
        assert get_detector_context(mid) is None
        assert not is_model_loaded(mid)

    def test_unload_not_loaded_returns_400(self, client):
        mid = self._register_trainable_model(client, "UnloadNot")
        res = client.post(f"/api/models/registry/{mid}/unload")
        assert res.status_code == 400

    def test_delete_cleans_up_context(self, client):
        from vtsearch.models.registry import is_model_loaded
        from vtsearch.utils.state_core import get_detector_context

        mid = self._register_trainable_model(client, "Delete")
        _load_model_and_wait(client, mid)
        assert get_detector_context(mid) is not None

        res = client.delete(f"/api/models/registry/{mid}")
        assert res.status_code == 200
        assert get_detector_context(mid) is None
        assert not is_model_loaded(mid)


# ---------------------------------------------------------------------------
# Model loading tasks endpoints
# ---------------------------------------------------------------------------


class TestModelLoadingTasks:
    """Test the model loading tasks progress API."""

    def test_loading_tasks_empty(self, client):
        res = client.get("/api/models/loading-tasks")
        assert res.status_code == 200
        assert res.get_json()["tasks"] == []

    def test_cancel_nonexistent_task(self, client):
        res = client.post("/api/models/cancel/nonexistent")
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

        res = client.post("/api/learned-sort")
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

        client.post("/api/learned-sort")
        det_ctx = get_active_detector_context()
        model1 = det_ctx.model

        # Add more votes and retrain
        good_votes[4] = None
        client.post("/api/learned-sort")

        # Model should have been updated
        model2 = det_ctx.model
        assert model2 is not model1
        assert len(det_ctx.training_medias) == 7  # 4 good + 3 bad

"""Tests for multi-dataset support: loading, activating, switching, and unloading
multiple datasets without re-resolving media or re-embedding vectors."""

import hashlib

import numpy as np

from tests import load_detector_and_wait as _load_detector_and_wait

from vtsearch.shim.state_proxies import _ProxyDict
from vtscore.state.core import (
    DatasetContext,
    DetectorContext,
    get_context,
    get_thread_dataset_context,
    list_loaded_dataset_ids,
    register_context,
    register_detector_context,
    set_thread_dataset_context,
    set_thread_detector_context,
    unregister_context,
)


# ---------------------------------------------------------------------------
# Unit tests: DatasetContext and proxy plumbing
# ---------------------------------------------------------------------------


class TestDatasetContext:
    """DatasetContext creation and field defaults."""

    def test_fresh_context_has_empty_containers(self):
        ctx = DatasetContext("ds1")
        assert ctx.dataset_id == "ds1"
        assert ctx.medias == {}
        assert ctx.diversity_tree is None
        assert ctx.dataset_display_name is None

    def test_contexts_are_independent(self):
        ctx_a = DatasetContext("a")
        ctx_b = DatasetContext("b")
        ctx_a.medias[1] = {"id": 1, "type": "audio"}
        ctx_b.medias[2] = {"id": 2, "type": "image"}
        assert 1 in ctx_a.medias and 2 not in ctx_a.medias
        assert 2 in ctx_b.medias and 1 not in ctx_b.medias


class TestDetectorContext:
    """DetectorContext creation and field defaults."""

    def test_fresh_context_has_empty_containers(self):
        ctx = DetectorContext("det1")
        assert ctx.detector_id == "det1"
        assert ctx.good_votes == {}
        assert ctx.bad_votes == {}
        assert ctx.label_history == []
        assert ctx.vote_click_times == {}
        assert ctx.click_counter == 0
        assert ctx.last_learned_scores == {}
        assert ctx.textsort_suggestions == []
        assert ctx.find_initial_labels == {}
        assert ctx.inclusion is None
        assert ctx.training_medias == {}
        assert ctx.model is None
        assert ctx.threshold == 0.5

    def test_contexts_are_independent(self):
        ctx_a = DetectorContext("da")
        ctx_b = DetectorContext("db")
        ctx_a.good_votes[1] = None
        ctx_b.good_votes[2] = None
        assert 1 in ctx_a.good_votes and 2 not in ctx_a.good_votes
        assert 2 in ctx_b.good_votes and 1 not in ctx_b.good_votes


class TestContextStore:
    """Context registration, lookup, and removal."""

    def test_register_and_get(self):
        ctx = DatasetContext("ds_reg")
        register_context(ctx)
        assert get_context("ds_reg") is ctx

    def test_list_loaded(self):
        ids = list_loaded_dataset_ids()
        # reset_state creates "_test_default"
        assert "_test_default" in ids

    def test_unregister_removes_context(self):
        ctx = DatasetContext("ds_unreg")
        register_context(ctx)
        assert get_context("ds_unreg") is ctx
        removed = unregister_context("ds_unreg")
        assert removed is ctx
        assert get_context("ds_unreg") is None

    def test_unregister_nonexistent_returns_none(self):
        assert unregister_context("nonexistent") is None

    def test_unregister_clears_active_if_match(self):
        ctx = DatasetContext("ds_active_unreg")
        register_context(ctx)
        set_thread_dataset_context(ctx)
        unregister_context("ds_active_unreg")
        assert get_thread_dataset_context() is None


class TestProxyDict:
    """_ProxyDict delegates to the active context."""

    def test_proxy_dict_reflects_active_context(self):
        from vtsearch.state import medias

        # The reset_state fixture creates _test_default with test medias.
        # Create a second context and switch.
        ctx2 = DatasetContext("proxy_test")
        ctx2.medias[999] = {"id": 999, "type": "test"}
        register_context(ctx2)
        set_thread_dataset_context(ctx2)

        assert 999 in medias
        assert len(medias) == 1
        assert list(medias.keys()) == [999]

        # Switch back
        set_thread_dataset_context(get_context("_test_default"))
        assert 999 not in medias
        assert len(medias) > 0  # test medias

    def test_proxy_dict_clear(self):
        from vtsearch.state import good_votes

        good_votes[1] = None
        assert 1 in good_votes
        good_votes.clear()
        assert len(good_votes) == 0

    def test_proxy_dict_pop(self):
        from vtsearch.state import bad_votes

        bad_votes[5] = None
        assert bad_votes.pop(5, "missing") is None
        assert bad_votes.pop(5, "missing") == "missing"

    def test_proxy_dict_update(self):
        from vtsearch.state import last_learned_scores

        last_learned_scores.update({1: 0.5, 2: 0.8})
        assert last_learned_scores[1] == 0.5
        assert last_learned_scores[2] == 0.8

    def test_proxy_dict_copy(self):
        from vtsearch.state import good_votes

        good_votes[10] = None
        c = good_votes.copy()
        assert isinstance(c, dict)
        assert not isinstance(c, _ProxyDict)
        assert 10 in c

    def test_proxy_dict_items_values_keys(self):
        from vtsearch.state import good_votes

        good_votes[1] = None
        good_votes[2] = None
        assert set(good_votes.keys()) == {1, 2}
        assert list(good_votes.values()) == [None, None]
        assert set(dict(good_votes.items()).keys()) == {1, 2}

    def test_proxy_dict_is_dict_instance(self):
        from vtsearch.state import medias

        assert isinstance(medias, dict)

    def test_proxy_dict_bool(self):
        from vtsearch.state import last_learned_scores

        last_learned_scores.clear()
        assert not last_learned_scores
        last_learned_scores[1] = 0.5
        assert last_learned_scores


class TestProxyList:
    """_ProxyList delegates to the active context."""

    def test_proxy_list_append_and_iter(self):
        from vtsearch.state import textsort_suggestions

        textsort_suggestions.clear()
        textsort_suggestions.append("hello")
        textsort_suggestions.append("world")
        assert list(textsort_suggestions) == ["hello", "world"]

    def test_proxy_list_clear(self):
        from vtsearch.state import label_history

        label_history.append((1, "good", 0.0))
        assert len(label_history) > 0
        label_history.clear()
        assert len(label_history) == 0

    def test_proxy_list_is_list_instance(self):
        from vtsearch.state import label_history

        assert isinstance(label_history, list)

    def test_proxy_list_remove(self):
        from vtsearch.state import textsort_suggestions

        textsort_suggestions.clear()
        textsort_suggestions.append("a")
        textsort_suggestions.append("b")
        textsort_suggestions.remove("a")
        assert list(textsort_suggestions) == ["b"]


# ---------------------------------------------------------------------------
# Integration tests: switching datasets preserves state
# ---------------------------------------------------------------------------


class TestMultiDatasetSwitching:
    """Verify that switching active datasets preserves medias."""

    def _make_test_media(self, media_id: int, media_type: str = "audio") -> dict:
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

    def test_switch_preserves_medias(self):
        from vtsearch.state import (
            medias,
            register_context,
            set_thread_dataset_context,
        )

        # Setup context A with medias
        ctx_a = DatasetContext("switch_a")
        register_context(ctx_a)
        set_thread_dataset_context(ctx_a)
        medias[1] = self._make_test_media(1)

        # Setup context B with different medias
        ctx_b = DatasetContext("switch_b")
        register_context(ctx_b)
        set_thread_dataset_context(ctx_b)
        medias[2] = self._make_test_media(2)

        # Verify B state
        assert 2 in medias
        assert 1 not in medias

        # Switch back to A — medias preserved
        set_thread_dataset_context(ctx_a)
        assert 1 in medias
        assert 2 not in medias

    def test_switch_detectors_preserves_votes(self):
        """Votes are per-detector: switching detectors preserves each one's votes."""
        from vtsearch.state import (
            good_votes,
            bad_votes,
            register_detector_context,
            set_thread_detector_context,
        )

        det_a = DetectorContext("det_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        good_votes[1] = None

        det_b = DetectorContext("det_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        bad_votes[2] = None

        # Verify B state
        assert 2 in bad_votes
        assert len(good_votes) == 0

        # Switch back to A — votes preserved
        set_thread_detector_context(det_a)
        assert 1 in good_votes
        assert len(bad_votes) == 0

        # Switch to B again — still intact
        set_thread_detector_context(det_b)
        assert 2 in bad_votes

    def test_switch_detectors_preserves_label_history(self):
        from vtsearch.state import (
            label_history,
            register_detector_context,
            set_thread_detector_context,
            toggle_vote,
        )

        det_a = DetectorContext("hist_det_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        toggle_vote(1, "good")

        det_b = DetectorContext("hist_det_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        toggle_vote(2, "bad")

        # B's history
        assert len(label_history) == 1
        assert label_history[0][0] == 2

        # Switch to A — A's history
        set_thread_detector_context(det_a)
        assert len(label_history) == 1
        assert label_history[0][0] == 1

    def test_switch_detectors_preserves_learned_scores(self):
        from vtsearch.state import (
            get_learned_scores,
            register_detector_context,
            set_thread_detector_context,
            update_learned_scores,
        )

        det_a = DetectorContext("scores_det_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        update_learned_scores({1: 0.9, 2: 0.1})

        det_b = DetectorContext("scores_det_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        update_learned_scores({3: 0.5})

        assert get_learned_scores() == {3: 0.5}

        set_thread_detector_context(det_a)
        assert get_learned_scores() == {1: 0.9, 2: 0.1}

    def test_unload_frees_context(self):
        from vtsearch.state import (
            medias,
            register_context,
            set_thread_dataset_context,
            unregister_context,
        )

        ctx = DatasetContext("unload_me")
        ctx.medias[100] = self._make_test_media(100)
        register_context(ctx)
        set_thread_dataset_context(ctx)
        assert 100 in medias

        unregister_context("unload_me")
        assert get_context("unload_me") is None
        assert get_thread_dataset_context() is None
        # After unload, medias delegates to empty fallback
        assert len(medias) == 0


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestMultiDatasetAPI:
    """Test the API endpoints for multi-dataset operations."""

    def _make_test_media(self, media_id: int) -> dict:
        rng = np.random.default_rng(media_id)
        return {
            "id": media_id,
            "type": "audio",
            "embedder": "clap",
            "duration": 1.0,
            "file_size": 100,
            "md5": hashlib.md5(f"api_media_{media_id}".encode()).hexdigest(),
            "embedding": rng.standard_normal(512).astype(np.float32),
            "media_bytes": b"fake",
            "filename": f"api_media_{media_id}.wav",
            "category": "test",
            "origin": {"importer": "test", "params": {}},
            "origin_name": f"api_media_{media_id}.wav",
        }

    def test_registry_shows_loaded(self, client):
        """Verify the registry endpoint shows loaded flag."""
        from vtscore.datasets.registry import register_dataset, add_loaded_id

        # Register two datasets in the registry.
        e1 = register_dataset(name="DS1", media_type="audio", num_items=5, pkl_path="/tmp/fake1.pkl")
        e2 = register_dataset(name="DS2", media_type="audio", num_items=3, pkl_path="/tmp/fake2.pkl")

        # Mark both as loaded.
        add_loaded_id(e1["id"])
        add_loaded_id(e2["id"])

        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        datasets = {d["id"]: d for d in data["datasets"]}

        assert datasets[e1["id"]]["loaded"] is True
        assert datasets[e2["id"]]["loaded"] is True

    def test_unload_removes_from_loaded(self, client):
        """POST /api/datasets/registry/<id>/unload removes the context."""
        from vtscore.datasets.registry import register_dataset, add_loaded_id, is_loaded

        e = register_dataset(name="Unload", media_type="audio", num_items=2, pkl_path="/tmp/fake_unload.pkl")
        ctx = DatasetContext(e["id"])
        ctx.medias[1] = self._make_test_media(1)
        register_context(ctx)
        add_loaded_id(e["id"])
        set_thread_dataset_context(ctx)

        resp = client.post(f"/api/datasets/registry/{e['id']}/unload")
        assert resp.status_code == 200
        assert not is_loaded(e["id"])
        assert get_context(e["id"]) is None

    def test_unload_not_loaded_returns_400(self, client):
        """Cannot unload a dataset that's not loaded."""
        from vtscore.datasets.registry import register_dataset

        e = register_dataset(name="X", media_type="audio", num_items=1, pkl_path="/tmp/fake_x.pkl")
        resp = client.post(f"/api/datasets/registry/{e['id']}/unload")
        assert resp.status_code == 400

    def test_preload_embedder_returns_resolved_name(self, client):
        """POST /api/datasets/registry/<id>/preload-embedder reports the
        embedder it warmed in the background."""
        from vtscore.datasets.registry import register_dataset

        e = register_dataset(name="Pre", media_type="audio", num_items=1, pkl_path="/tmp/fake_pre.pkl")
        resp = client.post(f"/api/datasets/registry/{e['id']}/preload-embedder")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        # The default audio embedder is registered in tests as "clap".
        assert data["embedder"] == "clap"

    def test_preload_embedder_unknown_dataset_returns_404(self, client):
        """Unknown dataset id is rejected with 404."""
        resp = client.post("/api/datasets/registry/does-not-exist/preload-embedder")
        assert resp.status_code == 404

    def test_preload_embedder_respects_entry_embedder_field(self, client):
        """When a dataset entry pins an embedder, that name is preferred
        over the media-type default."""
        from vtscore.datasets.registry import register_dataset

        e = register_dataset(
            name="Pinned",
            media_type="audio",
            num_items=1,
            pkl_path="/tmp/fake_pinned.pkl",
            embedder="clap",
        )
        resp = client.post(f"/api/datasets/registry/{e['id']}/preload-embedder")
        assert resp.status_code == 200
        assert resp.get_json()["embedder"] == "clap"

    def test_load_already_loaded_returns_instantly(self, client):
        """Loading an already-loaded dataset returns immediately."""
        from vtscore.datasets.registry import register_dataset, add_loaded_id

        e = register_dataset(name="Already", media_type="audio", num_items=2, pkl_path="/tmp/fake_already.pkl")
        ctx = DatasetContext(e["id"])
        ctx.medias[1] = self._make_test_media(1)
        register_context(ctx)
        add_loaded_id(e["id"])

        resp = client.post(f"/api/datasets/registry/{e['id']}/load")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "already loaded" in data["message"].lower()

    def test_dataset_status_reflects_active(self, client):
        """GET /api/dataset/status shows the active dataset's medias."""
        from vtsearch.state import medias

        # Default context has test medias from conftest.
        resp = client.get("/api/dataset/status")
        data = resp.get_json()
        assert data["loaded"] is True
        assert data["num_medias"] == len(medias)

    def test_clear_only_clears_active(self, client):
        """POST /api/dataset/clear removes only the active dataset."""
        from vtscore.datasets.registry import register_dataset, add_loaded_id, is_loaded

        e1 = register_dataset(name="Keep", media_type="audio", num_items=2, pkl_path="/tmp/fake_keep.pkl")
        e2 = register_dataset(name="Clear", media_type="audio", num_items=2, pkl_path="/tmp/fake_clear.pkl")

        ctx1 = DatasetContext(e1["id"])
        ctx1.medias[1] = self._make_test_media(1)
        register_context(ctx1)
        add_loaded_id(e1["id"])

        ctx2 = DatasetContext(e2["id"])
        ctx2.medias[2] = self._make_test_media(2)
        register_context(ctx2)
        add_loaded_id(e2["id"])

        set_thread_dataset_context(ctx2)

        resp = client.post("/api/dataset/clear")
        assert resp.status_code == 200

        # e2 should be gone, e1 should still be loaded
        assert not is_loaded(e2["id"])
        assert get_context(e2["id"]) is None
        assert is_loaded(e1["id"])
        assert get_context(e1["id"]) is ctx1


# ---------------------------------------------------------------------------
# Scalar state tests
# ---------------------------------------------------------------------------


class TestScalarContextState:
    """Per-context scalar fields switch correctly."""

    def test_click_counter_per_detector(self):
        """click_counter is per-detector (vote state)."""
        from vtscore.state.core import _get_click_counter, _set_click_counter

        det_a = DetectorContext("cc_det_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        _set_click_counter(10)
        assert _get_click_counter() == 10

        det_b = DetectorContext("cc_det_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        assert _get_click_counter() == 0  # fresh context
        _set_click_counter(20)

        set_thread_detector_context(det_a)
        assert _get_click_counter() == 10  # preserved

    def test_inclusion_per_detector(self):
        """inclusion is per-detector (training parameter)."""
        from vtscore.state.core import _get_inclusion, _set_inclusion

        det_a = DetectorContext("inc_det_a")
        register_detector_context(det_a)
        set_thread_detector_context(det_a)
        _set_inclusion(5)

        det_b = DetectorContext("inc_det_b")
        register_detector_context(det_b)
        set_thread_detector_context(det_b)
        _set_inclusion(-3)

        assert _get_inclusion() == -3
        set_thread_detector_context(det_a)
        assert _get_inclusion() == 5

    def test_display_name_per_dataset(self):
        from vtscore.state.core import _get_dataset_display_name, _set_dataset_display_name

        ctx_a = DatasetContext("dn_a")
        register_context(ctx_a)
        set_thread_dataset_context(ctx_a)
        _set_dataset_display_name("Dataset A")

        ctx_b = DatasetContext("dn_b")
        register_context(ctx_b)
        set_thread_dataset_context(ctx_b)
        _set_dataset_display_name("Dataset B")

        assert _get_dataset_display_name() == "Dataset B"
        set_thread_dataset_context(ctx_a)
        assert _get_dataset_display_name() == "Dataset A"


# ---------------------------------------------------------------------------
# Empty context fallback tests
# ---------------------------------------------------------------------------


class TestEmptyContextFallback:
    """When no context is active, proxies behave as empty containers."""

    def test_empty_medias_when_no_dataset(self):
        from vtsearch.state import medias

        set_thread_dataset_context(None)
        assert len(medias) == 0
        assert list(medias.keys()) == []

    def test_empty_votes_when_no_detector(self):
        from vtsearch.state import good_votes, bad_votes

        set_thread_detector_context(None)
        assert len(good_votes) == 0
        assert len(bad_votes) == 0


# ---------------------------------------------------------------------------
# Multi-dataset training: sync_labels protection
# ---------------------------------------------------------------------------


class TestSyncLabelsAcrossDatasets:
    """sync_labels_to_loaded_detector must not destroy training data when the
    active dataset has been switched (no votes in the new context)."""

    def test_load_model_route_skips_sync_when_no_votes(self, client, tmp_path):
        """load_model_route must skip sync when the active context has no
        votes, preventing destruction of the model's saved labelset from a
        prior training session on a different dataset."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.registry import add_loaded_detector_id, register_detector, reset_for_tests
        from vtscore.detectors.store import _read_detector, _write_detector
        from vtsearch.settings import get_detectors_dir, set_detectors_dir
        from vtsearch.state import (
            bad_votes,
            good_votes,
            set_thread_detector_context,
            snapshot_medias,
        )
        from vtscore.state.core import DetectorContext, register_detector_context

        reset_for_tests()

        # Phase 1: simulate training on "Dataset A" (6 labels)
        good_votes.update({k: None for k in [1, 2, 3]})
        bad_votes.update({k: None for k in [18, 19, 20]})

        snap = snapshot_medias()
        labelset = LabelSet.from_clips_and_votes(snap, good_votes, bad_votes, expand_dupes=False)
        original_label_count = len(labelset)
        assert original_label_count == 6

        original_dir = get_detectors_dir()
        set_detectors_dir(tmp_path)
        try:
            tm_name = "sync-protect-test"
            tm_path = tmp_path / f"{tm_name}.json"
            _write_detector(
                tm_path,
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "audio",
                    "examples": [],
                    "labelset": labelset.to_dict(),
                },
            )

            entry = register_detector(
                name="Sync Protect Test",
                media_type="audio",
            )
            detector_id = entry["id"]
            add_loaded_detector_id(detector_id)
            det_ctx = DetectorContext(detector_id)
            register_detector_context(det_ctx)
            set_thread_detector_context(det_ctx)

            # Phase 2: simulate switching to Dataset B (clear votes)
            good_votes.clear()
            bad_votes.clear()

            # Phase 3: re-load the same model via the API endpoint
            resp = _load_detector_and_wait(client, detector_id)
            assert resp.status_code == 200

            saved = _read_detector(tm_path)
            assert saved is not None
            saved_labels = saved["labelset"]["labels"]
            assert len(saved_labels) == original_label_count, (
                f"Expected {original_label_count} training labels but load_model "
                f"overwrote them with {len(saved_labels)} (empty votes context)"
            )
        finally:
            set_detectors_dir(original_dir)

    def test_load_model_on_new_dataset_preserves_labels(self, client, tmp_path):
        """Load a trained model on Dataset B via /api/detectors/registry/load.

        The model's saved labelset from Dataset A must survive the load
        even though Dataset B has no votes."""
        from vtscore.datasets.labelset import LabelSet
        from vtscore.detectors.registry import add_loaded_detector_id, register_detector, reset_for_tests
        from vtscore.detectors.store import _read_detector, _write_detector
        from vtsearch.settings import get_detectors_dir, set_detectors_dir
        from vtsearch.state import (
            bad_votes,
            good_votes,
            set_thread_detector_context,
            snapshot_medias,
        )
        from vtscore.state.core import DetectorContext, register_detector_context

        reset_for_tests()

        # Phase 1: simulate training on "Dataset A" (4 labels)
        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [19, 20]})

        snap = snapshot_medias()
        labelset = LabelSet.from_clips_and_votes(snap, good_votes, bad_votes, expand_dupes=False)
        original_label_count = len(labelset)
        assert original_label_count == 4

        original_dir = get_detectors_dir()
        set_detectors_dir(tmp_path)
        try:
            tm_name = "load-protect-test"
            tm_path = tmp_path / f"{tm_name}.json"
            _write_detector(
                tm_path,
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "audio",
                    "examples": [],
                    "labelset": labelset.to_dict(),
                },
            )

            entry = register_detector(
                name="Load Protect Test",
                media_type="audio",
            )
            detector_id = entry["id"]
            add_loaded_detector_id(detector_id)
            det_ctx = DetectorContext(detector_id)
            register_detector_context(det_ctx)
            set_thread_detector_context(det_ctx)

            # Phase 2: simulate switching to Dataset B (clear votes)
            good_votes.clear()
            bad_votes.clear()

            # Phase 3: re-load the same model (as the dashboard would)
            resp = _load_detector_and_wait(client, detector_id)
            assert resp.status_code == 200

            # The model file must still have the original labels
            saved = _read_detector(tm_path)
            assert saved is not None
            saved_labels = saved["labelset"]["labels"]
            assert len(saved_labels) == original_label_count, (
                f"Expected {original_label_count} training labels but load_model "
                f"overwrote them — got {len(saved_labels)}"
            )
        finally:
            set_detectors_dir(original_dir)

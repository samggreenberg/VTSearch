"""Tests for the labelset-driven right-pane API surface.

These cover three concerns introduced together so that a detector loaded
against a different dataset still shows its training labels on the right:

* ``GET /api/trainable-models/<name>/labels-detail`` — read.
* ``POST /api/trainable-models/<name>/labels/<id>/vote`` — toggle on a
  saved labelset element (origin-keyed, dataset-agnostic).
* ``sync_labels_to_loaded_model`` is now non-destructive across datasets:
  a vote in dataset B no longer wipes labels saved from dataset A.
"""

from __future__ import annotations

import shutil

import pytest

from vtsearch.settings import get_trainable_models_dir


@pytest.fixture(autouse=True)
def clean_trainable_models_dir():
    tm_dir = get_trainable_models_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)
    yield
    tm_dir = get_trainable_models_dir()
    if tm_dir.is_dir():
        shutil.rmtree(tm_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_cross_dataset_model(tm_name: str = "cross-ds-model") -> str:
    """Write a trainable model whose labelset references a *different* dataset.

    The labels' origins use importer ``"ds_a"`` so they don't resolve against
    the test medias (which use importer ``"test"``).  Returns the model's
    registry id with the model already loaded into the active context.
    """
    from vtsearch.models.registry import (
        add_loaded_model_id,
        register_model,
        reset_for_tests,
    )
    from vtsearch.models.trainable_model_store import _model_path, _write_model
    from vtsearch.utils.state_core import (
        DetectorContext,
        register_detector_context,
        set_thread_detector_context,
    )

    reset_for_tests()

    labelset = {
        "labels": [
            {
                "md5": "a1" * 16,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {"size": "100"}},
                "origin_name": "alpha.wav",
                "filename": "alpha.wav",
            },
            {
                "md5": "b2" * 16,
                "label": "good",
                "origin": {"importer": "ds_a", "params": {"size": "100"}},
                "origin_name": "beta.wav",
                "filename": "beta.wav",
            },
            {
                "md5": "c3" * 16,
                "label": "bad",
                "origin": {"importer": "ds_a", "params": {"size": "100"}},
                "origin_name": "gamma.wav",
                "filename": "gamma.wav",
            },
        ]
    }
    _write_model(
        _model_path(tm_name),
        {
            "name": tm_name,
            "text_query": "",
            "media_type": "audio",
            "examples": [],
            "labelset": labelset,
        },
    )

    entry = register_model(
        name=tm_name,
        media_type="audio",
        trainable=True,
        trainable_model_name=tm_name,
    )
    model_id = entry["id"]
    add_loaded_model_id(model_id)
    det_ctx = DetectorContext(model_id)
    register_detector_context(det_ctx)
    set_thread_detector_context(det_ctx)
    return model_id


# ---------------------------------------------------------------------------
# GET /api/trainable-models/<name>/labels-detail
# ---------------------------------------------------------------------------


class TestLabelsDetail:
    def test_returns_404_for_unknown_model(self, client):
        res = client.get("/api/trainable-models/does-not-exist/labels-detail")
        assert res.status_code == 404

    def test_lists_saved_labels_independent_of_loaded_dataset(self, client):
        """Right-pane data source must surface labels that don't resolve into
        the active dataset — the whole point of the labelset-driven pane."""
        _seed_cross_dataset_model()

        res = client.get("/api/trainable-models/cross-ds-model/labels-detail")
        assert res.status_code == 200
        body = res.get_json()

        assert body["media_type"] == "audio"
        assert len(body["good"]) == 2
        assert len(body["bad"]) == 1

        ids = {e["id"] for e in body["good"] + body["bad"]}
        assert len(ids) == 3, "ids must be unique per element"

        for elem in body["good"] + body["bad"]:
            # Cross-dataset: nothing resolves into the active dataset.
            assert elem["cid"] is None
            assert elem["media_type"] == "audio"
            assert elem["name"]  # display string non-empty


# ---------------------------------------------------------------------------
# POST /api/trainable-models/<name>/labels/<id>/vote
# ---------------------------------------------------------------------------


class TestLabelElementVote:
    def test_flip_label(self, client):
        _seed_cross_dataset_model()
        detail = client.get("/api/trainable-models/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]

        res = client.post(
            f"/api/trainable-models/cross-ds-model/labels/{target['id']}/vote",
            json={"vote": "bad"},
        )
        assert res.status_code == 200
        assert res.get_json()["action"] == "flipped"

        after = client.get("/api/trainable-models/cross-ds-model/labels-detail").get_json()
        assert len(after["good"]) == 1
        assert len(after["bad"]) == 2
        assert any(e["id"] == target["id"] for e in after["bad"])

    def test_same_vote_removes_element(self, client):
        _seed_cross_dataset_model()
        detail = client.get("/api/trainable-models/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]

        res = client.post(
            f"/api/trainable-models/cross-ds-model/labels/{target['id']}/vote",
            json={"vote": "good"},
        )
        assert res.status_code == 200
        assert res.get_json()["action"] == "removed"

        after = client.get("/api/trainable-models/cross-ds-model/labels-detail").get_json()
        assert len(after["good"]) == 1
        assert len(after["bad"]) == 1
        assert not any(e["id"] == target["id"] for e in after["good"] + after["bad"])

    def test_unknown_id_404(self, client):
        _seed_cross_dataset_model()
        res = client.post(
            "/api/trainable-models/cross-ds-model/labels/deadbeef/vote",
            json={"vote": "good"},
        )
        assert res.status_code == 404

    def test_invalid_vote_value_400(self, client):
        _seed_cross_dataset_model()
        detail = client.get("/api/trainable-models/cross-ds-model/labels-detail").get_json()
        target = detail["good"][0]
        res = client.post(
            f"/api/trainable-models/cross-ds-model/labels/{target['id']}/vote",
            json={"vote": "maybe"},
        )
        assert res.status_code == 400


# ---------------------------------------------------------------------------
# Cross-dataset: voting in dataset B preserves dataset A's labels on disk
# ---------------------------------------------------------------------------


class TestCrossDatasetVoteDoesNotWipeLabels:
    """Regression test for the bug where opening a detector against a fresh
    dataset and casting a single vote would overwrite the on-disk labelset
    with just that vote, destroying everything trained on the prior dataset.
    """

    def test_vote_in_active_dataset_merges_with_saved_labelset(self, client):
        from vtsearch.models.label_sync import sync_labels_to_loaded_model
        from vtsearch.models.trainable_model_store import _model_path, _read_model
        from vtsearch.utils import good_votes

        _seed_cross_dataset_model()

        # Active-dataset votes (test medias use importer "test").
        good_votes[1] = None
        good_votes[2] = None

        sync_labels_to_loaded_model()

        saved = _read_model(_model_path("cross-ds-model"))
        keys = {
            (el.get("origin", {}).get("importer", ""), el.get("origin_name", "")) for el in saved["labelset"]["labels"]
        }
        # Cross-dataset (ds_a) labels survived.
        assert ("ds_a", "alpha.wav") in keys
        assert ("ds_a", "beta.wav") in keys
        assert ("ds_a", "gamma.wav") in keys
        # Active-dataset votes were merged in.
        assert any(imp == "test" for imp, _ in keys)

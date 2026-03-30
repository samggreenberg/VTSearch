"""Tests for trainable model CRUD and label persistence."""

import json
import shutil

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


class TestCreateTrainableModel:
    def test_create_success(self, client):
        res = client.post(
            "/api/trainable-models",
            json={"name": "Dog Barks", "media_type": "audio", "text_query": "sounds of dogs barking"},
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["success"] is True
        assert data["name"] == "Dog Barks"
        assert data["text_query"] == "sounds of dogs barking"
        assert data["num_labels"] == 0

    def test_create_missing_name(self, client):
        res = client.post(
            "/api/trainable-models",
            json={"text_query": "sounds"},
        )
        assert res.status_code == 400
        assert "name" in res.get_json()["error"]

    def test_create_missing_text_query(self, client):
        res = client.post(
            "/api/trainable-models",
            json={"name": "Test", "media_type": "audio"},
        )
        assert res.status_code == 400
        assert "text_query" in res.get_json()["error"]

    def test_create_missing_media_type(self, client):
        res = client.post(
            "/api/trainable-models",
            json={"name": "Test", "text_query": "sounds"},
        )
        assert res.status_code == 400
        assert "media_type" in res.get_json()["error"]

    def test_create_rejects_any_media_type(self, client):
        res = client.post(
            "/api/trainable-models",
            json={"name": "Test", "media_type": "any", "text_query": "sounds"},
        )
        assert res.status_code == 400
        assert "media_type" in res.get_json()["error"]

    def test_create_duplicate(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Dog Barks", "media_type": "audio", "text_query": "dogs"},
        )
        res = client.post(
            "/api/trainable-models",
            json={"name": "Dog Barks", "media_type": "audio", "text_query": "dogs again"},
        )
        assert res.status_code == 409

    def test_file_created_on_disk(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Test Model", "media_type": "audio", "text_query": "test"},
        )
        files = list(get_trainable_models_dir().glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["name"] == "Test Model"
        assert data["text_query"] == "test"
        assert data["labelset"] == {"labels": []}


class TestListTrainableModels:
    def test_empty_list(self, client):
        res = client.get("/api/trainable-models")
        assert res.status_code == 200
        data = res.get_json()
        assert data["models"] == []

    def test_list_after_create(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Model A", "media_type": "audio", "text_query": "a"},
        )
        client.post(
            "/api/trainable-models",
            json={"name": "Model B", "media_type": "audio", "text_query": "b"},
        )
        res = client.get("/api/trainable-models")
        data = res.get_json()
        names = [m["name"] for m in data["models"]]
        assert "Model A" in names
        assert "Model B" in names


class TestGetTrainableModel:
    def test_get_existing(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "My Model", "media_type": "audio", "text_query": "test query"},
        )
        res = client.get("/api/trainable-models/My%20Model")
        assert res.status_code == 200
        data = res.get_json()
        assert data["name"] == "My Model"
        assert data["text_query"] == "test query"
        assert "labelset" in data

    def test_get_nonexistent(self, client):
        res = client.get("/api/trainable-models/nonexistent")
        assert res.status_code == 404


class TestDeleteTrainableModel:
    def test_delete_existing(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "To Delete", "media_type": "audio", "text_query": "test"},
        )
        res = client.delete("/api/trainable-models/To%20Delete")
        assert res.status_code == 200
        assert res.get_json()["success"] is True

        # Verify it's gone
        res = client.get("/api/trainable-models/To%20Delete")
        assert res.status_code == 404

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/trainable-models/nonexistent")
        assert res.status_code == 404


class TestRenameTrainableModel:
    def test_rename_success(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Old Name", "media_type": "audio", "text_query": "test"},
        )
        res = client.put(
            "/api/trainable-models/Old%20Name/rename",
            json={"new_name": "New Name"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["new_name"] == "New Name"

        # Old name should be gone
        res = client.get("/api/trainable-models/Old%20Name")
        assert res.status_code == 404

        # New name should exist
        res = client.get("/api/trainable-models/New%20Name")
        assert res.status_code == 200
        assert res.get_json()["name"] == "New Name"

    def test_rename_nonexistent(self, client):
        res = client.put(
            "/api/trainable-models/nonexistent/rename",
            json={"new_name": "Foo"},
        )
        assert res.status_code == 404

    def test_rename_missing_new_name(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Test", "media_type": "audio", "text_query": "test"},
        )
        res = client.put(
            "/api/trainable-models/Test/rename",
            json={},
        )
        assert res.status_code == 400

    def test_rename_updates_model_registry(self, client):
        """Renaming a trainable model should update registry references."""
        from vtsearch.models.registry import find_by_trainable_model_name, get_model

        # Create a trainable model and register it in the model registry
        client.post(
            "/api/trainable-models",
            json={"name": "Original", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={"name": "Original", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        assert res.status_code == 201
        model_id = res.get_json()["model"]["id"]

        # Rename the trainable model directly (not through the registry endpoint)
        res = client.put(
            "/api/trainable-models/Original/rename",
            json={"new_name": "Renamed"},
        )
        assert res.status_code == 200

        # Registry entry should now reference the new name
        entry = get_model(model_id)
        assert entry is not None
        assert entry["name"] == "Renamed"
        assert entry["trainable_model_name"] == "Renamed"

        # Look up by old name should fail
        assert find_by_trainable_model_name("Original") is None

        # Look up by new name should succeed
        assert find_by_trainable_model_name("Renamed") is not None

    def test_rename_conflict(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Model A", "media_type": "audio", "text_query": "a"},
        )
        client.post(
            "/api/trainable-models",
            json={"name": "Model B", "media_type": "audio", "text_query": "b"},
        )
        res = client.put(
            "/api/trainable-models/Model%20A/rename",
            json={"new_name": "Model B"},
        )
        assert res.status_code == 409


class TestSaveLabels:
    def test_save_labels_empty(self, client):
        """Save labels when there are no votes — should produce empty labelset."""
        client.post(
            "/api/trainable-models",
            json={"name": "Labeler", "media_type": "audio", "text_query": "test"},
        )
        res = client.post("/api/trainable-models/Labeler/labels")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["num_labels"] == 0

    def test_save_labels_with_votes(self, client):
        """Save labels after casting votes — labelset should contain the voted medias."""
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded for this test")

        # Cast a good vote on the first media
        first_id = next(iter(medias))
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})

        # Cast a bad vote on the second media
        media_ids = list(medias.keys())
        if len(media_ids) < 2:
            pytest.skip("Need at least 2 medias")
        second_id = media_ids[1]
        client.post(f"/api/medias/{second_id}/vote", json={"vote": "bad"})

        client.post(
            "/api/trainable-models",
            json={"name": "Voted Model", "media_type": "audio", "text_query": "test"},
        )
        res = client.post("/api/trainable-models/Voted%20Model/labels")
        assert res.status_code == 200
        data = res.get_json()
        assert data["num_labels"] == 2

        # Verify the labels are persisted on disk
        model_res = client.get("/api/trainable-models/Voted%20Model")
        model_data = model_res.get_json()
        labels = model_data["labelset"]["labels"]
        assert len(labels) == 2
        label_values = {lbl["label"] for lbl in labels}
        assert "good" in label_values
        assert "bad" in label_values

    def test_save_labels_nonexistent_model(self, client):
        res = client.post("/api/trainable-models/nonexistent/labels")
        assert res.status_code == 404

    def test_save_labels_does_not_expand_dupes(self, client):
        """Saving labels for a dupe-set representative should NOT expand members.

        Regression test: previously, a vote on a dupe-set representative
        with N members produced N label entries, inflating the stored
        label count.  Trainable model persistence should store one entry
        per vote, not one per duplicate.
        """
        import copy

        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded for this test")

        first_id = next(iter(medias))
        original = copy.deepcopy(medias[first_id])

        # Turn the first media into a dupe-set representative with 5 members
        medias[first_id]["origin"] = {
            "importer": "dupe_set",
            "params": {"name": original.get("filename", "a.wav")},
            "members": [
                {"origin": {"importer": "test", "params": {}}, "origin_name": f"dup_{i}.wav", "filename": f"dup_{i}.wav", "category": "c"}
                for i in range(5)
            ],
        }
        try:
            client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
            client.post("/api/trainable-models", json={"name": "DupeTest", "media_type": "audio", "text_query": "test"})

            res = client.post("/api/trainable-models/DupeTest/labels")
            assert res.status_code == 200
            data = res.get_json()
            # Should be 1 label (the vote), NOT 5 (the dupe members)
            assert data["num_labels"] == 1

            model_res = client.get("/api/trainable-models/DupeTest")
            labels = model_res.get_json()["labelset"]["labels"]
            assert len(labels) == 1
        finally:
            medias[first_id] = original


class TestLabelVoteIsolation:
    """Clearing votes before importing a model's labels prevents cross-contamination."""

    def test_clear_votes_before_import_prevents_leakage(self, client):
        """Votes from Model A must not persist into a Model B label session.

        Simulates the Label-button flow: clear votes, then import a model's
        labels.  Without the clear, votes from a prior session leak in.
        """
        from vtsearch.utils import good_votes, bad_votes, medias

        ids = list(medias.keys())
        if len(ids) < 4:
            pytest.skip("Need at least 4 medias")

        # Create two trainable models
        client.post("/api/trainable-models", json={"name": "Model A", "media_type": "audio", "text_query": "a"})
        client.post("/api/trainable-models", json={"name": "Model B", "media_type": "audio", "text_query": "b"})

        # Simulate labeling with Model A: vote on ids[0] and ids[1]
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[1]}/vote", json={"vote": "bad"})
        client.post("/api/trainable-models/Model%20A/labels")  # save 2 labels

        # Now clear votes (as the Label button should do) and import Model B's labels
        client.post("/api/votes/clear")
        assert len(good_votes) == 0
        assert len(bad_votes) == 0

        # Model B has no labels, so import is a no-op — votes should remain empty
        model_b = client.get("/api/trainable-models/Model%20B").get_json()
        assert len(model_b["labelset"]["labels"]) == 0

        client.post("/api/labels/import", json={"labels": model_b["labelset"]["labels"]})
        assert len(good_votes) == 0, "Model A's votes should not leak into Model B's session"
        assert len(bad_votes) == 0

    def test_import_after_clear_only_has_model_labels(self, client):
        """After clearing + importing, only the target model's labels are active."""
        from vtsearch.utils import good_votes, bad_votes, medias

        ids = list(medias.keys())
        if len(ids) < 4:
            pytest.skip("Need at least 4 medias")

        # Create model and label 2 items
        client.post("/api/trainable-models", json={"name": "Target", "media_type": "audio", "text_query": "t"})
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[1]}/vote", json={"vote": "bad"})
        client.post("/api/trainable-models/Target/labels")

        # Add extra votes that DON'T belong to the model (simulating stale state)
        client.post(f"/api/medias/{ids[2]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[3]}/vote", json={"vote": "bad"})
        assert len(good_votes) == 2  # ids[0] + ids[2]
        assert len(bad_votes) == 2  # ids[1] + ids[3]

        # Clear votes, then import only Target's labels
        client.post("/api/votes/clear")
        target_data = client.get("/api/trainable-models/Target").get_json()
        client.post("/api/labels/import", json={"labels": target_data["labelset"]["labels"]})

        # Should only have the 2 labels from Target, not the 4 from before
        assert len(good_votes) + len(bad_votes) == 2
        assert ids[2] not in good_votes, "Stale vote should be gone after clear+import"
        assert ids[3] not in bad_votes, "Stale vote should be gone after clear+import"


class TestDeleteRegisteredModel:
    """Tests for DELETE /api/models/registry/<model_id>."""

    def test_delete_registered_model(self, client):
        """Deleting a registered model removes it from the registry."""
        from vtsearch.models.registry import get_model

        res = client.post(
            "/api/models/registry",
            json={"name": "DelMe", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        assert res.status_code == 201
        model_id = res.get_json()["model"]["id"]

        res = client.delete(f"/api/models/registry/{model_id}")
        assert res.status_code == 200
        assert res.get_json()["ok"] is True
        assert get_model(model_id) is None

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/models/registry/nonexistent_id")
        assert res.status_code == 404

    def test_delete_loaded_model(self, client):
        """Deleting a loaded model should also unload it."""
        from vtsearch.models.registry import get_model, is_model_loaded

        client.post(
            "/api/trainable-models",
            json={"name": "LoadDel", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={"name": "LoadDel", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        model_id = res.get_json()["model"]["id"]
        _load_model_and_wait(client, model_id)
        assert is_model_loaded(model_id)

        res = client.delete(f"/api/models/registry/{model_id}")
        assert res.status_code == 200
        assert get_model(model_id) is None
        assert not is_model_loaded(model_id)

    def test_delete_removes_autorun_detector(self, client):
        """Deleting a model that has a detector_name cleans up autorun_detectors."""
        from vtsearch.models.registry import get_model
        from vtsearch.utils import autorun_detectors

        # Register with a detector_name
        res = client.post(
            "/api/models/registry",
            json={"name": "DetDel", "media_type": "audio", "trainable": False, "detector_name": "my_det"},
        )
        model_id = res.get_json()["model"]["id"]

        # Simulate an autorun detector being present
        autorun_detectors["my_det"] = lambda scores: scores

        res = client.delete(f"/api/models/registry/{model_id}")
        assert res.status_code == 200
        assert "my_det" not in autorun_detectors
        assert get_model(model_id) is None


class TestLoadModelEndpoint:
    """Tests for POST /api/models/registry/load."""

    def test_load_model(self, client):
        from vtsearch.models.registry import is_model_loaded

        res = client.post(
            "/api/models/registry",
            json={"name": "M", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        model_id = res.get_json()["model"]["id"]

        res = _load_model_and_wait(client, model_id)
        assert res.status_code == 200
        assert is_model_loaded(model_id)

    def test_unload_model(self, client):
        from vtsearch.models.registry import add_loaded_model_id, is_model_loaded

        add_loaded_model_id("fake")
        assert is_model_loaded("fake")
        res = client.post("/api/models/registry/load", json={"model_id": None})
        assert res.status_code == 200

    def test_load_nonexistent(self, client):
        res = client.post("/api/models/registry/load", json={"model_id": "nope"})
        assert res.status_code == 404

    def test_load_clears_previous_labels(self, client):
        """Loading model B must not carry over labels from model A."""
        from vtsearch.utils import bad_votes, good_votes, medias

        if not medias:
            pytest.skip("No medias loaded")

        ids = list(medias.keys())

        # Create two trainable models + registry entries.
        for name in ("ModelA", "ModelB"):
            client.post(
                "/api/trainable-models",
                json={"name": name, "media_type": "audio", "text_query": "test"},
            )
        res_a = client.post(
            "/api/models/registry",
            json={"name": "ModelA", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        mid_a = res_a.get_json()["model"]["id"]
        res_b = client.post(
            "/api/models/registry",
            json={"name": "ModelB", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        mid_b = res_b.get_json()["model"]["id"]

        # Load model A and cast some votes.
        _load_model_and_wait(client, mid_a)
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[1]}/vote", json={"vote": "bad"})
        assert ids[0] in good_votes
        assert ids[1] in bad_votes

        # Now load model B — votes from A must be gone.
        _load_model_and_wait(client, mid_b)
        assert ids[0] not in good_votes, "good vote from model A leaked into model B"
        assert ids[1] not in bad_votes, "bad vote from model A leaked into model B"

    def test_load_restores_saved_labels(self, client):
        """Loading a model that has a saved labelset should restore its labels."""
        from vtsearch.utils import good_votes, medias

        if not medias:
            pytest.skip("No medias loaded")

        ids = list(medias.keys())

        # Create model, load it, vote, then save labels.
        client.post(
            "/api/trainable-models",
            json={"name": "Persist", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={"name": "Persist", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        mid = res.get_json()["model"]["id"]
        _load_model_and_wait(client, mid)
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        # Labels auto-sync on vote, so the trainable model file now has 1 label.

        # Unload to clear votes, then reload — label should be restored.
        client.post("/api/models/registry/load", json={"model_id": None})
        assert ids[0] not in good_votes

        res = _load_model_and_wait(client, mid)
        assert res.status_code == 200
        assert ids[0] in good_votes, "saved label was not restored on model load"


class TestVoteSyncsToLoadedModel:
    """Voting while a trainable model is loaded should auto-update the model's labelset."""

    def test_vote_updates_model_labels(self, client):
        """Casting a vote with a loaded model should persist labels and update registry stats."""
        from vtsearch.models.registry import get_model
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded")

        # Create and register a trainable model
        client.post(
            "/api/trainable-models",
            json={"name": "AutoSync", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={"name": "AutoSync", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        model_id = res.get_json()["model"]["id"]

        # Load the model
        _load_model_and_wait(client, model_id)

        # Cast a vote
        first_id = next(iter(medias))
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})

        # Check that the model's labelset was updated
        model_data = client.get("/api/trainable-models/AutoSync").get_json()
        labels = model_data["labelset"]["labels"]
        assert len(labels) == 1
        assert labels[0]["label"] == "good"

        # Check that the registry entry was updated
        entry = get_model(model_id)
        assert entry["num_training"] == 1
        assert entry.get("last_trained_at") is not None

    def test_vote_toggle_off_updates_model(self, client):
        """Toggling a vote off should update the model labelset to reflect removal."""
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded")

        client.post(
            "/api/trainable-models",
            json={"name": "ToggleSync", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={"name": "ToggleSync", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        model_id = res.get_json()["model"]["id"]
        _load_model_and_wait(client, model_id)

        first_id = next(iter(medias))
        # Vote good
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
        model_data = client.get("/api/trainable-models/ToggleSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 1

        # Toggle off (vote good again)
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
        model_data = client.get("/api/trainable-models/ToggleSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 0

    def test_no_sync_without_loaded_model(self, client):
        """Voting with no loaded model should not create/update any model files."""
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded")

        client.post(
            "/api/trainable-models",
            json={"name": "NoSync", "media_type": "audio", "text_query": "test"},
        )

        first_id = next(iter(medias))
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})

        # Model should still have empty labelset
        model_data = client.get("/api/trainable-models/NoSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 0

    def test_label_import_syncs_to_loaded_model(self, client):
        """Importing labels with a loaded model should persist to the model."""
        from vtsearch.models.registry import get_model
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded")

        client.post(
            "/api/trainable-models",
            json={"name": "ImportSync", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/models/registry",
            json={"name": "ImportSync", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        model_id = res.get_json()["model"]["id"]
        _load_model_and_wait(client, model_id)

        # Get an MD5 from the first media
        first_id = next(iter(medias))
        media = medias[first_id]
        md5 = media.get("md5", "")

        # Import a label
        client.post("/api/labels/import", json={"labels": [{"md5": md5, "label": "good"}]})

        # Model should have the imported label
        model_data = client.get("/api/trainable-models/ImportSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 1

        entry = get_model(model_id)
        assert entry["num_training"] == 1


class TestSeedVotesFromExamples:
    """When loading a model with media examples, matching medias get auto-labeled Good."""

    @pytest.fixture(autouse=True)
    def _restore_medias(self):
        """Remove any media items inserted by seeding after each test."""
        from vtsearch.utils import medias

        saved = dict(medias)
        yield
        # Remove items that were added, restore any that were modified
        medias.clear()
        medias.update(saved)

    def _create_example_file(self, media_bytes: bytes, filename: str = "ex.wav") -> str:
        """Write *media_bytes* into data/example_media/<filename> and return the filename."""
        from vtsearch.config import DATA_DIR

        example_dir = DATA_DIR / "example_media"
        example_dir.mkdir(parents=True, exist_ok=True)
        dest = example_dir / filename
        dest.write_bytes(media_bytes)
        return filename

    # ---- POST /api/votes/seed-from-examples ----

    def test_seed_endpoint_adds_good_votes(self, client):
        """Media examples whose MD5 matches a loaded media should become good votes."""
        from vtsearch.utils import good_votes, medias

        if not medias:
            pytest.skip("No medias loaded")

        first_id = next(iter(medias))
        media = medias[first_id]
        media_bytes = media["media_bytes"]

        fname = self._create_example_file(media_bytes, "seed_test.wav")

        res = client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": fname}]},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["seeded"] == 1
        assert data["skipped"] == 0
        assert first_id in good_votes

    def test_seed_skips_text_examples(self, client):
        """Text examples should be skipped (only media examples are seeded)."""
        res = client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "text", "value": "dog barking"}]},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["seeded"] == 0
        assert data["skipped"] == 1

    def test_seed_skips_nonexistent_file(self, client):
        """A media example whose file doesn't exist should be skipped."""
        res = client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": "no_such_file.wav"}]},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["seeded"] == 0
        assert data["skipped"] == 1

    def test_seed_unmatched_inserts_new_media(self, client):
        """A media example not in the dataset should be embedded and inserted as a new media."""
        from vtsearch.utils import good_votes, medias

        original_count = len(medias)

        # Create a file whose content differs from all loaded medias
        fname = self._create_example_file(b"novel-example-content", "novel.wav")

        res = client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": fname}]},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["seeded"] == 1

        # A new media should have been inserted
        assert len(medias) == original_count + 1

        # The new media should be in good_votes
        new_id = max(medias.keys())
        assert new_id in good_votes

        # The new media should have the example_media origin (not a dataset origin)
        new_media = medias[new_id]
        assert new_media["origin"]["importer"] == "example_media"
        assert new_media["origin"]["params"]["filename"] == fname
        assert new_media["filename"] == fname
        assert new_media["embedding"] is not None

    def test_seed_preserves_original_origins(self, client):
        """Seeded medias should keep their original dataset origins."""
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded")

        first_id = next(iter(medias))
        media = medias[first_id]
        original_origin = media.get("origin")
        original_origin_name = media.get("origin_name", "")

        fname = self._create_example_file(media["media_bytes"], "origin_test.wav")
        client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": fname}]},
        )

        # Origin should be unchanged
        assert medias[first_id].get("origin") == original_origin
        assert medias[first_id].get("origin_name", "") == original_origin_name

    def test_seed_appears_in_label_export(self, client):
        """Seeded good votes should appear in the label export."""
        from vtsearch.utils import medias

        if not medias:
            pytest.skip("No medias loaded")

        first_id = next(iter(medias))
        media = medias[first_id]
        fname = self._create_example_file(media["media_bytes"], "export_test.wav")

        client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": fname}]},
        )

        res = client.get("/api/labels/export")
        assert res.status_code == 200
        labels = res.get_json()["labels"]
        assert len(labels) >= 1
        assert any(lbl["label"] == "good" for lbl in labels)

    def test_new_example_appears_in_label_export(self, client):
        """A non-dataset example inserted by seeding should appear in label export."""
        fname = self._create_example_file(b"export-novel-bytes", "export_novel.wav")

        client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": fname}]},
        )

        res = client.get("/api/labels/export")
        assert res.status_code == 200
        labels = res.get_json()["labels"]
        example_labels = [
            lbl for lbl in labels if isinstance(lbl.get("origin"), dict) and lbl["origin"].get("importer") == "example_media"
        ]
        assert len(example_labels) == 1
        assert example_labels[0]["label"] == "good"
        assert example_labels[0]["origin"]["params"]["filename"] == fname

    def test_new_example_usable_in_training(self, client):
        """Inserted examples should have embeddings usable by learned-sort."""
        from vtsearch.utils import good_votes, bad_votes, medias

        if not medias:
            pytest.skip("No medias loaded")

        # Seed a novel example as good
        fname = self._create_example_file(b"training-novel-bytes", "train_novel.wav")
        client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": fname}]},
        )
        assert len(good_votes) >= 1

        # Add a bad vote on the first dataset media so we have both good+bad
        first_id = next(iter(medias))
        # Make sure we don't vote bad on the newly inserted item
        new_id = max(medias.keys())
        target_id = first_id if first_id != new_id else list(medias.keys())[1]
        client.post(f"/api/medias/{target_id}/vote", json={"vote": "bad"})
        assert len(bad_votes) >= 1

        # Learned sort should work — it accesses the embedding from the inserted media
        res = client.post("/api/learned-sort")
        assert res.status_code == 200
        data = res.get_json()
        assert "results" in data
        assert len(data["results"]) > 0

    # ---- Model load auto-seeding ----

    def test_load_model_seeds_from_media_examples(self, client):
        """Loading a model with media examples should auto-seed good votes."""
        from vtsearch.utils import good_votes, medias

        if not medias:
            pytest.skip("No medias loaded")

        first_id = next(iter(medias))
        media = medias[first_id]
        fname = self._create_example_file(media["media_bytes"], "autoload.wav")

        # Create model with a media example
        client.post(
            "/api/trainable-models",
            json={
                "name": "AutoSeed",
                "media_type": "audio",
                "examples": [{"type": "media", "value": fname}],
            },
        )
        # Register in model registry
        res = client.post(
            "/api/models/registry",
            json={
                "name": "AutoSeed",
                "media_type": "audio",
                "trainable": True,
                "text_query": "",
                "media_example": fname,
            },
        )
        model_id = res.get_json()["model"]["id"]

        # Clear any prior votes
        client.post("/api/votes/clear")
        assert len(good_votes) == 0

        # Load model — should auto-seed
        res = _load_model_and_wait(client, model_id)
        assert res.status_code == 200
        assert first_id in good_votes, "example media should be seeded as good vote"

    def test_load_model_without_examples_seeds_nothing(self, client):
        """Loading a text-only model should seed 0 examples."""
        from vtsearch.utils import good_votes

        client.post(
            "/api/trainable-models",
            json={"name": "TextOnly", "media_type": "audio", "text_query": "dogs"},
        )
        res = client.post(
            "/api/models/registry",
            json={
                "name": "TextOnly",
                "media_type": "audio",
                "trainable": True,
                "text_query": "dogs",
            },
        )
        model_id = res.get_json()["model"]["id"]

        client.post("/api/votes/clear")
        res = _load_model_and_wait(client, model_id)
        assert res.status_code == 200
        assert len(good_votes) == 0

    def test_seeded_examples_enable_autopilot_skip(self, client):
        """If seeded examples meet the autopilot threshold, Good phase can be skipped.

        This tests the backend side: enough media examples seed enough
        good_votes that ``goodCount >= autopilot_top_greens``.
        """
        from vtsearch.utils import good_votes, medias

        ids = list(medias.keys())
        if len(ids) < 4:
            pytest.skip("Need at least 4 medias")

        # Create example files for 4 medias
        fnames = []
        for i, cid in enumerate(ids[:4]):
            fname = self._create_example_file(medias[cid]["media_bytes"], f"skip_{i}.wav")
            fnames.append(fname)

        examples = [{"type": "media", "value": fn} for fn in fnames]
        client.post(
            "/api/trainable-models",
            json={"name": "SkipGood", "media_type": "audio", "examples": examples},
        )
        res = client.post(
            "/api/models/registry",
            json={
                "name": "SkipGood",
                "media_type": "audio",
                "trainable": True,
                "text_query": "",
            },
        )
        model_id = res.get_json()["model"]["id"]

        client.post("/api/votes/clear")
        _load_model_and_wait(client, model_id)

        # With default autopilot_top_greens=3, 4 good votes is enough to skip Good phase
        assert len(good_votes) >= 4

    def test_load_model_seeds_novel_example(self, client):
        """Loading a model with a non-dataset example should embed and insert it."""
        from vtsearch.utils import good_votes, medias

        original_count = len(medias)
        fname = self._create_example_file(b"novel-load-bytes", "novel_load.wav")

        client.post(
            "/api/trainable-models",
            json={
                "name": "NovelSeed",
                "media_type": "audio",
                "examples": [{"type": "media", "value": fname}],
            },
        )
        res = client.post(
            "/api/models/registry",
            json={
                "name": "NovelSeed",
                "media_type": "audio",
                "trainable": True,
                "text_query": "",
            },
        )
        model_id = res.get_json()["model"]["id"]

        client.post("/api/votes/clear")
        res = _load_model_and_wait(client, model_id)
        assert res.status_code == 200

        # A new media should have been inserted
        assert len(medias) == original_count + 1
        new_id = max(medias.keys())
        assert new_id in good_votes
        assert medias[new_id]["origin"]["importer"] == "example_media"

    def test_seed_directory_traversal_blocked(self, client):
        """Path traversal attempts in example filenames should be rejected."""
        res = client.post(
            "/api/votes/seed-from-examples",
            json={"examples": [{"type": "media", "value": "../../etc/passwd"}]},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["seeded"] == 0
        assert data["skipped"] == 1


class TestLoadModelCrossDatasetResolution:
    """Loading a model trained on Dataset A while Dataset B is loaded should
    still resolve labels when the underlying files are the same."""

    def test_load_model_resolves_labels_via_origin(self, client, tmp_path):
        """Labels from Dataset A should resolve by origin→MD5 on Dataset B.

        Simulates: train detector on Dataset A (labels with folder origins),
        switch to Dataset B (same files, different origin keys), open Train
        mode.  The label restore should follow origin trails, compute MD5s,
        and match against loaded medias.
        """
        import hashlib

        import numpy as np

        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import good_votes, bad_votes, medias

        reset_for_tests()

        # --- Build files on disk (shared content between both datasets) ---
        label_folder = tmp_path / "dataset_a"
        label_folder.mkdir()
        good_file = label_folder / "good_0.wav"
        bad_file = label_folder / "bad_0.wav"
        good_file.write_bytes(b"shared_good_content")
        bad_file.write_bytes(b"shared_bad_content")

        good_md5 = hashlib.md5(b"shared_good_content").hexdigest()
        bad_md5 = hashlib.md5(b"shared_bad_content").hexdigest()

        label_origin = {
            "importer": "folder",
            "params": {"path": str(label_folder), "media_type": "sounds"},
        }

        # Labelset entries with Dataset A origin info and DIFFERENT MD5s
        # (simulating that the labelset was saved with old/different hashes)
        label_entries = [
            {
                "md5": "dataset_a_good_old_hash",
                "label": "good",
                "origin": label_origin,
                "origin_name": "good_0.wav",
                "filename": "good_0.wav",
            },
            {
                "md5": "dataset_a_bad_old_hash",
                "label": "bad",
                "origin": label_origin,
                "origin_name": "bad_0.wav",
                "filename": "bad_0.wav",
            },
        ]

        # --- Write trainable model ---
        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "cross-dataset-load"
            _write_model(tmp_path / f"{tm_name}.json", {
                "name": tm_name,
                "text_query": "",
                "media_type": "audio",
                "examples": [],
                "labelset": {"labels": label_entries},
            })

            entry = register_model(
                name="Cross Load Test",
                media_type="audio",
                trainable=True,
                trainable_model_name=tm_name,
            )
            model_id = entry["id"]

            # --- Replace medias with Dataset B (same file content, different origins) ---
            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(99)
            medias[1] = {
                "id": 1,
                "type": "audio",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "md5": good_md5,  # same content as good_0.wav
                "filename": "completely_different_name.wav",
                "origin": {"importer": "folder", "params": {"path": "/other/place"}},
                "origin_name": "completely_different_name.wav",
            }
            medias[2] = {
                "id": 2,
                "type": "audio",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "md5": bad_md5,  # same content as bad_0.wav
                "filename": "another_file.wav",
                "origin": {"importer": "folder", "params": {"path": "/other/place"}},
                "origin_name": "another_file.wav",
            }

            try:
                res = _load_model_and_wait(client, model_id)
                assert res.status_code == 200
                assert 1 in good_votes, "good label should be applied to media 1"
                assert 2 in bad_votes, "bad label should be applied to media 2"
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_trainable_models_dir(original_dir)

    def test_load_model_name_fallback(self, client, tmp_path):
        """Labels with matching origin_name should resolve even without origin/MD5 match."""
        import numpy as np

        from vtsearch.models.registry import register_model, reset_for_tests
        from vtsearch.routes.trainable_models import _write_model
        from vtsearch.settings import get_trainable_models_dir, set_trainable_models_dir
        from vtsearch.utils import good_votes, medias

        reset_for_tests()

        label_entries = [
            {
                "md5": "nonexistent_hash",
                "label": "good",
                "origin": {"importer": "folder", "params": {"path": "/gone"}},
                "origin_name": "shared_name.wav",
                "filename": "shared_name.wav",
            },
        ]

        original_dir = get_trainable_models_dir()
        set_trainable_models_dir(tmp_path)
        try:
            tm_name = "name-fallback"
            _write_model(tmp_path / f"{tm_name}.json", {
                "name": tm_name,
                "text_query": "",
                "media_type": "audio",
                "examples": [],
                "labelset": {"labels": label_entries},
            })

            entry = register_model(
                name="Name Fallback Test",
                media_type="audio",
                trainable=True,
                trainable_model_name=tm_name,
            )
            model_id = entry["id"]

            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(42)
            medias[1] = {
                "id": 1,
                "type": "audio",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "md5": "totally_different_md5",
                "filename": "shared_name.wav",
                "origin": {"importer": "folder", "params": {"path": "/different"}},
                "origin_name": "shared_name.wav",
            }

            try:
                res = _load_model_and_wait(client, model_id)
                assert res.status_code == 200
                assert 1 in good_votes, "origin_name fallback should have matched the label"
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_trainable_models_dir(original_dir)


"""Tests for trainable model CRUD and label persistence."""

import json
import shutil

import pytest

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


class TestLoadModelEndpoint:
    """Tests for POST /api/models/registry/load."""

    def test_load_model(self, client):
        from vtsearch.models.registry import get_loaded_id

        res = client.post(
            "/api/models/registry",
            json={"name": "M", "media_type": "audio", "trainable": True, "text_query": "test"},
        )
        model_id = res.get_json()["model"]["id"]

        res = client.post("/api/models/registry/load", json={"model_id": model_id})
        assert res.status_code == 200
        assert get_loaded_id() == model_id

    def test_unload_model(self, client):
        from vtsearch.models.registry import get_loaded_id, set_loaded_id

        set_loaded_id("fake")
        res = client.post("/api/models/registry/load", json={"model_id": None})
        assert res.status_code == 200
        assert get_loaded_id() is None

    def test_load_nonexistent(self, client):
        res = client.post("/api/models/registry/load", json={"model_id": "nope"})
        assert res.status_code == 404


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
        client.post("/api/models/registry/load", json={"model_id": model_id})

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
        client.post("/api/models/registry/load", json={"model_id": model_id})

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
        client.post("/api/models/registry/load", json={"model_id": model_id})

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

"""Tests for trainable model CRUD and label persistence."""

import json
import shutil

import pytest

from vtsearch.routes.trainable_models import TRAINABLE_MODELS_DIR


@pytest.fixture(autouse=True)
def clean_trainable_models_dir():
    """Remove the trainable models directory before and after each test."""
    if TRAINABLE_MODELS_DIR.is_dir():
        shutil.rmtree(TRAINABLE_MODELS_DIR)
    yield
    if TRAINABLE_MODELS_DIR.is_dir():
        shutil.rmtree(TRAINABLE_MODELS_DIR)


class TestCreateTrainableModel:
    def test_create_success(self, client):
        res = client.post(
            "/api/trainable-models",
            json={"name": "Dog Barks", "text_query": "sounds of dogs barking"},
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
            json={"name": "Test"},
        )
        assert res.status_code == 400
        assert "text_query" in res.get_json()["error"]

    def test_create_duplicate(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Dog Barks", "text_query": "dogs"},
        )
        res = client.post(
            "/api/trainable-models",
            json={"name": "Dog Barks", "text_query": "dogs again"},
        )
        assert res.status_code == 409

    def test_file_created_on_disk(self, client):
        client.post(
            "/api/trainable-models",
            json={"name": "Test Model", "text_query": "test"},
        )
        files = list(TRAINABLE_MODELS_DIR.glob("*.json"))
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
            json={"name": "Model A", "text_query": "a"},
        )
        client.post(
            "/api/trainable-models",
            json={"name": "Model B", "text_query": "b"},
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
            json={"name": "My Model", "text_query": "test query"},
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
            json={"name": "To Delete", "text_query": "test"},
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
            json={"name": "Old Name", "text_query": "test"},
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
            json={"name": "Test", "text_query": "test"},
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
            json={"name": "Original", "text_query": "test"},
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
            json={"name": "Model A", "text_query": "a"},
        )
        client.post(
            "/api/trainable-models",
            json={"name": "Model B", "text_query": "b"},
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
            json={"name": "Labeler", "text_query": "test"},
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
            json={"name": "Voted Model", "text_query": "test"},
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

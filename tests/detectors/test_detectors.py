"""Tests for detector CRUD and label persistence."""

import json
import shutil

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


class TestCreateDetector:
    def test_create_success(self, client):
        res = client.post(
            "/api/detectors",
            json={"name": "Dog Barks", "media_type": "audio", "text_query": "sounds of dogs barking"},
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["success"] is True
        assert data["name"] == "Dog Barks"
        assert data["text_query"] == "sounds of dogs barking"
        assert data["num_labels"] == 0

    def test_create_missing_name(self, client):
        # ``name`` is required by the schema → 422 with the standard
        # flask-smorest envelope (``errors`` per-field).
        res = client.post(
            "/api/detectors",
            json={"text_query": "sounds"},
        )
        assert res.status_code == 422
        assert "name" in res.get_json()["errors"]["json"]

    def test_create_missing_text_query(self, client):
        # ``name`` + ``media_type`` pass the schema; the
        # "at-least-one-example" check runs in the handler → 400 with
        # the standard error envelope (``message``).
        res = client.post(
            "/api/detectors",
            json={"name": "Test", "media_type": "audio"},
        )
        assert res.status_code == 400
        assert "text_query" in res.get_json()["message"]

    def test_create_missing_media_type(self, client):
        # ``media_type`` is required by the schema → 422.
        res = client.post(
            "/api/detectors",
            json={"name": "Test", "text_query": "sounds"},
        )
        assert res.status_code == 422
        assert "media_type" in res.get_json()["errors"]["json"]

    def test_create_rejects_any_media_type(self, client):
        # ``media_type="any"`` passes schema length check; rejection
        # happens in the handler → 400 with ``message``.
        res = client.post(
            "/api/detectors",
            json={"name": "Test", "media_type": "any", "text_query": "sounds"},
        )
        assert res.status_code == 400
        assert "media_type" in res.get_json()["message"]

    def test_create_duplicate(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Dog Barks", "media_type": "audio", "text_query": "dogs"},
        )
        res = client.post(
            "/api/detectors",
            json={"name": "Dog Barks", "media_type": "audio", "text_query": "dogs again"},
        )
        assert res.status_code == 409

    def test_file_created_on_disk(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Test Model", "media_type": "audio", "text_query": "test"},
        )
        files = list(get_detectors_dir().glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["name"] == "Test Model"
        assert data["text_query"] == "test"
        assert data["labelset"] == {"labels": []}


class TestListDetectors:
    def test_empty_list(self, client):
        res = client.get("/api/detectors")
        assert res.status_code == 200
        data = res.get_json()
        assert data["detectors"] == []

    def test_list_after_create(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Model A", "media_type": "audio", "text_query": "a"},
        )
        client.post(
            "/api/detectors",
            json={"name": "Model B", "media_type": "audio", "text_query": "b"},
        )
        res = client.get("/api/detectors")
        data = res.get_json()
        names = [m["name"] for m in data["detectors"]]
        assert "Model A" in names
        assert "Model B" in names


class TestGetDetector:
    def test_get_existing(self, client):
        client.post(
            "/api/detectors",
            json={"name": "My Model", "media_type": "audio", "text_query": "test query"},
        )
        res = client.get("/api/detectors/My%20Model")
        assert res.status_code == 200
        data = res.get_json()
        assert data["name"] == "My Model"
        assert data["text_query"] == "test query"
        assert "labelset" in data

    def test_get_nonexistent(self, client):
        res = client.get("/api/detectors/nonexistent")
        assert res.status_code == 404


class TestDeleteDetector:
    def test_delete_existing(self, client):
        client.post(
            "/api/detectors",
            json={"name": "To Delete", "media_type": "audio", "text_query": "test"},
        )
        res = client.delete("/api/detectors/To%20Delete")
        assert res.status_code == 200
        assert res.get_json()["success"] is True

        # Verify it's gone
        res = client.get("/api/detectors/To%20Delete")
        assert res.status_code == 404

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/detectors/nonexistent")
        assert res.status_code == 404


class TestRenameDetector:
    def test_rename_success(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Old Name", "media_type": "audio", "text_query": "test"},
        )
        res = client.put(
            "/api/detectors/Old%20Name/rename",
            json={"new_name": "New Name"},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["new_name"] == "New Name"

        # Old name should be gone
        res = client.get("/api/detectors/Old%20Name")
        assert res.status_code == 404

        # New name should exist
        res = client.get("/api/detectors/New%20Name")
        assert res.status_code == 200
        assert res.get_json()["name"] == "New Name"

    def test_rename_nonexistent(self, client):
        res = client.put(
            "/api/detectors/nonexistent/rename",
            json={"new_name": "Foo"},
        )
        assert res.status_code == 404

    def test_rename_missing_new_name(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Test", "media_type": "audio", "text_query": "test"},
        )
        # ``new_name`` is required by the schema → 422.
        res = client.put(
            "/api/detectors/Test/rename",
            json={},
        )
        assert res.status_code == 422

    def test_rename_updates_model_registry(self, client):
        """Renaming a detector should update registry references."""
        from vtscore.detectors.registry import find_by_name, get_detector

        # Create a detector and register it in the model registry
        client.post(
            "/api/detectors",
            json={"name": "Original", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "Original", "media_type": "audio", "text_query": "test"},
        )
        assert res.status_code == 201
        detector_id = res.get_json()["detector"]["id"]

        # Rename the detector directly (not through the registry endpoint)
        res = client.put(
            "/api/detectors/Original/rename",
            json={"new_name": "Renamed"},
        )
        assert res.status_code == 200

        # Registry entry should now reference the new name
        entry = get_detector(detector_id)
        assert entry is not None
        assert entry["name"] == "Renamed"

        # Look up by old name should fail
        assert find_by_name("Original") is None

        # Look up by new name should succeed
        assert find_by_name("Renamed") is not None

    def test_rename_conflict(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Model A", "media_type": "audio", "text_query": "a"},
        )
        client.post(
            "/api/detectors",
            json={"name": "Model B", "media_type": "audio", "text_query": "b"},
        )
        res = client.put(
            "/api/detectors/Model%20A/rename",
            json={"new_name": "Model B"},
        )
        assert res.status_code == 409


class TestRenameLabelsetSourceCleanup:
    """Renaming a detector with a {detector_name} labelset source should
    surface the orphaned file path so the user can move it."""

    def _create_registered_detector(self, client):
        client.post(
            "/api/detectors",
            json={"name": "Old Name", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "Old Name", "media_type": "audio", "text_query": "test"},
        )
        assert res.status_code == 201
        return res.get_json()["detector"]["id"]

    def _attach_labelset_source(self, detector_id, template):
        from vtscore.state.core import DetectorContext, register_detector_context

        ctx = DetectorContext(detector_id, name="Old Name", media_type="audio")
        ctx.labelset_source = {
            "source_name": "server_json_file",
            "field_values": {"filepath": template},
        }
        register_detector_context(ctx)
        return ctx

    def test_rename_returns_pending_move_when_old_file_exists(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        template = str(tmp_path / "{detector_name}.labels.json")
        self._attach_labelset_source(detector_id, template)

        # Pretend a previous sync wrote the old labelset file.
        old_file = tmp_path / "Old Name.labels.json"
        old_file.write_text('{"labels": []}')

        res = client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )
        assert res.status_code == 200
        body = res.get_json()
        assert body["ok"] is True
        assert body["name"] == "New Name"
        pending = body["pending_labelset_move"]
        assert pending is not None
        assert pending["old_path"].endswith("Old Name.labels.json")
        assert pending["new_path"].endswith("New Name.labels.json")

    def test_rename_updates_ctx_name_for_future_syncs(self, client, tmp_path):
        from vtscore.state.core import get_detector_context

        detector_id = self._create_registered_detector(client)
        template = str(tmp_path / "{detector_name}.labels.json")
        self._attach_labelset_source(detector_id, template)

        client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )

        ctx = get_detector_context(detector_id)
        assert ctx is not None
        assert ctx.name == "New Name"

    def test_rename_no_pending_when_old_file_missing(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        template = str(tmp_path / "{detector_name}.labels.json")
        self._attach_labelset_source(detector_id, template)

        # No old file on disk.
        res = client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )
        assert res.status_code == 200
        assert res.get_json()["pending_labelset_move"] is None

    def test_rename_no_pending_when_template_has_no_name_var(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        # Template without {detector_name} — old and new resolve to same path.
        template = str(tmp_path / "shared.labels.json")
        self._attach_labelset_source(detector_id, template)

        (tmp_path / "shared.labels.json").write_text('{"labels": []}')

        res = client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )
        assert res.status_code == 200
        assert res.get_json()["pending_labelset_move"] is None

    def test_rename_no_pending_when_destination_exists(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        template = str(tmp_path / "{detector_name}.labels.json")
        self._attach_labelset_source(detector_id, template)

        (tmp_path / "Old Name.labels.json").write_text('{"labels": []}')
        (tmp_path / "New Name.labels.json").write_text('{"labels": []}')

        res = client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )
        assert res.status_code == 200
        # We refuse to propose a move that would clobber an existing file.
        assert res.get_json()["pending_labelset_move"] is None

    def test_rename_no_pending_without_labelset_source(self, client):
        detector_id = self._create_registered_detector(client)
        # No source attached (no _attach_labelset_source call).
        res = client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )
        assert res.status_code == 200
        assert res.get_json()["pending_labelset_move"] is None

    def test_move_endpoint_moves_file(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        template = str(tmp_path / "{detector_name}.labels.json")
        self._attach_labelset_source(detector_id, template)

        old_file = tmp_path / "Old Name.labels.json"
        old_file.write_text('{"labels": [{"md5": "abc", "label": "good"}]}')
        new_file = tmp_path / "New Name.labels.json"

        res = client.put(
            f"/api/detectors/registry/{detector_id}/rename",
            json={"name": "New Name"},
        )
        pending = res.get_json()["pending_labelset_move"]

        res = client.post(
            f"/api/detectors/registry/{detector_id}/labelset-source/move-file",
            json={"old_path": pending["old_path"], "new_path": pending["new_path"]},
        )
        assert res.status_code == 200
        body = res.get_json()
        assert body["ok"] is True
        assert body["moved"] is True

        assert not old_file.exists()
        assert new_file.exists()
        # Content preserved.
        assert "abc" in new_file.read_text()

    def test_move_endpoint_idempotent_when_old_missing(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        self._attach_labelset_source(detector_id, str(tmp_path / "{detector_name}.labels.json"))

        res = client.post(
            f"/api/detectors/registry/{detector_id}/labelset-source/move-file",
            json={
                "old_path": str(tmp_path / "nope.json"),
                "new_path": str(tmp_path / "dest.json"),
            },
        )
        assert res.status_code == 200
        body = res.get_json()
        assert body["ok"] is True
        assert body["moved"] is False

    def test_move_endpoint_rejects_existing_destination(self, client, tmp_path):
        detector_id = self._create_registered_detector(client)
        self._attach_labelset_source(detector_id, str(tmp_path / "{detector_name}.labels.json"))

        old_file = tmp_path / "src.json"
        old_file.write_text("{}")
        new_file = tmp_path / "dst.json"
        new_file.write_text("{}")

        res = client.post(
            f"/api/detectors/registry/{detector_id}/labelset-source/move-file",
            json={"old_path": str(old_file), "new_path": str(new_file)},
        )
        assert res.status_code == 409
        # Neither file touched.
        assert old_file.exists()
        assert new_file.exists()

    def test_move_endpoint_404_for_unknown_detector(self, client, tmp_path):
        res = client.post(
            "/api/detectors/registry/nonexistent/labelset-source/move-file",
            json={
                "old_path": str(tmp_path / "x"),
                "new_path": str(tmp_path / "y"),
            },
        )
        assert res.status_code == 404


class TestSaveLabels:
    def test_save_labels_empty(self, client):
        """Save labels when there are no votes — should produce empty labelset."""
        client.post(
            "/api/detectors",
            json={"name": "Labeler", "media_type": "audio", "text_query": "test"},
        )
        res = client.post("/api/detectors/Labeler/labels")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["num_labels"] == 0

    def test_save_labels_with_votes(self, client):
        """Save labels after casting votes — labelset should contain the voted medias."""
        from vtsearch.state import medias

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
            "/api/detectors",
            json={"name": "Voted Model", "media_type": "audio", "text_query": "test"},
        )
        res = client.post("/api/detectors/Voted%20Model/labels")
        assert res.status_code == 200
        data = res.get_json()
        assert data["num_labels"] == 2

        # Verify the labels are persisted on disk
        model_res = client.get("/api/detectors/Voted%20Model")
        model_data = model_res.get_json()
        labels = model_data["labelset"]["labels"]
        assert len(labels) == 2
        label_values = {lbl["label"] for lbl in labels}
        assert "good" in label_values
        assert "bad" in label_values

    def test_save_labels_nonexistent_model(self, client):
        res = client.post("/api/detectors/nonexistent/labels")
        assert res.status_code == 404

    def test_save_labels_captures_active_clipper_into_input_spec(self, client):
        """When the active dataset has a clipper, save_detector_labels stamps it onto input_spec."""
        from vtscore.detectors.store import _detector_path, _read_detector
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded for this test")

        # Stamp every media's origin with the same clipper config — the
        # extractor scans the dict order until it finds a clipped media,
        # so doing it everywhere matches what the loader does in real life
        # and removes any iteration-order flakiness.
        originals: dict[int, object] = {}
        try:
            for mid, media in medias.items():
                originals[mid] = media.get("origin")
                media["origin"] = {
                    "importer": "test",
                    "params": {
                        "clipper": "sound_tiling",
                        "clipper_duration": "2.0",
                    },
                }

            client.post(
                "/api/detectors",
                json={"name": "ClipperCapture", "media_type": "audio", "text_query": "test"},
            )
            res = client.post("/api/detectors/ClipperCapture/labels")
            assert res.status_code == 200

            # Read straight off disk so we don't depend on the GET route's
            # serialisation behaviour for an unknown-include field.
            disk = _read_detector(_detector_path("ClipperCapture"))
            assert disk is not None
            assert disk["input_spec"] == {
                "clipper": "sound_tiling",
                "clipper_params": {"duration": "2.0"},
            }
        finally:
            for mid, origin in originals.items():
                if mid in medias:
                    medias[mid]["origin"] = origin

    def test_save_labels_drops_input_spec_when_dataset_has_no_clipper(self, client):
        """Re-saving labels against an unclipped dataset clears any stale input_spec."""
        from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

        client.post(
            "/api/detectors",
            json={"name": "DropSpec", "media_type": "audio", "text_query": "test"},
        )

        # Seed an old input_spec directly on disk (as if a previous save
        # had captured one from a clipped dataset).
        det_path = _detector_path("DropSpec")
        data = _read_detector(det_path) or {}
        data["input_spec"] = {"clipper": "sound_tiling"}
        _write_detector(det_path, data)

        # Re-save from a clean dataset — fixture medias have no clipper.
        res = client.post("/api/detectors/DropSpec/labels")
        assert res.status_code == 200

        updated = _read_detector(det_path)
        assert updated is not None
        assert "input_spec" not in updated

    def test_save_labels_does_not_expand_dupes(self, client):
        """Saving labels for a dupe-set representative should NOT expand members.

        Regression test: previously, a vote on a dupe-set representative
        with N members produced N label entries, inflating the stored
        label count.  Trainable model persistence should store one entry
        per vote, not one per duplicate.
        """
        import copy

        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded for this test")

        first_id = next(iter(medias))
        original = copy.deepcopy(medias[first_id])

        # Turn the first media into a dupe-set representative with 5 members
        medias[first_id]["origin"] = {
            "importer": "dupe_set",
            "params": {"name": original.get("filename", "a.wav")},
            "members": [
                {
                    "origin": {"importer": "test", "params": {}},
                    "origin_name": f"dup_{i}.wav",
                    "filename": f"dup_{i}.wav",
                    "category": "c",
                }
                for i in range(5)
            ],
        }
        try:
            client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
            client.post("/api/detectors", json={"name": "DupeTest", "media_type": "audio", "text_query": "test"})

            res = client.post("/api/detectors/DupeTest/labels")
            assert res.status_code == 200
            data = res.get_json()
            # Should be 1 label (the vote), NOT 5 (the dupe members)
            assert data["num_labels"] == 1

            model_res = client.get("/api/detectors/DupeTest")
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
        from vtsearch.state import (
            good_votes,
            bad_votes,
            medias,
        )

        ids = list(medias.keys())
        if len(ids) < 4:
            pytest.skip("Need at least 4 medias")

        # Create two detectors
        client.post("/api/detectors", json={"name": "Model A", "media_type": "audio", "text_query": "a"})
        client.post("/api/detectors", json={"name": "Model B", "media_type": "audio", "text_query": "b"})

        # Simulate labeling with Model A: vote on ids[0] and ids[1]
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[1]}/vote", json={"vote": "bad"})
        client.post("/api/detectors/Model%20A/labels")  # save 2 labels

        # Now clear votes (as the Label button should do) and import Model B's labels
        client.post("/api/votes/clear")
        assert len(good_votes) == 0
        assert len(bad_votes) == 0

        # Model B has no labels, so import is a no-op — votes should remain empty
        model_b = client.get("/api/detectors/Model%20B").get_json()
        assert len(model_b["labelset"]["labels"]) == 0

        client.post("/api/labels/import", json={"labels": model_b["labelset"]["labels"]})
        assert len(good_votes) == 0, "Model A's votes should not leak into Model B's session"
        assert len(bad_votes) == 0

    def test_import_after_clear_only_has_model_labels(self, client):
        """After clearing + importing, only the target model's labels are active."""
        from vtsearch.state import (
            good_votes,
            bad_votes,
            medias,
        )

        ids = list(medias.keys())
        if len(ids) < 4:
            pytest.skip("Need at least 4 medias")

        # Create model and label 2 items
        client.post("/api/detectors", json={"name": "Target", "media_type": "audio", "text_query": "t"})
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[1]}/vote", json={"vote": "bad"})
        client.post("/api/detectors/Target/labels")

        # Add extra votes that DON'T belong to the model (simulating stale state)
        client.post(f"/api/medias/{ids[2]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[3]}/vote", json={"vote": "bad"})
        assert len(good_votes) == 2  # ids[0] + ids[2]
        assert len(bad_votes) == 2  # ids[1] + ids[3]

        # Clear votes, then import only Target's labels
        client.post("/api/votes/clear")
        target_data = client.get("/api/detectors/Target").get_json()
        client.post("/api/labels/import", json={"labels": target_data["labelset"]["labels"]})

        # Should only have the 2 labels from Target, not the 4 from before
        assert len(good_votes) + len(bad_votes) == 2
        assert ids[2] not in good_votes, "Stale vote should be gone after clear+import"
        assert ids[3] not in bad_votes, "Stale vote should be gone after clear+import"


class TestDeleteRegisteredModel:
    """Tests for DELETE /api/detectors/registry/<detector_id>."""

    def test_delete_registered_model(self, client):
        """Deleting a registered model removes it from the registry."""
        from vtscore.detectors.registry import get_detector

        res = client.post(
            "/api/detectors/registry",
            json={"name": "DelMe", "media_type": "audio", "text_query": "test"},
        )
        assert res.status_code == 201
        detector_id = res.get_json()["detector"]["id"]

        res = client.delete(f"/api/detectors/registry/{detector_id}")
        assert res.status_code == 200
        assert res.get_json()["ok"] is True
        assert get_detector(detector_id) is None

    def test_delete_nonexistent(self, client):
        res = client.delete("/api/detectors/registry/nonexistent_id")
        assert res.status_code == 404

    def test_delete_loaded_model(self, client):
        """Deleting a loaded model should also unload it."""
        from vtscore.detectors.registry import get_detector, is_detector_loaded

        client.post(
            "/api/detectors",
            json={"name": "LoadDel", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "LoadDel", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)
        assert is_detector_loaded(detector_id)

        res = client.delete(f"/api/detectors/registry/{detector_id}")
        assert res.status_code == 200
        assert get_detector(detector_id) is None
        assert not is_detector_loaded(detector_id)

    def test_delete_removes_autorun_flag(self, client):
        """Deleting a model that is flagged for autorun clears it from settings."""
        from vtscore.detectors.registry import get_detector
        from vtsearch.settings import add_autorun_detector, get_autorun_detectors

        res = client.post(
            "/api/detectors/registry",
            json={"name": "DetDel", "media_type": "audio"},
        )
        detector_id = res.get_json()["detector"]["id"]
        add_autorun_detector("DetDel")

        res = client.delete(f"/api/detectors/registry/{detector_id}")
        assert res.status_code == 200
        assert "DetDel" not in get_autorun_detectors()
        assert get_detector(detector_id) is None


class TestLoadModelEndpoint:
    """Tests for POST /api/detectors/registry/load."""

    def test_load_model(self, client):
        from vtscore.detectors.registry import is_detector_loaded

        res = client.post(
            "/api/detectors/registry",
            json={"name": "M", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]

        res = _load_detector_and_wait(client, detector_id)
        assert res.status_code == 200
        assert is_detector_loaded(detector_id)

    def test_unload_model(self, client):
        from vtscore.detectors.registry import add_loaded_detector_id, is_detector_loaded

        add_loaded_detector_id("fake")
        assert is_detector_loaded("fake")
        res = client.post("/api/detectors/registry/load", json={"detector_id": None})
        assert res.status_code == 200

    def test_load_nonexistent(self, client):
        res = client.post("/api/detectors/registry/load", json={"detector_id": "nope"})
        assert res.status_code == 404

    def test_load_clears_previous_labels(self, client):
        """Loading model B must not carry over labels from model A."""
        from vtsearch.state import (
            bad_votes,
            good_votes,
            medias,
        )

        if not medias:
            pytest.skip("No medias loaded")

        ids = list(medias.keys())

        # Create two detectors + registry entries.
        for name in ("ModelA", "ModelB"):
            client.post(
                "/api/detectors",
                json={"name": name, "media_type": "audio", "text_query": "test"},
            )
        res_a = client.post(
            "/api/detectors/registry",
            json={"name": "ModelA", "media_type": "audio", "text_query": "test"},
        )
        mid_a = res_a.get_json()["detector"]["id"]
        res_b = client.post(
            "/api/detectors/registry",
            json={"name": "ModelB", "media_type": "audio", "text_query": "test"},
        )
        mid_b = res_b.get_json()["detector"]["id"]

        # Load model A and cast some votes.
        _load_detector_and_wait(client, mid_a)
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{ids[1]}/vote", json={"vote": "bad"})
        assert ids[0] in good_votes
        assert ids[1] in bad_votes

        # Now load model B — votes from A must be gone.
        _load_detector_and_wait(client, mid_b)
        assert ids[0] not in good_votes, "good vote from model A leaked into model B"
        assert ids[1] not in bad_votes, "bad vote from model A leaked into model B"

    def test_load_restores_saved_labels(self, client):
        """Loading a model that has a saved labelset should restore its labels."""
        from vtsearch.state import (
            good_votes,
            medias,
        )

        if not medias:
            pytest.skip("No medias loaded")

        ids = list(medias.keys())

        # Create model, load it, vote, then save labels.
        client.post(
            "/api/detectors",
            json={"name": "Persist", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "Persist", "media_type": "audio", "text_query": "test"},
        )
        mid = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, mid)
        client.post(f"/api/medias/{ids[0]}/vote", json={"vote": "good"})
        # Labels auto-sync on vote, so the detector file now has 1 label.

        # Unload to clear votes, then reload — label should be restored.
        client.post("/api/detectors/registry/load", json={"detector_id": None})
        assert ids[0] not in good_votes

        res = _load_detector_and_wait(client, mid)
        assert res.status_code == 200
        assert ids[0] in good_votes, "saved label was not restored on model load"

    def test_dataset_switch_clears_cross_dataset_cids(self, client):
        """Switching the active dataset must rehydrate the loaded detector's
        cid-keyed vote dicts from the on-disk labelset against the new dataset's
        medias.  Without this, ids voted in dataset A leak into dataset B's
        id-space and unrelated B-medias whose ids happen to coincide with
        A's voted ids appear as voted in B's labeling UI.
        """
        import hashlib

        import numpy as np

        from vtscore.detectors.dataset_sync import ensure_votes_match_active_dataset
        from vtsearch.state import (
            DatasetContext,
            bad_votes,
            good_votes,
            medias,
            register_context,
            set_thread_dataset_context,
        )

        if not medias:
            pytest.skip("No medias loaded")

        a_ids = list(medias.keys())
        if len(a_ids) < 2:
            pytest.skip("Need at least 2 medias")
        a_good = a_ids[0]
        a_bad = a_ids[1]

        # Create + load a detector while dataset A is active.
        client.post(
            "/api/detectors",
            json={"name": "CrossDS", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "CrossDS", "media_type": "audio", "text_query": "test"},
        )
        mid = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, mid)

        # Vote in dataset A.  Persists to good_votes/bad_votes and to the
        # detector's on-disk labelset via sync_labels_to_loaded_detector.
        client.post(f"/api/medias/{a_good}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{a_bad}/vote", json={"vote": "bad"})
        assert a_good in good_votes
        assert a_bad in bad_votes

        # Build dataset B with DIFFERENT media (different md5/origin) reusing
        # the same cids, exactly the situation that produced the bug: the
        # voted A ids would otherwise show up as votes in B's id-space.
        ctx_b = DatasetContext("ds_b_for_switch_test")
        for cid in (a_good, a_bad):
            ctx_b.medias[cid] = {
                "id": cid,
                "type": "audio",
                "embedder": "clap",
                "md5": hashlib.md5(f"ds_b_{cid}".encode()).hexdigest(),
                "embedding": np.zeros(512, dtype=np.float32),
                "media_bytes": b"fake-b",
                "filename": f"ds_b_{cid}.wav",
                "category": "test",
                "origin": {"importer": "test_b", "params": {"id": cid}},
                "origin_name": f"ds_b_{cid}.wav",
            }
        register_context(ctx_b)
        set_thread_dataset_context(ctx_b)

        # Simulate the before_request hook firing for a request whose active
        # dataset is now B.
        ensure_votes_match_active_dataset()

        assert a_good not in good_votes, "good cid from dataset A leaked into dataset B's id-space"
        assert a_bad not in bad_votes, "bad cid from dataset A leaked into dataset B's id-space"


class TestValidatedVoteSnapshot:
    """``validated_vote_snapshot`` must atomically pair medias + votes (H14).

    The H14 audit finding identified that ``good_votes`` / ``bad_votes`` are
    detector-scoped while ``medias`` is dataset-scoped, and composing them
    without an atomic snapshot allows a concurrent rehydrate on the same
    detector against a different dataset to slip in between the
    ``before_request`` rehydrate and the route body, leaking cross-dataset
    cids into the export and corrupting on-disk writes.

    The helper :func:`vtscore.detectors.dataset_sync.validated_vote_snapshot`
    captures both contexts under a single ``_state_lock`` acquisition and
    refuses to compose (``safe=False``) when ``votes_dataset_id`` doesn't
    match the active dataset.
    """

    def _setup_detector_with_votes(self, client):
        """Create a detector, load it, cast one good + one bad vote.

        Returns ``(detector_id, good_cid, bad_cid)``.
        """
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")
        ids = list(medias.keys())
        if len(ids) < 2:
            pytest.skip("Need at least 2 medias")
        good_cid, bad_cid = ids[0], ids[1]

        client.post(
            "/api/detectors",
            json={"name": "SnapshotTest", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "SnapshotTest", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)

        client.post(f"/api/medias/{good_cid}/vote", json={"vote": "good"})
        client.post(f"/api/medias/{bad_cid}/vote", json={"vote": "bad"})
        return detector_id, good_cid, bad_cid

    def test_snapshot_safe_when_aligned(self, client):
        """Happy path: votes_dataset_id matches active dataset → safe=True with full votes."""
        from vtscore.detectors.dataset_sync import validated_vote_snapshot

        _, good_cid, bad_cid = self._setup_detector_with_votes(client)

        snap = validated_vote_snapshot()
        assert snap.safe is True
        assert good_cid in snap.good_votes
        assert bad_cid in snap.bad_votes
        assert good_cid in snap.medias
        assert bad_cid in snap.medias

    def test_snapshot_unsafe_on_dataset_mismatch(self, client):
        """votes_dataset_id != active dataset_id → safe=False with empty vote dicts.

        Simulates the H14 race: another thread re-keyed the detector against
        a different dataset between ``ensure_votes_match_active_dataset()``
        and the snapshot copy.  Patches the rehydrate to a no-op so the
        forced mismatch survives the snapshot's own rehydrate call.
        """
        from vtscore.detectors import dataset_sync as _ds_sync
        from vtscore.detectors.dataset_sync import validated_vote_snapshot
        from vtscore.state.core import get_active_detector_context

        self._setup_detector_with_votes(client)

        det_ctx = get_active_detector_context()
        original = _ds_sync.ensure_votes_match_active_dataset
        _ds_sync.ensure_votes_match_active_dataset = lambda: None
        try:
            det_ctx.votes_dataset_id = "some_other_dataset_id"
            snap = validated_vote_snapshot()
        finally:
            _ds_sync.ensure_votes_match_active_dataset = original

        assert snap.safe is False
        assert snap.good_votes == {}
        assert snap.bad_votes == {}
        assert snap.vote_region_boxes == {}
        # ``medias`` is always populated from the live active dataset.
        assert snap.medias, "medias should still be populated on safe=False"

    def test_export_returns_empty_on_safe_false(self, client):
        """/api/labels/export must not leak cross-dataset cids when snapshot is unsafe."""
        from vtscore.state.core import get_active_detector_context

        self._setup_detector_with_votes(client)

        # Sanity: with aligned state we get the labels back.
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        assert len(resp.get_json()["labels"]) >= 2

        # Now corrupt votes_dataset_id and re-export.  The route's
        # ``ensure_votes_match_active_dataset`` will try to rehydrate but the
        # detector file has been written to disk, so it'll match the active
        # dataset again — verify by going around the rehydrate's mtime cache.
        # Simplest: directly corrupt the detector's vote dicts to simulate a
        # post-rehydrate mismatch (e.g. another thread re-flipped them).
        det_ctx = get_active_detector_context()
        # Force a state where votes_dataset_id can't match (the race outcome).
        # We monkey-patch ensure_votes_match_active_dataset to be a no-op so
        # the corruption survives the before_request hook.
        from vtscore.detectors import dataset_sync as _ds_sync

        original = _ds_sync.ensure_votes_match_active_dataset
        _ds_sync.ensure_votes_match_active_dataset = lambda: None
        try:
            det_ctx.votes_dataset_id = "stale_dataset_id"
            resp = client.get("/api/labels/export")
            assert resp.status_code == 200
            # safe=False degrades the vote dicts to empty, so no labels
            # are composed — no cross-dataset cid leakage.
            assert resp.get_json()["labels"] == []
        finally:
            _ds_sync.ensure_votes_match_active_dataset = original

    def test_sync_labels_to_disk_skips_on_safe_false(self, client):
        """``sync_labels_to_loaded_detector`` must not erase on-disk labels under race."""
        from vtscore.detectors import dataset_sync as _ds_sync
        from vtscore.detectors.label_sync import sync_labels_to_loaded_detector
        from vtscore.detectors.store import _detector_path, _read_detector
        from vtscore.state.core import get_active_detector_context

        _, good_cid, bad_cid = self._setup_detector_with_votes(client)

        # After the votes were cast, the detector JSON on disk has 2 entries.
        # Find the on-disk path and confirm.
        det_ctx = get_active_detector_context()
        path = _detector_path(det_ctx.name)
        data = _read_detector(path)
        assert data is not None
        labels_before = data.get("labelset", {}).get("labels", [])
        assert len(labels_before) >= 2

        # Simulate the race: votes_dataset_id mismatches active dataset, and
        # the rehydrate hook is bypassed (so the mismatch survives).  Without
        # the safe=False guard, ``_merge_labelsets_across_datasets`` would
        # drop the active dataset's existing entries and replace them with an
        # empty composition — erasing labels from disk.
        original = _ds_sync.ensure_votes_match_active_dataset
        _ds_sync.ensure_votes_match_active_dataset = lambda: None
        try:
            det_ctx.votes_dataset_id = "stale_dataset_id"
            sync_labels_to_loaded_detector()
        finally:
            _ds_sync.ensure_votes_match_active_dataset = original

        # The on-disk labelset must be unchanged.
        data_after = _read_detector(path)
        labels_after = data_after.get("labelset", {}).get("labels", [])
        assert len(labels_after) == len(labels_before), (
            "sync should have bailed on safe=False, not erased on-disk labels"
        )


class TestVoteSyncsToLoadedModel:
    """Voting while a detector is loaded should auto-update the model's labelset."""

    def test_vote_updates_model_labels(self, client):
        """Casting a vote with a loaded model should persist labels and update registry stats."""
        from vtscore.detectors.registry import get_detector
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        # Create and register a detector
        client.post(
            "/api/detectors",
            json={"name": "AutoSync", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "AutoSync", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]

        # Load the model
        _load_detector_and_wait(client, detector_id)

        # Cast a vote
        first_id = next(iter(medias))
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})

        # Check that the model's labelset was updated
        model_data = client.get("/api/detectors/AutoSync").get_json()
        labels = model_data["labelset"]["labels"]
        assert len(labels) == 1
        assert labels[0]["label"] == "good"

        # Check that the registry entry was updated
        entry = get_detector(detector_id)
        assert entry is not None
        assert entry["num_training"] == 1
        assert entry.get("last_trained_at") is not None

    def test_vote_toggle_off_updates_model(self, client):
        """Toggling a vote off should update the model labelset to reflect removal."""
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        client.post(
            "/api/detectors",
            json={"name": "ToggleSync", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "ToggleSync", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)

        first_id = next(iter(medias))
        # Vote good
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
        model_data = client.get("/api/detectors/ToggleSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 1

        # Toggle off (vote good again)
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
        model_data = client.get("/api/detectors/ToggleSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 0

    def test_no_sync_without_loaded_model(self, client):
        """Voting with no loaded model should not create/update any model files."""
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        client.post(
            "/api/detectors",
            json={"name": "NoSync", "media_type": "audio", "text_query": "test"},
        )

        first_id = next(iter(medias))
        client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})

        # Model should still have empty labelset
        model_data = client.get("/api/detectors/NoSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 0

    def test_label_import_syncs_to_loaded_model(self, client):
        """Importing labels with a loaded model should persist to the model."""
        from vtscore.detectors.registry import get_detector
        from vtsearch.state import medias

        if not medias:
            pytest.skip("No medias loaded")

        client.post(
            "/api/detectors",
            json={"name": "ImportSync", "media_type": "audio", "text_query": "test"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={"name": "ImportSync", "media_type": "audio", "text_query": "test"},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)

        # Get an MD5 from the first media
        first_id = next(iter(medias))
        media = medias[first_id]
        md5 = media.get("md5", "")

        # Import a label
        client.post("/api/labels/import", json={"labels": [{"md5": md5, "label": "good"}]})

        # Model should have the imported label
        model_data = client.get("/api/detectors/ImportSync").get_json()
        assert len(model_data["labelset"]["labels"]) == 1

        entry = get_detector(detector_id)
        assert entry is not None
        assert entry["num_training"] == 1


class TestSeedVotesFromExamples:
    """When loading a model with media examples, matching medias get auto-labeled Good."""

    @pytest.fixture(autouse=True)
    def _restore_medias(self):
        """Remove any media items inserted by seeding after each test."""
        from vtsearch.state import medias

        saved = dict(medias)
        yield
        # Remove items that were added, restore any that were modified
        medias.clear()
        medias.update(saved)

    def _create_example_file(self, media_bytes: bytes, filename: str = "ex.wav") -> str:
        """Write *media_bytes* into data/example_media/<filename> and return the filename."""
        from vtscore.config import DATA_DIR

        example_dir = DATA_DIR / "example_media"
        example_dir.mkdir(parents=True, exist_ok=True)
        dest = example_dir / filename
        dest.write_bytes(media_bytes)
        return filename

    # ---- POST /api/votes/seed-from-examples ----

    def test_seed_endpoint_adds_good_votes(self, client):
        """Media examples whose MD5 matches a loaded media should become good votes."""
        from vtsearch.state import (
            good_votes,
            medias,
        )

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
        from vtsearch.state import (
            good_votes,
            medias,
        )

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
        from vtsearch.state import medias

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
        from vtsearch.state import medias

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
            lbl
            for lbl in labels
            if isinstance(lbl.get("origin"), dict) and lbl["origin"].get("importer") == "example_media"
        ]
        assert len(example_labels) == 1
        assert example_labels[0]["label"] == "good"
        assert example_labels[0]["origin"]["params"]["filename"] == fname

    def test_new_example_usable_in_training(self, client):
        """Inserted examples should have embeddings usable by learned-sort."""
        from vtsearch.state import (
            good_votes,
            bad_votes,
            medias,
        )

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
        res = client.post("/api/learned-sort", json={"wait": True})
        assert res.status_code == 200
        data = res.get_json()
        assert "results" in data
        assert len(data["results"]) > 0

    # ---- Model load auto-seeding ----

    def test_load_model_seeds_from_media_examples(self, client):
        """Loading a model with media examples should auto-seed good votes."""
        from vtsearch.state import (
            good_votes,
            medias,
        )

        if not medias:
            pytest.skip("No medias loaded")

        first_id = next(iter(medias))
        media = medias[first_id]
        fname = self._create_example_file(media["media_bytes"], "autoload.wav")

        # Create model with a media example
        client.post(
            "/api/detectors",
            json={
                "name": "AutoSeed",
                "media_type": "audio",
                "examples": [{"type": "media", "value": fname}],
            },
        )
        # Register in model registry
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "AutoSeed",
                "media_type": "audio",
                "text_query": "",
                "media_example": fname,
            },
        )
        detector_id = res.get_json()["detector"]["id"]

        # Clear any prior votes
        client.post("/api/votes/clear")
        assert len(good_votes) == 0

        # Load model — should auto-seed
        res = _load_detector_and_wait(client, detector_id)
        assert res.status_code == 200
        assert first_id in good_votes, "example media should be seeded as good vote"

    def test_load_model_without_examples_seeds_nothing(self, client):
        """Loading a text-only model should seed 0 examples."""
        from vtsearch.state import good_votes

        client.post(
            "/api/detectors",
            json={"name": "TextOnly", "media_type": "audio", "text_query": "dogs"},
        )
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "TextOnly",
                "media_type": "audio",
                "text_query": "dogs",
            },
        )
        detector_id = res.get_json()["detector"]["id"]

        client.post("/api/votes/clear")
        res = _load_detector_and_wait(client, detector_id)
        assert res.status_code == 200
        assert len(good_votes) == 0

    def test_seeded_examples_enable_autopilot_skip(self, client):
        """If seeded examples meet the autopilot threshold, Good phase can be skipped.

        This tests the backend side: enough media examples seed enough
        good_votes that ``goodCount >= autopilot_top_greens``.
        """
        from vtsearch.state import (
            good_votes,
            medias,
        )

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
            "/api/detectors",
            json={"name": "SkipGood", "media_type": "audio", "examples": examples},
        )
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "SkipGood",
                "media_type": "audio",
                "text_query": "",
            },
        )
        detector_id = res.get_json()["detector"]["id"]

        client.post("/api/votes/clear")
        _load_detector_and_wait(client, detector_id)

        # With default autopilot_top_greens=3, 4 good votes is enough to skip Good phase
        assert len(good_votes) >= 4

    def test_load_model_seeds_novel_example(self, client):
        """Loading a model with a non-dataset example should embed and insert it."""
        from vtsearch.state import (
            good_votes,
            medias,
        )

        original_count = len(medias)
        fname = self._create_example_file(b"novel-load-bytes", "novel_load.wav")

        client.post(
            "/api/detectors",
            json={
                "name": "NovelSeed",
                "media_type": "audio",
                "examples": [{"type": "media", "value": fname}],
            },
        )
        res = client.post(
            "/api/detectors/registry",
            json={
                "name": "NovelSeed",
                "media_type": "audio",
                "text_query": "",
            },
        )
        detector_id = res.get_json()["detector"]["id"]

        client.post("/api/votes/clear")
        res = _load_detector_and_wait(client, detector_id)
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

        from vtscore.detectors.registry import register_detector, reset_for_tests
        from vtscore.detectors.store import _write_detector
        from vtsearch.settings import get_detectors_dir, set_detectors_dir
        from vtsearch.state import (
            good_votes,
            bad_votes,
            medias,
        )

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
            "importer": "server_folder",
            "params": {"path": str(label_folder), "media_type": "audio"},
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

        # --- Write detector ---
        original_dir = get_detectors_dir()
        set_detectors_dir(tmp_path)
        try:
            tm_name = "cross-dataset-load"
            _write_detector(
                tmp_path / f"{tm_name}.json",
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "audio",
                    "examples": [],
                    "labelset": {"labels": label_entries},
                },
            )

            entry = register_detector(
                name=tm_name,
                media_type="audio",
            )
            detector_id = entry["id"]

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
                "origin": {"importer": "server_folder", "params": {"path": "/other/place"}},
                "origin_name": "completely_different_name.wav",
            }
            medias[2] = {
                "id": 2,
                "type": "audio",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "md5": bad_md5,  # same content as bad_0.wav
                "filename": "another_file.wav",
                "origin": {"importer": "server_folder", "params": {"path": "/other/place"}},
                "origin_name": "another_file.wav",
            }

            try:
                res = _load_detector_and_wait(client, detector_id)
                assert res.status_code == 200
                assert 1 in good_votes, "good label should be applied to media 1"
                assert 2 in bad_votes, "bad label should be applied to media 2"
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_detectors_dir(original_dir)

    def test_load_model_does_not_silently_match_on_basename_collision(self, client, tmp_path):
        """Loading a detector trained on dataset B against dataset C must NOT
        silently apply labels to C's media when only the basename matches.

        Concretely: a labelset entry with origin+md5 that don't exist in the
        current dataset must NOT be applied to a media just because they
        share an ``origin_name``.  Different datasets routinely contain
        files with identical basenames (e.g. ``track1.wav``) but different
        underlying content; matching by basename alone produces silent
        mislabels.
        """
        import numpy as np

        from vtscore.detectors.registry import register_detector, reset_for_tests
        from vtscore.detectors.store import _write_detector
        from vtsearch.settings import get_detectors_dir, set_detectors_dir
        from vtsearch.state import (
            good_votes,
            medias,
        )

        reset_for_tests()

        label_entries = [
            {
                "md5": "nonexistent_hash",
                "label": "good",
                "origin": {"importer": "server_folder", "params": {"path": "/gone"}},
                "origin_name": "shared_name.wav",
                "filename": "shared_name.wav",
            },
        ]

        original_dir = get_detectors_dir()
        set_detectors_dir(tmp_path)
        try:
            tm_name = "name-fallback"
            _write_detector(
                tmp_path / f"{tm_name}.json",
                {
                    "name": tm_name,
                    "text_query": "",
                    "media_type": "audio",
                    "examples": [],
                    "labelset": {"labels": label_entries},
                },
            )

            entry = register_detector(
                name=tm_name,
                media_type="audio",
            )
            detector_id = entry["id"]

            saved = dict(medias)
            medias.clear()
            rng = np.random.default_rng(42)
            medias[1] = {
                "id": 1,
                "type": "audio",
                "embedding": rng.standard_normal(512).astype(np.float32),
                "md5": "totally_different_md5",
                "filename": "shared_name.wav",
                "origin": {"importer": "server_folder", "params": {"path": "/different"}},
                "origin_name": "shared_name.wav",
            }

            try:
                res = _load_detector_and_wait(client, detector_id)
                assert res.status_code == 200
                assert 1 not in good_votes, (
                    "label must NOT be applied: only the basename matches; the md5 "
                    "and origin disagree, so the underlying content is different"
                )
            finally:
                medias.clear()
                medias.update(saved)
        finally:
            set_detectors_dir(original_dir)


class TestRegisterModelFromLabelset:
    """Tests for POST /api/detectors/registry/from-labelset/<importer_name>.

    This endpoint creates a detector seeded with labels produced by
    a label importer in a single call: the importer runs, the trainable
    model JSON is written with the full labelset, and a registry entry is
    created.  The frontend then calls the load endpoint to resolve origins
    and train the MLP.

    Media type is inferred from the labels' origins — no ``media_type``
    request field is accepted.
    """

    @staticmethod
    def _audio_origin(path: str = "/tmp/audio_clips"):
        return {"importer": "server_folder", "params": {"path": path, "media_type": "audio"}}

    @staticmethod
    def _image_origin(path: str = "/tmp/image_clips"):
        return {"importer": "server_folder", "params": {"path": path, "media_type": "image"}}

    def test_creates_model_with_imported_labels(self, client, tmp_path):
        import app as app_module

        md5_good = app_module.medias[1]["md5"]
        md5_bad = app_module.medias[2]["md5"]
        origin = self._audio_origin()
        payload = json.dumps(
            {
                "labels": [
                    {"md5": md5_good, "label": "good", "origin": origin, "origin_name": "a.wav"},
                    {"md5": md5_bad, "label": "bad", "origin": origin, "origin_name": "b.wav"},
                ]
            }
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(payload)

        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "From LS", "filepath": str(labels_path)},
        )
        assert res.status_code == 201, res.get_json()
        data = res.get_json()
        assert data["ok"] is True
        assert data["applied"] == 2
        assert data["skipped"] == 0
        assert data["num_labels"] == 2
        # Media type inferred from the origin metadata.
        assert data["detector"]["media_type"] == "audio"
        assert data["detector"]["num_training"] == 2

        # Registry now has the model
        reg_res = client.get("/api/detectors/registry")
        names = [m["name"] for m in reg_res.get_json()["detectors"]]
        assert "From LS" in names

        # Trainable-model file has the labelset baked in
        tm_res = client.get("/api/detectors/From%20LS")
        tm_data = tm_res.get_json()
        assert len(tm_data["labelset"]["labels"]) == 2
        assert tm_data["text_query"] == ""
        assert tm_data["media_example"] == ""
        assert tm_data["examples"] == []

    def test_missing_name_returns_422(self, client, tmp_path):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text('{"labels": []}')
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"filepath": str(labels_path)},
        )
        assert res.status_code == 422
        assert "name" in res.get_json().get("errors", {}).get("json", {})

    def test_md5_only_labelset_returns_400(self, client, tmp_path):
        """Legacy md5-only labels have no origin — media type cannot be inferred."""
        import app as app_module

        md5 = app_module.medias[1]["md5"]
        payload = json.dumps({"labels": [{"md5": md5, "label": "good"}]})
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(payload)
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "NoOrigin", "filepath": str(labels_path)},
        )
        assert res.status_code == 400
        assert "media type" in res.get_json()["error"].lower()

    def test_mixed_media_types_returns_400(self, client, tmp_path):
        """Labels with conflicting origin media types are rejected."""
        import app as app_module

        md5_a = app_module.medias[1]["md5"]
        md5_b = app_module.medias[2]["md5"]
        payload = json.dumps(
            {
                "labels": [
                    {"md5": md5_a, "label": "good", "origin": self._audio_origin(), "origin_name": "a.wav"},
                    {"md5": md5_b, "label": "good", "origin": self._image_origin(), "origin_name": "b.jpg"},
                ]
            }
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(payload)
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "Mixed", "filepath": str(labels_path)},
        )
        assert res.status_code == 400
        err = res.get_json()["error"]
        assert "audio" in err and "image" in err

    def test_unknown_importer_returns_404(self, client):
        res = client.post(
            "/api/detectors/registry/from-labelset/no_such_importer",
            json={"name": "X"},
        )
        assert res.status_code == 404
        assert "no_such_importer" in res.get_json()["error"]

    def test_duplicate_name_returns_409(self, client, tmp_path):
        """Name collision with an existing detector file."""
        import app as app_module

        md5 = app_module.medias[1]["md5"]
        payload = json.dumps(
            {
                "labels": [
                    {"md5": md5, "label": "good", "origin": self._audio_origin(), "origin_name": "a.wav"},
                ]
            }
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(payload)
        client.post(
            "/api/detectors",
            json={"name": "Dup", "media_type": "audio", "text_query": "dup"},
        )
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "Dup", "filepath": str(labels_path)},
        )
        assert res.status_code == 409

    def test_skips_invalid_label_values(self, client, tmp_path):
        import app as app_module

        md5 = app_module.medias[1]["md5"]
        origin = self._audio_origin()
        payload = json.dumps(
            {
                "labels": [
                    {"md5": md5, "label": "good", "origin": origin, "origin_name": "a.wav"},
                    {"md5": md5, "label": "maybe", "origin": origin, "origin_name": "a.wav"},
                ]
            }
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(payload)
        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "SkipInvalid", "filepath": str(labels_path)},
        )
        assert res.status_code == 201
        data = res.get_json()
        assert data["applied"] == 1
        assert data["skipped"] == 1
        assert data["num_labels"] == 1
        assert data["detector"]["media_type"] == "audio"

    def test_loading_after_creation_restores_labels(self, client, tmp_path):
        """Loading the newly-created model resolves labels into the active dataset."""
        import app as app_module
        from vtsearch.state import (
            bad_votes,
            good_votes,
        )

        md5_good = app_module.medias[1]["md5"]
        md5_bad = app_module.medias[2]["md5"]
        origin = self._audio_origin()
        payload = json.dumps(
            {
                "labels": [
                    {"md5": md5_good, "label": "good", "origin": origin, "origin_name": "a.wav"},
                    {"md5": md5_bad, "label": "bad", "origin": origin, "origin_name": "b.wav"},
                ]
            }
        )
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(payload)

        res = client.post(
            "/api/detectors/registry/from-labelset/server_json_file",
            json={"name": "Loadable", "filepath": str(labels_path)},
        )
        detector_id = res.get_json()["detector"]["id"]
        _load_detector_and_wait(client, detector_id)

        # Labels resolved into the loaded detector's votes (matched by md5).
        assert 1 in good_votes
        assert 2 in bad_votes

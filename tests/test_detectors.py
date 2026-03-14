import io
import json

import numpy as np
import pytest

import app as app_module


class TestDetectorExport:
    def test_export_with_sufficient_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "weights" in data
        assert "threshold" in data
        assert isinstance(data["weights"], dict)
        assert isinstance(data["threshold"], (int, float))
        # Origin info for weight-free serialisation
        assert "good_origins" in data
        assert "bad_origins" in data
        assert "inclusion" in data
        assert len(data["good_origins"]) == 3
        assert len(data["bad_origins"]) == 3

    def test_export_requires_good_votes(self, client):
        app_module.bad_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "need at least one good and one bad vote" in data["error"]

    def test_export_requires_bad_votes(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 400

    def test_export_weights_structure(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        resp = client.post("/api/detector/export")
        data = resp.get_json()
        weights = data["weights"]
        # MLP has 4 layers: Linear, ReLU, Dropout, Linear
        # So we expect 4 keys: 0.weight, 0.bias, 3.weight, 3.bias
        assert "0.weight" in weights
        assert "0.bias" in weights
        assert "3.weight" in weights
        assert "3.bias" in weights


class TestDetectorSort:
    def test_sort_with_valid_detector(self, client):
        # First export a detector
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        # Now use it to sort
        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data
        assert len(data["results"]) == app_module.NUM_MEDIAS

    def test_sort_results_sorted_descending(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        scores = [e["score"] for e in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_sort_scores_in_valid_range(self, client):
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3, 4]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        for entry in data["results"]:
            assert 0.0 <= entry["score"] <= 1.0

    def test_sort_missing_detector(self, client):
        resp = client.post("/api/detector-sort", json={})
        assert resp.status_code == 400

    def test_sort_missing_weights(self, client):
        resp = client.post("/api/detector-sort", json={"detector": {"threshold": 0.5}})
        assert resp.status_code == 400

    def test_detector_roundtrip(self, client):
        """Export a detector and verify it produces reasonable scores."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

        # Export detector
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()

        # Use detector to sort
        resp = client.post("/api/detector-sort", json={"detector": detector})
        data = resp.get_json()
        score_map = {e["id"]: e["score"] for e in data["results"]}

        # Good medias should score higher than bad medias on average
        avg_good = np.mean([score_map[i] for i in app_module.good_votes])
        avg_bad = np.mean([score_map[i] for i in app_module.bad_votes])
        assert avg_good > avg_bad


class TestAutorunDetectors:
    """Tests for the autorun-detectors management endpoints."""

    def _export_detector(self, client):
        """Helper: vote on some medias and export a valid detector payload."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        resp = client.post("/api/detector/export")
        assert resp.status_code == 200
        return resp.get_json()

    def _post_autorun(self, client, name, detector):
        return client.post(
            "/api/autorun-detectors",
            json={
                "name": name,
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
            },
        )

    # -- GET list --

    def test_get_empty_list(self, client):
        resp = client.get("/api/autorun-detectors")
        assert resp.status_code == 200
        assert resp.get_json()["detectors"] == []

    def test_get_list_after_add(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "my-detector", det)

        resp = client.get("/api/autorun-detectors")
        data = resp.get_json()
        assert len(data["detectors"]) == 1
        d = data["detectors"][0]
        assert d["name"] == "my-detector"
        assert d["media_type"] == "audio"
        assert "threshold" in d

    # -- POST add --

    def test_add_detector_returns_success(self, client):
        det = self._export_detector(client)
        resp = self._post_autorun(client, "test-det", det)
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_add_missing_name_returns_400(self, client):
        det = self._export_detector(client)
        resp = client.post(
            "/api/autorun-detectors",
            json={"media_type": "audio", "weights": det["weights"]},
        )
        assert resp.status_code == 400

    def test_add_missing_media_type_returns_400(self, client):
        det = self._export_detector(client)
        resp = client.post(
            "/api/autorun-detectors",
            json={"name": "test", "weights": det["weights"]},
        )
        assert resp.status_code == 400

    def test_add_without_weights_creates_untrained(self, client):
        resp = client.post(
            "/api/autorun-detectors",
            json={"name": "test", "media_type": "audio"},
        )
        assert resp.status_code == 200

        detectors = client.get("/api/autorun-detectors").get_json()["detectors"]
        det = [d for d in detectors if d["name"] == "test"][0]
        assert det["weights"] is None
        assert det["autodetect"] is False

    def test_add_multiple_detectors(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "det-a", det)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        self._post_autorun(client, "det-b", det)

        resp = client.get("/api/autorun-detectors")
        names = {d["name"] for d in resp.get_json()["detectors"]}
        assert names == {"det-a", "det-b"}

    def test_add_overwrites_existing_name(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "dup", det)
        self._post_autorun(client, "dup", det)

        resp = client.get("/api/autorun-detectors")
        assert len(resp.get_json()["detectors"]) == 1

    # -- DELETE --

    def test_delete_detector(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "to-delete", det)

        resp = client.delete("/api/autorun-detectors/to-delete")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

        resp = client.get("/api/autorun-detectors")
        assert resp.get_json()["detectors"] == []

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/autorun-detectors/does-not-exist")
        assert resp.status_code == 404

    # -- RENAME --

    def test_rename_detector(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "old-name", det)

        resp = client.put(
            "/api/autorun-detectors/old-name/rename",
            json={"new_name": "new-name"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["new_name"] == "new-name"

        names = [d["name"] for d in client.get("/api/autorun-detectors").get_json()["detectors"]]
        assert "new-name" in names
        assert "old-name" not in names

    def test_rename_nonexistent_returns_400(self, client):
        resp = client.put(
            "/api/autorun-detectors/ghost/rename",
            json={"new_name": "anything"},
        )
        assert resp.status_code == 400

    def test_rename_to_existing_name_returns_400(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "det-a", det)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()
        self._post_autorun(client, "det-b", det)

        resp = client.put(
            "/api/autorun-detectors/det-a/rename",
            json={"new_name": "det-b"},
        )
        assert resp.status_code == 400

    def test_rename_missing_new_name_returns_400(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "some-det", det)

        resp = client.put(
            "/api/autorun-detectors/some-det/rename",
            json={},
        )
        assert resp.status_code == 400

    # -- import-pkl (detector JSON file) --

    def test_import_pkl_from_detector_json(self, client):
        det = self._export_detector(client)
        json_bytes = json.dumps(det).encode("utf-8")
        data = {
            "file": (io.BytesIO(json_bytes), "detector.json"),
            "name": "imported",
        }
        resp = client.post(
            "/api/autorun-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        result = resp.get_json()
        assert result["success"] is True
        assert result["name"] == "imported"

    def test_import_pkl_uses_filename_stem_as_default_name(self, client):
        det = self._export_detector(client)
        json_bytes = json.dumps(det).encode("utf-8")
        data = {"file": (io.BytesIO(json_bytes), "my_detector.json")}
        resp = client.post(
            "/api/autorun-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["name"] == "my_detector"

    def test_import_pkl_preserves_media_type_from_file(self, client):
        det = self._export_detector(client)
        # Embed explicit media_type in the "file" payload
        det["media_type"] = "image"
        json_bytes = json.dumps(det).encode("utf-8")
        data = {
            "file": (io.BytesIO(json_bytes), "image_detector.json"),
            "name": "img-det",
        }
        resp = client.post(
            "/api/autorun-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["media_type"] == "image"

    def test_import_pkl_no_file_returns_400(self, client):
        resp = client.post("/api/autorun-detectors/import-pkl", data={})
        assert resp.status_code == 400

    def test_import_pkl_invalid_format_returns_400(self, client):
        data = {"file": (io.BytesIO(b'{"not_a_detector": true}'), "bad.json")}
        resp = client.post(
            "/api/autorun-detectors/import-pkl",
            data=data,
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    # -- Detector data is stored correctly --

    def test_stored_detector_has_correct_fields(self, client):
        det = self._export_detector(client)
        self._post_autorun(client, "field-check", det)

        from vtsearch.utils.state import autorun_detectors

        assert "field-check" in autorun_detectors
        stored = autorun_detectors["field-check"]
        assert stored["name"] == "field-check"
        assert stored["media_type"] == "audio"
        assert "weights" in stored
        assert "threshold" in stored
        assert "created_at" in stored
        assert "good_origins" in stored
        assert "bad_origins" in stored
        assert "inclusion" in stored


class TestAutoDetect:
    """Tests for POST /api/auto-detect."""

    def _add_audio_detector(self, client, name="test-detector"):
        """Helper: create and save an audio detector with autodetect enabled."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        export_resp = client.post("/api/detector/export")
        assert export_resp.status_code == 200
        detector = export_resp.get_json()

        save_resp = client.post(
            "/api/autorun-detectors",
            json={
                "name": name,
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
                "autodetect": True,
            },
        )
        assert save_resp.status_code == 200
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

    # -- no matching detectors --

    def test_no_autorun_detectors_returns_400(self, client):
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 400
        assert "No autorun detectors" in resp.get_json()["error"]

    def test_no_matching_media_type_returns_400(self, client):
        """A detector for a different media type should not match audio medias."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})
        export_resp = client.post("/api/detector/export")
        detector = export_resp.get_json()
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Save as "image" type with autodetect — medias are audio, so it won't match
        client.post(
            "/api/autorun-detectors",
            json={
                "name": "image-detector",
                "media_type": "image",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
                "autodetect": True,
            },
        )
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 400

    # -- basic success --

    def test_returns_200_with_matching_detector(self, client):
        self._add_audio_detector(client)
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200

    def test_response_has_required_top_level_fields(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        assert "media_type" in data
        assert "detectors_run" in data
        assert "results" in data

    def test_media_type_matches_clips(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        assert data["media_type"] == "audio"

    def test_detectors_run_count(self, client):
        self._add_audio_detector(client, name="det-1")
        self._add_audio_detector(client, name="det-2")
        data = client.post("/api/auto-detect").get_json()
        assert data["detectors_run"] == 2

    # -- per-detector result structure --

    def test_each_result_has_required_fields(self, client):
        self._add_audio_detector(client, name="struct-check")
        data = client.post("/api/auto-detect").get_json()
        result = data["results"]["struct-check"]
        assert "detector_name" in result
        assert "threshold" in result
        assert "total_hits" in result
        assert "hits" in result

    def test_detector_name_matches_key(self, client):
        self._add_audio_detector(client, name="named-detector")
        data = client.post("/api/auto-detect").get_json()
        result = data["results"]["named-detector"]
        assert result["detector_name"] == "named-detector"

    def test_total_hits_matches_hits_length(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            assert result["total_hits"] == len(result["hits"])

    # -- hit data safety --

    def test_hits_do_not_contain_embeddings(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "embedding" not in hit

    def test_hits_do_not_contain_media_bytes(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "media_bytes" not in hit

    def test_hits_contain_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["hits"]:
                assert "score" in hit
                assert 0.0 <= hit["score"] <= 1.0

    def test_hits_sorted_descending_by_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            scores = [h["score"] for h in result["hits"]]
            assert scores == sorted(scores, reverse=True)

    # -- threshold correctness --

    def test_all_hits_score_at_or_above_threshold(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            threshold = result["threshold"]
            for hit in result["hits"]:
                assert hit["score"] >= threshold - 1e-6  # float tolerance

    # -- negative_hits --

    def test_each_result_has_negative_hits(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            assert "negative_hits" in result
            assert isinstance(result["negative_hits"], list)

    def test_negative_hits_score_below_threshold(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            threshold = result["threshold"]
            for hit in result["negative_hits"]:
                assert hit["score"] < threshold + 1e-6

    def test_negative_hits_do_not_contain_embeddings(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            for hit in result["negative_hits"]:
                assert "embedding" not in hit
                assert "media_bytes" not in hit

    def test_hits_and_negative_hits_cover_all_clips(self, client):
        """Positive + negative hits should cover every media in the dataset."""
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        from vtsearch.utils import medias

        total_clips = len(medias)
        for result in data["results"].values():
            total_returned = len(result["hits"]) + len(result["negative_hits"])
            assert total_returned == total_clips

    def test_negative_hits_sorted_descending_by_score(self, client):
        self._add_audio_detector(client)
        data = client.post("/api/auto-detect").get_json()
        for result in data["results"].values():
            scores = [h["score"] for h in result["negative_hits"]]
            assert scores == sorted(scores, reverse=True)


class TestServerDetectorExport:
    """Tests for the ServerFileProcessorExporter endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_server_dir(self, tmp_path):
        """Point detectors_dir at a temp directory for each test."""
        from vtsearch import settings

        det_dir = tmp_path / "detectors"
        settings.set_detectors_dir(str(det_dir))
        self._det_dir = det_dir
        yield

    def _vote(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

    # -- basic success --

    def test_export_server_creates_file(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "my_detector"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["name"] == "my_detector"
        assert (self._det_dir / "my_detector.json").exists()

    def test_export_server_file_contains_valid_json(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "valid_json"})
        content = json.loads((self._det_dir / "valid_json.json").read_text())
        # New format: origins instead of weights
        assert "good_origins" in content
        assert "bad_origins" in content
        assert "inclusion" in content
        assert "media_type" in content
        assert "name" in content
        # Weights must NOT be serialised to disk
        assert "weights" not in content

    def test_export_server_returns_name(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "path_check"})
        data = resp.get_json()
        assert data["success"] is True
        assert data["name"] == "path_check"

    # -- name validation --

    def test_export_server_missing_name_returns_400(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={})
        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"].lower()

    def test_export_server_empty_name_returns_400(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "  "})
        assert resp.status_code == 400

    # -- vote validation --

    def test_export_server_no_votes_returns_400(self, client):
        resp = client.post("/api/detector/export-server", json={"name": "no_votes"})
        assert resp.status_code == 400
        assert "vote" in resp.get_json()["error"].lower()

    # -- overwrite logic --

    def test_export_server_returns_409_when_exists(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "dup"})
        # Second export without overwrite
        resp = client.post("/api/detector/export-server", json={"name": "dup"})
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["exists"] is True
        assert "dup" in data["name"]

    def test_export_server_overwrite_succeeds(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "overwrite_me"})
        resp = client.post(
            "/api/detector/export-server",
            json={"name": "overwrite_me", "overwrite": True},
        )
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    # -- list server files --

    def test_list_server_files_empty(self, client):
        resp = client.get("/api/detector/server-files")
        assert resp.status_code == 200
        assert resp.get_json()["files"] == []

    def test_list_server_files_after_export(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "listed"})
        resp = client.get("/api/detector/server-files")
        data = resp.get_json()
        names = [f["name"] for f in data["files"]]
        assert "listed" in names

    def test_list_server_files_has_filename_and_size(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "sized"})
        resp = client.get("/api/detector/server-files")
        entry = resp.get_json()["files"][0]
        assert "filename" in entry
        assert "size_bytes" in entry
        assert entry["size_bytes"] > 0

    # -- name sanitisation --

    def test_export_server_sanitises_name(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "a/b\\c:d"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Special characters should be stripped
        assert "/" not in data["name"]
        assert "\\" not in data["name"]
        assert ":" not in data["name"]


class TestDetectorLabelsetExport:
    """Test the labelset export flow from the detector export context.

    The frontend Export Detector modal offers labelset exporters alongside
    the traditional detector weight export.  The flow is:
      1. GET /api/exporters  (list available exporters)
      2. GET /api/labels/export  (build labelset from current votes)
      3. POST /api/exporters/export  (run the chosen exporter on the labelset)
    """

    def test_exporters_list_available(self, client):
        """GET /api/exporters returns labelset exporters usable from the detector modal."""
        resp = client.get("/api/exporters")
        assert resp.status_code == 200
        names = {e["name"] for e in resp.get_json()}
        # At least the server JSON exporter should be present
        assert "server_json_file" in names

    def test_labelset_export_server_json_full(self, client, tmp_path):
        """Full flow: votes -> labelset -> server_json_file exporter -> file on disk."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19]})

        # Step 1: get the labelset
        labels_resp = client.get("/api/labels/export")
        assert labels_resp.status_code == 200
        labels_data = labels_resp.get_json()
        assert "labels" in labels_data
        assert len(labels_data["labels"]) == 5

        # Step 2: export through server_json_file exporter
        fpath = tmp_path / "exported_labels.json"
        export_resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(fpath)},
                "results": labels_data,
            },
        )
        assert export_resp.status_code == 200
        data = export_resp.get_json()
        assert data["success"] is True

        # The file should contain valid JSON with the labels
        downloaded = json.loads(fpath.read_text())
        assert "labels" in downloaded
        assert len(downloaded["labels"]) == 5

    def test_labelset_export_server_json(self, client, tmp_path):
        """Full flow: votes -> labelset -> server_json_file exporter -> file on disk."""
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3]})

        labels_resp = client.get("/api/labels/export")
        labels_data = labels_resp.get_json()

        fpath = tmp_path / "detector_labels.json"
        export_resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(fpath)},
                "results": labels_data,
            },
        )
        assert export_resp.status_code == 200
        assert export_resp.get_json()["success"] is True
        assert fpath.exists()

        written = json.loads(fpath.read_text())
        assert "labels" in written
        assert len(written["labels"]) == 3

    def test_labelset_roundtrip_via_exporter(self, client, tmp_path):
        """Labels exported through an exporter can be re-imported to restore votes."""
        app_module.good_votes.update({k: None for k in [1, 3]})
        app_module.bad_votes.update({k: None for k in [2]})

        # Export labelset
        labels_resp = client.get("/api/labels/export")
        labels_data = labels_resp.get_json()

        # Save via server_json_file exporter
        fpath = tmp_path / "labels_roundtrip.json"
        client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(fpath)},
                "results": labels_data,
            },
        )

        # Clear votes and re-import
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        saved = json.loads(fpath.read_text())
        resp = client.post("/api/labels/import", json=saved)
        data = resp.get_json()
        assert data["applied"] == 3
        assert set(app_module.good_votes) == {1, 3}
        assert set(app_module.bad_votes) == {2}

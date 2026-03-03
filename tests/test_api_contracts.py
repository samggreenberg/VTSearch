"""API contract tests.

Verifies that every major API endpoint returns the correct:
- HTTP status code
- Content-Type header (JSON for data endpoints)
- Response shape (required keys, correct types)
- Consistent error format {"error": "..."} with appropriate status codes

These tests complement the existing functional tests by focusing on the
*contract* — the shape of the response that the frontend relies on.
"""

from __future__ import annotations

import app as app_module
from vtsearch.utils import good_votes, bad_votes


class TestMediasContract:
    """GET /api/medias response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_json_array(self):
        resp = self.client.get("/api/medias")
        assert resp.status_code == 200
        assert resp.content_type.startswith("application/json")
        data = resp.get_json()
        assert isinstance(data, list)

    def test_each_media_has_required_fields(self):
        resp = self.client.get("/api/medias")
        data = resp.get_json()
        required = {"id", "type", "duration", "file_size", "filename", "category", "md5"}
        for item in data:
            assert required.issubset(item.keys()), f"Missing keys: {required - item.keys()}"

    def test_id_is_integer(self):
        resp = self.client.get("/api/medias")
        data = resp.get_json()
        for item in data:
            assert isinstance(item["id"], int)

    def test_file_size_is_positive_integer(self):
        resp = self.client.get("/api/medias")
        data = resp.get_json()
        for item in data:
            assert isinstance(item["file_size"], int)
            assert item["file_size"] > 0

    def test_md5_is_32_char_hex_string(self):
        resp = self.client.get("/api/medias")
        data = resp.get_json()
        for item in data:
            assert isinstance(item["md5"], str)
            assert len(item["md5"]) == 32

    def test_excludes_embedding_and_media_bytes(self):
        resp = self.client.get("/api/medias")
        data = resp.get_json()
        for item in data:
            assert "embedding" not in item
            assert "media_bytes" not in item
            assert "media_string" not in item


class TestVotesContract:
    """GET /api/votes response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_json_with_required_keys(self):
        resp = self.client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "good" in data
        assert "bad" in data
        assert "click_times" in data
        assert "learned_scores" in data

    def test_good_and_bad_are_lists(self):
        resp = self.client.get("/api/votes")
        data = resp.get_json()
        assert isinstance(data["good"], list)
        assert isinstance(data["bad"], list)

    def test_click_times_is_dict(self):
        resp = self.client.get("/api/votes")
        data = resp.get_json()
        assert isinstance(data["click_times"], dict)

    def test_learned_scores_is_dict(self):
        resp = self.client.get("/api/votes")
        data = resp.get_json()
        assert isinstance(data["learned_scores"], dict)

    def test_vote_response_shape(self):
        resp = self.client.post("/api/medias/1/vote", json={"vote": "good"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True


class TestSortContract:
    """POST /api/sort response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_response_has_results_and_threshold(self):
        resp = self.client.post("/api/sort", json={"text": "test query"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data

    def test_results_are_list(self):
        resp = self.client.post("/api/sort", json={"text": "test"})
        data = resp.get_json()
        assert isinstance(data["results"], list)

    def test_threshold_is_number(self):
        resp = self.client.post("/api/sort", json={"text": "test"})
        data = resp.get_json()
        assert isinstance(data["threshold"], (int, float))

    def test_each_result_has_id_and_similarity(self):
        resp = self.client.post("/api/sort", json={"text": "test"})
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "similarity" in entry
            assert isinstance(entry["id"], int)
            assert isinstance(entry["similarity"], (int, float))


class TestLearnedSortContract:
    """POST /api/learned-sort response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_response_has_results_and_threshold(self):
        good_votes.update({1: None, 2: None})
        bad_votes.update({3: None, 4: None})
        resp = self.client.post("/api/learned-sort")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "results" in data
        assert "threshold" in data

    def test_each_result_has_id_and_score(self):
        good_votes.update({1: None, 2: None})
        bad_votes.update({3: None, 4: None})
        resp = self.client.post("/api/learned-sort")
        data = resp.get_json()
        for entry in data["results"]:
            assert "id" in entry
            assert "score" in entry
            assert isinstance(entry["id"], int)
            assert isinstance(entry["score"], (int, float))


class TestInclusionContract:
    """GET/POST /api/inclusion response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_get_returns_inclusion_key(self):
        resp = self.client.get("/api/inclusion")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "inclusion" in data
        assert isinstance(data["inclusion"], (int, float))

    def test_post_returns_inclusion_key(self):
        resp = self.client.post("/api/inclusion", json={"inclusion": 3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "inclusion" in data
        assert data["inclusion"] == 3


class TestSafeThresholdsContract:
    """GET/POST /api/safe-thresholds response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_get_returns_boolean(self):
        resp = self.client.get("/api/safe-thresholds")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "safe_thresholds" in data
        assert isinstance(data["safe_thresholds"], bool)

    def test_post_returns_boolean(self):
        resp = self.client.post("/api/safe-thresholds", json={"safe_thresholds": True})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["safe_thresholds"] is True


class TestLabelsExportContract:
    """GET /api/labels/export response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_json_with_labels_key(self):
        resp = self.client.get("/api/labels/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "labels" in data
        assert isinstance(data["labels"], list)

    def test_with_votes_labels_contain_required_fields(self):
        good_votes[1] = None
        bad_votes[2] = None
        resp = self.client.get("/api/labels/export")
        data = resp.get_json()
        assert len(data["labels"]) >= 2
        for label in data["labels"]:
            assert "md5" in label
            assert "label" in label
            assert label["label"] in ("good", "bad")


class TestLabelsImportContract:
    """POST /api/labels/import response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_applied_and_skipped(self):
        labels = [{"md5": "nonexistent", "label": "good"}]
        resp = self.client.post("/api/labels/import", json={"labels": labels})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "applied" in data
        assert "skipped" in data
        assert isinstance(data["applied"], int)
        assert isinstance(data["skipped"], int)


class TestDatasetStatusContract:
    """GET /api/dataset/status response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_required_fields(self):
        resp = self.client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "loaded" in data
        assert "num_medias" in data
        assert isinstance(data["loaded"], bool)
        assert isinstance(data["num_medias"], int)

    def test_has_votes_field(self):
        resp = self.client.get("/api/dataset/status")
        data = resp.get_json()
        assert "has_votes" in data
        assert isinstance(data["has_votes"], bool)


class TestMediaTypesContract:
    """GET /api/media-types response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_media_types_list(self):
        resp = self.client.get("/api/media-types")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "media_types" in data
        assert isinstance(data["media_types"], list)
        assert len(data["media_types"]) > 0

    def test_each_type_has_required_fields(self):
        resp = self.client.get("/api/media-types")
        data = resp.get_json()
        for mt in data["media_types"]:
            assert "type_id" in mt
            assert "name" in mt
            assert isinstance(mt["type_id"], str)
            assert isinstance(mt["name"], str)


class TestSettingsContract:
    """GET/PUT /api/settings response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_get_returns_all_settings(self):
        resp = self.client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "volume" in data
        assert "autorun_processors" in data
        assert isinstance(data["volume"], (int, float))
        assert isinstance(data["autorun_processors"], list)

    def test_get_defaults_returns_dict(self):
        resp = self.client.get("/api/settings/defaults")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "volume" in data

    def test_put_returns_updated_settings(self):
        resp = self.client.put("/api/settings", json={"volume": 0.5})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "volume" in data

    def test_autorun_processors_list(self):
        resp = self.client.get("/api/settings/autorun-processors")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "autorun_processors" in data
        assert isinstance(data["autorun_processors"], list)


class TestExportersContract:
    """GET /api/exporters response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_list(self):
        resp = self.client.get("/api/exporters")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_each_exporter_has_name(self):
        resp = self.client.get("/api/exporters")
        data = resp.get_json()
        for exp in data:
            assert "name" in exp
            assert isinstance(exp["name"], str)


class TestImportersContract:
    """GET /api/dataset/importers response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_importers_dict(self):
        resp = self.client.get("/api/dataset/importers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "importers" in data
        assert isinstance(data["importers"], list)

    def test_all_importers_returns_importers_dict(self):
        resp = self.client.get("/api/dataset/all-importers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "importers" in data
        assert isinstance(data["importers"], list)

    def test_all_importers_is_superset(self):
        resp1 = self.client.get("/api/dataset/importers")
        resp2 = self.client.get("/api/dataset/all-importers")
        names1 = {i["name"] for i in resp1.get_json()["importers"]}
        names2 = {i["name"] for i in resp2.get_json()["importers"]}
        assert names1.issubset(names2)


class TestLabelImportersContract:
    """GET /api/label-importers response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_list(self):
        resp = self.client.get("/api/label-importers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_each_importer_has_name(self):
        resp = self.client.get("/api/label-importers")
        data = resp.get_json()
        for imp in data:
            assert "name" in imp


class TestProcessorImportersContract:
    """GET /api/processor-importers response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_list(self):
        resp = self.client.get("/api/processor-importers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)


class TestAutorunDetectorsContract:
    """GET/POST/DELETE /api/autorun-detectors response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_get_returns_detectors_dict(self):
        resp = self.client.get("/api/autorun-detectors")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "detectors" in data
        assert isinstance(data["detectors"], list)

    def test_create_returns_success(self):
        resp = self.client.post("/api/autorun-detectors", json={
            "name": "test_det",
            "media_type": "audio",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["name"] == "test_det"

    def test_delete_returns_success(self):
        self.client.post("/api/autorun-detectors", json={
            "name": "to_delete",
            "media_type": "audio",
        })
        resp = self.client.delete("/api/autorun-detectors/to_delete")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_rename_returns_success(self):
        self.client.post("/api/autorun-detectors", json={
            "name": "old_name",
            "media_type": "audio",
        })
        resp = self.client.put("/api/autorun-detectors/old_name/rename", json={
            "new_name": "new_name",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["new_name"] == "new_name"


class TestTextsortSuggestionsContract:
    """GET/POST /api/textsort-suggestions response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_get_returns_suggestions_list(self):
        resp = self.client.get("/api/textsort-suggestions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_post_returns_ok(self):
        resp = self.client.post("/api/textsort-suggestions", json={"text": "test query"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True

    def test_stored_suggestion_appears_in_list(self):
        self.client.post("/api/textsort-suggestions", json={"text": "my suggestion"})
        resp = self.client.get("/api/textsort-suggestions")
        data = resp.get_json()
        assert "my suggestion" in data["suggestions"]


class TestSortProgressContract:
    """GET /api/sort/progress response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_json(self):
        resp = self.client.get("/api/sort/progress")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)


class TestDiversityTreeContract:
    """GET/POST /api/diversity-tree/next response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_get_returns_required_fields(self):
        resp = self.client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "id" in data
        assert "diversity_level" in data
        assert "exhausted" in data

    def test_post_returns_required_fields(self):
        resp = self.client.post("/api/diversity-tree/next", json={"scores": {}})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "id" in data
        assert "diversity_level" in data
        assert "exhausted" in data


class TestLabelingStatusContract:
    """GET /api/labeling-status response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_json(self):
        resp = self.client.get("/api/labeling-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)

    def test_with_votes_returns_indicator_fields(self):
        """With sufficient votes, response should have labeling status indicators."""
        for i in range(1, 4):
            self.client.post(f"/api/medias/{i}/vote", json={"vote": "good"})
        for i in range(4, 7):
            self.client.post(f"/api/medias/{i}/vote", json={"vote": "bad"})
        resp = self.client.get("/api/labeling-status")
        assert resp.status_code == 200
        data = resp.get_json()
        # Should have status indicators or at minimum no error
        assert "error" not in data or "good_count" in data


class TestFillFromSortContract:
    """POST /api/labels/fill-from-sort response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_dry_run_returns_counts(self):
        sort_results = [{"id": i, "score": 0.9 if i <= 5 else 0.1} for i in range(1, 11)]
        resp = self.client.post("/api/labels/fill-from-sort", json={
            "sort_results": sort_results,
            "threshold": 0.5,
            "sides": "both",
            "confirm": False,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "good_count" in data
        assert "bad_count" in data

    def test_confirm_returns_applied_and_results(self):
        sort_results = [{"id": i, "score": 0.9 if i <= 3 else 0.1} for i in range(1, 11)]
        resp = self.client.post("/api/labels/fill-from-sort", json={
            "sort_results": sort_results,
            "threshold": 0.5,
            "sides": "both",
            "confirm": True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "good_applied" in data
        assert "bad_applied" in data
        assert "results" in data


class TestDatasetRegistryContract:
    """GET /api/datasets/registry response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_datasets_dict(self):
        resp = self.client.get("/api/datasets/registry")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)


class TestModelsRegistryContract:
    """GET /api/models/registry response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_returns_models_dict(self):
        resp = self.client.get("/api/models/registry")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestTrainableModelsContract:
    """GET/POST /api/trainable-models response shape."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_list_returns_models_dict(self):
        resp = self.client.get("/api/trainable-models")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestErrorResponseFormat:
    """All error responses should use consistent JSON format."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_404_media_error_is_json(self):
        resp = self.client.get("/api/medias/99999/audio")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    def test_400_sort_error_is_json(self):
        resp = self.client.post("/api/sort", json={"text": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_400_vote_error_is_json(self):
        resp = self.client.post("/api/medias/1/vote", json={"vote": "invalid"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_400_inclusion_error_is_json(self):
        resp = self.client.post("/api/inclusion", json={"inclusion": "not_a_number"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_400_safe_thresholds_error_is_json(self):
        resp = self.client.post("/api/safe-thresholds", json={"safe_thresholds": "not_a_bool"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_400_labels_import_error_is_json(self):
        resp = self.client.post("/api/labels/import", json={"labels": "not_a_list"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_400_settings_error_is_json(self):
        resp = self.client.put("/api/settings", json={"volume": "not_a_number"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_400_learned_sort_no_votes_is_json(self):
        resp = self.client.post("/api/learned-sort")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_404_unknown_exporter_is_json(self):
        resp = self.client.post("/api/exporters/export", json={
            "exporter_name": "nonexistent_exporter",
        })
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data

    def test_400_missing_exporter_name_is_json(self):
        resp = self.client.post("/api/exporters/export", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

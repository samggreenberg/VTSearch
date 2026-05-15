"""Error recovery tests.

Covers error handling and edge cases that the app should recover from
gracefully rather than crashing:
- Invalid / malformed request bodies
- Missing required fields
- Type mismatches (string where number expected, etc.)
- Empty state (no medias loaded, no votes)
- Nonexistent resources (media IDs, detector names, importers)
- Boundary values and edge cases
- Media type mismatches (requesting audio for a video, etc.)
- Path traversal / arbitrary file read prevention
"""

from __future__ import annotations

import io
import json

from vtsearch.state import (
    good_votes,
    bad_votes,
    medias,
)


class TestInvalidRequestBodies:
    """Routes should handle malformed or missing JSON gracefully."""

    def test_sort_with_no_body(self, client):
        resp = client.post("/api/sort", content_type="application/json")
        assert resp.status_code == 400

    def test_sort_with_non_json_string(self, client):
        resp = client.post("/api/sort", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_vote_with_empty_json(self, client):
        resp = client.post("/api/medias/1/vote", json={})
        assert resp.status_code == 400

    def test_vote_with_null_body(self, client):
        resp = client.post(
            "/api/medias/1/vote",
            data="null",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_inclusion_with_empty_json(self, client):
        resp = client.post("/api/inclusion", json={})
        # Missing 'inclusion' field → should be 400
        assert resp.status_code == 400

    def test_safe_thresholds_with_empty_json(self, client):
        resp = client.post("/api/safe-thresholds", json={})
        assert resp.status_code == 400

    def test_labels_import_with_null_body(self, client):
        resp = client.post(
            "/api/labels/import",
            data="null",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_settings_put_with_empty_body(self, client):
        resp = client.put("/api/settings", json={})
        # Empty dict is falsy → route rejects with 400
        assert resp.status_code == 400

    def test_settings_put_with_null_body(self, client):
        resp = client.put(
            "/api/settings",
            data="null",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_textsort_suggestion_with_null_body(self, client):
        resp = client.post(
            "/api/textsort-suggestions",
            data="null",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_register_model_with_null_body(self, client):
        resp = client.post(
            "/api/detectors/registry",
            data="null",
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestMissingRequiredFields:
    """Routes should reject requests missing required fields."""

    def test_sort_missing_text(self, client):
        resp = client.post("/api/sort", json={"query": "hello"})
        assert resp.status_code == 400

    def test_vote_missing_vote_field(self, client):
        resp = client.post("/api/medias/1/vote", json={"label": "good"})
        assert resp.status_code == 400

    def test_register_model_missing_name(self, client):
        resp = client.post(
            "/api/detectors/registry",
            json={"media_type": "audio"},
        )
        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"]

    def test_register_model_missing_media_type(self, client):
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "test"},
        )
        assert resp.status_code == 400
        assert "media_type" in resp.get_json()["error"]

    def test_register_model_rename_missing_new_name(self, client):
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "rename_test", "media_type": "audio"},
        )
        detector_id = resp.get_json()["detector"]["id"]
        resp = client.put(f"/api/detectors/registry/{detector_id}/rename", json={})
        assert resp.status_code == 400

    def test_autorun_flag_missing_value(self, client):
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "ad_test", "media_type": "audio"},
        )
        detector_id = resp.get_json()["detector"]["id"]
        resp = client.put(f"/api/detectors/registry/{detector_id}/autorun", json={})
        assert resp.status_code == 400

    def test_fill_from_sort_missing_threshold(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": [{"id": 1, "score": 0.5}],
            },
        )
        assert resp.status_code == 400

    def test_fill_from_sort_missing_sort_results(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "threshold": 0.5,
            },
        )
        assert resp.status_code == 400

    def test_exporter_missing_name(self, client):
        resp = client.post("/api/exporters/export", json={})
        assert resp.status_code == 400

    def test_textsort_suggestion_empty_text(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": ""})
        assert resp.status_code == 400

    def test_textsort_suggestion_whitespace_text(self, client):
        resp = client.post("/api/textsort-suggestions", json={"text": "   "})
        assert resp.status_code == 400

class TestTypeMismatches:
    """Routes should reject wrong-typed values."""

    def test_inclusion_string_value(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": "five"})
        assert resp.status_code == 400

    def test_inclusion_boolean_value(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": True})
        # Booleans are explicitly rejected even though bool is a subclass of int.
        assert resp.status_code == 400

    def test_safe_thresholds_string_value(self, client):
        resp = client.post("/api/safe-thresholds", json={"safe_thresholds": "yes"})
        assert resp.status_code == 400

    def test_safe_thresholds_number_value(self, client):
        resp = client.post("/api/safe-thresholds", json={"safe_thresholds": 1})
        assert resp.status_code == 400

    def test_settings_volume_string(self, client):
        resp = client.put("/api/settings", json={"volume": "loud"})
        assert resp.status_code == 400

    def test_settings_theme_invalid(self, client):
        resp = client.put("/api/settings", json={"theme": "neon"})
        assert resp.status_code == 400

    def test_settings_calibration_fraction_string(self, client):
        resp = client.put("/api/settings", json={"calibration_fraction": "half"})
        assert resp.status_code == 400

    def test_settings_calibrate_count_string(self, client):
        resp = client.put("/api/settings", json={"calibrate_count": "ten"})
        assert resp.status_code == 400

    def test_settings_autopilot_top_greens_string(self, client):
        resp = client.put("/api/settings", json={"autopilot_top_greens": "many"})
        assert resp.status_code == 400

    def test_settings_autopilot_hard_reds_string(self, client):
        resp = client.put("/api/settings", json={"autopilot_hard_reds": "few"})
        assert resp.status_code == 400

    def test_fill_from_sort_threshold_string(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": [{"id": 1, "score": 0.5}],
                "threshold": "high",
            },
        )
        assert resp.status_code == 400

    def test_fill_from_sort_invalid_sides(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": [{"id": 1, "score": 0.5}],
                "threshold": 0.5,
                "sides": "left",
            },
        )
        assert resp.status_code == 400

    def test_labels_import_labels_not_list(self, client):
        resp = client.post("/api/labels/import", json={"labels": "not a list"})
        assert resp.status_code == 400

    def test_labels_import_labels_is_dict(self, client):
        resp = client.post("/api/labels/import", json={"labels": {"1": "good"}})
        assert resp.status_code == 400


class TestNonexistentResources:
    """Routes should return 404 for resources that don't exist."""

    def test_media_audio_nonexistent(self, client):
        resp = client.get("/api/medias/99999/audio")
        assert resp.status_code == 404

    def test_media_video_nonexistent(self, client):
        resp = client.get("/api/medias/99999/video")
        assert resp.status_code == 404

    def test_media_image_nonexistent(self, client):
        resp = client.get("/api/medias/99999/image")
        assert resp.status_code == 404

    def test_media_paragraph_nonexistent(self, client):
        resp = client.get("/api/medias/99999/paragraph")
        assert resp.status_code == 404

    def test_media_generic_nonexistent(self, client):
        resp = client.get("/api/medias/99999/media")
        assert resp.status_code == 404

    def test_vote_nonexistent_media(self, client):
        resp = client.post("/api/medias/99999/vote", json={"vote": "good"})
        assert resp.status_code == 404

    def test_delete_nonexistent_model(self, client):
        resp = client.delete("/api/detectors/registry/does_not_exist")
        assert resp.status_code == 404

    def test_rename_nonexistent_model(self, client):
        resp = client.put(
            "/api/detectors/registry/does_not_exist/rename",
            json={"name": "new"},
        )
        assert resp.status_code == 404

    def test_autorun_nonexistent_model(self, client):
        resp = client.put(
            "/api/detectors/registry/does_not_exist/autorun",
            json={"autorun": False},
        )
        assert resp.status_code == 404

    def test_unknown_exporter(self, client):
        resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "nonexistent",
            },
        )
        assert resp.status_code == 404

    def test_unknown_label_importer(self, client):
        resp = client.post("/api/label-importers/import/nonexistent", json={})
        assert resp.status_code == 404

    def test_unknown_dataset_importer(self, client):
        resp = client.post("/api/dataset/import/nonexistent", json={})
        assert resp.status_code == 404

    def test_media_id_zero(self, client):
        resp = client.get("/api/medias/0/audio")
        assert resp.status_code == 404

    def test_media_id_negative(self, client):
        resp = client.get("/api/medias/-1/audio")
        # Flask's int converter does not match negative numbers → 404
        assert resp.status_code == 404


class TestEmptyState:
    """Routes should handle empty dataset state gracefully."""

    def test_medias_list_when_empty(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.get("/api/medias/ids")
            assert resp.status_code == 200
            assert resp.get_json() == []
        finally:
            medias.update(saved)

    def test_sort_when_no_medias(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.post("/api/sort", json={"text": "test"})
            assert resp.status_code == 400
            assert "No medias" in resp.get_json()["error"]
        finally:
            medias.update(saved)

    def test_learned_sort_no_votes(self, client):
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400
        assert "need at least" in resp.get_json()["error"]

    def test_learned_sort_only_good_votes(self, client):
        good_votes[1] = None
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_learned_sort_only_bad_votes(self, client):
        bad_votes[1] = None
        resp = client.post("/api/learned-sort", json={"wait": True})
        assert resp.status_code == 400

    def test_labeling_progress_no_votes(self, client):
        resp = client.post("/api/labeling-progress")
        assert resp.status_code == 400

    def test_labels_export_no_votes(self, client):
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["labels"] == []

    def test_votes_empty_state(self, client):
        resp = client.get("/api/votes")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["good"] == []
        assert data["bad"] == []

    def test_dataset_status_when_empty(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.get("/api/dataset/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["loaded"] is False
            assert data["num_medias"] == 0
        finally:
            medias.update(saved)

    def test_diversity_tree_when_not_built(self, client):
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["id"] is None
        assert data["diversity_level"] == 0

    def test_textsort_suggestions_empty(self, client):
        resp = client.get("/api/textsort-suggestions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["suggestions"] == []


class TestMediaTypeMismatch:
    """Requesting wrong media type for an item should return 400."""

    def test_video_endpoint_on_audio_media(self, client):
        # Default test medias are audio type
        resp = client.get("/api/medias/1/video")
        assert resp.status_code == 400
        assert "not a video" in resp.get_json()["error"]

    def test_image_endpoint_on_audio_media_returns_waveform(self, client):
        # Audio medias now return a waveform thumbnail PNG
        resp = client.get("/api/medias/1/image")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_paragraph_endpoint_on_audio_media(self, client):
        resp = client.get("/api/medias/1/paragraph")
        assert resp.status_code == 400
        assert "not a text media" in resp.get_json()["error"]


class TestBoundaryValues:
    """Edge case values for numeric inputs."""

    def test_inclusion_clamped_to_max(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] <= 10

    def test_inclusion_clamped_to_min(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": -100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["inclusion"] >= -10

    def test_inclusion_zero(self, client):
        resp = client.post("/api/inclusion", json={"inclusion": 0})
        assert resp.status_code == 200
        assert resp.get_json()["inclusion"] == 0

    def test_volume_zero(self, client):
        resp = client.put("/api/settings", json={"volume": 0})
        assert resp.status_code == 200

    def test_volume_one(self, client):
        resp = client.put("/api/settings", json={"volume": 1.0})
        assert resp.status_code == 200

    def test_fill_from_sort_empty_results(self, client):
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": [],
                "threshold": 0.5,
                "sides": "both",
                "confirm": False,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["good_count"] == 0
        assert data["bad_count"] == 0

    def test_labels_import_empty_labels_list(self, client):
        resp = client.post("/api/labels/import", json={"labels": []})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["applied"] == 0
        assert data["skipped"] == 0

    def test_labels_import_with_invalid_label_values(self, client):
        """Labels with value other than good/bad should be skipped, not crash."""
        resp = client.post(
            "/api/labels/import",
            json={
                "labels": [
                    {"md5": "abc", "label": "neutral"},
                    {"md5": "def", "label": 123},
                    {"md5": "ghi"},  # missing label
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["skipped"] == 3


class TestModelRegistryEdgeCases:
    """Edge cases for model-registry operations."""

    def test_create_model_with_empty_name(self, client):
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "", "media_type": "audio"},
        )
        assert resp.status_code == 400

    def test_create_model_with_whitespace_name(self, client):
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "   ", "media_type": "audio"},
        )
        assert resp.status_code == 400

    def test_double_delete_model(self, client):
        """Deleting a model twice: second should 404."""
        resp = client.post(
            "/api/detectors/registry",
            json={"name": "del_twice", "media_type": "audio"},
        )
        detector_id = resp.get_json()["detector"]["id"]
        resp1 = client.delete(f"/api/detectors/registry/{detector_id}")
        assert resp1.status_code == 200
        resp2 = client.delete(f"/api/detectors/registry/{detector_id}")
        assert resp2.status_code == 404


class TestSettingsEdgeCases:
    """Edge cases for settings operations."""

    def test_update_multiple_settings_at_once(self, client):
        resp = client.put(
            "/api/settings",
            json={
                "volume": 0.7,
                "swipe_animation": False,
                "enrich_descriptions": True,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "volume" in data

    def test_update_with_unknown_key_ignored(self, client):
        """Unknown keys should be ignored, not cause errors."""
        resp = client.put(
            "/api/settings",
            json={
                "volume": 0.5,
                "unknown_setting": "value",
            },
        )
        assert resp.status_code == 200



class TestVoteEdgeCases:
    """Edge cases in voting behavior."""

    def test_vote_with_extra_fields_ignored(self, client):
        """Extra fields in vote request should not cause errors."""
        resp = client.post(
            "/api/medias/1/vote",
            json={
                "vote": "good",
                "confidence": 0.95,
                "note": "very relevant",
            },
        )
        assert resp.status_code == 200

    def test_rapid_toggle_votes(self, client):
        """Rapidly toggling a vote should leave consistent state."""
        for _ in range(10):
            client.post("/api/medias/1/vote", json={"vote": "good"})
        # After 10 toggles (even number), vote should be off
        resp = client.get("/api/votes")
        data = resp.get_json()
        assert 1 not in data["good"]

    def test_vote_all_medias(self, client):
        """Voting on every media should work without errors."""
        num = len(medias)
        for i in range(1, num + 1):
            vote = "good" if i % 2 == 0 else "bad"
            resp = client.post(f"/api/medias/{i}/vote", json={"vote": vote})
            assert resp.status_code == 200

        resp = client.get("/api/votes")
        data = resp.get_json()
        assert len(data["good"]) + len(data["bad"]) == num


class TestPathTraversalPrevention:
    """Label-file endpoints must reject paths outside the data directory."""

    def _make_label_file(self, paths):
        """Build an in-memory label JSON file with the given file paths."""
        labels = []
        for i, p in enumerate(paths):
            labels.append({"path": p, "label": "good" if i % 2 == 0 else "bad"})
        content = json.dumps({"labels": labels}).encode("utf-8")
        return io.BytesIO(content)

    # -- /api/label-file-sort ------------------------------------------------

    def test_label_file_sort_rejects_absolute_escape(self, client):
        """Absolute paths outside DATA_DIR must be skipped."""
        buf = self._make_label_file(["/etc/passwd", "/etc/shadow"])
        resp = client.post(
            "/api/label-file-sort",
            data={"file": (buf, "labels.json")},
            content_type="multipart/form-data",
        )
        # All paths rejected → too few valid files → 400
        assert resp.status_code == 400
        assert "2 valid" in resp.get_json()["error"] or "loaded 0" in resp.get_json()["error"]

    def test_label_file_sort_rejects_relative_traversal(self, client):
        """Relative paths that traverse out of DATA_DIR must be skipped."""
        buf = self._make_label_file(["../../etc/passwd", "../../../etc/shadow"])
        resp = client.post(
            "/api/label-file-sort",
            data={"file": (buf, "labels.json")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


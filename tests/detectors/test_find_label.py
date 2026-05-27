"""Tests for POST /api/find-label and the auto-detect / multi-find pipelines.

After the detector→detector migration, every "model" used by
find-label is a detector.  Its MLP is trained on demand from the
labelset stored on disk.
"""

from __future__ import annotations

import app as app_module
from helpers import setup_trainable_model_in_registry
from vtsearch.state import snapshot_medias


class TestFindLabel:
    """``POST /api/find-label`` against a detector in the registry."""

    def test_find_label_marks_all_items(self, client):
        """find-label should mark every loaded media as good or bad."""
        detector_id = setup_trainable_model_in_registry(
            "find-label-model",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["ok"] is True
        total = data["good_count"] + data["bad_count"]
        assert total == app_module.NUM_MEDIAS

    def test_find_label_missing_model_id(self, client):
        resp = client.post("/api/find-label", json={})
        # Schema validation: missing required `detector_id` → 422 with the
        # standard flask-smorest errors envelope.
        assert resp.status_code == 422
        assert "detector_id" in resp.get_json()["errors"]["json"]

    def test_find_label_unknown_model_id(self, client):
        resp = client.post("/api/find-label", json={"detector_id": "does-not-exist"})
        assert resp.status_code == 404

    def test_find_label_no_medias(self, client):
        # Build the model first while medias are loaded so the labelset has md5s.
        detector_id = setup_trainable_model_in_registry(
            "no-medias",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        saved = dict(app_module.medias)
        app_module.medias.clear()
        try:
            resp = client.post("/api/find-label", json={"detector_id": detector_id})
            assert resp.status_code == 400
        finally:
            app_module.medias.update(saved)

    def test_find_label_body_dataset_id_is_rejected(self, client):
        """Regression for C5: the body must not carry a ``dataset_id`` field.

        The dataset to score against comes exclusively from the
        ``X-Dataset-Id`` header set by ``before_request``. An extra
        ``dataset_id`` in the body must fail schema validation rather
        than silently override ``g._dataset_context``; which used to
        let a confused client wipe one detector's votes while the UI
        thought it was labeling a different dataset.
        """
        detector_id = setup_trainable_model_in_registry(
            "c5-regression",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        resp = client.post(
            "/api/find-label",
            json={"detector_id": detector_id, "dataset_id": "spoofed"},
        )
        # ``additionalProperties: false`` on the request schema turns
        # the unknown field into a 422 from flask-smorest.
        assert resp.status_code == 422, resp.get_json()
        assert "dataset_id" in resp.get_json()["errors"]["json"]


class TestAutoDetect:
    """``POST /api/auto-detect`` iterates detectors flagged for autorun."""

    def test_no_autorun_models_returns_400(self, client):
        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 400

    def test_autorun_model_runs(self, client):
        from vtsearch.settings import add_autorun_detector

        setup_trainable_model_in_registry(
            "auto-detect-model",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        add_autorun_detector("auto-detect-model")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["media_type"] == "audio"
        assert data["detectors_run"] == 1
        assert "auto-detect-model" in data["results"]
        result = data["results"]["auto-detect-model"]
        assert "hits" in result
        assert "negative_hits" in result
        assert len(result["hits"]) + len(result["negative_hits"]) == app_module.NUM_MEDIAS

    def test_autorun_filters_by_media_type(self, client):
        from vtsearch.settings import add_autorun_detector

        # Image-only model should be skipped on an audio dataset.
        setup_trainable_model_in_registry(
            "image-only",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
            media_type="image",
        )
        add_autorun_detector("image-only")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 400

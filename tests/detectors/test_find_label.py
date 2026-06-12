"""Tests for POST /api/find-label and the auto-detect / multi-find pipelines.

After the detector→detector migration, every "model" used by
find-label is a detector.  Its MLP is trained on demand from the
labelset stored on disk.
"""

from __future__ import annotations

import app as app_module
import pytest
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

    def test_find_label_honors_active_detector_inclusion(self, client):
        """find-label trains at the active detector context's inclusion.

        Inclusion lives on the DetectorContext (seeded from the user-settings
        default).  A strongly-inclusive setting cannot label *fewer* items Good
        than a strongly-exclusive one for the same detector + dataset, since
        inclusion drives both the MLP class weights and the threshold.  Setting
        it via the API also invalidates the cached MLP, so the next find-label
        retrains at the new value rather than reusing a stale model.
        """
        from tests import load_detector_and_wait

        detector_id = setup_trainable_model_in_registry(
            "incl-context",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        load_detector_and_wait(client, detector_id)

        client.post("/api/inclusion", json={"inclusion": 10})
        inclusive = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()

        client.post("/api/inclusion", json={"inclusion": -10})
        exclusive = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()

        assert inclusive["good_count"] >= exclusive["good_count"]

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


class TestFindModeIsPerDetector:
    """Find mode is per-detector state, not a process-wide global.

    Running ``/api/find-label`` on detector A used to flip a process-global
    ``_find_mode`` flag that was only ever cleared by explicitly *unloading* a
    detector.  Loading detector B to train it (or creating a new detector)
    left the flag stuck True, so ``sync_labels_to_loaded_detector`` silently
    dropped every vote on B: the right-pane Good/Bad panels stayed empty and
    Learned sort stayed greyed out even though the stripe still showed the
    in-memory votes.  Find mode now lives on each ``DetectorContext``, so a
    find pass on A never blocks vote syncing on B.
    """

    def test_find_label_on_one_detector_does_not_block_sync_on_another(self, client):
        from tests import load_detector_and_wait
        from vtscore.detectors.store import _detector_path, _read_detector
        from vtsearch.state import medias

        if len(medias) < 5:
            pytest.skip("Need at least 5 medias")
        ids = list(medias.keys())
        snap = snapshot_medias()

        mid_a = setup_trainable_model_in_registry(
            "find-leak-A",
            good_ids=[ids[0], ids[1]],
            bad_ids=[ids[2], ids[3]],
            snap=snap,
        )
        mid_b = setup_trainable_model_in_registry(
            "find-leak-B",
            good_ids=[ids[0]],
            bad_ids=[ids[1]],
            snap=snap,
        )

        # Use detector A to score the whole dataset: A enters find mode so its
        # scoring labels are NOT synced back over its training labelset.
        load_detector_and_wait(client, mid_a)
        resp = client.post("/api/find-label", json={"detector_id": mid_a})
        assert resp.status_code == 200, resp.get_json()

        # Switch to detector B and cast a genuine training vote.
        load_detector_and_wait(client, mid_b)
        target_id = ids[4]
        resp = client.post(f"/api/medias/{target_id}/vote", json={"target": "good"})
        assert resp.status_code == 200, resp.get_json()

        # B's on-disk labelset must have recorded the vote.  Before the fix,
        # A's find mode leaked through the global flag and the sync was skipped,
        # leaving B's labelset (and the Good/Bad panels) empty.
        data = _read_detector(_detector_path("find-leak-B"))
        assert data is not None
        good_md5s = {el["md5"] for el in data["labelset"]["labels"] if el["label"] == "good"}
        assert medias[target_id]["md5"] in good_md5s


class TestAutoDetect:
    """``POST /api/auto-detect`` iterates detectors flagged for Auto-Find."""

    def test_no_autofind_models_returns_400(self, client):
        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 400

    def test_autofind_model_runs(self, client):
        from vtsearch.settings import add_autofind_detector

        setup_trainable_model_in_registry(
            "auto-detect-model",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        add_autofind_detector("auto-detect-model")

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

    def test_autofind_filters_by_media_type(self, client):
        from vtsearch.settings import add_autofind_detector

        # Image-only model should be skipped on an audio dataset.
        setup_trainable_model_in_registry(
            "image-only",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
            media_type="image",
        )
        add_autofind_detector("image-only")

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 400

    def test_missing_detector_is_reported_not_silently_skipped(self, client):
        """A stale Auto-Find entry (detector file deleted) shows up in
        ``missing_detectors`` while the surviving detectors still run."""
        from vtsearch.settings import add_autofind_detector

        setup_trainable_model_in_registry(
            "auto-detect-survivor",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        add_autofind_detector("auto-detect-survivor")
        add_autofind_detector("gone-detector")  # no file on disk

        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["missing_detectors"] == ["gone-detector"]
        assert "auto-detect-survivor" in data["results"]

    def test_all_detectors_missing_returns_400_naming_them(self, client):
        from vtsearch.settings import add_autofind_detector

        add_autofind_detector("gone-detector")
        resp = client.post("/api/auto-detect", json={})
        assert resp.status_code == 400
        assert "gone-detector" in resp.get_json()["message"]

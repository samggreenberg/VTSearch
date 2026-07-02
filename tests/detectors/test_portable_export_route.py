"""Tests for POST /api/detectors/<id>/portable-bundle.

Exercises the standalone-bundle export end-to-end: train a detector on the
active dataset, stream the zip, and verify it carries a valid ONNX scorer plus a
manifest/README describing the embedder and threshold - and nothing else.
"""

from __future__ import annotations

import io
import json
import zipfile

import app as app_module
import pytest
from helpers import setup_trainable_model_in_registry
from vtsearch.state import snapshot_medias


def _export(client, detector_id):
    return client.post(f"/api/detectors/{detector_id}/portable-bundle")


class TestPortableExport:
    def test_export_returns_valid_bundle(self, client):
        import onnx  # noqa: PLC0415

        detector_id = setup_trainable_model_in_registry(
            "portable-export",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        resp = _export(client, detector_id)
        assert resp.status_code == 200, resp.get_json()
        assert resp.mimetype == "application/zip"

        with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
            assert sorted(zf.namelist()) == ["README.md", "detector.onnx", "manifest.json"]
            manifest = json.loads(zf.read("manifest.json"))
            onnx_model = onnx.load_from_string(zf.read("detector.onnx"))

        onnx.checker.check_model(onnx_model)
        assert manifest["format"] == "vtsearch-portable-detector"
        assert manifest["detector_name"] == "portable-export"
        # Audio test medias bind the default CLAP embedder (512-dim).
        assert manifest["embedder"]["name"] == "clap"
        assert manifest["embedder"]["embedding_dim"] == 512
        assert 0.0 <= manifest["scoring"]["threshold"] <= 1.0
        assert manifest["training_labels"] == {"good": 3, "bad": 3}
        assert manifest["contains_media_data"] is False

    def test_export_onnx_scores_match_find(self, client):
        """The exported ONNX must score the dataset identically to the live MLP."""
        ort = pytest.importorskip("onnxruntime")
        import numpy as np  # noqa: PLC0415

        from vtscore.detectors.model_loading import resolve_or_train_detector  # noqa: PLC0415
        from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415
        from vtscore.detectors.training import score_media_with_model  # noqa: PLC0415

        detector_id = setup_trainable_model_in_registry(
            "portable-parity",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        resp = _export(client, detector_id)
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
            onnx_bytes = zf.read("detector.onnx")

        snap = snapshot_medias()
        det_data = _read_detector(_detector_path("portable-parity"))
        mlp, _threshold, _diag = resolve_or_train_detector(detector_id, det_data, "audio", snap)
        assert mlp is not None
        live = {r["id"]: r["score"] for r in score_media_with_model(mlp, snap, embedder_name="clap")}

        # Re-embed the same medias and run them through the exported graph.
        ids = sorted(live)
        x = np.stack([np.asarray(snap[i]["embeddings"]["clap"], dtype=np.float32) for i in ids])
        session = ort.InferenceSession(onnx_bytes)
        onnx_scores = session.run(["score"], {"embedding": x})[0].ravel()
        for i, s in zip(ids, onnx_scores, strict=True):
            assert s == pytest.approx(live[i], abs=1e-4)

    def test_export_unknown_detector(self, client):
        assert _export(client, "does-not-exist").status_code == 404

    def test_export_no_medias(self, client):
        detector_id = setup_trainable_model_in_registry(
            "portable-no-medias",
            good_ids=[1, 2, 3],
            bad_ids=[18, 19, 20],
            snap=snapshot_medias(),
        )
        saved = dict(app_module.medias)
        app_module.medias.clear()
        try:
            assert _export(client, detector_id).status_code == 400
        finally:
            app_module.medias.update(saved)

    def test_export_detector_without_labels(self, client):
        from vtscore.detectors.registry import register_detector  # noqa: PLC0415
        from vtscore.detectors.store import _detector_path, _write_detector  # noqa: PLC0415

        _write_detector(
            _detector_path("portable-empty"),
            {"name": "portable-empty", "media_type": "audio", "examples": [], "labelset": {"labels": []}},
        )
        entry = register_detector(name="portable-empty", media_type="audio", num_training=0)
        resp = _export(client, entry["id"])
        assert resp.status_code == 400

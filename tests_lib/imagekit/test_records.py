"""imagekit.face.records: FaceRecord / FaceDetection and their CSV rows."""

from __future__ import annotations

import numpy as np
import pytest

from imagekit.face.records import (
    EMBEDDING_DIM,
    FaceDetection,
    FaceRecord,
    as_landmarks,
    detection_csv_header,
    detection_from_row,
    detection_to_row,
    embedding_quality,
    oss_compat,
    vista_compat,
)

_LM = np.array([[10, 20], [30, 20], [20, 30], [12, 40], [28, 40]], np.float32)


def _record(embedding=None, landmarks: np.ndarray | None = _LM) -> FaceRecord:
    vec = np.arange(EMBEDDING_DIM, dtype=np.float32) if embedding is None else embedding
    return FaceRecord.from_detection(
        FaceDetection((1.0, 2.0, 41.0, 52.0), landmarks, 0.875), vec, "oss-test:1", "img.jpg"
    )


class TestHelpers:
    def test_compat_tags(self):
        assert oss_compat("magface-ir100", 1) == "oss-magface-ir100:1"
        assert vista_compat(3) == "vistaface:3"
        assert vista_compat("7") == "vistaface:7"

    def test_quality_is_the_l2_norm(self):
        assert embedding_quality(np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_as_landmarks(self):
        assert as_landmarks(None) is None
        assert as_landmarks([1, 2, 3]) is None
        assert as_landmarks(np.full((5, 2), np.nan)) is None
        flat = as_landmarks(_LM.reshape(-1).tolist())
        assert flat is not None and flat.shape == (5, 2)


class TestFaceRecord:
    def test_from_detection_fills_quality_and_keeps_raw_vector(self):
        rec = _record()
        assert rec.embedding.dtype == np.float32 and rec.embedding.shape == (EMBEDDING_DIM,)
        assert rec.quality == pytest.approx(float(np.linalg.norm(rec.embedding)))
        assert rec.compat == "oss-test:1" and rec.media_id == "img.jpg"
        assert rec.detection.bbox == rec.bbox and rec.detection.confidence == rec.confidence

    def test_unit_embedding_is_a_normalised_copy(self):
        rec = _record()
        unit = rec.unit_embedding()
        assert np.linalg.norm(unit) == pytest.approx(1.0)
        assert unit is not rec.embedding and rec.quality > 1.0
        zero = _record(embedding=np.zeros(EMBEDDING_DIM, np.float32))
        assert zero.quality == 0.0 and not zero.unit_embedding().any()

    def test_row_round_trip_without_the_embedding(self):
        rec = _record()
        row = rec.to_row(face_index=2)
        assert len(row) == len(FaceRecord.csv_header()) and row[1] == "2"
        back = FaceRecord.from_row(row)
        assert back.media_id == "img.jpg" and back.face_index == 2
        assert back.bbox == pytest.approx(rec.bbox) and back.confidence == pytest.approx(0.875)
        assert back.quality == pytest.approx(rec.quality, abs=1e-5) and back.compat == "oss-test:1"
        assert back.landmarks is not None
        np.testing.assert_allclose(back.landmarks, _LM)
        assert back.embedding.shape == (0,)
        filled = FaceRecord.from_row(row, embedding=rec.embedding)
        np.testing.assert_array_equal(filled.embedding, rec.embedding)

    def test_missing_landmarks_round_trip_as_empty_cells(self):
        rec = _record(landmarks=None)
        row = rec.to_row()
        assert row[-10:] == [""] * 10
        assert FaceRecord.from_row(row).landmarks is None

    def test_wrong_width_row_is_refused(self):
        with pytest.raises(ValueError, match="cells"):
            FaceRecord.from_row(["a", "b"])


class TestDetectionRows:
    def test_round_trip(self):
        det = FaceDetection((1.5, 2.5, 30.0, 40.0), _LM, 0.5)
        row = detection_to_row("m/1.png", 3, det)
        assert len(row) == len(detection_csv_header())
        media_id, index, back = detection_from_row(row)
        assert (media_id, index) == ("m/1.png", 3)
        assert back.bbox == pytest.approx(det.bbox) and back.confidence == 0.5
        assert back.landmarks is not None
        np.testing.assert_allclose(back.landmarks, _LM)
        assert back.width == pytest.approx(28.5) and back.height == pytest.approx(37.5)

    def test_without_landmarks(self):
        det = FaceDetection((0, 0, 1, 1), None, 1.0)
        assert not det.has_landmarks
        assert detection_from_row(detection_to_row("x", 0, det))[2].landmarks is None

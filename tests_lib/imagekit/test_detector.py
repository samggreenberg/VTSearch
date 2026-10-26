"""imagekit.face.detection: the SCRFD and RetinaFace decoders on synthetic ONNX graphs, and real weights when present."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from imagekit import RuntimeOptions
from imagekit.face.detection import DEFAULT_CONFIG_PATH, DetectorConfig, FaceDetector
from imagekit.runtime import CUDA_PROVIDER
from tests_lib.imagekit._synthetic_models import (
    build_retinaface_model,
    build_scrfd_model,
    scrfd_expected,
    scrfd_face,
    write_retinaface_config,
    write_scrfd_config,
)

KPS = ((-5, -5), (5, -5), (0, 0), (-4, 6), (4, 6))
FACE_A = scrfd_face(0, 20, 30, 0.9, (16, 16, 16, 16), kps=KPS)
FACE_B = scrfd_face(2, 10, 5, 0.7, (40, 40, 40, 40), kps=KPS)
#: Nearly the same box as FACE_A, one cell over, lower score: NMS must drop it.
FACE_A_DUP = scrfd_face(0, 21, 30, 0.6, (24, 16, 8, 16), kps=KPS)

BLACK = np.zeros((640, 640, 3), np.uint8)

_MODEL_DIR = os.environ.get("IMAGEKIT_TEST_MODEL_DIR")
_FACE_IMAGE = os.environ.get("IMAGEKIT_TEST_FACE_IMAGE")
needs_real_models = pytest.mark.skipif(not _MODEL_DIR, reason="IMAGEKIT_TEST_MODEL_DIR not set")


@pytest.fixture
def detector(tmp_path: Path) -> FaceDetector:
    build_scrfd_model(tmp_path / "det.onnx", [FACE_A, FACE_B, FACE_A_DUP])
    det = FaceDetector()
    det.initialize(RuntimeOptions(), tmp_path / "det.onnx")
    return det


class TestLifecycle:
    def test_detect_before_initialize_raises(self):
        det = FaceDetector()
        with pytest.raises(RuntimeError, match="Must initialize FaceDetector first!"):
            det.detect(BLACK)
        with pytest.raises(RuntimeError, match="Must initialize FaceDetector first!"):
            det.detect_buffer(b"")
        with pytest.raises(RuntimeError):
            _ = det.config
        assert not det.is_initialized

    def test_initialize_raises_on_a_missing_or_bad_model(self, tmp_path):
        det = FaceDetector()
        with pytest.raises(FileNotFoundError):
            det.initialize(RuntimeOptions(), tmp_path / "nope.onnx")
        (tmp_path / "bad.onnx").write_bytes(b"not a model")
        with pytest.raises(Exception):
            det.initialize(RuntimeOptions(), tmp_path / "bad.onnx")
        assert not det.is_initialized

    def test_gpu_requested_but_unavailable_raises(self, tmp_path):
        import onnxruntime as ort

        if CUDA_PROVIDER in ort.get_available_providers():
            pytest.skip("this onnxruntime has CUDA")
        build_scrfd_model(tmp_path / "det.onnx")
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
            FaceDetector().initialize(RuntimeOptions(use_gpu=True), tmp_path / "det.onnx")

    def test_packaged_default_config_is_used_without_a_sidecar(self, detector):
        cfg = detector.config
        assert cfg.name == "scrfd_10g_gnkps" and cfg.feat_stride == (8, 16, 32) and cfg.topk == 400
        assert DEFAULT_CONFIG_PATH.is_file()

    def test_sidecar_and_explicit_configs_win(self, tmp_path):
        build_scrfd_model(tmp_path / "det.onnx")
        write_scrfd_config(tmp_path / "det.json", topk=7)
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "det.onnx")
        assert det.config.topk == 7
        write_scrfd_config(tmp_path / "other.json", topk=3)
        det.initialize(RuntimeOptions(), tmp_path / "det.onnx", tmp_path / "other.json")
        assert det.config.topk == 3

    def test_a_bundle_directory_is_accepted(self, tmp_path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        build_scrfd_model(bundle / "model.onnx", [FACE_A])
        det = FaceDetector()
        det.initialize(RuntimeOptions(), bundle)
        assert det.model_path == bundle / "model.onnx"
        assert len(det.detect(BLACK)) == 1


class TestScrfdDecode:
    def test_boxes_landmarks_and_order(self, detector):
        dets = detector.detect(BLACK)
        assert [round(d.confidence, 3) for d in dets] == [0.9, 0.7]  # the duplicate is suppressed
        for det, face in zip(dets, (FACE_A, FACE_B)):
            expected = scrfd_expected(face)
            assert det.bbox == pytest.approx(expected["bbox"])
            assert det.landmarks is not None
            np.testing.assert_allclose(det.landmarks, expected["landmarks"])

    def test_coordinates_map_back_through_the_letterbox_scale(self, detector):
        dets = detector.detect(np.zeros((320, 320, 3), np.uint8))
        expected = np.asarray(scrfd_expected(FACE_A)["bbox"]) / 2
        assert dets[0].bbox == pytest.approx(expected.tolist())
        assert dets[0].landmarks is not None
        np.testing.assert_allclose(dets[0].landmarks, scrfd_expected(FACE_A)["landmarks"] / 2)

    def test_boxes_are_clamped_to_the_image(self, detector):
        # 640 wide, 240 tall: scale 1, so FACE_B's box (y up to 200) fits but FACE_A's (y to 256) is clamped.
        dets = detector.detect(np.zeros((240, 640, 3), np.uint8))
        assert dets[0].bbox[3] == 240.0

    def test_threshold_override_and_config_threshold(self, detector, tmp_path):
        assert len(detector.detect(BLACK, confidence_threshold=0.8)) == 1
        assert detector.detect(BLACK, confidence_threshold=0.95) == []
        build_scrfd_model(tmp_path / "strict.onnx", [FACE_A, FACE_B])
        write_scrfd_config(tmp_path / "strict.json", score_threshold=0.8)
        strict = FaceDetector()
        strict.initialize(RuntimeOptions(), tmp_path / "strict.onnx")
        assert len(strict.detect(BLACK)) == 1

    def test_topk_caps_the_result(self, tmp_path):
        build_scrfd_model(tmp_path / "det.onnx", [FACE_A, FACE_B])
        write_scrfd_config(tmp_path / "det.json", topk=1)
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "det.onnx")
        found = det.detect(BLACK)
        assert len(found) == 1 and found[0].confidence == pytest.approx(0.9)

    def test_images_below_min_image_size_give_nothing(self, detector):
        assert detector.detect(np.zeros((8, 100, 3), np.uint8)) == []

    def test_a_model_without_keypoints_gives_no_landmarks(self, tmp_path):
        build_scrfd_model(tmp_path / "det.onnx", [FACE_A], use_kps=False)
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "det.onnx")
        (found,) = det.detect(BLACK)
        assert found.landmarks is None and found.bbox == pytest.approx(scrfd_expected(FACE_A)["bbox"])

    def test_batched_head_outputs_decode_the_same(self, tmp_path):
        build_scrfd_model(tmp_path / "det.onnx", [FACE_A], batch_dim=True)
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "det.onnx")
        assert det.detect(BLACK)[0].bbox == pytest.approx(scrfd_expected(FACE_A)["bbox"])

    def test_a_stride_list_that_disagrees_with_the_heads_is_refused(self, tmp_path):
        build_scrfd_model(tmp_path / "det.onnx", [FACE_A])
        write_scrfd_config(tmp_path / "det.json", feat_stride=[8, 16])
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "det.onnx")
        with pytest.raises(ValueError, match="outputs"):
            det.detect(BLACK)

    def test_detect_buffer(self, detector):
        ok, png = cv2.imencode(".png", BLACK)
        assert ok
        assert len(detector.detect_buffer(png.tobytes())) == 2
        with pytest.raises(ValueError):
            detector.detect_buffer(b"garbage")


class TestRetinaFaceDecode:
    def test_a_prior_decodes_to_its_own_box(self, tmp_path):
        # Level 0 (stride 8) on a 64 px image: cell (x=4, y=3), first anchor (16 px).
        index = (3 * 8 + 4) * 2
        build_retinaface_model(tmp_path / "rf.onnx", (64, 64), [(index, 0.9)])
        write_retinaface_config(tmp_path / "rf.json")
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "rf.onnx")
        assert det.config.backend == "retinaface"
        (found,) = det.detect(np.zeros((64, 64, 3), np.uint8))
        assert found.bbox == pytest.approx((28.0, 20.0, 44.0, 36.0)) and found.confidence == pytest.approx(0.9)
        assert found.landmarks is not None
        np.testing.assert_allclose(found.landmarks, np.full((5, 2), (36.0, 28.0)))

    def test_without_landmark_head(self, tmp_path):
        build_retinaface_model(tmp_path / "rf.onnx", (64, 64), [(0, 0.8)], with_landmarks=False)
        write_retinaface_config(tmp_path / "rf.json")
        det = FaceDetector()
        det.initialize(RuntimeOptions(), tmp_path / "rf.onnx")
        (found,) = det.detect(np.zeros((64, 64, 3), np.uint8))
        assert found.landmarks is None


class TestDetectorConfig:
    def test_defaults_and_backend(self):
        cfg = DetectorConfig.from_dict({})
        assert cfg.backend == "scrfd" and cfg.iou_threshold == 0.45
        assert DetectorConfig.from_dict({"model": {"name": "RetinaFace-R50"}}).backend == "retinaface"

    def test_model_must_be_an_object(self):
        with pytest.raises(ValueError):
            DetectorConfig.from_dict({"model": 3})


@needs_real_models
class TestRealWeights:
    """Run only with ``IMAGEKIT_TEST_MODEL_DIR`` (holding ``det_10g.onnx``); never in the default suite."""

    @pytest.fixture
    def real(self) -> FaceDetector:
        assert _MODEL_DIR is not None
        det = FaceDetector()
        det.initialize(RuntimeOptions(num_threads=2), Path(_MODEL_DIR) / "det_10g.onnx")
        return det

    def test_noise_has_no_faces(self, real):
        noise = np.random.default_rng(42).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        assert real.detect(noise) == []

    @pytest.mark.skipif(not _FACE_IMAGE, reason="IMAGEKIT_TEST_FACE_IMAGE not set")
    def test_a_real_face_is_found_with_landmarks(self, real):
        from imagekit import read_image_tensor

        assert _FACE_IMAGE is not None
        image = read_image_tensor(_FACE_IMAGE)
        found = real.detect(image)
        assert found and found[0].confidence > 0.5
        assert found[0].landmarks is not None and found[0].landmarks.shape == (5, 2)
        x1, y1, x2, y2 = found[0].bbox
        assert 0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]

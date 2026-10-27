"""imagekit.face.extraction: the embedder on a synthetic ONNX network, and real weights when present."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from imagekit import RuntimeOptions
from imagekit.face.extraction import (
    ARCFACE_CONFIG_PATH,
    ARCFACE_MBF_CONFIG_PATH,
    MAGFACE_CONFIG_PATH,
    ExtractorConfig,
    FaceEmbeddingExtractor,
)
from imagekit.face.face_align import norm_crop
from imagekit.face.records import FaceDetection
from tests_lib.imagekit._real_models import FACE_IMAGE as _FACE_IMAGE
from tests_lib.imagekit._real_models import MODEL_DIR as _MODEL_DIR
from tests_lib.imagekit._real_models import detector_path, embedder_paths, expected_compat
from tests_lib.imagekit._synthetic_models import build_embedder_model, expected_embedding, write_embedder_config

LM = np.array([[200, 150], [260, 150], [230, 190], [205, 220], [255, 220]], np.float32)
BOX = (170.0, 110.0, 290.0, 250.0)


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(42).integers(0, 255, (400, 400, 3), dtype=np.uint8)


@pytest.fixture
def extractor(tmp_path: Path) -> FaceEmbeddingExtractor:
    build_embedder_model(tmp_path / "emb.onnx")
    write_embedder_config(tmp_path / "emb.json")
    ext = FaceEmbeddingExtractor()
    ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx")
    return ext


class TestLifecycle:
    def test_extract_before_initialize_raises(self, image):
        ext = FaceEmbeddingExtractor()
        with pytest.raises(RuntimeError, match="Must initialize FaceEmbeddingExtractor first!"):
            ext.extract(FaceDetection(BOX, LM, 1.0), image)
        with pytest.raises(RuntimeError, match="Must initialize"):
            ext.extract_buffer(FaceDetection(BOX, LM, 1.0), b"")
        with pytest.raises(RuntimeError):
            ext.get_compatibility()

    def test_a_config_naming_the_space_is_required(self, tmp_path):
        build_embedder_model(tmp_path / "emb.onnx")
        ext = FaceEmbeddingExtractor()
        with pytest.raises(FileNotFoundError, match="emb.json"):
            ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx")
        (tmp_path / "emb.json").write_text(json.dumps({"model": {"name": "x"}}))
        with pytest.raises(ValueError, match="embedding space"):
            ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx")
        assert not ext.is_initialized

    def test_name_plus_version_derives_the_tag(self, tmp_path):
        build_embedder_model(tmp_path / "emb.onnx")
        (tmp_path / "emb.json").write_text(json.dumps({"model": {"name": "magface-ir100", "version": 2}}))
        ext = FaceEmbeddingExtractor()
        ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx")
        assert ext.get_compatibility() == "oss-magface-ir100:2"

    def test_explicit_config_path_wins(self, tmp_path):
        build_embedder_model(tmp_path / "emb.onnx")
        write_embedder_config(tmp_path / "emb.json", compatibility="sidecar:1")
        write_embedder_config(tmp_path / "x.json", compatibility="explicit:1")
        ext = FaceEmbeddingExtractor()
        ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx", tmp_path / "x.json")
        assert ext.get_compatibility() == "explicit:1"

    def test_packaged_reference_configs_parse(self):
        for path in (MAGFACE_CONFIG_PATH, ARCFACE_CONFIG_PATH, ARCFACE_MBF_CONFIG_PATH):
            cfg = ExtractorConfig.from_dict(json.loads(path.read_text()))
            assert cfg.compatibility.startswith("oss-") and cfg.embedding_dim == 512
            assert [t["name"] for t in cfg.transforms] == ["Align", "Normalize", "ToTensor"]


class TestExtract:
    def test_raw_vector_matches_the_aligned_crop_and_is_not_normalised(self, extractor, image):
        vec = extractor.extract(FaceDetection(BOX, LM, 1.0), image)
        assert vec.shape == (512,) and vec.dtype == np.float32
        np.testing.assert_allclose(vec, expected_embedding(norm_crop(image, LM)), rtol=1e-4, atol=1e-4)
        assert not np.isclose(np.linalg.norm(vec), 1.0)

    def test_missing_landmarks_use_the_box_crop(self, extractor, image):
        vec = extractor.extract(FaceDetection(BOX, None, 1.0), image)
        np.testing.assert_allclose(vec, expected_embedding(norm_crop(image, None, bbox=BOX)), rtol=1e-4, atol=1e-4)

    def test_extract_many_equals_one_by_one(self, extractor, image):
        dets = [FaceDetection(BOX, LM, 1.0), FaceDetection((0, 0, 50, 50), None, 0.5), FaceDetection(BOX, LM + 3, 0.9)]
        batched = extractor.extract_many(dets, image)
        assert len(batched) == 3
        for det, vec in zip(dets, batched):
            np.testing.assert_allclose(vec, extractor.extract(det, image), rtol=1e-5)
        assert extractor.extract_many([], image) == []

    def test_a_fixed_batch_model_is_fed_one_crop_at_a_time(self, tmp_path, image):
        build_embedder_model(tmp_path / "emb.onnx", fixed_batch=1)
        write_embedder_config(tmp_path / "emb.json")
        ext = FaceEmbeddingExtractor()
        ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx")
        dets = [FaceDetection(BOX, LM, 1.0), FaceDetection(BOX, None, 1.0)]
        vecs = ext.extract_many(dets, image)
        np.testing.assert_allclose(vecs[0], expected_embedding(norm_crop(image, LM)), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(vecs[1], expected_embedding(norm_crop(image, None, bbox=BOX)), rtol=1e-4, atol=1e-4)

    def test_extract_buffer(self, extractor, image):
        ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert ok
        vec = extractor.extract_buffer(FaceDetection(BOX, LM, 1.0), png.tobytes())
        np.testing.assert_allclose(vec, extractor.extract(FaceDetection(BOX, LM, 1.0), image), rtol=1e-5)

    def test_a_width_the_config_did_not_promise_is_refused(self, tmp_path, image):
        build_embedder_model(tmp_path / "emb.onnx")
        write_embedder_config(tmp_path / "emb.json", embedding_dim=256)
        ext = FaceEmbeddingExtractor()
        ext.initialize(RuntimeOptions(), tmp_path / "emb.onnx")
        with pytest.raises(ValueError, match="512-d"):
            ext.extract(FaceDetection(BOX, LM, 1.0), image)


@pytest.mark.skipif(not _MODEL_DIR, reason="IMAGEKIT_TEST_MODEL_DIR not set")
class TestRealWeights:
    """Run only with ``IMAGEKIT_TEST_MODEL_DIR`` (`_real_models`); never in the default suite."""

    @pytest.fixture
    def real(self) -> FaceEmbeddingExtractor:
        assert _MODEL_DIR is not None
        ext = FaceEmbeddingExtractor()
        ext.initialize(RuntimeOptions(num_threads=2), *embedder_paths())
        return ext

    def test_shape_and_compat(self, real, image):
        vec = real.extract(FaceDetection(BOX, LM, 1.0), image)
        assert vec.shape == (512,) and vec.dtype == np.float32 and np.isfinite(vec).all()
        assert real.get_compatibility() == expected_compat()

    @pytest.mark.skipif(not _FACE_IMAGE, reason="IMAGEKIT_TEST_FACE_IMAGE not set")
    def test_a_face_and_its_mirror_embed_alike(self, real):
        from imagekit import read_image_tensor
        from imagekit.face.detection import FaceDetector

        assert _MODEL_DIR is not None and _FACE_IMAGE is not None
        det = FaceDetector()
        det.initialize(RuntimeOptions(num_threads=2), detector_path())
        image = read_image_tensor(_FACE_IMAGE)
        mirror = np.ascontiguousarray(image[:, ::-1])
        a = real.extract(det.detect(image)[0], image)
        b = real.extract(det.detect(mirror)[0], mirror)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos > 0.8

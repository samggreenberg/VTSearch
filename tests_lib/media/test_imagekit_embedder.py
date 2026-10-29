"""The ``face`` embedder (the imagekit provider seam) and the bridge behind it, on a fake pipeline."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from imagekit.face.compat import CompatibilityError
from imagekit.face.records import FaceDetection, FaceRecord
from vtscore.media import embedders_for_type, get_embedder
from vtscore.media.face import imagekit_bridge as bridge
from vtscore.media.face.embedder_imagekit import ImagekitFaceEmbedder

LM = np.array([[30, 30], [70, 30], [50, 55], [35, 75], [65, 75]], np.float32)


class FakePipeline:
    def __init__(self, n_faces=1):
        self.n_faces = n_faces
        self.seen: list[tuple] = []

    def process(self, image, *, confidence_threshold=None, media_id=""):
        self.seen.append((image.shape, media_id))
        rng = np.random.default_rng(7)
        return [
            FaceRecord.from_detection(
                FaceDetection((10 + i, 10, 90, 90), LM, 0.9 - i / 10),
                rng.standard_normal(512) * 4,
                "fake:1",
                media_id,
                i,
            )
            for i in range(self.n_faces)
        ]

    def compatibility(self):
        return "fake:1"


@pytest.fixture
def fake(monkeypatch) -> FakePipeline:
    pipe = FakePipeline()
    monkeypatch.setattr(bridge, "_pipeline", pipe)
    monkeypatch.setattr(bridge, "_compat", "fake:1")
    return pipe


def _crop(meta=None) -> dict:
    img = Image.fromarray(np.random.default_rng(3).integers(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    media = {"id": 7, "media_type": "face", "filename": "x_face_1.png", "media_bytes": buf.getvalue(), "embeddings": {}}
    if meta is not None:
        media["custom_metadata"] = meta
    return media


class TestDescriptor:
    def test_identity_and_flags(self):
        emb = ImagekitFaceEmbedder()
        assert emb.name == "face" and emb.media_type_id == "face" and emb.embedding_dim == 512
        assert emb.is_default is True and emb.supports_text is False and emb.supports_patch_regions is False
        assert emb.license_notice and "IMAGEKIT_FACE_PROVIDER" in emb.license_notice
        assert emb.embed_text("a face") is None

    def test_it_is_the_one_face_embedder(self):
        assert get_embedder("face").media_type_id == "face"
        assert [e.name for e in embedders_for_type("face")] == ["face"]

    def test_model_id_is_the_compat_once_loaded_and_never_loads(self, monkeypatch):
        emb = ImagekitFaceEmbedder()
        assert emb.model_id is None
        monkeypatch.setattr(bridge, "_compat", "fake:1")
        assert emb.model_id == "fake:1"


class TestBridge:
    def test_unconfigured_is_a_named_error(self, monkeypatch):
        monkeypatch.delenv(bridge.CONFIG_ENV_VAR, raising=False)
        with pytest.raises(bridge.FaceProviderUnconfigured, match=bridge.CONFIG_ENV_VAR):
            bridge.get_face_pipeline()
        assert bridge.loaded_compat() == ""

    def test_runtime_options_follow_the_app_device(self, monkeypatch):
        import vtscore.config as config

        monkeypatch.setattr(config, "resolve_device", lambda: "cuda:1")
        opts = bridge.runtime_options()
        assert opts.use_gpu is True and opts.gpu_device_id == 1 and opts.num_threads >= 1
        monkeypatch.setattr(config, "resolve_device", lambda: "cpu")
        assert bridge.runtime_options().use_gpu is False

    def test_builds_once_from_the_config_and_provider(self, monkeypatch, tmp_path):
        import json

        from imagekit.face import register_provider

        built: list[dict] = []

        class Recording:
            def initialize(self, opts, config):
                built.append(config)

            def process(self, image, *, confidence_threshold=None, media_id=""):
                return []

            def process_buffer(self, buffer, *, confidence_threshold=None, media_id=""):
                return []

            def compatibility(self):
                return "rec:2"

        register_provider("recording", Recording)
        cfg = tmp_path / "faces.json"
        (tmp_path / "d.onnx").write_bytes(b"")
        (tmp_path / "e.onnx").write_bytes(b"")
        cfg.write_text(json.dumps({"detector": {"model_bundle": "d.onnx"}, "embedder": {"model": "e.onnx"}}))
        monkeypatch.setenv(bridge.CONFIG_ENV_VAR, str(cfg))
        monkeypatch.setenv("IMAGEKIT_FACE_PROVIDER", "recording")
        first = bridge.get_face_pipeline()
        assert bridge.get_face_pipeline() is first and len(built) == 1
        assert built[0]["detector"]["model_bundle"] == str(tmp_path / "d.onnx")
        assert bridge.loaded_compat() == "rec:2" and bridge.provider_name() == "recording"


class TestEmbed:
    def test_embeds_the_best_face_the_provider_finds_in_the_crop(self, fake):
        emb = ImagekitFaceEmbedder()
        vec = emb.embed_media(_crop())
        assert vec is not None and vec.shape == (512,) and np.linalg.norm(vec) == pytest.approx(1.0)
        assert fake.seen == [((100, 100, 3), "7")]
        expected = fake.process(np.zeros((100, 100, 3), np.uint8))[0].unit_embedding()
        np.testing.assert_allclose(vec, expected, rtol=1e-5)

    def test_a_crop_with_no_face_gets_no_vector(self, monkeypatch):
        monkeypatch.setattr(bridge, "_pipeline", FakePipeline(n_faces=0))
        monkeypatch.setattr(bridge, "_compat", "fake:1")
        assert ImagekitFaceEmbedder().embed_media(_crop()) is None

    def test_the_same_space_passes_and_another_is_refused(self, fake):
        emb = ImagekitFaceEmbedder()
        assert emb.embed_media(_crop({"Face Compat": "fake:1"})) is not None
        with pytest.raises(CompatibilityError, match="fake:1"):
            emb.embed_media(_crop({"Face Compat": "extface:3"}))

    def test_bulk_matches_single(self, fake):
        emb = ImagekitFaceEmbedder()
        vectors = emb.embed_media_bulk([_crop(), _crop()])
        assert len(vectors) == 2 and vectors[0] is not None and vectors[1] is not None
        np.testing.assert_allclose(vectors[0], vectors[1])

    def test_the_embed_stage_reports_a_mixed_space(self, fake, monkeypatch):
        from vtscore.datasets.stages.embedding import embed_missing

        medias = {1: _crop({"Face Compat": "extface:3"})}
        failures: dict = {}
        # The conftest stubs ``load_models`` on every registered singleton, so
        # hand it the fake pipeline directly, as the other embedder tests do.
        monkeypatch.setattr(get_embedder("face"), "_model", fake)
        embed_missing(medias, "face", failures=failures)
        assert medias[1]["embeddings"] == {} and "CompatibilityError" in failures[1]

    def test_unreadable_bytes_decline_the_item(self, fake):
        media = _crop()
        media["media_bytes"] = b"garbage"
        assert ImagekitFaceEmbedder().embed_media(media) is None

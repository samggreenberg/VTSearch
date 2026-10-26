"""The image2face converter's ``imagekit`` provider: detect + embed in one pass, vectors ride with the crops.

The provider seam is exercised against a fake pipeline installed into the
bridge (no models, no ``IMAGEKIT_FACE_CONFIG``), so what is tested is the
app's side of the contract: the crop, its filed vector, its metadata, the
runner carrying the vector into the media dict, and the degrade-to-nothing
paths.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest
from PIL import Image

from imagekit.face.records import FaceDetection, FaceRecord
from vtscore.converters.image2face import PROVIDERS, Image2FaceMediaConverter
from vtscore.converters.runner import run_converters_on_folder
from vtscore.media.face import imagekit_bridge as bridge
from vtscore.media.provenance import SOURCE_BOX_FIELD

LM = np.array([[60, 60], [90, 60], [75, 80], [62, 100], [88, 100]], np.float32)


class FakePipeline:
    """Returns fixed faces for any image, best first, tagged ``fake:1``."""

    def __init__(self, records=None):
        self.calls: list[tuple] = []
        self.records = records

    def process(self, image, *, confidence_threshold=None, media_id=""):
        self.calls.append((image.shape, confidence_threshold, media_id))
        if self.records is not None:
            return list(self.records)
        rng = np.random.default_rng(42)
        return [
            FaceRecord.from_detection(
                FaceDetection((50, 40, 100, 110), LM, 0.9), rng.standard_normal(512) * 3, "fake:1", media_id, 0
            ),
            FaceRecord.from_detection(
                FaceDetection((150, 150, 190, 195), None, 0.6), rng.standard_normal(512), "fake:1", media_id, 1
            ),
        ]

    def compatibility(self):
        return "fake:1"


@pytest.fixture
def fake(monkeypatch) -> FakePipeline:
    pipe = FakePipeline()
    monkeypatch.setattr(bridge, "_pipeline", pipe)
    monkeypatch.setattr(bridge, "_compat", "fake:1")
    return pipe


def _png(width=200, height=200) -> bytes:
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _params(**overrides):
    params = {
        "threshold": 0.5,
        "padding": 0.0,
        "min_size": 10,
        "align": "none",
        "sideways": "false",
        "provider": "imagekit",
    }
    params.update(overrides)
    return params


class TestParams:
    def test_provider_defaults_to_mtcnn(self):
        conv = Image2FaceMediaConverter()
        assert conv._resolve_params(None).provider == "mtcnn"
        assert PROVIDERS == ("mtcnn", "imagekit")
        field = next(f for f in conv.fields if f.key == "provider")
        assert field.options == ["mtcnn", "imagekit"] and field.default == "mtcnn"

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(ValueError, match="provider"):
            Image2FaceMediaConverter()._resolve_params({"provider": "nope"})


class TestConvert:
    def test_each_face_leaves_with_its_vector_and_tags(self, fake):
        conv = Image2FaceMediaConverter()
        outputs = conv.convert_normalized({"filename": "group.png", "media_bytes": _png()}, _params())
        assert fake.calls == [((200, 200, 3), 0.5, "group.png")]
        assert [o["filename"] for o in outputs] == ["group_face_1.png", "group_face_2.png"]
        first = outputs[0]
        assert first["embedder"] == "imagekit_face" and set(first["embeddings"]) == {"imagekit_face"}
        vec = first["embeddings"]["imagekit_face"]
        assert vec.shape == (512,) and np.linalg.norm(vec) == pytest.approx(1.0)
        meta = first["custom_metadata"]
        assert meta["Face Compat"] == "fake:1" and meta["Detection Confidence"] == 0.9
        assert meta["Face Quality"] > 1.0
        assert first["width"] == 50 and first["height"] == 70
        assert first[SOURCE_BOX_FIELD] == pytest.approx([0.25, 0.2, 0.5, 0.55])

    def test_thresholds_and_min_size_still_apply(self, fake):
        conv = Image2FaceMediaConverter()
        kept = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(threshold=0.8))
        assert len(kept) == 1 and fake.calls[-1][1] == 0.8
        assert conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(min_size=60)) == []

    def test_alignment_uses_the_provider_landmarks(self, fake):
        conv = Image2FaceMediaConverter()
        plain = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(align="none"))
        turned = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(align="similarity"))
        assert turned[0]["width"] == turned[0]["height"]  # the similarity crop is a square at the fit's scale
        assert plain[0]["embeddings"]["imagekit_face"] is not None and turned[0]["embedder"] == "imagekit_face"

    def test_unconfigured_provider_yields_nothing_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.delenv(bridge.CONFIG_ENV_VAR, raising=False)
        conv = Image2FaceMediaConverter()
        with caplog.at_level(logging.WARNING):
            assert conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params()) == []
        assert bridge.CONFIG_ENV_VAR in caplog.text

    def test_a_failing_provider_yields_nothing_with_a_warning(self, monkeypatch, caplog):
        class Broken:
            def process(self, image, *, confidence_threshold=None, media_id=""):
                raise RuntimeError("model exploded")

            def compatibility(self):
                return "x:1"

        monkeypatch.setattr(bridge, "_pipeline", Broken())
        with caplog.at_level(logging.WARNING):
            assert (
                Image2FaceMediaConverter().convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params())
                == []
            )
        assert "face provider failed" in caplog.text

    def test_mtcnn_path_attaches_no_vector(self, monkeypatch):
        conv = Image2FaceMediaConverter()
        from vtscore.converters.image2face import FaceDetection as MtcnnFace

        monkeypatch.setattr(
            conv, "_detect_faces", lambda *a, **k: [MtcnnFace((10, 10, 60, 60), 0.8, (10, 10, 60, 60), None)]
        )
        (out,) = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(provider="mtcnn"))
        assert "embeddings" not in out and "embedder" not in out
        assert out["custom_metadata"] == {"Detection Confidence": 0.8}


class TestRunnerCarryThrough:
    def test_folder_conversion_files_the_vector_under_the_embedder(self, fake, tmp_path):
        (tmp_path / "photo.png").write_bytes(_png())
        medias: dict = {}
        run_converters_on_folder(
            tmp_path,
            target_media_type="face",
            medias=medias,
            converter_specs=[{"converter": "image2face", "params": _params()}],
        )
        assert len(medias) == 2
        for media in medias.values():
            assert media["embedder"] == "imagekit_face"
            assert set(media["embeddings"]) == {"imagekit_face"}
            assert media["embeddings"]["imagekit_face"].shape == (512,)
            assert media["custom_metadata"]["Face Compat"] == "fake:1"

    def test_embed_stage_finds_nothing_missing(self, fake, tmp_path):
        from vtscore.datasets.stages.embedding import embed_missing

        (tmp_path / "photo.png").write_bytes(_png())
        medias: dict = {}
        run_converters_on_folder(
            tmp_path,
            target_media_type="face",
            medias=medias,
            converter_specs=[{"converter": "image2face", "params": _params()}],
        )
        before = {mid: m["embeddings"]["imagekit_face"].copy() for mid, m in medias.items()}
        failures: dict = {}
        embed_missing(medias, failures=failures)
        assert failures == {}
        for mid, media in medias.items():
            np.testing.assert_array_equal(media["embeddings"]["imagekit_face"], before[mid])
            assert media["embedder"] == "imagekit_face"

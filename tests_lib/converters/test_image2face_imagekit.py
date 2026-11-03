"""The image2face converter over the face provider: detect + embed in one pass, vectors ride with the crops.

The provider seam is exercised against a fake pipeline installed into the
bridge (no models, no ``IMAGEKIT_FACE_CONFIG``), so what is tested is the
app's side of the contract: the crop, its filed vector, its metadata, the
runner carrying the vector into the media dict, a server without face models
refusing loudly, and one photo's failure yielding nothing.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest
from PIL import Image

from imagekit.face.records import FaceDetection, FaceRecord
from vtscore.converters.image2face import Image2FaceMediaConverter
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
    params = {"threshold": 0.5, "padding": 0.0, "min_size": 10, "align": "none"}
    params.update(overrides)
    return params


class TestConvert:
    def test_each_face_leaves_with_its_vector_and_tags(self, fake):
        conv = Image2FaceMediaConverter()
        outputs = conv.convert_normalized({"filename": "group.png", "media_bytes": _png()}, _params())
        assert fake.calls == [((200, 200, 3), 0.5, "group.png")]
        assert [o["filename"] for o in outputs] == ["group_face_1.png", "group_face_2.png"]
        first = outputs[0]
        assert first["embedder"] == "face" and set(first["embeddings"]) == {"face"}
        vec = first["embeddings"]["face"]
        assert vec.shape == (512,) and np.linalg.norm(vec) == pytest.approx(1.0)
        meta = first["custom_metadata"]
        assert meta["Face Compat"] == "fake:1" and meta["Detection Confidence"] == 0.9
        assert meta["Face Quality"] > 1.0
        assert first["width"] == 50 and first["height"] == 70
        assert first[SOURCE_BOX_FIELD] == pytest.approx([0.25, 0.2, 0.5, 0.55])

    def test_each_crop_carries_a_grid_thumbnail(self, fake):
        """A gallery of faces streams small tiles, not a decode per request (#5465)."""
        from vtscore.media.image.thumbnail import DEFAULT_MAX_DIM, make_image_thumbnail

        conv = Image2FaceMediaConverter()
        big = _png(1200, 1200)
        fake.records = [
            FaceRecord.from_detection(
                FaceDetection((100, 100, 900, 1000), None, 0.9),
                np.random.default_rng(42).standard_normal(512),
                "fake:1",
                "big.png",
                0,
            )
        ]
        (out,) = conv.convert_normalized({"filename": "big.png", "media_bytes": big}, _params())
        # The tile the thumbnail route would have generated, built once here.
        expected = make_image_thumbnail(out["media_bytes"])
        assert expected is not None and out["thumbnail_bytes"] == expected[0]
        with Image.open(io.BytesIO(out["thumbnail_bytes"])) as thumb:
            assert max(thumb.size) == DEFAULT_MAX_DIM
        assert (out["width"], out["height"]) == (800, 900)

    def test_thresholds_and_min_size_still_apply(self, fake):
        conv = Image2FaceMediaConverter()
        kept = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(threshold=0.8))
        assert len(kept) == 1 and fake.calls[-1][1] == 0.8
        assert conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(min_size=60)) == []

    def test_alignment_uses_the_provider_landmarks(self, fake):
        conv = Image2FaceMediaConverter()
        plain = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(align="none"))
        turned = conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params(align="roll"))
        # The crop turns on the provider's landmarks; the vector is the provider's, from the full photo.
        assert (turned[0]["width"], turned[0]["height"]) == (plain[0]["width"], plain[0]["height"])
        np.testing.assert_array_equal(turned[0]["embeddings"]["face"], plain[0]["embeddings"]["face"])

    def test_a_server_without_face_models_refuses_naming_what_is_missing(self):
        """Not an empty dataset: the run fails on its first photo, saying where models go."""
        conv = Image2FaceMediaConverter()
        with pytest.raises(bridge.FaceProviderUnconfigured, match="no face models") as err:
            conv.convert_normalized({"filename": "g.png", "media_bytes": _png()}, _params())
        assert bridge.CONFIG_ENV_VAR in str(err.value)

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
            assert media["embedder"] == "face"
            assert set(media["embeddings"]) == {"face"}
            assert media["embeddings"]["face"].shape == (512,)
            assert media["custom_metadata"]["Face Compat"] == "fake:1"
            # The crop's thumbnail rides along too, so no tile decodes on request.
            assert media["thumbnail_bytes"]

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
        before = {mid: m["embeddings"]["face"].copy() for mid, m in medias.items()}
        failures: dict = {}
        embed_missing(medias, failures=failures)
        assert failures == {}
        for mid, media in medias.items():
            np.testing.assert_array_equal(media["embeddings"]["face"], before[mid])
            assert media["embedder"] == "face"

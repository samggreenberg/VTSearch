"""Tests for the Face convert-in half media type, its FaceNet embedder, and
the image2face converter.

Face is the mirror of document in the half-media-type model: embeddable but
not importable (faces only ever arise from converting an image via
``image2face``).  These tests exercise the type's capability flags, the moved
FaceNet embedder, and the converter's crop logic against a stubbed MediaPipe
detector (no model weights, no mediapipe install required).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Half-media-type model
# ---------------------------------------------------------------------------


class TestHalfMediaTypeModel:
    def test_document_is_convert_out_half_type(self):
        from vtscore.media import get

        doc = get("document")
        assert doc.importable is True
        assert doc.embeddable is False
        assert doc.converts_to == ["image", "text"]

    def test_face_is_convert_in_half_type(self):
        from vtscore.media import get

        face = get("face")
        assert face.importable is False
        assert face.embeddable is True
        assert face.converts_to == []
        assert face.file_extensions == []

    def test_full_types_are_importable_and_embeddable(self):
        from vtscore.media import get

        for type_id in ("image", "audio", "text", "video"):
            mt = get(type_id)
            assert mt.importable is True, type_id
            assert mt.embeddable is True, type_id
            assert mt.converts_to == [], type_id

    def test_to_dict_surfaces_capability_flags(self):
        from vtscore.media import get

        d = get("document").to_dict()
        assert d["importable"] is True
        assert d["embeddable"] is False
        assert d["converts_to"] == ["image", "text"]
        f = get("face").to_dict()
        assert f["importable"] is False
        assert f["embeddable"] is True
        assert f["converts_to"] == []


# ---------------------------------------------------------------------------
# FaceMediaType
# ---------------------------------------------------------------------------


class TestFaceMediaType:
    def test_identity(self):
        from vtscore.media import get

        face = get("face")
        assert face.type_id == "face"
        assert face.name == "Face"
        assert face.icon == "face"
        assert face.has_thumbnail is True
        assert face.loops is False

    def test_no_demo_datasets(self):
        from vtscore.media import get

        assert get("face").demo_datasets == []

    def test_media_response_serves_image(self):
        from vtscore.media import get

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (200, 100, 50)).save(buf, format="PNG")
        media = {"id": 1, "filename": "x_face_1.png", "media_bytes": buf.getvalue()}
        resp = get("face").media_response(media)
        assert resp.mimetype == "image/png"
        assert resp.data == buf.getvalue()

    def test_image_response_returns_crop_bytes(self):
        from vtscore.media import get

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="PNG")
        media = {"id": 2, "filename": "x_face_1.png", "media_bytes": buf.getvalue()}
        resp = get("face").image_response(media)
        assert resp is not None
        assert resp.data == buf.getvalue()

    def test_image_response_none_when_no_bytes(self):
        from vtscore.media import get

        assert get("face").image_response({"id": 3, "filename": "a.png"}) is None


# ---------------------------------------------------------------------------
# FaceNet embedder (moved from the image type to the face type)
# ---------------------------------------------------------------------------


class TestFaceEmbedder:
    def test_name_and_media_type(self):
        from vtscore.media.face.embedder_facenet import FaceEmbedder

        emb = FaceEmbedder()
        assert emb.name == "face"
        assert emb.media_type_id == "face"

    def test_not_default_and_no_text(self):
        from vtscore.media.face.embedder_facenet import FaceEmbedder

        emb = FaceEmbedder()
        assert emb.is_default is False
        assert emb.supports_text is False
        assert emb.supports_patch_regions is False

    def test_to_dict(self):
        from vtscore.media.face.embedder_facenet import FaceEmbedder

        assert FaceEmbedder().to_dict() == {
            "name": "face",
            "display_name": "FaceNet (face identity, 512d)",
            "model_id": None,
            "media_type_id": "face",
            "is_default": False,
            "supports_text": False,
            "supports_patch_regions": False,
            "supports_geometric_verification": False,
            "license_notice": None,
        }

    def test_registered_under_face_type(self):
        from vtscore.media import get_embedder

        emb = get_embedder("face")
        assert emb.media_type_id == "face"

    def test_embed_text_returns_none(self):
        from vtscore.media.face.embedder_facenet import FaceEmbedder

        assert FaceEmbedder().embed_text("a face") is None

    def test_l2_normalises_output(self):
        from vtscore.media.face.embedder_facenet import FaceEmbedder

        emb = FaceEmbedder()
        out = emb._l2_normalise(np.array([3.0, 4.0, 0.0], dtype=np.float32))
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6
        z = emb._l2_normalise(np.zeros(3, dtype=np.float32))
        assert not np.isnan(z).any()

    def test_embed_media_returns_none_when_not_loaded(self):
        from vtscore.media.face.embedder_facenet import FaceEmbedder

        emb = FaceEmbedder()
        with patch.object(emb, "load_models"):
            assert emb.embed_media({"media_path": "/nonexistent.jpg"}) is None


# ---------------------------------------------------------------------------
# image2face converter
# ---------------------------------------------------------------------------


class _FakeBBox:
    def __init__(self, xmin, ymin, width, height):
        self.xmin = xmin
        self.ymin = ymin
        self.width = width
        self.height = height


class _FakeLocationData:
    def __init__(self, bbox):
        self.relative_bounding_box = bbox


class _FakeDetection:
    def __init__(self, score, bbox):
        self.score = [score]
        self.location_data = _FakeLocationData(bbox)


class _FakeResults:
    def __init__(self, detections):
        self.detections = detections


def _png_bytes(w=100, h=100):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


class TestImage2FaceConverter:
    def test_types(self):
        from vtscore.converters import get_converter

        conv = get_converter("image2face")
        assert conv is not None
        assert conv.name == "image2face"
        assert conv.source_type == "image"
        assert conv.target_type == "face"

    def test_registered_for_face_target(self):
        from vtscore.converters import list_converters_for_target

        names = {c.name for c in list_converters_for_target("face")}
        assert "image2face" in names

    def test_emits_one_crop_per_detection(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        detector = MagicMock()
        detector.process.return_value = _FakeResults(
            [
                _FakeDetection(0.9, _FakeBBox(0.1, 0.1, 0.3, 0.3)),
                _FakeDetection(0.8, _FakeBBox(0.5, 0.5, 0.3, 0.3)),
            ]
        )
        media = {"filename": "group.png", "media_bytes": _png_bytes()}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {"padding": "0", "min_size": "1"})
        assert len(out) == 2
        for o in out:
            assert o["media_bytes"]
            assert o["width"] > 0 and o["height"] > 0
            assert o["filename"].endswith(".png")
        # Highest-confidence face is first.
        assert "Detection Confidence" in out[0]["custom_metadata"]

    def test_no_faces_yields_no_output(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        detector = MagicMock()
        detector.process.return_value = _FakeResults([])
        media = {"filename": "empty.png", "media_bytes": _png_bytes()}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {})
        assert out == []

    def test_min_size_drops_tiny_detections(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        detector = MagicMock()
        # A 2px-wide box on a 100px image is below a 32px min size.
        detector.process.return_value = _FakeResults([_FakeDetection(0.95, _FakeBBox(0.1, 0.1, 0.02, 0.02))])
        media = {"filename": "tiny.png", "media_bytes": _png_bytes()}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {"padding": "0", "min_size": "32"})
        assert out == []

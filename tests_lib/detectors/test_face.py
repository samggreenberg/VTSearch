"""Tests for the Face convert-in half media type, its FaceNet embedder, and
the image2face converter.

Face is the mirror of document in the half-media-type model: embeddable but
not importable (faces only ever arise from converting an image via
``image2face``).  These tests exercise the type's capability flags, the moved
FaceNet embedder, and the converter's crop logic against a stubbed MTCNN
detector (no model weights, no facenet-pytorch install required).
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
        from vtscore.media.face.media_type import FaceMediaType

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="PNG")
        media = {"id": 2, "filename": "x_face_1.png", "media_bytes": buf.getvalue()}
        resp = FaceMediaType().image_response(media)
        assert resp is not None
        assert resp.data == buf.getvalue()

    def test_image_response_none_when_no_bytes(self):
        from vtscore.media.face.media_type import FaceMediaType

        assert FaceMediaType().image_response({"id": 3, "filename": "a.png"}) is None


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

    def test_weight_load_does_not_leak_progress_bar_to_console(self, capsys, monkeypatch):
        """facenet-pytorch downloads VGGFace2 weights via torch.hub, printing a
        tqdm bar. The embedder must wrap that in intercept_tqdm_progress so the
        bar is forwarded to the progress callback and never leaks to the console.

        See the "progress-bar-stdout-leak" fix: the ~107MB download bar used to
        print straight to stdout when starting the app.
        """
        import sys

        import tqdm.auto

        from vtscore.media.face.embedder_facenet import FaceEmbedder

        class _FakeInceptionResnetV1:
            """Stub whose construction emits a determinate tqdm bar, mimicking
            facenet-pytorch's torch.hub weight download."""

            def __init__(self, pretrained=None):
                bar = tqdm.auto.tqdm(total=107, desc="20180402-114759-vggface2.pt")
                bar.update(107)
                bar.close()

            def eval(self):
                return self

            def to(self, _device):
                return self

        fake_module = MagicMock()
        fake_module.InceptionResnetV1 = _FakeInceptionResnetV1

        calls: list[tuple] = []
        emb = FaceEmbedder()
        emb._on_progress = lambda status, message, current, total: calls.append((status, message, current, total))

        # ``setitem`` undoes only this one key.  ``patch.dict(sys.modules, …)``
        # restored the whole table on exit, so it also evicted the real
        # ``torch`` that ``_load_models_impl`` imports when this test was the
        # first to load it, and the next ``import torch`` re-initialised
        # ``torch._C`` and segfaulted the run.
        monkeypatch.setitem(sys.modules, "facenet_pytorch", fake_module)
        emb._load_models_impl()

        captured = capsys.readouterr()
        assert "vggface2" not in captured.out
        assert "vggface2" not in captured.err
        # The bar's progress should have been forwarded to the callback instead.
        assert any(c[3] == 107 for c in calls)


# ---------------------------------------------------------------------------
# image2face converter
# ---------------------------------------------------------------------------


def _fake_mtcnn(boxes, probs):
    """Build a stub MTCNN whose ``detect()`` returns ``(boxes, probs)``.

    ``boxes`` is a list of ``[x1, y1, x2, y2]`` pixel boxes (or ``None`` for
    "no faces"); ``probs`` the matching per-face confidences.
    """
    detector = MagicMock()
    detector.detect.return_value = (boxes, probs)
    return detector


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
        # Two faces on a 100x100 image, in pixel [x1, y1, x2, y2] coordinates.
        detector = _fake_mtcnn(
            boxes=[[10, 10, 40, 40], [50, 50, 80, 80]],
            probs=[0.9, 0.8],
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
        # MTCNN returns ``None`` for boxes when it finds no faces.
        detector = _fake_mtcnn(boxes=None, probs=[None])
        media = {"filename": "empty.png", "media_bytes": _png_bytes()}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {})
        assert out == []

    def test_min_size_drops_tiny_detections(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        # A 2px-wide box on a 100px image is below a 32px min size.
        detector = _fake_mtcnn(boxes=[[10, 10, 12, 12]], probs=[0.95])
        media = {"filename": "tiny.png", "media_bytes": _png_bytes()}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {"padding": "0", "min_size": "32"})
        assert out == []

    def test_below_threshold_detections_dropped(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        # A high-confidence face and a low-confidence one; only the first survives.
        detector = _fake_mtcnn(boxes=[[10, 10, 60, 60], [20, 20, 70, 70]], probs=[0.95, 0.2])
        media = {"filename": "mixed.png", "media_bytes": _png_bytes()}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {"threshold": "0.5", "padding": "0", "min_size": "1"})
        assert len(out) == 1

    def test_path_only_media_is_read_from_disk(self, tmp_path):
        """Reference (*thin*) mode passes only ``{filename, media_path}``.

        The runner's thin path hands the converter no ``media_bytes`` at all,
        so the converter has to read the file itself; otherwise every image in
        a thin folder import yields zero faces with no error.
        """
        from vtscore.converters.image2face import Image2FaceMediaConverter

        src = tmp_path / "group.png"
        src.write_bytes(_png_bytes())

        conv = Image2FaceMediaConverter()
        detector = _fake_mtcnn(boxes=[[10, 10, 40, 40]], probs=[0.9])
        media = {"filename": "group.png", "media_path": str(src)}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {"padding": "0", "min_size": "1"})
        assert len(out) == 1
        assert out[0]["media_bytes"]

    def test_missing_path_yields_no_output(self, tmp_path):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        media = {"filename": "gone.png", "media_path": str(tmp_path / "gone.png")}
        out = conv.convert_normalized(media, {})
        assert out == []


class TestImage2FaceSourceBox:
    """Each crop records where it sat in its photo, as a fraction of the frame (#4748)."""

    def test_each_crop_records_its_box_normalised_to_the_photo(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        # A wide photo, so x and y normalise against different sides.
        detector = _fake_mtcnn(boxes=[[20, 10, 80, 40], [100, 50, 160, 90]], probs=[0.9, 0.8])
        media = {"filename": "group.png", "media_bytes": _png_bytes(200, 100)}
        with patch.object(conv, "_make_detector", return_value=detector):
            out = conv.convert_normalized(media, {"padding": "0", "min_size": "1"})

        assert [o["source_box"] for o in out] == [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.9]]

    def test_the_box_is_the_padded_crop(self):
        """The box outlines what the crop shows, padding included, clamped to the photo."""
        from vtscore.converters.image2face import Image2FaceMediaConverter

        conv = Image2FaceMediaConverter()
        detector = _fake_mtcnn(boxes=[[10, 40, 30, 60]], probs=[0.9])
        media = {"filename": "edge.png", "media_bytes": _png_bytes(100, 100)}
        with patch.object(conv, "_make_detector", return_value=detector):
            (face,) = conv.convert_normalized(media, {"padding": "0.75", "min_size": "1"})

        # 20 px face + 15 px a side: x runs -5..45 and clamps to 0..45.
        assert face["source_box"] == [0.0, 0.25, 0.45, 0.75]
        assert (face["width"], face["height"]) == (45, 50)

    def test_a_downsampled_decode_still_gives_the_photo_fraction(self, monkeypatch):
        """Detection runs on a bounded decode, so the box must not be read as original pixels."""
        from vtscore.converters.image2face import Image2FaceMediaConverter
        from vtscore.media.image import decode

        monkeypatch.setattr(decode, "MAX_DECODE_PIXELS", 200 * 100)
        seen: dict[str, tuple[int, int]] = {}

        def detect(img):
            seen["size"] = img.size
            return [[20, 10, 60, 50]], [0.9]

        detector = MagicMock()
        detector.detect.side_effect = detect
        conv = Image2FaceMediaConverter()
        media = {"filename": "big.png", "media_bytes": _png_bytes(400, 200)}
        with patch.object(conv, "_make_detector", return_value=detector):
            (face,) = conv.convert_normalized(media, {"padding": "0", "min_size": "1"})

        assert seen["size"] == (200, 100)
        # Over the 400x200 original these pixels would read 0.05..0.15 wide.
        assert face["source_box"] == [0.1, 0.1, 0.3, 0.5]


class TestImage2FaceThroughTheRunner:
    """A folder conversion keeps the crop's box and its confidence row (#4748).

    The runner copies converter output through a named field list, unlike the
    importer-hook and clipper-chain paths, which keep the whole dict.
    """

    def test_folder_conversion_keeps_source_box_and_confidence(self, tmp_path):
        from vtscore.converters.image2face import CONVERTER
        from vtscore.converters.runner import run_converters_on_folder

        (tmp_path / "group.png").write_bytes(_png_bytes(200, 100))
        detector = _fake_mtcnn(boxes=[[20, 10, 80, 40]], probs=[0.9])
        medias: dict[int, dict] = {}
        with patch.object(CONVERTER, "_make_detector", return_value=detector):
            run_converters_on_folder(
                tmp_path,
                target_media_type="face",
                medias=medias,
                converter_specs=[{"converter": "image2face", "params": {"padding": "0", "min_size": "1"}}],
            )

        (face,) = medias.values()
        assert face["media_type"] == "face"
        assert face["source_box"] == [0.1, 0.1, 0.4, 0.4]
        assert face["custom_metadata"] == {"Detection Confidence": 0.9}

    def test_demo_conversion_keeps_source_box(self):
        from vtscore.converters.image2face import CONVERTER
        from vtscore.converters.runner import apply_converter_to_demo

        medias = {1: {"id": 1, "filename": "photo.png", "media_bytes": _png_bytes(200, 100), "category": "c"}}
        detector = _fake_mtcnn(boxes=[[20, 10, 80, 40]], probs=[0.9])
        with patch.object(CONVERTER, "_make_detector", return_value=detector):
            apply_converter_to_demo("image2face", "demo_photos", medias)

        (face,) = medias.values()
        # The demo path runs at the converter's defaults, so the 0.25 padding
        # applies: 15 px a side on x and 7.5 on y (rounded) of a 200x100 photo.
        assert face["source_box"] == [0.025, 0.02, 0.475, 0.48]

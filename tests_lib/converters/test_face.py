"""Tests for the Face convert-in half media type, its embedder, and the image2face converter.

Face is the mirror of document in the half-media-type model: embeddable but
not importable (faces only ever arise from converting an image via
``image2face``).  These tests exercise the type's capability flags, the face
embedder's identity, and the converter's crop logic against a fake face
provider installed on the bridge (no model weights).  The provider contract
itself (vectors riding with the crops, the compat guard) is
``tests_lib/converters/test_image2face_imagekit.py``'s.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image

from imagekit.face.records import FaceDetection as ProviderDetection, FaceRecord
from vtscore.media.face import imagekit_bridge as bridge


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
# The face embedder
# ---------------------------------------------------------------------------


class TestFaceEmbedder:
    def test_it_is_the_face_types_one_embedder_and_its_default(self):
        from vtscore.media import embedders_for_type, get_embedder

        (emb,) = embedders_for_type("face")
        assert emb is get_embedder("face")
        assert emb.name == "face" and emb.media_type_id == "face"
        assert emb.is_default is True
        assert emb.supports_text is False and emb.supports_patch_regions is False

    def test_to_dict(self):
        from vtscore.media import get_embedder

        d = get_embedder("face").to_dict()
        assert {k: d[k] for k in ("name", "display_name", "model_id", "media_type_id", "is_default")} == {
            "name": "face",
            "display_name": "Face identity (512d)",
            "model_id": None,
            "media_type_id": "face",
            "is_default": True,
        }
        assert d["license_notice"]

    def test_embed_text_returns_none(self):
        # A fresh instance: the conftest stubs the registered singletons' embed calls.
        from vtscore.media.face.embedder_imagekit import ImagekitFaceEmbedder

        assert ImagekitFaceEmbedder().embed_text("a face") is None


# ---------------------------------------------------------------------------
# image2face converter
# ---------------------------------------------------------------------------


class _Provider:
    """A fake face provider: the same faces for every photo, recording the image it saw."""

    def __init__(self, boxes, probs, points=None):
        self.faces = list(zip(boxes, probs, points or [None] * len(boxes)))
        self.seen: list[tuple[int, ...]] = []

    def process(self, image, *, confidence_threshold=None, media_id=""):
        self.seen.append(tuple(image.shape))
        rng = np.random.default_rng(42)
        return [
            FaceRecord.from_detection(
                ProviderDetection(tuple(box), pts, prob), rng.standard_normal(512), "fake:1", media_id, i
            )
            for i, (box, prob, pts) in enumerate(self.faces)
        ]

    def compatibility(self):
        return "fake:1"


@pytest.fixture
def provider(monkeypatch):
    """Install a fake provider on the bridge; call the result with ``(boxes, probs, points=None)``."""

    def install(boxes, probs, points=None) -> _Provider:
        pipe = _Provider(boxes, probs, points)
        monkeypatch.setattr(bridge, "_pipeline", pipe)
        monkeypatch.setattr(bridge, "_compat", "fake:1")
        return pipe

    return install


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

    def test_emits_one_crop_per_detection(self, provider):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        # Two faces on a 100x100 image, in pixel [x1, y1, x2, y2] coordinates.
        provider([[10, 10, 40, 40], [50, 50, 80, 80]], [0.8, 0.9])
        media = {"filename": "group.png", "media_bytes": _png_bytes()}
        out = Image2FaceMediaConverter().convert_normalized(media, {"padding": "0", "min_size": "1"})
        assert len(out) == 2
        for o in out:
            assert o["media_bytes"]
            assert o["width"] > 0 and o["height"] > 0
            assert o["filename"].endswith(".png")
        # Highest-confidence face is first.
        assert [o["custom_metadata"]["Detection Confidence"] for o in out] == [0.9, 0.8]

    def test_no_faces_yields_no_output(self, provider):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        provider([], [])
        media = {"filename": "empty.png", "media_bytes": _png_bytes()}
        assert Image2FaceMediaConverter().convert_normalized(media, {}) == []

    def test_min_size_drops_tiny_detections(self, provider):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        # A 2px-wide box on a 100px image is below a 32px min size.
        provider([[10, 10, 12, 12]], [0.95])
        media = {"filename": "tiny.png", "media_bytes": _png_bytes()}
        assert Image2FaceMediaConverter().convert_normalized(media, {"padding": "0", "min_size": "32"}) == []

    def test_below_threshold_detections_dropped(self, provider):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        # A high-confidence face and a low-confidence one; only the first survives.
        provider([[10, 10, 60, 60], [20, 20, 70, 70]], [0.95, 0.2])
        media = {"filename": "mixed.png", "media_bytes": _png_bytes()}
        out = Image2FaceMediaConverter().convert_normalized(
            media, {"threshold": "0.5", "padding": "0", "min_size": "1"}
        )
        assert len(out) == 1

    def test_path_only_media_is_read_from_disk(self, provider, tmp_path):
        """Reference (*thin*) mode passes only ``{filename, media_path}``.

        The runner's thin path hands the converter no ``media_bytes`` at all,
        so the converter has to read the file itself; otherwise every image in
        a thin folder import yields zero faces with no error.
        """
        from vtscore.converters.image2face import Image2FaceMediaConverter

        src = tmp_path / "group.png"
        src.write_bytes(_png_bytes())
        provider([[10, 10, 40, 40]], [0.9])
        media = {"filename": "group.png", "media_path": str(src)}
        out = Image2FaceMediaConverter().convert_normalized(media, {"padding": "0", "min_size": "1"})
        assert len(out) == 1
        assert out[0]["media_bytes"]

    def test_missing_path_yields_no_output(self, tmp_path):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        media = {"filename": "gone.png", "media_path": str(tmp_path / "gone.png")}
        assert Image2FaceMediaConverter().convert_normalized(media, {}) == []


class TestImage2FaceSourceBox:
    """Each crop records where it sat in its photo, as a fraction of the frame (#4748)."""

    def test_each_crop_records_its_box_normalised_to_the_photo(self, provider):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        # A wide photo, so x and y normalise against different sides.
        provider([[20, 10, 80, 40], [100, 50, 160, 90]], [0.9, 0.8])
        media = {"filename": "group.png", "media_bytes": _png_bytes(200, 100)}
        out = Image2FaceMediaConverter().convert_normalized(media, {"padding": "0", "min_size": "1"})

        assert [o["source_box"] for o in out] == [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.9]]

    def test_the_box_is_the_padded_crop(self, provider):
        """The box outlines what the crop shows, padding included, clamped to the photo."""
        from vtscore.converters.image2face import Image2FaceMediaConverter

        provider([[10, 40, 30, 60]], [0.9])
        media = {"filename": "edge.png", "media_bytes": _png_bytes(100, 100)}
        (face,) = Image2FaceMediaConverter().convert_normalized(media, {"padding": "0.75", "min_size": "1"})

        # 20 px face + 15 px a side: x runs -5..45 and clamps to 0..45.
        assert face["source_box"] == [0.0, 0.25, 0.45, 0.75]
        assert (face["width"], face["height"]) == (45, 50)

    def test_a_downsampled_decode_still_gives_the_photo_fraction(self, provider, monkeypatch):
        """Detection runs on a bounded decode, so the box must not be read as original pixels."""
        from vtscore.converters.image2face import Image2FaceMediaConverter
        from vtscore.media.image import decode

        monkeypatch.setattr(decode, "MAX_DECODE_PIXELS", 200 * 100)
        pipe = provider([[20, 10, 60, 50]], [0.9])
        media = {"filename": "big.png", "media_bytes": _png_bytes(400, 200)}
        (face,) = Image2FaceMediaConverter().convert_normalized(media, {"padding": "0", "min_size": "1"})

        assert pipe.seen == [(100, 200, 3)]
        # Over the 400x200 original these pixels would read 0.05..0.15 wide.
        assert face["source_box"] == [0.1, 0.1, 0.3, 0.5]


class TestImage2FaceThroughTheRunner:
    """A folder conversion keeps the crop's box and its confidence row (#4748).

    The runner copies converter output through a named field list, unlike the
    importer-hook and clipper-chain paths, which keep the whole dict.
    """

    def test_folder_conversion_keeps_source_box_and_confidence(self, provider, tmp_path):
        from vtscore.converters.runner import run_converters_on_folder

        (tmp_path / "group.png").write_bytes(_png_bytes(200, 100))
        provider([[20, 10, 80, 40]], [0.9])
        medias: dict[int, dict] = {}
        run_converters_on_folder(
            tmp_path,
            target_media_type="face",
            medias=medias,
            converter_specs=[{"converter": "image2face", "params": {"padding": "0", "min_size": "1"}}],
        )

        (face,) = medias.values()
        assert face["media_type"] == "face"
        assert face["source_box"] == [0.1, 0.1, 0.4, 0.4]
        assert face["custom_metadata"]["Detection Confidence"] == 0.9

    def test_demo_conversion_keeps_source_box(self, provider):
        from vtscore.converters.runner import apply_converter_to_demo

        medias = {1: {"id": 1, "filename": "photo.png", "media_bytes": _png_bytes(200, 100), "category": "c"}}
        provider([[20, 10, 80, 40]], [0.9])
        apply_converter_to_demo("image2face", "demo_photos", medias)

        (face,) = medias.values()
        # The demo path runs at the converter's defaults, so the 0.25 padding
        # applies: 15 px a side on x and 7.5 on y (rounded) of a 200x100 photo.
        assert face["source_box"] == [0.025, 0.02, 0.475, 0.48]


class TestImage2FaceAlign:
    """``align`` turns each stored crop upright by its five landmarks; it changes pixels only (#4777)."""

    COLOURS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255))

    def _face(self, points, size=(240, 240)):
        """A black photo with a coloured disc at each landmark."""
        from PIL import ImageDraw

        img = Image.new("RGB", size)
        draw = ImageDraw.Draw(img)
        for (x, y), colour in zip(points, self.COLOURS):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colour)
        return img

    def _template(self, side):
        """The ArcFace template scaled into a *side*-px box, upright, relative to the box's corner."""
        from imagekit.face.face_align import ARCFACE_TEMPLATE, TEMPLATE_SIZE

        return np.asarray(ARCFACE_TEMPLATE, dtype=np.float64) / TEMPLATE_SIZE * side

    def _landmarks(self, side, angle_deg, centre):
        """The template in a *side*-px box centred on *centre*, turned clockwise (y down) by *angle_deg*."""
        t = self._template(side) - side / 2
        th = np.radians(angle_deg)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return t @ rot.T + np.asarray(centre)

    def _assert_dots_at(self, crop, expected):
        arr = np.asarray(crop).astype(int)
        for (x, y), colour in zip(expected, self.COLOURS):
            got = arr[int(round(y)), int(round(x))]
            assert np.abs(got - np.asarray(colour)).max() < 60, (x, y, got.tolist(), colour)

    def test_the_fit_recovers_a_similarity(self):
        from vtscore.converters.image2face import fit_similarity

        src = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.9], [0.8, 0.9]])
        th, scale = np.radians(25.0), 3.0
        a, b = scale * np.cos(th), scale * np.sin(th)
        dst = np.column_stack([a * src[:, 0] - b * src[:, 1] + 7, b * src[:, 0] + a * src[:, 1] - 2])
        assert np.allclose(fit_similarity(src, dst), (a, b, 7, -2))

    def test_none_and_missing_landmarks_are_the_plain_crop(self):
        from vtscore.converters.image2face import FaceDetection, crop_face

        pts = self._landmarks(80, 30, (120, 120))
        img = self._face(pts)
        plain = img.crop((80, 80, 160, 160))
        with_pts = FaceDetection((80, 80, 160, 160), 0.9, (80.0, 80.0, 160.0, 160.0), pts)
        without = with_pts._replace(landmarks=None)
        for det, align in ((with_pts, "none"), (without, "roll")):
            assert crop_face(img, det, align, 0.0).tobytes() == plain.tobytes()

    def test_roll_turns_the_box_upright_about_its_centre(self):
        from vtscore.converters.image2face import FaceDetection, crop_face

        pts = self._landmarks(80, 35, (120, 120))
        det = FaceDetection((80, 80, 160, 160), 0.9, (80.0, 80.0, 160.0, 160.0), pts)
        crop = crop_face(self._face(pts), det, "roll", 0.0)
        assert crop.size == (80, 80)
        self._assert_dots_at(crop, self._template(80))

    def test_roll_of_an_upright_face_is_the_plain_crop(self):
        from vtscore.converters.image2face import FaceDetection, crop_face

        pts = self._landmarks(80, 0, (120, 120))
        img = self._face(pts)
        det = FaceDetection((80, 80, 160, 160), 0.9, (80.0, 80.0, 160.0, 160.0), pts)
        diff = np.abs(np.asarray(crop_face(img, det, "roll", 0.0), int) - np.asarray(img.crop((80, 80, 160, 160)), int))
        assert diff.max() <= 1

    def test_alignment_changes_pixels_not_faces(self, provider):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        pts = self._landmarks(80, 40, (120, 120))
        buf = io.BytesIO()
        self._face(pts).save(buf, format="PNG")
        media = {"filename": "tilted.png", "media_bytes": buf.getvalue()}
        provider([[80, 80, 160, 160]], [0.95], [pts])
        conv = Image2FaceMediaConverter()
        out = {align: conv.convert_normalized(media, {"align": align}) for align in ("none", "roll")}
        assert all(len(o) == 1 for o in out.values())
        assert out["roll"][0]["source_box"] == out["none"][0]["source_box"]
        assert out["roll"][0]["media_bytes"] != out["none"][0]["media_bytes"]
        # roll keeps the padded box's size.
        assert (out["roll"][0]["width"], out["roll"][0]["height"]) == (
            out["none"][0]["width"],
            out["none"][0]["height"],
        )
        # The vector is the provider's, from the full photo, whatever the crop's turn.
        np.testing.assert_array_equal(out["roll"][0]["embeddings"]["face"], out["none"][0]["embeddings"]["face"])

    def test_the_default_is_roll(self):
        """``roll`` won on the FHIBE face bench (#4777), so it is what a conversion gets unasked."""
        from vtscore.converters.image2face import Image2FaceMediaConverter

        assert Image2FaceMediaConverter()._resolve_params(None).align == "roll"

    def test_an_unknown_mode_is_refused(self):
        from vtscore.converters.image2face import Image2FaceMediaConverter

        for mode in ("tilt", "similarity"):
            with pytest.raises(ValueError):
                Image2FaceMediaConverter().convert_normalized(
                    {"filename": "x.png", "media_bytes": _png_bytes()}, {"align": mode}
                )

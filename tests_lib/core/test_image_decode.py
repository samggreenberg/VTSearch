"""Library-tier tests for :mod:`vtscore.media.image.decode`.

Pillow refuses to open an image past twice ``Image.MAX_IMAGE_PIXELS``, raising
``DecompressionBombError`` ("could be decompression bomb DOS attack") on the
header alone.  VTSearch lifts that ceiling — a user's gigapixel panorama is
large, not hostile — and replaces it with a bounded *decode* so peak memory
stays capped instead.  These tests pin both halves of that trade.

They also pin the module's second job: **applying EXIF display orientation**.
A phone photo is stored in sensor order with a tag saying how to rotate it, and
browsers honour that tag — so the picture a user sees is the upright one.  Every
decode here returns those upright pixels, and :func:`upright_size` reports the
matching dimensions from the header alone, so the embedder, the thumbnailer, the
extractors' boxes and a media's stored ``width``/``height`` all describe the
same image.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from vtscore.media.image import decode as decode_mod
from vtscore.media.image.decode import (
    apply_exif_orientation,
    configure_pil_limits,
    decode_bounded,
    decode_bounded_rgb,
    exif_orientation,
    exif_upright_size,
    open_image,
    open_upright,
    upright_size,
)


def _encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _oriented(orientation: int, size: tuple[int, int] = (40, 20)) -> bytes:
    """A JPEG whose stored bitmap is *size*, tagged with *orientation*.

    Orientation 6 is the ubiquitous one: a phone held upright writes a landscape
    sensor frame plus "rotate me", which is exactly the file that used to reach
    the embedder sideways.
    """
    exif = Image.Exif()
    exif[274] = orientation
    img = Image.new("RGB", size, color=(120, 60, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


@pytest.fixture
def restore_pil_limit():
    """Restore Pillow's global ceiling (and the configured-once latch)."""
    saved = Image.MAX_IMAGE_PIXELS
    saved_latch = decode_mod._limits_configured
    yield
    Image.MAX_IMAGE_PIXELS = saved
    decode_mod._limits_configured = saved_latch


class TestConfigurePilLimits:
    def test_lifts_the_decompression_bomb_ceiling(self):
        configure_pil_limits()
        assert Image.MAX_IMAGE_PIXELS is None

    def test_applied_at_package_import(self):
        """Importing the image media type is enough — no explicit call needed.

        The media registry auto-discovers ``vtscore.media.image`` at import, so
        the ceiling is lifted process-wide before anything decodes an image,
        including code that reaches for ``PIL.Image.open`` directly.
        """
        import vtscore.media.image  # noqa: F401,PLC0415

        assert Image.MAX_IMAGE_PIXELS is None

    def test_reopens_an_image_that_would_have_been_refused(self, restore_pil_limit):
        """The exact failure a user hit: a big-but-benign image, refused."""
        blob = _encode(Image.new("RGB", (200, 200), color=(10, 20, 30)))

        # Stand in for a 330 MP source by shrinking the ceiling instead of
        # allocating one: Pillow's check is purely (pixels > 2 * ceiling).
        decode_mod._limits_configured = False
        Image.MAX_IMAGE_PIXELS = 100
        with pytest.raises(Image.DecompressionBombError):
            Image.open(io.BytesIO(blob))

        configure_pil_limits()
        with open_image(blob) as img:
            assert img.size == (200, 200)

    def test_is_idempotent(self, restore_pil_limit):
        configure_pil_limits()
        configure_pil_limits()
        assert Image.MAX_IMAGE_PIXELS is None


class TestDecodeBounded:
    def test_downsamples_past_the_budget(self):
        blob = _encode(Image.new("RGB", (400, 200), color=(90, 40, 10)))
        img, scale = decode_bounded(blob, max_pixels=5_000)
        with img:
            assert img.width * img.height <= 5_000
            # Aspect ratio preserved.
            assert img.width == pytest.approx(img.height * 2, abs=1)
            assert scale == pytest.approx(img.width / 400)
            assert scale < 1.0

    def test_leaves_an_image_inside_the_budget_untouched(self):
        blob = _encode(Image.new("RGB", (64, 32), color=(1, 2, 3)))
        img, scale = decode_bounded(blob, max_pixels=1_000_000)
        with img:
            assert img.size == (64, 32)
            assert scale == 1.0

    def test_never_upscales_a_small_image(self):
        blob = _encode(Image.new("RGB", (8, 8), color=(4, 5, 6)))
        img, scale = decode_bounded(blob, max_pixels=1_000_000)
        with img:
            assert img.size == (8, 8)
            assert scale == 1.0

    def test_zero_budget_decodes_at_native_size(self):
        blob = _encode(Image.new("RGB", (300, 300), color=(7, 8, 9)))
        img, scale = decode_bounded(blob, max_pixels=0)
        with img:
            assert img.size == (300, 300)
            assert scale == 1.0

    def test_accepts_a_filesystem_path(self, tmp_path):
        path = tmp_path / "big.png"
        Image.new("RGB", (400, 400), color=(11, 22, 33)).save(path)
        img, scale = decode_bounded(path, max_pixels=2_500)
        with img:
            assert img.width * img.height <= 2_500
            assert scale < 1.0

    def test_bounds_a_jpeg_through_the_draft_path(self):
        """JPEG reduction goes via DCT-scaled ``draft``, so it must still fit."""
        blob = _encode(Image.new("RGB", (512, 512), color=(60, 60, 60)), "JPEG")
        img, scale = decode_bounded(blob, max_pixels=4_096)
        with img:
            assert img.width * img.height <= 4_096
            assert scale == pytest.approx(img.width / 512)

    def test_rgb_variant_converts_after_downsampling(self):
        blob = _encode(Image.new("RGBA", (400, 400), color=(1, 2, 3, 128)))
        img, scale = decode_bounded_rgb(blob, max_pixels=2_500)
        with img:
            assert img.mode == "RGB"
            assert img.width * img.height <= 2_500
            assert scale < 1.0

    def test_scale_maps_coordinates_back_to_the_original(self):
        """The contract extractors rely on: ``coord / scale`` is original space."""
        blob = _encode(Image.new("RGB", (1000, 500), color=(70, 70, 70)))
        img, scale = decode_bounded(blob, max_pixels=10_000)
        with img:
            # A box spanning the decoded image maps back to the full original.
            assert img.width / scale == pytest.approx(1000, abs=2)
            assert img.height / scale == pytest.approx(500, abs=2)


class TestOpenImage:
    def test_reads_dimensions_without_decoding(self):
        blob = _encode(Image.new("RGB", (321, 123), color=(9, 9, 9)))
        with open_image(blob) as img:
            assert (img.width, img.height) == (321, 123)


class TestOversizedImagesSurviveIngest:
    """End-to-end: a source past the ceiling is kept, not dropped."""

    def test_thumbnail_is_generated_for_an_oversized_source(self, restore_pil_limit):
        from vtscore.media.image.thumbnail import DEFAULT_MAX_DIM, make_image_thumbnail  # noqa: PLC0415

        blob = _encode(Image.new("RGB", (900, 600), color=(200, 40, 40)), "JPEG")
        decode_mod._limits_configured = False
        Image.MAX_IMAGE_PIXELS = 1_000  # 900*600 is well past 2x this

        configure_pil_limits()
        result = make_image_thumbnail(blob)
        assert result is not None
        thumb_bytes, mimetype = result
        assert mimetype == "image/jpeg"
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert max(out.size) <= DEFAULT_MAX_DIM

    def test_media_type_records_dimensions_for_an_oversized_source(self, restore_pil_limit, tmp_path):
        from vtscore.media.image.media_type import ImageMediaType  # noqa: PLC0415

        blob = _encode(Image.new("RGB", (900, 600), color=(30, 200, 90)), "JPEG")
        decode_mod._limits_configured = False
        Image.MAX_IMAGE_PIXELS = 1_000

        configure_pil_limits()
        data = ImageMediaType().load_media_data(tmp_path / "huge.jpg", media_bytes=blob)
        # Full dimensions, not the bounded decode's: the stored bytes are the
        # untouched original and crop boxes are expressed against them.
        assert (data["width"], data["height"]) == (900, 600)
        assert data["thumbnail_bytes"]

    def test_embedder_bulk_loader_decodes_an_oversized_source(self, restore_pil_limit):
        from vtscore.media.image._image_bulk import _load_pil  # noqa: PLC0415

        blob = _encode(Image.new("RGB", (900, 600), color=(20, 20, 200)), "JPEG")
        decode_mod._limits_configured = False
        Image.MAX_IMAGE_PIXELS = 1_000

        configure_pil_limits()
        img = _load_pil(blob)
        assert img is not None
        assert img.mode == "RGB"


class TestExifOrientationHelpers:
    def test_reports_the_tag_without_decoding(self):
        with open_image(_oriented(6)) as img:
            assert exif_orientation(img) == 6
            # Still lazy: reading the tag must not have pulled in the pixels.
            # ``_im`` is Pillow's backing buffer; the public ``im`` asserts on it.
            assert img._im is None

    def test_a_missing_or_unreadable_tag_reads_as_upright(self):
        with open_image(_encode(Image.new("RGB", (8, 8)))) as img:
            assert exif_orientation(img) == 1

        class _Hostile:
            format = None
            size = (8, 8)

            def getexif(self):
                raise ValueError("corrupt EXIF block")

        assert exif_orientation(_Hostile()) == 1  # type: ignore[arg-type]

    def test_upright_size_swaps_the_axes_for_a_quarter_turn(self):
        # 5-8 transpose; 1-4 are identity/flips that keep the axes as stored.
        for orientation in (5, 6, 7, 8):
            assert upright_size(_oriented(orientation, (40, 20))) == (20, 40)
        for orientation in (1, 2, 3, 4):
            assert upright_size(_oriented(orientation, (40, 20))) == (40, 20)

    def test_upright_size_matches_what_the_decode_actually_returns(self):
        """The invariant the stored ``width``/``height`` rest on."""
        for orientation in range(1, 9):
            blob = _oriented(orientation, (40, 20))
            img, _scale = decode_bounded(blob, max_pixels=0)
            with img:
                assert img.size == upright_size(blob) == exif_upright_size(open_image(blob))

    def test_apply_returns_the_same_object_when_there_is_nothing_to_do(self):
        """No unconditional full-bitmap copy on the overwhelmingly common path."""
        with open_image(_encode(Image.new("RGB", (16, 16)))) as img:
            assert apply_exif_orientation(img) is img

    def test_a_rotated_copy_keeps_its_format_and_drops_the_tag(self):
        with open_image(_oriented(6)) as img:
            upright = apply_exif_orientation(img)
            assert upright is not img
            # Format survives so a caller re-encoding a crop keeps the container
            # instead of silently switching to PNG.
            assert upright.format == "JPEG"
            # Tag stripped, so re-applying is a no-op rather than a second turn.
            assert exif_orientation(upright) == 1
            assert apply_exif_orientation(upright) is upright


class TestDecodeBoundedAppliesOrientation:
    def test_a_sideways_photo_decodes_upright(self):
        img, _scale = decode_bounded(_oriented(6, (40, 20)), max_pixels=0)
        with img:
            assert img.size == (20, 40)

    def test_the_rgb_variant_too(self):
        img, _scale = decode_bounded_rgb(_oriented(6, (40, 20)), max_pixels=0)
        with img:
            assert img.mode == "RGB"
            assert img.size == (20, 40)

    def test_scale_survives_the_transpose(self):
        """``scale`` is a linear ratio, so a quarter turn must not disturb it.

        Extractors divide reported coordinates by it to land in the media's
        stored (upright) space; if the transpose leaked into the ratio, every
        box on a rotated photo would be off by the aspect ratio.
        """
        img, scale = decode_bounded(_oriented(6, (400, 200)), max_pixels=5_000)
        with img:
            assert img.size[0] < img.size[1]  # upright: portrait
            assert img.width * img.height <= 5_000
            # Upright original is 200x400; the decoded box maps back onto it.
            assert img.width / scale == pytest.approx(200, abs=2)
            assert img.height / scale == pytest.approx(400, abs=2)

    def test_an_untagged_image_is_unaffected(self):
        img, scale = decode_bounded(_encode(Image.new("RGB", (40, 20))), max_pixels=0)
        with img:
            assert img.size == (40, 20)
            assert scale == 1.0


class TestOpenUpright:
    def test_decodes_at_native_size_rotated(self):
        with open_upright(_oriented(6, (400, 200))) as img:
            assert img.size == (200, 400)
            assert img.format == "JPEG"

    def test_accepts_a_filesystem_path(self, tmp_path):
        path = tmp_path / "sideways.jpg"
        path.write_bytes(_oriented(6, (40, 20)))
        with open_upright(path) as img:
            assert img.size == (20, 40)

    def test_open_image_stays_lazy_and_untransposed(self):
        """The deliberate exception: header reads must not force a decode."""
        with open_image(_oriented(6, (40, 20))) as img:
            assert img.size == (40, 20)
            assert img._im is None


class TestOrientationIsConsistentAcrossThePipeline:
    """The whole point of fixing this at decode: one coordinate space.

    A fix that transposed only in the embedder path would leave the thumbnail,
    the stored dimensions and the crop boxes disagreeing with it — worse than
    the original bug, because the disagreement is silent.
    """

    def test_stored_dimensions_match_the_thumbnail_and_the_embed_decode(self):
        from vtscore.media.image._image_bulk import _load_pil  # noqa: PLC0415
        from vtscore.media.image.media_type import ImageMediaType  # noqa: PLC0415

        blob = _oriented(6, (400, 200))
        data = ImageMediaType().load_media_data(Path("sideways.jpg"), media_bytes=blob)
        assert (data["width"], data["height"]) == (200, 400)

        embedded = _load_pil(blob)
        assert embedded is not None
        assert embedded.height > embedded.width

        with Image.open(io.BytesIO(data["thumbnail_bytes"])) as thumb:
            assert thumb.height > thumb.width

    def test_a_crop_box_in_stored_space_cuts_the_region_the_user_drew(self):
        from vtscore.media.image.clipper import ImageBboxClipper  # noqa: PLC0415

        blob = _oriented(6, (400, 200))  # upright: 200 wide x 400 tall
        clip = ImageBboxClipper((0, 0, 100, 400)).clip(
            {"id": 0, "media_type": "image", "media_bytes": blob, "width": 200, "height": 400}
        )[0]
        # The box is legal in upright space and would have been clamped to
        # nonsense (or cut the wrong half) against the 400x200 sensor bitmap.
        assert (clip["width"], clip["height"]) == (100, 400)
        with Image.open(io.BytesIO(clip["media_bytes"])) as cropped:
            assert cropped.size == (100, 400)

"""Library-tier tests for :mod:`vtscore.media.image.decode`.

Pillow refuses to open an image past twice ``Image.MAX_IMAGE_PIXELS``, raising
``DecompressionBombError`` ("could be decompression bomb DOS attack") on the
header alone.  VTSearch lifts that ceiling — a user's gigapixel panorama is
large, not hostile — and replaces it with a bounded *decode* so peak memory
stays capped instead.  These tests pin both halves of that trade.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from vtscore.media.image import decode as decode_mod
from vtscore.media.image.decode import (
    configure_pil_limits,
    decode_bounded,
    decode_bounded_rgb,
    open_image,
)


def _encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
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
        # Native dimensions, not the bounded decode's: the stored bytes are the
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

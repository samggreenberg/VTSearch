"""Library-tier tests for :func:`vtscore.media.image.thumbnail.make_image_thumbnail`.

The thumbnail helper bounds the decoded size of grid/list tiles so a gallery
of many high-resolution items can't force the browser to decode every
full-size bitmap at once (the cause of the image-set browsing freeze).
"""

from __future__ import annotations

import io

from PIL import Image

from vtscore.media.image.thumbnail import DEFAULT_MAX_DIM, make_image_thumbnail


def _encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestMakeImageThumbnail:
    def test_downscales_large_image_within_max_dim(self):
        big = Image.new("RGB", (2000, 1200), color=(120, 30, 200))
        result = make_image_thumbnail(_encode(big, "JPEG"))
        assert result is not None
        thumb_bytes, mimetype = result
        assert mimetype == "image/jpeg"
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert max(out.size) <= DEFAULT_MAX_DIM
            # Aspect ratio preserved: longest side hits the cap.
            assert max(out.size) == DEFAULT_MAX_DIM
            assert out.size == (DEFAULT_MAX_DIM, round(DEFAULT_MAX_DIM * 1200 / 2000))

    def test_respects_custom_max_dim(self):
        big = Image.new("RGB", (1000, 1000), color=(10, 10, 10))
        result = make_image_thumbnail(_encode(big, "JPEG"), max_dim=64)
        assert result is not None
        thumb_bytes, _ = result
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert max(out.size) == 64

    def test_does_not_upscale_small_image(self):
        small = Image.new("RGB", (40, 30), color=(1, 2, 3))
        result = make_image_thumbnail(_encode(small, "PNG"))
        assert result is not None
        thumb_bytes, _ = result
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert out.size == (40, 30)

    def test_preserves_alpha_as_png(self):
        rgba = Image.new("RGBA", (800, 800), color=(0, 0, 0, 0))
        result = make_image_thumbnail(_encode(rgba, "PNG"))
        assert result is not None
        thumb_bytes, mimetype = result
        assert mimetype == "image/png"
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert out.mode == "RGBA"
            assert max(out.size) <= DEFAULT_MAX_DIM

    def test_thumbnail_is_smaller_than_source(self):
        big = Image.new("RGB", (3000, 3000), color=(200, 100, 50))
        source = _encode(big, "JPEG")
        result = make_image_thumbnail(source)
        assert result is not None
        thumb_bytes, _ = result
        assert len(thumb_bytes) < len(source)

    def test_returns_none_for_undecodable_bytes(self):
        assert make_image_thumbnail(b"<svg></svg>") is None
        assert make_image_thumbnail(b"not an image at all") is None

    def test_applies_exif_orientation(self):
        # A landscape image tagged "rotate 90°" (orientation 6) should come back
        # portrait once the orientation is baked in.
        landscape = Image.new("RGB", (400, 200), color=(50, 50, 50))
        exif = landscape.getexif()
        exif[274] = 6  # 274 = Orientation tag; 6 = rotate 90° CW
        buf = io.BytesIO()
        landscape.save(buf, format="JPEG", exif=exif)
        result = make_image_thumbnail(buf.getvalue())
        assert result is not None
        thumb_bytes, _ = result
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            # Width/height swapped relative to the stored 400×200.
            assert out.size[1] > out.size[0]

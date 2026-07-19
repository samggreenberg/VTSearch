"""Library-tier tests for :func:`vtscore.media.image.thumbnail.make_image_thumbnail`.

The thumbnail helper bounds the decoded size of grid/list tiles so a gallery
of many high-resolution items can't force the browser to decode every
full-size bitmap at once (the cause of the image-set browsing freeze).
"""

from __future__ import annotations

import io

from PIL import Image

from vtscore.media.image.thumbnail import (
    DEFAULT_MAX_DIM,
    _trim_solid_edges,
    make_image_thumbnail,
    normalize_region_crop,
)


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


class TestTrimSolidEdges:
    """Per-edge trimming of near-solid white/black padding (helper level)."""

    def _content_box(self, w, h, box, bg, fg):
        # A `bg`-filled frame with an `fg` rectangle `box=(x0,y0,x1,y1)` painted
        # in — the content the trim should home in on.
        img = Image.new("RGB", (w, h), color=bg)
        x0, y0, x1, y1 = box
        img.paste(Image.new("RGB", (x1 - x0, y1 - y0), color=fg), (x0, y0))
        return img

    def test_trims_black_letterbox_top_and_bottom(self):
        # 100px black bars top & bottom, red content in the middle 200px.
        img = self._content_box(400, 400, (0, 100, 400, 300), (0, 0, 0), (200, 30, 30))
        out = _trim_solid_edges(img)
        assert out.size[0] == 400  # full width kept (no side bars)
        assert abs(out.size[1] - 200) <= 6  # ~top/bottom bars removed

    def test_trims_white_pillarbox_left_and_right(self):
        # 100px white bars left & right, content in the middle 200px column.
        img = self._content_box(400, 400, (100, 0, 300, 400), (255, 255, 255), (40, 90, 160))
        out = _trim_solid_edges(img)
        assert out.size[1] == 400  # full height kept (no top/bottom bars)
        assert abs(out.size[0] - 200) <= 6

    def test_trims_a_single_padded_edge_independently(self):
        # White margin only on the left; the other three edges are content.
        img = self._content_box(400, 200, (120, 0, 400, 200), (255, 255, 255), (10, 120, 40))
        out = _trim_solid_edges(img)
        assert out.size[1] == 200  # top/bottom untouched
        assert abs(out.size[0] - 280) <= 6  # only the ~120px left bar removed

    def test_leaves_a_content_filled_frame_unchanged(self):
        img = Image.new("RGB", (300, 200), color=(120, 60, 180))
        out = _trim_solid_edges(img)
        assert out.size == (300, 200)

    def test_leaves_a_wholly_solid_frame_unchanged(self):
        # Nothing but solid tone — no content to pull the box toward.
        assert _trim_solid_edges(Image.new("RGB", (300, 200), (255, 255, 255))).size == (300, 200)
        assert _trim_solid_edges(Image.new("RGB", (300, 200), (0, 0, 0))).size == (300, 200)

    def test_caps_trim_so_a_tiny_subject_does_not_explode(self):
        # A 10px dot centred in a 400px white field: an uncapped trim would crop
        # to ~10px, but the per-edge cap keeps the middle 10% each way (40px).
        img = self._content_box(400, 400, (195, 195, 205, 205), (255, 255, 255), (200, 0, 0))
        out = _trim_solid_edges(img)
        assert out.size == (40, 40)

    def test_does_not_trim_a_solid_coloured_border(self):
        # A saturated (non white/black) border is content, not padding.
        img = self._content_box(400, 200, (100, 0, 300, 200), (0, 200, 0), (0, 0, 200))
        assert _trim_solid_edges(img).size == (400, 200)


class TestMakeImageThumbnailTrim:
    def test_letterboxed_source_thumbnail_reflects_content_aspect(self):
        # A square source that is really wide content between black bars should
        # produce a landscape thumbnail, not a square one.
        img = Image.new("RGB", (400, 400), color=(0, 0, 0))
        img.paste(Image.new("RGB", (400, 200), color=(180, 40, 40)), (0, 100))
        result = make_image_thumbnail(_encode(img, "JPEG"))
        assert result is not None
        with Image.open(io.BytesIO(result[0])) as out:
            assert out.size[0] > out.size[1]

    def test_explicit_crop_skips_solid_edge_trim(self):
        # With a voted region the user has already chosen content, so the padded
        # top-left quadrant is honoured verbatim rather than being trimmed away.
        img = Image.new("RGB", (400, 400), color=(255, 255, 255))
        img.paste(Image.new("RGB", (40, 40), color=(200, 0, 0)), (0, 0))
        result = make_image_thumbnail(_encode(img, "PNG"), crop=(0.0, 0.0, 0.5, 0.5))
        assert result is not None
        with Image.open(io.BytesIO(result[0])) as out:
            # The crop is a 200×200 square (mostly white); trimming would have
            # collapsed it toward the 40px dot, so a square result proves skip.
            assert out.size[0] == out.size[1]


class TestNormalizeRegionCrop:
    def test_accepts_valid_box(self):
        assert normalize_region_crop([0.1, 0.2, 0.6, 0.8]) == (0.1, 0.2, 0.6, 0.8)

    def test_orders_and_clamps_coordinates(self):
        # Reversed + out-of-range coords are canonicalised to x0<x1, y0<y1 in [0,1].
        assert normalize_region_crop([0.6, 0.9, 0.1, -0.5]) == (0.1, 0.0, 0.6, 0.9)

    def test_rejects_none_and_wrong_length(self):
        assert normalize_region_crop(None) is None
        assert normalize_region_crop([0.1, 0.2, 0.3]) is None
        assert normalize_region_crop("0,0,1,1") is None

    def test_rejects_non_numbers_and_nan(self):
        assert normalize_region_crop(["a", "b", "c", "d"]) is None
        assert normalize_region_crop([0.0, 0.0, float("nan"), 1.0]) is None

    def test_rejects_degenerate_zero_area(self):
        assert normalize_region_crop([0.5, 0.2, 0.5, 0.8]) is None  # zero width
        assert normalize_region_crop([0.2, 0.5, 0.8, 0.5]) is None  # zero height

    def test_rejects_near_full_image_box(self):
        # A box covering essentially the whole frame is treated as "no crop".
        assert normalize_region_crop([0.0, 0.0, 1.0, 1.0]) is None
        assert normalize_region_crop([0.0, 0.0, 0.995, 0.995]) is None


class TestMakeImageThumbnailCrop:
    def test_crops_to_region_before_downscaling(self):
        # A 1000×1000 source cropped to the left quarter-width / full height
        # yields a portrait thumbnail (taller than wide).
        big = Image.new("RGB", (1000, 1000), color=(10, 20, 30))
        result = make_image_thumbnail(_encode(big, "JPEG"), crop=(0.0, 0.0, 0.25, 1.0))
        assert result is not None
        thumb_bytes, _ = result
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert out.size[1] > out.size[0]
            assert max(out.size) <= DEFAULT_MAX_DIM

    def test_crop_differs_from_full_thumbnail(self):
        # Two visually distinct halves: cropping each gives different bytes, and
        # both differ from the uncropped thumbnail.
        img = Image.new("RGB", (400, 200))
        for x in range(400):
            for y in range(200):
                img.putpixel((x, y), (255, 0, 0) if x < 200 else (0, 0, 255))
        src = _encode(img, "PNG")
        full = make_image_thumbnail(src)
        left = make_image_thumbnail(src, crop=(0.0, 0.0, 0.5, 1.0))
        right = make_image_thumbnail(src, crop=(0.5, 0.0, 1.0, 1.0))
        assert full and left and right
        assert left[0] != right[0]
        assert left[0] != full[0]

    def test_sliver_box_widened_to_one_pixel(self):
        # A box so thin it rounds to zero pixels is widened, not crashed.
        big = Image.new("RGB", (500, 500), color=(0, 0, 0))
        result = make_image_thumbnail(_encode(big, "JPEG"), crop=(0.5, 0.0, 0.5001, 1.0))
        assert result is not None
        thumb_bytes, _ = result
        with Image.open(io.BytesIO(thumb_bytes)) as out:
            assert out.size[0] >= 1 and out.size[1] >= 1

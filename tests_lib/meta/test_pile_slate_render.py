"""The side-inset slate framing: the photo stays uncovered, and a drawn box converts back.

A box drawn on a side-inset render is normalised to the padded canvas, not the photo.
Band is computed from box area, so an unconverted redraw would mis-band -- silently.
These pin the conversion and the one property the framing exists for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def sr():
    """``slate_render``, which has no import side effects (unlike make_positive_slate)."""
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import slate_render

    return slate_render


def _photo(tmp_path: Path, w: int, h: int, name: str = "p.jpg") -> Path:
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", (w, h), (0, 90, 200)).save(p, quality=100)
    return p


class TestConversion:
    def test_a_box_round_trips_from_photo_to_canvas_and_back(self, sr):
        geom = {"orig_w": 640, "orig_h": 480, "canvas_w": 960, "canvas_h": 480}
        photo = [0.10, 0.20, 0.55, 0.65]
        sx, sy = geom["canvas_w"] / geom["orig_w"], geom["canvas_h"] / geom["orig_h"]
        canvas = [photo[0] / sx, photo[1] / sy, photo[2] / sx, photo[3] / sy]
        assert sr.side_inset_to_original(canvas, geom) == pytest.approx(photo)

    def test_a_box_drawn_into_the_padding_is_clamped_to_the_photo(self, sr):
        geom = {"orig_w": 640, "orig_h": 480, "canvas_w": 960, "canvas_h": 480}
        out = sr.side_inset_to_original([0.5, 0.0, 1.0, 1.0], geom)  # runs to the canvas's right edge
        assert out[2] == 1.0 and all(0.0 <= v <= 1.0 for v in out)

    def test_unconverted_it_would_misband(self, sr):
        """Why the conversion is not optional: the canvas-normalised area is smaller."""
        geom = {"orig_w": 640, "orig_h": 480, "canvas_w": 960, "canvas_h": 480}
        canvas = [0.2, 0.2, 0.6, 0.5]
        photo = sr.side_inset_to_original(canvas, geom)
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
        assert area(photo) > area(canvas) * 1.4


class TestRender:
    def test_it_always_pads_to_the_right_so_the_photo_keeps_its_height(self, sr, tmp_path):
        """Wide screens: height limits the photo's drawn size, so never pad below it."""
        box = (0.1, 0.1, 0.3, 0.3)
        for w, h, name in ((640, 480, "l.jpg"), (480, 640, "t.jpg")):  # landscape and portrait
            g = sr.draw_with_side_inset(_photo(tmp_path, w, h, name), box, tmp_path / f"o{name}")
            assert g["side"] == "right" and g["canvas_h"] == h and g["canvas_w"] > w, name

    def test_the_photo_is_not_covered(self, sr, tmp_path):
        """The whole point: the corner the old inset pasted over is untouched here."""
        from PIL import Image

        src = _photo(tmp_path, 640, 480)
        dest = tmp_path / "o.jpg"
        g = sr.draw_with_side_inset(src, (0.05, 0.05, 0.20, 0.20), dest)  # box top-left
        with Image.open(dest) as im:
            assert im.size == (g["canvas_w"], g["canvas_h"])
            # bottom-right of the PHOTO is where the old corner inset would have gone
            px = im.getpixel((620, 460))
        # Pillow types getpixel() as float | tuple | None (it is a scalar for
        # single-band modes); this is an RGB JPEG, so narrow it for the unpack.
        assert isinstance(px, tuple)
        r, gg, b = px[:3]
        assert abs(r - 0) < 20 and abs(gg - 90) < 20 and abs(b - 200) < 20

    def test_several_instances_are_drawn_thin_under_the_red_union(self, sr, tmp_path):
        """The band comes from the union, so the union is the red box and each annotation is amber.

        VG holds 21 overlapping boxes for one bench; all-red made the image unreadable.
        """
        from PIL import Image

        src = _photo(tmp_path, 640, 480)
        dest = tmp_path / "multi.jpg"
        g = sr.draw_with_side_inset(src, (0.05, 0.05, 0.15, 0.15), dest, also=[(0.60, 0.60, 0.75, 0.75)])
        with Image.open(dest) as im:
            assert im.size == (g["canvas_w"], g["canvas_h"])
            second = im.getpixel((int(0.60 * 640), int(0.675 * 480)))  # the second annotation's edge
            union_left = im.getpixel((int(0.05 * 640) + 1, int(0.40 * 480)))  # union's left edge, between the two
        assert isinstance(second, tuple) and second[0] > 180 and second[1] > 140, "an annotation is amber"
        assert isinstance(union_left, tuple) and union_left[0] > 180 and union_left[1] < 90, "the union is red"

    def test_no_extra_boxes_renders_exactly_as_before(self, sr, tmp_path):
        src = _photo(tmp_path, 640, 480)
        a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
        sr.draw_with_side_inset(src, (0.1, 0.1, 0.3, 0.3), a)
        sr.draw_with_side_inset(src, (0.1, 0.1, 0.3, 0.3), b, also=())
        assert a.read_bytes() == b.read_bytes()

    def test_the_panel_keeps_the_crops_aspect_ratio(self, sr, tmp_path):
        """A tall crop stays tall: the corner inset squashed non-square crops into a square."""
        src = _photo(tmp_path, 333, 500)
        g = sr.draw_with_side_inset(src, (0.03, 0.06, 0.99, 0.80), tmp_path / "tall.jpg")
        # box + context padding reaches every edge, so the crop is the whole 333x500 photo
        gap = 4  # max(4, 2 * line width) at this size
        pw, ph = g["canvas_w"] - 333 - 2 * gap, 500 - 2 * gap
        assert g["canvas_h"] == 500
        assert abs(pw / ph - 333 / 500) < 0.01, f"panel {pw}x{ph} is not 333:500"

    def test_the_panel_carries_no_outline_at_all(self, sr, tmp_path):
        """The photo says WHERE the box is; the panel says what is UNDER it, so nothing is drawn on it."""
        from PIL import Image

        src = _photo(tmp_path, 640, 480)
        dest = tmp_path / "boxed.jpg"
        g = sr.draw_with_side_inset(src, (0.40, 0.40, 0.60, 0.60), dest)
        with Image.open(dest) as im:
            photo = [im.getpixel((x, 240)) for x in range(640)]
            panel = [im.getpixel((x, y)) for x in range(646, g["canvas_w"]) for y in range(10, 470, 7)]
        red = lambda p: isinstance(p, tuple) and p[0] > 150 and p[1] < 110 and p[2] < 110  # noqa: E731
        assert sum(map(red, photo)) >= 2, "the photo still carries the box"
        assert not any(map(red, panel)), "nothing red is drawn over the panel"

    def test_the_canvas_stays_within_the_wide_screen_aspect(self, sr, tmp_path):
        """A small box on a landscape photo would magnify to a panel as wide as the photo; it is capped."""
        src = _photo(tmp_path, 800, 534)
        g = sr.draw_with_side_inset(src, (0.05, 0.02, 0.08, 0.06), tmp_path / "wide.jpg")
        assert g["canvas_w"] / g["canvas_h"] <= sr.MAX_CANVAS_ASPECT + 0.01

    def test_near_duplicate_annotations_are_collapsed_for_display(self, sr):
        """VG annotates one object many times; only distinct ones are outlined."""
        big = [0.10, 0.10, 0.50, 0.50]
        nudged = [0.11, 0.11, 0.51, 0.51]  # the same object, annotated again
        inside = [0.20, 0.20, 0.25, 0.25]  # wholly within the big one
        other = [0.70, 0.70, 0.90, 0.90]

        kept = sr.distinct_boxes([big, nudged, inside, other])

        assert [list(b) for b in kept] == [big, other]

    def test_the_union_still_comes_from_every_annotation(self, sr, tmp_path):
        """Dedupe is for drawing only: a duplicate outside the kept boxes still widens the red union."""
        from PIL import Image

        src = _photo(tmp_path, 640, 480)
        dest = tmp_path / "dup.jpg"
        sr.draw_with_side_inset(src, (0.10, 0.10, 0.50, 0.50), dest, also=[(0.11, 0.11, 0.95, 0.95)])
        with Image.open(dest) as im:
            edge = im.getpixel((int(0.95 * 640) - 1, int(0.50 * 480)))
        assert isinstance(edge, tuple) and edge[0] > 180 and edge[1] < 90, "the union reaches the far annotation"


class TestCanvasBoxToOriginal:
    """A reviewer draws on whichever half shows the object best; both convert."""

    def test_a_box_drawn_on_the_photo_is_a_rescale(self, sr, tmp_path):
        src = _photo(tmp_path, 640, 480)
        shown = [(0.40, 0.40, 0.60, 0.60)]
        g = sr.draw_with_side_inset(src, shown[0], tmp_path / "a.jpg")
        canvas = [0.40 * 640 / g["canvas_w"], 0.40, 0.60 * 640 / g["canvas_w"], 0.60]
        out, where = sr.canvas_box_to_original(canvas, g, shown)
        assert where == "photo"
        assert out == pytest.approx([0.40, 0.40, 0.60, 0.60], abs=1e-6)

    def test_a_box_drawn_on_the_panel_lands_on_the_same_object(self, sr, tmp_path):
        """The panel magnifies a known rectangle, so a box on it maps back through that rectangle."""
        src = _photo(tmp_path, 640, 480)
        shown = [(0.40, 0.40, 0.60, 0.60)]
        g = sr.draw_with_side_inset(src, shown[0], tmp_path / "b.jpg")
        # the object as the reviewer sees it in the panel: re-derive the panel's own geometry
        W, H, cw, ch = g["orig_w"], g["orig_h"], g["canvas_w"], g["canvas_h"]
        cx0, cy0, cx1, cy1 = sr.side_crop_rect(W, H, (0.40 * W, 0.40 * H, 0.60 * W, 0.60 * H))
        lw = max(2, int(min(W, H) * 0.006))
        gap = max(4, 2 * lw)
        pw = cw - W - 2 * gap
        ph = max(1, round((cy1 - cy0) * (pw / (cx1 - cx0))))
        ix, iy = W + gap, (ch - ph) // 2
        # the box's own corners, expressed on the panel
        corners = []
        for ox, oy in ((0.40 * W, 0.40 * H), (0.60 * W, 0.60 * H)):
            corners += [(ix + (ox - cx0) * pw / (cx1 - cx0)) / cw, (iy + (oy - cy0) * ph / (cy1 - cy0)) / ch]
        out, where = sr.canvas_box_to_original(corners, g, shown)

        assert where == "panel"
        assert out == pytest.approx([0.40, 0.40, 0.60, 0.60], abs=0.01)

    def test_a_photo_box_dragged_into_the_padding_is_clipped(self, sr, tmp_path):
        src = _photo(tmp_path, 640, 480)
        shown = [(0.10, 0.10, 0.90, 0.95)]
        g = sr.draw_with_side_inset(src, shown[0], tmp_path / "c.jpg")
        out, where = sr.canvas_box_to_original([0.05, 0.10, 640 / g["canvas_w"] + 0.01, 0.99], g, shown)
        assert where == "across both"
        assert out[2] == 1.0 and all(0.0 <= v <= 1.0 for v in out)


class TestCornerInset:
    """The corner framing (#3961): one scale for both axes, and a grey surround.

    It used to cap the magnified crop's width and height to the same target
    independently, which resized every non-square crop into a square, and to frame
    the result in red -- around the padded context rather than around the box.
    """

    @pytest.mark.parametrize(("w", "h"), [(800, 600), (480, 800), (333, 500), (600, 600)])
    def test_a_frame_filling_crop_is_magnified_at_the_photos_own_aspect(self, sr, tmp_path, w, h):
        """Box plus context padding reaches every edge, so the crop IS the photo: no re-derivation."""
        from PIL import Image

        with Image.open(_photo(tmp_path, w, h, f"{w}x{h}.jpg")) as im:
            crop, _, _ = sr.inset_crop(im.convert("RGB"), (0.03, 0.03, 0.97, 0.97))
        assert abs(crop.width / crop.height / (w / h) - 1.0) < 0.02, f"{crop.size} is not {w}:{h}"

    def test_a_non_square_crop_is_not_squashed_into_a_square(self, sr, tmp_path):
        """The bug itself: on a SQUARE photo, only the crop's own shape can make the inset non-square."""
        from PIL import Image

        with Image.open(_photo(tmp_path, 600, 600, "sq.jpg")) as im:
            im = im.convert("RGB")
            wide, _, _ = sr.inset_crop(im, (0.20, 0.47, 0.80, 0.53))  # 10:1 box -> a wide crop
            tall, _, _ = sr.inset_crop(im, (0.47, 0.20, 0.53, 0.80))  # its transpose
        assert wide.width > wide.height, f"wide crop rendered {wide.size}"
        assert tall.height > tall.width, f"tall crop rendered {tall.size}"
        assert wide.size == (tall.height, tall.width), "transposed boxes should transpose the inset"

    def test_the_inset_fills_its_target_on_the_longer_side_and_never_exceeds_it(self, sr, tmp_path):
        """The magnification the old 3x floor was meant to give, without the squash paying for it."""
        from PIL import Image

        with Image.open(_photo(tmp_path, 800, 600)) as im:
            im = im.convert("RGB")
            target = int(600 * sr.INSET_FRAC)
            for box in ((0.10, 0.40, 0.60, 0.52), (0.50, 0.50, 0.52, 0.53), (0.40, 0.05, 0.47, 0.75)):
                crop, _, _ = sr.inset_crop(im, box)
                assert max(crop.size) == target, f"{box}: {crop.size} does not reach {target}"
                assert min(crop.size) <= target, f"{box}: {crop.size} exceeds {target}"

    def test_the_inset_is_framed_in_grey_not_red(self, sr, tmp_path):
        """A red frame around the padded context reads as the box, which is drawn red too."""
        from PIL import Image

        box = (0.40, 0.40, 0.60, 0.60)
        src = _photo(tmp_path, 640, 480)
        dest = tmp_path / "corner.jpg"
        sr.draw_with_inset(src, box, dest)
        with Image.open(dest) as im:
            assert im.size == (640, 480), "the corner framing leaves the canvas alone"
            crop, _, _ = sr.inset_crop(im.convert("RGB"), box)
            inset = [im.getpixel((x, y)) for x in range(640 - crop.width, 640) for y in range(480 - crop.height, 480)]
            photo_box_edge = [im.getpixel((x, int(0.40 * 480) + 1)) for x in range(640 - crop.width)]
        red = lambda p: isinstance(p, tuple) and p[0] > 150 and p[1] < 110 and p[2] < 110  # noqa: E731
        assert sum(map(red, photo_box_edge)) >= 2, "the photo still carries the red box"
        assert not any(map(red, inset)), "nothing red is drawn in or around the inset"

    def test_the_inset_goes_in_whichever_bottom_corner_is_furthest_from_the_box(self, sr, tmp_path):
        """So the magnifier never covers the thing it is magnifying."""
        from PIL import Image

        src = _photo(tmp_path, 640, 480)
        for box, left in (((0.70, 0.70, 0.90, 0.90), True), ((0.10, 0.70, 0.30, 0.90), False)):
            dest = tmp_path / f"c{left}.jpg"
            sr.draw_with_inset(src, box, dest)
            with Image.open(dest) as im:
                crop, _, _ = sr.inset_crop(im.convert("RGB"), box)
                ix = 0 if left else 640 - crop.width
                # The photo is a flat colour and the inset magnifies more of the same, so the
                # grey frame is what says which corner it landed in.
                edge = im.getpixel((ix + 1, 480 - crop.height))
                far = im.getpixel(((640 - crop.width if left else 0) + 2, 478))
            assert isinstance(edge, tuple) and all(abs(c - 110) < 45 for c in edge[:3]), (
                f"{box}: no grey frame in the {'left' if left else 'right'} corner, got {edge}"
            )
            assert isinstance(far, tuple) and all(
                abs(a - b) < 25 for a, b in zip(far[:3], (0, 90, 200), strict=False)
            ), f"{box}: the other corner should be untouched photo, got {far}"

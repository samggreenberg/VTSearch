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

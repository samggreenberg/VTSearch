"""The "Data, Set" card: one layout every dataset in the deck is shown in.

Every dataset slide in `hold-the-line` is a frame of this one card, so the
room learns where to look once and then reads each new dataset by comparison:
the **name** under the slide's headline, a few **counts** in the left column,
the **url** along the foot, and the dataset's own media on the right. Nothing
else. A caption that explains what the pictures show is a sentence the
presenter says, so the card has no caption.

The name is the slide's subtitle — the headline says *Data, Set*, the name
says which one — so it is set in the headline's own face, weight and size and
hung just under it, rather than as a heading of the column below.

A dataset we built has no url, because there is nowhere to download it from,
and a repo path set in the url's place reads as one. Its card leaves the foot
empty.

The right-hand side comes in two shapes:

* **A grid** — `cols x rows` real frames, optionally with their boxes drawn and
  a caption under each (a category name, where the dataset has one per image).
* **A zoom** — one frame of the grid blown up as large as the media area
  allows, with a column beside it listing categories: the ones in the picture
  in the box blue, each joined by a line to its boxes, and the ones that are
  not in grey. That is exhaustive annotation in one picture — a grey name is a
  *checked* absence, not an unmentioned one. The zoom is not held to the size
  of the grid cells it came from: the picture is the point of the frame, so it
  takes every pixel the category column does not need.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from slide_figure import INK, SOFT, TITLE_NOTCH_PX


def _installed(face: str) -> bool:
    return any(f.name == face for f in font_manager.fontManager.ttflist)


FIG_W, FIG_H = 12.8, 7.2
FLOOR_PT = 15

_nx, _ny, _nw, _nh = TITLE_NOTCH_PX
NOTCH_R = (_nx + _nw) / 1280.0
NOTCH_B = (_ny + _nh) / 720.0

CUT = "#2b6cb0"  # the deck's .cut blue: boxes, and the names of what is present
NEG = "#c0392b"
POS = "#2e7d51"
RULE = "#d5d9df"
ABSENT = "#9aa3ae"  # a checked absence: quieter than SOFT, still legible

LEFT_X = 0.047
LEFT_TOP = 1.0 - NOTCH_B - 0.03

#: The title notch a "Data, Set" card clears. Its headline is one line, whose
#: box measures 56.8px (`slide_figure.TITLE_NOTCH_PX`), plus one
#: `OBJECT_GAP_PT` — 16pt, 22px at the card's 100 dpi — so the subtitle below it
#: clears the headline by the deck's own standard gap. The trim
#: `slides/STYLE.md` allows a one-line headline, as `vote-boundary` takes it.
DATA_SET_NOTCH_PX = (_nx, _ny, _nw, 80.0)

#: The dataset's name, set as the slide's subtitle. The face, weight and size
#: are the theme's `section.full h2` (`themes/vtsearch.css`): 40px, weight 600
#: of the deck's sans stack — at the card's 100 dpi a pixel is 0.72pt, and
#: matplotlib has no 600, so bold is the nearest it can draw. The first face of
#: the stack that is installed is the one used, as a browser would. The
#: baseline is where the cap height lands just under `DATA_SET_NOTCH_PX`.
NAME_FAMILY = next(
    (face for face in ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans") if _installed(face)),
    "DejaVu Sans",
)
NAME_PT = 40 * 0.72
NAME_BASELINE = 1.0 - 156 / 720
NAME_LEADING = 44.8 / 720
#: Where the counts start, whatever the name does above them: the column below
#: the subtitle is the card's own, and it stays put between datasets.
STATS_TOP = LEFT_TOP - 0.12
#: How wide a line of the name may run before it wraps: up to the media area.
NAME_MAX_W = NOTCH_R + 0.03 - LEFT_X
LEFT_W = NOTCH_R - LEFT_X + 0.01
#: The media area: right of the left column, above the url line.
AREA_X0, AREA_X1 = NOTCH_R + 0.045, 0.975
AREA_Y0, AREA_Y1 = 0.13, 0.955
URL_Y = 0.055


@dataclass
class Tile:
    """One frame on the card: pixels, and optionally what to draw over them."""

    image: Any  # PIL image, already cropped to the tile aspect
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # x, y, w, h in image pixels
    caption: str | None = None
    #: Boxes drawn in a second colour, for a figure that contrasts two kinds.
    alt_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    alt_colour: str = NEG
    box_labels: list[str] = field(default_factory=list)


def nice(n: int) -> str:
    """A count rounded the way it would be said: `886,284` is "900K".

    Two significant figures, or one when the leading digit is 5 or more — so
    the relative precision stays roughly even — and a K or M suffix from ten
    thousand up. Counts under a thousand are usually exact by design (80
    classes, 36 marks) and are left alone.

    >>> [nice(n) for n in (886_284, 123_287, 199_855, 8_677, 2_260, 13_216_456, 400)]
    ['900K', '120K', '200K', '9,000', '2,300', '13M', '400']
    """
    if n < 1_000:
        return f"{n:,}"
    digits = len(str(n))
    keep = 1 if str(n)[0] >= "5" else 2
    rounded = round(n, keep - digits)
    if rounded >= 1_000_000:
        return f"{rounded / 1_000_000:g}M"
    if rounded >= 10_000:
        return f"{rounded / 1_000:g}K"
    return f"{rounded:,}"


#: The name a merged quarry class goes by on a slide. The config names the
#: union it is (`enclosed road vehicle`) so nobody mistakes it for COCO's own
#: `car`; a slide just says Car. Every class name is *a* definition anyway —
#: nobody writes "bird, alive or dead, not cooked" — so the long names buy the
#: room nothing.
CLASS_DISPLAY = {
    "enclosed road vehicle": "Car",
    "single serving drinking vessel": "Cup",
    "vase or potted plant": "Vase",
    "bag or luggage": "Bag",
    "tv": "TV",
}


def display_class(name: str) -> str:
    """A quarry class as a slide shows it: capitalised, merges by their short name.

    Capitalised on purpose. `Cup` reads as a proper name, which is the point —
    it is this dataset's definition of a cup, not the word's.
    """
    return CLASS_DISPLAY.get(name) or " ".join(word.capitalize() for word in name.split())


def blank() -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    return fig


def fit_tile(image: Any, aspect: float = 4 / 3) -> tuple[Any, tuple[int, int]]:
    """Centre-crop to `aspect`; returns the crop and the pixels it removed (left, top)."""
    width, height = image.size
    if width / height > aspect:
        side = int(round(height * aspect))
        left = (width - side) // 2
        box = (left, 0, left + side, height)
    else:
        side = int(round(width / aspect))
        top = (height - side) // 2
        box = (0, top, width, top + side)
    return image.convert("RGB").crop(box), (box[0], box[1])


def _name_width(fig: plt.Figure, text: str) -> float:
    probe = fig.text(0, 0, text, fontsize=NAME_PT, fontweight="bold", family=NAME_FAMILY)
    width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    probe.remove()
    return width


def _name_lines(fig: plt.Figure, name: str) -> list[str]:
    """The name on one line if it fits, else on the two most even lines that do.

    Balanced rather than greedy, for the reason `slides/STYLE.md` gives for
    headlines: the browser's greedy wrap is the most lopsided split available,
    and this is the headline's subtitle, set in the headline's type.
    """
    if _name_width(fig, name) <= NAME_MAX_W:
        return [name]
    words = name.split()
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    fits = [pair for pair in splits if max(_name_width(fig, line) for line in pair) <= NAME_MAX_W]
    if not fits:
        raise SystemExit(f"data card {name!r}: the name does not fit the column on two lines")
    return list(min(fits, key=lambda pair: abs(_name_width(fig, pair[0]) - _name_width(fig, pair[1]))))


def left_column(fig: plt.Figure, name: str, stats: list[tuple[str, str]], url: str | None) -> None:
    """Name, counts and url: the part of the card that stays put between frames.

    `url` is None for a dataset we built, which has nowhere to be fetched from.
    """
    lines = _name_lines(fig, name)
    for i, line in enumerate(lines):
        fig.text(
            LEFT_X,
            NAME_BASELINE - i * NAME_LEADING,
            line,
            fontsize=NAME_PT,
            color=INK,
            fontweight="bold",
            family=NAME_FAMILY,
            va="baseline",
        )
    y = STATS_TOP
    if NAME_BASELINE - (len(lines) - 1) * NAME_LEADING - 0.03 < y:
        raise SystemExit(f"data card {name!r}: the name runs into the counts under it")
    for value, label in stats:
        fig.text(LEFT_X, y, value, fontsize=FLOOR_PT + 9, color=INK, fontweight="bold", va="top")
        lines = textwrap.wrap(label, 26)
        fig.text(LEFT_X, y - 0.050, "\n".join(lines), fontsize=FLOOR_PT, color=SOFT, va="top", linespacing=1.3)
        y -= 0.058 + 0.040 * len(lines) + 0.030
    if y < URL_Y + 0.06:
        raise SystemExit(f"data card {name!r}: the left column runs into the url line (bottom at {y:.3f})")
    if url is not None:
        fig.text(LEFT_X, URL_Y, url, fontsize=FLOOR_PT + 1, color=CUT, va="center", family="monospace")


def _grid_geometry(cols: int, rows: int, aspect: float, caption: bool) -> tuple[float, float, float, float]:
    """`(x0, y_top, cell_w, cell_h)` in figure fractions, the grid centred in the media area."""
    cap = 0.045 if caption else 0.0
    area_w, area_h = AREA_X1 - AREA_X0, AREA_Y1 - AREA_Y0
    # A tile's height in figure fractions, for a given width: the figure is 16:9.
    cell_w = min(area_w / cols, (area_h / rows - cap) * aspect * FIG_H / FIG_W)
    cell_h = cell_w / aspect * FIG_W / FIG_H + cap
    x0 = AREA_X0 + (area_w - cell_w * cols) / 2
    y_top = AREA_Y1 - (area_h - cell_h * rows) / 2
    return x0, y_top, cell_w, cell_h


def _draw_tile(fig: plt.Figure, rect: list[float], tile: Tile, lw: float = 1.3) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.imshow(tile.image, aspect="auto")
    width, height = tile.image.size
    for x, y, w, h in tile.boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=CUT, lw=lw))
    for (x, y, w, h), label in zip(tile.boxes, tile.box_labels):
        # Above the box, unless that would leave the picture — then inside it.
        inside = y < 0.09 * height
        ax.text(
            x,
            y + (3 if inside else -4),
            label,
            color="white",
            fontsize=FLOOR_PT,
            fontweight="bold",
            va="top" if inside else "bottom",
            bbox={"boxstyle": "square,pad=0.15", "facecolor": CUT, "edgecolor": "none"},
            clip_on=True,
        )
    for x, y, w, h in tile.alt_boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=tile.alt_colour, lw=lw, ls="--"))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(RULE)
    return ax


def grid(fig: plt.Figure, tiles: list[Tile], cols: int, rows: int, aspect: float = 4 / 3) -> None:
    """`cols x rows` frames, uniform, with captions under them if the tiles carry any."""
    caption = any(t.caption for t in tiles)
    x0, y_top, cell_w, cell_h = _grid_geometry(cols, rows, aspect, caption)
    pad = 0.006
    cap = 0.045 if caption else 0.0
    for i, tile in enumerate(tiles[: cols * rows]):
        r, c = divmod(i, cols)
        bottom = y_top - (r + 1) * cell_h
        _draw_tile(fig, [x0 + c * cell_w, bottom + cap + pad, cell_w - pad, cell_h - cap - pad], tile)
        if tile.caption:
            fig.text(
                x0 + c * cell_w + (cell_w - pad) / 2,
                bottom + cap - 0.004,
                tile.caption,
                ha="center",
                va="top",
                fontsize=FLOOR_PT,
                color=SOFT,
            )


def _nearest(px: float, py: float, box: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return min(max(px, x0), x1), min(max(py, y0), y1)


def interleave(
    categories: list[tuple[str, list[tuple[float, float, float, float]]]],
) -> list[tuple[str, list[tuple[float, float, float, float]]]]:
    """Present names in the order their boxes sit top to bottom, absent ones spread between.

    Ordering by the boxes is what keeps the connectors from crossing: a name
    halfway down the column is joined to something halfway down the picture.
    The absent names keep their given order and are dealt into the gaps evenly,
    so the column reads as one list rather than two.
    """
    present = sorted(
        (c for c in categories if c[1]),
        key=lambda c: sum(y + h / 2 for _, y, _, h in c[1]) / len(c[1]),
    )
    absent = [c for c in categories if not c[1]]
    out: list[tuple[str, list[tuple[float, float, float, float]]]] = []
    for i, item in enumerate(present):
        out.append(item)
        take = len(absent) * (i + 1) // len(present) - len(absent) * i // len(present) if present else 0
        out.extend(absent[len(absent) * i // len(present) : len(absent) * i // len(present) + take])
    return out if present else absent


#: The zoom's right-hand limit — the category column may run nearly to the
#: slide's edge, further than the grid does — and the gap between the picture
#: and the names.
ZOOM_X1 = 0.985
ZOOM_COL_GAP = 0.025


def zoom(
    fig: plt.Figure,
    tile: Tile,
    categories: list[tuple[str, list[tuple[float, float, float, float]]]],
    *,
    aspect: float = 4 / 3,
    more: bool = True,
    sort: bool = True,
) -> None:
    """One frame as large as the media area allows, and a category column beside it.

    `categories` is `[(name, boxes)]` in the order to list them; an empty box
    list means *checked absent* and the name is set in grey with no line. The
    boxes are drawn once, on the image, and each present name is joined to
    every one of its boxes at the point of the box nearest the name.
    """
    if sort:
        categories = interleave(categories)
    # The column's width is measured, and the picture takes the rest of the
    # media area — as large as that width and the area's height allow, centred
    # on the height it leaves.
    renderer = fig.canvas.get_renderer()
    col_w = 0.0
    for name, _ in categories:
        probe = fig.text(0, 0, name, fontsize=FLOOR_PT + 1, fontweight="bold")
        col_w = max(col_w, probe.get_window_extent(renderer).width / fig.bbox.width)
        probe.remove()
    pic_w = min(ZOOM_X1 - AREA_X0 - ZOOM_COL_GAP - col_w, (AREA_Y1 - AREA_Y0) * aspect * FIG_H / FIG_W)
    pic_h = pic_w / aspect * FIG_W / FIG_H
    rect = [AREA_X0, AREA_Y0 + (AREA_Y1 - AREA_Y0 - pic_h) / 2, pic_w, pic_h]
    shown = Tile(tile.image, [b for _, boxes in categories for b in boxes])
    _draw_tile(fig, rect, shown, lw=2.2)

    width, height = tile.image.size

    def to_fig(x: float, y: float) -> tuple[float, float]:
        return rect[0] + x / width * rect[2], rect[1] + (1 - y / height) * rect[3]

    col_x = rect[0] + rect[2] + ZOOM_COL_GAP
    n = len(categories) + (1 if more else 0)
    top, bottom = rect[1] + rect[3], rect[1]
    pitch = min(0.052, (top - bottom) / max(n, 1))
    y = top - pitch / 2 - (top - bottom - pitch * n) / 2
    overlay = fig.add_axes([0, 0, 1, 1], facecolor="none")
    overlay.set_xlim(0, 1)
    overlay.set_ylim(0, 1)
    overlay.axis("off")
    for name, boxes in categories:
        present = bool(boxes)
        label = overlay.text(
            col_x,
            y,
            name,
            fontsize=FLOOR_PT + 1,
            color=CUT if present else ABSENT,
            fontweight="bold" if present else "normal",
            va="center",
            ha="left",
        )
        right = label.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
        if right > 0.99:
            raise SystemExit(f"zoom column: {name!r} runs off the slide (right edge {right:.3f})")
        for bx, by, bw, bh in boxes:
            fx0, fy1 = to_fig(bx, by)
            fx1, fy0 = to_fig(bx + bw, by + bh)
            tx, ty = _nearest(col_x - 0.008, y, (fx0, fy0, fx1, fy1))
            overlay.add_line(Line2D([col_x - 0.008, tx], [y, ty], color=CUT, lw=1.3, alpha=0.9))
        y -= pitch
    if more:
        overlay.text(col_x, y, "…", fontsize=FLOOR_PT + 1, color=ABSENT, va="center", ha="left")

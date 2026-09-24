#!/usr/bin/env python
"""The "Data, Set" frames: every dataset in the deck, drawn as one card.

Run from the repo root:

    python slides/figs/src/make-data-cards.py            # everything
    python slides/figs/src/make-data-cards.py --only coco

Each dataset is a frame of the same card (`data_card.py`): name, counts and
url on the left, the dataset's own media on the right. Where a dataset has
boxes, a second frame zooms one picture into the grid's 2x2 corner and lists
categories beside it — present ones in blue, joined to their boxes, absent ones
in grey — because that is the one picture that shows both what a box is and
what exhaustive annotation means.

Media come from `dataset_samples.py` (Caltech-101, COCO val2017) and
`docmarks_media.py` (the DocMarks sources), both of which download into the
gitignored `data/` and never into the tree. The selections are pinned by id, so
re-running redraws the same frames.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import io
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_card as dc  # noqa: E402
from slide_figure import FULL_BLEED, save  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "slides" / "figs"


def _save(fig: Any, name: str) -> None:
    """Photograph-carrying frames go out as WebP (slides/README.md), through `save`'s checks."""
    from PIL import Image

    png = f"{name}.png"
    # A "Data, Set" card's headline is one line, and its subtitle — the
    # dataset's name — sits in the half of the reserve that line leaves.
    notch = dc.DATA_SET_NOTCH_PX if name.startswith("data-set-") else True
    with contextlib.redirect_stdout(io.StringIO()):
        save(fig, OUT, png, column=FULL_BLEED, tight=False, notch=notch)
    with Image.open(OUT / png) as image:
        image.convert("RGB").save(OUT / f"{name}.webp", quality=88, method=6)
    (OUT / png).unlink()
    print(f"wrote figs/{name}.webp")


# --------------------------------------------------------------------------
# Caltech-101
# --------------------------------------------------------------------------

#: How each sampled directory is named on the slide: a capitalised plural,
#: whatever the directory happens to be called (the source mixes `Leopards`
#: with `laptop` and `grand_piano`).
CALTECH_NAMES = {
    "airplanes": "Airplanes",
    "Leopards": "Leopards",
    "Motorbikes": "Motorbikes",
    "chandelier": "Chandeliers",
    "grand_piano": "Grand Pianos",
    "sunflower": "Sunflowers",
    "laptop": "Laptops",
    "umbrella": "Umbrellas",
    "watch": "Watches",
    "starfish": "Starfish",
    "ketch": "Ketches",
    "brain": "Brains",
}


def frame_caltech() -> Any:
    from PIL import Image

    import dataset_samples

    picks = dataset_samples.caltech101_samples()
    root = dataset_samples.caltech101_dir()
    # `BACKGROUND_Google` is a 102nd directory and not one of the 101.
    categories = [c for c in dataset_samples.caltech101_categories() if not c.startswith("BACKGROUND")]
    images = sum(len(list((root / c).glob("*.jpg"))) for c in categories)

    fig = dc.blank()
    dc.left_column(
        fig,
        "Caltech-101",
        [(dc.nice(images), "images"), (f"{len(categories)}", "categories, one per image"), ("0", "boxes")],
        "data.caltech.edu/records/mzrjq-6wc02",
    )
    tiles = [dc.Tile(dc.fit_tile(Image.open(p))[0], caption=CALTECH_NAMES[c]) for c, p in picks]
    dc.grid(fig, tiles, cols=4, rows=3)
    return fig


# --------------------------------------------------------------------------
# COCO
# --------------------------------------------------------------------------

#: COCO 2017, train and val together, counted from `instances_train2017.json`
#: and `instances_val2017.json` (2026-09-24). The card draws val2017 frames
#: only — that is the half `coco_fixture.py` keeps on disk — so these three
#: are the one set of numbers here that is typed rather than counted live.
COCO_IMAGES = 123_287
COCO_OBJECTS = 886_284  # boxes, crowd regions excluded

#: Six val2017 frames for the 3x2 grid: rooms, streets and tables with several
#: classes each. `COCO_ZOOM` is one of them, the one the next frame blows up.
COCO_GRID = (139, 67616, 96001, 139099, 350148, 246968)
#: The zoom: a museum case holding Mary Poppins' umbrella, beside an open book
#: and a bowl — three classes, every box plainly what it says it is (the desk
#: this replaced had a card reader boxed as a `remote`, #4176) — and, just as
#: plainly, things nobody boxed: a glass paperweight, a placard, a glass hen.
#: Those are the frame's second point: COCO answers for its 80 classes on every
#: image, and *only* for those, so an object outside them is simply not there
#: as far as the annotation goes.
COCO_ZOOM = 96001
#: Drawn whole, at the photograph's own 3:2 rather than the grid's 4:3: the
#: umbrella's box runs the full width of the frame, and a 4:3 crop cut its two
#: ends off and left it as a pair of loose lines.
COCO_ZOOM_ASPECT = 3 / 2
#: The category column for the zoom, in the order listed: what is in the case,
#: interleaved with what a display like it plausibly *could* hold and does not.
#: As long as the column has room for — the zoom is as tall as the media area,
#: and every grey name is one more checked absence the room can read. No `cat`
#: or `dog`, though both are absent as boxes: the case holds china figurines of
#: both, and a grey "cat" beside a china cat asks a question this frame is not
#: about.
COCO_ZOOM_LIST = (
    "book",
    "cup",
    "bowl",
    "vase",
    "umbrella",
    "clock",
    "scissors",
    "bottle",
    "wine glass",
    "cell phone",
    "spoon",
    "teddy bear",
    "handbag",
    "potted plant",
)


@functools.cache
def _coco() -> tuple[Path, dict, dict[int, list[dict]], dict[int, str]]:
    import dataset_samples

    images, coco = dataset_samples.coco_val()
    by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for a in coco["annotations"]:
        if not a["iscrowd"]:
            by_image[a["image_id"]].append(a)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    return images, coco, by_image, names


def _coco_tile(image_id: int, keep=None, rename=None, aspect: float = 4 / 3) -> tuple[dc.Tile, list[tuple[str, tuple]]]:
    """A val2017 frame cropped to `aspect` (4:3), with its boxes moved by the same crop.

    `keep(name)` filters which classes are drawn; `rename(name)` maps a COCO
    class to the name the card shows. Returns the tile and `(name, box)` pairs.
    """
    from PIL import Image

    images, coco, by_image, names = _coco()
    meta = next(i for i in coco["images"] if i["id"] == image_id)
    with Image.open(images / meta["file_name"]) as frame:
        tile, (dx, dy) = dc.fit_tile(frame, aspect)
    pairs = []
    for a in by_image[image_id]:
        name = names[a["category_id"]]
        if keep is not None and not keep(name):
            continue
        x, y, w, h = a["bbox"]
        pairs.append(((rename or (lambda n: n))(name), (x - dx, y - dy, w, h)))
    return dc.Tile(tile, [b for _, b in pairs]), pairs


def _coco_left(fig: Any) -> None:
    _, coco, _, _ = _coco()
    dc.left_column(
        fig,
        "COCO 2017",
        [
            (dc.nice(COCO_IMAGES), "images"),
            (f"{len(coco['categories'])}", "classes, every one checked on every image"),
            (dc.nice(COCO_OBJECTS), "objects, each boxed"),
        ],
        "cocodataset.org",
    )


def frame_coco_grid() -> Any:
    fig = dc.blank()
    _coco_left(fig)
    dc.grid(fig, [_coco_tile(i)[0] for i in COCO_GRID], cols=3, rows=2)
    return fig


def frame_coco_zoom() -> Any:
    fig = dc.blank()
    _coco_left(fig)
    tile, pairs = _coco_tile(COCO_ZOOM, aspect=COCO_ZOOM_ASPECT)
    present = collections.defaultdict(list)
    for name, box in pairs:
        present[name].append(box)
    missing = set(present) - set(COCO_ZOOM_LIST)
    if missing:
        raise SystemExit(f"coco zoom: {sorted(missing)} are boxed in {COCO_ZOOM} but not listed")
    dc.zoom(fig, tile, [(name, present.get(name, [])) for name in COCO_ZOOM_LIST], aspect=COCO_ZOOM_ASPECT)
    return fig


# --------------------------------------------------------------------------
# What COCO does not give you for free
# --------------------------------------------------------------------------

#: The four problems, in the order the slide takes them. The left column lists
#: all four on every frame and sets the one being drawn at full weight — the
#: deck's outline trick, so the room always knows which of the four it is on.
COCO_PROBLEMS = (
    "Two names, one thing",
    "One set, many cuts",
    "Which one sets the size?",
    "A pile is not large",
)

#: `(annotation id, COCO's label)` pairs: the same kind of object, filed under
#: two classes. Picked by eye from val2017 crops; each id is one box.
#:
#: Drawn as a grid, not as a row of crops each at its own aspect: every
#: picture fills one cell of one size (`_crop_to`), so the three rows line up
#: column for column and every photograph is as large as its cell.
COCO_MERGE_PAIRS = (
    (((354441, "car"), (399097, "truck")), "enclosed road vehicle"),
    (((677789, "cup"), (2097457, "wine glass")), "single serving drinking vessel"),
    (((1156773, "vase"), (1954267, "potted plant")), "vase or potted plant"),
)
#: Frames b–d stay on the talk's running example, the book. A desk of books at
#: several sizes: the largest is the one on the left, under the dog.
COCO_SHELF = 309938
#: Two val2017 `book` boxes, each the largest non-crowd `book` box in its frame
#: and so the one that sets the frame's size, and both in the **large** band:
#: one round a single book held up to the camera, and one COCO drew round the
#: whole bottom shelf of a bookcase — a row of small books, large only by area.
#: `(image id, annotation id)`; the band is checked against `pile_config` when
#: the frame is drawn, so a band change that moves either fails here.
COCO_PILE_ONE = (551439, 1140019)
COCO_PILE_ROW = (183049, 1150908)


#: Where a problems frame's pictures start: right of the list, with an object
#: gap to spare after its longest line ("Which one sets the size?").
PROBLEMS_MEDIA_X0 = 0.345


def _problems_column(fig: Any, active: int, items: tuple[str, ...]) -> None:
    y = dc.LEFT_TOP - 0.01
    for i, item in enumerate(items):
        lit = i == active
        fig.text(
            dc.LEFT_X,
            y,
            f"{i + 1}  {item}",
            fontsize=dc.FLOOR_PT + 3,
            color=dc.INK if lit else dc.ABSENT,
            fontweight="bold" if lit else "normal",
            va="top",
        )
        y -= 0.085


def _ann(ann_id: int) -> tuple[dict, dict]:
    _, coco, _, _ = _coco()
    ann = next(a for a in coco["annotations"] if a["id"] == ann_id)
    meta = next(i for i in coco["images"] if i["id"] == ann["image_id"])
    return ann, meta


def _picture(fig: Any, rect: list[float], image: Any, boxes=(), dashed=(), labels=()) -> Any:
    """`image` letterboxed into `rect`, aspect kept, with boxes over it."""
    from matplotlib.patches import Rectangle

    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (x, y, w, h), colour, lw in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=colour, lw=lw))
    for x, y, w, h in dashed:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=dc.NEG, lw=3, ls="--"))
    for x, y, text, colour in labels:
        ax.text(
            x,
            y,
            text,
            color="white",
            fontsize=dc.FLOOR_PT + 2,
            fontweight="bold",
            va="bottom",
            bbox={"boxstyle": "square,pad=0.2", "facecolor": colour, "edgecolor": "none"},
        )
    return ax


def _crop_to(ann_id: int, aspect: float, pad: float = 0.06) -> Any:
    """One box's crop, grown to `aspect` (width / height) around the box.

    The padded box is widened or heightened — whichever it is short of — and
    slid back inside the frame where it runs off an edge, so the crop fills its
    cell with picture rather than being letterboxed into it. Where the frame is
    too small to grow into, it takes all there is.
    """
    from PIL import Image

    images, _, _, _ = _coco()
    ann, meta = _ann(ann_id)
    x, y, w, h = ann["bbox"]
    with Image.open(images / meta["file_name"]) as frame:
        frame = frame.convert("RGB")
    cw, ch = w * (1 + 2 * pad), h * (1 + 2 * pad)
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    scale = min(1.0, frame.width / cw, frame.height / ch)
    cw, ch = cw * scale, ch * scale
    left = min(max(0.0, x + w / 2 - cw / 2), frame.width - cw)
    top = min(max(0.0, y + h / 2 - ch / 2), frame.height - ch)
    return frame.crop((int(left), int(top), int(left + cw), int(top + ch)))


def frame_problem_merge() -> Any:
    """Three rows: two crops COCO files under two classes, and the one class they become.

    A grid of equal cells, so the rows align column for column, sized to the
    whole media area: two picture columns, then the arrow and the merged name.
    """
    fig = dc.blank()
    _problems_column(fig, 0, COCO_PROBLEMS)
    x0, gap, cell_w = PROBLEMS_MEDIA_X0, 0.018, 0.2385
    label_h, row_gap = 0.05, 0.035
    cell_h = (0.955 - 0.045 - 3 * label_h - 2 * row_gap) / 3
    aspect = cell_w / cell_h * dc.FIG_W / dc.FIG_H
    for row, (pair, merged) in enumerate(COCO_MERGE_PAIRS):
        top = 0.955 - row * (cell_h + label_h + row_gap)
        for col, (ann_id, label) in enumerate(pair):
            x = x0 + col * (cell_w + gap)
            _picture(fig, [x, top - cell_h, cell_w, cell_h], _crop_to(ann_id, aspect))
            fig.text(
                x + cell_w / 2,
                top - cell_h - 0.008,
                f"“{label}”",
                ha="center",
                va="top",
                fontsize=dc.FLOOR_PT + 1,
                color=dc.SOFT,
            )
        arrow_x = x0 + 2 * cell_w + gap + 0.02
        fig.text(arrow_x, top - cell_h / 2, "→", va="center", fontsize=dc.FLOOR_PT + 8, color=dc.CUT)
        fig.text(
            arrow_x + 0.04,
            top - cell_h / 2,
            dc.display_class(merged),
            va="center",
            fontsize=dc.FLOOR_PT + 3,
            color=dc.CUT,
            fontweight="bold",
        )
    return fig


def frame_problem_cuts() -> Any:
    import numpy as np
    from matplotlib.patches import Rectangle

    fig = dc.blank()
    _problems_column(fig, 1, COCO_PROBLEMS)
    fig.text(0.40, 0.935, "the same photographs, cut four ways", fontsize=dc.FLOOR_PT + 3, color=dc.SOFT, va="top")

    def dots(rect: list[float], cols: int, rows: int, hits: int, size: float, title: str) -> None:
        ax = fig.add_axes(rect)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        colours = np.array([dc.RULE] * (cols * rows), dtype=object)
        rng = np.random.default_rng(7)
        colours[rng.choice(cols * rows, hits, replace=False)] = dc.POS
        order = np.argsort(colours == dc.POS)
        ax.scatter(xs.ravel()[order], ys.ravel()[order], s=size, c=list(colours[order]), linewidths=0)
        fig.text(
            rect[0] + rect[2] / 2,
            rect[1] - 0.02,
            title,
            ha="center",
            va="top",
            fontsize=dc.FLOOR_PT + 3,
            color=dc.INK,
            fontweight="bold",
        )

    def frame(rect: list[float], side: float, title: str) -> None:
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=dc.SOFT, lw=2))
        ax.add_patch(Rectangle(((1 - side) / 2, (1 - side) / 2), side, side, facecolor=dc.CUT, ec="none"))
        fig.text(
            rect[0] + rect[2] / 2,
            rect[1] - 0.02,
            title,
            ha="center",
            va="top",
            fontsize=dc.FLOOR_PT + 3,
            color=dc.INK,
            fontweight="bold",
        )

    dots([0.40, 0.52, 0.25, 0.33], 10, 10, 10, 60, "1 in 10 is a book")
    dots([0.70, 0.52, 0.25, 0.33], 40, 25, 1, 9, "1 in 1,000 is a book")
    frame([0.445, 0.14, 0.16, 0.28], 0.07, "small books")
    frame([0.745, 0.14, 0.16, 0.28], 0.75, "large books")
    return fig


def frame_problem_largest() -> Any:
    from PIL import Image

    fig = dc.blank()
    _problems_column(fig, 2, COCO_PROBLEMS)
    images, coco, by_image, names = _coco()
    meta = next(i for i in coco["images"] if i["id"] == COCO_SHELF)
    cars = [a["bbox"] for a in by_image[COCO_SHELF] if names[a["category_id"]] == "book"]
    largest = max(cars, key=lambda b: b[2] * b[3])
    x0 = min(b[0] for b in cars)
    y0 = min(b[1] for b in cars)
    x1 = max(b[0] + b[2] for b in cars)
    y1 = max(b[1] + b[3] for b in cars)
    with Image.open(images / meta["file_name"]) as frame:
        image = frame.convert("RGB")
    boxes = [(b, "white", 1.4) for b in cars if b is not largest] + [(largest, dc.CUT, 3.5)]
    _picture(
        fig,
        [0.39, 0.13, 0.585, 0.80],
        image,
        boxes=boxes,
        dashed=[(x0, y0, x1 - x0, y1 - y0)],
        labels=[(largest[0], largest[1] - 6, "largest", dc.CUT), (x0, y0 - 6, "all of them", dc.NEG)],
    )
    return fig


def frame_problem_pile() -> Any:
    """Two photographs, two `book` boxes, both large by area: one book, and a row.

    Each box is the largest `book` box in its frame, so each is the one that
    sets that frame's size (the frame before), and both land in the **large**
    band. On the left that is right — one book, held up to the camera. On the
    right it is a whole shelf of small books under one box: large by area, and
    no book in it is.
    """
    from PIL import Image

    images, coco, _, _ = _coco()
    fig = dc.blank()
    _problems_column(fig, 3, COCO_PROBLEMS)
    shown = []
    for (image_id, ann_id), colour, dashed in ((COCO_PILE_ONE, dc.CUT, False), (COCO_PILE_ROW, dc.NEG, True)):
        ann, meta = _ann(ann_id)
        if ann["image_id"] != image_id:
            raise SystemExit(f"coco pile: annotation {ann_id} is not on image {image_id}")
        box = tuple(ann["bbox"])
        if _band(box[2] * box[3] / (meta["width"] * meta["height"])) != "large":
            raise SystemExit(f"coco pile: annotation {ann_id} is not in the large band")
        with Image.open(images / meta["file_name"]) as frame:
            shown.append((frame.convert("RGB"), box, colour, dashed))
    # As tall as the media area, unless the two side by side would be wider
    # than it; then as wide. Centred in it either way. It starts where the
    # merge frame's grid does, clear of the longest line in the column.
    gap, left = 0.03, PROBLEMS_MEDIA_X0
    widths = [image.width / image.height * dc.FIG_H / dc.FIG_W for image, *_ in shown]
    height = min(dc.AREA_Y1 - dc.AREA_Y0, (dc.AREA_X1 - left - gap) / sum(widths))
    x = left + (dc.AREA_X1 - left - gap - height * sum(widths)) / 2
    y = dc.AREA_Y0 + (dc.AREA_Y1 - dc.AREA_Y0 - height) / 2
    for (image, box, colour, dashed), unit in zip(shown, widths, strict=True):
        _picture(
            fig,
            [x, y, height * unit, height],
            image,
            boxes=[] if dashed else [(box, colour, 3.5)],
            dashed=[box] if dashed else [],
            labels=[(box[0] + 3, box[1] - 6, "“book”: large", colour)],
        )
        x += height * unit + gap
    return fig


# --------------------------------------------------------------------------
# coco_quarry — "COCO Quarry" on a slide
# --------------------------------------------------------------------------

#: `(val2017 image id, class, band)`: six quarry positives, each drawn with the
#: one region the cell carries — the class's largest instance. The class and
#: band are checked against `pile_config` when the frame is drawn, so a roster
#: or band change that moves one of these fails here rather than mislabelling it.
QUARRY_GRID = (
    (200839, "bus", "large"),
    (245513, "bird", "small"),
    (404484, "dog", "medium"),
    (484760, "clock", "small"),
    (367680, "enclosed road vehicle", "medium"),
    (358923, "umbrella", "large"),
)
#: The dog@medium frame from the grid, blown up: a dog, a person, a tv and a
#: potted plant — and a teddy bear, which is not a quarry class, so it gets no
#: box and no line. Each class shows its one region, the largest instance.
QUARRY_ZOOM = 404484
QUARRY_ZOOM_ABSENT = (
    "bird",
    "bus",
    "book",
    "chair",
    "bench",
    "clock",
    "bicycle",
    "enclosed road vehicle",
    "umbrella",
    "bag or luggage",
    "laptop",
)


@functools.cache
def _pile_config() -> Any:
    import importlib.util

    path = REPO / "scripts" / "experiments" / "pile" / "pile_config.py"
    spec = importlib.util.spec_from_file_location("_cards_pile_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quarry_name(coco_name: str) -> str | None:
    """The quarry class a COCO class is filed under, or None if it is not in *C*."""
    pc = _pile_config()
    merged = {member: name for name, members in pc.SCALE_CLASS_MERGES.items() for member in members}
    name = merged.get(coco_name, coco_name)
    return name if name in pc.SCALE_CLASSES else None


def _band(area_fraction: float) -> str:
    for band, (lo, hi) in _pile_config().BOX_BANDS.items():
        if lo <= area_fraction < hi:
            return band
    raise SystemExit(f"quarry: an area fraction of {area_fraction:.3f} is in no band")


def _largest_by_class(image_id: int) -> tuple[dc.Tile, dict[str, tuple], dict[str, float]]:
    """The 4:3 tile, the largest box of each quarry class in it, and each one's area fraction."""
    tile, pairs = _coco_tile(image_id, keep=lambda n: _quarry_name(n) is not None, rename=_quarry_name)
    _, coco, _, _ = _coco()
    meta = next(i for i in coco["images"] if i["id"] == image_id)
    area = meta["width"] * meta["height"]
    largest: dict[str, tuple] = {}
    for name, box in pairs:
        if name not in largest or box[2] * box[3] > largest[name][2] * largest[name][3]:
            largest[name] = box
    return tile, largest, {n: b[2] * b[3] / area for n, b in largest.items()}


def _quarry_left(fig: Any) -> None:
    pc = _pile_config()
    cells = len(pc.SCALE_CLASSES) * len(pc.BOX_BANDS) - len(pc.SCALE_DROPPED_CELLS)
    # Its name as a person says it, not as the config spells it; and no url,
    # because we built it and there is nowhere to download it from.
    dc.left_column(
        fig,
        "COCO Quarry",
        [
            (dc.nice(COCO_IMAGES), "images, all of COCO 2017"),
            (f"{len(pc.SCALE_CLASSES)}", "classes"),
            (f"{cells}", "cells: a class at a size"),
        ],
        None,
    )


def frame_quarry_grid() -> Any:
    fig = dc.blank()
    _quarry_left(fig)
    tiles = []
    for image_id, cls, band in QUARRY_GRID:
        base, largest, fractions = _largest_by_class(image_id)
        if cls not in largest or _band(fractions[cls]) != band:
            raise SystemExit(f"quarry grid: image {image_id} is not a {cls}@{band} positive")
        tiles.append(dc.Tile(base.image, [largest[cls]], caption=f"{dc.display_class(cls)}@{band}"))
    dc.grid(fig, tiles, cols=3, rows=2)
    return fig


def frame_quarry_zoom() -> Any:
    fig = dc.blank()
    _quarry_left(fig)
    tile, largest, _ = _largest_by_class(QUARRY_ZOOM)
    clash = set(largest) & set(QUARRY_ZOOM_ABSENT)
    if clash:
        raise SystemExit(f"quarry zoom: {sorted(clash)} are listed absent but present in {QUARRY_ZOOM}")
    categories = [(dc.display_class(name), [box]) for name, box in largest.items()] + [
        (dc.display_class(name), []) for name in QUARRY_ZOOM_ABSENT
    ]
    dc.zoom(fig, tile, categories)
    return fig


# --------------------------------------------------------------------------
# DocMarks: its four sources, what they do not give you, and the result
# --------------------------------------------------------------------------


def _datasheet_sources() -> dict[str, dict[str, int]]:
    """`{source: {pages, instances}}` from DATASHEET.md's Sources table."""
    import re

    text = (REPO / "scripts" / "experiments" / "docmarks" / "DATASHEET.md").read_text()
    section = text[text.index("### Sources") :]
    out = {}
    for line in section.splitlines()[4:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        name = re.sub(r"[*`]", "", cells[0]).split()[0]
        out[name] = {"pages": int(re.sub(r"\D", "", cells[2])), "instances": int(re.sub(r"\D", "", cells[4]))}
    return out


def _page_tile(image: Any, boxes: list[tuple], aspect: float = 3 / 4, top: bool = True) -> dc.Tile:
    """A document page cropped to `aspect`, keeping its top, boxes moved with it."""
    width, height = image.size
    if top and width / height < aspect:
        crop = image.crop((0, 0, width, int(round(width / aspect))))
        dx, dy = 0, 0
    else:
        crop, (dx, dy) = dc.fit_tile(image, aspect)
    return dc.Tile(crop, [(x - dx, y - dy, w, h) for x, y, w, h in boxes])


def _letterbox(image: Any, aspect: float = 4 / 3, pad: float = 0.12) -> Any:
    """`image` centred on white at `aspect`, with a margin: a mark shown whole."""
    from PIL import Image

    w, h = image.size
    W = max(w, h * aspect) * (1 + pad)
    H = W / aspect
    canvas = Image.new("RGB", (int(W), int(H)), "white")
    canvas.paste(image, (int((W - w) / 2), int((H - h) / 2)))
    return canvas


#: The SPODS and Tobacco800 cards draw the anchor pages of eight roster
#: classes each — real pages, with the boxes the source ships.
SPODS_PAGES = ("00003", "00011", "00014", "00023", "00129", "00293", "00514", "00546")
TOBACCO800_PAGES = (
    "aah97e00-page02_1",
    "aeq93a00",
    "afm90c00-first_1",
    "ajj10e00",
    "ald41a00-ernest",
    "bea6aa00",
    "ciy01a00-page02_1",
    "kan00d00",
)
STAVER_PAGES = (
    "stampds-00213",
    "stampds-00230",
    "stampds-00010",
    "stampds-00050",
    "stampds-00100",
    "stampds-00150",
    "stampds-00300",
    "stampds-00380",
)


def frame_spods() -> Any:
    import docmarks_media as dm

    counts = _datasheet_sources()["SPODS"]
    fig = dc.blank()
    dc.left_column(
        fig,
        "SPODS",
        [
            (dc.nice(counts["pages"]), "made-up official documents"),
            ("4", "masks a page: logo, stamp, signature, text"),
            ("0", "names for the marks"),
        ],
        "facweb.iitkgp.ac.in/~jay/spods",
    )
    tiles = []
    for stem in SPODS_PAGES:
        image, marks = dm.page("spods", stem)
        tiles.append(_page_tile(image, [b for k, b in marks if k in ("logo", "stamp")]))
    dc.grid(fig, tiles, cols=4, rows=2, aspect=3 / 4)
    return fig


def frame_tobacco800() -> Any:
    import docmarks_media as dm

    counts = _datasheet_sources()["Tobacco800"]
    fig = dc.blank()
    dc.left_column(
        fig,
        "Tobacco800",
        [
            (dc.nice(counts["pages"]), "scanned business letters, 1980s–90s"),
            ("412", "with a logo, boxed"),
            ("21", "logos seen more than once"),
        ],
        "tc11.cvc.uab.es/datasets/Tobacco800_1",
    )
    tiles = []
    for stem in TOBACCO800_PAGES:
        image, marks = dm.page("tobacco800", stem)
        tiles.append(_page_tile(image, [b for k, b in marks if k == "logo"]))
    dc.grid(fig, tiles, cols=4, rows=2, aspect=3 / 4)
    return fig


def frame_staver() -> Any:
    import docmarks_media as dm

    counts = _datasheet_sources()["StaVer"]
    fig = dc.blank()
    dc.left_column(
        fig,
        "StaVer",
        [(dc.nice(counts["pages"]), "German invoices, rubber-stamped"), ("0", "names for the stamps")],
        "madm.dfki.de/downloads-ds-staver",
    )
    tiles = []
    for stem in STAVER_PAGES:
        image, marks = dm.page("staver", stem)
        tiles.append(_page_tile(image, [b for _, b in marks]))
    dc.grid(fig, tiles, cols=4, rows=2, aspect=3 / 4)
    return fig


def frame_ucsf() -> Any:
    import docmarks_media as dm

    fig = dc.blank()
    dc.left_column(
        fig,
        "UCSF Industry Documents",
        # Measured live against the IDL Solr index (docmarks/README.md).
        [(dc.nice(13_216_456), "short tobacco-industry documents"), ("0", "boxes")],
        "industrydocuments.ucsf.edu",
    )
    tiles = [_page_tile(dm.ucsf_page(doc_id), []) for doc_id in dm.UCSF_PAGES]
    dc.grid(fig, tiles, cols=4, rows=2, aspect=3 / 4)
    return fig


DOCMARKS_PROBLEMS = (
    "Which marks are the same?",
    "Masks, not marks",
    "The haystack holds needles",
)
#: Five roster classes, every one a different mark by ruling: three leaf marks
#: and two chief engravings, one a near-mirror of the other.
LOOKALIKES = (
    ("tobacco800/logo_ald41a00-ernest_1", "tobacco800/logo_azb11c00_1", "tobacco800/logo_asg54f00_1"),
    ("tobacco800/logo_afm90c00-first_1_0", "tobacco800/logo_ciy01a00-page02_1_0"),
)
#: A SPODS stamp whose mask arrives in pieces: "Dy.Manager / NewEastZone".
FRAGMENTED = ("00129", 1)
#: The Lorillard logo as a Tobacco800 positive, and a UCSF letter carrying it.
NEEDLE = ("tobacco800/logo_ajj10e00_1", "ffbb0108")


def frame_docmarks_same() -> Any:
    import docmarks_media as dm

    fig = dc.blank()
    _problems_column(fig, 0, DOCMARKS_PROBLEMS)
    height = 0.27
    for row, ids in enumerate(LOOKALIKES):
        top = 0.93 - row * 0.44
        x = 0.42
        for class_id in ids:
            crop = _letterbox(dm.mark_crop(class_id), aspect=1.0, pad=0.05)
            width = height * dc.FIG_H / dc.FIG_W
            _picture(fig, [x, top - height, width, height], crop)
            x += width + 0.03
        if x - 0.03 > 0.975:
            raise SystemExit(f"look-alikes row {row}: runs to {x - 0.03:.3f}, off the slide")
    return fig


def frame_docmarks_mask() -> Any:
    from PIL import Image

    import docmarks_media as dm

    cfg, common, spods, _, _ = dm._modules()
    stem, index = FRAGMENTED
    image, marks = dm.page("spods", stem)
    kind, (x, y, w, h) = marks[index]
    _, gt = spods.find_tree(dm.spods_root())
    with Image.open(gt / kind / f"image ({int(stem)}).png") as mask:
        mask = mask.convert("L")
        pieces = [c.box for c in common.mask_components(mask)]
    pad = 0.15
    box = (int(x - pad * w), int(y - pad * h), int(x + w * (1 + pad)), int(y + h * (1 + pad)))
    inside = [
        (bx - box[0], by - box[1], bw, bh)
        for bx, by, bw, bh in pieces
        if bx >= box[0] and by >= box[1] and bx + bw <= box[2] and by + bh <= box[3]
    ]
    shown = mask.crop(box).point(lambda v: 255 if v > 127 else 0).convert("RGB")

    fig = dc.blank()
    _problems_column(fig, 1, DOCMARKS_PROBLEMS)
    rect_h = 0.36
    rect_w = rect_h * (box[2] - box[0]) / (box[3] - box[1]) * dc.FIG_H / dc.FIG_W
    _picture(fig, [0.42, 0.53, rect_w, rect_h], shown, boxes=[(b, dc.NEG, 1.8) for b in inside])
    fig.text(
        0.42 + rect_w / 2,
        0.51,
        f"the mask: {len(inside)} pieces",
        ha="center",
        va="top",
        fontsize=dc.FLOOR_PT + 2,
        color=dc.NEG,
        fontweight="bold",
    )
    _picture(fig, [0.42, 0.08, rect_w, rect_h], image.crop(box), boxes=[((x - box[0], y - box[1], w, h), dc.CUT, 3)])
    fig.text(
        0.42 + rect_w / 2,
        0.06,
        "the mark: one stamp",
        ha="center",
        va="top",
        fontsize=dc.FLOOR_PT + 2,
        color=dc.CUT,
        fontweight="bold",
    )
    return fig


def frame_docmarks_needle() -> Any:
    import docmarks_media as dm

    fig = dc.blank()
    _problems_column(fig, 2, DOCMARKS_PROBLEMS)
    class_id, doc_id = NEEDLE
    crop = _letterbox(dm.mark_crop(class_id), aspect=4 / 3, pad=0.1)
    width = 0.25
    _picture(fig, [0.40, 0.50, width, width * 0.75 * dc.FIG_W / dc.FIG_H], crop)
    fig.text(
        0.40 + width / 2,
        0.48,
        "Tobacco800: a positive",
        ha="center",
        va="top",
        fontsize=dc.FLOOR_PT + 2,
        color=dc.CUT,
        fontweight="bold",
    )
    page = dm.ucsf_page(doc_id)
    top = page.crop((0, 0, page.width, int(page.width * 0.75)))
    width = 0.34
    height = width * 0.75 * dc.FIG_W / dc.FIG_H
    _picture(fig, [0.64, 0.95 - height, width, height], top)
    fig.text(
        0.64 + width / 2,
        0.93 - height,
        "UCSF: a “negative”?",
        ha="center",
        va="top",
        fontsize=dc.FLOOR_PT + 2,
        color=dc.NEG,
        fontweight="bold",
    )
    fig.text(
        0.64 + width / 2,
        0.87 - height,
        "same archive, same letterhead",
        ha="center",
        va="top",
        fontsize=dc.FLOOR_PT,
        color=dc.SOFT,
    )
    return fig


#: `(class id, name on the slide)`: six roster marks, one per tile, two from
#: each kind of source — made-up documents, real invoices, real letters.
DOCMARKS_GRID = (
    ("spods/stamp_00293_1", "Elephant Stamp"),
    ("spods/stamp_00716_1", "Not Delivered"),
    ("staver/stamp_stampds-00230_0", "DFKI Receipt"),
    ("tobacco800/logo_aah97e00-page02_1_0", "Philip Morris Crest"),
    ("tobacco800/logo_kan00d00_1", "Rockefeller Seal"),
    ("ucsf/logo_rjr_script", "RJR Script"),
)
#: The zoom: the Philip Morris crest's anchor page, and marks it does not hold.
DOCMARKS_ZOOM = "tobacco800/logo_aah97e00-page02_1_0"
DOCMARKS_ZOOM_ABSENT = (
    "Elephant Stamp",
    "RJR Script",
    "Lorillard Crest",
    "Rockefeller Seal",
    "Mount Sinai",
)


def _docmarks_left(fig: Any) -> None:
    import docmarks_media as dm

    sources = _datasheet_sources()
    dc.left_column(
        fig,
        "DocMarks",
        [
            (dc.nice(sum(s["pages"] for s in sources.values())), "pages"),
            (f"{len(dm.datasheet_roster())}", "marks to find"),
            (dc.nice(sum(s["instances"] for s in sources.values())), "copies, every one checked"),
        ],
        None,  # ours, like COCO Quarry: nowhere to download it from
    )


def frame_docmarks_grid() -> Any:
    import docmarks_media as dm

    roster = set(dm.datasheet_roster())
    fig = dc.blank()
    _docmarks_left(fig)
    tiles = []
    for class_id, name in DOCMARKS_GRID:
        if class_id not in roster:
            raise SystemExit(f"docmarks grid: {class_id} is not on the datasheet's roster")
        tiles.append(dc.Tile(_letterbox(dm.mark_crop(class_id)), caption=name))
    dc.grid(fig, tiles, cols=3, rows=2)
    return fig


def frame_docmarks_zoom() -> Any:
    import docmarks_media as dm

    fig = dc.blank()
    _docmarks_left(fig)
    image, (x, y, w, h) = dm.anchor(DOCMARKS_ZOOM)
    # The top of the page, at the zoom's 4:3: letterheads live there.
    top = image.crop((0, 0, image.width, int(image.width * 3 / 4)))
    name = dict(DOCMARKS_GRID)[DOCMARKS_ZOOM]
    dc.zoom(fig, dc.Tile(top), [(name, [(x, y, w, h)])] + [(n, []) for n in DOCMARKS_ZOOM_ABSENT])
    return fig


# --------------------------------------------------------------------------


FRAMES = {
    "caltech": [("data-set-caltech", frame_caltech)],
    "coco": [("data-set-coco-grid", frame_coco_grid), ("data-set-coco-zoom", frame_coco_zoom)],
    "quarry": [("data-set-quarry-grid", frame_quarry_grid), ("data-set-quarry-zoom", frame_quarry_zoom)],
    "docmarks-sources": [
        ("data-set-spods", frame_spods),
        ("data-set-tobacco800", frame_tobacco800),
        ("data-set-staver", frame_staver),
        ("data-set-ucsf", frame_ucsf),
    ],
    "docmarks-problems": [
        ("docmarks-problems-same", frame_docmarks_same),
        ("docmarks-problems-mask", frame_docmarks_mask),
        ("docmarks-problems-needle", frame_docmarks_needle),
    ],
    "docmarks": [("data-set-docmarks-grid", frame_docmarks_grid), ("data-set-docmarks-zoom", frame_docmarks_zoom)],
    "coco-problems": [
        ("coco-problems-merge", frame_problem_merge),
        ("coco-problems-cuts", frame_problem_cuts),
        ("coco-problems-largest", frame_problem_largest),
        ("coco-problems-pile", frame_problem_pile),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=sorted(FRAMES), action="append")
    args = ap.parse_args()
    for group in args.only or FRAMES:
        for name, make in FRAMES[group]:
            _save(make(), name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

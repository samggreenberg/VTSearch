#!/usr/bin/env python
"""A contact sheet of what a detector found, for the Find slide.

    python slides/figs/src/results_grid.py <width> <height> <cols> <rows> <plan>

Takes `cols x rows` frames from the `book` folder of the named corpus and writes
one PNG to stdout, sized to
the same box a raw app screenshot occupies so `shoot-ui-figs.mjs` can pad it out
to the deck's 16:9 frame with exactly the same arithmetic. That is the point of
matching the box: the Find slide is a two-page build whose first page is the
dashboard, and a reveal is supposed to change what is *in* the frame rather
than move the frame (`slides/STYLE.md`).

The slide it serves used to be the verification screen with the results shoved
into a left-hand panel, which is a picture of a person checking their answers.
Not looking at your results in the tool is a *feature* — an autorun detector
mails you a list of references — so what the room should see is the pictures,
with no chrome anywhere near them (#3779).

**Named frames from the corpus's own `book` folder, rather than read off the
live ranking**, and that is a deliberate choice about what this deck is. It is a
cartoon of how the tool works, not a transcript of one session: the page has to
say "the few minutes bought you these", and a sheet whose bottom row is whatever
a twenty-three-vote head happened to rank eleventh spends the room's attention
on the ranking's mistakes — which are the subject of the *next* slide and of the
whole second half, and are not this page's argument to make.

Named rather than sampled because in *this* corpus the two are very different
pictures, and the reason is worth writing down. COCO files a frame under `book`
when its largest annotated box happens to be one, and `photos-prod` is
deliberately disjoint from `photos` (`coco_fixture.DISJOINT_FROM`), so the
training pile has already taken the forty-four frames where a book fills the
picture and what is left is mostly living rooms with a shelf somewhere in them.
Twelve at random came out as two people playing Wii, a man with a sandwich and a
cat on a duvet; twelve by largest box area came out barely better, because the
biggest book in a wide shot of a hallway is still a hallway. So the twelve are
listed, the way `make-book-figs.TILES` lists the ten frames of the intro figure
and for the same reason — a slide figure in this deck picks its pictures.

Each frame is centre-cropped square. A pile of results is not a set of
thumbnails at fifteen aspect ratios, and a ragged grid makes the eye read the
shapes before the subjects.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_fixture import FIXTURES  # noqa: E402

#: The sheet, in reading order, as COCO file names from `photos-prod/book/`.
#: Twelve of the production pile's forty that a person would call a photograph
#: *of* books: shelves, a case of paperbacks, an open book, four people reading.
#: Every one of them is a frame the detector really does rank highly; what is
#: chosen here is which twelve of them the slide spends its grid on.
SHEET = (
    "000000031248.jpg",  # bookcases either side of a fireplace
    "000000578545.jpg",  # reading in bed
    "000000172617.jpg",  # a case of paperbacks, spilled
    "000000379441.jpg",  # a wall of shelves, lamp-lit
    "000000125129.jpg",  # reading at a table
    "000000491497.jpg",  # bedroom shelves, floor to ceiling
    "000000233825.jpg",  # a bay-windowed room, shelves down one wall
    "000000458255.jpg",  # reading in bed, with a cat
    "000000571893.jpg",  # a shelf of spines over a piano
    "000000416534.jpg",  # a room with shelves behind a screen
    "000000547336.jpg",  # writing in a notebook, outdoors
    "000000484351.jpg",  # a classroom, shelves along the back
)

#: The white line between two frames, as a fraction of a cell. Thin on purpose:
#: this is a contact sheet and the subject is the pile, so the gutters should
#: separate the frames without becoming a grid the eye reads first.
GUTTER = 0.045


def sheet(width: int, height: int, cols: int, rows: int, paths: list[str]) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    gutter = round(GUTTER * width / cols)
    # Square cells, so the binding axis decides the size — the box is wider than
    # 3:2 and a 4x3 sheet is taller than it, so which one binds is not a
    # constant and cannot be assumed.
    cell = min((width - (cols - 1) * gutter) // cols, (height - (rows - 1) * gutter) // rows)
    top = (height - (rows * cell + (rows - 1) * gutter)) // 2
    left = (width - (cols * cell + (cols - 1) * gutter)) // 2
    for index, path in enumerate(paths[: cols * rows]):
        with Image.open(path) as frame:
            frame = frame.convert("RGB")
            side = min(frame.size)
            box = ((frame.width - side) // 2, (frame.height - side) // 2)
            frame = frame.crop((box[0], box[1], box[0] + side, box[1] + side))
            frame = frame.resize((cell, cell), Image.LANCZOS)
        row, col = divmod(index, cols)
        canvas.paste(frame, (left + col * (cell + gutter), top + row * (cell + gutter)))
    return canvas


def pick(plan: str, count: int) -> list[str]:
    """The sheet's frames, checked against the corpus that is supposed to hold them."""
    folder = FIXTURES / plan / "book"
    if len(SHEET) != count:
        raise SystemExit(f"the sheet lists {len(SHEET)} frames and the grid holds {count}")
    frames = [folder / name for name in SHEET]
    missing = [f.name for f in frames if not f.exists()]
    if missing:
        raise SystemExit(f"{folder} does not hold {', '.join(missing)} — re-pick the sheet or rebuild the corpus")
    return [str(f) for f in frames]


def main() -> None:
    width, height, cols, rows = (int(a) for a in sys.argv[1:5])
    sheet(width, height, cols, rows, pick(sys.argv[5], cols * rows)).save(sys.stdout.buffer, "PNG")


if __name__ == "__main__":
    main()

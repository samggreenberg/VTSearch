#!/usr/bin/env python
"""A contact sheet of what a detector found, for the Find slide.

    python slides/figs/src/results_grid.py <width> <height> <cols> <rows> <img>... > sheet.png

Reads the images in ranking order and writes one PNG to stdout, sized to the
same box a raw app screenshot occupies so `shoot-ui-figs.mjs` can pad it out to
the deck's 16:9 frame with exactly the same arithmetic. That is the point of
matching the box: the Find slide is a two-page build whose first page is the
dashboard, and a reveal is supposed to change what is *in* the frame rather
than move the frame (`slides/STYLE.md`).

The slide it serves used to be the verification screen with the results shoved
into a left-hand panel, which is a picture of a person checking their answers.
Not looking at your results in the tool is a *feature* — an autorun detector
mails you a list of references — so what the room should see is the pictures,
with no chrome anywhere near them (#3779).

Each frame is centre-cropped square. A ranking is not a set of thumbnails at
fifteen aspect ratios; it is a list, and a ragged grid makes the eye read the
shapes before the subjects.
"""

from __future__ import annotations

import sys

from PIL import Image

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


def main() -> None:
    width, height, cols, rows = (int(a) for a in sys.argv[1:5])
    paths = sys.argv[5:]
    if len(paths) < cols * rows:
        raise SystemExit(f"the ranking gave {len(paths)} frames — the sheet is {cols}x{rows}")
    sheet(width, height, cols, rows, paths).save(sys.stdout.buffer, "PNG")


if __name__ == "__main__":
    main()

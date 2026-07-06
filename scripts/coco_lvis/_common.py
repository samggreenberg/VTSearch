#!/usr/bin/env python3
"""Shared helpers for the COCO / LVIS annotation-visualization scripts.

COCO 2017 (val) and LVIS v1 (val) are staged under
``/exp/scale26/datasets/external/{COCO,LVIS}`` with an identical flattened
schema (see each dir's README), so the three viz scripts
(``render_annotations.py``, ``annotate_category.py``, ``show_image.py``) share
this module rather than duplicating the palette / zip-reading / drawing logic.

The staged layout both scripts rely on::

    <ds>/derived/objects_flat_*.jsonl.gz   one row per annotation, box
                                           normalized to [0,1] as (x0,y0,x1,y1),
                                           plus split + file_name
    <ds>/images/{val2017,train2017}.zip    the JPEGs, KEPT ZIPPED; a row's
                                           ``file_name`` (e.g. val2017/xxx.jpg)
                                           is exactly the zip member path

LVIS reuses the COCO image zips (its ``images/`` is a symlink to
``../COCO/images``) and its val images span BOTH train2017 and val2017, which is
why each row carries its own ``split``.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_EXTERNAL = Path("/exp/scale26/datasets/external")

# dataset name -> (derived extract, images dir) defaults.
DATASETS: dict[str, tuple[Path, Path]] = {
    "coco": (
        DEFAULT_EXTERNAL / "COCO" / "derived" / "objects_flat_val2017.jsonl.gz",
        DEFAULT_EXTERNAL / "COCO" / "images",
    ),
    "lvis": (
        DEFAULT_EXTERNAL / "LVIS" / "derived" / "objects_flat_lvis_v1_val.jsonl.gz",
        DEFAULT_EXTERNAL / "LVIS" / "images",
    ),
}


def resolve_paths(
    dataset: str,
    extract: Path | None,
    images_dir: Path | None,
) -> tuple[Path, Path]:
    """Resolve (extract, images_dir), letting explicit flags override defaults."""
    ext_default, img_default = DATASETS[dataset]
    ext = extract or ext_default
    img = images_dir or img_default
    for p in (ext, img):
        if not p.exists():
            raise SystemExit(
                f"missing {p}\nExpected the staged {dataset.upper()} dataset; run "
                f"fetch_{dataset}.sbatch under {DEFAULT_EXTERNAL}/{dataset.upper()} first."
            )
    return ext, img


# High-contrast palette; a category's box color is chosen deterministically so a
# given category always renders in the same color within and across images.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 215, 0),  # gold
    (40, 170, 200),  # cyan
    (220, 50, 50),  # red
    (60, 200, 90),  # green
    (160, 90, 220),  # purple
    (255, 140, 0),  # orange
    (0, 160, 160),  # teal
    (230, 80, 160),  # pink
    (110, 200, 40),  # lime
    (70, 120, 240),  # blue
    (200, 170, 60),  # mustard
    (250, 100, 100),  # salmon
    (140, 140, 140),  # grey
    (180, 220, 40),  # chartreuse
)


def color_for(name: str) -> tuple[int, int, int]:
    """Stable palette color for a category (same name -> same color)."""
    return _PALETTE[(hash(name) & 0x7FFFFFFF) % len(_PALETTE)]


def font() -> ImageFont.ImageFont:
    """A legible default font, robust across PIL builds (no bundled TTF)."""
    try:
        return ImageFont.load_default(size=14)  # Pillow 10+
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None  # type: ignore[return-value]


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white text depending on background luminance."""
    r, g, b = bg
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luma > 140 else (255, 255, 255)


def norm_category(s: str) -> str:
    """Canonicalize a category for matching: COCO 'traffic light' == LVIS
    'traffic_light'.  Lowercase, collapse spaces/underscores to one space."""
    return re.sub(r"[ _]+", " ", s.strip().lower())


class ZipImageReader:
    """Read JPEG bytes from the staged image zips, caching one handle per split.

    A row's ``file_name`` (``val2017/000000123.jpg`` / ``train2017/...``) is the
    exact zip member path; the split prefix selects ``<images_dir>/<split>.zip``.
    """

    def __init__(self, images_dir: Path) -> None:
        self._images_dir = images_dir
        self._handles: dict[str, zipfile.ZipFile] = {}

    def _zip_for(self, split: str) -> zipfile.ZipFile:
        zf = self._handles.get(split)
        if zf is None:
            zp = self._images_dir / f"{split}.zip"
            if not zp.exists():
                raise FileNotFoundError(f"no image zip for split {split!r}: {zp}")
            zf = zipfile.ZipFile(zp)
            self._handles[split] = zf
        return zf

    def open_rgb(self, split: str, file_name: str) -> Image.Image:
        raw = self._zip_for(split).read(file_name)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def close(self) -> None:
        for zf in self._handles.values():
            zf.close()
        self._handles.clear()

    def __enter__(self) -> ZipImageReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def draw_box(
    draw: ImageDraw.ImageDraw,
    box_px: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    ft: ImageFont.ImageFont,
    img_w: int,
    img_h: int,
) -> None:
    """Draw one outlined box plus a filled label chip clamped inside the image."""
    x0, y0, x1, y1 = box_px
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    if not label:
        return
    try:
        tb = draw.textbbox((0, 0), label, font=ft)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = 7 * len(label), 12
    pad = 2
    chip_w, chip_h = tw + 2 * pad, th + 2 * pad
    cx = min(max(0, int(x0)), max(0, img_w - chip_w))
    cy = int(y0) - chip_h
    if cy < 0:  # no room above the box; tuck the chip just inside the top edge
        cy = min(int(y0) + 1, max(0, img_h - chip_h))
    draw.rectangle((cx, cy, cx + chip_w, cy + chip_h), fill=color)
    draw.text((cx + pad, cy + pad), label, fill=_text_color(color), font=ft)


def draw_rows(
    img: Image.Image,
    rows: list[dict],
    ft: ImageFont.ImageFont,
    max_boxes: int | None,
    *,
    label: bool = True,
    forced_color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Denormalize each row's box against the image size and draw it.

    Boxes are drawn largest-area first so the ``max_boxes`` cap keeps the most
    prominent objects.  ``forced_color`` (used when highlighting a single
    category) overrides the per-category palette color.
    """
    w, h = img.width, img.height
    draw = ImageDraw.Draw(img)
    boxes = []
    for r in rows:
        x0, y0, x1, y1 = r["x0"] * w, r["y0"] * h, r["x1"] * w, r["y1"] * h
        area = (x1 - x0) * (y1 - y0)
        boxes.append((area, (x0, y0, x1, y1), str(r.get("name", ""))))
    boxes.sort(key=lambda b: b[0], reverse=True)
    for _area, box_px, name in boxes[:max_boxes]:
        color = forced_color or color_for(name)
        draw_box(draw, box_px, name if label else "", color, ft, w, h)
    return img


def collect_boxes(extract_path: Path, chosen: set[int]) -> dict[int, list[dict]]:
    """Stream the flattened extract, keeping only rows for ``chosen`` images.

    Cheap prefix filter: every line begins ``{"image_id":<id>,`` so we test the
    line prefix as a string before paying for ``json.loads``.
    """
    prefixes = {f'{{"image_id":{iid},': iid for iid in chosen}
    max_pref = max((len(p) for p in prefixes), default=0)
    by_image: dict[int, list[dict]] = {iid: [] for iid in chosen}
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            head = line[:max_pref]
            for pref, iid in prefixes.items():
                if head.startswith(pref):
                    by_image[iid].append(json.loads(line))
                    break
    return by_image


def load_all(extract_path: Path) -> dict[int, list[dict]]:
    """Stream the whole extract into image_id -> object rows (one pass)."""
    by_image: dict[int, list[dict]] = {}
    with gzip.open(extract_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            by_image.setdefault(int(row["image_id"]), []).append(row)
    return by_image


def file_name_of(rows: list[dict]) -> tuple[str, str]:
    """(split, file_name) for an image, taken from its first row."""
    r = rows[0]
    return str(r["split"]), str(r["file_name"])

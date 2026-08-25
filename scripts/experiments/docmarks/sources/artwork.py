"""Artwork pools for the synthetic stratum — the marks that get pasted.

Synthesis needs a supply of *distinct, real* marks.  Two pools are supported:

* **LogoDet-3K** — 3,000 logo categories, 158,652 images, ~200k boxed logo
  instances, Pascal-VOC XML annotations.  The upstream GitHub is reachable only
  from China; the Kaggle mirror is the practical route.  Cropping one clean
  instance per category yields a pool in the thousands, each a genuine logo
  rather than a synthetic glyph.
* **a local directory** — any PNGs, ideally with alpha.  This is the escape
  hatch for pools that need a manual download (the ICDAR 2023 ReST seal set,
  10,000 real Chinese official seals, is the obvious one: it sits behind an RRC
  competition registration, so it cannot be fetched unattended).

Whatever the pool, the contract is the same: RGBA images with the mark's ink in
the colour channels and everything else transparent.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # noqa: S405 - local dataset annotations
from pathlib import Path
from typing import Any, Iterator, Optional

from . import _common

KAGGLE_SLUG_LOGODET3K = "lyly99/logodet3k"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def fetch_logodet3k(raw_root: Path) -> Path:
    dest = raw_root / "logodet3k"
    _common.kaggle_download(KAGGLE_SLUG_LOGODET3K, dest)
    return dest


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def parse_voc(xml_text: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Pascal-VOC XML -> ``[(class name, (x, y, w, h)), ...]``."""
    root = ET.fromstring(xml_text)  # noqa: S314 - local dataset annotation file
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for obj in root.iter("object"):
        name_el = obj.find("name")
        box_el = obj.find("bndbox")
        if name_el is None or box_el is None or not name_el.text:
            continue
        try:
            xmin = int(round(float(box_el.findtext("xmin", ""))))
            ymin = int(round(float(box_el.findtext("ymin", ""))))
            xmax = int(round(float(box_el.findtext("xmax", ""))))
            ymax = int(round(float(box_el.findtext("ymax", ""))))
        except (TypeError, ValueError):
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        out.append((name_el.text.strip(), (xmin, ymin, xmax - xmin, ymax - ymin)))
    return out


def iter_logodet_instances(root: Path) -> Iterator[tuple[str, Path, tuple[int, int, int, int]]]:
    """Yield ``(class name, image path, box)`` for every annotated logo."""
    for xml_path in sorted(root.rglob("*.xml")):
        image_path = next(
            (xml_path.with_suffix(s) for s in (".jpg", ".jpeg", ".png") if xml_path.with_suffix(s).exists()),
            None,
        )
        if image_path is None:
            continue
        try:
            for name, box in parse_voc(xml_path.read_text(encoding="utf-8", errors="replace")):
                yield name, image_path, box
        except ET.ParseError:
            continue


def build_pool_from_logodet(
    root: Path,
    out_dir: Path,
    *,
    max_classes: int = 200,
    min_side_px: int = 64,
) -> dict[str, Path]:
    """Crop one clean instance per LogoDet-3K category into *out_dir*.

    "Clean" means the largest annotated instance of that category, since a big
    crop degrades gracefully to a small paste and the reverse does not.
    """
    from PIL import Image

    best: dict[str, tuple[int, Path, tuple[int, int, int, int]]] = {}
    for name, image_path, box in iter_logodet_instances(root):
        area = box[2] * box[3]
        if min(box[2], box[3]) < min_side_px:
            continue
        key = _slug(name)
        if key and (key not in best or area > best[key][0]):
            best[key] = (area, image_path, box)

    out_dir.mkdir(parents=True, exist_ok=True)
    pool: dict[str, Path] = {}
    for key, (_area, image_path, box) in sorted(best.items())[:max_classes]:
        dest = out_dir / f"{key}.png"
        if not dest.exists():
            with Image.open(image_path) as im:
                x, y, w, h = box
                im.convert("RGBA").crop((x, y, x + w, y + h)).save(dest)
        pool[key] = dest
    return pool


def load_pool_dir(directory: Path, *, limit: Optional[int] = None) -> dict[str, Path]:
    """Every image in *directory* as a ``{class name: path}`` artwork pool."""
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if limit is not None:
        files = files[:limit]
    return {_slug(p.stem): p for p in files}


def to_rgba_mark(image: Any, *, white_threshold: int = 245) -> Any:
    """Coerce artwork to RGBA with the paper knocked out.

    A LogoDet crop is a photo region with an opaque white-ish background; pasted
    as-is it lands on the page as a conspicuous white rectangle that any matcher
    could find from the rectangle alone.  Knocking out near-white pixels leaves
    the ink, which is what a real mark contributes.
    """
    import numpy as np

    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    if arr[..., 3].min() < 255:
        return rgba  # already has real alpha; trust it
    near_white = (arr[..., :3] >= white_threshold).all(axis=2)
    arr[..., 3] = np.where(near_white, 0, 255)
    from PIL import Image as _Image

    return _Image.fromarray(arr, mode="RGBA")

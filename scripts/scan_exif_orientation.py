#!/usr/bin/env python3
"""How many images in a corpus actually carry a non-trivial EXIF orientation?

VTSearch applies EXIF display orientation at decode
(:mod:`vtscore.media.image.decode`), so a phone photo shot sideways is embedded,
thumbnailed, OCR'd and cropped the way the user sees it.  That is a correctness
fix for ordinary uploads — but whether it *moves a number* on any corpus we
benchmark against is a separate, empirical question, and the two should not be
conflated in a report.

This is the measurement.  Point it at a folder of images, a dataset pickle, or
both, and it prints the histogram of orientation tags:

    python scripts/scan_exif_orientation.py /data/photos
    python scripts/scan_exif_orientation.py data/datasets/*.pkl
    python scripts/scan_exif_orientation.py --json /exp/scale26/images

Orientation ``1`` (and a missing tag) means upright; ``2``-``4`` are flips and
half-turns, which change the picture without changing its shape; ``5``-``8`` are
the quarter turns, which also transpose the stored ``width``/``height``.  Only
the last group can move a bounding box, so it is reported separately.

Reading the tag costs a header parse per file — no pixels are decoded — so a
large corpus scans at disk speed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Orientations whose upright form transposes the stored dimensions.
QUARTER_TURNS = (5, 6, 7, 8)

#: What Pillow will open without a plugin install.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif", ".heic", ".heif"}


def _orientation_of_bytes(blob: bytes) -> int | None:
    """Return the orientation tag, or ``None`` when the payload won't open."""
    import io

    from vtscore.media.image.decode import exif_orientation, open_image

    try:
        with open_image(io.BytesIO(blob)) as img:
            return exif_orientation(img)
    except Exception:
        return None


def _orientation_of_path(path: Path) -> int | None:
    from vtscore.media.image.decode import exif_orientation, open_image

    try:
        with open_image(path) as img:
            return exif_orientation(img)
    except Exception:
        return None


def _scan_folder(root: Path) -> Iterator[tuple[str, int | None]]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield str(path.relative_to(root)), _orientation_of_path(path)


def _scan_pickle(path: Path) -> Iterator[tuple[str, int | None]]:
    """Scan a dataset pickle's medias, preferring stored bytes over the origin path."""
    with open(path, "rb") as fh:
        payload: Any = pickle.load(fh)  # noqa: S301 - a dataset the operator chose
    medias = payload.get("medias", payload) if isinstance(payload, dict) else payload
    items = medias.values() if isinstance(medias, dict) else medias
    for media in items:
        if not isinstance(media, dict) or media.get("media_type") != "image":
            continue
        name = str(media.get("filename") or media.get("id") or "?")
        blob = media.get("media_bytes")
        if isinstance(blob, (bytes, bytearray)) and blob:
            yield name, _orientation_of_bytes(bytes(blob))
            continue
        media_path = media.get("media_path")
        yield name, (_orientation_of_path(Path(media_path)) if media_path else None)


def scan(target: Path) -> list[tuple[str, int | None]]:
    if target.is_dir():
        return list(_scan_folder(target))
    if target.suffix.lower() in {".pkl", ".pickle"}:
        return list(_scan_pickle(target))
    return [(target.name, _orientation_of_path(target))]


def summarise(rows: list[tuple[str, int | None]]) -> dict[str, Any]:
    counts = Counter(orientation for _name, orientation in rows)
    readable = sum(n for orientation, n in counts.items() if orientation is not None)
    rotated = sum(n for orientation, n in counts.items() if orientation not in (None, 1))
    transposed = sum(counts.get(o, 0) for o in QUARTER_TURNS)
    return {
        "images": len(rows),
        "unreadable": counts.get(None, 0),
        "readable": readable,
        "rotated": rotated,
        "transposed": transposed,
        "histogram": {str(k): v for k, v in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "examples": [name for name, o in rows if o not in (None, 1)][:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", type=Path, help="image folders, dataset pickles, or single files")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per target instead of a table")
    args = parser.parse_args()

    exit_code = 0
    for target in args.targets:
        if not target.exists():
            print(f"skip {target}: does not exist", file=sys.stderr)
            exit_code = 1
            continue
        report = summarise(scan(target))
        if args.json:
            print(json.dumps({"target": str(target), **report}))
            continue
        print(f"\n{target}")
        print(f"  images        {report['images']}")
        print(f"  unreadable    {report['unreadable']}")
        print(f"  rotated       {report['rotated']}  (orientation tag other than 1)")
        print(f"  transposed    {report['transposed']}  (quarter turns; these also swap width/height)")
        print(f"  histogram     {report['histogram']}")
        if report["examples"]:
            print(f"  e.g.          {', '.join(report['examples'])}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

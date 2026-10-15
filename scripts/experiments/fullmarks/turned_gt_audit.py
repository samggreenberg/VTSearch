#!/usr/bin/env python
"""Find Tobacco800 pages whose GEDI boxes are in the other frame (#4871).

    python turned_gt_audit.py --out turned_gt_audit.csv

One page's XML was drawn on the page turned the right way up while its TIF is
stored upside down, and the TIF's Orientation tag says "normal", so nothing
turns it.  Read verbatim, its logo box sat on blank paper and its signature box
on the memo header.  A blank crop hashes like every faint one, so the box was a
hub in every Tobacco800 re-cluster.

For every zone this measures the share of dark pixels in the box as the XML
gives it and in the box's 180-degree image.  A **logo** zone that is near-empty
where it is and inked once turned is flagged: logos are dense, so that is a
strong signal.  Signature zones are written to the CSV but not flagged.  A
signature is a few strokes, and its turned box lands on body text about as
often as not.

A flagged page goes in ``sources/tobacco800.TURNED_180`` once a person has
looked at it turned upright; then ``audit_to_corrections.py --turn-pages``
brings the built corpus into line.  On the 2026-10-14 mirror this flags one
page, ``dic45f00_1``, of 432 logo zones.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fullmarks_config as cfg  # noqa: E402
from sources.tobacco800 import _IMAGE_SUFFIXES, TURNED_180, read_gedi, turn_180  # noqa: E402

#: A pixel darker than this is ink (the scans are bilevel; any mid-point works).
INK_LEVEL = 128
#: Dark share at or under which a logo box is "near-empty".  The emptiest real
#: logo zone on the mirror is 5.6%; the flagged one is 0.0%.
EMPTY = 0.01
#: Dark share a turned logo box must reach to call the page turned.
INKED = 0.03


def dark_share(page: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    crop = page[max(y, 0) : y + h, max(x, 0) : x + w]
    return float((crop < INK_LEVEL).mean()) if crop.size else 0.0


def audit(unpacked: Path) -> list[dict[str, object]]:
    from PIL import Image  # noqa: PLC0415

    images = {p.stem.lower(): p for p in unpacked.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES}
    rows: list[dict[str, object]] = []
    for stem, marks in sorted(read_gedi(unpacked).items()):
        if not marks or stem not in images:
            continue
        with Image.open(images[stem]) as im:
            page = np.asarray(im.convert("L"))
        height, width = page.shape
        for index, mark in enumerate(marks):
            here = dark_share(page, mark.box)
            turned = dark_share(page, turn_180(mark.box, width, height))
            rows.append(
                {
                    "stem": stem,
                    "zone": index,
                    "kind": mark.kind,
                    "box": " ".join(map(str, mark.box)),
                    "dark": round(here, 4),
                    "dark_turned": round(turned, 4),
                    "flagged": mark.kind == "logo" and here <= EMPTY and turned >= INKED,
                    "in_table": stem in TURNED_180,
                }
            )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=cfg.RAW / "tobacco800", help="the unpacked Tobacco800 mirror")
    ap.add_argument("--out", type=Path, required=True, help="CSV, one row per GEDI zone")
    args = ap.parse_args(argv)

    rows = audit(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logos = [r for r in rows if r["kind"] == "logo"]
    flagged = sorted({str(r["stem"]) for r in rows if r["flagged"]})
    print(f"{len(rows)} zones on {len({r['stem'] for r in rows})} pages; {len(logos)} logo zones; wrote {args.out}")
    print(f"flagged: {flagged or 'none'}")
    missing = [s for s in flagged if s not in TURNED_180]
    if missing:
        print(f"NOT in TURNED_180 yet (look at each turned upright first): {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

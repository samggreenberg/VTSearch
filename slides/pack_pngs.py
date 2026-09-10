#!/usr/bin/env python
"""Pack a deck's rendered slide images into zips small enough to send.

    ./pack_pngs.py _out/scale-readout-png scale-readout --max-mb 25

`render.sh <deck> png` renders one PNG per page and then calls this. The zips
exist because the pile is the deliverable: a deck exported as images, every one
the same size, to be dragged into somebody *else's* slide template and dropped
in as pictures. That is the one export where the receiving deck owns the layout
and this one owns nothing but the pixels.

**Each part is a complete, ordinary zip**, not one volume of a split archive.
`zip -s` would be fewer lines and produces `.z01`, `.z02`, … which cannot be
opened without every sibling present — a bad trade for a file whose whole
purpose is being emailed to somebody. So pages are dealt into consecutive parts
under the cap, in order, and any one part opens on its own.

The uniform-size check is the other half of the job. "Every image is the same
size" is what lets the receiving deck place them with one drag and no
per-picture fiddling, and it is a property of the render rather than of this
script — so it is asserted here, where the whole set is in hand, rather than
assumed.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from pathlib import Path

#: A part's ceiling, in megabytes. 25 because that is the smallest attachment
#: limit these decks actually meet in the wild — mail gateways and chat apps
#: sit at 25 or 30 — and a part that clears the smallest clears them all.
DEFAULT_MAX_MB = 25.0


def png_size(path: Path) -> tuple[int, int]:
    """The pixel size in a PNG's IHDR, read without decoding the image."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise SystemExit(f"{path.name} is not a PNG")
    return struct.unpack(">II", header[16:24])


def pack(src: Path, prefix: str, max_bytes: int) -> list[Path]:
    pages = sorted(src.glob("*.png"))
    if not pages:
        raise SystemExit(f"no PNGs in {src}")

    sizes = {png_size(page) for page in pages}
    if len(sizes) != 1:
        raise SystemExit(
            f"the pages are not all one size ({', '.join(f'{w}x{h}' for w, h in sorted(sizes))}) — "
            "a pile that has to be dragged into another deck cannot be, so this is a render bug"
        )
    (width, height) = sizes.pop()

    oversized = [p for p in pages if p.stat().st_size > max_bytes]
    if oversized:
        raise SystemExit(f"{oversized[0].name} is bigger than one whole part — raise --max-mb")

    parts: list[list[Path]] = [[]]
    running = 0
    for page in pages:
        size = page.stat().st_size
        if parts[-1] and running + size > max_bytes:
            parts.append([])
            running = 0
        parts[-1].append(page)
        running += size

    written = []
    for index, part in enumerate(parts, start=1):
        name = f"{prefix}-pngs.zip" if len(parts) == 1 else f"{prefix}-pngs-part{index}.zip"
        out = src.parent / name
        # Stored, not deflated: a PNG is already compressed, so deflating it
        # again buys a percent or so and costs the time twice over.
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for page in part:
                archive.write(page, page.name)
        written.append(out)
        print(f"  {out.name}  {len(part)} pages, {out.stat().st_size / 1e6:.1f} MB")

    print(f"{len(pages)} pages at {width}x{height} -> {len(written)} zip(s)")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("src", type=Path, help="directory of rendered page PNGs")
    parser.add_argument("prefix", help="deck name, used for the zip file names")
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB, help="ceiling for one zip")
    args = parser.parse_args()
    pack(args.src, args.prefix, int(args.max_mb * 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())

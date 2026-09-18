#!/usr/bin/env python3
"""Stage the COCO image zips `coco_quarry` needs, resumably.

**This is the prerequisite nothing in the repo had, and it is why the embed
(#3988) cannot start before it.** `coco_anchor.py` fetches COCO's
*annotations* -- 241 MB of `instances_*2017.json` -- because until now the
*pixels* came from Visual Genome's own copies of the same photographs. Moving
the image pool off VG (#3983) makes the pixels a hard dependency, and 118,287 of
`coco_quarry`'s 123,287 images live in a zip that has never been downloaded.

Only ``val2017.zip`` (0.8 GB, 5,000 images) is staged today. ``train2017.zip`` is
**18.0 GB**, so this is an hours-long, purely I/O-bound job with no GPU in it --
which is exactly what makes it the right thing to start when the next decision
is not yet made. Nothing about #3985, #3986 or #3987 can invalidate a downloaded
image.

**Resumable on purpose.** An 18 GB transfer over a long window will be
interrupted, so the download uses an HTTP range request against a ``.part`` file
and is renamed into place only once complete -- the same shape as
``coco_anchor._fetch``, plus the resume. Re-running after any interruption
continues rather than restarting.

**Zips, not directories.** The builder reads members straight out of the archive,
as ``coco.py`` already does for val. That is deliberate: #3299 cost a day when
the rebuild canary checked an extracted directory while the builder opened a
zip, and reported a perfectly intact staging area as REBUILD-BROKEN.

    python stage_coco_images.py                 # stage whatever is missing
    python stage_coco_images.py --verify        # check what is there, fetch nothing
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402

BASE_URL = "http://images.cocodataset.org/zips"
#: ``(destination, expected member count)``. The counts are COCO 2017's own split
#: sizes and are what makes a truncated-but-openable archive detectable.
TARGETS: tuple[tuple[Path, int], ...] = (
    (pc.COCO_VAL_ZIP, 5_000),
    (pc.COCO_TRAIN_ZIP, 118_287),
)


def verify(path: Path, expected: int) -> str:
    """``ok`` / a description of what is wrong. Never raises on a bad archive."""
    if not path.exists():
        part = path.with_suffix(path.suffix + ".part")
        if part.exists():
            return f"MISSING (partial download present: {part.stat().st_size / 1e9:.1f} GB)"
        return "MISSING"
    try:
        with zipfile.ZipFile(path) as zf:
            n = sum(1 for m in zf.namelist() if m.lower().endswith(".jpg"))
    except zipfile.BadZipFile:
        return "CORRUPT (not a readable zip)"
    if n != expected:
        return f"SHORT ({n:,} images, expected {expected:,})"
    return f"ok ({n:,} images, {path.stat().st_size / 1e9:.1f} GB)"


def fetch(url: str, dest: Path) -> None:
    """Download *url* to *dest*, resuming a ``.part`` file if one is there."""
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)  # noqa: S310 - fixed http URL above
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"  resuming at {have / 1e9:.1f} GB")
    with urllib.request.urlopen(req) as r:  # noqa: S310
        if have and r.status != 206:
            # The server ignored the range; start over rather than append garbage.
            print("  server refused the range request; restarting from zero")
            have = 0
            part.unlink(missing_ok=True)
        total = int(r.headers.get("Content-Length", 0)) + have
        with part.open("ab" if have else "wb") as fh:
            done = have
            while chunk := r.read(1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e9:6.1f} / {total / 1e9:.1f} GB", end="", flush=True)
    print()
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="report what is staged and exit")
    args = ap.parse_args()

    bad = False
    for dest, expected in TARGETS:
        status = verify(dest, expected)
        print(f"{dest.name:<16} {status}")
        if status.startswith("ok"):
            continue
        bad = True
        if args.verify:
            continue
        if status.startswith(("CORRUPT", "SHORT")):
            print(f"  refusing to overwrite {dest}; delete it and re-run")
            continue
        print(f"  fetching {BASE_URL}/{dest.name}")
        fetch(f"{BASE_URL}/{dest.name}", dest)
        print(f"{dest.name:<16} {verify(dest, expected)}")

    if args.verify and bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

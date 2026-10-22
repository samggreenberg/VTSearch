#!/usr/bin/env python3
"""Gzip a finished run's cell frames in place (#5092).

``CALIB_CELLS_GZIP`` has been on by default since #5092, but every run before it
that did not opt in wrote its frames as plain text: about 3.3 MB a COCO Better
cell against ~180 KB gzipped.  The State of the App's 2026-10-16 review alone
was 84 GB of them, and the runs that other tools still read had to stay.  This
converts such a run to the gzipped spelling, which ``_cells_paths`` already
reads, so nothing downstream changes.

**One file at a time, and never two spellings left behind.**  ``_cells_paths``
refuses a cell that exists both plain and gzipped (a resume with the knob
flipped would count it twice), so each frame is written to a temporary name,
read back and compared byte for byte with the original, renamed into place,
and only then is the plain file removed.  An interrupted pass leaves at most
one frame in both spellings; the next pass finds it, checks that the two hold
the same bytes, and removes the plain one.  Two spellings that DIFFER are
reported and left alone: that is a real conflict, not an interrupted pass.

What it leaves alone:

* **Empty frames.**  A 0-byte ``task_NNNN.csv`` is how a failed cell shows, and
  the launchers find failures with ``find -size 0``; a gzipped empty file is not
  0 bytes, so converting one would hide the failure.
* **Anything that is not a cell frame**: the ``__pframes`` / ``__testscores``
  npz archives (already compressed), ``prepare_info.json``, crops, analysis
  output.  Only ``task_NNNN.csv`` and ``task_NNNN__<frame>.csv`` are touched.
* **A cells directory written to in the last ``--quiet-minutes``** (30), unless
  ``--force``: a run that is still writing would race the conversion.

**Linked frames are re-pointed, not converted.**  A ``-common`` directory holds
symlinks into another run's cells (2026-10-06-b*-common, 966 a preset).  Reading
through a link and writing a gzipped copy would duplicate the data, and once the
target run was converted its plain file would be gone and the link dangling --
the first pass of #5092 stopped on exactly that.  So real files are converted
first, in every directory given, and only then is each linked
``task_NNNN.csv -> T`` replaced by ``task_NNNN.csv.gz -> T.gz``, once ``T.gz``
exists.  A link whose target is still plain is left for a later pass.  (A
``cells`` directory that is itself a link is not walked into; its target is
converted on its own.)

Dry run unless ``--apply``.  Each argument is a run directory (its ``cells``
directories are found below it, e.g. ``results/cells`` or
``results/<arm>/cells``) or a ``cells`` directory itself::

    python gzip_cells.py /expscratch/$USER/state-of-the-app/2026-10-16-b1
    python gzip_cells.py --apply --jobs 8 /expscratch/$USER/state-of-the-app/2026-10-0[58]-b*
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

#: A cell frame in its plain spelling: the main frame or a ``__`` side frame.
FRAME = re.compile(r"^task_\d{4,}(?:__[A-Za-z0-9]+)?\.csv$")
#: How deep below a run directory a ``cells`` directory may sit.
MAX_DEPTH = 4


@dataclass
class Tally:
    """What one cells directory held, and what a pass did to it."""

    cells_dir: Path
    converted: int = 0
    resumed: int = 0
    empty: int = 0
    before: int = 0
    after: int = 0
    relinked: int = 0
    links_waiting: int = 0
    conflicts: list[str] = field(default_factory=list)
    broken: list[str] = field(default_factory=list)
    skipped: str = ""


def cells_dirs(root: Path) -> list[Path]:
    """*root* if it is a ``cells`` directory, else every ``cells`` directory below it."""
    if root.name == "cells":
        return [root]
    found = []
    base = len(root.parts)
    for here, dirs, _ in os.walk(root):
        depth = len(Path(here).parts) - base
        if Path(here).name == "cells":
            found.append(Path(here))
            dirs[:] = []
        elif depth >= MAX_DEPTH:
            dirs[:] = []
    return sorted(found)


def _write_checked(src: Path, data: bytes) -> int:
    """Write *data* gzipped beside *src*, verified, under its final name; return its size."""
    dst = src.with_name(src.name + ".gz")
    part = src.with_name(f".{src.name}.gz.part")
    with open(part, "wb") as fh:
        fh.write(gzip.compress(data, compresslevel=6))
        fh.flush()
        os.fsync(fh.fileno())
    if gzip.decompress(part.read_bytes()) != data:
        part.unlink()
        raise OSError(f"{dst}: the gzipped copy does not read back as the original")
    os.replace(part, dst)
    return dst.stat().st_size


def convert_one(src: Path, apply: bool) -> tuple[str, int, int]:
    """``(outcome, bytes before, bytes after)`` for one plain frame."""
    size = src.stat().st_size
    if size == 0:
        return "empty", 0, 0
    dst = src.with_name(src.name + ".gz")
    if dst.exists():
        # An interrupted pass, or a real conflict: only identical bytes are the former.
        same = gzip.decompress(dst.read_bytes()) == src.read_bytes()
        if not same:
            return "conflict", size, dst.stat().st_size
        if apply:
            src.unlink()
        return "resumed", size, dst.stat().st_size
    if not apply:
        return "converted", size, 0
    after = _write_checked(src, src.read_bytes())
    src.unlink()
    return "converted", size, after


def convert_dir(cells: Path, *, apply: bool, quiet_minutes: float, force: bool, jobs: int) -> Tally:
    tally = Tally(cells)
    if apply:
        # Only this tool writes a ``.part``; one left over is a pass killed mid-write.
        for stale in cells.glob(".*.gz.part"):
            stale.unlink()
    frames = sorted(p for p in cells.iterdir() if FRAME.match(p.name) and not p.is_symlink())
    if frames and not force:
        newest = max(p.stat().st_mtime for p in frames)
        if time.time() - newest < quiet_minutes * 60:
            tally.skipped = f"a frame was written {(time.time() - newest) / 60:.0f} min ago (still running?)"
            return tally
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for src, (outcome, before, after) in zip(frames, pool.map(lambda p: convert_one(p, apply), frames)):
            if outcome == "empty":
                tally.empty += 1
            elif outcome == "conflict":
                tally.conflicts.append(src.name)
            else:
                setattr(tally, outcome, getattr(tally, outcome) + 1)
                tally.before += before
                tally.after += after
    return tally


def relink_dir(cells: Path, tally: Tally, *, apply: bool) -> None:
    """Re-point each linked plain frame at its target's gzipped spelling, once that exists."""
    for link in sorted(p for p in cells.iterdir() if FRAME.match(p.name) and p.is_symlink()):
        target = Path(os.readlink(link))
        packed = target.with_name(target.name + ".gz")
        if not (link.parent / packed).exists():
            if (link.parent / target).exists():
                tally.links_waiting += 1
            else:
                tally.broken.append(link.name)
            continue
        if apply:
            new = link.with_name(link.name + ".gz")
            if new.is_symlink() or new.exists():
                new.unlink()
            new.symlink_to(packed)
            link.unlink()
        tally.relinked += 1


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="run directories or cells directories")
    ap.add_argument("--apply", action="store_true", help="convert (default: report what would be converted)")
    ap.add_argument("--jobs", type=int, default=4, help="frames compressed at once")
    ap.add_argument("--quiet-minutes", type=float, default=30.0)
    ap.add_argument("--force", action="store_true", help="convert even a directory written to recently")
    args = ap.parse_args(argv)

    rc = 0
    total_before = total_after = 0
    done: list[Tally] = []
    for root in args.roots:
        found = cells_dirs(root)
        if not found:
            print(f"{root}: no cells directory")
            rc = 1
            continue
        for cells in found:
            t = convert_dir(cells, apply=args.apply, quiet_minutes=args.quiet_minutes, force=args.force, jobs=args.jobs)
            if t.skipped:
                print(f"SKIP {cells}: {t.skipped}", flush=True)
                rc = 1
                continue
            verb = "gzipped" if args.apply else "would gzip"
            after = f" -> {_gb(t.after)}" if args.apply else ""
            print(
                f"{cells}: {verb} {t.converted} frame(s), {_gb(t.before)}{after}"
                + (f"; {t.resumed} left from an interrupted pass" if t.resumed else "")
                + (f"; {t.empty} empty frame(s) left plain" if t.empty else ""),
                flush=True,  # a batch log is block-buffered; the first pass logged nothing for 28 min
            )
            for name in t.conflicts:
                print(f"  CONFLICT {name}: plain and gzipped copies differ; both left in place")
                rc = 1
            total_before += t.before
            total_after += t.after
            done.append(t)
    # Links last: a link's target may sit in a directory converted after the link's own.
    for t in done:
        relink_dir(t.cells_dir, t, apply=args.apply)
        if t.relinked or t.links_waiting or t.broken:
            verb = "re-pointed" if args.apply else "would re-point"
            print(
                f"{t.cells_dir}: {verb} {t.relinked} linked frame(s)"
                + (f"; {t.links_waiting} wait for their target's conversion" if t.links_waiting else "")
            )
        for name in t.broken:
            print(f"  BROKEN LINK {name}: neither its target nor the target's .gz exists")
            rc = 1
    if args.apply:
        print(f"total: {_gb(total_before)} -> {_gb(total_after)}")
    else:
        print(f"total: {_gb(total_before)} of plain frames (dry run; --apply converts)")
    return rc


if __name__ == "__main__":
    sys.exit(main())

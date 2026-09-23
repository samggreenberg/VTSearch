#!/usr/bin/env python3
"""Rewrite a pile cell whose whole-image vectors are not unit-norm (#4095, #4099).

The app normalises every vector at ingest; a cell built around that chokepoint
(`coco_val__siglip`, norms 12-19, from `build_coco_pickle.py` before its fix)
makes every harness run on it measure a detector nobody ships. The harness now
repairs such vectors on load (`_cells_io.repair_norms`), but the pile should not
depend on that: `build_pile.py --verify` fails a non-unit cell.

The original is copied aside first (the pile is shared; a later reader of an old
report may want the exact bytes it ran on), the new cell is written aside and
swapped in, and the sidecar gains a ``repairs`` entry plus the new fingerprint.

    python normalise_cell.py coco_val siglip --archive /expscratch/$USER/keep/<dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402

from pilebuild.env import cells_io, log  # noqa: E402
from pilebuild.provenance import cell_fingerprint, code_record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("embedder")
    ap.add_argument("--archive", type=Path, required=True, help="directory to copy the original cell and sidecar into")
    args = ap.parse_args()

    out = pc.cell_path(args.dataset, args.embedder)
    side = pc.provenance_path(args.dataset, args.embedder)
    io = cells_io()
    medias = io.load_medias(out, repair=False)
    before = cell_fingerprint(args.dataset, args.embedder, medias)
    fixed = io.repair_norms(medias, where=out.name)
    if not fixed:
        log(f"{out.name}: every vector already unit-norm; nothing to do")
        return 0

    args.archive.mkdir(parents=True, exist_ok=True)
    for f in (out, side):
        if f.exists():
            shutil.copy2(f, args.archive / f.name)
    tmp = out.with_name(out.name + ".normalise-tmp")
    io.dump_medias(medias, tmp)
    os.replace(tmp, out)

    record = json.loads(side.read_text()) if side.exists() else {}
    after = cell_fingerprint(args.dataset, args.embedder, medias)
    record.setdefault("repairs", []).append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "what": "l2-normalised whole-image vectors (#4095, #4099)",
            "vectors_normalised": fixed,
            "fingerprint_before": before,
            "archived_to": str(args.archive),
            "code": code_record(),
        }
    )
    record["fingerprint"] = after
    side.write_text(json.dumps(record, indent=2) + "\n")
    log(f"{out.name}: normalised {fixed} of {len(medias)} vectors; original in {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

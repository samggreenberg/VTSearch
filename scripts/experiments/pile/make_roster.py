"""Freeze the membership of the cells as they stand, so a review keeps covering them.

Run this against a built cell *before* changing how membership is selected. The
builder's selection is now hash-stable, but stability only helps from the moment
it is adopted: the hash draw and the earlier random draw agree on a couple of
hundred of 3,900 negatives, so adopting it without a roster would retire almost
every image a human had already judged.

Idempotent and read-only with respect to the pile: it reads one cell and writes
the roster the builder will honour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pile_config as pc

pc.setup_env()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--out", default=str(pc.ROSTER))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    medias = load_medias(Path(args.cell))
    cells: dict[str, list[int]] = {}
    for iid, m in medias.items():
        for cell in m.get("categories") or []:
            cells.setdefault(cell, []).append(int(iid))
    for v in cells.values():
        v.sort()

    # A negative is designated when it is evaluable in every cell; the rest of
    # the category-less medias are the spares held back for backfill.
    all_cells = {pc.scale_cell(c, b) for c in pc.SCALE_CLASSES for b in pc.BOX_BANDS}
    negatives, spares = [], []
    for iid, m in medias.items():
        if m.get("categories"):
            continue
        (negatives if set(m.get("evaluable_categories") or []) >= all_cells else spares).append(int(iid))
    negatives.sort()
    spares.sort()

    Path(args.out).write_text(json.dumps({"cells": cells, "negatives": negatives, "spares": spares}, indent=1) + "\n")
    print(f"roster written to {args.out}")
    print(f"  {len(cells)} cells, {sum(len(v) for v in cells.values())} positives")
    print(f"  {len(negatives)} designated negatives + {len(spares)} spares")
    short = [c for c, v in cells.items() if len(v) != pc.SCALE_N_POS]
    print(f"  cells not at n_pos={pc.SCALE_N_POS}: {short or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

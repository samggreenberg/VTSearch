"""What did a rebuild actually change, per category?

A pile rebuild is not a no-op on the studies that already ran against it. #3281
rebuilt `vg_scale` to repair 130 boxes and moved the positive set of **22 of 36
categories** -- because each cell takes exactly 100 positives from a ranked
candidate pool, so repairing one band re-selects the others in the same class.
Six categories changed that held no repaired box at all.

That distinction decides whether an existing run is still usable. A category
whose positive set, evaluable set and boxes are all identical before and after
is one whose old cells still describe the live dataset, and its runs can be
compared across the rebuild. One that moved cannot: pooling it would compare two
different sets of images and attribute the difference to the fix.

Answering it needs the OLD pickle, so copy the cell aside before rebuilding::

    cp $VTS_PILE/datadir/embeddings/vg_scale__siglip.pkl  /somewhere/prefix/
    # ... rebuild ...
    python diff_labels.py --old /somewhere/prefix/vg_scale__siglip.pkl \\
                          --new $VTS_PILE/datadir/embeddings/vg_scale__siglip.pkl

The whole-image cell is enough: boxes and labels come from one build path per
dataset, and it is 26 MB against the patch cell's 2.4 GB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    return load_medias(path)


def _sets(medias: dict, category: str) -> tuple[set[int], set[int]]:
    """``(positives, evaluable)`` for one category.

    Evaluable matters as much as positive: a banded dataset excludes an image
    that holds the object at the wrong size, so the negative pool is
    ``evaluable - positive`` and a change in either moves the cell.
    """
    from vtscore.eval.labels import media_is_evaluable, media_is_positive  # noqa: PLC0415

    pos, ev = set(), set()
    for mid, m in medias.items():
        try:
            if media_is_evaluable(m, category):
                ev.add(mid)
        except TypeError:  # single-label datasets have no evaluability notion
            ev.add(mid)
        if media_is_positive(m, category):
            pos.add(mid)
    return pos, ev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True, help="pickle saved BEFORE the rebuild")
    ap.add_argument("--new", required=True, help="the live pickle")
    ap.add_argument("--quiet", action="store_true", help="print only the verdict lists")
    args = ap.parse_args(argv)

    old, new = _load(Path(args.old)), _load(Path(args.new))
    from vtscore.eval.labels import region_box_for_category  # noqa: PLC0415

    print(
        f"medias: old {len(old)}  new {len(new)}  dropped {len(set(old) - set(new))}  added {len(set(new) - set(old))}"
    )
    cats = sorted(
        {c for m in old.values() for c in (m.get("categories") or [])}
        | {c for m in new.values() for c in (m.get("categories") or [])}
    )

    hdr = f"{'category':<20}{'pos o/n':>10}{'pos +/-':>10}{'eval +/-':>11}{'box moved':>11}{'verdict':>12}"
    if not args.quiet:
        print(hdr)
        print("-" * len(hdr))
    unchanged, changed = [], []
    for c in cats:
        po, eo = _sets(old, c)
        pn, en = _sets(new, c)
        shared = po & pn
        moved = sum(1 for mid in shared if region_box_for_category(old[mid], c) != region_box_for_category(new[mid], c))
        same = po == pn and eo == en and moved == 0
        (unchanged if same else changed).append(c)
        if not args.quiet:
            print(
                f"{c:<20}{f'{len(po)}/{len(pn)}':>10}{f'+{len(pn - po)}/-{len(po - pn)}':>10}"
                f"{f'+{len(en - eo)}/-{len(eo - en)}':>11}{moved:>11}"
                f"{'unchanged' if same else 'CHANGED':>12}"
            )

    print(f"\nUNCHANGED ({len(unchanged)}/{len(cats)}) - old runs still describe the live dataset:")
    print("  " + (", ".join(unchanged) or "(none)"))
    print(f"\nCHANGED ({len(changed)}/{len(cats)}) - old runs describe a dataset that no longer exists:")
    print("  " + (", ".join(changed) or "(none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

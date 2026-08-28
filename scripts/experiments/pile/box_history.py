"""Was this finding manufactured by the rebuild? One media's box, before and after.

`diff_labels.py` answers the question at the level of a category: which cells a
rebuild moved, and so which of them can still be quoted. This answers it for a
single image, which is what you need when a *report* names an exemplar and you
have to decide whether the row it supports survives.

It was written for #3284. #3156's overview named six worst-case exemplars, and
the only way to settle each one was to look the media up on both sides of the
#3281 rebuild rather than reason from the defect's footprint::

    python box_history.py --old /somewhere/prefix/vg_scale__siglip.pkl \\
                          --new $VTS_PILE/datadir/embeddings/vg_scale__siglip.pkl \\
                          --media 2381555 2322075 1222

Two of the six turned out to hold a **crushed** box (`--census` counts them
pile-wide), and both had therefore been filed in the wrong size band -- which is
how a box defect reaches a whole-image arm that never reads a box. Three others
were byte-identical across the rebuild, so the findings resting on them stand.
One, `bird@small` 1222, had a clean box in a cell that was 42% corrupt: the run
is unquotable because of its *pool*, not its seed. That distinction is the whole
point of looking per-media, and no per-category summary shows it.

A box is called CRUSHED on the #3281 signature: sub-pixel area (< 1e-5 of the
frame) with every corner inside the top-left 1%. A real object can be tiny
anywhere; one whose coordinates are all crushed toward the origin has been
scaled, not observed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CRUSHED_AREA = 1e-5
CRUSHED_CORNER = 0.01


def _load(path: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    return load_medias(path)


def _area(box) -> float:
    x0, y0, x1, y1 = box[:4]
    return abs((x1 - x0) * (y1 - y0))


def _is_crushed(box) -> bool:
    return _area(box) < CRUSHED_AREA and max(box[:4]) < CRUSHED_CORNER


def _fmt(box) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in box[:4]) + "]"


def _boxes(media: dict) -> list[tuple[str, list]]:
    """``(category, box)`` for every category this media is a positive of."""
    from vtscore.eval.labels import region_box_for_category  # noqa: PLC0415

    out = []
    for cat in media.get("categories") or []:
        box = region_box_for_category(media, cat)
        if box is not None:
            out.append((cat, box))
    return out


def _census(medias: dict) -> tuple[int, set]:
    """``(box count, medias holding a crushed box)``."""
    n, bad = 0, set()
    for mid, m in medias.items():
        for _cat, box in _boxes(m):
            n += 1
            if _is_crushed(box):
                bad.add(mid)
    return n, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True, help="pickle saved BEFORE the rebuild")
    ap.add_argument("--new", required=True, help="the live pickle")
    ap.add_argument("--media", nargs="*", type=int, default=[], help="media ids to trace")
    ap.add_argument("--census", action="store_true", help="also count crushed boxes pile-wide")
    args = ap.parse_args(argv)

    old, new = _load(Path(args.old)), _load(Path(args.new))
    print(f"medias: old {len(old)}  new {len(new)}")

    old_bad: set = set()
    new_bad: set = set()
    if args.census or args.media:
        n_old, old_bad = _census(old)
        n_new, new_bad = _census(new)
        print(f"boxes:  old {n_old} ({len(old_bad)} crushed)  new {n_new} ({len(new_bad)} crushed)")

    for mid in args.media:
        print(f"\n--- {mid} ---")
        for tag, medias, bad in (("OLD", old, old_bad), ("NEW", new, new_bad)):
            m = medias.get(mid)
            if m is None:
                print(f"  {tag}: not in this pile")
                continue
            print(f"  {tag}: {'CRUSHED' if mid in bad else 'clean'}")
            for cat, box in _boxes(m):
                mark = "  <- CRUSHED" if _is_crushed(box) else ""
                print(f"        {cat:<20} {_fmt(box)}  area {100 * _area(box):.4f}%{mark}")
        o, n = old.get(mid), new.get(mid)
        if o is not None and n is not None:
            same = _boxes(o) == _boxes(n)
            print(
                f"  verdict: {'unmoved -- findings on this media survive the rebuild' if same else 'MOVED -- re-check every finding that names it'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

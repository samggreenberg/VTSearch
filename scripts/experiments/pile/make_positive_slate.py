"""Re-issue the positive stratum with its box drawn, so small objects are answerable.

A bare thumbnail cannot settle "is there a backpack in this picture?" when the
backpack is a sub-patch object. Measured on the first three reviewed classes,
the reviewer rejected **43%** of small-band positives against 20% medium and
10% large — a clean monotonic function of how many pixels the object occupies,
which is a property of the *review protocol*, not of the labels. Taking those
rejections at face value would have deleted nearly half the small band, the very
band the study exists to measure.

So for positives — and only for positives — the image is re-issued with the
ground-truth box drawn on it, plus a magnified inset of the box's contents in
the corner. The question stops being "find the backpack" and becomes "is the
thing in this box a backpack, and is the box around the right thing?", which is
answerable at any scale.

Drawing the box on a *negative* would be meaningless (there is nothing to draw)
and would bias the answer, so the ranked and random strata keep bare images.

Usage::

    python make_positive_slate.py --out /expscratch/$USER/vgscale-3156/slates_pos
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pile_config as pc

pc.setup_env()

#: How big the inset may get, as a fraction of the image's shorter side.
INSET_FRAC = 0.42
#: The inset magnifies at least this much, so a sub-patch box is actually visible.
MIN_ZOOM = 3.0


def log(msg: str) -> None:
    print(f"[posslate] {msg}", flush=True)


def draw_with_inset(src: Path, box: tuple[float, float, float, float], dest: Path) -> tuple[int, int]:
    """Write *src* with *box* outlined and a magnified inset of its contents."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    with Image.open(src) as im:
        im = im.convert("RGB")
        W, H = im.size
        # VG boxes are not guaranteed to lie inside their image -- some run past
        # the edge, and a few are inverted -- so clamp before any arithmetic
        # that assumes a well-formed rectangle.
        x0, x1 = sorted((box[0] * W, box[2] * W))
        y0, y1 = sorted((box[1] * H, box[3] * H))
        x0, x1 = max(0.0, min(x0, W - 1.0)), max(1.0, min(x1, float(W)))
        y0, y1 = max(0.0, min(y0, H - 1.0)), max(1.0, min(y1, float(H)))
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)

        # The crop is padded around the box so the object keeps its context --
        # a box cropped exactly to its edges is often unrecognisable.
        pad = max(bw, bh) * 0.6
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
        cx1, cy1 = max(cx1, cx0 + 2), max(cy1, cy0 + 2)
        crop = im.crop((cx0, cy0, min(cx1, W), min(cy1, H)))

        target = int(min(W, H) * INSET_FRAC)
        zoom = max(MIN_ZOOM, target / max(crop.width, crop.height))
        iw, ih = int(crop.width * zoom), int(crop.height * zoom)
        iw, ih = min(iw, target), min(ih, target)
        crop = crop.resize((max(1, iw), max(1, ih)), Image.LANCZOS)

        out = im.copy()
        d = ImageDraw.Draw(out)
        lw = max(2, int(min(W, H) * 0.006))
        d.rectangle([x0, y0, x1, y1], outline=(255, 32, 32), width=lw)

        # Inset in whichever bottom corner is furthest from the box, so the
        # magnifier never covers the thing it is magnifying.
        ix = 0 if (x0 + x1) / 2 > W / 2 else W - crop.width
        iy = H - crop.height
        out.paste(crop, (ix, iy))
        d.rectangle([ix, iy, ix + crop.width - 1, iy + crop.height - 1], outline=(255, 32, 32), width=lw)
        out.save(dest, quality=92)
        return out.size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=str(pc.EMBEDDINGS / "vg_scale__siglip.pkl"))
    ap.add_argument("--slates", default=str(pc.PILE.parent / "vgscale-3156" / "slates"))
    ap.add_argument("--out", default=str(pc.PILE.parent / "vgscale-3156" / "slates_pos"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))
    from _cells_io import load_medias  # noqa: PLC0415

    from build_pile import _vg_image_paths  # noqa: PLC0415

    medias = load_medias(Path(args.cell))
    paths = _vg_image_paths()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    index = []
    for man in sorted(Path(args.slates).glob("*/manifest.csv")):
        rows = [r for r in csv.DictReader(man.open()) if r["stratum"] == "positive"]
        if not rows:
            continue
        cls = rows[0]["class"]
        cdir = out_root / man.parent.name
        cdir.mkdir(parents=True, exist_ok=True)
        written = []
        for r in rows:
            iid = int(r["image_id"])
            cell = r["cell"]
            media = medias.get(iid)
            src = paths.get(iid)
            if media is None or src is None:
                continue
            boxes = [x["box"] for x in (media.get("regions") or []) if x.get("label") == cell]
            if not boxes:
                continue
            box = (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            draw_with_inset(src, box, cdir / f"{iid}.jpg")
            written.append(
                {
                    "image_id": iid,
                    "class": cls,
                    # A distinct stratum so a boxed verdict supersedes the bare
                    # one the reviewer cast on the same pair.
                    "stratum": "positive_boxed",
                    "cell": cell,
                    "text_score": r["text_score"],
                    "reference": "present",
                    "exhaustive": r["exhaustive"],
                    "n_boxes": r["n_boxes"],
                    "detector": f"{cls} positives",
                }
            )
        with (cdir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(written[0]))
            w.writeheader()
            w.writerows(written)
        index.append({"class": cls, "dir": str(cdir), "n": len(written), "detector": f"{cls} positives"})
        log(f"  {cls:<12} {len(written):3d} boxed positives -> {cdir}")

    (out_root / "slates.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"\n{sum(e['n'] for e in index)} boxed images across {len(index)} classes under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

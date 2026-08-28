"""Look at what a region-voting Good vote would actually drag.

A box is the one field in a pile cell that nothing downstream reads back in a
form a person can check. Labels get adjudicated on contact sheets; geometry gets
a number. #3281 lived through three studies because a corrupted box lands its
image in `@small` and `@small` is exactly where a sub-pixel area belongs -- the
cell name and its boxes stayed consistent with each other, and no aggregate
could see it. The first thing that did see it was a picture.

So: one tile per positive, the full frame with the stored box marked, and the
box's own pixels enlarged beside it. A box that describes its object looks
obviously right; one that does not looks obviously wrong, at a glance, which is
the property no summary statistic has.

    python box_sheets.py --dataset vg_scale --category bird@small --out sheet.jpg
    python box_sheets.py --dataset vg_scale --category bird@small --order area --n 12

``--order area`` puts the smallest boxes first, which is where geometry bugs
surface: a scaling error shrinks a box, it does not grow one.

Pixels come from the dataset's original source, not from the cell pickle (which
drops ``media_bytes``): extracted directories for Visual Genome, and
``val2017.zip`` read member-wise for COCO. If none of them resolve, this exits
non-zero instead of writing a sheet -- an empty sheet is the one output that
looks like an answer and contains none (#3305).
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import common  # noqa: E402

common.setup_env()

import pile_config as pc  # noqa: E402

#: Extracted image directories, per dataset kind. Checked first: a loose JPEG is
#: cheaper to read than a zip member, and VG is only ever staged this way.
IMAGE_DIRS = {
    "vg": [pc.DEMO_CACHE / "visual_genome" / "VG_100K", pc.DEMO_CACHE / "visual_genome" / "VG_100K_2"],
    "coco": [pc.COCO_IMAGES],
}

#: Zipped image archives, per dataset kind, read member-wise when no extracted
#: directory holds the file. COCO needs this: the staging area holds
#: `val2017.zip` and has never held `val2017/`, so a directory-only resolver
#: found nothing for every COCO media and drew an empty sheet (#3305).
#: `build_pile._load_coco` has always read pixels out of the same zip.
IMAGE_ZIPS = {
    "vg": [],
    "coco": [pc.COCO_VAL_ZIP],
}


def _kind_for(dataset: str) -> str:
    return "coco" if pc.DATASETS.get(dataset, {}).get("kind", "") == "coco" else "vg"


class ImageSource:
    """Where a sheet's pixels come from -- and what to say when there are none.

    Resolution is by basename against extracted directories first, then against
    the archives, mirroring the member lookup in ``build_pile._load_coco``. The
    cell pickles drop ``media_bytes`` (`_cells_io._DROP_FIELDS`), so the
    original source really is the only place the pixels are.
    """

    def __init__(self, dirs: list[Path], zips: list[Path]) -> None:
        self.dirs = [Path(d) for d in dirs]
        self.zips = [Path(z) for z in zips]
        self._open: dict[Path, zipfile.ZipFile] = {}
        self._members: dict[Path, dict[str, str]] = {}

    def _members_of(self, z: Path) -> dict[str, str]:
        if z not in self._members:
            zf = zipfile.ZipFile(z)
            self._open[z] = zf
            self._members[z] = {Path(n).name: n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
        return self._members[z]

    def exists(self) -> bool:
        """True if any configured source is actually present on disk."""
        return any(d.is_dir() for d in self.dirs) or any(z.is_file() for z in self.zips)

    def describe(self) -> str:
        parts = [f"{d} ({'ok' if d.is_dir() else 'MISSING'})" for d in self.dirs]
        parts += [f"{z} ({'ok' if z.is_file() else 'MISSING'})" for z in self.zips]
        return "; ".join(parts) or "(no source configured)"

    def locate(self, mid, m) -> Path | tuple[Path, str] | None:
        """A handle :meth:`stream` can open, or None if the pixels are unreachable."""
        for cand in (m.get("filename"), m.get("media_string"), f"{mid}.jpg"):
            if not cand:
                continue
            name = Path(str(cand)).name
            for d in self.dirs:
                p = d / name
                if p.exists():
                    return p
            for z in self.zips:
                if not z.is_file():
                    continue
                member = self._members_of(z).get(name)
                if member is not None:
                    return (z, member)
        return None

    def stream(self, handle: Path | tuple[Path, str]):
        """Something ``PIL.Image.open`` accepts: a path, or the member's bytes."""
        if isinstance(handle, tuple):
            z, member = handle
            return io.BytesIO(self._open[z].read(member))
        return handle


def source_for(dataset: str) -> ImageSource:
    kind = _kind_for(dataset)
    return ImageSource(IMAGE_DIRS[kind], IMAGE_ZIPS[kind])


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="vg_scale")
    ap.add_argument("--embedder", default="siglip", help="which cell to read labels from (the small one)")
    ap.add_argument("--category", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--order", default="area", choices=["area", "id"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from PIL import Image, ImageDraw, ImageFont

    from vtscore.datasets import loader as _loader
    from vtscore.eval.labels import media_is_positive, region_box_for_category

    from _cells_io import load_medias  # noqa: PLC0415

    medias = load_medias(_loader.EMBEDDINGS_DIR / f"{args.dataset}__{args.embedder}.pkl")
    images = source_for(args.dataset)
    # Checked before the scan so the message names the *configuration* problem
    # rather than reporting it as an absence of positives.
    if not images.exists():
        raise SystemExit(f"no image source exists for {args.dataset}: {images.describe()}")

    items = []
    positives = boxless = unreachable = 0
    for mid, m in medias.items():
        if not media_is_positive(m, args.category):
            continue
        positives += 1
        box = region_box_for_category(m, args.category)
        if box is None:
            boxless += 1
            continue
        handle = images.locate(mid, m)
        if handle is None:
            unreachable += 1
            continue
        items.append((_area(box), mid, box, handle))
    print(
        f"{args.dataset} {args.category}: {len(items)} of {positives} positives drawable "
        f"({boxless} with no box for this category, {unreachable} with unreachable pixels)"
    )
    # A sheet exists to *show* the reader the images; one drawn with none of
    # them answers the question with nothing while looking like an answer, so
    # unreachable pixels are an error rather than a thinner sheet (#3305).
    if unreachable and not items:
        raise SystemExit(
            f"no image resolved for any of {positives} positives -- refusing to write an empty sheet.\n"
            f"  looked in: {images.describe()}"
        )
    if unreachable:
        print(f"WARNING: {unreachable} positives dropped, pixels not found in {images.describe()}")
    if not items:
        raise SystemExit(f"nothing to draw: {args.dataset} has no {args.category} positive with a box")
    items.sort(key=(lambda it: it[0]) if args.order == "area" else (lambda it: it[1]))
    items = items[: args.n]

    TILE, IMG_H, CROP, PAD, HDR = 320, 230, 230, 12, 24
    cols = 4
    rows = (len(items) + cols - 1) // cols
    tile_h = HDR + IMG_H + PAD + CROP + PAD
    sheet = Image.new("RGB", (cols * (TILE + PAD) + PAD, rows * tile_h + PAD), "#fcfcfb")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for k, (a, mid, box, handle) in enumerate(items):
        r, c = divmod(k, cols)
        x0, y0 = PAD + c * (TILE + PAD), PAD + r * tile_h
        img = Image.open(images.stream(handle)).convert("RGB")
        W, H = img.size
        px = [box[0] * W, box[1] * H, box[2] * W, box[3] * H]
        full = img.copy()
        d = ImageDraw.Draw(full)
        # A sub-pixel box is invisible at thumbnail scale, which is the case
        # this sheet exists for -- so it gets a crosshair as well as an outline.
        d.rectangle([px[0] - 1, px[1] - 1, px[2] + 1, px[3] + 1], outline="#e11d48", width=max(2, W // 200))
        cx, cy = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
        d.line([(cx, 0), (cx, H)], fill="#e11d48", width=max(1, W // 600))
        d.line([(0, cy), (W, cy)], fill="#e11d48", width=max(1, W // 600))
        full.thumbnail((TILE, IMG_H))
        sheet.paste(full, (x0 + (TILE - full.width) // 2, y0 + HDR))
        m = max(px[2] - px[0], px[3] - px[1]) * 1.6 + 10
        cb = (max(0, cx - m), max(0, cy - m), min(W, cx + m), min(H, cy + m))
        crop = img.crop([int(v) for v in cb]).resize((CROP, CROP), Image.LANCZOS)
        dc = ImageDraw.Draw(crop)
        sx, sy = CROP / max(1e-6, cb[2] - cb[0]), CROP / max(1e-6, cb[3] - cb[1])
        dc.rectangle(
            [(px[0] - cb[0]) * sx, (px[1] - cb[1]) * sy, (px[2] - cb[0]) * sx, (px[3] - cb[1]) * sy],
            outline="#e11d48",
            width=2,
        )
        sheet.paste(crop, (x0 + (TILE - CROP) // 2, y0 + HDR + IMG_H + PAD))
        # Both the fraction and the pixels: a percentage alone cannot tell you
        # the box is smaller than one pixel, and that is the failure mode.
        draw.text(
            (x0, y0 + 5),
            f"{mid}   {a * 100:.4f}% of frame   {px[2] - px[0]:.1f}x{px[3] - px[1]:.1f}px",
            fill="#292524",
            font=font,
        )

    sheet.save(args.out, quality=92)
    print(f"wrote {args.out} {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

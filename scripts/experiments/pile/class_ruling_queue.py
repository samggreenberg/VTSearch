#!/usr/bin/env python3
"""Put a definitional ruling to the owner as a VTSearch dataset of single crops.

Two classes in *C* are named for something they mostly are not, measured by
``coco_class_purity.py`` against LVIS's defined synsets:

* ``dining table`` -- only **10.4%** of its boxes are `dining_table`; 34.2% are
  `tablecloth` and 30.5% generic `table`. What COCO boxed is the covered surface.
* ``tv`` -- **50.4%** `television_set` against **46.9%** computer monitor. It is
  a screen class, not a television class.

#4056 shipped briefs resolving both, written from the measurement. They are the
measurer's calls, not the owner's, and the plan marks a boundary contest of this
shape (`truck`/`car`) as **human**. So this asks.

**One image per question, one dataset per class, no contact sheets.** The owner
answers Good/Bad per crop in VTSearch -- "would you call this a `<class>`?" --
and the dataset name carries the question, because the dashboard is the worklist.
Nothing in a filename says which stratum a crop came from, so the answer cannot
be anchored by the label; ``manifest.json`` holds the mapping and the vote is
read back against it.

**The strata are the attention control.** Each queue mixes the canonical
sub-population (an unmistakable television; a table COCO's annotators and LVIS
both call `dining_table`) with the contested one. A reviewer who is reading gives
different answers to the two arms; a tired sweep of one button gives the same,
and the canonical arm is what distinguishes them.

Usage::

    python class_ruling_queue.py --class tv --n 40 --out /expscratch/.../queues
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io as _io
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402

IOU_MIN = 0.5
#: Context around the box, as a fraction of its longer side. A crop with no
#: surroundings cannot answer "is this a dining table" -- the setting IS the
#: evidence -- and a crop of the whole image cannot be read at a glance.
CONTEXT = 0.45
MAX_SIDE = 900
#: A crop is only useful if the reviewer can SEE the thing. Cropping to the box
#: is already an auto-zoom, but the first build of these queues only ever scaled
#: DOWN, so a small box stayed a small picture: 88% of the `tv` crops came out
#: under 400px and the smallest was 20px, which is not a question anyone can
#: answer. Two fixes, in this order, because they fail differently:
#:
#: * widen the window until it is at least ``MIN_WINDOW`` of REAL pixels, so a
#:   small object is judged from its surroundings rather than from mush;
#: * only then upscale, to ``MIN_SIDE``.
#:
#: Widening makes "which object is the question?" ambiguous, so the target is
#: outlined -- the same convention the FullMarks queues use.
MIN_WINDOW = 480
MIN_SIDE = 640
QUALITY = 85


def _boxes(path: Path) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    out: dict[int, list] = collections.defaultdict(list)
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        out[int(a["image_id"])].append((a["bbox"], cats[a["category_id"]], int(a["id"])))
    # LVIS v1 has no `file_name`; it carries `coco_url`. Only the COCO side is
    # ever asked for a filename, so this stays tolerant rather than branching.
    sizes = {
        int(im["id"]): (
            im["width"],
            im["height"],
            im.get("file_name") or im.get("coco_url", "").rsplit("/", 1)[-1],
        )
        for im in d["images"]
    }
    return out, sizes


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / (aw * ah + bw * bh - inter)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--klass", "--class", dest="klass", required=True)
    ap.add_argument("--annotations", type=Path, default=Path("/expscratch/sgreenberg/vts-cache/coco_anchor"))
    ap.add_argument(
        "--lvis", type=Path, default=Path("/exp/scale26/datasets/external/LVIS/annotations/lvis_v1_val.json")
    )
    ap.add_argument("--n", type=int, default=40, help="questions per queue")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    from PIL import Image, ImageDraw  # noqa: PLC0415

    coco: dict[int, list] = collections.defaultdict(list)
    sizes: dict[int, tuple] = {}
    for split in ("val2017", "train2017"):
        b, s = _boxes(args.annotations / f"instances_{split}.json")
        for k, v in b.items():
            coco[k] += v
        sizes.update(s)
    lvis, lsizes = _boxes(args.lvis)
    print(f"COCO images {len(coco):,}  LVIS images {len(lvis):,}")

    # Mutual best match, exactly as coco_class_purity does, so the strata here
    # are the same populations the purity note quotes.
    rows = []
    for iid, cbs in coco.items():
        lbs = lvis.get(iid)
        if not lbs:
            continue
        mine = [i for i, (_, cls, _) in enumerate(cbs) if cls == args.klass]
        if not mine:
            continue
        grid = [[_iou(cb, lb) for lb, _, _ in lbs] for cb, _, _ in cbs]
        for i in mine:
            if not grid[i]:
                continue
            j = max(range(len(lbs)), key=lambda k: grid[i][k])
            if grid[i][j] < IOU_MIN or max(range(len(cbs)), key=lambda k: grid[k][j]) != i:
                continue
            rows.append({"image_id": iid, "bbox": cbs[i][0], "ann_id": cbs[i][2], "lvis": lbs[j][1]})

    by_name = collections.Counter(r["lvis"] for r in rows)
    print(f"{len(rows)} matched `{args.klass}` boxes over {len(by_name)} LVIS names")
    for n, c in by_name.most_common(6):
        print(f"   {n:<42} {c:>5}  {100 * c / len(rows):5.1f}%")

    # Stratified sample: every name with >=3% of the class, proportional, so the
    # contested sub-population is present by construction rather than by luck.
    rng = random.Random(args.seed)
    keep = [n for n, c in by_name.items() if c / len(rows) >= 0.03]
    per = {n: max(4, round(args.n * by_name[n] / sum(by_name[k] for k in keep))) for n in keep}
    picked = []
    for n in keep:
        pool = [r for r in rows if r["lvis"] == n]
        rng.shuffle(pool)
        for r in pool[: per[n]]:
            r["stratum"] = n
            picked.append(r)
    rng.shuffle(picked)
    print(f"\nsampled {len(picked)} across {len(keep)} strata: " + ", ".join(f"{n}={per[n]}" for n in keep))

    slug = args.klass.replace(" ", "_")
    outdir = args.out / slug / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    members = {}
    for zp in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.endswith(".jpg"):
                    members[Path(name).name] = (zp, name)

    manifest, kb = [], []
    zcache: dict = {}
    for idx, r in enumerate(picked):
        w, h, fname = sizes[r["image_id"]]
        zp, member = members[fname]
        if zp not in zcache:
            zcache[zp] = zipfile.ZipFile(zp)
        with zcache[zp].open(member) as fh:
            im = Image.open(_io.BytesIO(fh.read())).convert("RGB")
        x, y, bw, bh = r["bbox"]
        pad = CONTEXT * max(bw, bh)
        # Widen symmetrically until the window has MIN_WINDOW real pixels on its
        # longer side, or we run out of image.
        while max(bw + 2 * pad, bh + 2 * pad) < MIN_WINDOW and (pad < w or pad < h):
            pad *= 1.5
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
        # Outline the target, since a widened window no longer says which object
        # is being asked about. Two strokes so it reads on light and dark alike.
        draw = ImageDraw.Draw(crop)
        bx0, by0, bx1, by1 = x - x0, y - y0, x + bw - x0, y + bh - y0
        lw = max(2, int(0.004 * max(crop.size)))
        draw.rectangle([bx0 - lw, by0 - lw, bx1 + lw, by1 + lw], outline=(255, 255, 255), width=lw)
        draw.rectangle([bx0, by0, bx1, by1], outline=(255, 64, 0), width=lw)
        scale = None
        if max(crop.size) > MAX_SIDE:
            scale = MAX_SIDE / max(crop.size)
        elif max(crop.size) < MIN_SIDE:
            scale = MIN_SIDE / max(crop.size)
        if scale:
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
        # The filename must not leak the stratum: a reviewer who can see which
        # arm a crop is from is answering a different question.
        stem = hashlib.sha1(f"{args.seed}:{r['ann_id']}".encode()).hexdigest()[:12]  # noqa: S324
        path = outdir / f"{slug}_{stem}.jpg"
        crop.save(path, quality=QUALITY, optimize=True)
        kb.append(path.stat().st_size / 1024)
        manifest.append({"file": path.name, "order": idx, **r})

    kb.sort()
    (args.out / slug / "manifest.json").write_text(
        json.dumps({"class": args.klass, "seed": args.seed, "strata": per, "items": manifest}, indent=1)
    )
    print(f"\nwrote {len(manifest)} crops to {outdir}")
    print(f"size: median {kb[len(kb) // 2]:.0f} KB, p90 {kb[int(0.9 * len(kb))]:.0f} KB, max {kb[-1]:.0f} KB")
    print("(budget from the FullMarks passes: ~124 KB median / ~169 KB p90)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Is COCO's box around ONE object, or around a group? Ask the owner.

`coco_box_granularity.py` measures that COCO's boxes are larger than LVIS's for
the same pixels, and splits the disagreement into three faults. What it cannot
do is say which of them is an ERROR:

* **lumping** -- `banana` 7.58 count / 8.15 area. One box round a bunch. Almost
  certainly wrong for a benchmark whose bands are the size of ONE object.
* **wider extent** -- `potted plant` 1.40 / 5.92. The same objects, drawn around
  more. Plausibly pot-plus-plant against LVIS's foliage, in which case COCO is
  right and LVIS is the odd one out.
* **structural pairing** -- `skis` 2.09. A pair of skis is arguably ONE object
  in use, so this may be no error at all.

So this puts the box in front of a human: **Good if the red box is around one
object.** One image per question, one dataset per class, no contact sheets.

**Stratified on the per-image ratio, which is also the attention control.** Each
queue mixes images the measurement calls lumped (ratio >= 1.6) with images it
calls clean (< 1.2). A reviewer who is reading answers those two differently; a
sweep of one button does not. Nothing in a filename says which arm a crop came
from -- the stem is a hash of the annotation id.
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
from pilebuild.loaders.coco_better import _share_inside  # noqa: E402
from pilebuild.scale_core import band_for, largest_box  # noqa: E402

CONTEXT = 0.45
MIN_WINDOW = 480
MIN_SIDE = 640
MAX_SIDE = 900
QUALITY = 85
LUMPED, CLEAN = 1.6, 1.2


def _read(path: Path):
    d = json.loads(path.read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    per: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in d["annotations"]:
        if not a.get("iscrowd"):
            per[int(a["image_id"])][cats[a["category_id"]]].append((a["bbox"], int(a["id"])))
    # LVIS v1 has no `file_name`; it carries `coco_url`. Only the COCO side is
    # ever asked for a filename, so this stays tolerant rather than branching.
    dims = {
        int(i["id"]): (
            i["width"],
            i["height"],
            i.get("file_name") or i.get("coco_url", "").rsplit("/", 1)[-1],
        )
        for i in d["images"]
    }
    return per, dims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--klass", "--class", dest="klass", required=True)
    ap.add_argument("--lvis-names", required=True, help="comma-separated, from coco_box_granularity.SAME")
    ap.add_argument("--annotations", type=Path, default=Path("/expscratch/sgreenberg/vts-cache/coco_anchor"))
    ap.add_argument(
        "--lvis",
        type=Path,
        nargs="+",
        default=[pc.LVIS_DIR / f"lvis_v1_{s}.json" for s in pc.LVIS_SPLITS],
        help="LVIS annotation files; default both splits, since val alone is 16%% of COCO",
    )
    ap.add_argument(
        "--banded-only",
        action="store_true",
        help="sample only images `band_for` puts in a band, i.e. ones a cell could take. Without it a "
        "class the scatter guard mostly rejects (`book`: 3,223 of 5,562) spends votes on images no "
        "cell will ever hold.",
    )
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument(
        "--inside-strata",
        default="",
        help="e.g. '0,2-4,5-9,10+': stratify on how many LVIS instances lie inside the picked (largest) "
        "COCO box instead of on the area ratio, --n per stratum. The picked box and the band follow "
        "`largest_box`, which is what the build drags since #4096. Images LVIS never boxed are "
        "included, since stratum 0 is a question too.",
    )
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="manifests whose annotations are not re-asked")
    args = ap.parse_args()

    coco: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    dims: dict = {}
    for split in ("val2017", "train2017"):
        per, d = _read(args.annotations / f"instances_{split}.json")
        dims.update(d)
        for iid, m in per.items():
            for cls, bs in m.items():
                coco[iid][cls] += bs
    lvis: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in args.lvis:
        per, _ = _read(path)
        for iid, m in per.items():
            for cls, bs in m.items():
                lvis[iid][cls] += bs
    names = [n.strip() for n in args.lvis_names.split(",")]

    strata = [t.strip() for t in args.inside_strata.split(",") if t.strip()]
    asked = {i["ann_id"] for m in args.exclude for i in json.loads(m.read_text())["items"]}

    def stratum(n: int) -> str | None:
        for t in strata:
            lo_, _, hi_ = t.rstrip("+").partition("-")
            if n >= int(lo_) and (t.endswith("+") or n <= int(hi_ or lo_)):
                return t
        return None

    rows = []
    for iid, m in coco.items():
        cb = m.get(args.klass)
        lb = [b for n in names for b in (lvis.get(iid, {}).get(n) or [])]
        if strata and cb:
            w, h, _ = dims[iid]
            xyxy = {(x, y, x + bw, y + bh): ann for (x, y, bw, bh), ann in cb}
            pick = largest_box([list(k) for k in xyxy])
            ann = xyxy[tuple(pick)]
            if ann in asked or band_for([pick], w, h) not in pc.BOX_BANDS:
                continue
            inside = sum(
                _share_inside([x, y, x + bw, y + bh], pick) >= pc.SCALE_LUMP_CONTAIN for (x, y, bw, bh), _ in lb
            )
            t = stratum(inside)
            if t is not None:
                bx = [pick[0], pick[1], pick[2] - pick[0], pick[3] - pick[1]]
                rows.append(
                    {
                        "image_id": iid,
                        "bbox": bx,
                        "ann_id": ann,
                        "ratio": float(inside),
                        "n_coco": len(cb),
                        "n_lvis": len(lb),
                        "arm": t,
                        "lvis_inside": inside,
                    }
                )
            continue
        if not cb or not lb:
            continue
        w, h, _ = dims[iid]
        if args.banded_only:
            xyxy = [[x, y, x + bw, y + bh] for (x, y, bw, bh), _ in cb]
            if band_for(xyxy, w, h) not in pc.BOX_BANDS:
                continue
        ca = sum(b[2] * b[3] for b, _ in cb) / len(cb) / (w * h)
        la = sum(b[2] * b[3] for b, _ in lb) / len(lb) / (w * h)
        if la <= 0:
            continue
        # The BIGGEST COCO box is the one a lump would be, so that is the one asked about.
        box, ann = max(cb, key=lambda t: t[0][2] * t[0][3])
        rows.append(
            {"image_id": iid, "bbox": box, "ann_id": ann, "ratio": ca / la, "n_coco": len(cb), "n_lvis": len(lb)}
        )

    rng = random.Random(args.seed)
    if strata:
        by: dict = collections.defaultdict(list)
        for r in rows:
            by[r["arm"]].append(r)
        print(f"{args.klass}: {len(rows)} banded images by LVIS-inside stratum:", {t: len(by[t]) for t in strata})
        picked = []
        for t in strata:
            rng.shuffle(by[t])
            picked += by[t][: args.n]
        rng.shuffle(picked)
    else:
        picked = _ratio_arms(rows, args, rng)
    return _render(picked, args, dims)


def _ratio_arms(rows: list, args, rng: random.Random) -> list:
    """Half lumped (ratio >= LUMPED), half clean (< CLEAN) -- the #3985 design."""
    hi = [r for r in rows if r["ratio"] >= LUMPED]
    lo = [r for r in rows if r["ratio"] < CLEAN]
    print(f"{args.klass}: {len(rows)} comparable images -- {len(hi)} lumped (>={LUMPED}), {len(lo)} clean (<{CLEAN})")
    if len(hi) < 4 or len(lo) < 4:
        raise SystemExit("not enough on one side to make a control arm")

    rng.shuffle(hi)
    rng.shuffle(lo)
    half = args.n // 2
    picked = [dict(r, arm="lumped") for r in hi[:half]] + [dict(r, arm="clean") for r in lo[: args.n - half]]
    rng.shuffle(picked)
    return picked


def _render(picked: list, args, dims: dict) -> int:
    """Crop, outline and write each picked box, and the manifest beside them."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    slug = args.klass.replace(" ", "_")
    outdir = args.out / slug / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    members: dict = {}
    for zp in (pc.COCO_VAL_ZIP, pc.COCO_TRAIN_ZIP):
        with zipfile.ZipFile(zp) as zf:
            for nm in zf.namelist():
                if nm.lower().endswith(".jpg"):
                    members[Path(nm).name] = (zp, nm)

    manifest, kb, zc = [], [], {}
    for r in picked:
        w, h, fname = dims[r["image_id"]]
        zp, member = members[fname]
        if zp not in zc:
            zc[zp] = zipfile.ZipFile(zp)
        with zc[zp].open(member) as fh:
            im = Image.open(_io.BytesIO(fh.read())).convert("RGB")
        x, y, bw, bh = r["bbox"]
        pad = CONTEXT * max(bw, bh)
        while max(bw + 2 * pad, bh + 2 * pad) < MIN_WINDOW and (pad < w or pad < h):
            pad *= 1.5
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
        draw = ImageDraw.Draw(crop)
        bx0, by0, bx1, by1 = x - x0, y - y0, x + bw - x0, y + bh - y0
        lw = max(2, int(0.004 * max(crop.size)))
        draw.rectangle([bx0 - lw, by0 - lw, bx1 + lw, by1 + lw], outline=(255, 255, 255), width=lw)
        draw.rectangle([bx0, by0, bx1, by1], outline=(255, 64, 0), width=lw)
        scale = (
            MAX_SIDE / max(crop.size)
            if max(crop.size) > MAX_SIDE
            else (MIN_SIDE / max(crop.size) if max(crop.size) < MIN_SIDE else None)
        )
        if scale:
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
        stem = hashlib.sha1(f"{args.seed}:{r['ann_id']}".encode()).hexdigest()[:12]  # noqa: S324
        p = outdir / f"{slug}_{stem}.jpg"
        crop.save(p, quality=QUALITY, optimize=True)
        kb.append(p.stat().st_size / 1024)
        manifest.append({"file": p.name, **r})

    kb.sort()
    (args.out / slug / "manifest.json").write_text(
        json.dumps(
            {
                "class": args.klass,
                "lvis_names": [n.strip() for n in args.lvis_names.split(",")],
                "seed": args.seed,
                "items": manifest,
            },
            indent=1,
        )
    )
    print(f"  wrote {len(manifest)} crops; size median {kb[len(kb) // 2]:.0f} KB, p90 {kb[int(0.9 * len(kb))]:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

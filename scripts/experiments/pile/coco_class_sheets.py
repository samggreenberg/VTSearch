"""Show what COCO actually means by each class, in its own annotators' examples.

Prose definitions of an annotation class are worth less than the annotations.
"Is a messenger bag a `backpack`?" is not settled by a dictionary; it is settled
by what COCO's annotators did, repeatedly, across 118k images -- and the only
way to transfer that to a human reviewer is to show them.

So: for each class, real COCO instances cropped with context, **four per size
band**, because a reviewer of this study has to recognise the class at every
scale and a `bus` at 0.3% of the frame does not look like the bus in anyone's
mental image. Alongside each target class, the sibling classes that steal from
it (`handbag`/`suitcase` from `backpack`, `motorcycle` from `bicycle`), since a
boundary is only learnable by seeing both sides of it.

Also reports, per class, the measured size distribution and the share of
instances below one DINOv3 patch -- which is how a suspicion like "COCO labels
wristwatches as clocks" becomes a number rather than an argument.

Usage::

    python coco_class_sheets.py --out /expscratch/$USER/vgscale-3156/sheets
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pile_config as pc

pc.setup_env()

#: Sibling classes worth showing beside a target, because COCO puts them in a
#: different class and a reviewer who conflates them corrupts both.
CONFUSABLE: dict[str, tuple[str, ...]] = {
    "backpack": ("handbag", "suitcase"),
    "bicycle": ("motorcycle",),
    "boat": ("surfboard",),
    "bus": ("truck", "train", "car"),
    "knife": ("scissors", "fork"),
    "kite": ("umbrella", "airplane"),
    "dog": ("cat", "horse", "sheep", "bear"),
    "bird": ("kite",),
    "clock": ("cell phone",),
    "book": ("laptop",),
    "umbrella": ("kite",),
    "stop sign": ("traffic light",),
}

TILE = 240
PER_BAND = 4


def log(msg: str) -> None:
    print(f"[sheets] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(pc.PILE.parent / "vgscale-3156" / "sheets"))
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()

    from PIL import Image, ImageDraw  # noqa: PLC0415

    import coco_anchor as ca  # noqa: PLC0415
    from build_pile import _vg_image_paths  # noqa: PLC0415

    wanted = set(pc.SCALE_CLASSES) | {s for v in CONFUSABLE.values() for s in v}
    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, wanted)
    with image_data.open() as fh:
        vg_of = {int(m["coco_id"]): int(m["image_id"]) for m in json.load(fh) if m.get("coco_id")}
    paths = _vg_image_paths()
    log(f"{len(truth)} COCO images, {len(vg_of)} with a VG copy on disk")

    # class -> band -> [(coco_id, box, area_frac)]
    pool: dict[str, dict[str, list]] = {c: {b: [] for b in pc.BOX_BANDS} for c in wanted}
    areas: dict[str, list[float]] = {c: [] for c in wanted}
    for cid, ref in truth.items():
        wh = ca.COCO_DIMS.get(cid)
        vg = vg_of.get(cid)
        if not ref or wh is None or vg is None or vg not in paths:
            continue
        area = float(wh[0] * wh[1])
        for name, boxes in ref.items():
            for b in boxes:
                frac = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])) / area
                areas[name].append(frac)
                for band, (lo, hi) in pc.BOX_BANDS.items():
                    if lo <= frac < hi:
                        pool[name][band].append((cid, b, frac))
                        break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    stats = {}

    for name in sorted(wanted):
        rows = []
        for band in pc.BOX_BANDS:
            items = pool[name][band]
            rows.append((band, rng.sample(items, min(PER_BAND, len(items)))))
        if not any(r[1] for r in rows):
            continue
        sheet = Image.new("RGB", (TILE * PER_BAND, TILE * len(rows)), (18, 20, 21))
        d = ImageDraw.Draw(sheet)
        for ri, (band, items) in enumerate(rows):
            for ci, (cid, box, frac) in enumerate(items):
                src = paths[vg_of[cid]]
                try:
                    with Image.open(src) as im:
                        im = im.convert("RGB")
                        W, H = im.size
                        x0, x1 = sorted((box[0], box[2]))
                        y0, y1 = sorted((box[1], box[3]))
                        pad = max((x1 - x0), (y1 - y0)) * 0.35 + min(W, H) * 0.03
                        crop = im.crop(
                            (
                                max(0, int(x0 - pad)),
                                max(0, int(y0 - pad)),
                                min(W, int(x1 + pad) + 2),
                                min(H, int(y1 + pad) + 2),
                            )
                        )
                        crop.thumbnail((TILE, TILE), Image.LANCZOS)
                except Exception:  # noqa: BLE001 - a corrupt file just leaves a gap
                    continue
                ox, oy = ci * TILE, ri * TILE
                sheet.paste(crop, (ox + (TILE - crop.width) // 2, oy + (TILE - crop.height) // 2))
                d.text((ox + 5, oy + 5), f"{band} {frac * 100:.2f}%", fill=(255, 210, 90))
            d.line([(0, ri * TILE), (TILE * PER_BAND, ri * TILE)], fill=(90, 96, 98), width=1)
        fname = name.replace(" ", "_") + ".jpg"
        sheet.save(out / fname, quality=86)
        a = sorted(areas[name])
        stats[name] = {
            "n_instances": len(a),
            "median_area_pct": round(100 * a[len(a) // 2], 3) if a else None,
            "sub_patch_pct": round(100 * sum(1 for x in a if x < pc.PATCH_AREA) / len(a), 1) if a else None,
            "sheet": fname,
            "confusable_with": list(CONFUSABLE.get(name, ())),
        }
        log(
            f"  {name:<14} {len(a):6d} instances  median {stats[name]['median_area_pct']}%  "
            f"sub-patch {stats[name]['sub_patch_pct']}%"
        )

    (out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print(f"\n{len(stats)} sheets under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

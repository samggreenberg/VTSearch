#!/usr/bin/env python3
"""Render COCO `vase` instances so we can see whether COCO counts a flower pot.

The empty-planter question cannot be answered from the rule text -- `bowl`
refuses it ("made to hold food"), `potted plant` needs a plant, and `vase`'s
inclusion of planters is the very line under review. COCO annotated these
images, so what COCO drew a `vase` box around is the evidence.

Deliberately drawn from images COCO gives NO `potted plant`, so anything
plant-pot-shaped here is a pot COCO chose to call a vase rather than one it
declined to annotate.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402

pc.setup_env()
import coco_anchor as ca  # noqa: E402

OUT = Path("/expscratch/sgreenberg/vlm-3720/coco_vase_crops")
OUT.mkdir(parents=True, exist_ok=True)
image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
truth = ca.coco_truth(instances, {"vase", "potted plant"})

# coco_id -> a VG path we can actually open
vg_of = {}
with image_data.open() as fh:
    for m in json.load(fh):
        if m.get("coco_id"):
            vg_of[int(m["coco_id"])] = m
cands = []
for cid, d in truth.items():
    v, pp = d.get("vase") or [], d.get("potted plant") or []
    if v and not pp and cid in vg_of:
        cands.append((cid, v))
print(f"{len(cands):,} COCO images with a vase and no potted plant")

rng = random.Random(3778)
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path("scripts/experiments/pile").resolve()))
from make_positive_slate import draw_with_inset  # noqa: E402

picked = rng.sample(cands, 8)
for cid, boxes in picked:
    m = vg_of[cid]
    iid = int(m["image_id"])
    # image_data carries ids, not paths; VG splits its images over two folders
    root = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")
    src = next((root / d / f"{iid}.jpg" for d in ("VG_100K", "VG_100K_2") if (root / d / f"{iid}.jpg").exists()), None)
    if src is None:
        continue
    with Image.open(src) as im:
        W, H = im.size
    b = max(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
    nb = (b[0] / W, b[1] / H, b[2] / W, b[3] / H)
    dest = OUT / f"coco{cid}.jpg"
    try:
        draw_with_inset(src, nb, dest)
        print(f"  wrote {dest.name}  box {tuple(round(v, 2) for v in nb)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {cid}: {exc}")

#!/usr/bin/env python3
"""What does COCO actually call a `vase`? (#3778)

The empty-planter question is not answerable from the rule text, and three
eyeballed crops are not an answer either. This crops every COCO `vase` box in
images COCO gives NO `potted plant` -- so nothing here is a pot COCO declined to
annotate -- and asks an open-vocabulary detector which of a handful of words the
crop best matches.

It measures what the objects LOOK like, not what COCO meant, so it can only
support a reading rather than prove one. But if COCO's vase set were largely
flower pots, that would show, and it is the difference between a ruling with
evidence and a ruling with a hunch.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

QUERIES = [
    "a photo of a vase",
    "a photo of a flower pot",
    "a photo of a potted plant",
    "a photo of a jug or pitcher",
    "a photo of a bowl",
    "a photo of a cup",
]
SHORT = ["vase", "flower pot", "potted plant", "pitcher", "bowl", "cup"]
ROOT = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")


def main() -> int:
    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    pc.setup_env()
    import coco_anchor as ca  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: PLC0415

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, {"vase", "potted plant"})
    vg_of = {}
    with image_data.open() as fh:
        for m in json.load(fh):
            if m.get("coco_id"):
                vg_of[int(m["coco_id"])] = int(m["image_id"])

    jobs = []
    for cid, d in truth.items():
        v, pp = d.get("vase") or [], d.get("potted plant") or []
        if not v or pp or cid not in vg_of:
            continue
        iid = vg_of[cid]
        src = next(
            (ROOT / s / f"{iid}.jpg" for s in ("VG_100K", "VG_100K_2") if (ROOT / s / f"{iid}.jpg").exists()), None
        )
        if src:
            jobs.append((cid, src, v))
    print(f"[what] {len(jobs)} COCO vase images with no potted plant", flush=True)

    m = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(m)
    model = Owlv2ForObjectDetection.from_pretrained(m).to("cuda").eval()

    tally, skipped, rows = Counter(), 0, []
    for n, (cid, src, boxes) in enumerate(jobs, 1):
        try:
            im = Image.open(src).convert("RGB")
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        W, H = im.size
        b = max(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        # COCO boxes arrive in COCO's pixel frame, and 88% of anchored pairs are
        # a uniform rescale of it (median 1.28x). Aspect gating cannot see a pure
        # rescale, so the box must be carried across by the DIMENSION RATIO --
        # dividing by the VG size instead is a silent 28% error.
        cw, ch = ca.COCO_DIMS.get(cid, (0, 0))
        if not cw or not ch:
            skipped += 1
            continue
        sx, sy = W / cw, H / ch
        b = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        if b[2] > W + 2 or b[3] > H + 2 or b[0] < -2 or b[1] < -2:
            skipped += 1
            continue
        pad = 0.15 * max(b[2] - b[0], b[3] - b[1])
        crop = im.crop((max(0, b[0] - pad), max(0, b[1] - pad), min(W, b[2] + pad), min(H, b[3] + pad)))
        if crop.width < 8 or crop.height < 8:
            skipped += 1
            continue
        inputs = proc(text=[QUERIES], images=[crop], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs=o, target_sizes=torch.tensor([[crop.height, crop.width]], device="cuda"), threshold=0.02
        )[0]
        if not len(res["scores"]):
            tally["no detection"] += 1
            continue
        k = SHORT[int(res["labels"][int(res["scores"].argmax())])]
        tally[k] += 1
        rows.append({"coco_id": cid, "top": k})
        if n % 100 == 0:
            print(f"[what] {n}/{len(jobs)}", flush=True)

    Path("/expscratch/sgreenberg/vlm-3720/coco_vase_what2.json").write_text(
        json.dumps({"tally": dict(tally), "skipped": skipped, "rows": rows}, indent=1) + "\n"
    )
    total = sum(tally.values())
    print(f"\nwhat COCO's `vase` boxes look like ({total} crops, {skipped} skipped as mis-scaled or tiny):")
    for k, c in tally.most_common():
        print(f"   {k:<14}{c:>6}  {100 * c / total:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

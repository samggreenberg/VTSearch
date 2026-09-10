#!/usr/bin/env python3
"""Are any pre-slate `bowl` or `vase` positives actually planters? (#3778)

`bowl` never admitted planters, so by the rule nothing is owed. But the reviewer
reported thinking of pots AS bowls, and a verdict follows the person, not the
text -- so the question is what the boxed vessels actually are. The old `vase`
rows are the other half: under the wording in force when they were cast, a
planter WAS a vase, so any planter among them is now out of C.

Boxes here are already normalised (`box_space: normalised`) against the VG image,
so no COCO rescale applies -- unlike the join that trap caught earlier.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

QUERIES = [
    "a photo of a bowl",
    "a photo of a vase",
    "a photo of a flower pot",
    "a photo of a potted plant",
    "a photo of a cup",
    "a photo of a plate",
]
SHORT = ["bowl", "vase", "flower pot", "potted plant", "cup", "plate"]
ROOT = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")


def main() -> int:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: PLC0415

    rows = json.loads(Path("/expscratch/sgreenberg/vts-cache/corrections.json").read_text())
    jobs = []
    for r in rows:
        if r.get("class") not in ("bowl", "vase") or not r.get("present"):
            continue
        boxes = r.get("boxes") or []
        if not boxes:
            continue
        iid = int(r["image_id"])
        src = next(
            (ROOT / s / f"{iid}.jpg" for s in ("VG_100K", "VG_100K_2") if (ROOT / s / f"{iid}.jpg").exists()), None
        )
        if src:
            jobs.append((r["class"], iid, src, boxes[0]))
    print(f"[pots] {len(jobs)} old positives with a box", flush=True)

    m = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(m)
    model = Owlv2ForObjectDetection.from_pretrained(m).to("cuda").eval()

    tally = {"bowl": Counter(), "vase": Counter()}
    flagged = []
    for cls, iid, src, nb in jobs:
        im = Image.open(src).convert("RGB")
        W, H = im.size
        x0, y0, x1, y1 = nb[0] * W, nb[1] * H, nb[2] * W, nb[3] * H
        pad = 0.15 * max(x1 - x0, y1 - y0)
        crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))
        if crop.width < 8 or crop.height < 8:
            tally[cls]["too small"] += 1
            continue
        inputs = proc(text=[QUERIES], images=[crop], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs=o, target_sizes=torch.tensor([[crop.height, crop.width]], device="cuda"), threshold=0.02
        )[0]
        if not len(res["scores"]):
            tally[cls]["no detection"] += 1
            continue
        k = SHORT[int(res["labels"][int(res["scores"].argmax())])]
        tally[cls][k] += 1
        if k in ("flower pot", "potted plant"):
            flagged.append({"class": cls, "image_id": iid, "top": k})

    for cls in ("bowl", "vase"):
        tot = sum(tally[cls].values())
        print(f"\nold `{cls}` positives ({tot}):")
        for k, n in tally[cls].most_common():
            print(f"   {k:<14}{n:>4}  {100 * n / tot:5.1f}%")
    print(f"\nplanter-ish among the old positives: {len(flagged)}")
    for f in flagged:
        print(f"   {f['class']:<6}{f['image_id']}  {f['top']}")
    Path("/expscratch/sgreenberg/vlm-3720/old_pots.json").write_text(json.dumps({"flagged": flagged}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

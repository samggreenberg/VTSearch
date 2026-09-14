#!/usr/bin/env python3
"""How many of the vase rejects are actually planters? (#3778)

The planter ruling is worth whatever it changes, and nothing so far says how
many images it touches. This scores the finished vase slate against queries the
class list does not carry -- `potted plant`, `flower pot`, `pitcher`, `lamp` --
so the reviewer's own good/bad split can be read against what each image most
likely holds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

QUERIES = [
    "a photo of a potted plant",
    "a photo of a flower pot",
    "a photo of a vase",
    "a photo of a pitcher",
    "a photo of a lamp",
]
SHORT = ["potted plant", "flower pot", "vase", "pitcher", "lamp"]


def main() -> int:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: PLC0415

    slate = Path("/expscratch/sgreenberg/vlm-3720/slates/vase/images")
    src = json.loads(Path("/expscratch/sgreenberg/vlm-3720/slates/slates.json").read_text())["vase"]
    paths = {int(r["image_id"]): r["path"] for r in src["rows"]}
    ids = sorted(int(p.stem) for p in slate.glob("*.jpg"))
    print(f"[planter] {len(ids)} vase-slate images", flush=True)

    m = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(m)
    model = Owlv2ForObjectDetection.from_pretrained(m).to("cuda").eval()

    out = Path("/expscratch/sgreenberg/vlm-3720/vase_planter.jsonl").open("w")
    t0 = time.time()
    B = 8
    for i in range(0, len(ids), B):
        chunk = ids[i : i + B]
        imgs, keep = [], []
        for iid in chunk:
            try:
                imgs.append(Image.open(paths[iid]).convert("RGB"))
                keep.append(iid)
            except Exception:  # noqa: BLE001
                continue
        if not imgs:
            continue
        inputs = proc(text=[QUERIES] * len(imgs), images=imgs, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = model(**inputs)
        sizes = torch.tensor([[im.height, im.width] for im in imgs], device="cuda")
        res = proc.post_process_grounded_object_detection(outputs=o, target_sizes=sizes, threshold=0.03)
        for iid, r in zip(keep, res):
            best = {}
            for s, lab in zip(r["scores"], r["labels"]):
                k = SHORT[int(lab)]
                best[k] = max(best.get(k, 0.0), round(float(s), 4))
            out.write(json.dumps({"image_id": iid, "best": best}) + "\n")
        out.flush()
        if (i // B) % 10 == 0:
            print(f"[planter] {i + len(keep)}/{len(ids)}", flush=True)
    out.close()
    print(f"[planter] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

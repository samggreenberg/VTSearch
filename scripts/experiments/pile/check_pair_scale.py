#!/usr/bin/env python3
"""Do anchored VG<->COCO pairs actually share the same pixel dimensions? (#3720)

`anchor_to_coco` gates a pairing on ASPECT-RATIO drift (MAX_ASPECT_DRIFT = 0.01),
which is scale-invariant by construction: a 500x375 image and a 1000x750 one pass
it. That is fine for anything comparing shapes, and wrong for anything that maps
a COCO BOX onto the VG image, because the box arrives in COCO's pixel frame.

Rendering COCO vase boxes onto their VG images turned up boxes running past the
image edge, so this measures how often the two frames disagree, and by how much.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/exp/scale26/datasets/external/vtsearch-demos/visual_genome")
SAMPLE = 4000


def main() -> int:
    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    pc.setup_env()
    import coco_anchor as ca  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    image_data, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    ca.coco_truth(instances, set(pc.SCALE_CLASSES))  # populates COCO_DIMS
    with image_data.open() as fh:
        pairs = [(int(m["image_id"]), int(m["coco_id"])) for m in json.load(fh) if m.get("coco_id")]
    rng = random.Random(3720)
    pairs = rng.sample(pairs, min(SAMPLE, len(pairs)))
    print(f"checking {len(pairs)} anchored pairs")

    same = diff = missing = 0
    ratios = []
    tally = Counter()
    for vg_id, cid in pairs:
        dims = ca.COCO_DIMS.get(cid)
        src = next(
            (ROOT / s / f"{vg_id}.jpg" for s in ("VG_100K", "VG_100K_2") if (ROOT / s / f"{vg_id}.jpg").exists()), None
        )
        if not dims or src is None:
            missing += 1
            continue
        try:
            with Image.open(src) as im:  # header only, no decode
                vw, vh = im.size
        except Exception:  # noqa: BLE001
            missing += 1
            continue
        cw, ch = dims
        if (cw, ch) == (vw, vh):
            same += 1
            tally["identical"] += 1
            continue
        diff += 1
        r = (cw / vw + ch / vh) / 2
        ratios.append(r)
        tally["COCO larger" if r > 1.02 else "VG larger" if r < 0.98 else "off by <2%"] += 1

    tot = same + diff
    print(f"\n{tot} pairs compared ({missing} unreadable)")
    print(f"  identical dimensions : {same:,}  ({100 * same / tot:.1f}%)")
    print(f"  different dimensions : {diff:,}  ({100 * diff / tot:.1f}%)")
    for k, n in tally.most_common():
        print(f"     {k:<16}{n:>7}")
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        print(f"\n  median COCO/VG scale where they differ: {med:.3f}")
        print(f"  10th/90th percentile: {ratios[len(ratios) // 10]:.3f} / {ratios[9 * len(ratios) // 10]:.3f}")
    print(
        "\nreading: a COCO box mapped onto a VG image is only valid where the "
        "dimensions\nmatch. Aspect-ratio gating cannot catch a pure rescale."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

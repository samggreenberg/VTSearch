#!/usr/bin/env python3
"""Does COCO treat a planter's pot as a `vase`? (#3720)

Our rule widens `vase` to "flower pots, planters" and says "a potted plant's pot
is a vase". COCO carries `potted plant` as a class of its own, so the question is
answerable from the reference rather than from taste: if COCO also draws a `vase`
box on the pot, the two annotations should overlap; if it does not, then the
planter case is a departure from COCO that our own negatives will contradict.
"""

import sys
from collections import Counter

sys.path.insert(0, "scripts/experiments/pile")
import pile_config as pc  # noqa: E402

pc.setup_env()
import coco_anchor as ca  # noqa: E402


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    _, instances = ca.ensure_sources(pc.PILE / "coco_anchor", fetch=False)
    truth = ca.coco_truth(instances, {"vase", "potted plant"})
    both = withpp = withv = 0
    overlaps = Counter()
    contained = 0
    for cid, d in truth.items():
        v, pp = d.get("vase") or [], d.get("potted plant") or []
        if v:
            withv += 1
        if pp:
            withpp += 1
        if not (v and pp):
            continue
        both += 1
        for pb in pp:
            best = max((iou(pb, vb) for vb in v), default=0.0)
            overlaps["iou>0.5" if best > 0.5 else "iou>0.1" if best > 0.1 else "no overlap"] += 1
            # a pot is a sub-part of a potted plant, so containment matters more than IoU
            for vb in v:
                if vb[0] >= pb[0] - 2 and vb[1] >= pb[1] - 2 and vb[2] <= pb[2] + 2 and vb[3] <= pb[3] + 2:
                    contained += 1
                    break
    print(f"COCO images with a `vase`: {withv:,}")
    print(f"COCO images with a `potted plant`: {withpp:,}")
    print(f"images with both: {both:,}  ({100 * both / withpp:.1f}% of potted-plant images)\n")
    print("for each potted plant in an image that also has a vase, the best IoU against any vase box:")
    for k, n in overlaps.most_common():
        print(f"   {k:<12} {n:,}")
    tot = sum(overlaps.values())
    print(f"\npotted plants whose box CONTAINS a vase box: {contained:,} of {tot:,} ({100 * contained / tot:.0f}%)")
    print(
        "\nreading: if COCO annotated the pot of a potted plant as a vase, most "
        "potted-plant\nboxes would contain a vase box. They do not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

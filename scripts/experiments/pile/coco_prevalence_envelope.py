#!/usr/bin/env python3
"""If prevalence stops being a build-time designation, what range can COCO serve?

`SCALE_PREVALENCE` is fixed when the pile is built, so changing it is a rebuild
and a re-embed -- and `pile_config` already records that the shipped constant is
the *designed* prevalence rather than the realised one (#3681). Under an
exhaustively annotated source every ``(image, class)`` pair is answered, so a
cell is a **filter over a fixed set** and prevalence becomes a sampling
parameter: one meta-dataset can export a version at any pi the data supports, and
prevalence turns into an experimental axis instead of a constant.

This measures the envelope. ``pi = P / (P + N)``: positives are capped by a
cell's supply, negatives by the images that lack the class. **The low end is the
hard one** -- 0.1% at 100 positives needs 99,900 negatives -- and it is where the
interesting question lives, because a rare-needle haystack is what the app faces.

The headline the run prints: at the shipped ``SCALE_N_POS`` every class but
`person` reaches 0.1%, and halving the positive count reaches it for all of them.

**The embed is sized by the lowest prevalence, not by the class list.** Once the
images are embedded, every ``(class, band, prevalence)`` version is a filter that
costs nothing -- so the only real decision is how deep the haystack must go, and
embedding the whole source once settles it permanently.

    python coco_prevalence_envelope.py --annotations <dir holding instances_*2017.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402
from pilebuild.loaders.vg_scale import band_for  # noqa: E402

BANDS = ("small", "medium", "large")
#: Prevalences to report reach for. 5% is roughly today's realised figure; 0.1%
#: is the rare-needle end a calibration study wants.
TARGETS = (0.05, 0.01, 0.005, 0.002, 0.001, 0.0005)
#: Positive counts to try. The first is the shipped `SCALE_N_POS`.
POSITIVE_COUNTS = (pc.SCALE_N_POS, 50, 25)
#: Minutes per 6,000 images across all five embedders, measured in #3670.
MINUTES_PER_6K = 15.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dims: dict[int, tuple[int, int]] = {}
    boxes: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for split in ("val2017", "train2017"):
        with (args.annotations / f"instances_{split}.json").open() as fh:
            data = json.load(fh)
        cats = {c["id"]: c["name"] for c in data["categories"]}
        for im in data["images"]:
            dims[im["id"]] = (int(im["width"]), int(im["height"]))
        for a in data["annotations"]:
            if a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            if w > 0 and h > 0:
                boxes[a["image_id"]][cats[a["category_id"]]].append([x, y, x + w, y + h])
        del data

    cell: collections.Counter = collections.Counter()
    holders: dict[str, set] = collections.defaultdict(set)
    for iid, per in boxes.items():
        w, h = dims[iid]
        for cls, bs in per.items():
            holders[cls].add(iid)
            band = band_for(bs, w, h)
            if band in BANDS:
                cell[(cls, band)] += 1

    total = len(dims)
    roster = [c for c in sorted({c for c, _ in cell}) if min(cell[(c, b)] for b in BANDS) >= pc.SCALE_N_POS]
    # A negative need only lack its own class (#3986), which is what makes the
    # low end reachable at all: "holds none of C" would leave 16,058 images.
    neg = {c: total - len(holders[c]) for c in roster}

    print(f"COCO = {total:,} images; |C| = {len(roster)}\n")
    print("classes reaching each prevalence, at a fixed positive count per cell:")
    header = "".join(f"{t * 100:>9.2f}%" for t in TARGETS)
    print(f"{'positives':>11}{header}")
    print("-" * (11 + 10 * len(TARGETS)))
    reach = {}
    for p in POSITIVE_COUNTS:
        row = []
        for target in TARGETS:
            need = p * (1 - target) / target
            row.append(sum(1 for c in roster if neg[c] >= need))
        reach[p] = row
        print(f"{p:>11}" + "".join(f"{n:>10}" for n in row))

    floor_target = 0.001
    need = pc.SCALE_N_POS * (1 - floor_target) / floor_target
    short = sorted(((c, neg[c]) for c in roster if neg[c] < need), key=lambda t: t[1])
    print(f"\nshort of {floor_target * 100:.1f}% at {pc.SCALE_N_POS} positives (needs {need:,.0f} negatives):")
    for c, n in short or []:
        print(f"  {c:<15}{n:>9,} negatives -> floor {100 * pc.SCALE_N_POS / (pc.SCALE_N_POS + n):.3f}%")
    if not short:
        print("  none")

    hours = total / 6000 * MINUTES_PER_6K / 60
    print(f"\nembedding the whole source once: {total:,} images, ~{hours:.1f} h for all five")
    print("embedders. Every (class, band, prevalence) version is then a filter costing nothing.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "images": total,
                    "roster": roster,
                    "negatives": neg,
                    "reach": {str(p): dict(zip(map(str, TARGETS), r)) for p, r in reach.items()},
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

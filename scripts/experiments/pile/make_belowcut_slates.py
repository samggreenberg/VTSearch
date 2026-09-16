#!/usr/bin/env python3
"""Probe the images the screen never flagged, to bound what the cut is losing (#3879).

The per-class pass (#3720) asked "is this box one?" of every candidate the OWLv2
screen flagged at or above the class cut. It asked nothing at all about the rest
of the corpus, and the cut was set to ~95% recall on the pilot, so by
construction it drops some positives. This is the complementary probe: a sample
of the images a class was NOT shown, asked as "is there one anywhere in this
image?" with no box to confirm -- there is nothing to draw a box around when the
premise is that the screen missed it.

**The name is misleading and the population is not what it says.** "Below cut"
suggests the detections that scored under the threshold. It is not: of the 100
images in each of the six probes already run, only 11-25 carry *any* detection
for their class. The rest were never flagged at all. So the population is the
class's whole unswept remainder -- every image in the pass that is not a known A
and not in that class's slate -- which is ~2,200-3,200 per class, an order of
magnitude more than the few hundred sub-threshold detections.

Six classes have such a probe (`bicycle`, `boat`, `dog`, `fire hydrant`, `sink`,
`stop sign`) and all six returned **0 good / 100 bad**. This builds the same
probe for the other nineteen, so the resulting miss-rate ceiling is per-class
rather than pooled over the six smallest-slate classes.

The six were built by a script that was never committed, which is why this one
is: the population rule above had to be reverse-engineered from the manifest it
left behind, and five of the six reproduce exactly under it.

Usage::

    python make_belowcut_slates.py --out /expscratch/$USER/vlm-3720/belowcut19 --dry-run
    python make_belowcut_slates.py --out /expscratch/$USER/vlm-3720/belowcut19
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pile_config as pc

#: Matches the six probes already run, so the new nineteen are the same
#: experiment rather than a differently-powered one.
PER_CLASS = 100

#: Fixed so a rebuild draws the same images and can be diffed rather than
#: re-argued -- the reason `make_pass25.CONTROL_SEED` is fixed too.
SEED = 3879

#: Classes whose probe is already banked as LABELSETS__belowcut__<class>.json.
ALREADY = ("bicycle", "boat", "dog", "fire hydrant", "sink", "stop sign")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="/expscratch/sgreenberg/vlm-3720/owl_queue.jsonl")
    ap.add_argument("--slates", default="/expscratch/sgreenberg/vlm-3720/slates/slates.json")
    ap.add_argument("--queue", default="/expscratch/sgreenberg/vgscale-3156/annotation_queue.jsonl")
    ap.add_argument("--banked", default="/expscratch/sgreenberg/vlm-3720/banked_labels.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_pile import _vg_image_paths  # noqa: PLC0415

    images = set()
    for line in Path(args.dets).read_text().splitlines():
        if line.strip():
            images.add(int(json.loads(line)["image_id"]))

    # The designated positives: the question there was answered by the
    # designation, so re-asking it is duplicated labour.
    known: dict[str, set[int]] = defaultdict(set)
    for line in Path(args.queue).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            for c in r["classes"]:
                known[c].add(int(r["image_id"]))

    # Everything the class already showed the reviewer, under its own slate.
    slates = json.loads(Path(args.slates).read_text())
    in_slate: dict[str, set[int]] = {
        c: {int(r["image_id"]) for r in slates.get(c, {}).get("rows", [])} for c in pc.SCALE_CLASSES
    }

    # And anything answered under the earlier one-dataset pass. Only `bicycle`
    # and `bench` have such a set; `bicycle` is already probed, so in practice
    # this only moves `bench`. A verdict is a verdict whatever slate it was cast
    # in -- the same rule `make_class_slates.py` applies.
    answered: dict[str, set[int]] = defaultdict(set)
    if Path(args.banked).exists():
        name2cls = {"bicycle incl trikes not motorcycles": "bicycle", "bench not chairs": "bench"}
        for det, body in json.loads(Path(args.banked).read_text()).items():
            cls = name2cls.get(det)
            if not cls:
                continue
            for side in ("good", "bad"):
                for row in body.get("labels-detail", {}).get(side, []):
                    try:
                        answered[cls].add(int(str(row["filename"]).split(".")[0]))
                    except (ValueError, KeyError, TypeError):
                        continue

    paths = _vg_image_paths()
    rng = random.Random(args.seed)
    out = Path(args.out)
    manifest: dict[str, dict] = {}
    todo = [c for c in pc.SCALE_CLASSES if c not in ALREADY]

    print(f"{'class':<13}{'unswept':>10}{'sampled':>9}{'missing img':>13}  detector")
    print("-" * 95)
    total = 0
    for cls in todo:
        pool = sorted(images - known[cls] - in_slate[cls] - answered[cls])
        pick = rng.sample(pool, min(args.per_class, len(pool)))
        missing = [i for i in pick if i not in paths]
        name = f"{pc.review_name(cls)} (any in image, no box)"
        manifest[cls] = {
            "below": len(pool),
            "sampled": len(pick),
            "detector": name,
            "seed": args.seed,
            "ids": sorted(pick),
        }
        total += len(pick)
        print(f"{cls:<13}{len(pool):>10,}{len(pick):>9}{len(missing):>13}  {name}")
        if args.dry_run:
            continue
        d = out / cls.replace(" ", "_") / "images"
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jpg"):
            old.unlink()
        for iid in pick:
            src = paths.get(iid)
            if src is None:
                continue
            (d / f"{iid}.jpg").symlink_to(src)
    print("-" * 95)
    print(f"{'TOTAL':<13}{'':>10}{total:>9}")
    if args.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "belowcut.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"\nwrote {out}/belowcut.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

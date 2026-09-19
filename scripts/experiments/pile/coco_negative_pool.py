#!/usr/bin/env python3
"""Does the negative pool have to be SHARED across classes? No, and it costs a lot.

`band_candidates` admits an image to the shared pool only when it holds **no
class in C**:

    if not by_name:
        # Only a true negative for every class in C may join the shared pool.

That rule is stricter than the thing it protects. A negative for `book` need only
lack `book`; whether it holds a `person` is irrelevant to scoring `book`. Sharing
*within* a class is what the construction actually needs -- `book@small`,
`book@medium` and `book@large` scored against identical negatives make
small-vs-large a paired contrast rather than two datasets of different difficulty
-- and per-class pools preserve that exactly. Sharing *across* classes was never
argued for.

Two costs, and the second is the serious one:

* **Supply.** "Holds none of C" shrinks fast as C grows (#3604 measured four
  classes evicting 34% of it on VG), so the pool -- not per-class positives --
  is what caps the class list. Per-class it does not bind at all.
* **Representativeness.** A shared-pool negative holds no common object by
  construction: **0%** of it co-occurs with another class in C, where a
  representative negative set for the same class would be ~86%. The pool is
  therefore a sample of unusually *barren* images, and a detector scored against
  it is being asked an easier question than the app poses -- and can learn that
  an image containing a B cannot contain an A.

#3667 already found the shortcut half of this and fixed it partially, by adding
the other classes' positives to each class's negatives. This measures what is
left: the designated pool is still drawn barren, and it is the large majority of
what a detector faces.

**One pool, filtered per class** is the whole change, and
``evaluable_categories`` already expresses it per media -- draw one set *P* and
let class *A* use ``{p in P : p holds no A}``. |P| is then set by the class
present in the most images.

    python coco_negative_pool.py --annotations <dir holding instances_*2017.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pile_config as pc  # noqa: E402
from pilebuild.scale_core import band_for  # noqa: E402

BANDS = ("small", "medium", "large")
#: Pool sizes to report. The interesting one is whichever first serves every class.
CANDIDATE_SIZES = (16_058, 20_000, 24_000, 30_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
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

    everything = sorted(dims)
    roster = [c for c in sorted({c for c, _ in cell}) if min(cell[(c, b)] for b in BANDS) >= pc.SCALE_N_POS]
    need = pc.SCALE_N_NEG + pc.SCALE_N_NEG_SPARE
    shared = set(everything) - set().union(*(holders[c] for c in roster))

    print(f"|C| = {len(roster)}   COCO images = {len(everything):,}   pool + spares needed = {need:,}\n")
    print(f"SHARED pool ('holds none of C'): {len(shared):,} ({len(shared) / need:.1f}x)")
    print("  0.0% of it co-occurs with another class in C, by construction.\n")

    in_c: collections.Counter = collections.Counter()
    for c in roster:
        for i in holders[c]:
            in_c[i] += 1
    co = []
    print("PER-CLASS negatives ('holds no A'):")
    print(f"{'class':<16}{'candidates':>12}{'x needed':>10}{'co-occurring':>14}")
    print("-" * 54)
    rows = []
    for c in roster:
        cand = len(everything) - len(holders[c])
        other = sum(1 for i in everything if i not in holders[c] and in_c[i] > 0)
        rows.append((c, cand, 100 * other / cand))
        co.append(100 * other / cand)
    rows.sort(key=lambda r: r[1])
    for c, cand, pct in rows[:3] + rows[-2:]:
        print(f"{c:<16}{cand:>12,}{cand / need:>9.0f}x{pct:>13.0f}%")
    print(f"\nmean co-occurrence of a representative negative set: {statistics.mean(co):.0f}%")
    print("(the shared pool's is 0% -- it samples unusually barren images)\n")

    # One pool, filtered per class. |P| is set by the most common class in C.
    worst = max(roster, key=lambda c: len(holders[c]))
    prev = len(holders[worst]) / len(everything)
    print(f"ONE POOL, FILTERED PER CLASS. Most common class: `{worst}` in {100 * prev:.0f}% of COCO,")
    print(f"so |P| >= {need:,} / {1 - prev:.3f} = {need / (1 - prev):,.0f}.\n")
    for n in CANDIDATE_SIZES:
        rng = random.Random(args.seed)
        pool = rng.sample(everything, min(n, len(everything)))
        valid = {c: sum(1 for i in pool if i not in holders[c]) for c in roster}
        short = [c for c in roster if valid[c] < need]
        thin = min(valid, key=lambda c: valid[c])
        print(
            f"|P| = {n:>6,}: thinnest `{thin}` at {valid[thin]:>6,}; "
            f"short of {need:,}: {len(short)}" + (f" ({', '.join(short)})" if 0 < len(short) <= 3 else "")
        )


if __name__ == "__main__":
    main()

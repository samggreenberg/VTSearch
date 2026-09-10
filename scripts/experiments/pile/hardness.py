#!/usr/bin/env python3
"""Was class hardness predictable BEFORE the human pass, and by what? (#3720)

Three candidate predictors of the precision a class actually turned in, scored
against the ten classes now measured:

* **anchored precision** -- what the detector scored on the COCO-anchored half at
  that class's own cut. Available before a single human click.
* **sibling confusion** -- of the detector's anchored false positives for C, the
  share that are a true instance of some OTHER class in C. This is the
  near-miss reading: a class is hard when the things that look like it are
  common, whether or not anyone wrote them down.
* **exclusion count** -- how many "not X" clauses the rule text carries. My
  claim, and the one under suspicion of being post hoc.

Spearman rank correlation, computed by hand to avoid a scipy dependency. Ten
points is few, so the honest use is to separate "predicts" from "does not",
not to rank the survivors finely.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

OBSERVED = {  # precision on the finished/партial human slates
    "dog": 0.808,
    "fork": 0.642,
    "boat": 0.461,
    "bicycle": 0.459,
    "sink": 0.458,
    "vase": 0.294,
    "fire hydrant": 0.283,
    "bus": 0.272,
    "stop sign": 0.117,
    "bench": 0.096,
}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> int:
    sys.path.insert(0, "scripts/experiments/pile")
    import pile_config as pc  # noqa: PLC0415

    cuts = {
        c: v["cut"]
        for c, v in json.loads(Path("/expscratch/sgreenberg/vlm-3720/slates/slates.json").read_text()).items()
    }
    rows = [
        json.loads(x)
        for x in Path("/expscratch/sgreenberg/vlm-3720/owl_big.jsonl").read_text().splitlines()
        if x.strip()
    ]
    rows = [r for r in rows if "dets" in r]

    anch_p, sibling = {}, {}
    for c in OBSERVED:
        t = cuts.get(c, 0.1)
        tp = fp = 0
        fp_is_sibling = 0
        for r in rows:
            truth = set(r["truth"])
            if not any(d["cls"] == c and d["score"] >= t for d in r["dets"]):
                continue
            if c in truth:
                tp += 1
            else:
                fp += 1
                # is the image a true instance of some OTHER class in C?
                if truth:
                    fp_is_sibling += 1
        anch_p[c] = tp / (tp + fp) if tp + fp else float("nan")
        sibling[c] = fp_is_sibling / fp if fp else float("nan")

    excl = {}
    for c in OBSERVED:
        test = getattr(pc.SCALE_CLASS_RULES.get(c), "test", "") or ""
        bad = test.split("Bad:", 1)[1] if "Bad:" in test else ""
        excl[c] = len(re.findall(r",|--|\band\b", bad)) if bad else 0

    print(f"{'class':<14}{'observed':>10}{'anchored':>10}{'sibling':>9}{'excl':>6}")
    print("-" * 49)
    for c in sorted(OBSERVED, key=lambda k: -OBSERVED[k]):
        print(f"{c:<14}{OBSERVED[c]:>10.3f}{anch_p[c]:>10.3f}{sibling[c]:>9.2f}{excl[c]:>6}")
    print("-" * 49)
    obs = [OBSERVED[c] for c in OBSERVED]
    print("\nSpearman rank correlation with observed precision (n=10):")
    print(f"  anchored precision : {spearman([anch_p[c] for c in OBSERVED], obs):+.2f}")
    print(f"  sibling confusion  : {spearman([sibling[c] for c in OBSERVED], obs):+.2f}")
    print(f"  exclusion count    : {spearman([excl[c] for c in OBSERVED], obs):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

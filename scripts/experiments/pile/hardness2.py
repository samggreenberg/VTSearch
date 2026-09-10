#!/usr/bin/env python3
"""Test the near-miss reading properly, after a broken first attempt (#3720).

The first version defined "sibling confusion" as "the false-positive image holds
any class in C", which is true of nearly every anchored image and so returned
1.00 for all ten classes -- a constant, not a measurement.

Two better operationalisations, both computable before any human click:

* **AUC** -- how separable the detector's scores are between images that hold the
  class and images that do not. This is the near-miss reading stated directly: a
  class is hard when the things that score like it are not it, whatever the
  reason and whether or not anyone wrote the confusions down.
* **top-confusion share** -- among false positives for C, the largest share
  attributable to any single other class in C. High means one specific
  neighbour; low means diffuse, which is the "blobs and buildings" failure
  rather than a class boundary.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

OBSERVED = {
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
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def auc(pos, neg):
    """P(score of a positive > score of a negative), ties counted half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum = 0.0
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if allv[k][1] == 1:
                rank_sum += avg
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    sys.path.insert(0, "scripts/experiments/pile")
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

    aucs, topshare, topname = {}, {}, {}
    for c in OBSERVED:
        pos, neg = [], []
        conf = Counter()
        nfp = 0
        for r in rows:
            truth = set(r["truth"])
            s = max((d["score"] for d in r["dets"] if d["cls"] == c), default=0.0)
            (pos if c in truth else neg).append(s)
            if c not in truth and s >= cuts.get(c, 0.1):
                nfp += 1
                for o in truth:
                    conf[o] += 1
        aucs[c] = auc(pos, neg)
        if nfp and conf:
            k, v = conf.most_common(1)[0]
            topshare[c], topname[c] = v / nfp, k
        else:
            topshare[c], topname[c] = float("nan"), "-"

    print(f"{'class':<14}{'observed':>10}{'AUC':>8}{'topconf':>9}  most-confused-with")
    print("-" * 62)
    for c in sorted(OBSERVED, key=lambda k: -OBSERVED[k]):
        print(f"{c:<14}{OBSERVED[c]:>10.3f}{aucs[c]:>8.3f}{topshare[c]:>9.2f}  {topname[c]}")
    print("-" * 62)
    obs = [OBSERVED[c] for c in OBSERVED]
    print("\nSpearman vs observed precision (n=10):")
    print(f"  AUC on the anchored half : {spearman([aucs[c] for c in OBSERVED], obs):+.2f}")
    print(f"  top-confusion share      : {spearman([topshare[c] for c in OBSERVED], obs):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

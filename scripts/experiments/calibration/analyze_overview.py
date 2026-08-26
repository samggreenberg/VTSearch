"""What is VTSearch good at, what is it bad at, and why — descriptively.

No arms, no winners. Three modes a real user could pick (`siglip` whole-image,
`siglip2_l` whole-image, `dinov3_patch` region voting) run many times under
shipped defaults, characterised rather than ranked.

The "why" comes from a decomposition the harness already emits per step:

    cost  =  oracle_cost  +  regret

`oracle_cost` is the best any threshold could achieve **on that run's own
ranking**, so it is the ranking's own limit. `regret` is what the shipped cut
rule gives away on top of it. A cell that is expensive because `oracle_cost` is
high needs a better ranking — different embedder, more votes, region geometry.
A cell that is expensive because `regret` is high has a ranking that already
knows the answer and a cut rule that cannot find it. Those two want completely
different fixes, and averaging cost alone cannot tell them apart.

Reported as distributions, not means: the tail is the product problem. A mode
whose median run is fine and whose worst decile never leaves the floor is a mode
that fails some users completely, and a mean hides exactly that.

Usage::

    python analyze_overview.py --exp /expscratch/$USER/scale-3156
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

STEPS = (20, 50, 150)


def q(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (sum(xs) / n if n else float("nan")), float("nan")
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def f(v: float) -> str:
    return "n/a" if v != v else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156")
    ap.add_argument("--floor", type=float, default=0.9, help="cost at/above this counts as 'never got going'")
    args = ap.parse_args()

    cells = Path(args.exp) / "results" / "cells"
    rows = []
    for path in sorted(glob.glob(str(cells / "task_*.csv"))):
        if "__" in Path(path).name:
            continue
        with open(path, newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))
    print(f"{len(rows)} rows from {len({r['category'] for r in rows})} cells\n")

    def mode(r):
        return f"{r.get('embedder', '')}/{r.get('style', '')}".strip("/")

    # last row at or before each step, per run
    snap: dict[tuple, dict] = {}
    for r in rows:
        try:
            t = int(r["t"])
        except (KeyError, ValueError):
            continue
        for s in STEPS:
            if t > s:
                continue
            k = (mode(r), r["category"], r["seed"], s)
            if k not in snap or int(snap[k]["t"]) < t:
                snap[k] = r

    modes = sorted({mode(r) for r in rows})

    print("=== what a run looks like: cost distribution (lower is better) ===")
    print(f"{'mode':<26}{'votes':>6}{'p10':>7}{'median':>8}{'p90':>7}{'worst':>7}{'n':>5}")
    print("-" * 66)
    for m in modes:
        for s in STEPS:
            xs = [float(v["cost"]) for k, v in snap.items() if k[0] == m and k[3] == s and v.get("cost")]
            print(
                f"{m:<26}{s:>6}{f(q(xs, 0.1)):>7}{f(q(xs, 0.5)):>8}{f(q(xs, 0.9)):>7}{f(max(xs)) if xs else 'n/a':>7}{len(xs):>5}"
            )
        print()

    print(f"=== runs that never got going (cost >= {args.floor} at 150 votes) ===")
    print(f"{'mode':<26}{'stuck':>7}{'of':>5}{'rate':>8}   worst cells")
    print("-" * 78)
    stuck_cells: dict[str, list[str]] = defaultdict(list)
    for m in modes:
        vs = [(k[1], float(v["cost"])) for k, v in snap.items() if k[0] == m and k[3] == 150 and v.get("cost")]
        stuck = [c for c, x in vs if x >= args.floor]
        for c in stuck:
            stuck_cells[m].append(c)
        top = sorted({c for c in stuck})[:3]
        rate = len(stuck) / len(vs) if vs else float("nan")
        print(f"{m:<26}{len(stuck):>7}{len(vs):>5}{f(rate):>8}   {', '.join(top) if top else '-'}")

    print()
    print("=== why: is the ranking the limit, or the cut rule? ===")
    print("cost = oracle_cost (the ranking's own limit) + regret (what the cut gives away)")
    print(f"{'mode':<26}{'cost':>8}{'oracle':>8}{'regret':>8}{'regret share':>14}")
    print("-" * 66)
    for m in modes:
        c, o, g = [], [], []
        for k, v in snap.items():
            if k[0] != m or k[3] != 150:
                continue
            try:
                c.append(float(v["cost"]))
                o.append(float(v["oracle_cost"]))
                g.append(float(v["regret"]))
            except (KeyError, ValueError, TypeError):
                continue
        mc, oc, gc = mean_se(c)[0], mean_se(o)[0], mean_se(g)[0]
        share = gc / mc if mc else float("nan")
        print(f"{m:<26}{f(mc):>8}{f(oc):>8}{f(gc):>8}{f(share):>14}")

    print()
    print("=== the same split, by target size ===")
    print(f"{'mode':<26}{'band':<8}{'cost':>7}{'oracle':>8}{'regret':>8}")
    print("-" * 60)
    for m in modes:
        for b in ("small", "medium", "large"):
            c, o, g = [], [], []
            for k, v in snap.items():
                if k[0] != m or k[3] != 150 or not k[1].endswith("@" + b):
                    continue
                try:
                    c.append(float(v["cost"]))
                    o.append(float(v["oracle_cost"]))
                    g.append(float(v["regret"]))
                except (KeyError, ValueError, TypeError):
                    continue
            print(f"{m:<26}{b:<8}{f(mean_se(c)[0]):>7}{f(mean_se(o)[0]):>8}{f(mean_se(g)[0]):>8}")
        print()

    print("=== hard for everyone, or hard for one mode? ===")
    per_cell: dict[str, dict[str, float]] = defaultdict(dict)
    for k, v in snap.items():
        if k[3] != 150 or not v.get("cost"):
            continue
        per_cell[k[1]].setdefault(k[0], 0.0)
        per_cell[k[1]][k[0]] = max(per_cell[k[1]][k[0]], float(v["cost"]))
    universal = [c for c, d in per_cell.items() if len(d) == len(modes) and min(d.values()) >= args.floor]
    print(f"cells at/above {args.floor} for EVERY mode (intrinsically hard): {len(universal)}")
    for c in sorted(universal)[:10]:
        print(f"   {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

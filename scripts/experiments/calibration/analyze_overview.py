"""What is VTSearch good at, what is it bad at, and why — descriptively.

No arms, no winners. Three modes a real user could pick (`siglip` whole-image,
`siglip2_l` whole-image, `dinov3_patch` region voting) run many times under
shipped defaults, characterised rather than ranked.

The "why" comes from a decomposition the harness already emits per step:

    cost  =  oracle_cost  +  regret
    regret  =  rule_inefficiency  +  calibration_shift

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

**Depth changes which questions are answerable.** At three seeds a cell's stuck
runs are an anecdote; at sixty they are a *rate*, so "is `bicycle@small`
reliably bad, or was one seed unlucky?" becomes a measurement. Anything here
that reduces over seeds therefore reports a rate with its spread, never a
single draw — and `--min-seeds` refuses to print a per-cell rate that too few
runs support rather than quoting a confident-looking fraction of three.

Usage::

    python analyze_overview.py --exp /expscratch/$USER/scale-3156-final
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
DEEP = 150
BANDS = ("small", "medium", "large")


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


def pm(xs: list[float]) -> str:
    """mean ±SE, two significant digits — the only honest form for a difference."""
    m, se = mean_se(xs)
    if m != m:
        return "n/a"
    return f"{m:.2f}" if se != se else f"{m:.2f}±{se:.2f}"


def fnum(r: dict, key: str) -> float:
    try:
        return float(r[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156-final")
    ap.add_argument("--floor", type=float, default=0.9, help="cost at/above this counts as 'never got going'")
    ap.add_argument("--expect", type=int, default=0, help="expected cell count; 0 = infer from the grid")
    ap.add_argument("--min-seeds", type=int, default=10, help="per-cell rates need at least this many runs")
    ap.add_argument("--top", type=int, default=15, help="rows in the per-cell listings")
    args = ap.parse_args()

    cells = Path(args.exp) / "results" / "cells"
    paths = [p for p in sorted(glob.glob(str(cells / "task_*.csv"))) if "__" not in Path(p).name]
    rows: list[dict] = []
    unreadable: list[str] = []
    for path in paths:
        try:
            with open(path, newline="") as fh:
                got = list(csv.DictReader(fh))
        except (OSError, csv.Error):
            unreadable.append(Path(path).name)
            continue
        if not got:
            unreadable.append(Path(path).name)
            continue
        rows.extend(got)

    def mode(r: dict) -> str:
        return f"{r.get('embedder', '')}/{r.get('style', '')}".strip("/")

    modes = sorted({mode(r) for r in rows})
    cats = sorted({r["category"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    expected = args.expect or len(cats) * len(modes) * len(seeds)

    # --- what was dropped ---------------------------------------------------
    # A silently-short analysis reads exactly like a complete one, which is how
    # a disk incident becomes a wrong verdict.  State the numbers first.
    print("=== coverage: what this analysis is actually over ===")
    print(f"cell files found      {len(paths)}")
    print(f"expected              {expected}   ({len(cats)} cells x {len(modes)} modes x {len(seeds)} seeds)")
    print(f"unreadable/empty      {len(unreadable)}" + (f"   e.g. {', '.join(unreadable[:5])}" if unreadable else ""))
    missing = expected - (len(paths) - len(unreadable))
    print(f"MISSING               {missing}" + ("   <- report this in the writeup" if missing > 0 else ""))
    print(f"rows                  {len(rows)}\n")
    if not rows:
        print("no rows; nothing to analyse")
        return 1

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

    deep = {k: v for k, v in snap.items() if k[3] == DEEP}

    print("=== what a run looks like: cost distribution (lower is better) ===")
    print(f"{'mode':<26}{'votes':>6}{'p10':>7}{'median':>8}{'p90':>7}{'worst':>7}{'n':>6}")
    print("-" * 67)
    for m in modes:
        for s in STEPS:
            xs = [fnum(v, "cost") for k, v in snap.items() if k[0] == m and k[3] == s]
            xs = [x for x in xs if x == x]
            print(
                f"{m:<26}{s:>6}{f(q(xs, 0.1)):>7}{f(q(xs, 0.5)):>8}"
                f"{f(q(xs, 0.9)):>7}{f(max(xs)) if xs else 'n/a':>7}{len(xs):>6}"
            )
        print()

    print(f"=== runs that never got going (cost >= {args.floor} at {DEEP} votes) ===")
    print(f"{'mode':<26}{'stuck':>7}{'of':>6}{'rate':>8}")
    print("-" * 47)
    for m in modes:
        xs = [fnum(v, "cost") for k, v in deep.items() if k[0] == m]
        xs = [x for x in xs if x == x]
        stuck = [x for x in xs if x >= args.floor]
        print(f"{m:<26}{len(stuck):>7}{len(xs):>6}{f(len(stuck) / len(xs)) if xs else 'n/a':>8}")

    # --- the profile of a stuck run -----------------------------------------
    print()
    print(f"=== what distinguishes a stuck run (>= {args.floor}) from a healthy one ===")
    metrics = ("n_good", "n_bad", "average_precision", "auroc", "oracle_cost", "regret")
    bad = [v for k, v in deep.items() if fnum(v, "cost") >= args.floor]
    good = [v for k, v in deep.items() if fnum(v, "cost") < args.floor]
    print(f"{'metric':<22}{'stuck':>14}{'healthy':>14}   n={len(bad)} vs {len(good)}")
    print("-" * 62)
    for met in metrics:
        b = [x for x in (fnum(v, met) for v in bad) if x == x]
        g = [x for x in (fnum(v, met) for v in good) if x == x]
        print(f"{met:<22}{pm(b):>14}{pm(g):>14}")

    # Which Autopilot phase a stuck run died in decides which knob can help it:
    # `good` never escaped the text sort, `hard` starved under the learned one.
    print()
    print("=== the two failure modes: which phase was the stuck run still in? ===")
    ph_bad: dict[str, int] = defaultdict(int)
    ph_good: dict[str, int] = defaultdict(int)
    for v in bad:
        ph_bad[v.get("phase", "?")] += 1
    for v in good:
        ph_good[v.get("phase", "?")] += 1
    print(f"{'phase':<12}{'stuck':>8}{'healthy':>9}   what it means")
    print("-" * 72)
    meaning = {
        "good": "never found GOOD_TARGET positives — still on the text sort",
        "bad": "still collecting the initial negatives",
        "hard": "escaped seeding; the learned selector starved it",
        "new": "exploring the coverage atlas",
        "done": "ran to completion",
        "exhausted": "nothing left to label",
    }
    for p in sorted(set(ph_bad) | set(ph_good), key=lambda x: -ph_bad.get(x, 0)):
        print(f"{p:<12}{ph_bad.get(p, 0):>8}{ph_good.get(p, 0):>9}   {meaning.get(p, '')}")

    print()
    print("=== why: is the ranking the limit, or the cut rule? ===")
    print("cost = oracle_cost (the ranking's own limit) + regret (what the cut gives away)")
    print(f"{'mode':<26}{'cost':>13}{'oracle':>13}{'regret':>13}{'regret share':>14}")
    print("-" * 79)
    for m in modes:
        sel = [v for k, v in deep.items() if k[0] == m]
        c = [x for x in (fnum(v, "cost") for v in sel) if x == x]
        o = [x for x in (fnum(v, "oracle_cost") for v in sel) if x == x]
        g = [x for x in (fnum(v, "regret") for v in sel) if x == x]
        mc = mean_se(c)[0]
        share = mean_se(g)[0] / mc if mc else float("nan")
        print(f"{m:<26}{pm(c):>13}{pm(o):>13}{pm(g):>13}{f(share):>14}")

    print()
    print("=== and inside regret: a bad rule, or a shifted calibration? ===")
    print(f"{'mode':<26}{'regret':>13}{'rule_ineff':>13}{'cal_shift':>13}")
    print("-" * 66)
    for m in modes:
        sel = [v for k, v in deep.items() if k[0] == m]
        cols = ("regret", "rule_inefficiency", "calibration_shift")
        vals = [[x for x in (fnum(v, c) for v in sel) if x == x] for c in cols]
        print(f"{m:<26}" + "".join(f"{pm(v):>13}" for v in vals))

    print()
    print("=== the same split, by target size ===")
    print(f"{'mode':<26}{'band':<8}{'cost':>13}{'oracle':>13}{'regret':>13}{'stuck':>8}")
    print("-" * 82)
    for m in modes:
        for b in BANDS:
            sel = [v for k, v in deep.items() if k[0] == m and k[1].endswith("@" + b)]
            c = [x for x in (fnum(v, "cost") for v in sel) if x == x]
            o = [x for x in (fnum(v, "oracle_cost") for v in sel) if x == x]
            g = [x for x in (fnum(v, "regret") for v in sel) if x == x]
            rate = (sum(1 for x in c if x >= args.floor) / len(c)) if c else float("nan")
            print(f"{m:<26}{b:<8}{pm(c):>13}{pm(o):>13}{pm(g):>13}{f(rate):>8}")
        print()

    # --- per-cell reliability ------------------------------------------------
    # The question depth buys: is this cell reliably hard, or did one seed draw
    # badly?  A max over seeds cannot tell those apart (and saturates as seeds
    # grow); a rate over seeds is exactly the distinction.
    print(f"=== reliably hard, or unlucky? per-cell stuck rate at {DEEP} votes ===")
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for k, v in deep.items():
        x = fnum(v, "cost")
        if x == x:
            per[(k[1], k[0])].append(x)
    scored = [
        (cat, m, sum(1 for x in xs if x >= args.floor) / len(xs), q(xs, 0.5), len(xs))
        for (cat, m), xs in per.items()
        if len(xs) >= args.min_seeds
    ]
    skipped = len(per) - len(scored)
    if skipped:
        print(f"({skipped} cell x mode combinations had < {args.min_seeds} runs and are not rated)")
    for cat, m, rate, med, n in sorted(scored, key=lambda r: -r[2])[: args.top]:
        bar = "#" * int(round(rate * 20))
        print(f"{cat:<20}{m:<26}rate {rate:>5.2f}  median {med:>5.2f}  n={n:<4} {bar}")

    print()
    print("=== hard for everyone, or hard for one mode? ===")
    med_by: dict[str, dict[str, float]] = defaultdict(dict)
    for (cat, m), xs in per.items():
        if len(xs) >= args.min_seeds:
            med_by[cat][m] = q(xs, 0.5)
    universal = [c for c, d in med_by.items() if len(d) == len(modes) and min(d.values()) >= args.floor]
    mode_specific = [
        c
        for c, d in med_by.items()
        if len(d) == len(modes) and max(d.values()) >= args.floor and min(d.values()) < args.floor
    ]
    print(f"median cost >= {args.floor} for EVERY mode (intrinsically hard): {len(universal)}")
    for c in sorted(universal)[: args.top]:
        print(f"   {c}   " + "  ".join(f"{m.split('/')[0]}={med_by[c][m]:.2f}" for m in modes))
    print(f"\nhard for some mode but not others (a mode choice would fix it): {len(mode_specific)}")
    for c in sorted(mode_specific)[: args.top]:
        print(f"   {c}   " + "  ".join(f"{m.split('/')[0]}={med_by[c][m]:.2f}" for m in modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

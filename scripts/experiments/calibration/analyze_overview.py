"""What is VTSearch good at, what is it bad at, and why — descriptively.

No arms, no winners. Five columns, run many times under shipped defaults and
characterised rather than ranked: `siglip` whole-image (the shipped default),
`siglip2_l` whole-image (the premium encoder), `clip` whole-image (a second
lineage), `clip_l` whole-image, and `siglip+dinov3_patch` region voting.

**Four of the five are modes a user could pick; `clip_l` is not.** It is
`eval_only` (#3292) and is not offered in the app, so it belongs in a column
that says "here is what a bigger CLIP does" and never in a sentence of the form
"users should choose X". It is here because its 768-d output matches `siglip`'s
exactly, which is what stops a SigLIP-vs-CLIP difference from being read as
"CLIP's vectors are narrower".

The region mode is a **pair** (#3276): SigLIP embeds the typed query and ranks
the opening, DINOv3 does the learning. DINOv3 has no text tower, so bare
`dinov3_patch` cannot open on a text sort at all -- it falls back to three
random known-goods, which would put a seeding difference inside the voting-mode
comparison this file draws. Every column now opens the same way and differs
only in the space the detector learns in.

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
import math
import os
from collections import defaultdict
from pathlib import Path

from _cells_paths import main_frame_files

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


def click_zero_section(path: str, rows: list[dict], mode, modes: list[str], mw: int, floor: float) -> None:
    """What the clicking bought over typing the query and stopping.

    Click 0 is not a zero: it is the whole product's cheap path -- type a query,
    read the ranked haystack under the same cut rule -- and it costs nothing.
    Every curve here is therefore only worth the votes it spends once it has
    moved past it, and a mode that never gets there is a mode whose clicking is
    ceremony. This is the one comparison in the report a user would make without
    being asked, so it is the one the analyzer must not leave to the figures.

    Reported per mode as a level, and per cell as a CROSSING: the median cell's
    first click at which the mean cost is at or below its own text sort, plus
    how many cells never get there. A level alone hides the cells that start
    ahead and stay ahead.
    """
    import csv as _csv
    from collections import defaultdict as _dd

    base: dict[tuple[str, str], list[float]] = _dd(list)
    try:
        with open(path, newline="") as fh:
            for r in _csv.DictReader(fh):
                if r.get("supports_text") not in (None, "", "1"):
                    continue
                try:
                    base[(r["embedder"], r["category"])].append(float(r["text_cost"]))
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        print(f"\n(no zero-click baseline at {path})")
        return
    if not base:
        print(f"\n(zero-click baseline at {path} carried no usable rows)")
        return
    anchor = {k: sum(v) / len(v) for k, v in base.items()}

    # (mode, category, t) -> costs, so a crossing is read off the same mean the
    # level is.
    by: dict[tuple[str, str, int], list[float]] = _dd(list)
    for r in rows:
        try:
            t = int(r["t"])
        except (KeyError, ValueError, TypeError):
            continue
        c = fnum(r, "cost")
        if c == c:
            by[(mode(r), r.get("category", ""), t)].append(c)

    print()
    print("=== what the clicking bought over the free text sort ===")
    print("Click 0 is the typed query alone, cut the same way and costing nothing.")
    print(f"{'mode':<{mw}}{'text sort':>10}{'@20':>8}{'@150':>8}{'crossing':>10}{'never':>8}")
    print("-" * (mw + 44))
    for m in modes:
        emb = m.split("/")[0]
        cats = sorted({c for (mm, c, _t) in by if mm == m})
        cats = [c for c in cats if (emb, c) in anchor]
        if not cats:
            continue

        def lvl(t: int, cats=cats, m=m) -> float:
            xs = [x for c in cats for x in by.get((m, c, t), [])]
            return sum(xs) / len(xs) if xs else float("nan")

        crossings, never = [], 0
        for c in cats:
            a = anchor[(emb, c)]
            ts = sorted({t for (mm, cc, t) in by if mm == m and cc == c and t >= 1})
            hit = None
            for t in ts:
                xs = by.get((m, c, t), [])
                if xs and sum(xs) / len(xs) <= a:
                    hit = t
                    break
            if hit is None:
                never += 1
            else:
                crossings.append(hit)
        med = q(crossings, 0.5) if crossings else float("nan")
        anchor_lvl = sum(anchor[(emb, c)] for c in cats) / len(cats)
        cross = f"{med:.0f}" if crossings else "-"
        print(f"{m:<{mw}}{anchor_lvl:>10.2f}{lvl(20):>8.2f}{lvl(150):>8.2f}{cross:>10}{never:>4}/{len(cats):<3}")
    print()
    print("crossing = the median cell's first click whose mean cost is at or below its own")
    print("text sort; 'never' counts cells that do not get there within the horizon.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156-final")
    ap.add_argument("--floor", type=float, default=0.9, help="cost at/above this counts as 'never got going'")
    ap.add_argument("--expect", type=int, default=0, help="expected cell count; 0 = infer from the grid")
    ap.add_argument("--min-seeds", type=int, default=10, help="per-cell rates need at least this many runs")
    ap.add_argument("--top", type=int, default=15, help="rows in the per-cell listings")
    ap.add_argument(
        "--baseline",
        default=None,
        help="text_baseline.py CSV: the zero-click text sort. Without it the report cannot say "
        "whether the clicking beat typing the query, which is the first thing a reader asks.",
    )
    args = ap.parse_args()

    cells = Path(args.exp) / "results" / "cells"
    paths = main_frame_files(cells)
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

    def mode_w(names, floor: int = 26) -> int:
        """Width of the mode column, from the widest name actually present.

        A constant 26 was wide enough for `dinov3_patch/max_patch` and is two
        short of `siglip+dinov3_patch/max_patch`, so the by-band table printed
        `...max_patchsmall` -- the band welded onto the arm. Tables are how this
        study is read, so the width follows the data.
        """
        return max(floor, max((len(str(n)) for n in names), default=floor) + 2)

    modes = sorted({mode(r) for r in rows})
    MW = mode_w(modes)
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
    hdr = f"{'mode':<{MW}}{'votes':>6}{'p10':>7}{'median':>8}{'p90':>7}{'worst':>7}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        for s in STEPS:
            xs = [fnum(v, "cost") for k, v in snap.items() if k[0] == m and k[3] == s]
            xs = [x for x in xs if x == x]
            print(
                f"{m:<{MW}}{s:>6}{f(q(xs, 0.1)):>7}{f(q(xs, 0.5)):>8}"
                f"{f(q(xs, 0.9)):>7}{f(max(xs)) if xs else 'n/a':>7}{len(xs):>6}"
            )
        print()

    print(f"=== runs that never got going (cost >= {args.floor} at {DEEP} votes) ===")
    hdr = f"{'mode':<{MW}}{'stuck':>7}{'of':>6}{'rate':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        xs = [fnum(v, "cost") for k, v in deep.items() if k[0] == m]
        xs = [x for x in xs if x == x]
        stuck = [x for x in xs if x >= args.floor]
        print(f"{m:<{MW}}{len(stuck):>7}{len(xs):>6}{f(len(stuck) / len(xs)) if xs else 'n/a':>8}")

    # --- the profile of a stuck run -----------------------------------------
    print()
    print(f"=== what distinguishes a stuck run (>= {args.floor}) from a healthy one ===")
    metrics = ("n_good", "n_bad", "average_precision", "auroc", "oracle_cost", "regret")
    bad = [v for k, v in deep.items() if fnum(v, "cost") >= args.floor]
    good = [v for k, v in deep.items() if fnum(v, "cost") < args.floor]
    hdr = f"{'metric':<22}{'stuck':>14}{'healthy':>14}   n={len(bad)} vs {len(good)}"
    print(hdr)
    print("-" * len(hdr))
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
    hdr = f"{'phase':<12}{'stuck':>8}{'healthy':>9}   what it means"
    print(hdr)
    print("-" * len(hdr))
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
    hdr = f"{'mode':<{MW}}{'cost':>13}{'oracle':>13}{'regret':>13}{'regret share':>14}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        sel = [v for k, v in deep.items() if k[0] == m]
        c = [x for x in (fnum(v, "cost") for v in sel) if x == x]
        o = [x for x in (fnum(v, "oracle_cost") for v in sel) if x == x]
        g = [x for x in (fnum(v, "regret") for v in sel) if x == x]
        mc = mean_se(c)[0]
        share = mean_se(g)[0] / mc if mc else float("nan")
        print(f"{m:<{MW}}{pm(c):>13}{pm(o):>13}{pm(g):>13}{f(share):>14}")

    print()
    print("=== and inside regret: a bad rule, or a shifted calibration? ===")
    hdr = f"{'mode':<{MW}}{'regret':>13}{'rule_ineff':>13}{'cal_shift':>13}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        sel = [v for k, v in deep.items() if k[0] == m]
        cols = ("regret", "rule_inefficiency", "calibration_shift")
        vals = [[x for x in (fnum(v, c) for v in sel) if x == x] for c in cols]
        print(f"{m:<{MW}}" + "".join(f"{pm(v):>13}" for v in vals))

    print()
    print("=== the same split, by target size ===")
    hdr = f"{'mode':<{MW}}{'band':<8}{'cost':>13}{'oracle':>13}{'regret':>13}{'stuck':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        for b in BANDS:
            sel = [v for k, v in deep.items() if k[0] == m and k[1].endswith("@" + b)]
            c = [x for x in (fnum(v, "cost") for v in sel) if x == x]
            o = [x for x in (fnum(v, "oracle_cost") for v in sel) if x == x]
            g = [x for x in (fnum(v, "regret") for v in sel) if x == x]
            rate = (sum(1 for x in c if x >= args.floor) / len(c)) if c else float("nan")
            print(f"{m:<{MW}}{b:<8}{pm(c):>13}{pm(o):>13}{pm(g):>13}{f(rate):>8}")
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
        print(f"{cat:<20}{m:<{MW}}rate {rate:>5.2f}  median {med:>5.2f}  n={n:<4} {bar}")

    if args.baseline:
        click_zero_section(args.baseline, rows, mode, modes, MW, args.floor)

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

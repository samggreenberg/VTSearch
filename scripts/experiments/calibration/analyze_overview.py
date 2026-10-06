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

**The headline metric is the decision metric** (#4584): the objective, F-beta
of the withheld half above the threshold the app holds at the run's own beta,
on a run that drew its line at a balance; cost on one that did not (the
Inclusion arm, a Cost-era run).  Higher is better for the objective, lower for
cost, and every table below says which.  The "why" split stays on cost: it is
an identity of cost, and the frame carries no oracle for the objective.

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

import objective
from _cells_paths import main_frame_files

from study_paths import require_study_dir

STEPS = (20, 50, 150)
DEEP = 150
BANDS = ("small", "medium", "large")

#: Where a run counts as "never got going", per decision metric: cost at or
#: above 0.9 (FPR + FNR near a coin's), or the objective at or below 0.1 (it
#: found next to nothing).  ``--floor`` overrides either.
STUCK_FLOOR: dict[str, float] = {objective.COST: 0.9, objective.OBJECTIVE: 0.1}


class Metric:
    """The decision metric a run is read on: its value of a row, its direction, and its stuck test."""

    def __init__(self, name: str, floor: float | None) -> None:
        self.name = name
        self.lower_is_better = name == objective.COST
        self.floor = STUCK_FLOOR[name] if floor is None else floor

    def of(self, r: dict) -> float:
        return fnum(r, "cost") if self.name == objective.COST else objective.row_objective(r)

    def stuck(self, x: float) -> bool:
        return x >= self.floor if self.lower_is_better else x <= self.floor

    def at_least_as_good(self, x: float, anchor: float) -> bool:
        return x <= anchor if self.lower_is_better else x >= anchor

    @property
    def stuck_words(self) -> str:
        return f"{self.name} {'>=' if self.lower_is_better else '<='} {self.floor}"

    @property
    def direction(self) -> str:
        return "lower is better" if self.lower_is_better else "higher is better"


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


def click_zero_section(path: str, rows: list[dict], mode, modes: list[str], mw: int, metric: Metric) -> None:
    """What the clicking bought over typing the query and stopping.

    Click 0 is not a zero: it is the whole product's cheap path -- type a query,
    read the ranked haystack under the same cut rule -- and it costs nothing.
    Every curve here is therefore only worth the votes it spends once it has
    moved past it, and a mode that never gets there is a mode whose clicking is
    ceremony. This is the one comparison in the report a user would make without
    being asked, so it is the one the analyzer must not leave to the figures.

    Reported per mode as a level, and per cell as a CROSSING: the median cell's
    first click at which the mean of the decision metric is at least as good as
    its own text sort, plus how many cells never get there. A level alone hides
    the cells that start ahead and stay ahead.  The objective's anchor is the
    text sort's at the run's preset (``text_fbeta_b1`` at beta 1).
    """
    import csv as _csv
    from collections import defaultdict as _dd

    if metric.name == objective.COST:
        anchor_col = "text_cost"
    else:
        from vtscore.eval.voting_columns import beta_tag  # noqa: PLC0415

        betas = sorted({fnum(r, "beta") for r in rows if fnum(r, "beta") == fnum(r, "beta")})
        if len(betas) != 1:
            print(f"\n(no zero-click anchor: the runs drew their lines at {len(betas)} balances, not one)")
            return
        anchor_col = f"text_fbeta_{beta_tag(betas[0])}"

    base: dict[tuple[str, str], list[float]] = _dd(list)
    try:
        with open(path, newline="") as fh:
            for r in _csv.DictReader(fh):
                if r.get("supports_text") not in (None, "", "1"):
                    continue
                try:
                    base[(r["embedder"], r["category"])].append(float(r[anchor_col]))
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        print(f"\n(no zero-click baseline at {path})")
        return
    if not base:
        print(f"\n(zero-click baseline at {path} carried no usable `{anchor_col}` rows)")
        return
    anchor = {k: sum(v) / len(v) for k, v in base.items()}

    # (mode, category, t) -> values, so a crossing is read off the same mean the
    # level is.
    by: dict[tuple[str, str, int], list[float]] = _dd(list)
    for r in rows:
        try:
            t = int(r["t"])
        except (KeyError, ValueError, TypeError):
            continue
        c = metric.of(r)
        if c == c:
            by[(mode(r), r.get("category", ""), t)].append(c)

    print()
    print(f"=== what the clicking bought over the free text sort ({metric.name}, {metric.direction}) ===")
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
                if xs and metric.at_least_as_good(sum(xs) / len(xs), a):
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
    print(f"crossing = the median cell's first click whose mean {metric.name} is at least as good as")
    print("its own text sort; 'never' counts cells that do not get there within the horizon.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156-final")
    ap.add_argument(
        "--metric",
        choices=(objective.OBJECTIVE, objective.COST),
        default=None,
        help="decision metric (default: the objective, fbeta, when the runs carry a beta; else cost)",
    )
    ap.add_argument(
        "--floor",
        type=float,
        default=None,
        help="a run 'never got going' at cost >= this (default 0.9) or objective <= this (default 0.1)",
    )
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

    args.exp = str(require_study_dir(args.exp, "--exp"))

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
    name = args.metric or (objective.OBJECTIVE if objective.rows_carry_beta(rows) else objective.COST)
    metric = Metric(name, args.floor)
    print(f"decision metric       {metric.name} ({metric.direction}); stuck at {metric.stuck_words}\n")

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

    print(f"=== what a run looks like: {metric.name} distribution ({metric.direction}) ===")
    hdr = f"{'mode':<{MW}}{'votes':>6}{'p10':>7}{'median':>8}{'p90':>7}{'worst':>7}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        for s in STEPS:
            xs = [metric.of(v) for k, v in snap.items() if k[0] == m and k[3] == s]
            xs = [x for x in xs if x == x]
            worst = (max(xs) if metric.lower_is_better else min(xs)) if xs else float("nan")
            print(
                f"{m:<{MW}}{s:>6}{f(q(xs, 0.1)):>7}{f(q(xs, 0.5)):>8}"
                f"{f(q(xs, 0.9)):>7}{f(worst) if xs else 'n/a':>7}{len(xs):>6}"
            )
        print()

    print(f"=== runs that never got going ({metric.stuck_words} at {DEEP} votes) ===")
    hdr = f"{'mode':<{MW}}{'stuck':>7}{'of':>6}{'rate':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        xs = [metric.of(v) for k, v in deep.items() if k[0] == m]
        xs = [x for x in xs if x == x]
        stuck = [x for x in xs if metric.stuck(x)]
        print(f"{m:<{MW}}{len(stuck):>7}{len(xs):>6}{f(len(stuck) / len(xs)) if xs else 'n/a':>8}")

    # --- the profile of a stuck run -----------------------------------------
    print()
    print(f"=== what distinguishes a stuck run ({metric.stuck_words}) from a healthy one ===")
    metrics = ("n_good", "n_bad", "average_precision", "auroc", "cost", "oracle_cost", "regret")
    scored_deep = [(v, metric.of(v)) for v in deep.values()]
    bad = [v for v, x in scored_deep if x == x and metric.stuck(x)]
    good = [v for v, x in scored_deep if x == x and not metric.stuck(x)]
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
    print("=== why: is the ranking the limit, or the cut rule? (on cost, the diagnostic) ===")
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
    print(f"(stuck is the share at {metric.stuck_words}; the rest is the cost split)")
    print(hdr)
    print("-" * len(hdr))
    for m in modes:
        for b in BANDS:
            sel = [v for k, v in deep.items() if k[0] == m and k[1].endswith("@" + b)]
            c = [x for x in (fnum(v, "cost") for v in sel) if x == x]
            o = [x for x in (fnum(v, "oracle_cost") for v in sel) if x == x]
            g = [x for x in (fnum(v, "regret") for v in sel) if x == x]
            vals = [x for x in (metric.of(v) for v in sel) if x == x]
            rate = (sum(1 for x in vals if metric.stuck(x)) / len(vals)) if vals else float("nan")
            print(f"{m:<{MW}}{b:<8}{pm(c):>13}{pm(o):>13}{pm(g):>13}{f(rate):>8}")
        print()

    # --- per-cell reliability ------------------------------------------------
    # The question depth buys: is this cell reliably hard, or did one seed draw
    # badly?  A max over seeds cannot tell those apart (and saturates as seeds
    # grow); a rate over seeds is exactly the distinction.
    print(f"=== reliably hard, or unlucky? per-cell stuck rate at {DEEP} votes ===")
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for k, v in deep.items():
        x = metric.of(v)
        if x == x:
            per[(k[1], k[0])].append(x)
    scored = [
        (cat, m, sum(1 for x in xs if metric.stuck(x)) / len(xs), q(xs, 0.5), len(xs))
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
        click_zero_section(args.baseline, rows, mode, modes, MW, metric)

    print()
    print("=== hard for everyone, or hard for one mode? ===")
    med_by: dict[str, dict[str, float]] = defaultdict(dict)
    for (cat, m), xs in per.items():
        if len(xs) >= args.min_seeds:
            med_by[cat][m] = q(xs, 0.5)
    universal = [c for c, d in med_by.items() if len(d) == len(modes) and all(metric.stuck(x) for x in d.values())]
    mode_specific = [
        c
        for c, d in med_by.items()
        if len(d) == len(modes)
        and any(metric.stuck(x) for x in d.values())
        and not all(metric.stuck(x) for x in d.values())
    ]
    print(f"median {metric.stuck_words} for EVERY mode (intrinsically hard): {len(universal)}")
    for c in sorted(universal)[: args.top]:
        print(f"   {c}   " + "  ".join(f"{m.split('/')[0]}={med_by[c][m]:.2f}" for m in modes))
    print(f"\nhard for some mode but not others (a mode choice would fix it): {len(mode_specific)}")
    for c in sorted(mode_specific)[: args.top]:
        print(f"   {c}   " + "  ".join(f"{m.split('/')[0]}={med_by[c][m]:.2f}" for m in modes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

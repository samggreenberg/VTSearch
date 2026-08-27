"""When a run ends badly, is it the same run for every mode?

The tail is the product problem — the per-run figures show a bottom decile that
never leaves the floor whichever mode you pick — but "a bad tail" has two very
different causes and the same appearance:

* **a property of the mode.** Region voting fails a tenth of runs that
  whole-image handles. Then the fix is the mode, and switching mode helps those
  users.
* **a property of the data.** The same images defeat every mode. Then the fix is
  the dataset or the query, no mode switch helps anyone, and reporting it as a
  mode difference sends the next study after a ghost.

#3156 got this wrong once already, by eye: two "stuck exemplars" that survived a
change of representation were taken as evidence of hard images, and the
heuristic behind that shortlist — *a model failure should not survive a change
of representation* — turned out to be false (one image was correctly labelled
and genuinely hard, one was a mislabel, and both were seeding artefacts anyway).
Cross-mode persistence is a **shortlist generator, not a verdict.**

So this measures the persistence instead of asserting it, and it is careful in
two ways that matter:

* each mode's tail is cut at **its own** decile, not at a shared cost. Cutting
  at a shared threshold hands every bad run to the worst mode by construction
  and measures nothing.
* the overlap is quoted against **chance**, because two decile sets over the
  same runs overlap ~10% for free. A Jaccard of 0.05 is what independence looks
  like here, not a weak signal.

Usage::

    python analyze_tail_overlap.py --exp /expscratch/$USER/scale-3156-pair
    python analyze_tail_overlap.py --exp ... --metric cost --quantile 0.1
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

from figures_overview import band_of, class_of

#: For each metric, whether a LARGE value is the bad end.
BAD_IS_HIGH = {"cost": True, "regret": True, "average_precision": False, "auroc": False}


def final_values(exp: str, metric: str, step: int) -> dict[str, dict[tuple[str, str], float]]:
    """``mode -> {(category, seed): value at the deepest row <= step}``."""
    best: dict[str, dict[tuple[str, str], tuple[int, float]]] = defaultdict(dict)
    for path in sorted(glob.glob(str(Path(exp) / "results" / "cells" / "task_*.csv"))):
        if "__" in Path(path).name:
            continue
        try:
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        t, v = int(r["t"]), float(r[metric])
                    except (KeyError, ValueError, TypeError):
                        continue
                    if t > step:
                        continue
                    mode = f"{r.get('embedder', '')}/{r.get('style', '')}".strip("/")
                    key = (r["category"], r["seed"])
                    prev = best[mode].get(key)
                    if prev is None or prev[0] < t:
                        best[mode][key] = (t, v)
        except (OSError, csv.Error):
            continue
    return {m: {k: v for k, (_, v) in d.items()} for m, d in best.items()}


def tail_set(values: dict[tuple[str, str], float], q: float, bad_high: bool) -> set[tuple[str, str]]:
    """The worst *q* fraction of runs, by this mode's own distribution."""
    if not values:
        return set()
    ordered = sorted(values, key=lambda k: values[k], reverse=bad_high)
    return set(ordered[: max(1, int(round(q * len(ordered))))])


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, ties averaged — no scipy in this environment."""

    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156-pair")
    ap.add_argument("--metric", default="average_precision")
    ap.add_argument("--step", type=int, default=150)
    ap.add_argument("--quantile", type=float, default=0.10)
    args = ap.parse_args(argv)

    bad_high = BAD_IS_HIGH.get(args.metric, True)
    per_mode = final_values(args.exp, args.metric, args.step)
    if not per_mode:
        print(f"no rows under {args.exp}")
        return 1
    modes = sorted(per_mode)
    shared = set.intersection(*(set(per_mode[m]) for m in modes))
    print(f"metric={args.metric}  step<={args.step}  tail=worst {args.quantile:.0%} of each mode's OWN runs")
    print(f"{len(shared)} runs present in all {len(modes)} modes (of {max(len(per_mode[m]) for m in modes)} max)\n")

    tails = {m: tail_set({k: v for k, v in per_mode[m].items() if k in shared}, args.quantile, bad_high) for m in modes}

    print("=== do the modes fail the SAME runs? ===")
    print("Jaccard of the two tail sets, against what independence would give.")
    hdr = f"{'pair':<62}{'|A&B|':>7}{'jaccard':>9}{'chance':>8}{'ratio':>7}"
    print(hdr)
    print("-" * len(hdr))
    q = args.quantile
    chance_j = q / (2 - q)  # two independent q-fractions of one population
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            inter = len(tails[a] & tails[b])
            union = len(tails[a] | tails[b])
            j = inter / union if union else float("nan")
            print(f"{a + '  vs  ' + b:<62}{inter:>7}{j:>9.2f}{chance_j:>8.2f}{j / chance_j:>7.1f}x")
    all_three = set.intersection(*(tails[m] for m in modes))
    print(f"\nin EVERY mode's tail: {len(all_three)} runs  (chance ~{q ** (len(modes) - 1) * len(shared):.0f})")

    print("\n=== is it the CATEGORY, or the seed? ===")
    print("Each mode's per-category tail rate, and how those rates rank together.")
    cats = sorted({c for c, _ in shared})
    rate = {m: {c: 0.0 for c in cats} for m in modes}
    for m in modes:
        n_by_cat: dict[str, int] = defaultdict(int)
        bad_by_cat: dict[str, int] = defaultdict(int)
        for key in shared:
            n_by_cat[key[0]] += 1
            if key in tails[m]:
                bad_by_cat[key[0]] += 1
        for c in cats:
            rate[m][c] = bad_by_cat[c] / n_by_cat[c] if n_by_cat[c] else float("nan")
    hdr2 = f"{'modes compared':<62}{'spearman rho':>14}"
    print(hdr2)
    print("-" * len(hdr2))
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            rho = spearman([rate[a][c] for c in cats], [rate[b][c] for c in cats])
            print(f"{a + '  vs  ' + b:<62}{rho:>14.2f}")

    worst = sorted(cats, key=lambda c: -sum(rate[m][c] for m in modes))[:10]
    print(f"\n{'category':<20}{'band':<8}" + "".join(f"{m.split('/')[0][:14]:>16}" for m in modes))
    print("-" * (28 + 16 * len(modes)))
    for c in worst:
        print(f"{class_of(c):<20}{band_of(c):<8}" + "".join(f"{rate[m][c]:>16.2f}" for m in modes))
    covered = sum(1 for key in all_three if key[0] in set(worst))
    print(f"\n{covered}/{len(all_three)} of the universally-bad runs fall in those 10 categories.")
    print("A tail concentrated in a few categories is a DATA problem; one spread")
    print("evenly across categories but shared across modes is a HARNESS one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

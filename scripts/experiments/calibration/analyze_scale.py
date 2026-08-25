"""Analyse the scale study: does cost rise as the target shrinks? (#3156)

The contrast is the **band**, and the design makes it a paired one: the same
twelve classes appear at all three sizes, against identical negatives at
identical prevalence (0.0250 by construction). So every comparison here is
within `(class, seed, embedder)` and differs only in band -- which is exactly
what the published `vg_box_*` sets could not do, since their vocabularies are
disjoint and a small-vs-large gap there confounds size with class identity.

Reported paired, with a standard error, to two significant digits. A difference
smaller than twice its SE is called unresolvable rather than dressed up: "not
resolvable at three seeds" is a finding, and a more useful one than a decimal
the sample cannot support.

Encoders are a **blocking factor**, not a contrast: the question is whether the
band effect survives all three, so each is reported separately and they are
never pooled into one number.

Usage::

    python analyze_scale.py --exp /expscratch/$USER/scale-3156
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import defaultdict
from pathlib import Path

BANDS = ("small", "medium", "large")


def freshness_report(cells_dir: Path, expected: int) -> tuple[int, list[str]]:
    """How many cells exist, and how many share the newest run's generation.

    File existence is not evidence of a result: a task that dies leaves its
    PREVIOUS output in place, so a directory can hold a full set of cells from
    two different runs and look complete. Cells more than a few hours older than
    the newest are reported as suspect rather than silently averaged in.
    """
    import time

    files = [f for f in sorted(cells_dir.glob("task_*.csv")) if "__" not in f.name]
    if not files:
        return 0, []
    newest = max(f.stat().st_mtime for f in files)
    stale = [f.name for f in files if newest - f.stat().st_mtime > 6 * 3600]
    print(f"cells present: {len(files)} of {expected} expected")
    print(f"newest cell written: {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))}")
    if stale:
        print(f"WARNING: {len(stale)} cells are >6h older than the newest — from an earlier run?")
        print(f"         e.g. {', '.join(stale[:5])}")
    if len(files) != expected:
        print(f"WARNING: {expected - len(files)} cells missing; results below are a SUBSET")
    return len(files), stale


def load_rows(cells_dir: Path) -> list[dict]:
    import csv

    rows = []
    dropped = 0
    files = sorted(glob.glob(str(cells_dir / "task_*.csv")))
    for f in files:
        if "__" in Path(f).name:  # sweep/cutdiag sidecars, not the main rows
            continue
        try:
            with open(f, newline="") as fh:
                rows.extend(list(csv.DictReader(fh)))
        except Exception:
            dropped += 1
    return rows, len(files), dropped


def band_of(category: str) -> str:
    return category.rsplit("@", 1)[1] if "@" in category else ""


def class_of(category: str) -> str:
    return category.rsplit("@", 1)[0]


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def fmt(v: float, se: float | None = None) -> str:
    if v != v:
        return "n/a"
    if se is None or se != se:
        return f"{v:.2f}"
    return f"{v:.2f} ± {se:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156")
    ap.add_argument("--at-step", type=int, default=150, help="votes spent at which to read the headline")
    ap.add_argument("--expect", type=int, default=324, help="cells the grid should have produced")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cells = Path(args.exp) / "results" / "cells"
    freshness_report(cells, args.expect)
    rows, n_files, dropped = load_rows(cells)
    print(f"loaded {len(rows)} rows from {n_files} cell files ({dropped} unreadable)")

    # The endpoint of each trajectory: the last row at or before --at-step for
    # each (embedder, category, seed, style).
    last: dict[tuple, dict] = {}
    for r in rows:
        try:
            t = int(r["t"])
        except (KeyError, ValueError):
            continue
        if t > args.at_step:
            continue
        key = (r.get("embedder", ""), r["category"], r["seed"], r.get("style", ""))
        prev = last.get(key)
        if prev is None or int(prev["t"]) < t:
            last[key] = r

    # Encoder is the blocking factor and style is its arm (whole_image for the
    # single-vector encoders, max_patch for the patch one), so the two are kept
    # together as one label rather than pooled.
    per_band: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_key: dict[tuple, float] = {}
    styles = set()
    for (emb, cat, seed, style), r in last.items():
        try:
            cost = float(r["cost"])
        except (KeyError, ValueError, TypeError):
            continue
        b = band_of(cat)
        if b not in BANDS:
            continue
        arm = f"{emb}/{style}" if style else emb
        styles.add(arm)
        per_band[(arm, b)].append(cost)
        by_key[(arm, class_of(cat), seed, b)] = cost

    print(f"arms present: {sorted(styles)}")
    print()
    print(f"=== cost at t={args.at_step}, by band (lower is better) ===")
    hdr = f"{'arm':<26}" + "".join(f"{b:>16}" for b in BANDS) + f"{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for style in sorted(styles):
        line = f"{style:<26}"
        n = 0
        for b in BANDS:
            xs = per_band[(style, b)]
            n = max(n, len(xs))
            m, se = mean_se(xs)
            line += f"{fmt(m, se):>16}"
        print(line + f"{n:>6}")

    print()
    print("=== paired small - large, within (class, seed) ===")
    print(f"{'arm':<26}{'mean diff':>16}{'n pairs':>9}{'resolvable?':>14}")
    print("-" * 65)
    for style in sorted(styles):
        diffs = []
        for (st, cls, seed, b), v in by_key.items():
            if st != style or b != "small":
                continue
            other = by_key.get((st, cls, seed, "large"))
            if other is not None:
                diffs.append(v - other)
        m, se = mean_se(diffs)
        verdict = "yes" if (se == se and abs(m) > 2 * se) else "NOT RESOLVABLE"
        print(f"{style:<26}{fmt(m, se):>16}{len(diffs):>9}{verdict:>14}")

    print()
    print("Cost is the harness's operating-point cost; every comparison above is")
    print("paired on (class, seed) and differs only in band. A difference smaller")
    print("than twice its standard error is not resolvable at this seed count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

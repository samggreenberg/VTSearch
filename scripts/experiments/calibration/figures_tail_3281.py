"""Did the #3156 stuck tail survive the #3281 box repair? Two panels that answer it.

Draws `docs/experiments/2026-08-25-vg-scale/figures/stuck_tail_after_3281.png` for #3284.

Three grids of the same design differ in known ways, which is what makes the
question answerable at all::

    scale-3156-final  crop seeding, head=linear,     corrupt boxes
    scale-3156-pair   text seeding, head=linear_svm, corrupt boxes
    scale-3156-fixed  text seeding, head=linear_svm, REPAIRED boxes

`final -> pair` moves three things at once and attributes nothing. `pair ->
fixed` moves only the pile rebuild, so the right-hand panel is a clean read on
the boxes.

**The right-hand panel plots median cost, not the worst-decile rate**, and that
is deliberate. A share-of-the-worst-decile is a fixed 10% budget: when the
corrupted classes leave the tail some other class must fill it, so `stop
sign@small` reads as 0.22 -> 0.55 "worse" while its median cost moves 0.43 ->
0.44. A share of a fixed tail is zero-sum across the things sharing it; plot the
absolute quantity.

    python figures_tail_3281.py --out docs/experiments/2026-08-25-vg-scale/figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STUCK = 0.9
HORIZON = 150

# corrupt boxes per category, from #3281's census
CRUSHED = {"backpack@small": 44, "bird@small": 42, "bicycle@small": 34}

GRIDS = [
    ("final", "scale-3156-final", "dinov3_patch/max_patch", "crop seeding\ncorrupt boxes\n(this report)", "#b23a48"),
    (
        "pair",
        "scale-3156-pair",
        "siglip+dinov3_patch/max_patch",
        "text seeding\ncorrupt boxes\n(#3276 pair run)",
        "#d99058",
    ),
    (
        "fixed",
        "scale-3156-fixed",
        "siglip+dinov3_patch/max_patch",
        "text seeding\nREPAIRED boxes\n(scale-3156-fixed)",
        "#4c8577",
    ),
]


def _f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _load(cells: Path) -> dict:
    """Last row at or before the horizon, per ``(mode, category, seed)`` run."""
    runs: dict = {}
    for fp in sorted(glob.glob(str(cells / "task_[0-9]*.csv"))):
        if "__" in os.path.basename(fp):  # cutdiag / cutincl sidecars
            continue
        with open(fp, newline="") as fh:
            for row in csv.DictReader(fh):
                t = _f(row.get("t"))
                if t != t or t > HORIZON:
                    continue
                key = (f"{row.get('embedder', '')}/{row.get('style', '')}", row["category"], int(_f(row["seed"])))
                prev = runs.get(key)
                if prev is None or t >= prev[0]:
                    runs[key] = (t, _f(row["cost"]), _f(row["average_precision"], 1.0))
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/expscratch/{user}/".format(user=os.environ.get("USER", "")),
        help="directory holding the three grid dirs",
    )
    ap.add_argument("--out", required=True, help="figures directory to write into")
    args = ap.parse_args(argv)

    root = Path(args.root)
    loaded = {}
    for tag, sub, _mode, _lab, _c in GRIDS:
        cells = root / sub / "results" / "cells"
        if not cells.is_dir():
            print(f"missing: {cells}")
            return 1
        loaded[tag] = _load(cells)
        print(f"{tag}: {len(loaded[tag])} runs")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # --- panel 1: the catastrophic tail across the three grids ---
    ax = axes[0]
    rates, notes = [], []
    for tag, _sub, _mode, _lab, _c in GRIDS:
        runs = loaded[tag]
        stuck = sum(1 for v in runs.values() if v[1] >= STUCK)
        rates.append(100.0 * stuck / len(runs))
        notes.append(f"{stuck} / {len(runs)}")
    bars = ax.bar(range(len(GRIDS)), rates, color=[g[4] for g in GRIDS], width=0.62)
    for rate, note, bar in zip(rates, notes, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{rate:.2f}%\n{note}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(range(len(GRIDS)))
    ax.set_xticklabels([g[3] for g in GRIDS], fontsize=9)
    ax.set_ylabel(f"runs ending at cost $\\geq$ {STUCK}  (%)")
    ax.set_title("The catastrophic tail, and what removed it", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.35)
    ax.grid(axis="y", alpha=0.3)

    # --- panel 2: region arm, small band, corrupt vs repaired ---
    ax = axes[1]

    def medians(tag: str, mode: str) -> dict:
        per = defaultdict(list)
        for (m, cat, _seed), v in loaded[tag].items():
            if m == mode:
                per[cat].append(v[1])
        return {c: st.median(vs) for c, vs in per.items() if vs}

    mp = medians("pair", GRIDS[1][2])
    mf = medians("fixed", GRIDS[2][2])
    cats = sorted((c for c in set(mp) | set(mf) if c.endswith("@small")), key=lambda c: -mp.get(c, 0.0))
    height = 0.38
    ax.barh(
        [i + height / 2 for i in range(len(cats))],
        [mp.get(c, 0) for c in cats],
        height=height,
        label="corrupt boxes (pair run)",
        color=GRIDS[1][4],
    )
    ax.barh(
        [i - height / 2 for i in range(len(cats))],
        [mf.get(c, 0) for c in cats],
        height=height,
        label="repaired boxes (fixed run)",
        color=GRIDS[2][4],
    )
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{c}  ({CRUSHED[c]} crushed)" if c in CRUSHED else c for c in cats], fontsize=9)
    for i, c in enumerate(cats):
        if c in CRUSHED:
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(f"median cost at {HORIZON} votes (lower is better)")
    ax.set_title("Region arm, small band: same head, same seeding,\nonly the boxes differ", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "stuck_tail_after_3281.png"
    fig.savefig(path, dpi=130)
    print("wrote", path)
    for c in cats:
        print(f"  {c:<18} median cost  pair {mp.get(c, 0):.2f}  fixed {mf.get(c, 0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What does each Autopilot phase actually do, across the whole grid?

The session sheets show one run. This is the same question over every run: how
many of the user's clicks each phase spends, how often those clicks land on a
positive, and where in the opening sort they come from.

Descriptive only -- it ranks nothing and recommends nothing. It exists because
"the app spends 80% of a session in one phase" is the kind of fact that is
obvious in the data and invisible in every summary metric, which all average
over the phases rather than separating them.

Reads the per-click log `run_cells.py` writes (`task_*__picks.csv`).

    python analyze_phases.py --exp /expscratch/$USER/scale-3156-fixed

Caveat carried from `pick_sheets.py`: `picked_seed_percentile` is a position in
the text sort over ALL medias, while a run walks only the simulation split
(~half at the default `sim_fraction`), so it overstates depth-into-the-pool.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from _cells_paths import side_frame_files

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156-fixed")
EXP = _ap.parse_args().exp
rows_by = defaultdict(lambda: {"n": 0, "good": 0, "pct": [], "runs": set()})
files = side_frame_files(Path(EXP) / "results" / "cells", "__picks")
for path in files:
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                mode = f"{r.get('embedder', '')}/{r.get('style', '')}".strip("/")
                ph = r.get("phase") or "?"
                d = rows_by[(mode, ph)]
                d["n"] += 1
                d["good"] += 1 if r.get("picked_label") == "1" else 0
                d["runs"].add((r.get("category"), r.get("seed")))
                try:
                    d["pct"].append(float(r["picked_seed_percentile"]))
                except (KeyError, ValueError, TypeError):
                    pass
    except (OSError, csv.Error):
        continue


def q(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


print(f"{len(files)} cells\n")
hdr = f"{'mode':<32}{'phase':<8}{'clicks':>9}{'clicks/run':>12}{'hit rate':>10}{'seed pctile p10/p50/p90':>26}"
print(hdr)
print("-" * len(hdr))
for mode in sorted({k[0] for k in rows_by}):
    for ph in ("good", "bad", "hard", "new", "done"):
        d = rows_by.get((mode, ph))
        if not d or not d["n"]:
            continue
        pr = d["n"] / max(1, len(d["runs"]))
        pcts = f"{q(d['pct'], 0.1):.2f} / {q(d['pct'], 0.5):.2f} / {q(d['pct'], 0.9):.2f}" if d["pct"] else "-"
        print(f"{mode:<32}{ph:<8}{d['n']:>9}{pr:>12.1f}{d['good'] / d['n']:>10.2f}{pcts:>26}")
    print()
print("hit rate = fraction of that phase's clicks that landed on a positive.")
print("seed pctile = where in the opening text sort the pick came from (0 = top).")

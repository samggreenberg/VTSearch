"""Head-strategy (or any-arm) comparison straight from results.jsonl — trace-free.

Avoids the /exp ENOSPC + the same-out-dir results.jsonl clobber (run each class to its own
out-dir, no ``--labeling-trace``). Computes, per arm dir (globs all results.jsonl under it):
the **deep-spike rate** (Δcost>0.1 between consecutive ``t`` within a (class,seed), MLP-regime
only = ``calib_mode != cosine_coldstart``, ``t >= deep_t``), the median spike magnitude, and
**cost=FNR+FPR / FNR / FPR** at fixed vote budgets. Usage: ``results_eval.py DIR LABEL [...]``.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _load(root: Path):
    g = defaultdict(list)
    for f in root.rglob("results.jsonl"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            g[(r.get("class"), r.get("seed"))].append(r)
    return g


def analyze(root: Path, label: str, deep_t: int = 20, thresh: float = 0.1):
    g = _load(root)
    mlp_steps = deep_steps = deep_spikes = 0
    ups = []
    budgets = [20, 40, 60]
    cost = {b: [] for b in budgets}
    fnr = {b: [] for b in budgets}
    fpr = {b: [] for b in budgets}
    for seq in g.values():
        seq.sort(key=lambda r: r["t"])
        for i in range(1, len(seq)):
            if seq[i].get("calib_mode") == "cosine_coldstart":
                continue
            dc = seq[i]["cost"] - seq[i - 1]["cost"]
            mlp_steps += 1
            if (seq[i].get("t") or 0) >= deep_t:
                deep_steps += 1
                if dc > thresh:
                    deep_spikes += 1
                    ups.append(dc)
        for b in budgets:
            row = None
            for r in seq:
                if (r.get("t") or 0) <= b:
                    row = r
                else:
                    break
            if row is not None:
                cost[b].append(row["cost"])
                fnr[b].append(row["fnr"])
                fpr[b].append(row["fpr"])

    def m(x):
        v = [a for a in x if a is not None]
        return statistics.fmean(v) if v else float("nan")

    dr = deep_spikes / deep_steps if deep_steps else float("nan")
    print(f"{label:<11} cells={len(g):<4} deep-spike rate(t>={deep_t}) {dr:.3f}  "
          f"med-up {statistics.median(ups) if ups else float('nan'):.3f}  ({deep_spikes}/{deep_steps})")  # fmt: skip
    for b in budgets:
        print(f"    @t={b:<3} cost {m(cost[b]):.3f}  FNR {m(fnr[b]):.3f}  FPR {m(fpr[b]):.3f}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    for i in range(0, len(argv), 2):
        analyze(Path(argv[i]), argv[i + 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

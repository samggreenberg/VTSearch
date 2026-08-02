"""Evaluate positive-seeking 'soft' acquisition on the metrics that matter for rare needles.

Per the rare-needle, don't-miss-any use case: report **n_good growth** (does soft break past
the ~3-5 starvation?), **FNR** (needles missed = 1-recall), **FPR**, and **cost = FNR+FPR**
(prevalence-independent, recall-inclusive) at matched vote budgets — NOT F1 (which, under the
100x imbalance, collapses to precision and under-weights FNR). Also the realized **soft-pick
good-rate** (did soft actually surface positives?). Compare baseline / soft / g6b4-ceiling.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sod"))  # noqa: E402
from spike_analysis import _f  # type: ignore  # noqa: E402


def _at(tr, t):
    best = None
    for e in tr:
        if (e.get("t") or 0) <= t:
            best = e
        else:
            break
    return best


def evaluate(root: Path, label: str):
    budgets = [20, 40, 60]
    ng = {b: [] for b in budgets}
    fnr = {b: [] for b in budgets}
    fpr = {b: [] for b in budgets}
    cost = {b: [] for b in budgets}
    soft_good = soft_tot = 0
    n = 0
    for tj in root.rglob("trace.json"):
        tr = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        n += 1
        for e in tr:
            if e.get("select_mode") == "soft":
                soft_tot += 1
                soft_good += 1 if e.get("gt_label") == "good" else 0
        for b in budgets:
            e = _at(tr, b)
            if e:
                ng[b].append(e.get("n_good"))
                fnr[b].append(_f(e.get("fnr")))
                fpr[b].append(_f(e.get("fpr")))
                cost[b].append(_f(e.get("cost")))

    def m(x):
        v = [a for a in x if a is not None and a == a]
        return statistics.fmean(v) if v else float("nan")

    print(
        f"{label}: n_traces={n}"
        + (f"  soft-pick good-rate {soft_good}/{soft_tot}={soft_good / soft_tot:.3f}" if soft_tot else "")
    )
    print(f"  {'budget':<8}{'n_good':>8}{'FNR':>8}{'FPR':>8}{'cost':>8}   (cost=FNR+FPR; FNR=needles missed)")
    for b in budgets:
        print(f"  t={b:<6}{m(ng[b]):>8.1f}{m(fnr[b]):>8.3f}{m(fpr[b]):>8.3f}{m(cost[b]):>8.3f}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    for i in range(0, len(argv), 2):
        evaluate(Path(argv[i]), argv[i + 1])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

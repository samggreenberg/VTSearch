"""Learned-detector QUALITY at/after the Text→Learned switch, vs how long we stay in TextHard.

"It seemed like we had horrible scores when we switched from Text to Learned." Does
collecting more bads (higher ``bad_to_start``) — or more goods — before the MLP switches on
make the *first* learned detector less garbage, and/or the detector better at a matched
annotation budget? For each trace this finds the switch step (first ``head==mlp``) and reports:

* the **handoff jump** = cost at the first MLP step − cost at the last cosine step (the
  "horrible switch"), and the absolute cost at the switch;
* cost/F1 a few steps **after** the switch (does the young MLP recover fast?);
* cost/F1 at fixed **total-vote budgets** t ∈ {20,40,60} so arms whose switch lands at
  different t are compared at equal annotation cost.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sod"))  # noqa: E402
from spike_analysis import _f  # type: ignore  # noqa: E402


def _at_budget(tr, t):
    best = None
    for e in tr:
        if (e.get("t") or 0) <= t:
            best = e
        else:
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Text→Learned handoff quality (#2790).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    budgets = [20, 40, 60]
    after = [0, 3, 6]  # steps after the switch
    jumps, at_switch, switch_ts = [], [], []
    cost_after = {k: [] for k in after}
    cost_at = {b: [] for b in budgets}
    f1_at = {b: [] for b in budgets}
    final_c, final_f = [], []
    n = 0
    for tj in sorted(Path(args.root).rglob("trace.json")):
        tr = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        sw = next((i for i, e in enumerate(tr) if e.get("head") == "mlp"), None)
        if sw is None or sw == 0:
            continue
        n += 1
        cb, ca = _f(tr[sw - 1].get("cost")), _f(tr[sw].get("cost"))
        jumps.append(ca - cb)
        at_switch.append(ca)
        switch_ts.append(tr[sw].get("t"))
        for k in after:
            if sw + k < len(tr):
                cost_after[k].append(_f(tr[sw + k].get("cost")))
        for b in budgets:
            e = _at_budget(tr, b)
            if e:
                cost_at[b].append(_f(e.get("cost")))
                f1_at[b].append(_f(e.get("f1")))
        final_c.append(_f(tr[-1].get("cost")))
        final_f.append(_f(tr[-1].get("f1")))

    def m(x):
        vals = [v for v in x if v == v]
        return statistics.fmean(vals) if vals else float("nan")

    print(f"{args.label or args.root}: n_traces={n}")
    print(f"  switch_t (first MLP)      : {m(switch_ts):.1f}")
    print(f"  cost AT switch (1st MLP)  : {m(at_switch):.3f}   handoff jump Δ {m(jumps):+.3f}")
    print(f"  cost after switch  +0/+3/+6: {m(cost_after[0]):.3f} / {m(cost_after[3]):.3f} / {m(cost_after[6]):.3f}")
    for b in budgets:
        print(f"  @ t={b:<3} (equal budget) cost {m(cost_at[b]):.3f}   f1 {m(f1_at[b]):.3f}")
    print(f"  final (t=60)              cost {m(final_c):.3f}   f1 {m(final_f):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

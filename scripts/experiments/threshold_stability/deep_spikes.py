"""The ACTUAL #2790 spikes: deep-run transient excursions (converged MLP jumps up, snaps back).

Not the start-up cosine->MLP handoff — the violent single-step jumps *while the MLP is
already trained and improving* (the issue's t~24: cost 0.088 -> 0.424 in one retrain, then
recovers). For every Bad vote once the MLP is on (``n_bad >= bad_to_start``, default 4),
tracks the **up-jump** (Δcost > thresh) AND its **recovery** — does cost snap back near the
pre-spike level within a few steps ("jump back to good") or run away? Sliced by depth ``t``
and reported with the state at the spike (n_good, n_bad, surface_margin), plus a few full
before→spike→after cost trajectories.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sod"))  # noqa: E402
from spike_analysis import _class_seed, _f  # type: ignore  # noqa: E402


def collect(root: Path, thresh: float, bad_to_start: int):
    events = []  # every MLP-regime bad vote
    for tj in sorted(root.rglob("trace.json")):
        cls, seed = _class_seed(tj)
        tr = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        costs = [_f(e.get("cost")) for e in tr]
        for i in range(1, len(tr)):
            cur = tr[i]
            if cur.get("gt_label") != "bad":
                continue
            if (cur.get("n_bad") or 0) < bad_to_start:  # MLP not on yet (cosine phase)
                continue
            up = costs[i] - costs[i - 1]
            is_spike = up > thresh
            rec_steps = None  # steps until cost returns within 0.05 of pre-spike level
            down = float("nan")
            if is_spike:
                pre = costs[i - 1]
                for k in range(1, 6):
                    if i + k < len(tr) and costs[i + k] <= pre + 0.05:
                        rec_steps = k
                        break
                if i + 1 < len(tr):
                    down = costs[i + 1] - costs[i]
            events.append(
                {
                    "cls": cls,
                    "seed": seed,
                    "t": cur.get("t"),
                    "n_good": tr[i - 1].get("n_good"),
                    "n_bad": tr[i - 1].get("n_bad"),
                    "surface_margin": _f(cur.get("surface_margin")),
                    "pre_cost": round(costs[i - 1], 3),
                    "spike_cost": round(costs[i], 3),
                    "up": up,
                    "down": down,
                    "is_spike": is_spike,
                    "rec_steps": rec_steps,
                }  # fmt: skip
            )
    return events


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deep-run transient spike dynamics (#2790).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--bad-to-start", type=int, default=4)
    ap.add_argument("--deep-t", type=int, default=20)
    args = ap.parse_args(argv)

    ev = collect(Path(args.root), args.thresh, args.bad_to_start)
    spikes = [e for e in ev if e["is_spike"]]
    print(f"MLP-regime bad votes: {len(ev)}   up-spikes (Δcost>{args.thresh}): {len(spikes)} "
          f"({len(spikes) / max(1, len(ev)):.3f})\n")  # fmt: skip

    print("== spike rate & recovery by depth t ==")
    print(
        f"  {'t band':<10} {'votes':>7} {'spikes':>7} {'rate':>6} {'transient':>10} {'runaway':>8} {'med_up':>7} {'med_rec_steps':>14}"
    )
    bands = [(4, 10), (10, 20), (20, 30), (30, 45), (45, 999)]
    for lo, hi in bands:
        vv = [e for e in ev if lo <= (e["t"] or 0) < hi]
        ss = [e for e in vv if e["is_spike"]]
        if not vv:
            continue
        trans = [e for e in ss if e["rec_steps"] is not None and e["rec_steps"] <= 2]
        runaway = [e for e in ss if e["rec_steps"] is None]
        med_up = statistics.median(e["up"] for e in ss) if ss else float("nan")
        rec_vals = [e["rec_steps"] for e in ss if e["rec_steps"] is not None]
        med_rec = statistics.median(rec_vals) if rec_vals else float("nan")
        tp = f"{len(trans)}({len(trans) / len(ss):.0%})" if ss else "-"
        rp = f"{len(runaway)}({len(runaway) / len(ss):.0%})" if ss else "-"
        print(
            f"  t[{lo:>2},{hi if hi < 999 else '+':>3}) {len(vv):>7} {len(ss):>7} {len(ss) / len(vv):>6.3f} {tp:>10} {rp:>8} {med_up:>7.3f} {med_rec:>14.1f}"
        )

    deep = [e for e in spikes if (e["t"] or 0) >= args.deep_t]
    if deep:
        print(f"\n== DEEP spikes (t>={args.deep_t}): n={len(deep)} ==")
        trans = [e for e in deep if e["rec_steps"] is not None and e["rec_steps"] <= 2]
        print(f"  transient (snap back <=2 steps): {len(trans)}/{len(deep)} = {len(trans) / len(deep):.0%}")
        print(f"  median up-jump Δcost   : +{statistics.median(e['up'] for e in deep):.3f}")
        downs = [e["down"] for e in deep if e["down"] == e["down"]]
        print(f"  median next-step recover: {statistics.median(downs):+.3f}" if downs else "  (no recover data)")
        print(f"  n_good at deep spike   : median {statistics.median(e['n_good'] for e in deep):.0f} "
              f"(min {min(e['n_good'] for e in deep)}, max {max(e['n_good'] for e in deep)})")  # fmt: skip
        print(f"  surface_margin>=0 share: {sum(1 for e in deep if e['surface_margin'] >= 0) / len(deep):.0%}")
        print("\n  example deep transient spikes (pre → spike → recovery):")
        exs = sorted((e for e in trans), key=lambda e: -e["up"])[:6]
        for e in exs:
            print(f"    {e['cls']:<14} s{e['seed']} t{e['t']:>2}  g{e['n_good']} b{e['n_bad']}  "
                  f"cost {e['pre_cost']:.3f} → {e['spike_cost']:.3f} (Δ+{e['up']:.3f}) → back {e['down']:+.3f} in {e['rec_steps']} step(s)")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

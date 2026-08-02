"""Necessary + sufficient conditions for the #2790 calibration catastrophe (v2: the gap).

v1 refuted the training-recall-collapse hypothesis: at spikes the labeled positives stay
ABOVE the cut (recall_frac 0.996), and Bad votes essentially never outscore a labeled
positive. The labeled positives are unrepresentatively high-scoring; the cut lives in the
GAP between the top labeled bad and the bottom labeled positive, and the *test* positives
live in that gap. Refined hypothesis: a spike is the cut being pushed UP within that gap
(``cut_in_gap`` ↑), triggered by a Bad scored high in the gap (``vote_above_badmax`` — it
raises the conformal gap floor). This tests it on the instrumented trace.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sod"))  # noqa: E402
from spike_analysis import _f  # type: ignore  # noqa: E402


def collect(root: Path, thresh: float):
    ev = []
    for tj in sorted(root.rglob("trace.json")):
        tr = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        costs = [_f(e.get("cost")) for e in tr]
        for i in range(1, len(tr)):
            cur, prev = tr[i], tr[i - 1]
            if cur.get("head") != "mlp":
                continue
            cig, pcig = cur.get("cut_in_gap"), prev.get("cut_in_gap")
            ev.append(
                {
                    "is_spike": int(costs[i] - costs[i - 1] > thresh),
                    "label": cur.get("gt_label"),
                    "vote_above_badmax": cur.get("vote_above_badmax"),
                    "vote_beats_bad": cur.get("vote_beats_bad"),
                    "n_bad_lab": (prev.get("n_bad") or 0),
                    "cut_in_gap": cig,
                    "d_cut_in_gap": (cig - pcig if cig is not None and pcig is not None else None),
                    "d_thr": cur.get("delta_threshold"),
                }
            )
    return ev


def _rate(ev, cond):
    s = [e for e in ev if cond(e)]
    k = sum(e["is_spike"] for e in s)
    return k, len(s), (k / len(s) if s else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="#2790 spike conditions (gap hypothesis).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--thresh", type=float, default=0.1)
    args = ap.parse_args(argv)
    ev = collect(Path(args.root), args.thresh)
    n = len(ev)
    sp = [e for e in ev if e["is_spike"]]
    ns = [e for e in ev if not e["is_spike"]]
    print(f"MLP-regime steps: {n}   up-spikes: {len(sp)} ({len(sp) / n:.3f})\n")

    def m(x):
        v = [a for a in x if a is not None and a == a]
        return statistics.fmean(v) if v else float("nan")

    print("== (A) where the cut sits in the bad→positive gap (0=top bad, 1=bottom positive) ==")
    print(
        f"  cut_in_gap    spikes {m([e['cut_in_gap'] for e in sp]):.3f}   non-spikes {m([e['cut_in_gap'] for e in ns]):.3f}"
    )
    print(
        f"  Δcut_in_gap   spikes {m([e['d_cut_in_gap'] for e in sp]):+.3f}   non-spikes {m([e['d_cut_in_gap'] for e in ns]):+.3f}"
    )
    print(f"  Δthreshold    spikes {m([e['d_thr'] for e in sp]):+.4f}   non-spikes {m([e['d_thr'] for e in ns]):+.4f}")

    print("\n== (B) trigger: spike rate by vote-above-topbad (raises the gap floor) ==")
    for lab in ("bad", "good"):
        for name, cond in [
            (f"{lab} above top bad", lambda e, lab=lab: e["label"] == lab and e["vote_above_badmax"] == 1),
            (f"{lab} below top bad", lambda e, lab=lab: e["label"] == lab and e["vote_above_badmax"] == 0),
        ]:
            k, tot, r = _rate(ev, cond)
            if tot:
                print(f"    {name:<20} {k:>4}/{tot:<6} = {r:.3f}")

    print("\n== (C) candidate rules (precision = P(spike|rule); recall = spikes caught) ==")
    tot_sp = len(sp)
    rules = [
        ("vote above top bad (any)", lambda e: e["vote_above_badmax"] == 1),
        ("Bad above top bad", lambda e: e["label"] == "bad" and e["vote_above_badmax"] == 1),
        ("Δcut_in_gap > 0.1 (cut pushed up)", lambda e: (e["d_cut_in_gap"] or 0) > 0.1),
        ("Δthreshold > 0 (cut moved up)", lambda e: (e["d_thr"] or 0) > 0),
        ("above-topbad AND Δthr>0", lambda e: e["vote_above_badmax"] == 1 and (e["d_thr"] or 0) > 0),
    ]
    print(f"  {'rule':<38} {'precision':>10} {'recall':>8} {'n':>8}")
    for nm, rule in rules:
        k, tot, prec = _rate(ev, rule)
        rec = k / tot_sp if tot_sp else float("nan")
        print(f"  {nm:<38} {prec:>10.3f} {rec:>8.3f} {tot:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

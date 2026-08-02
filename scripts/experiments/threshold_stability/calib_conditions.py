"""Necessary + sufficient conditions for the #2790 calibration catastrophe.

Hypothesis: a deep spike is a **training-recall collapse** — the conformal cut jumping ABOVE
the sparse labeled positives — triggered by a vote scored among/above them. The instrumented
trace (region_curve `_calib_diag`) records, per MLP-regime step:
  ``n_pos_above_cut`` (labeled positives still ≥ cut), ``n_pos_lab``, ``vote_beats_pos``
  (positives the just-voted item outscored), ``gt_label``.
This script tests:
  (A) does a spike ⟺ training-recall collapse (n_pos_above_cut/n_pos_lab → 0)?
  (B) the trigger: spike rate by (vote label, vote_beats_pos) — is "a Bad that outscores ≥1
      positive" necessary (spike~0 otherwise) and sufficient (spike~1 when it holds)?
  (C) precision/recall of candidate necessary+sufficient rules.
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
            cur = tr[i]
            if cur.get("head") != "mlp":
                continue
            up = costs[i] - costs[i - 1]
            npl = cur.get("n_pos_lab") or 0
            nab = cur.get("n_pos_above_cut")
            ev.append(
                {
                    "is_spike": int(up > thresh),
                    "label": cur.get("gt_label"),
                    "vote_beats_pos": cur.get("vote_beats_pos"),
                    "n_pos_lab": npl,
                    "recall_frac": (nab / npl if npl and nab is not None else None),
                    "fnr": _f(cur.get("fnr")),
                }
            )
    return ev


def _rate(ev, cond):
    s = [e for e in ev if cond(e)]
    k = sum(e["is_spike"] for e in s)
    return k, len(s), (k / len(s) if s else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Necessary+sufficient conditions for #2790 spikes.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--thresh", type=float, default=0.1)
    args = ap.parse_args(argv)
    ev = collect(Path(args.root), args.thresh)
    n = len(ev)
    sp = [e for e in ev if e["is_spike"]]
    print(f"MLP-regime steps: {n}   up-spikes: {len(sp)} ({len(sp) / n:.3f})\n")

    def m(x):
        v = [a for a in x if a is not None and a == a]
        return statistics.fmean(v) if v else float("nan")

    print("== (A) training-recall at the step: frac of labeled positives still >= cut ==")
    print(f"  spikes    : recall_frac mean {m([e['recall_frac'] for e in sp]):.3f}  (→0 = cut above the positives)")
    print(f"  non-spikes: recall_frac mean {m([e['recall_frac'] for e in ev if not e['is_spike']]):.3f}")
    print(f"  spikes with ALL positives below cut (recall_frac==0): "
          f"{sum(1 for e in sp if e['recall_frac'] == 0)}/{len(sp)}")  # fmt: skip

    print("\n== (B) trigger: spike rate by vote label × how many positives it outscored ==")
    for lab in ("bad", "good"):
        print(f"  {lab} votes:")
        for bp in ("0", "1", "2", ">=3"):

            def c(e, lab=lab, bp=bp):
                if e["label"] != lab or e["vote_beats_pos"] is None:
                    return False
                v = e["vote_beats_pos"]
                return {"0": v == 0, "1": v == 1, "2": v == 2, ">=3": v >= 3}[bp]

            k, tot, r = _rate(ev, c)
            if tot:
                print(f"    beats {bp:<3} positives: {k:>4}/{tot:<6} = {r:.3f}")

    print("\n== (C) candidate necessary+sufficient rules (precision = spike rate | rule; recall = spikes caught) ==")
    rules = [
        ("Bad AND beats >=1 pos", lambda e: e["label"] == "bad" and (e["vote_beats_pos"] or 0) >= 1),
        ("beats >=1 pos (any label)", lambda e: (e["vote_beats_pos"] or 0) >= 1),
        (
            "Bad AND beats >=1 AND n_pos_lab<=5",
            lambda e: e["label"] == "bad" and (e["vote_beats_pos"] or 0) >= 1 and (e["n_pos_lab"] or 0) <= 5,
        ),  # noqa: E501
        (
            "Bad AND beats ALL pos",
            lambda e: e["label"] == "bad" and e["n_pos_lab"] and e["vote_beats_pos"] == e["n_pos_lab"],
        ),
    ]
    tot_sp = len(sp)
    print(f"  {'rule':<38} {'precision':>10} {'recall':>8} {'n':>7}")
    for name, rule in rules:
        k, tot, prec = _rate(ev, rule)
        rec = k / tot_sp if tot_sp else float("nan")
        print(f"  {name:<38} {prec:>10.3f} {rec:>8.3f} {tot:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

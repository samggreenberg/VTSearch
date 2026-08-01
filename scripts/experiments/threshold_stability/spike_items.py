"""Per-*item* spike analysis (#2790): which media items cause the cost spikes?

The rate-level analysis (`spike_analysis.py`) treats spikes as sporadic events. This
asks the other question: each spike was triggered by a vote on a *specific image* —
are there images that cause spikes disproportionately, what are they (hard/false
negatives?), and what precedes them?

For every labeling trace it:
- accumulates, per (class, image_id): how many times the item was **voted** across
  seeds (`n_votes`), and how many of those votes triggered an up-spike (`n_spikes`),
  giving a per-item `spike_rate` — the repeat-offender signal;
- records the item's ground-truth label, and the model's **surface_score** when it
  was surfaced (how positive the model thought it was — a Bad vote with a high
  surface score is a *hard/false* negative: it looks like the class);
- records the **preceding context** (the label of the previous vote, and the run
  length of consecutive same-label votes before the spike) — a click-time signal.

Writes `spike_items.json` (ranked offenders + summary) next to --root and prints a
summary. `--top N` controls how many offenders to emit for the visual report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from spike_analysis import _class_seed, _f  # reuse trace parsing helpers


def analyze(root: Path, thresh: float) -> dict:
    # Per (class, iid): votes and spike-culprit records.
    votes: dict[tuple, int] = defaultdict(int)
    spikes: dict[tuple, list[dict]] = defaultdict(list)
    n_traces = 0
    for tj in sorted(root.rglob("trace.json")):
        n_traces += 1
        cls, seed = _class_seed(tj)
        trace = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        # consecutive same-label run length up to each step
        run = 0
        prev_label = None
        for i, e in enumerate(trace):
            iid = int(e["image_id"])
            lbl = e["gt_label"]
            votes[(cls, iid)] += 1
            run = run + 1 if lbl == prev_label else 1
            if i >= 1:
                dcost = _f(e.get("cost")) - _f(trace[i - 1].get("cost"))
                if dcost > thresh:  # this vote triggered an up-spike
                    spikes[(cls, iid)].append(
                        {
                            "seed": seed,
                            "t": e["t"],
                            "gt_label": lbl,
                            "dcost": round(dcost, 4),
                            "d_fnr": round(_f(e.get("fnr")) - _f(trace[i - 1].get("fnr")), 4),
                            "surface_score": e.get("surface_score"),
                            "surface_margin": e.get("surface_margin"),
                            "n_good": trace[i - 1].get("n_good"),
                            "prev_label": prev_label,
                            "run_len": run,
                        }
                    )
            prev_label = lbl

    items = []
    for (cls, iid), recs in spikes.items():
        nv = votes[(cls, iid)]
        surf = [r["surface_score"] for r in recs if isinstance(r["surface_score"], (int, float))]
        items.append(
            {
                "cls": cls,
                "image_id": iid,
                "gt_label": recs[0]["gt_label"],
                "n_spikes": len(recs),
                "n_votes": nv,
                "spike_rate": round(len(recs) / nv, 3) if nv else None,
                "seeds": sorted({r["seed"] for r in recs}),
                "mean_surface_score": round(statistics.fmean(surf), 4) if surf else None,
                "mean_dcost": round(statistics.fmean([r["dcost"] for r in recs]), 4),
                "mean_dfnr": round(statistics.fmean([r["d_fnr"] for r in recs]), 4),
                "mean_n_good": round(statistics.fmean([r["n_good"] or 0 for r in recs]), 1),
                "prev_labels": _counts(r["prev_label"] for r in recs),
            }
        )
    items.sort(key=lambda d: (-d["n_spikes"], -(d["spike_rate"] or 0)))

    total_spikes = sum(d["n_spikes"] for d in items)
    repeat = [d for d in items if d["n_spikes"] >= 2]
    # Confidence tiers: an item that spiked in >=K of its seeds is spiky by design, not
    # coincidence. With 30 seeds, >=5 is a robust offender.
    tiers = {f"items_ge{k}": sum(1 for d in items if d["n_spikes"] >= k) for k in (3, 5, 10)}
    classes = {d["cls"] for d in items}
    classes_with_offender = {d["cls"] for d in items if d["n_spikes"] >= 5}
    summary = {
        "n_traces": n_traces,
        "n_distinct_culprit_items": len(items),
        "total_spikes": total_spikes,
        "spikes_from_repeat_offenders": sum(d["n_spikes"] for d in repeat),
        "repeat_offender_items": len(repeat),
        **tiers,
        "n_classes": len(classes),
        "classes_with_robust_offender": len(classes_with_offender),
        "bad_vote_share": round(sum(d["n_spikes"] for d in items if d["gt_label"] == "bad") / total_spikes, 3)
        if total_spikes
        else None,
        # A Bad culprit the model scored high (>0.5) is a hard/false negative.
        "hard_neg_share": round(
            sum(d["n_spikes"] for d in items if d["gt_label"] == "bad" and (d["mean_surface_score"] or 0) > 0.5)
            / total_spikes,
            3,
        )
        if total_spikes
        else None,
    }
    return {"summary": summary, "items": items}


def _counts(vals) -> dict:
    c: dict = defaultdict(int)
    for v in vals:
        c[str(v)] += 1
    return dict(c)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-item spike analysis (#2790).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    res = analyze(root, args.thresh)
    out = Path(args.out) if args.out else root / "spike_items.json"
    out.write_text(json.dumps(res, indent=2))

    s = res["summary"]
    print(f"traces={s['n_traces']}  total_spikes={s['total_spikes']}  "
          f"distinct culprit items={s['n_distinct_culprit_items']}")  # fmt: skip
    print(f"repeat-offender items (≥2 spikes)={s['repeat_offender_items']} "
          f"→ {s['spikes_from_repeat_offenders']}/{s['total_spikes']} of spikes")  # fmt: skip
    print(f"bad-vote culprits={s['bad_vote_share']:.0%}   "
          f"hard/false-negative (bad but surf>0.5)={s['hard_neg_share']:.0%}")  # fmt: skip
    print(f"\nTop {args.top} spike-causing items (cls / id / label / n_spikes / n_votes / rate / surf / seeds):")
    for d in res["items"][: args.top]:
        print(
            f"  {d['cls']:<14} {d['image_id']:>7} {d['gt_label']:<4} "
            f"spk={d['n_spikes']:>2} votes={d['n_votes']:>2} rate={d['spike_rate']} "
            f"surf={d['mean_surface_score']} n_good={d['mean_n_good']} seeds={len(d['seeds'])}"
        )
    print(f"\n[-> {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

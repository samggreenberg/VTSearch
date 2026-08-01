"""Per-EVENT spike analysis (#2790): when does a Bad vote spike, and when does a spiker NOT?

Earlier analysis treated spikers as *items*. But a spike is really a **(class, seed, step)
VOTE EVENT**: the same item is voted Bad in many trajectories and spikes in only some of
them. This tool rebuilds every Bad-vote event from the ``--labeling-trace`` ``trace.json``
files and records the **state at vote time** — no re-run needed, it's all in the traces:

* loop position: ``n_good``, ``n_bad``, ``t``, ``phase``, ``select_mode``;
* the surfaced item's ``surface_score`` and ``surface_margin`` (score minus the cut);
* **context geometry** against the *labeled-so-far* sets (what the model has actually
  seen): nearest labeled bad (``nb_cos_max``), context good/bad centroids and margin.
  The static "distance to all bads" said spikers aren't outliers; the *context* distance
  to the handful of already-labeled bads is the quantity the intuition is really about.

and whether the vote **spiked** (``Δcost > thresh``). Then three analyses:

  A. conditional spike rates -> which conditions are *necessary* (spike rate ~0 when
     violated) and how far short of *sufficient* they fall (max spike rate when all hold);
  B. within-item -> for items that spike in >=K seeds, compare the occurrences where the
     SAME item spiked vs didn't (holds the item fixed, isolates the context);
  C. logistic fit on standardized features -> coefficients ranking the drivers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sod"))  # noqa: E402
from spike_analysis import _class_seed, _f  # type: ignore  # noqa: E402


def _load_vec(regions: Path, iid) -> np.ndarray | None:
    p = regions / f"{iid}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        if "whole_vec" not in z:
            return None
        v = np.asarray(z["whole_vec"], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-12) if n else v


def build_events(root: Path, regions: Path, thresh: float) -> list[dict]:
    """One row per Bad-vote event across all traces, with state-at-vote-time + geometry."""
    events: list[dict] = []
    for tj in sorted(root.rglob("trace.json")):
        cls, seed = _class_seed(tj)
        trace = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        # cache vectors for the (<=60) voted ids in this trace
        vcache: dict = {}
        for e in trace:
            iid = e.get("image_id")
            if iid is not None and iid not in vcache:
                vcache[iid] = _load_vec(regions, iid)
        good_ids: list = []  # labeled BEFORE the current step (running)
        bad_ids: list = []
        for i, cur in enumerate(trace):
            label = cur.get("gt_label")
            iid = cur.get("image_id")
            if i >= 1 and label == "bad":
                prev = trace[i - 1]
                dcost = _f(cur.get("cost")) - _f(prev.get("cost"))
                v = vcache.get(iid)
                # context geometry vs the sets labeled BEFORE this vote
                gv = [vcache[g] for g in good_ids if vcache.get(g) is not None]
                bv = [vcache[b] for b in bad_ids if vcache.get(b) is not None]
                nb_cos_max = nb_cos_mean = ng_cos_max = ctx_cos_G = ctx_cos_B = ctx_margin = float("nan")
                if v is not None:
                    if bv:
                        bsim = np.array([float(v @ b) for b in bv])
                        nb_cos_max, nb_cos_mean = float(bsim.max()), float(bsim.mean())
                        bc = np.mean(bv, axis=0)
                        ctx_cos_B = float(v @ (bc / (np.linalg.norm(bc) + 1e-12)))
                    if gv:
                        gsim = np.array([float(v @ g) for g in gv])
                        ng_cos_max = float(gsim.max())
                        gc = np.mean(gv, axis=0)
                        ctx_cos_G = float(v @ (gc / (np.linalg.norm(gc) + 1e-12)))
                    if gv and bv:
                        ctx_margin = ctx_cos_G - ctx_cos_B
                events.append(
                    {
                        "cls": cls,
                        "seed": seed,
                        "t": cur.get("t"),
                        "image_id": iid,
                        "is_spike": int(dcost > thresh),
                        "dcost": round(dcost, 4),
                        "d_fnr": round(_f(cur.get("fnr")) - _f(prev.get("fnr")), 4),
                        "d_fpr": round(_f(cur.get("fpr")) - _f(prev.get("fpr")), 4),
                        "n_good": prev.get("n_good"),
                        "n_bad": prev.get("n_bad"),
                        "phase": cur.get("phase"),
                        "select_mode": cur.get("select_mode"),
                        "calib_mode": cur.get("calib_mode"),
                        "surface_score": _f(cur.get("surface_score")),
                        "surface_margin": _f(cur.get("surface_margin")),
                        "nb_cos_max": nb_cos_max,
                        "nb_cos_mean": nb_cos_mean,
                        "ng_cos_max": ng_cos_max,
                        "ctx_cos_G": ctx_cos_G,
                        "ctx_cos_B": ctx_cos_B,
                        "ctx_margin": ctx_margin,
                    }
                )
            # advance the labeled sets to include this vote
            if label == "good":
                good_ids.append(iid)
            elif label == "bad":
                bad_ids.append(iid)
    return events


def _rate(rows: list[dict], cond) -> tuple[int, int, float]:
    sub = [r for r in rows if cond(r)]
    k = sum(r["is_spike"] for r in sub)
    return k, len(sub), (k / len(sub) if sub else float("nan"))


def _bin_table(rows: list[dict], key: str, edges: list[float]) -> str:
    lines = [f"  spike rate by {key}:"]
    labels = [f"<{edges[0]}"] + [f"[{edges[j]},{edges[j + 1]})" for j in range(len(edges) - 1)] + [f">={edges[-1]}"]
    bins: list = [[] for _ in labels]
    for r in rows:
        x = r.get(key)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        b = 0
        while b < len(edges) and x >= edges[b]:
            b += 1
        bins[b].append(r)
    for lab, b in zip(labels, bins):
        if b:
            k = sum(rr["is_spike"] for rr in b)
            lines.append(f"    {lab:<12} {k:>4}/{len(b):<5} = {k / len(b):.3f}")
    return "\n".join(lines)


def within_item(events: list[dict], min_seeds: int) -> tuple[list[dict], str]:
    """For items that spike in >=min_seeds seeds, compare spike vs no-spike occurrences."""
    by_item: dict = defaultdict(list)
    for e in events:
        by_item[(e["cls"], e["image_id"])].append(e)
    feats = ["n_good", "n_bad", "t", "surface_margin", "nb_cos_max", "ng_cos_max", "ctx_margin", "ctx_cos_B"]
    diffs: dict = defaultdict(list)
    kept = 0
    ambivalent = 0  # items that both spike AND don't-spike across their occurrences
    for _, occ in by_item.items():
        n_sp = sum(o["is_spike"] for o in occ)
        if n_sp < min_seeds:
            continue
        kept += 1
        sp = [o for o in occ if o["is_spike"]]
        no = [o for o in occ if not o["is_spike"]]
        if sp and no:
            ambivalent += 1
            for f in feats:
                a = [o[f] for o in sp if o[f] is not None and not (isinstance(o[f], float) and np.isnan(o[f]))]
                b = [o[f] for o in no if o[f] is not None and not (isinstance(o[f], float) and np.isnan(o[f]))]
                if a and b:
                    diffs[f].append(statistics.fmean(a) - statistics.fmean(b))
    lines = [
        f"  robust spiker items (>={min_seeds} spike-seeds): {kept}; of those, "
        f"{ambivalent} ALSO have no-spike occurrences (same item, both outcomes).",
        "  mean(spike) - mean(no-spike) within the same item, averaged over ambivalent items:",
    ]
    for f in feats:
        if diffs[f]:
            lines.append(f"    Δ{f:<15} {statistics.fmean(diffs[f]):+.4f}   (n_items={len(diffs[f])})")
    return [], "\n".join(lines)


def logistic(events: list[dict]) -> str:
    try:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    except ImportError:
        return "  (sklearn unavailable — skipping logistic fit)"
    feats = ["n_good", "n_bad", "t", "surface_score", "surface_margin", "nb_cos_max", "ng_cos_max", "ctx_margin"]
    X, y = [], []
    for e in events:
        row = [e.get(f) for f in feats]
        vals = [float(v) if v is not None else np.nan for v in row]
        # impute "no labeled bad yet" (nb_cos_max NaN) as -1 (maximally far) + flag
        no_bad = 1.0 if np.isnan(vals[feats.index("nb_cos_max")]) else 0.0
        vals = [(-1.0 if (i == feats.index("nb_cos_max") and np.isnan(v)) else v) for i, v in enumerate(vals)]
        if any(np.isnan(v) for v in vals):
            continue
        X.append(vals + [no_bad])
        y.append(e["is_spike"])
    if len(set(y)) < 2:
        return "  (only one class present — skipping logistic fit)"
    names = feats + ["no_prior_bad"]
    Xs = StandardScaler().fit_transform(np.array(X))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs, y)
    coefs = sorted(zip(names, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    lines = [f"  logistic (standardized, class-balanced) on n={len(y)} events, {sum(y)} spikes:",
             "  coef>0 => raises spike odds:"]  # fmt: skip
    for nm, c in coefs:
        lines.append(f"    {nm:<15} {c:+.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-event spike analysis (#2790).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True, help="cache dir (for regions/<ds>/<embedder>/whole)")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--embedder", default="siglip")
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--out", default=None, help="write events as JSON")
    args = ap.parse_args(argv)

    regions = Path(args.cache) / "regions" / args.dataset / args.embedder / "whole"
    events = build_events(Path(args.root), regions, args.thresh)
    if args.out:
        Path(args.out).write_text(json.dumps(events))

    n = len(events)
    n_sp = sum(e["is_spike"] for e in events)
    print(f"Bad-vote events: {n}   spikes: {n_sp}   base spike rate: {n_sp / n:.3f}\n")

    print("== A. Conditional spike rates ==")
    print(_bin_table(events, "n_good", [3, 4, 6, 9, 15]))
    print(_bin_table(events, "n_bad", [1, 2, 4, 8]))
    print(_bin_table(events, "surface_margin", [-0.5, -0.1, 0.0, 0.1, 0.5]))
    print(_bin_table(events, "nb_cos_max", [0.2, 0.4, 0.5, 0.6, 0.7]))
    print(_bin_table(events, "ctx_margin", [-0.1, 0.0, 0.1, 0.2]))
    for name, cond in [
        ("select_mode == hard", lambda r: r["select_mode"] == "hard"),
        ("select_mode != hard", lambda r: r["select_mode"] != "hard"),
        ("surface_margin >= 0 (item AT/ABOVE cut)", lambda r: _f(r["surface_margin"]) >= 0),
        ("surface_margin < 0 (item below cut)", lambda r: _f(r["surface_margin"]) < 0),
        ("first bad (n_bad==0)", lambda r: (r["n_bad"] or 0) == 0),
        ("first bad AND margin>=0", lambda r: (r["n_bad"] or 0) == 0 and _f(r["surface_margin"]) >= 0),
        ("sparse pos (n_good<=4)", lambda r: (r["n_good"] or 0) <= 4),
        ("margin>=0 AND n_good<=4", lambda r: _f(r["surface_margin"]) >= 0 and (r["n_good"] or 0) <= 4),
        (
            "margin>=0 AND n_good<=4 AND hard",
            lambda r: _f(r["surface_margin"]) >= 0 and (r["n_good"] or 0) <= 4 and r["select_mode"] == "hard",
        ),  # noqa: E501
        ("margin<0 AND NOT first-bad (safe?)", lambda r: _f(r["surface_margin"]) < 0 and (r["n_bad"] or 0) > 0),
    ]:
        k, tot, rate = _rate(events, cond)
        print(f"  {name:<38} {k:>4}/{tot:<6} = {rate:.3f}")

    print("\n== B. Within-item: when does the SAME item spike vs not ==")
    _, wtxt = within_item(events, args.min_seeds)
    print(wtxt)

    print("\n== C. Logistic drivers ==")
    print(logistic(events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

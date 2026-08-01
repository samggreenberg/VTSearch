"""Attribute the #2790 cost spikes to the individual votes that cause them.

Reads ``--labeling-trace`` ``trace.json`` files (one per class/seed) and finds every
step where the held-out **cost jumps UP** by more than ``--thresh`` — the single-step
excursions #2790 reports. Each up-spike is attributed to the vote added *that* step
(the image whose label triggered the retrain), and the tool aggregates the culprits
to look for a blockable pattern:

* is the culprit a **good** or a **bad** vote?
* which way does the **threshold** move (up = reject more -> FNR spike)?
* is the spike **FNR-driven** (cut jumped above real matches) or **FPR-driven**?
* is it **narrow** (recovers the next step, as #2790 observed)?
* where in the loop (vote count ``t``), which ``calib_mode`` / ``phase``, and how
  marginal was the culprit's surfacing score?

Usage: ``python spike_analysis.py --root <dir with labeling_trace/...> [--thresh 0.1]``
Writes a summary to stdout and ``spike_rows.csv`` next to ``--root``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _class_seed(trace_path: Path) -> tuple[str, str]:
    # .../labeling_trace/<slug>/coco_<class>_<embedder>_<proposal>[_a<alpha>]/seed<N>/trace.json
    import re  # noqa: PLC0415

    seed = trace_path.parent.name.removeprefix("seed")
    cfg = trace_path.parent.parent.name
    # Strip the coco_ prefix and the _<embedder>_<proposal>... suffix so the class is
    # comparable across embedders/proposals (whole/siglip2 vs hac/dinov3, etc.).
    m = re.match(r"^coco_(.*?)_(siglip2|siglip|dinov2|dinov3|clip)_(whole|hac|sliding|dino)", cfg)
    cls = m.group(1) if m else cfg.replace("coco_", "").replace("_siglip2_whole", "")
    return cls, seed


def collect_spikes(root: Path, thresh: float) -> list[dict]:
    rows: list[dict] = []
    for tj in sorted(root.rglob("trace.json")):
        cls, seed = _class_seed(tj)
        trace = sorted(json.loads(tj.read_text()), key=lambda e: e["t"])
        for i in range(1, len(trace)):
            prev, cur = trace[i - 1], trace[i]
            dcost = _f(cur.get("cost")) - _f(prev.get("cost"))
            if not (dcost > thresh):  # up-spikes only (NaN-safe)
                continue
            nxt = trace[i + 1] if i + 1 < len(trace) else None
            recover = _f(nxt.get("cost")) - _f(cur.get("cost")) if nxt else float("nan")
            # Steps until cost returns near the pre-spike level (within 0.05), up to 5
            # steps out — measures how "narrow" the excursion actually is.
            pre_cost = _f(prev.get("cost"))
            recover_steps = None
            for k in range(1, 6):
                if i + k < len(trace) and _f(trace[i + k].get("cost")) <= pre_cost + 0.05:
                    recover_steps = k
                    break
            rows.append(
                {
                    "cls": cls,
                    "seed": seed,
                    "t": cur.get("t"),
                    "culprit_label": cur.get("gt_label"),
                    "culprit_id": cur.get("image_id"),
                    "select_mode": cur.get("select_mode"),
                    "phase": cur.get("phase"),
                    "calib_mode": cur.get("calib_mode"),
                    "dcost": round(dcost, 4),
                    "d_threshold": round(_f(cur.get("threshold")) - _f(prev.get("threshold")), 4),
                    "d_fnr": round(_f(cur.get("fnr")) - _f(prev.get("fnr")), 4),
                    "d_fpr": round(_f(cur.get("fpr")) - _f(prev.get("fpr")), 4),
                    "recover_dcost": round(recover, 4),
                    "recover_steps": recover_steps,  # None = still elevated 5 steps later
                    "narrow": recover_steps is not None and recover_steps <= 2,
                    "surface_score": cur.get("surface_score"),
                    "surface_margin": cur.get("surface_margin"),
                    "n_good": prev.get("n_good"),
                    "n_bad": prev.get("n_bad"),
                }
            )
    return rows


def summarize(rows: list[dict], n_traces: int, thresh: float) -> str:
    if not rows:
        return f"No up-spikes (Δcost > {thresh}) found across {n_traces} traces."
    n = len(rows)

    def pct(cond) -> str:
        k = sum(1 for r in rows if cond(r))
        return f"{k}/{n} ({100 * k / n:.0f}%)"

    good = pct(lambda r: r["culprit_label"] == "good")
    bad = pct(lambda r: r["culprit_label"] == "bad")
    thr_up = pct(lambda r: r["d_threshold"] > 0)
    fnr_driven = pct(lambda r: r["d_fnr"] > r["d_fpr"])
    narrow = pct(lambda r: r["narrow"])
    rec = Counter(("never>5" if r["recover_steps"] is None else r["recover_steps"]) for r in rows)
    early = pct(lambda r: (r["t"] or 0) < 20)
    bad_hard = pct(lambda r: r["culprit_label"] == "bad" and r["select_mode"] == "hard")
    sparse_pos = pct(lambda r: (r["n_good"] or 0) <= 6)
    runaway = pct(lambda r: _f(r.get("d_fnr")) > 0.2)  # cut jumped over a big chunk of positives
    goods = sorted(r["n_good"] for r in rows if r["n_good"] is not None)
    med_good = goods[len(goods) // 2] if goods else "-"
    by_phase = Counter(r["phase"] for r in rows)
    by_calib = Counter(r["calib_mode"] for r in rows)
    by_select = Counter(r["select_mode"] for r in rows)
    by_cls = Counter(r["cls"] for r in rows)
    ts = sorted(r["t"] for r in rows if r["t"] is not None)
    worst = sorted(rows, key=lambda r: -r["dcost"])[:8]

    lines = [
        f"# Spike attribution — {n} up-spikes (Δcost > {thresh}) across {n_traces} traces",
        "",
        f"Culprit vote label:   good={good}   bad={bad}",
        f"  ... bad vote AND hard-selected: {bad_hard}",
        f"Raw threshold 'up':   {thr_up}   (NOT model-comparable — MLP retrained each vote; "
        f"use the FNR operating-point move below, which is test-set and sound)",
        f"FNR-driven (Δfnr>Δfpr): {fnr_driven}",
        f"Early (t<20): {early}",
        f"Sparse positives at spike (n_good<=6): {sparse_pos}   (median n_good at spike = {med_good})",
        f"Runaway (Δfnr>0.2 — cut vaulted over many positives): {runaway}",
        f"Narrow (recovers to pre-spike within 2 steps): {narrow}",
        f"Recovery-steps distribution: {dict(sorted(rec.items(), key=lambda kv: str(kv[0])))}",
        "",
        f"By phase:  {dict(by_phase.most_common())}",
        f"By calib_mode: {dict(by_calib.most_common())}",
        f"By select_mode: {dict(by_select.most_common())}",
        f"By class:  {dict(by_cls.most_common())}",
        f"Vote-count t: min={ts[0] if ts else '-'} median={ts[len(ts) // 2] if ts else '-'} max={ts[-1] if ts else '-'}",
        "",
        "Worst spikes (by Δcost):",
        "  cls / seed / t / label / Δcost / Δthr / Δfnr / Δfpr / narrow / calib / phase / surf_margin",
    ]
    for r in worst:
        lines.append(
            f"  {r['cls']} s{r['seed']} t{r['t']} {r['culprit_label']} "
            f"Δcost={r['dcost']} Δthr={r['d_threshold']} Δfnr={r['d_fnr']} Δfpr={r['d_fpr']} "
            f"narrow={r['narrow']} {r['calib_mode']} {r['phase']} surf_margin={r['surface_margin']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Attribute #2790 cost spikes to their votes.")
    ap.add_argument("--root", required=True, help="Dir containing labeling_trace/.../trace.json")
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--out", default=None, help="CSV of spike rows (default <root>/spike_rows.csv)")
    args = ap.parse_args(argv)

    root = Path(args.root)
    n_traces = sum(1 for _ in root.rglob("trace.json"))
    rows = collect_spikes(root, args.thresh)
    out = Path(args.out) if args.out else root / "spike_rows.csv"
    if rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(summarize(rows, n_traces, args.thresh))
    print(f"\n[{len(rows)} rows -> {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

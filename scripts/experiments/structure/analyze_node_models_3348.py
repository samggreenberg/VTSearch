#!/usr/bin/env python
"""Analyse the #3348 node-model run against its pre-registration.

    python analyze_node_models_3348.py --results DIR --out DIR

Reads every ``pit_*.csv`` / ``shift_*.csv`` / ``timing_*.json`` the run wrote and
emits, per (embedder, variant), the PREREG measures -- median KS, rate below
0.05, false alarm, detection, separation -- then applies the decision rule.
Writes ``table.csv`` (every variant), ``verdict.json`` (gates, passers, winner,
P1-P4) and ``tables.md`` (the report's tables).  Figures: ``figures_node_models_3348.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: Source of each dataset, for the different-/same-source split (PREREG "Grid").
SOURCES = {
    "coco_val": "coco",
    "coco_better": "coco",
    "visual_genome_m": "vg",
    "caltech101_m": "caltech",
    "o365_better": "o365",
    # #3329's datasets, so the analyzer also reads that grid's shape.
    "vg_scale_any": "vg",
    "vg_box_large": "vg",
}

SHIPPED = "grid_mean_shipped"
CONTROL = "knn_global"
PATCH = "dinov3_patch"

# Decision rule, PREREG.md.
KS_MAX = 0.05
RATE_LO, RATE_HI = 0.03, 0.075
FALSE_ALARM_MAX = 0.10
PATCH_SEP_MIN = 0.50
SEP_SLACK = 0.10
#: PREREG tie rule: within this much of the best minimum separation, the cheaper
#: variant wins -- scalar node state (vmf) before node directions (mix) before
#: node principal axes (ppca) before node vectors at query time (knn).
TIE = 0.05
COST_RANK = {"vmf": 0, "mix": 1, "ppca": 2, "knn": 3}


def load(results: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pit = pd.concat([pd.read_csv(f) for f in sorted(results.glob("pit_*.csv"))], ignore_index=True)
    shift = pd.concat([pd.read_csv(f) for f in sorted(results.glob("shift_*.csv"))], ignore_index=True)
    timing = {f.stem: json.loads(f.read_text()) for f in sorted(results.glob("timing_*.json"))}
    return pit, shift, timing


def measures(pit: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    hold = pit[pit["scope"].isin(["holdout", "in_class_holdout"])]
    wide = hold.pivot_table(
        index=["embedder", "family", "dataset", "seed"], columns="statistic", values="value"
    ).reset_index()
    cal = (
        wide.groupby(["embedder", "family"])
        .agg(
            ks=("ks_uniform", "median"),
            rate05=("frac_below_0.05", "mean"),
            sd=("sd", "mean"),
            cells=("ks_uniform", "size"),
        )
        .reset_index()
        .rename(columns={"family": "variant"})
    )

    s = shift.copy()
    s["src_b"] = s["build_dataset"].map(SOURCES)
    s["src_q"] = s["query_dataset"].map(SOURCES)
    s["kind"] = np.where(s["is_self"], "self", np.where(s["src_b"] == s["src_q"], "same_source", "diff_source"))
    rates = s.pivot_table(index=["embedder", "variant"], columns="kind", values="shifted", aggfunc="mean").reset_index()
    counts = s.pivot_table(
        index=["embedder", "variant"], columns="kind", values="shifted", aggfunc="size"
    ).reset_index()
    frac = s.pivot_table(
        index=["embedder", "variant"], columns="kind", values="frac_atypical", aggfunc="mean"
    ).reset_index()
    rates = rates.rename(columns={"self": "false_alarm", "diff_source": "detect", "same_source": "same_source_fire"})
    counts = counts.rename(columns={"self": "n_self", "diff_source": "n_diff", "same_source": "n_same"})
    frac = frac.rename(columns={"self": "frac_self", "diff_source": "frac_diff", "same_source": "frac_same"})
    out = rates.merge(counts, on=["embedder", "variant"]).merge(frac, on=["embedder", "variant"])
    out["separation"] = out["detect"] - out["false_alarm"]
    return cal.merge(out, on=["embedder", "variant"], how="outer")


def decide(table: pd.DataFrame) -> dict:
    t = table.set_index(["embedder", "variant"])
    embedders = sorted(table["embedder"].unique())
    variants = sorted(v for v in table["variant"].unique() if v != "class_conformal")
    shipped_sep = {e: float(t.loc[(e, SHIPPED), "separation"]) for e in embedders if (e, SHIPPED) in t.index}
    gates: dict[str, dict] = {}
    for v in variants:
        fails: list[str] = []
        for e in embedders:
            if (e, v) not in t.index:
                fails.append(f"{e}: missing")
                continue
            r = t.loc[(e, v)]
            if not r["ks"] <= KS_MAX:
                fails.append(f"{e}: KS {r['ks']:.3f}")
            if not RATE_LO <= r["rate05"] <= RATE_HI:
                fails.append(f"{e}: rate@0.05 {r['rate05']:.3f}")
            if not r["false_alarm"] <= FALSE_ALARM_MAX:
                fails.append(f"{e}: false alarm {r['false_alarm']:.2f}")
            if e == PATCH:
                if not r["separation"] >= PATCH_SEP_MIN:
                    fails.append(f"{e}: separation {r['separation']:.2f}")
            elif e in shipped_sep and not r["separation"] >= shipped_sep[e] - SEP_SLACK:
                fails.append(f"{e}: separation {r['separation']:.2f} < shipped {shipped_sep[e]:.2f} - {SEP_SLACK}")
        min_sep = float(table[table["variant"] == v]["separation"].min())
        gates[v] = {"passes": not fails, "fails": fails, "min_separation": min_sep}
    passers = sorted((v for v, g in gates.items() if g["passes"]), key=lambda v: -gates[v]["min_separation"])
    tree_passers = [v for v in passers if v != CONTROL]
    best_tree = None
    if tree_passers:
        top = gates[tree_passers[0]]["min_separation"]
        tied = [v for v in tree_passers if gates[v]["min_separation"] >= top - TIE]
        best_tree = min(tied, key=lambda v: (COST_RANK[v.split("_")[0]], -gates[v]["min_separation"]))

    def row(e: str, v: str) -> dict:
        if (e, v) not in t.index:
            return {}
        r = t.loc[(e, v)]
        return {k: float(r[k]) for k in ("ks", "rate05", "false_alarm", "detect", "separation") if pd.notna(r[k])}

    patch = {v: row(PATCH, v) for v in variants}
    preds = {}
    s = patch.get(SHIPPED, {})
    others_fa = [row(e, SHIPPED).get("false_alarm", np.nan) for e in embedders if e != PATCH]
    preds["P1"] = bool(s and s["rate05"] > 0.10 and s["separation"] < 0.30 and np.nanmax(others_fa) <= 0.2)
    c = [row(e, CONTROL) for e in embedders]
    preds["P2"] = bool(c and all(r and r["ks"] <= 0.05 and 0.035 <= r["rate05"] <= 0.065 for r in c))
    vs = patch.get("vmf_split_deepest", {})
    preds["P3"] = bool(vs and 0.035 <= vs["rate05"] <= 0.065)
    calibrated = {
        v: r for v, r in patch.items() if r and 0.035 <= r.get("rate05", 1) <= 0.065 and r.get("ks", 1) <= KS_MAX
    }
    knn_best = max(
        (r["separation"] for v, r in calibrated.items() if v.startswith("knn_") and v != CONTROL), default=np.nan
    )
    vmf_best = max((r["separation"] for v, r in calibrated.items() if v.startswith("vmf_")), default=np.nan)
    preds["P4"] = bool(np.isfinite(knn_best) and np.isfinite(vmf_best) and knn_best - vmf_best >= 0.10)
    return {
        "gates": gates,
        "passers": passers,
        "tree_passers": tree_passers,
        # The highest minimum separation of all, control included, and the tree
        # variant the tie rule picks; PREREG sends a knn_global win to the owner.
        "top": passers[0] if passers else None,
        "winner": best_tree if best_tree else (CONTROL if CONTROL in passers else None),
        "predictions": preds,
        "patch_best_calibrated": {"knn": knn_best, "vmf": vmf_best},
        "shipped_separation": shipped_sep,
    }


def fmt_md(table: pd.DataFrame, verdict: dict) -> str:
    cols = ["variant", "ks", "rate05", "false_alarm", "detect", "same_source_fire", "separation"]
    lines = []
    for e in sorted(table["embedder"].unique()):
        sub = table[table["embedder"] == e].sort_values("separation", ascending=False)
        lines.append(f"\n### {e}\n")
        lines.append("| " + " | ".join(cols) + " | passes |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for _, r in sub.iterrows():
            cells = [str(r["variant"])] + ["" if pd.isna(r.get(k)) else f"{r[k]:.3f}" for k in cols[1:]]
            ok = verdict["gates"].get(r["variant"], {}).get("passes")
            lines.append("| " + " | ".join(cells) + f" | {'yes' if ok else ''} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    pit, shift, timing = load(Path(args.results))
    table = measures(pit, shift)
    verdict = decide(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table.csv", index=False)
    q = pd.DataFrame([{"task": k, **v["query_seconds"]} for k, v in timing.items()])
    if not q.empty:
        q.to_csv(out / "query_seconds.csv", index=False)
    fit = pit[pit["family"] == "fit"].pivot_table(index="scope", columns="statistic", values="value", aggfunc="sum")
    fit.to_csv(out / "fit_totals.csv")
    verdict["tasks"] = sorted(timing)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float))
    (out / "tables.md").write_text(fmt_md(table, verdict))
    print(json.dumps({k: verdict[k] for k in ("passers", "top", "winner", "predictions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

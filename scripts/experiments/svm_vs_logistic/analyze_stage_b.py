"""#3197 Stage B analysis: the heads inside the Autopilot loop.

Reads each arm's base rows (``_cells_io.load_arm``, the production rows only)
and pairs every arm against the shipped ``svm`` arm on the CELL - (dataset,
embedder, category, seed) - never the step: a different head collects different
votes from its first retrain, so two arms share a cell and nothing finer.

The decomposition the issue asks for, read straight off the rows:

* ``average_precision`` / ``auroc`` / ``oracle_cost`` are threshold-free (the
  last is the best cut the test labels allow) - **the ranking**;
* ``regret = cost - oracle_cost`` is what the shipped cut loses against that
  best cut - **the threshold**;
* ``cost`` is both.

Every difference is reported as ``other - svm`` (so for cost, regret and
oracle cost a POSITIVE number means the shipped SVM is better; for AP/AUROC a
NEGATIVE one does), with a standard error clustered on (dataset, embedder,
category) because five seeds of one category are not five independent cells.

    python analyze_stage_b.py --root /expscratch/$USER/svmlog-3197/stageB --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))

ARMS = ("svm", "linear", "mlp", "linconv", "svmc01", "svmc10")
CHECKPOINTS = (10, 20, 40, 80, 150)
METRICS = ("cost", "oracle_cost", "regret", "average_precision", "auroc", "f1", "fnr", "fpr", "n_good")
KEYS = ["dataset", "embedder", "category", "seed"]
CLUSTER = ["dataset", "embedder", "category"]


def env_of(df: pd.DataFrame) -> pd.Series:
    return (
        df["dataset"].str.replace("visual_genome_m", "vg").str.replace("caltech101_m", "caltech") + "/" + df["embedder"]
    )


def load(root: Path) -> tuple[pd.DataFrame, dict]:
    import _cells_io

    parts, prov = [], {}
    for arm in ARMS:
        d = root / arm / "results"
        if not (d / "cells").exists():
            continue
        df, p = _cells_io.load_arm(d)
        prov[arm] = {k: (len(v) if isinstance(v, list) else v) for k, v in p.items()}
        if df.empty:
            continue
        cols = [c for c in [*KEYS, "t", *METRICS, "threshold", "head", "phase"] if c in df.columns]
        df = df[cols].copy()
        df["arm"] = arm
        parts.append(df)
    return pd.concat(parts, ignore_index=True), prov


def at_checkpoint(df: pd.DataFrame, t: int) -> pd.DataFrame:
    """Each cell's row at click ``t`` (the last row at or before it)."""
    sub = df[df["t"] <= t]
    return sub.sort_values("t").groupby([*KEYS, "arm"], as_index=False).tail(1)


def paired(df: pd.DataFrame, metric: str, arm: str, ref: str = "svm") -> dict:
    p = df.pivot_table(index=KEYS, columns="arm", values=metric)
    if arm not in p or ref not in p:
        return {"n": 0}
    x = (p[arm] - p[ref]).dropna()
    if x.empty:
        return {"n": 0}
    g = x.groupby(level=CLUSTER).mean()
    se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
    return {
        "mean": float(x.mean()),
        "se": se,
        "n": int(len(x)),
        "n_clusters": int(len(g)),
        "frac_ref_better": float(np.mean(x > 0))
        if metric in ("cost", "oracle_cost", "regret", "fnr", "fpr")
        else float(np.mean(x < 0)),
    }


def aulc(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of *metric* over clicks 1..150 per cell (area under the curve / 150).

    A cell's curve is forward-filled across clicks with no row (there is a row
    at every step once both classes exist); clicks before the first trainable
    step are excluded identically for every arm on the cell by taking the union
    of steps... they differ by arm, so instead every cell is read over the
    SAME window, the clicks at which ALL arms of that cell have a row.
    """
    w = df.pivot_table(index=[*KEYS, "t"], columns="arm", values=metric)
    w = w.dropna()
    return w.groupby(level=KEYS).mean().reset_index().melt(id_vars=KEYS, var_name="arm", value_name=metric)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, prov = load(args.root)
    df["env"] = env_of(df)
    arms = [a for a in ARMS if a in set(df["arm"])]
    rows = []
    for t in CHECKPOINTS:
        snap = at_checkpoint(df, t)
        for env in [*sorted(snap["env"].unique()), "ALL"]:
            s = snap if env == "ALL" else snap[snap["env"] == env]
            for arm in arms:
                if arm == "svm":
                    continue
                for m in METRICS:
                    if m not in s:
                        continue
                    r = paired(s, m, arm)
                    if r["n"]:
                        rows.append({"window": f"t{t}", "env": env, "arm": arm, "metric": m, **r})
    for m in ("cost", "oracle_cost", "regret", "average_precision", "auroc"):
        a = aulc(df, m)
        a["env"] = env_of(a)
        for env in [*sorted(a["env"].unique()), "ALL"]:
            s = a if env == "ALL" else a[a["env"] == env]
            for arm in arms:
                if arm == "svm":
                    continue
                r = paired(s, m, arm)
                if r["n"]:
                    rows.append({"window": "aulc", "env": env, "arm": arm, "metric": m, **r})
    out = pd.DataFrame(rows)
    out.to_csv(args.out / "stageB_paired.csv", index=False)

    levels = []
    for t in CHECKPOINTS:
        snap = at_checkpoint(df, t)
        g = snap.groupby("arm")[[m for m in METRICS if m in snap]].mean()
        g["window"] = f"t{t}"
        levels.append(g.reset_index())
    pd.concat(levels).to_csv(args.out / "stageB_levels.csv", index=False)
    (args.out / "stageB_provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    def fmt(r):
        return f"{r['mean']:+.3f} ± {r['se']:.3f}"

    print("Stage B: other - svm, clustered SE (positive cost = SVM better; negative AP = SVM better)")
    for win in ("aulc", "t20", "t40", "t150"):
        sub = out[(out.window == win) & (out.env == "ALL")]
        print(f"\n[{win}]")
        print(sub.pivot_table(index="arm", columns="metric", values="mean", aggfunc="first").round(4).to_string())
        print(sub.pivot_table(index="arm", columns="metric", values="se", aggfunc="first").round(4).to_string())
    print(json.dumps(prov, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

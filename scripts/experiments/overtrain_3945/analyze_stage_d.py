"""#3945 Stage D analysis: 400-click Autopilot sessions, shipped C = 1 and C = 0.1.

    python analyze_stage_d.py --root /expscratch/$USER/overtrain-3945/stageD \
        --prefix-root /expscratch/$USER/svmlog-3197/stageB --out DIR

Four readings, in PLAN.md's order of dependence:

* ``D_prefix.csv``   validity: does each 400-click session's first 150 picks
                     reproduce #3197's 150-click session exactly?  A mismatch
                     breaks nothing below (every contrast here is within this
                     run), but it is reported, not patched.
* ``D_paired.csv``   ``svmc01 - svm`` at click checkpoints and over windows,
                     paired on the session, SE clustered on (dataset, embedder,
                     category).  Ranking (``oracle_cost``), cut (``regret``), both.
* ``D_curve.csv``    mean metric per click per arm per dataset: is there a
                     late turn-up in the average session?
* ``D_degrade.csv``  per session, how much worse the last 50 clicks are than
                     the session's best 50-click stretch - the literal "more
                     votes made it worse", on a smoothed curve so one noisy
                     click cannot fake it.
* ``D_stop.csv``     the app's own stopping rules (``stopping.stopping_points``):
                     cost and oracle cost at the stop vs at click 400.  Does
                     clicking past the app's stop signal hurt?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "calibration"))

ARMS = ("svm", "svmc01")
KEYS = ["dataset", "embedder", "category", "seed"]
CLUSTER = ["dataset", "embedder", "category"]
METRICS = ("cost", "oracle_cost", "regret", "average_precision", "auroc", "n_good")
CHECKPOINTS = (20, 40, 80, 150, 200, 300, 400)
WINDOWS = {"1-150": (1, 150), "151-400": (151, 400), "1-400": (1, 400)}
SMOOTH = 50


def clustered(x: pd.Series) -> tuple[float, float, int]:
    x = x.dropna()
    if x.empty:
        return float("nan"), float("nan"), 0
    g = x.groupby(level=CLUSTER)
    sums, counts = g.sum(), g.count()
    n, k = counts.sum(), len(counts)
    mean = sums.sum() / n
    resid = sums - counts * mean
    se = float(np.sqrt(k / max(k - 1, 1) * (resid**2).sum()) / n) if k > 1 else float("nan")
    return float(mean), se, int(n)


def load(root: Path) -> pd.DataFrame:
    import _cells_io

    parts = []
    for arm in ARMS:
        d = root / arm / "results"
        df, prov = _cells_io.load_arm(d)
        print(f"{arm}: {prov}")
        cols = [c for c in [*KEYS, "t", "phase", *METRICS, "n_test_pos", "fpr", "fnr", "threshold"] if c in df.columns]
        parts.append(df[cols].assign(arm=arm))
    return pd.concat(parts, ignore_index=True)


def prefix_check(root: Path, prefix_root: Path) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        a = root / arm / "picks_all.csv.gz"
        b = prefix_root / arm / "picks_all.csv.gz"
        if not (a.exists() and b.exists()):
            continue
        pa = pd.read_csv(a)
        pb = pd.read_csv(b)
        pa = pa[pa["t"] <= 150]
        m = pa.merge(pb, on=[*KEYS, "t"], how="outer", suffixes=("_d", "_b"), indicator=True)
        same = (m["_merge"] == "both") & (m["picked_id_d"] == m["picked_id_b"])
        per = same.groupby([m[k] for k in KEYS]).all()
        first_diff = m[~same].groupby(KEYS)["t"].min()
        rows.append(
            {
                "arm": arm,
                "sessions": int(len(per)),
                "identical_first_150": int(per.sum()),
                "median_first_divergence": float(first_diff.median()) if len(first_diff) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--prefix-root", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.root)
    df["ds"] = (
        df["dataset"]
        .str.replace("visual_genome_m", "vg")
        .str.replace("caltech101_m", "caltech")
        .str.replace("coco_val", "coco")
    )

    if args.prefix_root is not None:
        pc = prefix_check(args.root, args.prefix_root)
        pc.to_csv(args.out / "D_prefix.csv", index=False)

    # ---- paired svmc01 - svm ------------------------------------------------------------
    rows = []
    wide = {m: df.pivot_table(index=[*KEYS, "t"], columns="arm", values=m) for m in METRICS}
    for m, w in wide.items():
        w = w.dropna()
        tt = w.index.get_level_values("t")
        for name, (lo, hi) in WINDOWS.items():
            sub = w[(tt >= lo) & (tt <= hi)].groupby(level=KEYS).mean()
            d = sub["svmc01"] - sub["svm"]
            for scope in ("all", *sorted(df["dataset"].unique())):
                x = d if scope == "all" else d[d.index.get_level_values("dataset") == scope]
                mean, se, n = clustered(x)
                rows.append({"window": name, "scope": scope, "metric": m, "mean": mean, "se": se, "n": n})
        for t in CHECKPOINTS:
            sub = w[tt == t].droplevel("t")
            d = sub["svmc01"] - sub["svm"]
            mean, se, n = clustered(d)
            rows.append({"window": f"t{t}", "scope": "all", "metric": m, "mean": mean, "se": se, "n": n})
    pd.DataFrame(rows).to_csv(args.out / "D_paired.csv", index=False)

    # ---- mean curves --------------------------------------------------------------------
    curve = df.groupby(["arm", "ds", "t"])[list(METRICS)].mean().reset_index()
    curve_all = df.groupby(["arm", "t"])[list(METRICS)].mean().reset_index().assign(ds="all")
    pd.concat([curve_all, curve]).to_csv(args.out / "D_curve.csv", index=False)
    # Every session's own curve, for the per-run figure (not committed; large).
    run_cols = [c for c in ["arm", *KEYS, "t", "cost", "oracle_cost", "n_good", "fpr", "fnr", "threshold"] if c in df]
    df[run_cols].to_csv(args.out / "D_runs.csv.gz", index=False)

    # ---- along-session degradation, smoothed ------------------------------------------------
    rows = []
    for (arm, *key), g in df.groupby(["arm", *KEYS]):
        g = g.sort_values("t").set_index("t")
        if g.index.max() < 400:
            continue
        for m in ("oracle_cost", "cost"):
            s = g[m].reindex(range(1, 401)).ffill().rolling(SMOOTH, min_periods=SMOOTH).mean().dropna()
            if s.empty:
                continue
            # #4121's prediction: past exhaustion the fused cut drifts into the
            # negatives, so a late cost rise should be an FPR rise, not an FNR one.
            fx = {}
            for r in ("fpr", "fnr"):
                if r in g.columns:
                    rr = g[r].reindex(range(1, 401)).ffill().rolling(SMOOTH, min_periods=SMOOTH).mean()
                    fx[f"{r}_last_minus_at_best"] = float(rr.iloc[-1] - rr.loc[s.idxmin()])
            rows.append(
                {
                    "arm": arm,
                    **dict(zip(KEYS, key, strict=True)),
                    "metric": m,
                    **fx,
                    "last_minus_best": float(s.iloc[-1] - s.min()),
                    "t_best_end": int(s.idxmin()),
                    "n_good_400": float(g["n_good"].iloc[-1]),
                    # The sim half holds about as many positives as the test half, so this
                    # is the share of findable positives the session had voted by click 400.
                    "harvest": float(g["n_good"].iloc[-1] / max(g["n_test_pos"].iloc[-1], 1)),
                }
            )
    deg = pd.DataFrame(rows)
    deg.to_csv(args.out / "D_degrade_sessions.csv", index=False)
    deg["ds"] = deg["dataset"]
    summ = []
    for (arm, m), g in deg.groupby(["arm", "metric"]):
        scopes = [("all", g), *list(g.groupby("ds"))]
        scopes += [("harvest>=0.8", g[g["harvest"] >= 0.8]), ("harvest<0.8", g[g["harvest"] < 0.8])]
        for scope, sub in scopes:
            summ.append(
                {
                    "arm": arm,
                    "metric": m,
                    "scope": scope,
                    "n": len(sub),
                    "share_worse_0.02": float((sub["last_minus_best"] > 0.02).mean()),
                    "share_worse_0.05": float((sub["last_minus_best"] > 0.05).mean()),
                    "median": float(sub["last_minus_best"].median()),
                    "share_best_before_150": float((sub["t_best_end"] <= 150).mean()),
                    "mean_dfpr": float(sub["fpr_last_minus_at_best"].mean())
                    if "fpr_last_minus_at_best" in sub
                    else np.nan,
                    "mean_dfnr": float(sub["fnr_last_minus_at_best"].mean())
                    if "fnr_last_minus_at_best" in sub
                    else np.nan,
                }
            )
    pd.DataFrame(summ).to_csv(args.out / "D_degrade.csv", index=False)

    # Literal rows for the late cost rise: per dataset, the two shipped-arm sessions
    # with the largest rise among those that harvested >= 80% of the positives, read
    # at the end of their best 50-click stretch and at click 400 (raw, unsmoothed).
    ex = []
    top = deg[(deg["arm"] == "svm") & (deg["metric"] == "cost") & (deg["harvest"] >= 0.8)]
    # Ranked by the RAW rise between the two rows printed, so a row pair never
    # shows a smoothing artefact the reader cannot see.
    svm_runs = df[df["arm"] == "svm"].set_index([*KEYS, "t"])["cost"]
    top = top.assign(
        raw_rise=[
            svm_runs.get((*[r[k] for k in KEYS], 400), np.nan)
            - svm_runs.get((*[r[k] for k in KEYS], int(r["t_best_end"])), np.nan)
            for _, r in top.iterrows()
        ]
    )
    top = top.sort_values("raw_rise", ascending=False).groupby("dataset").head(2)
    for _, r in top.iterrows():
        g = df[(df["arm"] == "svm") & np.logical_and.reduce([df[k] == r[k] for k in KEYS])].set_index("t")
        for t in (int(r["t_best_end"]), 400):
            row = g.loc[t]
            ex.append(
                {
                    **{k: r[k] for k in KEYS},
                    "t": t,
                    **{
                        c: row[c]
                        for c in ("n_good", "n_test_pos", "threshold", "fpr", "fnr", "cost", "oracle_cost")
                        if c in g
                    },
                }
            )
    pd.DataFrame(ex).to_csv(args.out / "D_examples.csv", index=False)

    # ---- the app's own stopping rules ---------------------------------------------------------
    from stopping import stopping_points

    rows = []
    for arm in ARMS:
        sp = stopping_points(df[df["arm"] == arm], keys=KEYS, metrics=("cost", "oracle_cost", "n_good"))
        sp.to_csv(args.out / f"D_stop_sessions_{arm}.csv", index=False)
        st = sp[sp["stopped"]].set_index(KEYS)
        for scope in ("all", *sorted(df["dataset"].unique())):
            sub = st if scope == "all" else st[st.index.get_level_values("dataset") == scope]
            r = {
                "arm": arm,
                "scope": scope,
                "sessions": int(len(sp) if scope == "all" else (sp["dataset"] == scope).sum()),
                "stopped": int(len(sub)),
                "median_t_stop": float(sub["t_stop"].median()) if len(sub) else np.nan,
            }
            for m in ("cost", "oracle_cost"):
                mean, se, n = clustered(sub[f"{m}_delta"])
                r[f"{m}_final_minus_stop"] = mean
                r[f"{m}_se"] = se
                r[f"{m}_share_worse_0.02"] = float((sub[f"{m}_delta"] > 0.02).mean()) if len(sub) else np.nan
            rows.append(r)
    pd.DataFrame(rows).to_csv(args.out / "D_stop.csv", index=False)

    for f in ("D_prefix", "D_paired", "D_degrade", "D_stop", "D_examples"):
        p = args.out / f"{f}.csv"
        if p.exists():
            t = pd.read_csv(p)
            if f == "D_paired":
                t = t[t["metric"].isin(["cost", "oracle_cost", "regret", "n_good"])]
            print(f"\n== {f}\n" + t.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

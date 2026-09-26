"""#3945 Stage R analysis: the quantities PLAN.md pre-registers, in its order.

    python analyze_c_path.py --results DIR --out DIR

Reads every ``*.csv.gz`` ``c_path.py`` wrote under ``--results``.  A *session*
is one (dataset, embedder, category, seed); a *cut* is one click count ``t``.
Every contrast is paired within the session and cut, and its standard error is
clustered on (environment, category), as #3197's analyzers do.

Tables written:

* ``R_path.csv``       mean held-out metrics per (t, C), pooled and per dataset;
* ``R_penalty.csv``    Q1: oracle cost at C = 1 minus at the cross-fitted C*(t);
* ``R_bestC.csv``      Q2: C*(t) as each seed half picked it;
* ``R_schedule.csv``   Q3: each vote-count schedule vs fixed C* and vs C = 1;
* ``R_detect.csv``     Q4: each vote-only gauge, tracking and C-picking;
* ``R_degrade.csv``    Q5: sessions whose C = 1 ranking got worse with votes.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

SESSION = ["dataset", "embedder", "category", "seed"]
CLUSTER = ["env", "category"]
HALVES = ((0, 1, 2), (3, 4))
SHIPPED_C = 1.0


def load(d: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(d / "*.csv.gz")))
    if not files:
        raise SystemExit(f"no results under {d}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df[df["one_class"] == 0].copy()
    short = {"visual_genome_m": "vg", "caltech101_m": "caltech", "coco_val": "coco"}
    df["ds"] = df["dataset"].map(short).fillna(df["dataset"])
    df["env"] = df["ds"] + "/" + df["embedder"]
    return df


def clustered(values: pd.Series, clusters: pd.DataFrame) -> tuple[float, float, int]:
    """Mean and cluster-robust SE of a paired difference (one row per session)."""
    v = pd.DataFrame({"v": values.to_numpy()}, index=values.index).join(clusters)
    v = v.dropna(subset=["v"]).reset_index(drop=True)
    if v.empty:
        return float("nan"), float("nan"), 0
    g = v.groupby(CLUSTER)["v"]
    sums, counts = g.sum(), g.count()
    n, k = counts.sum(), len(counts)
    mean = sums.sum() / n
    resid = sums - counts * mean
    se = float(np.sqrt(k / max(k - 1, 1) * (resid**2).sum()) / n) if k > 1 else float("nan")
    return float(mean), se, int(n)


def other_half(seed: int) -> tuple[int, ...]:
    return HALVES[1] if seed in HALVES[0] else HALVES[0]


def best_c_by_half(df: pd.DataFrame, metric: str = "test_oracle_cost") -> dict[tuple[int, ...], pd.Series]:
    """For each seed half: per t, the single C with the lowest pooled mean ``metric``."""
    out = {}
    for half in HALVES:
        m = df[df["seed"].isin(half)].groupby(["t", "C"])[metric].mean().reset_index()
        out[half] = m.loc[m.groupby("t")[metric].idxmin()].set_index("t")["C"]
    return out


def at(df: pd.DataFrame, C_of_row: pd.Series, metric: str = "test_oracle_cost") -> pd.Series:
    """``metric`` of each (session, t) at the C named per (session, t) in ``C_of_row``."""
    key = [*SESSION, "t"]
    want = C_of_row.rename("Cw").reset_index()
    m = df.merge(want, on=key)
    m = m[np.isclose(m["C"], m["Cw"])]
    return m.set_index(key)[metric]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.results)
    key = [*SESSION, "t"]
    grid = np.array(sorted(df["C"].unique()))
    sess = df.drop_duplicates(key).set_index(key)
    clusters = pd.DataFrame({"env": sess["env"], "category": sess.index.get_level_values("category")}, index=sess.index)
    print(f"{df.drop_duplicates(SESSION).shape[0]} sessions, {len(sess)} (session, t) cuts, C grid {grid.tolist()}")

    # ---- the path itself -------------------------------------------------------
    metrics = [
        "test_oracle_cost",
        "test_ap",
        "test_auroc",
        "train_auroc",
        "active_frac",
        "wnorm",
        "app_auroc",
        "cv_auroc",
    ]
    path = df.groupby(["t", "C"])[metrics].mean().reset_index().assign(ds="all")
    path_ds = df.groupby(["ds", "t", "C"])[metrics].mean().reset_index()
    pd.concat([path, path_ds]).to_csv(args.out / "R_path.csv", index=False)

    # ---- Q2: C*(t), cross-fitted --------------------------------------------------
    best = best_c_by_half(df)
    pd.DataFrame({f"seeds{''.join(map(str, h))}": s for h, s in best.items()}).to_csv(args.out / "R_bestC.csv")

    def crossfit_c(frame_index: pd.MultiIndex, table: dict) -> pd.Series:
        idx = frame_index.to_frame(index=False)
        cs = [table[other_half(int(s))].get(int(t), np.nan) for s, t in zip(idx["seed"], idx["t"])]
        return pd.Series(cs, index=frame_index)

    c_star = crossfit_c(sess.index, best)
    oc_star = at(df, c_star)
    oc_one = at(df, pd.Series(SHIPPED_C, index=sess.index))

    # ---- Q1: penalty of C = 1 vs C*(t), per t and per dataset ------------------------
    pen = (oc_one - oc_star).rename("pen")
    rows = []
    for scope, mask in [("all", None), *[(d, sess["ds"] == d) for d in sorted(sess["ds"].unique())]]:
        for t in sorted(sess.index.get_level_values("t").unique()):
            sel = pen.index.get_level_values("t") == t
            if mask is not None:
                sel &= mask.reindex(pen.index).to_numpy()
            mean, se, n = clustered(pen[sel], clusters)
            n_pos = sess.loc[pen[sel].index, "n_pos_votes"].mean() if n else np.nan
            rows.append({"scope": scope, "t": t, "penalty": mean, "se": se, "n": n, "mean_goods": n_pos})
    pd.DataFrame(rows).to_csv(args.out / "R_penalty.csv", index=False)

    # Growth: the PLAN's "t = 150 value above the t = 20 value by 2 SE", paired within
    # the session (the two cuts share votes, so their SEs cannot just be combined).
    # Also read at the deepest cut, which is t = 400 on a Stage D replay.
    wide = pen.unstack("t")
    rows = []
    for t_hi in sorted({150, int(wide.columns.max())}):
        if t_hi not in wide.columns or 20 not in wide.columns:
            continue
        d = (wide[t_hi] - wide[20]).dropna()
        ix = d.index.to_frame(index=False)
        cl = pd.DataFrame(
            {"env": (ix["dataset"] + "/" + ix["embedder"]).to_numpy(), "category": ix["category"].to_numpy()},
            index=d.index,
        )
        ds_of = d.index.get_level_values("dataset")
        for scope in ("all", *sorted(set(ds_of))):
            sub = d if scope == "all" else d[ds_of == scope]
            mean, se, n = clustered(sub, cl)
            rows.append({"scope": scope, "contrast": f"pen(t={t_hi}) - pen(t=20)", "mean": mean, "se": se, "n": n})
    pd.DataFrame(rows).to_csv(args.out / "R_growth.csv", index=False)

    # C*(t) per dataset (descriptive; the pooled cross-fitted one is the pre-registered one).
    per_ds = df.groupby(["ds", "t", "C"])["test_oracle_cost"].mean().reset_index()
    per_ds = per_ds.loc[per_ds.groupby(["ds", "t"])["test_oracle_cost"].idxmin()]
    per_ds.pivot(index="t", columns="ds", values="C").to_csv(args.out / "R_bestC_by_ds.csv")

    # ---- Q3: schedules --------------------------------------------------------------
    # C = c0 * n^-alpha, snapped to the nearest grid C in log space; c0 fitted on one
    # seed half (lowest mean oracle cost over every cut), scored on the other.
    lg = np.log10(grid)

    def snap(c: np.ndarray) -> np.ndarray:
        return grid[np.abs(np.log10(np.clip(c, 1e-12, None))[:, None] - lg[None, :]).argmin(1)]

    idx = sess.index.to_frame(index=False)
    n_votes = idx["t"].to_numpy(dtype=float)
    n_goods = np.maximum(sess["n_pos_votes"].to_numpy(dtype=float), 1.0)
    c0s = 10.0 ** np.arange(-4, 3.01, 0.25)
    oc_table = df.set_index([*key, "C"])["test_oracle_cost"]

    def oc_for(cvec: np.ndarray) -> pd.Series:
        ix = pd.MultiIndex.from_arrays([*(idx[k] for k in key), cvec], names=[*key, "C"])
        return pd.Series(oc_table.reindex(ix).to_numpy(), index=sess.index)

    sched_defs = {"fixed": (None, 0.0)}
    for base_name, base in (("votes", n_votes), ("goods", n_goods)):
        for alpha in (0.5, 1.0):
            sched_defs[f"{base_name}^-{alpha:g}"] = (base, alpha)
    sched_oc, sched_c0 = {}, {}
    seeds = idx["seed"].to_numpy()
    for name, (base, alpha) in sched_defs.items():
        out = pd.Series(np.nan, index=sess.index)
        for half in HALVES:
            train = np.isin(seeds, half)
            best_c0, best_val = None, np.inf
            for c0 in c0s:
                cvec = snap(np.full(len(idx), c0) if base is None else c0 * base**-alpha)
                val = oc_for(cvec)[train].mean()
                if val < best_val:
                    best_c0, best_val = c0, val
            cvec = snap(np.full(len(idx), best_c0) if base is None else best_c0 * base**-alpha)
            test = ~train
            out[test] = oc_for(cvec)[test].to_numpy()
            sched_c0[(name, half)] = best_c0
        sched_oc[name] = out
    rows = []
    for name, s in sched_oc.items():
        for ref_name, ref in (("C=1", oc_one), ("fixed C*", sched_oc["fixed"]), ("C*(t)", oc_star)):
            if name == "fixed" and ref_name == "fixed C*":
                continue
            d = (s - ref).groupby(level=SESSION).mean()  # average over cuts first
            cl = d.index.to_frame(index=False).assign(env=lambda f: f["dataset"] + "/" + f["embedder"])
            cl.index = d.index
            mean, se, n = clustered(d, cl[CLUSTER])
            rows.append(
                {
                    "schedule": name,
                    "minus": ref_name,
                    "mean": mean,
                    "se": se,
                    "n_sessions": n,
                    "c0_half012": sched_c0[(name, HALVES[0])],
                    "c0_half34": sched_c0[(name, HALVES[1])],
                }
            )
    pd.DataFrame(rows).to_csv(args.out / "R_schedule.csv", index=False)

    # ---- Q4: detection by vote-only gauges ----------------------------------------------
    rows = []
    for gauge in ("app_auroc", "cv_auroc", "train_auroc"):
        # (a) within-session tracking at C = 1, across cuts
        one = df[np.isclose(df["C"], SHIPPED_C)]
        rhos = []
        for _, g in one.groupby(SESSION):
            g = g.dropna(subset=[gauge])
            if len(g) >= 4 and g[gauge].nunique() > 1 and g["test_auroc"].nunique() > 1:
                rhos.append(g[gauge].rank().corr(g["test_auroc"].rank()))
        # (b) the C the gauge picks: highest gauge, ties to the C nearest the shipped one
        g = df.dropna(subset=[gauge]).copy()
        g["dist"] = np.abs(np.log10(g["C"]) - np.log10(SHIPPED_C))
        g = g.sort_values([*key, gauge, "dist"], ascending=[True] * len(key) + [False, True])
        pick = g.drop_duplicates(key).set_index(key)["C"]
        oc_pick = at(df, pick)
        common = oc_pick.index.intersection(oc_star.index)
        mean_s, se_s, n_s = clustered((oc_pick - oc_star).loc[common], clusters)
        mean_1, se_1, _ = clustered((oc_pick - oc_one).loc[common], clusters)
        # (c) exploratory: across sessions at one cut, does the gauge's C=1-vs-C* gap
        # predict the held-out one?
        gap_g = at(df, pd.Series(SHIPPED_C, index=sess.index), gauge) - at(df, c_star, gauge)
        gap_t = -(oc_one - oc_star)  # positive = C = 1 ranks better held out
        cc = pd.concat([gap_g.rename("g"), gap_t.rename("t")], axis=1).dropna()
        rows.append(
            {
                "gauge": gauge,
                "median_rho_within_session": float(np.median(rhos)) if rhos else np.nan,
                "n_sessions_rho": len(rhos),
                "picked_minus_Cstar": mean_s,
                "se": se_s,
                "picked_minus_C1": mean_1,
                "se_C1": se_1,
                "n_cuts": n_s,
                "share_pick_ge_1": float((pick >= SHIPPED_C).mean()),
                "rank_corr_gap_across_cuts": float(cc["g"].rank().corr(cc["t"].rank())) if len(cc) > 10 else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(args.out / "R_detect.csv", index=False)

    # ---- Q5: along-session degradation at C = 1 (and at C*) -------------------------------
    rows = []
    for label, series in (("C=1", oc_one), ("C*(t)", oc_star)):
        s = series.rename("oc").reset_index()
        worse = []
        for _, g in s.groupby(SESSION):
            g = g.sort_values("t")
            if len(g) < 3:
                continue
            last = g["oc"].iloc[-1]
            best_before = g["oc"].iloc[:-1].min()
            worse.append(
                {
                    "t_last": int(g["t"].iloc[-1]),
                    "worse": float(last - best_before),
                    "ds": g["dataset"].iloc[0],
                }
            )
        w = pd.DataFrame(worse)
        for scope, sub in [("all", w), *list(w.groupby("ds"))]:
            rows.append(
                {
                    "C": label,
                    "scope": scope,
                    "n_sessions": len(sub),
                    "share_worse_0.02": float((sub["worse"] > 0.02).mean()),
                    "share_worse_0.05": float((sub["worse"] > 0.05).mean()),
                    "median_last_minus_best": float(sub["worse"].median()),
                }
            )
    pd.DataFrame(rows).to_csv(args.out / "R_degrade.csv", index=False)
    for f in ("R_penalty", "R_growth", "R_bestC", "R_bestC_by_ds", "R_schedule", "R_detect", "R_degrade"):
        print(f"\n== {f}\n" + pd.read_csv(args.out / f"{f}.csv").round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

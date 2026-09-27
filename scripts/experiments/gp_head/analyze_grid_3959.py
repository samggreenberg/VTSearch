"""#3959 analysis: the GP head at GRID scale, paired on the cell.

Reads each (env, arm)'s base rows (``_cells_io.load_arm``) from the layout
``launch_grid_3959.sh`` writes - ``<root>/<env>/<arm>/results/cells`` - and pairs
arms on the CELL, ``(dataset, embedder, category, seed)``, never the step: the
head and the cut drive Autopilot's Hard pick, so two arms collect different
votes from their first retrain and share nothing finer than the cell.

Every contrast is read three ways, straight off the rows:

* ``oracle_cost`` / ``average_precision`` / ``auroc`` - threshold-free, **the
  ranking** (``oracle_cost`` is the best cut the test labels allow);
* ``regret = cost - oracle_cost`` - **the cut**'s own loss against that;
* ``cost`` - both, the number a user lives with.

Differences are ``arm - ref`` with a standard error clustered on (dataset,
embedder, category), because the seeds of one category are not independent
cells.  For cost-like metrics a NEGATIVE difference means *arm* is better; for
AP/AUROC a POSITIVE one does.

    python analyze_grid_3959.py --root /expscratch/$USER/gp-grid-3959 --out DIR
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

ENVS = ("better", "natural")
ARMS = (
    "app",
    "app_xcal",
    "app_gmm",
    "gp_rbf_anch",
    "gp_rbf_xcal",
    "gp_rbf_gmm",
    "gp_dot_anch",
    "gp_rbf_anch_maxvar",
)
#: The contrasts the study exists for, as ``(arm, ref, what it isolates)``.
#: The first three hold the RULE fixed and change the head; the next two are the
#: issue's own framing (everything against the plain cross-calibrated SVM); the
#: rest hold the head fixed and change the rule, the kernel or the pick.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("gp_rbf_anch", "app", "head, at the fold-anchored cut"),
    ("gp_rbf_xcal", "app_xcal", "head, at the plain x-cal cut"),
    ("gp_rbf_gmm", "app_gmm", "head, at the GMM-midpoint cut"),
    ("app", "app_xcal", "shipped rule vs plain x-cal (SVM head)"),
    ("gp_rbf_anch", "app_xcal", "GP-native vs the plain x-cal SVM"),
    ("app_gmm", "app", "GMM midpoint vs shipped rule (SVM head)"),
    ("gp_rbf_anch", "gp_rbf_xcal", "fold-anchored vs plain x-cal (GP head)"),
    ("gp_rbf_anch", "gp_rbf_gmm", "fold-anchored vs GMM midpoint (GP head)"),
    ("gp_dot_anch", "gp_rbf_anch", "dot-product vs RBF kernel"),
    ("gp_rbf_anch_maxvar", "gp_rbf_anch", "max-variance vs rank-nearest-cut Hard pick"),
    ("gp_rbf_anch_maxvar", "app", "best GP arm vs the shipped detector"),
)
CHECKPOINTS = (10, 25, 50, 100, 150)
METRICS = ("cost", "oracle_cost", "regret", "average_precision", "auroc", "f1", "fnr", "fpr", "n_good")
LOWER_IS_BETTER = {"cost", "oracle_cost", "regret", "fnr", "fpr"}
KEYS = ["dataset", "embedder", "category", "seed"]
CLUSTER = ["dataset", "embedder", "category"]
#: The window an area-under-the-curve is read over.  From click 8 every arm of
#: every cell has trained (a Good and a Bad exist from the text opening), so the
#: window is the same for every arm and pairing is exact.
AULC_FROM = 8


def env_label(ds: str, emb: str) -> str:
    return f"{ds}/{emb}"


def load(root: Path) -> tuple[pd.DataFrame, dict]:
    import _cells_io

    parts, prov = [], {}
    for env in ENVS:
        for arm in ARMS:
            d = root / env / arm / "results"
            if not (d / "cells").exists():
                continue
            df, p = _cells_io.load_arm(d)
            prov[f"{env}/{arm}"] = {k: (len(v) if isinstance(v, list) else v) for k, v in p.items()}
            if df.empty:
                continue
            cols = [
                c
                for c in [*KEYS, "t", *METRICS, "threshold", "threshold_provenance", "trainer", "strategy", "n_flagged"]
                if c in df.columns
            ]
            df = df[cols].copy()
            df["arm"] = arm
            df["study_env"] = env
            parts.append(df)
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not frame.empty:
        frame["env"] = [env_label(d, e) for d, e in zip(frame["dataset"], frame["embedder"], strict=True)]
    return frame, prov


def at_checkpoint(df: pd.DataFrame, t: int) -> pd.DataFrame:
    """Each cell's row at click ``t`` (the last row at or before it)."""
    sub = df[df["t"] <= t]
    return sub.sort_values("t").groupby([*KEYS, "arm"], as_index=False).tail(1)


def aulc(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of *metric* over clicks ``AULC_FROM``..max, forward-filled per cell and arm."""
    t_max = int(df["t"].max())
    grid = np.arange(AULC_FROM, t_max + 1)
    out = []
    for key, g in df.groupby([*KEYS, "arm"], sort=False):
        s = g.set_index("t")[metric].sort_index()
        s = s[~s.index.duplicated(keep="last")].reindex(np.union1d(s.index, grid)).ffill().reindex(grid)
        out.append((*key, float(s.mean())))
    return pd.DataFrame(out, columns=[*KEYS, "arm", metric])


def paired(df: pd.DataFrame, metric: str, arm: str, ref: str) -> dict:
    p = df.pivot_table(index=KEYS, columns="arm", values=metric)
    if arm not in p or ref not in p:
        return {"n": 0}
    x = (p[arm] - p[ref]).dropna()
    if x.empty:
        return {"n": 0}
    g = x.groupby(level=CLUSTER).mean()
    se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
    better = (x < 0) if metric in LOWER_IS_BETTER else (x > 0)
    return {
        "ref_mean": float(p.loc[x.index, ref].mean()),
        "arm_mean": float(p.loc[x.index, arm].mean()),
        "mean": float(x.mean()),
        "se": se,
        "n": int(len(x)),
        "n_clusters": int(len(g)),
        "frac_arm_better": float(np.mean(better)),
        "frac_tied": float(np.mean(x == 0)),
    }


def snapshots(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    snaps = {f"t{t}": at_checkpoint(df, t) for t in CHECKPOINTS}
    a = None
    for m in METRICS:
        if m not in df:
            continue
        am = aulc(df, m)
        a = am if a is None else a.merge(am, on=[*KEYS, "arm"])
    snaps["aulc"] = a
    return snaps


def paired_table(snaps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for win, s in snaps.items():
        s = s.copy()
        s["env"] = [env_label(d, e) for d, e in zip(s["dataset"], s["embedder"], strict=True)]
        scopes = [("ALL", s)]
        scopes += [(ds, s[s["dataset"] == ds]) for ds in sorted(s["dataset"].unique())]
        scopes += [(env, s[s["env"] == env]) for env in sorted(s["env"].unique())]
        for scope, sub in scopes:
            for arm, ref, what in PAIRS:
                for m in METRICS:
                    if m not in sub:
                        continue
                    r = paired(sub, m, arm, ref)
                    if r["n"]:
                        rows.append(
                            {"window": win, "scope": scope, "arm": arm, "ref": ref, "isolates": what, "metric": m, **r}
                        )
    return pd.DataFrame(rows)


def levels_table(snaps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for win, s in snaps.items():
        s = s.copy()
        s["env"] = [env_label(d, e) for d, e in zip(s["dataset"], s["embedder"], strict=True)]
        for scope_col in ("dataset", "env"):
            g = s.groupby([scope_col, "arm"])[[m for m in METRICS if m in s]].mean().reset_index()
            g = g.rename(columns={scope_col: "scope"})
            g["n_cells"] = s.groupby([scope_col, "arm"]).size().values
            g["window"] = win
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def cut_health(df: pd.DataFrame) -> pd.DataFrame:
    """Share of steps whose cut flagged everything / nothing on the test half, per arm and dataset."""
    d = df.copy()
    d["flag_all"] = d["fpr"] >= 0.999
    d["flag_none"] = d["fnr"] >= 0.999
    prov = d.get("threshold_provenance", pd.Series("", index=d.index)).fillna("").astype(str)
    d["anchored"] = prov.str.startswith("fold_anchored")
    return (
        d.groupby(["dataset", "arm"])
        .agg(
            steps=("t", "size"),
            flagged_everything=("flag_all", "mean"),
            flagged_nothing=("flag_none", "mean"),
            fold_anchored_share=("anchored", "mean"),
            mean_threshold=("threshold", "mean"),
        )
        .reset_index()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, prov = load(args.root)
    if df.empty:
        print(f"no rows under {args.root}")
        return 2
    df = df[df["t"] > 0]  # the skyline row, where one exists, belongs to no step
    snaps = snapshots(df)
    paired_df = paired_table(snaps)
    paired_df.to_csv(args.out / "paired.csv", index=False)
    levels_table(snaps).to_csv(args.out / "levels.csv", index=False)
    cut_health(df).to_csv(args.out / "cut_health.csv", index=False)
    snaps["aulc"].to_csv(args.out / "aulc_per_cell.csv", index=False)

    cells = df.groupby(["study_env", "arm"])[KEYS].apply(lambda g: len(g.drop_duplicates())).rename("cells_with_rows")
    cells.reset_index().to_csv(args.out / "coverage.csv", index=False)
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    def fmt(r: pd.Series) -> str:
        flag = "" if abs(r["mean"]) >= 2 * r["se"] else "  (not resolvable)"
        return f"{r['mean']:+.3f} ± {r['se']:.3f}  n={r['n']}{flag}"

    print("coverage (cells with rows):")
    print(cells.unstack(0).to_string())
    for win in ("aulc", "t25", "t50", "t150"):
        print(f"\n[{win}] arm - ref, clustered SE (cost-like: negative = arm better)")
        sub = paired_df[(paired_df.window == win) & (paired_df.scope == "ALL")]
        for (arm, ref), g in sub.groupby(["arm", "ref"], sort=False):
            print(f"  {arm} - {ref}:")
            for m in ("cost", "oracle_cost", "regret", "average_precision"):
                r = g[g.metric == m]
                if len(r):
                    print(f"      {m:>18}: {fmt(r.iloc[0])}")
    print(json.dumps(prov, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

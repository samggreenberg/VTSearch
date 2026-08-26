"""Where does a combine-rule effect actually live? (#3115 follow-up)

Every headline in ``folds_combine_3115`` is a mean of cell-means: 12 categories x
4 seeds x ~150 steps collapsed to one number per (voting mode, contrast). That is
the right unit for *deciding*, and the wrong unit for *understanding* - a
``+0.028`` is equally "twelve categories at +0.028" and "two categories at +0.15
and ten at zero", and those imply different work.

This walks the same contrast down the hierarchy the study is built from:

    env  ->  category  ->  run (category x seed)  ->  step

and stops at the step, where the diagnostic columns say what was *different*
about it - how many positives were in hand, where the threshold sat in the score
distribution, which error the cut was paying. The point is not to find a winner;
it is to find the situations that produce the loss, so the losing rule can be
improved rather than merely avoided.

Reads the same per-step cells the tables come from, so a drill-down and a
headline cannot disagree.

Usage::

    python drill_folds_3115.py --results <dir> --arm qmean --ref xcal
    python drill_folds_3115.py --results <dir> --arm qmean --ref xcal --env vg_scale_any/siglip/whole_image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import folds_combine_3115 as fc  # noqa: E402
from analyze_folds_2897 import fold_frame, load_cells  # noqa: E402

#: Columns that describe the *situation* a step was in, as opposed to how well it
#: scored. These are what turn "this run is bad" into "this run is bad *when*".
SITUATION = [
    "n_good",
    "n_bad",
    "threshold",
    "threshold_percentile",
    "fpr",
    "fnr",
    "degenerate",
    "n_cal_scores",
    "auroc",
    "average_precision",
]


def _paired(v: pd.DataFrame, arm: str, ref: str, k: int) -> pd.DataFrame:
    """Step-paired arm-vs-ref at one fold count, carrying the situation columns.

    The situation columns are taken from the **reference** side on purpose: they
    describe the state the trajectory was in when the two rules disagreed, and
    that state is a property of the run, not of the rule being tested.
    """
    keys = [*fc.STEP_KEYS, "k"]
    a = v[(v["arm"] == arm) & (v["k"] == k)].set_index(keys)
    b = v[(v["arm"] == ref) & (v["k"] == k)].set_index(keys)
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    have = [c for c in SITUATION if c in b.columns]
    j = pd.concat(
        [a[["regret", "cost", "threshold"]].add_suffix("_a"), b[["regret", "cost", *have]].add_suffix("_b")],
        axis=1,
        join="inner",
    )
    if j.empty:
        return j
    j = j.reset_index()
    j["d_regret"] = j["regret_a"] - j["regret_b"]
    j["moved"] = (j["threshold_a"] != j["threshold_b"]).astype(float)
    j["env"] = j["env"].astype(str)
    return j


def by_category(j: pd.DataFrame, deep_min: int, v: pd.DataFrame) -> pd.DataFrame:
    """Per-category effect, with the spread ACROSS seeds that a mean would hide."""
    step_window = v[["env", "category", "seed", "t", "window_hi"]].drop_duplicates()
    j = j.merge(step_window, on=["env", "category", "seed", "t"], how="left")
    deep = j[j["window_hi"] >= deep_min]
    per_run = deep.groupby(["env", "category", "seed"], observed=True)["d_regret"].mean().reset_index()
    g = (
        per_run.groupby(["env", "category"], observed=True)["d_regret"]
        .agg(mean="mean", worst="max", best="min", sd="std", n_seeds="size")
        .reset_index()
        .sort_values(["env", "mean"], ascending=[True, False])
    )
    # How much of the env's total effect this one category carries.
    tot = g.groupby("env", observed=True)["mean"].transform("sum")
    g["share_of_env"] = g["mean"] / tot.replace(0, np.nan)
    return g


def by_run(j: pd.DataFrame, deep_min: int, v: pd.DataFrame, env: str, category: str) -> pd.DataFrame:
    """Every run of one category, so a category mean can be checked against its runs."""
    step_window = v[["env", "category", "seed", "t", "window_hi"]].drop_duplicates()
    j = j.merge(step_window, on=["env", "category", "seed", "t"], how="left")
    sel = j[(j["env"] == env) & (j["category"] == category) & (j["window_hi"] >= deep_min)]
    if sel.empty:
        return sel
    return (
        sel.groupby("seed", observed=True)
        .agg(
            d_regret=("d_regret", "mean"),
            worst_step=("d_regret", "max"),
            n_steps=("d_regret", "size"),
            moved_rate=("moved", "mean"),
            n_good=("n_good_b", "mean"),
            thr_pct=("threshold_percentile_b", "mean"),
            fnr=("fnr_b", "mean"),
            fpr=("fpr_b", "mean"),
        )
        .reset_index()
        .sort_values("d_regret", ascending=False)
    )


def worst_steps(j: pd.DataFrame, env: str, category: str, seed: int, n: int = 12) -> pd.DataFrame:
    """The individual steps carrying a run's loss, with the state they were in."""
    sel = j[(j["env"] == env) & (j["category"] == category) & (j["seed"] == seed)]
    if sel.empty:
        return sel
    cols = ["t", "d_regret", "regret_a", "regret_b", "moved"] + [
        c for c in (f"{s}_b" for s in SITUATION) if c in sel.columns
    ]
    return sel.nlargest(n, "d_regret")[cols].sort_values("t")


def _md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:  # noqa: BLE001 - tabulate not installed
        return df.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--arm", default="qmean")
    ap.add_argument("--ref", default="xcal")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--deep-min", type=int, default=100)
    ap.add_argument("--env", default=None, help="restrict the run/step drill-down to one env")
    ap.add_argument("--top", type=int, default=3, help="how many extreme categories to open up")
    a = ap.parse_args(argv)

    v = fold_frame(load_cells(a.results / "cells"))
    if v.empty:
        print("no fold rows")
        return 1
    j = _paired(v, a.arm, a.ref, a.k)
    if j.empty:
        print(f"no paired rows for {a.arm} vs {a.ref} at K={a.k}")
        return 1

    print(f"\n# {a.arm} vs {a.ref}, K={a.k}, deep windows (>= {a.deep_min} votes)\n")
    cats = by_category(j, a.deep_min, v)
    print("## Per category — is the effect uniform, or carried by a few?\n")
    print(_md(cats))
    print()
    for env, g in cats.groupby("env", observed=True):
        top = g.nlargest(1, "mean").iloc[0]
        bot = g.nsmallest(1, "mean").iloc[0]
        spread = g["mean"].max() - g["mean"].min()
        print(
            f"  {env}: mean {g['mean'].mean():+.4f}, but categories span {spread:.4f} "
            f"({bot['category']} {bot['mean']:+.4f} .. {top['category']} {top['mean']:+.4f})"
        )
    print()

    envs = [a.env] if a.env else sorted(cats["env"].unique())
    for env in envs:
        g = cats[cats["env"] == env]
        for _, row in g.nlargest(a.top, "mean").iterrows():
            cat = row["category"]
            print(f"\n## {env} — category `{cat}` (mean {row['mean']:+.4f}), run by run\n")
            runs = by_run(j, a.deep_min, v, env, cat)
            print(_md(runs))
            if len(runs):
                worst = runs.iloc[0]
                print(f"\n### its worst run (seed {int(worst['seed'])}) — the steps carrying it\n")
                print(_md(worst_steps(j, env, cat, int(worst["seed"]))))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Tables for #3839: what the anchored refits that exit on ``max_iter`` cost.

    python analyze_3839.py --analysis <dir> --shards 12

Reads the sharded gate frames ``gate_3839.py`` wrote and refuses a set with a
shard missing - a dropped shard is 7 cells nobody would notice were gone.

Every arm is read two ways.  Against ``shipped`` (the rule on ``dev``), which is
the question "what would changing it do"; and against ``limit`` (the same loop
run to 1e-13 with 20,000 iterations), which is the question the issue actually
asks - "what does a capped fit *cost*" - because a fit that ran out of
iterations is only wrong to the extent that finishing would have moved the
line.

Writes ``agg/{fits,gate,capped,env,examples}.csv``, ``agg/summary.json`` and
``agg/TABLES.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE = ["cell", "style", "kind", "case"]
FOLD = [*CASE, "fold"]
PARAMS = ["w_lo", "mu_lo", "var_lo", "w_hi", "mu_hi", "var_hi"]
LOOP_ARMS = {"shipped", "limit", "cap400", "cap1000", "cap2000", "cap5000", "ll1e-7", "ll1e-6", "fallback"}
INCL = 0


def _fmt(v, digits: int) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.{digits}g}"
    return str(v)


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    """A markdown table without ``tabulate`` (not in the GRID venv)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def load(analysis: Path, shards: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cuts, fits = [], []
    for i in range(shards):
        c, f = analysis / f"gate3839_cuts.{i}.csv", analysis / f"gate3839_fits.{i}.csv"
        if not (c.exists() and f.exists()):
            raise SystemExit(f"shard {i} missing under {analysis}")
        cuts.append(pd.read_csv(c))
        fits.append(pd.read_csv(f))
    return pd.concat(cuts, ignore_index=True), pd.concat(fits, ignore_index=True)


def env_of(df: pd.DataFrame) -> pd.Series:
    return df["dataset"] + "/" + df["embedder"] + "/" + df["style"]


def pair_to(df: pd.DataFrame, keys: list[str], ref: str, cols: list[str]) -> pd.DataFrame:
    """Every arm's row beside *ref*'s row for the same key (many arms to one ref)."""
    r = df[df["arm"] == ref][[*keys, *cols]].rename(columns={c: f"ref_{c}" for c in cols})
    return df.merge(r, on=keys, how="inner", validate="many_to_one")


def fit_table(fits: pd.DataFrame) -> pd.DataFrame:
    anch = fits["provenance"] == "anchored"
    rows = []
    base = fits[fits["arm"] == "shipped"].set_index(FOLD)
    lim = fits[fits["arm"] == "limit"].set_index(FOLD)
    for arm, g in fits.groupby("arm", sort=False):
        a = g[anch.loc[g.index]]
        gi = g.set_index(FOLD)
        both = gi.join(base, rsuffix="_b", how="inner")
        both = both[(both["provenance"] == "anchored") & (both["provenance_b"] == "anchored")]
        d_obj = both["loglik_anchored"] - both["loglik_anchored_b"]
        vl = gi.join(lim, rsuffix="_l", how="inner")
        vl = vl[(vl["provenance"] == "anchored") & (vl["provenance_l"] == "anchored")]
        gap = vl["loglik_anchored_l"] - vl["loglik_anchored"]
        rows.append(
            {
                "arm": arm,
                "folds": len(g),
                "anchored": int(len(a)),
                "capped_pct": 100.0 * float((a["converged"] == 0).mean()),
                "iter_median": float(a["n_iter"].median()),
                "iter_mean": float(a["n_iter"].mean()),
                "iter_p90": float(a["n_iter"].quantile(0.9)),
                "iter_max": int(a["n_iter"].max()),
                "refit_ms_median": 1e3 * float(a["refit_seconds"].median()) if arm in LOOP_ARMS else np.nan,
                "lost_vs_shipped": int(
                    ((gi["provenance"] != "anchored") & (base.reindex(gi.index)["provenance"] == "anchored")).sum()
                ),
                "obj_better": int((d_obj > 1e-13).sum()),
                "obj_worse": int((d_obj < -1e-13).sum()),
                "gap_to_limit_median": float(gap.median()),
                "gap_to_limit_p99": float(gap.quantile(0.99)),
                "gap_to_limit_max": float(gap.max()),
            }
        )
    return pd.DataFrame(rows)


def move_table(cuts: pd.DataFrame, ref: str, subset: "pd.Index | None" = None) -> pd.DataFrame:
    col = f"n_admitted_i{INCL}"
    p = pair_to(cuts, CASE, ref, [col])
    if subset is not None:
        p = p.set_index(CASE).loc[lambda x: x.index.isin(subset)].reset_index()
    p["move"] = (p[col] - p[f"ref_{col}"]).abs() / p["n"]
    rows = []
    for arm, g in p.groupby("arm", sort=False):
        if arm == ref:
            continue
        rows.append(
            {
                "arm": arm,
                "cases": len(g),
                "moved_pct": 100.0 * float((g["move"] > 0).mean()),
                "move_median_pct": 100.0 * float(g["move"].median()),
                "move_p90_pct": 100.0 * float(g["move"].quantile(0.9)),
                "move_p99_pct": 100.0 * float(g["move"].quantile(0.99)),
                "move_max_pct": 100.0 * float(g["move"].max()),
                "one_order_stat_pct": 100.0 * float((g[col] - g[f"ref_{col}"]).abs().le(1).mean()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: "list[str] | None" = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--shards", type=int, default=12)
    args = ap.parse_args(list(argv) if argv is not None else None)
    analysis = Path(args.analysis)
    agg = analysis / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    cuts, fits = load(analysis, args.shards)
    cuts["env"] = env_of(cuts)
    fits["env"] = env_of(fits)
    md = ["# #3839 tables (generated by `analyze_3839.py`)\n"]
    summary: dict = {}

    # 0. The driver must BE the loop, or every driven arm is two changes.
    s = fits[fits["arm"] == "shipped"].set_index(FOLD)[PARAMS]
    d = fits[fits["arm"] == "drv_shipped"].set_index(FOLD)[PARAMS].reindex(s.index)
    same = int((s.eq(d) | (s.isna() & d.isna())).all(axis=1).sum())
    summary["driver_identical"] = [same, len(s)]
    if same != len(s):
        raise SystemExit(f"drv_shipped differs from shipped on {len(s) - same} of {len(s)} folds")
    md.append(f"`drv_shipped` reproduces `shipped` on **{same} of {len(s)}** folds, all six parameters bit for bit.\n")
    n_arms = cuts["arm"].nunique()
    n_cases = cuts.groupby(CASE).ngroups
    if len(cuts) != n_cases * n_arms:
        raise SystemExit(f"{len(cuts)} cut rows for {n_cases} cases x {n_arms} arms")
    summary["cases"] = n_cases
    summary["folds"] = int(len(s))

    ft = fit_table(fits)
    ft.to_csv(agg / "fits.csv", index=False)
    md += ["## Estimator: iterations, capped share, objective vs shipped and vs the limit\n", _md(ft), "\n"]

    gs = move_table(cuts, "shipped")
    gl = move_table(cuts, "limit")
    gs.to_csv(agg / "gate_vs_shipped.csv", index=False)
    gl.to_csv(agg / "gate_vs_limit.csv", index=False)
    md += ["## Gate vs `shipped` (inclusion 0, move = |Δ admitted| / haystack)\n", _md(gs), "\n"]
    md += ["## Gate vs `limit` - distance from the converged answer\n", _md(gl), "\n"]

    # The population the issue is about: cases where the shipped chain had a capped fold.
    sc = cuts[cuts["arm"] == "shipped"]
    capped_cases = pd.MultiIndex.from_frame(sc[sc["n_unconverged"].fillna(0) > 0][CASE])
    summary["capped_cases"] = [int(len(capped_cases)), int(len(sc))]
    gc = move_table(cuts, "limit", capped_cases)
    gc.to_csv(agg / "capped.csv", index=False)
    md += [
        f"## Gate vs `limit`, only the {len(capped_cases)} of {len(sc)} cases with a capped fold under `shipped`\n",
        _md(gc),
        "\n",
    ]

    # Per environment: where the capped folds are and what finishing them moves.
    fa = fits[(fits["arm"] == "shipped") & (fits["provenance"] == "anchored")]
    e1 = fa.groupby("env").agg(
        folds=("n_iter", "size"), capped_pct=("converged", lambda c: 100.0 * float((c == 0).mean()))
    )
    col = f"n_admitted_i{INCL}"
    pl = pair_to(cuts, CASE, "limit", [col])
    pl = pl[pl["arm"] == "shipped"]
    pl["move"] = (pl[col] - pl[f"ref_{col}"]).abs() / pl["n"]
    e2 = pl.groupby("env").agg(
        cases=("move", "size"),
        to_limit_moved_pct=("move", lambda m: 100.0 * float((m > 0).mean())),
        to_limit_p90_pct=("move", lambda m: 100.0 * float(m.quantile(0.9))),
        to_limit_max_pct=("move", lambda m: 100.0 * float(m.max())),
    )
    env = e1.join(e2).reset_index()
    env.to_csv(agg / "env.csv", index=False)
    md += ["## Per environment (`shipped`)\n", _md(env), "\n"]

    # Literal examples: the cases where finishing moves the line most.
    lim_it = fits[fits["arm"] == "limit"].groupby(CASE)["n_iter"].max().rename("limit_iters_max_fold")
    ex = pl.sort_values("move", ascending=False).head(8).merge(lim_it.reset_index(), on=CASE, how="left")
    ex = ex[[*CASE, "env", "category", "seed", "n", "n_unconverged", col, f"ref_{col}", "move", "limit_iters_max_fold"]]
    ex = ex.rename(columns={col: "admitted_shipped", f"ref_{col}": "admitted_limit"})
    ex.to_csv(agg / "examples.csv", index=False)
    md += ["## Largest shipped-vs-limit moves (literal cases)\n", _md(ex), "\n"]

    # Remaining gain on the capped folds specifically.
    base = fits[fits["arm"] == "shipped"].set_index(FOLD)
    lim = fits[fits["arm"] == "limit"].set_index(FOLD)
    j = base.join(lim, rsuffix="_l", how="inner")
    j = j[(j["provenance"] == "anchored") & (j["provenance_l"] == "anchored")]
    cap = j[j["converged"] == 0]
    summary["capped_folds"] = {
        "n": int(len(cap)),
        "remaining_gain_median": float((cap["loglik_anchored_l"] - cap["loglik_anchored"]).median()),
        "remaining_gain_p90": float((cap["loglik_anchored_l"] - cap["loglik_anchored"]).quantile(0.9)),
        "remaining_gain_max": float((cap["loglik_anchored_l"] - cap["loglik_anchored"]).max()),
        "midpoint_move_median": float((cap["midpoint_l"] - cap["midpoint"]).abs().median()),
        "midpoint_move_p90": float((cap["midpoint_l"] - cap["midpoint"]).abs().quantile(0.9)),
        "midpoint_move_max": float((cap["midpoint_l"] - cap["midpoint"]).abs().max()),
        "limit_iters_median": float(cap["n_iter_l"].median()),
        "limit_iters_p90": float(cap["n_iter_l"].quantile(0.9)),
        "limit_iters_max": int(cap["n_iter_l"].max()),
        "limit_capped": int((cap["converged_l"] == 0).sum()),
    }
    conv = j[j["converged"] == 1]
    summary["converged_folds"] = {
        "n": int(len(conv)),
        "midpoint_move_median": float((conv["midpoint_l"] - conv["midpoint"]).abs().median()),
        "midpoint_move_p90": float((conv["midpoint_l"] - conv["midpoint"]).abs().quantile(0.9)),
        "remaining_gain_median": float((conv["loglik_anchored_l"] - conv["loglik_anchored"]).median()),
    }
    (agg / "summary.json").write_text(json.dumps(summary, indent=2))
    md += ["## Summary\n", "```json\n" + json.dumps(summary, indent=2) + "\n```\n"]
    (agg / "TABLES.md").write_text("\n".join(md))
    print((agg / "TABLES.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

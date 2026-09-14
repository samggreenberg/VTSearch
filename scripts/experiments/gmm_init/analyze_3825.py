#!/usr/bin/env python
"""Turn the #3825 gate frames into the tables the decision needs.

    python analyze_3825.py --analysis <dir>

**Table 1, the gate.** Per arm: how often does the candidate admit a different
set of medias than the shipped stopping rule does, and by how much of the
haystack?  That is the criterion the issue names.  ``|delta threshold|`` rides
along because a large threshold move with no admitted-set change is a different
situation from one with neither - ``snap_cut_to_sample`` is designed to
collapse the first, and how much of a move it collapses is the question the
gate is really asking.

**Table 2, the incumbent.** How the *shipped* rule behaves: iterations run, and
what share of folds leave the loop because they ran out of them rather than
because they converged.  Nothing reported this before #3825, and it is half the
issue.

**Table 3, the estimator.** When two arms differ, which fit is better?  Both are
ascending the same weighted semi-supervised objective, so the sign of its
difference says whether an arm's cut moved toward the data or merely elsewhere.
Both objectives are scored for every arm, whichever it stopped on - an arm
stopped on the free-sample likelihood still has to be priced on the one its EM
was climbing, or the comparison is a tautology.

**Table 4, monotonicity.** A stopping rule that watches a quantity the loop does
not monotonically increase is not a convergence criterion.  The weighted
objective has a theorem; the free sample's likelihood under an anchored M-step
does not, and this table is whether that matters in practice or only on paper.

Pairing is exact: every arm sees the identical captured input, so a row pairs
with its baseline on the case identity with nothing matched approximately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: The inclusion whose columns the gate turns on: the shipped default.
GATE_INCLUSION = 0

#: Every inclusion the corpus was re-cut at; the gate turns on 0 and the others
#: say whether a difference invisible at the midpoint appears once the fitted
#: *variances* are read (the ``mid_tilt`` rule mixes the rate cut in off-centre).
INCLUSIONS = (-3, 0, 3)

#: Below this many paired cases an arm's rate is not reported as a rate.  A
#: "0% of 12" reads like a result and is not one.
MIN_CASES = 30

#: What identifies one captured case, and therefore what an arm's row pairs
#: with its baseline on.
CASE_KEYS = ["cell", "dataset", "embedder", "style", "kind", "case"]

#: A log-likelihood difference below this is a tie: it is what float64 resolves
#: on a mean over ~1e3-1e4 points, and calling those wins inflates every arm.
LL_TIE = 1e-9


def _num(df: pd.DataFrame, cols: "list[str]") -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pair(df: pd.DataFrame, keys: "list[str]", value_cols: "list[str]") -> pd.DataFrame:
    """Join every arm's rows to the baseline's on *keys*."""
    base = df[df["arm"] == "baseline"][keys + value_cols].rename(columns={c: f"base_{c}" for c in value_cols})
    return df[df["arm"] != "baseline"].merge(base, on=keys, how="inner", validate="many_to_one")


def gate_table(cuts: pd.DataFrame) -> pd.DataFrame:
    """Per arm: admitted-set changes at the shipped inclusion, and what moved them."""
    tcol, acol = f"threshold_i{GATE_INCLUSION}", f"n_admitted_i{GATE_INCLUSION}"
    df = _num(cuts.copy(), [tcol, acol, "seconds", "n", "n_unconverged"])
    paired = _pair(df, CASE_KEYS, [tcol, acol, "seconds", "provenance"])
    paired["d_threshold"] = (paired[tcol] - paired[f"base_{tcol}"]).abs()
    paired["d_admitted"] = (paired[acol] - paired[f"base_{acol}"]).abs()
    paired["d_admitted_frac"] = paired["d_admitted"] / paired["n"]
    paired["changed"] = paired["d_admitted"] > 0
    paired["thr_moved"] = paired["d_threshold"] > 0
    paired["speedup"] = paired["base_seconds"] / paired["seconds"].replace(0.0, np.nan)

    rows = []
    for arm, g in paired.groupby("arm", sort=False):
        n = len(g)
        rows.append(
            {
                "arm": arm,
                "cases": n,
                "rate_reportable": n >= MIN_CASES,
                "thr_moved_pct": 100.0 * g["thr_moved"].mean(),
                "admitted_moved_pct": 100.0 * g["changed"].mean(),
                "d_thr_median": g["d_threshold"].median(),
                "d_thr_p90": g["d_threshold"].quantile(0.90),
                "d_adm_frac_median": g["d_admitted_frac"].median(),
                "d_adm_frac_p90": g["d_admitted_frac"].quantile(0.90),
                "d_adm_frac_max": g["d_admitted_frac"].max(),
                "d_adm_median_of_changed": g.loc[g["changed"], "d_admitted"].median(),
                "prov_changed": int((g["provenance"] != g["base_provenance"]).sum()),
                "cut_speedup_median": g["speedup"].median(),
            }
        )
    return pd.DataFrame(rows)


def gate_by_inclusion(cuts: pd.DataFrame) -> pd.DataFrame:
    """The same rate at each inclusion - is the midpoint hiding a variance move?"""
    rows = []
    for v in INCLUSIONS:
        tcol, acol = f"threshold_i{v}", f"n_admitted_i{v}"
        df = _num(cuts.copy(), [tcol, acol, "n"])
        paired = _pair(df, CASE_KEYS, [tcol, acol])
        paired["changed"] = (paired[acol] - paired[f"base_{acol}"]).abs() > 0
        paired["frac"] = (paired[acol] - paired[f"base_{acol}"]).abs() / paired["n"]
        for arm, g in paired.groupby("arm", sort=False):
            rows.append(
                {
                    "arm": arm,
                    "inclusion": v,
                    "cases": len(g),
                    "admitted_moved_pct": 100.0 * g["changed"].mean(),
                    "d_adm_frac_median": g["frac"].median(),
                    "d_adm_frac_p90": g["frac"].quantile(0.9),
                }
            )
    return pd.DataFrame(rows).sort_values(["arm", "inclusion"])


def gate_by_env(cuts: pd.DataFrame) -> pd.DataFrame:
    """The gate rate split by environment - which is where it varies."""
    tcol, acol = f"threshold_i{GATE_INCLUSION}", f"n_admitted_i{GATE_INCLUSION}"
    df = _num(cuts.copy(), [tcol, acol, "n"])
    paired = _pair(df, CASE_KEYS, [tcol, acol])
    paired["changed"] = (paired[acol] - paired[f"base_{acol}"]).abs() > 0
    paired["frac"] = (paired[acol] - paired[f"base_{acol}"]).abs() / paired["n"]
    g = paired.groupby(["arm", "dataset", "embedder", "style"], sort=False)
    return (
        g.agg(
            cases=("changed", "size"),
            changed=("changed", "sum"),
            d_adm_frac_median=("frac", "median"),
            d_adm_frac_p90=("frac", lambda s: s.quantile(0.9)),
        )
        .assign(admitted_moved_pct=lambda d: 100.0 * d["changed"] / d["cases"])
        .reset_index()
    )


def incumbent_table(fits: pd.DataFrame) -> pd.DataFrame:
    """What the SHIPPED anchored refit does, per environment - half of the issue.

    A fit that exits on ``max_iter`` is not the estimator anyone specified, and
    the only reason anybody knows it happens is that a cost measurement went
    looking.  This is that measurement, on the whole corpus rather than the
    five folds the issue sampled.
    """
    df = _num(fits.copy(), ["n_iter", "converged", "n", "n_anchors", "init_seconds", "refit_seconds", "total_seconds"])
    df = df[(df["arm"] == "baseline") & (df["provenance"] == "anchored")]
    rows = []
    for keys, g in [(("ALL", "ALL", "ALL"), df), *df.groupby(["dataset", "embedder", "style"], sort=False)]:
        rows.append(
            {
                "dataset": keys[0],
                "embedder": keys[1],
                "style": keys[2],
                "folds": len(g),
                "n_median": g["n"].median(),
                "anchors_median": g["n_anchors"].median(),
                "iter_median": g["n_iter"].median(),
                "iter_p90": g["n_iter"].quantile(0.9),
                "hit_cap_pct": 100.0 * (1.0 - g["converged"].mean()),
                "init_ms": 1000 * g["init_seconds"].median(),
                "refit_ms": 1000 * g["refit_seconds"].median(),
                "refit_share_pct": 100.0 * (g["refit_seconds"].sum() / g["total_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Per arm: iterations, cost, and who wins each of the two objectives."""
    df = _num(
        fits.copy(),
        [
            "n_iter",
            "converged",
            "loglik_anchored",
            "loglik_free",
            "init_seconds",
            "refit_seconds",
            "total_seconds",
            "midpoint",
        ],
    )
    keys = [*CASE_KEYS, "fold"]
    paired = _pair(
        df,
        keys,
        ["loglik_anchored", "loglik_free", "refit_seconds", "total_seconds", "midpoint", "provenance", "n_iter"],
    )
    rows = []
    for arm, g in paired.groupby("arm", sort=False):
        anchored_only = g[(g["provenance"] == "anchored") & (g["base_provenance"] == "anchored")]
        d_a = anchored_only["loglik_anchored"] - anchored_only["base_loglik_anchored"]
        d_f = anchored_only["loglik_free"] - anchored_only["base_loglik_free"]
        d_mid = (anchored_only["midpoint"] - anchored_only["base_midpoint"]).abs()
        rows.append(
            {
                "arm": arm,
                "folds": len(g),
                "anchored_both": len(anchored_only),
                "lost_anchored_fit": int(
                    ((g["provenance"] != "anchored") & (g["base_provenance"] == "anchored")).sum()
                ),
                "iter_median": g["n_iter"].median(),
                "iter_p90": g["n_iter"].quantile(0.9),
                "hit_cap_pct": 100.0 * (1.0 - g["converged"].mean()),
                "refit_speedup": g["base_refit_seconds"].sum() / max(g["refit_seconds"].sum(), 1e-12),
                "fit_speedup": g["base_total_seconds"].sum() / max(g["total_seconds"].sum(), 1e-12),
                "obj_better": int((d_a > LL_TIE).sum()),
                "obj_worse": int((d_a < -LL_TIE).sum()),
                "obj_d_median": d_a.median(),
                "free_better": int((d_f > LL_TIE).sum()),
                "free_worse": int((d_f < -LL_TIE).sum()),
                "d_midpoint_median": d_mid.median(),
                "d_midpoint_max": d_mid.max(),
            }
        )
    return pd.DataFrame(rows)


def monotonicity_table(trace: pd.DataFrame) -> pd.DataFrame:
    """Per objective: how often an EM iteration made it *worse*.

    A decrease of more than :data:`LL_TIE` is a real one - below that it is
    float64 noise on a mean.  The weighted objective is what the M-step
    maximises, so it should never fall; the free sample's likelihood has no such
    guarantee once anchors pull the components, and "in practice it is fine" is
    a measurement, not an argument.
    """
    df = trace.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    rows = []
    for objective, g in df.groupby("objective", sort=False):
        steps = decreases = worst = 0.0
        traces = dropped = 0
        for _key, t in g.groupby(["cell", "case", "fold"], sort=False):
            v = t.sort_values("iteration")["value"].to_numpy(dtype=float)
            if v.size < 2:
                continue
            d = np.diff(v)
            traces += 1
            steps += d.size
            decreases += int((d < -LL_TIE).sum())
            if (d < -LL_TIE).any():
                dropped += 1
                worst = min(worst, float(d.min()))
        rows.append(
            {
                "objective": objective,
                "traces": traces,
                "iterations": int(steps),
                "decreasing_iterations": int(decreases),
                "decreasing_pct": 100.0 * decreases / steps if steps else float("nan"),
                "traces_with_a_decrease": dropped,
                "worst_single_step": worst,
            }
        )
    return pd.DataFrame(rows)


def bench_table(bench: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, n): min-of-k seconds for the refit, and the speedup over baseline."""
    df = _num(bench.copy(), ["init_seconds", "total_seconds", "refit_seconds", "n", "n_iter"])
    edges = [0, 1_000, 5_000, 20_000, 10**9]
    labels = ["<1k", "1k-5k", "5k-20k", ">20k"]
    df["n_bucket"] = np.where(
        df["resampled"] == 1,
        df["n"].astype(int).astype(str),
        pd.cut(df["n"], bins=edges, labels=labels, right=False).astype(str),
    )
    base = (
        df[df["arm"] == "baseline"]
        .groupby(["sample", "n_bucket"], sort=False)[["refit_seconds", "total_seconds"]]
        .median()
        .rename(columns={"refit_seconds": "base_refit", "total_seconds": "base_total"})
        .reset_index()
    )
    m = df.merge(base, on=["sample", "n_bucket"], how="left")
    m["refit_speedup"] = m["base_refit"] / m["refit_seconds"].replace(0.0, np.nan)
    m["fit_speedup"] = m["base_total"] / m["total_seconds"].replace(0.0, np.nan)
    g = m.groupby(["arm", "n_bucket", "resampled"], sort=False)
    return (
        g.agg(
            samples=("total_seconds", "size"),
            n_median=("n", "median"),
            init_ms=("init_seconds", lambda s: 1000 * s.median()),
            refit_ms=("refit_seconds", lambda s: 1000 * s.median()),
            fit_ms=("total_seconds", lambda s: 1000 * s.median()),
            iter_median=("n_iter", "median"),
            refit_speedup_median=("refit_speedup", "median"),
            fit_speedup_median=("fit_speedup", "median"),
        )
        .reset_index()
        .sort_values(["resampled", "n_median", "arm"])
    )


def _cell(v: object, sig: int) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{sig}g}"
    return str(v)


def _md(df: pd.DataFrame, sig: int = 3) -> str:
    """A markdown table, hand-rolled.

    ``DataFrame.to_markdown`` needs ``tabulate``, which this venv does not have
    and which is not worth a dependency for six tables - and a report that dies
    at the formatting step after an hour of analysis is the wrong failure to
    design in.  Significant digits rather than decimal places: a speedup and an
    admitted-set fraction cannot support the same precision.
    """
    cols = list(df.columns)
    rows = [[_cell(v, sig) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    out += ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows]
    return "\n".join(out)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True, help="directory holding the gate3825_*.csv frames")
    ap.add_argument("--out", default=None, help="where the aggregates go (default: <analysis>/agg)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    analysis = Path(args.analysis)
    out = Path(args.out) if args.out else analysis / "agg"
    out.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_csv(analysis / "gate3825_cuts.csv")
    fits = pd.read_csv(analysis / "gate3825_fits.csv")
    tables = {
        "gate_by_arm.csv": gate_table(cuts),
        "gate_by_inclusion.csv": gate_by_inclusion(cuts),
        "gate_by_env.csv": gate_by_env(cuts),
        "incumbent.csv": incumbent_table(fits),
        "fits_by_arm.csv": fit_table(fits),
    }
    trace_path = analysis / "gate3825_trace.csv"
    if trace_path.exists() and trace_path.stat().st_size > 0:
        trace = pd.read_csv(trace_path)
        if not trace.empty:
            tables["monotonicity.csv"] = monotonicity_table(trace)
    bench_path = analysis / "bench3825.csv"
    if bench_path.exists():
        tables["bench_by_n.csv"] = bench_table(pd.read_csv(bench_path))
    else:
        print(f"no {bench_path} - skipping the cost table")

    lines = ["# #3825 gate tables", ""]
    for name, df in tables.items():
        df.to_csv(out / name, index=False)
        lines += [f"## {name.removesuffix('.csv')}", "", _md(df), ""]
    (out / "TABLES.md").write_text("\n".join(lines))

    summary = {
        "cells": int(cuts["cell"].nunique()),
        "cases": int(cuts.groupby(["cell", "case"]).ngroups),
        "fold_fits": int(len(fits[fits["arm"] == "baseline"])),
        "arms": sorted(cuts["arm"].unique().tolist()),
        "gate_inclusion": GATE_INCLUSION,
    }
    (out / "summary_3825.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))
    print(f"wrote {out}/ ({', '.join(tables)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Turn the #3585 gate frames into the two tables the decision needs.

    python analyze_3585.py --analysis <dir>

**Table 1, the gate.** Per arm and per path (fold cut vs cosine/text sort): how
often does the candidate admit a different set of medias than the baseline
does?  That is the issue's stated criterion - not the size of the threshold
move, which the ``snap_cut_to_sample`` canonicalisation makes uninformative on
its own, but whether any media crosses the line.  ``|delta threshold|`` rides
along because a change with a large threshold move and no admitted-set change is
a different situation from one with neither.

**Table 2, the estimator.** When the two do differ, which one is the better fit?
Both arms maximise the same mean log-likelihood, so the sign of its difference
says whether the swap moved the cut toward the data or merely elsewhere.  A
candidate that changed 3% of admitted sets while winning the likelihood
comparison 3% of the time and losing it 0% is a different verdict from one that
changed 3% and split them evenly.

Pairing is exact: every arm sees the identical captured input, so a row pairs
with its baseline on ``(cell, kind, case)`` with nothing to match approximately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: The inclusion whose columns the gate turns on: the shipped default.
GATE_INCLUSION = 0

#: Below this many paired cases an arm's rate is not reported as a rate.  A
#: "0% of 12" reads like a result and is not one.
MIN_CASES = 30


def _pair(df: pd.DataFrame, keys: list[str], value_cols: list[str]) -> pd.DataFrame:
    """Join every arm's rows to the baseline's on *keys*."""
    base = df[df["arm"] == "baseline"][keys + value_cols].rename(columns={c: f"base_{c}" for c in value_cols})
    out = df[df["arm"] != "baseline"].merge(base, on=keys, how="inner", validate="many_to_one")
    return out


def gate_table(cuts: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, kind): admitted-set changes and the threshold moves behind them."""
    tcol, acol = f"threshold_i{GATE_INCLUSION}", f"n_admitted_i{GATE_INCLUSION}"
    df = cuts.copy()
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    paired = _pair(df, ["cell", "kind", "case"], [tcol, acol, "seconds", "provenance"])
    paired["d_threshold"] = (paired[tcol] - paired[f"base_{tcol}"]).abs()
    paired["d_admitted"] = (paired[acol] - paired[f"base_{acol}"]).abs()
    paired["changed"] = paired["d_admitted"] > 0
    paired["prov_changed"] = paired["provenance"] != paired["base_provenance"]
    paired["speedup"] = paired["base_seconds"] / paired["seconds"].replace(0.0, np.nan)

    rows = []
    for (arm, kind), g in paired.groupby(["arm", "kind"], sort=False):
        n = len(g)
        changed = int(g["changed"].sum())
        rows.append(
            {
                "arm": arm,
                "kind": kind,
                "cases": n,
                "changed": changed,
                "changed_pct": 100.0 * changed / n if n else float("nan"),
                "rate_reportable": n >= MIN_CASES,
                "prov_changed": int(g["prov_changed"].sum()),
                "d_thr_median": g["d_threshold"].median(),
                "d_thr_p90": g["d_threshold"].quantile(0.90),
                "d_thr_max": g["d_threshold"].max(),
                "d_adm_median_of_changed": g.loc[g["changed"], "d_admitted"].median(),
                "d_adm_max": g["d_admitted"].max(),
                "d_adm_frac_max": (g["d_admitted"] / g["n"]).max(),
                "speedup_median": g["speedup"].median(),
            }
        )
    return pd.DataFrame(rows)


def gate_by_env(cuts: pd.DataFrame) -> pd.DataFrame:
    """The same rate, split by the environment - which is where it varies."""
    tcol, acol = f"threshold_i{GATE_INCLUSION}", f"n_admitted_i{GATE_INCLUSION}"
    df = cuts.copy()
    df[acol] = pd.to_numeric(df[acol], errors="coerce")
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    paired = _pair(df, ["cell", "kind", "case"], [tcol, acol])
    paired["changed"] = (paired[acol] - paired[f"base_{acol}"]).abs() > 0
    paired["d_threshold"] = (paired[tcol] - paired[f"base_{tcol}"]).abs()
    g = paired.groupby(["arm", "kind", "dataset", "embedder", "style"], sort=False)
    return (
        g.agg(
            cases=("changed", "size"), changed=("changed", "sum"), d_thr_p90=("d_threshold", lambda s: s.quantile(0.9))
        )
        .assign(changed_pct=lambda d: 100.0 * d["changed"] / d["cases"])
        .reset_index()
    )


def fit_table(fits: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, kind): who wins the likelihood the two arms are both maximising."""
    df = fits.copy()
    df["loglik"] = pd.to_numeric(df["loglik"], errors="coerce")
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df["midpoint"] = pd.to_numeric(df["midpoint"], errors="coerce")
    paired = _pair(df, ["cell", "kind", "case", "sample"], ["loglik", "seconds", "midpoint"])
    paired["d_loglik"] = paired["loglik"] - paired["base_loglik"]
    paired["speedup"] = paired["base_seconds"] / paired["seconds"].replace(0.0, np.nan)
    # A tie is a difference below what float64 resolves on a mean over ~1e4
    # points; calling those "wins" would inflate every arm equally.
    tie = 1e-9
    rows = []
    for (arm, kind), g in paired.groupby(["arm", "kind"], sort=False):
        d = g["d_loglik"]
        rows.append(
            {
                "arm": arm,
                "kind": kind,
                "samples": len(g),
                "better": int((d > tie).sum()),
                "tied": int(d.abs().le(tie).sum()),
                "worse": int((d < -tie).sum()),
                "failed_here_only": int(g["loglik"].isna().sum() - g["base_loglik"].isna().sum()),
                "d_loglik_median": d.median(),
                "d_loglik_min": d.min(),
                "speedup_median": g["speedup"].median(),
                "seconds_median": g["seconds"].median(),
                "base_seconds_median": g["base_seconds"].median(),
            }
        )
    return pd.DataFrame(rows)


def bench_table(bench: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, n): min-of-k seconds and the speedup over the baseline."""
    df = bench.copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    df["n_bucket"] = df["n"].astype(int)
    base = (
        df[df["arm"] == "baseline"]
        .groupby(["sample", "n_bucket"], sort=False)["seconds"]
        .median()
        .rename("base_seconds")
        .reset_index()
    )
    merged = df.merge(base, on=["sample", "n_bucket"], how="left")
    merged["speedup"] = merged["base_seconds"] / merged["seconds"].replace(0.0, np.nan)
    g = merged.groupby(["arm", "n_bucket", "resampled"], sort=False)
    return (
        g.agg(
            samples=("seconds", "size"),
            seconds_median=("seconds", "median"),
            seconds_p90=("seconds", lambda s: s.quantile(0.9)),
            speedup_median=("speedup", "median"),
            speedup_min=("speedup", "min"),
        )
        .reset_index()
        .sort_values(["n_bucket", "arm"])
    )


def _cell(v: object, sig: int) -> str:
    """One table cell: integers as integers, floats at *sig* significant digits."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{sig}g}"
    return str(v)


def sort_latency(cuts: pd.DataFrame, corpus: Path) -> pd.DataFrame:
    """What a whole cosine/text sort costs, before and after.

    The gate times the fit; the sort capture times the rest of the sort
    (``cosine_sort_with_boxes`` - the scoring pass, the result dicts and the
    sort itself) for the same query on the same hardware.  Together they turn a
    speedup on the fit into the number a user would feel, which is the one this
    issue sized itself on: the fit was measured at 91-95% of a sort, so the
    interesting question is what fraction it is *afterwards*.
    """
    seconds: dict[tuple[str, str, str], float] = {}
    fits: dict[tuple[str, str, str], dict] = {}
    for path in sorted(corpus.glob("sorts*.npz")):
        with np.load(path) as z:
            if "_meta" not in z:
                continue
            meta = json.loads(bytes(z["_meta"].tobytes()).decode("utf-8"))
        for key, value in (meta.get("sort_seconds") or {}).items():
            _kind, dataset, embedder, category = key.split("|")
            seconds[(dataset, embedder, category)] = float(value)
        for key, value in (meta.get("fit_seconds") or {}).items():
            _kind, dataset, embedder, category = key.split("|")
            fits[(dataset, embedder, category)] = value
    if not seconds:
        return pd.DataFrame()

    df = cuts[cuts["kind"] == "sort"].copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    ids = list(zip(df["dataset"].astype(str), df["embedder"].astype(str), df["case"].astype(str), strict=True))
    df["rest_seconds"] = [seconds.get(k, float("nan")) for k in ids]
    # Prefer the timings taken in the SAME process as the sort they are a
    # fraction of (min of 5).  Timing the fit in one job and the rest of the
    # sort in another puts a cross-node difference straight into the ratio, and
    # this cluster has nodes that differ (#3160).  An arm the capture did not
    # time falls back to the gate's single call, and the column says how many
    # rows are which rather than mixing them silently.
    df["same_process"] = [1 if str(a) in fits.get(k, {}) else 0 for k, a in zip(ids, df["arm"], strict=True)]
    df["seconds"] = [
        float(fits[k][str(a)]) if str(a) in fits.get(k, {}) else sec
        for k, a, sec in zip(ids, df["arm"], df["seconds"], strict=True)
    ]
    df = df.dropna(subset=["rest_seconds"])
    if df.empty:
        return pd.DataFrame()
    base = df[df["arm"] == "baseline"].set_index(["cell", "case"])["seconds"].rename("base_fit")
    df = df.join(base, on=["cell", "case"])
    df["sort_before"] = df["rest_seconds"] + df["base_fit"]
    df["sort_after"] = df["rest_seconds"] + df["seconds"]
    rows = []
    for arm, g in df.groupby("arm", sort=False):
        rows.append(
            {
                "arm": arm,
                "sorts": len(g),
                "same_process_timing": int(g["same_process"].sum()),
                "n_median": int(g["n"].median()),
                "rest_ms": 1000 * g["rest_seconds"].median(),
                "fit_ms": 1000 * g["seconds"].median(),
                "fit_share_pct": 100 * (g["seconds"] / g["sort_after"]).median(),
                "sort_ms": 1000 * g["sort_after"].median(),
                "sort_speedup": (g["sort_before"] / g["sort_after"]).median(),
            }
        )
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, sig: int = 3) -> str:
    """A markdown table, hand-rolled.

    ``DataFrame.to_markdown`` needs ``tabulate``, which this venv does not
    have and which is not worth a dependency for four tables - and a report
    that dies at the formatting step after an hour of analysis is the wrong
    failure to design in.  Significant digits rather than decimal places,
    because a fixed ``%.3f`` prints a speedup and an admitted-set fraction with
    the same precision and only one of them can support it.
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
    ap.add_argument("--analysis", required=True, help="directory holding gate_cuts.csv / gate_fits.csv")
    ap.add_argument("--out", default=None, help="where the aggregates go (default: <analysis>/agg)")
    ap.add_argument("--corpus", default=None, help="the capture dir, for the whole-sort latency table")
    args = ap.parse_args(list(argv) if argv is not None else None)

    analysis = Path(args.analysis)
    out = Path(args.out) if args.out else analysis / "agg"
    out.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_csv(analysis / "gate_cuts.csv")
    fits = pd.read_csv(analysis / "gate_fits.csv")
    tables = {
        "gate_by_arm.csv": gate_table(cuts),
        "gate_by_env.csv": gate_by_env(cuts),
        "fits_by_arm.csv": fit_table(fits),
    }
    if args.corpus:
        latency = sort_latency(cuts, Path(args.corpus))
        if latency.empty:
            print(f"no per-sort timings under {args.corpus} - skipping the latency table", file=sys.stderr)
        else:
            tables["sort_latency.csv"] = latency
    bench_path = analysis / "bench.csv"
    if bench_path.exists():
        tables["bench_by_n.csv"] = bench_table(pd.read_csv(bench_path))
    else:
        print(f"no {bench_path} - skipping the cost table", file=sys.stderr)

    lines = ["# #3585 gate tables", ""]
    for name, df in tables.items():
        df.to_csv(out / name, index=False)
        lines += [f"## {name.removesuffix('.csv')}", "", _md(df), ""]
    (out / "TABLES.md").write_text("\n".join(lines))

    summary = {
        "cells": int(cuts["cell"].nunique()),
        "cases": int(cuts.groupby(["cell", "kind", "case"]).ngroups),
        "arms": sorted(cuts["arm"].unique().tolist()),
        "gate_inclusion": GATE_INCLUSION,
    }
    (out / "summary_3585.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))
    print(f"wrote {out}/ ({', '.join(tables)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

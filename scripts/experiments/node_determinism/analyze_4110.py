"""Pair #4110's arms cell by cell: is a cell bit-identical across node types?

Every arm ran the same frozen commit on the same grid, so a cell is identified by
its output file name (``task_<index>.csv.gz``) in each arm.  A pair of cells is

* ``identical``   - every printed value matches, timing columns aside, in the
  steps file and in every sidecar (picks, cut diagnostics, line counts, ...);
* ``trajectory``  - a different item was picked at some vote, after which the
  two sessions are different sessions;
* ``objective``   - the same picks, but a vote's F-beta, precision, recall,
  flagged count or cost differs on the base row (the row every study reads);
* ``threshold``   - the same picks and objective, but a base row's printed
  threshold or cut differs (the #3840 shape: one unit in the last digit);
* ``diagnostics`` - the base rows' picks, threshold and objective all agree; what
  differs is a counterfactual variant row or a measurement column, such as the
  honest oracle, whose resampling is seeded from a hash of the score array and
  so redraws on any last-bit change.

Timing columns (``*_seconds``, ``backend``, ``device``) are the only ones left out.

    python analyze_4110.py --base /expscratch/$USER/det-4110 --out /expscratch/$USER/det-4110/analysis
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import _cells_io  # noqa: E402  # the one cell reader: base rows, spot check set apart

#: (left, right, what the pair answers).  ``a_base`` doubles as AMD + numpy pin:
#: the pin disables AVX-512 dispatch, which the EPYC 7713 does not have.
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("a_base", "a_rep", "the same arm twice on one node type"),
    ("i_base", "a_base", "today's dev, Intel vs AMD"),
    ("i_npy", "a_base", "numpy pin only, Intel vs AMD"),
    ("i_pin", "a_pin", "numpy + MKL pins, Intel vs AMD"),
    ("i_full", "a_full", "numpy + MKL + ATen pins, Intel vs AMD"),
    ("i_ship", "i_full", "the shipped pins vs the tested ones, Intel"),
    ("a_ship", "a_full", "the shipped pins vs the tested ones, AMD"),
    ("i_ship", "a_ship", "the shipped pins, Intel vs AMD"),
    ("a_base", "a_full", "what the three pins move on AMD"),
    ("i_base", "i_full", "what the three pins move on Intel"),
    ("i_base", "i_npy", "what the numpy pin moves on Intel"),
    ("a_base", "a_pin", "what the MKL pin moves on AMD"),
)
TIMING = re.compile(r"(_seconds$|^backend$|^device$)")
#: Printed metrics whose largest gap is reported for a diverged pair.
METRICS = ("threshold", "fbeta", "cost", "average_precision")
#: Base-row columns a study reads as the outcome of a vote.
OBJECTIVE = ("fbeta", "fbeta_b025", "fbeta_b1", "fbeta_b4", "precision", "recall", "n_flagged", "cost")
#: Base-row columns that are the cut itself.
CUTS = ("threshold", "acq_threshold", "gmm_cut", "live_threshold")


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df[[c for c in df.columns if not TIMING.search(c)]]


def _cells(arm_dir: Path) -> dict[str, Path]:
    d = arm_dir / "results" / "cells"
    return {p.name.split(".")[0]: p for p in sorted(d.glob("task_[0-9][0-9][0-9][0-9].csv*")) if p.stat().st_size > 0}


def _sidecars(main: Path) -> dict[str, Path]:
    stem = main.name.split(".")[0]
    return {p.name[len(stem) :]: p for p in main.parent.glob(f"{stem}__*") if p.stat().st_size > 0}


def _first_diff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int | None, str | None]:
    """(row, column) of the first printed difference, or (None, None) if equal."""
    if list(a.columns) != list(b.columns):
        return 0, "columns"
    n = min(len(a), len(b))
    neq = a.iloc[:n].to_numpy() != b.iloc[:n].to_numpy()
    rows = np.flatnonzero(neq.any(axis=1))
    if rows.size:
        r = int(rows[0])
        return r, str(a.columns[int(np.flatnonzero(neq[r])[0])])
    if len(a) != len(b):
        return n, "rows"
    return None, None


def _max_gap(a: pd.DataFrame, b: pd.DataFrame, col: str) -> float | None:
    if col not in a.columns or len(a) != len(b):
        return None
    x = pd.to_numeric(a[col], errors="coerce").to_numpy()
    y = pd.to_numeric(b[col], errors="coerce").to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.max(np.abs(x[ok] - y[ok]))) if ok.any() else None


def _clicks(path: Path) -> pd.DataFrame:
    """One row a vote: the production rows of a cell, in vote order (``_cells_io``'s contract)."""
    df = _cells_io._base_rows(pd.read_csv(path, low_memory=False))
    df = df[~_cells_io.check_rows(df)]
    if "prevalence_arm" in df.columns:  # the grid's one pool, should a cell carry two
        df = df[df["prevalence_arm"] == df["prevalence_arm"].iloc[0]] if len(df) else df
    return df.sort_values("t", kind="stable").reset_index(drop=True)


def compare_cell(left: Path, right: Path) -> dict:
    a, b = _read(left), _read(right)
    row = {"cell": left.name.split(".")[0], "category": a["category"].iat[0] if len(a) else ""}
    r, col = _first_diff(a, b)
    row["steps_identical"] = r is None
    row["first_diff_t"] = int(a["t"].iat[r]) if r is not None and r < len(a) and "t" in a else None
    row["first_diff_col"] = col
    for m in METRICS:
        row[f"max_gap_{m}"] = _max_gap(a, b, m) if r is not None else 0.0
    # What a reader of the objective sees: the cell's base rows (one a vote,
    # read as every study reads them), the share of votes whose F-beta moved, and
    # the paired gap (left - right) averaged over the votes and at the last one.
    # Paired on the vote: two sessions that part need not write a base row at the same votes.
    ba, bb = _clicks(left), _clicks(right)
    m = ba[["t", "fbeta"]].merge(bb[["t", "fbeta"]], on="t", suffixes=("_l", "_r"))
    if len(m):
        x, y = m["fbeta_l"].to_numpy(dtype=float), m["fbeta_r"].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        row["frac_votes_fbeta_moved"] = float(np.mean(x[ok] != y[ok])) if ok.any() else 0.0
        row["mean_gap_fbeta"] = float(np.mean(x[ok] - y[ok])) if ok.any() else 0.0
        row["mean_abs_gap_fbeta"] = float(np.mean(np.abs(x[ok] - y[ok]))) if ok.any() else 0.0
        row["final_gap_fbeta"] = float(x[-1] - y[-1]) if np.isfinite(x[-1]) and np.isfinite(y[-1]) else None
    ls, rs = _sidecars(left), _sidecars(right)
    differing = sorted(
        k
        for k in ls.keys() | rs.keys()
        if k not in ls or k not in rs or _first_diff(_read(ls[k]), _read(rs[k]))[0] is not None
    )
    row["sidecars_differing"] = ",".join(differing)
    picks = "__picks.csv.gz" if "__picks.csv.gz" in ls else "__picks.csv"
    if picks in ls and picks in rs:
        pa, pb = _read(ls[picks]), _read(rs[picks])
        ida, idb = pa["picked_id"].tolist(), pb["picked_id"].tolist()
        div = next((i for i, (x, y) in enumerate(zip(ida, idb, strict=False)) if x != y), None)
        if div is None and len(ida) != len(idb):
            div = min(len(ida), len(idb))
        row["pick_diverge_t"] = (
            int(pa["t"].iat[div]) if div is not None and div < len(pa) else (None if div is None else -1)
        )
    else:
        row["pick_diverge_t"] = None
    mt = ba[["t", "threshold"]].merge(bb[["t", "threshold"]], on="t", suffixes=("_l", "_r"))
    gap_t = (mt["threshold_l"] - mt["threshold_r"]).abs()
    row["base_max_gap_threshold"] = float(gap_t.max()) if gap_t.notna().any() else 0.0
    row["base_votes_threshold_moved"] = int((gap_t > 0).sum())
    moved = _base_cols_differing(ba, bb)
    row["base_cols_differing"] = ",".join(moved)
    if row["pick_diverge_t"] is not None:
        row["verdict"] = "trajectory"
    elif row["steps_identical"] and not differing:
        row["verdict"] = "identical"
    elif any(c in moved for c in OBJECTIVE):
        row["verdict"] = "objective"
    elif any(c in moved for c in CUTS):
        row["verdict"] = "threshold"
    else:
        row["verdict"] = "diagnostics"
    return row


def _base_cols_differing(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Columns of the base rows (one a vote) whose printed value differs at any vote."""
    if len(a) != len(b):
        return ["rows"]
    out = []
    for c in a.columns:
        if c not in b.columns or TIMING.search(c):
            continue
        x, y = a[c], b[c]
        if not ((x == y) | (x.isna() & y.isna())).all():
            out.append(c)
    return out


def cell_seconds(arm_dir: Path) -> pd.Series:
    """Wall seconds per cell: the largest ``elapsed_seconds`` its steps file printed."""
    out = {}
    for name, p in _cells(arm_dir).items():
        df = pd.read_csv(p, usecols=["elapsed_seconds"])
        out[name] = float(df["elapsed_seconds"].max())
    return pd.Series(out, dtype=float)


def nodes(arm_dir: Path) -> pd.DataFrame:
    cols = [
        "job",
        "task",
        "host",
        "cpu",
        "avx512",
        "np_avx512",
        "torch_cap",
        "openblas",
        "npy_disable",
        "mkl_cbwr",
        "numpy",
    ]
    p = arm_dir / "logs" / "nodes.tsv"
    return pd.read_csv(p, sep="\t", names=cols, dtype=str) if p.exists() else pd.DataFrame(columns=cols)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    arms = sorted({x for p in PAIRS for x in p[:2]})
    hw = {}
    for arm in arms:
        n = nodes(args.base / arm)
        hw[arm] = {
            "cpus": sorted(n["cpu"].unique().tolist()),
            "torch_cap": sorted(n["torch_cap"].unique().tolist()),
            "np_avx512": sorted(n["np_avx512"].unique().tolist()),
            "mkl_cbwr": sorted(n["mkl_cbwr"].unique().tolist()),
            "npy_disable": sorted(n["npy_disable"].unique().tolist()),
        }
    timing = {arm: cell_seconds(args.base / arm) for arm in arms}

    summary, rows = [], []
    for left, right, what in PAIRS:
        lc, rc = _cells(args.base / left), _cells(args.base / right)
        common = sorted(lc.keys() & rc.keys())
        per = [compare_cell(lc[c], rc[c]) | {"left": left, "right": right} for c in common]
        rows.extend(per)
        df = pd.DataFrame(per)
        counts = df["verdict"].value_counts().to_dict() if len(df) else {}
        traj = df[df["verdict"] == "trajectory"] if len(df) else df
        summary.append(
            {
                "pair": f"{left} ~ {right}",
                "question": what,
                "cells": len(common),
                "identical": counts.get("identical", 0),
                "trajectory": counts.get("trajectory", 0),
                "objective": counts.get("objective", 0),
                "threshold": counts.get("threshold", 0),
                "diagnostics": counts.get("diagnostics", 0),
                "base_cols_differing": df["base_cols_differing"]
                .str.split(",")
                .explode()
                .loc[lambda c: c.fillna("") != ""]
                .value_counts()
                .head(12)
                .to_dict()
                if len(df)
                else {},
                "median_first_diff_t": float(df["first_diff_t"].dropna().median())
                if len(df) and df["first_diff_t"].notna().any()
                else None,
                "median_pick_diverge_t": float(traj["pick_diverge_t"].median()) if len(traj) else None,
                "max_gap_fbeta": float(df["max_gap_fbeta"].max())
                if len(df) and df["max_gap_fbeta"].notna().any()
                else None,
                "frac_votes_fbeta_moved": float(df["frac_votes_fbeta_moved"].mean())
                if "frac_votes_fbeta_moved" in df
                else None,
                # Mean over cells of each cell's mean paired gap, and its standard error.
                "mean_gap_fbeta": float(df["mean_gap_fbeta"].mean()) if "mean_gap_fbeta" in df else None,
                "se_gap_fbeta": float(df["mean_gap_fbeta"].std(ddof=1) / np.sqrt(len(df)))
                if "mean_gap_fbeta" in df and len(df) > 1
                else None,
                "mean_abs_gap_fbeta": float(df["mean_abs_gap_fbeta"].mean()) if "mean_abs_gap_fbeta" in df else None,
                "final_gap_fbeta_mean": float(df["final_gap_fbeta"].mean()) if "final_gap_fbeta" in df else None,
                "final_gap_fbeta_max_abs": float(df["final_gap_fbeta"].abs().max())
                if "final_gap_fbeta" in df
                else None,
                "median_max_gap_threshold": float(df["max_gap_threshold"].median())
                if len(df) and df["max_gap_threshold"].notna().any()
                else None,
                # On the base rows only (one a vote): the cut the user's detector draws.
                "base_threshold_gap_median": float(df["base_max_gap_threshold"].median())
                if "base_max_gap_threshold" in df
                else None,
                "base_threshold_gap_p90": float(df["base_max_gap_threshold"].quantile(0.9))
                if "base_max_gap_threshold" in df
                else None,
                "base_threshold_gap_max": float(df["base_max_gap_threshold"].max())
                if "base_max_gap_threshold" in df
                else None,
                "base_votes_threshold_moved_median": float(df["base_votes_threshold_moved"].median())
                if "base_votes_threshold_moved" in df
                else None,
                "first_diff_cols": df["first_diff_col"].dropna().value_counts().head(5).to_dict() if len(df) else {},
            }
        )
    pd.DataFrame(rows).to_csv(args.out / "pairs.csv", index=False)
    secs = {
        arm: {
            "cells": int(s.size),
            "median_s": float(s.median()) if s.size else None,
            "mean_s": float(s.mean()) if s.size else None,
        }
        for arm, s in timing.items()
    }
    # Paired per-cell time ratios: what a pin costs a whole cell on one node type.
    ratios = {}
    for a, b in (
        ("i_npy", "i_base"),
        ("i_pin", "i_base"),
        ("i_full", "i_base"),
        ("a_pin", "a_base"),
        ("a_full", "a_base"),
        ("a_rep", "a_base"),
        ("i_base", "a_base"),
    ):
        j = timing[a].index.intersection(timing[b].index)
        if len(j):
            r = timing[a][j] / timing[b][j]
            ratios[f"{a}/{b}"] = {"cells": len(j), "median": float(r.median()), "mean": float(r.mean())}
    result = {"pairs": summary, "hardware": hw, "cell_seconds": secs, "time_ratios": ratios}
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Read Rico Suave runs (#4356): does the app work on app screens, per class and balance preset.

Each run directory is one balance preset (``launch.sh``, ``runs/<date>-b<beta>``). Every cell is one
class on one seed. Written under ``--out``:

* ``runs.csv`` - one row per run (preset x class x seed): the session means, AP, the full-label
  ceiling's AP (its rows carry no F-beta), the class's prevalence on the withheld half, and when a
  detector was first shown.
* ``curves.csv`` - the mean over runs of F-beta, precision and recall at every vote 1-150, per preset
  (and per class).
* ``summary.md`` - the tables a reader starts from.

**The objective is what Find returns**: at each vote, F-beta at the session's balance of the withheld
half above the threshold of the detector a Find at that vote gives. Every run is scored at every vote;
between rows the last value carries, and a vote before the first row scores 0. **The session mean**
is the mean of F-beta over votes 1-150, the owner's single number (2026-10-09, #4668).

**Prevalence is the class's, not chosen.** The pool is the twelve classes' agreed screens, so
settings is about a third of it and cart about 1%. Each class is therefore also read against the
F-beta of returning the whole withheld half (precision = prevalence, recall = 1), the floor a
detector has to clear to be doing anything.

``--ref label=<run dir>`` reads another dataset's runs with the same reader (the State of the App's
coco_better cells) as a photo reference at the same preset.

Usage::

    python analyze.py --run b025=<runs/...-b025> --run b1=<runs/...-b1> --run b4=<runs/...-b4> \\
        [--ref b1=<state-of-the-app/...-b1> ...] --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "calibration"))
import _cells_io  # noqa: E402

HORIZON = 150
SHORT = 50
CEILING = "skyline_train_full"
COLS = [
    "dataset", "category", "embedder", "seed", "t", "phase", "beta", "fbeta", "precision", "recall",
    "average_precision", "app_trained", "n_test_pos", "n_test_neg", "pool_variant", "gmm_variant",
    "schedule", "seed_mode",
]  # fmt: skip


def _carry(t: np.ndarray, v: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Values at votes 1..horizon, each the last row at or before it; 0 before the first row."""
    out = np.zeros(horizon)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    idx = np.searchsorted(t, np.arange(1, horizon + 1), side="right") - 1
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return out


def everything_fbeta(prevalence: float, beta: float) -> float:
    """F-beta of returning the whole withheld half: precision = prevalence, recall = 1."""
    b2 = beta * beta
    return (1 + b2) * prevalence / (b2 * prevalence + 1) if prevalence > 0 else 0.0


def read_cell(path: str) -> dict | None:
    """One cell file -> one run's curves and scalars, or ``None`` for an empty cell."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except (pd.errors.EmptyDataError, OSError):
        return None
    if df.empty:
        return None
    sky = df[df["gmm_variant"].fillna("").astype(str) == CEILING]
    base = _cells_io._base_rows(df)
    check = base[_cells_io.check_rows(base)]
    ordinary = base[~_cells_io.check_rows(base)]
    ordinary = ordinary[ordinary["t"] <= HORIZON].groupby("t", as_index=False).last()
    t = ordinary["t"].to_numpy(dtype=float)
    curves = {m: _carry(t, ordinary[m].fillna(0.0).to_numpy(dtype=float)) for m in ("fbeta", "precision", "recall")}
    ap = _carry(t, ordinary["average_precision"].fillna(0.0).to_numpy(dtype=float))
    shown = ordinary.loc[ordinary["app_trained"].fillna(0).astype(float) > 0, "t"]
    first = base.iloc[0]
    n_pos, n_neg = float(first["n_test_pos"]), float(first.get("n_test_neg", np.nan))
    beta = float(first["beta"]) if pd.notna(first.get("beta")) else 1.0
    prev = n_pos / (n_pos + n_neg) if n_pos + n_neg > 0 else np.nan
    return {
        "dataset": first["dataset"],
        "category": first["category"],
        "embedder": first["embedder"],
        "seed": int(first["seed"]),
        "beta": beta,
        "seed_mode": first.get("seed_mode"),
        "n_test_pos": n_pos,
        "n_test_neg": n_neg,
        "prevalence": prev,
        "everything_fbeta": everything_fbeta(prev, beta),
        "fbeta_first": float(curves["fbeta"][0]),
        "fbeta_mean": float(curves["fbeta"].mean()),
        "fbeta_mean_short": float(curves["fbeta"][:SHORT].mean()),
        "fbeta_at_end": float(curves["fbeta"][-1]),
        "fbeta_after_check": float(check.sort_values("t")["fbeta"].iloc[-1])
        if len(check)
        else float(curves["fbeta"][-1]),
        "ap_at_end": float(ap[-1]),
        "ceiling_fbeta": float(sky["fbeta"].iloc[-1]) if len(sky) else np.nan,
        "ceiling_ap": float(sky["average_precision"].iloc[-1]) if len(sky) else np.nan,
        "first_shown": float(shown.min()) if len(shown) else np.nan,
        "_curves": curves,
    }


def load_run(label: str, run: Path, workers: int) -> tuple[pd.DataFrame, list[dict], int]:
    """``(runs, curve records, empty cells)`` for one run directory."""
    files = [str(f) for f in _cells_io.main_frame_files(run / "results" / "cells")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(read_cell, files, chunksize=8))
    rows = [g for g in got if g is not None]
    recs = []
    for r in rows:
        c = r.pop("_curves")
        r["preset"] = label
        recs.append({"preset": label, "dataset": r["dataset"], "category": r["category"], **c})
    return pd.DataFrame(rows), recs, len(got) - len(rows)


def _fmt(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2f}"


def summary(runs: pd.DataFrame, ref: pd.DataFrame, empty: dict[str, int], labels: list[str]) -> str:
    rico = runs[runs["dataset"] == "rico_suave"]
    lines = [
        "# Rico Suave v0 (#4356)",
        "",
        "F-beta at the session's preset of what Find returns on the withheld half, every run at every vote "
        f"(split by app). Runs per preset: {rico.groupby('preset').size().to_dict()}. Empty cells: {empty}.",
        "",
        "## Per preset, every class",
        "",
        "| preset | dataset | runs | session mean F-beta (1-150) | short (1-50) | at vote 1 | at 150 | after the check "
        "| AP at 150 | ceiling AP | return-everything F-beta | first detector (median vote) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in labels:
        for name, g in (("rico_suave", rico[rico["preset"] == k]), ("coco_better (ref)", ref[ref["preset"] == k])):
            if g.empty:
                continue
            lines.append(
                f"| {k} | {name} | {len(g)} | {_fmt(g['fbeta_mean'].mean())} | {_fmt(g['fbeta_mean_short'].mean())} | "
                f"{_fmt(g['fbeta_first'].mean())} | {_fmt(g['fbeta_at_end'].mean())} | "
                f"{_fmt(g['fbeta_after_check'].mean())} | {_fmt(g['ap_at_end'].mean())} | "
                f"{_fmt(g['ceiling_ap'].mean())} | "
                f"{_fmt(g['everything_fbeta'].mean())} | {g['first_shown'].median():.0f} |"
            )
    lines += [
        "",
        "## Per class (session mean F-beta over votes 1-150, the floor of returning everything, AP at 150 "
        "and the full-label ceiling's AP; mean over seeds)",
        "",
        "| class | prevalence | withheld positives | "
        + " | ".join(f"{k} mean | {k} everything" for k in labels)
        + " | AP at 150 | ceiling AP |",
        "|---|---|---|" + "---|---|" * len(labels) + "---|---|",
    ]
    order = rico.groupby("category")["prevalence"].mean().sort_values(ascending=False).index
    for c in order:
        g = rico[rico["category"] == c]
        cells = []
        for k in labels:
            h = g[g["preset"] == k]
            cells += [_fmt(h["fbeta_mean"].mean()), _fmt(h["everything_fbeta"].mean())]
        cells += [_fmt(g["ap_at_end"].mean()), _fmt(g["ceiling_ap"].mean())]
        lines.append(
            f"| {c} | {g['prevalence'].mean():.1%} | {g['n_test_pos'].mean():.0f} "
            f"[{g['n_test_pos'].min():.0f}-{g['n_test_pos'].max():.0f}] | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="preset=run_dir, e.g. b1=runs/2026-10-14-b1")
    ap.add_argument("--ref", action="append", default=[], help="preset=run_dir of another dataset at that preset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    frames, refs, recs, empty, labels = [], [], [], {}, []
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs, r, n_empty = load_run(label, Path(path), args.workers)
        frames.append(runs)
        recs += r
        empty[label] = n_empty
        labels.append(label)
    for spec in args.ref:
        label, _, path = spec.partition("=")
        runs, r, _ = load_run(label, Path(path), args.workers)
        refs.append(runs)
        recs += r
    runs = pd.concat(frames, ignore_index=True)
    ref = pd.concat(refs, ignore_index=True) if refs else pd.DataFrame(columns=runs.columns)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat([runs, ref], ignore_index=True).to_csv(args.out / "runs.csv", index=False)
    rows = []
    groups: dict[tuple, list[dict]] = {}
    for rec in recs:
        groups.setdefault((rec["preset"], rec["dataset"], "all"), []).append(rec)
        if rec["dataset"] == "rico_suave":
            groups.setdefault((rec["preset"], rec["dataset"], rec["category"]), []).append(rec)
    for (preset, ds, cat), rs in groups.items():
        means = {m: np.mean(np.stack([r[m] for r in rs]), axis=0) for m in ("fbeta", "precision", "recall")}
        for i in range(HORIZON):
            rows.append(
                {"preset": preset, "dataset": ds, "category": cat, "t": i + 1, "n": len(rs)}
                | {m: means[m][i] for m in means}
            )
    pd.DataFrame(rows).to_csv(args.out / "curves.csv", index=False)
    (args.out / "summary.md").write_text(summary(runs, ref, empty, labels))
    print((args.out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

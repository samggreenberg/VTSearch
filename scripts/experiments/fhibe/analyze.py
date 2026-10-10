#!/usr/bin/env python
"""Read FHIBE baseline runs (#4699): what Find returns per person, at every vote, per arm and example count.

Each run directory is one example count K (``launch.sh``, ``runs/<date>-k<K>``). Every cell is one
person on one arm. Written under ``--out``:

* ``runs.csv`` - one row per run (K x arm x person x seed): the session means, the end-check result,
  AP, the ceiling, whether Autopilot left its Good phase and when a detector was first shown, and
  the Goods found by clicks.
* ``curves.csv`` - the mean over people of F-beta, precision and recall at every vote 1-150, per
  K x arm.
* ``summary.md`` - the tables a reader starts from, with paired contrasts.

**The objective is what Find returns.** At each vote, a row's ``precision``/``recall``/``fbeta`` are
the withheld half above the threshold of the detector a Find at that vote gives (the Goods' centroid
under the label quota, #4643, then the trained head), at the session's beta. Every run is scored at
every vote. Between rows (the weak-separation check casts its round at once) the last value
carries. The end-of-run check is not a vote; it is reported on its own.

**Votes, not clicks.** The K examples are the run's first K votes (phase ``example``), so K=4 has
spent 4 votes on photos the user brought. ``curves.csv`` keys on the vote ``t``; clicks after the
examples are ``t - K``.

**The session mean** is the mean of F-beta over votes 1-150 (and 1-50 for short sessions), the
owner's single number for "did it improve" (2026-10-09, #4668).

Usage::

    python analyze.py --run k1=<runs/...-k1> --run k4=<runs/...-k4> --out <dir>
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
    "average_precision", "app_trained", "n_test_pos", "pool_variant", "gmm_variant", "schedule",
]  # fmt: skip

#: The arms in reading order, and how a table names them.
ARMS = {
    "fhibe_1024": "photo 1024 (SigLIP)",
    "fhibe_640": "photo 640 (SigLIP)",
    "fhibe_faces_1024": "face 1024 (FaceNet)",
    "fhibe_faces_640": "face 640 (FaceNet)",
}


def _carry(t: np.ndarray, v: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Values at votes 1..horizon, each the last row at or before it; 0 before the first row."""
    out = np.zeros(horizon)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    idx = np.searchsorted(t, np.arange(1, horizon + 1), side="right") - 1
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return out


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
    picks_path = Path(path.replace(".csv", "__picks.csv").replace(".gz__picks", "__picks"))
    picks = pd.read_csv(picks_path) if picks_path.exists() and picks_path.stat().st_size else pd.DataFrame()
    left_good = np.nan
    found = 0
    if not picks.empty:
        clicks = picks[(picks["t"] <= HORIZON) & (picks["phase"].astype(str) != _cells_io.CHECK_PHASE)]
        past = clicks.loc[~clicks["phase"].astype(str).isin(["example", "good"]), "t"]
        left_good = float(past.min()) if len(past) else np.nan
        found = int(((clicks["phase"].astype(str) != "example") & (clicks["picked_label"] == 1)).sum())
    first = base.iloc[0]
    return {
        "dataset": first["dataset"],
        "category": first["category"],
        "embedder": first["embedder"],
        "seed": int(first["seed"]),
        "beta": first.get("beta"),
        "n_test_pos": float(first["n_test_pos"]),
        "fbeta_mean": float(curves["fbeta"].mean()),
        "fbeta_mean_short": float(curves["fbeta"][:SHORT].mean()),
        "fbeta_at_end": float(curves["fbeta"][-1]),
        "fbeta_after_check": float(check.sort_values("t")["fbeta"].iloc[-1])
        if len(check)
        else float(curves["fbeta"][-1]),
        "ap_mean": float(ap.mean()),
        "ap_at_end": float(ap[-1]),
        "ceiling_fbeta": float(sky["fbeta"].iloc[-1]) if len(sky) else np.nan,
        "ceiling_ap": float(sky["average_precision"].iloc[-1]) if len(sky) else np.nan,
        "first_shown": float(shown.min()) if len(shown) else np.nan,
        "left_good": left_good,
        "goods_found": found,
        "_curves": curves,
    }


def load_run(label: str, run: Path, workers: int) -> tuple[pd.DataFrame, dict, int]:
    """``(runs, curves by (K, arm), empty cells)`` for one run directory."""
    files = [str(f) for f in _cells_io.main_frame_files(run / "results" / "cells")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(read_cell, files, chunksize=8))
    rows = [g for g in got if g is not None]
    curves: dict[tuple, dict[str, list[np.ndarray]]] = {}
    for r in rows:
        c = r.pop("_curves")
        r["K"] = label
        for m, v in c.items():
            curves.setdefault((label, r["dataset"]), {}).setdefault(m, []).append(v)
    return pd.DataFrame(rows), curves, len(got) - len(rows)


def paired(runs: pd.DataFrame, a: tuple, b: tuple, metric: str, reps: int = 2000, seed: int = 0) -> dict:
    """Mean of (b - a) over people both cells ran, with a 95% bootstrap interval over people."""
    key = ["category", "seed"]
    sa = runs[(runs["K"] == a[0]) & (runs["dataset"] == a[1])].set_index(key)[metric]
    sb = runs[(runs["K"] == b[0]) & (runs["dataset"] == b[1])].set_index(key)[metric]
    d = (sb - sa).dropna().to_numpy()
    if len(d) == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    return {"n": len(d), "mean": d.mean(), "lo": np.quantile(boots, 0.025), "hi": np.quantile(boots, 0.975)}


def _fmt(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2f}"


def summary(runs: pd.DataFrame, empty: dict[str, int], labels: list[str]) -> str:
    beta = sorted({str(b) for b in runs["beta"].dropna().unique()}) or ["?"]
    lines = [
        "# FHIBE baseline (#4699)",
        "",
        f"F-beta at the session's beta ({', '.join(beta)}) of what Find returns on the withheld half, every run at "
        f"every vote. People per cell: {runs.groupby(['K', 'dataset']).size().min()}. Empty cells: "
        + ", ".join(f"{k} {v}" for k, v in empty.items())
        + ".",
        "",
        "| K | arm | session mean F-beta (1-150) | short session (1-50) | at 150 | after the check | AP at 150 "
        "| ceiling AP | left the Good phase | first detector shown (median vote) | Goods found by clicks |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in labels:
        for ds, name in ARMS.items():
            g = runs[(runs["K"] == k) & (runs["dataset"] == ds)]
            if g.empty:
                continue
            lines.append(
                f"| {k} | {name} | {_fmt(g['fbeta_mean'].mean())} | {_fmt(g['fbeta_mean_short'].mean())} | "
                f"{_fmt(g['fbeta_at_end'].mean())} | {_fmt(g['fbeta_after_check'].mean())} | {_fmt(g['ap_at_end'].mean())} | "
                f"{_fmt(g['ceiling_ap'].mean())} | {g['left_good'].notna().mean():.0%} | "
                f"{g['first_shown'].median():.0f} | {g['goods_found'].mean():.1f} |"
            )
    lines += [
        "",
        "**Paired contrasts** (session mean F-beta, votes 1-150; mean of the difference over people, 95% bootstrap)",
        "",
    ]
    lines += ["| contrast | people | difference | 95% interval |", "|---|---|---|---|"]
    contrasts = []
    for k in labels:
        contrasts += [
            (f"{k}: face - photo, 1024", (k, "fhibe_1024"), (k, "fhibe_faces_1024")),
            (f"{k}: face - photo, 640", (k, "fhibe_640"), (k, "fhibe_faces_640")),
            (f"{k}: 640 - 1024, photo", (k, "fhibe_1024"), (k, "fhibe_640")),
            (f"{k}: 640 - 1024, face", (k, "fhibe_faces_1024"), (k, "fhibe_faces_640")),
        ]
    if len(labels) == 2:
        a, b = labels
        contrasts += [(f"{b} - {a}, {name}", (a, ds), (b, ds)) for ds, name in ARMS.items()]
    for name, x, y in contrasts:
        p = paired(runs, x, y, "fbeta_mean")
        lines.append(f"| {name} | {p['n']} | {p['mean']:+.3f} | [{p['lo']:+.3f}, {p['hi']:+.3f}] |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="label=run_dir, e.g. k1=runs/2026-10-09-k1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    all_runs, all_curves, empty, labels = [], {}, {}, []
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs, curves, n_empty = load_run(label, Path(path), args.workers)
        all_runs.append(runs)
        all_curves.update(curves)
        empty[label] = n_empty
        labels.append(label)
    runs = pd.concat(all_runs, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / "runs.csv", index=False)
    rows = []
    for (label, ds), ms in all_curves.items():
        means = {m: np.mean(np.stack(v), axis=0) for m, v in ms.items()}
        for i in range(HORIZON):
            rows.append(
                {"K": label, "dataset": ds, "t": i + 1, **{m: means[m][i] for m in means}, "n": len(ms["fbeta"])}
            )
    pd.DataFrame(rows).to_csv(args.out / "curves.csv", index=False)
    (args.out / "summary.md").write_text(summary(runs, empty, labels))
    print((args.out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

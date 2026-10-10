#!/usr/bin/env python
"""Price the Goods' centroid's line (#4732) from a run's tagged variant rows.

Below the label quota a Find gives the Goods' centroid, cut at the two-Gaussian midpoint of its
corpus cosines. ``CALIB_CENTROID_LINE_VARIANTS`` writes, at every click that Find gives the centroid,
one extra row per ``(rule, beta)``: the same centroid on the same withheld half with its line drawn by
*rule* at *beta*, and F-beta weighted at *beta*. The centroid never picks and the opening reads no
balance, so the variants pair exactly: two lines differ only on the centroid's clicks, and only by
which media they keep.

**What a Find returns, at every vote.** Each run's curve over votes 1..H is the untagged rows' F-beta
(the centroid, then the trained head), the last row carried between rows and 0 before the first, as
``fhibe/analyze.py`` reads it. A rule's curve swaps in its own rows on the centroid's clicks. At a
beta other than the run's, the trained rows are not at that beta, so only the difference to the
midpoint is read there: it is exactly 0 off the centroid's clicks, whatever the trained head scores.

**Session means.** Over votes 1-10, 1-50 and 1-150. A run cut at H < 150 votes still gives the
1-150 difference exactly when its centroid tier ended inside H (checked: ``tier_past_h``), because
the two lines' curves agree from there on. Absolute means are read at the run's beta only, over
votes 1-10 and 1-H.

**The typed query (COCO only).** With ``--text-baseline``, the typed query's own set at the app's line
for the same cell and seed (``text_baseline.csv``'s ``text_line_*``), which is what the session shows
through the opening. It is a reference, not an arm.

Writes ``cells.csv`` (one row per run x rule x beta), ``curves.csv`` (mean over runs per vote: F-beta,
precision and recall at the run's beta, and ``d_fbeta``, the difference to the midpoint, at every beta)
and ``summary.md`` under ``--out``.

Usage::

    python analyze_centroid_line_4732.py --run fhibe-k1=<run dir> --run coco-b1=<run dir> \\
        [--text-baseline coco-b1=<text_baseline.csv>] --out <dir> [--horizon 40]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cells_io  # noqa: E402

TAG = "centroid_line:"
COLS = [
    "dataset", "category", "embedder", "seed", "t", "phase", "beta", "fbeta", "precision", "recall",
    "n_flagged", "n_test_pos", "detector_tier", "gmm_variant", "pool_variant", "schedule", "app_trained",
]  # fmt: skip
METRICS = ("fbeta", "precision", "recall", "n_flagged")
SPANS = {"1-10": 10, "1-50": 50, "1-150": 150}


def _carry(t: np.ndarray, v: np.ndarray, horizon: int) -> np.ndarray:
    """Values at votes 1..horizon, each the last row at or before it; 0 before the first row."""
    out = np.zeros(horizon)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    idx = np.searchsorted(t, np.arange(1, horizon + 1), side="right") - 1
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return out


def _fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return 0.0 if not (p > 0 or r > 0) else (1 + b2) * p * r / (b2 * p + r)


def _no_good_cell(path: str, horizon: int, variants: list[tuple[str, float]], run_beta: float) -> list[dict] | None:
    """A run that found no Good inside H: Find gives nothing at every vote, under every line.

    The main frame has no rows (no detector is ever scored), but the pick log names the run.
    Every rule returns the same nothing, so each difference is exactly 0 and each F-beta 0; the
    run still counts in every average (#4631).  ``None`` when there is no pick log either.
    """
    picks = Path(path.replace(".csv", "__picks.csv"))
    try:
        p = pd.read_csv(picks, nrows=1)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return None
    if p.empty:
        return None
    first = p.iloc[0]
    zero = np.zeros(horizon)
    out = []
    for rule, vb in variants:
        rec = {
            "dataset": first["dataset"],
            "category": first["category"],
            "embedder": first.get("embedder", ""),
            "seed": int(first["seed"]),
            "n_test_pos": np.nan,
            "run_beta": run_beta,
            "rule": rule,
            "beta": vb,
            "n_tier": 0,
            "tier_first": np.nan,
            "tier_last": np.nan,
            "tier_past_h": False,
            "no_good": True,
            **{f"tier_{m}": np.nan for m in METRICS},
            "tier_n_median": np.nan,
            "_curves": {m: zero for m in ("fbeta", "precision", "recall")} if vb == run_beta else {},
            "_dcurve": zero,
        }
        for span, n in SPANS.items():
            rec[f"d_fbeta_{span}"] = 0.0
            if vb == run_beta and n <= horizon:
                rec[f"fbeta_{span}"] = 0.0
        if vb == run_beta:
            rec[f"fbeta_1-{horizon}"] = 0.0
        out.append(rec)
    return out


def read_cell(args: tuple[str, int, list[tuple[str, float]], float]) -> list[dict] | None:
    """One cell file -> one row per (rule, beta) with its curves and session means, or ``None``."""
    path, horizon, variants, run_beta_default = args
    try:
        df = pd.read_csv(path, usecols=lambda c: c in COLS, low_memory=False)
    except pd.errors.EmptyDataError:
        return _no_good_cell(path, horizon, variants, run_beta_default)
    except (OSError, ValueError):
        return None
    if df.empty:
        return _no_good_cell(path, horizon, variants, run_beta_default)
    if "gmm_variant" not in df.columns:
        return None
    df = df[~_cells_io.check_rows(df)]
    tag = df["gmm_variant"].fillna("").astype(str)
    var = df[tag.str.startswith(TAG)].copy()
    base = _cells_io._base_rows(df)
    base = base[base["t"] <= horizon].groupby("t", as_index=False).last()
    if base.empty:
        return None
    first = base.iloc[0]
    run_beta = float(first["beta"]) if pd.notna(first["beta"]) else run_beta_default
    tier = base.loc[base["detector_tier"].astype(str) == "centroid", "t"].to_numpy(dtype=int)
    all_tier = df.loc[(tag == "") & (df["detector_tier"].astype(str) == "centroid"), "t"]
    t_base = base["t"].to_numpy(dtype=float)
    var["rule"] = var["gmm_variant"].str[len(TAG) :].str.split("@").str[0]
    var["vbeta"] = var["gmm_variant"].str.split("@").str[1].astype(float)
    var = var[var["t"] <= horizon]
    out = []
    mids: dict[float, dict[str, np.ndarray]] = {}
    rows_by: dict[tuple[str, float], pd.DataFrame] = {
        (r, b): g.groupby("t", as_index=False).last() for (r, b), g in var.groupby(["rule", "vbeta"])
    }
    for (rule, vb), g in sorted(rows_by.items(), key=lambda kv: (kv[0][0] != "midpoint", kv[0])):
        curves = {}
        for m in ("fbeta", "precision", "recall"):
            v = base.set_index("t")[m].astype(float).fillna(0.0)
            v.loc[g["t"].to_numpy()] = g.set_index("t")[m].astype(float).fillna(0.0).to_numpy()
            curves[m] = _carry(t_base, v.reindex(base["t"]).to_numpy(), horizon)
        if rule == "midpoint":
            mids[vb] = curves
        rec = {
            "dataset": first["dataset"],
            "category": first["category"],
            "embedder": first["embedder"],
            "seed": int(first["seed"]),
            "n_test_pos": float(first["n_test_pos"]),
            "run_beta": run_beta,
            "rule": rule,
            "beta": vb,
            "n_tier": int(len(tier)),
            "tier_first": int(tier.min()) if len(tier) else np.nan,
            "tier_last": int(tier.max()) if len(tier) else np.nan,
            "tier_past_h": bool(len(all_tier) and all_tier.max() >= horizon),
            "no_good": False,
            # What the centroid returns on its own clicks, averaged over them.
            **{f"tier_{m}": float(g[m].astype(float).mean()) if len(g) else np.nan for m in METRICS},
            "tier_n_median": float(g["n_flagged"].astype(float).median()) if len(g) else np.nan,
            "_curves": curves,
        }
        out.append(rec)
    for rec in out:
        mid = mids.get(rec["beta"])
        c = rec["_curves"]
        for span, n in SPANS.items():
            if mid is not None:
                # Exact past H when the tier ended inside it: the two curves agree from there on.
                rec[f"d_fbeta_{span}"] = float((c["fbeta"] - mid["fbeta"])[: min(n, horizon)].sum() / n)
            if rec["beta"] == run_beta and n <= horizon:
                rec[f"fbeta_{span}"] = float(c["fbeta"][:n].mean())
        if rec["beta"] == run_beta:
            rec[f"fbeta_1-{horizon}"] = float(c["fbeta"].mean())
        if mid is not None:
            rec["_dcurve"] = c["fbeta"] - mid["fbeta"]
        if rec["beta"] != run_beta:
            # Off the run's beta the trained rows are at the run's beta: only the difference is read.
            rec["_curves"] = {}
    return out


def load_run(
    label: str, run: Path, horizon: int, workers: int, variants: list[tuple[str, float]], run_beta: float
) -> tuple[pd.DataFrame, dict]:
    files = [str(f) for f in _cells_io.main_frame_files(run / "results" / "cells")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(read_cell, [(f, horizon, variants, run_beta) for f in files], chunksize=8))
    rows, curves = [], {}
    for cell in got:
        for r in cell or []:
            c = r.pop("_curves")
            d = r.pop("_dcurve", None)
            r["run"] = label
            rows.append(r)
            key = (label, r["dataset"], r["rule"], r["beta"])
            for m, v in c.items():
                curves.setdefault(key, {}).setdefault(m, []).append(v)
            if d is not None:
                curves.setdefault(key, {}).setdefault("d_fbeta", []).append(d)
    n_empty = sum(1 for c in got if not c)
    n_no_good = sum(1 for c in got if c and c[0].get("no_good"))
    print(f"{label}: {len(files)} cells, {n_no_good} with no Good inside H, {n_empty} unreadable", file=sys.stderr)
    return pd.DataFrame(rows), curves


def add_text_reference(cells: pd.DataFrame, label: str, baseline: Path) -> pd.DataFrame:
    """The typed query's set at the app's line, per cell and seed, at each row's beta."""
    tb = pd.read_csv(baseline)
    tags = {0.25: "b025", 1.0: "b1", 4.0: "b4"}
    keep = cells["run"] == label
    ref = tb.set_index(["dataset", "embedder", "category", "seed"])
    vals = []
    for _, r in cells[keep].iterrows():
        tag = tags.get(float(r["beta"]))
        key = (r["dataset"], r["embedder"], r["category"], int(r["seed"]))
        if tag is None or key not in ref.index:
            vals.append(np.nan)
            continue
        row = ref.loc[key]
        vals.append(_fbeta(float(row[f"text_line_precision_{tag}"]), float(row[f"text_line_recall_{tag}"]), r["beta"]))
    cells.loc[keep, "text_fbeta"] = vals
    return cells


def _mean_se(x: pd.Series, clusters: pd.Series) -> tuple[float, float, int]:
    """Mean and its standard error with *clusters* (category) as the unit; the cluster count."""
    x = x.astype(float)
    ok = x.notna()
    x, clusters = x[ok], clusters[ok]
    if x.empty:
        return np.nan, np.nan, 0
    per = x.groupby(clusters).mean()
    n = len(per)
    return float(x.mean()), float(per.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan, n


def summarize(cells: pd.DataFrame, horizon: int) -> str:
    lines = [f"# #4732: the Goods' centroid's line, priced on tagged rows (H = {horizon} votes)", ""]
    lines.append(
        "Differences are paired per run against the midpoint at the same beta; SE clusters on the category "
        "(the person on FHIBE). Session means are over votes; on FHIBE the K examples are the first K votes."
    )
    lines.append("")
    one = cells.drop_duplicates(["run", "dataset", "category", "seed"])
    for run, g in one.groupby("run"):
        lines.append(
            f"- {run}: {len(g)} runs; {int(g['no_good'].sum())} found no Good by vote {horizon} (every line returns "
            f"nothing there: difference 0, counted); centroid tier still on at vote {horizon} in "
            f"{g['tier_past_h'].mean():.1%}"
        )
    lines.append("")
    for (run, ds), g in cells.groupby(["run", "dataset"], sort=False):
        lines.append(f"## {run} / {ds}")
        lines.append("")
        lines.append(
            "| rule | beta | runs | tier clicks (mean) | kept on tier (median) | tier precision | tier recall "
            "| tier F-beta | dF 1-10 | dF 1-50 | dF 1-150 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for (rule, beta), h in g.groupby(["rule", "beta"]):
            cl = h["category"]
            d = {s: _mean_se(h[f"d_fbeta_{s}"], cl) for s in SPANS}
            lines.append(
                f"| {rule} | {beta:g} | {len(h)} | {h['n_tier'].mean():.1f} | {h['tier_n_median'].median():.0f} "
                f"| {h['tier_precision'].mean():.3f} | {h['tier_recall'].mean():.3f} | {h['tier_fbeta'].mean():.3f} | "
                + " | ".join(f"{m:+.4f} ± {se:.4f}" for m, se, _ in d.values())
                + " |"
            )
        if "text_fbeta" in g.columns and g["text_fbeta"].notna().any():
            lines.append("")
            lines.append("Typed query's own set at the app's line (what the session shows), F-beta by beta:")
            for beta, h in g[g["rule"] == "midpoint"].groupby("beta"):
                lines.append(f"- beta {beta:g}: {h['text_fbeta'].mean():.3f}")
        rb = g[g["beta"] == g["run_beta"]]
        if len(rb):
            lines.append("")
            lines.append(f"Absolute session means at the run's beta ({rb['run_beta'].iloc[0]:g}):")
            for rule, h in rb.groupby("rule"):
                lines.append(
                    f"- {rule}: votes 1-10 {h['fbeta_1-10'].mean():.3f}, votes 1-{horizon} "
                    f"{h[f'fbeta_1-{horizon}'].mean():.3f}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="label=<run dir>")
    ap.add_argument("--text-baseline", action="append", default=[], help="label=<text_baseline.csv>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--variants",
        default="midpoint@0.25,midpoint@1,midpoint@4,guarded@0.25,guarded@1,guarded@4,count@0.25,count@1,count@4",
        help="the runs' CALIB_CENTROID_LINE_VARIANTS, for the runs that found no Good",
    )
    ap.add_argument("--run-beta", type=float, default=1.0, help="the runs' balance, for the runs that found no Good")
    args = ap.parse_args()
    variants = [(v.split("@")[0], float(v.split("@")[1])) for v in args.variants.split(",") if v]
    args.out.mkdir(parents=True, exist_ok=True)
    frames, curves = [], {}
    for spec in args.run:
        label, _, run = spec.partition("=")
        df, cv = load_run(label, Path(run), args.horizon, args.workers, variants, args.run_beta)
        frames.append(df)
        curves.update(cv)
    cells = pd.concat(frames, ignore_index=True)
    for spec in args.text_baseline:
        label, _, path = spec.partition("=")
        cells = add_text_reference(cells, label, Path(path))
    cells.to_csv(args.out / "cells.csv", index=False)
    rows = []
    for (label, ds, rule, beta), c in curves.items():
        for m, vs in c.items():
            mean = np.mean(np.stack(vs), axis=0)
            rows += [
                {"run": label, "dataset": ds, "rule": rule, "beta": beta, "metric": m, "t": t + 1, "value": float(v)}
                for t, v in enumerate(mean)
            ]
    pd.DataFrame(rows).to_csv(args.out / "curves.csv", index=False)
    (args.out / "summary.md").write_text(summarize(cells, args.horizon) + "\n")
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()

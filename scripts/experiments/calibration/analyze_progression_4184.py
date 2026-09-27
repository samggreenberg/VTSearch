#!/usr/bin/env python3
"""#4184: the deck's calibration ladder, one cost curve per rung, on COCO Better.

Reads every rung's arm directory (``launch_progression_4184.sh``), checks each
one ran the rung it claims to, and writes:

* ``progression_curve.csv`` - per rung, per click: the mean cost over every
  cell, as a user would experience it (see *filled* below), with its SE and the
  share of cells that had a detector on screen.  **This is the slide's input**:
  ``slides/figs/src/make-progression-fig.py`` reads a committed copy of it.
* ``paired.csv`` - each rung against the one before it, and against rung 1, at
  fixed clicks and over the whole trajectory: paired mean difference and SE.
* ``REPORT_progression.md`` - the machine summary: what loaded, what was
  dropped, whether each rung's premise held, and the two tables above.
* ``figures/`` and ``viewer.html`` - the mandatory quality-over-clicks pair
  (``curves.py``) and interactive viewer (``viewer.py``), unmodified.

**Filled, not survivor-averaged.**  A cell has no row until it holds a Good
and a Bad vote, and ``app_trained`` is 0 on a row the app would not yet have
shown.  In both cases the simulated user is still reading the typed query's
ranking, so the cell's cost at that click IS its text-sort cost - which is
also exactly the t=0 notch.  Averaging only the cells that happen to have a
detector would put an easier subset of the grid on the left of every curve.
A cell that produced no row in one rung (a header-only file: it never found a
positive) is filled the same way at every click, but only when that arm lost
no file; a lost file is data loss, not a user outcome, and the arm's filled
mean then drops those cells and says so.

    python analyze_progression_4184.py --arms DIR=r1_xcal,DIR=r2_gmm,... \\
        --baseline text_baseline.csv --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402
import curves  # noqa: E402

CELL = ["dataset", "embedder", "category", "seed"]

#: What each rung must have run with, read back off its own rows.  ``acq`` is
#: whether acquisition may aim anywhere but the reporting cut: only the rung
#: that turns the offset on may ever show ``acq_threshold != threshold``.
RUNGS: dict[str, dict] = {
    "r1_xcal": {"live_threshold": "xcal_mincost", "calibration_fraction": 0.5, "acq": False},
    "r2_gmm": {"live_threshold": "gmm_mid", "calibration_fraction": 0.5, "acq": False},
    "r3_blend": {"live_threshold": "blend", "calibration_fraction": 0.5, "acq": False},
    "r4_rawmean": {"live_threshold": "anchored_rawmean", "calibration_fraction": 0.5, "acq": False},
    "r5_anchored": {"live_threshold": "shipped", "calibration_fraction": 0.5, "acq": False},
    "r6_split70": {"live_threshold": "shipped", "calibration_fraction": 0.3, "acq": False},
    "r7_acq4": {"live_threshold": "shipped", "calibration_fraction": 0.3, "acq": True},
}

#: The clicks the paired table reads.  150 is the grid's horizon.
CHECKPOINTS = (10, 20, 30, 50, 100, 150)


def parse_arms(spec: str) -> list[tuple[Path, str]]:
    """``dir=label,dir=label`` -> ``[(dir, label)]``, order kept (it is the rung order)."""
    out = []
    for part in [p for p in spec.replace(",", " ").split() if p]:
        d, _, label = part.partition("=")
        out.append((Path(d), label or Path(d).parent.name))
    return out


def premise_failures(label: str, frame: pd.DataFrame) -> list[str]:
    """Every way *frame* is not the rung *label* names.  Empty = the premise held."""
    want = RUNGS.get(label)
    if want is None:
        return [f"{label}: not a known rung; premise unchecked"]
    bad = []
    for col in ("live_threshold", "calibration_fraction"):
        if col not in frame.columns:
            bad.append(f"{label}: no `{col}` column - these cells predate #4184's harness")
            continue
        got = sorted({str(v) for v in frame[col].dropna().unique()})
        expect = str(want[col])
        if col == "calibration_fraction":
            got = sorted({f"{float(v):g}" for v in frame[col].dropna().unique()})
            expect = f"{float(want[col]):g}"
        if got != [expect]:
            bad.append(f"{label}: {col} = {got}, expected [{expect!r}]")
    trained = frame[np.isfinite(frame["threshold"]) & np.isfinite(frame["acq_threshold"])]
    moved = (trained["acq_threshold"] - trained["threshold"]).abs() > 1e-9
    if want["acq"] and not moved.any():
        bad.append(f"{label}: acquisition never left the reporting cut - the offset did not reach the run")
    if not want["acq"] and moved.any():
        bad.append(f"{label}: acquisition left the reporting cut on {int(moved.sum())} rows - the offset leaked in")
    heads = sorted({str(h) for h in frame.get("head", pd.Series(dtype=str)).dropna().unique()})
    if len(heads) > 1:
        bad.append(f"{label}: more than one head {heads}")
    return bad


def filled_matrix(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    text_cost: dict[tuple, float],
    horizon: int,
    metric: str = "cost",
    baseline_metric: dict[tuple, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``cell x t`` (t = 0..horizon) of *metric* as the user experiences it, and where a detector was shown.

    t=0 and every click with no app-visible detector take the cell's text-sort
    value; after a cell's last row its last value carries (it cannot happen
    inside the pool at this horizon, but a silent NaN would be worse).  A cell
    with no text-sort value cannot be anchored and is left NaN at those clicks.

    The second frame is the same shape, True where the app had a detector on
    screen - its column mean is the rung's coverage at that click.
    """
    base = baseline_metric if baseline_metric is not None else text_cost
    idx = pd.MultiIndex.from_frame(cells[CELL])
    out = pd.DataFrame(np.nan, index=idx, columns=range(horizon + 1), dtype=float)
    shown = frame[frame["app_trained"].fillna(0).astype(int) == 1] if "app_trained" in frame.columns else frame
    shown = shown[shown["t"] <= horizon]
    wide = shown.pivot_table(index=CELL, columns="t", values=metric, aggfunc="mean")
    wide = wide.reindex(index=idx, columns=range(horizon + 1))
    first = wide.notna().to_numpy().argmax(axis=1)
    has_any = wide.notna().to_numpy().any(axis=1)
    values = wide.to_numpy(copy=True)
    for i, key in enumerate(idx):
        anchor = base.get(tuple(key), np.nan)
        stop = first[i] if has_any[i] else horizon + 1
        values[i, :stop] = anchor
        if has_any[i]:
            row = pd.Series(values[i, stop:]).ffill().to_numpy()
            values[i, stop:] = row
    out.loc[:, :] = values
    shown_mask = pd.DataFrame(wide.notna().to_numpy(), index=idx, columns=out.columns)
    return out, shown_mask


def summarize(m: pd.DataFrame) -> pd.DataFrame:
    """Per-click mean, SE and count of a ``cell x t`` matrix."""
    n = m.notna().sum(axis=0)
    mean = m.mean(axis=0)
    se = m.std(axis=0, ddof=1) / np.sqrt(n.clip(lower=1))
    return pd.DataFrame({"t": m.columns.astype(int), "mean": mean.to_numpy(), "se": se.to_numpy(), "n": n.to_numpy()})


def paired(a: pd.DataFrame, b: pd.DataFrame, horizon: int) -> list[dict]:
    """``b - a`` on the cells both measured, at each checkpoint and over clicks 1..horizon."""
    common_idx = a.index.intersection(b.index)
    a, b = a.loc[common_idx], b.loc[common_idx]
    rows = []
    points = [(f"t={t}", [t]) for t in CHECKPOINTS if t <= horizon]
    points.append((f"mean t=1..{horizon}", list(range(1, horizon + 1))))
    for name, ts in points:
        d = (b[ts].mean(axis=1) - a[ts].mean(axis=1)).dropna().to_numpy(float)
        k = len(d)
        mean = float(d.mean()) if k else float("nan")
        se = float(d.std(ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
        rows.append({"at": name, "n": k, "mean": mean, "se": se, "resolvable": bool(k > 1 and abs(mean) > 2 * se)})
    return rows


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", required=True, help="dir=label,... in rung order; dir holds cells/")
    ap.add_argument("--baseline", required=True, help="text_baseline.py CSV: the click-0 notch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = parse_arms(args.arms)
    labels = [label for _d, label in arms]
    baseline = curves.text_sort_baseline(args.baseline)
    text_cost = {tuple(k): float(v) for k, v in baseline.groupby(CELL)["text_cost"].mean().items()}
    text_ap = {tuple(k): float(v) for k, v in baseline.groupby(CELL)["text_AP"].mean().items()}

    frames: dict[str, pd.DataFrame] = {}
    provs: dict[str, dict] = {}
    lines = ["# #4184 progression - machine summary", ""]
    failures: list[str] = []
    for d, label in arms:
        frame, prov = _cells_io.load_arm(d)
        provs[label] = {**prov, "dir": str(d)}
        lines.append(f"- `{label}` ({d}): {_cells_io.describe_load(prov)}")
        if frame.empty:
            failures.append(f"{label}: no rows")
            continue
        frame = frame.copy()
        frame["arm"] = label
        frames[label] = frame
        failures += premise_failures(label, frame)
    lines.append("")
    if failures:
        lines += ["## Premise failures", "", *[f"- {f}" for f in failures], ""]
    (out / "provenance.json").write_text(json.dumps(provs, indent=2, default=str))

    # The grid: every cell any rung measured, plus every cell the baseline
    # anchors.  A cell missing from one rung is either starved there (filled)
    # or lost there (dropped from that rung, and counted).
    seen = pd.concat([f[CELL] for f in frames.values()], ignore_index=True)
    grid = pd.concat([seen, baseline[CELL]], ignore_index=True).drop_duplicates().reset_index(drop=True)
    lines.append(f"Grid: {len(grid)} cells (union of every rung's cells and the baseline's).")
    lines.append("")

    mats: dict[str, pd.DataFrame] = {}
    ap_mats: dict[str, pd.DataFrame] = {}
    curve_rows = []
    lines += ["| rung | cells | lost | filled-only | coverage@10 | coverage@50 |", "|---|---|---|---|---|---|"]
    for order, label in enumerate(labels, start=1):
        if label not in frames:
            continue
        frame = frames[label]
        lost = len(provs[label].get("unreadable") or []) + len(provs[label].get("zero_byte") or [])
        present = frame[CELL].drop_duplicates()
        cells = grid if lost == 0 else present
        m, shown_mask = filled_matrix(frame, cells, text_cost, args.horizon)
        mats[label] = m
        ap_mats[label], _ = filled_matrix(
            frame, cells, text_cost, args.horizon, metric="average_precision", baseline_metric=text_ap
        )
        cov = shown_mask.mean(axis=0)
        s = summarize(m)
        s_ap = summarize(ap_mats[label])
        s["rung"] = label
        s["rung_order"] = order
        s["coverage"] = cov.to_numpy()
        s["ap_mean"] = s_ap["mean"].to_numpy()
        s["ap_se"] = s_ap["se"].to_numpy()
        curve_rows.append(s)
        filled_only = len(cells) - len(present) if lost == 0 else 0
        lines.append(
            f"| {label} | {len(cells)} | {lost} | {filled_only} | {cov.get(10, np.nan):.2f} | {cov.get(50, np.nan):.2f} |"
        )
    lines.append("")
    curve = pd.concat(curve_rows, ignore_index=True)[
        ["rung_order", "rung", "t", "mean", "se", "n", "coverage", "ap_mean", "ap_se"]
    ]
    curve.to_csv(out / "progression_curve.csv", index=False, float_format="%.6g")

    # Levels at the checkpoints, then the paired steps.
    pts = [0, *[t for t in CHECKPOINTS if t <= args.horizon]]
    lines += ["## Mean cost (filled) at checkpoints", ""]
    lines += ["| rung | " + " | ".join(f"t={t}" for t in pts) + " |", "|---|" + "---|" * len(pts)]
    for label in mats:
        c = curve[curve["rung"] == label].set_index("t")
        lines.append(f"| {label} | " + " | ".join(f"{c.loc[t, 'mean']:.2g}" for t in pts) + " |")
    lines.append("")

    pair_rows = []
    present_labels = [label for label in labels if label in mats]
    for prev, cur in zip(present_labels, present_labels[1:], strict=False):
        for r in paired(mats[prev], mats[cur], args.horizon):
            pair_rows.append({"from": prev, "to": cur, "kind": "step", **r})
    if present_labels:
        first = present_labels[0]
        for cur in present_labels[1:]:
            for r in paired(mats[first], mats[cur], args.horizon):
                pair_rows.append({"from": first, "to": cur, "kind": "vs_first", **r})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "paired.csv", index=False, float_format="%.6g")
    lines += ["## Each rung against the one before (Δcost = to − from; negative = the step helped)", ""]
    lines += ["| from → to | at | n | Δ mean | SE |", "|---|---|---|---|---|"]
    for r in pairs[pairs["kind"] == "step"].itertuples(index=False) if not pairs.empty else []:
        star = "**" if r.resolvable else ""
        lines.append(f"| {r[0]} → {r[1]} | {r.at} | {r.n} | {star}{r.mean:+.2g}{star} | {r.se:.2g} |")
    lines += ["", "Bold = more than 2 SE from zero.", ""]
    (out / "REPORT_progression.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.no_figures or not args.no_viewer:
        main_frame = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
        denominator = pd.concat([grid.assign(arm=label) for label in frames], ignore_index=True)
        if not args.no_figures and not main_frame.empty:
            for metric in ("cost", "average_precision"):
                curves.quality_vs_clicks(
                    main_frame,
                    out / "figures",
                    arms=list(frames),
                    metric=metric,
                    denominator=denominator,
                    baseline=baseline,
                    lower_is_better=(metric == "cost"),
                )
        if not args.no_viewer and not main_frame.empty:
            import viewer  # noqa: PLC0415

            viewer.build_viewer(
                main_frame,
                out / "viewer.html",
                arms=list(frames),
                denominator=denominator,
                baseline=baseline,
                title="COCO Better: the calibration ladder",
                subtitle="Binary voting, SigLIP, 144 cells x 5 seeds; one arm per rung of the deck (#4184)",
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

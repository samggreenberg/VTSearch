#!/usr/bin/env python3
"""#4671: two Autopilot pieces priced on the objective against today's app, at every preset.

Reads the nine arms ``launch_autopilot_4671.sh`` wrote under ``--base`` (``ctl_*``, ``open_*``, ``nowalk_*`` at
``b025`` / ``b1`` / ``b4``) and writes, to ``--out``:

* ``curves.csv`` - per preset, arm and vote: the objective (F-beta at the preset's own beta of the withheld half
  above the line the app shows), AP, precision, recall, the returned set's median size, the shares returning more
  than 200 and returning nothing, and the share of runs showing a detector.
* ``paired.csv`` - each test arm minus the control at its preset, paired by (category, seed): the session means
  over votes 1-150, 1-50, 51-150 and 1-25, and the points at votes 10, 25, 50, 100 and 150, for the objective and AP.
* ``paired_curve.csv`` - the paired difference in the objective at every vote, with its SE.
* ``REPORT_autopilot.md`` - the machine summary; ``provenance.json``; ``figures/``.

**Every run counts at every vote** (#4631; owner 2026-10-09). Until a run shows a detector, Find returns the
typed query's own set at today's per-preset line (#4603), and that is what the run scores. After the hand-over it
scores the last line shown. The single number per arm is the session mean, the area under the curve divided by
the votes (owner, 2026-10-09).

    python analyze_autopilot_4671.py --base /expscratch/$USER/autopilot-4671 --out OUTDIR
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
from _rank_metrics import beta_tag  # noqa: E402
from analyze_buildup_4668 import logged_knobs, paired_rows, text_fbeta, text_set  # noqa: E402
from analyze_progression_4184 import CELL, baseline_mismatches, fbeta, filled_matrix  # noqa: E402

PRESETS: dict[str, float] = {"b025": 0.25, "b1": 1.0, "b4": 4.0}
#: (arm, label, what its run logs must say).
ARMS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ("ctl", "today's app", {"startup_schedule": "app", "new_walk": "atlas"}),
    ("open", "the opening before #4288 (g3@top,b4@mid)", {"startup_schedule": "g3@top,b4@mid", "new_walk": "atlas"}),
    ("nowalk", "New on the Hard pick, no atlas walk", {"startup_schedule": "app", "new_walk": "hard"}),
)
WINDOWS = ((1, 150), (1, 50), (51, 150), (1, 25))


def premise_failures(arm: str, frame: pd.DataFrame, knobs: dict[str, set[str]], want: dict[str, str], beta: float):
    out = []
    if not knobs:
        return [f"{arm}: no run_cells header line found in logs/cells-*.out"]
    for k, v in want.items():
        got = knobs.get(k)
        # The header prints an unset schedule as "app default"; the regex reads its first word.
        if k == "startup_schedule" and v == "app":
            ok = got == {"app"}
        else:
            ok = got == {v}
        if not ok:
            out.append(f"{arm}: run logs say {k}={sorted(got) if got else 'nothing'}, expected {v}")
    b = pd.to_numeric(frame.get("beta"), errors="coerce").dropna().unique()
    if not np.allclose(b, beta):
        out.append(f"{arm}: rows carry beta {sorted(b)}, expected {beta}")
    pools = sorted({str(v) for v in frame.get("prevalence_arm", pd.Series(dtype=str)).dropna().unique()})
    if pools != ["haystack_0.01"]:
        out.append(f"{arm}: pool {pools}, expected haystack_0.01")
    return out


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--baseline", help="text_baseline.py CSV at today's lines (default BASE/text_baseline.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seeds", help="comma-separated seeds to read (an interim read); default all")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    bl_path = Path(args.baseline) if args.baseline else base / "text_baseline.csv"
    bl = curves.text_sort_baseline(bl_path)
    raw = pd.read_csv(bl_path)
    H = args.horizon

    lines = ["# #4671 Autopilot pieces - machine summary", ""]
    failures: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    provs: dict[str, dict] = {}
    for p, beta in PRESETS.items():
        for arm, _label, want in ARMS:
            name = f"{arm}_{p}"
            if not (base / name / "results" / "cells").is_dir():
                failures.append(f"{name}: no cells dir")
                continue
            frame, prov = _cells_io.load_arm(base / name / "results")
            provs[name] = {**prov, "dir": f"<base>/{name}/results"}
            lines.append(f"- `{name}`: {_cells_io.describe_load(prov)}")
            if frame.empty:
                failures.append(f"{name}: no rows")
                continue
            if seeds is not None:
                frame = frame[frame["seed"].isin(seeds)]
            failures += premise_failures(name, frame, logged_knobs(base / name), want, beta)
            frames[name] = frame
    failures += [f"baseline: {m}" for m in baseline_mismatches(raw, frames)]
    lines.append("")
    if failures:
        lines += ["## Premise failures", "", *[f"- {f}" for f in failures], ""]
    (out / "provenance.json").write_text(json.dumps(provs, indent=2, default=str) + "\n")

    anchored = bl[CELL] if seeds is None else bl.loc[bl["seed"].isin(seeds), CELL]
    grid = pd.concat([*(f[CELL] for f in frames.values()), anchored], ignore_index=True).drop_duplicates()
    grid = grid.reset_index(drop=True)
    lines += [f"Grid: {len(grid)} cells.", ""]

    curve_rows, pair_rows, pcurve_rows = [], [], []
    for p, beta in PRESETS.items():
        tag = beta_tag(beta)
        obj_anchor = text_fbeta(bl, f"text_line_precision_{tag}", f"text_line_recall_{tag}", beta)
        ap_anchor = {tuple(k): float(v) for k, v in bl.groupby(CELL)["text_AP"].mean().items()}
        sets = text_set(raw, f"text_line_precision_{tag}", f"text_line_recall_{tag}", f"text_line_fpr_{tag}")
        mats: dict[str, dict[str, pd.DataFrame]] = {}
        for arm, label, _want in ARMS:
            name = f"{arm}_{p}"
            if name not in frames:
                continue
            frame = frames[name].copy()
            frame["objective"] = fbeta(frame["precision"], frame["recall"], beta)
            frame["precision"] = np.nan_to_num(frame["precision"].to_numpy(dtype=float))
            starved = len(provs[name].get("no_positive_found") or [])
            present = frame[CELL].drop_duplicates()
            lost = len(provs[name].get("unreadable") or []) + len(provs[name].get("zero_byte") or [])
            if seeds is None:
                lost += max(len(grid) - len(present) - starved, 0)
            cells = grid if lost == 0 else present
            m, shown = filled_matrix(frame, cells, {}, H, metric="objective", baseline_metric=obj_anchor)
            # Once a run hands over, a detector stays on screen (a check's steps carry no headline row).
            shown = shown.astype(int).cummax(axis=1).astype(bool)
            got = {"objective": m}
            got["ap"], _ = filled_matrix(frame, cells, {}, H, metric="average_precision", baseline_metric=ap_anchor)
            for col in ("precision", "recall", "n_flagged"):
                got[col], _ = filled_matrix(frame, cells, {}, H, metric=col, baseline_metric=sets[col])
            mats[arm] = got
            k = got["n_flagged"]
            curve_rows.append(
                pd.DataFrame(
                    {
                        "t": m.columns.astype(int),
                        "objective": m.mean(axis=0).to_numpy(),
                        "objective_se": (m.std(axis=0, ddof=1) / np.sqrt(m.notna().sum(axis=0))).to_numpy(),
                        "ap": got["ap"].mean(axis=0).to_numpy(),
                        "precision": got["precision"].mean(axis=0).to_numpy(),
                        "recall": got["recall"].mean(axis=0).to_numpy(),
                        "returned_median": k.median(axis=0).to_numpy(),
                        "over_200": (k > 200).mean(axis=0).to_numpy(),
                        "returned_none": (k < 0.5).mean(axis=0).to_numpy(),
                        "showing_detector": shown.mean(axis=0).to_numpy(),
                        "runs": m.notna().sum(axis=0).to_numpy(),
                    }
                ).assign(preset=p, beta=beta, arm=arm, label=label)
            )
            if lost:
                lines.append(f"- {name}: {lost} cells lost (unreadable, zero-byte or missing); read on what is present")
        if "ctl" not in mats:
            continue
        for arm, label, _want in ARMS[1:]:
            if arm not in mats:
                continue
            for metric in ("objective", "ap"):
                for r in paired_rows(mats["ctl"][metric], mats[arm][metric]):
                    pair_rows.append({"preset": p, "arm": arm, "label": label, "metric": metric, **r})
            a, b = mats["ctl"]["objective"], mats[arm]["objective"]
            idx = a.index.intersection(b.index)
            d = b.loc[idx] - a.loc[idx]
            pcurve_rows.append(
                pd.DataFrame(
                    {
                        "t": d.columns.astype(int),
                        "delta": d.mean(axis=0).to_numpy(),
                        "se": (d.std(axis=0, ddof=1) / np.sqrt(d.notna().sum(axis=0))).to_numpy(),
                        "n": d.notna().sum(axis=0).to_numpy(),
                    }
                ).assign(preset=p, arm=arm)
            )

    curve = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    pairs = pd.DataFrame(pair_rows)
    pcurve = pd.concat(pcurve_rows, ignore_index=True) if pcurve_rows else pd.DataFrame()
    curve.to_csv(out / "curves.csv", index=False, float_format="%.6g")
    pairs.to_csv(out / "paired.csv", index=False, float_format="%.6g")
    pcurve.to_csv(out / "paired_curve.csv", index=False, float_format="%.6g")

    lines += ["## Session means (every run, every vote)", ""]
    lines += [
        "| preset | arm | objective 1-150 | 1-50 | 51-150 | AP 1-150 | precision 1-150 | recall 1-150 | over 200 "
        "| showing a detector |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (p, arm), c in curve.groupby(["preset", "arm"], sort=False):
        w = c[(c["t"] >= 1) & (c["t"] <= H)]
        e = c[(c["t"] >= 1) & (c["t"] <= 50)]
        late = c[(c["t"] >= 51) & (c["t"] <= H)]
        lines.append(
            f"| {p} | {arm} | {w['objective'].mean():.3f} | {e['objective'].mean():.3f} | {late['objective'].mean():.3f} "
            f"| {w['ap'].mean():.3f} | {w['precision'].mean():.2f} | {w['recall'].mean():.2f} "
            f"| {w['over_200'].mean():.0%} | {w['showing_detector'].mean():.0%} |"
        )
    lines += ["", "## Each arm minus today's app (paired; bold = beyond 2 SE)", ""]
    cols = ["mean 1-150", "mean 1-50", "mean 51-150", "mean 1-25", "t=150"]
    lines += ["| preset | arm | metric | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for (p, arm, metric), g in pairs.groupby(["preset", "arm", "metric"], sort=False):
        g = g.set_index("at")
        cells_ = []
        for at in cols:
            r = g.loc[at]
            star = "**" if r["resolvable"] else ""
            cells_.append(f"{star}{r['mean']:+.3f}{star} ± {r['se']:.3f}")
        lines.append(f"| {p} | {arm} | {metric} | " + " | ".join(cells_) + " |")
    (out / "REPORT_autopilot.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    print("\n".join(lines))

    if not args.no_figures and not curve.empty:
        figures(curve, pcurve, out / "figures")
    return 1 if failures else 0


def figures(curve: pd.DataFrame, pcurve: pd.DataFrame, figdir: Path) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figdir.mkdir(parents=True, exist_ok=True)
    colours = {"ctl": "#2a78d6", "open": "#eb6834", "nowalk": "#1baf7a"}
    labels = {a: lab for a, lab, _w in ARMS}
    presets = list(PRESETS)
    fig, axes = plt.subplots(2, len(presets), figsize=(5.4 * len(presets), 7.2), squeeze=False)
    for j, p in enumerate(presets):
        ax = axes[0][j]
        for arm, *_ in ARMS:
            c = curve[(curve["preset"] == p) & (curve["arm"] == arm)]
            if not c.empty:
                ax.plot(c["t"], c["objective"], color=colours[arm], lw=1.6, label=labels[arm])
        ax.set_title(f"objective at beta {PRESETS[p]:g}")
        ax.grid(alpha=0.3)
        ax = axes[1][j]
        ax.axhline(0, color="#888", lw=1)
        for arm, *_ in ARMS[1:]:
            c = pcurve[(pcurve["preset"] == p) & (pcurve["arm"] == arm)]
            if c.empty:
                continue
            ax.plot(c["t"], c["delta"], color=colours[arm], lw=1.6, label=f"{labels[arm]} - today")
            ax.fill_between(c["t"], c["delta"] - 2 * c["se"], c["delta"] + 2 * c["se"], color=colours[arm], alpha=0.15)
        ax.set_title(f"paired difference from today's app, beta {PRESETS[p]:g} (±2 SE)")
        ax.set_xlabel("votes")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    axes[1][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "autopilot_curves.png", dpi=110)
    plt.close(fig)

    rows = (("precision", False), ("recall", False), ("returned_median", True), ("ap", False))
    fig, axes = plt.subplots(len(rows), len(presets), figsize=(5.2 * len(presets), 2.9 * len(rows)), squeeze=False)
    for j, p in enumerate(presets):
        for i, (col, log) in enumerate(rows):
            ax = axes[i][j]
            for arm, *_ in ARMS:
                c = curve[(curve["preset"] == p) & (curve["arm"] == arm)]
                if not c.empty:
                    ax.plot(c["t"], c[col], color=colours[arm], lw=1.4, label=labels[arm])
            if log:
                ax.set_yscale("log")
            if i == 0:
                ax.set_title(f"beta {PRESETS[p]:g}")
            if j == 0:
                ax.set_ylabel(col.replace("_", " "))
            if i == len(rows) - 1:
                ax.set_xlabel("votes")
            ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "autopilot_returned.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

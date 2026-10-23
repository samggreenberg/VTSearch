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

``--arms`` reads a subset (``ctl`` always among them): #4739's legs run ``ctl`` and ``nowalk`` only.
``--categories`` reads only the cells a file names, one ``class@band`` a line, in the baseline as in the arms: #4739's
Objects365 leg reads the 107 cells both corpora hold (#4815).

On arms run with a second corpus to Find on (#4960, ``CALIB_FIND_CORPUS``; #4739's legs), the ``find_*`` columns
are read the same way, every run at every vote, paired by (category, seed), into ``find_curves.csv`` (the objective
and AP on that corpus, and how many it returns), the ``find_objective`` / ``find_ap`` / ``find_n_flagged`` rows of
``paired.csv``, ``find_paired_curve.csv`` (the paired difference in the Find objective at every vote), and
``flags.csv``: per arm and snapshot click, the share of runs each domain-shift flag fires in, with the atypical and
unsupported shares' means. Before a run shows a detector the Find corpus is read at the typed query's own set there,
off the run's own typed-query rows, as the in-corpus read takes ``text_baseline.csv``'s.

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
#: The flag columns the snapshot clicks carry (#4960): each headline verdict, and the share behind it.
FLAGS = (
    ("shift", "shift_shifted", "shift_frac_atypical"),
    ("evidence", "evidence_unsupported", "evidence_frac_unsupported"),
)


def has_find(frame: pd.DataFrame) -> bool:
    """Whether *frame*'s runs Found on a second corpus (#4960)."""
    return "find_n_flagged" in frame.columns and bool(frame["find_n_flagged"].notna().any())


def find_anchor(frame: pd.DataFrame, metric: str) -> dict[tuple, float]:
    """Per cell, *metric* on the Find corpus before a detector shows: the typed query's own set there.

    The in-corpus read fills those votes from ``text_baseline.csv``; the Find corpus has no baseline file, so it
    reads the run's own typed-query rows (``detector_tier == "query"``, #4758), the same set on that corpus, at
    the first one. A run with none (no typed query) leaves those votes NaN, as a cell missing from the baseline is.
    """
    if "detector_tier" not in frame.columns:
        return {}
    q = frame[frame["detector_tier"] == "query"].sort_values("t").drop_duplicates(CELL)
    return {tuple(k): float(v) for k, v in zip(q[CELL].itertuples(index=False), q[metric], strict=True)}


def find_premise_failures(frames: dict[str, pd.DataFrame], preset: str) -> list[str]:
    """The paired arms of *preset* must Find on one corpus, drawn one way, or none of them may."""
    seen: dict[str, tuple] = {}
    for name, frame in frames.items():
        if not name.endswith(f"_{preset}"):
            continue
        corpus = tuple(sorted({str(v) for v in frame.get("find_corpus", pd.Series(dtype=str)).fillna("")}))
        mix = tuple(sorted({str(v) for v in frame.get("find_mix", pd.Series(dtype=str)).fillna("")}))
        seen[name] = (has_find(frame), corpus, mix)
    if len(set(seen.values())) > 1:
        return [f"{preset}: the arms Find on different corpora: " + "; ".join(f"{n} {v}" for n, v in seen.items())]
    return []


def flag_rows(frame: pd.DataFrame) -> list[dict]:
    """Per snapshot click, the share of runs each flag fires in and the mean share behind it (#4960)."""
    if "flags_at" not in frame.columns:
        return []
    snap = frame.dropna(subset=["flags_at"]).drop_duplicates([*CELL, "flags_at"])
    out = []
    for at, g in snap.groupby("flags_at"):
        row: dict = {"flags_at": int(at), "runs": int(len(g))}
        for name, verdict, share in FLAGS:
            read = g[verdict].dropna()
            row[f"{name}_runs"] = int(len(read))
            row[f"{name}_fires"] = float(read.mean()) if len(read) else np.nan
            row[f"{name}_{share.split('_', 1)[1]}"] = float(g[share].mean()) if g[share].notna().any() else np.nan
        row["evidence_frac_low_trust"] = (
            float(g["evidence_frac_low_trust"].mean()) if g["evidence_frac_low_trust"].notna().any() else np.nan
        )
        out.append(row)
    return out


def read_categories(path: "Path | None") -> "set[str] | None":
    """The cells *path* names, one ``class@band`` a line; None (every cell) with no path."""
    return None if path is None else set(path.read_text().split("\n")) - {""}


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
    ap.add_argument("--arms", default=",".join(a for a, *_ in ARMS), help="the arms to read; ctl must be one")
    ap.add_argument("--categories", type=Path, help="read only these cells, one class@band a line")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    wanted = {a.strip() for a in args.arms.split(",") if a.strip()}
    unknown = wanted - {a for a, *_ in ARMS}
    if unknown or "ctl" not in wanted:
        ap.error(f"--arms must name ctl and only known arms; got {sorted(wanted)}")
    arms = tuple(a for a in ARMS if a[0] in wanted)
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    keep = read_categories(args.categories)
    bl_path = Path(args.baseline) if args.baseline else base / "text_baseline.csv"
    bl = curves.text_sort_baseline(bl_path)
    raw = pd.read_csv(bl_path)
    if keep is not None:
        bl, raw = bl[bl["category"].isin(keep)], raw[raw["category"].isin(keep)]
    H = args.horizon

    lines = ["# #4671 Autopilot pieces - machine summary", ""]
    failures: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    provs: dict[str, dict] = {}
    for p, beta in PRESETS.items():
        for arm, _label, want in arms:
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
            if keep is not None:
                frame = frame[frame["category"].isin(keep)]
            failures += premise_failures(name, frame, logged_knobs(base / name), want, beta)
            frames[name] = frame
    failures += [f"baseline: {m}" for m in baseline_mismatches(raw, frames)]
    for p in PRESETS:
        failures += find_premise_failures(frames, p)
    lines.append("")
    if failures:
        lines += ["## Premise failures", "", *[f"- {f}" for f in failures], ""]
    (out / "provenance.json").write_text(json.dumps(provs, indent=2, default=str) + "\n")

    anchored = bl[CELL] if seeds is None else bl.loc[bl["seed"].isin(seeds), CELL]
    grid = pd.concat([*(f[CELL] for f in frames.values()), anchored], ignore_index=True).drop_duplicates()
    grid = grid.reset_index(drop=True)
    lines += [f"Grid: {len(grid)} cells.", ""]

    curve_rows, pair_rows, pcurve_rows, find_pcurve_rows = [], [], [], []
    find_curve_rows: list[pd.DataFrame] = []
    flags: list[dict] = []
    for p, beta in PRESETS.items():
        tag = beta_tag(beta)
        obj_anchor = text_fbeta(bl, f"text_line_precision_{tag}", f"text_line_recall_{tag}", beta)
        ap_anchor = {tuple(k): float(v) for k, v in bl.groupby(CELL)["text_AP"].mean().items()}
        sets = text_set(raw, f"text_line_precision_{tag}", f"text_line_recall_{tag}", f"text_line_fpr_{tag}")
        mats: dict[str, dict[str, pd.DataFrame]] = {}
        find_mats: dict[str, dict[str, pd.DataFrame]] = {}
        for arm, label, _want in arms:
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
            if has_find(frame):
                # #4960: the second corpus, read as the in-corpus columns are; F-beta off the rows (NaN with no positive).
                fgot = {
                    key: filled_matrix(frame, cells, {}, H, metric=col, baseline_metric=find_anchor(frame, col))[0]
                    for key, col in (
                        ("objective", "find_fbeta"),
                        ("ap", "find_average_precision"),
                        ("n_flagged", "find_n_flagged"),
                    )
                }
                find_mats[arm] = fgot
                fm, fk = fgot["objective"], fgot["n_flagged"]
                find_curve_rows.append(
                    pd.DataFrame(
                        {
                            "t": fm.columns.astype(int),
                            "objective": fm.mean(axis=0).to_numpy(),
                            "objective_se": (fm.std(axis=0, ddof=1) / np.sqrt(fm.notna().sum(axis=0))).to_numpy(),
                            "ap": fgot["ap"].mean(axis=0).to_numpy(),
                            "returned_median": fk.median(axis=0).to_numpy(),
                            "returned_mean": fk.mean(axis=0).to_numpy(),
                            "returned_none": (fk < 0.5).mean(axis=0).to_numpy(),
                            "runs": fm.notna().sum(axis=0).to_numpy(),
                            "runs_counted": fk.notna().sum(axis=0).to_numpy(),
                        }
                    ).assign(preset=p, beta=beta, arm=arm, label=label)
                )
                flags += [{"preset": p, "arm": arm, **r} for r in flag_rows(frame)]
        if "ctl" not in mats:
            continue
        for arm, label, _want in arms[1:]:
            if arm not in mats:
                continue
            for metric in ("objective", "ap"):
                for r in paired_rows(mats["ctl"][metric], mats[arm][metric]):
                    pair_rows.append({"preset": p, "arm": arm, "label": label, "metric": metric, **r})
            if "ctl" in find_mats and arm in find_mats:
                # The returned count too: on a corpus with no positive (`only` on a foreign dataset) it is the read.
                for metric in ("objective", "ap", "n_flagged"):
                    for r in paired_rows(find_mats["ctl"][metric], find_mats[arm][metric]):
                        pair_rows.append({"preset": p, "arm": arm, "label": label, "metric": f"find_{metric}", **r})
                find_pcurve_rows.append(
                    paired_curve(find_mats["ctl"]["objective"], find_mats[arm]["objective"], p, arm)
                )
            pcurve_rows.append(paired_curve(mats["ctl"]["objective"], mats[arm]["objective"], p, arm))

    curve = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    pairs = pd.DataFrame(pair_rows)
    pcurve = pd.concat(pcurve_rows, ignore_index=True) if pcurve_rows else pd.DataFrame()
    curve.to_csv(out / "curves.csv", index=False, float_format="%.6g")
    pairs.to_csv(out / "paired.csv", index=False, float_format="%.6g")
    pcurve.to_csv(out / "paired_curve.csv", index=False, float_format="%.6g")
    fcurve = pd.concat(find_curve_rows, ignore_index=True) if find_curve_rows else pd.DataFrame()
    flag_frame = pd.DataFrame(flags)
    if not fcurve.empty:
        fcurve.to_csv(out / "find_curves.csv", index=False, float_format="%.6g")
        flag_frame.to_csv(out / "flags.csv", index=False, float_format="%.6g")
    if find_pcurve_rows:
        pd.concat(find_pcurve_rows, ignore_index=True).to_csv(
            out / "find_paired_curve.csv", index=False, float_format="%.6g"
        )

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
    lines += find_report(fcurve, flag_frame, H)
    (out / "REPORT_autopilot.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    print("\n".join(lines))

    if not args.no_figures and not curve.empty:
        figures(curve, pcurve, out / "figures")
    return 1 if failures else 0


def paired_curve(a: pd.DataFrame, b: pd.DataFrame, preset: str, arm: str) -> pd.DataFrame:
    """``b - a`` at every vote on the runs both measured, with its SE: one arm against the control."""
    idx = a.index.intersection(b.index)
    d = b.loc[idx] - a.loc[idx]
    return pd.DataFrame(
        {
            "t": d.columns.astype(int),
            "delta": d.mean(axis=0).to_numpy(),
            "se": (d.std(axis=0, ddof=1) / np.sqrt(d.notna().sum(axis=0))).to_numpy(),
            "n": d.notna().sum(axis=0).to_numpy(),
        }
    ).assign(preset=preset, arm=arm)


def find_report(fcurve: pd.DataFrame, flag_frame: pd.DataFrame, horizon: int) -> list[str]:
    """The report's #4960 sections: the Find corpus's session means, and the flag table; nothing without one."""
    if fcurve.empty:
        return []
    lines = [
        "",
        "## Find on the second corpus (every run, every vote; #4960)",
        "",
        "Objective NaN where the corpus holds no positive: read the returned count there.",
        "",
        "| preset | arm | objective 1-150 | 1-50 | 51-150 | AP 1-150 | returned (median) 1-150 | returning none |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (p, arm), c in fcurve.groupby(["preset", "arm"], sort=False):
        w = c[(c["t"] >= 1) & (c["t"] <= horizon)]
        e = c[(c["t"] >= 1) & (c["t"] <= 50)]
        late = c[(c["t"] >= 51) & (c["t"] <= horizon)]
        lines.append(
            f"| {p} | {arm} | {w['objective'].mean():.3f} | {e['objective'].mean():.3f} | "
            f"{late['objective'].mean():.3f} | {w['ap'].mean():.3f} | {w['returned_median'].mean():.1f} "
            f"| {w['returned_none'].mean():.0%} |"
        )
    if flag_frame.empty:
        return [*lines, "", "No snapshot click carried the domain-shift flags (`CALIB_FIND_FLAGS_AT`)."]
    lines += [
        "",
        "## Domain-shift flags against the second corpus, at the snapshot clicks (#4960)",
        "",
        "Share of runs each flag fires in; the shares behind them are means over runs. "
        "The training-domain overlap is NaN on a patch embedder, which the app refuses.",
        "",
        "| preset | arm | click | runs | overlap fires | atypical share | evidence fires | unsupported share "
        "| low-trust share |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in flag_frame.itertuples(index=False):
        lines.append(
            f"| {r.preset} | {r.arm} | {r.flags_at} | {r.runs} | {r.shift_fires:.0%} | {r.shift_frac_atypical:.3f} "
            f"| {r.evidence_fires:.0%} | {r.evidence_frac_unsupported:.3f} | {r.evidence_frac_low_trust:.3f} |"
        )
    return lines


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

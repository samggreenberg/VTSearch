#!/usr/bin/env python3
"""#4668: the F-beta era as a cumulative build-up, each preset scored at its own beta.

Reads the rung arms ``launch_buildup_4668.sh`` wrote under ``--base`` and writes,
to ``--out``:

* ``buildup_curve.csv`` - per preset, rung and click: the objective's filled mean
  and SE (F-beta at the preset's own beta), AP, and the share of runs with a
  detector on screen.  **The slide's input** (``slides/figs/src/make-buildup-fig.py``).
* ``paired.csv`` - each rung against the one before it, and against
  cross-calibration, per preset: at fixed clicks and over windows of clicks.
* ``returned.csv`` - what the user gets at the horizon: precision, recall and the
  returned set's size, per preset and rung.
* ``REPORT_buildup.md`` - the machine summary: what loaded, what each rung ran
  with (read back off its rows and its run logs), and the tables.
* ``figures/`` - the build-up curves and steps, plus the mandatory AP pair
  (``curves.py``).

**The rungs.**  b1 cross-calibration (one set of sessions, scored at every beta);
then per preset b2 the labels line, b3 the relative floor, b4 the weak check,
b5 even-odds asking (today's app), b6 = b5's sessions with the typed query's set
drawn at today's per-preset line before the hand-over, and b7 the detector walk,
scored with the typed query on screen until Hard.

**Filled, as #4519.**  Until a run shows a detector (``app_trained``), the user
reads the typed query's set; its F-beta is the cell's text baseline at the line of
the rung's era: the mixture midpoint for b1-b5 (``--midpoint``), today's
per-preset line for b6-b7 (``--today``, #4603's ``text_line_*_<tag>`` columns).

    python analyze_buildup_4668.py --base /expscratch/$USER/buildup-4668 --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
import re
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
from analyze_progression_4184 import CELL, baseline_mismatches, fbeta, filled_matrix, summarize  # noqa: E402

PRESETS: dict[str, float] = {"b025": 0.25, "b1": 1.0, "b4": 4.0}
#: Autopilot's learned phases: where the app shows a detector when the walk is not shown (#4605, #4637).
TRAINED = ("hard", "new", "done", "exhausted")

#: (key, label, arm dir, typed-query baseline, what is shown).  ``{p}`` is the preset.
STEPS: list[tuple[str, str, str, str, str]] = [
    ("b1", "cross-calibration", "b1_xcal", "midpoint", "app"),
    ("b2", "the labels line at the preset", "b2_labels_{p}", "midpoint", "app"),
    ("b3", "+ the relative spread floor", "b3_floor_{p}", "midpoint", "app"),
    ("b4", "+ the weak-separation check", "b4_check_{p}", "midpoint", "app"),
    ("b5", "+ even-odds asking (today's sessions)", "b5_app_{p}", "midpoint", "app"),
    ("b6", "+ the typed query's per-preset line (today's app)", "b5_app_{p}", "today", "app"),
    ("b7", "+ the detector walk (not shipped)", "b7_walk_{p}", "today", "hard"),
]

#: What each arm's run log must say it ran with (``run_cells.py``'s header line).
EXPECT: dict[str, dict[str, str]] = {
    "b1_xcal": {"live_threshold": "xcal_mincost", "acq_inclusion_offset": "0.0", "sigma_floor": "relative"},
    "b2_labels": {"acq_target_p": "off", "sigma_floor": "absolute", "more_walk": "seed"},
    "b3_floor": {"acq_target_p": "off", "sigma_floor": "relative", "more_walk": "seed"},
    "b4_check": {"acq_target_p": "off", "sigma_floor": "relative", "more_walk": "seed"},
    # Unset under a balance is the app's even odds (#3546); the header prints it as None.
    "b5_app": {"acq_target_p": "None", "sigma_floor": "relative", "more_walk": "seed"},
    "b7_walk": {"acq_target_p": "None", "sigma_floor": "relative", "more_walk": "detector"},
}
#: Which arms run a check: the weak check comes in at b4.
CHECKS = {"b1_xcal": False, "b2_labels": False, "b3_floor": False, "b4_check": True, "b5_app": True, "b7_walk": True}

CHECKPOINTS = (10, 25, 50, 100, 150)
WINDOWS = ((1, 25), (1, 50), (51, 150), (1, 150))
#: The step table's columns, the session mean first: one number for "did this step improve the app" is the area
#: under the objective over votes, i.e. its mean (owner, 2026-10-09). Points on the curve come after.
STEP_COLUMNS = ("mean 1-150", "mean 1-50", "mean 51-150", "t=50", "t=150")
RETURNED_NOTE = (
    "Each row reads only the runs showing a detector at that vote, a subset that differs by rung before the "
    "hand-over (a check's steps carry no headline row). Compare rungs at vote 150; the curves carry the early votes."
)


def text_fbeta(baseline: pd.DataFrame, p_col: str, r_col: str, beta: float) -> dict[tuple, float]:
    per = baseline.groupby(CELL)[[p_col, r_col]].mean()
    vals = fbeta(per[p_col].to_numpy(), per[r_col].to_numpy(), beta)
    return {tuple(k): float(v) for k, v in zip(per.index, vals, strict=True)}


def logged_knobs(arm_dir: Path) -> dict[str, set[str]]:
    """Every ``key=value`` the arm's run logs printed in run_cells' header line, with the values seen."""
    seen: dict[str, set[str]] = {}
    logs = sorted((arm_dir / "logs").glob("cells-*.out"))
    for log in logs[:: max(1, len(logs) // 40)]:  # a spread of tasks is enough to catch a mixed arm
        for line in log.read_text(errors="replace").splitlines():
            if "dataset=" not in line or "sigma_floor=" not in line:
                continue
            for k, v in re.findall(r"(\w+)=(\S+)", line):
                seen.setdefault(k, set()).add(v)
    return seen


def premise_failures(arm: str, frame: pd.DataFrame, prov: dict, knobs: dict[str, set[str]], beta: float | None):
    out = []
    step = arm if arm == "b1_xcal" else arm.rsplit("_", 1)[0]
    for k, want in EXPECT[step].items():
        got = knobs.get(k)
        if got != {want}:
            out.append(f"{arm}: run logs say {k}={sorted(got) if got else 'nothing'}, expected {want}")
    if not knobs:
        out.append(f"{arm}: no run_cells header line found in logs/cells-*.out")
    b = pd.to_numeric(frame.get("beta"), errors="coerce")
    if beta is None:
        if b.notna().any():
            out.append(f"{arm}: rows carry a beta ({sorted(b.dropna().unique())}); the Inclusion arm has none")
        # On the rows the app shows: the opening's young labelsets take the centroid line (#4643), whose
        # acquisition is the text sort's, and they are scored as the typed query anyway.
        shown = frame["app_trained"].fillna(0).astype(int) == 1
        trained = frame[shown & np.isfinite(pd.to_numeric(frame["threshold"], errors="coerce"))]
        off = (trained["acq_threshold"] - trained["threshold"]).abs() > 1e-6
        if off.any():
            out.append(f"{arm}: {int(off.sum())} rows ask away from the reporting cut")
    elif not np.allclose(b.dropna().unique(), beta):
        out.append(f"{arm}: rows carry beta {sorted(b.dropna().unique())}, expected {beta}")
    checks = int(prov.get("check_rows") or 0)
    if CHECKS[step] != (checks > 0):
        out.append(f"{arm}: {checks} spot-check rows, expected {'some' if CHECKS[step] else 'none'}")
    pools = sorted({str(v) for v in frame.get("prevalence_arm", pd.Series(dtype=str)).dropna().unique()})
    if pools != ["haystack_0.01"]:
        out.append(f"{arm}: pool {pools}, expected haystack_0.01")
    return out


def paired_rows(a: pd.DataFrame, b: pd.DataFrame) -> list[dict]:
    """``b - a`` on the runs both measured: at each checkpoint and as a mean over each window of clicks."""
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    rows = []
    points = [(f"t={t}", [t]) for t in CHECKPOINTS] + [
        (f"mean {lo}-{hi}", list(range(lo, hi + 1))) for lo, hi in WINDOWS
    ]
    for name, ts in points:
        d = (b[ts].mean(axis=1) - a[ts].mean(axis=1)).dropna().to_numpy(float)
        k = len(d)
        mean = float(d.mean()) if k else float("nan")
        se = float(d.std(ddof=1) / np.sqrt(k)) if k > 1 else float("nan")
        better = float((d > 0.05).mean()) if k else float("nan")
        worse = float((d < -0.05).mean()) if k else float("nan")
        rows.append(
            {
                "at": name,
                "n": k,
                "mean": mean,
                "se": se,
                "resolvable": bool(k > 1 and abs(mean) > 2 * se),
                "share_better_005": better,
                "share_worse_005": worse,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="launch_buildup_4668.sh's BASE: one dir per arm")
    ap.add_argument(
        "--midpoint", help="text_baseline.py CSV at the mixture midpoint (default BASE/text_baseline_midpoint.csv)"
    )
    ap.add_argument("--today", help="text_baseline.py CSV at today's lines (default BASE/text_baseline.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--seeds", help="comma-separated seeds to read (an interim read of the first seeds); default all")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    presets = [p for p in args.presets.split(",") if p]
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    bl = {
        "midpoint": curves.text_sort_baseline(args.midpoint or base / "text_baseline_midpoint.csv"),
        "today": curves.text_sort_baseline(args.today or base / "text_baseline.csv"),
    }

    lines = ["# #4668 build-up - machine summary", ""]
    failures: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    provs: dict[str, dict] = {}
    arms_needed = sorted({d.format(p=p) for _k, _l, d, _b, _s in STEPS for p in presets})
    for arm in arms_needed:
        if not (base / arm / "results" / "cells").is_dir():
            failures.append(f"{arm}: no cells dir under {base / arm}")
            continue
        frame, prov = _cells_io.load_arm(base / arm / "results")
        provs[arm] = {**prov, "dir": f"<base>/{arm}/results"}
        lines.append(f"- `{arm}`: {_cells_io.describe_load(prov)}")
        if frame.empty:
            failures.append(f"{arm}: no rows")
            continue
        if seeds is not None:
            frame = frame[frame["seed"].isin(seeds)]
        beta = None if arm == "b1_xcal" else PRESETS[arm.rsplit("_", 1)[1]]
        failures += premise_failures(arm, frame, prov, logged_knobs(base / arm), beta)
        frames[arm] = frame
    for name, path in (
        ("midpoint", args.midpoint or base / "text_baseline_midpoint.csv"),
        ("today", args.today or base / "text_baseline.csv"),
    ):
        failures += [f"{name} baseline: {m}" for m in baseline_mismatches(pd.read_csv(path), frames)]
    lines.append("")
    if failures:
        lines += ["## Premise failures", "", *[f"- {f}" for f in failures], ""]
    (out / "provenance.json").write_text(json.dumps(provs, indent=2, default=str) + "\n")

    seen = pd.concat([f[CELL] for f in frames.values()], ignore_index=True)
    anchored = bl["today"][CELL] if seeds is None else bl["today"].loc[bl["today"]["seed"].isin(seeds), CELL]
    grid = pd.concat([seen, anchored], ignore_index=True).drop_duplicates().reset_index(drop=True)
    lines += [f"Grid: {len(grid)} cells.", ""]

    curve_rows, pair_rows, ret_rows = [], [], []
    mats: dict[tuple[str, str], pd.DataFrame] = {}
    lines += ["| preset | rung | arm | cells | lost | coverage@25 | coverage@50 |", "|---|---|---|---|---|---|---|"]
    for p in presets:
        beta = PRESETS[p]
        tag = beta_tag(beta)
        anchors = {
            "midpoint": text_fbeta(bl["midpoint"], "text_precision", "text_recall", beta),
            "today": text_fbeta(bl["today"], f"text_line_precision_{tag}", f"text_line_recall_{tag}", beta),
        }
        ap_anchor = {tuple(k): float(v) for k, v in bl["today"].groupby(CELL)["text_AP"].mean().items()}
        for order, (key, label, d, which, shown) in enumerate(STEPS, start=1):
            arm = d.format(p=p)
            if arm not in frames:
                continue
            frame = frames[arm].copy()
            frame["objective"] = fbeta(frame["precision"], frame["recall"], beta)
            if shown == "hard":
                frame["app_trained"] = frame["phase"].astype(str).isin(TRAINED).astype(int)
            present = frame[CELL].drop_duplicates()
            starved_files = provs[arm].get("no_positive_found") or []
            if seeds is not None:
                # Seed-major grid (CALIB_CELL_ORDER=seed): task index // cells-per-seed is the seed.
                per_seed = len(grid) // len(seeds)
                starved_files = [f for f in starved_files if int(re.findall(r"\d+", f)[0]) // per_seed in seeds]
            starved = len(starved_files)
            missing = max(len(grid) - len(present) - starved, 0)
            lost = len(provs[arm].get("unreadable") or []) + len(provs[arm].get("zero_byte") or []) + missing
            cells = grid if lost == 0 else present
            m, shown_mask = filled_matrix(
                frame, cells, {}, args.horizon, metric="objective", baseline_metric=anchors[which]
            )
            m_ap, _ = filled_matrix(
                frame, cells, {}, args.horizon, metric="average_precision", baseline_metric=ap_anchor
            )
            mats[(p, key)] = m
            s = summarize(m)
            s_ap = summarize(m_ap)
            cov = shown_mask.mean(axis=0)
            s = s.assign(
                preset=p,
                beta=beta,
                rung_order=order,
                rung=key,
                label=label,
                arm=arm,
                coverage=cov.to_numpy(),
                ap_mean=s_ap["mean"].to_numpy(),
                ap_se=s_ap["se"].to_numpy(),
            )
            curve_rows.append(s)
            lines.append(
                f"| {p} | {key} | {arm} | {len(cells)} | {lost} | {cov.get(25, np.nan):.2f} | {cov.get(50, np.nan):.2f} |"
            )
            # What the user gets at votes 25 and 50 and at the horizon, on runs showing a detector by then.
            for at in sorted({25, 50, args.horizon}):
                last = frame[(frame["t"] == at) & (frame["app_trained"].fillna(0).astype(int) == 1)]
                if last.empty:
                    continue
                ret_rows.append(
                    {
                        "preset": p,
                        "rung": key,
                        "label": label,
                        "t": at,
                        "runs": len(last),
                        "precision": float(np.nan_to_num(last["precision"]).mean()),
                        "recall": float(last["recall"].mean()),
                        "returned_median": float(last["n_flagged"].median()),
                        "returned_mean": float(last["n_flagged"].mean()),
                        "over_200": float((last["n_flagged"] > 200).mean()),
                    }
                )
        keys = [k for k, *_ in STEPS if (p, k) in mats]
        for prev, cur in zip(keys, keys[1:], strict=False):
            for r in paired_rows(mats[(p, prev)], mats[(p, cur)]):
                pair_rows.append({"preset": p, "from": prev, "to": cur, "kind": "step", **r})
        for cur in keys[1:]:
            for r in paired_rows(mats[(p, keys[0])], mats[(p, cur)]):
                pair_rows.append({"preset": p, "from": keys[0], "to": cur, "kind": "vs_b1", **r})
    lines.append("")

    curve = pd.concat(curve_rows, ignore_index=True)[
        ["preset", "beta", "rung_order", "rung", "label", "arm", "t", "mean", "se", "n", "coverage", "ap_mean", "ap_se"]
    ]
    curve.to_csv(out / "buildup_curve.csv", index=False, float_format="%.6g")
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "paired.csv", index=False, float_format="%.6g")
    returned = pd.DataFrame(ret_rows)
    returned.to_csv(out / "returned.csv", index=False, float_format="%.6g")

    pts = [0, *CHECKPOINTS]
    for p in presets:
        c = curve[curve["preset"] == p]
        # The session mean leads: the area under the curve over votes, divided by the votes (owner, 2026-10-09).
        lines += [f"## Beta {PRESETS[p]:g}: the objective (filled), the session mean first", ""]
        lines += [
            "| rung | mean 1-150 | mean 1-50 | " + " | ".join(f"t={t}" for t in pts) + " |",
            "|---|" + "---|" * (len(pts) + 2),
        ]
        for key, label, *_ in STEPS:
            ck = c[c["rung"] == key].set_index("t")
            if ck.empty:
                continue
            avg = ck.loc[1 : args.horizon, "mean"].mean()
            early = ck.loc[1:50, "mean"].mean()
            lines.append(
                f"| {key} {label} | {avg:.2g} | {early:.2g} | "
                + " | ".join(f"{ck.loc[t, 'mean']:.2g}" for t in pts)
                + " |"
            )
        lines += ["", "Each rung against the one before (positive = the step helped; bold = beyond 2 SE):", ""]
        lines += [
            "| step | " + " | ".join(r for r in STEP_COLUMNS) + " |",
            "|---|---|---|---|---|---|",
        ]
        pp = pairs[(pairs["preset"] == p) & (pairs["kind"] == "step")]
        for (f, t), g in pp.groupby(["from", "to"], sort=False):
            g = g.set_index("at")
            cells_ = []
            for at in STEP_COLUMNS:
                r = g.loc[at]
                star = "**" if r["resolvable"] else ""
                cells_.append(f"{star}{r['mean']:+.3f}{star} ± {r['se']:.3f}")
            lines.append(f"| {f} → {t} | " + " | ".join(cells_) + " |")
        lines.append("")
    if not returned.empty:
        lines += ["## What the user gets at votes 25, 50 and 150 (runs showing a detector)", "", RETURNED_NOTE, ""]
        lines += [
            "| preset | rung | vote | runs | precision | recall | returned, median | over 200 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in returned.itertuples(index=False):
            lines.append(
                f"| {r.preset} | {r.rung} | {r.t} | {r.runs} | {r.precision:.2f} | {r.recall:.2f} "
                f"| {r.returned_median:.0f} | {r.over_200:.0%} |"
            )
        lines.append("")
    (out / "REPORT_buildup.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    print("\n".join(lines))

    if not args.no_figures and frames:
        figures(curve, out / "figures", presets)
        main_frame = pd.concat([f.assign(arm=a) for a, f in frames.items()], ignore_index=True)
        denominator = pd.concat([grid.assign(arm=a) for a in frames], ignore_index=True)
        curves.quality_vs_clicks(
            main_frame,
            out / "figures",
            arms=list(frames),
            metric="average_precision",
            denominator=denominator,
            baseline=bl["today"],
            lower_is_better=False,
        )
    if not args.no_viewer and frames:
        import viewer  # noqa: PLC0415

        main_frame = pd.concat([f.assign(arm=a) for a, f in frames.items()], ignore_index=True)
        denominator = pd.concat([grid.assign(arm=a) for a in frames], ignore_index=True)
        viewer.build_viewer(
            main_frame,
            out / "viewer.html",
            arms=list(frames),
            denominator=denominator,
            baseline=bl["today"],
            title="COCO Better: the F-beta era, built up",
            subtitle="Binary voting, SigLIP, 144 cells x 5 seeds, the user's pool at 1%; one arm per rung and preset (#4668)",
        )
    return 1 if failures else 0


def figures(curve: pd.DataFrame, figdir: Path, presets: list[str]) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figdir.mkdir(parents=True, exist_ok=True)
    keys = [k for k, *_ in STEPS]
    cmap = plt.get_cmap("viridis")
    colours = {k: cmap(i / max(1, len(keys) - 1)) for i, k in enumerate(keys)}
    fig, axes = plt.subplots(1, len(presets), figsize=(5.2 * len(presets), 4.2), sharey=False, squeeze=False)
    for ax, p in zip(axes[0], presets, strict=True):
        c = curve[curve["preset"] == p]
        for key, label, *_ in STEPS:
            ck = c[c["rung"] == key]
            if ck.empty:
                continue
            ax.plot(ck["t"], ck["mean"], color=colours[key], lw=1.6, label=f"{key} {label}")
        ax.set_title(f"F-beta {PRESETS[p]:g}, at its own preset")
        ax.set_xlabel("clicks")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("objective (withheld half, the line the app shows)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "buildup_curves.png", dpi=130)
    plt.close(fig)

    # Two windows: the whole session (the area under the curve, as a mean) and a short one, which is most sessions.
    windows = (("1-150", 1, 150, "#2a78d6"), ("1-50", 1, 50, "#eb6834"))
    fig, axes = plt.subplots(1, len(presets), figsize=(5.2 * len(presets), 3.8), squeeze=False)
    for ax, p in zip(axes[0], presets, strict=True):
        for name, lo, hi, colour in windows:
            c = curve[(curve["preset"] == p) & (curve["t"] >= lo) & (curve["t"] <= hi)]
            means = c.groupby("rung", sort=False)["mean"].mean().reindex(keys).dropna()
            ax.step(range(len(means)), means.to_numpy(), where="mid", color=colour)
            ax.plot(range(len(means)), means.to_numpy(), "o", color=colour, label=f"mean over votes {name}")
            ax.set_xticks(range(len(means)), list(means.index))
        ax.set_title(f"F-beta {PRESETS[p]:g}, at its own preset")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "buildup_steps.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

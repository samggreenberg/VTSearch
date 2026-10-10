#!/usr/bin/env python3
"""#4746: Autopilot's acquisition target precision per preset, priced on the objective against today's app.

Reads the arms ``launch_acqtarget_4746.sh`` wrote under ``--base`` (``<arm>_<preset>``: ``ctl`` is today's app,
target 0.5; ``tp75`` samples where the labels line's corpus posterior falls below 0.75, ``tp25`` below 0.25) and
pairs each test arm with the control at its preset on (category, seed), the SE clustered on category.  Writes, to
``--out``:

* ``curves.csv`` - per preset, arm and vote: the objective (F-beta at the preset's own beta of the withheld half
  above the line the app shows), AP, precision, recall, the returned set's median size, and the share of runs
  showing a detector.
* ``paired.csv`` - each test arm minus the control: the objective's session means over votes 1-150, 1-50, 1-25 and
  51-150, its points at votes 25, 50, 100 and 150, and after the end-of-session check; AP at vote 150 and over
  1-150; Goods by vote 150; the share of Boundary (``hard``) and New picks that are Good; the returned set's size
  and precision after the check.
* ``paired_curve.csv`` - the paired difference in the objective at every vote, with its clustered SE.
* ``levels.csv`` - each arm's own level on the same reads.
* ``REPORT_acqtarget.md`` - the machine summary; ``provenance.json``; ``figures/``.

**Every run counts at every vote** (#4631): until a run shows a detector, Find returns the typed query's own set
at today's per-preset line (#4603), and that is what it scores.  The single number is the session mean (the area
under the curve over the votes, owner 2026-10-09).

    python analyze_acqtarget_4746.py --base /expscratch/$USER/acqtarget-4746 --out OUTDIR
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
import analyze_calsplit_4583 as A  # noqa: E402
import curves  # noqa: E402
from _rank_metrics import beta_tag  # noqa: E402
from analyze_acqcut_3546 import harvest  # noqa: E402
from analyze_buildup_4668 import logged_knobs, text_fbeta, text_set  # noqa: E402
from analyze_progression_4184 import CELL, baseline_mismatches, fbeta, filled_matrix  # noqa: E402

PRESETS: dict[str, float] = {"b025": 0.25, "b1": 1.0, "b4": 4.0}
#: What each arm's run logs must print for ``acq_target_p`` (the header prints the app's ``None``).
TARGETS: dict[str, str] = {"ctl": "None", "tp75": "0.75", "tp25": "0.25"}
LABELS: dict[str, str] = {"ctl": "today's app (target 0.5)", "tp75": "target 0.75", "tp25": "target 0.25"}
WINDOWS = ((1, 150), (1, 50), (1, 25), (51, 150))
POINTS = (25, 50, 100, 150)
RUN = ["category", "seed"]


def premise_failures(arm: str, frame: pd.DataFrame, knobs: dict[str, set[str]], want: str, beta: float) -> list[str]:
    out = []
    if not knobs:
        return [f"{arm}: no run_cells header line found in logs/cells-*.out"]
    got = knobs.get("acq_target_p")
    if got != {want}:
        out.append(f"{arm}: run logs say acq_target_p={sorted(got) if got else 'nothing'}, expected {want}")
    b = pd.to_numeric(frame.get("beta"), errors="coerce").dropna().unique()
    if not np.allclose(b, beta):
        out.append(f"{arm}: rows carry beta {sorted(b)}, expected {beta}")
    pools = sorted({str(v) for v in frame.get("prevalence_arm", pd.Series(dtype=str)).dropna().unique()})
    if pools != ["haystack_0.01"]:
        out.append(f"{arm}: pool {pools}, expected haystack_0.01")
    return out


def by_run(m: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Re-index a CELL-indexed matrix on (category, seed): one dataset and embedder here."""
    return m.droplevel([lv for lv in m.index.names if lv not in RUN])


def new_pick_share(cells_dir: Path) -> pd.Series:
    """Per run: the share of New-phase picks (the atlas probe) that are Good."""
    parts = []
    for p in _cells_io.side_frame_files(cells_dir, "__picks"):
        try:
            k = pd.read_csv(p, usecols=["category", "seed", "phase", "picked_label"])
        except (ValueError, OSError, pd.errors.EmptyDataError):
            continue
        parts.append(k[k["phase"] == "new"])
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True).groupby(RUN)["picked_label"].mean()


def arm_reads(frame: pd.DataFrame, cells: pd.DataFrame, anchors: dict, beta: float, cells_dir: Path, H: int) -> dict:
    """One arm's per-run matrices and per-run scalars, every run filled from the typed query until it hands over."""
    is_check = frame["phase"].fillna("").astype(str).str.strip() == "check"
    votes = frame[~is_check].copy()
    votes["objective"] = fbeta(votes["precision"], votes["recall"], beta)
    votes["precision"] = np.nan_to_num(votes["precision"].to_numpy(dtype=float))
    m, shown = filled_matrix(votes, cells, {}, H, metric="objective", baseline_metric=anchors["objective"])
    got: dict = {"objective": m, "shown": shown.astype(int).cummax(axis=1).astype(bool)}
    got["ap"], _ = filled_matrix(votes, cells, {}, H, metric="average_precision", baseline_metric=anchors["ap"])
    for col in ("precision", "recall", "n_flagged"):
        got[col], _ = filled_matrix(votes, cells, {}, H, metric=col, baseline_metric=anchors[col])
    got = {k: by_run(v) for k, v in got.items()}
    # After the check: the last check row, else the run's own vote-H value (a run the check never reached).
    after = got["objective"][H].copy()
    k_after = got["n_flagged"][H].copy()
    p_after = got["precision"][H].copy()
    checked = frame[is_check]
    if len(checked):
        last = checked.sort_values("t").groupby(RUN).tail(1).set_index(RUN)
        last = last[last.index.isin(after.index)]
        after.loc[last.index] = fbeta(last["precision"].to_numpy(float), last["recall"].to_numpy(float), beta)
        k_after.loc[last.index] = last["n_flagged"].astype(float)
        p_after.loc[last.index] = np.nan_to_num(last["precision"].to_numpy(dtype=float))
    got["after"], got["k_after"], got["p_after"] = after, k_after, p_after
    h = harvest(frame, cells_dir)
    for col in ("goods", "hard_good_share", "hard_picks", "acq_pct", "line_pct"):
        got[col] = h[col] if col in h else pd.Series(dtype=float)
    got["new_good_share"] = new_pick_share(cells_dir)
    return got


def scalar_reads(g: dict, H: int) -> dict[str, pd.Series]:
    """The per-run numbers a contrast is read on."""
    obj = g["objective"]
    out = {f"objective, mean {lo}-{hi}": A.window_mean(obj, lo, hi) for lo, hi in WINDOWS}
    out |= {f"objective, vote {t}": obj[t] for t in POINTS if t <= H}
    out["objective, after the check"] = g["after"]
    out[f"AP, vote {H}"] = g["ap"][H]
    out[f"AP, mean 1-{H}"] = A.window_mean(g["ap"], 1, H)
    out[f"Goods by vote {H}"] = g["goods"]
    out["Boundary picks Good"] = g["hard_good_share"]
    out["New picks Good"] = g["new_good_share"]
    out["returned, after the check"] = g["k_after"]
    out["precision, after the check"] = g["p_after"]
    out[f"cut's pool percentile, vote {H}"] = g["acq_pct"]
    return out


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--baseline", help="text_baseline.py CSV at today's lines (default BASE/text_baseline.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="ctl,tp75,tp25", help="arm stems; the ones with no cells are skipped")
    ap.add_argument("--control", default="ctl")
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seeds", help="comma-separated seeds to read (an interim read); default all")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    stems = [a for a in args.arms.split(",") if a]
    presets = [p for p in args.presets.split(",") if p]
    bl_path = Path(args.baseline) if args.baseline else base / "text_baseline.csv"
    bl = curves.text_sort_baseline(bl_path)
    raw = pd.read_csv(bl_path)
    H = args.horizon

    lines = ["# #4746 acquisition target per preset - machine summary", ""]
    failures: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    provs: dict[str, dict] = {}
    for p in presets:
        for stem in stems:
            name = f"{stem}_{p}"
            if not (base / name / "results" / "cells").is_dir():
                continue
            frame, prov = _cells_io.load_arm(base / name / "results", keep_check=True)
            provs[name] = {**prov, "dir": f"<base>/{name}/results"}
            lines.append(f"- `{name}`: {_cells_io.describe_load(prov)}")
            if frame.empty:
                failures.append(f"{name}: no rows")
                continue
            if seeds is not None:
                frame = frame[frame["seed"].isin(seeds)]
            failures += premise_failures(name, frame, logged_knobs(base / name), TARGETS[stem], PRESETS[p])
            frames[name] = frame
    failures += [f"baseline: {m}" for m in baseline_mismatches(raw, frames)]
    lines.append("")
    if failures:
        lines += ["## Premise failures", "", *[f"- {f}" for f in failures], ""]
    (out / "provenance.json").write_text(json.dumps(provs, indent=2, default=str) + "\n")

    anchored = bl[CELL] if seeds is None else bl.loc[bl["seed"].isin(seeds), CELL]
    grid = pd.concat([*(f[CELL] for f in frames.values()), anchored], ignore_index=True).drop_duplicates()
    grid = grid.reset_index(drop=True)
    lines += [f"Grid: {len(grid)} runs.", ""]

    curve_rows, pair_rows, pcurve_rows, level_rows = [], [], [], []
    for p in presets:
        beta = PRESETS[p]
        tag = beta_tag(beta)
        sets = text_set(raw, f"text_line_precision_{tag}", f"text_line_recall_{tag}", f"text_line_fpr_{tag}")
        anchors = {
            "objective": text_fbeta(bl, f"text_line_precision_{tag}", f"text_line_recall_{tag}", beta),
            "ap": {tuple(k): float(v) for k, v in bl.groupby(CELL)["text_AP"].mean().items()},
            **sets,
        }
        reads: dict[str, dict] = {}
        for stem in stems:
            name = f"{stem}_{p}"
            if name not in frames:
                continue
            frame = frames[name]
            present = frame[CELL].drop_duplicates()
            lost = len(provs[name].get("unreadable") or []) + len(provs[name].get("zero_byte") or [])
            starved = len(provs[name].get("no_positive_found") or [])
            if seeds is None:
                lost += max(len(grid) - len(present) - starved, 0)
            cells = grid if lost == 0 else present
            if lost:
                lines.append(f"- {name}: {lost} runs lost or not yet written; read on the {len(present)} present")
            g = arm_reads(frame, cells, anchors, beta, base / name / "results" / "cells", H)
            reads[stem] = g
            m, k = g["objective"], g["n_flagged"]
            curve_rows.append(
                pd.DataFrame(
                    {
                        "t": m.columns.astype(int),
                        "objective": m.mean(axis=0).to_numpy(),
                        "objective_se": (m.std(axis=0, ddof=1) / np.sqrt(m.notna().sum(axis=0))).to_numpy(),
                        "ap": g["ap"].mean(axis=0).to_numpy(),
                        "precision": g["precision"].mean(axis=0).to_numpy(),
                        "recall": g["recall"].mean(axis=0).to_numpy(),
                        "returned_median": k.median(axis=0).to_numpy(),
                        "showing_detector": g["shown"].mean(axis=0).to_numpy(),
                        "runs": m.notna().sum(axis=0).to_numpy(),
                    }
                ).assign(preset=p, beta=beta, arm=stem)
            )
            for read, s in scalar_reads(g, H).items():
                s = s.dropna()
                level_rows.append({"preset": p, "beta": beta, "arm": stem, "read": read, "n": len(s), "mean": s.mean()})
        if args.control not in reads:
            continue
        ctl = reads[args.control]
        c_reads = scalar_reads(ctl, H)
        for stem in stems:
            if stem == args.control or stem not in reads:
                continue
            g = reads[stem]
            for read, s in scalar_reads(g, H).items():
                d = (s - c_reads[read]).dropna()
                mean, se, sd, n = A.clustered(d) if len(d) else (np.nan, np.nan, np.nan, 0)
                pair_rows.append(
                    {"preset": p, "beta": beta, "arm": stem, "read": read, "n": n, "delta": mean, "se": se, "sd": sd}
                )
            a, b = ctl["objective"], g["objective"]
            idx = a.index.intersection(b.index)
            d = b.loc[idx] - a.loc[idx]
            per_vote = [A.clustered(d[t]) for t in d.columns]
            pcurve_rows.append(
                pd.DataFrame(
                    {
                        "t": d.columns.astype(int),
                        "delta": [x[0] for x in per_vote],
                        "se": [x[1] for x in per_vote],
                        "n": [x[3] for x in per_vote],
                    }
                ).assign(preset=p, arm=stem)
            )

    curve = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs["resolved"] = pairs["delta"].abs() > 2 * pairs["se"]
    pcurve = pd.concat(pcurve_rows, ignore_index=True) if pcurve_rows else pd.DataFrame()
    levels = pd.DataFrame(level_rows)
    curve.to_csv(out / "curves.csv", index=False, float_format="%.6g")
    pairs.to_csv(out / "paired.csv", index=False, float_format="%.6g")
    pcurve.to_csv(out / "paired_curve.csv", index=False, float_format="%.6g")
    levels.to_csv(out / "levels.csv", index=False, float_format="%.6g")

    if not levels.empty:
        lines += ["## Levels", ""]
        wide = levels.pivot_table(index=["preset", "arm"], columns="read", values="mean", sort=False)
        keep = [
            f"objective, mean 1-{H}",
            "objective, mean 1-50",
            "objective, after the check",
            f"AP, vote {H}",
            f"Goods by vote {H}",
            "Boundary picks Good",
            "New picks Good",
            f"cut's pool percentile, vote {H}",
        ]
        keep = [c for c in keep if c in wide.columns]
        lines += ["| preset | arm | " + " | ".join(keep) + " |", "|---|---|" + "---|" * len(keep)]
        for (p, arm), r in wide.iterrows():
            lines.append(f"| {p} | {arm} | " + " | ".join(f"{r[c]:.3f}" for c in keep) + " |")
        lines.append("")
    if not pairs.empty:
        lines += [f"## Each arm minus `{args.control}` (paired on category x seed, SE clustered on category)", ""]
        lines += ["| preset | arm | read | n | delta | SE |", "|---|---|---|---|---|---|"]
        for _, r in pairs.iterrows():
            star = "**" if r["resolved"] else ""
            lines.append(
                f"| {r['preset']} | {r['arm']} | {r['read']} | {r['n']} | {star}{r['delta']:+.4f}{star} | {r['se']:.4f} |"
            )
    (out / "REPORT_acqtarget.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    print("\n".join(lines))

    if not args.no_figures and not curve.empty:
        figures(curve, pcurve, out / "figures", args.control)
    return 1 if failures else 0


def figures(curve: pd.DataFrame, pcurve: pd.DataFrame, figdir: Path, control: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figdir.mkdir(parents=True, exist_ok=True)
    colours = {"ctl": "#2a78d6", "tp75": "#eb6834", "tp25": "#1baf7a"}
    presets = list(dict.fromkeys(curve["preset"]))
    fig, axes = plt.subplots(2, len(presets), figsize=(5.4 * len(presets), 7.2), squeeze=False)
    for j, p in enumerate(presets):
        ax = axes[0][j]
        for arm, c in curve[curve["preset"] == p].groupby("arm", sort=False):
            ax.plot(c["t"], c["objective"], color=colours.get(arm, "#777"), lw=1.6, label=LABELS.get(arm, arm))
        ax.set_title(f"objective at beta {PRESETS[p]:g}")
        ax.grid(alpha=0.3)
        ax = axes[1][j]
        ax.axhline(0, color="#888", lw=1)
        for arm, c in pcurve[pcurve["preset"] == p].groupby("arm", sort=False):
            col = colours.get(arm, "#777")
            ax.plot(c["t"], c["delta"], color=col, lw=1.6, label=f"{LABELS.get(arm, arm)} - {control}")
            ax.fill_between(c["t"], c["delta"] - 2 * c["se"], c["delta"] + 2 * c["se"], color=col, alpha=0.15)
        ax.set_title(f"paired difference, beta {PRESETS[p]:g} (±2 SE)")
        ax.set_xlabel("votes")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    axes[1][0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "acqtarget_curves.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

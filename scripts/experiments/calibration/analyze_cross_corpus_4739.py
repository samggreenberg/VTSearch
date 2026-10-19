#!/usr/bin/env python3
"""#4739: the Coverage Atlas walk priced on a cross-corpus Find, read across the legs.

Reads the legs ``launch_cross_corpus_4739.sh`` wrote under ``--root`` (``<leg>/{ctl,nowalk}_{b025,b1,b4}``) and
writes, to ``--out``:

* ``<leg>/`` - ``analyze_autopilot_4671``'s whole read of that leg (``--arms ctl,nowalk``): the in-corpus and the
  Find corpus's curves, the paired differences, the flag table, its figures.
* ``legs.csv`` - per leg, preset and metric, ``nowalk`` minus ``ctl`` paired on (category, seed): the in-corpus
  objective and AP, and the Find corpus's objective, AP and returned count, over the session windows and at vote 150.
  Positive is the Hard pick ahead of the walk.
* ``flags.csv`` - per leg, preset, arm and snapshot click: the share of runs each domain-shift flag fires in, and the
  atypical / unsupported shares behind them (#4960).
* ``span.csv`` - per preset and arm: when Span goes green and when the session reaches Done, which the walk drives.
* ``identity.csv`` - per preset and arm, each leg against the first one read: how many runs' in-corpus rows differ.
  The Find corpus is only ever scored, so every leg runs the same trajectories, and a difference is a premise failure.
* ``REPORT_cross_corpus.md``, ``figures/``, and the mandatory quality-over-clicks pair and ``viewer.html`` over the
  in-corpus read of the first leg (the trajectories every leg shares).

    python analyze_cross_corpus_4739.py --root /expscratch/$USER/cross-corpus-4739 --out OUTDIR
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
import analyze_autopilot_4671 as ap4671  # noqa: E402
import curves  # noqa: E402
from analyze_progression_4184 import CELL  # noqa: E402

PRESETS: dict[str, float] = ap4671.PRESETS
ARMS = ("ctl", "nowalk")
#: (leg, what it Finds on), in the order the report reads them.
LEGS: tuple[tuple[str, str], ...] = (
    ("screens0", "the withheld half alone (no shift)"),
    ("o365", "Objects365 Better's cell of the same name"),
    ("screens_equal", "the withheld half plus Apptitude's 3,141 screens"),
    ("screens_only", "Apptitude's screens alone (no positive)"),
)
LEG_LABEL = dict(LEGS)
#: The paired rows the legs table carries, and the windows it reads them over.
METRICS = ("objective", "ap", "find_objective", "find_ap", "find_n_flagged")
AT = ("mean 1-150", "mean 1-50", "mean 51-150", "t=150")
#: The columns a trajectory is: every leg must write them identically (timings and the Find read excepted).
TRAJECTORY = (
    "t",
    "n_good",
    "n_bad",
    "phase",
    "span",
    "span_level",
    "app_trained",
    "detector_tier",
    "threshold",
    "precision",
    "recall",
    "fbeta",
    "n_flagged",
    "average_precision",
)
SPAN_BY = (25, 50, 100, 150)


def leg_frames(root: Path, leg: str, seeds: "list[int] | None") -> dict[str, pd.DataFrame]:
    """Every arm's base rows in *leg*, keyed ``<arm>_<preset>``; an arm with no cells is absent."""
    out = {}
    for p in PRESETS:
        for arm in ARMS:
            d = root / leg / f"{arm}_{p}" / "results"
            if not (d / "cells").is_dir():
                continue
            frame, _prov = _cells_io.load_arm(d)
            if frame.empty:
                continue
            if seeds is not None:
                frame = frame[frame["seed"].isin(seeds)]
            out[f"{arm}_{p}"] = frame
    return out


def span_rows(frames: dict[str, pd.DataFrame], horizon: int) -> list[dict]:
    """Per arm: the vote Span first goes green and the session first reaches Done, over the runs."""
    rows = []
    for name, frame in frames.items():
        arm, p = name.rsplit("_", 1)
        f = frame[frame["t"] <= horizon]
        runs = f[CELL].drop_duplicates()
        green = f[f["span"] == "green"].groupby(CELL)["t"].min()
        done = f[f["phase"] == "done"].groupby(CELL)["t"].min()
        idx = pd.MultiIndex.from_frame(runs)
        green = green.reindex(idx)
        done = done.reindex(idx)
        last = f.sort_values("t").groupby(CELL).tail(1).set_index(CELL)
        row = {"preset": p, "arm": arm, "runs": len(runs)}
        for t in SPAN_BY:
            row[f"span_green_by_{t}"] = float((green <= t).mean())
            row[f"done_by_{t}"] = float((done <= t).mean())
        row["span_green_median_vote"] = float(green.median()) if green.notna().any() else np.nan
        row["done_median_vote"] = float(done.median()) if done.notna().any() else np.nan
        row["span_level_last_mean"] = float(pd.to_numeric(last["span_level"], errors="coerce").mean())
        row["span_target"] = float(pd.to_numeric(last["span_target"], errors="coerce").median())
        rows.append(row)
    return rows


def trajectory_key(frame: pd.DataFrame) -> pd.Series:
    """Per run, one string holding its in-corpus rows in order: equal across legs exactly when the runs are."""
    cols = [c for c in TRAJECTORY if c in frame.columns]
    f = frame[[*CELL, *cols]].copy()
    for c in cols:
        if f[c].dtype.kind == "f":
            f[c] = f[c].round(9)
    f = f.astype(str)
    return f.groupby(CELL, sort=True)[cols].apply(lambda g: "|".join(g.to_numpy().ravel()))


def identity_rows(ref_leg: str, ref: dict[str, pd.DataFrame], leg: str, other: dict[str, pd.DataFrame]) -> list[dict]:
    """Per arm, how many runs of *leg* wrote in-corpus rows other than *ref_leg*'s."""
    rows = []
    for name in sorted(set(ref) & set(other)):
        arm, p = name.rsplit("_", 1)
        a, b = trajectory_key(ref[name]), trajectory_key(other[name])
        both = a.index.intersection(b.index)
        rows.append(
            {
                "preset": p,
                "arm": arm,
                "reference": ref_leg,
                "leg": leg,
                "runs_compared": len(both),
                "runs_differing": int((a.loc[both] != b.loc[both]).sum()),
                "only_in_reference": len(a.index.difference(b.index)),
                "only_in_leg": len(b.index.difference(a.index)),
            }
        )
    return rows


def fmt_pair(r: pd.Series, digits: int = 3) -> str:
    star = "**" if r["resolvable"] else ""
    return f"{star}{r['mean']:+.{digits}f}{star} ± {r['se']:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="launch_cross_corpus_4739.sh's ROOT: one dir per leg")
    ap.add_argument("--baseline", help="text_baseline.py CSV at today's lines (default ROOT/text_baseline.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--legs", help="comma-separated legs to read; default every leg with cells")
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--seeds", help="comma-separated seeds to read (an interim read); default all")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bl_path = Path(args.baseline) if args.baseline else root / "text_baseline.csv"
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    wanted = args.legs.split(",") if args.legs else [leg for leg, _ in LEGS]
    legs = [
        leg
        for leg in wanted
        if any((root / leg / f"{a}_{p}" / "results" / "cells").is_dir() for a in ARMS for p in PRESETS)
    ]
    if not legs:
        print(f"no leg under {root} has cells", file=sys.stderr)
        return 1

    status = 0
    pairs, flags = [], []
    for leg in legs:
        leg_args = [
            "--base",
            str(root / leg),
            "--baseline",
            str(bl_path),
            "--out",
            str(out / leg),
            "--arms",
            ",".join(ARMS),
        ]
        if args.seeds:
            leg_args += ["--seeds", args.seeds]
        if args.no_figures:
            leg_args.append("--no-figures")
        rc = ap4671.main(leg_args)
        status |= rc
        p = pd.read_csv(out / leg / "paired.csv")
        pairs.append(p.assign(leg=leg))
        if (out / leg / "flags.csv").is_file() and (out / leg / "flags.csv").stat().st_size > 1:
            flags.append(pd.read_csv(out / leg / "flags.csv").assign(leg=leg))
    pairs_frame = pd.concat(pairs, ignore_index=True)
    pairs_frame = pairs_frame[pairs_frame["metric"].isin(METRICS) & pairs_frame["at"].isin(AT)]
    pairs_frame.to_csv(out / "legs.csv", index=False, float_format="%.6g")
    flag_frame = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()
    if not flag_frame.empty:
        flag_frame.to_csv(out / "flags.csv", index=False, float_format="%.6g")

    # The trajectories: one read of Span, and every leg checked against the first.
    ref_leg = legs[0]
    ref = leg_frames(root, ref_leg, seeds)
    span = pd.DataFrame(span_rows(ref, args.horizon))
    span.to_csv(out / "span.csv", index=False, float_format="%.6g")
    ident: list[dict] = []
    for leg in legs[1:]:
        ident += identity_rows(ref_leg, ref, leg, leg_frames(root, leg, seeds))
    ident_frame = pd.DataFrame(ident)
    ident_frame.to_csv(out / "identity.csv", index=False)
    differing = int(ident_frame["runs_differing"].sum()) if not ident_frame.empty else 0
    if differing:
        status = 1

    lines = ["# #4739 cross-corpus Find - machine summary", ""]
    lines += [f"Legs read: {', '.join(f'`{leg}` ({LEG_LABEL.get(leg, leg)})' for leg in legs)}.", ""]
    lines += [
        "## nowalk minus ctl, paired on (category, seed); positive = the Hard pick ahead (bold = beyond 2 SE)",
        "",
    ]
    lines += ["| leg | preset | metric | " + " | ".join(AT) + " | runs |", "|---|---|---|" + "---|" * (len(AT) + 1)]
    for leg in legs:
        for p in PRESETS:
            for metric in METRICS:
                g = pairs_frame[
                    (pairs_frame["leg"] == leg) & (pairs_frame["preset"] == p) & (pairs_frame["metric"] == metric)
                ]
                if g.empty:
                    continue
                g = g.set_index("at")
                digits = 1 if metric == "find_n_flagged" else 3
                cells_ = [fmt_pair(g.loc[at], digits) if at in g.index else "" for at in AT]
                n = int(g.loc["mean 1-150", "n"]) if "mean 1-150" in g.index else 0
                lines.append(f"| {leg} | {p} | {metric} | " + " | ".join(cells_) + f" | {n} |")
    if not flag_frame.empty:
        lines += ["", "## Domain-shift flags against each Find corpus, at the snapshot clicks", ""]
        lines += [
            "| leg | preset | arm | click | runs | overlap fires | atypical share | evidence fires | unsupported share |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in flag_frame.itertuples(index=False):
            lines.append(
                f"| {r.leg} | {r.preset} | {r.arm} | {r.flags_at} | {r.runs} | {r.shift_fires:.0%} "
                f"| {r.shift_frac_atypical:.3f} | {r.evidence_fires:.0%} | {r.evidence_frac_unsupported:.3f} |"
            )
    lines += ["", f"## Span and Done (read on `{ref_leg}`; every leg runs these trajectories)", ""]
    lines += [
        "| preset | arm | runs | " + " | ".join(f"Span green by {t}" for t in SPAN_BY) + " | Span median vote "
        "| " + " | ".join(f"Done by {t}" for t in SPAN_BY) + " | Done median vote | Span level at the end (target) |",
        "|---|---|---|" + "---|" * (2 * len(SPAN_BY) + 3),
    ]
    for r in span.itertuples(index=False):
        d = r._asdict()
        lines.append(
            f"| {r.preset} | {r.arm} | {r.runs} | "
            + " | ".join(f"{d[f'span_green_by_{t}']:.0%}" for t in SPAN_BY)
            + f" | {r.span_green_median_vote:.0f} | "
            + " | ".join(f"{d[f'done_by_{t}']:.0%}" for t in SPAN_BY)
            + f" | {r.done_median_vote:.0f} | {r.span_level_last_mean:.1f} ({r.span_target:.0f}) |"
        )
    lines += ["", "## Trajectory identity across legs", ""]
    if ident_frame.empty:
        lines.append("Only one leg read; nothing to compare.")
    else:
        lines.append(
            f"{int(ident_frame['runs_compared'].sum())} runs compared against `{ref_leg}`, {differing} differ"
            + (" -- PREMISE FAILURE: a Find corpus moved a trajectory." if differing else ".")
        )
        lines += [
            "",
            "| preset | arm | leg | compared | differing | only in reference | only in leg |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in ident_frame.itertuples(index=False):
            lines.append(
                f"| {r.preset} | {r.arm} | {r.leg} | {r.runs_compared} | {r.runs_differing} "
                f"| {r.only_in_reference} | {r.only_in_leg} |"
            )
    (out / "REPORT_cross_corpus.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    (out / "provenance.json").write_text(
        json.dumps({"root": "<root>", "legs": legs, "reference_leg": ref_leg, "seeds": seeds}, indent=2) + "\n"
    )
    print("\n".join(lines))

    if not args.no_figures:
        figures(out, legs)
        bl = curves.text_sort_baseline(bl_path)
        grid = pd.concat([f[CELL] for f in ref.values()], ignore_index=True).drop_duplicates()
        main_frame = pd.concat([f.assign(arm=a) for a, f in ref.items()], ignore_index=True)
        denominator = pd.concat([grid.assign(arm=a) for a in ref], ignore_index=True)
        curves.quality_vs_clicks(
            main_frame, out / "figures", arms=list(ref), denominator=denominator, baseline=bl, lower_is_better=False
        )
    if not args.no_viewer:
        import viewer  # noqa: PLC0415

        bl = curves.text_sort_baseline(bl_path)
        grid = pd.concat([f[CELL] for f in ref.values()], ignore_index=True).drop_duplicates()
        main_frame = pd.concat([f.assign(arm=a) for a, f in ref.items()], ignore_index=True)
        denominator = pd.concat([grid.assign(arm=a) for a in ref], ignore_index=True)
        viewer.build_viewer(
            main_frame,
            out / "viewer.html",
            arms=list(ref),
            denominator=denominator,
            baseline=bl,
            title="COCO Better: the atlas walk against the Hard pick",
            subtitle="Binary voting, SigLIP, 144 cells x 5 seeds, the user's pool at 1%; in-corpus read, "
            "the trajectories every #4739 leg shares",
            default_metric="fbeta",
            hide_metrics=("cost",),
            # 4,320 runs bust the repo's 4 MB file cap even at the coarsest per-seed grid (4.6 MB): the averaged
            # view alone (#4651), which the page says.
            runs_budget_mb=0,
        )
    return status


def figures(out: Path, legs: list[str]) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    presets = list(PRESETS)
    # 1. The walk's price on each Find corpus with positives: nowalk - ctl at every vote, beside the in-corpus one.
    scored = [leg for leg in legs if (out / leg / "find_paired_curve.csv").is_file() and leg != "screens_only"]
    if scored:
        fig, axes = plt.subplots(
            len(scored), len(presets), figsize=(5.0 * len(presets), 3.0 * len(scored)), squeeze=False, sharex=True
        )
        for i, leg in enumerate(scored):
            fc = pd.read_csv(out / leg / "find_paired_curve.csv")
            ic = pd.read_csv(out / leg / "paired_curve.csv")
            for j, p in enumerate(presets):
                ax = axes[i][j]
                ax.axhline(0, color="#888", lw=1)
                c = ic[(ic["preset"] == p) & (ic["arm"] == "nowalk")]
                ax.plot(c["t"], c["delta"], color="#888", lw=1.2, ls="--", label="in-corpus (withheld half)")
                c = fc[(fc["preset"] == p) & (fc["arm"] == "nowalk")]
                ax.plot(c["t"], c["delta"], color="#1baf7a", lw=1.6, label="on the Find corpus")
                ax.fill_between(c["t"], c["delta"] - 2 * c["se"], c["delta"] + 2 * c["se"], color="#1baf7a", alpha=0.15)
                ax.set_title(f"{leg}, beta {PRESETS[p]:g}", fontsize=9)
                ax.grid(alpha=0.3)
                if j == 0:
                    ax.set_ylabel("objective: Hard pick - walk")
                if i == len(scored) - 1:
                    ax.set_xlabel("votes")
        axes[0][0].legend(fontsize=7, loc="lower left")
        fig.suptitle(
            "The walk's price on the objective, by Find corpus (paired, ±2 SE; above 0 = the walk costs)", fontsize=10
        )
        fig.tight_layout()
        fig.savefig(figdir / "walk_price_by_corpus.png", dpi=130)
        plt.close(fig)
    # 2. The screens alone: how many each arm returns among them, mean over runs, every vote. (Every run returns at
    # least one at every vote: the line on a corpus with no positive still keeps its top.)
    if "screens_only" in legs and (out / "screens_only" / "find_curves.csv").is_file():
        fc = pd.read_csv(out / "screens_only" / "find_curves.csv")
        fig, axes = plt.subplots(1, len(presets), figsize=(5.0 * len(presets), 3.4), squeeze=False)
        for j, p in enumerate(presets):
            ax = axes[0][j]
            for arm, colour in (("ctl", "#2a78d6"), ("nowalk", "#1baf7a")):
                c = fc[(fc["preset"] == p) & (fc["arm"] == arm)]
                ax.plot(c["t"], c["returned_mean"], color=colour, lw=1.6, label=arm)
            ax.set_title(f"screens alone, beta {PRESETS[p]:g}", fontsize=9)
            ax.set_xlabel("votes")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("screens returned (mean over runs)")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "screens_returned.png", dpi=130)
        plt.close(fig)
    # 3. The flags' operating range: the share of runs each fires in, by Find corpus, at each snapshot click.
    if (out / "flags.csv").is_file():
        ff = pd.read_csv(out / "flags.csv")
        ff = ff[ff["preset"] == "b1"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), squeeze=False)
        palette = {"screens0": "#2a78d6", "o365": "#eb6834", "screens_equal": "#9c5fd0", "screens_only": "#d03c5f"}
        for j, (flag, title) in enumerate(
            (("shift_fires", "Training-domain overlap"), ("evidence_fires", "Evidence coverage"))
        ):
            ax = axes[0][j]
            for leg in legs:
                for arm, ls in (("ctl", "-"), ("nowalk", "--")):
                    c = ff[(ff["leg"] == leg) & (ff["arm"] == arm)].sort_values("flags_at")
                    if not c.empty:
                        ax.plot(
                            c["flags_at"],
                            c[flag],
                            color=palette.get(leg, "#444"),
                            ls=ls,
                            marker="o",
                            ms=3,
                            label=f"{leg} {arm}",
                        )
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{title}: share of runs it fires in (beta 1)", fontsize=9)
            ax.set_xlabel("vote")
            ax.grid(alpha=0.3)
        axes[0][1].legend(fontsize=7, loc="center right")
        fig.tight_layout()
        fig.savefig(figdir / "flags_operating_range.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

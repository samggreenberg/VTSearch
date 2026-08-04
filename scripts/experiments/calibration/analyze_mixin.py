"""Mix-in schedule study (#2841): rank the schedules, then judge the A/B.

Two modes, matching the two phases of the design:

``--mode screen``
    Reads the counterfactual ``schedule`` rows from one production-trajectory
    run.  Every schedule was re-cut on the *same* step of the *same* model, so
    the comparison is exactly paired and needs no modelling of the trajectory.
    Produces the promotion list for phase 2.

``--mode ab --arms a,b,c``
    Reads one full run per schedule and pairs them per **cell**
    ``(dataset, embedder, style, category, seed)``.  Pairing per *step* would be
    wrong here: the blended threshold feeds Autopilot's Hard pick, so two arms
    label different items and their step *t* are not the same state.  This is
    the mode that decides.

Both modes report region voting and binary voting **separately** - #2841 allows
them to want different curves, and pooling would hide exactly that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: The window every headline number is computed over: the app's first trained
#: detector appears at 7 votes, and the production ramp ends at 20.  Below 7 no
#: user sees a threshold at all; above 20 the production schedule has handed
#: over completely, so differences there are pure acquisition feedback (which is
#: real, and reported separately, but is not the blend acting directly).
RAMP_LO, RAMP_HI = 7, 20

#: The incumbent every arm is compared against.
BASELINE = "prod"


def _voting_mode(row: pd.Series) -> str:
    """Region voting vs binary voting - the axis #2841 may split its answer on."""
    return "region" if row["style"] != "whole_image" else "binary"


def load_cells(results: Path) -> pd.DataFrame:
    """Concatenate every ``task_*.csv`` under *results*/cells."""
    files = sorted((results / "cells").glob("task_*.csv"))
    files = [f for f in files if not f.name.endswith("__sweep.csv")]
    if not files:
        raise SystemExit(f"no cell CSVs under {results / 'cells'}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    for col in ("schedule", "gmm_variant"):
        df[col] = df[col].fillna("")
    df["n_votes"] = df["n_good"] + df["n_bad"]
    df["mode"] = df.apply(_voting_mode, axis=1)
    return df


def assert_screen_fidelity(df: pd.DataFrame) -> dict:
    """The ``prod`` counterfactual row must reproduce the live blend exactly.

    This is the study's load-bearing check: every schedule row is produced by
    the same code path as ``prod``'s, so if ``prod`` reproduces the threshold
    the run actually used, the other rows are being computed correctly too.  A
    mismatch means the counterfactual and the live blend have drifted apart and
    nothing downstream can be trusted.
    """
    base = df[(df.schedule == "") & (df.gmm_variant == "")]
    prod = df[df.schedule == BASELINE]
    keys = ["dataset", "embedder", "style", "category", "seed", "t"]
    m = base.merge(prod, on=keys, suffixes=("_b", "_p"))
    if m.empty:
        raise SystemExit("fidelity check found no paired rows - is this a screen run?")
    dt = (m.threshold_b - m.threshold_p).abs().max()
    dc = (m.cost_b - m.cost_p).abs().max()
    if not (dt < 1e-9 and dc < 1e-9):
        raise SystemExit(f"FIDELITY FAILURE: prod variant != live blend (max dthreshold={dt:.3e}, dcost={dc:.3e})")
    return {"paired_rows": int(len(m)), "max_threshold_diff": float(dt), "max_cost_diff": float(dc)}


def cell_means(df: pd.DataFrame, lo: int = RAMP_LO, hi: int = RAMP_HI) -> pd.DataFrame:
    """Collapse each cell's steps in the window to one number per metric.

    Steps within a cell are strongly correlated (same model lineage, same
    split), so they are not independent observations.  Aggregating to the cell
    first makes the paired test's n the number of *cells*, which is the unit
    that actually replicates.
    """
    w = df[(df.n_votes >= lo) & (df.n_votes <= hi)]
    keys = ["mode", "dataset", "embedder", "style", "category", "seed", "schedule"]
    metrics = ["cost", "fnr", "fpr", "regret", "average_precision", "auroc", "degenerate"]
    have = [m for m in metrics if m in w.columns]
    return w.groupby(keys, as_index=False)[have].mean()


def paired_vs_baseline(cells: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Per (mode, schedule): paired deltas against *baseline* over shared cells."""
    from scipy import stats  # noqa: PLC0415

    keys = ["mode", "dataset", "embedder", "style", "category", "seed"]
    base = cells[cells.schedule == baseline].set_index(keys)
    out = []
    for (mode, sched), grp in cells.groupby(["mode", "schedule"]):
        if sched == baseline:
            continue
        g = grp.set_index(keys)
        shared = g.index.intersection(base.index)
        if len(shared) < 3:
            continue
        a, b = g.loc[shared], base.loc[shared]
        d = (a["cost"] - b["cost"]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if len(d) < 3:
            continue
        # Wilcoxon: the per-cell cost deltas are heavy-tailed (a handful of cells
        # swing hugely when a cut goes degenerate), so a rank test is the honest
        # default; the t-test rides along for comparability with #2799.
        try:
            w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            w_p = 1.0
        t_p = float(stats.ttest_rel(a["cost"], b["cost"], nan_policy="omit").pvalue)
        row = {
            "mode": mode,
            "schedule": sched,
            "n_cells": int(len(d)),
            "cost": float(a["cost"].mean()),
            "cost_baseline": float(b["cost"].mean()),
            "d_cost": float(np.mean(d)),
            "pct_improved": float(np.mean(d < 0) * 100),
            "p_wilcoxon": w_p,
            "p_ttest": t_p,
        }
        for metric in ("fnr", "fpr", "regret", "average_precision", "degenerate"):
            if metric in a.columns:
                row[f"d_{metric}"] = float((a[metric] - b[metric]).mean())
        out.append(row)
    return pd.DataFrame(out).sort_values(["mode", "d_cost"]).reset_index(drop=True)


#: Cost weightings the verdict is stress-tested under.  The scored metric is
#: ``wf*FPR + wn*FNR`` at inclusion 0, where both weights are 1 - so a schedule
#: can win simply by cutting lower, trading a lot of FNR for a little FPR.  That
#: is a real win at this operating point, but it would reverse for a user who
#: cares more about false alarms, and #2790 flagged exactly this trap.  Re-scoring
#: the same cells under asymmetric weights separates "genuinely better calibrated"
#: from "merely more permissive".
COST_WEIGHTS: tuple[tuple[str, float, float], ...] = (
    ("fpr x1 (shipped)", 1.0, 1.0),
    ("fpr x2", 2.0, 1.0),
    ("fpr x4", 4.0, 1.0),
    ("fnr x2", 1.0, 2.0),
)


def weight_sensitivity(cells: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Paired cost delta vs *baseline* under each weighting in :data:`COST_WEIGHTS`.

    A schedule whose advantage survives ``fpr x4`` is better calibrated; one
    whose advantage flips is only exploiting the symmetric weights.
    """
    keys = ["mode", "dataset", "embedder", "style", "category", "seed"]
    base = cells[cells.schedule == baseline].set_index(keys)
    out = []
    for (mode, sched), grp in cells.groupby(["mode", "schedule"]):
        if sched == baseline:
            continue
        g = grp.set_index(keys)
        shared = g.index.intersection(base.index)
        if len(shared) < 3:
            continue
        a, b = g.loc[shared], base.loc[shared]
        row = {"mode": mode, "schedule": sched, "n_cells": int(len(shared))}
        for label, wf, wn in COST_WEIGHTS:
            ca = wf * a["fpr"] + wn * a["fnr"]
            cb = wf * b["fpr"] + wn * b["fnr"]
            row[label] = float((ca - cb).mean())
        out.append(row)
    return pd.DataFrame(out).sort_values(["mode", "fpr x1 (shipped)"]).reset_index(drop=True)


def past_ramp_effect(df: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Paired cost delta over steps **past** the production ramp (21+ votes).

    In an A/B run the schedules have all converged by 21 votes on the
    ``prod``-shaped curves, so a surviving difference there cannot be the blend
    acting directly - it is the trajectory, i.e. the blend having steered which
    items Autopilot asked the user to label.  #2799 found that channel carries
    real gain, so it is measured rather than assumed away.
    """
    cells = cell_means(df, lo=21, hi=10_000)
    if cells.empty:
        return pd.DataFrame()
    return paired_vs_baseline(cells, baseline)


def promotion_list(deltas: pd.DataFrame, top_n: int = 3) -> list[str]:
    """The pre-registered phase-2 arm set: top *top_n* per mode + fixed anchors.

    ``pure_xcal`` is deliberately absent - it is safe-thresholds OFF, already
    measured and rejected by #2799, so spending a full arm on it would buy a
    number we have.
    """
    picks: list[str] = []
    for _mode, grp in deltas.groupby("mode"):
        picks.extend(grp.nsmallest(top_n, "d_cost")["schedule"].tolist())
    for anchor in (BASELINE, "pure_gmm"):
        if anchor not in picks:
            picks.append(anchor)
    seen: set[str] = set()
    return [p for p in picks if not (p in seen or seen.add(p))]


def _fmt(deltas: pd.DataFrame, mode: str) -> str:
    g = deltas[deltas["mode"] == mode]
    if g.empty:
        return "_(no cells)_\n"
    cols = ["schedule", "n_cells", "cost", "d_cost", "pct_improved", "p_wilcoxon", "d_fnr", "d_fpr"]
    cols = [c for c in cols if c in g.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in g.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "schedule":
                cells.append(str(v))
            elif c == "n_cells":
                cells.append(f"{int(v)}")
            elif c.startswith("p_"):
                cells.append(f"{v:.2g}")
            else:
                cells.append(f"{v:+.4f}" if c.startswith(("d_", "pct")) else f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _fmt_generic(df: pd.DataFrame, mode: str) -> str:
    g = df[df["mode"] == mode] if "mode" in df.columns else df
    if g.empty:
        return "_(none)_\n"
    cols = [c for c in g.columns if c != "mode"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in g.iterrows():
        cells = [
            str(r[c]) if isinstance(r[c], str) else (f"{int(r[c])}" if c == "n_cells" else f"{r[c]:+.4f}") for c in cols
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: Path, mode: str, deltas: pd.DataFrame, extra: dict) -> None:
    baseline_note = (
        "Every schedule was re-cut on the **same** production trajectory, so rows are exactly "
        "paired within a step. This ranks candidates; it cannot see that another schedule would "
        "have labelled different items."
        if mode == "screen"
        else "Each arm is a **full independent trajectory**, paired per cell. This includes the "
        "blend's effect on which items Autopilot chose to label, which the screen structurally cannot see."
    )
    body = [
        f"# Mix-in schedule study (#2841) - {mode}",
        "",
        baseline_note,
        "",
        f"Window: {RAMP_LO}-{RAMP_HI} votes (app's first trained step -> end of the production ramp). "
        f"Baseline: `{BASELINE}`. Negative `d_cost` beats the incumbent.",
        "",
        "## Region voting",
        "",
        _fmt(deltas, "region"),
        "",
        "## Binary voting",
        "",
        _fmt(deltas, "binary"),
        "",
    ]
    sens = extra.pop("_sensitivity", None)
    if sens is not None and not sens.empty:
        body += [
            "## Cost-weighting sensitivity",
            "",
            "Same cells, re-scored under asymmetric weights. A winner that survives `fpr x4` is "
            "better calibrated; one that flips was only cutting lower.",
            "",
            "### Region voting",
            "",
            _fmt_generic(sens, "region"),
            "",
            "### Binary voting",
            "",
            _fmt_generic(sens, "binary"),
            "",
        ]
    past = extra.pop("_past_ramp", None)
    if past is not None and not past.empty:
        body += [
            "## Past the ramp (21+ votes)",
            "",
            "Where every schedule has converged, so a surviving difference is acquisition "
            "feedback: the blend having steered which items got labelled.",
            "",
            "### Region voting",
            "",
            _fmt(past, "region"),
            "",
            "### Binary voting",
            "",
            _fmt(past, "binary"),
            "",
        ]
    body += [
        "## Provenance",
        "",
        "```json",
        json.dumps(extra, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(body))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("screen", "ab"), default="screen")
    ap.add_argument("--arms", default="", help="ab mode: comma-separated schedule names")
    ap.add_argument("--results", default=None, help="override CALIB_RESULTS")
    args = ap.parse_args(argv)

    results = Path(args.results) if args.results else common.RESULTS
    extra: dict = {"mode": args.mode, "results": str(results), "window": [RAMP_LO, RAMP_HI]}

    if args.mode == "screen":
        df = load_cells(results)
        extra["fidelity"] = assert_screen_fidelity(df)
        extra["n_rows"] = int(len(df))
        cells = cell_means(df[df.schedule != ""])
        deltas = paired_vs_baseline(cells)
        promote = promotion_list(deltas)
        extra["promote_to_ab"] = promote
        sens = weight_sensitivity(cells)
        sens.to_csv(results / "screen_sensitivity.csv", index=False)
        extra["_sensitivity"] = sens
        write_report(results / "REPORT_screen.md", "screen", deltas, extra)
        deltas.to_csv(results / "screen_deltas.csv", index=False)
        (results / "promote.json").write_text(json.dumps(promote, indent=2))
        print("promotion list:", " ".join(promote))
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        if not arms:
            raise SystemExit("--arms is required in ab mode")
        frames = []
        for arm in arms:
            d = load_cells(results / arm)
            # In an A/B run the *live* threshold is the arm; the counterfactual
            # rows are noise here, so keep only the run's own base rows.
            d = d[(d.schedule == "") & (d.gmm_variant == "")].copy()
            d["schedule"] = arm
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        extra["arms"] = arms
        extra["n_rows"] = int(len(df))
        cells = cell_means(df)
        deltas = paired_vs_baseline(cells)
        sens = weight_sensitivity(cells)
        sens.to_csv(results / "ab_sensitivity.csv", index=False)
        extra["_sensitivity"] = sens
        past = past_ramp_effect(df)
        if not past.empty:
            past.to_csv(results / "ab_past_ramp.csv", index=False)
            extra["_past_ramp"] = past
        write_report(results / "REPORT_ab.md", "ab", deltas, extra)
        deltas.to_csv(results / "ab_deltas.csv", index=False)
        print(deltas.to_string(index=False))

    print(f"wrote report under {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

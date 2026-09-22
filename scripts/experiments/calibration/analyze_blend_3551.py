#!/usr/bin/env python3
"""#3551 analyzer: tuned `rare` / `corridor` blend schedules against today's stack.

Reads a screen grid (``launch_blend_3551.sh screen``): one production trajectory
per cell, with every schedule re-cut counterfactually at every step.  Two
questions, never pooled (see ``docs/experiments/2026-09-22-blend-endpoints-3551/PLAN.md``):

* **Q1 fallback** - on steps where the shipped fused cut fell back to the
  schedule blend (``threshold_provenance == "gmm_blend"``), what would each
  schedule have shipped?  ``diluted`` = averaged over every step of the
  trajectory (0 on fused steps); ``conditional`` = over fallback steps only.
* **Q2 replacement** - on every step, the schedule's blend of the raw x-cal cut
  and the GMM midpoint against the fused cut the base row carries.

Every contrast is cell-paired: per-step differences are averaged within a cell
(environment x category x seed x calibration draw) first, and the SE is taken
across cells.  Nothing is reported unless the **fidelity gate** passes: the
shipped fallback schedule's row must equal the base row's threshold on every
fallback step.

    python analyze_blend_3551.py [--results DIR] [--out DIR] [--baseline CSV]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402

#: The fallback schedule production resolves per voting mode.  Read from the
#: registry rather than restated, so a future ship cannot leave this stale.
from vtscore.training.blend_schedules import production_schedule_for  # noqa: E402

SHIPPED = {
    "region": production_schedule_for(region_voting=True),
    "binary": production_schedule_for(region_voting=False),
}

#: Cost weightings: (w_fpr, w_fnr).  1:1 is inclusion 0; the other two are the
#: #2841 robustness check, on both sides of it.
WEIGHTS: dict[str, tuple[float, float]] = {"1:1": (1.0, 1.0), "fpr x4": (4.0, 1.0), "fnr x4": (1.0, 4.0)}

VOTE_BANDS: list[tuple[str, int, int]] = [("1-20", 1, 20), ("21-50", 21, 50), ("51-150", 51, 10**9)]
POS_BANDS: list[tuple[str, int, int]] = [("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16+", 16, 10**9)]

CELL = ["env", "mode", "dataset", "embedder", "category", "seed", "calibration_seed"]
KEEP = [
    "dataset",
    "embedder",
    "category",
    "seed",
    "calibration_seed",
    "t",
    "n_good",
    "n_bad",
    "schedule",
    "threshold",
    "threshold_provenance",
    "shipped_provenance",
    "fold_fallback",
    "cost",
    "fpr",
    "fnr",
    "average_precision",
    "gmm_variant",
    "pool_variant",
    "seed_mode",
]


def family(name: str) -> str:
    """Which family a schedule name belongs to, for the promotion cap."""
    if name.startswith("rare"):
        return "rare"
    if name.startswith("corridor"):
        return "corridor"
    return "reference"


def mode_of(embedder: str) -> str:
    """Region voting iff the learning half of the embedder is a patch embedder."""
    return "region" if "dinov3" in str(embedder) else "binary"


def _keep_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Base rows and schedule rows only; drop the #2799 cut-variant rows at read time."""
    df = df[[c for c in KEEP if c in df.columns]]
    if "gmm_variant" in df.columns:
        df = df[_cells_io._blank(df["gmm_variant"])]
    if "pool_variant" in df.columns:
        pv = df["pool_variant"].fillna("").astype(str).str.strip()
        df = df[pv.isin(_cells_io.BASE_POOL_VARIANTS)]
    return df


def load(results: Path) -> tuple[pd.DataFrame, dict]:
    frame, prov = _cells_io.load_cells(results / "cells", where="blend-3551 screen", per_file=_keep_rows)
    return frame, prov


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, step, schedule): the schedule row joined to its base row.

    Adds ``fallback`` (the base row fell back), and ``c0_<w>`` / ``cs_<w>``, the
    base and schedule cost under each weighting.
    """
    f = frame.copy()
    f["schedule"] = f["schedule"].fillna("").astype(str).str.strip()
    f["calibration_seed"] = f["calibration_seed"].fillna(-1).astype(int)
    f["mode"] = f["embedder"].map(mode_of)
    f["env"] = f["dataset"].astype(str) + " / " + f["mode"]
    key = [*CELL, "t"]
    base = f[f["schedule"] == ""]
    dup = base.duplicated(key)
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate base rows on {key}: two grids in one results dir?")
    base = base[[*key, "n_good", "n_bad", "threshold", "threshold_provenance", "cost", "fpr", "fnr"]].rename(
        columns={"threshold": "thr0", "cost": "cost0", "fpr": "fpr0", "fnr": "fnr0"}
    )
    sched = f[f["schedule"] != ""][[*key, "schedule", "threshold", "shipped_provenance", "fold_fallback", "fpr", "fnr"]]
    sched = sched.rename(columns={"threshold": "thrs", "fpr": "fprs", "fnr": "fnrs"})
    m = sched.merge(base, on=key, how="inner", validate="many_to_one")
    m["fallback"] = m["threshold_provenance"].astype(str) == "gmm_blend"
    for w, (wf, wn) in WEIGHTS.items():
        m[f"c0_{w}"] = wf * m["fpr0"] + wn * m["fnr0"]
        m[f"cs_{w}"] = wf * m["fprs"] + wn * m["fnrs"]
    return m


def fidelity(m: pd.DataFrame) -> dict:
    """The gate: the shipped schedule's row reproduces every fallback step exactly."""
    shipped = m[m["schedule"] == m["mode"].map(SHIPPED)]
    fb = shipped[shipped["fallback"]]
    mismatch = fb[(fb["thrs"] - fb["thr0"]).abs() > 0]
    prov_mismatch = m[m["shipped_provenance"].astype(str) != m["threshold_provenance"].astype(str)]
    cost_check = m.drop_duplicates([*CELL, "t"])
    cost_err = float((cost_check["c0_1:1"] - cost_check["cost0"]).abs().max()) if len(cost_check) else 0.0
    return {
        "fallback_steps_checked": int(len(fb)),
        "threshold_mismatches": int(len(mismatch)),
        "provenance_mismatches": int(len(prov_mismatch)),
        "max_abs_cost_minus_fpr_plus_fnr": cost_err,
        "passed": bool(len(fb) > 0 and len(mismatch) == 0 and len(prov_mismatch) == 0 and cost_err < 1e-5),
    }


def cell_contrasts(m: pd.DataFrame, band: tuple[str, int, int] | None = None, axis: str = "t") -> pd.DataFrame:
    """Per (cell, schedule): Q2, Q1-diluted and Q1-conditional means, per weighting."""
    d = m
    if band is not None:
        _, lo, hi = band
        d = d[(d[axis] >= lo) & (d[axis] <= hi)]
    out = d[[*CELL, "schedule"]].copy()
    cols = []
    for w in WEIGHTS:
        delta = d[f"cs_{w}"] - d[f"c0_{w}"]
        out[f"q2_{w}"] = delta
        out[f"q1d_{w}"] = np.where(d["fallback"], delta, 0.0)
        out[f"q1c_{w}"] = np.where(d["fallback"], delta, np.nan)
        cols += [f"q2_{w}", f"q1d_{w}", f"q1c_{w}"]
    out["fallback"] = d["fallback"].astype(float)
    g = out.groupby([*CELL, "schedule"], sort=False)
    agg = g[cols + ["fallback"]].mean()  # nan-skipping mean: q1c over fallback steps only
    agg["n_steps"] = g.size()
    return agg.reset_index()


def summarise(cells: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Per (env, schedule): mean, SE, n cells and share improved for one contrast column."""
    rows = []
    for (env, mode, sched), d in cells.groupby(["env", "mode", "schedule"], sort=True):
        x = d[stat].dropna().to_numpy(dtype=float)
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append(
            {
                "env": env,
                "mode": mode,
                "schedule": sched,
                "family": family(sched),
                "mean": mean,
                "se": se,
                "n": n,
                "improved": float((x < 0).mean()) if n else float("nan"),
                "resolvable": bool(n > 1 and abs(mean) > 2 * se),
            }
        )
    return pd.DataFrame(rows)


def pooled_within_mode(cells: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Within-mode pooled mean and 95% CI (cells from every environment of the mode)."""
    rows = []
    for (mode, sched), d in cells.groupby(["mode", "schedule"], sort=True):
        x = d[stat].dropna().to_numpy(dtype=float)
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        rows.append({"mode": mode, "schedule": sched, "mean": mean, "se": se, "n": n, "ci_hi": mean + 1.96 * se})
    return pd.DataFrame(rows)


def promote(cells: pd.DataFrame, question: str) -> pd.DataFrame:
    """Apply the pre-registered promotion rule; returns every candidate with its verdict.

    *question* is ``"q2"`` (trajectory mean) or ``"q1c"`` (conditional on fallback).
    Promoted iff, in every environment of the mode, the 1:1 contrast is negative
    and resolvable and neither reweighting is resolvably worse; then at most the
    best two per family (by within-mode mean).
    """
    s = {w: summarise(cells, f"{question}_{w}") for w in WEIGHTS}
    base = s["1:1"].copy()
    rows = []
    for (mode, sched), d in base.groupby(["mode", "schedule"]):
        if sched == SHIPPED[mode]:
            continue
        wins = bool(len(d) and ((d["mean"] < 0) & d["resolvable"]).all())
        robust = True
        for w in ("fpr x4", "fnr x4"):
            dw = s[w][(s[w]["mode"] == mode) & (s[w]["schedule"] == sched)]
            if ((dw["mean"] > 0) & dw["resolvable"]).any():
                robust = False
        rows.append(
            {
                "mode": mode,
                "schedule": sched,
                "family": family(sched),
                "mean_over_envs": float(d["mean"].mean()),
                "worst_env": float(d["mean"].max()),
                "wins_every_env": wins,
                "robust_to_reweighting": robust,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["eligible"] = out["wins_every_env"] & out["robust_to_reweighting"]
    out["promoted"] = False
    for (_mode, _fam), d in out[out["eligible"]].groupby(["mode", "family"]):
        out.loc[d.nsmallest(2, "mean_over_envs").index, "promoted"] = True
    return out.sort_values(["mode", "mean_over_envs"]).reset_index(drop=True)


def census(m: pd.DataFrame) -> pd.DataFrame:
    """How often the shipped path falls back, where, and which class is short."""
    steps = m.drop_duplicates([*CELL, "t"])
    rows = []
    for env, d in steps.groupby("env"):
        fb = d[d["fallback"]]
        row = {
            "env": env,
            "steps": len(d),
            "cells": d[CELL].drop_duplicates().shape[0],
            "fallback_share": float(d["fallback"].mean()),
            "cells_with_fallback": float(d.groupby(CELL)["fallback"].any().mean()),
            "fallback_one_good": float((fb["n_good"] <= 1).mean()) if len(fb) else float("nan"),
            "fallback_one_bad": float((fb["n_bad"] <= 1).mean()) if len(fb) else float("nan"),
            "fallback_past_20": float((fb["t"] > 20).mean()) if len(fb) else float("nan"),
            "cells_whole_run_fallback": float(d.groupby(CELL)["fallback"].all().mean()),
        }
        for name, lo, hi in VOTE_BANDS:
            b = d[(d["t"] >= lo) & (d["t"] <= hi)]
            row[f"share_{name}"] = float(b["fallback"].mean()) if len(b) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def rare_grid(summary: pd.DataFrame) -> pd.DataFrame:
    """The uncapped `rare:lo=L:hi=H` points of a summary, with L and H as columns."""
    pat = re.compile(r"^rare:lo=(\d+(?:\.\d+)?):hi=(\d+(?:\.\d+)?)$")
    s = summary.copy()
    parsed = s["schedule"].str.extract(pat)
    s["lo"] = pd.to_numeric(parsed[0])
    s["hi"] = pd.to_numeric(parsed[1])
    return s.dropna(subset=["lo", "hi"])


def corridor_width(name: str) -> tuple[str, float] | None:
    """``(kind, width)`` of a corridor schedule, or None."""
    if name == "corridor":
        return ("constant", 1.0)
    if name == "corridor_ramp":
        return ("ramped", 1.0)
    m = re.match(r"^(corridor|corridor_ramp):w=([0-9.]+)$", name)
    if not m:
        return None
    return ("constant" if m.group(1) == "corridor" else "ramped", float(m.group(2)))


# --------------------------------------------------------------------------- figures


def figures(
    m: pd.DataFrame, cells: pd.DataFrame, outdir: Path, frame: pd.DataFrame, baseline_csv: Path | None
) -> list[str]:  # noqa: C901
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    envs = sorted(m["env"].unique(), key=lambda e: (e.split(" / ")[1], e))

    # 1. fallback share over clicks
    steps = m.drop_duplicates([*CELL, "t"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for env in envs:
        d = steps[steps["env"] == env].groupby("t")["fallback"].mean()
        ax.plot(d.index, d.values, label=env, lw=1.6, ls="-" if env.endswith("region") else "--")
    ax.set_xlabel("votes")
    ax.set_ylabel("share of cells on the fallback")
    ax.set_title("How often the shipped fused cut falls back to the schedule blend")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fallback_share.png", dpi=130)
    plt.close(fig)
    written.append("fallback_share.png")

    # 2. rare tuning surfaces, per question x env
    for q, label in (("q1c", "Q1 fallback steps"), ("q2", "Q2 replacement")):
        s = rare_grid(summarise(cells, f"{q}_1:1"))
        if s.empty:
            continue
        fig, axes = plt.subplots(1, len(envs), figsize=(3.2 * len(envs), 3.2), squeeze=False)
        vmax = float(np.nanmax(np.abs(s["mean"]))) or 1e-3
        for ax, env in zip(axes[0], envs, strict=False):
            d = s[s["env"] == env]
            los, his = sorted(d["lo"].unique()), sorted(d["hi"].unique())
            grid = np.full((len(los), len(his)), np.nan)
            for _, r in d.iterrows():
                grid[los.index(r["lo"]), his.index(r["hi"])] = r["mean"]
            ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for i, lo in enumerate(los):
                for j, hi in enumerate(his):
                    r = d[(d["lo"] == lo) & (d["hi"] == hi)]
                    if len(r):
                        mark = "*" if bool(r["resolvable"].iloc[0]) else ""
                        ax.text(j, i, f"{r['mean'].iloc[0]:+.3f}{mark}", ha="center", va="center", fontsize=6.5)
            ax.set_xticks(range(len(his)), [f"{h:g}" for h in his])
            ax.set_yticks(range(len(los)), [f"{lo:g}" for lo in los])
            ax.set_xlabel("hi (rarer-class votes)")
            ax.set_ylabel("lo")
            ax.set_title(env, fontsize=8)
        fig.suptitle(f"`rare` endpoints, {label}: Δcost vs shipped (blue = better; * = |Δ| > 2 SE)", fontsize=9)
        fig.tight_layout()
        name = f"rare_grid_{q}.png"
        fig.savefig(outdir / name, dpi=130)
        plt.close(fig)
        written.append(name)

    # 3. corridor width curves
    for q, label in (("q1c", "Q1 fallback steps"), ("q2", "Q2 replacement")):
        s = summarise(cells, f"{q}_1:1")
        parsed = s["schedule"].map(corridor_width)
        s = s[parsed.notna()].copy()
        if s.empty:
            continue
        s["kind"] = [p[0] for p in parsed[parsed.notna()]]
        s["w"] = [p[1] for p in parsed[parsed.notna()]]
        fig, axes = plt.subplots(1, len(envs), figsize=(3.2 * len(envs), 3.0), squeeze=False, sharey=True)
        for ax, env in zip(axes[0], envs, strict=False):
            for kind, mk in (("constant", "o-"), ("ramped", "s--")):
                d = s[(s["env"] == env) & (s["kind"] == kind)].sort_values("w")
                ax.errorbar(d["w"], d["mean"], yerr=2 * d["se"], fmt=mk, ms=3, capsize=2, label=kind)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlabel("corridor width w")
            ax.set_title(env, fontsize=8)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("Δcost vs shipped (±2 SE)")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"Corridor width, {label}", fontsize=9)
        fig.tight_layout()
        name = f"corridor_width_{q}.png"
        fig.savefig(outdir / name, dpi=130)
        plt.close(fig)
        written.append(name)

    # 4. the mandatory quality-over-clicks pair, on the Q2 counterfactual trajectories
    try:
        import curves  # noqa: PLC0415

        q2 = summarise(cells, "q2_1:1")
        best = {}
        for fam in ("rare", "corridor"):
            f = q2[q2["family"] == fam].groupby("schedule")["mean"].mean()
            if len(f):
                best[fam] = f.idxmin()
        show = [s for s in dict.fromkeys([*SHIPPED.values(), *best.values()]) if s]
        base = frame[frame["schedule"].fillna("").astype(str).str.strip() == ""].copy()
        base["arm"] = "shipped (fused)"
        parts = [base]
        for s in show:
            r = frame[frame["schedule"] == s].copy()
            r["arm"] = f"blend {s}"
            parts.append(r)
        d = pd.concat(parts, ignore_index=True)
        # A cell is (seed, calibration draw); fold the draw into the seed key so
        # the per-run figure draws one line per trajectory, not one per seed.
        d["seed"] = d["seed"].astype(int) * 1000 + d["calibration_seed"].fillna(0).astype(int)
        d["dataset"] = d["dataset"].astype(str) + " · " + d["embedder"].map(mode_of)
        arms = ["shipped (fused)", *[f"blend {s}" for s in show]]
        baseline = None
        if baseline_csv and Path(baseline_csv).exists():
            b = curves.text_sort_baseline(baseline_csv)
            draws = sorted(frame["calibration_seed"].dropna().astype(int).unique())
            reps = []
            for dr in draws:
                bb = b.copy()
                bb["seed"] = bb["seed"].astype(int) * 1000 + dr
                bb["dataset"] = bb["dataset"].astype(str) + " · " + bb["embedder"].map(mode_of)
                reps.append(bb)
            baseline = pd.concat(reps, ignore_index=True)
        # Every arm re-cuts the same trajectories, so each arm's denominator is
        # the same cell list; the viewer groups it by arm.
        denominator = d[["arm", "dataset", "embedder", "category", "seed"]].drop_duplicates()
        written += curves.quality_vs_clicks(
            d, outdir, arms=arms, metric="cost", denominator=denominator, baseline=baseline, lower_is_better=True
        )
        import viewer  # noqa: PLC0415

        viewer.build_viewer(
            d,
            outdir.parent / "viewer.html",
            arms=arms,
            denominator=denominator,
            baseline=baseline,
            title="#3551 blend schedules vs the fused cut",
            subtitle="Screen rows: every arm re-cuts the SAME shipped trajectory; blend arms are counterfactual.",
        )
        written.append("../viewer.html")
    except Exception as exc:  # noqa: BLE001 - a figure must not cost the analysis
        print(f"WARNING: curves/viewer failed: {exc!r}")
    return written


# --------------------------------------------------------------------------- report


def _f(x: float, sig: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:+.{max(sig - 1 - int(np.floor(np.log10(abs(x)))), 0)}f}"


def _md(df: pd.DataFrame) -> str:
    """A markdown table without the optional ``tabulate`` dependency (2 sig. digits)."""

    def cell(v: object) -> str:
        if isinstance(v, (float, np.floating)):
            return "n/a" if not np.isfinite(v) else f"{v:.2g}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _table(summary: pd.DataFrame, schedules: list[str], envs: list[str]) -> list[str]:
    lines = ["| schedule | " + " | ".join(envs) + " |", "|---|" + "---|" * len(envs)]
    for s in schedules:
        cells = []
        for env in envs:
            r = summary[(summary["schedule"] == s) & (summary["env"] == env)]
            if r.empty:
                cells.append("")
                continue
            r = r.iloc[0]
            star = "**" if r["resolvable"] else ""
            cells.append(f"{star}{_f(r['mean'])}{star} ± {abs(r['se']):.2g}")
        lines.append(f"| `{s}` | " + " | ".join(cells) + " |")
    return lines


def examples(m: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Literal fallback steps where the candidate and the shipped schedule disagree most."""
    fb = m[m["fallback"]].copy()
    fb["delta"] = fb["cs_1:1"] - fb["c0_1:1"]
    fb = fb[fb["schedule"].isin(["rare:lo=1:hi=8", "corridor:w=0.2", "pure_gmm"])]
    cols = [*CELL, "t", "n_good", "n_bad", "schedule", "thr0", "thrs", "fpr0", "fnr0", "fprs", "fnrs", "delta"]
    return pd.concat([fb.nsmallest(n, "delta")[cols], fb.nlargest(n, "delta")[cols]], ignore_index=True)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(common.RESULTS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out) if args.out else results.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    frame, prov = load(results)
    print(f"cells: {_cells_io.describe_load(prov)}")
    if frame.empty:
        print("nothing to analyse")
        return 1
    m = paired(frame)
    fid = fidelity(m)
    (out / "fidelity.json").write_text(json.dumps(fid, indent=2))
    print(f"fidelity: {fid}")

    cells_all = cell_contrasts(m)
    cells_all.to_csv(out / "cell_contrasts.csv", index=False)
    band_frames = []
    for band in VOTE_BANDS:
        c = cell_contrasts(m, band, "t")
        c["band"] = f"votes {band[0]}"
        band_frames.append(c)
    for band in POS_BANDS:
        c = cell_contrasts(m, band, "n_good")
        c["band"] = f"positives {band[0]}"
        band_frames.append(c)
    bands = pd.concat(band_frames, ignore_index=True)
    bands.to_csv(out / "cell_contrasts_banded.csv", index=False)

    cen = census(m)
    cen.to_csv(out / "census.csv", index=False)
    summaries = {}
    for q in ("q2", "q1d", "q1c"):
        for w in WEIGHTS:
            s = summarise(cells_all, f"{q}_{w}")
            s["weighting"] = w
            summaries[(q, w)] = s
    pd.concat([s.assign(question=q) for (q, _w), s in summaries.items()]).to_csv(out / "summary.csv", index=False)
    pooled = pd.concat(
        [pooled_within_mode(cells_all, f"{q}_1:1").assign(question=q) for q in ("q2", "q1d", "q1c")], ignore_index=True
    )
    pooled.to_csv(out / "pooled_within_mode.csv", index=False)
    promo = {q: promote(cells_all, q) for q in ("q1c", "q2")}
    for q, p in promo.items():
        p.to_csv(out / f"promotion_{q}.csv", index=False)
    ex = examples(m)
    ex.to_csv(out / "examples.csv", index=False)

    figs = figures(m, cells_all, out / "figures", frame, Path(args.baseline) if args.baseline else None)

    envs = sorted(m["env"].unique(), key=lambda e: (e.split(" / ")[1], e))
    lines = ["# #3551 screen — machine summary", ""]
    lines.append(f"Cells read: {_cells_io.describe_load(prov)}")
    lines.append(f"Fidelity gate: {'PASSED' if fid['passed'] else '**FAILED**'} {fid}")
    lines.append("")
    lines.append("## Census")
    lines.append(_md(cen))
    for q, title in (("q1c", "Q1 conditional (fallback steps)"), ("q1d", "Q1 diluted"), ("q2", "Q2 replacement")):
        s = summaries[(q, "1:1")]
        order = s.groupby("schedule")["mean"].mean().sort_values().index.tolist()
        lines += ["", f"## {title}, 1:1 (bold = |Δ| > 2 SE)", ""]
        lines += _table(s, order, envs)
    for q in ("q1c", "q2"):
        lines += ["", f"## Promotion ({q})", "", _md(promo[q]) if len(promo[q]) else "(none)"]
    lines += ["", "## Figures", *[f"- {f}" for f in figs]]
    (out / "REPORT_screen.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'REPORT_screen.md'}")
    if not fid["passed"]:
        print("FIDELITY GATE FAILED - no verdict may be read off this run", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

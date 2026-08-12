"""Stage 2 (safe-threshold study, #2799): aggregate the GMM-variant cells.

Consumes the same ``results/cells/task_*.csv`` files as ``analyze.py`` but from
a run with ``CALIB_SAFE_THRESHOLDS=1``, where every step additionally emits one
row per safe-threshold GMM variant (``gmm_variant`` column; see
``vtscore.eval.voting_iterations._SAFE_GMM_VARIANTS``).  Computes the
pre-registered #2799 deliverables:

* FPR / FNR / cost vs vote count per variant, per arm - the 6-20-vote ramp
  window (and the sub-6 pure-GMM window) are the cells that matter.
* Paired contrasts on the same steps: ``pooled_mid - image_mid`` (the geometry
  bias #2797 removed), ``pooled_cross - pooled_mid`` (the #2798 cut-rule
  change), ``pooled_cross_logit - pooled_cross`` (the logit-space idea), and
  ``pooled_cross - xcal_only`` (does the blend beat raw conformal at all?).
* Threshold diagnostics: mean cut vs the oracle cut, degenerate rate.
* A sanity check that the ``pooled_mid`` variant reproduces the production
  blend (its threshold must equal the base row's).  This was ``pooled_cross``
  while #2801's crossing cut shipped; #2833 reverted production to the midpoint.

Writes ``results/summary.json``, ``results/agg/*.csv``,
``results/figures/*.png``, and a ``results/REPORT.md`` draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import main_frame_files  # noqa: E402

#: Vote-count windows (inclusive) the deliverables aggregate over.  Below 6
#: votes the blend is pure GMM; 6-20 is the ramp; above 20 the blend is pure
#: cross-cal and the #2781 study already covers it.
WINDOWS: dict[str, tuple[int, int]] = {"pure_gmm_2_5": (2, 5), "ramp_6_20": (6, 20)}

#: Pre-registered paired contrasts: name -> (variant_a, variant_b); each is
#: measured as mean(a - b) on identical (arm, category, seed, t) steps.
CONTRASTS: dict[str, tuple[str, str]] = {
    "pooled_fit_vs_image_fit": ("pooled_mid", "image_mid"),  # the #2797 geometry bias
    "crossing_vs_midpoint": ("pooled_cross", "pooled_mid"),  # the #2798 cut rule
    "logit_vs_sigmoid": ("pooled_cross_logit", "pooled_cross"),  # the open logit idea
    "blend_vs_xcal_only": ("pooled_cross", "xcal_only"),  # does the blend help at all?
    "crossing_vs_midpoint_image": ("image_cross", "image_mid"),  # cut rule, single-vector geometry
}


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a plain fixed-width dump."""
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def load_cells(cells_dir: Path) -> pd.DataFrame:
    files = main_frame_files(cells_dir)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {len(df)} rows from {len(files)} cells")
    return df


def production_blend_sanity(df: pd.DataFrame) -> dict:
    """The ``pooled_mid`` variant must reproduce the production blended cut.

    This tracked ``pooled_cross`` while #2801's crossing cut was shipped; #2833
    reverted production to the midpoint, so the mirror is ``pooled_mid`` now.
    """
    keys = ["arm", "category", "seed", "t"]
    base = df[(df["pool_variant"] == "max") & (df["gmm_variant"] == "")].set_index(keys)["threshold"]
    pm = df[df["gmm_variant"] == "pooled_mid"].set_index(keys)["threshold"]
    joined = base.to_frame("base").join(pm.to_frame("pooled_mid"), how="inner")
    if joined.empty:
        return {"n_steps": 0, "max_abs_diff": None, "ok": None}
    diff = (joined["base"] - joined["pooled_mid"]).abs()
    return {
        "n_steps": int(len(joined)),
        "max_abs_diff": float(diff.max()),
        # thresholds are rounded to 6 dp on emit, so equality holds to that grain
        "ok": bool(diff.max() <= 1e-6),
    }


def variant_curves(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Mean FPR/FNR/cost vs vote count, per (arm, variant)."""
    v = df[df["gmm_variant"] != ""]
    g = (
        v.groupby(["arm", "gmm_variant", "n_votes"])
        .agg(
            cost=("cost", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            threshold=("threshold", "mean"),
            gmm_cut=("gmm_cut", "mean"),
            oracle_threshold=("oracle_threshold", "mean"),
            degenerate_rate=("degenerate", "mean"),
            n=("cost", "size"),
        )
        .reset_index()
    )
    g.to_csv(agg_dir / "variant_vs_votes.csv", index=False)
    return g


def window_table(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Per-window mean metrics per (arm, variant) - the headline table."""
    v = df[df["gmm_variant"] != ""]
    out = []
    for wname, (lo, hi) in WINDOWS.items():
        w = v[(v["n_votes"] >= lo) & (v["n_votes"] <= hi)]
        g = (
            w.groupby(["arm", "gmm_variant"])
            .agg(
                cost=("cost", "mean"),
                fpr=("fpr", "mean"),
                fnr=("fnr", "mean"),
                regret=("regret", "mean"),
                degenerate_rate=("degenerate", "mean"),
                n_steps=("cost", "size"),
            )
            .reset_index()
        )
        g.insert(0, "window", wname)
        out.append(g)
    tbl = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    tbl.to_csv(agg_dir / "window_by_variant.csv", index=False)
    return tbl


def _paired_cells(v: pd.DataFrame, a: str, b: str, lo: int, hi: int) -> pd.DataFrame:
    """Per-(arm, category, seed) mean deltas of a-b on identical steps in [lo, hi]."""
    keys = ["arm", "category", "seed", "t"]
    w = v[(v["n_votes"] >= lo) & (v["n_votes"] <= hi)]
    va = w[w["gmm_variant"] == a].set_index(keys)[["cost", "fpr", "fnr", "threshold"]]
    vb = w[w["gmm_variant"] == b].set_index(keys)[["cost", "fpr", "fnr", "threshold"]]
    j = va.join(vb, how="inner", lsuffix="_a", rsuffix="_b")
    if j.empty:
        return pd.DataFrame()
    for col in ("cost", "fpr", "fnr", "threshold"):
        j[f"d_{col}"] = j[f"{col}_a"] - j[f"{col}_b"]
    j = j.reset_index()
    # Collapse the t axis first: one delta per (arm, category, seed) cell, so
    # the Wilcoxon's units are independent cells rather than autocorrelated steps.
    return j.groupby(["arm", "category", "seed"])[["d_cost", "d_fpr", "d_fnr", "d_threshold"]].mean().reset_index()


def contrast_tables(df: pd.DataFrame, agg_dir: Path) -> dict:
    from scipy.stats import wilcoxon  # noqa: PLC0415

    v = df[df["gmm_variant"] != ""]
    out: dict = {}
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        for cname, (a, b) in CONTRASTS.items():
            for arm, sub in v.groupby("arm"):
                cells = _paired_cells(sub, a, b, lo, hi)
                if cells.empty:
                    continue
                entry: dict = {
                    "window": wname,
                    "contrast": cname,
                    "variant_a": a,
                    "variant_b": b,
                    "arm": arm,
                    "n_cells": int(len(cells)),
                }
                for col in ("d_cost", "d_fpr", "d_fnr", "d_threshold"):
                    vals = cells[col].to_numpy()
                    entry[f"mean_{col}"] = float(np.mean(vals))
                    if len(vals) >= 3 and np.any(vals != 0):
                        _stat, p = wilcoxon(vals)
                        entry[f"p_{col}"] = float(p)
                    else:
                        entry[f"p_{col}"] = None
                rows.append(entry)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "contrasts.csv", index=False)
    out["rows"] = rows
    return out


def decision_rules(contrasts: dict) -> dict:
    """Evaluate the pre-registered decisions on the ramp window's patch arm.

    Rules:

    * ``keep_crossing`` - #2801 stays unless the crossing cut is *significantly
      worse* than the midpoint on cost (p < 0.05 with a positive delta); the
      revert is a one-line fallback swap.  (This rule fired ``False`` on the
      #2799 run and #2833 performed the revert; production cuts at the midpoint.)
    * ``adopt_logit`` - the logit-space fit ships only if it beats the sigmoid
      crossing by >= 0.02 mean cost at p < 0.05.
    * ``blend_helps_cold_start`` - if the production blend is *worse* than the
      raw conformal cut on the ramp, the cold-start item in
      inclusion-calibration-bias.md absorbs a GMM-specific note.
    """

    def _find(contrast: str, window: str = "ramp_6_20") -> dict | None:
        matches = [
            r
            for r in contrasts.get("rows", [])
            if r["contrast"] == contrast and r["window"] == window and "whole_image" not in r["arm"]
        ]
        return matches[0] if matches else None

    out: dict = {}
    cross = _find("crossing_vs_midpoint")
    out["keep_crossing"] = (
        None
        if cross is None
        else not (cross["p_d_cost"] is not None and cross["p_d_cost"] < 0.05 and cross["mean_d_cost"] > 0)
    )
    logit = _find("logit_vs_sigmoid")
    out["adopt_logit"] = (
        None
        if logit is None
        else bool(logit["p_d_cost"] is not None and logit["p_d_cost"] < 0.05 and logit["mean_d_cost"] <= -0.02)
    )
    blend = _find("blend_vs_xcal_only")
    out["blend_helps_cold_start"] = None if blend is None else bool(blend["mean_d_cost"] < 0)
    out["evidence"] = {"crossing_vs_midpoint": cross, "logit_vs_sigmoid": logit, "blend_vs_xcal_only": blend}
    return out


def make_figures(curves: pd.DataFrame, fig_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        common.log(f"matplotlib unavailable ({e}); skipping figures")
        return []
    saved = []
    for metric in ("cost", "fpr", "fnr"):
        fig, axes = plt.subplots(1, max(1, curves["arm"].nunique()), figsize=(13, 4.5), sharey=True, squeeze=False)
        for ax, (arm, sub) in zip(axes[0], curves.groupby("arm"), strict=False):
            for variant, vs in sub.groupby("gmm_variant"):
                ax.plot(vs["n_votes"], vs[metric], label=variant, lw=1.2)
            ax.axvspan(6, 20, alpha=0.08, color="gray")
            ax.set_title(arm, fontsize=8)
            ax.set_xlabel("votes")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(f"mean {metric}")
        axes[0][-1].legend(fontsize=6)
        p = fig_dir / f"safe_{metric}_vs_votes.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p.name)
    return saved


def write_report(summary: dict, tables: dict, report_path: Path) -> None:
    lines = ["# Safe-threshold GMM study — auto-generated summary (issue #2799)", ""]
    lines.append(f"Rows: {summary.get('n_rows')} · variant rows: {summary.get('n_variant_rows')}")
    lines.append("")
    lines.append("## Production-blend sanity (pooled_mid == base threshold)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["sanity"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Window means by (arm, variant)")
    lines.append("")
    lines.append(_md(tables["window"]))
    lines.append("")
    lines.append("## Pre-registered contrasts")
    lines.append("")
    lines.append(_md(pd.DataFrame(summary["contrasts"]["rows"])))
    lines.append("")
    lines.append("## Decision rules")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["decisions"], indent=2))
    lines.append("```")
    lines.append("")
    report_path.write_text("\n".join(lines))
    common.log(f"wrote {report_path}")


def main() -> int:
    cells_dir = common.RESULTS / "cells"
    agg_dir = common.RESULTS / "agg"
    fig_dir = common.RESULTS / "figures"
    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_cells(cells_dir)
    if df.empty:
        common.log("no cell CSVs found; nothing to analyze")
        return 1
    if (df["gmm_variant"] != "").sum() == 0:
        common.log("no gmm_variant rows - was the run launched with CALIB_SAFE_THRESHOLDS=1?")
        return 1

    sanity = production_blend_sanity(df)
    curves = variant_curves(df, agg_dir)
    window = window_table(df, agg_dir)
    contrasts = contrast_tables(df, agg_dir)
    decisions = decision_rules(contrasts)
    figs = make_figures(curves, fig_dir)

    summary = {
        "n_rows": int(len(df)),
        "n_variant_rows": int((df["gmm_variant"] != "").sum()),
        "n_cells": int(df[["dataset", "embedder", "category", "seed"]].drop_duplicates().shape[0]),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "sanity": sanity,
        "contrasts": contrasts,
        "decisions": decisions,
        "figures": figs,
    }
    (common.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, {"window": window}, common.RESULTS / "REPORT.md")
    common.log("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

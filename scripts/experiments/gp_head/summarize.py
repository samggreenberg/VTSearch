"""Turn the pilot's CSVs into the study's tables, figures and viewer.

Deterministic from ``stage_a.csv``, ``text_baseline.csv`` and the per-arm cell
CSVs under the results root; writes into the study directory
(``docs/experiments/2026-09-17-gp-head-3954/``):

* ``tables.md`` - every table the report quotes, in markdown;
* ``stage_a_summary.csv`` / ``stage_a_paired.csv`` - the label-curve sweep,
  per trainer x label count, and paired against ``svm_linear``;
* ``budget.csv`` / ``paired.csv`` / ``crossover.csv`` / ``cut_health.csv`` -
  the voting simulation at fixed vote counts, paired against the two app arms,
  the click at which each arm beats the text sort, and how often each arm's
  cut flagged nothing or everything;
* ``figures/`` - the quality-over-clicks pair for ``cost`` and
  ``average_precision`` (``scripts/experiments/calibration/curves.py``, the
  one implementation);
* ``viewer.html`` - the interactive viewer (``calibration/viewer.py``).

Every arm-vs-arm difference is **paired** on ``(dataset, category, seed)`` and
carries its standard error and a Wilcoxon p-value; the report quotes nothing
that is not in a file here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import common

common.setup_env()

import experiment_config as cfg  # noqa: E402

# The shared quality-over-clicks implementation and the cell loader live in
# the calibration study's directory; every study imports them from there.
_CALIB = common.REPO / "scripts" / "experiments" / "calibration"
if str(_CALIB) not in sys.path:
    sys.path.insert(0, str(_CALIB))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUDGETS = (25, 50, 100, 150)
AULC_FROM = 8
KEYS = ["dataset", "category", "seed"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """A GitHub-markdown table; floats at three digits (the report rounds further)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("nan" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _paired(
    df: pd.DataFrame, value: str, arm_col: str, ref: str, keys: list[str], *, arms: list[str] | None = None
) -> pd.DataFrame:
    """Mean paired difference (arm - ref) with SE, n and a Wilcoxon p-value."""
    from scipy.stats import wilcoxon

    wide = df.pivot_table(index=keys, columns=arm_col, values=value, aggfunc="mean")
    rows = []
    for arm in arms or [a for a in wide.columns if a != ref]:
        if arm == ref or arm not in wide.columns or ref not in wide.columns:
            continue
        pair = wide[[arm, ref]].dropna()
        d = (pair[arm] - pair[ref]).to_numpy(dtype=float)
        n = int(d.size)
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        p = float("nan")
        if n > 1 and np.any(d != 0):
            try:
                p = float(wilcoxon(d).pvalue)
            except ValueError:
                p = float("nan")
        rows.append(
            {
                "metric": value,
                "arm": arm,
                "ref": ref,
                "n": n,
                "ref_mean": float(pair[ref].mean()) if n else float("nan"),
                "arm_mean": float(pair[arm].mean()) if n else float("nan"),
                "delta": float(d.mean()) if n else float("nan"),
                "se": se,
                "wilcoxon_p": p,
            }
        )
    return pd.DataFrame(rows)


def _at_budget(frame: pd.DataFrame, t: int, metrics: list[str]) -> pd.DataFrame:
    """Each cell's last row at or before vote *t* (forward-filled), one row per cell."""
    sub = frame[frame["t"] <= t].sort_values("t")
    last = sub.groupby(["arm", *KEYS], as_index=False).tail(1)
    return last[["arm", *KEYS, "t", *metrics]]


def _aulc(frame: pd.DataFrame, metric: str, t_from: int, t_to: int) -> pd.DataFrame:
    """Area under the metric-vs-clicks curve per cell, forward-filled on the integer grid."""
    rows = []
    grid = np.arange(t_from, t_to + 1)
    for (arm, ds, cat, seed), g in frame.groupby(["arm", *KEYS]):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        v = g[metric].to_numpy(dtype=float)
        idx = np.searchsorted(t, grid, side="right") - 1
        ok = idx >= 0
        if not ok.any():
            continue
        vals = v[np.clip(idx, 0, len(v) - 1)]
        rows.append(
            {"arm": arm, "dataset": ds, "category": cat, "seed": seed, f"aulc_{metric}": float(vals[ok].mean())}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage A
# ---------------------------------------------------------------------------


def stage_a_tables(stage_a: pd.DataFrame, out: Path) -> list[str]:
    md: list[str] = []
    metrics = ["auroc", "average_precision", "best_f1", "f1_at_xcal", "brier", "ece", "std_mean", "train_seconds"]
    metrics = [m for m in metrics if m in stage_a.columns]
    summ = stage_a.groupby(["trainer", "n_labels"], as_index=False)[metrics].mean()
    summ["n_cells"] = stage_a.groupby(["trainer", "n_labels"]).size().to_numpy()
    summ.to_csv(out / "stage_a_summary.csv", index=False)
    md.append("### Stage A - label curve, mean over categories x seeds\n")
    md.append(_md(summ))

    ref = "svm_linear"
    pairs = []
    for m in ["auroc", "average_precision", "f1_at_xcal", "brier", "ece"]:
        if m not in stage_a.columns:
            continue
        for n_labels, g in stage_a.groupby("n_labels"):
            p = _paired(g, m, "trainer", ref, ["dataset", "category", "seed"])
            p.insert(0, "n_labels", n_labels)
            pairs.append(p)
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    paired.to_csv(out / "stage_a_paired.csv", index=False)
    md.append(f"\n### Stage A - paired against `{ref}` (arm - ref; negative Brier/ECE = better)\n")
    md.append(_md(paired))
    return md


# ---------------------------------------------------------------------------
# stage B
# ---------------------------------------------------------------------------


def load_main(results: Path, arms: list[str]) -> tuple[pd.DataFrame, dict]:
    import _cells_io

    parts, prov = [], {}
    for arm in arms:
        arm_dir = results / arm
        if not (arm_dir / "cells").exists():
            prov[arm] = {"missing": True}
            continue
        df, p = _cells_io.load_arm(arm_dir)
        prov[arm] = p
        if not df.empty:
            df = df.copy()
            df["arm"] = arm
            parts.append(df)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), prov


def stage_b_tables(main: pd.DataFrame, cells: pd.DataFrame, out: Path, baseline: pd.DataFrame | None) -> list[str]:
    md: list[str] = []
    arms = [a for a in cfg.STAGE_B_ARMS if a in set(main["arm"])]
    metrics = ["cost", "fpr", "fnr", "average_precision", "auroc"]

    # --- budget table ---
    parts = []
    for t in BUDGETS:
        at = _at_budget(main, t, metrics)
        at["budget"] = t
        parts.append(at)
    at_all = pd.concat(parts, ignore_index=True)
    aulc = _aulc(main, "cost", AULC_FROM, cfg.MAX_STEPS)
    budget = at_all.groupby(["arm", "budget"])[metrics].mean().unstack("budget")
    budget.columns = [f"{m}@{t}" for m, t in budget.columns]
    budget = budget.reset_index()
    budget = budget.merge(aulc.groupby("arm", as_index=False)["aulc_cost"].mean(), on="arm", how="left")
    budget["n_cells"] = budget["arm"].map(main.groupby("arm").apply(lambda g: g[KEYS].drop_duplicates().shape[0]))
    budget = budget.set_index("arm").loc[arms].reset_index()
    keep = [
        "arm",
        "n_cells",
        *[f"cost@{t}" for t in BUDGETS],
        f"fnr@{BUDGETS[1]}",
        f"fnr@{BUDGETS[-1]}",
        f"average_precision@{BUDGETS[1]}",
        f"average_precision@{BUDGETS[-1]}",
        "aulc_cost",
    ]
    budget = budget[[c for c in keep if c in budget.columns]]
    budget.to_csv(out / "budget.csv", index=False)
    md.append("### Stage B - Autopilot voting, mean over cells (categories x seeds)\n")
    md.append(_md(budget))

    # --- paired ---
    pairs = []
    for ref in ("app_xcal", "app"):
        if ref not in arms:
            continue
        for t in BUDGETS:
            at = _at_budget(main, t, metrics)
            for m in ("cost", "fnr", "average_precision"):
                p = _paired(at, m, "arm", ref, KEYS, arms=arms)
                p.insert(0, "budget", t)
                pairs.append(p)
        p = _paired(aulc, "aulc_cost", "arm", ref, KEYS, arms=arms)
        p.insert(0, "budget", "AULC")
        pairs.append(p)
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    paired.to_csv(out / "paired.csv", index=False)
    for ref in ("app_xcal", "app"):
        sub = paired[paired["ref"] == ref] if not paired.empty else paired
        if sub.empty:
            continue
        md.append(
            f"\n### Stage B - paired against `{ref}` (arm - ref; negative cost/FNR = better, positive AP = better)\n"
        )
        md.append(_md(sub))

    # --- cut health ---
    health_rows = []
    for arm, g in main.groupby("arm"):
        n = len(g)
        none = int((g["n_flagged"] == 0).sum()) if "n_flagged" in g.columns else 0
        n_test = (g["n_test_pos"] + g["n_test_neg"]) if {"n_test_pos", "n_test_neg"} <= set(g.columns) else None
        all_ = int((g["n_flagged"] >= n_test).sum()) if n_test is not None else 0
        health_rows.append(
            {
                "arm": arm,
                "steps": n,
                "flagged_nothing": none / n if n else float("nan"),
                "flagged_everything": all_ / n if n else float("nan"),
                "mean_threshold": float(g["threshold"].mean()) if "threshold" in g.columns else float("nan"),
                "mean_train_seconds": float(g["train_seconds"].mean())
                if "train_seconds" in g.columns
                else float("nan"),
                "mean_xcal_seconds": float(g["xcal_seconds"].mean()) if "xcal_seconds" in g.columns else float("nan"),
            }
        )
    health = pd.DataFrame(health_rows).set_index("arm").loc[arms].reset_index()
    health.to_csv(out / "cut_health.csv", index=False)
    md.append(
        "\n### Stage B - cut health and cost per step (fraction of steps whose cut flagged nothing / everything)\n"
    )
    md.append(_md(health, "{:.4f}"))

    # --- figures + crossover ---
    import curves

    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric, lower in (("cost", True), ("average_precision", False)):
        written += curves.quality_vs_clicks(
            main, figdir, arms=arms, metric=metric, denominator=cells, baseline=baseline, lower_is_better=lower
        )
    md.append("\n### Figures\n")
    md.append("\n".join(f"- `figures/{Path(w).name}`" for w in written))
    curve_csv = figdir / "cost_vs_clicks.csv"
    if curve_csv.exists():
        x = curves.crossover(pd.read_csv(curve_csv), lower_is_better=True)
        x.to_csv(out / "crossover.csv", index=False)
        md.append("\n### Crossover - first click at which the arm's cost beats the zero-click text sort\n")
        md.append(_md(x))
    return md


def build_viewer(main: pd.DataFrame, cells: pd.DataFrame, out: Path, baseline: pd.DataFrame | None) -> None:
    import viewer

    arms = [a for a in cfg.STAGE_B_ARMS if a in set(main["arm"])]
    viewer.build_viewer(
        main,
        out / "viewer.html",
        arms=arms,
        denominator=cells,
        baseline=baseline,
        skyline=None,
        title="GP head vs the shipped linear SVM (#3954)",
        subtitle=f"{', '.join(cfg.DATASETS)} / {cfg.EMBEDDER}; Autopilot voting to {cfg.MAX_STEPS} clicks",
        build={"results": str(Path(common.RESULTS).resolve()), "arms": ",".join(arms)},
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(common.RESULTS))
    ap.add_argument("--out", default=str(common.STUDY))
    args = ap.parse_args(argv)
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prepare = json.loads((results / "prepare_info.json").read_text())
    md: list[str] = ["<!-- generated by scripts/experiments/gp_head/summarize.py; do not edit by hand -->\n"]
    md.append("## Grid\n")
    md.append("```json\n" + json.dumps(prepare.get("config", {}), indent=2) + "\n```\n")
    for ds, info in prepare["datasets"].items():
        sel = info["selected_categories"]
        md.append(
            f"- `{ds}`: {info['n_medias']} medias; categories "
            + ", ".join(f"`{c}` ({info['category_counts'][c]})" for c in sel)
        )

    stage_a_csv = results / "stage_a.csv"
    if stage_a_csv.exists():
        stage_a = pd.read_csv(stage_a_csv)
        md.append("")
        md += stage_a_tables(stage_a, out)

    import curves

    baseline_csv = results / "text_baseline.csv"
    baseline = curves.text_sort_baseline(baseline_csv) if baseline_csv.exists() else None

    main_df, prov = load_main(results, cfg.STAGE_B_ARMS)
    (out / "load_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if not main_df.empty:
        cells = pd.DataFrame(
            [
                {"arm": arm, **cell}
                for arm in cfg.STAGE_B_ARMS
                for cell in cfg.cells({ds: i["selected_categories"] for ds, i in prepare["datasets"].items()})
            ]
        )
        md.append("")
        md += stage_b_tables(main_df, cells, out, baseline)
        build_viewer(main_df, cells, out, baseline)
        md.append("\n### Cells loaded per arm\n")
        md.append(
            _md(
                pd.DataFrame(
                    [
                        {
                            "arm": a,
                            "n_read": p.get("n_read", 0),
                            "no_positive_found": len(p.get("no_positive_found", []) or []),
                            "unreadable": len(p.get("unreadable", []) or []),
                        }
                        for a, p in prov.items()
                    ]
                ),
                "{:.0f}",
            )
        )

    (out / "tables.md").write_text("\n".join(md) + "\n")
    common.log(f"wrote {out / 'tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

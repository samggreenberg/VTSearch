"""Stage 2: aggregate the calibration cells into figures, a summary, and a report.

Concatenates ``results/cells/task_*.csv`` (+ the ``*__sweep.csv`` side files) and
computes the pre-registered #2781 deliverables:

* Regret vs t per arm (and its rule-inefficiency / calibration-shift split).
* Trained-vs-oracle cost at the final step, per arm.
* The tree verdict: does ``max_patch_pca_hac`` tie/beat ``max_patch`` at the
  **oracle** while carrying larger regret (paired Wilcoxon over (category, seed))?
* Remedial closure: do ``topk`` / ``pnorm`` close the trained-cost gap to
  ``max_patch``?
* Degenerate-threshold incidence vs vote count, with a provenance histogram
  (the #2781 runaway-threshold bug).
* Inclusion-budget compliance: measured FNR vs ``alpha(k)`` for the grouped
  (patch) vs ungrouped (whole-image) calibration paths.

Writes ``results/summary.json``, ``results/agg/*.csv``, ``results/figures/*.png``
(when matplotlib is available), and a ``results/REPORT.md`` draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import describe_load, load_cells, side_frame_files  # noqa: E402

import experiment_config as cfg  # noqa: E402

FINAL_T = cfg.MAX_STEPS


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a plain fixed-width dump."""
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def _arm(row) -> str:
    base = f"{row['dataset']}/{row['embedder']}/{row['style']}"
    return base if row["pool_variant"] == "max" else f"{base}::{row['pool_variant']}"


def load_all(cells_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main, prov = load_cells(cells_dir, where="analyze.py")
    sweep_files = side_frame_files(cells_dir, "__sweep")
    sweep = pd.concat([pd.read_csv(p) for p in sweep_files], ignore_index=True) if sweep_files else pd.DataFrame()
    if not main.empty:
        main["arm"] = main.apply(_arm, axis=1)
    common.log(f"loaded {describe_load(prov)}, {len(sweep)} sweep rows")
    return main, sweep


def _near_final(df: pd.DataFrame) -> pd.DataFrame:
    """Rows at the last step each (arm, category, seed) trajectory reached (<= FINAL_T)."""
    keys = ["dataset", "embedder", "style", "pool_variant", "category", "seed"]
    capped = df[df["t"] <= FINAL_T]
    idx = capped.groupby(keys)["t"].idxmax()
    return df.loc[idx].copy()


def regret_curves(main: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    g = (
        main.groupby(["arm", "t"])
        .agg(
            regret=("regret", "mean"),
            cost=("cost", "mean"),
            oracle_cost=("oracle_cost", "mean"),
            rule_inefficiency=("rule_inefficiency", "mean"),
            calibration_shift=("calibration_shift", "mean"),
            n=("regret", "size"),
        )
        .reset_index()
    )
    g.to_csv(agg_dir / "regret_vs_t.csv", index=False)
    return g


def final_cost_table(main: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    fin = _near_final(main)
    tbl = (
        fin.groupby("arm")
        .agg(
            n_cells=("cost", "size"),
            trained_cost=("cost", "mean"),
            oracle_cost=("oracle_cost", "mean"),
            regret=("regret", "mean"),
            rule_inefficiency=("rule_inefficiency", "mean"),
            calibration_shift=("calibration_shift", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            auroc=("auroc", "mean"),
            average_precision=("average_precision", "mean"),
            n_pool_rows=("n_pool_rows", "mean"),
        )
        .reset_index()
        .sort_values("arm")
    )
    tbl.to_csv(agg_dir / "final_cost_by_arm.csv", index=False)
    return tbl


def _paired(fin: pd.DataFrame, arm_a: str, arm_b: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Values of *col* for two arms aligned on (category, seed)."""
    a = fin[fin["arm"] == arm_a].set_index(["category", "seed"])[col]
    b = fin[fin["arm"] == arm_b].set_index(["category", "seed"])[col]
    common_idx = a.index.intersection(b.index)
    return a.loc[common_idx].to_numpy(), b.loc[common_idx].to_numpy()


def tree_verdict(main: pd.DataFrame) -> dict:
    """Compare the raw-patch tree arm to plain max_patch at trained + oracle cost."""
    fin = _near_final(main)
    arms = set(fin["arm"])
    # Identify the VG dinov3 arms.
    tree = next((a for a in arms if a.endswith("/dinov3_patch/max_patch_pca_hac")), None)
    flat = next((a for a in arms if a.endswith("/dinov3_patch/max_patch")), None)
    out: dict = {"tree_arm": tree, "flat_arm": flat}
    if not tree or not flat:
        out["note"] = "tree and/or max_patch arm missing"
        return out
    from scipy.stats import wilcoxon  # noqa: PLC0415

    for label, col in [("trained_cost", "cost"), ("oracle_cost", "oracle_cost"), ("regret", "regret")]:
        ta, fa = _paired(fin, tree, flat, col)
        if len(ta) >= 3 and np.any(ta - fa != 0):
            _stat, p = wilcoxon(ta, fa)
            out[label] = {
                "tree_mean": float(np.mean(ta)),
                "flat_mean": float(np.mean(fa)),
                "delta_tree_minus_flat": float(np.mean(ta - fa)),
                "wilcoxon_p": float(p),
                "n_pairs": int(len(ta)),
            }
        else:
            out[label] = {"tree_mean": float(np.mean(ta)) if len(ta) else None, "n_pairs": int(len(ta))}
    # Remedial closure: fraction of the tree-vs-flat trained-cost gap each variant closes.
    gap = out.get("trained_cost", {}).get("delta_tree_minus_flat")
    out["remedial"] = {}
    for variant in cfg.REPOOL_VARIANTS:
        varm = f"{tree}::{variant}"
        if varm in arms and gap:
            va, fa = _paired(fin, varm, flat, "cost")
            if len(va):
                var_gap = float(np.mean(va - fa))
                out["remedial"][variant] = {
                    "trained_cost_mean": float(np.mean(va)),
                    "gap_to_flat": var_gap,
                    "fraction_of_tree_gap_closed": float((gap - var_gap) / gap) if gap else None,
                    "beats_flat_trained": bool(np.mean(va) < np.mean(fa)),
                }
    return out


def degenerate_incidence(main: pd.DataFrame, agg_dir: Path) -> dict:
    base = main[main["pool_variant"] == "max"]
    prov = base["threshold_provenance"].value_counts().to_dict()
    by_t = base.groupby("t")["degenerate"].mean().reset_index(name="degenerate_rate")
    by_t.to_csv(agg_dir / "degenerate_vs_t.csv", index=False)
    deg = base[base["degenerate"] == 1]
    # Self-heal: for each degenerate (cell, t), is the next step non-degenerate?
    heal_num = heal_den = 0
    dmap = {(r.dataset, r.embedder, r.style, r.category, r.seed, r.t): r.degenerate for r in base.itertuples()}
    for r in deg.itertuples():
        nxt = dmap.get((r.dataset, r.embedder, r.style, r.category, r.seed, r.t + 1))
        if nxt is not None:
            heal_den += 1
            heal_num += int(nxt == 0)
    return {
        "provenance_histogram": {str(k): int(v) for k, v in prov.items()},
        "n_degenerate_steps": int(len(deg)),
        "n_base_steps": int(len(base)),
        "degenerate_rate_overall": float(len(deg) / max(1, len(base))),
        "degenerate_vote_counts": deg["n_good"].add(deg["n_bad"]).describe().to_dict() if len(deg) else {},
        "self_heal_rate": float(heal_num / heal_den) if heal_den else None,
        "provenance_of_degenerate": deg["threshold_provenance"].value_counts().to_dict() if len(deg) else {},
    }


def budget_compliance(sweep: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    if sweep.empty:
        return pd.DataFrame()
    sweep = sweep.copy()
    sweep["grouped"] = sweep["style"].isin(cfg.PATCH_STYLES)
    late = sweep[sweep["t"] >= 100]
    tbl = (
        late.groupby(["grouped", "inclusion_k"])
        .agg(
            alpha=("alpha", "mean"),
            measured_fnr=("sweep_fnr", "mean"),
            excess_fnr=("excess_fnr", "mean"),
            n=("sweep_fnr", "size"),
        )
        .reset_index()
    )
    tbl.to_csv(agg_dir / "budget_compliance.csv", index=False)
    return tbl


def make_figures(main: pd.DataFrame, regret_g: pd.DataFrame, fig_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        common.log(f"matplotlib unavailable ({e}); skipping figures")
        return []
    saved = []
    # Regret vs t per arm
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, sub in regret_g.groupby("arm"):
        ax.plot(sub["t"], sub["regret"], label=arm, lw=1.3)
    ax.set_xlabel("votes (t)")
    ax.set_ylabel("mean regret (cost - oracle_cost)")
    ax.set_title("Calibration regret vs votes, per arm")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    p = fig_dir / "regret_vs_t.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)

    # FPR tail vs n_pool_rows (final step, base pooling)
    fin = _near_final(main[main["pool_variant"] == "max"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(fin["n_pool_rows"], fin["fpr"], s=8, alpha=0.4)
    ax.set_xlabel("n_pool_rows (nodes max-pooled per image)")
    ax.set_ylabel("test FPR at trained threshold (final step)")
    ax.set_title("FPR tail vs pool size")
    ax.grid(alpha=0.3)
    p = fig_dir / "fpr_tail_vs_npool.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p.name)
    return saved


def write_report(summary: dict, tables: dict, report_path: Path) -> None:
    lines = ["# Calibration study — auto-generated summary (issue #2781)", ""]
    lines.append(
        f"Cells: {summary.get('n_cells')} · main rows: {summary.get('n_main_rows')} · "
        f"sweep rows: {summary.get('n_sweep_rows')}"
    )
    lines.append("")
    lines.append("## Final-step cost by arm")
    lines.append("")
    lines.append(_md(tables["final_cost"]))
    lines.append("")
    lines.append("## Tree verdict (max_patch_pca_hac vs max_patch)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["tree_verdict"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Runaway-threshold bug (degenerate incidence)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["degenerate"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Inclusion-budget compliance (t >= 100)")
    lines.append("")
    if not tables["budget"].empty:
        lines.append(_md(tables["budget"]))
    lines.append("")
    report_path.write_text("\n".join(lines))
    common.log(f"wrote {report_path}")


def main() -> int:
    cells_dir = common.RESULTS / "cells"
    agg_dir = common.RESULTS / "agg"
    fig_dir = common.RESULTS / "figures"
    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    main, sweep = load_all(cells_dir)
    if main.empty:
        common.log("no cell CSVs found; nothing to analyze")
        return 1

    regret_g = regret_curves(main, agg_dir)
    final_tbl = final_cost_table(main, agg_dir)
    verdict = tree_verdict(main)
    degen = degenerate_incidence(main, agg_dir)
    budget = budget_compliance(sweep, agg_dir)
    figs = make_figures(main, regret_g, fig_dir)

    summary = {
        "n_cells": int(main[["dataset", "embedder", "category", "seed"]].drop_duplicates().shape[0]),
        "n_main_rows": int(len(main)),
        "n_sweep_rows": int(len(sweep)),
        "final_t": FINAL_T,
        "tree_verdict": verdict,
        "degenerate": degen,
        "figures": figs,
    }
    (common.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, {"final_cost": final_tbl, "budget": budget}, common.RESULTS / "REPORT.md")
    common.log("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

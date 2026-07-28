"""Generate REPORT.md (+ figures) from the completed Max-Patch cell CSVs.

Deterministic given the CSVs: concatenates ``results/cells/task_*.csv``,
aggregates per arm (``embedder/style``), and writes tables for the three
headline questions - ranking quality (AP), operating cost at the trained
threshold (the inclusion-weighted ``fpr_weight*FPR + fnr_weight*FNR`` the live
tool optimises), and runtime (embed / train / score wall clocks).  Per-dataset
break-downs and vote-budget curves show *where* each style wins, not just
whether it wins on average.
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import pandas as pd  # noqa: E402

#: Vote budgets at which the curves are tabulated.
BUDGETS = [10, 20, 50, 100, 150]

#: Row keys identifying one voting trajectory.
TRAJ_KEYS = ["dataset", "category", "seed", "embedder", "style"]


def _load_cells() -> pd.DataFrame:
    files = sorted((common.RESULTS / "cells").glob("task_*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("no cell CSVs found under results/cells - run run_cells.py first")
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df["embedder"] + "/" + df["style"]
    return df


def _final_steps(df: pd.DataFrame) -> pd.DataFrame:
    """The last (highest-t) row of every trajectory."""
    return df.sort_values("t").groupby(TRAJ_KEYS, as_index=False).tail(1)


def _fmt(x: float, digits: int = 3) -> str:
    return "-" if pd.isna(x) else f"{x:.{digits}f}"


def _arm_table(final: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = final.groupby(by).agg(
        n=("cost", "size"),
        AP=("average_precision", "mean"),
        cost=("cost", "mean"),
        fpr=("fpr", "mean"),
        fnr=("fnr", "mean"),
        auroc=("auroc", "mean"),
        train_s=("train_seconds", "mean"),
        score_s=("test_score_seconds", "mean"),
    )
    return agg.reset_index()


def _budget_table(df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for b in budgets:
        at_b = df[df["t"] == b]
        if at_b.empty:
            continue
        for arm, sub in at_b.groupby("arm"):
            rows.append(
                {
                    "t": b,
                    "arm": arm,
                    "n": len(sub),
                    "AP": sub["average_precision"].mean(),
                    "cost": sub["cost"].mean(),
                }
            )
    return pd.DataFrame(rows)


def _write_figures(df: pd.DataFrame, outdir: Path) -> list[str]:
    """Cost-vs-t and AP-vs-t line plots per dataset; returns relative paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for metric, label in (("cost", "cost @ trained threshold"), ("average_precision", "average precision")):
        for ds, ds_df in df.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for arm, sub in ds_df.groupby("arm"):
                curve = sub.groupby("t")[metric].mean()
                ax.plot(curve.index, curve.values, label=str(arm), linewidth=1.5)
            ax.set_xlabel("votes (t)")
            ax.set_ylabel(label)
            ax.set_title(f"{ds}: {label} vs vote budget")
            ax.legend(fontsize=7)
            fig.tight_layout()
            name = f"fig_{metric}_{ds}.png"
            fig.savefig(outdir / name, dpi=130)
            plt.close(fig)
            written.append(f"figures/{name}")
    return written


def _markdown_table(frame: pd.DataFrame, float_cols: dict[str, int]) -> list[str]:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(_fmt(v, float_cols[c]) if c in float_cols else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    df = _load_cells()
    final = _final_steps(df)

    lines: list[str] = []
    lines.append("# Max-Patch experiment - report")
    lines.append("")
    lines.append(
        "Arms are `embedder/style`: `max_hac` is the production HAC-tree patch pipeline, "
        "`max_patch` is the tree-free raw-patch max-pool alternative, `whole_image` is the "
        "single-vector pipeline (the CLS-only control on the DINO embedders; the standard "
        "baseline on SigLIP).  Metrics come from the Autopilot voting simulation "
        "(`vtscore.eval.voting_iterations`): each trajectory votes one item per step, retrains, "
        "and is scored on a held-out split.  `cost` is the inclusion-weighted "
        "`FPR + FNR` at the cross-calibrated (trained) threshold; `AP` is threshold-free "
        "ranking quality.  Timing columns are per-retrain wall clocks."
    )
    lines.append("")

    n_traj = len(final)
    lines.append(f"Trajectories: **{n_traj}** (dataset x category x seed x arm).")
    lines.append("")

    float_cols = {"AP": 3, "cost": 3, "fpr": 3, "fnr": 3, "auroc": 3, "train_s": 2, "score_s": 3}

    lines.append("## Overall (final vote budget)")
    lines.append("")
    lines.extend(_markdown_table(_arm_table(final, ["arm"]), float_cols))
    lines.append("")

    lines.append("## Per dataset (final vote budget)")
    lines.append("")
    lines.extend(_markdown_table(_arm_table(final, ["dataset", "arm"]), float_cols))
    lines.append("")

    lines.append("## Vote-budget curves")
    lines.append("")
    lines.append("Mean over the trajectories that reached each budget.")
    lines.append("")
    budget = _budget_table(df, BUDGETS)
    if len(budget):
        lines.extend(_markdown_table(budget, {"AP": 3, "cost": 3}))
    lines.append("")

    figures = _write_figures(df, common.RESULTS / "figures")
    if figures:
        lines.append("## Figures")
        lines.append("")
        for f in figures:
            lines.append(f"![{f}]({f})")
        lines.append("")

    # --- embed-time break-down from prepare_info ---
    info_path = common.RESULTS / "prepare_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        lines.append("## Embed runtime (prepare stage)")
        lines.append("")
        lines.append("| dataset | embedder | images | load+embed s | s/image |")
        lines.append("|---|---|---|---|---|")
        for ds, per_emb in sorted(info.get("datasets", {}).items()):
            for emb, entry in sorted(per_emb.items()):
                lines.append(
                    f"| {ds} | {emb} | {entry.get('n_medias')} | {entry.get('load_seconds')} | "
                    f"{entry.get('embed_seconds_per_image')} |"
                )
        lines.append("")
        lines.append(
            "Note: `load+embed s` includes download/IO on the first run; re-runs from a warm "
            "demo cache understate true embed time.  Treat the *relative* per-embedder numbers "
            "from a cold run as the meaningful comparison."
        )
        lines.append("")

    # --- quick verdict scaffold ---
    lines.append("## Reading the results")
    lines.append("")
    lines.append(
        "- **MaxPatch vs MaxHAC**: compare `*/max_patch` and `*/max_hac` rows per dataset; the "
        "vote-budget curves show whether one dominates early (few votes) or only asymptotically."
    )
    lines.append(
        "- **Does patch machinery pay at all?** Compare both patch styles against the same "
        "embedder's `whole_image` control and against `siglip/whole_image`."
    )
    lines.append(
        "- **Runtime**: `score_s` is the per-retrain full-test-set scoring cost - MaxPatch scores "
        "~8x more rows than MaxHAC (196-256 raw patches vs ~24 tree nodes per image)."
    )
    lines.append("")

    out = common.RESULTS / "REPORT.md"
    out.write_text("\n".join(lines))
    common.log(f"wrote {out} ({n_traj} trajectories, {len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

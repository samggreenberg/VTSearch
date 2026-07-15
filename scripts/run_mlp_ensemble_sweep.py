"""MLP-ensemble uncertainty sweep.

Throwaway experiment script for issue #2497 ("MLP uncertainty").  It
exercises the two ensemble axes added to the eval harness and writes the
results (CSVs, plots, a README) under ``docs/experiments/mlp-ensemble/``:

1. **Label-curve axis** (`vtscore.eval.label_curve`) - trains the single
   MLP, the SVM baselines, and the ``mlp_ens{3,5,7,10}`` ensembles over a
   growing label budget.  Shows whether averaging N seed-varied MLPs buys
   any ranking (AUROC / F1@xcal) over one MLP, and traces the ensemble's
   reported per-item uncertainty (``std_mean``) as labels accumulate.

2. **Voting-order axis** (`vtscore.eval.voting_iterations`) - simulates
   voting under ``vote_order ∈ {shuffle, balanced, ensemble_std}`` and
   plots the inclusion-weighted cost against the number of votes cast.
   Answers "does choosing the next vote by ensemble disagreement (active
   learning) reach a low cost in fewer votes than random or class-balanced
   ordering?"  Per-step cost is always the single production MLP; the
   ensemble only *selects* the next vote.

Usage::

    python scripts/run_mlp_ensemble_sweep.py \\
        --datasets urbansound8k_s \\
        --out-dir docs/experiments/mlp-ensemble \\
        --label-counts 5 10 20 50 \\
        --seeds 0 1 2 \\
        --max-votes 40

Designed to be deleted once the results are committed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vtscore.eval.label_curve import run_label_curve_eval, summarise
from vtscore.eval.voting_iterations import VOTE_ORDERS, run_voting_iterations_eval

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_DATASETS = ("urbansound8k_s",)
DEFAULT_LABEL_COUNTS = (5, 10, 20, 50)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ENSEMBLE_SIZES = (3, 5, 7, 10)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_dataset(demo_id: str) -> dict[int, dict[str, Any]]:
    """Load one demo dataset into a fresh medias dict."""
    from vtscore.datasets.loader import load_demo_dataset

    medias: dict[int, dict[str, Any]] = {}
    load_demo_dataset(demo_id, medias)
    return medias


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _setup_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_label_curve(summary: "pd.DataFrame", out_dir: Path) -> list[Path]:
    """One line per trainer: AUROC and F1@xcal vs label count, ±1 std band.

    Curves are averaged over every (dataset, category) cell so the figure
    reads as "trainer X's ranking quality at N labels", which is the axis
    the ensemble question lives on.
    """
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    if summary.empty:
        return generated

    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    trainers = sorted(summary["trainer"].unique())

    for metric, ylabel in (("auroc", "AUROC"), ("f1_at_xcal", "F1 @ cross-cal threshold")):
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, trainer in enumerate(trainers):
            sub = summary[summary["trainer"] == trainer]
            agg = sub.groupby("n_labels")[[mean_col, std_col]].mean().reset_index()
            colour = palette[idx % len(palette)]
            n = agg["n_labels"].to_numpy()
            mean = agg[mean_col].to_numpy()
            std = agg[std_col].fillna(0).to_numpy()
            ax.plot(n, mean, marker="o", label=trainer, color=colour, linewidth=1.5)
            if (std > 0).any():
                ax.fill_between(n, mean - std, mean + std, alpha=0.12, color=colour)
        ax.set_xlabel("Training labels")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Label curve: {ylabel} by trainer")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        path = out_dir / f"label_curve_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        generated.append(path)

    # Ensemble uncertainty: std_mean vs label count, ensemble trainers only.
    if "std_mean_mean" in summary.columns:
        ens = summary[summary["trainer"].str.startswith("mlp_ens")]
        if not ens.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for idx, trainer in enumerate(sorted(ens["trainer"].unique())):
                sub = ens[ens["trainer"] == trainer]
                agg = sub.groupby("n_labels")["std_mean_mean"].mean().reset_index()
                colour = palette[idx % len(palette)]
                ax.plot(
                    agg["n_labels"].to_numpy(),
                    agg["std_mean_mean"].to_numpy(),
                    marker="o",
                    label=trainer,
                    color=colour,
                    linewidth=1.5,
                )
            ax.set_xlabel("Training labels")
            ax.set_ylabel("Mean per-item ensemble std")
            ax.set_title("Ensemble uncertainty (std_mean) by label count")
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            path = out_dir / "label_curve_std_mean.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            generated.append(path)

    return generated


def plot_voting_orders(df: "pd.DataFrame", out_dir: Path) -> list[Path]:
    """Cost vs votes cast, one line per vote_order (averaged over cells)."""
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    if df.empty:
        return generated

    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, order in enumerate(sorted(df["vote_order"].unique())):
        sub = df[df["vote_order"] == order]
        agg = sub.groupby("t")["cost"].agg(["mean", "std"]).reset_index()
        colour = palette[idx % len(palette)]
        t = agg["t"].to_numpy()
        mean = agg["mean"].to_numpy()
        std = agg["std"].fillna(0).to_numpy()
        ax.plot(t, mean, label=order, color=colour, linewidth=1.5)
        if (std > 0).any():
            ax.fill_between(t, mean - std, mean + std, alpha=0.12, color=colour)
    ax.set_xlabel("Votes cast (t)")
    ax.set_ylabel("Inclusion-weighted cost")
    ax.set_title("Voting cost by vote_order")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = out_dir / "voting_cost_by_order.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)
    return generated


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    *,
    report_path: Path,
    datasets: tuple[str, ...],
    label_counts: tuple[int, ...],
    seeds: tuple[int, ...],
    ensemble_sizes: tuple[int, ...],
    vote_orders: tuple[str, ...],
    max_votes: int | None,
    n_ensemble: int,
    label_summary: "pd.DataFrame",
    voting_df: "pd.DataFrame",
) -> None:
    lines: list[str] = []
    lines.append("# MLP-ensemble uncertainty sweep")
    lines.append("")
    lines.append(
        "Throwaway experiment for issue #2497.  Regenerate with "
        "`python scripts/run_mlp_ensemble_sweep.py` (see the module "
        "docstring for flags).  All CSVs and PNGs in this directory are "
        "produced by that script."
    )
    lines.append("")
    lines.append(
        f"Datasets: {', '.join(f'`{d}`' for d in datasets)} · "
        f"label counts: {list(label_counts)} · seeds: {list(seeds)} · "
        f"ensemble sizes: {list(ensemble_sizes)} · "
        f"vote orders: {list(vote_orders)} · "
        f"max_votes: {max_votes} · ensemble_std n_ensemble: {n_ensemble}."
    )
    lines.append("")

    # ---- Label-curve axis --------------------------------------------------
    lines.append("## 1. Label curve - does an MLP ensemble out-rank one MLP?")
    lines.append("")
    lines.append(
        "`mlp` is the single production MLP; `mlp_ens{N}` averages N "
        "seed-varied MLPs and reports member disagreement as per-item "
        "uncertainty.  VTSearch ranks by score and learns its own "
        "threshold, so the metrics that matter are **AUROC** (ranking) "
        "and **F1@xcal** (production-path F1); Brier / F1@0.5 stay "
        "diagnostic.  `std_mean` is the ensemble's mean per-item "
        "uncertainty - it is `nan` for the non-ensemble trainers."
    )
    lines.append("")
    lines.append("![AUROC by trainer](label_curve_auroc.png)")
    lines.append("")
    lines.append("![F1@xcal by trainer](label_curve_f1_at_xcal.png)")
    lines.append("")
    lines.append("![Ensemble uncertainty](label_curve_std_mean.png)")
    lines.append("")
    lines.append("Mean AUROC / F1@xcal per trainer, collapsed over seeds and label counts:")
    lines.append("")
    lines.extend(_trainer_headline_table(label_summary))
    lines.append("")

    # ---- Voting-order axis -------------------------------------------------
    lines.append("## 2. Voting order - does ensemble-uncertainty selection help?")
    lines.append("")
    lines.append(
        "Cost is the inclusion-weighted `fpr_weight·FPR + fnr_weight·FNR` "
        "on a held-out test split, measured with the single production MLP "
        "at every step.  `shuffle` votes in random order, `balanced` keeps "
        "the running label set class-balanced, and `ensemble_std` votes "
        "next on the item an N-member ensemble disagrees about most (active "
        "learning).  A lower curve, or the same cost reached at a smaller "
        "`t`, means the ordering is more label-efficient."
    )
    lines.append("")
    lines.append("![Cost by vote_order](voting_cost_by_order.png)")
    lines.append("")
    lines.extend(_voting_headline_table(voting_df))
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `label_curve.csv` - one row per (dataset, category, trainer, n_labels, seed).")
    lines.append("- `voting_iterations.csv` - one row per (vote_order, seed, dataset, category, t).")
    lines.append("- `*.png` - the plots embedded above.")
    lines.append("")

    report_path.write_text("\n".join(lines))


def _trainer_headline_table(summary: "pd.DataFrame") -> list[str]:
    if summary.empty:
        return ["_(no rows - every cell was skipped)_"]
    rows = ["| trainer | AUROC | F1@xcal | mean std |", "|---|---|---|---|"]
    for trainer in sorted(summary["trainer"].unique()):
        sub = summary[summary["trainer"] == trainer]
        auroc = float(sub["auroc_mean"].mean()) if "auroc_mean" in sub else float("nan")
        f1 = float(sub["f1_at_xcal_mean"].mean()) if "f1_at_xcal_mean" in sub else float("nan")
        std_mean = float(sub["std_mean_mean"].mean()) if "std_mean_mean" in sub else float("nan")
        std_txt = "—" if math.isnan(std_mean) else f"{std_mean:.3f}"
        rows.append(f"| `{trainer}` | {auroc:.3f} | {f1:.3f} | {std_txt} |")
    return rows


def _voting_headline_table(df: "pd.DataFrame") -> list[str]:
    if df.empty:
        return ["_(no rows - every cell was skipped)_"]
    rows = ["| vote_order | steps | final cost | min cost |", "|---|---|---|---|"]
    for order in sorted(df["vote_order"].unique()):
        sub = df[df["vote_order"] == order]
        # Final cost = mean cost at the largest t reached per (seed, dataset, category).
        finals = sub.sort_values("t").groupby(["seed", "dataset", "category"]).tail(1)
        rows.append(
            f"| `{order}` | {int(sub['t'].max())} | "
            f"{float(finals['cost'].mean()):.4f} | {float(sub['cost'].min()):.4f} |"
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), metavar="DEMO_ID")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/experiments/mlp-ensemble"))
    ap.add_argument("--label-counts", nargs="+", type=int, default=list(DEFAULT_LABEL_COUNTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--ensemble-sizes", nargs="+", type=int, default=list(DEFAULT_ENSEMBLE_SIZES))
    ap.add_argument(
        "--vote-orders",
        nargs="+",
        default=list(VOTE_ORDERS),
        choices=list(VOTE_ORDERS),
        help="Vote orders to compare in the voting-iterations axis.",
    )
    ap.add_argument("--categories", nargs="+", default=None, metavar="CAT")
    ap.add_argument("--inclusion", type=int, default=0)
    ap.add_argument(
        "--max-votes",
        type=int,
        default=40,
        help="Cap on votes cast per cell (keeps the ensemble_std order tractable). "
        "Pass 0 for no cap.",
    )
    ap.add_argument(
        "--n-ensemble",
        type=int,
        default=5,
        help="Ensemble size used for ensemble_std vote selection (default 5).",
    )
    ap.add_argument("--no-voting", action="store_true", help="Skip the voting-iterations axis.")
    ap.add_argument("--no-label-curve", action="store_true", help="Skip the label-curve axis.")
    args = ap.parse_args(argv)

    import pandas as pd

    from vtscore.embedding import initialize_models

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()
    initialize_models()

    print("Loading datasets…", flush=True)
    dataset_clips: dict[str, dict[int, dict[str, Any]]] = {}
    for demo_id in args.datasets:
        print(f"  {demo_id}", flush=True)
        dataset_clips[demo_id] = _load_dataset(demo_id)

    categories = {ds: args.categories for ds in dataset_clips} if args.categories else None
    max_votes = None if args.max_votes in (0, None) else args.max_votes

    label_summary = pd.DataFrame()
    voting_df = pd.DataFrame()

    # ---- Label-curve axis --------------------------------------------------
    if not args.no_label_curve:
        trainers = ["mlp", "svm_linear", *[f"mlp_ens{n}" for n in args.ensemble_sizes]]
        print(f"\nLabel-curve sweep: trainers={trainers}", flush=True)
        label_df = run_label_curve_eval(
            dataset_clips=dataset_clips,
            trainers=trainers,
            label_counts=tuple(args.label_counts),
            seeds=tuple(args.seeds),
            categories=categories,
            inclusion_value=args.inclusion,
            progress=True,
        )
        label_df.to_csv(args.out_dir / "label_curve.csv", index=False)
        print(f"  wrote {len(label_df)} rows -> label_curve.csv", flush=True)
        label_summary = summarise(label_df, include_diagnostics=True)
        for p in plot_label_curve(label_summary, args.out_dir):
            print(f"  plot -> {p.name}", flush=True)

    # ---- Voting-order axis -------------------------------------------------
    if not args.no_voting:
        print(f"\nVoting-iterations sweep: vote_orders={args.vote_orders}", flush=True)
        frames: list[pd.DataFrame] = []
        for order in args.vote_orders:
            print(f"  vote_order={order}", flush=True)
            df = run_voting_iterations_eval(
                dataset_clips=dataset_clips,
                seeds=list(args.seeds),
                categories=categories,
                inclusion=args.inclusion,
                vote_order=order,
                n_ensemble=args.n_ensemble,
                max_votes=max_votes,
            )
            df["vote_order"] = order
            frames.append(df)
        voting_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        voting_df.to_csv(args.out_dir / "voting_iterations.csv", index=False)
        print(f"  wrote {len(voting_df)} rows -> voting_iterations.csv", flush=True)
        for p in plot_voting_orders(voting_df, args.out_dir):
            print(f"  plot -> {p.name}", flush=True)

    # ---- Report ------------------------------------------------------------
    report_path = args.out_dir / "README.md"
    write_report(
        report_path=report_path,
        datasets=tuple(args.datasets),
        label_counts=tuple(args.label_counts),
        seeds=tuple(args.seeds),
        ensemble_sizes=tuple(args.ensemble_sizes),
        vote_orders=tuple(args.vote_orders),
        max_votes=max_votes,
        n_ensemble=args.n_ensemble,
        label_summary=label_summary,
        voting_df=voting_df,
    )
    print(f"\nwrote report -> {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

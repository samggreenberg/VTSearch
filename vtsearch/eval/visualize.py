"""Generate evaluation visualizations from eval results.

Produces charts summarising text-sort and learned-sort performance
across datasets and categories.  All functions accept the same data
structures returned by the runner / voting_iterations modules and
write PNG files to a specified output directory.

Charts produced by :func:`plot_eval_results`:

1. **mAP by dataset** — horizontal bar chart of mean Average Precision.
2. **AP by query** — grouped bar chart of per-query Average Precision,
   one group per dataset.
3. **P@k curves** — line chart of Precision@k for each query.
4. **R@k curves** — line chart of Recall@k for each query.
5. **Learned-sort F1 by category** — grouped bar chart of F1 per
   category, one group per dataset.
6. **Learned-sort metrics breakdown** — grouped bar chart showing
   accuracy, precision, recall, and F1 side-by-side per category.

Charts produced by :func:`plot_voting_iterations`:

1. **Cost over voting iterations** — line chart showing how the
   inclusion-weighted cost decreases as more votes are cast.  One
   line per (dataset, category) combination; multiple seeds are
   averaged with a shaded ±1 std-dev band.
2. **FPR / FNR over voting iterations** — similar layout, with
   separate lines for FPR and FNR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vtsearch.eval.metrics import DatasetResult


def _setup_style() -> None:
    """Apply a clean default style to matplotlib figures."""
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


# ------------------------------------------------------------------
# Standard eval plots
# ------------------------------------------------------------------


def plot_eval_results(
    results: list[DatasetResult],
    output_dir: str | Path = "eval_output",
) -> list[Path]:
    """Generate visualisation PNGs from standard eval results.

    Args:
        results: List of :class:`DatasetResult` from :func:`run_eval`.
        output_dir: Directory to write PNG files into (created if needed).

    Returns:
        List of paths to the generated PNG files.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Collect datasets that have text-sort / learned-sort results
    text_results = [r for r in results if r.text_sort]
    learned_results = [r for r in results if r.learned_sort]

    # ---- 1. mAP by dataset (horizontal bar) ----
    if text_results:
        path = _plot_map_by_dataset(text_results, output_dir)
        generated.append(path)

    # ---- 2. AP by query (grouped bar) ----
    if text_results:
        path = _plot_ap_by_query(text_results, output_dir)
        generated.append(path)

    # ---- 3. P@k curves ----
    if text_results:
        path = _plot_precision_at_k(text_results, output_dir)
        generated.append(path)

    # ---- 4. R@k curves ----
    if text_results:
        path = _plot_recall_at_k(text_results, output_dir)
        generated.append(path)

    # ---- 5. Learned-sort F1 by category ----
    if learned_results:
        path = _plot_learned_f1_by_category(learned_results, output_dir)
        generated.append(path)

    # ---- 6. Learned-sort metrics breakdown ----
    if learned_results:
        path = _plot_learned_metrics_breakdown(learned_results, output_dir)
        generated.append(path)

    plt.close("all")
    return generated


def _plot_map_by_dataset(results: list[DatasetResult], out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.7)))
    ds_ids = [r.dataset_id for r in results]
    maps = [r.mean_average_precision for r in results]

    y_pos = np.arange(len(ds_ids))
    bars = ax.barh(y_pos, maps, color="#4C72B0", edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_ids)
    ax.set_xlabel("Mean Average Precision (mAP)")
    ax.set_title("Text Sort: mAP by Dataset")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    path = out / "text_sort_map_by_dataset.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_ap_by_query(results: list[DatasetResult], out: Path) -> Path:
    import matplotlib.pyplot as plt

    # Flatten all queries across datasets
    labels: list[str] = []
    values: list[float] = []
    colours: list[str] = []
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, r in enumerate(results):
        colour = palette[i % len(palette)]
        for qm in r.text_sort:
            labels.append(f"{r.dataset_id}\n{qm.target_category}")
            values.append(qm.average_precision)
            colours.append(colour)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colours, edgecolor="white", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Average Precision")
    ax.set_title("Text Sort: AP by Query")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", fontsize=7)

    fig.tight_layout()
    path = out / "text_sort_ap_by_query.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_precision_at_k(results: list[DatasetResult], out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    line_idx = 0

    for r in results:
        for qm in r.text_sort:
            if not qm.precision_at_k:
                continue
            ks = sorted(qm.precision_at_k.keys())
            ps = [qm.precision_at_k[k] for k in ks]
            colour = palette[line_idx % len(palette)]
            ax.plot(ks, ps, marker="o", label=f"{r.dataset_id}: {qm.target_category}", color=colour, linewidth=1.5)
            line_idx += 1

    ax.set_xlabel("k")
    ax.set_ylabel("Precision@k")
    ax.set_title("Text Sort: Precision@k")
    ax.set_ylim(0, 1.05)
    if line_idx <= 15:
        ax.legend(fontsize=7, loc="best", ncol=max(1, line_idx // 8))

    fig.tight_layout()
    path = out / "text_sort_precision_at_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_recall_at_k(results: list[DatasetResult], out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]
    line_idx = 0

    for r in results:
        for qm in r.text_sort:
            if not qm.recall_at_k:
                continue
            ks = sorted(qm.recall_at_k.keys())
            rs = [qm.recall_at_k[k] for k in ks]
            colour = palette[line_idx % len(palette)]
            ax.plot(ks, rs, marker="s", label=f"{r.dataset_id}: {qm.target_category}", color=colour, linewidth=1.5)
            line_idx += 1

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Text Sort: Recall@k")
    ax.set_ylim(0, 1.05)
    if line_idx <= 15:
        ax.legend(fontsize=7, loc="best", ncol=max(1, line_idx // 8))

    fig.tight_layout()
    path = out / "text_sort_recall_at_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_learned_f1_by_category(results: list[DatasetResult], out: Path) -> Path:
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[float] = []
    colours: list[str] = []
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, r in enumerate(results):
        colour = palette[i % len(palette)]
        for lm in r.learned_sort:
            labels.append(f"{r.dataset_id}\n{lm.target_category}")
            values.append(lm.f1)
            colours.append(colour)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colours, edgecolor="white", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title("Learned Sort: F1 by Category")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}", ha="center", fontsize=7)

    fig.tight_layout()
    path = out / "learned_sort_f1_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_learned_metrics_breakdown(results: list[DatasetResult], out: Path) -> Path:
    import matplotlib.pyplot as plt

    categories: list[str] = []
    acc_vals: list[float] = []
    prec_vals: list[float] = []
    rec_vals: list[float] = []
    f1_vals: list[float] = []

    for r in results:
        for lm in r.learned_sort:
            categories.append(f"{r.dataset_id}\n{lm.target_category}")
            acc_vals.append(lm.accuracy)
            prec_vals.append(lm.precision)
            rec_vals.append(lm.recall)
            f1_vals.append(lm.f1)

    n = len(categories)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))
    ax.bar(x - 1.5 * width, acc_vals, width, label="Accuracy", color="#4C72B0")
    ax.bar(x - 0.5 * width, prec_vals, width, label="Precision", color="#55A868")
    ax.bar(x + 0.5 * width, rec_vals, width, label="Recall", color="#C44E52")
    ax.bar(x + 1.5 * width, f1_vals, width, label="F1", color="#8172B2")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Learned Sort: Metrics Breakdown")
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    path = out / "learned_sort_metrics_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------
# Voting iterations plots
# ------------------------------------------------------------------


def plot_voting_iterations(
    df: "Any",
    output_dir: str | Path = "eval_output",
) -> list[Path]:
    """Generate visualisation PNGs from voting-iterations eval results.

    When multiple seeds are present, lines are averaged per
    ``(dataset, category)`` and a shaded ±1 std-dev band is drawn.

    Args:
        df: :class:`pandas.DataFrame` with columns
            ``seed, dataset, category, t, cost, fpr, fnr`` as returned
            by :func:`run_voting_iterations_eval`.
        output_dir: Directory to write PNG files into (created if needed).

    Returns:
        List of paths to the generated PNG files.
    """
    import pandas as pd

    _setup_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if not isinstance(df, pd.DataFrame) or df.empty:
        return generated

    # ---- 1. Cost over voting iterations ----
    path = _plot_iterations_metric(df, "cost", "Inclusion-Weighted Cost", output_dir)
    generated.append(path)

    # ---- 2. FPR / FNR over voting iterations ----
    path = _plot_iterations_fpr_fnr(df, output_dir)
    generated.append(path)

    import matplotlib.pyplot as plt

    plt.close("all")
    return generated


def _plot_iterations_metric(
    df: "Any",
    metric: str,
    ylabel: str,
    out: Path,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

    groups = df.groupby(["dataset", "category"])
    for idx, ((ds, cat), group) in enumerate(groups):
        colour = palette[idx % len(palette)]
        agg = group.groupby("t")[metric].agg(["mean", "std"]).reset_index()
        t = agg["t"].values
        mean = agg["mean"].values
        std = agg["std"].fillna(0).values

        ax.plot(t, mean, label=f"{ds}: {cat}", color=colour, linewidth=1.5)
        if (std > 0).any():
            ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=colour)

    ax.set_xlabel("Voting Iteration (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Voting Iterations: {ylabel}")
    n_groups = len(list(groups))
    if n_groups <= 15:
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    path = out / f"voting_iterations_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_iterations_fpr_fnr(df: "Any", out: Path) -> Path:
    import matplotlib.pyplot as plt

    groups = list(df.groupby(["dataset", "category"]))
    n_groups = len(groups)

    if n_groups <= 4:
        ncols = min(n_groups, 2)
        nrows = (n_groups + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for idx, ((ds, cat), group) in enumerate(groups):
            ax = axes_flat[idx]
            agg = (
                group.groupby("t")
                .agg(
                    fpr_mean=("fpr", "mean"),
                    fpr_std=("fpr", "std"),
                    fnr_mean=("fnr", "mean"),
                    fnr_std=("fnr", "std"),
                )
                .reset_index()
            )
            t = agg["t"].values

            ax.plot(t, agg["fpr_mean"].values, label="FPR", color="#C44E52", linewidth=1.5)
            fpr_std = agg["fpr_std"].fillna(0).values
            if (fpr_std > 0).any():
                ax.fill_between(
                    t, agg["fpr_mean"].values - fpr_std, agg["fpr_mean"].values + fpr_std, alpha=0.15, color="#C44E52"
                )

            ax.plot(t, agg["fnr_mean"].values, label="FNR", color="#4C72B0", linewidth=1.5)
            fnr_std = agg["fnr_std"].fillna(0).values
            if (fnr_std > 0).any():
                ax.fill_between(
                    t, agg["fnr_mean"].values - fnr_std, agg["fnr_mean"].values + fnr_std, alpha=0.15, color="#4C72B0"
                )

            ax.set_title(f"{ds}: {cat}", fontsize=10)
            ax.set_xlabel("Voting Iteration (t)")
            ax.set_ylabel("Rate")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)
    else:
        # Many groups: single chart with combined FPR/FNR lines
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for idx, ((ds, cat), group) in enumerate(groups):
            colour = palette[idx % len(palette)]
            agg = group.groupby("t").agg(fpr_mean=("fpr", "mean"), fnr_mean=("fnr", "mean")).reset_index()
            t = agg["t"].values
            ax.plot(t, agg["fpr_mean"].values, linestyle="--", color=colour, linewidth=1, label=f"{ds}:{cat} FPR")
            ax.plot(t, agg["fnr_mean"].values, linestyle="-", color=colour, linewidth=1, label=f"{ds}:{cat} FNR")

        ax.set_xlabel("Voting Iteration (t)")
        ax.set_ylabel("Rate")
        ax.set_title("Voting Iterations: FPR & FNR")
        ax.set_ylim(-0.05, 1.05)
        if n_groups <= 8:
            ax.legend(fontsize=6, loc="best", ncol=2)

    fig.tight_layout()
    path = out / "voting_iterations_fpr_fnr.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

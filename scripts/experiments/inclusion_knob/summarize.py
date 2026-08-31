"""Stage 2: aggregate the sweep CSV into figures + markdown tables for the report.

Reads ``docs/experiments/2026-07-27-inclusion-knob/sweep.csv`` and writes, next to it:

* ``fig_knob_response.png``   - included fraction vs inclusion, per design/arm
* ``fig_recall_fpr.png``      - recall & FPR vs inclusion, per design
* ``summary_tables.md``       - the responsiveness + saturation tables the report embeds

The headline metric is ``flat_frac``: the fraction of (seed x category) sweeps
where throwing the knob from -10 to +10 changes the included set size by less
than 1% of the pool - issue #2693's literal "we failed" criterion.
"""

from __future__ import annotations

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (dataviz skill), light mode: slots 1-4 in fixed order.
DESIGN_COLORS = {
    "argmin": "#2a78d6",
    "bayes": "#eb6834",
    "bayes_temp": "#1baf7a",
    "conformal": "#eda100",
}
DESIGN_LABELS = {
    "argmin": "argmin (production)",
    "bayes": "bayes",
    "bayes_temp": "bayes + temperature",
    "conformal": "conformal quantile",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

ARM_GROUPS = ("agnews", "synth:easy", "synth:medium", "synth:hard")


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(True, color=GRID, linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _arm_group(arm: str) -> str:
    return "agnews" if arm.startswith("agnews:") else arm


def _sweep_stats(g: pd.DataFrame) -> pd.Series:
    """Per-(cell, design) sweep statistics; ``g`` is one inclusion sweep."""
    g = g.sort_values("inclusion")
    frac = g.n_included.to_numpy() / g.pool_size.to_numpy()
    span = frac[-1] - frac[0]
    return pd.Series(
        {
            "flat": abs(span) < 0.01,
            "span_frac": span,
            "n_distinct_sizes": len(np.unique(g.n_included)),
            "mono_violation": bool((np.diff(g.n_included.to_numpy()) < 0).any()),
            "recall_at_+10": g[g.inclusion == 10].recall.iloc[0],
            "fpr_at_+10": g[g.inclusion == 10].fpr.iloc[0],
            "recall_at_-10": g[g.inclusion == -10].recall.iloc[0],
            "precision_at_-10": g[g.inclusion == -10].precision.iloc[0],
        }
    )


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(common.RESULTS / "sweep.csv")
    df["group"] = df.arm.map(_arm_group)
    df["frac_included"] = df.n_included / df.pool_size

    # ------------------------------------------------------------------
    # Table: responsiveness per (group, treatment, design) at n_votes=50
    # ------------------------------------------------------------------
    cell_keys = ["group", "treatment", "design", "arm", "seed", "n_votes"]
    stats = df.groupby(cell_keys, sort=False)[df.columns.tolist()].apply(_sweep_stats).reset_index()
    lines: list[str] = []
    for n_votes in sorted(df.n_votes.unique()):
        sub = stats[stats.n_votes == n_votes]
        agg = (
            sub.groupby(["group", "treatment", "design"], sort=False)
            .agg(
                flat_frac=("flat", "mean"),
                span_frac=("span_frac", "mean"),
                distinct_sizes=("n_distinct_sizes", "mean"),
                mono_viol=("mono_violation", "mean"),
                recall_p10=("recall_at_+10", "mean"),
                fpr_p10=("fpr_at_+10", "mean"),
                recall_m10=("recall_at_-10", "mean"),
                prec_m10=("precision_at_-10", "mean"),
            )
            .reset_index()
        )
        lines.append(f"\n### n_votes = {n_votes}\n")
        lines.append(
            "| group | treatment | design | flat% | span | distinct sizes | mono viol% "
            "| recall@+10 | FPR@+10 | recall@-10 | precision@-10 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in agg.iterrows():
            lines.append(
                f"| {r.group} | {r.treatment} | {r.design} | {r.flat_frac:.0%} | {r.span_frac:.3f} "
                f"| {r.distinct_sizes:.1f} | {r.mono_viol:.0%} | {r.recall_p10:.3f} | {r.fpr_p10:.3f} "
                f"| {r.recall_m10:.3f} | {r.prec_m10:.3f} |"
            )
    (common.RESULTS / "summary_tables.md").write_text("\n".join(lines) + "\n")
    common.log(f"wrote {common.RESULTS / 'summary_tables.md'}")

    # ------------------------------------------------------------------
    # Figure 1: knob response (included fraction vs inclusion), n_votes=50, raw
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURFACE, sharey=True)
    for ax, group in zip(axes, ARM_GROUPS, strict=True):
        sub = df[(df.group == group) & (df.n_votes == 50) & (df.treatment == "raw")]
        for design in DESIGN_COLORS:
            agg = sub[sub.design == design].groupby("inclusion").frac_included.mean()
            # bayes is dashed: bayes_temp coincides with it wherever T=1 and
            # would otherwise hide it entirely.
            ls = "--" if design == "bayes" else "-"
            ax.plot(
                agg.index,
                agg.values,
                color=DESIGN_COLORS[design],
                linewidth=2,
                linestyle=ls,
                label=DESIGN_LABELS[design],
            )
        _style(ax, group)
        ax.set_xlabel("inclusion", fontsize=9, color=MUTED)
    axes[0].set_ylabel("fraction of pool included", fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("Knob response by design (n_votes=50, raw training)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(common.RESULTS / "fig_knob_response.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: recall and FPR vs inclusion (agnews + synth:hard), n_votes=50, raw
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), facecolor=SURFACE)
    for col, group in enumerate(("agnews", "synth:hard")):
        sub = df[(df.group == group) & (df.n_votes == 50) & (df.treatment == "raw")]
        for row, metric in enumerate(("recall", "fpr")):
            ax = axes[row][col]
            for design in DESIGN_COLORS:
                agg = sub[sub.design == design].groupby("inclusion")[metric].mean()
                ls = "--" if design == "bayes" else "-"
                ax.plot(
                    agg.index,
                    agg.values,
                    color=DESIGN_COLORS[design],
                    linewidth=2,
                    linestyle=ls,
                    label=DESIGN_LABELS[design],
                )
            _style(ax, f"{group}: {metric} vs inclusion")
            if row == 1:
                ax.set_xlabel("inclusion", fontsize=9, color=MUTED)
            ax.set_ylabel(metric, fontsize=9, color=MUTED)
    axes[0][0].legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(common.RESULTS / "fig_recall_fpr.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Saturation is reported as a table (no third figure): in this harness the
    # scores never saturate (sat_frac_mid = 1.0 throughout), which is itself a
    # finding - see the report's "saturation is not the root cause" section.
    sat = (
        df[df.group == "agnews"]
        .groupby(["n_votes", "treatment"])[
            ["sat_frac_extreme", "sat_frac_mid", "sat_mean_abs_logit", "cal_mean_abs_logit"]
        ]
        .mean()
        .round(3)
    )
    with (common.RESULTS / "summary_tables.md").open("a") as f:
        f.write("\n### AG News score-saturation stats (pool / calibration mean |logit|)\n\n")
        f.write("| n_votes | treatment | extreme frac | mid frac | pool mean abs logit | cal mean abs logit |\n")
        f.write("|---|---|---|---|---|---|\n")
        for (n_votes, treatment), r in sat.iterrows():
            f.write(
                f"| {n_votes} | {treatment} | {r.sat_frac_extreme:.3f} | {r.sat_frac_mid:.3f} "
                f"| {r.sat_mean_abs_logit:.3f} | {r.cal_mean_abs_logit:.3f} |\n"
            )

    common.log("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""State of the App (#4159): the report's figures, from ``analyze.py``'s tables.

    python figures.py --analysis <exp>/analysis --out <report dir>/figures

* ``ap_over_clicks.png`` / ``goods_over_clicks.png`` -- the mean curve per path,
  click 0 (text only) to the last click, with each path's full-label ceiling AP
  dashed. Two figures, not one with two y-axes: AP and a count of Goods are
  different measures.
* ``line_at_floors.png`` -- the line on a fresh corpus, one panel per floor the
  app offers: how right the set the line keeps is, from click 0 to the final
  line, against the floor (dotted) and the full-label ceiling's line (dashed).
  Each panel's title carries the share of sessions whose final line meets P.
* ``returned_at_p.png`` -- the returned set at each floor P over clicks: its
  precision against P, its recall against the oracle's recall at P (#4408).
* ``objective_over_clicks.png`` -- the objective (#4427): F-beta of the withheld
  set above the app's threshold over clicks, per path, the check's end marked.
* ``in_hand.png`` -- positives in hand over clicks (#4427): the Goods voted, the
  positives inside the set the line keeps on the user's own unvoted corpus, and
  their sum, with that set's precision beside them.
* ``returned_at_beta.png`` -- the returned set at each balance over clicks: its
  F-beta as a share of the best cut's (#4413).
* ``f1_over_clicks.png`` -- the F1 of the set the line keeps (the returned set),
  over clicks, at the default floor P = 50% and at 10%, drawn only at the
  clicks a rank frame was recorded (``line_steps.csv``), with the text sort's
  F1 at click 0 and the full-label ceiling's line dashed (owner, 2026-09-30:
  AP is all ranking; F1 is the returned set).
* ``compare_ap.png`` (with ``--compare <other analysis dir>``) -- both paths'
  mean AP on the SAME seeds, the ones this analysis has.
* ``per_cell.png`` -- every class x band: text only (hollow), after the clicks
  (filled), full labels (tick), one panel per path.

No FPR + FNR anywhere (owner, 2026-09-30, #4357). Colours are the dataviz
reference palette's first two categorical slots, which validate for both CVD
and normal vision (validate_palette.js, 2026-09-23).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"SigLIP binary": "#2a78d6", "DINOv3 region": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
#: The line's points left to right, as ``analyze.py`` names them; the ceiling is drawn apart.
LINE_POINTS = ("text", "10", "25", "50", "100", "150", "final")


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def over_clicks(curves: pd.DataFrame, cells: pd.DataFrame, metric: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _axes(ax)
    mean = curves.groupby(["arm", "t"])[metric].mean().unstack(0)
    for arm, color in COLORS.items():
        if arm not in mean:
            continue
        ax.plot(mean.index, mean[arm], color=color, linewidth=2)
        fmt = f"{mean[arm].iloc[-1]:.2f}" if metric == "ap" else f"{mean[arm].iloc[-1]:.1f}"
        ax.annotate(
            f"{arm}  {fmt}",
            (mean.index[-1], mean[arm].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
        if metric == "ap":
            ceil = cells[cells["arm"] == arm]["ceiling_ap"].mean()
            if pd.notna(ceil):
                ax.axhline(ceil, color=color, linewidth=1.2, linestyle="--")
                ax.annotate(
                    f"full labels {ceil:.2f}",
                    (2, ceil),
                    xytext=(0, 4),
                    textcoords="offset points",
                    color=INK,
                    fontsize=8,
                )
    if metric == "ap":
        t0 = mean.iloc[0].mean()
        ax.plot([0], [t0], marker="o", markersize=8, color=INK, zorder=5)
        ax.annotate(f"text only {t0:.2f}", (0, t0), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=8)
        ax.set_ylim(0, 1.02)
    ax.set_xlim(0, mean.index.max() * 1.17)
    ax.set_xlabel("clicks", color=INK)
    label = {
        "ap": "average precision on the test half (higher is better)",
        "goods": "Goods found (higher is better)",
    }[metric]
    ax.set_ylabel(label, color=INK)
    n = cells.groupby("arm").size().to_dict()
    name = {"ap": "AP", "goods": "Goods found"}[metric]
    ax.set_title(
        f"Mean {name} over clicks  (runs: " + ", ".join(f"{k} {v}" for k, v in n.items()) + ")",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def line_at_floors(lines: pd.DataFrame, out: Path) -> bool:
    """One panel per floor: the kept set's precision from click 0 to the final line.

    Returns False (and draws nothing) when the run recorded no rank frames.
    """
    known = lines[(lines["point"] != "text") & lines["precision"].notna()]
    if known.empty:
        return False
    floors = sorted(lines["floor"].unique())
    fig, axes = plt.subplots(1, len(floors), figsize=(4.0 * len(floors), 3.8), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(floors) > 1 else [axes]
    for ax, x in zip(axes, floors, strict=True):
        _axes(ax)
        sub = lines[lines["floor"] == x]
        met = []
        for arm, color in COLORS.items():
            a = sub[sub["arm"] == arm]
            if a.empty:
                continue
            m = a.groupby("point")["precision"].mean().reindex(LINE_POINTS)
            ax.plot(range(len(LINE_POINTS)), m.to_numpy(), color=color, linewidth=2, marker="o", markersize=5)
            ceil = a[a["point"] == "ceiling"]["precision"].mean()
            if pd.notna(ceil):
                ax.axhline(ceil, color=color, linewidth=1.2, linestyle="--")
            final = a[a["point"] == "final"]["meets"].mean()
            if pd.notna(final):
                met.append(f"{arm} {final:.0%}")
        ax.axhline(x, color=INK, linewidth=1, linestyle=":")
        ax.annotate(f"floor {x:.0%}", (0, x), xytext=(2, 4), textcoords="offset points", color=INK, fontsize=8)
        ax.set_xticks(range(len(LINE_POINTS)))
        ax.set_xticklabels(["text", *LINE_POINTS[1:-1], "final"], fontsize=8)
        ax.set_xlabel("clicks", color=INK)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"P = {x:.0%}: meets P at the end: " + ", ".join(met), color=INK, fontsize=9, loc="left")
    axes[0].set_ylabel("share of the kept set that is right", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def f1_over_clicks(curves: pd.DataFrame, cells: pd.DataFrame, steps: pd.DataFrame, out: Path) -> bool:
    """The returned set's F1 over clicks, at the default floor (solid) and at 10% (light).

    Read only at click 0 and at the clicks a rank frame was recorded: between
    frames ``curves.csv`` carries the last value, which is not a measurement.
    Returns False (and draws nothing) when the run recorded no rank frames.
    """
    if steps is None or steps.empty or "f1" not in curves:
        return False
    at = sorted({0, *steps["t"].astype(int).unique().tolist()})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _axes(ax)
    for arm, color in COLORS.items():
        c = curves[(curves["arm"] == arm) & curves["t"].isin(at)]
        if c.empty:
            continue
        for col, floor, lw, alpha in (("f1", "50%", 2.0, 1.0), ("f1_p10", "10%", 1.4, 0.45)):
            if col not in c:
                continue
            m = c.groupby("t")[col].mean()
            ax.plot(m.index, m.to_numpy(), color=color, linewidth=lw, alpha=alpha, marker="o", markersize=2.5)
            ax.annotate(
                f"{arm}, P = {floor}  {m.iloc[-1]:.2f}",
                (m.index[-1], m.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK,
                fontsize=8,
            )
        ceil = cells[cells["arm"] == arm]["ceiling_f1"].mean() if "ceiling_f1" in cells else float("nan")
        if pd.notna(ceil):
            ax.axhline(ceil, color=color, linewidth=1.2, linestyle="--")
            ax.annotate(
                f"full labels, P = 50%  {ceil:.2f}", (2, ceil), xytext=(0, 4), textcoords="offset points",
                color=INK, fontsize=8,
            )  # fmt: skip
        t0 = cells[cells["arm"] == arm]["text_f1"].mean()
        if pd.notna(t0):
            ax.plot([0], [t0], marker="o", markersize=8, color=INK, zorder=5)
            ax.annotate(
                f"text only {t0:.2f}", (0, t0), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=8
            )
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(at) * 1.3)
    ax.set_xlabel("clicks", color=INK)
    ax.set_ylabel("F1 of the returned set on the test half", color=INK)
    n = cells.groupby("arm").size().to_dict()
    ax.set_title(
        "Mean F1 of the line's set over clicks  (runs: " + ", ".join(f"{k} {v}" for k, v in n.items()) + ")",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def per_cell(cells: pd.DataFrame, out: Path) -> None:
    cells = cells.copy()
    order = cells.groupby("category")["final_ap"].mean().sort_values(ascending=False).index.tolist()
    arms = [(a, c) for a, c in COLORS.items() if a in set(cells["arm"])]
    fig, axes = plt.subplots(
        1, len(arms), figsize=(5.2 * len(arms), 0.2 * len(order) + 1.4), sharey=True, facecolor=SURFACE, squeeze=False
    )
    axes = axes[0]
    for ax, (arm, color) in zip(axes, arms, strict=True):
        _axes(ax)
        sub = cells[cells["arm"] == arm].assign(
            never_trained=lambda d: d["never_trained"].astype("boolean").fillna(False).astype(bool)
        )
        # Mean over seeds; a cell "never found a positive" only if it failed in every seed.
        a = (
            sub.groupby("category")
            .agg(
                text_ap=("text_ap", "mean"),
                final_ap=("final_ap", "mean"),
                ceiling_ap=("ceiling_ap", "mean"),
                never_trained=("never_trained", "all"),
            )
            .reindex(order)
        )
        y = range(len(order))
        for yi, (fa, ca) in zip(y, a[["final_ap", "ceiling_ap"]].itertuples(index=False), strict=True):
            ax.plot([fa, ca], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(
            a["text_ap"], y, s=36, facecolors="none", edgecolors=MUTED, linewidths=1.2, zorder=2, label="text only"
        )
        ax.scatter(a["final_ap"], y, s=40, color=color, zorder=3, label="after the clicks")
        ax.scatter(a["ceiling_ap"], y, s=60, marker="|", color=INK, zorder=3, label="full labels")
        never = a[a["never_trained"].astype("boolean").fillna(False).astype(bool)]
        for cat in never.index:
            ax.annotate(
                "never found a positive",
                (never.loc[cat, "text_ap"], order.index(cat)),
                xytext=(10, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=INK,
            )
        ax.set_title(arm, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("average precision (higher is better)", color=INK)
        ax.set_xlim(0, 1.02)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=6 if len(order) > 60 else 8)
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def returned_at_p(curves: pd.DataFrame, cells: pd.DataFrame, steps: pd.DataFrame, out: Path) -> bool:
    """The returned set at each floor over clicks (#4408): precision against P, recall against the oracle.

    One column per floor (10/50/90%). Top: the set's precision, with P dashed;
    bottom: its recall, with the oracle's recall at P (the most any cut of the
    same ranking returns at or above P) dashed. Read at click 0 and at the
    clicks a rank frame was recorded. A floor the run's sessions did not aim at
    is titled so: it is read off sessions that aimed elsewhere.
    """
    if steps is None or steps.empty or "precision_p50" not in curves:
        return False
    at = sorted({0, *steps["t"].astype(int).unique().tolist()})
    own = set(cells["session_floor"].dropna().round(4)) if "session_floor" in cells else set()
    floors = (0.1, 0.5, 0.9)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), facecolor=SURFACE, sharex=True)
    for j, floor in enumerate(floors):
        tag = f"p{round(floor * 100)}"
        for arm, color in COLORS.items():
            c = curves[(curves["arm"] == arm) & curves["t"].isin(at)]
            if c.empty or f"precision_{tag}" not in c:
                continue
            m = c.groupby("t")[[f"precision_{tag}", f"recall_{tag}", f"oracle_recall_{tag}"]].mean()
            axes[0, j].plot(m.index, m[f"precision_{tag}"], color=color, marker="o", markersize=2.5, lw=2)
            axes[1, j].plot(m.index, m[f"recall_{tag}"], color=color, marker="o", markersize=2.5, lw=2, label=arm)
            axes[1, j].plot(m.index, m[f"oracle_recall_{tag}"], color=color, ls="--", lw=1.2, label="oracle at P")
            for row, col in ((0, f"precision_{tag}"), (1, f"recall_{tag}")):
                v = m[col].iloc[-1]
                axes[row, j].annotate(f"{v:.2f}", (m.index[-1], v), xytext=(4, 0), textcoords="offset points",
                                      va="center", color=INK, fontsize=8)  # fmt: skip
        axes[0, j].axhline(floor, color=INK, ls=":", lw=1.2)
        axes[0, j].annotate(f"P = {floor:.0%}", (at[-1], floor), xytext=(-40, 4), textcoords="offset points",
                            color=INK, fontsize=8)  # fmt: skip
        aimed = "these sessions' P" if round(floor, 4) in own else "read off sessions at another P"
        axes[0, j].set_title(f"P = {floor:.0%} ({aimed})", color=INK, fontsize=10, loc="left")
        for row in (0, 1):
            _axes(axes[row, j])
            axes[row, j].set_ylim(0, 1.02)
        axes[1, j].set_xlabel("clicks", color=INK)
    axes[0, 0].set_ylabel("precision of the returned set", color=INK)
    axes[1, 0].set_ylabel("recall of the returned set", color=INK)
    axes[1, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def returned_at_beta(curves: pd.DataFrame, cells: pd.DataFrame, balance_steps: pd.DataFrame, out: Path) -> bool:
    """The returned set at each balance over clicks (#4413): its F-beta as a share of the best cut.

    One panel per beta the analysis read (the app's presets, 1/4 / 1 / 4 since
    #4448; an older analysis has 0.5 / 1 / 2): the share, with 1.0 (the best
    cut) dotted; read at click 0 and at the clicks a rank frame was recorded.
    A balance the run's sessions did not aim at is titled so.
    """
    if balance_steps is None or balance_steps.empty or "fb_share_b1" not in curves:
        return False
    at = sorted({0, *balance_steps["t"].astype(int).unique().tolist()})
    own = set(cells["session_beta"].dropna().round(4)) if "session_beta" in cells else set()
    betas = [
        (b, "b" + (f"{b:g}".replace(".", "") if b < 1 else f"{b:g}")) for b in sorted(balance_steps["beta"].unique())
    ]
    fig, axes = plt.subplots(
        1, len(betas), figsize=(13 * len(betas) / 3, 3.9), facecolor=SURFACE, sharey=True, squeeze=False
    )
    axes = axes[0]
    for j, (beta, tag) in enumerate(betas):
        ax = axes[j]
        for arm, color in COLORS.items():
            c = curves[(curves["arm"] == arm) & curves["t"].isin(at)]
            if c.empty or f"fb_share_{tag}" not in c:
                continue
            m = c.groupby("t")[f"fb_share_{tag}"].mean()
            ax.plot(m.index, m.to_numpy(), color=color, marker="o", markersize=2.5, lw=2, label=arm)
            ax.annotate(f"{m.iloc[-1]:.2f}", (m.index[-1], m.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8)  # fmt: skip
        ax.axhline(1.0, color=INK, ls=":", lw=1.2)
        aimed = "these sessions' balance" if round(beta, 4) in own else "read off sessions at another preference"
        ax.set_title(f"beta = {beta:g} ({aimed})", color=INK, fontsize=10, loc="left")
        _axes(ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("clicks", color=INK)
    axes[0].set_ylabel("F-beta of the returned set / best cut", color=INK)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def objective_over_clicks(curves: pd.DataFrame, cells: pd.DataFrame, out: Path) -> bool:
    """The objective over clicks (#4427): F-beta of the withheld set above the app's threshold, per path,
    the unchecked line at each click, with the post-check value marked at the end."""
    if "thr_fbeta" not in curves or not curves["thr_fbeta"].notna().any():
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _axes(ax)
    mean = curves.groupby(["arm", "t"])["thr_fbeta"].mean().unstack(0)
    for arm, color in COLORS.items():
        if arm not in mean:
            continue
        ax.plot(mean.index, mean[arm], color=color, linewidth=2, label=f"{arm}: unchecked line")
        c = cells[(cells["arm"] == arm) & ~cells["never_trained"].astype(bool)]
        if "thr_fbeta_final" in c and c["thr_fbeta_final"].notna().any():
            end = float(c["thr_fbeta_final"].mean())
            ax.plot([mean.index.max()], [end], marker="D", markersize=7, color=color, zorder=5)
            ax.annotate(f"after the check {end:.2f}", (mean.index.max(), end), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=8)  # fmt: skip
        ax.annotate(f"{mean[arm].dropna().iloc[-1]:.2f}", (mean.index.max(), mean[arm].dropna().iloc[-1]),
                    xytext=(6, 10), textcoords="offset points", va="center", color=INK, fontsize=8)  # fmt: skip
    ax.set_xlim(0, mean.index.max() * 1.25)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("clicks", color=INK)
    ax.set_ylabel("F-beta of the withheld set above the threshold", color=INK)
    ax.set_title("The objective over clicks: unchecked line, then the check's end", color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def in_hand(cells: pd.DataFrame, pool_steps: pd.DataFrame, out: Path) -> bool:
    """Positives in hand over clicks (#4427): Goods, the kept set's positives on the user's own unvoted
    corpus, and their sum, per path; beside them the kept set's precision there.
    """
    if pool_steps is None or pool_steps.empty or not pool_steps["in_hand"].notna().any():
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE)
    for arm, color in COLORS.items():
        p = pool_steps[pool_steps["arm"] == arm]
        if p.empty:
            continue
        g = p.groupby("t")[["goods", "pool_tp", "in_hand", "pool_precision"]].mean()
        axes[0].plot(g.index, g["in_hand"], color=color, lw=2.2, label=f"{arm}: in hand")
        axes[0].plot(g.index, g["goods"], color=color, lw=1.2, ls="--", label=f"{arm}: Goods voted")
        axes[0].plot(g.index, g["pool_tp"], color=color, lw=1.2, ls=":", label=f"{arm}: in the kept set")
        axes[0].annotate(f"{g['in_hand'].iloc[-1]:.1f}", (g.index[-1], g["in_hand"].iloc[-1]), xytext=(4, 0),
                         textcoords="offset points", va="center", color=INK, fontsize=8)  # fmt: skip
        axes[1].plot(g.index, g["pool_precision"], color=color, lw=2, label=arm)
    axes[0].set_title("positives in hand on the user's own corpus", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("mean over runs", color=INK)
    axes[1].set_title("precision of the kept set there", color=INK, fontsize=10, loc="left")
    axes[1].set_ylim(0, 1.0)
    for ax in axes:
        _axes(ax)
        ax.set_xlabel("clicks", color=INK)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None, help="the other path's analysis dir, for compare_ap.png")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.analysis / "cells.csv")
    curves = pd.read_csv(args.analysis / "curves.csv")
    lines = pd.read_csv(args.analysis / "lines.csv", dtype={"point": str})
    if not objective_over_clicks(curves, cells, args.out / "objective_over_clicks.png"):
        print("no session thresholds in this run: objective_over_clicks.png skipped")
    over_clicks(curves, cells, "ap", args.out / "ap_over_clicks.png")
    over_clicks(curves, cells, "goods", args.out / "goods_over_clicks.png")
    if not line_at_floors(lines, args.out / "line_at_floors.png"):
        print("no rank frames in this run: line_at_floors.png skipped")
    steps_path = args.analysis / "line_steps.csv"
    steps = pd.read_csv(steps_path) if steps_path.exists() and steps_path.stat().st_size > 1 else pd.DataFrame()
    if not f1_over_clicks(curves, cells, steps, args.out / "f1_over_clicks.png"):
        print("no rank frames in this run: f1_over_clicks.png skipped")
    if not returned_at_p(curves, cells, steps, args.out / "returned_at_p.png"):
        print("no rank frames in this run: returned_at_p.png skipped")
    bs_path = args.analysis / "balance_steps.csv"
    balance_steps = pd.read_csv(bs_path) if bs_path.exists() and bs_path.stat().st_size > 1 else pd.DataFrame()
    if not returned_at_beta(curves, cells, balance_steps, args.out / "returned_at_beta.png"):
        print("no rank frames in this run: returned_at_beta.png skipped")
    ps_path = args.analysis / "pool_steps.csv"
    pool_steps = pd.read_csv(ps_path) if ps_path.exists() and ps_path.stat().st_size > 1 else pd.DataFrame()
    if not in_hand(cells, pool_steps, args.out / "in_hand.png"):
        print("no rank frames in this run: in_hand.png skipped")
    per_cell(cells, args.out / "per_cell.png")
    if args.compare is not None:
        seeds = set(cells["seed"])
        other_cells = pd.read_csv(args.compare / "cells.csv")
        other_curves = pd.read_csv(args.compare / "curves.csv")
        both_cells = pd.concat([cells, other_cells[other_cells["seed"].isin(seeds)]], ignore_index=True)
        both_curves = pd.concat([curves, other_curves[other_curves["seed"].isin(seeds)]], ignore_index=True)
        over_clicks(both_curves, both_cells, "ap", args.out / "compare_ap.png")
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

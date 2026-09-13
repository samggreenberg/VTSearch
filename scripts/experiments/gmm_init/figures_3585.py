#!/usr/bin/env python
"""Figures for the #3585 report, from the gate frames the report's tables use.

    python figures_3585.py --analysis <dir> --out <dir>/figures

Three, each answering a question the tables state but cannot show:

``arms.png`` - the trade-off itself.  One point per candidate: how much faster
it is against how much of the admitted set it moves.  The whole argument of the
study is that one candidate sits in the corner (fast *and* faithful) and that
the alternatives - including a three-character re-init of the incumbent - do
not, and a scatter says that in one look where a table needs two columns and a
reader who trusts both.

``change_ecdf.png`` - the distribution behind the mean.  "0.25% of the haystack
on average" is compatible with "every sort moves a little" and with "19 sorts
move by nothing and one moves by 5%", and those are different risks.

``cost_vs_n.png`` - cost against sample size, which is what says whether the
saving survives to the sizes the app fits on.  The measured sizes and the
bootstrap-resampled ones are drawn differently, because only one of them is a
measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Arm -> (colour, marker).  Fixed per arm so the same candidate is the same
#: colour in every figure of the report, and so deselecting one elsewhere does
#: not repaint the others.
STYLE = {
    "native": ("#1f77b4", "o"),
    "native_ll1e-4": ("#4e9ad4", "s"),
    "native_ll1e-5": ("#8fbfe0", "^"),
    "native_param1e-8": ("#d62728", "v"),
    "native_iter50": ("#ff9896", "D"),
    "sklearn_kmeanspp": ("#2ca02c", "P"),
    "sklearn_spherical": ("#98df8a", "X"),
    "native_10k": ("#9467bd", "*"),
    "baseline": ("#444444", "."),
}


def _style(arm: str) -> tuple[str, str]:
    return STYLE.get(arm, ("#777777", "o"))


#: One captured case's identity - the same key ``analyze_3585.CASE_KEYS`` uses.
_CASE_KEYS = ["cell", "dataset", "embedder", "style", "kind", "case"]


def _paired(cuts: pd.DataFrame) -> pd.DataFrame:
    """Every arm's cases joined to the baseline's, with the two deltas."""
    tcol, acol = "threshold_i0", "n_admitted_i0"
    df = cuts.copy()
    for c in (tcol, acol, "seconds"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keys = _CASE_KEYS
    base = df[df["arm"] == "baseline"][keys + [tcol, acol, "seconds"]].rename(
        columns={tcol: "base_t", acol: "base_a", "seconds": "base_s"}
    )
    out = df[df["arm"] != "baseline"].merge(base, on=keys, how="inner", validate="many_to_one")
    out["d_thr"] = (out[tcol] - out["base_t"]).abs()
    out["d_frac"] = (out[acol] - out["base_a"]).abs() / out["n"]
    out["speedup"] = out["base_s"] / out["seconds"].replace(0.0, np.nan)
    return out


def fig_arms(paired: pd.DataFrame, path: Path) -> None:
    keys = _CASE_KEYS
    kinds = [k for k in ("sort", "fold") if k in set(paired["kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(6.2 * len(kinds), 5.0), squeeze=False)
    for ax, kind in zip(axes[0], kinds, strict=True):
        g = paired[paired["kind"] == kind]
        # Grouped on the whole identity, not (cell, case): a sort capture holds
        # a grid, so a category name repeats inside one file and the short key
        # under-counts the panel by a third.
        n_cases = g.groupby(keys).ngroups
        # The linear part of the symlog axis is sized to the data: fixed at 0.01
        # it gave half the panel to a region holding one point (the arm that
        # changes nothing) and squashed every other arm into the top strip.
        means = 100.0 * g.groupby("arm")["d_frac"].mean()
        positive = means[means > 0]
        linthresh = max(float(positive.min()) / 3.0, 1e-3) if len(positive) else 0.01
        for arm, gg in g.groupby("arm"):
            colour, marker = _style(str(arm))
            x = gg["speedup"].median()
            y = 100.0 * gg["d_frac"].mean()
            ax.scatter(x, y, s=150, color=colour, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
            # Labelled in place rather than in a legend: eight arms is more than
            # a legend can carry without the reader counting marker shapes, and
            # the whole figure is one point per arm.
            ax.annotate(
                str(arm),
                (x, y),
                textcoords="offset points",
                xytext=(9, 3),
                fontsize=7.5,
                color=colour,
                zorder=4,
            )
        ax.axvline(1.0, color="#999999", lw=1, ls="--", zorder=1)
        ax.set_xscale("log")
        # Non-negative by construction, so the scale is log above a floor and
        # linear below it - a symlog axis that dips under zero is drawing a
        # region the quantity cannot reach.
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(bottom=-0.001)
        ax.set_xlabel("median speedup over the sklearn fit  (right is faster)")
        ax.set_ylabel("mean % of the haystack whose verdict changes")
        ax.set_title(f"{kind} path — {g['cell'].nunique()} captures, {n_cases} cases")
        ax.grid(alpha=0.25, zorder=0)
    fig.suptitle(
        "Faster and more faithful is the bottom right.  Dashed line: the incumbent's own speed; "
        "`sklearn_kmeanspp` is the incumbent re-initialised, and is the scale for 'how much does a fit move'.",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_change_ecdf(paired: pd.DataFrame, path: Path) -> None:
    kinds = [k for k in ("sort", "fold") if k in set(paired["kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.4 * len(kinds), 4.4), squeeze=False)
    for ax, kind in zip(axes[0], kinds, strict=True):
        for arm, gg in paired[paired["kind"] == kind].groupby("arm"):
            colour, _ = _style(str(arm))
            v = np.sort(100.0 * gg["d_frac"].to_numpy())
            if v.size == 0:
                continue
            ax.step(np.maximum(v, 1e-4), np.arange(1, v.size + 1) / v.size, where="post", color=colour, label=str(arm))
        ax.set_xscale("log")
        ax.set_xlabel("% of the haystack whose verdict changes (log; left edge is 'none')")
        ax.set_ylabel("fraction of cases at or below")
        ax.set_title(f"{kind} path")
        ax.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("A curve that reaches 1.0 at the left edge changes nothing on any case", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_cost(bench: pd.DataFrame, path: Path) -> None:
    df = bench.copy()
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    # Measured sizes are banded by decade before the median is taken.  Every
    # captured array has its own n, so plotting the raw value draws one vertical
    # spike per dataset instead of a curve - the spikes are the spread *within*
    # a size, which is a different question from how cost scales with size.
    edges = [0, 1_000, 5_000, 20_000, 10**9]
    measured = df[df["resampled"] == 0].copy()
    measured["band"] = pd.cut(measured["n"], bins=edges, right=False)
    resampled = df[df["resampled"] == 1]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for arm, g in df.groupby("arm"):
        colour, marker = _style(str(arm))
        # The incumbent is the line every other line is read against, so it is
        # drawn heavier: at these sizes two other arms sit on top of it.
        width = 3.0 if arm == "baseline" else 1.5
        m = measured[measured["arm"] == arm]
        if not m.empty:
            by_band = m.groupby("band", observed=True).agg(n=("n", "median"), s=("seconds", "median"))
            ax.plot(by_band["n"], 1000 * by_band["s"], marker=marker, color=colour, ls="-", lw=width, label=str(arm))
        r = resampled[resampled["arm"] == arm]
        if not r.empty:
            by_n = r.groupby("n")["seconds"].median().sort_index()
            # Joined to the last measured point so the projection reads as a
            # continuation of the same curve rather than a second series.
            xs = list(by_n.index)
            ys = [1000 * v for v in by_n.to_numpy()]
            if not m.empty:
                xs = [float(by_band["n"].iloc[-1]), *xs]
                ys = [1000 * float(by_band["s"].iloc[-1]), *ys]
            ax.plot(xs, ys, marker=marker, color=colour, ls=":", lw=width, alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scores in the fit")
    ax.set_ylabel("milliseconds per fit (min of 5)")
    ax.set_title("Cost against sample size")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, loc="upper left")
    fig.text(
        0.5,
        0.02,
        "Solid: measured, median over the arrays in each decade band.  Dotted: the same arrays bootstrap-resampled\n"
        "to 20k and 50k - a projection of the SHAPE, which is what drives iteration count, not a measurement.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    analysis = Path(args.analysis)
    out = Path(args.out) if args.out else analysis / "figures"
    out.mkdir(parents=True, exist_ok=True)

    paired = _paired(pd.read_csv(analysis / "gate_cuts.csv"))
    fig_arms(paired, out / "arms.png")
    fig_change_ecdf(paired, out / "change_ecdf.png")
    bench = analysis / "bench.csv"
    if bench.exists():
        fig_cost(pd.read_csv(bench), out / "cost_vs_n.png")
    else:
        print(f"no {bench} - skipping the cost figure")
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

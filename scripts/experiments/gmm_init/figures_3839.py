#!/usr/bin/env python
"""The #3839 report's figures, from the gate frames ``analyze_3839.py`` read.

    python figures_3839.py --analysis <dir>

1. ``tradeoff.png`` - every arm's cost (mean refit iterations, relative to
   ``shipped``) against its distance from the converged answer on the cases
   that had a capped fold (p90 admitted-set move against ``limit``).  The
   three options of the issue sit in three different places on it.
2. ``capped_gap.png`` - each capped fold: how many iterations it needed to
   converge against how far its midpoint moved getting there.  A cap of 200 is
   binding on real migrations, not on noise.
3. ``move_to_limit_ecdf.png`` - on the capped cases, the ECDF of the
   admitted-set distance from ``limit`` per arm; a median hides a tail.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CASE = ["cell", "style", "kind", "case"]
FOLD = [*CASE, "fold"]
COL = "n_admitted_i0"
INK, MUTED = "#222222", "#8a8a85"
#: One hue per option of the issue (dataviz reference palette, fixed order).
OPTION = {
    "shipped": INK,
    "cap": "#2a78d6",  # option 1 by budget
    "ll": "#1baf7a",  # option 1 by tolerance
    "stall": "#eb6834",  # option 2
    "rel": "#eb6834",
    "aitken": "#eb6834",
    "fallback": "#e87ba4",  # option 3
}


#: Arms labelled on the trade-off plot (the rest sit in the cluster at shipped's
#: height and are identified by colour), with label offsets that do not collide.
LABELS = {
    "shipped": (6, -12),
    "fallback": (6, 3),
    "stall1e-3": (6, 3),
    "stall1e-4": (6, 3),
    "ll1e-6": (-10, 8),
    "cap400": (6, 3),
    "cap1000": (-40, -14),
    "cap2000": (-4, 8),
}

#: The ECDF's curves, each with its own line style so no two share colour AND dash.
ECDF_STYLE = {
    "shipped": (INK, "-"),
    "cap400": (OPTION["cap"], ":"),
    "cap1000": (OPTION["cap"], "--"),
    "cap2000": (OPTION["cap"], "-"),
    "stall1e-4": (OPTION["stall"], "-"),
    "fallback": (OPTION["fallback"], "-"),
}


def _color(arm: str) -> str:
    for k, c in OPTION.items():
        if arm.startswith(k):
            return c
    return MUTED


def _style(ax) -> None:  # noqa: ANN001
    ax.grid(True, color="#e6e6e3", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    a = Path(args.analysis)
    figs = a / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    cuts = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(a / "gate3839_cuts.*.csv"))))
    fits = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(a / "gate3839_fits.*.csv"))))

    sc = cuts[cuts["arm"] == "shipped"].set_index(CASE)
    capped = sc[sc["n_unconverged"].fillna(0) > 0].index
    lim = cuts[cuts["arm"] == "limit"].set_index(CASE)[COL]
    anch = fits[fits["provenance"] == "anchored"]
    base_iters = anch[anch["arm"] == "shipped"]["n_iter"].mean()

    # 1. trade-off
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ecdf = {}
    for arm, g in cuts.groupby("arm", sort=False):
        if arm in ("limit", "drv_shipped"):
            continue
        g = g.set_index(CASE)
        mv = ((g[COL] - lim.reindex(g.index)).abs() / g["n"]).reindex(capped)
        ecdf[arm] = np.sort(mv.to_numpy())
        cost = anch[anch["arm"] == arm]["n_iter"].mean() / base_iters
        y = 100 * mv.quantile(0.9)
        ax.scatter(cost, y, s=70, color=_color(arm), zorder=3, marker="s" if arm == "shipped" else "o")
        if arm in LABELS:
            ax.annotate(arm, (cost, y), xytext=LABELS[arm], textcoords="offset points", fontsize=8, color=INK)
    for lab, c in (
        ("option 1: budget", OPTION["cap"]),
        ("option 1: tolerance", OPTION["ll"]),
        ("option 2: stall / relative / Aitken", OPTION["stall"]),
        ("option 3: fallback", OPTION["fallback"]),
        ("shipped", INK),
    ):
        ax.scatter([], [], color=c, s=40, label=lab, marker="s" if lab == "shipped" else "o")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_xlabel("mean refit iterations, relative to shipped (cost)", color=INK)
    ax.set_ylabel("p90 distance from the converged answer\n(% of haystack, cases with a capped fold)", color=INK)
    ax.set_title("Only a bigger budget moves TOWARD the converged fit", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(figs / "tradeoff.png", dpi=150)
    plt.close(fig)

    # 2. capped folds: iterations to converge vs midpoint distance
    s = fits[fits["arm"] == "shipped"].set_index(FOLD)
    lf = fits[fits["arm"] == "limit"].set_index(FOLD)
    j = s.join(lf, rsuffix="_l", how="inner")
    j = j[(j["provenance"] == "anchored") & (j["provenance_l"] == "anchored")]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for conv, c, lab in ((1, MUTED, "converged under shipped"), (0, "#eb6834", "capped at 200 under shipped")):
        g = j[j["converged"] == conv]
        ax.scatter(
            g["n_iter_l"],
            (g["midpoint"] - g["midpoint_l"]).abs().clip(lower=1e-7),
            s=8,
            alpha=0.6,
            color=c,
            label=f"{lab} ({len(g)})",
        )
    for cap in (200, 1000, 2000):
        ax.axvline(cap, color=MUTED, lw=0.8, ls="--")
        ax.text(cap, 2e-7, f" {cap}", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iterations the fold needs to reach 1e-13 (limit)", color=INK)
    ax.set_ylabel("|midpoint(shipped) − midpoint(limit)|", color=INK)
    ax.set_title("The capped folds are slow migrations, not noise at the tolerance", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "capped_gap.png", dpi=150)
    plt.close(fig)

    # 3. ECDF on capped cases
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for arm, (c, ls) in ECDF_STYLE.items():
        if arm not in ecdf:
            continue
        x = np.clip(100 * ecdf[arm], 1e-3, None)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=1.6, ls=ls, label=arm)
    ax.set_xscale("log")
    ax.set_xlabel("admitted-set distance from limit, % of haystack (floored at 0.001)", color=INK)
    ax.set_ylabel(f"share of the {len(capped)} capped cases", color=INK)
    ax.set_title("How far each rule leaves the capped cases from the converged answer", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "move_to_limit_ecdf.png", dpi=150)
    plt.close(fig)
    print(f"wrote {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figure for the #2846 Grid re-measure: what dropping the ordering guard buys.

Two panels, because the result is two facts that only mean something together:
the repair *fires* (fallback rate falls) but what it buys depends on the tilt.
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz palette, validated (categorical slots 1-2; diverging blue<->red)
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"

RESULTS = Path("/exp/sgreenberg/calibration-cut2846/results")
ARMS = [
    ("visual_genome_m/dinov3_patch/max_patch", "production\n(dinov3 x max_patch)"),
    ("visual_genome_m/siglip/whole_image", "control\n(siglip whole_image)"),
]
KEYS = ["arm", "category", "seed", "t"]
rng = np.random.default_rng(2846)


def load() -> pd.DataFrame:
    files = sorted(
        p for p in glob.glob(str(RESULTS / "cells" / "task_*.csv")) if "__sweep" not in p and "__cutdiag" not in p
    )
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    return df[(df.n_votes >= 6) & (df.n_votes <= 20)]


def ci(v: np.ndarray, n: int = 10000) -> tuple[float, float]:
    b = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return tuple(np.percentile(b, [2.5, 97.5]))


def main() -> None:
    r = load()
    fb, eff = [], []
    for arm, arm_label in ARMS:
        for tilt in ("priorfree", "cross"):
            a = r[(r.gmm_variant == f"pooled_gumbel_any_{tilt}") & (r.arm == arm)].set_index(KEYS)
            b = r[(r.gmm_variant == f"pooled_gumbel_{tilt}") & (r.arm == arm)].set_index(KEYS)
            j = a[["cost", "gmm_cut", "cut_fallback"]].join(
                b[["cost", "gmm_cut", "cut_fallback"]], lsuffix="_a", rsuffix="_b", how="inner"
            )
            fb.append(
                dict(
                    label=f"{tilt}\n{arm_label}",
                    old=j.cut_fallback_b.mean(),
                    new=j.cut_fallback_a.mean(),
                )
            )
            m = (j.gmm_cut_a - j.gmm_cut_b).abs() > 1e-9
            d = (j[m].cost_a - j[m].cost_b).values
            lo, hi = ci(d)
            eff.append(
                dict(
                    label=f"{tilt}\n{arm_label}",
                    tilt=tilt,
                    d=d.mean(),
                    lo=lo,
                    hi=hi,
                    fires=m.mean(),
                    n=len(d),
                )
            )
    fb, eff = pd.DataFrame(fb), pd.DataFrame(eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"wspace": 0.42})
    fig.patch.set_facecolor("#fcfcfb")
    y = np.arange(len(fb))[::-1]
    h = 0.34

    # --- Panel A: the guard does come off ---
    ax1.barh(y + h / 2 + 0.02, fb.old, h, color=BLUE, label="`gumbel_*` (#2836, guard on)")
    ax1.barh(y - h / 2 - 0.02, fb.new, h, color=ORANGE, label="`gumbel_any_*` (#2846, guard off)")
    for yi, o, n in zip(y, fb.old, fb.new, strict=True):
        ax1.text(o + 0.012, yi + h / 2 + 0.02, f"{o:.1%}", va="center", fontsize=9.5, color=INK)
        ax1.text(n + 0.012, yi - h / 2 - 0.02, f"{n:.1%}", va="center", fontsize=9.5, color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(fb.label, fontsize=9.5, color=INK)
    ax1.set_xlim(0, 0.82)
    ax1.set_xlabel("share of ramp steps that fell back to the midpoint", fontsize=10, color=INK2)
    ax1.set_title("A · The crossing fires more often", fontsize=12.5, color=INK, loc="left", pad=12)

    # --- Panel B: but what it buys flips sign with the tilt ---
    for _, row in eff.iterrows():
        yi = y[list(eff.label).index(row.label)]
        c = BLUE if row.d < 0 else RED
        ax2.plot([row.lo, row.hi], [yi, yi], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax2.plot([row.d], [yi], "o", ms=9, color=c, mec="#fcfcfb", mew=2, zorder=3)
        ax2.text(
            row.hi + 0.0035,
            yi,
            f"{row.d:+.4f}   (fires on {row.fires:.1%}, n={row.n})",
            va="center",
            fontsize=9.5,
            color=INK,
        )
    ax2.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(eff.label, fontsize=9.5, color=INK)
    ax2.set_xlim(-0.055, 0.105)
    ax2.set_xlabel(
        "paired Δ cost on the steps the repair changes  (95% CI)\n← keeping the fit is better        worse →",
        fontsize=10,
        color=INK2,
    )
    ax2.set_title(
        "B · …and that only helps under the prior-free tilt",
        fontsize=12.5,
        color=INK,
        loc="left",
        pad=12,
    )

    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)

    # Title and legend live above the axes as figure-level elements, because the
    # bars span 20-70% of panel A and carry a value label on the right of each --
    # there is no in-panel position for a legend that does not collide with one.
    fig.suptitle(
        "Dropping #2836's Gumbel-is-the-low-component guard, on real Visual Genome scores",
        fontsize=14,
        color=INK,
        x=0.09,
        ha="left",
        y=1.10,
    )
    fig.legend(
        *ax1.get_legend_handles_labels(),
        frameon=False,
        fontsize=9.5,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.088, 1.045),
    )
    out = RESULTS / "figures" / "cut_gumbel_any_2846.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print("wrote", out)
    print(fb.to_string(index=False))
    print(eff.to_string(index=False))


if __name__ == "__main__":
    main()

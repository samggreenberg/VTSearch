#!/usr/bin/env python3
"""Figures for #4056, from the JSON the probes wrote.

    python figures.py            # regenerate every figure into figures/

Inputs live beside this script under `measurements/`, so the figures rebuild
without the GRID, the pile, or a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

# Validated categorical slots 1, 2 and 3 (worst adjacent CVD dE 9.2 deutan, 27.6
# normal). Slot 3's contrast against the surface is 2.74:1, a WARN the validator
# relieves only with visible labels -- every bar here carries its value.
SIM = "#eb6834"
SHIPPED = "#2a78d6"
MATCHED = "#1baf7a"
BARREN = "#8a8a84"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"
BANDS = ("small", "medium", "large")
EMB = "siglip"


def band_mean(rows: list, band: str, a: str, b: str) -> float:
    sel = [r for r in rows if r["cell"].endswith("@" + band)]
    return sum(r[b] - r[a] for r in sel) / len(sel)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    sim = json.loads((MEAS / f"roster53-{EMB}.json").read_text())
    reb = json.loads((MEAS / f"rebuild-{EMB}.json").read_text())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 4.5), gridspec_kw={"width_ratios": [2.3, 1.0], "wspace": 0.32})

    # --- A. what the simulation said, and what the rebuild did ---------------
    series = [
        ("simulated pool (superseded)", SIM, [band_mean(sim["cells"], b, "ap_shipped", "ap_wide") for b in BANDS]),
        ("rebuilt, as shipped", SHIPPED, [band_mean(reb["cells"], b, "ap_old", "ap_new") for b in BANDS]),
        (
            "rebuilt, size-matched",
            MATCHED,
            [band_mean(reb["cells"], b, "ap_old_matched", "ap_new_matched") for b in BANDS],
        ),
    ]
    w = 0.24
    for k, (label, colour, ys) in enumerate(series):
        xs = [i + (k - 1) * (w + 0.02) for i in range(len(BANDS))]
        ax.bar(xs, ys, width=w, color=colour, label=label, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:+.2f}" if abs(y) >= 0.1 else f"{y:+.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4 if y >= 0 else -11),
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.axhline(0, color=INK, lw=1.0, zorder=4)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([f"@{b}" for b in BANDS], fontsize=10, color=INK)
    ax.set_ylabel("paired $\\Delta$AP,  $C$ = 25 $\\rightarrow$ 53", fontsize=9.5, color=MUTED)
    ax.set_title("The pool empties — and the benchmark barely moves", fontsize=11.5, color=INK, loc="left", pad=10)
    ax.set_ylim(-0.10, 0.46)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    ax.text(
        0.0,
        -0.155,
        f"`{EMB}`; `siglip2_l` agrees to 0.002 AP on the shipped arm and also straddles zero when size-matched",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
        clip_on=False,
    )

    # --- B. why it does not reach: the barren draw is capped -----------------
    rows = reb["cells"]
    import statistics

    n_old = int(statistics.median(r["n_neg_old"] for r in rows))
    n_new = int(statistics.median(r["n_neg_new"] for r in rows))
    barren = 9900  # SCALE_N_NEG, the cap, identical in both builds
    cross = [n_old - barren, n_new - barren]
    xs = [0, 1]
    bx.bar(xs, [barren, barren], width=0.46, color=BARREN, label="barren (capped at SCALE_N_NEG)", zorder=3)
    # 2px surface gap between stacked segments, per the mark spec.
    bx.bar(xs, cross, width=0.46, bottom=[barren + 180] * 2, color=SHIPPED, label="#3667 cross-class", zorder=3)
    for x, c in zip(xs, cross):
        bx.annotate(f"{barren:,}", (x, barren / 2), ha="center", va="center", fontsize=8.5, color=SURFACE, zorder=5)
        bx.annotate(
            f"{c:,}", (x, barren + 180 + c / 2), ha="center", va="center", fontsize=8.5, color=SURFACE, zorder=5
        )
    bx.set_xticks(xs)
    bx.set_xticklabels([f"$|C|$ = 25\n{n_old:,} neg", f"$|C|$ = 53\n{n_new:,} neg"], fontsize=9.5, color=INK)
    bx.set_ylabel("negatives per cell (median)", fontsize=9.5, color=MUTED)
    bx.set_title("...the barren draw is capped", fontsize=11, color=INK, loc="left", pad=10)
    bx.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    bx.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        bx.spines[s].set_visible(False)
    bx.spines["left"].set_color(GRID)
    bx.tick_params(colors=MUTED, length=0)
    bx.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    bx.set_ylim(0, 30000)

    fig.subplots_adjust(bottom=0.21, top=0.86, left=0.075, right=0.97)
    out = FIGS / "widening-empties-the-pool.png"
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

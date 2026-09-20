#!/usr/bin/env python3
"""Figures for #4056, from the JSON the roster-width probe wrote.

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

# Validated categorical slots 1 and 2 (CVD dE 24.7 protan, 33.6 normal), the same
# two the #3986 report uses, so the two studies read as one visual language.
SIGLIP = "#2a78d6"
SIGLIP2 = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
BANDS = ("small", "medium", "large")

#: #3986's headline, the same quantity measured with C held still. Printed as the
#: comparator because the whole point is the two orders of magnitude between them.
REF_3986 = -0.002


def load(stem: str) -> dict | None:
    for name in (f"roster53-{stem}.json", f"roster-{stem}.json"):
        p = MEAS / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def band_dap(d: dict, band: str) -> float:
    rows = [r for r in d["cells"] if r["cell"].endswith("@" + band)]
    return sum(r["ap_wide"] - r["ap_shipped"] for r in rows) / len(rows)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    runs = [(s, load(s)) for s in ("siglip", "siglip2_l")]
    runs = [(s, d) for s, d in runs if d]
    if not runs:
        raise SystemExit(f"no roster measurement JSON under {MEAS}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": [2.45, 1.0], "wspace": 0.30})

    # --- A. the effect, paired, by band ------------------------------------
    w = 0.30
    for k, (stem, d) in enumerate(runs):
        xs = [i + (k - 0.5) * (w + 0.018) for i in range(len(BANDS))]
        ys = [band_dap(d, b) for b in BANDS]
        ax.bar(xs, ys, width=w, color=SIGLIP if k == 0 else SIGLIP2, label=f"`{stem}`".strip("`"), zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"+{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8.5,
                color=INK,
            )
    ax.axhline(0, color=INK, lw=1.0, zorder=4)
    ax.axhline(REF_3986, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=4)
    # The comparator sits ON the zero line, which is the point. Every in-axes
    # placement collided with a bar or a value label, so it reads as a caption
    # under the panel, where the dashed rule above it is the only referent.
    ax.text(
        0.0,
        -0.165,
        "dashed: #3986, the same pool question with $C$ held still  \u2014  $-0.002$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
        clip_on=False,
    )
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([f"@{b}" for b in BANDS], fontsize=10, color=INK)
    ax.set_ylabel("paired $\\Delta$AP,  shipped pool $\\rightarrow$ widened pool", fontsize=9.5, color=MUTED)
    ax.set_title("Widening $C$ makes every cell look better", fontsize=11.5, color=INK, loc="left", pad=10)
    ax.set_ylim(-0.022, 0.47)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK)

    # --- B. the mechanism ---------------------------------------------------
    d = runs[0][1]
    rows = d["cells"]
    e_ship = sum(r["emptiness_shipped"] for r in rows) / len(rows)
    e_wide = sum(r["emptiness_wide"] for r in rows) / len(rows)
    n_s, n_w = d["clean_shipped"], d["clean_wide"]
    c_s, c_w = d["n_classes_shipped"], d["n_classes_wide"]

    bx.bar([0, 1], [e_ship, e_wide], width=0.46, color=[MUTED, SIGLIP], zorder=3)
    # e_wide is exactly 0 by construction. Draw a visible rule on the baseline so
    # the reader sees a MEASURED zero rather than a bar that failed to render.
    bx.plot([1 - 0.23, 1 + 0.23], [0, 0], color=SIGLIP, lw=2.4, solid_capstyle="butt", zorder=5)
    for x, y, n, c in ((0, e_ship, n_s, c_s), (1, e_wide, n_w, c_w)):
        bx.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9.5,
            color=INK,
        )
        bx.annotate(
            f"{n:,} clean",
            (x, 0),
            textcoords="offset points",
            xytext=(0, -26),
            ha="center",
            fontsize=8.5,
            color=MUTED,
        )
    bx.set_xticks([0, 1])
    bx.set_xticklabels([f"$|C|$ = {c_s}", f"$|C|$ = {c_w}"], fontsize=10, color=INK)
    bx.set_ylabel("mean classes of $C$ held\nper pool image", fontsize=9.5, color=MUTED)
    bx.set_title("...because the pool empties", fontsize=11.5, color=INK, loc="left", pad=10)
    bx.set_ylim(0, 1.42)
    bx.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    bx.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        bx.spines[s].set_visible(False)
    bx.spines["left"].set_color(GRID)
    bx.tick_params(colors=MUTED, length=0)

    fig.subplots_adjust(bottom=0.20, top=0.86, left=0.075, right=0.965)
    out = FIGS / "widening-empties-the-pool.png"
    fig.savefig(out, dpi=DPI, facecolor="#fcfcfb")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

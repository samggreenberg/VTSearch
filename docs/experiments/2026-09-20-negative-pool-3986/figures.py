#!/usr/bin/env python3
"""Figures for #3986, from the JSON the trained-head probe wrote.

    python figures.py            # regenerate every figure into figures/

Inputs live beside this script under `measurements/`, so the figures rebuild
without the GRID, the pile, or a GPU.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
FIGS = HERE / "figures"
DPI = 130

# Validated categorical slots 1 and 2 (CVD dE 24.7 protan, 33.6 normal).
BARREN = "#eb6834"
SHIPPED = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
BANDS = ("small", "medium", "large")
#: #3667's headline, measured on vg_scale with the same contrast.
REF_3667 = 1.88


def cooccur_over_barren(row: dict) -> float | None:
    """The 100%-vs-0% co-occurrence contrast, whatever the baseline's mixture.

    The baseline is a mixture -- a fraction w of it co-occurs with another class
    in C -- so `fpr_base = w*fpr_co + (1-w)*fpr_barren` recovers the barren-subset
    FPR exactly. When w is 0 the row already IS the contrast.
    """
    w = row["cooccur_rate_baseline"]
    if w <= 0:
        return row["ratio_cooccur"]
    if w >= 1:
        return None
    barren = (row["fpr_shared"] - w * row["fpr_cooccur"]) / (1 - w)
    return row["fpr_cooccur"] / barren if barren > 0 else None


def by_band(rows: list[dict]) -> dict[str, float]:
    out = {}
    for b in BANDS:
        vals = [v for r in rows if r["cell"].endswith("@" + b) if (v := cooccur_over_barren(r)) is not None]
        out[b] = statistics.mean(vals)
    return out


def fig_shortcut(barren: dict, shipped: dict) -> None:
    """Two panels: the head still learns the shortcut; switching pools no longer pays."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [2.0, 1.0]})

    b_band, s_band = by_band(barren["cells"]), by_band(shipped["cells"])
    xs = range(len(BANDS))
    w = 0.36
    r1 = ax1.bar([x - w / 2 - 0.01 for x in xs], [b_band[b] for b in BANDS], w, color=BARREN)
    r2 = ax1.bar([x + w / 2 + 0.01 for x in xs], [s_band[b] for b in BANDS], w, color=SHIPPED)
    for rects in (r1, r2):
        for rect in rects:
            ax1.annotate(
                f"{rect.get_height():.1f}x",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
                color=INK,
            )
    # Reference lines carry their identity in the legend rather than inline
    # annotations, which collided with the leftmost bar at every y they needed.
    ref_none = ax1.axhline(1.0, color=MUTED, lw=1.0, ls="-", zorder=0)
    ref_3667 = ax1.axhline(REF_3667, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax1.set_xlim(-0.55, 2.55)
    ax1.set_ylim(0, 4.7)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([f"@{b}" for b in BANDS])
    ax1.set_ylabel("FPR ratio, co-occurring vs barren negatives")
    ax1.set_title("The head still learns the shortcut", fontsize=11, color=INK)
    ax1.legend(
        [r1, r2, ref_none, ref_3667],
        [
            "trained on barren pool (pre-#3667)",
            "trained on shipped pool",
            "1.0 = no shortcut",
            f"#3667 on vg_scale, {REF_3667:.1f}x",
        ],
        fontsize=8.5,
        loc="upper right",
    )
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_axisbelow(True)

    d = [barren["d_ap_mean"], shipped["d_ap_mean"]]
    rects = ax2.bar([0, 1], d, 0.5, color=[BARREN, SHIPPED])
    for rect, v in zip(rects, d, strict=True):
        ax2.annotate(
            f"{v:+.3f}",
            (rect.get_x() + rect.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, -13 if v < 0 else 3),
            ha="center",
            fontsize=10,
            color=INK,
        )
    ax2.axhline(0, color=MUTED, lw=1.0)
    ax2.set_ylim(-0.215, 0.013)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["pre-#3667", "shipped"], fontsize=9)
    ax2.set_ylabel("change in AP from a representative pool")
    ax2.set_title("But switching pools no longer pays", fontsize=11, color=INK)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "The scene-clutter shortcut is undiminished on coco_better — "
        "and #3667's fix already absorbed what it cost the benchmark",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "shortcut-already-absorbed.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")
    print("  panel A (co-occurring / barren):")
    for b in BANDS:
        print(f"    @{b:<7} pre-#3667 {b_band[b]:.2f}x   shipped {s_band[b]:.2f}x")
    print(f"  panel B (dAP): pre-#3667 {d[0]:+.3f}   shipped {d[1]:+.3f}")


def main() -> None:
    barren = json.loads((MEAS / "shortcut-siglip-barren.json").read_text())
    shipped = json.loads((MEAS / "shortcut-siglip-shipped.json").read_text())
    fig_shortcut(barren, shipped)


if __name__ == "__main__":
    main()

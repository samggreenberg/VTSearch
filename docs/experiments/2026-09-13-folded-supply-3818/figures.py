#!/usr/bin/env python3
"""The figures for #3818, from `measurements/folded_supply.json` and nothing else.

Deterministic and source-free: redrawing needs no pile, no VG source, no GPU and
no cluster, so a figure and the report's tables cannot disagree.

Usage::

    python figures.py                          # beside this file
    python figures.py --json <file> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

#: One colour per outcome, and the ordering is the argument: the benign one is
#: the muted bulk of every bar, and the two that would be faults are the ones
#: that carry ink.
BENIGN = "#b9c6d2"
SCATTERED = "#2c6fa8"
OVERSIZE = "#d1603d"
CUT = "#8a8a8a"

SPLIT = (("cell_full", BENIGN), ("scattered", SCATTERED), ("oversize", OVERSIZE))


def apportionment(doc: dict, out: Path) -> Path:
    """The 441 beside the population they were drawn from, as shares.

    **Paired, because a share of a selected sample is not a finding on its own.**
    The 441 are undesignated by construction and a scattered image can never be
    designated, so scatter is over-represented among them however the build
    behaves. The control bar carries the same condition -- every undesignated
    queue image holding the class -- so what is left between the two bars is the
    screen and the reviewer. The figure's whole content is that nothing is.
    """
    rows = sorted(doc["classes"], key=lambda r: -r["folded"])
    ctrl = {c["class"]: c for c in doc["control"]}
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    h = 0.38

    for i, r in enumerate(rows):
        c = ctrl[r["class"]]
        for y, counts, n in ((i + h / 2 + 0.02, r, r["folded"]), (i - h / 2 - 0.02, c, c["undesignated"])):
            left = 0.0
            for key, colour in SPLIT:
                w = 100 * counts[key] / n if n else 0.0
                ax.barh(y, w, left=left, height=h, color=colour, zorder=2)
                left += w

    p, t = doc["pooled"], doc["control"]
    tot = {k: sum(c[k] for c in t) for k, _ in SPLIT}
    tot["undesignated"] = sum(c["undesignated"] for c in t)
    y0 = len(rows) + 0.6
    for y, counts, n in ((y0 + h / 2 + 0.02, p, p["folded"]), (y0 - h / 2 - 0.02, tot, tot["undesignated"])):
        left = 0.0
        for key, colour in SPLIT:
            w = 100 * counts[key] / n if n else 0.0
            ax.barh(y, w, left=left, height=h, color=colour, zorder=2, edgecolor="#222", lw=0.8)
            left += w

    labels = [f"{r['class']}  {r['folded']}/{ctrl[r['class']]['undesignated']}" for r in rows]
    ax.set_yticks([*range(len(rows)), y0])
    ax.set_yticklabels([*labels, f"POOLED  {p['folded']}/{tot['undesignated']}"], fontsize=9)
    for lbl in ax.get_yticklabels()[-1:]:
        lbl.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the class's undesignated images (%)")
    ax.set_title(
        f"Why {p['folded']} confirmed positives are not designated\n"
        "upper bar: the confirmed errors   lower bar: every undesignated queue image holding the class",
        fontsize=10,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in SPLIT]
    ax.legend(
        handles,
        ["cell already full (not a fault)", "scatter filter", "missed a band (oversize)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.115),
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )
    ax.grid(axis="x", color="#eee", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = out / "fig_apportionment.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def filters(doc: dict, out: Path) -> Path:
    """Every image the two filters dropped, against the cut that dropped it.

    The question a count cannot answer: 77 near-misses would be an argument for
    moving ``BAND_MAX_INFLATION``, and 77 images whose instances are scattered
    across the frame are the filter working. Each image is one dot at its own
    value, so the reader sees the distribution rather than a summary of it.
    """
    rows = doc["rows"]
    sc = sorted((r["inflation"] for r in rows if r["outcome"] == "scattered"))
    ov = sorted((r["union_area"] for r in rows if r["outcome"] == "oversize"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.55, 1]})

    ax1.plot(range(len(sc)), sc, "o", ms=4.5, color=SCATTERED, zorder=3)
    ax1.axhline(1.5, color=CUT, ls="--", lw=1.2, zorder=2)
    ax1.text(len(sc) * 0.44, 1.53, "BAND_MAX_INFLATION = 1.5", fontsize=8, color=CUT, va="bottom")
    ax1.set_yscale("log")
    ax1.set_yticks([1.5, 2, 3, 5, 10, 20, 50])
    ax1.set_yticklabels(["1.5x", "2x", "3x", "5x", "10x", "20x", "50x"], fontsize=8)
    ax1.set_ylabel("union box / largest instance")
    ax1.set_xlabel(f"the {len(sc)} images the scatter filter dropped, sorted")
    near = sum(1 for x in sc if x < 2.0)
    ax1.set_title(f"scatter: {near} of {len(sc)} are within 2x of the cut", fontsize=10)

    ax2.plot(range(len(ov)), [100 * x for x in ov], "o", ms=5.5, color=OVERSIZE, zorder=3)
    ax2.axhline(80, color=CUT, ls="--", lw=1.2, zorder=2)
    ax2.text(len(ov) * 0.30, 80.4, "MAX_VOTED_AREA = 80%", fontsize=8, color=CUT, va="bottom")
    ax2.set_ylim(75, 102)
    ax2.set_ylabel("union box, % of the frame")
    ax2.set_xlabel(f"the {len(ov)} images that missed every band")
    ax2.set_title("oversize: a box this big is the image", fontsize=10)

    for ax in (ax1, ax2):
        ax.grid(color="#eee", zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path = out / "fig_filters.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def seats(doc: dict, out: Path) -> Path:
    """Supply against seats, one row per cell.

    The context ``cell_full`` has to be read against, and the reason this study
    closes rather than filing a repair: a cell that draws 100 positives from
    2,185 candidates is not short of supply, and nothing a repair could add to it
    would be designated either.
    """
    rows = sorted(doc["cells"], key=lambda c: c["supply"])
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ys = range(len(rows))
    ax.barh(list(ys), [c["supply"] for c in rows], color=BENIGN, height=0.66, zorder=2)
    ax.axvline(rows[0]["seats"], color=OVERSIZE, lw=1.4, zorder=4)
    ax.text(
        rows[0]["seats"] * 1.15,
        len(rows) - 1.2,
        f"SCALE_N_POS = {rows[0]['seats']} seats",
        fontsize=8.5,
        color=OVERSIZE,
    )
    for y, c in zip(ys, rows):
        ax.text(c["supply"] * 1.04, y, f"{c['oversubscription']:.1f}x", va="center", fontsize=7.5, color="#444")
    ax.set_xscale("log")
    ax.set_xlim(60, 4200)
    # Plain counts: a reader comparing a cell's supply against 100 seats should
    # not have to evaluate a power of ten to do it.
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["100", "200", "500", "1,000", "2,000"], fontsize=8.5)
    ax.minorticks_off()
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c["cell"] for c in rows], fontsize=8)
    total = sum(c["supply"] for c in rows)
    ax.set_xlabel("banded candidates available to the cell (log scale)")
    ax.set_title(
        f"Every cell is over-subscribed: {total:,} candidates for {sum(c['seats'] for c in rows):,} seats",
        fontsize=10,
    )
    ax.grid(axis="x", color="#eee", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = out / "fig_oversubscription.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(HERE / "measurements" / "folded_supply.json"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    doc = json.loads(Path(args.json).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (apportionment, filters, seats):
        print(f"wrote {fn(doc, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

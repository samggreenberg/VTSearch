#!/usr/bin/env python3
"""The figures for #3696, from the two measurement JSONs and nothing else.

Deterministic and source-free: it reads `measurements/silence_rate.json` and
`measurements/silence_source.json`, so a figure and the report's tables cannot
disagree, and redrawing after the pass finishes another class needs no pile, no
GPU and no cluster. The source split is drawn only when its file is there, so
the two figures that need the cheap measurement alone still render without the
expensive one.

Usage::

    python figures.py                       # beside this file
    python figures.py --rate <json> --source <json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

#: One colour for a class in the pooled figure and one for a class that is not,
#: because the difference between them is the whole reading of the chart -- a
#: 25% bar is an unreviewed class, not a dirty one.
POOLED = "#2c6fa8"
EXCLUDED = "#c9c9c9"
BOUND = "#d1603d"


def by_class(rate: dict, out: Path) -> Path:
    """Measured rate with its Wilson interval, and the bound, one row a class.

    **Only the classes somebody has voted on.** A class nobody has started has no
    rate and a bound that is just its share of the queue -- 13.7% to 25.9% today,
    which is five times the range everything measured sits in. Plotting them
    together squashes the entire result into the left fifth of the axis to make
    room for thirteen bars that say "not started"; the count belongs in the
    subtitle, where it cannot crowd out what was measured.
    """
    rows = sorted((r for r in rate["classes"] if r["answered"]), key=lambda r: r["rate"])
    unstarted = [r for r in rate["classes"] if not r["answered"]]
    eligible = set(rate["pooled"]["classes"])
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ys = range(len(rows))

    # Scaled to the finished classes. A part-banked class's bound is dominated by
    # what nobody has voted on yet -- `chair` reads 25% at 300 of 841 -- so
    # letting it set the axis would compress every measured class into the left
    # quarter to plot one number that is about the backlog, not about silence.
    limit = max(r["bound"] for r in rows if r["class"] in eligible) * 100 * 1.35

    for y, r in zip(ys, rows):
        lo, hi = r["wilson95"]
        colour = POOLED if r["class"] in eligible else EXCLUDED
        ax.barh(y, r["rate"] * 100, color=colour, height=0.62, zorder=2)
        ax.plot([lo * 100, hi * 100], [y, y], color="#222", lw=1.4, zorder=4)
        # The bound is drawn as a tick rather than a second bar: it is the same
        # quantity read conservatively, not a rival measurement.
        if r["bound"] * 100 <= limit:
            ax.plot([r["bound"] * 100], [y], marker="|", ms=13, color=BOUND, mew=2.2, zorder=5)
        else:
            ax.plot([limit * 0.985], [y], marker=">", ms=7, color=BOUND, zorder=5)
            ax.text(limit * 0.965, y, f"{r['bound']:.0%}", ha="right", va="center", fontsize=8, color=BOUND, zorder=6)

    p = rate["pooled"]
    ax.axvline(p["rate"] * 100, color=POOLED, ls="--", lw=1.1, zorder=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["class"] for r in rows], fontsize=9)
    ax.set_xlabel("VG silent on a class that is really there (% of the class's silent pairs)")
    lo_u = min((r["bound"] for r in unstarted), default=0.0)
    hi_u = max((r["bound"] for r in unstarted), default=0.0)
    ax.set_title(
        f"VG's silence rate, off the exhaustive pass's answers\n"
        f"pooled over {len(eligible)} finished, rule-current classes: "
        f"{p['rate']:.1%} [{p['wilson95'][0]:.1%}, {p['wilson95'][1]:.1%}], bound {p['bound']:.1%}\n"
        f"{len(unstarted)} classes nobody has started yet are off the chart "
        f"(bounds {lo_u:.1%}–{hi_u:.1%}, all of it unreviewed)",
        fontsize=10.5,
    )
    ax.set_xlim(0, limit)
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=POOLED),
        plt.Rectangle((0, 0), 1, 1, color=EXCLUDED),
        plt.Line2D([], [], color="#222", lw=1.4),
        plt.Line2D([], [], color=BOUND, marker="|", ms=11, mew=2.2, ls="none"),
    ]
    ax.legend(
        handles,
        ["in the pooled figure", "unfinished or stale rule", "95% Wilson", "upper bound"],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.95,
    )
    fig.tight_layout()
    path = out / "fig_silence_by_class.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def budget(rate: dict, out: Path) -> Path:
    """Where the bound's headroom comes from, for the pooled nine.

    The point the table cannot make: almost all of the gap between the measured
    rate and the bound is *screening* -- pairs nobody was shown -- and none of it
    is uncertainty about the verdicts that were cast.
    """
    p = rate["pooled"]
    hi = p["wilson95"][1]
    parts = [
        ("confirmed by a human", p["rate"] * 100, POOLED),
        ("sampling (Wilson, to the upper end)", (hi - p["rate"]) * 100, "#7fa8c9"),
        ("below the cut, nobody was shown (#3768)", (p["bound"] - hi) * 100, BOUND),
        ("...if those samples are not pooled", (p["bound_unpooled"] - p["bound"]) * 100, "#e8b4a0"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    left = 0.0
    for label, width, colour in parts:
        ax.barh(0, width, left=left, color=colour, height=0.5, label=f"{label} ({width:.2f} pts)")
        left += width
    ax.set_xlim(0, left * 1.02)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    # No x-label: the axis is percentage points and the title says so, and a
    # label here lands on top of the legend below at every figure height tried.
    ax.set_title(
        "What the bound is made of (percentage points)\n"
        "none of the headroom is doubt about the verdicts that were cast",
        fontsize=10.5,
    )
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=8.5, ncol=2, frameon=False)
    fig.tight_layout()
    path = out / "fig_bound_budget.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def source_split(src: dict, out: Path) -> Path:
    """What the confirmed errors actually are, per class.

    The correction that halves the headline twice over: most of a "silence
    error" is not silence. Drawn as shares of each class's own error count
    rather than as counts, because the question is *what kind* of error the
    class makes and a count would just redraw the class-size ranking.
    """
    rows = sorted(src["classes"], key=lambda r: -(r["coverage"] / max(r["found_present"], 1)))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ys = range(len(rows))
    palette = {"coverage": BOUND, "withheld": "#e0a458", "folded": "#9bb7c9"}
    labels = {
        "coverage": "coverage — VG named nothing (can contaminate a pool)",
        "withheld": "withheld — VG used a name the build refuses (#3605)",
        "folded": "folded — VG used a name the build folds; just not designated",
    }

    for kind in ("coverage", "withheld", "folded"):
        left = [
            sum(r[k] for k in ("coverage", "withheld", "folded")[: ("coverage", "withheld", "folded").index(kind)])
            / max(r["found_present"], 1)
            * 100
            for r in rows
        ]
        width = [r[kind] / max(r["found_present"], 1) * 100 for r in rows]
        ax.barh(list(ys), width, left=left, color=palette[kind], height=0.66, label=labels[kind], zorder=2)

    p = src["pooled"]
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r['class']}  ({r['found_present']})" for r in rows], fontsize=9)
    ax.set_xlabel("share of the class's confirmed errors (%)")
    ax.set_xlim(0, 100)
    ax.set_title(
        f"Most of a “silence error” is not silence\n"
        f"{p['withheld'] + p['folded']} of {p['found_present']} are images VG DID name, so the rate falls "
        f"{p['designation_rate']:.2%} → {p['coverage_rate']:.2%}",
        fontsize=10.5,
    )
    ax.grid(axis="x", color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5, frameon=False)
    fig.tight_layout()
    path = out / "fig_error_source.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", default=str(HERE / "measurements" / "silence_rate.json"))
    ap.add_argument("--source", default=str(HERE / "measurements" / "silence_source.json"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()

    rate = json.loads(Path(args.rate).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = [by_class(rate, out), budget(rate, out)]
    if Path(args.source).exists():
        made.append(source_split(json.loads(Path(args.source).read_text()), out))
    for path in made:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

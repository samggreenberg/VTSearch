#!/usr/bin/env python
"""#3688's figures, from ``flip_3688_ab.py``'s tables: one row per path (Binary, Region, the record).

* ``fig_by_vote.png``: F at every vote, one wrong vote minus none, paired, with a 95% band clustered on the class;
* ``fig_spread.png``: the share of runs one wrong vote moves by more than one (and two) of its cell's seed SDs,
  beside the same share for a clean run against the next seed.

    python flip_3688_figures.py --tables binary=<dir> [--tables region=<dir>] [--tables arm1=<dir>] --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order: the Good flips take 1-4, the Bad flips 5-6, the record 7.
COLOR = {
    "g1": "#2a78d6",
    "g2": "#eb6834",
    "g5": "#1baf7a",
    "g20": "#eda100",
    "b1": "#e87ba4",
    "b5": "#008300",
    "asbuilt": "#6250d6",
}
LABEL = {
    "g1": "1st Good voted Bad",
    "g2": "2nd Good voted Bad",
    "g5": "5th Good voted Bad",
    "g20": "20th Good voted Bad",
    "b1": "1st negative voted Good",
    "b5": "5th negative voted Good",
    "asbuilt": "the record's 349 errors left in",
}
INK, MUTED, GRID = "#3d3d3a", "#73726c", "#e5e4df"
TITLE = {"binary": "Binary (SigLIP)", "region": "Region (SigLIP + DINOv3 patches)", "arm1": "Binary, the real errors"}
PANELS = {
    "binary": (("g1", "g2", "g5", "g20"), ("b1", "b5")),
    "region": (("g1", "g5"), ("b1",)),
    "arm1": (("asbuilt",),),
}


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def by_vote(tables: dict, out: Path) -> None:
    rows = [(stem, sets) for stem in tables for sets in PANELS[stem]]
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.4), sharey=True, squeeze=False,
                             constrained_layout=True)  # fmt: skip
    fig.suptitle("F at every vote, one wrong vote minus none, paired (band: 95%, clustered on the class)",
                 color=INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    for ax, (stem, sets) in zip(axes[0], rows):
        style(ax)
        cur = pd.read_csv(tables[stem] / f"flip_3688_{stem}_curves.csv")
        ax.axhline(0, color=MUTED, linewidth=1)
        for s in sets:
            c = cur[cur["set"] == s]
            if c.empty:
                continue
            ax.fill_between(c["vote"], c["d_F"] - 1.96 * c["se"], c["d_F"] + 1.96 * c["se"], color=COLOR[s],
                            alpha=0.15, linewidth=0)  # fmt: skip
            ax.plot(c["vote"], c["d_F"], color=COLOR[s], linewidth=2, label=LABEL[s])
        ax.set_title(TITLE[stem], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("vote", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    axes[0, 0].set_ylabel("F, wrong vote minus none", color=MUTED, fontsize=9)
    fig.savefig(out / "fig_by_vote.png", dpi=110)
    plt.close(fig)


def spread(tables: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(tables), figsize=(5.2 * len(tables), 3.6), sharey=True, squeeze=False,
                             constrained_layout=True)  # fmt: skip
    fig.suptitle("Runs moved by more than 1 and 2 seed SDs at the stop, against a clean run vs the next seed",
                 color=INK, fontsize=11, x=0.01, ha="left")  # fmt: skip
    for ax, stem in zip(axes[0], tables):
        style(ax)
        s = pd.read_csv(tables[stem] / f"flip_3688_{stem}_summary.csv")
        s = s[s["band"] == "all"]
        ref = s[s["set"] == "reseed"].iloc[0]
        arms = [a for a in COLOR if a in set(s["set"])]
        x = range(len(arms))
        one = [float(s[s["set"] == a]["over_1sd_F_stop"].iloc[0]) for a in arms]
        two = [float(s[s["set"] == a]["over_2sd_F_stop"].iloc[0]) for a in arms]
        ax.bar([i - 0.2 for i in x], one, width=0.38, color=[COLOR[a] for a in arms], edgecolor="white")
        ax.bar([i + 0.2 for i in x], two, width=0.38, color=[COLOR[a] for a in arms], alpha=0.45, edgecolor="white")
        ax.axhline(ref["over_1sd_F_stop"], color=INK, linewidth=1.2, linestyle="--",
                   label=f"new seed, >1 SD: {ref['over_1sd_F_stop']:.0%}")  # fmt: skip
        ax.axhline(ref["over_2sd_F_stop"], color=INK, linewidth=1.2, linestyle=":",
                   label=f"new seed, >2 SD: {ref['over_2sd_F_stop']:.0%}")  # fmt: skip
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
        ax.set_xlim(-0.7, max(len(arms), 3) - 0.3)
        for i, (o, t) in enumerate(zip(one, two)):
            ax.text(i - 0.2, o, f"{o:.0%}", color=INK, fontsize=7, ha="center", va="bottom")
            ax.text(i + 0.2, t, f"{t:.0%}", color=MUTED, fontsize=7, ha="center", va="bottom")
        ax.set_xticks(list(x), [LABEL[a].replace(" voted ", "\nvoted ").replace(" errors ", "\nerrors ") for a in arms],
                      fontsize=7, color=INK)  # fmt: skip
        ax.set_ylim(0, 0.75)
        ax.set_title(f"{TITLE[stem]} (solid: >1 SD, pale: >2 SD)", color=INK, fontsize=10, loc="left")
    axes[0, 0].set_ylabel("share of runs", color=MUTED, fontsize=9)
    fig.savefig(out / "fig_spread.png", dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", action="append", required=True, help="STEM=DIR")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {k: Path(v) for k, _, v in (t.partition("=") for t in args.tables)}
    by_vote(tables, args.out)
    spread(tables, args.out)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""#3546 figures, from the CSVs ``analyze_acqcut_3546.py`` wrote.

    python figures_3546.py --data ANALYSIS_DIR --out FIGURES_DIR

* ``rules.png`` - every rule against today's cut (line - 4), ±2 SE, per preset:
  Goods found by vote 150, AP at vote 150, and the objective after the check.
* ``curves.png`` - the objective over votes per preset: today's cut, the target
  precision 0.5 that ships, and sampling at the line.
* ``hard_picks.png`` - the share of Hard picks that are Good, per rule and preset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BETAS = (0.25, 1.0, 4.0)
BTAG = {0.25: "b025", 1.0: "b1", 4.0: "b4"}
BETA_LABEL = {0.25: "beta 1/4", 1.0: "beta 1", 4.0: "beta 4"}
RULE_LABEL = {
    "ctl": "today: line - 4",
    "tp50": "target precision 0.5 (ships)",
    "tp25": "target precision 0.25",
    "line": "at the line",
    "off2": "line - 2",
    "off6": "line - 6",
    "pin985": "fixed 98.5th-percentile pin",
    "inc0": "old origin: Inclusion 0 - 4",
}
RULES = [r for r in RULE_LABEL if r != "ctl"]


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def rules(data: Path, out: Path) -> None:
    p = pd.read_csv(data / "paired.csv")
    reads = (
        ("goods", "Goods found by vote 150"),
        ("AP, vote 150", "AP at vote 150"),
        ("objective, after the check", "objective after the check"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (read, title) in zip(axes, reads, strict=True):
        for j, beta in enumerate(BETAS):
            q = p[(p.read == read) & (p.beta == beta)].set_index("rule")
            for i, r in enumerate(RULES):
                if r not in q.index:
                    continue
                y = i + (j - 1) * 0.25
                ax.errorbar(
                    q.loc[r, "delta"],
                    y,
                    xerr=2 * q.loc[r, "se"],
                    fmt="o",
                    color=SERIES[j],
                    ms=4,
                    capsize=2,
                    lw=1.3,
                    label=BETA_LABEL[beta] if i == 0 else None,
                )
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(RULES)))
        ax.set_yticklabels([RULE_LABEL[r] for r in RULES], fontsize=8)
        ax.set_title(f"{title}: rule − today", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("right of 0: the rule is better", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(
        "Target precision 0.5 finds more Goods and ranks better at every preset, at no cost to the objective",
        fontsize=10,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "rules.png", dpi=130)
    plt.close(fig)


def curves(data: Path, out: Path) -> None:
    c = pd.read_csv(data / "curves.csv").set_index("t")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, beta in zip(axes, BETAS, strict=True):
        for i, r in enumerate(("ctl", "tp50", "line")):
            col = f"{r}_{BTAG[beta]}"
            if col in c.columns:
                ax.plot(c.index, c[col], color=SERIES[i], lw=1.6, label=RULE_LABEL[r])
        ax.set_title(f"{BETA_LABEL[beta]}: the objective over votes", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("votes", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)


def hard_picks(data: Path, out: Path) -> None:
    lv = pd.read_csv(data / "levels.csv")
    lv["rule"] = lv["arm"].str.rsplit("_", n=1).str[0]
    lv["btag"] = lv["arm"].str.rsplit("_", n=1).str[1]
    order = ["ctl", *RULES]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.26
    for j, beta in enumerate(BETAS):
        q = lv[lv.btag == BTAG[beta]].set_index("rule").reindex(order)
        xs = [i + (j - 1) * width for i in range(len(order))]
        ax.bar(xs, q["hard picks Good"], width=width - 0.02, color=SERIES[j], label=BETA_LABEL[beta])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([RULE_LABEL[r] for r in order], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("share of Hard picks that are Good", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "hard_picks.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rules(args.data, args.out)
    curves(args.data, args.out)
    hard_picks(args.data, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Figures for #3911, from ``measurements/`` alone.

python figures.py    # writes fig_rankers.png, fig_budget.png, fig_shortlist.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAS = HERE / "measurements"
SOURCES = ("spods", "staver", "tobacco800")
BUDGETS = (4096, 8192, 16384)


def _rows(name: str) -> list[dict]:
    return list(csv.DictReader((MEAS / name).open(encoding="utf-8")))


def _mean_ap(rows: list[dict], method: str, source: str | None = None) -> float:
    sel = [float(r["ap"]) for r in rows if r["method"] == method and (source is None or r["source"] == source)]
    return float(np.mean(sel)) if sel else float("nan")


def fig_rankers() -> None:
    sift = _rows("sift_rank_s_16384.csv")
    splg = _rows("splg_rank_s_2048.csv")
    arms = [
        ("SigLIP", sift, "siglip", "#9cc3de"),
        ("VLAD (~5.9k kp)", sift, "vlad", "#d9a3a3"),
        ("SP+LG, every page", splg, "splg", "#8a5a1f"),
        ("SIFT ~5.9k kp, every page", sift, "sift", "#2f6f9f"),
    ]
    groups = ("all", *SOURCES)
    x = np.arange(len(groups))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (label, rows, method, colour) in enumerate(arms):
        ys = [_mean_ap(rows, method, None if g == "all" else g) for g in groups]
        ax.bar(x + (i - 1.5) * width, ys, width, color=colour, label=label)
    counts = {g: len({r["class_id"] for r in sift if g == "all" or r["source"] == g}) for g in groups}
    ax.set_xticks(x, [f"{g} ({counts[g]})" for g in groups])
    ax.set_ylabel("mean AP, tier s, headline pool")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_rankers.png", dpi=150)


def fig_budget() -> None:
    budgets = json.loads((MEAS / "sift_budgets.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ap = [_mean_ap(_rows(f"sift_rank_s_{b}.csv"), "sift") for b in BUDGETS]
    ax.plot(BUDGETS, ap, marker="o", color="#2f6f9f", label="SIFT, every page")
    for b, a in zip(BUDGETS, ap):
        kp = budgets[str(b)]
        ax.annotate(
            f"AP {a:.2f}\nmedian {kp['median']:,} kp, p90 {kp['p90']:,}",
            (b, a),
            textcoords="offset points",
            xytext=(0, -30),
            ha="center",
            fontsize=7,
        )
    ax.axhline(_mean_ap(_rows("sift_rank_s_16384.csv"), "siglip"), color="#9cc3de", ls="--", lw=1, label="SigLIP")
    ax.set_xscale("log", base=2)
    ax.set_xlim(2800, 24000)
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.set_xlabel("SIFT max_features per page (2 MP detection cap applies)")
    ax.set_ylabel("mean AP, tier s")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_budget.png", dpi=150)


def fig_shortlist() -> None:
    rows = _rows("sift_rank_s_16384.csv")
    ks = (100, 500, 1000)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for stage1, colour in (("siglip", "#2f6f9f"), ("vlad", "#d08c3c")):
        ys = [_mean_ap(rows, f"{stage1}{k}_sift") for k in ks]
        ax.plot(
            ks,
            ys,
            marker="o",
            color=colour,
            label=f"{stage1.upper() if stage1 == 'vlad' else 'SigLIP'} top K, SIFT verifies",
        )
    ax.axhline(_mean_ap(rows, "sift"), color="#333333", ls="--", lw=1, label="SIFT, every page")
    ax.axhline(_mean_ap(rows, "siglip"), color="#9cc3de", ls=":", lw=1, label="SigLIP alone")
    ax.set_xlabel("shortlist size K (of ~4,900 pages)")
    ax.set_ylabel("mean AP")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_shortlist.png", dpi=150)


if __name__ == "__main__":
    fig_rankers()
    fig_budget()
    fig_shortlist()

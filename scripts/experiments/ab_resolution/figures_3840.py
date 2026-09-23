#!/usr/bin/env python
"""The #3840 report's figures, from ``resolution_3840.py``'s tables.

    python figures_3840.py --analysis <dir>

1. ``se_vs_cells.png`` - the curve itself: SE of the paired mean against the
   number of paired cells, one line per arm pair, the σ/√k law drawn through
   each, the subsampled points on it, the validation grid's observed SE at 399
   against its pre-registered interval, and the δ a grid of that size resolves
   read off the right-hand axis labels.
2. ``sigma_vs_divergence.png`` - why σ is what it is: each pair's σ against the
   median vote at which its two trajectories stop agreeing.
3. ``env_sigma.png`` - σ per environment per pair, with each environment's cell
   cost, which is what a reallocation spends.
4. ``steps.png`` - SE at equal compute when every trajectory is cut at T steps.
5. ``cell_deltas.png`` - every cell's Δ for #3825's pair, by environment: the
   heavy tail the mean sits on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Fixed categorical order (the dataviz reference palette), assigned by pair, never cycled.
COLORS = {
    "native-sklearn": "#2a78d6",
    "ll1e-3-baseline": "#eb6834",
    "ll1e-6-baseline": "#1baf7a",
    "ll1e-8-baseline": "#eda100",
    "ll1e-6-ll1e-8": "#e87ba4",
    "baseline-native": "#7a7a74",
    "v:ll1e-8-baseline": "#008300",
}
INK, MUTED = "#222222", "#8a8a85"


def _style(ax) -> None:  # noqa: ANN001
    ax.grid(True, color="#e6e6e3", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def fig_curve(a: Path, figs: Path) -> None:
    pairs = pd.read_csv(a / "pairs.csv")
    curve = pd.read_csv(a / "curve.csv")
    val = json.loads((a / "validation.json").read_text())
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    k = np.logspace(np.log10(6), np.log10(3000), 100)
    for _, p in pairs.iterrows():
        if p["pair"] not in COLORS or p["kind"] == "validation":
            continue
        c = COLORS[p["pair"]]
        ax.plot(k, p["sd"] / np.sqrt(k), color=c, lw=1.4, alpha=0.8, label=f"{p['pair']}  (σ={p['sd']:.2g})")
        g = curve[curve["pair"] == p["pair"]]
        ax.scatter(g["k"], g["se_true"], color=c, s=22, zorder=3)
    if "validation" in val:
        v = val["validation"]
        lo, hi = val["predicted_se_90"]
        ax.errorbar(
            [v["n"]],
            [val["predicted_se"]],
            yerr=[[val["predicted_se"] - lo], [hi - val["predicted_se"]]],
            fmt="none",
            ecolor=MUTED,
            elinewidth=6,
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            [v["n"]],
            [v["se"]],
            marker="D",
            s=60,
            color=COLORS["v:ll1e-8-baseline"],
            zorder=4,
            label=f"validation grid, observed (n={v['n']})",
        )
    for d in (0.002, 0.004, 0.01):
        ax.axhline(d / 2, color=MUTED, lw=0.8, ls="--")
        ax.text(3000, d / 2, f"  resolves δ={d:g}", va="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("paired cells", color=INK)
    ax.set_ylabel("SE of the paired mean Δcost", color=INK)
    ax.set_title("A/B resolution against grid size (dots: subsampled; lines: σ/√k)", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "se_vs_cells.png", dpi=150)
    plt.close(fig)


def fig_divergence(a: Path, figs: Path) -> None:
    p = pd.read_csv(a / "pairs.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, r in p.iterrows():
        c = COLORS.get(r["pair"], INK)
        ax.scatter(r["median_divergence_step"], r["sd"], s=60, color=c, zorder=3)
        ax.annotate(
            r["pair"],
            (r["median_divergence_step"], r["sd"]),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("median step at which the two arms' thresholds first differ", color=INK)
    ax.set_ylabel("σ of the per-cell Δcost", color=INK)
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.set_title(
        "Every arm pair parts by step 7 and lands at σ 0.03-0.07; only a late-parting pair is quiet",
        fontsize=9,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(figs / "sigma_vs_divergence.png", dpi=150)
    plt.close(fig)


def fig_env(a: Path, figs: Path) -> None:
    e = pd.read_csv(a / "env.csv")
    envs = sorted(e["arm"].unique())
    pairs = [p for p in COLORS if p in set(e["pair"])]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    width = 0.8 / len(pairs)
    for i, p in enumerate(pairs):
        g = e[e["pair"] == p].set_index("arm").reindex(envs)
        ax.bar(np.arange(len(envs)) + i * width, g["sd"], width=width * 0.9, color=COLORS[p], label=p)
    cost = e.groupby("arm")["cell_seconds"].first().reindex(envs)
    labels = [
        f"{x.replace('/whole_image', '/whole').replace('visual_genome_m', 'vg_m')}\n{c:.0f} s/cell"
        for x, c in zip(envs, cost)
    ]
    ax.set_xticks(np.arange(len(envs)) + 0.4 - width / 2)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("σ of Δcost within the environment", color=INK)
    ax.set_title("Where the A/B's noise lives, and what a cell there costs", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "env_sigma.png", dpi=150)
    plt.close(fig)


def fig_steps(a: Path, figs: Path) -> None:
    s = pd.read_csv(a / "steps.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p, g in s.groupby("pair"):
        if p not in COLORS or p == "baseline-native":
            continue
        ax.plot(
            g["T"],
            g["se_same_cost"] / g[g["T"] == 100]["se_same_cost"].iloc[0],
            marker="o",
            ms=4,
            lw=1.6,
            color=COLORS[p],
            label=p,
        )
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xlabel("steps each trajectory is run for (T)", color=INK)
    ax.set_ylabel("SE at equal compute, relative to T=100", color=INK)
    ax.set_title(
        "Shorter trajectories buy resolution per cell-hour (of an earlier-window estimand)", fontsize=9, color=INK
    )
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "steps.png", dpi=150)
    plt.close(fig)


def fig_cells(a: Path, figs: Path) -> None:
    f = a / "cell_deltas.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    envs = sorted(d["arm"].unique())
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    rng = np.random.default_rng(0)
    for i, env in enumerate(envs):
        for grp, c, off in (
            ("0913", COLORS["ll1e-8-baseline"], -0.15),
            ("validation", COLORS["v:ll1e-8-baseline"], 0.15),
        ):
            g = d[(d["arm"] == env) & (d["set"] == grp)]
            ax.scatter(
                i + off + rng.uniform(-0.08, 0.08, len(g)),
                g["delta"],
                s=9,
                color=c,
                alpha=0.7,
                label=("seeds 0-1 (2026-09-13)" if grp == "0913" else "seeds 2-8 (validation)") if i == 0 else None,
            )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(
        [x.replace("visual_genome_m", "vg_m").replace("/whole_image", "/whole") for x in envs],
        fontsize=7,
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("per-cell Δcost (ll1e-8 − baseline)", color=INK)
    ax.set_title("Every cell of #3825's pair: the mean sits on a heavy two-sided tail", fontsize=10, color=INK)
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "cell_deltas.png", dpi=150)
    plt.close(fig)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    a = Path(args.analysis)
    figs = a / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for f in (fig_curve, fig_divergence, fig_env, fig_steps, fig_cells):
        f(a, figs)
        print(f"{f.__name__} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

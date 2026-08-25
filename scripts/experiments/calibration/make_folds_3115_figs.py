"""Figures for the fold combine-rule study (#3115) and its fold-count sibling.

Built from the same per-step cell CSVs the report's tables come from, so a
figure and a table cannot disagree.  Four things the tables cannot show:

* **The shape over the axis the user spends.**  A deep-regime mean is one number
  per arm; what a user has is a vote budget, and a crossover between two rules
  is exactly what an average across it hides.
* **How unlike the mean a single run is.**  Averaged over categories and seeds
  every rule descends smoothly.  Individual trajectories plateau and spike, and
  the spread is usually the real finding.
* **The axis this study's mechanism runs on** - the fold count.  The combine
  rule cannot matter at K<3 by construction; whether it starts mattering *at* 3,
  saturates, or grows with K is the whole question.
* **The contamination channel's exposure.**  A robustness result means nothing
  without the rate at which the hazard it is robust to actually occurs.

Usage::

    python make_folds_3115_figs.py --results /exp/$USER/calibration-folds-3115/results \\
        --out docs/experiments/calibration-fold-combine/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

import folds_combine_3115 as fc  # noqa: E402
from analyze_folds_2897 import fold_frame, load_cells  # noqa: E402

#: One colour per rule, fixed across every figure so a colour means the same
#: thing everywhere in the report.  The pooled control is deliberately the
#: neutral dark line - it is the thing every other rule is being read against.
COLOR = {
    "xcal": "#333333",
    "tmean": "#3B6EA8",
    "tmedian": "#7FA8D0",
    "qmean": "#D08428",
    "qmedian": "#E0B278",
    "anchored": "#3E8A6E",
    "anchored_qmedian": "#8CC0AC",
}
LABEL = {
    "xcal": "pooled (shipped)",
    "tmean": "tmean (score space)",
    "tmedian": "tmedian",
    "qmean": "qmean (quantile space)",
    "qmedian": "qmedian",
    "anchored": "anchored (production rule)",
    "anchored_qmedian": "anchored, qmedian",
}
DPI = 130


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.png")


def fig_regret_over_votes(v: pd.DataFrame, out: Path, k: int) -> None:
    """Paired delta vs the pooled control over the vote axis, one line per rule.

    The figure that answers "what do I get after 20 clicks?".  Paired at the
    step, so the band is the standard error of a *difference* and not of two
    independently noisy levels.
    """
    w = v[v["k"] == k]
    if w.empty:
        return
    for voting, g in w.groupby("voting", observed=True):
        base = g[g["arm"] == fc.POOLED].set_index([*fc.STEP_KEYS])["regret"]
        base = base[~base.index.duplicated()]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        drawn = 0
        for arm in ("tmean", "tmedian", "qmean", "qmedian", "anchored", "anchored_qmedian"):
            a = g[g["arm"] == arm].set_index([*fc.STEP_KEYS])["regret"]
            a = a[~a.index.duplicated()]
            j = pd.concat([a.rename("a"), base.rename("b")], axis=1, join="inner")
            if j.empty:
                continue
            j = j.reset_index()
            j["d"] = j["a"] - j["b"]
            # Bin the vote axis so each point averages a comparable number of
            # trajectories; raw steps are far too sparse at the deep end.
            j["bin"] = (j["t"] // 10) * 10
            s = j.groupby("bin")["d"].agg(["mean", "sem", "size"])
            s = s[s["size"] >= 5]
            if s.empty:
                continue
            ax.plot(s.index, s["mean"], color=COLOR[arm], lw=1.6, label=LABEL[arm])
            ax.fill_between(
                s.index, s["mean"] - s["sem"], s["mean"] + s["sem"], color=COLOR[arm], alpha=0.18, linewidth=0
            )
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        ax.axhline(0.0, color=COLOR["xcal"], lw=1.2, ls="--", label=LABEL["xcal"])
        ax.set_xlabel("votes cast")
        ax.set_ylabel(f"paired $\\Delta$ regret vs pooled (K={k})")
        ax.set_title(f"{voting} voting: combine rule vs pooling, K={k}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        _save(fig, out, f"regret_over_votes_{voting}_k{k}")


def fig_per_run(v: pd.DataFrame, out: Path, k: int, arm: str = "qmean") -> None:
    """The same delta, one line per run - because the mean hides the spread.

    A rule that is worth 0.01 on average and swings +-0.15 run to run is a
    different proposition from one that is worth 0.01 on every run, and no
    averaged panel can tell them apart.
    """
    w = v[v["k"] == k]
    if w.empty:
        return
    for voting, g in w.groupby("voting", observed=True):
        base = g[g["arm"] == fc.POOLED].set_index([*fc.STEP_KEYS])["regret"]
        a = g[g["arm"] == arm].set_index([*fc.STEP_KEYS])["regret"]
        base, a = base[~base.index.duplicated()], a[~a.index.duplicated()]
        j = pd.concat([a.rename("a"), base.rename("b")], axis=1, join="inner")
        if j.empty:
            continue
        j = j.reset_index()
        j["d"] = j["a"] - j["b"]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for _key, run in j.groupby(["env", "category", "seed"], observed=True):
            run = run.sort_values("t")
            ax.plot(run["t"], run["d"], color=COLOR[arm], lw=0.6, alpha=0.30)
        s = j.groupby((j["t"] // 10) * 10)["d"].mean()
        ax.plot(s.index, s.to_numpy(), color="#111111", lw=2.0, label="mean over runs")
        ax.axhline(0.0, color="#888888", lw=1.0, ls="--")
        ax.set_xlabel("votes cast")
        ax.set_ylabel(f"paired $\\Delta$ regret, {LABEL[arm]} vs pooled")
        ax.set_title(f"{voting} voting: every run, K={k} (mean in black)")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        _save(fig, out, f"per_run_{voting}_{arm}_k{k}")


def fig_over_k(t: pd.DataFrame, out: Path, deep_min: int) -> None:
    """The axis the mechanism runs on: does the combine rule matter more with K?

    The pooling argument predicts a *growing* gap - a pooled quantile estimates
    the quantile of a mixture of K half-trained models, and that mixture widens
    with K.  A flat curve refutes that mechanism whatever the level says.
    """
    deep = t[t["window_hi"] >= deep_min]
    if deep.empty:
        return
    for voting, g in deep.groupby("voting", observed=True):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        drawn = 0
        for name, sub in g.groupby("contrast", observed=True):
            s = sub.groupby("k", observed=True).agg(d=("d_regret", "mean"), se=("se_regret", "mean")).sort_index()
            if s.empty:
                continue
            ax.errorbar(s.index, s["d"], yerr=s["se"], marker="o", ms=3.5, lw=1.4, capsize=2.5, label=name)
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        ax.axhline(0.0, color="#888888", lw=1.0, ls="--")
        ax.axvspan(0.5, fc.COLLAPSE_BELOW_K - 0.5, color="#000000", alpha=0.05, lw=0)
        ax.text(
            1.0,
            ax.get_ylim()[1],
            " mean == median here",
            fontsize=7,
            va="top",
            color="#666666",
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("fold count K")
        ax.set_ylabel(f"paired $\\Delta$ regret (votes >= {deep_min})")
        ax.set_title(f"{voting} voting: each contrast over the fold count")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.grid(alpha=0.25, lw=0.5)
        _save(fig, out, f"contrasts_over_k_{voting}")


def fig_degenerate(degen: pd.DataFrame, out: Path) -> None:
    """The contamination channel's exposure: how often a fold has no cut to give.

    Without this the median arms' result is unreadable: "the robust rule changed
    nothing" means one thing when a quarter of steps drop a fold and something
    else entirely when none do.
    """
    if degen.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for (voting, arm), g in degen.groupby(["voting", "arm"], observed=True):
        s = g.groupby("k", observed=True)["any_dropped_rate"].mean().sort_index()
        ax.plot(s.index, s.to_numpy(), marker="o", ms=3.5, lw=1.4, label=f"{voting} / {arm}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fold count K")
    ax.set_ylabel("fraction of steps with >=1 fold dropped")
    ax.set_title("Degenerate folds: the hazard the median combine is robust to")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    _save(fig, out, "degenerate_folds_over_k")


def fig_threshold_sd(v: pd.DataFrame, out: Path, deep_min: int) -> None:
    """sd(threshold) across seeds per K, one line per rule (#3116's instrument).

    Whether a combine rule makes the cut *steadier* is a different question from
    whether it makes it better, and it is the one the regret decomposition
    provably cannot answer.  Steps carrying a single seed contribute nothing: an
    sd of one observation is undefined, not zero.
    """
    w = v[v["window_hi"] >= deep_min]
    if w.empty:
        return
    per_step = (
        w.groupby(["voting", "arm", "k", "env", "category", "t"], observed=True)["threshold"]
        .agg(sd="std", n="count")
        .reset_index()
    )
    per_step = per_step[(per_step["n"] >= 2) & per_step["sd"].notna()]
    if per_step.empty:
        print("  sd(threshold): no step carries >=2 seeds; figure skipped")
        return
    for voting, g in per_step.groupby("voting", observed=True):
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for arm, a in g.groupby("arm", observed=True):
            if arm not in COLOR:
                continue
            s = a.groupby("k", observed=True)["sd"].mean().sort_index()
            ax.plot(s.index, s.to_numpy(), marker="o", ms=3.5, lw=1.4, color=COLOR[arm], label=LABEL[arm])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("fold count K")
        ax.set_ylabel("sd(threshold) across seeds")
        ax.set_title(f"{voting} voting: does the rule make the cut steadier?")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
        _save(fig, out, f"threshold_sd_over_k_{voting}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True, help="the run's results/ dir")
    ap.add_argument("--out", type=Path, required=True, help="figures/ dir to write")
    ap.add_argument("--deep-min", type=int, default=100)
    ap.add_argument("--k", type=int, default=4, help="fold count for the vote-axis figures")
    a = ap.parse_args(argv)

    v = fold_frame(load_cells(a.results / "cells"))
    if v.empty:
        print("no fold rows found")
        return 1
    agg = a.results / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    t = fc.contrast_table(v, agg)
    degen = fc.degenerate_table(v, agg)

    fig_regret_over_votes(v, a.out, a.k)
    fig_per_run(v, a.out, a.k)
    fig_over_k(t, a.out, a.deep_min)
    fig_degenerate(degen, a.out)
    fig_threshold_sd(v, a.out, a.deep_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

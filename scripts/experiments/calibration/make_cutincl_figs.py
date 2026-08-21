"""Figures for the #2865 cut-rule x Inclusion sweep.

Every figure reads the analyzer's own loader and the analyzer's own aggregates,
so a figure and the table beside it cannot disagree.  (`make_linhead_figs.py`
learned this the expensive way: a private "drop the tagged rows" filter in a
figure script loaded 1.41 M rows where the analyzer saw 42 k.)

What the tables cannot show, and why this script exists:

* **The shape of the knob.** The decision table is one interval per (arm, env,
  k).  Whether a rule is uniformly a little worse or fine in the middle and
  wrong at the ends is the whole question, and only the curve says which.
* **What a slider drag actually does.** `knob_yield` is a count; the admitted
  fraction against `k`, beside the *fold quantile* against `k`, is what
  separates "the rule never moves the cut" from "the cut moves and the haystack
  has nothing there" - two failures with different fixes.
* **What one run looks like.** Averaged over categories and seeds every arm
  behaves; individual cells do not.
* **Where `q_tilt`'s free parameter should sit** - the axis that candidate's
  mechanism runs on, and the reason it is a candidate rather than a constant.

Usage:

    python make_cutincl_figs.py --out docs/experiments/cut-inclusion-2865/figures \\
        --results /expscratch/$USER/cut-incl-2865/results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["path.simplify"] = True
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 130

#: One colour per cut rule, fixed across every figure so a colour means one
#: thing.  `q_tilt`'s five step sizes share a hue and are separated by dash.
COLOR = {
    "mid": "#b2182b",
    "mid_tilt": "#2166ac",
    "rate": "#1a9850",
    "cross_tilt": "#762a83",
    "q_tilt": "#e08214",
}
LABEL = {
    "mid": "mid — candidate 4, the inclusion-blind null",
    "mid_tilt": "mid_tilt — the shipped rule (incumbent)",
    "rate": "rate — candidate 2 as described",
    "cross_tilt": "cross_tilt — candidate 2 as written (keeps the priors)",
    "q_tilt": "q_tilt — candidate 3, fixed quantile shift",
}
RULE_ORDER = ("mid", "mid_tilt", "rate", "cross_tilt", "q_tilt")
#: Dash pattern per `q_tilt` step size, so the five expansions stay separable.
QDASH = {0.005: (1, 2), 0.01: (2, 2), 0.02: (), 0.04: (5, 2), 0.08: (7, 2, 1, 2)}


#: `fold_anchored_w<kappa>_<rule>_<combine>[_s<step>]`.  Both the weight and the
#: rule can contain characters that a naive `split("_")` mis-counts (`w0.3`,
#: `mid_tilt`, `q_tilt`), so the prefix is stripped by pattern, not by position.
_ARM_RE = re.compile(r"^fold_anchored_w[^_]+_(?P<rule>.+?)_(?:qmean|qmedian)(?:_s(?P<step>[0-9.]+))?$")


def _rule_of(arm: str) -> str:
    """`fold_anchored_w0.3_<rule>_qmean[_s<step>]` -> `<rule>`."""
    m = _ARM_RE.match(arm)
    if not m:
        raise ValueError(f"unrecognised arm name: {arm!r}")
    return m.group("rule")


def _step_of(arm: str) -> float | None:
    """The `q_tilt` step size an arm carries, or ``None``."""
    m = _ARM_RE.match(arm)
    step = m.group("step") if m else None
    return float(step) if step else None


def _style(rule: str, step: float | None) -> dict:
    st: dict = {"color": COLOR.get(rule, "0.4"), "lw": 1.7}
    if rule == "q_tilt" and step is not None and not np.isnan(step):
        dash = QDASH.get(round(float(step), 4), ())
        if dash:
            st["dashes"] = dash
        st["lw"] = 1.3
    return st


def _envs(df: pd.DataFrame) -> list[str]:
    """Region-voting environments first: they are the ones fusion pays on."""
    envs = sorted(str(e) for e in df["env"].unique())
    return sorted(envs, key=lambda e: (0 if "dinov3" in e else 1, e))


def _short(env: str) -> str:
    ds, emb, style = env.split("/")
    mode = "region" if "dinov3" in emb else "binary"
    return f"{ds} × {emb}\n{style} ({mode} voting)"


# --------------------------------------------------------------------------


def fig_regret_vs_k(reg: pd.DataFrame, tol: float, out: Path) -> str:
    """(a) The decision number, as a curve: paired regret across the knob."""
    envs = sorted(reg["env"].unique(), key=lambda e: (0 if "dinov3" in e else 1, e))
    fig, axes = plt.subplots(1, len(envs), figsize=(4.1 * len(envs), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, env in zip(axes, envs):
        sub = reg[reg["env"] == env]
        for arm, g in sub.groupby("arm"):
            g = g.sort_values("inclusion_k")
            rule = _rule_of(str(arm))
            step = _step_of(str(arm))
            st = _style(rule, step)
            ax.plot(g["inclusion_k"], g["d_regret"], **st)
            ax.fill_between(g["inclusion_k"], g["ci_lo"], g["ci_hi"], color=st["color"], alpha=0.10, lw=0)
        ax.axhline(0, color="k", lw=1)
        ax.axhline(tol, color="0.4", ls="--", lw=1)
        ax.axhline(-tol, color="0.4", ls="--", lw=1)
        ax.axvline(0, color="0.7", ls=":", lw=1)
        ax.set_title(_short(env), fontsize=9)
        ax.set_xlabel("Inclusion k")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("regret − incumbent (rate scale)")
    handles = [plt.Line2D([], [], color=COLOR[r], lw=1.8, label=LABEL[r]) for r in RULE_ORDER]
    fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Paired regret against the shipped rule at each stop of the Inclusion knob, mean ± bootstrap CI over cells.\n"
        f"Dashed lines are the ±{tol} harm tolerance. Below 0 favours the challenger; the incumbent is the zero line.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    p = out / "fig1_regret_vs_k.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_knob_liveness(df: pd.DataFrame, out: Path) -> str:
    """(b) What a slider drag does — to the cut, and to the admitted set."""
    envs = _envs(df)
    fig, axes = plt.subplots(2, len(envs), figsize=(4.1 * len(envs), 7.0), sharex=True)
    axes = np.atleast_2d(axes)
    for col, env in enumerate(envs):
        sub = df[df["env"] == env]
        for (arm, rule), g in sub.groupby(["arm", "cut_rule"], observed=True):
            step = _step_of(str(arm))
            st = _style(str(rule), step)
            a = g.groupby("inclusion_k")["admitted_frac"].mean()
            q = g.groupby("inclusion_k")["fold_quantile"].mean()
            axes[0, col].plot(a.index, a.values, **st)
            axes[1, col].plot(q.index, q.values, **st)
        axes[0, col].set_title(_short(env), fontsize=9)
        for row in (0, 1):
            axes[row, col].grid(alpha=0.25)
            axes[row, col].axvline(0, color="0.7", ls=":", lw=1)
        axes[1, col].set_xlabel("Inclusion k")
    axes[0, 0].set_ylabel("admitted fraction of the test pool")
    axes[1, 0].set_ylabel("combined fold quantile the cut sits at")
    handles = [plt.Line2D([], [], color=COLOR[r], lw=1.8, label=LABEL[r]) for r in RULE_ORDER]
    fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Top: what the user sees the slider do. Bottom: what the rule did to the cut.\n"
        "A rule that moves in the bottom row and not the top has been defeated by the haystack, not by its own maths.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out / "fig2_knob_liveness.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_regret_vs_votes(df: pd.DataFrame, ks: tuple[int, ...], out: Path) -> str:
    """The headline metric over the axis the user spends: votes."""
    envs = _envs(df)
    fig, axes = plt.subplots(len(ks), len(envs), figsize=(4.1 * len(envs), 3.2 * len(ks)), sharex=True)
    axes = np.atleast_2d(axes)
    for row, k in enumerate(ks):
        for col, env in enumerate(envs):
            ax = axes[row, col]
            sub = df[(df["env"] == env) & (df["inclusion_k"] == k)]
            for (arm, rule), g in sub.groupby(["arm", "cut_rule"], observed=True):
                step = _step_of(str(arm))
                if str(rule) == "q_tilt" and step is not None and round(step, 4) != 0.02:
                    continue  # one q_tilt line here; its step axis is fig5
                st = _style(str(rule), None)
                m = g.groupby("t")["regret_rate"].mean()
                se = g.groupby("t")["regret_rate"].sem()
                ax.plot(m.index, m.values, **st)
                ax.fill_between(m.index, m - se, m + se, color=st["color"], alpha=0.15, lw=0)
            ax.grid(alpha=0.25)
            if row == 0:
                ax.set_title(_short(env), fontsize=9)
            if row == len(ks) - 1:
                ax.set_xlabel("votes (labels spent)")
            if col == 0:
                ax.set_ylabel(f"regret at k={k:+d}\n(rate scale)")
    handles = [plt.Line2D([], [], color=COLOR[r], lw=1.8, label=LABEL[r]) for r in RULE_ORDER]
    fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Regret over the ramp at three stops of the knob, mean ± SE over categories and seeds.\n"
        "`q_tilt` is shown at its shipped placeholder step (0.02); the step axis is its own figure.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out / "fig3_regret_vs_votes.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_per_run(df: pd.DataFrame, k: int, arms: list[str], out: Path) -> str:
    """The same metric, one line per run — the spread the mean hides."""
    fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    sub = df[df["inclusion_k"] == k]
    for ax, arm in zip(axes, arms):
        a = sub[sub["arm"] == arm]
        rule = _rule_of(arm)
        for _, g in a.groupby("cell", observed=True):
            g = g.sort_values("t")
            ax.plot(g["t"], g["regret_rate"], color=COLOR.get(rule, "0.4"), lw=0.4, alpha=0.20)
        m = a.groupby("t")["regret_rate"].mean()
        ax.plot(m.index, m.values, color="k", lw=2.0, label="mean")
        ax.set_title(LABEL.get(rule, rule), fontsize=8)
        ax.set_xlabel("votes")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel(f"regret at k={k:+d} (rate scale)")
    fig.suptitle(
        f"One line per run at k={k:+d}, all environments pooled. The mean is the black line; "
        "the spread around it is what a per-arm average cannot say.",
        fontsize=10,
    )
    fig.tight_layout()
    p = out / "fig4_per_run.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_qtilt_step(reg: pd.DataFrame, live: pd.DataFrame, out: Path) -> str:
    """Candidate 3's free parameter — the axis its mechanism runs on."""
    q = reg[reg["arm"].str.contains("_q_tilt_")].copy()
    if q.empty:
        return ""
    q["step"] = q["arm"].map(_step_of)
    ql = live[live["arm"].str.contains("_q_tilt_")].copy()
    ql["step"] = ql["arm"].map(_step_of)
    envs = sorted(q["env"].unique(), key=lambda e: (0 if "dinov3" in e else 1, e))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for env in envs:
        s = q[q["env"] == env].groupby("step")["d_regret"].mean()
        axes[0].plot(s.index, s.values, marker="o", ms=4, lw=1.6, label=_short(env).replace("\n", " "))
        y = ql[ql["env"] == env].set_index("step")["knob_yield"].sort_index()
        axes[1].plot(y.index, y.values, marker="o", ms=4, lw=1.6)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("q_tilt step (fold quantile per inclusion step)")
    axes[0].set_ylabel("regret − incumbent (rate scale), pooled over k")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log")
    axes[1].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[1].set_xlabel("q_tilt step")
    axes[1].set_ylabel("knob yield (distinct admitted sets / stops)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        "The free parameter candidate 3 costs. Left: what the step buys or spends against the shipped rule.\n"
        "Right: how much of the slider it delivers. A rule that only wins at one hand-picked step is not a result.",
        fontsize=10,
    )
    fig.tight_layout()
    p = out / "fig5_qtilt_step.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_knob_yield(live: pd.DataFrame, n_ks: int, out: Path) -> str:
    """The binding constraint: how many distinct answers the slider has."""
    envs = sorted(live["env"].unique(), key=lambda e: (0 if "dinov3" in e else 1, e))
    arms = sorted(live["arm"].unique(), key=lambda a: (RULE_ORDER.index(_rule_of(a)), a))
    fig, ax = plt.subplots(figsize=(max(9.0, 1.1 * len(arms)), 4.4))
    width = 0.8 / len(envs)
    x = np.arange(len(arms))
    for i, env in enumerate(envs):
        s = live[live["env"] == env].set_index("arm")["distinct_admitted"].reindex(arms)
        ax.bar(x + i * width, s.values, width, label=_short(env).replace("\n", " "))
    ax.axhline(n_ks, color="0.3", ls="--", lw=1)
    ax.text(0.01, n_ks, f" every stop distinct ({n_ks})", va="bottom", fontsize=7.5, color="0.3")
    ax.axhline(1.0, color="#b2182b", ls=":", lw=1)
    ax.text(0.01, 1.0, " inert (1)", va="bottom", fontsize=7.5, color="#b2182b")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(
        [a.replace("fold_anchored_w0.3_", "").replace("_qmean", "") for a in arms], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("distinct admitted sets across the knob")
    ax.set_ylim(0, n_ks * 1.12)
    ax.grid(alpha=0.25, axis="y")
    # Below the axes, not inside it: every bar here is near the top of the
    # range, so an in-axes legend covers the arms it is labelling.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    fig.suptitle(
        f"How many different answers dragging the slider produces, out of {n_ks} stops, averaged over steps and cells.",
        fontsize=10,
    )
    fig.tight_layout()
    p = out / "fig6_knob_yield.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def write_examples(df: pd.DataFrame, incumbent: str, out: Path) -> str:
    """The literal rows behind the rates: one real slider drag, per environment.

    Every aggregate here is a count of admitted sets, and a reader has to be
    able to check one.  So for each environment this prints one actual cell at
    one actual step - the threshold each rule chose at each stop of the knob and
    how many of the test pool it admitted - with the *most* and *least* live
    cell of that environment named, since the mean of a bimodal liveness
    distribution describes no cell at all.

    The cell shown is the one whose incumbent knob yield is the environment's
    median, so it is representative rather than flattering.
    """
    lines = ["# What one slider drag actually did (#2865)", ""]
    deep = df[df["n_votes"] >= 100]
    for env in _envs(deep):
        sub = deep[deep["env"] == env]
        inc = sub[sub["arm"] == incumbent]
        # Knob yield per (cell, step) for the incumbent, then the median cell.
        yield_by_cell = (
            inc.groupby(["cell", "t"], observed=True)["n_admitted"].nunique().groupby("cell", observed=True).mean()
        )
        yield_by_cell = yield_by_cell[yield_by_cell.notna()].sort_values()
        if yield_by_cell.empty:
            continue
        cell = str(yield_by_cell.index[len(yield_by_cell) // 2])
        worst, best = str(yield_by_cell.index[0]), str(yield_by_cell.index[-1])
        cs = sub[sub["cell"] == cell]
        t = int(cs["t"].max())
        cs = cs[cs["t"] == t]
        n_test = int(cs["n_test"].iloc[0])
        lines += [
            f"## `{env}`",
            "",
            f"Cell `{cell}` at step t={t} ({int(cs['n_votes'].iloc[0])} votes, test pool {n_test} items).",
            "Chosen as the environment's **median** cell by the incumbent's knob yield;",
            f"the least live cell here is `{worst}` ({yield_by_cell.iloc[0]:.1f} distinct sets)",
            f"and the most live is `{best}` ({yield_by_cell.iloc[-1]:.1f}).",
            "",
            "Each row is one stop of the Inclusion slider; each cell is `threshold -> items admitted`.",
            "",
        ]
        arms = [a for a in sorted(cs["arm"].unique(), key=lambda a: (RULE_ORDER.index(_rule_of(str(a))), str(a)))]
        header = (
            "| k | " + " | ".join(str(a).replace("fold_anchored_w0.3_", "").replace("_qmean", "") for a in arms) + " |"
        )
        lines += [header, "|" + "---|" * (len(arms) + 1)]
        for k in sorted(cs["inclusion_k"].unique()):
            row = [f"| {int(k):+d} "]
            for a in arms:
                r = cs[(cs["arm"] == a) & (cs["inclusion_k"] == k)]
                if r.empty:
                    row.append("| — ")
                    continue
                row.append(f"| {float(r['cut_threshold'].iloc[0]):.3g} → {int(r['n_admitted'].iloc[0])} ")
            lines.append("".join(row) + "|")
        lines.append("")
    p = out / "examples_slider.md"
    p.write_text("\n".join(lines))
    return p.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Figures for the #2865 cut-rule x inclusion sweep.")
    ap.add_argument("--results", default=None, help="Run's results dir (default: the study env's).")
    ap.add_argument("--out", required=True, help="Directory the PNGs are written to.")
    ap.add_argument("--votes-ks", default="-3,0,3", help="Knob stops the vote-axis figure shows.")
    args = ap.parse_args()

    import common  # noqa: PLC0415 - after argparse so --help works without the study env

    common.setup_env()
    import analyze_cutincl as A  # noqa: PLC0415

    results = Path(args.results) if args.results else common.RESULTS
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = A.load_cutincl(results / "cells")
    if df.empty:
        print("no cut-inclusion rows")
        return 1
    deep = df[df["n_votes"] >= A.DEEP_VOTES_MIN]
    print(f"deep rows (>= {A.DEEP_VOTES_MIN} votes): {len(deep):,} of {len(df):,}")
    reg = pd.read_csv(results / "agg" / "cutincl_regret_vs_incumbent.csv")
    live = pd.read_csv(results / "agg" / "cutincl_liveness.csv")
    n_ks = int(df.groupby(["arm", "cell", "t"], observed=True)["inclusion_k"].size().max())

    made = [
        fig_regret_vs_k(reg, A.HARM_TOLERANCE, out),
        fig_knob_liveness(deep, out),
        fig_regret_vs_votes(df, tuple(int(k) for k in args.votes_ks.split(",")), out),
        fig_per_run(df, 3, [A.incumbent_arm(), "fold_anchored_w0.3_mid_qmean", "fold_anchored_w0.3_rate_qmean"], out),
        fig_qtilt_step(reg, live, out),
        fig_knob_yield(live, n_ks, out),
        write_examples(df, A.incumbent_arm(), out),
    ]
    for name in made:
        if name:
            print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

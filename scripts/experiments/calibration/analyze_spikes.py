"""#2847: do the MLP-era cost spikes survive today's stack?

Issue #2847 reran the #2790 threshold study on COCO ``cat`` with a SigLIP2
whole-image embedder and found single steps where the operating cost jumps to
0.25-0.68 while the **oracle** cost for the same ranking stays flat near 0.05 -
a threshold failure, not a ranking failure.  Since then the head became linear
(#2790/#2809) and the threshold became the fold-anchored GMM cut
(#2852/#2861/#2865).

This analyzer consumes the 2x2 that separates those two causes:

===============  =========  ================  ================================
arm              head       threshold         role
===============  =========  ================  ================================
``A_mlp_xcal``   mlp        conformal only    the #2847-era configuration
``B_mlp_fused``  mlp        fold-anchored     threshold change alone
``C_lin_xcal``   linear     conformal only    head change alone
``D_lin_fused``  linear     fold-anchored     **today's production**
===============  =========  ================  ================================

**Arm A is the positive control.**  A run in which arm A does not spike cannot
distinguish "we fixed it" from "this harness never showed it", so the report
leads with A's incidence and refuses a verdict if it is zero.

**Pairing.** Both knobs steer acquisition (the blended threshold feeds
Autopilot's Hard pick), so the four arms are four different trajectories and are
*not* step-paired.  Every comparison here pairs at the ``(category, seed)``
level on one summary number per trajectory - the statistical unit is a
trajectory, never a step (steps within a run are strongly autocorrelated).

Writes ``agg/*.csv``, ``spike_summary.json``, ``figures/*.png`` and
``REPORT_spikes.md`` under ``$CALIB_EXP/analysis``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import main_frame_files  # noqa: E402

try:  # scipy is in the grid venv, but the analyzer must not die without it
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # noqa: BLE001
    _wilcoxon = None

#: The 2x2 above is the default.  ``SPIKE_ARMS`` re-points this analyzer at a
#: different arm set that answers the same question with the same pre-registered
#: spike rules - #2808 reuses it for the head/epoch contrast.  The FIRST arm is
#: always the positive control and the LAST is the arm treated as production,
#: so a caller that reorders them changes which arm the no-verdict guard reads.
ARMS: tuple[str, ...] = tuple(
    a for a in os.environ.get("SPIKE_ARMS", "A_mlp_xcal B_mlp_fused C_lin_xcal D_lin_fused").replace(",", " ").split()
)
ARM_LABEL: dict[str, str] = {
    "A_mlp_xcal": "mlp + conformal (#2847-era)",
    "B_mlp_fused": "mlp + fold-anchored",
    "C_lin_xcal": "linear + conformal",
    "D_lin_fused": "linear + fold-anchored (production)",
    # #2808: head x training-budget, all on the fold-anchored (production) cut.
    "C_mlp": "mlp, 200 epochs (reference)",
    "A_shipped": "linear, 200 epochs / patience 10 (production, early-stopped)",
    "B_converged": "linear, 2000 epochs / no early-stop (converged)",
}
if not ARMS:
    raise SystemExit("SPIKE_ARMS is set but empty - refusing to analyze zero arms")
CONTROL_ARM = os.environ.get("SPIKE_CONTROL_ARM") or ARMS[0]
PRODUCTION_ARM = os.environ.get("SPIKE_PRODUCTION_ARM") or ARMS[-1]

# --- Pre-registered spike rules -------------------------------------------
#: Steps before this are the cold start - every arm in #2847's figure humps
#: there, and it is a different phenomenon (no model yet) from the mid-run
#: blips the issue is about.  Reported separately, never mixed in.
WARM_T = int(os.environ.get("SPIKE_WARM_T", "20"))
#: A "deep spike": absolute cost this high *and* this far above the oracle for
#: the same ranking.  Calibrated to the figure, whose spikes are 0.25/0.65/0.68
#: against an oracle near 0.05 - i.e. deliberately conservative, so a run that
#: merely wobbles does not count.
DEEP_COST = float(os.environ.get("SPIKE_DEEP_COST", "0.25"))
DEEP_EXCESS = float(os.environ.get("SPIKE_DEEP_EXCESS", "0.20"))
#: A "local jump": a step that leaps this far above its own trailing median.
#: Catches the same events without an absolute scale, so a category whose costs
#: all sit low is not exempted by construction.
JUMP_WINDOW = int(os.environ.get("SPIKE_JUMP_WINDOW", "5"))
JUMP_DELTA = float(os.environ.get("SPIKE_JUMP_DELTA", "0.15"))

OUT = Path(os.environ.get("SPIKE_OUT", str(common.EXP / "analysis")))

#: Figure resolution.  These are read at full width in the report and the
#: published artifact, and the four-panel comparison only works if a single
#: spike is legible against its neighbours - so they are rendered at print
#: density rather than screen density.  The repo's ``check-added-large-files``
#: cap was raised to 2 MB to accommodate them (it exists to keep datasets and
#: model weights out of git, not to ration figure quality).
FIG_DPI = int(os.environ.get("SPIKE_FIG_DPI", "200"))


# --- loading ---------------------------------------------------------------
def _blank(s: pd.Series) -> pd.Series:
    """True where a tag column is empty/NaN - i.e. the arm's own base row."""
    return s.isna() | (s.astype(str).str.strip().isin(("", "nan", "None")))


#: The base row's ``pool_variant``.  Whole-image styles emit ``"max"`` here (not
#: blank) - the re-pool variants #2781 added are ``topk``/``pnorm``, and only the
#: raw-patch tree arm emits them.  Filtering on "blank" instead drops *every*
#: row, which is a silent empty analysis, so the accepted set is explicit.
BASE_POOL_VARIANTS = ("", "max")


def load_arm(arm_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Concatenate one arm's cell CSVs, keeping only its base rows.

    Returns ``(frame, provenance)``; provenance counts the files read, the
    unreadable ones, and the zero-byte ones, because an analysis that silently
    drops cells is how a disk incident becomes a wrong verdict.
    """
    files = main_frame_files(arm_dir / "cells")
    files = [f for f in files if "__" not in f.name]  # skip __sweep / __cutdiag
    frames, bad, empty, headless = [], [], [], []
    for f in files:
        if f.stat().st_size == 0:
            empty.append(f.name)
            continue
        try:
            fr = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            bad.append(f.name)
            continue
        # A header-only cell is not a failure: the simulator emits a row only
        # once it has at least one good AND one bad vote, so a rare category
        # whose 100 votes never turned up a positive legitimately writes none.
        # That is the extreme of the positive-starvation regime this study is
        # about, so it is counted and reported rather than silently dropped -
        # and it differs per arm, which is why paired tests lose those cells.
        if fr.empty:
            headless.append(f.name)
            continue
        frames.append(fr)
    prov = {
        "n_files": len(files),
        "n_read": len(frames),
        "unreadable": bad,
        "zero_byte": empty,
        "no_positive_found": headless,
    }
    if not frames:
        return pd.DataFrame(), prov
    df = pd.concat(frames, ignore_index=True)
    prov["n_rows_all"] = int(len(df))
    for col in ("gmm_variant", "schedule"):
        if col in df.columns:
            df = df[_blank(df[col])]
    if "pool_variant" in df.columns:
        pv = df["pool_variant"].fillna("").astype(str).str.strip()
        df = df[pv.isin(BASE_POOL_VARIANTS)]
    prov["n_rows"] = int(len(df))
    # A base-row filter that keeps nothing is a bug, not a result.
    if prov["n_rows_all"] and not prov["n_rows"]:
        raise SystemExit(f"{arm_dir}: base-row filter kept 0 of {prov['n_rows_all']} rows - check tag columns")
    return df.reset_index(drop=True), prov


def load_all(results_root: Path) -> tuple[pd.DataFrame, dict]:
    parts, prov = [], {}
    for arm in ARMS:
        df, p = load_arm(results_root / arm)
        prov[arm] = p
        if not df.empty:
            df = df.copy()
            df["arm"] = arm
            parts.append(df)
    if not parts:
        return pd.DataFrame(), prov
    return pd.concat(parts, ignore_index=True), prov


# --- per-trajectory summaries ---------------------------------------------
def trajectory_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(arm, dataset, embedder, category, seed)`` trajectory."""
    keys = ["arm", "dataset", "embedder", "category", "seed"]
    keys = [k for k in keys if k in df.columns]
    out = []
    for key, g in df.groupby(keys, dropna=False):
        g = g.sort_values("t")
        cost = g["cost"].to_numpy(dtype=float)
        orc = g["oracle_cost"].to_numpy(dtype=float)
        t = g["t"].to_numpy(dtype=float)
        warm = t >= WARM_T
        cold = ~warm
        excess = cost - orc

        # Local-jump rule: leap above the trailing median of the previous
        # JUMP_WINDOW steps.  Only defined once a full window exists.
        jump = np.full(cost.shape, np.nan)
        for i in range(JUMP_WINDOW, len(cost)):
            jump[i] = cost[i] - float(np.median(cost[i - JUMP_WINDOW : i]))
        ojump = np.full(orc.shape, np.nan)
        for i in range(JUMP_WINDOW, len(orc)):
            ojump[i] = orc[i] - float(np.median(orc[i - JUMP_WINDOW : i]))

        deep = warm & (cost >= DEEP_COST) & (excess >= DEEP_EXCESS)
        # Parenthesised deliberately: `&` binds tighter than `>=` in Python, so
        # the unbracketed form silently compares a bool/float mix.
        jumped = warm & (np.nan_to_num(jump, nan=-np.inf) >= JUMP_DELTA)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=False))
        rec.update(
            n_steps=int(len(g)),
            n_good_final=int(g["n_good"].iloc[-1]) if "n_good" in g.columns else -1,
            realized_prevalence=(
                float(g["realized_prevalence"].iloc[-1]) if "realized_prevalence" in g.columns else np.nan
            ),
            n_warm=int(warm.sum()),
            # Headline incidence
            has_deep=bool(deep.any()),
            n_deep=int(deep.sum()),
            has_jump=bool(jumped.any()),
            n_jump=int(jumped.sum()),
            # Continuous endpoints (more power than a binary flag)
            max_cost_warm=float(np.nanmax(cost[warm])) if warm.any() else np.nan,
            max_excess_warm=float(np.nanmax(excess[warm])) if warm.any() else np.nan,
            mean_excess_warm=float(np.nanmean(excess[warm])) if warm.any() else np.nan,
            final_cost=float(cost[-1]) if len(cost) else np.nan,
            # Smoothness, and the ranking control: if the oracle jumps too, the
            # step is not a threshold failure at all.
            max_jump_cost=float(np.nanmax(jump[warm])) if warm.any() and np.isfinite(jump[warm]).any() else np.nan,
            max_jump_oracle=float(np.nanmax(ojump[warm])) if warm.any() and np.isfinite(ojump[warm]).any() else np.nan,
            # Cold start, reported separately - not part of the #2847 question.
            max_cost_cold=float(np.nanmax(cost[cold])) if cold.any() else np.nan,
        )
        out.append(rec)
    return pd.DataFrame(out)


def spike_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Every step that trips the deep rule, with the diagnostics for *why*."""
    keys = [k for k in ("arm", "dataset", "embedder", "category", "seed") if k in df.columns]
    rows = []
    for _, g in df.groupby(keys, dropna=False):
        g = g.sort_values("t")
        m = (g["t"] >= WARM_T) & (g["cost"] >= DEEP_COST) & ((g["cost"] - g["oracle_cost"]) >= DEEP_EXCESS)
        if m.any():
            rows.append(g[m])
    if not rows:
        return pd.DataFrame()
    keep = [
        *keys,
        "t",
        "n_good",
        "n_bad",
        "phase",
        "cost",
        "fpr",
        "fnr",
        "oracle_cost",
        "threshold",
        "xcal_threshold",
        "gmm_cut",
        "threshold_provenance",
        "degenerate",
        "auroc",
        "average_precision",
    ]
    out = pd.concat(rows, ignore_index=True)
    return out[[c for c in keep if c in out.columns]]


# --- statistics ------------------------------------------------------------
def paired_vs(traj: pd.DataFrame, metric: str, arm_a: str, arm_b: str) -> dict:
    """Wilcoxon on ``(category, seed)``-paired trajectory summaries.

    The pair is a *cell*, not a step: arm A and arm D ran different trajectories
    on the same category+seed, so the pairing controls the environment and the
    exemplar draw but nothing downstream of the first vote.
    """
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in traj.columns]
    a = traj[traj["arm"] == arm_a].set_index(keys)[metric]
    b = traj[traj["arm"] == arm_b].set_index(keys)[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joined.empty:
        return {"n_pairs": 0}
    d = joined["b"] - joined["a"]
    res = {
        "n_pairs": int(len(joined)),
        f"{arm_a}_median": float(joined["a"].median()),
        f"{arm_b}_median": float(joined["b"].median()),
        "median_delta": float(d.median()),
        "mean_delta": float(d.mean()),
        "frac_b_lower": float((d < 0).mean()),
    }
    if _wilcoxon is not None and (d != 0).any():
        try:
            res["p_wilcoxon"] = float(_wilcoxon(joined["a"], joined["b"]).pvalue)
        except Exception:  # noqa: BLE001
            pass
    return res


def mcnemar_incidence(traj: pd.DataFrame, arm_a: str, arm_b: str, flag: str = "has_deep") -> dict:
    """Discordant-pair test on a binary per-trajectory flag."""
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in traj.columns]
    a = traj[traj["arm"] == arm_a].set_index(keys)[flag]
    b = traj[traj["arm"] == arm_b].set_index(keys)[flag]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if j.empty:
        return {"n_pairs": 0}
    n01 = int(((~j["a"].astype(bool)) & j["b"].astype(bool)).sum())  # only b spikes
    n10 = int((j["a"].astype(bool) & (~j["b"].astype(bool))).sum())  # only a spikes
    out = {
        "n_pairs": int(len(j)),
        f"{arm_a}_rate": float(j["a"].astype(bool).mean()),
        f"{arm_b}_rate": float(j["b"].astype(bool).mean()),
        "only_a": n10,
        "only_b": n01,
    }
    n = n01 + n10
    if n:
        # Exact binomial (two-sided) on the discordant pairs.
        from math import comb  # noqa: PLC0415

        k = min(n01, n10)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2.0**n)
        out["p_exact"] = float(min(1.0, 2 * tail))
    return out


# --- figures ---------------------------------------------------------------
#: One distinct hue per seed.  A cycle shorter than the seed count silently
#: draws two trajectories in the same colour, which reads as one run that
#: teleports - exactly the artefact this figure exists to rule out.
SEED_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
)


def _plot_category(df: pd.DataFrame, arm: str, category: str, ax) -> None:
    g = df[(df["arm"] == arm) & (df["category"] == category)]
    seeds = sorted(g["seed"].unique())
    if len(seeds) > len(SEED_COLORS):
        raise SystemExit(f"{len(seeds)} seeds but only {len(SEED_COLORS)} colours - extend SEED_COLORS")
    for i, s in enumerate(seeds):
        gs = g[g["seed"] == s].sort_values("t")
        c = SEED_COLORS[i]
        ax.plot(gs["t"], gs["cost"], color=c, lw=1.2, label=f"seed {s}")
        ax.plot(gs["t"], gs["oracle_cost"], color=c, lw=0.9, ls="--", alpha=0.4)
        # Ring the steps the deep rule actually flags, so the eye and the
        # incidence table are reading the same events.
        m = (gs["t"] >= WARM_T) & (gs["cost"] >= DEEP_COST) & ((gs["cost"] - gs["oracle_cost"]) >= DEEP_EXCESS)
        if m.any():
            ax.scatter(gs["t"][m], gs["cost"][m], s=22, facecolors="none", edgecolors=c, lw=1.1, zorder=5)
    n_deep = int(
        ((g["t"] >= WARM_T) & (g["cost"] >= DEEP_COST) & ((g["cost"] - g["oracle_cost"]) >= DEEP_EXCESS)).sum()
    )
    ax.set_title(f"{ARM_LABEL.get(arm, arm)}  -  {n_deep} deep-spike steps", fontsize=9)
    ax.set_xlabel("total annotations t")
    ax.set_ylabel("cost (FPR+FNR)")
    ax.grid(alpha=0.25, ls=":")


def make_figures(df: pd.DataFrame, traj: pd.DataFrame, outdir: Path, category: str = "cat") -> list[str]:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    have = df[df["category"] == category]
    if not have.empty:
        # Fig 1: the issue's own figure, one panel per arm, shared y so the
        # arms are visually comparable rather than each auto-scaled to itself.
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
        for ax, arm in zip(axes.ravel(), ARMS, strict=False):
            _plot_category(have, arm, category, ax)
        axes.ravel()[0].legend(fontsize=7, ncol=2)
        fig.suptitle(
            f"coco: {category} - cost vs t (solid) with oracle_cost (dashed), siglip2/whole",
            fontsize=11,
        )
        fig.tight_layout()
        p = outdir / f"fig1_{category}_arms.png"
        fig.savefig(p, dpi=FIG_DPI)
        plt.close(fig)
        made.append(p.name)

        # Fig 2: production alone, styled like the issue's figure so the two
        # can be put side by side.
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        _plot_category(have, PRODUCTION_ARM, category, ax)
        ax.legend(fontsize=8)
        ax.set_title(f"coco: {category} - cost vs t, TODAY'S PRODUCTION (linear + fold-anchored)")
        fig.tight_layout()
        p = outdir / f"fig2_{category}_production.png"
        fig.savefig(p, dpi=FIG_DPI)
        plt.close(fig)
        made.append(p.name)

    # Fig 3: incidence + magnitude by arm across every category.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    order = [a for a in ARMS if a in set(traj["arm"])]
    inc = [100 * traj[traj["arm"] == a]["has_deep"].mean() for a in order]
    axes[0].bar(range(len(order)), inc, color="#4c72b0")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("% of trajectories with a deep spike")
    axes[0].set_title(f"deep spike: cost>={DEEP_COST}, excess>={DEEP_EXCESS}, t>={WARM_T}", fontsize=9)
    axes[0].grid(alpha=0.25, axis="y", ls=":")

    data = [traj[traj["arm"] == a]["max_excess_warm"].dropna().to_numpy() for a in order]
    axes[1].boxplot(data, tick_labels=order, showfliers=True)
    axes[1].set_ylabel("max (cost - oracle_cost), t>=%d" % WARM_T)
    axes[1].set_title("worst-step regret per trajectory", fontsize=9)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].grid(alpha=0.25, axis="y", ls=":")

    d1 = [traj[traj["arm"] == a]["max_jump_cost"].dropna().to_numpy() for a in order]
    d2 = [traj[traj["arm"] == a]["max_jump_oracle"].dropna().to_numpy() for a in order]
    pos = np.arange(len(order))
    axes[2].boxplot(
        d1, positions=pos - 0.17, widths=0.3, showfliers=False, patch_artist=True, boxprops={"facecolor": "#c44e52"}
    )
    axes[2].boxplot(
        d2, positions=pos + 0.17, widths=0.3, showfliers=False, patch_artist=True, boxprops={"facecolor": "#8c8c8c"}
    )
    axes[2].set_xticks(pos)
    axes[2].set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    axes[2].set_ylabel("max jump above trailing median")
    axes[2].set_title("cost (red) vs oracle_cost (grey) - the ranking control", fontsize=9)
    axes[2].grid(alpha=0.25, axis="y", ls=":")

    fig.tight_layout()
    p = outdir / "fig3_incidence_and_magnitude.png"
    fig.savefig(p, dpi=FIG_DPI)
    plt.close(fig)
    made.append(p.name)
    return made


# --- report ----------------------------------------------------------------
def build_summary(df: pd.DataFrame, traj: pd.DataFrame, prov: dict) -> dict:
    per_arm = {}
    for a in ARMS:
        t = traj[traj["arm"] == a]
        if t.empty:
            continue
        per_arm[a] = {
            "label": ARM_LABEL.get(a, a),
            "n_trajectories": int(len(t)),
            "n_steps": int(t["n_steps"].sum()),
            "deep_spike_trajectory_rate": float(t["has_deep"].mean()),
            "deep_spike_step_rate": float(t["n_deep"].sum() / max(1, t["n_warm"].sum())),
            "jump_trajectory_rate": float(t["has_jump"].mean()),
            "median_max_excess_warm": float(t["max_excess_warm"].median()),
            "p90_max_excess_warm": float(t["max_excess_warm"].quantile(0.90)),
            "median_max_cost_warm": float(t["max_cost_warm"].median()),
            "median_max_jump_cost": float(t["max_jump_cost"].median()),
            "median_max_jump_oracle": float(t["max_jump_oracle"].median()),
            "median_max_cost_cold": float(t["max_cost_cold"].median()),
            "median_final_cost": float(t["final_cost"].median()),
            # Positive starvation is the #2790/#2825 spike mechanism, so the
            # regime each arm actually ran in belongs in the table, not a
            # footnote: a run that never finds positives cannot spike the same
            # way one that does.
            "median_n_good_final": float(t["n_good_final"].median()),
            "median_realized_prevalence": float(t["realized_prevalence"].median()),
        }
    contrasts = {}
    for a in ARMS:
        if a == CONTROL_ARM or a not in per_arm or CONTROL_ARM not in per_arm:
            continue
        contrasts[f"{a}_vs_{CONTROL_ARM}"] = {
            "max_excess_warm": paired_vs(traj, "max_excess_warm", CONTROL_ARM, a),
            "max_cost_warm": paired_vs(traj, "max_cost_warm", CONTROL_ARM, a),
            "max_jump_cost": paired_vs(traj, "max_jump_cost", CONTROL_ARM, a),
            "deep_incidence": mcnemar_incidence(traj, CONTROL_ARM, a),
        }
    ctrl = per_arm.get(CONTROL_ARM, {})
    return {
        "coverage": {
            "n_base_rows": int(len(df)),
            "categories": sorted(df["category"].astype(str).unique().tolist()),
            "seeds": sorted(int(s) for s in df["seed"].unique()),
        },
        "config": {
            "warm_t": WARM_T,
            "deep_cost": DEEP_COST,
            "deep_excess": DEEP_EXCESS,
            "jump_window": JUMP_WINDOW,
            "jump_delta": JUMP_DELTA,
        },
        "provenance": prov,
        "control_reproduces_phenomenon": bool(ctrl.get("deep_spike_trajectory_rate", 0) > 0),
        "per_arm": per_arm,
        "contrasts_vs_control": contrasts,
    }


def _fmt(x, digits=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def write_report(summary: dict, spikes: pd.DataFrame, figs: list[str], outdir: Path) -> Path:
    L: list[str] = []
    A = L.append
    cfgc = summary["config"]
    A("# #2847 - do the MLP-era cost spikes survive today's stack?\n")
    A(
        "Deep spike = a step at `t >= {warm_t}` with `cost >= {deep_cost}` **and** "
        "`cost - oracle_cost >= {deep_excess}`; local jump = `cost` above its own "
        "trailing-{jump_window} median by `>= {jump_delta}`. Cold start (`t < {warm_t}`) "
        "is reported separately - every arm humps there and it is a different "
        "phenomenon.\n".format(**cfgc)
    )

    if not summary["control_reproduces_phenomenon"]:
        A(
            f"> **NO VERDICT.** The control arm `{CONTROL_ARM}` - the #2847-era "
            "configuration - produced no deep spike in this harness, so this run "
            "cannot distinguish a fix from a harness that never showed the "
            "phenomenon. Everything below is descriptive only.\n"
        )

    A("\n## Per-arm\n")
    A(
        "| arm | trajectories | deep-spike runs | deep-spike steps | median worst-step regret | p90 | median max jump (cost / oracle) | median final cost | median positives found |"
    )
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, v in summary["per_arm"].items():
        A(
            f"| `{a}` - {v['label']} | {v['n_trajectories']} | "
            f"{100 * v['deep_spike_trajectory_rate']:.1f}% | "
            f"{100 * v['deep_spike_step_rate']:.2f}% | "
            f"{_fmt(v['median_max_excess_warm'], 3)} | {_fmt(v['p90_max_excess_warm'], 3)} | "
            f"{_fmt(v['median_max_jump_cost'], 3)} / {_fmt(v['median_max_jump_oracle'], 3)} | "
            f"{_fmt(v['median_final_cost'], 3)} | {_fmt(v['median_n_good_final'], 1)} |"
        )

    A(f"\n## Paired against the control (`{CONTROL_ARM}`)\n")
    A("Pairs are `(category, seed)` cells, not steps - the arms are separate trajectories.\n")
    A("| arm | metric | n pairs | control median | arm median | median delta | frac lower | p |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, block in summary["contrasts_vs_control"].items():
        arm = name.split("_vs_")[0]
        for metric, r in block.items():
            if metric == "deep_incidence" or not r.get("n_pairs"):
                continue
            A(
                f"| `{arm}` | {metric} | {r['n_pairs']} | "
                f"{_fmt(r.get(f'{CONTROL_ARM}_median'), 3)} | {_fmt(r.get(f'{arm}_median'), 3)} | "
                f"{_fmt(r.get('median_delta'), 3)} | {100 * r.get('frac_b_lower', float('nan')):.0f}% | "
                f"{_fmt(r.get('p_wilcoxon'), 5)} |"
            )
    A("\n| arm | deep-spike incidence (control -> arm) | only control | only arm | p exact |")
    A("|---|---|---:|---:|---:|")
    for name, block in summary["contrasts_vs_control"].items():
        arm = name.split("_vs_")[0]
        r = block.get("deep_incidence", {})
        if not r.get("n_pairs"):
            continue
        A(
            f"| `{arm}` | {100 * r.get(f'{CONTROL_ARM}_rate', 0):.1f}% -> {100 * r.get(f'{arm}_rate', 0):.1f}% | "
            f"{r.get('only_a')} | {r.get('only_b')} | {_fmt(r.get('p_exact'), 5)} |"
        )

    if not spikes.empty:
        A("\n## What the surviving spikes look like\n")
        by_arm = spikes.groupby("arm")
        A("| arm | spike steps | median n_good | median FNR | median FPR | fallback provenance |")
        A("|---|---:|---:|---:|---:|---|")
        for arm, g in by_arm:
            prov_top = ""
            if "threshold_provenance" in g.columns:
                vc = g["threshold_provenance"].astype(str).value_counts()
                prov_top = ", ".join(f"{k} x{v}" for k, v in vc.head(3).items())
            A(
                f"| `{arm}` | {len(g)} | {g['n_good'].median():.0f} | "
                f"{g['fnr'].median():.3f} | {g['fpr'].median():.3f} | {prov_top} |"
            )

    p = summary["provenance"]
    A("\n## Data read\n")
    A("| arm | cell files | trajectories | never found a positive | unreadable | zero-byte | base rows |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for arm, v in p.items():
        A(
            f"| `{arm}` | {v.get('n_files', 0)} | {v.get('n_read', 0)} | "
            f"{len(v.get('no_positive_found', []))} | "
            f"{len(v.get('unreadable', []))} | {len(v.get('zero_byte', []))} | {v.get('n_rows', 0)} |"
        )
    A(
        "\n`never found a positive` = 100 votes, zero positives, so the simulator "
        "never trained and the cell emits no step. Not a failure - the extreme of "
        "the same positive-starvation regime the spikes live in. These cells differ "
        "per arm, so the paired tests above drop them.\n"
    )

    if figs:
        A("\n## Figures\n")
        for f in figs:
            A(f"![{f}](figures/{f})\n")

    out = outdir / "REPORT_spikes.md"
    out.write_text("\n".join(L) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="#2847 spike analysis across the 2x2 arms.")
    ap.add_argument("--results-root", default=str(common.EXP / "results"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--category", default="cat", help="Category to draw the per-seed figures for.")
    args = ap.parse_args(argv)

    root = Path(args.results_root)
    outdir = Path(args.out)
    (outdir / "agg").mkdir(parents=True, exist_ok=True)

    df, prov = load_all(root)
    if df.empty:
        print(f"no cells under {root}")
        return 1
    print(f"loaded {len(df)} base rows across {df['arm'].nunique()} arms")

    traj = trajectory_stats(df)
    spikes = spike_steps(df)
    traj.to_csv(outdir / "agg" / "trajectories.csv", index=False)
    if not spikes.empty:
        spikes.to_csv(outdir / "agg" / "spike_steps.csv", index=False)

    summary = build_summary(df, traj, prov)
    (outdir / "spike_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    figs = make_figures(df, traj, outdir / "figures", category=args.category)
    rep = write_report(summary, spikes, figs, outdir)
    print(f"wrote {rep}")
    for a, v in summary["per_arm"].items():
        print(
            f"  {a:14s} n={v['n_trajectories']:4d}  deep-runs={100 * v['deep_spike_trajectory_rate']:5.1f}%  "
            f"median max-regret={v['median_max_excess_warm']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

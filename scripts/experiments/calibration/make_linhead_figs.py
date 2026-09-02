"""Figures for the #2808 linear-head convergence study.

Every figure comes from the same per-step cell CSVs as the report's tables, so a
figure and a table cannot disagree.  Four things the tables cannot show:

* **What a user gets over the ramp.** The tables are deep-regime summaries; the
  question is "what is my cost after 20 clicks, and after 100?"
* **What a single run looks like.** Averaged, every arm descends smoothly.
  Individually they plateau and spike, and the spread is the real finding.
* **Where the convergence effect actually lives.** The pooled paired delta hides
  that it is absent on the shipped default embedder and concentrated on VG.
* **The binding constraint.** Positives found, not spike incidence, is what
  separates these arms in the regime users occupy.

Usage:

    python make_linhead_figs.py --results /expscratch/$USER/linhead-2808/results \\
        --out docs/experiments/2026-08-19-linhead-convergence-2808/figures
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["path.simplify"] = True
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: One colour per arm, fixed across every figure so a colour means one thing.
COLOR = {
    "C_mlp": "#b2182b",
    "A_shipped": "#2166ac",
    "B_converged": "#1a9850",
}
LABEL = {
    "C_mlp": "C_mlp — mlp, 200 ep (reference)",
    "A_shipped": "A_shipped — linear, 200 ep / pat 10 (production)",
    "B_converged": "B_converged — linear, 2000 ep / no early stop",
}
ARMS = ("C_mlp", "A_shipped", "B_converged")
WARM_T = 20
DEEP_COST, DEEP_EXCESS = 0.25, 0.20
DPI = 130


def load(results: Path) -> pd.DataFrame:
    """Concatenate every arm's base rows, via the ANALYZER's own loader.

    Deliberately not a second implementation.  The cells carry counterfactual
    variant rows alongside each arm's base row, and a private "drop the tagged
    rows" filter here loaded 1.41M rows where the analyzer sees 42k - which
    would have made every figure disagree with the table beside it while looking
    entirely plausible.  Importing ``load_arm`` makes that class of drift
    impossible rather than merely unlikely.
    """
    import _cells_io  # noqa: PLC0415 - needs the study env set up first

    frames = []
    for arm in ARMS:
        d, prov = _cells_io.load_arm(results / arm)
        if d.empty:
            print(f"  {arm}: NO ROWS")
            continue
        d["arm"] = arm
        # Report what was dropped; an analysis that silently excludes cells is
        # how a disk incident becomes a wrong verdict.
        print(
            f"  {arm}: {len(d)} base rows from {prov.get('n_files', '?')} cells"
            f" | unreadable={len(prov.get('unreadable', []))}"
            f" zero_byte={len(prov.get('zero_byte', []))}"
            f" no_positive={len(prov.get('no_positive_found', []))}"
        )
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "embedder" not in df.columns:
        df["embedder"] = "unknown"
    return df


def _cellkeys(df: pd.DataFrame) -> list[str]:
    return [k for k in ("dataset", "embedder", "category", "seed") if k in df.columns]


def fig_ramp(df: pd.DataFrame, out: Path) -> str:
    """Headline metric over the axis the user spends: votes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, sub) in zip(axes, df.groupby("dataset")):
        for arm in ARMS:
            a = sub[sub.arm == arm]
            g = a.groupby("t")["cost"]
            m, se = g.mean(), g.sem()
            ax.plot(m.index, m.values, color=COLOR[arm], lw=1.8, label=LABEL[arm])
            ax.fill_between(m.index, m - se, m + se, color=COLOR[arm], alpha=0.18, lw=0)
        ax.axvline(WARM_T, color="0.4", ls=":", lw=1)
        ax.set_title(f"{name}", fontsize=10)
        ax.set_xlabel("votes (labels spent)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("operating cost")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Cost over the ramp, mean ± SE across categories/seeds/embedders", fontsize=11)
    fig.tight_layout()
    p = out / "fig1_ramp.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_per_run(df: pd.DataFrame, out: Path) -> str:
    """The same metric, one line per run - the spread the mean hides."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    keys = _cellkeys(df)
    for ax, arm in zip(axes, ARMS):
        a = df[df.arm == arm]
        for _, g in a.groupby(keys):
            ax.plot(g["t"], g["cost"], color=COLOR[arm], lw=0.5, alpha=0.25)
        m = a.groupby("t")["cost"].mean()
        ax.plot(m.index, m.values, color="k", lw=2.0, label="mean")
        ax.axhline(DEEP_COST, color="0.3", ls="--", lw=1)
        ax.set_title(LABEL[arm], fontsize=8.5)
        ax.set_xlabel("votes")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("operating cost")
    fig.suptitle(
        "One line per run. The dashed line is the deep-spike cost floor (0.25); "
        "runs that ride above it are the finding, not the mean.",
        fontsize=10,
    )
    fig.tight_layout()
    p = out / "fig2_per_run.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def _traj(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["arm"] + _cellkeys(df)
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy(float)
        cost = g["cost"].to_numpy(float)
        orc = g["oracle_cost"].to_numpy(float)
        warm = t >= WARM_T
        if not warm.any():
            continue
        excess = cost - orc
        rows.append(
            dict(
                zip(keys, key),
                max_excess_warm=float(np.nanmax(excess[warm])),
                deep=bool(((cost >= DEEP_COST) & (excess >= DEEP_EXCESS) & warm).any()),
                n_good_final=float(g["n_good"].iloc[-1]) if "n_good" in g else np.nan,
                prevalence=float(g["realized_prevalence"].median()) if "realized_prevalence" in g else np.nan,
            )
        )
    return pd.DataFrame(rows)


def fig_where(traj: pd.DataFrame, out: Path) -> str:
    """Where convergence actually helps - the axis the mechanism runs on."""
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in traj.columns]
    cells = [k for k in keys if k in ("category", "seed")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    labels, means, ses = [], [], []
    for (ds, emb), sub in traj.groupby(["dataset", "embedder"]):
        piv = sub.pivot_table(index=cells, columns="arm", values="max_excess_warm").dropna()
        if not {"A_shipped", "B_converged"} <= set(piv.columns) or len(piv) < 5:
            continue
        d = piv["B_converged"] - piv["A_shipped"]
        labels.append(f"{ds}\n×{emb}  (n={len(d)})")
        means.append(d.mean())
        ses.append(d.std(ddof=1) / np.sqrt(len(d)))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=ses, color=["#1a9850" if m < 0 else "#b2182b" for m in means], alpha=0.85, capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("paired Δ worst-step regret,  B_converged − A_shipped")
    ax.set_title("Negative = convergence helps. Bars whose error bar\ncrosses 0 are not resolvable here.", fontsize=9)
    ax.grid(alpha=0.25, axis="x")

    ax = axes[1]
    inc = traj.groupby(["dataset", "embedder", "arm"])["deep"].mean().unstack("arm")
    idx = np.arange(len(inc))
    w = 0.26
    for i, arm in enumerate(ARMS):
        if arm in inc:
            ax.bar(idx + (i - 1) * w, inc[arm].values, w, color=COLOR[arm], label=LABEL[arm], alpha=0.9)
    ax.set_xticks(idx, [f"{a}\n×{b}" for a, b in inc.index], fontsize=8)
    ax.set_ylabel("fraction of runs with a deep spike")
    ax.set_title("Deep-spike incidence per cell type", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    p = out / "fig3_where.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def fig_binding(traj: pd.DataFrame, out: Path) -> str:
    """The binding constraint: positives found, and how it tracks prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    data = [traj[traj.arm == a]["n_good_final"].dropna().values for a in ARMS]
    bp = ax.boxplot(data, tick_labels=["C_mlp", "A_shipped", "B_converged"], patch_artist=True, widths=0.55)
    for patch, arm in zip(bp["boxes"], ARMS):
        patch.set_facecolor(COLOR[arm])
        patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("k")
    ax.set_ylabel("positives found at the final vote")
    ax.set_title("Positives found — the binding constraint", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for arm in ARMS:
        a = traj[traj.arm == arm]
        ax.scatter(
            a["prevalence"], a["n_good_final"], s=12, alpha=0.5, color=COLOR[arm], label=LABEL[arm], edgecolors="none"
        )
    ax.set_xscale("log")
    ax.set_xlabel("realized prevalence (log)")
    ax.set_ylabel("positives found")
    ax.set_title("Starvation tracks prevalence, not the head", fontsize=9)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    p = out / "fig4_binding.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"/expscratch/{os.environ.get('USER', '')}/linhead-2808/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.results))
    print(f"loaded {len(df)} base rows, {df.arm.nunique()} arms")
    traj = _traj(df)
    print(f"{len(traj)} trajectories")
    for name in (fig_ramp(df, out), fig_per_run(df, out), fig_where(traj, out), fig_binding(traj, out)):
        print("wrote", out / name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""#5089: the Evidence coverage verdict's candidate rules, priced on #4739's bench.

Reads the cells ``launch_cross_corpus_4739.sh`` wrote under ``--root`` (``<leg>/{ctl,nowalk}_{b025,b1,b4}``) from the
#5089 re-run, whose snapshot-click rows carry the counts behind the Evidence share (``evidence_n*``,
``evidence_pool_n*``, ``evidence_n_{pos,neg}_beyond_pool``), and writes, to ``--out``:

* ``flags.csv`` - one row per run and snapshot click: the counts, and every rule's verdict.
* ``rules.csv`` - per leg, arm, preset and click: the share of runs each rule fires in, and the mean shares behind them.
* ``shares.csv`` - per leg, arm and click: the spread over runs of today's share and of the share beyond the pool.
* ``examples.csv`` - literal runs: the in-domain runs today's verdict fires on hardest, the in-domain runs closest to
  firing under the chosen rule (every one it fires on, if any), and the screens-only runs closest to staying silent.
* ``figures/rules_by_click.png`` (each rule's fire rate by click, per corpus) and ``figures/shares_per_run.png`` (every
  run's share at vote 150, today's against the chosen rule's).

The rules, each a verdict from one run's counts at one click, alpha = 0.05:

* ``today`` - the shipped verdict: the share with support p-value ``D < alpha`` (``D`` ranked against the votes' own
  leave-one-out spread), z > 3 against alpha and at least 2 alpha. Re-derived from the counts, and checked against the
  recorded ``evidence_unsupported``.
* ``beyond_pool`` - the chosen rule: the share farther from its predicted class's votes than 95% of the sim pool's
  unvoted media called that class, z > 3 and at least 2 alpha.
* ``vs_pool`` - today's share against the pool's own share: two-proportion z > 3 and an excess of at least alpha.
* ``pos_only`` - today's test over the predicted positives alone.
* ``bar70`` - today's share at or above 0.7.

    python analyze_evidence_5089.py --root /expscratch/$USER/evidence-5089/runs --out OUTDIR [--jobs 8]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05
LEGS = ("screens0", "screens_equal", "screens_only")
LEG_NAMES = {
    "screens0": "withheld half (in-domain)",
    "screens_equal": "withheld half + 22% screens",
    "screens_only": "screens only",
}
RULES = ("today", "beyond_pool", "vs_pool", "pos_only", "bar70")
RULE_NAMES = {
    "today": "today: D vs the votes' spread",
    "beyond_pool": "ranked against the pool (chosen)",
    "vs_pool": "share vs the pool's share",
    "pos_only": "predicted positives only",
    "bar70": "share >= 0.7",
}
COLUMNS = [
    "seed",
    "category",
    "t",
    "n_good",
    "n_bad",
    "detector_tier",
    "find_n_flagged",
    "flags_at",
    "evidence_frac_unsupported",
    "evidence_z",
    "evidence_unsupported",
    "evidence_n",
    "evidence_n_pred_pos",
    "evidence_n_pos_unsupported",
    "evidence_n_neg_unsupported",
    "evidence_pool_n",
    "evidence_pool_n_pred_pos",
    "evidence_pool_n_pos_unsupported",
    "evidence_pool_n_neg_unsupported",
    "evidence_n_pos_beyond_pool",
    "evidence_n_neg_beyond_pool",
]


def _read_cell(path: Path) -> pd.DataFrame:
    leg, arm_preset = path.parts[-5], path.parts[-4]
    arm, preset = arm_preset.rsplit("_", 1)
    frame = pd.read_csv(path, usecols=lambda c: c in COLUMNS, low_memory=False)
    snap = frame.dropna(subset=["flags_at"]).drop_duplicates("flags_at")
    return snap.assign(leg=leg, arm=arm, preset=preset, task=path.name.split(".")[0])


def read_runs(root: Path, jobs: int) -> pd.DataFrame:
    paths = sorted(
        p
        for leg in LEGS
        for p in (root / leg).glob("*_b*/results/cells/task_*.csv.gz")
        if "__" not in p.name and not p.parts[-4].startswith("sizing")
    )
    with ProcessPoolExecutor(jobs) as ex:
        frames = list(ex.map(_read_cell, paths, chunksize=16))
    return pd.concat(frames, ignore_index=True)


def _z_vs_alpha(k: pd.Series, n: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    n = n.astype(float)
    frac = np.where(n > 0, k / n.clip(lower=1), 0.0)
    z = np.where(n > 0, (frac - ALPHA) / np.sqrt(ALPHA * (1 - ALPHA) / n.clip(lower=1)), 0.0)
    return frac, z


def price(runs: pd.DataFrame) -> pd.DataFrame:
    """Every rule's verdict, and the shares behind them, for each run at each snapshot click."""
    f = runs.copy()
    n, n_pool = f["evidence_n"], f["evidence_pool_n"]
    k = f["evidence_n_pos_unsupported"] + f["evidence_n_neg_unsupported"]
    k_pool = f["evidence_pool_n_pos_unsupported"] + f["evidence_pool_n_neg_unsupported"]

    f["share"], f["z"] = _z_vs_alpha(k, n)
    f["today"] = (f["z"] > 3) & (f["share"] >= 2 * ALPHA)
    f["today_matches_recorded"] = f["today"] == (f["evidence_unsupported"] == 1)

    f["beyond"], f["z_beyond"] = _z_vs_alpha(f["evidence_n_pos_beyond_pool"] + f["evidence_n_neg_beyond_pool"], n)
    f["beyond_pool"] = (f["z_beyond"] > 3) & (f["beyond"] >= 2 * ALPHA)

    f["pool_share"] = k_pool / n_pool.clip(lower=1)
    pooled = (k + k_pool) / (n + n_pool).clip(lower=1)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n.clip(lower=1) + 1 / n_pool.clip(lower=1)))
    z_two = np.where(se > 0, (f["share"] - f["pool_share"]) / se.where(se > 0, 1), 0.0)
    f["vs_pool"] = (z_two > 3) & (f["share"] - f["pool_share"] >= ALPHA)

    pos_share, z_pos = _z_vs_alpha(f["evidence_n_pos_unsupported"], f["evidence_n_pred_pos"])
    f["pos_only"] = (f["evidence_n_pred_pos"] > 0) & (z_pos > 3) & (pos_share >= 2 * ALPHA)
    f["bar70"] = f["share"] >= 0.7
    return f


def rule_table(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby(["leg", "arm", "preset", "flags_at"])
    out = g[list(RULES)].mean()
    out["runs"] = g.size()
    out["share"] = g["share"].mean()
    out["pool_share"] = g["pool_share"].mean()
    out["beyond"] = g["beyond"].mean()
    out["today_matches_recorded"] = g["today_matches_recorded"].mean()
    return out.reset_index()


def share_table(f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (leg, arm, at), g in f.groupby(["leg", "arm", "flags_at"]):
        row = {"leg": leg, "arm": arm, "flags_at": int(at), "runs": len(g)}
        for col in ("share", "beyond"):
            for q in (0.05, 0.5, 0.95):
                row[f"{col}_p{int(q * 100):02d}"] = float(g[col].quantile(q))
            row[f"{col}_max"] = float(g[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def examples(f: pd.DataFrame) -> pd.DataFrame:
    """Literal runs at vote 150, one preset (beta 1) per list so a run is not repeated; and every in-domain fire, any click."""
    at150 = f[f["flags_at"] == 150]
    keep = [
        "why",
        "leg",
        "arm",
        "preset",
        "category",
        "seed",
        "flags_at",
        "n_good",
        "n_bad",
        "evidence_n",
        "evidence_n_pred_pos",
        "evidence_pool_n",
        "evidence_pool_n_pred_pos",
        "share",
        "z",
        "pool_share",
        "beyond",
        "z_beyond",
        "today",
        "beyond_pool",
    ]
    home = at150[at150["leg"] == "screens0"]
    home_b1 = home[home["preset"] == "b1"]
    away_b1 = at150[(at150["leg"] == "screens_only") & (at150["preset"] == "b1")]
    picks = [
        home_b1.nlargest(6, "share").assign(why="in-domain, today's share highest"),
        home_b1.nlargest(6, "beyond").assign(why="in-domain, closest to firing under the chosen rule"),
        f[(f["leg"] == "screens0") & f["beyond_pool"]].assign(why="in-domain, the chosen rule fires"),
        away_b1.nsmallest(6, "beyond").assign(why="screens only, closest to silent under the chosen rule"),
    ]
    return pd.concat(picks, ignore_index=True)[keep]


def figures(f: pd.DataFrame, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    colors = {
        "today": "#d03c5f",
        "beyond_pool": "#2a78d6",
        "vs_pool": "#9c5fd0",
        "pos_only": "#e0a020",
        "bar70": "#777",
    }
    ctl = f[f["arm"] == "ctl"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, leg in zip(axes, LEGS, strict=True):
        rates = ctl[ctl["leg"] == leg].groupby("flags_at")[list(RULES)].mean()
        for rule in RULES:
            ax.plot(
                rates.index,
                rates[rule],
                marker="o",
                ms=3,
                color=colors[rule],
                lw=2.2 if rule in ("today", "beyond_pool") else 1.2,
                label=RULE_NAMES[rule],
            )
        ax.set_title(LEG_NAMES[leg], fontsize=9)
        ax.set_xlabel("vote")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of runs the verdict fires in")
    axes[2].legend(fontsize=7, loc="center right")
    fig.suptitle("Evidence coverage verdict by rule, today's app (all three presets)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "rules_by_click.png", dpi=130)
    plt.close(fig)

    legcolors = {"screens0": "#2a78d6", "screens_equal": "#9c5fd0", "screens_only": "#d03c5f"}
    at150 = f[f["flags_at"] == 150]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, (col, title) in zip(
        axes,
        (("share", "today: share with D < alpha"), ("beyond", "chosen: share beyond 95% of the pool")),
        strict=True,
    ):
        for leg in LEGS:
            for arm, ls in (("ctl", "-"), ("nowalk", "--")):
                x = np.sort(at150[(at150["leg"] == leg) & (at150["arm"] == arm)][col].to_numpy())
                if x.size:
                    ax.step(
                        x,
                        np.arange(1, x.size + 1) / x.size,
                        where="post",
                        color=legcolors[leg],
                        ls=ls,
                        label=f"{LEG_NAMES[leg]}, {arm}",
                    )
        ax.axvline(2 * ALPHA, color="#444", lw=0.8, ls=":")
        ax.set_title(f"{title}, every run at vote 150", fontsize=9)
        ax.set_xlabel("share of the Find corpus")
        ax.set_xlim(-0.01, 1.01)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("share of runs at or below")
    axes[1].legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    fig.savefig(figdir / "shares_per_run.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    f = price(read_runs(args.root, args.jobs))
    f.to_csv(args.out / "flags.csv", index=False)
    rule_table(f).to_csv(args.out / "rules.csv", index=False)
    share_table(f).to_csv(args.out / "shares.csv", index=False)
    examples(f).to_csv(args.out / "examples.csv", index=False)
    figures(f, args.out / "figures")
    bad = int((~f["today_matches_recorded"]).sum())
    print(
        f"{len(f)} snapshot rows from {f.groupby(['leg', 'arm', 'preset', 'task']).ngroups} runs; "
        f"today's verdict re-derived from the counts differs from the recorded one in {bad}"
    )
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

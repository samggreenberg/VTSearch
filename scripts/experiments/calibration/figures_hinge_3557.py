"""Figures for the #3557 hinge study, from ``analyze_hinge_3557.py``'s CSVs only.

    python figures_hinge_3557.py --analysis $BASE/analysis --out docs/experiments/2026-09-22-hinge-tilt-3557/figures

Writes:

``fig1_delta_vs_k.png``
    Per environment, the paired Δ against the incumbent at every stop of the
    knob: the full ship (arm B ``hinge`` vs arm A ``mid_tilt``, cost) and the
    reporting-only re-cuts (arm A, regret) of ``hinge``, ``hinge_raw``,
    ``hinge_cont`` and ``cross_tilt``.  Dashed lines are the ±0.010 bar.
``fig2_seam.png``
    Per environment, the distribution over deep cell-steps of
    ``q_cross(0) - q_mid`` - negative is exactly where the literal hinge breaks
    nesting - with the share below zero printed on each panel.
``fig3_mechanism.png``
    Per cell, the fitted prior odds (bits) against the re-cut hinge's gain at
    k=-1: does the hinge win where ``cross_tilt`` reads the most prior?
``cost_vs_clicks*.png`` / ``average_precision_vs_clicks*.png``
    The mandatory quality-over-clicks pair (``curves.quality_vs_clicks``) for the
    two run-level arms at the reporting inclusion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common

common.setup_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import curves  # noqa: E402

DPI = 130
BAR = 0.010
ENV_ORDER = [
    ("visual_genome_m", "siglip+dinov3_patch", "max_patch"),
    ("coco_val", "siglip+dinov3_patch", "max_patch"),
    ("visual_genome_m", "siglip", "whole_image"),
    ("coco_val", "siglip", "whole_image"),
    ("caltech101_m", "siglip", "whole_image"),
]
SERIES = [
    # (contrast, arm_b, label, colour, linestyle)
    ("ship_cost", "hinge", "hinge, full ship (cost)", "#1f5fbf", "-"),
    ("recut", "hinge", "hinge, re-cut (regret)", "#1f5fbf", ":"),
    ("recut", "hinge_raw", "hinge_raw, re-cut", "#c0392b", ":"),
    ("recut", "hinge_cont", "hinge_cont, re-cut", "#7f8c8d", ":"),
    ("recut", "cross_tilt", "cross_tilt, re-cut", "#e08e0b", "--"),
    ("trajectory_only", "mid_tilt", "trajectory only (mid_tilt on arm B's votes)", "#27ae60", "-."),
]


def _label(env: tuple[str, str, str]) -> str:
    ds, emb, sty = env
    mode = "region" if sty == "max_patch" else "binary"
    return f"{ds} x {emb}\n({mode})"


def _envs(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    have = {(r.dataset, r.embedder, r.style) for r in df.itertuples()}
    return [e for e in ENV_ORDER if e in have] + sorted(have - set(ENV_ORDER))


def fig_delta(paired: pd.DataFrame, out: Path) -> None:
    envs = _envs(paired)
    fig, axes = plt.subplots(1, len(envs), figsize=(4.2 * len(envs), 4.0), sharey=True, squeeze=False)
    for ax, env in zip(axes[0], envs, strict=True):
        sub = paired[(paired["dataset"] == env[0]) & (paired["embedder"] == env[1]) & (paired["style"] == env[2])]
        for contrast, arm_b, label, colour, ls in SERIES:
            s = sub[(sub["contrast"] == contrast) & (sub["arm_b"] == arm_b)].sort_values("inclusion_k")
            if s.empty:
                continue
            ax.plot(s["inclusion_k"], s["d_mean"], ls=ls, color=colour, lw=1.6, label=label)
            if contrast == "ship_cost":
                ax.fill_between(s["inclusion_k"], s["d_lo"], s["d_hi"], color=colour, alpha=0.18, lw=0)
        ax.axhline(0, color="black", lw=0.8)
        for y in (-BAR, BAR):
            ax.axhline(y, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="grey", lw=0.6, alpha=0.5)
        ax.set_title(_label(env), fontsize=9)
        ax.set_xlabel("Inclusion k")
    axes[0][0].set_ylabel("Δ vs mid_tilt (rate scale; < 0 favours challenger)")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig1_delta_vs_k.png", dpi=DPI)
    plt.close(fig)


def fig_seam(seam_steps: pd.DataFrame, out: Path) -> None:
    d = seam_steps[seam_steps["deep"] & (seam_steps["arm"] == "incumbent")].copy()
    d["gap"] = d["seam_q_cross0"] - d["seam_q_mid"]
    envs = _envs(d)
    fig, axes = plt.subplots(1, len(envs), figsize=(4.0 * len(envs), 3.4), sharey=False, squeeze=False)
    for ax, env in zip(axes[0], envs, strict=True):
        g = d[(d["dataset"] == env[0]) & (d["embedder"] == env[1]) & (d["style"] == env[2])]["gap"].dropna()
        lo, hi = np.quantile(g, [0.005, 0.995]) if len(g) else (-0.1, 0.1)
        ax.hist(g.clip(lo, hi), bins=60, color="#1f5fbf", alpha=0.8)
        ax.axvline(0, color="#c0392b", lw=1.2)
        ax.set_title(f"{_label(env)}\nbelow 0 (raw hinge breaks): {(g < 0).mean():.0%}", fontsize=9)
        ax.set_xlabel("q_cross(0) - q_mid  (fold quantile)")
    axes[0][0].set_ylabel("deep cell-steps")
    fig.tight_layout()
    fig.savefig(out / "fig2_seam.png", dpi=DPI)
    plt.close(fig)


def fig_mechanism(deep_cells: pd.DataFrame, seam_steps: pd.DataFrame, out: Path) -> None:
    a = deep_cells[(deep_cells["arm"] == "incumbent") & (deep_cells["inclusion_k"] == -1)]
    keys = ["dataset", "embedder", "style", "category", "seed"]
    inc = a[a["cut_rule"] == "mid_tilt"].set_index(keys)["rregret"]
    hin = a[a["cut_rule"] == "hinge"].set_index(keys)["rregret"]
    gain = (hin - inc).rename("gain").reset_index()
    s = seam_steps[seam_steps["deep"] & (seam_steps["arm"] == "incumbent")]
    odds = s.groupby(keys)["fit_log2_prior_odds"].median().rename("odds").reset_index()
    m = gain.merge(odds, on=keys)
    envs = _envs(m)
    fig, axes = plt.subplots(1, len(envs), figsize=(4.0 * len(envs), 3.4), sharey=True, squeeze=False)
    for ax, env in zip(axes[0], envs, strict=True):
        g = m[(m["dataset"] == env[0]) & (m["embedder"] == env[1]) & (m["style"] == env[2])]
        ax.scatter(g["odds"], g["gain"], s=12, alpha=0.7, color="#1f5fbf")
        ax.axhline(0, color="black", lw=0.8)
        if len(g) > 2:
            r = np.corrcoef(g["odds"], g["gain"])[0, 1]
            ax.set_title(f"{_label(env)}\nPearson r = {r:.2f}", fontsize=9)
        else:
            ax.set_title(_label(env), fontsize=9)
        ax.set_xlabel("fitted prior odds log2(w_lo/w_hi), deep median")
    axes[0][0].set_ylabel("hinge - mid_tilt regret at k=-1, per cell")
    fig.tight_layout()
    fig.savefig(out / "fig3_mechanism.png", dpi=DPI)
    plt.close(fig)


def fig_clicks(traj: pd.DataFrame, out: Path, baseline: Path | None) -> list[str]:
    main = traj.copy()
    # One line per (arm, voting mode): curves.py panels per DATASET, and VG/COCO
    # each carry a region and a binary embedder - averaging those would describe
    # no system anyone runs (the embedders-are-never-averaged rule).
    mode = np.where(main["style"] == "max_patch", "region", "binary")
    main["arm"] = main["arm"] + " · " + mode
    arms = [f"{a} · {m}" for m in ("region", "binary") for a in ("incumbent", "hinge")]
    arms = [a for a in arms if a in set(main["arm"])]
    keys = ["dataset", "embedder", "category", "seed"]
    denominator = main[["arm", *keys]].drop_duplicates()
    base = curves.text_sort_baseline(baseline) if baseline and baseline.exists() else None
    written = []
    for metric, lower in (("cost", True), ("average_precision", False)):
        written += curves.quality_vs_clicks(
            main,
            out,
            arms=arms,
            metric=metric,
            denominator=denominator,
            baseline=base,
            lower_is_better=lower,
        )
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None, help="text_baseline.py CSV (the click-0 anchor)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(args.analysis / "hinge3557_paired.csv")
    seam_steps = pd.read_csv(args.analysis / "hinge3557_seam_steps.csv.gz")
    deep_cells = pd.read_csv(args.analysis / "hinge3557_deep_cells.csv.gz")
    traj = pd.read_csv(args.analysis / "hinge3557_traj.csv.gz")
    fig_delta(paired, args.out)
    fig_seam(seam_steps, args.out)
    fig_mechanism(deep_cells, seam_steps, args.out)
    written = fig_clicks(traj, args.out, args.baseline)
    print("wrote fig1-3 and", len(written), "quality-over-clicks files into", args.out)


if __name__ == "__main__":
    main()

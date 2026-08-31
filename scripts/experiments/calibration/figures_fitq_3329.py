#!/usr/bin/env python
"""Figures for the #3329 goodness-of-fit run.

Three things, in the order a reader needs them:

1. **The mandatory quality-over-clicks pair**, drawn by the one implementation
   (:mod:`curves`) so it is literally the same figure as in every other study.
2. **The four pre-registered statistics over clicks**, because every bar in
   ``PREREG.md`` is a median over a run and a median hides whether the thing
   moves.  ``anchor_mass_frac`` is the case in point: it crosses its bar
   somewhere around click 20, which no single number can say.
3. **The worked cell** - the figure #3329 actually asked for: the score
   histogram with the fitted components drawn over it and the TRUE CLASSES
   coloured underneath.  No user and no prior run has seen it, because it needs
   ground truth.

Called by ``analyze_fitq_3329.py`` unless ``--no-figures``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import curves

#: Bad (negative) then Good (positive).  Kept apart from the fitted-component
#: colours below on purpose: the whole point of the worked-cell panel is to see
#: whether the fitted LOW component lines up with the true BAD class, so the two
#: must never be drawn in the same hue.
CLASS_COLOURS = ("#8fa8c8", "#e8913a")
FIT_COLOURS = ("#1a3a5c", "#a03a0a")

ARM_COLOURS = {
    "siglip/whole_image": "#2b6cb0",
    "siglip+dinov3_patch/whole_image": "#38a169",
    "siglip+dinov3_patch/max_patch": "#c0392b",
}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _median_band(g: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median and inter-quartile band of *col* against ``t``."""
    by = g.groupby("t")[col]
    ts = np.asarray(sorted(by.groups), dtype=float)
    med = by.median().reindex(sorted(by.groups)).to_numpy(dtype=float)
    lo = by.quantile(0.25).reindex(sorted(by.groups)).to_numpy(dtype=float)
    hi = by.quantile(0.75).reindex(sorted(by.groups)).to_numpy(dtype=float)
    return ts, med, lo, hi


def quality_pair(main: pd.DataFrame, outdir: Path, baseline_csv: str | Path | None) -> list[str]:
    """The two mandatory figures. One dataset, three geometries -> arm is the hue."""
    if main.empty or "arm" not in main.columns:
        return []
    baseline = curves.text_sort_baseline(baseline_csv) if baseline_csv and Path(baseline_csv).exists() else None
    denominator = main[["dataset", "embedder", "category", "seed"]].drop_duplicates()
    arms = [a for a in ARM_COLOURS if (main["arm"] == a).any()]
    written: list[str] = []
    for metric in ("cost", "average_precision"):
        if metric not in main.columns:
            continue
        written += curves.quality_vs_clicks(
            main,
            outdir,
            arms=arms,
            metric=metric,
            denominator=denominator,
            baseline=baseline,
            lower_is_better=(metric == "cost"),
        )
    return written


def statistics_over_clicks(fq: pd.DataFrame, outdir: Path, bars: dict[str, float], dpi: int = 130) -> list[str]:
    """The four pre-registered statistics against the axis the user spends.

    A median over the whole run is what PREREG asked for and what the tables
    report; this is the picture that says whether that median describes a
    plateau or an average across a crossing.
    """
    plt = _plt()
    pooled = fq[fq["scope"] == "sim:pooled"]
    folds = fq[fq["scope"].astype(str).str.startswith("fold")]
    if pooled.empty:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), layout="constrained")
    panels: Sequence[tuple[Any, pd.DataFrame, str, str, float | None, bool]] = (
        (axes[0][0], pooled, "tail_ratio", "H1  tail calibration at the cut\n(empirical / predicted)", 1.0, False),
        (axes[0][1], pooled, "shape_skew_neg", "H2  skewness of the true Bad mode", bars.get("h2"), False),
        (axes[1][0], folds, "anchor_mass_frac", "H3  anchors' share of M-step mass", bars.get("h3_mass"), True),
        (axes[1][1], folds, "anchored_dmu_lo_abs", "H3  movement of the fitted Bad mean", bars.get("h3_dmu"), True),
    )
    for ax, frame, col, title, bar, logy in panels:
        d = frame
        if col == "anchored_dmu_lo_abs":
            d = frame.assign(anchored_dmu_lo_abs=frame["anchored_dmu_lo"].abs())
            d = d[d["anchor_n"] > 0]
        if col not in d.columns or d.empty:
            ax.set_visible(False)
            continue
        for arm, colour in ARM_COLOURS.items():
            g = d[d["arm"] == arm].dropna(subset=[col])
            if g.empty:
                continue
            ts, med, lo, hi = _median_band(g, col)
            ax.plot(ts, med, color=colour, lw=1.8, label=arm, zorder=3)
            ax.fill_between(ts, lo, hi, color=colour, alpha=0.13, lw=0, zorder=2)
        if bar is not None:
            ax.axhline(bar, color="#444", ls="--", lw=1.0, zorder=1)
            ax.annotate(
                "pre-registered bar" if col != "tail_ratio" else "a perfect fit",
                xy=(0.99, bar),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=7,
                color="#444",
            )
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("clicks")
        ax.grid(alpha=0.25, lw=0.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    outdir.mkdir(parents=True, exist_ok=True)
    name = "fit_statistics_over_clicks.png"
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)
    return [name]


def _mixture_density(xs: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w_lo, mu_lo, var_lo, w_hi, mu_hi, var_hi = (float(v) for v in p)
    lo = w_lo * np.exp(-0.5 * (xs - mu_lo) ** 2 / max(var_lo, 1e-12)) / math.sqrt(2 * math.pi * max(var_lo, 1e-12))
    hi = w_hi * np.exp(-0.5 * (xs - mu_hi) ** 2 / max(var_hi, 1e-12)) / math.sqrt(2 * math.pi * max(var_hi, 1e-12))
    return lo, hi


def worked_cell(npz_paths: dict[str, Path], outdir: Path, checkpoints: Sequence[int], dpi: int = 130) -> list[str]:
    """THE figure #3329 asked for, for one worked cell of each geometry.

    Rows are geometries, columns are click checkpoints.  Under each panel is the
    score histogram **coloured by ground truth**; over it, the two fitted
    Gaussian components and their sum; the dashed line is the threshold the app
    would ship at that click.  The question the panel answers by eye is the one
    no relative diagnostic can: does the fitted LOW component actually sit on
    the true Bad mass, or only on whatever the EM found?
    """
    plt = _plt()
    rows: list[tuple[str, Any, str]] = []
    for label, path in npz_paths.items():
        if not Path(path).exists():
            continue
        z = np.load(path)
        for key in sorted({k.rsplit("|", 1)[0] for k in z.files}):
            style, scope, t = key.split("|")
            if scope.startswith("fold") or int(t) not in checkpoints:
                continue
            if (style == "whole_image" and scope != "sim:image") or (style == "max_patch" and scope != "sim:pooled"):
                continue
            rows.append((f"{label} · {style}", z, key))
    if not rows:
        return []

    geometries = sorted({r[0] for r in rows})
    fig, axes = plt.subplots(
        len(geometries),
        len(checkpoints),
        figsize=(3.0 * len(checkpoints), 2.5 * len(geometries)),
        sharex="row",
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    for i, geom in enumerate(geometries):
        for j, t in enumerate(checkpoints):
            ax = axes[i][j]
            match = [r for r in rows if r[0] == geom and r[2].endswith(f"|{t}")]
            if not match:
                ax.set_visible(False)
                continue
            _, z, key = match[0]
            scores = z[f"{key}|scores"]
            labels = z[f"{key}|labels"]
            fit = z[f"{key}|fit"]
            cut = float(z[f"{key}|cut"][0])
            edges = np.histogram_bin_edges(scores, bins=44)
            ax.hist(
                [scores[labels != 1.0], scores[labels == 1.0]],
                bins=edges,
                stacked=True,
                density=True,
                color=CLASS_COLOURS,
                label=("true Bad", "true Good"),
                lw=0,
            )
            xs = np.linspace(float(edges[0]), float(edges[-1]), 400)
            lo, hi = _mixture_density(xs, fit)
            # The true Good class is ~7% of the mass, so stacked it is a sliver.
            # Draw its PRIOR-WEIGHTED class-conditional density as well, on the
            # same scale as the fitted components - that is what makes "is the
            # fitted high component the Good class?" answerable by eye, and it
            # is the question the whole panel exists for.
            good = scores[labels == 1.0]
            if good.size:
                dens, _ = np.histogram(good, bins=edges, density=True)
                prior = float(good.size) / float(scores.size)
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.step(
                    centres,
                    dens * prior,
                    where="mid",
                    color="#b35900",
                    lw=1.2,
                    alpha=0.9,
                    label="true Good (prior-weighted)",
                )
            ax.plot(xs, lo, color=FIT_COLOURS[0], lw=1.5, label="fitted low")
            ax.plot(xs, hi, color=FIT_COLOURS[1], lw=1.5, label="fitted high")
            ax.plot(xs, lo + hi, color="#222", lw=1.0, ls=":", label="mixture")
            ax.axvline(cut, color="#c0392b", ls="--", lw=1.2, label="shipped cut")
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{t} clicks", fontsize=9)
            if j == 0:
                ax.set_ylabel(geom, fontsize=8)
            ax.tick_params(labelsize=7)
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    outdir.mkdir(parents=True, exist_ok=True)
    name = "worked_cell_fit_overlay.png"
    fig.savefig(outdir / name, dpi=dpi)
    plt.close(fig)
    return [name]

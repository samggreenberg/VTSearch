"""Figures for the #3146 report, from the study's own CSVs.

    python make_fastproc_figs.py --results <dir> [--svg]

Generated, never hand-drawn, and read from the same files the tables are built
from — a figure sourced separately from its table is a second copy of the
finding that drifts from the first.

Four figures, each answering a question the prose cannot:

1. **cost_speedup** — end-to-end embed cost per arm with the measured spread.
   The point of drawing it is the error bar: the first version of this
   measurement was one run per arm, and the same code run twice differed by 8%,
   which is the size of the effect. The bars carry that history.
2. **cost_breakdown** — where a pile cell's wall clock actually goes. A stage
   that is 3.8x faster in isolation and 9% of the cell is a different decision
   from one that is 3.8x faster and 68% of it, and the issue's projection was
   the second.
3. **pixel_vs_vector** — the perturbation at the input against the perturbation
   at the output, per arm and per embedder. The two invert between models, so
   any single-model headline is wrong for the other one.
4. **topk_and_top1** — the user-visible quantity. Cosine drift is not what a
   user sees; whether the first result changed is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fastproc_config as fcfg  # noqa: E402

#: One colour per arm, stable across every figure so a reader can track an arm
#: between them without re-reading a legend.
COLOR = {
    "tv_cpu": "#2b6cb0",
    "tv_cpu_rep": "#999999",
    "pil_cpu": "#b7791f",
    "tv_cuda": "#2f855a",
}
LABEL = {
    "tv_cpu": "torchvision/cpu\n(shipped today)",
    "tv_cpu_rep": "torchvision/cpu\n(same code, rerun)",
    "pil_cpu": "pil/cpu\n(transformers<5)",
    "tv_cuda": "torchvision/cuda\n(candidate)",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def save(fig, outdir: Path, name: str, svg: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=130, bbox_inches="tight")
    if svg:
        fig.savefig(outdir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {name}.png" + (" + .svg" if svg else ""))


def fig_cost_speedup(results: Path, outdir: Path, svg: bool) -> None:
    path = results / "timing_arms.json"
    if not path.exists():
        log("  (no timing_arms.json — skipping cost_speedup)")
        return
    d = json.loads(path.read_text())
    samples = {tuple(k.split("|")): v for k, v in d["samples"].items()}
    embs = sorted({e for e, _ in samples})
    fig, axes = plt.subplots(1, len(embs), figsize=(4.2 * len(embs), 3.8), squeeze=False)
    for ax, emb in zip(axes[0], embs):
        arms = [a for e, a in samples if e == emb]
        arms.sort(key=lambda a: -np.median(samples[(emb, a)]))
        xs = np.arange(len(arms))
        meds = [np.median(samples[(emb, a)]) for a in arms]
        ses = [np.std(samples[(emb, a)], ddof=1) / np.sqrt(len(samples[(emb, a)])) for a in arms]
        ax.bar(xs, meds, yerr=ses, capsize=4, color=[COLOR.get(a, "#666") for a in arms])
        # Every rep drawn, not just the summary: the whole reason this figure
        # exists is that a summary of one run hid an 8% spread.
        for x, a in zip(xs, arms):
            ys = samples[(emb, a)]
            ax.scatter(
                np.full(len(ys), x) + (np.random.default_rng(0).random(len(ys)) - 0.5) * 0.25,
                ys,
                s=12,
                color="k",
                zorder=3,
                alpha=0.7,
            )
        ax.set_xticks(xs, [LABEL.get(a, a) for a in arms], fontsize=7)
        ax.set_ylabel(f"seconds for {d['n_medias']} medias")
        ax.set_title(f"{emb}\n{d['reps']} interleaved reps, {d['gpu']}", fontsize=9)
        base = np.median(samples.get((emb, fcfg.REFERENCE_ARM), [np.nan]))
        for x, m in zip(xs, meds):
            ax.text(x, m, f"{base / m:.2f}x", ha="center", va="bottom", fontsize=8)
    fig.suptitle("End-to-end embed cost — bars are median ± SE, dots are individual reps", fontsize=10)
    save(fig, outdir, "cost_speedup", svg)


def fig_cost_breakdown(results: Path, outdir: Path, svg: bool) -> None:
    """Where a pile cell's wall clock goes: the denominator that decides this."""
    timing = results / "timing_arms.json"
    pixels = results / "pixel_drift.csv"
    if not (timing.exists() and pixels.exists()):
        log("  (missing timing_arms.json or pixel_drift.csv — skipping cost_breakdown)")
        return
    d = json.loads(timing.read_text())
    samples = {tuple(k.split("|")): v for k, v in d["samples"].items()}
    px = pd.read_csv(pixels)
    n_pile = 4193  # medias in the visual_genome_m cell these arms built

    embs = sorted({e for e, _ in samples})
    fig, ax = plt.subplots(figsize=(7.5, 0.9 * len(embs) + 2.2))
    ys, labels = [], []
    for i, emb in enumerate(embs):
        ref = samples.get((emb, fcfg.REFERENCE_ARM))
        if not ref:
            continue
        embed_s = np.median(ref) * n_pile / d["n_medias"]
        row = px[(px.embedder == emb) & (px.backend_requested == "torchvision") & (px.device == "cpu")]
        proc_s = float(row["ms_per_image"].iloc[0]) * n_pile / 1000 if len(row) else np.nan
        cell_s = {"siglip": 86.8, "siglip2_l": 182.4}.get(emb, np.nan)  # measured, from provenance
        ys.append((i, proc_s, embed_s, cell_s))
        labels.append(emb)
    for i, proc_s, embed_s, cell_s in ys:
        ax.barh(i, cell_s, color="#e2e8f0", label="rest of the pile cell" if i == 0 else None)
        ax.barh(i, embed_s, color="#90cdf4", label="embed path (decode+processor+forward)" if i == 0 else None)
        ax.barh(i, proc_s, color="#2b6cb0", label="image processor alone" if i == 0 else None)
        ax.text(
            cell_s,
            i,
            f"  processor = {proc_s / cell_s * 100:.0f}% of the cell, {proc_s / embed_s * 100:.0f}% of the embed path",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(f"seconds to build one {fcfg.DATASET} cell ({n_pile} medias)")
    ax.set_title(
        "The processor is a large share of the embed path and a small share of the cell\n"
        "#3146 projected 28-68%; that projection was of the embed path, and assumed the PIL backend",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower right")
    save(fig, outdir, "cost_breakdown", svg)


def fig_pixel_vs_vector(results: Path, outdir: Path, svg: bool) -> None:
    dr = results / "drift.csv"
    px = results / "pixel_drift.csv"
    if not (dr.exists() and px.exists()):
        log("  (missing drift.csv or pixel_drift.csv — skipping pixel_vs_vector)")
        return
    drift = pd.read_csv(dr)
    pixel = pd.read_csv(px)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    markers = {"siglip": "o", "siglip2_l": "s"}
    for _, r in drift.iterrows():
        arm = r["arm"]
        p = pixel[
            (pixel.embedder == r["embedder"])
            & (pixel.backend_requested == r["backend"])
            & (pixel.device == r["proc_device"])
        ]
        if p.empty or "drift_max_median" not in p or not np.isfinite(p["drift_max_median"].iloc[0]):
            continue
        ax.scatter(
            p["drift_max_median"].iloc[0],
            max(r["median"], 1e-17),
            s=90,
            color=COLOR.get(arm, "#666"),
            marker=markers.get(r["embedder"], "^"),
            edgecolor="k",
            linewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            f"{arm}\n{r['embedder']}",
            (p["drift_max_median"].iloc[0], max(r["median"], 1e-17)),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7,
        )
    # One 8-bit level: the quantum both a backend change and a CPU-dispatch
    # change produce, so it is the natural unit for the x axis (#3160).
    ax.axvline(2 / 255, color="#c53030", ls="--", lw=1)
    ax.text(2 / 255, ax.get_ylim()[1], " one 8-bit level (2/255)", color="#c53030", fontsize=7, va="top", rotation=90)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("perturbation at the input (max |Δpixel| in the median image)")
    ax.set_ylabel("perturbation at the output (median 1 − cos)")
    ax.set_title(
        "Input perturbation does not predict output drift\n"
        "the two models invert: pil hurts siglip2_l, cuda hurts siglip",
        fontsize=9,
    )
    save(fig, outdir, "pixel_vs_vector", svg)


def fig_topk_and_top1(results: Path, outdir: Path, svg: bool) -> None:
    path = results / "rank_stability.csv"
    if not path.exists():
        log("  (no rank_stability.csv — skipping topk_and_top1)")
        return
    ranks = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, source in zip(axes, sorted(ranks["source"].unique())):
        sub = ranks[ranks["source"] == source]
        groups = sorted(sub.groupby(["embedder", "arm"]).groups)
        xs = np.arange(len(groups))
        top1 = [sub[(sub.embedder == e) & (sub.arm == a)]["top1_same"].mean() * 100 for e, a in groups]
        top10 = [sub[(sub.embedder == e) & (sub.arm == a)]["top10_overlap"].mean() * 100 for e, a in groups]
        ax.bar(xs - 0.2, top1, width=0.38, color=[COLOR.get(a, "#666") for _, a in groups], label="top-1 unchanged")
        ax.bar(
            xs + 0.2,
            top10,
            width=0.38,
            color=[COLOR.get(a, "#666") for _, a in groups],
            alpha=0.45,
            label="top-10 overlap",
        )
        ax.set_xticks(xs, [f"{e}\n{a}" for e, a in groups], fontsize=6.5)
        ax.set_ylim(90, 100.6)
        ax.axhline(100, color="#999", ls=":", lw=1)
        ax.set_ylabel("% of categories")
        ax.set_title(f"{source}", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle(
        "What a user would notice — solid = the first result is unchanged, faded = top-10 overlap\n"
        "y axis starts at 90%: every arm is high, and the gaps are what matter",
        fontsize=9,
    )
    save(fig, outdir, "topk_and_top1", svg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(fcfg.results_dir()))
    ap.add_argument("--out", default=None)
    ap.add_argument("--svg", action="store_true", help="also emit vector copies for the HTML reading copy")
    args = ap.parse_args(argv)

    results = Path(args.results)
    outdir = Path(args.out) if args.out else results / "figures"
    log(f"figures from {results} -> {outdir}")
    fig_cost_speedup(results, outdir, args.svg)
    fig_cost_breakdown(results, outdir, args.svg)
    fig_pixel_vs_vector(results, outdir, args.svg)
    fig_topk_and_top1(results, outdir, args.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

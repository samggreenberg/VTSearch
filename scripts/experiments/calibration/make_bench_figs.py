"""Figures for the production-defaults overview benchmark.

Every figure here comes from the same per-step cell CSVs the report's tables
come from, so a figure and a table can never disagree. Two things the tables
genuinely cannot show, and the reason this script exists:

* **What happens over the ramp.** A deep-regime mean is one number per arm; the
  question a user has is "what do I get after 20 clicks, and after 100?" — and
  an average across a crossover is precisely the number that hides it.
* **What a single run looks like.** Averaged over seeds and categories, every
  arm descends smoothly. Individual traces do not: they plateau, spike, and
  sometimes never leave the floor. A benchmark that only reports means is
  claiming a typical run behaves like the mean, which is the claim these
  figures exist to check.

Usage (paths default to the study's scratch dirs):

    python make_bench_figs.py --out docs/experiments/overview-bench/figures \\
        --wave1 /expscratch/$USER/bench-overview/results \\
        --vgbox /expscratch/$USER/bench-vgbox2/results \\
        --binary /expscratch/$USER/bench-binary/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Vector output settings, for the SVG pass the HTML build consumes: keep text as
# text (smaller, and selectable) and let matplotlib drop collinear points from
# long line paths, which is most of the weight of a 170-trace plot.
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bench_cells import CELL_KEY, load_cells  # noqa: E402

#: One colour per representation, held fixed across every figure so a colour
#: means the same thing everywhere in the report.
COLOR = {
    "siglip": "#3B6EA8",
    "siglip2_l": "#D08428",
    "dinov3_patch": "#3E8A6E",
    "dinov3_patch (binary)": "#B3574A",
}
#: Datasets in the order the report discusses them: easy → hard, then the
#: purpose-built box-area axis, largest box first.
DATASET_ORDER = [
    "caltech101_m",
    "coco_val",
    "visual_genome_m",
    "vg_box_large",
    "vg_box_medium",
    "vg_box_small",
]
BAND_ORDER = ["vg_box_large", "vg_box_medium", "vg_box_small"]
T_GRID = np.arange(2, 151)


def interp_cell(cell: pd.DataFrame, metric: str, grid: np.ndarray = T_GRID) -> np.ndarray:
    """One cell's *metric* on the shared vote grid, NaN outside its own range.

    A cell that stopped at t=63 (it found its first positive at vote 87) must
    not be extended flat to 150: that would quietly turn a starved run into a
    well-behaved one at exactly the votes where the claim is being made.
    """
    s = cell.sort_values("t")
    vals = np.interp(grid, s["t"].to_numpy(), s[metric].to_numpy(), left=np.nan, right=np.nan)
    return np.where((grid >= s["t"].min()) & (grid <= s["t"].max()), vals, np.nan)


def stack_cells(sub: pd.DataFrame, metric: str) -> np.ndarray:
    """(n_cells, n_grid) matrix of *metric*, one row per (category, seed)."""
    rows = [interp_cell(g, metric) for _, g in sub.groupby(["category", "seed"])]
    return np.vstack(rows) if rows else np.empty((0, len(T_GRID)))


def boot_band(stack: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap 95% band for the mean over cells (resampling cells, not steps)."""
    n = stack.shape[0]
    if n < 3:
        m = np.nanmean(stack, axis=0)
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    with np.errstate(invalid="ignore"):
        means = np.nanmean(stack[idx], axis=1)
    return np.nanpercentile(means, 2.5, axis=0), np.nanpercentile(means, 97.5, axis=0)


#: Also emit SVG beside each PNG. The HTML build picks whichever is smaller per
#: figure, so a line plot ships as crisp vector art while the 170-trace spaghetti
#: plot (huge as SVG) stays a bitmap.
ALSO_SVG = False


def _finish(fig, out: Path, note: str | None = None) -> None:
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=8, color="#555")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    if ALSO_SVG:
        svg = out.with_suffix(".svg")
        # No Date in the metadata: otherwise every regeneration rewrites all
        # eight files with a new timestamp, and a committed artifact churns for
        # reasons that have nothing to do with the data.
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        # matplotlib leaves trailing spaces, which the repo's whitespace hook
        # would rewrite - and a hook-rewritten artifact no longer matches what
        # the generator produces.
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().split("\n")))
    plt.close(fig)
    print(f"  wrote {out}{' (+svg)' if ALSO_SVG else ''}")


def _label_bottom_row(axes, xlabel: str, ncols: int = 3) -> None:
    """Label x on the last panel of each column, and re-show the shared ticks.

    `sharex` hides tick labels on every row but the last, so a panel in the top
    row would otherwise carry an axis label above no numbers at all.
    """
    for i, ax in enumerate(axes):
        ax.tick_params(labelbottom=True)
        if i + ncols >= len(axes):
            ax.set_xlabel(xlabel)


def _panel_grid(datasets: list[str], ncols: int = 3, height: float = 2.9):
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, height * nrows), squeeze=False, sharex=True)
    return fig, axes.ravel()[: len(datasets)], axes.ravel()


# --------------------------------------------------------------------------
# 1. cost over the ramp, averaged over seeds and categories
# --------------------------------------------------------------------------
def fig_cost_vs_votes(df: pd.DataFrame, out: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    fig, axes, allaxes = _panel_grid(datasets)
    for ax, ds in zip(axes, datasets, strict=True):
        sub_ds = df[df["dataset"] == ds]
        for emb in sorted(sub_ds["embedder"].unique()):
            sub = sub_ds[sub_ds["embedder"] == emb]
            stack = stack_cells(sub, "cost")
            if not stack.size:
                continue
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(stack, axis=0)
            lo, hi = boot_band(stack)
            color = COLOR.get(emb, "#777")
            ax.plot(T_GRID, mean, color=color, lw=1.9, label=f"{emb} (n={stack.shape[0]})")
            ax.fill_between(T_GRID, lo, hi, color=color, alpha=0.15, lw=0)
        ax.set_title(ds, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False)
    for ax in allaxes[len(datasets) :]:
        ax.axis("off")
    _label_bottom_row(axes, "votes cast (t)")
    axes[0].set_ylabel("cost = fpr + fnr")
    fig.suptitle("Cost over the voting ramp, mean over seeds x categories (band: bootstrap 95% CI over cells)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(
        fig, out, "Categories differ in prevalence, so read down a panel (arms on the same data), not across panels."
    )


# --------------------------------------------------------------------------
# 2. the same thing, one line per run
# --------------------------------------------------------------------------
def fig_cost_traces(df: pd.DataFrame, out: Path, datasets: list[str]) -> None:
    embs = [e for e in ("siglip", "siglip2_l", "dinov3_patch") if e in set(df["embedder"])]
    fig, axes = plt.subplots(
        len(datasets), len(embs), figsize=(4.0 * len(embs), 2.9 * len(datasets)), squeeze=False, sharex=True
    )
    for r, ds in enumerate(datasets):
        for c, emb in enumerate(embs):
            ax = axes[r][c]
            sub = df[(df["dataset"] == ds) & (df["embedder"] == emb)]
            color = COLOR.get(emb, "#777")
            n = 0
            for _, g in sub.groupby(["category", "seed"]):
                s = g.sort_values("t")
                ax.plot(s["t"], s["cost"], color=color, lw=0.7, alpha=0.45)
                n += 1
                # A run that stopped early ended because it had no positive to
                # train on until late; mark where it actually began.
                if s["t"].min() > 10:
                    ax.plot(s["t"].iloc[0], s["cost"].iloc[0], "o", ms=3.5, color="#B3574A", zorder=5)
            stack = stack_cells(sub, "cost")
            if stack.size:
                with np.errstate(invalid="ignore"):
                    ax.plot(T_GRID, np.nanmedian(stack, axis=0), color="black", lw=2.0, label="median run")
                ax.plot(T_GRID, np.nanmean(stack, axis=0), color=color, lw=2.0, ls="--", label="mean")
            ax.set_title(f"{ds} x {emb}  ({n} runs)", fontsize=9)
            ax.grid(alpha=0.25)
            ax.set_ylim(0, min(1.6, max(0.6, float(np.nanmax(stack)) if stack.size else 1.0)))
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False)
            if r == len(datasets) - 1:
                ax.set_xlabel("votes cast (t)")
            if c == 0:
                ax.set_ylabel("cost = fpr + fnr")
    fig.suptitle("Every individual run (one line per category x seed); red dot = the run's first scored step", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, out, "A late first step means Autopilot needed that many votes to surface a first positive.")


# --------------------------------------------------------------------------
# 3. positives found: the binding constraint
# --------------------------------------------------------------------------
def fig_positives(df: pd.DataFrame, out: Path) -> None:
    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    fig, axes, allaxes = _panel_grid(datasets)
    for ax, ds in zip(axes, datasets, strict=True):
        sub_ds = df[df["dataset"] == ds]
        for emb in sorted(sub_ds["embedder"].unique()):
            sub = sub_ds[sub_ds["embedder"] == emb]
            stack = stack_cells(sub, "n_good")
            if not stack.size:
                continue
            color = COLOR.get(emb, "#777")
            with np.errstate(invalid="ignore"):
                ax.plot(T_GRID, np.nanmedian(stack, axis=0), color=color, lw=1.9, label=emb)
                ax.fill_between(
                    T_GRID,
                    np.nanpercentile(stack, 10, axis=0),
                    np.nanpercentile(stack, 90, axis=0),
                    color=color,
                    alpha=0.12,
                    lw=0,
                )
        ax.plot(T_GRID, T_GRID, color="#999", lw=0.8, ls=":", label="every vote a positive")
        ax.set_title(ds, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")
        ax.legend(fontsize=7, frameon=False)
    for ax in allaxes[len(datasets) :]:
        ax.axis("off")
    _label_bottom_row(axes, "votes cast (t)")
    axes[0].set_ylabel("positives held (median, 10-90%)")
    fig.suptitle("What the loop is actually given: positives accumulated per 150 votes", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, out, "Log y axis. The dotted diagonal is the unreachable ceiling where every vote is a positive.")


# --------------------------------------------------------------------------
# 4. the scale axis
# --------------------------------------------------------------------------
def fig_scale_bands(df: pd.DataFrame, out: Path) -> None:
    deep = df[(df["t"] >= 100) & df["dataset"].isin(BAND_ORDER)]
    if deep.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    x = np.arange(len(BAND_ORDER))
    for metric, ax, label in zip(
        ("cost", "average_precision", "fnr"),
        axes,
        ("cost = fpr + fnr (lower better)", "average precision (higher better)", "false-negative rate"),
        strict=True,
    ):
        for emb in sorted(deep["embedder"].unique()):
            per_cell = deep[deep["embedder"] == emb].groupby(CELL_KEY)[metric].mean().reset_index()
            means = [per_cell[per_cell["dataset"] == b][metric].mean() for b in BAND_ORDER]
            errs = [
                per_cell[per_cell["dataset"] == b][metric].sem() if len(per_cell[per_cell["dataset"] == b]) > 1 else 0.0
                for b in BAND_ORDER
            ]
            ax.errorbar(x, means, yerr=errs, marker="o", lw=1.9, capsize=3, color=COLOR.get(emb, "#777"), label=emb)
        ax.set_xticks(x, ["large\n>1/12", "medium\n1/196-1/12", "small\n<1/196"], fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Target scale: box area as a fraction of the image (deep regime, t >= 100)", y=1.02)
    fig.tight_layout()
    _finish(fig, out, "10 prevalence-spread categories per band, 3 seeds; bars are standard error over cells.")


# --------------------------------------------------------------------------
# 5. where the error budget sits
# --------------------------------------------------------------------------
def fig_error_budget(df: pd.DataFrame, out: Path) -> None:
    deep = df[df["t"] >= 100]
    datasets = [d for d in DATASET_ORDER if d in set(deep["dataset"])]
    embs = sorted(deep["embedder"].unique(), key=lambda e: list(COLOR).index(e) if e in COLOR else 9)
    fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 2, 4.0))
    width = 0.8 / max(len(embs), 1)
    for i, emb in enumerate(embs):
        xs, fprs, fnrs = [], [], []
        for j, ds in enumerate(datasets):
            sub = deep[(deep["dataset"] == ds) & (deep["embedder"] == emb)]
            if sub.empty:
                continue
            xs.append(j + (i - (len(embs) - 1) / 2) * width)
            fprs.append(sub["fpr"].mean())
            fnrs.append(sub["fnr"].mean())
        color = COLOR.get(emb, "#777")
        ax.bar(xs, fprs, width * 0.92, color=color, label=f"{emb} — fpr")
        ax.bar(xs, fnrs, width * 0.92, bottom=fprs, color=color, alpha=0.45, hatch="//", label=f"{emb} — fnr")
    ax.set_xticks(range(len(datasets)), datasets, fontsize=8, rotation=12)
    ax.set_ylabel("cost = fpr + fnr (solid = fpr, hatched = fnr)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, frameon=False, ncol=len(embs))
    ax.set_title("Which error each configuration makes (deep regime, t >= 100)", fontsize=11)
    fig.tight_layout()
    _finish(fig, out, "Bar height is cost; the split says whether an arm over-includes (fpr) or misses (fnr).")


# --------------------------------------------------------------------------
# 6. boxes vs no boxes: the interaction axis, isolated
# --------------------------------------------------------------------------
def fig_binary_vs_boxes(wave1: pd.DataFrame, binary: pd.DataFrame, out: Path) -> None:
    datasets = ["visual_genome_m", "coco_val"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 3.6), squeeze=False)
    for ax, ds in zip(axes[0], datasets, strict=True):
        series = [
            (
                wave1[(wave1["dataset"] == ds) & (wave1["embedder"] == "siglip")],
                "siglip",
                "siglip (no box possible)",
                "-",
            ),
            (
                wave1[(wave1["dataset"] == ds) & (wave1["embedder"] == "siglip2_l")],
                "siglip2_l",
                "siglip2_l (no box possible)",
                "-",
            ),
            (
                binary[binary["dataset"] == ds],
                "dinov3_patch (binary)",
                "dinov3_patch — box NOT drawn",
                "-",
            ),
            (
                wave1[(wave1["dataset"] == ds) & (wave1["embedder"] == "dinov3_patch")],
                "dinov3_patch",
                "dinov3_patch — box drawn",
                "--",
            ),
        ]
        for sub, key, label, style in series:
            stack = stack_cells(sub, "cost")
            if not stack.size:
                continue
            with np.errstate(invalid="ignore"):
                ax.plot(T_GRID, np.nanmean(stack, axis=0), color=COLOR[key], lw=2.0, ls=style, label=label)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("votes cast (t)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("cost = fpr + fnr")
    fig.suptitle("The same cells, with and without a drawn box (whole-image arms shown for scale)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(
        fig,
        out,
        "Cell-for-cell paired: same datasets, categories, seeds and splits; only the Good-vote vector differs.",
    )


# --------------------------------------------------------------------------
# 7. typing vs clicking
# --------------------------------------------------------------------------
def fig_text_vs_detector(df: pd.DataFrame, text_csv: Path, out: Path) -> None:
    if not text_csv.exists():
        print(f"  (no {text_csv}; skipping the text figure)")
        return
    text = pd.read_csv(text_csv)
    text = text[text.get("supports_text", 1) == 1]
    datasets = [d for d in DATASET_ORDER if d in set(text["dataset"]) and d in set(df["dataset"])]
    embs = ["siglip", "siglip2_l"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets, strict=True):
        for emb in embs:
            sub = df[(df["dataset"] == ds) & (df["embedder"] == emb)]
            stack = stack_cells(sub, "cost")
            color = COLOR[emb]
            if stack.size:
                with np.errstate(invalid="ignore"):
                    ax.plot(T_GRID, np.nanmean(stack, axis=0), color=color, lw=1.9, label=f"{emb} — clicked")
            t = text[(text["dataset"] == ds) & (text["embedder"] == emb)]
            if not t.empty:
                ax.axhline(t["text_cost"].mean(), color=color, lw=1.4, ls=":", label=f"{emb} — typed (0 clicks)")
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("votes cast (t)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    axes[0][0].set_ylabel("cost = fpr + fnr")
    fig.suptitle("Zero-click typed query (dotted) against the clicked detector's ramp (solid)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(fig, out, "Text queries are raw category names; `dinov3_patch` has no text tower and cannot appear here.")


# --------------------------------------------------------------------------
# 8. what regret is made of
# --------------------------------------------------------------------------
def _grouped_bars(ax, deep: pd.DataFrame, metric: str, datasets: list[str], embs: list[str]) -> None:
    width = 0.8 / max(len(embs), 1)
    for i, emb in enumerate(embs):
        xs, vals = [], []
        for j, ds in enumerate(datasets):
            sub = deep[(deep["dataset"] == ds) & (deep["embedder"] == emb)]
            if sub.empty:
                continue
            xs.append(j + (i - (len(embs) - 1) / 2) * width)
            vals.append(sub[metric].mean())
        ax.bar(xs, vals, width * 0.92, color=COLOR.get(emb, "#777"), label=emb)
    ax.axhline(0, color="#444", lw=0.9)
    ax.set_xticks(range(len(datasets)), datasets, fontsize=8, rotation=14)
    ax.grid(alpha=0.3, axis="y")


def fig_regret_decomposition(df: pd.DataFrame, out: Path) -> None:
    deep = df[(df["t"] >= 100)].dropna(subset=["rule_inefficiency", "calibration_shift"])
    if deep.empty:
        print("  (no decomposition rows; skipping the regret figure)")
        return
    datasets = [d for d in DATASET_ORDER if d in set(deep["dataset"])]
    embs = [e for e in COLOR if e in set(deep["embedder"])]
    # Deliberately NOT a stacked bar: rule inefficiency is negative nearly
    # everywhere, and stacking a negative term on a positive one draws the two
    # overlapping - which reads as a large positive contribution, the exact
    # opposite of the finding.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 3.8), sharex=True)
    _grouped_bars(axes[0], deep, "calibration_shift", datasets, embs)
    axes[0].set_ylabel("calibration shift (sim -> test)")
    axes[0].set_title("What regret is made of: the sim->test move", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    _grouped_bars(axes[1], deep, "rule_inefficiency", datasets, embs)
    axes[1].set_ylabel("rule inefficiency")
    axes[1].set_title("...and what the cut rule itself costs (negative = the shipped cut wins)", fontsize=10)
    fig.suptitle(
        "Regret decomposition, deep regime (t >= 100). Both panels are in cost units — note the different y scales.",
        y=1.02,
    )
    fig.tight_layout()
    _finish(
        fig,
        out,
        "A negative rule inefficiency means the shipped cut beats the best threshold fitted on the sim half — "
        "so acquisition, not the cut rule, is where the remaining headroom is.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wave1", default="/expscratch/sgreenberg/bench-overview/results")
    ap.add_argument("--vgbox", default="/expscratch/sgreenberg/bench-vgbox2/results")
    ap.add_argument("--binary", default="/expscratch/sgreenberg/bench-binary/results")
    ap.add_argument("--text-csv", default=None, help="text_baseline.csv (defaults to the wave-1 results dir's copy)")
    ap.add_argument("--svg", action="store_true", help="also write an SVG beside each PNG (for the HTML build)")
    args = ap.parse_args()

    global ALSO_SVG
    ALSO_SVG = args.svg

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("loading cells:")
    wave1, _ = load_cells(Path(args.wave1))
    vgbox, _ = load_cells(Path(args.vgbox))
    binary, _ = load_cells(Path(args.binary), embedder_suffix=" (binary)")
    boxed = pd.concat([wave1, vgbox], ignore_index=True)
    text_csv = Path(args.text_csv) if args.text_csv else Path(args.wave1) / "text_baseline.csv"

    print("figures:")
    fig_cost_vs_votes(boxed, out / "fig_cost_vs_votes.png")
    fig_cost_traces(boxed, out / "fig_cost_traces.png", ["visual_genome_m", "vg_box_small"])
    fig_positives(boxed, out / "fig_positives.png")
    fig_scale_bands(vgbox, out / "fig_scale_bands.png")
    fig_error_budget(boxed, out / "fig_error_budget.png")
    if not binary.empty:
        fig_binary_vs_boxes(wave1, binary, out / "fig_binary_vs_boxes.png")
    fig_text_vs_detector(wave1, text_csv, out / "fig_text_vs_detector.png")
    fig_regret_decomposition(boxed, out / "fig_regret_decomposition.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

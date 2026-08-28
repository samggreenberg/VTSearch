"""Learning curves: how the simulated user's detector improves as they click.

The question these answer is the one a user actually asks — *if I keep voting,
does this get better, and how fast?* — and it is asked twice, because the two
readings want different pictures:

1. **Averaged** (``learning_<metric>.png``, and ``_by_band``): one line per arm,
   the mean over every run on the dataset, with a ±SE ribbon. This is "how does
   this arm TEND to do", and it is the picture for comparing arms.
2. **Individual** (``learning_<metric>_runs.png``, and ``_by_band``): one PANEL
   per arm, every run on the dataset drawn as its own thin line, with the median
   and the 10–90% band over the top. This is "what actually happens to people",
   and it is the picture a mean cannot show: an arm whose median is fine and
   whose bottom decile never leaves the floor fails those users completely.

Splitting the runs into one panel per arm rather than overlaying every arm's
spaghetti in one is deliberate — at 60 seeds x 36 categories an overlay is 2160
lines in three hues, which encodes nothing.

Both are drawn per *band* as well, because band is this study's axis: a curve
pooled over `small`, `medium` and `large` is an average over the very thing the
grid was built to separate.

Colours, band names and the run-splitting helpers come from
:mod:`figures_overview` rather than being restated here — an arm's hue has to
mean the same thing in every figure of the report, and two copies of a colour
table is how it stops.

Usage::

    python figures_trajectory.py --exp /expscratch/$USER/scale-3156-pair
    python figures_trajectory.py --exp ... --metric average_precision
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

from figures_overview import BANDS, MODE_COLORS, band_of, mean_se

#: Metrics worth a curve, and which direction is good.  ``cost`` is the study's
#: headline (the harness's operating-point cost); ``average_precision`` is the
#: ranking's own quality, which is what improves first when the detector learns
#: something the cut cannot yet exploit.
METRICS = {
    "cost": ("detection cost (lower is better)", False),
    "average_precision": ("average precision (higher is better)", True),
    "auroc": ("AUROC (higher is better)", True),
    "oracle_cost": ("oracle cost — the ranking's own limit (lower is better)", False),
}


def load_runs(exp: str, metric: str) -> dict[tuple, list[tuple[int, float]]]:
    """``(mode, category, seed) -> [(t, value), ...]`` sorted by t.

    Side frames (``task_*__sweep.csv`` and friends) are excluded by the ``__``
    test, the same guard ``_cells_io.SIDE_FRAME_SUFFIXES`` exists to centralise:
    they are separate long-format tables and concatenating them yields a ragged
    frame whose extra rows enter every aggregate silently.
    """
    runs: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for path in sorted(glob.glob(str(Path(exp) / "results" / "cells" / "task_*.csv"))):
        if "__" in Path(path).name:
            continue
        try:
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        t, v = int(r["t"]), float(r[metric])
                    except (KeyError, ValueError, TypeError):
                        continue
                    mode = f"{r.get('embedder', '')}/{r.get('style', '')}".strip("/")
                    runs[(mode, r["category"], r["seed"])].append((t, v))
        except (OSError, csv.Error):
            continue
    for v in runs.values():
        v.sort()
    return runs


def carry_forward(series: list[tuple[int, float]], steps: list[int]) -> list[float | None]:
    """The run's value at each of *steps*, carried forward from its last row.

    A run does not emit a row at every t — the harness skips steps with too few
    votes to train — so sampling "the row where t == step" silently drops runs
    from the average at exactly the steps where runs differ most.  Carrying the
    last value forward is what a user would see on screen: the detector does not
    stop existing between votes.

    Done as ONE merge over both sorted lists rather than a scan per step. The
    scan-per-step version is quadratic in the run's length, which is invisible
    at 1000 runs and is an hour of a chained, unattended job at 6480.
    """
    out: list[float | None] = []
    i, last = 0, None
    for step in steps:
        while i < len(series) and series[i][0] <= step:
            last = series[i][1]
            i += 1
        out.append(last)
    return out


def q(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def _style_axes(ax, xlabel: str, ylabel: str) -> None:
    """Hairline, solid, recessive — grid and axes are chrome, not data."""
    ax.grid(True, color="#e6e4e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c6c1")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#57534e", labelsize=8, length=3, width=0.8)
    ax.set_xlabel(xlabel, fontsize=9, color="#44403c")
    ax.set_ylabel(ylabel, fontsize=9, color="#44403c")


def _subset(runs, modes, band):
    """Runs restricted to one band, or all of them when *band* is None."""
    return {k: v for k, v in runs.items() if k[0] in modes and (band is None or band_of(k[1]) == band)}


def _grid(runs, steps):
    """``key -> [value at each step]``, computed once and reused by every figure."""
    return {k: carry_forward(v, steps) for k, v in runs.items()}


def _mean_curve(grid, mode, steps):
    """(steps, mean, se, n) for one arm — the ±SE is the mean's own uncertainty."""
    series = [v for k, v in grid.items() if k[0] == mode]
    xs, ms, ses, ns = [], [], [], []
    for i, s in enumerate(steps):
        vals = [row[i] for row in series if row[i] is not None]
        if len(vals) < 2:
            continue
        m, se = mean_se(vals)
        xs.append(s)
        ms.append(m)
        ses.append(se)
        ns.append(len(vals))
    return xs, ms, ses, ns


def load_anchor(path: str, metric: str) -> dict[tuple[str, str], float]:
    """``(embedder, category) -> the zero-click text sort's value of *metric*``.

    The free path through the product -- type a query, read the ranked haystack
    under the same cut -- is what every one of these curves has to beat, and a
    figure that omits it invites the reader to compare the arms only with each
    other. Averaged over whatever seeds the baseline was computed on: the text
    sort does not depend on the votes, only on the split, so a handful of seeds
    fixes it.
    """
    from curves import BASELINE_COLUMNS  # one metric -> baseline-column map, not two

    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    cols = (*BASELINE_COLUMNS.get(metric, ()), metric)
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("supports_text") not in (None, "", "1"):
                    continue
                col = next((c for c in cols if c in r and r[c] not in (None, "")), None)
                if col is None:
                    continue
                try:
                    out[(r["embedder"], r["category"])].append(float(r[col]))
                except (KeyError, ValueError, TypeError):
                    continue
    except OSError:
        return {}
    return {k: sum(v) / len(v) for k, v in out.items()}


def _stamp(fig, note: str) -> None:
    """Say so, on the picture, when the data behind it is not the whole run."""
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color="#a8a29e", ha="left", va="bottom")


def figure_average(
    grid,
    modes,
    steps,
    out: Path,
    metric: str,
    label: str,
    band: str | None,
    plt,
    note: str = "",
    anchor: dict[tuple[str, str], float] | None = None,
) -> None:
    """One panel: every arm's mean curve on one axis."""
    sub = _subset(grid, modes, band)
    grid = sub
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    drew_anchor: list[bool] = []
    n_by_mode = {}
    ends: list[tuple[float, float, str]] = []
    for m in modes:
        xs, ms, ses, ns = _mean_curve(sub, m, steps)
        if not xs:
            continue
        n_by_mode[m] = max(ns)
        colour = MODE_COLORS.get(m, "#57534e")
        ax.fill_between(
            xs,
            [a - b for a, b in zip(ms, ses)],
            [a + b for a, b in zip(ms, ses)],
            color=colour,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(xs, ms, color=colour, linewidth=2.0, zorder=3, label=f"{m}  (n={max(ns)})")
        ends.append((ms[-1], xs[-1], colour))
        # Click 0 is this arm's own text sort, drawn as a point rather than as a
        # horizontal rule across the panel -- the same convention the viewer
        # uses, and for the same reason: a rule implies a level that holds at
        # every click when it holds at one, and it dominates the figure to make
        # a point the leftmost marker already makes.
        if anchor:
            emb = m.split("/")[0]
            vals = [anchor[(emb, c)] for c in {k[1] for k in grid if k[0] == m} if (emb, c) in anchor]
            if vals:
                ax.plot([0], [sum(vals) / len(vals)], marker="o", markersize=5.5, color=colour, zorder=7)
                drew_anchor.append(True)
    # Direct-label the endpoints -- three series, so identity never rests on the
    # legend alone -- but de-collide them first.  Two arms that FINISH IN A TIE
    # is the most interesting thing this figure can show, and it is exactly the
    # case where two labels land on the same pixel and render as mush.
    ends.sort()
    # The gap has to be a TEXT height, so it is measured against the axis range,
    # not against the spread of the endpoints. Two arms finishing 0.003 apart is
    # the interesting case, and scaling the gap by that spread reproduces the
    # collision it exists to prevent.
    lo_y, hi_y = ax.get_ylim()
    gap = (hi_y - lo_y) * 0.038
    # `ends` is sorted ascending, so "at least `gap` above the previous label"
    # is a max(), not a search. It was written as a `while` that recomputed
    # `y = placed[-1] + gap` until the gap was satisfied -- which never
    # terminates the moment `placed[-1] + gap == placed[-1]` in floating point,
    # because then `y` stops moving while the condition stays true. That is a
    # LIVE lock with no error and no output: it cost two hours of a dedicated
    # node and three wrong explanations (a quadratic scan, matplotlib artist
    # overhead, node contention) before a faulthandler traceback named the line.
    #
    # A loop whose termination depends on floating-point addition making
    # progress is a loop that should not be a loop.
    placed: list[float] = []
    for value, x, colour in ends:
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        ax.annotate(
            f"{value:.2f}",
            (x, y),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=colour,
            va="center",
            annotation_clip=False,
        )
        if abs(y - value) > 1e-12:
            # A displaced label needs a leader back to the line it belongs to,
            # or it reads as a value the curve never had.
            ax.annotate(
                "",
                (x, y),
                xytext=(x, value),
                textcoords="data",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.7, alpha=0.7),
            )
    _style_axes(ax, "votes spent (clicks)", label)
    title = f"How the detector improves with clicking — {band or 'all bands'}"
    ax.set_title(title, fontsize=11, color="#292524", loc="left", pad=10)
    if drew_anchor:
        ax.set_xlim(left=-3)
        ax.annotate(
            "dot at 0 = the free text sort",
            (0.01, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            color="#57534e",
        )
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    _stamp(fig, note)
    name = f"learning_{metric}{'_' + band if band else ''}.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    counts = ", ".join(f"{k} n={v}" for k, v in n_by_mode.items())
    print(f"  wrote {name}  ({counts})")


def figure_runs(
    runs,
    grid,
    modes,
    steps,
    out: Path,
    metric: str,
    label: str,
    band: str | None,
    plt,
    note: str = "",
    max_lines: int = 0,
) -> None:
    """One panel per arm: every run, with the median and the 10-90% band."""
    from matplotlib.collections import LineCollection  # noqa: PLC0415

    sub = _subset(runs, modes, band)
    gsub = _subset(grid, modes, band)
    fig, axes = plt.subplots(1, len(modes), figsize=(4.6 * len(modes), 4.4), sharey=True, sharex=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, m in zip(axes, modes):
        colour = MODE_COLORS.get(m, "#57534e")
        series = [v for k, v in sub.items() if k[0] == m]
        drawn = series
        if max_lines and len(series) > max_lines:
            # Deterministic stride, not a random sample: the same runs are drawn
            # every time the figure is regenerated, so two versions of it differ
            # only where the DATA differs.
            stride = len(series) / max_lines
            drawn = [series[int(i * stride)] for i in range(max_lines)]
        # One LineCollection, not one plot() per run. Matplotlib builds a Line2D
        # artist per call and the spaghetti is thousands of them: at 2000 runs
        # the per-artist overhead dominated everything else in this script, and
        # the grid this feeds has 6480. The drawn result is identical.
        segs = [[(t, v) for t, v in run] for run in drawn if len(run) > 1]
        if segs:
            ax.add_collection(LineCollection(segs, colors=colour, linewidths=0.3, alpha=0.05, zorder=2))
        rows = [v for k, v in gsub.items() if k[0] == m]
        lo, mid, hi, xs = [], [], [], []
        for i, s in enumerate(steps):
            vals = [row[i] for row in rows if row[i] is not None]
            if len(vals) < 2:
                continue
            xs.append(s)
            lo.append(q(vals, 0.10))
            mid.append(q(vals, 0.50))
            hi.append(q(vals, 0.90))
        # add_collection does not update the data limits, so the axes would keep
        # matplotlib's default 0-1 range and clip every curve.
        ax.autoscale_view()
        if xs:
            # The band has to read THROUGH the spaghetti it summarises, so it
            # carries hairline edges rather than relying on fill alpha alone,
            # and the median gets a surface-coloured halo -- the same trick as a
            # 2px surface ring on overlapping marks.
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.20, linewidth=0, zorder=3)
            for edge in (lo, hi):
                ax.plot(xs, edge, color=colour, linewidth=0.9, alpha=0.85, zorder=4)
            ax.plot(xs, mid, color="#fcfcfb", linewidth=3.6, zorder=5)
            ax.plot(xs, mid, color=colour, linewidth=2.0, zorder=6)
        _style_axes(ax, "votes spent (clicks)", label if ax is axes[0] else "")
        # Say when the texture is a sample. A spaghetti plot silently showing a
        # fifth of the runs is a different claim from one showing all of them,
        # and the picture cannot be told apart by looking.
        shown = "" if len(drawn) == len(series) else f", {len(drawn)} drawn"
        ax.set_title(f"{m}\n{len(series)} runs{shown}", fontsize=9.5, color="#292524", loc="left", pad=8)
    fig.suptitle(
        f"Every run, one line each — {band or 'all bands'}  (band = 10–90%, line = median)",
        fontsize=11,
        color="#292524",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    _stamp(fig, note)
    name = f"learning_{metric}_runs{'_' + band if band else ''}.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    print(f"  wrote {name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default=f"/expscratch/{os.environ.get('USER', 'sgreenberg')}/scale-3156-pair")
    ap.add_argument("--out", default="")
    ap.add_argument("--metric", default="cost,average_precision", help="comma-separated; see METRICS")
    ap.add_argument(
        "--note",
        default="",
        help="stamped on every figure -- use it to say a run is still in flight, "
        "since a partial figure that looks final is how a preview becomes a fact",
    )
    ap.add_argument(
        "--max-run-lines",
        type=int,
        default=1200,
        help="cap on the spaghetti drawn per panel (0 = all). The 10-90%% band and the "
        "median are always computed over EVERY run; only the texture is sampled.",
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="text_baseline.py CSV; adds each arm's zero-click text sort as its own point at click 0",
    )
    ap.add_argument("--by-band", action="store_true", default=True)
    ap.add_argument("--no-by-band", dest="by_band", action="store_false")
    args = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out or (Path(args.exp) / "figures"))
    out.mkdir(parents=True, exist_ok=True)

    for metric in [m.strip() for m in args.metric.split(",") if m.strip()]:
        label = METRICS.get(metric, (metric, False))[0]
        runs = load_runs(args.exp, metric)
        if not runs:
            print(f"{metric}: no rows under {args.exp}")
            continue
        modes = sorted({k[0] for k in runs})
        seeds = sorted({k[2] for k in runs}, key=lambda s: int(s) if s.isdigit() else s)
        steps = sorted({t for v in runs.values() for t, _ in v})
        print(f"{metric}: {len(runs)} runs, {len(modes)} arms, {len(seeds)} seeds present")
        note = args.note or f"{len(seeds)} seeds present, {len(runs)} runs"
        grid = _grid(runs, steps)
        anchor = load_anchor(args.baseline, metric) if args.baseline else None
        figure_average(grid, modes, steps, out, metric, label, None, plt, note, anchor)
        figure_runs(runs, grid, modes, steps, out, metric, label, None, plt, note, args.max_run_lines)
        if args.by_band:
            for band in BANDS:
                if any(band_of(k[1]) == band for k in runs):
                    figure_average(grid, modes, steps, out, metric, label, band, plt, note, anchor)
                    figure_runs(runs, grid, modes, steps, out, metric, label, band, plt, note, args.max_run_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

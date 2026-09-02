"""The standard **quality-over-clicks** figures for a simulated-user study.

Every study that drives the Autopilot loop is, from the user's side, the same
story: *"I keep clicking — is the thing I am building getting any better, and is
it better than what I got for free by typing?"*  A deep-regime table cannot
answer that, and neither can a mining curve: positives found is what the
**acquisition** did, while cost is what the **detector** is worth.  An arm can
mine well and still rank badly.

So this module is the one implementation of the two figures every such report
owes, and every analyzer calls it rather than rolling its own:

``{metric}_vs_clicks.png``
    **The averages.**  One panel per dataset, one line per arm, averaged over
    every seed and category on that dataset, with an inter-quartile band.  This
    is the figure a reader looks at to decide which arm to ship.

``{metric}_vs_clicks_runs__{dataset}.png``
    **The individuals.**  One file per dataset, one panel per arm, and inside
    each panel **every seed of that arm on that dataset as its own line**.  A
    mean hides that some runs never leave the floor, and on this axis the
    spread is routinely the finding: two arms with the same mean can be "every
    run is mediocre" and "half the runs are excellent and half never start".

**Click 0 is the zero-click text sort.**  There is no detector at the far left,
so the natural thing is to start the axis at the first trainable step — and that
throws away the comparison the reader actually wants, because typing a query and
reading the ranked haystack is *free* and is what clicking has to beat.  Pass
``baseline`` (see :func:`text_sort_baseline`) and every curve is anchored at
``t=0`` on that cell's own text-sort quality: the far left is what the query got
for nothing, the far right is what the clicking got, and the distance between
them is the whole value of the loop.

The anchor is **each series' own leftmost point**, not a rule across the panel.
A horizontal reference line dominated the figure to make a point the leftmost
marker already makes, and it implied a level that holds at every click when it
holds at one.  "How many clicks before this beat typing?" is a number, so it is
reported as one — :func:`crossover` — rather than left to be eyeballed off a
crossing.

Three things it refuses to do quietly, because each one turns a failing arm
into a good-looking curve:

1. **Average over a shrinking denominator.**  The main metric frame starts at
   the first *trainable* step — before one Good and one Bad vote coexist there
   is no model, no threshold and no row.  So a starved cell simply is not in
   the average, and an arm that starves on a third of its cells gets its mean
   computed over the two thirds that worked.  Every mean is therefore dashed
   wherever coverage — the fraction of that arm's cells measured at that click
   — is below :data:`SOLID_COVERAGE`.  A dashed line means "this level
   describes a subset".  The stretch between ``t=0`` and the first trainable
   click is dashed by the same rule and for the same reason: nothing was
   measured in there.

   A **coverage strip** under the panel draws that fraction outright, but only
   when it is not already told by the dashing (:func:`_strip_worth_drawing`):
   on a healthy grid every arm ramps to full inside a handful of clicks and
   then holds a flat 100% line across the rest of the axis, which spends a
   quarter of the figure restating one number per arm.  Suppressed, the panel
   title names the click from which everything is measured, and ``coverage``
   is in the CSV either way.
2. **Count a cell that never trained as absent.**  Pass ``denominator`` — one
   row per ``(arm, *keys)`` cell the run *attempted* — and coverage is measured
   against the cells that exist rather than against the cells that produced
   rows.  Without it the two are the same number by construction and the
   coverage strip reads 100% for an arm that starved everywhere.
3. **Silently subsample the per-run panel.**  ``max_runs`` defaults to 0 (draw
   them all).  A caller that sets it gets the cap written into the panel title,
   because a hairball with a third of its lines removed looks like a tighter
   arm, not a truncated figure.

The per-run lines are coloured by category **prevalence** when a prevalence
table is supplied, which is what turns the hairball into an explanation: the
runs sitting along the top are the scarce categories, not random bad luck.

Also writes ``{metric}_vs_clicks.csv`` — the plotted curves as numbers, so the
report can quote a level off the figure and a later study can re-plot it
without re-reading the cells.

Standalone use (regenerate the figures for a finished study without redoing its
whole analysis)::

    python curves.py --results "$CALIB_RESULTS" --arms prod,top_long \\
        --baseline "$OUT/text_baseline.csv" --out ./figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Below this fraction of an arm's cells carrying a detector, the mean is drawn
#: dashed: it describes a subset of the grid, not the grid.
SOLID_COVERAGE = float(os.environ.get("CURVE_SOLID_COVERAGE", "0.95"))

#: How far into the click axis a shortfall in coverage has to reach before the
#: coverage strip is drawn at all.  See :func:`_strip_worth_drawing`; set to 0
#: to get the strip on every figure unconditionally.
STRIP_SPAN = float(os.environ.get("CURVE_STRIP_SPAN", "0.25"))

#: Default identity of a cell.  Filtered to the columns actually present, so a
#: frame without an ``embedder`` column keys on the other three.
KEYS: tuple[str, ...] = ("dataset", "embedder", "category", "seed")

FIG_DPI = int(os.environ.get("CURVE_FIG_DPI", "130"))

#: ``"mean"`` matches the paired tables every report quotes; ``"median"`` is
#: available for a metric whose distribution is heavy-tailed enough that the
#: mean is not a level anyone experiences.
STAT = os.environ.get("CURVE_STAT", "mean")

#: What the zero-click anchor is called on the figure.  It is the seed sort the
#: run actually opened on, which for every text-seeded study is the typed query.
BASELINE_LABEL = os.environ.get("CURVE_BASELINE_LABEL", "text sort, 0 clicks")

#: Where to look for a metric's zero-click value in a ``text_baseline.py`` frame.
#: Tried in order; the metric's own name is the last resort so a caller can hand
#: over a frame that simply uses the same column name.
BASELINE_COLUMNS: dict[str, tuple[str, ...]] = {
    "cost": ("text_cost",),
    "oracle_cost": ("text_oracle_cost",),
    "average_precision": ("text_AP", "text_average_precision"),
    "auroc": ("text_auroc",),
    "precision": ("text_precision",),
    "recall": ("text_recall",),
    "f1": ("text_f1",),
    "fpr": ("text_fpr",),
    "fnr": ("text_fnr",),
}


def _keys(df: pd.DataFrame, keys: Sequence[str] = KEYS) -> list[str]:
    return [k for k in keys if k in df.columns]


def text_sort_baseline(path: str | Path, keys: Sequence[str] = KEYS) -> pd.DataFrame:
    """Load ``text_baseline.py``'s CSV as a baseline frame keyed by cell.

    Rows where the embedder has no text tower (``supports_text == 0``) carry no
    measurement and are dropped rather than read as zeros.
    """
    df = pd.read_csv(path)
    if "supports_text" in df.columns:
        df = df[df["supports_text"] == 1]
    wanted = set(keys)
    return df.loc[:, [c for c in df.columns if c in wanted or c.startswith("text_")]].copy()


def baseline_map(
    baseline: pd.DataFrame | None,
    metric: str,
    keys: list[str],
    baseline_col: str | None = None,
) -> dict[tuple, float]:
    """``cell key -> the metric's zero-click value``, or ``{}``.

    Keyed on the cell columns the *baseline* frame shares with the metric frame,
    so a baseline computed per ``(dataset, category, seed)`` lines up with a
    metric frame keyed the same way.  A baseline lacking the metric entirely
    yields ``{}`` — the figure then simply has no anchor, rather than an anchor
    invented from the wrong column.
    """
    if baseline is None or baseline.empty:
        return {}
    col = baseline_col
    if col is None:
        for cand in (*BASELINE_COLUMNS.get(metric, ()), metric):
            if cand in baseline.columns:
                col = cand
                break
    if col is None or col not in baseline.columns:
        return {}
    kk = [k for k in keys if k in baseline.columns]
    if not kk:
        return {}
    g = baseline.groupby(kk, dropna=False)[col].mean()
    return {(k if isinstance(k, tuple) else (k,)): float(v) for k, v in g.items() if np.isfinite(v)}


def _cell_columns(cells: Sequence[tuple], keys: list[str]) -> pd.Index:
    """Column index for a ``t`` x cell matrix over *cells*."""
    if len(keys) > 1:
        return pd.MultiIndex.from_tuples(list(cells), names=keys)
    return pd.Index([c[0] for c in cells], name=keys[0])


def _wide(
    g: pd.DataFrame,
    metric: str,
    keys: list[str],
    t_index: np.ndarray,
    base: dict[tuple, float] | None = None,
    cells: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    """``t`` x cell matrix of *metric* for one (arm, dataset) slice.

    Reindexed onto the full click axis so a cell that trained late contributes
    NaN — not a shifted curve — for the clicks before it had a model, and onto
    the full *cells* list so a cell that **never** trained is a NaN column
    rather than no column at all.  That distinction is the whole coverage
    strip: without it the denominator quietly shrinks to the cells that worked.

    When *base* is given, row ``t=0`` is that cell's zero-click (text-sort)
    value — including for a cell that never trained, which does have a text
    sort even though it never got a detector.
    """
    wide = g.pivot_table(index="t", columns=keys, values=metric, aggfunc="mean")
    wide = wide.reindex(t_index)
    if cells:
        wide = wide.reindex(columns=_cell_columns(cells, keys))
    if base and 0 in wide.index:
        wide.loc[0] = [base.get(col if isinstance(col, tuple) else (col,), np.nan) for col in wide.columns]
    return wide


def _bridge(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the un-measured clicks so a line draws straight through them.

    Used for the thin dashed line only, and that is the whole reason it is
    dashed: it is what carries a series from its click-0 text-sort point to its
    first trained click, across a stretch where nothing was measured.  The solid
    line never bridges — a solid segment has to be a level.
    """
    ok = np.isfinite(y)
    return t[ok], y[ok]


def _coverage(wide: pd.DataFrame, n_cells: int) -> np.ndarray:
    """Fraction of the arm's cells that are **measured** at each click.

    At ``t=0`` that is every cell, because every cell has a text sort even if it
    never went on to train a detector; from ``t=1`` it is the cells that have a
    detector, which is the only thing the metric frame records.
    """
    if n_cells <= 0:
        return np.zeros(len(wide.index))
    return wide.notna().sum(axis=1).to_numpy(dtype=float) / float(n_cells)


def _strip_worth_drawing(covs: Sequence[np.ndarray], t_index: np.ndarray) -> bool:
    """Does a coverage strip say anything the mean above it does not?

    The mean is already drawn dashed below :data:`SOLID_COVERAGE` and solid
    above it, so the *crossing* is on the figure twice: once as the strip's
    ramp and once as the line thickening.  On a healthy grid that is the whole
    content of the strip — every arm ramps to full inside the first handful of
    clicks and then draws a flat 100% line across the rest of the axis, which
    is a quarter of the figure spent restating one number per arm.

    What the dashed/solid switch cannot show is **how far** below full the
    denominator went and **whether it came back**, and that only matters where
    a reader would otherwise trust the average:

    * a shortfall reaching past :data:`STRIP_SPAN` of the click axis — an arm
      starving over a meaningful part of the run, or never recovering; and
    * coverage that goes *down* at any point after the first click — cells
      dropping out of the average partway is a different failure from starting
      late, and it is worth its depth being drawn however early it happens.

    Either shape keeps the strip; neither present, the panel gets the height
    and the click from which everything is measured is written in its title.
    ``coverage`` stays in the CSV regardless, so suppressing the strip drops a
    drawing, not a record.
    """
    if STRIP_SPAN <= 0:
        return True
    if len(t_index) == 0:
        return False
    late = float(t_index.max()) * STRIP_SPAN
    # Click 0 is fully covered by construction (every cell has a text sort) and
    # click 1 has no detectors yet, so that one step down is the axis, not a
    # denominator shrinking.  Monotonicity is asked of the clicked stretch.
    clicked = t_index >= 1
    for cov in covs:
        cov = np.asarray(cov, dtype=float)
        short = cov < SOLID_COVERAGE
        if short.any() and float(t_index[np.flatnonzero(short)].max()) > late:
            return True
        if bool((np.diff(cov[clicked]) < -1e-9).any()):
            return True
    return False


def _fully_measured_from(covs: Sequence[np.ndarray], t_index: np.ndarray) -> int | None:
    """The click from which **every** series here is measured on every cell.

    ``None`` when some series never gets there, or when there was never a
    shortfall to report — in the first case the strip is being drawn anyway,
    and in the second there is nothing for a reader to discount.
    """
    firsts: list[int] = []
    for cov in covs:
        short = np.asarray(cov, dtype=float) < SOLID_COVERAGE
        if not short.any():
            continue
        after = int(np.flatnonzero(short).max()) + 1
        if after >= len(t_index):
            return None
        firsts.append(int(t_index[after]))
    return max(firsts) if firsts else None


def _cell_index(
    main: pd.DataFrame,
    keys: list[str],
    denominator: pd.DataFrame | None,
) -> dict[tuple[str, str], list[tuple]]:
    """``(arm, dataset) -> every cell attempted``, as key tuples.

    From *denominator* when given — one row per attempted cell, including the
    ones that never trained and therefore appear nowhere in *main*.  Falling
    back to *main* makes coverage identically 1.0 at the arm's own maximum,
    which is exactly the reassurance this figure exists to withhold.
    """
    src = denominator if denominator is not None and not denominator.empty else main
    if "dataset" not in src.columns or "arm" not in src.columns:
        return {}
    out: dict[tuple[str, str], list[tuple]] = {}
    for (arm, ds), d in src.groupby(["arm", "dataset"]):
        rows = d.loc[:, keys].drop_duplicates()
        out[(str(arm), str(ds))] = [tuple(r) for r in rows.itertuples(index=False, name=None)]
    return out


def _prevalence_norm(prevalence: dict[tuple[str, str], float] | None):
    """A log norm over the supplied prevalences, or ``None`` to colour flat."""
    if not prevalence:
        return None
    vals = [v for v in prevalence.values() if v and v > 0]
    if len(vals) < 2:
        return None
    import matplotlib.colors as mcolors

    return mcolors.LogNorm(vmin=min(vals), vmax=max(vals))


def _stop_clicks(stops: pd.DataFrame | None, panels: Iterable[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Median first-fire click per ``(dataset, arm)``, for the panels being drawn.

    Only where **at least half** the arm's runs on that dataset ever fired.
    Below that the median of the firers is a survivorship artefact — the runs it
    leaves out are exactly the slow ones — and a marker drawn from it would put
    an authoritative glyph on a number the sample cannot support.  Those arms
    are named in the caption instead (see :func:`_stop_caption`), which is
    itself the finding.
    """
    if stops is None or stops.empty or "t_stop" not in stops.columns:
        return {}
    want = set(panels)
    out: dict[tuple[str, str], float] = {}
    for (arm, ds), g in stops.groupby(["arm", "dataset"], dropna=False):
        if (str(ds), str(arm)) not in want:
            continue
        fired = g[g["stopped"]] if "stopped" in g.columns else g[g["t_stop"].notna()]
        if len(g) == 0 or len(fired) / len(g) < 0.5:
            continue
        med = float(pd.to_numeric(fired["t_stop"], errors="coerce").median())
        if np.isfinite(med):
            out[(str(ds), str(arm))] = med
    return out


def _stop_caption(stops: pd.DataFrame, stop_at: dict[tuple[str, str], float], panels: Iterable[tuple[str, str]]) -> str:
    """The one line the stopping marker needs to be readable.

    Says what the glyph is, and — the part that is easy to omit and expensive to
    omit — names the arms that carry no glyph *because their runs mostly never
    stopped*, so an absent marker cannot be read as an oversight.
    """
    drawn = "▽ = median click at which the app's stopping rules first fired"
    missing = sorted({arm for (ds, arm) in set(panels) if (ds, arm) not in stop_at})
    if not missing:
        return drawn
    return drawn + f"; no marker for {', '.join(missing)} — fewer than half those runs ever fired"


def mean_figure(  # noqa: C901
    main: pd.DataFrame,
    outdir: Path,
    *,
    arms: Sequence[str],
    metric: str = "cost",
    keys: Sequence[str] = KEYS,
    denominator: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    baseline_col: str | None = None,
    baseline_label: str = BASELINE_LABEL,
    stat: str = STAT,
    lower_is_better: bool = True,
    stops: pd.DataFrame | None = None,
    dpi: int = FIG_DPI,
) -> tuple[str | None, pd.DataFrame]:
    """Per-dataset mean curves with an inter-quartile band.

    Carries a coverage strip under each panel when coverage says something the
    dashed/solid switch does not — see :func:`_strip_worth_drawing`; otherwise
    the panel takes the height and its title names the click from which every
    arm is fully measured.

    *stops* is :func:`stopping.stopping_points`' output (issue #3560).  When
    given, each arm's curve carries a marker at the click where the app's
    stopping rules fired for **half** the runs that ever fired — the point on
    this very axis past which the user was being told they could stop.  It is
    drawn as a marker on the curve rather than a rule across the panel for the
    same reason the click-0 anchor is: a rule implies a level that holds
    everywhere, and this one holds at one click.  Arms where fewer than half
    the runs ever fired get no marker and are named in the caption instead,
    because there is no click to put one at.

    Returns ``(filename, curves)``; ``curves`` is the plotted numbers, long
    format, one row per ``(arm, dataset, t)``, and always carries ``coverage``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kk = _keys(main, keys)
    if main.empty or metric not in main.columns or "dataset" not in main.columns:
        return None, pd.DataFrame()
    datasets = sorted(str(d) for d in main["dataset"].dropna().unique())
    arms_present = [a for a in arms if (main["arm"] == a).any()]
    if not datasets or not arms_present:
        return None, pd.DataFrame()

    base = baseline_map(baseline, metric, kk, baseline_col)
    t_index = np.arange(0 if base else 1, int(main["t"].max()) + 1)
    index = _cell_index(main, kk, denominator)
    rows: list[dict] = []

    # Everything the figure plots, computed before a single axis exists: the
    # coverage strip only gets a row of the gridspec when it has something to
    # say (see _strip_worth_drawing), and that cannot be known until every
    # arm's coverage has been measured.
    series: dict[tuple[str, str], dict] = {}
    for ds in datasets:
        for arm in arms_present:
            g = main[(main["arm"] == arm) & (main["dataset"] == ds)]
            if g.empty:
                continue
            wide = _wide(g, metric, kk, t_index, base, index.get((arm, ds)))
            centre = wide.mean(axis=1) if stat == "mean" else wide.median(axis=1)
            n_cells = int(wide.shape[1])
            series[(ds, arm)] = {
                "y": centre.to_numpy(dtype=float),
                "cov": _coverage(wide, n_cells),
                "lo": wide.quantile(0.25, axis=1).to_numpy(dtype=float),
                "hi": wide.quantile(0.75, axis=1).to_numpy(dtype=float),
                "n_cells": n_cells,
                "base_level": float(centre.loc[0]) if (base and 0 in wide.index) else float("nan"),
            }
    if not series:
        return None, pd.DataFrame()
    for (ds, arm), sr in series.items():
        for t, yy, c, l_, h_ in zip(t_index, sr["y"], sr["cov"], sr["lo"], sr["hi"], strict=False):
            rows.append(
                {
                    "arm": arm,
                    "dataset": ds,
                    "t": int(t),
                    stat: yy,
                    "q25": float(l_),
                    "q75": float(h_),
                    "coverage": float(c),
                    "n_cells": int(sr["n_cells"]),
                    "baseline": sr["base_level"],
                }
            )

    stop_at = _stop_clicks(stops, series.keys())
    strip = _strip_worth_drawing([sr["cov"] for sr in series.values()], t_index)
    fig = plt.figure(figsize=(6.6 * len(datasets), 5.4 if strip else 4.4), layout="constrained")
    gs = (
        fig.add_gridspec(2, len(datasets), height_ratios=[3.1, 1.0], hspace=0.04)
        if strip
        else fig.add_gridspec(1, len(datasets))
    )
    palette = {a: f"C{i}" for i, a in enumerate(arms_present)}
    top_axes = []
    for di, ds in enumerate(datasets):
        ax = fig.add_subplot(gs[0, di])
        axc = fig.add_subplot(gs[1, di], sharex=ax) if strip else None
        top_axes.append(ax)
        for arm in arms_present:
            sr = series.get((ds, arm))
            if sr is None:
                continue
            y, cov = sr["y"], sr["cov"]
            # The whole curve, thin and dashed: this is the level over whatever
            # subset of cells had a detector at that click.  Bridged across the
            # clicks with no measurement, which is what carries the line from
            # its click-0 text-sort point to its first trained click - dashed
            # exactly because nothing was measured in there.
            ax.plot(*_bridge(t_index, y), color=palette[arm], lw=1.0, ls=(0, (2, 2)), alpha=0.9)
            # The same curve, solid, only where it describes (nearly) the whole
            # grid.  A solid line is the one a reader may quote a level off.
            solid = np.where(cov >= SOLID_COVERAGE, y, np.nan)
            ax.plot(t_index, solid, color=palette[arm], lw=1.8, label=arm)
            band = np.where(cov >= SOLID_COVERAGE, 1.0, np.nan)
            ax.fill_between(
                t_index,
                sr["lo"] * band,
                sr["hi"] * band,
                color=palette[arm],
                alpha=0.09,
                lw=0,
            )
            # Click 0 is drawn as a point, not joined: every cell is measured
            # there (it has a text sort) and none has a detector at click 1, so
            # a joined line would render that as a spike rather than as two
            # different facts.
            clicked = t_index > 0
            # Click 0 IS the text sort, drawn as this arm's own first point.
            # It used to be a rule across the whole panel, which dominated the
            # figure to make a point the leftmost marker already makes - and
            # implied a level that holds at every click when it holds at one.
            if base and np.isfinite(y[0]):
                ax.plot(0, y[0], marker="o", ms=4.5, color=palette[arm], zorder=5)
            # Where the app told the user they could stop (#3560).  On the
            # curve, at the median first-fire click, so it reads as a point on
            # this arm rather than as a level across the panel.
            ts = stop_at.get((ds, arm))
            if ts is not None:
                yi = np.interp(ts, t_index, y, left=np.nan, right=np.nan)
                if np.isfinite(yi):
                    ax.plot(
                        ts,
                        yi,
                        marker="v",
                        ms=7,
                        mfc="none",
                        mew=1.6,
                        color=palette[arm],
                        zorder=6,
                        ls="none",
                    )
            if axc is not None:
                axc.plot(t_index[clicked], 100.0 * cov[clicked], color=palette[arm], lw=1.2)
                if not clicked.all():
                    axc.plot(0, 100.0 * cov[0], marker="o", ms=2.5, color=palette[arm])
        title = ds
        if axc is None:
            # The one fact the suppressed strip was carrying: every arm in this
            # panel is measured on every cell from here on, so the dashed
            # stretch to its left is the only part a reader has to discount.
            from_t = _fully_measured_from([sr["cov"] for (d, _a), sr in series.items() if d == ds], t_index)
            if from_t:
                title = f"{ds}\nall cells measured from click {from_t}"
        ax.set_title(title, fontsize=10)
        if axc is not None:
            ax.tick_params(labelbottom=False)
            axc.set_ylim(0, 105)
            axc.axhline(100.0 * SOLID_COVERAGE, color="#888", lw=0.8, ls=":")
        xlabel = f"clicks spent  (0 = {baseline_label})" if base else "clicks spent"
        (axc or ax).set_xlabel(xlabel)
        if di == 0:
            ax.set_ylabel(f"{metric} ({stat} over cells)", fontsize=9)
            if axc is not None:
                axc.set_ylabel("% of cells\nmeasured", fontsize=8)
    top_axes[0].legend(fontsize=7, ncol=2)
    better = "lower is better" if lower_is_better else "higher is better"
    caption = (
        f"Is the user's detector getting better as they click?  —  {metric} vs clicks ({better})\n"
        f"dashed where fewer than {SOLID_COVERAGE:.0%} of that arm's cells are measured at that click"
    )
    if stops is not None and not stops.empty:
        caption += "\n" + _stop_caption(stops, stop_at, series.keys())
    fig.suptitle(caption, fontsize=10)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{metric}_vs_clicks.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    curves = pd.DataFrame(rows)
    if not curves.empty:
        curves.to_csv(outdir / f"{metric}_vs_clicks.csv", index=False)
    return p.name, curves


def per_run_figures(  # noqa: C901
    main: pd.DataFrame,
    outdir: Path,
    *,
    arms: Sequence[str],
    metric: str = "cost",
    keys: Sequence[str] = KEYS,
    denominator: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    baseline_col: str | None = None,
    prevalence: dict[tuple[str, str], float] | None = None,
    max_runs: int = 0,
    lower_is_better: bool = True,
    dpi: int = FIG_DPI,
) -> list[str]:
    """One file per dataset: a panel per arm holding **every** seed's own curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kk = _keys(main, keys)
    if main.empty or metric not in main.columns or "dataset" not in main.columns:
        return []
    arms_present = [a for a in arms if (main["arm"] == a).any()]
    if not arms_present:
        return []

    norm = _prevalence_norm(prevalence)
    cmap = plt.get_cmap("viridis_r")
    index = _cell_index(main, kk, denominator)
    base = baseline_map(baseline, metric, kk, baseline_col)
    t_max = int(main["t"].max())
    written: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    for ds in sorted(str(d) for d in main["dataset"].dropna().unique()):
        sub = main[main["dataset"] == ds]
        fig, axes = plt.subplots(
            1,
            len(arms_present),
            figsize=(2.45 * len(arms_present), 3.9),
            sharey=True,
            sharex=True,
            layout="constrained",
        )
        axes = np.atleast_1d(axes)
        for ax, arm in zip(axes, arms_present, strict=False):
            g = sub[sub["arm"] == arm]
            runs = list(g.groupby(kk, dropna=False)) if not g.empty else []
            dropped = 0
            if max_runs and len(runs) > max_runs:
                step = len(runs) / float(max_runs)
                keep = {int(i * step) for i in range(max_runs)}
                dropped = len(runs) - len(keep)
                runs = [r for i, r in enumerate(runs) if i in keep]
            trained_cells = set()
            for key, run in runs:
                run = run.sort_values("t")
                cell = key if isinstance(key, tuple) else (key,)
                trained_cells.add(cell)
                cat = dict(zip(kk, cell, strict=False)).get("category")
                prev = (prevalence or {}).get((ds, cat)) if cat is not None else None
                colour = cmap(norm(prev)) if (norm is not None and prev) else "#2b6cb0"
                xs = run["t"].to_numpy(dtype=float)
                ys = run[metric].to_numpy(dtype=float)
                # Anchor the run at its OWN zero-click text-sort quality, so the
                # left end of every line is what that cell's typed query got for
                # free and the line itself is what the clicking bought.
                b = base.get(cell)
                if b is not None and np.isfinite(b):
                    xs, ys = np.concatenate(([0.0], xs)), np.concatenate(([b], ys))
                    ax.plot(0.0, b, marker="o", ms=1.8, color=colour, alpha=0.45)
                ax.plot(xs, ys, lw=0.5, alpha=0.35, color=colour)
            # ...and a run that NEVER trained still has a text sort, so its
            # anchor dot is drawn on its own.  A column of lone dots at x=0 is
            # exactly what total Good-starvation looks like.
            for cell in index.get((arm, ds)) or []:
                if cell in trained_cells:
                    continue
                b = base.get(cell)
                if b is None or not np.isfinite(b):
                    continue
                cat = dict(zip(kk, cell, strict=False)).get("category")
                prev = (prevalence or {}).get((ds, cat)) if cat is not None else None
                colour = cmap(norm(prev)) if (norm is not None and prev) else "#2b6cb0"
                ax.plot(0.0, b, marker="x", ms=3.0, mew=0.7, color=colour, alpha=0.55)
            # A starved run draws nothing beyond its t=0 anchor - it never
            # trained, so it emitted no row - and an absent line is
            # indistinguishable from an arm that simply has fewer seeds.  Name
            # the count in the title, and dash the median wherever it is a
            # median over a subset.
            n_trained = int(g.groupby(kk, dropna=False).ngroups) if not g.empty else 0
            cells = index.get((arm, ds))
            n_cells = len(cells) if cells else n_trained
            if not g.empty:
                t_index = np.arange(0 if base else 1, t_max + 1)
                wide = _wide(g, metric, kk, t_index, base, cells)
                cov = _coverage(wide, int(wide.shape[1]))
                med = wide.median(axis=1).to_numpy(dtype=float)
                ax.plot(*_bridge(t_index, med), color="#111", lw=0.9, ls=(0, (2, 2)))
                ax.plot(t_index, np.where(cov >= SOLID_COVERAGE, med, np.nan), color="#111", lw=1.6, label="median")
            bits = [arm]
            if dropped:
                # Deterministic thinning, SAID SO: a hairball with a third of
                # its lines removed reads as a tighter arm, not a cropped one.
                bits.append(f"{len(runs)} of {len(runs) + dropped} runs drawn")
            never = n_cells - n_trained
            if never > 0:
                bits.append(f"{never}/{n_cells} never trained")
            ax.set_title(bits[0] if len(bits) == 1 else bits[0] + "\n" + "; ".join(bits[1:]), fontsize=8)
            ax.set_xlabel("clicks  (0 = text sort)" if base else "clicks", fontsize=8)
        better = "lower is better" if lower_is_better else "higher is better"
        axes[0].set_ylabel(f"{metric} — {better}")
        axes[0].legend(fontsize=7)
        if norm is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.015, pad=0.01)
            cb.set_label("category prevalence", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            cb.ax.yaxis.set_major_formatter(lambda v, _p: f"{v * 100:g}%")
            cb.ax.yaxis.set_minor_formatter(lambda v, _p: f"{v * 100:g}%")
        fig.suptitle(
            f"{ds}: every run's own {metric} curve (a mean hides the runs that never start)",
            fontsize=10,
        )
        p = outdir / f"{metric}_vs_clicks_runs__{ds}.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        written.append(p.name)
    return written


def quality_vs_clicks(
    main: pd.DataFrame,
    outdir: Path,
    *,
    arms: Sequence[str],
    metric: str = "cost",
    keys: Sequence[str] = KEYS,
    denominator: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    baseline_col: str | None = None,
    baseline_label: str = BASELINE_LABEL,
    prevalence: dict[tuple[str, str], float] | None = None,
    stat: str = STAT,
    max_runs: int = 0,
    lower_is_better: bool = True,
    stops: pd.DataFrame | None = None,
    dpi: int = FIG_DPI,
) -> list[str]:
    """Both standard figures (averaged, and per run) plus the curve CSV.

    The one call an analyzer needs: pass the main metric frame tagged with
    ``arm``, the cell list as *denominator*, and the text-sort *baseline*.
    Pass *stops* (from ``stopping.stopping_points``) to mark where the app's
    stopping rules fired on the averaged panel — the click past which every
    further click on the axis was one the app had already said was optional.
    """
    written: list[str] = []
    name, _curves = mean_figure(
        main,
        outdir,
        arms=arms,
        metric=metric,
        keys=keys,
        denominator=denominator,
        baseline=baseline,
        baseline_col=baseline_col,
        baseline_label=baseline_label,
        stat=stat,
        lower_is_better=lower_is_better,
        stops=stops,
        dpi=dpi,
    )
    if name:
        written.append(name)
    written.extend(
        per_run_figures(
            main,
            outdir,
            arms=arms,
            metric=metric,
            keys=keys,
            denominator=denominator,
            baseline=baseline,
            baseline_col=baseline_col,
            prevalence=prevalence,
            max_runs=max_runs,
            lower_is_better=lower_is_better,
            dpi=dpi,
        )
    )
    return written


def crossover(curves: pd.DataFrame, stat: str = STAT, lower_is_better: bool = True) -> pd.DataFrame:
    """First click at which each ``(arm, dataset)`` curve beats its ``t=0`` anchor.

    The number the zero-click anchor exists to make computable: *"how many
    clicks before the thing I am training is worth more than the query I
    typed?"*  ``NaN`` where the arm never crosses, which is itself a result and
    the reason this returns a column rather than a printed sentence.
    """
    if curves.empty or "baseline" not in curves.columns or stat not in curves.columns:
        return pd.DataFrame()
    out = []
    for (arm, ds), g in curves.groupby(["arm", "dataset"]):
        g = g.sort_values("t")
        b = float(g["baseline"].iloc[0])
        if not np.isfinite(b):
            continue
        better = (g[stat] < b) if lower_is_better else (g[stat] > b)
        hit = g.loc[better & (g["t"] > 0), "t"]
        out.append(
            {
                "arm": arm,
                "dataset": ds,
                "baseline": b,
                "final": float(g[stat].iloc[-1]),
                "crossover_t": int(hit.iloc[0]) if len(hit) else float("nan"),
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Standalone: regenerate a finished study's curves without re-running analysis
# ---------------------------------------------------------------------------


def _load(results: Path, arms: Sequence[str]) -> pd.DataFrame:
    import _cells_io

    parts = []
    for arm in arms:
        df, _prov = _cells_io.load_arm(results / arm)
        if not df.empty:
            df = df.copy()
            df["arm"] = arm
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def prevalence_from(path: str | Path) -> dict[tuple[str, str], float]:
    """``(dataset, category) -> prevalence`` from a run's ``prepare_info.json``."""
    prevalence: dict[tuple[str, str], float] = {}
    info = json.loads(Path(path).read_text())
    for ds, embs in info.get("datasets", {}).items():
        for _emb, d in embs.items():
            n = int(d.get("n_medias") or 0)
            cc = d.get("category_counts") or {}
            for cat in d.get("selected_categories") or []:
                if n:
                    prevalence[(ds, cat)] = float(cc.get(cat, 0)) / n
    return prevalence


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(common.RESULTS), help="results root holding one dir per arm")
    ap.add_argument("--arms", required=True, help="comma-separated arm directories, in report order")
    ap.add_argument("--out", default=None, help="figure directory (default: <results>/../analysis/figures)")
    ap.add_argument("--metric", default="cost", help="metric column to curve (default: cost)")
    ap.add_argument("--stat", default=STAT, choices=("mean", "median"))
    ap.add_argument("--max-runs", type=int, default=0, help="cap per-run lines per panel (0 = draw all)")
    ap.add_argument("--prevalence", default=None, help="prepare_info.json, to colour runs by category prevalence")
    ap.add_argument("--baseline", default=None, help="text_baseline.py CSV: the zero-click anchor at t=0")
    args = ap.parse_args(list(argv) if argv is not None else None)

    results = Path(args.results)
    arms = [a for a in args.arms.replace(",", " ").split() if a]
    outdir = Path(args.out) if args.out else results.parent / "analysis" / "figures"
    frame = _load(results, arms)
    if frame.empty:
        print(f"no rows under {results} for arms {arms}")
        return 2

    prevalence = prevalence_from(args.prevalence) if args.prevalence else None
    baseline = text_sort_baseline(args.baseline) if args.baseline else None
    lower_is_better = args.metric not in ("average_precision", "auroc")

    written = quality_vs_clicks(
        frame,
        outdir,
        arms=arms,
        metric=args.metric,
        prevalence=prevalence,
        baseline=baseline,
        stat=args.stat,
        max_runs=args.max_runs,
        lower_is_better=lower_is_better,
    )
    print(f"wrote {len(written)} figure(s) to {outdir}:")
    for name in written:
        print(f"  {name}")
    curve_csv = outdir / f"{args.metric}_vs_clicks.csv"
    if curve_csv.exists():
        x = crossover(pd.read_csv(curve_csv), stat=args.stat, lower_is_better=lower_is_better)
        if not x.empty:
            print("\nclicks to beat the zero-click text sort (NaN = never):")
            print(x.to_string(index=False))
    if not args.baseline:
        print("\nNOTE: no --baseline, so the curves start at the first trainable click and there is")
        print("      nothing to compare the far right against. Run text_baseline.py and pass its CSV.")
    print(
        "\nNOTE: run standalone, coverage is measured against the cells that PRODUCED rows, "
        "so a starved cell is invisible.\n      analyze_startup.py passes its own cell list as the "
        "denominator and is the reading that counts starvation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

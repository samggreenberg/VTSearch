"""#3796: how wide is the distribution the pinned Train/Calibrate split draws from?

Production splits each calibration fold's labelset off a fresh
``RandomState(CALIBRATION_SPLIT_SEED)`` and #2934 pinned that on purpose, so
every user gets the same arbitrary draw and every eval number in this repo is
one sample from a distribution nobody has measured.  This reads the #3796 grid,
which holds the cell seed (same medias voted, same order, same held-out test
split) and redraws only the split, and answers the three questions the issue
asks, plus the one it implies:

1. **the headline** - the sd and range of cost across calibration draws, per
   (class, cell seed) block, at fixed vote count;
2. **against what studies already average over** - the same spread across *cell*
   seeds at the pinned draw, as a variance decomposition rather than two
   unrelated sds, because the two are nested and the ratio is the question;
3. **does it shrink with votes** - the same spread as a curve over the horizon,
   since the mechanics say a bigger labelset makes any one split less pivotal;
4. **where does 42 sit** - the pin's own percentile among its competitors, which
   is the only thing that could make production's draw *biased* rather than
   merely noisy, and the only part of this a per-cell bootstrap could never see.

Every number is computed on cell-band means, never on raw steps: steps inside
one trajectory share a model and a labelset prefix, so an sd over steps is an sd
over a hundred copies of the same draw.

    python analyze_calseed_3796.py --results <run>/results --out <base>/analysis \\
        [--baseline <base>/analysis/text_baseline.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _cells_io  # noqa: E402
import curves  # noqa: E402
from _cells_io import assert_one_opening  # noqa: E402

#: A *block* is everything this study holds fixed while it redraws the split:
#: one class, in one geometry, at one cell seed.  The draws inside a block are
#: the measurement; anything computed across blocks is a summary of many
#: measurements and is labelled as such.
BLOCK_KEYS: tuple[str, ...] = ("dataset", "geometry", "category", "seed")

#: Vote bands, matching #3287 so the two studies' tables read the same way.  The
#: spread is expected to fall across them, which is deliverable 3.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("early 1-25", 1, 25),
    ("mid 26-60", 26, 60),
    ("late 61-100", 61, 100),
    ("deep 101-150", 101, 150),
)

#: Clicks the spread-vs-votes curve is quoted at in the report's table.
CHECKPOINTS: tuple[int, ...] = (20, 50, 100, 150)

#: What gets a spread.  `cost` is the headline - it is what a user pays.  The
#: other five separate the two ways a redrawn split can move it:
#:
#: * `threshold` is the DIRECT effect: a different split is a different conformal
#:   quantile, read off the same scores.
#: * `auroc` and `oracle_cost` are properties of the RANKING, so they move only
#:   if the trajectories diverged - a different cut is a different acquisition
#:   rank, which is a different next vote.  An sd of zero here with an sd above
#:   zero on `cost` says the whole spread is the cut; both above zero says the
#:   loop compounded, which the issue names as part of the measurand.
#: * `regret` is cost minus the oracle's, i.e. the part of the spread that is
#:   the cut rule's rather than the ranking's, per step.
#: * `f1` is there because the #3794 probe quoted its answer in F1 and a
#:   comparison needs the same unit.
#: * `n_good` is not a quality metric at all - it is how many positives the
#:   trajectory had found by that band.  It rides along because it is the one
#:   thing that makes the answer PORTABLE: a threshold is a quantile of the
#:   calibration set, so the spread should fall as the set that sets it grows,
#:   and a reader in a different environment needs that relation rather than
#:   this grid's constant.
METRICS: tuple[str, ...] = ("cost", "f1", "auroc", "threshold", "oracle_cost", "regret", "n_good")

#: The headline metric.  Named once so the report and the summary agree.
HEADLINE = "cost"

#: The contrast a reader is most likely to be holding this spread against: the
#: cost difference #3287 reported as its result, on the shipped configuration.
#: Not a threshold this study applies to anything - it is a yardstick, quoted so
#: "is the spread big?" has an answer in units someone here already has.
REFERENCE_CONTRAST = 0.013


def geometry_of(row) -> str:
    """``dinov3_patch/max_patch``-style label for one row's (embedder, style).

    The pair name collapses to its LEARN half: that is the space the detector
    trains, scores and sorts in, and the text half only ranked the opening,
    which this grid holds fixed.  Copied deliberately from ``analyze_calfrac``
    rather than imported - #3287's module owns a study, and a shared helper that
    two studies' tables depend on belongs in the shared layer or nowhere.
    """
    emb = str(row["embedder"])
    learn = emb.partition("+")[2] or emb
    return f"{learn}/{row['style']}"


def voting_mode(geometry: str) -> str:
    """Region voting is a property of the STYLE, not of the dataset (#2877/#2905)."""
    return "region" if geometry.endswith("/max_patch") else "binary"


def band_of(t: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=t.index, dtype="object")
    for name, lo, hi in BANDS:
        out = out.mask((t >= lo) & (t <= hi), name)
    return out


def load(results: Path) -> tuple[pd.DataFrame, dict]:
    """The run's base rows, with geometry/mode, checked for the things that void it."""
    frame, prov = _cells_io.load_arm(results)
    if frame.empty:
        raise SystemExit(f"no rows under {results}/cells")
    if "calibration_seed" not in frame.columns:
        raise SystemExit(
            f"{results} has no `calibration_seed` column: it was produced by a run_cells.py "
            "that predates #3794, so which draw a row came from cannot be established."
        )
    frame = frame.copy()
    frame["calibration_seed"] = pd.to_numeric(frame["calibration_seed"], errors="coerce").astype("Int64")
    if frame["calibration_seed"].isna().any():
        raise SystemExit("some rows carry no calibration_seed; refusing to pool a draw that cannot be named")
    frame["geometry"] = frame.apply(geometry_of, axis=1)
    frame["mode"] = frame["geometry"].map(voting_mode)
    frame["band"] = band_of(frame["t"])
    # One opening for the whole study, or every pooled number below describes two
    # different experiments (#3278).
    assert_one_opening(frame, where="calibration-seed study")
    if frame["calibration_seed"].nunique() < 2:
        raise SystemExit(
            "this run holds one calibration draw: it is a default run, not a #3796 sweep. "
            "Set CALIB_CALIBRATION_SEEDS and re-run."
        )
    return frame, prov


def coverage(prov: dict) -> str:
    """`describe_load`'s sentence, plus the count `load_arm` renames out of it.

    `load_arm` moves `header_only` to `no_positive_found` - a starved cell is a
    legitimate result rather than data loss, and the rename says so - but
    `describe_load` still looks for the old key, so calling it on `load_arm`'s
    provenance reports *zero* starved cells however many there were.  Naming the
    count here is the smaller change: six studies across the tree read that
    shared sentence, and it is the sentence that makes "N of M cells" mean the
    same thing in two reports.  Filed as a follow-up rather than fixed in place.
    """
    line = _cells_io.describe_load(prov)
    starved = len(prov.get("no_positive_found") or ())
    if starved:
        line += f", {starved} header-only (no positive found)"
    return line


def resolve_pin(results: Path, frame: pd.DataFrame) -> int:
    """The draw production actually ships, taken from the APP, not from the grid.

    The alternative - "whichever draw the launcher listed first" - would keep
    reporting this study's own reference point on the day the app's pin moved,
    which is the failure `preflight.sh` check 12 exists to prevent one level up.
    A grid that does not contain the shipped pin is still analysable; it just
    cannot say where production sits, and says so rather than picking a stand-in.
    """
    from vtscore.training.thresholds import CALIBRATION_SPLIT_SEED

    pin = int(CALIBRATION_SPLIT_SEED)
    if pin not in set(frame["calibration_seed"].dropna().astype(int)):
        print(
            f"WARNING: the app's pinned split seed ({pin}) is not among this grid's draws "
            f"{sorted(frame['calibration_seed'].dropna().astype(int).unique())}; "
            "the spread is still measured, but nothing here locates production inside it."
        )
    return pin


def block_band_means(frame: pd.DataFrame) -> pd.DataFrame:
    """One value per (block, draw, band, metric): that trajectory's band mean.

    Collapsing steps BEFORE any spread is what makes the spread a spread over
    *draws*.  150 steps of one trajectory are 150 readings of one draw, sharing
    a model and a labelset prefix, so an sd taken over them measures the
    trajectory's own wobble and would be reported as if it were the split's.
    """
    present = [m for m in METRICS if m in frame.columns]
    d = frame[frame["band"].notna()]
    grouped = d.groupby([*BLOCK_KEYS, "mode", "calibration_seed", "band"], dropna=False)
    out = grouped[present].mean().reset_index()
    out["n_steps"] = grouped.size().to_numpy()
    return out


def _spread(values: pd.Series) -> dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if v.size < 2:
        return {"n": float(v.size), "mean": float(v.mean()) if v.size else np.nan, "sd": np.nan, "range": np.nan}
    return {
        "n": float(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "range": float(v.max() - v.min()),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
    }


def draw_spread(bb: pd.DataFrame, pin: int) -> pd.DataFrame:
    """Per (block, band, metric): the spread across calibration draws - the headline.

    Also carries the pin's own value and its percentile among the draws, which
    is deliverable 4.  The percentile is computed with the pin INCLUDED, so a
    grid of 20 draws puts a typical pin near 0.5 and an extreme one near 0 or 1;
    excluding it would make the null depend on the grid size.
    """
    rows: list[dict] = []
    for keys, g in bb.groupby([*BLOCK_KEYS, "mode", "band"], dropna=False):
        base = dict(zip([*BLOCK_KEYS, "mode", "band"], keys))
        for metric in (m for m in METRICS if m in bb.columns):
            s = _spread(g[metric])
            pin_rows = g[g["calibration_seed"] == pin]
            pin_val = float(pin_rows[metric].iloc[0]) if len(pin_rows) == 1 else np.nan
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if np.isfinite(pin_val) and vals.size > 1:
                # Mid-rank, so ties (a degenerate metric) land at 0.5 rather than
                # at whichever end the comparison operator happens to pick.
                below = float((vals < pin_val).sum())
                equal = float((vals == pin_val).sum())
                pin_pct = (below + 0.5 * equal) / vals.size
            else:
                pin_pct = np.nan
            rows.append({**base, "metric": metric, **s, "pin_value": pin_val, "pin_percentile": pin_pct})
    return pd.DataFrame(rows)


def variance_components(bb: pd.DataFrame) -> pd.DataFrame:
    """Split the cell-level variance into its cell-seed and calibration-draw parts.

    The design is nested and balanced - draws *inside* cell seeds, inside a
    (geometry, class) - so the classical estimator applies and needs no fitting:

        MS_within  = mean over seeds of var_draws(y)            -> sigma^2_draw
        MS_between = D * var_seeds(mean_draw y)                 -> sigma^2_draw + D * sigma^2_seed

    Two unrelated sds would not answer the issue's question.  "Is the split noise
    a rounding error next to the data noise, or a fraction of it?" is a question
    about *components of one variance*, and reading it off two sds computed on
    different denominators is how a ratio comes out wrong by the factor D.

    ``sigma^2_seed`` is clipped at zero.  A negative estimate is legal in this
    estimator (it means the between-seed mean square came in under the within),
    and it is information - it says the cell seed moved the metric less than the
    draw did - but a negative variance has no square root, so the clip is
    recorded in ``seed_var_was_negative`` rather than hidden.
    """
    # Named, so an empty result is still a frame with these columns.  A study
    # whose grid holds one cell seed legitimately produces no decomposition, and
    # a bare `DataFrame([])` then has no `geometry` to filter on - which turns a
    # missing deliverable into a KeyError three functions later.
    columns = [
        "geometry",
        "mode",
        "category",
        "band",
        "metric",
        "n_seeds",
        "n_draws",
        "sd_draw",
        "sd_seed",
        "seed_var_was_negative",
        "share_draw",
    ]
    rows: list[dict] = []
    for (geom, mode, cat, band), g in bb.groupby(["geometry", "mode", "category", "band"], dropna=False):
        for metric in (m for m in METRICS if m in bb.columns):
            wide = g.pivot_table(index="seed", columns="calibration_seed", values=metric, aggfunc="mean")
            wide = wide.dropna(axis=0, how="any").dropna(axis=1, how="any")
            n_seeds, n_draws = wide.shape
            if n_seeds < 2 or n_draws < 2:
                continue
            ms_within = float(wide.var(axis=1, ddof=1).mean())
            ms_between = float(n_draws * wide.mean(axis=1).var(ddof=1))
            var_seed = (ms_between - ms_within) / n_draws
            rows.append(
                {
                    "geometry": geom,
                    "mode": mode,
                    "category": cat,
                    "band": band,
                    "metric": metric,
                    "n_seeds": n_seeds,
                    "n_draws": n_draws,
                    "sd_draw": float(np.sqrt(max(ms_within, 0.0))),
                    "sd_seed": float(np.sqrt(max(var_seed, 0.0))),
                    "seed_var_was_negative": bool(var_seed < 0),
                    "share_draw": (
                        float(max(ms_within, 0.0) / (max(ms_within, 0.0) + max(var_seed, 0.0)))
                        if (max(ms_within, 0.0) + max(var_seed, 0.0)) > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows, columns=pd.Index(columns))


def spread_by_step(frame: pd.DataFrame, metric: str = HEADLINE) -> pd.DataFrame:
    """Per (geometry, t): the across-draw sd, summarised over blocks.

    Deliverable 3.  Taken per step rather than per band because "where is the
    shallow end" is a question about the shape of the curve, and a four-point
    band table cannot show a knee.

    Only steps where **every** block still has all its draws contribute, so the
    curve cannot fall merely because cheap blocks dropped out of it.
    """
    # `geometry` is already one of BLOCK_KEYS; naming it twice gives the frame a
    # duplicate column and `groupby` a non-1-dimensional grouper.
    d = frame[["mode", *BLOCK_KEYS, "calibration_seed", "t", metric]].dropna(subset=[metric])
    per_block = d.groupby(["mode", *BLOCK_KEYS, "t"])[metric].agg(sd=lambda s: s.std(ddof=1), n_draws="count")
    per_block = per_block.reset_index()
    full = per_block["n_draws"] == per_block["n_draws"].max()
    per_block = per_block[full]
    out = (
        per_block.groupby(["geometry", "mode", "t"])["sd"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n_blocks="count")
        .reset_index()
    )
    return out


def divergence_onset(results: Path, pin: int) -> pd.DataFrame:
    """The click at which two draws of one block first vote on different media.

    The issue's "the loop is closed, so a threshold that moves moves the next
    pick too" is the one claim here that a metric column cannot settle: cost can
    differ because the cut moved on an identical ranking.  The pick log can, and
    since #3796 it carries the draw.

    Reported as the first ``t`` at which the block's draws do not all agree on
    ``picked_id``.  A block whose draws never disagree returns NaN and is
    counted separately - that is a real outcome (the split moved the threshold
    but never enough to reorder the acquisition rank) and averaging it in as the
    horizon would understate exactly the thing being measured.
    """
    files = _cells_io.side_frame_files(results / "cells", "__picks")
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if df.empty or "calibration_seed" not in df.columns:
            continue
        parts.append(df[["dataset", "category", "seed", "style", "embedder", "calibration_seed", "t", "picked_id"]])
    if not parts:
        return pd.DataFrame()
    picks = pd.concat(parts, ignore_index=True)
    picks["geometry"] = picks.apply(geometry_of, axis=1)
    rows: list[dict] = []
    for keys, g in picks.groupby([*BLOCK_KEYS], dropna=False):
        base = dict(zip(BLOCK_KEYS, keys))
        per_t = g.groupby("t")["picked_id"].agg(["nunique", "count"])
        # Only clicks every draw reached: a block whose draws stop at different
        # steps would otherwise read as "diverged" at the first ragged click.
        per_t = per_t[per_t["count"] == per_t["count"].max()]
        disagree = per_t[per_t["nunique"] > 1]
        n_draws = int(g["calibration_seed"].nunique())
        rows.append(
            {
                **base,
                "n_draws": n_draws,
                "first_divergent_click": float(disagree.index.min()) if len(disagree) else np.nan,
                "last_click_compared": float(per_t.index.max()) if len(per_t) else np.nan,
                "never_diverged": not len(disagree),
            }
        )
    return pd.DataFrame(rows)


def cost_decomposition(bb: pd.DataFrame) -> pd.DataFrame:
    """Was it the CUT the split was drawn for, or the run the cut then steered?

    ``cost = oracle_cost + regret`` holds row by row - the oracle is the best cut
    available on *this* trajectory's ranking, so the first term is a property of
    what the run FOUND and the second of what the rule then did with it.  Across
    draws inside a block that gives a variance identity:

        Var(cost) = Var(oracle_cost) + Var(regret) + 2 Cov(oracle_cost, regret)

    which is the only form worth reporting.  **The terms are sum-pinned and they
    slide against each other** - the covariance here is negative everywhere - so
    reading one alone manufactures an effect, the trap #2897 fell into and #3287
    quantified.  Shares are therefore POOLED (summed variances over blocks, then
    divided) rather than averaged per block: pooled shares sum to exactly 1 and
    a median of shares does not, and a share above 1 is then legible as what it
    is, the other side of a negative covariance.
    """
    rows: list[dict] = []
    for (geom, mode, band, cat, seed), k in bb.groupby(["geometry", "mode", "band", "category", "seed"], dropna=False):
        if len(k) < 2 or not {"oracle_cost", "regret"} <= set(k.columns):
            continue
        rows.append(
            {
                "geometry": geom,
                "mode": mode,
                "band": band,
                "var_cost": float(k["cost"].var(ddof=1)),
                "var_ranking": float(k["oracle_cost"].var(ddof=1)),
                "var_cut": float(k["regret"].var(ddof=1)),
                "cov2": float(2.0 * k["oracle_cost"].cov(k["regret"])),
            }
        )
    per_block = pd.DataFrame(
        rows, columns=pd.Index(["geometry", "mode", "band", "var_cost", "var_ranking", "var_cut", "cov2"])
    )
    if per_block.empty:
        return per_block
    agg = per_block.groupby(["geometry", "mode", "band"], dropna=False).agg(
        blocks=("var_cost", "size"),
        var_cost=("var_cost", "sum"),
        var_ranking=("var_ranking", "sum"),
        var_cut=("var_cut", "sum"),
        cov2=("cov2", "sum"),
    )
    out = pd.DataFrame(
        {
            "blocks": agg["blocks"],
            "rms_sd_cost": np.sqrt(agg["var_cost"] / agg["blocks"]),
            "share_ranking": agg["var_ranking"] / agg["var_cost"],
            "share_cut": agg["var_cut"] / agg["var_cost"],
            "share_cov": agg["cov2"] / agg["var_cost"],
        }
    ).reset_index()
    return out


def by_class(spread: pd.DataFrame) -> pd.DataFrame:
    """The spread per class, because it is not one number.

    A single median over five classes would be quoted as "the noise floor"; the
    classes differ by a factor of several, and a reader deciding whether a
    contrast of theirs survives needs the spread of the class they are working
    on, not the grid's average.
    """
    c = spread[spread["metric"] == HEADLINE]
    return c.pivot_table(index="category", columns="geometry", values="sd", aggfunc="median").round(4).reset_index()


def study_variance(bb: pd.DataFrame) -> pd.DataFrame:
    """The report's headline table: the two components, pooled, and the identity.

    :func:`variance_components` answers the question per (geometry, class, band),
    which is where a reader checks it.  This answers it for the STUDY, and it is
    a different computation rather than a summary of that one: sds do not average
    and medians of sds are not the median of anything, so the components are
    pooled as **variances** and square-rooted once at the end.

    It also carries the pre-registered reading as two columns rather than as a
    sentence.  ``sd_seen`` is what a single-draw study actually observes - the sd
    across cell seeds at one draw, averaged over which draw it happened to be -
    and ``sd_predicted`` is ``sqrt(var_draw + var_seed)``.  If the split noise is
    already inside a study's own cell-to-cell variation, those two agree; if it
    were somehow an extra term on top, they would not.  Checking it beats
    asserting it, and it costs one column.
    """
    rows: list[dict] = []
    for (geom, mode, cat, band), g in bb.groupby(["geometry", "mode", "category", "band"], dropna=False):
        wide = g.pivot_table(index="seed", columns="calibration_seed", values=HEADLINE, aggfunc="mean")
        wide = wide.dropna(axis=0, how="any").dropna(axis=1, how="any")
        n_seeds, n_draws = wide.shape
        if n_seeds < 2 or n_draws < 2:
            continue
        ms_within = float(wide.var(axis=1, ddof=1).mean())
        ms_between = float(n_draws * wide.mean(axis=1).var(ddof=1))
        rows.append(
            {
                "geometry": geom,
                "mode": mode,
                "var_draw": ms_within,
                "var_seed": (ms_between - ms_within) / n_draws,
                # The mean over draws of the variance across seeds: an unbiased
                # estimate of what a study pinned to ANY one draw is resampling.
                "var_seen": float(wide.var(axis=0, ddof=1).mean()),
            }
        )
    per_cell = pd.DataFrame(rows, columns=pd.Index(["geometry", "mode", "var_draw", "var_seed", "var_seen"]))
    if per_cell.empty:
        return per_cell

    def _summarise(d: pd.DataFrame, label: str, mode: str) -> dict:
        vd, vs, vseen = float(d["var_draw"].mean()), float(d["var_seed"].mean()), float(d["var_seen"].mean())
        total = vd + max(vs, 0.0)
        return {
            "geometry": label,
            "mode": mode,
            "cells": int(len(d)),
            "sd_draw": float(np.sqrt(max(vd, 0.0))),
            "sd_seed": float(np.sqrt(max(vs, 0.0))),
            "share_draw": float(vd / total) if total > 0 else np.nan,
            "sd_seen": float(np.sqrt(max(vseen, 0.0))),
            "sd_predicted": float(np.sqrt(max(total, 0.0))),
        }

    out = [_summarise(d, str(geom), str(d["mode"].iloc[0])) for geom, d in per_cell.groupby("geometry")]
    out.append(_summarise(per_cell, "POOLED", "both"))
    return pd.DataFrame(out)


def pair_exceedance(bb: pd.DataFrame, margin: float) -> pd.DataFrame:
    """How often two draws of ONE block differ by more than *margin*.

    An sd is a summary; this is the question a reader actually has, which is
    whether a contrast the size of a published finding is inside the noise of a
    single cell.  Every unordered pair of the block's draws is compared, and the
    fraction exceeding *margin* is summarised over blocks - so the answer is
    about a cell, not about a grid mean, and it does not depend on the metric
    being normally distributed, which #3329 gives no reason to assume.
    """
    rows: list[dict] = []
    for (geom, mode, band, cat, seed), k in bb.groupby(["geometry", "mode", "band", "category", "seed"], dropna=False):
        v = pd.to_numeric(k[HEADLINE], errors="coerce").dropna().to_numpy(dtype=float)
        if v.size < 2:
            continue
        iu = np.triu_indices(v.size, 1)
        diff = np.abs(v[:, None] - v[None, :])[iu]
        rows.append({"geometry": geom, "mode": mode, "band": band, "frac_over": float((diff > margin).mean())})
    per_block = pd.DataFrame(rows, columns=pd.Index(["geometry", "mode", "band", "frac_over"]))
    if per_block.empty:
        return per_block
    out = (
        per_block.groupby(["geometry", "mode"], dropna=False)["frac_over"]
        .agg(blocks="size", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    out["margin"] = margin
    return out


def worked_blocks(bb: pd.DataFrame, pin: int, band: str) -> pd.DataFrame:
    """Every block's best draw, worst draw and the shipped pin, in one band.

    The report owes literal examples, and for a study whose subject is a spread
    the literal example IS the pair of draws at the ends of one: two runs on data
    that could not move, named by draw number so a reader can go back to the
    cells.  Emitted for every block rather than for the few the prose quotes, so
    the quoted ones can be checked against the ones that were not.
    """
    rows: list[dict] = []
    for (geom, mode, cat, seed), k in bb[bb["band"] == band].groupby(
        ["geometry", "mode", "category", "seed"], dropna=False
    ):
        k = k.dropna(subset=[HEADLINE]).sort_values(HEADLINE)
        if len(k) < 2:
            continue
        pin_rows = k[k["calibration_seed"] == pin]
        rows.append(
            {
                "geometry": geom,
                "mode": mode,
                "category": cat,
                "seed": seed,
                "band": band,
                "n_draws": int(len(k)),
                "best_draw": int(k.iloc[0]["calibration_seed"]),
                "best": float(k.iloc[0][HEADLINE]),
                "worst_draw": int(k.iloc[-1]["calibration_seed"]),
                "worst": float(k.iloc[-1][HEADLINE]),
                "range": float(k.iloc[-1][HEADLINE] - k.iloc[0][HEADLINE]),
                "pin": float(pin_rows[HEADLINE].iloc[0]) if len(pin_rows) == 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["geometry", "range"], ascending=[True, False])


def spread_vs_positives(spread: pd.DataFrame, bb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does the spread fall as the calibration set grows?  The portability handle.

    A conformal threshold is a quantile of the held-out half of the votes, and
    the anchors that fix it are the POSITIVES among them.  So the mechanism
    predicts the spread falls with the positive count, and if it does, a reader
    in a different environment can carry that relation across rather than
    carrying this grid's constant - which is the thing a number measured at one
    prevalence cannot do.

    Returns ``(per-geometry rank correlation, a table by tercile of positives)``.
    Rank correlation rather than Pearson: the relation is expected to go like
    one over a square root, and nothing here needs it to be a line.
    """
    c = spread[spread["metric"] == HEADLINE][["geometry", "mode", "category", "seed", "band", "sd"]]
    pos = bb.groupby(["geometry", "category", "seed", "band"], dropna=False)["n_good"].mean().reset_index()
    j = c.merge(pos, on=["geometry", "category", "seed", "band"], how="inner").dropna(subset=["sd", "n_good"])
    if j.empty:
        return pd.DataFrame(), pd.DataFrame()
    corr = (
        j.groupby(["geometry", "mode"], dropna=False)
        .apply(lambda d: pd.Series({"n": len(d), "spearman": d["sd"].corr(d["n_good"], method="spearman")}))
        .reset_index()
    )
    j = j.copy()
    j["positives"] = j.groupby("geometry")["n_good"].transform(
        lambda v: pd.qcut(v, 3, labels=["fewest", "middle", "most"], duplicates="drop")
    )
    table = (
        j.groupby(["geometry", "positives"], dropna=False, observed=True)
        .agg(cells=("sd", "size"), median_positives=("n_good", "median"), median_sd=("sd", "median"))
        .reset_index()
    )
    return corr, table


def implications(spread: pd.DataFrame, vc: pd.DataFrame, n_cells_per_arm: int = 60) -> pd.DataFrame:
    """What the measured spread does to a study that never varies the split.

    The arithmetic that matters is NOT "the spread is the error bar".  A study
    bootstraps over cells at the pinned draw, and each of those cells carries
    its own realisation of the split noise, so the split is already inside the
    SE it reports.  What the sweep adds is two things a bootstrap cannot get at:

    * **the floor on a PAIRED contrast.**  Two arms paired on (class, seed) are
      not paired on the split: a different knob is a different trajectory and
      therefore a different realisation.  So the paired difference carries the
      draw variance TWICE, and pairing - which is what buys those studies their
      precision - cannot remove it.  ``sqrt(2) * sd_draw / sqrt(N)`` is the
      smallest SE a study of N cells per arm can have from this source alone.
    * **the share**, which is the design lever: if most of a cell's variance is
      the draw rather than the data, a study buys precision with more cells and
      not with more cell seeds, because a second seed re-rolls both and a second
      cell re-rolls both too, but the *composition* says which one is nearly
      free to add.

    *n_cells_per_arm* is a reference grid size, not a claim about any particular
    study; the column scales as 1/sqrt(N) and the report names the N it used.
    """
    rows: list[dict] = []
    for (geom, band), g in spread[spread["metric"] == HEADLINE].groupby(["geometry", "band"], dropna=False):
        sd = float(g["sd"].median())
        share = vc.loc[(vc["geometry"] == geom) & (vc["band"] == band) & (vc["metric"] == HEADLINE), "share_draw"]
        rows.append(
            {
                "geometry": geom,
                "band": band,
                "n_blocks": int(len(g)),
                "median_sd_draw": sd,
                "median_range_draw": float(g["range"].median()),
                "paired_floor_se": float(np.sqrt(2.0) * sd / np.sqrt(n_cells_per_arm)),
                "n_cells_per_arm": n_cells_per_arm,
                "median_share_draw": float(share.median()) if len(share) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def figures(
    frame: pd.DataFrame, spread: pd.DataFrame, by_step: pd.DataFrame, pin: int, outdir: Path, baseline_csv: Path | None
) -> list[str]:
    """The study's own pictures, plus the mandatory quality-over-clicks pair.

    `curves` is the one implementation of the latter and is not re-written here;
    it is handed a two-arm view (the pin against every other draw) because
    twenty labelled series in one panel is a legend, not a figure.  The spread
    itself needs pictures `curves` has no notion of, and those are drawn below.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # --- 1. does the spread shrink with votes? ------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for geom, g in by_step.groupby("geometry"):
        g = g.sort_values("t")
        (line,) = ax.plot(g["t"], g["median"], label=f"{geom} (n={int(g['n_blocks'].max())} blocks)")
        ax.fill_between(g["t"], g["q25"], g["q75"], alpha=0.15, color=line.get_color())
    ax.set_xlabel("votes")
    ax.set_ylabel(f"sd of {HEADLINE} across calibration draws")
    ax.set_title("The calibration-split noise floor, over a session")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "spread_vs_votes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(str(p))

    # --- 2. the fan: every draw of a few blocks -----------------------------
    blocks = (
        spread[spread["metric"] == HEADLINE]
        .sort_values("sd", ascending=False)
        .drop_duplicates(subset=["geometry", "category", "seed"])
    )
    picks = []
    for geom in sorted(frame["geometry"].unique()):
        sub = blocks[blocks["geometry"] == geom]
        if len(sub):
            picks.append(sub.iloc[0])
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(5.0 * len(picks), 4.0), squeeze=False)
        for ax, row in zip(axes[0], picks):
            sel = frame[
                (frame["geometry"] == row["geometry"])
                & (frame["category"] == row["category"])
                & (frame["seed"] == row["seed"])
            ]
            for draw, g in sel.groupby("calibration_seed"):
                g = g.sort_values("t")
                if int(draw) == pin:
                    ax.plot(g["t"], g[HEADLINE], color="crimson", lw=2.0, zorder=3, label=f"draw {pin} (shipped)")
                else:
                    ax.plot(g["t"], g[HEADLINE], color="0.55", lw=0.8, alpha=0.7, zorder=1)
            ax.set_title(f"{row['geometry']} · {row['category']} · seed {int(row['seed'])}", fontsize=9)
            ax.set_xlabel("votes")
            ax.set_ylabel(HEADLINE)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("One block, every calibration draw (the widest block of each geometry)", fontsize=10)
        fig.tight_layout()
        p = outdir / "draw_fan.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    # --- 3. where does the shipped draw sit? --------------------------------
    pct = spread[(spread["metric"] == HEADLINE) & spread["pin_percentile"].notna()]["pin_percentile"]
    if len(pct):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.hist(pct, bins=10, range=(0, 1), color="0.45", edgecolor="white")
        ax.axhline(len(pct) / 10.0, color="crimson", ls="--", lw=1.2, label="uniform (no bias)")
        ax.set_xlabel(f"percentile of draw {pin}'s {HEADLINE} among its block's draws")
        ax.set_ylabel("blocks")
        ax.set_title(f"Is the shipped draw lucky?  n={len(pct)} (block, band) cells")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = outdir / "pin_percentile.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    # --- 4. the mandatory quality-over-clicks pair --------------------------
    d = frame.copy()
    d["arm"] = np.where(d["calibration_seed"] == pin, f"draw {pin} (shipped)", "redrawn (19 draws)")
    d["dataset"] = d["dataset"].astype(str) + " · " + d["geometry"].astype(str).str.replace("/", "-", regex=False)
    baseline = curves.text_sort_baseline(baseline_csv) if baseline_csv and Path(baseline_csv).exists() else None
    if baseline is not None:
        reps = []
        for geom in sorted(frame["geometry"].unique()):
            b = baseline.copy()
            b["dataset"] = b["dataset"].astype(str) + " · " + geom.replace("/", "-")
            reps.append(b)
        baseline = pd.concat(reps, ignore_index=True)
    denominator = d[["dataset", "embedder", "category", "seed"]].drop_duplicates()
    for metric in ("cost", "average_precision"):
        if metric not in d.columns:
            continue
        try:
            written += curves.quality_vs_clicks(
                d,
                outdir,
                arms=[f"draw {pin} (shipped)", "redrawn (19 draws)"],
                metric=metric,
                denominator=denominator,
                baseline=baseline,
                lower_is_better=(metric == "cost"),
            )
        except Exception as exc:  # noqa: BLE001 - a figure must not cost the analysis
            print(f"WARNING: curves.quality_vs_clicks({metric}) failed: {exc!r}")
    return written


# --------------------------------------------------------------------------- report


def _fmt(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x):.{digits}f}"


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_(no rows)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: _fmt(v, digits))
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.to_numpy())
    return f"{head}\n{rule}\n{body}\n"


def write_report(
    out: Path,
    *,
    pin,
    prov,
    spread,
    vc,
    by_step,
    onset,
    impl,
    figs,
    shape,
    decomp,
    study,
    pairs,
    worked,
    cls,
    corr,
    pos_table,
) -> None:
    head = spread[spread["metric"] == HEADLINE]
    by_geo_band = (
        head.groupby(["geometry", "band"], dropna=False)
        .agg(
            blocks=("sd", "size"),
            median_sd=("sd", "median"),
            p90_sd=("sd", lambda s: s.quantile(0.90)),
            median_range=("range", "median"),
            max_range=("range", "max"),
            median_pin_pct=("pin_percentile", "median"),
        )
        .reset_index()
    )
    by_metric = (
        spread.groupby(["geometry", "metric"], dropna=False)["sd"].median().reset_index(name="median_sd_over_bands")
    )
    vc_pool = (
        vc[vc["metric"] == HEADLINE]
        .groupby(["geometry", "band"], dropna=False)
        .agg(
            cells=("sd_draw", "size"),
            median_sd_draw=("sd_draw", "median"),
            median_sd_seed=("sd_seed", "median"),
            median_share_draw=("share_draw", "median"),
            n_negative_seed_var=("seed_var_was_negative", "sum"),
        )
        .reset_index()
    )
    ck = by_step[by_step["t"].isin(CHECKPOINTS)].pivot_table(
        index="geometry", columns="t", values="median", aggfunc="first"
    )
    lines: list[str] = []
    lines.append("# #3796 — the calibration-split noise floor\n")
    lines.append(
        "Generated by `analyze_calseed_3796.py`. Every number below is computed on cell-band means "
        "(one value per trajectory per band), never on raw steps.\n"
    )
    lines.append("## What ran\n")
    lines.append("```\n" + json.dumps(shape, indent=2) + "\n```\n")
    lines.append(f"Cells read: {coverage(prov)}\n")
    lines.append(f"Production's pinned split seed: **{pin}**\n")
    lines.append("## 1. The spread across calibration draws (headline: `cost`)\n")
    lines.append(_md(by_geo_band))
    lines.append(
        f"\nFraction of draw PAIRS inside one block differing by more than {REFERENCE_CONTRAST} - "
        "the cost contrast #3287 reported as its headline. This is the question about a single cell, "
        "which is what a user is.\n"
    )
    lines.append(_md(pairs))
    lines.append(f"\n### 1a. The widest and narrowest blocks ({BANDS[-1][0]} votes)\n")
    lines.append(
        "Two runs on data that could not move, named by draw so a reader can go back to the cells. "
        "Full table: `worked_blocks.csv`.\n"
    )
    if not worked.empty:
        ends = pd.concat([worked.groupby("geometry").head(2), worked.groupby("geometry").tail(1)])
        lines.append(_md(ends.sort_values(["geometry", "range"], ascending=[True, False])))
    lines.append("\n## 2. Against the cell-seed spread — a variance decomposition\n")
    lines.append(
        "`sd_draw` is the within-(class, cell seed) sd across calibration draws; `sd_seed` is the "
        "between-cell-seed sd with the draw variance removed. `share_draw` is "
        "`var_draw / (var_draw + var_seed)`: the fraction of a cell's variance the split owns.\n"
    )
    lines.append(_md(study, digits=4))
    lines.append(
        "\n`sd_seen` is what a single-draw study observes - the sd across cell seeds at one draw, averaged "
        "over which draw it happened to be - and `sd_predicted` is `sqrt(var_draw + var_seed)`. They agree "
        "because the split noise is **already inside** a study's own cell-to-cell variation, which is the "
        "pre-registered reading, checked rather than asserted: restating published numbers as plus-or-minus "
        "this spread would count the same variance twice.\n"
    )
    lines.append("\nPer (geometry, class, band):\n")
    lines.append(_md(vc_pool))
    lines.append("\n## 3. Does it shrink with votes?\n")
    lines.append(f"Median across blocks of the across-draw sd of `{HEADLINE}`, at four click counts.\n")
    lines.append(_md(ck.reset_index()))
    lines.append("\n## 4. Where does the shipped draw sit?\n")
    lines.append(
        f"Percentile of draw {pin} among its own block's draws, pooled over (block, band). "
        "A median near 0.5 means the pin is an ordinary sample; a systematic offset would be the one "
        "error no per-cell bootstrap could ever see.\n"
    )
    # Per BLOCK, not per (block, band): the four bands of one block are four
    # readings of one trajectory pair, so an SE over 300 of them would be an SE
    # over 75 things counted four times.  The SE is the OBSERVED one rather than
    # the uniform null's, because the bands within a block are not independent
    # and the data knows how much they are not.
    pct_block = head.dropna(subset=["pin_percentile"]).groupby(list(BLOCK_KEYS))["pin_percentile"].mean()
    if len(pct_block):
        se = float(pct_block.std(ddof=1) / np.sqrt(len(pct_block)))
        z = (float(pct_block.mean()) - 0.5) / se if se > 0 else float("nan")
        lines.append(
            f"- n = {len(pct_block)} blocks (the four bands of a block averaged first); mean percentile "
            f"**{_fmt(pct_block.mean())}** against the 0.50 an unbiased pin gives, SE {_fmt(se)} "
            f"(z = {_fmt(z, 1)}). Two SE bounds any bias at {_fmt(2 * se)} in percentile terms.\n"
        )
    lines.append("\n### 1b. The spread is a property of the CLASS, not a constant\n")
    lines.append(_md(cls))
    lines.append("\n### 1c. Does it fall as the calibration set grows?\n")
    lines.append(
        "A conformal threshold is a quantile of the held-out votes and the positives are its anchors, so "
        "the mechanism predicts the spread falls with the positive count. This is the relation a reader in "
        "another environment can carry across; the constant above is not.\n"
    )
    lines.append(_md(corr))
    lines.append(_md(pos_table))
    lines.append("\n## 5. Did the trajectories diverge, or only the cut?\n")
    lines.append(
        "`auroc` and `oracle_cost` are properties of the ranking, so an sd above zero on them is the "
        "closed loop compounding — a redrawn split moved the cut, the cut moved the acquisition rank, "
        "and the run went on to vote on different media.\n"
    )
    lines.append(_md(by_metric))
    lines.append(
        "\n`cost = oracle_cost + regret` row by row - the oracle is the best cut available on **this** "
        "trajectory's ranking - so across draws the variance telescopes. Shares are pooled (variances "
        "summed over blocks, then divided), so they sum to exactly 1; the terms are sum-pinned and slide "
        "against each other, so a share above 1 is the other side of the negative covariance beside it and "
        "**no term may be read alone** (#2897's trap, #3287's measurement of it).\n"
    )
    lines.append(_md(decomp))
    if not onset.empty:
        never = int(onset["never_diverged"].sum())
        first = onset["first_divergent_click"].dropna()
        lines.append(
            f"\nPick-log divergence: **{len(onset) - never} of {len(onset)} blocks** had two draws pick "
            f"different media, first at a median click of **{_fmt(first.median(), 1)}** "
            f"(min {_fmt(first.min(), 1)}, max {_fmt(first.max(), 1)}); {never} never diverged.\n"
        )
    lines.append("\n## 6. What it means for every other study here\n")
    lines.append(
        "The spread is **not** an extra error bar to add to published numbers: a study that bootstraps "
        "over cells at the pinned draw already resamples cells that each carry their own realisation of "
        "it. What it is, is a floor that pairing cannot remove — two arms paired on (class, seed) are "
        "not paired on the split, because a different knob is a different trajectory — and a statement "
        "about where precision comes from.\n"
    )
    lines.append(_md(impl))
    if figs:
        lines.append("\n## Figures\n")
        for f in figs:
            lines.append(f"- `{Path(f).name}`\n")
    out.write_text("".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the run's results dir (holds cells/)")
    ap.add_argument("--out", required=True, help="where the aggregates, figures and report go")
    ap.add_argument("--baseline", default=None, help="text_baseline.csv for the click-0 anchor")
    ap.add_argument("--n-cells-per-arm", type=int, default=60, help="reference grid size for the paired floor")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame, prov = load(results)
    print(f"cells: {coverage(prov)}")
    pin = resolve_pin(results, frame)

    shape_path = results / "grid_shape.json"
    try:
        shape = json.loads(shape_path.read_text())
    except (OSError, ValueError):
        shape = {"note": f"no readable {shape_path}"}
    shape["draws_seen"] = sorted(int(x) for x in frame["calibration_seed"].dropna().unique())
    shape["blocks_seen"] = int(frame[list(BLOCK_KEYS)].drop_duplicates().shape[0])

    bb = block_band_means(frame)
    spread = draw_spread(bb, pin)
    vc = variance_components(bb)
    by_step = spread_by_step(frame)
    onset = divergence_onset(results, pin)
    decomp = cost_decomposition(bb)
    study = study_variance(bb)
    pairs = pair_exceedance(bb, REFERENCE_CONTRAST)
    worked = worked_blocks(bb, pin, BANDS[-1][0])
    cls = by_class(spread)
    corr, pos_table = spread_vs_positives(spread, bb)
    impl = implications(spread, vc, n_cells_per_arm=args.n_cells_per_arm)

    bb.to_csv(out / "block_band_means.csv", index=False)
    spread.to_csv(out / "draw_spread.csv", index=False)
    vc.to_csv(out / "variance_components.csv", index=False)
    by_step.to_csv(out / "spread_by_step.csv", index=False)
    if not onset.empty:
        onset.to_csv(out / "divergence_onset.csv", index=False)
    impl.to_csv(out / "implications.csv", index=False)
    decomp.to_csv(out / "cost_decomposition.csv", index=False)
    study.to_csv(out / "study_variance.csv", index=False)
    pairs.to_csv(out / "pair_exceedance.csv", index=False)
    worked.to_csv(out / "worked_blocks.csv", index=False)
    cls.to_csv(out / "spread_by_class.csv", index=False)
    if not pos_table.empty:
        corr.to_csv(out / "spread_vs_positives_corr.csv", index=False)
        pos_table.to_csv(out / "spread_vs_positives.csv", index=False)

    figs: list[str] = []
    if not args.no_figures:
        try:
            figs = figures(frame, spread, by_step, pin, out / "figures", Path(args.baseline) if args.baseline else None)
        except Exception as exc:  # noqa: BLE001 - the tables are the result; a figure is not
            print(f"WARNING: figures failed: {exc!r}")

    write_report(
        out / "REPORT_calseed.md",
        pin=pin,
        prov=prov,
        spread=spread,
        vc=vc,
        by_step=by_step,
        onset=onset,
        impl=impl,
        figs=figs,
        shape=shape,
        decomp=decomp,
        study=study,
        pairs=pairs,
        worked=worked,
        cls=cls,
        corr=corr,
        pos_table=pos_table,
    )
    head = spread[spread["metric"] == HEADLINE]
    summary = {
        "pin": pin,
        "shape": shape,
        "cells": {k: (len(v) if isinstance(v, list) else v) for k, v in prov.items()},
        "median_sd_cost": float(head["sd"].median()),
        "median_range_cost": float(head["range"].median()),
        "by_geometry": {
            str(g): {
                "median_sd_cost": float(d["sd"].median()),
                "median_range_cost": float(d["range"].median()),
                "median_pin_percentile": float(d["pin_percentile"].median()),
            }
            for g, d in head.groupby("geometry")
        },
        "figures": [Path(f).name for f in figs],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'REPORT_calseed.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the **interactive quality-over-clicks viewer** a study's report links to.

The two committed PNGs (:mod:`curves`) answer the questions a report *asks*.
They cannot answer the questions a reader has after reading it — *"is that true
on `visual_genome_m` too?"*, *"is it the scarce categories doing all the work?"*,
*"does it hold on recall, or only on cost?"* — because each of those is a
different slice and a PNG is one slice, chosen in advance by whoever wrote the
analyzer.  This builds a single self-contained HTML page that carries **every**
slice, so a reader can ask their own question instead of asking for a re-run.

What the page lets a reader pick:

* **dataset** — one, or all of them averaged together;
* **category** — one within that dataset, or all averaged;
* **embedder** — any non-empty subset, drawn as **one panel each**.  Embedders
  are never averaged with one another: two embedders are two different
  representations of the haystack, and their mean is a number describing no
  system anyone could run.  Faceting makes that structural rather than a rule
  someone has to remember;
* **arms** — any non-empty subset.  Non-empty for the same reason the embedder
  subset is: an empty selection has no honest rendering, since the page either
  goes blank or falls back to "all" and a reader who misses that takes a chart
  of everything for a chart of nothing.  The page locks the last remaining chip
  rather than snapping it silently back on;
* **seeds** — averaged, or every seed as its own line;
* **metric** — cost, precision, recall, F1, FPR, FNR, average precision, AUROC:
  whatever the run emitted, from one shared definition
  (:data:`vtscore.eval.calibration_metrics.DETECTION_METRICS`);
* **overlay** — off (the default), every varying dimension is its own chart
  carrying one bold line and the ±1 SD shadow of the population under it; on,
  they all land on one chart in distinct hues and the shadows come off.  The
  shadow only has an honest home in the first: two translucent bands over one
  another are a third shape nobody can read the overlap of.  Overlaying two
  embedders is **not** averaging them — the ban that keeps them out of one
  number is a ban on pooling, and two lines pool nothing;
* **oracle** — off (the default), or the same model's score at the cut the test
  labels say it should have used, drawn dotted beside the solid performance
  line.  The gap between them is the calibration regret.

Four reference quantities, and each is drawn as what it is
------------------------------------------------------------

The chart carries two **lines** and two **points**, and the whole redesign is
about not letting a point be read as a line:

``performance`` (a line)
    What the loop actually achieved: the momentary model at its own computed
    threshold, at every click.  Solid where it describes the whole grid, dashed
    where it describes a subset.
``oracle`` (a line, behind a checkbox)
    The same momentary model with a **cheating threshold** — the cut a reader
    would have picked knowing the test labels.  Same hue, dotted.  It is not a
    rival system; it is the ceiling this system's *threshold rule* left on the
    table, so it shares the colour and differs only in style.
``text sort`` (a point, notched in the **left** margin)
    What typing the query got for free, at zero clicks.
``skyline`` (a point, notched in the **right** margin)
    The same head, through the same trainer, with every training label handed
    to it (issue #3322) — the learnability floor of this embedding space.
    Vote-independent, so it does not move as the reader clicks.

**Nothing joins the text-sort notch to the curve.**  The dotted bridge that used
to run from click 0 to the first trained click was the most prominent mark on
the chart and it stood for a measurement nobody made: between the two there is
no detector, so there is no level to draw.  A gap says exactly that.

The two notches sit in the margins rather than as rules across the panel for the
same reason: each is a level that holds at *one* x, and a horizontal rule across
the chart asserts it holds at every x.  The skyline is the sharper case — drawn
as a rule it would read as "the floor was reachable at click 3", which is the
one reading the number exists to prevent.

Payload
-------

Two arrays, both embedded, because they answer different questions and neither
is derivable from the other at an acceptable size:

``agg``
    Per ``(group, arm, metric, click)``: ``mean``, ``sd`` and ``n``, at **full**
    click resolution, for every metric.  A group is one
    ``(dataset, embedder, category)``.  Storing ``n`` beside the moments is what
    makes "all categories" and "all datasets" *exact* rather than a mean of
    means — the page pools them properly, weighted by the cells that actually
    contributed.
``agg.omean`` / ``agg.on``
    The oracle companion, on the same axes.  The harness emits the oracle
    **cut** and the two rates it pays there, never the confusion-matrix metrics
    at that cut — so precision / recall / F1 at the oracle threshold are
    *reconstructed* here from ``(oracle_fpr, oracle_fnr, n_test_pos,
    n_test_neg)``, which is a full confusion matrix.  That is why the oracle is
    offered on every metric that is a statement about one cut, rather than on
    cost alone: "the cut cost us 0.1" and "the cut cost us 11 points of recall"
    are the same fact in the two units a reader thinks in.  It carries its own
    ``n``, because an oracle that declines to flag anything has no precision at
    a click where the trained cut's precision is perfectly well defined.
    ``average_precision`` and ``auroc`` get none: they integrate over every
    threshold, so re-cutting cannot move them and the "oracle" would be the
    performance line drawn twice.
``skyline``
    Per ``(group, arm, metric)``: one number, not a series.  Read straight out
    of the cell CSVs (:func:`load_skyline`) rather than out of the main frame,
    because ``analyze_spikes.load_arm`` filters skyline rows out by design — a
    skyline reads ground-truth labels the app can never see, so it must never
    land in a mean of what the app achieved.
``runs``
    Per ``(group, arm, seed, metric, click)``: the raw series behind the
    per-seed lines, on a **thinned** click grid chosen to fit
    ``runs_budget_mb``.  A 42-seed grid across 24 environments and 8 arms is
    ~13 million numbers at full resolution and no amount of packing makes that a
    committable file; thinning the click axis is the one lossy choice that costs
    a reader nothing they would have seen (per-seed lines are read for spread,
    not for fine structure).  **The chosen grid is written into the payload and
    shown in the page**, and if even the coarsest grid busts the budget the
    per-seed mode is disabled with the reason on screen — never silently
    dropped, and never quietly subsampled down to a tidier-looking set of runs.

Both are quantised to int16, NaN-masked, delta-coded along the click axis and
gzipped; the page inflates them with ``DecompressionStream``.

Usage
-----

::

    python viewer.py --results "$CALIB_RESULTS" --arms prod,top_long \\
        --baseline "$OUT/text_baseline.csv" --out "$OUT/viewer.html"

A study with a single results directory names it and relabels it, so the page
says what the arm IS rather than where it sits::

    python viewer.py --results "$CALIB_EXP" --arms results=prod \\
        --baseline "$OUT/text_baseline.csv" --out "$OUT/viewer.html"
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import curves  # noqa: E402

#: NaN in the int16 encoding.  Values are masked separately, so this is only the
#: filler that keeps the delta stream smooth across a gap.
FILL = 0

#: Quantisation: values are stored as ``round(value * SCALE)``.  1/4000 is a
#: quarter of a thousandth on a [0,1] metric — an order of magnitude finer than
#: anything a reader can resolve on screen, and coarse enough that the delta
#: stream still compresses.
SCALE = int(os.environ.get("VIEWER_SCALE", "4000"))

#: Quantisation for the **per-seed** payload, coarser than :data:`SCALE` on
#: purpose.  The aggregate means are the numbers a reader quotes and keep the
#: fine grid; per-seed lines are read for spread and for which runs never leave
#: the floor, and 1/1000 is already finer than a screen pixel on any of these
#: axes.  It matters because the entropy in this payload is the per-step jitter,
#: so two fewer bits per sample buys most of a click grid's worth of budget.
RUNS_SCALE = int(os.environ.get("VIEWER_RUNS_SCALE", "1000"))

#: Byte budget for the per-seed payload, before base64.  The click grid is
#: thinned until it fits.  Raise it for a study small enough to afford full
#: resolution; the page always says which grid it got.  The ceiling that matters
#: is the repo's 4 MB large-file gate, which the aggregate payload also draws on.
RUNS_BUDGET_MB = float(os.environ.get("VIEWER_RUNS_BUDGET_MB", "2.0"))

TEMPLATE = Path(__file__).with_name("viewer_template.html")

#: The one placeholder the template carries; see the note in its header comment.
TOKEN = "__VIEWER" + "_PAYLOAD__"

#: The **supervised-skyline** arms (issue #3322), in preference order.  These
#: rows are tagged in ``gmm_variant``, so ``analyze_spikes.load_arm`` filters
#: them out of the main frame by construction and the viewer re-reads the cell
#: CSVs for them (:func:`load_skyline`).
#:
#: ``skyline_train_full`` is the primary: the same head, through the same
#: trainer, on the entire sim split with full ground-truth labels.
#: ``skyline_test_xfit`` is its cross-fitted test-side bracket partner and is
#: used only when the primary was not run.
SKYLINE_ARMS: tuple[str, ...] = ("skyline_train_full", "skyline_test_xfit")

#: What each skyline arm is called on the page.  Spelled out rather than shown
#: as its arm name, because "skyline_test_xfit" tells a reader nothing about the
#: one thing they need to know: whether the number in front of them was fitted
#: on the training split or cross-fitted on the test split.
SKYLINE_LABELS: dict[str, str] = {
    "skyline_train_full": "skyline — fully-supervised head",
    "skyline_test_xfit": "skyline — cross-fitted on the test split",
}


def _b64gz(buf: bytes) -> str:
    return base64.b64encode(gzip.compress(buf, 9)).decode("ascii")


def _encode(arr: np.ndarray, scale: int = SCALE) -> dict:
    """Quantise, mask, delta-code along the last axis, gzip, base64.

    The mask is what lets the delta stream stay smooth across a gap: a NaN run
    is filled with its neighbour's value so it contributes zero deltas, and the
    separate 1-bit-per-sample mask (which is all runs, so it compresses to
    nearly nothing) restores the NaNs on the other side.
    """
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    valid = np.isfinite(flat)
    q = np.where(valid, np.clip(np.rint(flat * scale), -32000, 32000), np.nan)
    # Forward-fill the gaps, then zero whatever is still NaN (a leading gap).
    idx = np.where(valid, np.arange(flat.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    q = np.take_along_axis(np.nan_to_num(q, nan=FILL), idx, axis=1)
    qi = q.astype(np.int32)
    deltas = np.diff(qi, axis=1, prepend=0).astype(np.int16)
    return {
        "scale": scale,
        "shape": list(arr.shape),
        "v": _b64gz(deltas.tobytes()),
        "m": _b64gz(np.packbits(valid, axis=None).tobytes()),
    }


def _payload_bytes(enc: dict) -> int:
    return len(enc["v"]) + len(enc["m"])


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


#: Prefix of the derived per-row oracle columns :func:`add_oracle_columns` writes.
OCOL = "__oracle__"

#: Metrics the **oracle cut cannot move**, and therefore has no separate value
#: for.  ``average_precision`` and ``auroc`` are properties of the *ranking*:
#: they integrate over every threshold, so re-cutting the same scores at the
#: cost-minimising point leaves them exactly where they were.  Drawing an
#: "oracle AP" would be drawing the performance line twice, in two styles, and
#: inviting a reader to read a gap that is zero by construction.
#:
#: Every other metric in :data:`~vtscore.eval.calibration_metrics.DETECTION_METRICS`
#: *is* a statement about one cut, so each one has an oracle value and gets one.
RANKING_METRICS: frozenset[str] = frozenset({"average_precision", "auroc"})


def add_oracle_columns(main: pd.DataFrame) -> list[str]:
    """Fill ``__oracle__<metric>`` in place; return the metrics that got one.

    The harness emits the oracle **cut** and the two rates it pays there
    (``oracle_threshold`` / ``oracle_cost`` / ``oracle_fpr`` / ``oracle_fnr``)
    but not the confusion-matrix metrics at that cut.  It does not need to: an
    ``(FPR, FNR)`` pair plus the split's own class counts — ``n_test_pos`` and
    ``n_test_neg``, on every row — is a full confusion matrix, so precision,
    recall and F1 at the oracle cut are *reconstructed exactly* here rather than
    left off the page.  Deriving them beats re-running the grid to emit four
    more columns, and it beats offering the oracle on cost alone: "the cut cost
    us 0.1" and "the cut cost us 11 points of recall" are the same fact in the
    two units a reader actually thinks in.

    The conventions are :func:`~vtscore.eval.calibration_metrics.detection_metrics`'s,
    because a derived precision that treated "flagged nothing" as 0 rather than
    NaN would read as a catastrophically bad oracle where the truth is that the
    oracle declined to flag anything — which on a rare class is often the
    cost-minimising move.

    Returns the metric keys that ended up with an oracle column, so
    :func:`_metric_list` can flag them and the page can hide the control where
    there is nothing to show.
    """
    have = set(main.columns)
    got: list[str] = []
    if "oracle_cost" in have:
        main[OCOL + "cost"] = pd.to_numeric(main["oracle_cost"], errors="coerce")
        got.append("cost")
    if not {"oracle_fpr", "oracle_fnr"} <= have:
        return got
    fpr = pd.to_numeric(main["oracle_fpr"], errors="coerce")
    fnr = pd.to_numeric(main["oracle_fnr"], errors="coerce")
    main[OCOL + "fpr"] = fpr
    main[OCOL + "fnr"] = fnr
    main[OCOL + "recall"] = 1.0 - fnr
    got += ["fpr", "fnr", "recall"]
    if not {"n_test_pos", "n_test_neg"} <= have:
        return got
    n_pos = pd.to_numeric(main["n_test_pos"], errors="coerce")
    n_neg = pd.to_numeric(main["n_test_neg"], errors="coerce")
    tp = n_pos * (1.0 - fnr)
    fn = n_pos * fnr
    fp = n_neg * fpr
    flagged = tp + fp
    f1_denom = 2.0 * tp + fp + fn
    main[OCOL + "precision"] = (tp / flagged).where(flagged > 0)
    main[OCOL + "f1"] = (2.0 * tp / f1_denom).where(f1_denom > 0)
    got += ["precision", "f1"]
    return [k for k in got if main[OCOL + k].notna().any()]


def _metric_list(main: pd.DataFrame, oracle_keys: Sequence[str] = ()) -> list[dict]:
    """Every metric the run emitted, in the shared canonical order.

    Read from :data:`vtscore.eval.calibration_metrics.DETECTION_METRICS` so a
    metric's label and its direction come from the same place the harness
    computes it — a viewer that decided for itself which way is "better" is how
    "lower is better" gets attached to recall.
    """
    from vtscore.eval.calibration_metrics import DETECTION_METRICS

    oracle = {k for k in oracle_keys if k not in RANKING_METRICS}
    out = []
    for key, (label, lower, domain) in DETECTION_METRICS.items():
        if key in main.columns and main[key].notna().any():
            out.append(
                {
                    "key": key,
                    "label": label,
                    "lower": bool(lower),
                    "lo": domain[0],
                    "hi": domain[1],
                    "oracle": key in oracle,
                    "ranking": key in RANKING_METRICS,
                }
            )
    return out


def _thin(t_full: np.ndarray, keep: int) -> np.ndarray:
    """*keep* clicks out of *t_full*, always including click 0 and the horizon.

    Weighted toward the early clicks, which is where every one of these curves
    does its moving: an evenly spaced grid spends half its points on a plateau.
    """
    if keep >= len(t_full):
        return t_full
    lo, hi = float(t_full[0]), float(t_full[-1])
    # Even spacing in sqrt(t): dense at the start, sparse at the horizon.
    want = np.unique(np.rint(lo + (np.linspace(0.0, 1.0, keep) ** 2) * (hi - lo)).astype(int))
    return np.array([t for t in t_full if t in set(want.tolist())], dtype=int)


class _Shape:
    """The index a payload is built against: groups, arms, seeds, metrics."""

    def __init__(
        self,
        main: pd.DataFrame,
        arms: Sequence[str],
        denominator: pd.DataFrame | None,
        oracle_keys: Sequence[str] = (),
    ):
        self.arms = [a for a in arms if (main["arm"] == a).any()]
        self.metrics = _metric_list(main, oracle_keys)
        self.datasets = sorted(str(d) for d in main["dataset"].dropna().unique())
        self.embedders = (
            sorted(str(e) for e in main["embedder"].dropna().unique()) if "embedder" in main.columns else [""]
        )
        self.categories = sorted(str(c) for c in main["category"].dropna().unique())
        self.seeds = sorted(int(s) for s in main["seed"].dropna().unique())
        src = denominator if denominator is not None and not denominator.empty else main
        cols = [c for c in ("dataset", "embedder", "category") if c in src.columns]
        pairs = src.loc[:, cols].drop_duplicates()
        self.groups: list[tuple[str, str, str]] = sorted(
            (str(r[0]), str(r[1]) if "embedder" in cols else "", str(r[-1]))
            for r in pairs.itertuples(index=False, name=None)
        )
        self.gi = {g: i for i, g in enumerate(self.groups)}
        self.ai = {a: i for i, a in enumerate(self.arms)}
        self.si = {s: i for i, s in enumerate(self.seeds)}


#: Joins ``(dataset, embedder, category)`` into one groupby key.  A control
#: character rather than a space or a slash: COCO categories are things like
#: "baseball glove" and "hair drier", so any printable separator is a category
#: name away from splitting in the wrong place.
GSEP = "\u001f"


def _gkey(dataset: str, embedder: str, category: str) -> str:
    return GSEP.join((str(dataset), str(embedder or ""), str(category)))


def _group_key(df: pd.DataFrame) -> pd.Series:
    emb = df["embedder"].astype(str) if "embedder" in df.columns else pd.Series("", index=df.index)
    return df["dataset"].astype(str) + GSEP + emb + GSEP + df["category"].astype(str)


def _agg_arrays(  # noqa: C901
    main: pd.DataFrame,
    shape: _Shape,
    t_full: np.ndarray,
    base: dict[str, dict[tuple, float]],
    cells: dict[tuple[str, str], int],
) -> dict[str, np.ndarray]:
    """``mean`` / ``sd`` / ``n`` / ``cells`` over ``(group, arm, metric, click)``.

    ``n`` is stored beside the moments on purpose: it is what makes an "all
    categories" or "all datasets" selection *exact* in the page.  Pooling means
    of means would weight a category that trained on 3 cells the same as one
    that trained on 42, which is precisely the survivorship the coverage strip
    exists to expose.

    ``omean`` / ``on`` are the same thing for the **oracle cut** — the value the
    very same model would have scored had its threshold been chosen with the
    test labels in hand.  It carries its own ``n`` rather than borrowing the
    metric's, because the two genuinely differ: an oracle that declines to flag
    anything has an undefined precision at a click where the trained cut's
    precision is perfectly well defined, and pooling that cell in at weight 1
    with a NaN would poison the whole average.
    """
    nG, nA, nM, nT = len(shape.groups), len(shape.arms), len(shape.metrics), len(t_full)
    mean = np.full((nG, nA, nM, nT), np.nan)
    sd = np.full((nG, nA, nM, nT), np.nan)
    n = np.zeros((nG, nA, nM, nT))
    omean = np.full((nG, nA, nM, nT), np.nan)
    on = np.zeros((nG, nA, nM, nT))
    ncells = np.zeros((nG, nA))
    t_pos = {int(t): i for i, t in enumerate(t_full)}

    for (gk, arm), g in main.groupby(["__group", "arm"], dropna=False):
        gi, ai = shape.gi.get(tuple(str(gk).split(GSEP))), shape.ai.get(arm)
        if gi is None or ai is None:
            continue
        rows = g["t"].map(t_pos)
        ok = rows.notna()
        cols = rows[ok].astype(int).to_numpy()
        for mi, spec in enumerate(shape.metrics):
            vals = g.loc[ok, spec["key"]].to_numpy(dtype=float)
            good = np.isfinite(vals)
            if not good.any():
                continue
            # Sum / sumsq / count per click, accumulated with bincount so a
            # cell contributing at only some clicks lands only there.
            cnt = np.bincount(cols[good], minlength=nT).astype(float)
            s1 = np.bincount(cols[good], weights=vals[good], minlength=nT)
            s2 = np.bincount(cols[good], weights=vals[good] ** 2, minlength=nT)
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
                var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mu**2, np.nan)
            mean[gi, ai, mi] = mu
            sd[gi, ai, mi] = np.sqrt(np.clip(var, 0.0, None))
            n[gi, ai, mi] = cnt
        for mi, spec in enumerate(shape.metrics):
            ocol = OCOL + spec["key"]
            if not spec.get("oracle") or ocol not in g.columns:
                continue
            ovals = g.loc[ok, ocol].to_numpy(dtype=float)
            ogood = np.isfinite(ovals)
            if not ogood.any():
                continue
            ocnt = np.bincount(cols[ogood], minlength=nT).astype(float)
            os1 = np.bincount(cols[ogood], weights=ovals[ogood], minlength=nT)
            with np.errstate(invalid="ignore", divide="ignore"):
                omean[gi, ai, mi] = np.where(ocnt > 0, os1 / np.maximum(ocnt, 1), np.nan)
            on[gi, ai, mi] = ocnt

    for (ds, emb, cat), gi in shape.gi.items():
        for arm, ai in shape.ai.items():
            ncells[gi, ai] = cells.get((arm, _gkey(ds, emb, cat)), 0)

    # Click 0 is the zero-click text sort: every attempted cell has one, whether
    # or not it ever trained a detector.  Written here rather than left to the
    # page so the anchor obeys the same pooling as everything else.
    if 0 in t_pos:
        z = t_pos[0]
        for mi, spec in enumerate(shape.metrics):
            bm = base.get(spec["key"]) or {}
            if not bm:
                continue
            for (ds, emb, cat), gi in shape.gi.items():
                vals = [v for (d, e, c, _s), v in bm.items() if (d, e, c) == (ds, emb, cat)]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                for ai in range(nA):
                    if ncells[gi, ai] <= 0:
                        continue
                    mean[gi, ai, mi, z] = float(arr.mean())
                    sd[gi, ai, mi, z] = float(arr.std())
                    n[gi, ai, mi, z] = float(ncells[gi, ai])
    return {"mean": mean, "sd": sd, "n": n, "cells": ncells, "omean": omean, "on": on}


def _runs_arrays(
    main: pd.DataFrame,
    shape: _Shape,
    t_grid: np.ndarray,
    base: dict[str, dict[tuple, float]],
) -> tuple[np.ndarray, list[list[int]]]:
    """``values[(run, metric, click)]`` plus the ``(group, arm, seed)`` index.

    Built once at full click resolution; :func:`build_viewer` slices columns out
    of the result for each candidate grid rather than re-running this, which is
    the difference between a few seconds and several minutes on a grid this size.
    """
    t_pos = {int(t): i for i, t in enumerate(t_grid)}
    index: list[list[int]] = []
    nM, nT = len(shape.metrics), len(t_grid)
    seen: dict[tuple[int, int, int], int] = {}
    blocks: list[np.ndarray] = []

    def _slot(gi: int, ai: int, si: int) -> int:
        key = (gi, ai, si)
        if key not in seen:
            seen[key] = len(blocks)
            blocks.append(np.full((nM, nT), np.nan))
            index.append([gi, ai, si])
        return seen[key]

    keys = [spec["key"] for spec in shape.metrics]
    for (gk, arm, seed), g in main.groupby(["__group", "arm", "seed"], dropna=False, sort=False):
        gi, ai, si = shape.gi.get(tuple(str(gk).split(GSEP))), shape.ai.get(arm), shape.si.get(int(seed))
        if gi is None or ai is None or si is None:
            continue
        cols = g["t"].map(t_pos).to_numpy()
        ok = np.isfinite(pd.to_numeric(cols, errors="coerce").astype(float))
        if not ok.any():
            continue
        ci = cols[ok].astype(int)
        blocks[_slot(gi, ai, si)][:, ci] = g.loc[ok, keys].to_numpy(dtype=float).T

    # Every attempted cell gets its click-0 anchor, including one that never
    # trained: a lone point at the far left is exactly what total starvation
    # looks like, and dropping it would render that run as simply absent.
    if 0 in t_pos:
        z = t_pos[0]
        for mi, spec in enumerate(shape.metrics):
            for (ds, emb, cat, seed), val in (base.get(spec["key"]) or {}).items():
                gi = shape.gi.get((ds, emb, cat))
                si = shape.si.get(int(seed))
                if gi is None or si is None or not np.isfinite(val):
                    continue
                for ai in range(len(shape.arms)):
                    blocks[_slot(gi, ai, si)][mi, z] = val
    return (np.stack(blocks) if blocks else np.zeros((0, nM, nT))), index


def load_skyline(results: Path, dirs: Sequence[str], arms: Sequence[str]) -> pd.DataFrame:
    """The **supervised-skyline** rows (issue #3322) under *results*, if any.

    These have to be read separately because
    :func:`analyze_spikes.load_arm` — which every analyzer and
    :func:`curves._load` go through — keeps only *base* rows, and a skyline row
    is tagged in ``gmm_variant`` precisely so it cannot be mistaken for one.
    That filter is right: a skyline reads ground-truth labels the app can never
    see, so it must never land in a mean of what the app achieved.  It is also
    why the skyline needs its own door into the page.

    One row per ``(cell, skyline arm)``, at ``t = 0`` and vote-independent, so
    the frame is tiny next to the main one.  ``arm`` is set to the *results*
    arm's label, matching the main frame; the skyline arm's own name lands in
    ``gmm_variant``.
    """
    import _cells_io
    import analyze_spikes as sp  # noqa: F401  (kept beside _cells_io: same reader family)

    want = set(SKYLINE_ARMS)
    parts = []
    for d, label in zip(dirs, arms, strict=True):
        for f in _cells_io.main_frame_files(Path(results) / d / "cells"):
            if f.stat().st_size == 0:
                continue
            try:
                fr = pd.read_csv(f)
            except Exception:  # noqa: BLE001, S112
                continue
            if fr.empty or "gmm_variant" not in fr.columns:
                continue
            fr = fr[fr["gmm_variant"].astype(str).str.strip().isin(want)]
            if fr.empty:
                continue
            fr = fr.copy()
            fr["arm"] = label
            parts.append(fr)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _skyline_arrays(skyline: pd.DataFrame | None, shape: _Shape) -> tuple[np.ndarray, np.ndarray, str]:
    """``(mean, n, arm_name)`` over ``(group, arm, metric)`` — one number, not a line.

    The skyline is **vote-independent**: it is what the same head reaches on the
    same cell with every training label handed to it, so it does not move as the
    reader clicks and it is drawn as a notch at the horizon rather than as a
    line across the chart.  A line would assert that the floor was reachable at
    click 3, which is exactly the reading the point exists to prevent.

    Only one skyline arm reaches the page — :data:`SKYLINE_ARMS` in preference
    order — because the two are a *bracket* on the same quantity rather than two
    findings, and two notches a hair apart at the same x read as a comparison
    that was never run.
    """
    nG, nA, nM = len(shape.groups), len(shape.arms), len(shape.metrics)
    mean = np.full((nG, nA, nM), np.nan)
    n = np.zeros((nG, nA, nM))
    if skyline is None or skyline.empty:
        return mean, n, ""
    tags = skyline["gmm_variant"].astype(str).str.strip()
    pick = next((a for a in SKYLINE_ARMS if (tags == a).any()), "")
    if not pick:
        return mean, n, ""
    df = skyline[tags == pick].copy()
    if "embedder" not in df.columns:
        df["embedder"] = ""
    df["__group"] = _group_key(df)
    for (gk, arm), g in df.groupby(["__group", "arm"], dropna=False):
        gi, ai = shape.gi.get(tuple(str(gk).split(GSEP))), shape.ai.get(arm)
        if gi is None or ai is None:
            continue
        for mi, spec in enumerate(shape.metrics):
            if spec["key"] not in g.columns:
                continue
            vals = pd.to_numeric(g[spec["key"]], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            mean[gi, ai, mi] = float(vals.mean())
            n[gi, ai, mi] = float(vals.size)
    return mean, n, pick


def _baselines(baseline: pd.DataFrame | None, shape: _Shape) -> dict[str, dict[tuple, float]]:
    """``metric -> {(dataset, embedder, category, seed): value}``.

    Uses :func:`curves.baseline_map` for the column lookup, so the page's click-0
    anchor and the PNG's click-0 anchor read the same column of the same file.
    """
    if baseline is None or baseline.empty:
        return {}
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in baseline.columns]
    out: dict[str, dict[tuple, float]] = {}
    for spec in shape.metrics:
        m = curves.baseline_map(baseline, spec["key"], keys)
        if not m:
            continue
        fixed: dict[tuple, float] = {}
        for k, v in m.items():
            rec = dict(zip(keys, k, strict=False))
            fixed[
                (
                    str(rec.get("dataset", "")),
                    str(rec.get("embedder", "")),
                    str(rec.get("category", "")),
                    int(rec.get("seed", 0)),
                )
            ] = v
        out[spec["key"]] = fixed
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_viewer(  # noqa: C901
    main: pd.DataFrame,
    out_path: Path,
    *,
    arms: Sequence[str],
    denominator: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    skyline: pd.DataFrame | None = None,
    title: str = "Quality over clicks",
    subtitle: str = "",
    runs_budget_mb: float = RUNS_BUDGET_MB,
    anchor_label: str = curves.BASELINE_LABEL,
    template: Path = TEMPLATE,
) -> Path:
    """Write the self-contained viewer HTML.  Returns *out_path*."""
    if main.empty:
        raise SystemExit("viewer: no rows to build from")
    main = main.copy()
    if "embedder" not in main.columns:
        main["embedder"] = ""
    main["__group"] = _group_key(main)
    oracle_keys = add_oracle_columns(main)

    shape = _Shape(main, arms, denominator, oracle_keys)
    if not shape.metrics:
        raise SystemExit("viewer: the frame carries none of the known metric columns")

    den = denominator if denominator is not None and not denominator.empty else main
    den = den.copy()
    if "embedder" not in den.columns:
        den["embedder"] = ""
    den["__group"] = _group_key(den)
    cells = {
        (str(arm), str(gk)): int(d.loc[:, ["seed"]].drop_duplicates().shape[0])
        for (arm, gk), d in den.groupby(["arm", "__group"])
    }

    base = _baselines(baseline, shape)
    has_anchor = bool(base)
    t_full = np.arange(0 if has_anchor else 1, int(main["t"].max()) + 1)

    ag = _agg_arrays(main, shape, t_full, base, cells)
    agg = {
        "mean": _encode(ag["mean"]),
        "sd": _encode(ag["sd"]),
        "n": _encode(ag["n"], scale=1),
        "cells": _encode(ag["cells"].reshape(len(shape.groups), len(shape.arms), 1), scale=1),
        "omean": _encode(ag["omean"]),
        "on": _encode(ag["on"], scale=1),
    }

    sky_mean, sky_n, sky_arm = _skyline_arrays(skyline, shape)
    sky = (
        {
            "arm": sky_arm,
            "label": SKYLINE_LABELS.get(sky_arm, sky_arm),
            # Trailing axis of length 1: the encoder delta-codes along the last
            # axis, and a scalar per (group, arm, metric) is a one-click series.
            "mean": _encode(sky_mean[:, :, :, None]),
            "n": _encode(sky_n[:, :, :, None], scale=1),
        }
        if sky_arm
        else None
    )

    # Per-seed payload: thin the click axis until it fits, and say which grid it
    # landed on.  Never drop runs, never drop metrics - a reader comparing two
    # arms must be looking at the same set of seeds for both, and a metric that
    # silently vanished from one view is worse than a coarser axis in all of
    # them.  Built once at full resolution and sliced, because rebuilding the
    # frame per candidate grid is minutes rather than seconds at this size.
    budget = int(runs_budget_mb * 1024 * 1024)
    runs = None
    runs_note = ""
    full_vals, index = _runs_arrays(main, shape, t_full, base)
    sizes: dict[str, int] = {k: _payload_bytes(v) for k, v in agg.items()}
    if len(index):
        candidates = [len(t_full), 120, 80, 60, 45, 34, 26, 20, 15]
        candidates = [c for i, c in enumerate(candidates) if c <= len(t_full) and c not in candidates[:i]]
        pos = {int(t): i for i, t in enumerate(t_full)}
        for keep in candidates:
            grid = _thin(t_full, keep)
            enc = _encode(full_vals[:, :, [pos[int(t)] for t in grid]], scale=RUNS_SCALE)
            fits = _payload_bytes(enc) <= budget
            if fits or keep == candidates[-1]:
                if not fits:
                    runs_note = (
                        f"Per-seed lines are drawn at {len(grid)} of {len(t_full)} clicks — the coarsest grid "
                        f"available, and still over the {runs_budget_mb:g} MB payload budget."
                    )
                elif len(grid) < len(t_full):
                    runs_note = (
                        f"Per-seed lines are drawn at {len(grid)} of {len(t_full)} clicks, thinned to fit the "
                        f"{runs_budget_mb:g} MB payload budget (denser at the start, where these curves move). "
                        f"The averaged view is at full resolution."
                    )
                else:
                    runs_note = f"Per-seed lines are at full click resolution ({len(t_full)} clicks)."
                runs = {"t": [int(x) for x in grid], "index": index, "values": enc}
                sizes["runs"] = _payload_bytes(enc)
                break
    else:
        runs_note = "No per-seed rows were found, so per-seed lines are unavailable."

    payload = {
        "schema": 1,
        "title": title,
        "subtitle": subtitle,
        "anchor": {"has": has_anchor, "label": anchor_label},
        "skyline": sky,
        "solid_coverage": curves.SOLID_COVERAGE,
        "datasets": shape.datasets,
        "embedders": shape.embedders,
        "categories": shape.categories,
        "arms": shape.arms,
        "seeds": shape.seeds,
        "metrics": shape.metrics,
        "groups": [list(g) for g in shape.groups],
        "t": [int(x) for x in t_full],
        "agg": agg,
        "runs": runs,
        "runs_note": runs_note,
        "n_cells": int(sum(cells.values())),
        "oracle_metrics": [k for k in oracle_keys if k not in RANKING_METRICS],
        "payload_kb": {k: round(v / 1024) for k, v in sizes.items()},
    }

    html = template.read_text(encoding="utf-8")
    # Exactly one, or the page silently doubles: the substitution is a plain
    # string replace, and the token used to appear in the template's own header
    # comment too - which cost 3 MB and looked like a payload problem.
    if html.count(TOKEN) != 1:
        raise SystemExit(f"{template}: expected exactly 1 {TOKEN}, found {html.count(TOKEN)}")
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html.replace(TOKEN, blob), encoding="utf-8")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(common.RESULTS), help="results root holding one dir per arm")
    ap.add_argument(
        "--arms",
        required=True,
        help="comma-separated arm directories, in report order; `dir=label` renames one on the page",
    )
    ap.add_argument("--out", required=True, help="path to write the HTML to")
    ap.add_argument("--baseline", default=None, help="text_baseline.py CSV: the click-0 anchor")
    ap.add_argument("--title", default="Quality over clicks")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--runs-budget-mb", type=float, default=RUNS_BUDGET_MB)
    ap.add_argument(
        "--no-skyline",
        action="store_true",
        help="skip the supervised-skyline pass over the cell CSVs (issue #3322)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    # `dir=label` exists for the single-arm studies. A study that sweeps a knob
    # has one results directory per arm and the directory name IS the arm name,
    # which is why this started as a bare list. A descriptive study has one
    # directory called `results`, and that word would then be the label on its
    # only chip and in every caption -- naming the filesystem where the reader
    # needs the configuration ("prod"). The pairing is positional so the arm
    # ORDER, which sets colours and report order, still comes from one place.
    specs = [a for a in args.arms.replace(",", " ").split() if a]
    dirs = [spec.partition("=")[0] for spec in specs]
    arms = [spec.partition("=")[2] or spec.partition("=")[0] for spec in specs]
    frame = curves._load(Path(args.results), dirs)
    if frame.empty:
        print(f"no rows under {args.results} for arms {dirs}")
        return 2
    if arms != dirs:
        frame["arm"] = frame["arm"].map(dict(zip(dirs, arms, strict=True)))
    baseline = curves.text_sort_baseline(args.baseline) if args.baseline else None
    skyline = None if args.no_skyline else load_skyline(Path(args.results), dirs, arms)
    out = build_viewer(
        frame,
        Path(args.out),
        arms=arms,
        baseline=baseline,
        skyline=skyline,
        title=args.title,
        subtitle=args.subtitle,
        runs_budget_mb=args.runs_budget_mb,
    )
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    if not args.baseline:
        print("NOTE: no --baseline, so the page has no click-0 anchor and nothing to compare the far right against.")
    if skyline is None or skyline.empty:
        print(
            "NOTE: no supervised-skyline rows under --results, so the page has no learnability floor. "
            "Re-run with CALIB_SKYLINE_ARMS=skyline_train_full to get one (issue #3322)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

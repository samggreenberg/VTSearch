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
* **target size** — on a study whose categories are ``<class>@<band>``: one
  size, all of them averaged, or each as its own line (#4635);
* **category** — one within that dataset, or all averaged;
* **embedder** — any non-empty subset, drawn as **one panel each**.  Embedders
  are never averaged with one another: two embedders are two different
  representations of the haystack, and their mean is a number describing no
  system anyone could run.  Faceting makes that structural rather than a rule
  someone has to remember.  An embedder is named by its key unless the study
  gives it a label (``--embedder-label``): the State of the App's region path
  is the composite key ``siglip+dinov3_patch``, which read bare looks like two
  embedders compared (#4655);
* **arms** — any non-empty subset.  Non-empty for the same reason the embedder
  subset is: an empty selection has no honest rendering, since the page either
  goes blank or falls back to "all" and a reader who misses that takes a chart
  of everything for a chart of nothing.  The page locks the last remaining chip
  rather than snapping it silently back on.  On a page built from a review's
  session sets (``--beta-run``, #4636) the arms are the **beta the sessions ran
  at**, one chip per preset, and the control says so;
* **seeds** — averaged, or every seed as its own line;
* **metric** — cost, precision, recall, F1/4, F1, F4 (the returned set at
  each preset balance), the objective (F-beta at the run's own balance), FPR,
  FNR, average precision, AUROC: whatever the run emitted, from
  one shared definition
  (:data:`vtscore.eval.calibration_metrics.DETECTION_METRICS`).  The page opens
  on F1 (:data:`DEFAULT_METRIC`, #4635); a run that emitted no F1 opens on the
  objective when it drew its line at a balance (#4584), else on the first
  metric.  A study can say otherwise: ``--default-metric``
  picks the one it opens on and ``--hide-metrics`` takes some off the menu, for
  a study whose report has retired one (the State of the App dropped cost,
  #4576).  Both live in the payload's ``view`` block, so a later ``--reskin``
  keeps them, and ``--reskin`` takes the same two flags to set them on a page
  that is already built.  Hiding is not dropping: the numbers stay in the
  payload, so a hidden metric comes back with a reskin rather than a rebuild;
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
    What typing the query got for free, at zero clicks.  A metric about the
    returned set is read off the set the text sort's own line returns, at the
    balance the arm's line was drawn at (:func:`curves.text_line_values`), so
    the notch and the curve are one rule (#4474) and each session set starts
    from the line the app shows at its beta.
``skyline`` (a point, notched in the **right** margin)
    The same head, through the same trainer, with every training label handed
    to it (issue #3322) — the learnability floor of this embedding space.
    Vote-independent, so it does not move as the reader clicks.

**Nothing joins the text-sort notch to the curve.**  The dotted bridge that used
to run from click 0 to the first trained click was the most prominent mark on
the chart and it stood for a measurement nobody made: between the two there is
no detector, so there is no level to draw.  A gap says exactly that.

**The axis ends at the last click, not at the spot check.**  A floor-era run
ends on its spot check (#4272), whose rows sit past ``max_steps`` and come from
a model retrained on the check's own picks.  ``_cells_io.load_arm`` sets them
apart (#4364), so the page never draws them as clicks.

The two notches sit in the margins rather than as rules across the panel for the
same reason: each is a level that holds at *one* x, and a horizontal rule across
the chart asserts it holds at every x.  The skyline is the sharper case — drawn
as a rule it would read as "the floor was reachable at click 3", which is the
one reading the number exists to prevent.

**Through Autopilot's opening the line is the detector, not the session**
(owner, 2026-10-07, #4640).  The app stays on the text sort until the Hard
phase (``app_trained``, #4605), but every vote is saved as it is cast, and
Export labels and Test have no phase gate: a user can take the labels away at
any click, and an import or a Test builds a detector from them.  So from a
run's first Good the page draws the harness's detector, which is what that
gives: since #4643, the Goods' centroid until the labels hold 3 Goods and 4
Bads (the label quota), the trained head from there.  The harness writes each
row as that detector (``detector_tier`` names which); nothing here reads
``detector_tier`` or ``app_trained``.  Before the first Good, Test is refused
and the click is the empty set (:func:`curves.score_empty_sets`).  A page built
from results written before #4643 starts its rows at the first Good and Bad and
draws a trained head there, the app as it was.  A report that scores the
session (``state_of_app/analyze.py``) differs from the page through the opening
on purpose.

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
    contributed.  ``cells`` beside them, per ``(group, arm)``, is what coverage
    divides by: every run *attempted*, the ones that never trained included,
    read off the caller's cell list or the text-sort baseline
    (:func:`curves.attempted_cells`).  Every one of those runs is scored at
    every click, a click with no trained detector as the empty returned set
    (:func:`curves.score_empty_sets`), so a run that never trained is in
    the mean as the loss it was rather than out of it.  That needs the
    baseline (for each run's prevalence, AP's chance level), so a committed
    page built before it is rebuilt from its results, not reskinned.
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
    because ``_cells_io.load_arm`` filters skyline rows out by design — a
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

A review that ran one set of sessions per preset (``SOTA_BETA``, #4413) puts
them on one page, a chip per beta, each run directory read from its
``results/`` and refused if its rows were drawn at another beta (#4636)::

    python viewer.py --beta-run 0.25=<b025> --beta-run 1=<b1> --beta-run 4=<b4> \\
        --baseline <b1>/text_baseline.csv --out "$OUT/viewer.html"

A study whose report has retired a metric opens on another and hides it, and
the same two flags set that on a page that is already built::

    python viewer.py ... --default-metric average_precision --hide-metrics cost
    python viewer.py --reskin path/to/viewer.html \\
        --default-metric average_precision --hide-metrics cost

A page names an embedder by a label rather than its key, and a reskin can
re-label a committed page and rewrite its subtitle (#4655)::

    python viewer.py ... --embedder-label "siglip+dinov3_patch=DINOv3 region"
    python viewer.py --reskin path/to/viewer.html --subtitle "..." \\
        --embedder-label "siglip+dinov3_patch=DINOv3 region"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import base64
import gzip
import json
import os
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import common

common.setup_env()

import numpy as np  # noqa: E402
import objective  # noqa: E402
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

#: The metric a page opens on when its study names none (#4635).  The template
#: makes the choice (its ``OPEN_ON``), so a committed page picks it up on a plain
#: reskin; the builder only needs it to know when the template cannot.
DEFAULT_METRIC = "f1"

#: The one placeholder the template carries; see the note in its header comment.
TOKEN = "__VIEWER" + "_PAYLOAD__"

#: The **supervised-skyline** arms (issue #3322), in preference order.  These
#: rows are tagged in ``gmm_variant``, so ``_cells_io.load_arm`` filters
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


def _decode(enc: dict) -> np.ndarray:
    """The inverse of :func:`_encode`, as the page decodes it: NaN where the mask says so."""
    shape = enc["shape"]
    n_t = int(shape[-1])
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    deltas = np.frombuffer(gzip.decompress(base64.b64decode(enc["v"])), dtype=np.int16).reshape(rows, n_t)
    vals = np.cumsum(deltas.astype(np.int64), axis=1) / float(enc["scale"])
    mask = np.unpackbits(np.frombuffer(gzip.decompress(base64.b64decode(enc["m"])), dtype=np.uint8))
    valid = mask[: rows * n_t].reshape(rows, n_t).astype(bool)
    return np.where(valid, vals, np.nan).reshape(shape)


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


def opening_view(keys: Sequence[str], *, metric: str | None = None, hide: Sequence[str] = ()) -> dict:
    """The payload's ``view`` block: the metric the page opens on, and the ones it hides.

    *keys* are the metrics the payload carries.  The block says how the page
    **opens**, never what it carries, so it can be rewritten on a built page
    (:func:`reskin`) without touching a number.

    Returns ``{}`` when there is nothing to say, so a page built without either
    choice has no ``view`` key and reads exactly as it did before there was one.
    Raises rather than letting the page fall back on its own, because a page
    that quietly opened on its first metric is the bug this block exists to
    fix (#4576): a misspelt name, a default the page cannot open on, or a
    ``hide`` that leaves nothing to offer each fail the build here.  Hiding a
    known metric the run never emitted is allowed, since the page then has
    nothing to hide.
    """
    from vtscore.eval.calibration_metrics import DETECTION_METRICS

    unknown = [k for k in hide if k not in DETECTION_METRICS]
    if unknown:
        raise SystemExit(
            f"viewer: no such metric to hide: {', '.join(unknown)} (known: {', '.join(DETECTION_METRICS)})"
        )
    hidden = [k for k in DETECTION_METRICS if k in set(hide)]
    offered = [k for k in keys if k not in hidden]
    if not offered:
        raise SystemExit(f"viewer: hiding {', '.join(hidden)} leaves the page no metric to offer")
    if metric and metric not in keys:
        raise SystemExit(f"viewer: cannot open on {metric!r}: the payload carries only {', '.join(keys)}")
    if metric and metric in hidden:
        raise SystemExit(f"viewer: cannot open on {metric!r} and hide it")
    view: dict = {}
    if metric:
        view["metric"] = metric
    if hidden:
        view["hide"] = hidden
    return view


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
    base: dict[str, dict[str, dict[tuple, float]]],
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
    # page so the anchor obeys the same pooling as everything else.  Each arm
    # reads its own anchor: a cut metric's is the text sort's line at the beta
    # that arm's line was drawn at (`_baselines`).
    if 0 in t_pos:
        z = t_pos[0]
        for arm, ai in shape.ai.items():
            for mi, spec in enumerate(shape.metrics):
                by_group: dict[tuple[str, str, str], list[float]] = {}
                for (d, e, c, _s), v in (base.get(arm, {}).get(spec["key"]) or {}).items():
                    by_group.setdefault((d, e, c), []).append(v)
                for group, vals in by_group.items():
                    gi = shape.gi.get(group)
                    if gi is None or ncells[gi, ai] <= 0:
                        continue
                    arr = np.asarray(vals, dtype=float)
                    mean[gi, ai, mi, z] = float(arr.mean())
                    sd[gi, ai, mi, z] = float(arr.std())
                    n[gi, ai, mi, z] = float(ncells[gi, ai])
    return {"mean": mean, "sd": sd, "n": n, "cells": ncells, "omean": omean, "on": on}


def _runs_arrays(
    main: pd.DataFrame,
    shape: _Shape,
    t_grid: np.ndarray,
    base: dict[str, dict[str, dict[tuple, float]]],
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
        for arm, ai in shape.ai.items():
            for mi, spec in enumerate(shape.metrics):
                for (ds, emb, cat, seed), val in (base.get(arm, {}).get(spec["key"]) or {}).items():
                    gi = shape.gi.get((ds, emb, cat))
                    si = shape.si.get(int(seed))
                    if gi is None or si is None or not np.isfinite(val):
                        continue
                    blocks[_slot(gi, ai, si)][mi, z] = val
    return (np.stack(blocks) if blocks else np.zeros((0, nM, nT))), index


def load_skyline(results: Path, dirs: Sequence[str], arms: Sequence[str]) -> pd.DataFrame:
    """The **supervised-skyline** rows (issue #3322) under *results*, if any.

    These have to be read separately because
    :func:`_cells_io.load_arm` — which every analyzer and
    :func:`curves._load` go through — keeps only *base* rows, and a skyline row
    is tagged in ``gmm_variant`` precisely so it cannot be mistaken for one.
    That filter is right: a skyline reads ground-truth labels the app can never
    see, so it must never land in a mean of what the app achieved.  It is also
    why the skyline needs its own door into the page.

    One row per ``(cell, skyline arm)``, at ``t = 0`` and vote-independent, so
    the frame is tiny next to the main one.  ``arm`` is set to the *results*
    arm's label, matching the main frame; the skyline arm's own name lands in
    ``gmm_variant``.

    *results* need not be the root the curves came from (``--skyline-results``).
    Vote-independence is what makes that sound: the floor is what the same head
    reaches on the same split with every label handed to it, so a later pass
    over the same cells measures the same quantity as the original run would
    have, had it been asked.  The alternative -- re-running the loop to collect
    a floor -- **replaces** the performance rows a finished report's tables were
    read off, which is a worse trade than the one this avoids.

    What it does not do is check that the two roots describe the same study: a
    skyline row lands by ``(dataset, embedder, category)``, so rows from a
    foreign grid are dropped rather than merged, but rows from the same grid at
    a different configuration would be taken at face value.  Point it at a pass
    over the same cells.
    """
    import _cells_io

    want = set(SKYLINE_ARMS)
    parts = []
    for d, label in zip(dirs, arms, strict=True):
        for f in _cells_io.main_frame_files(Path(results) / d / "cells"):
            if f.stat().st_size == 0:
                continue
            try:
                fr = _cells_io.legacy_datasets(pd.read_csv(f))
            except Exception:  # noqa: BLE001
                # A cell whose CSV is truncated costs its skyline, never the
                # page: the main frame's own loader reports unreadable cells,
                # and this pass must not be the thing that fails the build.
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


#: What the arms control says on a page whose arms are a review's session sets,
#: one per balance (#4636): the reader is choosing the beta the sessions ran
#: at, which is a different thing from the beta a metric scores at.  A short
#: title (the page sets control titles in capitals) and the hint beside it.
BETA_ARMS_CONTROL = {
    "title": "Sessions' beta",
    "hint": "one or more; the balance each set of sessions ran at, which F1/4, F1 and F4 then score",
}


def parse_embedder_labels(specs: Sequence[str]) -> dict[str, str]:
    """``KEY=LABEL`` pairs; refuses a malformed pair and a key named twice.

    Split at the first ``=``: an embedder key never holds one, a label may.
    """
    labels: dict[str, str] = {}
    for spec in specs:
        key, sep, label = spec.partition("=")
        key, label = key.strip(), label.strip()
        if not sep or not key or not label:
            raise SystemExit(f"--embedder-label wants KEY=LABEL, got {spec!r}")
        if key in labels:
            raise SystemExit(f"--embedder-label names {key!r} twice")
        labels[key] = label
    return labels


def page_embedder_labels(embedders: Sequence[str], labels: Mapping[str, str] | None) -> dict[str, str]:
    """The labels for the embedders a page carries, in the page's order.

    A label for an embedder the page does not carry is dropped, so one set of
    labels serves every page of a study: the State of the App names both paths
    and each path's page keeps its own (#4655).
    """
    return {e: labels[e] for e in embedders if labels and e in labels}


def _with_embedder_labels(payload: dict, labels: Mapping[str, str]) -> dict:
    """*payload* with its ``embedder_labels`` block replaced, in place.

    The block sits right after ``embedders``, where a build puts it, so a
    built and a relabelled page agree key for key; an empty mapping drops it.
    """
    items = [(k, v) for k, v in payload.items() if k != "embedder_labels"]
    if labels:
        at = next(i for i, (k, _v) in enumerate(items) if k == "embedders") + 1
        items.insert(at, ("embedder_labels", dict(labels)))
    payload.clear()
    payload.update(items)
    return payload


def beta_label(beta: float) -> str:
    """``β 1/4``, ``β 1``, ``β 4``: a session set's chip, in the fraction the app's presets are named by."""
    return f"β {Fraction(beta).limit_denominator(64)}"


def parse_beta_runs(specs: Sequence[str]) -> list[tuple[float, Path]]:
    """``BETA=DIR`` pairs, sorted by beta; refuses a malformed pair and a beta named twice."""
    runs: list[tuple[float, Path]] = []
    for spec in specs:
        beta_s, sep, d = spec.partition("=")
        try:
            beta = float(beta_s)
        except ValueError:
            beta = float("nan")
        if not sep or not d or not (np.isfinite(beta) and beta > 0):
            raise SystemExit(f"viewer: --beta-run wants BETA=DIR with a positive beta, got {spec!r}")
        runs.append((beta, Path(d)))
    betas = [b for b, _ in runs]
    if len(set(betas)) != len(betas):
        raise SystemExit(f"viewer: a beta is named twice in --beta-run: {', '.join(f'{b:g}' for b in betas)}")
    return sorted(runs)


def load_beta_runs(
    runs: Sequence[tuple[float, Path]], *, skyline: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """``(frame, skyline, arms)``: one arm per session set, each chip named for the beta it ran at (#4636).

    A review runs one set of sessions per preset (``SOTA_BETA``, #4413), each
    in its own run directory, and the app at each beta trains, checks and
    draws its line differently, so the sets are three configurations of the
    app rather than three readings of one.  Each *runs* entry is ``(beta,
    run directory)``, read from the directory's ``results/`` as ``analyze.sh``
    reads it.  A set whose rows were drawn at another beta, at several, or at
    none is refused: a swapped pair would put the beta-4 sessions under the
    ``β 1/4`` chip, which nothing on screen would show and which inverts every
    comparison the page exists for.
    """
    parts, skies, arms = [], [], []
    for beta, run_dir in runs:
        label = beta_label(beta)
        frame = curves._load(run_dir, ["results"])
        if frame.empty:
            raise SystemExit(f"viewer: no rows under {run_dir / 'results'} for the beta {beta:g} sessions")
        found = objective.frame_betas(frame)
        if len(found) != 1 or abs(found[0] - beta) > 1e-9:
            raise SystemExit(
                f"viewer: {run_dir} was named as the beta {beta:g} sessions, but its rows were drawn at "
                f"{', '.join(f'{b:g}' for b in found) or 'no balance'}"
            )
        parts.append(frame.assign(arm=label))
        arms.append(label)
        if skyline:
            skies.append(load_skyline(run_dir, ["results"], [label]))
    sky = pd.concat([s for s in skies if not s.empty], ignore_index=True) if any(not s.empty for s in skies) else None
    return pd.concat(parts, ignore_index=True), sky, arms


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


def _baselines(
    baseline: pd.DataFrame | None, shape: _Shape, arm_betas: Mapping[str, float | None]
) -> dict[str, dict[str, dict[tuple, float]]]:
    """``arm -> metric -> {(dataset, embedder, category, seed): value}``.

    Uses :func:`curves.baseline_map` for the lookup, so the page's click-0
    anchor and the PNG's click-0 anchor read the same columns of the same file.
    Per arm, because a cut metric's anchor is the set the text sort's line
    returns at the balance that arm's line was drawn at (*arm_betas*, from
    :func:`objective.frame_beta`): a page whose arms are a review's session
    sets at beta 1/4, 1 and 4 (#4636) starts each from the line the app shows
    at that beta.  Arms at one beta share one set of maps.
    """
    if baseline is None or baseline.empty:
        return {}
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in baseline.columns]
    by_beta: dict[float | None, dict[str, dict[tuple, float]]] = {}
    out: dict[str, dict[str, dict[tuple, float]]] = {}
    for arm in shape.arms:
        beta = arm_betas.get(arm)
        if beta not in by_beta:
            maps: dict[str, dict[tuple, float]] = {}
            for spec in shape.metrics:
                m = curves.baseline_map(baseline, spec["key"], keys, beta=beta)
                if m:
                    maps[spec["key"]] = {_anchor_key(dict(zip(keys, k, strict=False))): v for k, v in m.items()}
            by_beta[beta] = maps
        if by_beta[beta]:
            out[arm] = by_beta[beta]
    return out


def _anchor_key(rec: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(rec.get("dataset", "")),
        str(rec.get("embedder", "")),
        str(rec.get("category", "")),
        int(rec.get("seed", 0)),
    )


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
    build: dict | None = None,
    default_metric: str | None = None,
    hide_metrics: Sequence[str] = (),
    fill_gaps: bool = True,
    score_empty_sets: bool = True,
    arms_control: Mapping[str, str] | None = None,
    embedder_labels: Mapping[str, str] | None = None,
) -> Path:
    """Write the self-contained viewer HTML.  Returns *out_path*.

    *arms_control* (``title`` and ``hint``) relabels the arms control when the
    arms are not a study's configurations: a review's session sets, one per
    preset, are a balance the reader picks (:data:`BETA_ARMS_CONTROL`,
    :func:`load_beta_runs`, #4636).

    *embedder_labels* names an embedder on the page by what it is rather than
    by its key (:func:`page_embedder_labels`, #4655); an embedder without one
    shows its key.

    *default_metric* and *hide_metrics* set the page's opening ``view``; see
    :func:`opening_view`.  Without a *default_metric* the page opens on
    :data:`DEFAULT_METRIC`, F1 (#4635), and the ``view`` says nothing about it.
    A frame that offers no F1 but carries a beta opens on the objective (#4584),
    its columns filled from the rates where the cells predate them; any other
    frame opens on the first metric.

    *fill_gaps* carries each run's last scored row through the clicks it has
    no row for (:func:`curves.fill_gaps`, #4624): a run inside a prompted
    spot check is scored once per round of picks, and without the carry the
    averaged line skips it between rounds, which on a review is a survivor's
    mean over the sessions the check did not prompt.  Off only for a study of
    the rows themselves; the page says in its reading note when it is on
    (``gaps_filled``).

    *denominator* is one row per attempted cell, the cells coverage is
    measured against.  Without one, a *baseline* supplies it
    (:func:`curves.attempted_cells`): *main* holds only the runs that
    trained, so a denominator read off it calls a starving group fully
    measured.  With neither, it is the cells of *main*.

    *score_empty_sets* scores every empty returned set as one
    (:func:`curves.score_empty_sets`): every attempted run at every click it
    had no trained detector at, and the undefined precision of a detector
    that flags nothing, as 0.  The user got nothing back, so the click is a
    loss, and leaving it out of the mean averaged over the sessions that
    worked.  The oracle companion follows the same rule: a click with no
    model gets the same values (there is no cut to move), and an oracle cut
    that flags nothing counts its precision as 0.  The page says so in its
    reading note (``empty_sets_scored``).
    """
    if main.empty:
        raise SystemExit("viewer: no rows to build from")
    main = objective.with_objective(main).copy()
    if "embedder" not in main.columns:
        main["embedder"] = ""
    main["__group"] = _group_key(main)
    oracle_keys = add_oracle_columns(main)
    if fill_gaps:
        main = curves.fill_gaps(main, ("__group", "arm", "seed"))
    if (denominator is None or denominator.empty) and baseline is not None and not baseline.empty:
        denominator = curves.attempted_cells(main, baseline)
    if score_empty_sets:
        main = curves.score_empty_sets(main, denominator, baseline)
        main["__group"] = _group_key(main)
        added = main[curves.NO_DETECTOR] == 1
        for k in oracle_keys:
            if k in main.columns:
                main.loc[added, OCOL + k] = main.loc[added, k]
        main = curves.zero_empty_precision(main, OCOL + "precision", OCOL + "recall", OCOL + "fpr")

    shape = _Shape(main, arms, denominator, oracle_keys)
    if not shape.metrics:
        raise SystemExit("viewer: the frame carries none of the known metric columns")
    emb_labels = page_embedder_labels(shape.embedders, embedder_labels)
    offered = [m["key"] for m in shape.metrics]
    shown = [k for k in offered if k not in set(hide_metrics)]
    if default_metric is None and DEFAULT_METRIC not in shown and objective.carries_beta(main):
        if objective.OBJECTIVE in shown:
            default_metric = objective.OBJECTIVE
    # Checked before the expensive part, so a misspelt flag fails in a second.
    view = opening_view(offered, metric=default_metric, hide=hide_metrics)

    den = denominator if denominator is not None and not denominator.empty else main
    den = den.copy()
    if "embedder" not in den.columns:
        den["embedder"] = ""
    den["__group"] = _group_key(den)
    cells = {
        (str(arm), str(gk)): int(d.loc[:, ["seed"]].drop_duplicates().shape[0])
        for (arm, gk), d in den.groupby(["arm", "__group"])
    }

    arm_betas = {a: objective.frame_beta(main[main["arm"] == a]) for a in shape.arms}
    base = _baselines(baseline, shape, arm_betas)
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
        # 2 adds `agg.omean` / `agg.on` (the oracle companion) and `skyline`.
        # Nothing branches on it - the page treats both as optional, which is
        # what lets `--reskin` push a template improvement onto a schema-1 page
        # built before either existed.  It is here so a reader can tell which
        # reports still predate them.
        "schema": 2,
        "title": title,
        "subtitle": subtitle,
        "anchor": {"has": has_anchor, "label": anchor_label},
        "skyline": sky,
        "solid_coverage": curves.SOLID_COVERAGE,
        "datasets": shape.datasets,
        "embedders": shape.embedders,
        # What the page calls each embedder, where its key would mislead
        # (#4655).  Absent otherwise, and the page then shows the keys.
        **({"embedder_labels": emb_labels} if emb_labels else {}),
        "categories": shape.categories,
        "arms": shape.arms,
        # What the arm chips choose between, when it is not a configuration:
        # a review's session sets, one per balance (#4636).  Absent otherwise,
        # and the page then calls them arms.
        **({"arms_control": dict(arms_control)} if arms_control else {}),
        "seeds": shape.seeds,
        "metrics": shape.metrics,
        "groups": [list(g) for g in shape.groups],
        "t": [int(x) for x in t_full],
        "agg": agg,
        "runs": runs,
        "runs_note": runs_note,
        "n_cells": int(sum(cells.values())),
        "oracle_metrics": [k for k in oracle_keys if k not in RANKING_METRICS],
        # How this page was built, when the builder knew (the CLI does; an
        # analyzer calling `build_viewer` is itself in the tree, so its
        # invocation is already recoverable and it passes nothing).
        #
        # Rebuilding a committed page is a routine job -- it is the whole of
        # #3326 -- and until now the arguments were nowhere: `inclusion-knob-3196`
        # ships two chips called `linear` and `logistic` over two results
        # directories called `svm` and `linear`, and which carried which had to
        # be settled by building BOTH orders and diffing them against the
        # committed numbers.  A swap is invisible on screen and inverts the
        # study's finding.
        **({"build": build} if build else {}),
        # When every empty returned set was scored as one (no detector yet, or
        # one that flags nothing); absent on a page built before, where those
        # runs left the mean.
        # Ahead of `gaps_filled`, which a reskin inserts just before
        # `payload_kb`, so a built and a reskinned page agree key for key.
        **({"empty_sets_scored": _now()} if score_empty_sets else {}),
        # When the per-run gaps were carried (#4624); absent on a page that
        # was built without the carry and never reskinned with it.
        **({"gaps_filled": _now()} if fill_gaps else {}),
        "payload_kb": {k: round(v / 1024) for k, v in sizes.items()},
        # Last, because that is where `reskin` appends it: a view added to a
        # built page sits where a build would have put it.
        **({"view": view} if view else {}),
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


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fill_payload_gaps(payload: dict) -> dict:
    """Carry the per-seed lines through their gaps and re-average them (#4624), in place.

    The reskin counterpart of :func:`curves.fill_gaps`, for a committed page
    whose results directory is on the GRID or gone: the ``runs`` payload holds
    every run at every click, so the carry can be done on it and the averaged
    ``agg`` (``mean`` / ``sd`` / ``n``) recomputed from the result.  Click 0 is
    the text-sort anchor, not a detector, so it is neither a span's start nor
    recomputed; the oracle companion (``omean`` / ``on``) is not in the per-seed
    payload and is left as built.  The per-seed values are quantised to
    ``1/RUNS_SCALE``, so a re-averaged mean can differ from a rebuilt one by up
    to that much; on a [0, 1] metric that is a thousandth.

    Refuses a page whose per-seed lines were thinned to fit the byte budget:
    the carry needs every click, so such a page is rebuilt from its results.
    """
    runs = payload.get("runs")
    if not runs:
        raise SystemExit("no per-seed payload on this page: rebuild it from its results to fill the gaps")
    t = [int(x) for x in payload["t"]]
    if [int(x) for x in runs["t"]] != t:
        raise SystemExit(
            f"the per-seed lines are on a thinned grid ({len(runs['t'])} of {len(t)} clicks); "
            "rebuild the page from its results to fill the gaps"
        )
    vals = _decode(runs["values"])
    present = np.isfinite(vals).any(axis=1)
    anchor = t.index(0) if 0 in t else -1
    if anchor >= 0:
        present[:, anchor] = False
    filled = vals.copy()
    for r in range(vals.shape[0]):
        idx = np.flatnonzero(present[r])
        if len(idx) < 2:
            continue
        inside = np.arange(idx[0], idx[-1] + 1)
        gap = ~present[r, inside]
        if not gap.any():
            continue
        src = idx[np.searchsorted(idx, inside, side="right") - 1]
        filled[r][:, inside[gap]] = vals[r][:, src[gap]]

    agg = payload["agg"]
    mean, sd, n = _decode(agg["mean"]), _decode(agg["sd"]), _decode(agg["n"])
    new_mean, new_sd, new_n = np.full_like(mean, np.nan), np.full_like(sd, np.nan), np.zeros_like(n)
    index = np.asarray(runs["index"], dtype=int)
    for gi, ai in {(int(g), int(a)) for g, a, _s in index}:
        v = filled[(index[:, 0] == gi) & (index[:, 1] == ai)]
        good = np.isfinite(v)
        cnt = good.sum(axis=0).astype(float)
        z = np.where(good, v, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(cnt > 0, z.sum(axis=0) / np.maximum(cnt, 1), np.nan)
            var = np.where(cnt > 0, (z**2).sum(axis=0) / np.maximum(cnt, 1) - mu**2, np.nan)
        new_mean[gi, ai], new_sd[gi, ai], new_n[gi, ai] = mu, np.sqrt(np.clip(var, 0.0, None)), cnt
    if anchor >= 0:
        new_mean[..., anchor], new_sd[..., anchor], new_n[..., anchor] = (
            mean[..., anchor],
            sd[..., anchor],
            n[..., anchor],
        )
    agg["mean"], agg["sd"], agg["n"] = _encode(new_mean), _encode(new_sd), _encode(new_n, scale=1)
    runs["values"] = _encode(filled, scale=RUNS_SCALE)
    sizes = payload.get("payload_kb") or {}
    for k in ("mean", "sd", "n"):
        sizes[k] = round(_payload_bytes(agg[k]) / 1024)
    sizes["runs"] = round(_payload_bytes(runs["values"]) / 1024)
    payload["payload_kb"] = sizes
    # Lands where a build puts it: just before `payload_kb`.
    items = [(k, v) for k, v in payload.items() if k != "gaps_filled"]
    at = next(i for i, (k, _v) in enumerate(items) if k == "payload_kb")
    items.insert(at, ("gaps_filled", _now()))
    payload.clear()
    payload.update(items)
    return payload


def reskin(
    page: Path,
    template: Path = TEMPLATE,
    *,
    default_metric: str | None = None,
    hide_metrics: Sequence[str] | None = None,
    fill_gaps: bool = False,
    subtitle: str | None = None,
    embedder_labels: Mapping[str, str] | None = None,
) -> Path:
    """Re-substitute *page*'s own payload into the current template, in place.

    The template is the whole of the viewer's behaviour — every control, every
    drawing rule, every word of the reading note — and the payload is just the
    study's numbers.  So a change to how the page *reads* can be pushed onto a
    committed report without the results directory it was built from, which for
    a finished study lives on the GRID and may be gone.

    It cannot add data the payload never carried: a page rebuilt this way keeps
    whatever ``schema`` it had, so a study that measured no oracle cut and no
    skyline still shows neither, and the page disables those controls rather
    than drawing an empty one.  Re-running the analyzer is the only way to get
    them.

    *default_metric* and *hide_metrics* rewrite the payload's ``view`` block
    (:func:`opening_view`), each only when given: ``""`` and ``[]`` clear their
    half.  *fill_gaps* carries the per-seed lines through their gaps and
    re-averages them (:func:`fill_payload_gaps`, #4624).  *subtitle* replaces
    the page's subtitle and *embedder_labels* its ``embedder_labels`` block
    (:func:`page_embedder_labels`; an empty mapping drops it), each only when
    given: a label that said the wrong thing is a fix that needs no results
    directory (#4655).  With none of the five, the payload is copied byte for
    byte, ``view`` included, so a template push never undoes a study's choice.
    """
    page = Path(page)
    html = page.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        raise SystemExit(f"{page}: no payload script tag - not a viewer page")
    blob = m.group(1)
    relabel = subtitle is not None or embedder_labels is not None
    if default_metric is not None or hide_metrics is not None or fill_gaps or relabel:
        payload = json.loads(blob)
        if fill_gaps:
            fill_payload_gaps(payload)
        if subtitle is not None:
            payload["subtitle"] = subtitle
        if embedder_labels is not None:
            _with_embedder_labels(payload, page_embedder_labels(payload["embedders"], embedder_labels))
        if default_metric is not None or hide_metrics is not None:
            was = payload.pop("view", None) or {}
            view = opening_view(
                [spec["key"] for spec in payload["metrics"]],
                metric=was.get("metric") if default_metric is None else default_metric,
                hide=was.get("hide", []) if hide_metrics is None else hide_metrics,
            )
            if view:
                payload["view"] = view
        blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    fresh = template.read_text(encoding="utf-8")
    if fresh.count(TOKEN) != 1:
        raise SystemExit(f"{template}: expected exactly 1 {TOKEN}, found {fresh.count(TOKEN)}")
    page.write_text(fresh.replace(TOKEN, blob), encoding="utf-8")
    return page


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--reskin",
        nargs="+",
        metavar="PAGE",
        help="re-substitute each built viewer.html's own payload into the current template, in place",
    )
    ap.add_argument("--results", default=str(common.RESULTS), help="results root holding one dir per arm")
    ap.add_argument(
        "--arms",
        help="comma-separated arm directories, in report order; `dir=label` renames one on the page",
    )
    ap.add_argument(
        "--beta-run",
        action="append",
        metavar="BETA=DIR",
        help="a review's sessions at one balance (#4636): DIR is the run directory whose results/ holds the "
        "sessions run at BETA (SOTA_BETA). Repeat it per preset; each set becomes a chip named for its beta, "
        "the arms control reads as the sessions' beta, and a set whose rows carry another beta is refused. "
        "In place of --results / --arms.",
    )
    ap.add_argument("--out", help="path to write the HTML to")
    ap.add_argument("--baseline", default=None, help="text_baseline.py CSV: the click-0 anchor")
    ap.add_argument("--title", default="Quality over clicks")
    ap.add_argument("--subtitle", default=None, help="the line under the title; with --reskin, replaces the page's")
    ap.add_argument(
        "--embedder-label",
        action="append",
        metavar="KEY=LABEL",
        help="what the page calls the embedder KEY (#4655); repeat it per embedder. A label for an embedder the "
        "page does not carry is dropped, so one set serves every page of a study. With --reskin, replaces the "
        "page's labels",
    )
    ap.add_argument(
        "--default-metric",
        metavar="KEY",
        help=f"the metric the page opens on (default: {DEFAULT_METRIC}; without it, the objective, fbeta, on a "
        "run with a balance, else the first it offers); with --reskin, '' drops the page's own choice",
    )
    ap.add_argument(
        "--hide-metrics",
        metavar="KEYS",
        help="comma-separated metrics the page does not offer, though it still carries them; "
        "with --reskin, '' offers every one again",
    )
    ap.add_argument(
        "--fill-gaps",
        action="store_true",
        help="with --reskin, carry each run's last value through the clicks it has no row for (a spot check's "
        "rounds, #4624) and re-average the page from its per-seed lines; refused on a page whose per-seed "
        "lines were thinned. A build does this by itself.",
    )
    ap.add_argument("--runs-budget-mb", type=float, default=RUNS_BUDGET_MB)
    ap.add_argument(
        "--no-skyline",
        action="store_true",
        help="skip the supervised-skyline pass over the cell CSVs (issue #3322)",
    )
    ap.add_argument(
        "--skyline-results",
        default=None,
        metavar="ROOT",
        help="read the skyline rows from a SECOND results root over the same cells "
        "(default: --results), for a floor measured after the run it describes",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    hide = None if args.hide_metrics is None else [k for k in args.hide_metrics.replace(",", " ").split() if k]

    labels = None if args.embedder_label is None else parse_embedder_labels(args.embedder_label)
    if args.reskin:
        for page in args.reskin:
            out = reskin(
                Path(page),
                default_metric=args.default_metric,
                hide_metrics=hide,
                fill_gaps=args.fill_gaps,
                subtitle=args.subtitle,
                embedder_labels=labels,
            )
            print(f"reskinned {out}  ({out.stat().st_size / 1e6:.2f} MB)")
        return 0
    if args.fill_gaps:
        ap.error("--fill-gaps goes with --reskin; a build carries the gaps by itself")
    if args.beta_run and args.arms:
        ap.error("--beta-run names the arms itself; give it or --arms, not both")
    if not (args.arms or args.beta_run) or not args.out:
        ap.error("--arms (or --beta-run) and --out are required unless --reskin is given")
    baseline = curves.text_sort_baseline(args.baseline) if args.baseline else None
    common_kw = {
        "baseline": baseline,
        "title": args.title,
        "subtitle": args.subtitle or "",
        "runs_budget_mb": args.runs_budget_mb,
        "embedder_labels": labels,
        "default_metric": args.default_metric or None,
        "hide_metrics": hide or (),
    }
    if args.beta_run:
        if args.skyline_results:
            ap.error("--skyline-results goes with --results; each --beta-run directory carries its own skyline")
        runs = parse_beta_runs(args.beta_run)
        frame, skyline, arms = load_beta_runs(runs, skyline=not args.no_skyline)
        out = build_viewer(
            frame,
            Path(args.out),
            arms=arms,
            skyline=skyline,
            arms_control=BETA_ARMS_CONTROL,
            build={
                "beta_runs": [f"{b:g}={d.resolve()}" for b, d in runs],
                "baseline": str(Path(args.baseline).resolve()) if args.baseline else None,
                "built": _now(),
            },
            **common_kw,
        )
        _report(out, args, skyline, "each --beta-run directory's results/")
        return 0

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
    # The skyline may live in its own results root -- see `load_skyline`'s note
    # on why a floor is allowed to arrive later than the curve it sits beside.
    sky_root = Path(args.skyline_results or args.results)
    skyline = None if args.no_skyline else load_skyline(sky_root, dirs, arms)
    if args.skyline_results:
        print(f"skyline from {sky_root} ({0 if skyline is None else len(skyline)} rows)")
    out = build_viewer(
        frame,
        Path(args.out),
        arms=arms,
        skyline=skyline,
        build={
            "results": str(Path(args.results).resolve()),
            "arms": args.arms,
            "baseline": str(Path(args.baseline).resolve()) if args.baseline else None,
            "skyline_results": str(sky_root.resolve()) if args.skyline_results else None,
            "built": _now(),
        },
        **common_kw,
    )
    _report(out, args, skyline, "--skyline-results" if args.skyline_results else "--results")
    return 0


#: The repo's large-file cap (``check-added-large-files --maxkb=4000`` in
#: ``.pre-commit-config.yaml``), which a committed page has to fit under.
COMMIT_CAP_KB = 4000


def _report(out: Path, args: argparse.Namespace, skyline: pd.DataFrame | None, sky_from: str) -> None:
    """The build's closing lines: the page's size, and what it lacks and why."""
    size = out.stat().st_size
    print(f"wrote {out}  ({size / 1e6:.2f} MB)")
    # The hook's own arithmetic: KiB, rounded up.
    if -(-size // 1024) > COMMIT_CAP_KB:
        print(
            f"NOTE: the page is over the repo's {COMMIT_CAP_KB} KB large-file cap, so it cannot be "
            "committed as is. Rebuild with a smaller --runs-budget-mb: the per-seed lines thin to fit, and the "
            "averaged view keeps every click."
        )
    if not args.baseline:
        print("NOTE: no --baseline, so the page has no click-0 anchor and nothing to compare the far right against.")
    if args.no_skyline:
        print("NOTE: --no-skyline, so the cell CSVs were not read for a learnability floor and the page has none.")
    elif skyline is None or skyline.empty:
        print(
            f"NOTE: no supervised-skyline rows under {sky_from}, so the page has no "
            "learnability floor. Re-run with CALIB_SKYLINE_ARMS=skyline_train_full to get one, over "
            "these cells or over a subset of them passed as --skyline-results (issue #3322)."
        )


if __name__ == "__main__":
    raise SystemExit(main())

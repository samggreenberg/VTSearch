#!/usr/bin/env python3
"""State of the App (#4159): per-cell quality, and what each clicked image did.

Reads one State of the App run (``launch.sh``) and writes, under ``--out``:

* ``cells.csv`` -- one row per run (arm x cell x seed): the **ranking**, as
  average precision at click 0 (the text sort), at fixed clicks and at the end,
  and the full-label ceiling's (``skyline_train_full``); the **harvest**, Goods
  found by each checkpoint; and the **spot check** the run ends on: its verdict,
  its likely range and whether that range covered the truth.
* ``lines.csv`` -- the **line on a fresh corpus**, one row per run x point x
  floor: at each floor *P* the app offers (``_rank_metrics.FLOORS``), how right
  the set the line keeps on the test half is, how far short of *P*, whether it
  meets it, its recall next to the best recall any cut of the same ranking gets
  at *P*, and its **F1** next to the best F1 any cut gets.  Points run left to
  right: ``text`` (click 0), each checkpoint, ``final`` and ``ceiling``.
* ``line_steps.csv`` -- the same line at every click a rank frame was recorded
  (``CALIB_RANK_FRAME_STEPS``), one row per run x click x floor: what the F1
  curve is drawn from.
* ``curves.csv`` -- every run's AP, Goods found, and the returned set at each
  floor on a common click grid: its precision and recall and the oracle's
  recall at that floor (``precision_p50``, ``recall_p50``,
  ``oracle_recall_p50``, ... for each of 10/50/90%), and its F1 (``f1`` at the
  default floor, ``f1_p10``, ``f1_p90``).
* ``influence.csv`` -- one row per click: the change in held-out AP that the
  click is credited with.
* ``images.csv`` / ``image_detector.csv`` -- the influence rolled up per image,
  and per image x detector, early and late separately.
* ``harmful_pairs.csv`` -- the (image, class, label) pairs that hurt past
  -``Z_FLAG``, net of cell, label and phase: the hand-review list (#4179).
* ``stops.csv`` / ``margins.csv`` -- one row per run: where the app's stopping
  rules first fired (the click Autopilot says *All quality indicators are
  green*), the objective and AP there, at the budget and at the run's own
  best, and how far short each gate was over the clicks it held (#3560,
  ``calibration/stopping.py``).  A run that never trained is kept as one the
  rules never stopped.
* ``summary.md`` -- the tables a reader starts from.

**The metrics (owner, 2026-09-30, #4357).**  #4223 retired FPR + FNR as the
objective and #4272 made the default arm's line the floor's set: the top *K*
unvoted at the default floor.  A review scored in cost would read the change of
objective as a regression, so nothing here is cost: the ranking is AP, and the
line is read at each floor off the rank frames (``task_NNNN__rankframes.csv``,
``CALIB_RANK_FRAME_STEPS``).  **F1 is back, at the line (owner, 2026-09-30):**
"Showing AP is nice, but it's entirely about the ranking. Using the
returned-set threshold, show F1 over time, too."  It is the F1 of the set the
floor keeps on the test half (its top *K*), from the same rank frames -- never
the harness rows' ``f1``, which carries the session pool's line *score* over to
the test half instead of keeping its top *K*.  The precision floor is written
*P* (owner, 2026-09-30), not *X*.  A run recorded without rank frames still gets
every AP, harvest and check column; its line points other than ``text`` are
blank rather than guessed.

**Precision and recall at P (owner, 2026-10-01, #4408).**  F1 cannot see P: the
50% and 90% lines keep nearly the same set and get the same F1.  So the
returned set is scored at the floor it aimed for: its **precision against P**
(below P is a broken promise, far above P is recall left behind) and its
**recall against the oracle's recall at P** (the most a cut of the same ranking
returns while staying at or above P).  A run's sessions aim at one floor
(``session_floor``, the run's ``CALIB_MIN_PRECISION``); the other floors are
read off the same sessions, which is exact only while a session ignores P, so
a review runs one set of sessions per P and reads each at its own.

**The spot check is not a click.**  The default arm checks the line once the
voting steps are spent (``spot_check="end"``): its rounds are cast as votes and
emit ``phase == "check"`` rows past ``max_steps``.  They are read for the
check's own columns only.  "Final" is the last *ordinary* step, the curves stop
there, and no check pick is credited as a click.  The check's truth is the
share right of the set its range describes: the top ``check_k`` of the pool's
unvoted ranking at the last ordinary step (the ``last`` rank frame), which is
the candidate the check drew its picks from.  Under the advisory check that is
the set it audited (``check_audited``, #4427), not the set the line keeps.  What the check *should* certify
is #4358's ruling; until then this reports what it does certify.

**Every attempted run is in every average (owner, 2026-10-07, #4631).**  A run
that never trains a detector (150 clicks and no Good) writes no metric row, and
a mean over the runs with a value drops exactly the sessions the app failed.  So
every run is scored at every click as the user meets it: the typed query's own
set (the text sort at its line in the app) until the app shows the run's
detector, which for such a run is never (the owner's pick over the viewer's
empty set, whose 4th-decimal difference on these runs is the typed query's
F-beta of 0.006 to 0.03).  An **empty returned set** (a line that keeps nothing,
a detector that flags nothing) scores precision 0 beside its recall and F-beta
of 0, never an undefined precision that leaves the mean (``curves.EMPTY_SET``).

**The report is the session, not Test (#4643).**  The harness scores the test
split as Test would at each click: from the first Good, the Goods' centroid
until the labels hold 3 Goods and 4 Bads (the label quota), the trained head
from there (``detector_tier``).  The session shows neither until the Hard phase
(``app_trained``, #4605): Autopilot's opening is the text sort, so Test differs
from the session through the whole opening - the centroid before the quota, the
trained head between the quota and the Hard phase.  This report follows the
session there (:func:`_shown_from`): every user-facing number before the hand-off
is the typed query's set.  The viewer draws what Test gives instead
(``calibration/viewer.py``), so the two differ through the opening on purpose.

**How a click is credited (owner, 2026-09-23).** The harness scores the test
split after every click once a Good exists (a Good and a Bad, in results run
before #4643); before that there is no detector to score. So a scored step's change is split EQUALLY among every click
since the previous scored step, and the first scored step's change is measured
from the text-only score (click 0) and split among the opening clicks. That is
exact when every click is scored and still honest when some are not. A click
after the last scored step is credited nothing.

**What the credit means.** It is the marginal change along THIS trajectory --
the head is refit from scratch each click, so part of any single delta is refit
noise. It becomes a statement about an image only in aggregate, over the runs
and detectors that clicked it; ``images.csv`` carries ``n_obs`` so a reader can
see how much aggregate there is.

**Is an image's effect its own?** Early positives hurt on average (the first
detector is worse than the text sort) and some classes are harder than others,
so a raw per-image mean mostly says *when* and *where* the image was clicked.
``resid`` removes both: the click's credit minus the mean credit of every click
with the same run cell, label and phase. ``resid_z`` is that residual's mean
over the image's clicks in standard errors; ``summary.md`` compares the count
with |z| > ``Z_FLAG`` to the same count after shuffling images within each
(cell, label, phase) bucket, which is what chance gives.

    python analyze.py --exp /expscratch/$USER/state-of-the-app/<date> \\
        --baseline <exp>/text_baseline.csv --out <exp>/analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

import _cells_io  # noqa: E402
from _rank_metrics import (  # noqa: E402
    BETAS,
    FLOORS,
    balance_metrics,
    beta_tag,
    floor_tag,
    frame_beta_k,
    frame_k,
    line_metrics,
    parse_ranks,
)
import objective  # noqa: E402
import stopping  # noqa: E402

#: The two production paths, as the harness names them.
ARMS = {("siglip", "whole_image"): "SigLIP binary", ("siglip+dinov3_patch", "max_patch"): "DINOv3 region"}
#: The State of the App reports are one per production path (owner, 2026-09-24):
#: "Binary Photo" and "Region Photo" (and later e.g. "Document Logo").
PATHS = {"binary": ("siglip", "whole_image"), "region": ("siglip+dinov3_patch", "max_patch")}
#: A spot check's round of picks (``vtscore.training.thresholds.spot_check.CHECK_MIN_PICKS``, which a test pins):
#: how far past a checkpoint a run's next rank frame may be read when the run handed over inside a check round
#: and has no frame on screen by the checkpoint (#4631).
CHECK_ROUND = 5
#: Clicks at which a curve is sampled into ``cells.csv`` and ``lines.csv``.
#: ``launch.sh`` records rank frames at the same clicks.
CHECKPOINTS = (10, 25, 50, 100, 150)
#: What "early" and "late" mean for a click, in clicks.
EARLY, LATE = 30, 90
RUN_KEY = ["dataset", "category", "embedder", "style", "seed"]
#: The per-image test: an image needs this many clicks, and flags past this |z|.
MIN_OBS_Z, Z_FLAG = 10, 3.0
_BUCKET = ["arm", "category", "label", "when"]
#: The ceiling's arm, and the kind its rank frame carries.
CEILING = "skyline_train_full"
#: The line's metrics, as ``_rank_metrics.line_metrics`` names them.
LINE_METRICS = ("k", "precision", "shortfall", "meets", "recall", "oracle_recall", "f1", "oracle_f1")
#: The balance's metrics, as ``_rank_metrics.balance_metrics`` names them (#4413).
BALANCE_METRICS = ("k", "precision", "recall", "fbeta", "oracle_fbeta", "fb_share")
#: How a balance row was thresholded (owner, 2026-10-04: apples to apples - a comparison
#: between the text sort and the detector uses ONE rule on both sides).
#: ``app line``: what each sort's own line in the app returns - the text sort's blind GMM
#: cut, the detector's labels line (#4452), and for the full-label ceiling Find's labels
#: line drawn from its full labels (#4486; a run from before #4486 has none, and the row is
#: blank rather than read off the skyline's oracle cut on the test labels).
#: ``top-K``: set-constant, the balance's old cap (32 at beta <= 1, 128 above) on every sort;
#: shown for both sorts or neither, never as the text sort's stand-in for a line.
RULE_APP, RULE_TOP = "app line", "top-K"
RULES = (RULE_APP, RULE_TOP)


def balance_cap(beta: float) -> int:
    return 32 if beta <= 1.0 else 128


#: THE objective (owner, 2026-10-01): the withheld images above the app's threshold, as F-beta at the session's
#: beta.  Every session row carries the test set's precision / recall at that row's threshold; the last ordinary
#: row is the unchecked line, the last ``check`` row the line after the walk and its votes.  Unlike the rank-count
#: reading (``balances.csv``, the balance's rule re-drawn on the fresh ranking) this sees the threshold the app
#: actually holds, so a harvest that thins the user's unvoted top shows up here (#4427).
THRESHOLD_METRICS = ("thr_precision", "thr_recall", "thr_fbeta", "thr_returned")
#: A diagnostic, not the objective (#4427; owner 2026-10-01: the goal is the F-beta of what the line keeps,
#: scored on the withheld test half).  The user's own corpus: the line the app actually kept over the
#: session's UNVOTED pool at that click (``pool_k``, the harness's ``floor_count``), what it holds, and the
#: positives IN HAND - the Goods voted so far plus the positives inside that kept set.  It explains what the
#: withheld half cannot see: a harvesting acquisition strips the unvoted top, so the walk, which only sees
#: the user's pool, ends shallow (``docs/experiments/2026-10-01-acquisition-fbeta-4409``).
POOL_METRICS = (
    "pool_k",
    "pool_positives",
    "pool_tp",
    "pool_precision",
    "pool_recall",
    "pool_fbeta",
    "pool_oracle_fbeta",
    "pool_fb_share",
    "goods",
    "in_hand",
)
#: The floor a cell's F1 columns and the headline F1 curve are read at: the app's
#: default (``DEFAULT_MIN_PRECISION``), which is every session's floor until the
#: user moves it.
DEFAULT_FLOOR = 0.5
#: The floors ``curves.csv`` carries the returned set's curves for: every floor
#: the app offers (#4408).
CURVE_FLOORS = (0.5, 0.1, 0.9)
#: The returned set's metrics a curve is kept for, per floor.
CURVE_METRICS = ("f1", "precision", "recall", "oracle_recall")


#: The balance's curves: the returned set's F-beta and its share of the best cut, with precision and recall.
BALANCE_CURVE_METRICS = ("fbeta", "fb_share", "precision", "recall")

#: What the stopping block reads at the app's stop (#3560): the objective (a run that drew its line at a
#: balance), and AP beside it.  Never cost (#4357), so not ``stopping.DEFAULT_METRICS``.
STOP_METRICS = (objective.OBJECTIVE, "average_precision")


def curve_col(metric: str, floor: Any) -> str:
    """``curves.csv``'s column for *metric* at *floor* (a float) or at a balance ``("b", beta)``.

    ``f1`` alone is the default floor's F1, as it always was.
    """
    if isinstance(floor, tuple):
        return f"{metric}_{beta_tag(float(floor[1]))}"
    if metric == "f1" and floor == DEFAULT_FLOOR:
        return "f1"
    return f"{metric}_{floor_tag(floor)}"


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


#: The check's rows, or picks: the one rule ``_cells_io.load_arm`` sets them apart by.
_is_check = _cells_io.check_rows


def load(exp: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(base, skyline, picks, rank_frames)`` for every cell of the run."""
    cells = exp / "results" / "cells"
    frames, sky, picks, ranks = [], [], [], []
    for f in _cells_io.main_frame_files(cells):
        try:
            df = _cells_io.legacy_datasets(pd.read_csv(f, low_memory=False))
        except (pd.errors.EmptyDataError, OSError):
            continue
        if df.empty:
            continue
        if "gmm_variant" in df.columns:
            sky.append(df[df["gmm_variant"].astype(str) == CEILING])
        frames.append(_cells_io._base_rows(df))
    for p in _cells_io.side_frame_files(cells, "__picks"):
        if p.stat().st_size:
            picks.append(_cells_io.legacy_datasets(pd.read_csv(p)))
    for r in _cells_io.side_frame_files(cells, "__rankframes"):
        if r.stat().st_size:
            rf = pd.read_csv(r, dtype={"test_pos_ranks": str, "pool_pos_ranks": str})
            if not rf.empty:
                ranks.append(_cells_io.legacy_datasets(rf))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return cat(frames), cat(sky), cat(picks), cat(ranks)


def text_scores(baseline: Path | None) -> dict[tuple, dict[str, float]]:
    """``{(dataset, category, embedder, seed): {text_ap, text_<metric>_x<NN>...}}``.

    A baseline written before #4357 has no line columns; they come back NaN.
    """
    if baseline is None or not baseline.exists():
        return {}
    tb = _cells_io.legacy_datasets(pd.read_csv(baseline))
    tb = tb[tb.get("supports_text", 1) == 1]
    cols = [f"text_{m}_{floor_tag(x)}" for x in FLOORS for m in LINE_METRICS]
    cols += [f"text_{m}_{beta_tag(b)}" for b in BETAS for m in BALANCE_METRICS]  # the balance's (#4413)
    out = {}
    for r in tb.to_dict("records"):
        key = (r["dataset"], r["category"], r["embedder"], int(r["seed"]))
        out[key] = {
            "text_ap": _f(r.get("text_AP")),
            **{c: _f(r.get(c)) for c in cols},
            # The text sort's own line in the app (the blind GMM cut), as text_baseline.py scored it.
            "text_gmm_precision": _f(r.get("text_precision")),
            "text_gmm_recall": _f(r.get("text_recall")),
            "text_gmm_fpr": _f(r.get("text_fpr")),
            # The line the app draws at each preset (#4603), where the baseline records it.
            **{
                f"text_line_{m}_{beta_tag(b)}": _f(r.get(f"text_line_{m}_{beta_tag(b)}"))
                for b in BETAS
                for m in ("precision", "recall", "fpr")
            },  # fmt: skip
            "n_test": _f(r.get("n_test")),
            "n_test_pos": _f(r.get("n_test_pos")),
        }
    return out


def _text_for(ts: dict, ds: str, cat: str, emb: str, seed: int) -> dict[str, float]:
    # A paired region arm opens on its text half's sort, so its click-0 score is
    # the text embedder's when the pair has no row of its own.
    return ts.get((ds, cat, emb, seed)) or ts.get((ds, cat, emb.split("+")[0], seed)) or {"text_ap": np.nan}


def _text_line(text: dict[str, float], floor: float) -> dict[str, float]:
    return {m: text.get(f"text_{m}_{floor_tag(floor)}", np.nan) for m in LINE_METRICS}


def _text_balance(text: dict[str, float], beta: float) -> dict[str, float]:
    """The text sort's top-K (the cap) at *beta*, as ``text_baseline.py`` scored it."""
    return {m: text.get(f"text_{m}_{beta_tag(beta)}", np.nan) for m in BALANCE_METRICS}


def _pr_balance(
    precision: float, recall: float, fpr: float, n_pos: float, n_neg: float, beta: float, oracle: float
) -> dict[str, float]:
    """A returned set known by its precision, recall and false-positive rate: F-beta, its size, its share of *oracle*."""
    nan = float("nan")
    if not (np.isfinite(precision) or np.isfinite(recall)) or not np.isfinite(n_pos):
        return {m: nan for m in BALANCE_METRICS}
    recall = recall if np.isfinite(recall) else 0.0
    k = recall * n_pos + (fpr if np.isfinite(fpr) else 0.0) * n_neg
    f = _fbeta_pr(precision, recall, beta)
    return {
        "k": k,
        # A set that keeps nothing scores precision 0, not undefined (#4631, as `balance_metrics`).
        "precision": precision if k > 0 else (0.0 if n_pos > 0 else nan),
        "recall": recall,
        "fbeta": f,
        "oracle_fbeta": oracle,
        "fb_share": f / oracle if (np.isfinite(oracle) and oracle > 0) else nan,
    }


def _text_app_line(text: dict[str, float], beta: float) -> dict[str, float]:
    """The text sort's own line in the app (``text_sort_threshold`` at *beta*), scored at *beta*.

    Since #4603 the app draws a different line at each preset; a baseline that records it
    (``text_line_*_<beta>``) is read at it, and an older one at its single beta-blind line.
    """
    tag = beta_tag(beta)
    per_beta = np.isfinite(text.get(f"text_line_recall_{tag}", np.nan))
    pre = f"text_line_{{}}_{tag}" if per_beta else "text_gmm_{}"
    return _pr_balance(
        text.get(pre.format("precision"), np.nan),
        text.get(pre.format("recall"), np.nan),
        text.get(pre.format("fpr"), np.nan),
        text.get("n_test_pos", np.nan),
        text.get("n_test", np.nan) - text.get("n_test_pos", np.nan),
        beta,
        text.get(f"text_oracle_fbeta_{beta_tag(beta)}", np.nan),
    )


def _balance_at(frame: dict | None, beta: float, k: int | None = -1) -> dict[str, float]:
    """The returned set off a rank frame at *beta*: the frame's recorded line (*k* = -1), or the top *k*."""
    if frame is None:
        return {m: np.nan for m in BALANCE_METRICS}
    return balance_metrics(
        parse_ranks(frame["test_pos_ranks"]),
        int(frame["n_test"]),
        int(frame["n_test_pos"]),
        beta,
        frame_beta_k(frame, beta) if k == -1 else k,
    )


def _balance_rows(
    text: dict[str, float], frame: dict | None, use_text: bool, beta: float, sky_m: dict | None = None
) -> list[tuple[str, dict[str, float]]]:
    """``[(rule, metrics)]`` for one point at *beta*: the app's own line and the top-K cap."""
    if use_text:
        return [(RULE_APP, _text_app_line(text, beta)), (RULE_TOP, _text_balance(text, beta))]
    top = _balance_at(frame, beta, balance_cap(beta))
    if sky_m is not None and (frame is None or frame_beta_k(frame, beta) is None):
        # A ceiling frame from before #4486 records no line: the harness cut the skyline at
        # the retired oracle on the TEST labels, which is neither Train's threshold (Find
        # cannot see it) nor Find's line. The row carries only the ranking's best cut.
        app = {**{m: float("nan") for m in BALANCE_METRICS}, "oracle_fbeta": top["oracle_fbeta"]}
    else:
        # The detector's labels line, or the ceiling's Find line from its full labels (#4486).
        app = _balance_at(frame, beta)
    return [(RULE_APP, app), (RULE_TOP, top)]


def _goods_by(picks: pd.DataFrame, upto: float) -> int:
    """Goods found by click *upto* (the pick log's ``t`` counts the click itself)."""
    if picks is None or picks.empty:
        return 0
    return int(picks.loc[picks["t"] <= upto, "picked_label"].sum())


def _check_columns(check: pd.DataFrame | None, last_frame: dict | None, check_picks: pd.DataFrame | None) -> dict:
    """The spot check's verdict, range and truth, from its last row and the ``last`` rank frame."""
    nan = float("nan")
    if check is None or check.empty:
        return {"check_status": "", "check_votes": 0}
    end = check.sort_values("t").iloc[-1]
    # The range describes the set the walk audited (#4427): under the advisory
    # check (every balance since #4452) that is not the set the line keeps, so
    # the truth is read on ``check_audited`` when the row records one. A floor
    # walk records none; its end is the kept set (``floor_count``).
    audited = _f(end.get("check_audited"))
    kept = _f(end.get("floor_count"))
    k = int(audited) if np.isfinite(audited) and audited >= 1 else (int(kept) if np.isfinite(kept) else -1)
    lo, hi = _f(end.get("range_lo")), _f(end.get("range_hi"))
    truth = nan
    if last_frame is not None and k > 0 and int(last_frame["n_pool"]) > 0:
        # The candidate is the top K unvoted at the check's start; a pool with
        # fewer unvoted items is its own candidate.
        k_eff = min(k, int(last_frame["n_pool"]))
        truth = float(np.count_nonzero(parse_ranks(last_frame["pool_pos_ranks"]) < k_eff) / k_eff)
    covered = nan
    if np.isfinite(truth) and np.isfinite(lo) and np.isfinite(hi):
        covered = float(lo - 1e-9 <= truth <= hi + 1e-9)
    return {
        "check_status": str(end.get("floor_status", "")),
        "check_floor": _f(end.get("min_precision")),
        "check_k": k,
        "check_votes": int(len(check_picks)) if check_picks is not None else 0,
        "check_labelled": int(_f(end.get("check_labelled"))) if np.isfinite(_f(end.get("check_labelled"))) else -1,
        "check_right": int(_f(end.get("check_right"))) if np.isfinite(_f(end.get("check_right"))) else -1,
        "range_lo": lo,
        "range_hi": hi,
        "check_truth": truth,
        "check_covered": covered,
    }


def _fbeta_pr(precision: float, recall: float, beta: float) -> float:
    """F-beta from precision and recall; an empty returned set (no precision, recall 0) scores 0, not NaN.

    A line that keeps nothing of a corpus that holds positives found none of
    them: its F-beta is 0.  Reading it as NaN dropped the cell from every
    mean, which flattered whichever line kept nothing most often (#4452: 18
    cells of 702 at beta 1 for the count line, 27 for the labels line).
    """
    if np.isfinite(recall) and not np.isfinite(precision) and recall == 0:
        return 0.0
    if not (np.isfinite(precision) and np.isfinite(recall)):
        return float("nan")
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) > 0 else 0.0


def _threshold_at(row: pd.Series | None, beta: float) -> dict[str, float]:
    """The withheld set above *row*'s threshold (#4427): its precision, recall, F-beta and size.

    A detector that flags nothing has its precision left undefined by the harness; it is an empty
    returned set, so it scores precision 0 (#4631), by ``curves.zero_empty_precision``'s rule: an
    undefined precision with recall 0 and, where recorded, FPR 0.
    """
    nan = float("nan")
    if row is None:
        return {m: nan for m in THRESHOLD_METRICS}
    p, r = _f(row.get("precision")), _f(row.get("recall"))
    n_pos, n_neg, fpr = _f(row.get("n_test_pos")), _f(row.get("n_test_neg")), _f(row.get("fpr"))
    if not np.isfinite(p) and r == 0 and (fpr == 0 or not np.isfinite(fpr)):
        p = 0.0
    returned = r * n_pos + fpr * n_neg if np.isfinite(r * n_pos + fpr * n_neg) else nan
    return {"thr_precision": p, "thr_recall": r, "thr_fbeta": _fbeta_pr(p, r, beta), "thr_returned": returned}


def _pool_at(frame: dict | None, count: float, goods: int, beta: float) -> dict[str, float]:
    """The line over the session's unvoted pool at *count* (the app's own kept set), and the positives in hand (#4427)."""
    nan = float("nan")
    if frame is None or not np.isfinite(count) or count < 1 or int(frame.get("n_pool", -1)) <= 0:
        return {**{m: nan for m in POOL_METRICS}, "goods": goods, "in_hand": nan}
    n_pool, n_pos = int(frame["n_pool"]), int(frame["n_pool_pos"])
    ranks = parse_ranks(frame["pool_pos_ranks"])
    m = balance_metrics(ranks, n_pool, n_pos, beta, int(count))
    tp = int(round(m["precision"] * m["k"])) if np.isfinite(m["precision"]) else 0
    return {
        "pool_k": m["k"],
        "pool_positives": n_pos,
        "pool_tp": tp,
        "pool_precision": m["precision"],
        "pool_recall": m["recall"],
        "pool_fbeta": m["fbeta"],
        "pool_oracle_fbeta": m["oracle_fbeta"],
        "pool_fb_share": m["fb_share"],
        "goods": goods,
        "in_hand": goods + tp,
    }


def _counts_by_t(ordinary: pd.DataFrame | None, check: pd.DataFrame | None) -> tuple[dict[int, float], float]:
    """``(the line's kept count at each ordinary click, the count at the end)``: the harness's ``floor_count``.

    The end is the check's end when the session ran one (the set the app keeps
    once the walk has finished, over the ranking the ``last`` frame recorded),
    else the last ordinary click's.
    """
    out: dict[int, float] = {}
    if ordinary is not None and not ordinary.empty and "floor_count" in ordinary:
        for t, g in ordinary.groupby("t"):
            v = pd.to_numeric(g["floor_count"], errors="coerce").dropna()
            if len(v) and v.iloc[-1] >= 1:
                out[int(t)] = float(v.iloc[-1])
    end = out[max(out)] if out else float("nan")
    if check is not None and not check.empty and "floor_count" in check:
        v = pd.to_numeric(check.sort_values("t")["floor_count"], errors="coerce").dropna()
        if len(v) and v.iloc[-1] >= 1:
            end = float(v.iloc[-1])
    return out, end


def _frame_dict(r: pd.Series | None) -> dict | None:
    return None if r is None else r.to_dict()


def _run_frames(frames: pd.DataFrame) -> dict[tuple, dict[str, pd.DataFrame]]:
    """Per run key, its rank frames by kind."""
    if frames.empty:
        return {}
    out: dict[tuple, dict[str, pd.DataFrame]] = {}
    for key, g in frames.groupby(RUN_KEY):
        out[tuple(key)] = {k: kg for k, kg in g.groupby("kind")}
    return out


def _shown_from(ordinary: pd.DataFrame) -> dict[tuple, float]:
    """Per run key, the first click at which the app shows a detector (#4605); ``inf`` when it never does.

    The app stays on the text sort through Autopilot's opening (Good, Bad, More) and sorts by the
    detector only from the Hard phase on. The harness marks the steps the app shows as
    ``app_trained == 1`` (:func:`vtscore.eval.autopilot_flow.app_has_detector`): a detector it scores
    earlier - the Goods' centroid under the label quota, or a trained head inside the opening
    (#4643) - is what Test would give there, but not what the session shows. So every user-facing
    number before this click is the text sort's.
    Rows without the flag (recorded before it existed, or a fixture) are shown from the first row.
    """
    out: dict[tuple, float] = {}
    for key, g in ordinary.groupby(RUN_KEY):
        flag = pd.to_numeric(g["app_trained"], errors="coerce") if "app_trained" in g.columns else None
        shown = g.loc[flag == 1, "t"] if flag is not None and flag.notna().any() else g["t"]
        out[tuple(key)] = float(shown.min()) if len(shown) else float("inf")
    return out


def _app_shows(row: pd.Series, default: bool) -> bool:
    """Whether the app shows the detector of *row* (``app_trained == 1``); *default* for a row without the flag."""
    flag = pd.to_numeric(pd.Series([row.get("app_trained")]), errors="coerce").iloc[0]
    return default if pd.isna(flag) else bool(flag == 1)


def _line_at(frame: dict | None, floor: float) -> dict[str, float]:
    if frame is None:
        return {m: np.nan for m in LINE_METRICS}
    return line_metrics(
        parse_ranks(frame["test_pos_ranks"]),
        int(frame["n_test"]),
        int(frame["n_test_pos"]),
        floor,
        frame_k(frame, floor),
    )


def run_tables(
    base: pd.DataFrame, sky: pd.DataFrame, ts: dict, picks: pd.DataFrame, frames: pd.DataFrame
) -> tuple[pd.DataFrame, ...]:
    """``(cells, lines, steps, balances, balance_steps, pools, pool_steps, thresholds)``: one row per run, one
    per run x point x floor, one per run x recorded click x floor (every ``step`` rank frame, for the F1
    curve), the same two per balance (#4413: the returned set's F-beta over the best cut, at each preset
    beta), the same two over the session's own unvoted pool at the line the app kept (#4427: the positives in
    hand), and the objective: the withheld set above the app's threshold at every click, unchecked and after
    the check (``thresholds``, :data:`THRESHOLD_METRICS`).

    A run that never found a positive has no scored steps (the head cannot
    train without a Good), and it is the review's most important row, so it is
    listed from its picks with the text sort as its score at every click rather
    than dropped.  In a two-pass run (``launch.sh`` ``SOTA_PASS``) such a run is
    known only from its ceiling row: the clicks pass writes a header-only file.
    """
    have_frames = not frames.empty
    by_frames = _run_frames(frames)
    if base.empty:
        base = pd.DataFrame(columns=[*RUN_KEY, "t", "phase", "average_precision"])
    ordinary = base[~_is_check(base)]
    checks = {tuple(k): g for k, g in base[_is_check(base)].groupby(RUN_KEY)}
    ordinary_by = {tuple(k): g for k, g in ordinary.groupby(RUN_KEY)}
    pk_all = picks if not picks.empty else pd.DataFrame(columns=[*RUN_KEY, "t", "picked_label", "phase"])
    pk_click = pk_all[~_is_check(pk_all)]
    pk_check = pk_all[_is_check(pk_all)]
    clicks_by = {tuple(k): g for k, g in pk_click.groupby(RUN_KEY)}
    check_picks_by = {tuple(k): g for k, g in pk_check.groupby(RUN_KEY)}
    series = {tuple(k): g.groupby("t")["average_precision"].mean() for k, g in ordinary.groupby(RUN_KEY)}
    shown_from = _shown_from(ordinary)
    # The floor each run's sessions aimed at (the run's CALIB_MIN_PRECISION), off its own rows.
    floor_of: dict[tuple, float] = {}
    beta_of: dict[tuple, float] = {}
    if "min_precision" in ordinary:
        for k, g in ordinary.groupby(RUN_KEY):
            mp = pd.to_numeric(g["min_precision"], errors="coerce").dropna()
            if len(mp):
                floor_of[tuple(k)] = float(mp.iloc[0])
    if "beta" in ordinary:  # the balance arm (#4413): NaN on a floor arm
        for k, g in ordinary.groupby(RUN_KEY):
            b = pd.to_numeric(g["beta"], errors="coerce").dropna()
            if len(b):
                beta_of[tuple(k)] = float(b.iloc[0])
    # A run that never trained writes no row to read its beta off, and a review runs one set of sessions
    # per beta (`SOTA_BETA`), so it takes the beta every other run carries; F1 (beta 1) when they disagree
    # or carry none, as for a floor-era run.  Read at beta 1, a beta-4 review's starved runs would score
    # the typed query at the wrong preset (#4631).
    study_betas = set(beta_of.values())
    study_beta = study_betas.pop() if len(study_betas) == 1 else float("nan")
    skyd: dict[tuple, float] = {}
    skym: dict[tuple, dict] = {}
    for r in sky.to_dict("records") if not sky.empty else []:
        k_ = (r["dataset"], r["category"], r["embedder"], r["style"], int(r["seed"]))
        skyd[k_] = _f(r.get("average_precision"))
        skym[k_] = {c: _f(r.get(c)) for c in ("precision", "recall", "fpr", "n_test_pos", "n_test_neg")}

    keys = set(series) | set(clicks_by) | set(skyd)
    cells, lines, steps_out, balances, balance_steps, pools, pool_steps, thresholds = [], [], [], [], [], [], [], []
    for key in sorted(keys, key=lambda k: tuple(str(x) for x in k)):
        ds, cat, emb, style, seed = key
        text = _text_for(ts, ds, cat, emb, int(seed))
        s = series.get(key)
        trained = s is not None and len(s) > 0
        clicks = clicks_by.get(key)
        kinds = by_frames.get(key, {})
        last = kinds.get("last")
        last_frame = _frame_dict(last.iloc[-1]) if last is not None and len(last) else None
        row: dict = {
            "arm": ARMS.get((emb, style), f"{emb}/{style}"),
            "dataset": ds,
            "category": cat,
            "class": cat.split("@")[0],
            "band": cat.split("@")[1] if "@" in cat else "",
            "seed": int(seed),
            "never_trained": not trained,
            "session_floor": floor_of.get(key, float("nan")),
            "session_beta": beta_of.get(key, study_beta),
            "text_ap": text["text_ap"],
        }
        final_t = int(s.index.max()) if trained else (int(clicks["t"].max()) if clicks is not None else 0)
        # The first click the app shows a detector: the end of Autopilot's opening (#4605).
        shown = shown_from.get(key, float("inf"))
        row["shown_from"] = shown
        for c in CHECKPOINTS:
            at = s[(s.index <= c) & (s.index >= shown)] if trained else None
            # Before the app shows a detector, what the user sees IS the text sort
            # (the opening's own detectors are on no screen, #4605), so the text
            # score stands in -- never a gap, which would let the mean at a
            # checkpoint silently skip the runs that are still starving.
            row[f"ap_{c}"] = at.iloc[-1] if at is not None and len(at) else text["text_ap"]
            row[f"goods_{c}"] = _goods_by(clicks, c)
        row["final_t"] = final_t
        row["final_ap"] = s.iloc[-1] if trained and s.index.max() >= shown else text["text_ap"]
        row["positives_found"] = _goods_by(clicks, final_t)
        row["ceiling_ap"] = skyd.get(key, np.nan)
        row["clicks_bought"] = row["final_ap"] - row["text_ap"]
        row["headroom"] = row["ceiling_ap"] - row["final_ap"]
        row.update(_check_columns(checks.get(key), last_frame, check_picks_by.get(key)))
        # The session's own preference, for the pool-side line (#4427): its beta, or F1 under a floor.
        own_beta = beta_of.get(key, study_beta if np.isfinite(study_beta) else 1.0)
        counts, end_count = _counts_by_t(ordinary_by.get(key), checks.get(key))
        # The objective (#4427): the withheld set above the app's threshold, from the session's own rows.
        ord_rows = ordinary_by.get(key)
        ord_rows = ord_rows.sort_values("t") if ord_rows is not None else None
        chk_rows = checks.get(key)
        last_ord = ord_rows.iloc[-1] if ord_rows is not None and len(ord_rows) else None
        last_chk = chk_rows.sort_values("t").iloc[-1] if chk_rows is not None and len(chk_rows) else None
        # Before the app shows a detector the user has the typed query's own set: the text sort at its line.
        text_line = _text_app_line(text, own_beta)
        text_thr = {f"thr_{m}": text_line[k] for m, k in (("precision", "precision"), ("recall", "recall"),
                                                          ("fbeta", "fbeta"), ("returned", "k"))}  # fmt: skip
        # The objective at click 0, and wherever the app shows no detector: the curves start from it.
        for m in THRESHOLD_METRICS:
            row[f"text_{m}"] = text_thr[m]
        # A run with no detector on screen at its last click has the typed query's set there: one still in the
        # opening, or one that never trained (#4631: in every average, at what its session shows).
        last_shown = last_ord is not None and int(last_ord["t"]) >= shown
        unchecked = _threshold_at(last_ord, own_beta) if last_shown else text_thr
        # The check moves the set only when the app shows the detector it checked. A run still in the opening
        # checks one no screen shows - since #4643 the Goods' centroid cut at its midpoint, half the corpus - so
        # it keeps the typed query's set after the check, and the check's effect is 0 (#4631, #4653).
        chk_shown = last_chk is not None and _app_shows(last_chk, last_shown)
        after = _threshold_at(last_chk, own_beta) if chk_shown else unchecked
        for c in CHECKPOINTS:
            at = ord_rows[(ord_rows["t"] <= c) & (ord_rows["t"] >= shown)] if ord_rows is not None else None
            point = _threshold_at(at.iloc[-1], own_beta) if at is not None and len(at) else text_thr
            # The returned set's path through the session (#4519): F-beta, and the
            # precision, recall and size behind it, at each checkpoint.
            for m in ("fbeta", "precision", "recall", "returned"):
                row[f"thr_{m}_{c}"] = point[f"thr_{m}"]
        row["thr_fbeta_unchecked"] = unchecked["thr_fbeta"]
        row["thr_returned_unchecked"] = unchecked["thr_returned"]
        row["thr_fbeta_final"] = after["thr_fbeta"]
        row["thr_precision_final"] = after["thr_precision"]
        row["thr_recall_final"] = after["thr_recall"]
        row["thr_returned_final"] = after["thr_returned"]
        row["thr_walk_effect"] = after["thr_fbeta"] - unchecked["thr_fbeta"]
        cells.append(row)

        # The line, left to right.  Without rank frames only click 0 is known.
        steps = kinds.get("step")
        step_at = {int(r["t"]): r for r in steps.to_dict("records")} if steps is not None else {}
        sky_frame = kinds.get(CEILING)
        first_t = int(shown) if trained and np.isfinite(shown) else None
        # (point, t, frame, read the text sort instead).  A point with neither is
        # unknown and stays blank rather than borrowing a later value.
        points: list[tuple[str, float, dict | None, bool]] = [("text", 0, None, True)]
        for c in CHECKPOINTS:
            on_screen_by_c = [t for t in step_at if first_t is not None and first_t <= t <= c]
            ahead = [t for t in step_at if first_t is not None and first_t <= t and c < t <= c + CHECK_ROUND]
            if not have_frames:
                points.append((str(c), c, None, False))
            elif c in step_at and first_t is not None and c >= first_t:
                points.append((str(c), c, step_at[c], False))
            elif last_frame is not None and int(last_frame["t"]) <= c and first_t is not None and c >= first_t:
                # The pool ran out before click c: the line stays where it ended.
                points.append((str(c), c, last_frame, False))
            elif on_screen_by_c:
                # No frame at click c itself: the run is inside a spot check, which scores it once per round of
                # picks, and the user has its last detector all the while (#4624).  Blank, the point dropped
                # exactly the weak sessions the check prompts in from the mean (#4631: 157 of 1,440 at click 50).
                points.append((str(c), c, step_at[max(on_screen_by_c)], False))
            elif first_t is not None and c >= first_t and ahead:
                # Handed over a click or two before c and straight into a check round, so no frame is on screen
                # yet (frames are sparse: every 5 clicks, and only at the checkpoints on the region path).  Its next
                # frame, at most a round ahead, is the nearest reading of the detector the user has; the last one
                # before the hand-over can be 50 clicks stale and was on no screen (#4605).
                points.append((str(c), c, step_at[min(ahead)], False))
            else:
                # No detector on screen yet at click c: the user still has the text sort.
                points.append((str(c), c, None, first_t is None or c < first_t))
        final_text = have_frames and (not trained or first_t is None)
        points.append(("final", final_t, last_frame if have_frames and not final_text else None, final_text))
        points.append(("ceiling", np.nan, _frame_dict(sky_frame.iloc[-1]) if sky_frame is not None else None, False))
        ident = {k: row[k] for k in ("arm", "dataset", "category", "class", "band", "seed", "never_trained")}
        at_default: dict[str, dict[str, float]] = {}
        in_hand_at: dict[str, dict[str, float]] = {}
        for point, t, frame, use_text in points:
            for x in FLOORS:
                m = _text_line(text, x) if use_text else _line_at(frame, x)
                lines.append({**ident, "point": point, "t": t, "floor": x, **m})
                if x == DEFAULT_FLOOR:
                    at_default[point] = m
            for b in BETAS:  # the returned set at each balance (#4413), under each rule (apples to apples)
                for rule, mb in _balance_rows(text, frame, use_text, b, skym.get(key) if point == "ceiling" else None):
                    balances.append({**ident, "point": point, "t": t, "beta": b, "rule": rule, **mb})
            # The user's own corpus at that click (#4427): the ceiling has no session, the text sort no line.
            if point != "ceiling":
                count = (
                    float("nan") if use_text else (end_count if point == "final" else counts.get(int(t), float("nan")))
                )
                mp = _pool_at(None if use_text else frame, count, _goods_by(clicks, t), own_beta)
                pools.append({**ident, "point": point, "t": t, "beta": own_beta, **mp})
                in_hand_at[point] = mp
        for c in CHECKPOINTS:
            row[f"in_hand_{c}"] = in_hand_at[str(c)]["in_hand"]
        row["final_in_hand"] = in_hand_at["final"]["in_hand"]
        row["final_pool_k"] = in_hand_at["final"]["pool_k"]
        row["final_pool_precision"] = in_hand_at["final"]["pool_precision"]
        row["final_pool_positives"] = in_hand_at["final"]["pool_positives"]
        # The returned set's F1 at the default floor, beside AP (owner, 2026-09-30).
        row["text_f1"] = at_default["text"]["f1"]
        for c in CHECKPOINTS:
            row[f"f1_{c}"] = at_default[str(c)]["f1"]
        row["final_f1"] = at_default["final"]["f1"]
        row["final_oracle_f1"] = at_default["final"]["oracle_f1"]
        row["ceiling_f1"] = at_default["ceiling"]["f1"]
        # The line at every click a frame was recorded, for the F1 curve.
        for fr in steps.to_dict("records") if steps is not None else []:
            ranks = parse_ranks(fr["test_pos_ranks"])
            for x in FLOORS:
                m = line_metrics(ranks, int(fr["n_test"]), int(fr["n_test_pos"]), x, frame_k(fr, x))
                steps_out.append({**ident, "t": int(fr["t"]), "shown": int(fr["t"]) >= shown, "floor": x, **m})
            for b in BETAS:
                for rule, kk in ((RULE_APP, frame_beta_k(fr, b)), (RULE_TOP, balance_cap(b))):
                    mb = balance_metrics(ranks, int(fr["n_test"]), int(fr["n_test_pos"]), b, kk)
                    balance_steps.append(
                        {**ident, "t": int(fr["t"]), "shown": int(fr["t"]) >= shown, "beta": b, "rule": rule, **mb}
                    )
            t_fr = int(fr["t"])
            mp = _pool_at(fr, counts.get(t_fr, float("nan")), _goods_by(clicks, t_fr), own_beta)
            pool_steps.append({**ident, "t": t_fr, "shown": t_fr >= shown, "beta": own_beta, **mp})
        # The withheld set above the threshold at every ordinary click, then unchecked and after the check.
        if ord_rows is not None:
            for r_ in ord_rows.to_dict("records"):
                thresholds.append(
                    {
                        **ident,
                        "point": "step",
                        "t": int(r_["t"]),
                        "shown": int(r_["t"]) >= shown,
                        "beta": own_beta,
                        **_threshold_at(pd.Series(r_), own_beta),
                    }
                )
            thresholds.append({**ident, "point": "unchecked", "t": int(last_ord["t"]), "beta": own_beta, **unchecked})
            thresholds.append({**ident, "point": "final", "t": final_t, "beta": own_beta, **after})
        else:  # never trained: its ends are the typed query's set (#4631)
            thresholds.append({**ident, "point": "unchecked", "t": final_t, "beta": own_beta, **unchecked})
            thresholds.append({**ident, "point": "final", "t": final_t, "beta": own_beta, **after})
    return (
        pd.DataFrame(cells),
        pd.DataFrame(lines),
        pd.DataFrame(steps_out),
        pd.DataFrame(balances),
        pd.DataFrame(balance_steps),
        pd.DataFrame(pools),
        pd.DataFrame(pool_steps),
        pd.DataFrame(thresholds),
    )


def _band(category: pd.Series) -> pd.Series:
    """The size half of a ``class@band`` category; blank where there is none."""
    return category.astype(str).str.split("@").str[1].fillna("")


def stopping_tables(base: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(stops, margins)``: per run, where the app's stopping rules fired and how close each gate came (#3560).

    The app announces "All quality indicators are green" the first time Smart,
    Stable and Span are all green, and a user may stop there; the session keeps
    clicking to the budget either way.  ``stopping.stopping_points`` reads that
    stop and the objective there off the ordinary clicks (the end-of-run check's
    rows are not clicks), and ``stopping.margins`` how far short each gate was
    over the clicks it held.

    **A run that never trained a detector is kept.**  It has no row to read, but
    it is a session the rules never stopped, so it enters the fire rate as a run
    that did not fire, censored at its last click.  Dropping it would raise the
    fire rate by exactly the runs the review most needs to see.
    """
    stops = mg = pd.DataFrame()
    ordinary = base[~_is_check(base)] if not base.empty and "phase" in base.columns else pd.DataFrame()
    if not ordinary.empty:
        arm = [ARMS.get((e, s), f"{e}/{s}") for e, s in zip(ordinary["embedder"], ordinary["style"], strict=True)]
        ordinary = ordinary.assign(arm=arm)
        keys = ("arm", *RUN_KEY)
        stops = stopping.stopping_points(ordinary, keys=keys, metrics=STOP_METRICS)
        mg = stopping.margins(ordinary, keys=keys)
    starved = cells[cells["never_trained"].astype(bool)] if not cells.empty else cells
    if not starved.empty:
        rows = []
        for r in starved.to_dict("records"):
            emb, style = _ARM_OF.get(r["arm"]) or tuple(str(r["arm"]).split("/", 1))
            rows.append(
                {
                    "arm": r["arm"],
                    "dataset": r["dataset"],
                    "category": r["category"],
                    "embedder": emb,
                    "style": style,
                    "seed": int(r["seed"]),
                    "n_steps": 0,
                    "t_budget": int(r["final_t"]),
                    "stopped": False,
                    "t_stop": np.nan,
                    "t_sustained": np.nan,
                    "n_done_episodes": 0,
                    "clicks_after_stop": np.nan,
                }
            )
        stops = pd.concat([stops, pd.DataFrame(rows)], ignore_index=True) if not stops.empty else pd.DataFrame(rows)
    if not stops.empty:
        stops["stopped"] = stops["stopped"].astype(bool)
    for df in (stops, mg):
        if not df.empty:
            df["band"] = _band(df["category"])
    return stops, mg


def stopping_md(stops: pd.DataFrame, mg: pd.DataFrame) -> list[str]:
    """The summary's stopping block: where the app said stop, what the line scored there, and what held it."""
    if stops.empty:
        return []
    # The objective where the sessions drew their line at a balance, else AP; decided on the runs, not on
    # whether any of them fired, so a review where nothing fired still names the objective.
    obj_at = f"{objective.OBJECTIVE}_at_stop"
    metric = objective.OBJECTIVE if obj_at in stops.columns else "average_precision"
    by_arm = stopping.summarise(stops, by=("arm",), metrics=STOP_METRICS)
    out = [
        "## Where the app said stop (#3560)",
        "",
        "The app announces *All quality indicators are green* the first time Smart, Stable and Span are all "
        "green, and a user may stop there; the session keeps clicking to the budget either way. A run's **stop** "
        "is that first click. `fired` counts the runs the rules stopped at all, and every column after it "
        "describes only those, so read it first. The KM median carries the runs that never fired as censored "
        "at their last click, and is blank when fewer than half fired. "
        f"`{metric} at stop` and `at budget` are the line's {metric} at the stop and at the last click, Δ "
        "paired within run; `short of run's best` is how far the stop fell below the best the run ever "
        "reached, and `clicks past best` how many clicks after that best it fired (negative: before it). "
        "The best is the top of a noisy series, so read those two together.",
        "",
        stopping.stopping_table(by_arm, metric=metric),
        "",
    ]
    note = stopping.binding_note(by_arm)
    if note:
        out += [note, "", "Per band:", ""]
    else:
        out += ["Per band:", ""]
    out += [stopping.stopping_table(stopping.summarise(stops, by=("arm", "band"), metrics=STOP_METRICS), metric=metric)]
    out += [
        "",
        "### How close each gate came, over the clicks it held",
        "",
        "The median margin to green over each run's held clicks (before its stop, or all of them when it never "
        "stopped): positive is satisfied with that much room, negative is short by that much. In brackets, the "
        "share of held clicks the gate was green at: a margin just under zero at about half is a rule flapping, "
        "at 0% a wall. Smart's two gates are an either-or.",
        "",
        stopping.margin_table(stopping.summarise_margins(mg, by=("arm",))),
        "",
    ]
    return out


def _on_screen(d: pd.DataFrame | None) -> pd.DataFrame | None:
    """A per-click table's rows at clicks the app shows a detector (``shown``, #4605); all rows without the flag."""
    if d is None or d.empty or "shown" not in d.columns:
        return d
    return d[d["shown"].fillna(True).astype(bool)]


def curves(
    cells: pd.DataFrame,
    base: pd.DataFrame,
    picks: pd.DataFrame,
    lines: pd.DataFrame | None = None,
    steps: pd.DataFrame | None = None,
    balances: pd.DataFrame | None = None,
    balance_steps: pd.DataFrame | None = None,
    thresholds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every run on a common click grid 0..horizon, as the USER would see it.

    Click 0 is the text-only AP; until the app shows a detector -- the end of
    Autopilot's opening, not the first scored click (#4605, :func:`_shown_from`)
    -- the user still sees the text sort, so it carries forward; after that the
    last scored value carries forward. A run that never trained stays at its text AP.  Goods found
    count the clicks only, never the spot check's picks.

    The returned set at each floor (precision, recall, the oracle's recall at
    that floor and F1; :func:`curve_col` names the columns) follows the
    same rule off the ``step`` rank frames: the text sort's until the first
    frame on screen, then the last frame's.  Between two recorded frames that is
    a carried value, not a measurement, so a figure should read the curve only at
    the clicks ``line_steps.csv`` holds.  A trained run with no frames at all (a
    run recorded without them) is blank past click 0; one whose frames all fall
    in the opening (it never handed over) reads the text sort throughout.

    The objective (``thr_*``) starts at the run's typed query (``text_thr_*``,
    the text sort's own set at the session's beta) and holds it until the app
    shows the run's detector, then carries its line (#4631).  So every run is
    in the mean at every click, the ones that never trained included.
    """
    ordinary = base[~_is_check(base)] if not base.empty else pd.DataFrame(columns=[*RUN_KEY, "t"])
    # The runs that recorded rank frames at all, shown or not: a run with frames only in the opening is on the
    # text sort, not unknown.
    framed = {
        (a, c, int(s))
        for d in (steps, balance_steps)
        if d is not None and not d.empty
        for a, c, s in d[["arm", "category", "seed"]].drop_duplicates().itertuples(index=False, name=None)
    }
    # Only what the app shows (#4605): the opening's own detectors are on no screen.
    shown_from = _shown_from(ordinary)
    if ordinary.empty:
        on_screen = ordinary
    else:
        first = pd.Series(shown_from).reindex(pd.MultiIndex.from_frame(ordinary[list(RUN_KEY)])).to_numpy()
        on_screen = ordinary[ordinary["t"].to_numpy(dtype=float) >= first]
    by_run = {tuple(k): g.groupby("t")["average_precision"].mean() for k, g in on_screen.groupby(RUN_KEY)}
    steps, balance_steps, thresholds = (_on_screen(d) for d in (steps, balance_steps, thresholds))
    pk = picks[~_is_check(picks)] if not picks.empty else picks
    goods_run = {tuple(k): g.sort_values("t") for k, g in pk.groupby(RUN_KEY)} if not pk.empty else {}
    horizon = max(CHECKPOINTS[-1], int(ordinary["t"].max()) if not ordinary.empty else 0)
    grid = np.arange(0, horizon + 1)
    text_val: dict[tuple, dict[str, float]] = {}
    if lines is not None and not lines.empty:
        for r in lines[lines["point"].astype(str) == "text"].to_dict("records"):
            text_val[(r["arm"], r["category"], int(r["seed"]), float(r["floor"]))] = {m: r[m] for m in CURVE_METRICS}
    run_val: dict[tuple, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    if steps is not None and not steps.empty:
        for (arm, cat, seed, floor), g in steps.groupby(["arm", "category", "seed", "floor"]):
            g = g.sort_values("t")
            run_val[(arm, cat, int(seed), float(floor))] = (
                g["t"].to_numpy(),
                {m: g[m].to_numpy(dtype=float) for m in CURVE_METRICS},
            )
    # The balance's curves (#4413), keyed by ("b", beta) beside the floors.
    if balances is not None and not balances.empty:
        app = balances[balances["rule"] == RULE_APP] if "rule" in balances else balances
        for r in app[app["point"].astype(str) == "text"].to_dict("records"):
            text_val[(r["arm"], r["category"], int(r["seed"]), ("b", float(r["beta"])))] = {
                m: r[m] for m in BALANCE_CURVE_METRICS
            }
    if balance_steps is not None and not balance_steps.empty:
        app_steps = balance_steps[balance_steps["rule"] == RULE_APP] if "rule" in balance_steps else balance_steps
        for (arm, cat, seed, beta), g in app_steps.groupby(["arm", "category", "seed", "beta"]):
            g = g.sort_values("t")
            run_val[(arm, cat, int(seed), ("b", float(beta)))] = (
                g["t"].to_numpy(),
                {m: g[m].to_numpy(dtype=float) for m in BALANCE_CURVE_METRICS},
            )
    # The objective and the precision and recall behind it (the right panel's per-click path, #4605).
    thr_cols = ("thr_fbeta", "thr_precision", "thr_recall")
    thr_run: dict[tuple, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    if thresholds is not None and not thresholds.empty:
        for (arm, cat, seed), g in thresholds[thresholds["point"] == "step"].groupby(["arm", "category", "seed"]):
            g = g.sort_values("t")
            thr_run[(arm, cat, int(seed))] = (g["t"].to_numpy(), {m: g[m].to_numpy(dtype=float) for m in thr_cols})
    rows = []
    for r in cells.itertuples():
        key = (r.dataset, r.category, _emb_of(r.arm), _style_of(r.arm), int(r.seed))
        s = by_run.get(key)
        ap = np.full(len(grid), r.text_ap, dtype=float)
        if s is not None and len(s):
            idx = np.searchsorted(s.index.to_numpy(), grid, side="right") - 1
            have = idx >= 0
            ap[have] = s.to_numpy()[idx[have]]
            ap[0] = r.text_ap
        g = goods_run.get(key)
        goods = np.zeros(len(grid))
        if g is not None and len(g):
            cum = g["picked_label"].cumsum().to_numpy()
            idx = np.searchsorted(g["t"].to_numpy(), grid, side="right") - 1
            goods[idx >= 0] = cum[idx[idx >= 0]]
        f1_cols = {}
        families: list[tuple[Any, tuple[str, ...]]] = [(floor, CURVE_METRICS) for floor in CURVE_FLOORS]
        families += [(("b", b), BALANCE_CURVE_METRICS) for b in BETAS]
        for floor, metrics in families:
            starts = text_val.get((r.arm, r.category, int(r.seed), floor), {})
            have_run = run_val.get((r.arm, r.category, int(r.seed), floor))
            for metric in metrics:
                start = starts.get(metric, np.nan)
                v = np.full(len(grid), start, dtype=float)
                if have_run is not None:
                    ft, fvals = have_run
                    idx = np.searchsorted(ft, grid, side="right") - 1
                    v[idx >= 0] = fvals[metric][idx[idx >= 0]]
                    v[0] = start
                elif not r.never_trained and (r.arm, r.category, int(r.seed)) not in framed:
                    v[1:] = np.nan
                f1_cols[curve_col(metric, floor)] = v
        # The objective over clicks (#4427): the typed query's set until the app shows a detector (#4605), then
        # the line carried forward; with it the precision and recall of the same set.
        thr = {m: np.full(len(grid), getattr(r, f"text_{m}", np.nan), dtype=float) for m in thr_cols}
        have_thr = thr_run.get((r.arm, r.category, int(r.seed)))
        if have_thr is not None:
            tt, tv = have_thr
            idx = np.searchsorted(tt, grid, side="right") - 1
            for m in thr_cols:
                thr[m][idx >= 0] = tv[m][idx[idx >= 0]]
        rows.append(
            pd.DataFrame(
                {
                    "arm": r.arm,
                    "category": r.category,
                    "seed": r.seed,
                    "t": grid,
                    "ap": ap,
                    "goods": goods,
                    **thr,
                    **f1_cols,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


_ARM_OF = {v: k for k, v in ARMS.items()}


def _emb_of(arm: str) -> str:
    return _ARM_OF[arm][0]


def _style_of(arm: str) -> str:
    return _ARM_OF[arm][1]


def attribute(base: pd.DataFrame, picks: pd.DataFrame, ts: dict) -> pd.DataFrame:
    """One row per click: the share of the test-AP change it is credited with.

    The spot check's rows and picks are left out: the check is not a click, and
    its retrains would otherwise hand its picks the credit for moving the line.
    """
    ordinary = base[~_is_check(base)]
    series = ordinary.groupby(RUN_KEY + ["t"], as_index=False)["average_precision"].mean().sort_values(RUN_KEY + ["t"])
    out = []
    pk = picks[~_is_check(picks)] if not picks.empty else picks
    pk = pk.sort_values(RUN_KEY + ["t"]) if not pk.empty else pk
    groups = dict(tuple(pk.groupby(RUN_KEY))) if not pk.empty else {}
    for key, s in series.groupby(RUN_KEY):
        clicks = groups.get(key)
        if clicks is None or clicks.empty:
            continue
        ds, cat, emb, style, seed = key
        start = _text_for(ts, ds, cat, emb, int(seed))["text_ap"]
        prev_t, prev = 0, (start if np.isfinite(start) else None)
        ct = clicks["t"].to_numpy()
        for t, ap in s[["t", "average_precision"]].itertuples(index=False):
            grp = clicks[(ct > prev_t) & (ct <= t)]
            if prev is not None and len(grp):
                d_ap = (ap - prev) / len(grp)
                for r in grp.itertuples():
                    out.append(
                        {
                            "dataset": ds,
                            "category": cat,
                            "embedder": emb,
                            "style": style,
                            "seed": int(seed),
                            "t": int(r.t),
                            "image_id": int(r.picked_id),
                            "label": int(r.picked_label),
                            "phase": getattr(r, "phase", ""),
                            "shared_with": len(grp),
                            "d_ap": d_ap,
                        }
                    )
            prev_t, prev = int(t), ap
    inf = pd.DataFrame(out)
    if not inf.empty:
        inf["arm"] = [ARMS.get((e, s), f"{e}/{s}") for e, s in zip(inf["embedder"], inf["style"], strict=True)]
        inf["class"] = inf["category"].str.split("@").str[0]
        inf["band"] = inf["category"].str.split("@").str[1]
        # Positive = helped: a click that RAISED held-out AP.
        inf["help_ap"] = inf["d_ap"]
    return inf


def _with_resid(inf: pd.DataFrame) -> pd.DataFrame:
    inf = inf.assign(when=np.where(inf["t"] <= EARLY, "early", np.where(inf["t"] > LATE, "late", "mid")))
    inf["resid"] = inf["help_ap"] - inf.groupby(_BUCKET)["help_ap"].transform("mean")
    return inf


def roll_up(inf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if inf.empty:
        return inf, inf
    inf = _with_resid(inf)
    agg = dict(
        n_obs=("help_ap", "size"),
        help_ap=("help_ap", "mean"),
        help_ap_sd=("help_ap", "std"),
    )
    img = inf.groupby("image_id").agg(
        n_obs=("help_ap", "size"),
        n_detectors=("category", "nunique"),
        n_classes=("class", "nunique"),
        clicked_as_positive=("label", "sum"),
        help_ap=("help_ap", "mean"),
        help_ap_sd=("help_ap", "std"),
    )
    img = img.join(_resid_z(inf))
    for w in ("early", "late"):
        sub = inf[inf["when"] == w].groupby("image_id")["help_ap"].agg(["mean", "size"])
        img[f"help_{w}"] = sub["mean"]
        img[f"n_{w}"] = sub["size"]
    img = img.reset_index().sort_values("help_ap", ascending=False)
    det = inf.groupby(["image_id", "arm", "class", "label"]).agg(**agg)
    return img, det.reset_index()


def _resid_z(inf: pd.DataFrame) -> pd.DataFrame:
    g = inf.groupby("image_id")["resid"].agg(["mean", "std", "size"])
    z = g["mean"] / (g["std"] / np.sqrt(g["size"]))
    ok = (g["size"] >= MIN_OBS_Z) & (g["std"] > 0)
    return pd.DataFrame({"resid": g["mean"], "resid_z": z.where(ok)})


def harmful_pairs(inf: pd.DataFrame) -> pd.DataFrame:
    """Per image x class x label, the net credit (``resid``) and its z; the pairs past -``Z_FLAG``.

    An image hurts particular classes, not everything, so a hand review (#4179)
    works from this list rather than from ``images.csv``: each row is one
    question, "is this image really a <label> for <class>?", strongest first.
    """
    inf = _with_resid(inf)
    g = inf.groupby(["image_id", "class", "label"])["resid"].agg(["mean", "std", "size"])
    g = g[(g["size"] >= MIN_OBS_Z) & (g["std"] > 0)]
    g["z"] = g["mean"] / (g["std"] / np.sqrt(g["size"]))
    out = g[g["z"] < -Z_FLAG].rename(columns={"mean": "resid", "size": "n_clicks"}).reset_index()
    return out.sort_values("z")


def image_null(inf: pd.DataFrame, reps: int = 5, seed: int = 0) -> tuple[int, int, list[tuple[int, int]]]:
    """Images flagged helpful / harmful, and the same counts with images shuffled within buckets."""

    def count(df: pd.DataFrame) -> tuple[int, int]:
        z = _resid_z(df)["resid_z"]
        return int((z > Z_FLAG).sum()), int((z < -Z_FLAG).sum())

    inf = _with_resid(inf)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(reps):
        shuf = inf.groupby(_BUCKET)["image_id"].transform(lambda s: rng.permutation(s.to_numpy()))
        null.append(count(inf.assign(image_id=shuf)))
    return (*count(inf), null)


def _md(df: pd.DataFrame, index: bool = True) -> str:
    """A GitHub markdown table, without pandas' optional `tabulate` dependency."""
    d = df.reset_index() if index else df
    head = [str(c) for c in d.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in d.itertuples(index=False)]
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


#: The line's points the headline reads, left to right.
HEADLINE_POINTS = ("text", "25", "50", "final", "ceiling")


def line_table(lines: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The line per *by* x point x floor: mean precision, share meeting P, shortfall, recall, oracle recall, F1."""
    lines = lines[lines["point"].isin(HEADLINE_POINTS)].copy()
    lines["point"] = pd.Categorical(lines["point"], HEADLINE_POINTS, ordered=True)
    g = lines.groupby([*by, "floor", "point"], observed=True)
    out = g.agg(
        k=("k", "mean"),
        precision=("precision", "mean"),
        meets=("meets", "mean"),
        shortfall=("shortfall", "mean"),
        recall=("recall", "mean"),
        oracle_recall=("oracle_recall", "mean"),
        f1=("f1", "mean"),
        oracle_f1=("oracle_f1", "mean"),
        runs=("precision", "count"),
    )
    return out.round(3)


def returned_at_p(lines: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The returned set per *by* x floor x point, scored at the floor it aimed for (#4408).

    ``precision`` against its target ``floor`` (``gap`` = precision - P; ``meets``
    the share of runs at or above P), and ``recall`` against ``oracle_recall``,
    the most any cut of the same ranking returns at or above P (``share`` =
    recall / oracle recall).
    """
    lines = lines[lines["point"].isin(HEADLINE_POINTS)].copy()
    lines["point"] = pd.Categorical(lines["point"], HEADLINE_POINTS, ordered=True)
    out = lines.groupby([*by, "floor", "point"], observed=True).agg(
        k=("k", "mean"),
        precision=("precision", "mean"),
        meets=("meets", "mean"),
        recall=("recall", "mean"),
        oracle_recall=("oracle_recall", "mean"),
        runs=("precision", "count"),
    )
    out.insert(2, "gap", out["precision"] - out.index.get_level_values("floor").to_numpy(dtype=float))
    out.insert(6, "share", out["recall"] / out["oracle_recall"])
    return out.round(3)


def session_floors(cells: pd.DataFrame) -> list[float]:
    """The floors this run's sessions aimed at (``session_floor``), most runs first."""
    if "session_floor" not in cells:
        return []
    return [float(x) for x in cells["session_floor"].dropna().value_counts().index]


def returned_at_p_md(cells: pd.DataFrame, lines: pd.DataFrame) -> list[str]:
    """The summary's first line section: precision against P and recall against the oracle, per floor."""
    own = session_floors(cells)
    who = (
        "These sessions aimed at P = " + ", ".join(f"{x:.0%}" for x in own) + "; the other floors are read off "
        "the same sessions, which is exact only while a session ignores P (a review runs one set of sessions "
        "per P, #4408)."
        if own
        else "These sessions aimed at no floor (a balance, since #4413); every P here is read off them."
    )
    if lines.empty or not lines[lines["point"] != "text"]["precision"].notna().any():
        return ["## The returned set at each P", "", who, "", "*No rank frames: only click 0 is known.*", ""]
    return [
        "## The returned set at each P: precision against P, recall against the oracle",
        "",
        "The set the app returns when it aims for P, on the fresh test half. `precision` is the share of it "
        "that is right, against the target P (`gap` = precision - P: below 0 is a broken promise, far above "
        "0 is recall left behind); `meets` is the share of runs at or above P. `recall` is the share of the "
        "corpus's positives it returns, against `oracle_recall`, the most any cut of the same ranking returns "
        "while staying at or above P (`share` = recall / oracle recall). " + who,
        "",
        _md(returned_at_p(lines, ["arm"])),
        "",
    ]


def check_table(cells: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The spot check per *by*: how often it confirms, its range, and whether the range held the truth."""
    ran = cells[cells["check_status"].fillna("").astype(str) != ""]
    if ran.empty:
        return pd.DataFrame()
    g = ran.assign(
        confirmed=(ran["check_status"] == "confirmed").astype(float),
        width=ran["range_hi"] - ran["range_lo"],
    ).groupby(by)
    return g.agg(
        checks=("confirmed", "size"),
        confirm_rate=("confirmed", "mean"),
        k=("check_k", "mean"),
        votes=("check_votes", "mean"),
        range_lo=("range_lo", "mean"),
        range_hi=("range_hi", "mean"),
        width=("width", "mean"),
        truth=("check_truth", "mean"),
        covered=("check_covered", "mean"),
    ).round(3)


def objective_table(cells: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The objective per *by* (#4427): F-beta of the withheld set above the app's threshold, unchecked and checked.

    Over every run, a run that never trained at its typed query's set (#4631); its check's effect is 0.
    """
    out = (
        cells.groupby(by)
        .agg(
            runs=("thr_fbeta_final", "count"),
            fbeta_25=("thr_fbeta_25", "mean"),
            fbeta_50=("thr_fbeta_50", "mean"),
            fbeta_100=("thr_fbeta_100", "mean"),
            fbeta_unchecked=("thr_fbeta_unchecked", "mean"),
            fbeta_checked=("thr_fbeta_final", "mean"),
            walk_effect=("thr_walk_effect", "mean"),
            walk_effect_se=("thr_walk_effect", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan),
            precision=("thr_precision_final", "mean"),
            recall=("thr_recall_final", "mean"),
            returned_unchecked=("thr_returned_unchecked", "mean"),
            returned_checked=("thr_returned_final", "mean"),
            check_votes=("check_votes", "mean"),
        )
        .round(3)
    )
    return out


def objective_md(cells: pd.DataFrame) -> list[str]:
    """The headline (owner, 2026-10-01): the withheld images above the app's threshold, as F-beta at the session's beta."""
    own = session_betas(cells)
    aim = ("beta = " + ", ".join(f"{x:g}" for x in own)) if own else "a floor (read as F1 here)"
    if cells.empty or "thr_fbeta_final" not in cells or not cells["thr_fbeta_final"].notna().any():
        return [
            "## The objective: the withheld set above the app's threshold",
            "",
            "*No session rows with a threshold.*",
            "",
        ]
    c = cells
    by_band = (
        c.groupby(["arm", "band"])[["thr_fbeta_unchecked", "thr_fbeta_final", "thr_walk_effect"]].mean().round(3)
        if "band" in c and c["band"].astype(bool).any()
        else None
    )
    out = [
        "## The objective: the withheld set above the app's threshold",
        "",
        "The images of the withheld half that score above the threshold the app holds, as F-beta at the "
        f"session's own preference ({aim}): `fbeta_unchecked` at the last click before the spot check (the "
        "line as AutoFind or an unchecked session leaves it), `fbeta_checked` after the check and its votes, "
        "`walk_effect` their paired difference, with the set's precision, recall and size (`returned_*`) and "
        "the check's votes. Fixed-click columns read the unchecked line at that click. This, not the "
        "rank-count line below (the balance's rule re-drawn on the fresh ranking), is what a user's next "
        "corpus gets, and it is where a harvesting acquisition shows: a thinned unvoted top pushes the kept "
        "set's edge score up and few fresh images clear it (#4427). Every run counts: until the app shows a "
        "run's detector, and throughout a run that never trained, it scores the typed query's own set (#4605, "
        "#4631).",
        "",
        _md(objective_table(cells, ["arm"])),
        "",
    ]
    if by_band is not None:
        out += ["By size band:", "", _md(by_band), ""]
    return out


def in_hand_table(pools: pd.DataFrame, balances: pd.DataFrame | None, by: list[str]) -> pd.DataFrame:
    """Positives in hand per *by* x point (#4427): Goods + the kept set's positives on the session's own pool.

    Beside them the kept set on that pool (size, precision, recall, F-beta
    share of its best cut) and, read off the same frames, the fresh test
    half's F-beta share at the session's own beta.
    """
    pts = [p for p in HEADLINE_POINTS if p != "ceiling"]
    p = pools[pools["point"].isin(pts)].copy()
    p["point"] = pd.Categorical(p["point"], pts, ordered=True)
    out = (
        p.groupby([*by, "point"], observed=True)
        .agg(
            goods=("goods", "mean"),
            pool_tp=("pool_tp", "mean"),
            in_hand=("in_hand", "mean"),
            pool_k=("pool_k", "mean"),
            pool_precision=("pool_precision", "mean"),
            pool_recall=("pool_recall", "mean"),
            pool_fb_share=("pool_fb_share", "mean"),
            pool_positives=("pool_positives", "mean"),
            runs=("goods", "count"),
        )
        .round(3)
    )
    if balances is not None and not balances.empty:
        own = p[[*by, "point", "beta", "category", "seed"]].drop_duplicates()
        app = balances[balances["rule"] == RULE_APP] if "rule" in balances else balances
        b = app.merge(own, on=[*by, "point", "beta", "category", "seed"], how="inner")
        if not b.empty:
            b["point"] = pd.Categorical(b["point"], pts, ordered=True)
            fresh = b.groupby([*by, "point"], observed=True)["fb_share"].mean().round(3).rename("fresh_fb_share")
            out = out.join(fresh, how="left")
    return out


def in_hand_md(cells: pd.DataFrame, pools: pd.DataFrame | None, balances: pd.DataFrame | None) -> list[str]:
    """The diagnostic section (#4427): the user's own corpus, beside the withheld half's reading."""
    own = session_betas(cells)
    aim = ("beta = " + ", ".join(f"{x:g}" for x in own)) if own else "a floor (scored as F1 here)"
    if pools is None or pools.empty or not pools[pools["point"] != "text"]["in_hand"].notna().any():
        return [
            "## Diagnostic: the user's own corpus and the positives in hand",
            "",
            "*No rank frames: nothing to read.*",
            "",
        ]
    return [
        "## Diagnostic: the user's own corpus and the positives in hand",
        "",
        "Not the objective (that is the F-beta of the kept set on the withheld test half, above), but what "
        "the screen shows at each click on the session's own corpus: the Goods voted so far plus the "
        "positives inside the set the line keeps over the UNVOTED ranking (`in_hand` = `goods` + `pool_tp`), "
        "with that kept set's size, precision, recall and F-beta share of its best cut there (`pool_*`), and "
        "beside it the withheld half's share (`fresh_fb_share`). A harvesting acquisition strips the unvoted "
        "top, and the walk only sees the user's pool, so this is where a shallow line gets explained. The "
        "line keeps what the app kept (the unchecked count, then the check's end at the last click). These "
        f"sessions aimed at {aim}.",
        "",
        _md(in_hand_table(pools, balances, ["arm"])),
        "",
    ]


def returned_at_beta(balances: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The returned set per *by* x beta x point under the balance (#4413): F-beta over the best cut."""
    b = balances[balances["point"].isin(HEADLINE_POINTS)].copy()
    b["point"] = pd.Categorical(b["point"], HEADLINE_POINTS, ordered=True)
    if "rule" not in b:
        b["rule"] = RULE_APP
    b["rule"] = pd.Categorical(b["rule"], RULES, ordered=True)
    return (
        b.groupby([*by, "rule", "beta", "point"], observed=True)
        .agg(
            k=("k", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            fbeta=("fbeta", "mean"),
            oracle_fbeta=("oracle_fbeta", "mean"),
            fb_share=("fb_share", "mean"),
            # The labels line (#4452) can keep nothing: an empty set, scored precision 0 and F-beta 0 (#4631).
            empty=("k", lambda k: float((k == 0).mean())),
            runs=("fbeta", "count"),
        )
        .round(3)
    )


def session_balance_steps(balance_steps: pd.DataFrame, balances: pd.DataFrame) -> pd.DataFrame:
    """Every run's returned set at every recorded click, per beta and rule, as its session shows it (#4631).

    ``balance_steps.csv`` has a row only where a run recorded a rank frame, so a mean over its rows at a
    click is over the runs with a frame there: never a run that never trained, and every one of the
    opening's detectors, which the app does not show (#4605).  This reads every run ``balances`` lists at
    every click any run recorded: its last frame on screen at or before the click, else the typed query's
    row of the same rule (the ``text`` point).  ``framed`` says which.
    """
    keys = ["arm", "category", "seed", "beta", "rule"]
    metrics = list(BALANCE_METRICS)
    if balance_steps is None or balance_steps.empty or balances is None or balances.empty:
        return pd.DataFrame(columns=[*keys, "t", "framed", *metrics])
    steps = balance_steps if "rule" in balance_steps else balance_steps.assign(rule=RULE_APP)
    bal = balances if "rule" in balances else balances.assign(rule=RULE_APP)
    text = bal.loc[bal["point"].astype(str) == "text", [*keys, *metrics]].drop_duplicates(keys)
    on = _on_screen(steps)
    on = on.loc[:, [*keys, "t", *metrics]].assign(t=lambda d: d["t"].astype(int))
    on = on.drop_duplicates([*keys, "t"], keep="last").assign(t_frame=lambda d: d["t"]).sort_values("t")
    clicks = pd.DataFrame({"t": np.sort(steps["t"].astype(int).unique())})
    grid = text.loc[:, keys].merge(clicks, how="cross").sort_values("t")
    out = pd.merge_asof(grid, on, on="t", by=keys, direction="backward")
    out = out.merge(text, on=keys, suffixes=("", "_text"))
    framed = out["t_frame"].notna()
    for m in metrics:
        out[m] = out[m].where(framed, out[f"{m}_text"])
    out["framed"] = framed
    return out.loc[:, [*keys, "t", "framed", *metrics]].sort_values([*keys, "t"]).reset_index(drop=True)


def session_betas(cells: pd.DataFrame) -> list[float]:
    if "session_beta" not in cells:
        return []
    return [float(x) for x in cells["session_beta"].dropna().value_counts().index]


def returned_at_beta_md(cells: pd.DataFrame, balances: pd.DataFrame) -> list[str]:
    """The balance's section (#4413): F-beta of the returned set over the best F-beta of any cut, per beta."""
    own = session_betas(cells)
    who = (
        "These sessions aimed at beta = " + ", ".join(f"{x:g}" for x in own) + "; the other balances are read off "
        "the same sessions (a review runs one set of sessions per beta)."
        if own
        else "These sessions aimed at a floor, not a balance; every beta here is read off them."
    )
    if balances is None or balances.empty or not balances[balances["point"] != "text"]["fbeta"].notna().any():
        return ["## The returned set at each balance", "", who, "", "*No rank frames: only click 0 is known.*", ""]
    return [
        "## The returned set at each balance: F-beta over the best cut",
        "",
        "The set each sort returns at a balance (F-beta's beta at the app's presets: "
        + ", ".join(f"{b:g}" for b in BETAS)
        + "; #4448), on the fresh test half, under ONE rule per row (owner, 2026-10-04: apples to apples). "
        "`app line`: the text sort's own line in the app (the blind GMM cut), the detector's labels line "
        "(#4452) with its corpus side fitted on the test half as Find draws it (at the sessions' own beta, the "
        "objective's returned set), and for the full-label ceiling Find's labels line from its full labels "
        "(#4486; blank on runs from before it). `top-K`: set-constant at the old cap "
        "(32 at beta <= 1, 128 above) on every sort. Compare text and detector within a rule, never across. "
        "`fbeta` against `oracle_fbeta`, the best any cut of the same ranking reaches (`fb_share` = fbeta / "
        "oracle); `k`, `precision` and `recall` beside it, and `empty`, the share of runs whose line keeps "
        "nothing (an empty set: F-beta and precision 0, #4631). " + who,
        "",
        _md(returned_at_beta(balances, ["arm"])),
        "",
    ]


def summary(
    cells: pd.DataFrame,
    lines: pd.DataFrame,
    img: pd.DataFrame,
    out: Path,
    null: tuple | None = None,
    balances: pd.DataFrame | None = None,
    pools: pd.DataFrame | None = None,
    stops: pd.DataFrame | None = None,
    margins: pd.DataFrame | None = None,
) -> None:
    lines_md = ["# State of the App -- summary tables", ""]
    if not cells.empty and cells["never_trained"].any():
        nt = cells[cells["never_trained"]][["arm", "category", "seed", "positives_found", "text_ap", "ceiling_ap"]]
        lines_md += ["## Runs that never found a positive (no detector was ever trained)", "", _md(nt, index=False), ""]
    if not cells.empty:
        lines_md += objective_md(cells)
        rank = ["text_ap", "ap_25", "ap_50", "final_ap", "ceiling_ap", "goods_25", "goods_50", "positives_found"]
        lines_md += [
            "## The ranking: AP and Goods found, per path",
            "",
            _md(cells.groupby("arm")[rank].mean().round(3)),
            "",
        ]
        lines_md += returned_at_p_md(cells, lines)
        lines_md += returned_at_beta_md(cells, balances)
        lines_md += in_hand_md(cells, pools, balances)
        f1s = ["text_f1", *[f"f1_{c}" for c in CHECKPOINTS if c in (25, 50, 100)], "final_f1", "ceiling_f1"]
        f1s += ["final_oracle_f1"]
        lines_md += [
            f"## The returned set: F1 of the line at P = {DEFAULT_FLOOR:.0%}, per path",
            "",
            "F1 of the set the default floor keeps on the test half (the top K, unchecked), at click 0 "
            "(the text sort), at fixed clicks, at the end, and for the full-label model. "
            "`final_oracle_f1` is the best F1 any cut of the final ranking reaches.",
            "",
            _md(cells.groupby("arm")[f1s].mean().round(3)),
            "",
        ]
        lines_md += [
            "## The line on a fresh corpus (the test half), per path",
            "",
            "At each floor P the line keeps the top K of the ranking (the floor's unchecked candidate). "
            "`precision` is the share of them that is right, `meets` the share of runs at or above P, "
            "`shortfall` the mean max(0, P - precision), `oracle_recall` the best recall any cut of "
            "the same ranking reaches at precision >= P, `f1` the kept set's F1 and `oracle_f1` the best "
            "F1 any cut reaches. `ceiling` is the full-label model's ranking.",
            "",
        ]
        if lines[lines["point"] != "text"]["precision"].notna().any():
            lines_md += [_md(line_table(lines, ["arm"])), ""]
        else:
            lines_md += ["*This run recorded no rank frames (`CALIB_RANK_FRAME_STEPS`); only click 0 is known.*", ""]
            lines_md += [_md(line_table(lines[lines["point"] == "text"], ["arm"])), ""]
        chk = check_table(cells, ["arm"])
        lines_md += ["## The spot check at the end of the session", ""]
        if chk.empty:
            lines_md += ["*No run checked its line.*", ""]
        else:
            lines_md += [
                "`truth` is the share right of the set the range describes (the check's candidate, "
                "the top K unvoted of the session's pool); `covered` is how often the range held it.",
                "",
                _md(chk),
                "",
            ]
        if stops is not None:
            lines_md += stopping_md(stops, margins if margins is not None else pd.DataFrame())
        by_band = ["text_ap", "final_ap", "ceiling_ap", "text_f1", "final_f1", "ceiling_f1", "positives_found"]
        lines_md += ["## Per path and band", "", _md(cells.groupby(["arm", "band"])[by_band].mean().round(3)), ""]
        band_line = lines[(lines["point"] == "final") & (lines["floor"] == 0.5)]
        if band_line["precision"].notna().any():
            lines_md += [
                "## The final line at 50%, per path and band",
                "",
                _md(
                    band_line.groupby(["arm", "band"])[["precision", "meets", "recall", "oracle_recall", "f1"]]
                    .mean()
                    .round(3)
                ),
                "",
            ]
        for arm, a in cells.groupby("arm"):
            by = a.groupby("class")[
                ["text_ap", "final_ap", "ceiling_ap", "clicks_bought", "headroom", "text_f1", "final_f1", "ceiling_f1"]
            ].mean()
            lines_md += [
                f"## {arm}: the 10 hardest classes (final AP)",
                "",
                _md(by.sort_values("final_ap").head(10).round(3)),
                "",
            ]
            lines_md += [
                f"## {arm}: the 10 easiest classes",
                "",
                _md(by.sort_values("final_ap", ascending=False).head(10).round(3)),
                "",
            ]
            lines_md += [
                f"## {arm}: most headroom (full-label ceiling AP minus final AP)",
                "",
                _md(by.sort_values("headroom", ascending=False).head(10).round(3)),
                "",
            ]
            lines_md += [
                f"## {arm}: what the clicks bought least (final AP minus text AP)",
                "",
                _md(by.sort_values("clicks_bought").head(10).round(3)),
                "",
            ]
    if not img.empty:
        seen = img[img["n_obs"] >= 3]
        lines_md += [f"## Images clicked 3+ times ({len(seen)} of {len(img)})", ""]
        lines_md += [
            "### Most helpful (mean held-out AP added per click)",
            "",
            _md(seen.head(15).round(4), index=False),
            "",
        ]
        lines_md += ["### Most harmful", "", _md(seen.tail(15).iloc[::-1].round(4), index=False), ""]
        both = seen.dropna(subset=["help_early", "help_late"])
        flips = both[np.sign(both["help_early"]) != np.sign(both["help_late"])]
        lines_md += [f"### Early/late sign flips ({len(flips)} images seen both early and late)", ""]
        lines_md += [_md(flips.sort_values("n_obs", ascending=False).head(15).round(4), index=False), ""]
    if null is not None:
        hp, hn, perm = null
        tested = int(img["resid_z"].notna().sum())
        lines_md += [
            f"## Images with an effect of their own (|resid_z| > {Z_FLAG:g}, {MIN_OBS_Z}+ clicks)",
            "",
            f"{tested} images tested: **{hp} helpful, {hn} harmful**. With images shuffled within each "
            f"(cell, label, phase) bucket: " + ", ".join(f"{a}/{b}" for a, b in perm) + " (helpful/harmful).",
            "",
        ]
        flagged = img[img["resid_z"].abs() > Z_FLAG].sort_values("resid_z")
        lines_md += [_md(flagged.round(4), index=False), ""]
    (out / "summary.md").write_text("\n".join(lines_md))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None, help="text_baseline.csv (the click-0 score)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--path", choices=["all", *PATHS], default="all", help="one production path per report")
    ap.add_argument("--seeds", type=int, default=0, help="keep seeds 0..N-1 only (0 = all); for a snapshot mid-run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base, sky, picks, frames = load(args.exp)
    # A two-pass run keeps the full-label ceiling in its own directory
    # (launch.sh SOTA_PASS=ceiling); read its skyline rows and frames alongside.
    ceiling_dir = args.exp / "ceiling"
    if (ceiling_dir / "results" / "cells").exists():
        _, sky2, _, frames2 = load(ceiling_dir)
        sky = pd.concat([sky, sky2], ignore_index=True) if not sky.empty else sky2
        frames2 = frames2[frames2["kind"] == CEILING] if not frames2.empty else frames2
        frames = pd.concat([frames, frames2], ignore_index=True) if not frames.empty else frames2
    if args.path != "all":
        emb, style = PATHS[args.path]
        keep = lambda df: df[(df["embedder"] == emb) & (df["style"] == style)] if not df.empty else df  # noqa: E731
        base, sky, picks, frames = keep(base), keep(sky), keep(picks), keep(frames)
    if args.seeds:
        within = lambda df: df[df["seed"] < args.seeds] if not df.empty else df  # noqa: E731
        base, sky, picks, frames = within(base), within(sky), within(picks), within(frames)
    if base.empty and sky.empty:
        raise SystemExit(f"no cells under {args.exp}/results/cells")
    ts = text_scores(args.baseline)
    cells, lines, steps, balances, balance_steps, pools, pool_steps, thresholds = run_tables(
        base, sky, ts, picks, frames
    )
    inf = attribute(base, picks, ts) if not base.empty else pd.DataFrame()
    img, det = roll_up(inf)
    cells.to_csv(args.out / "cells.csv", index=False)
    lines.to_csv(args.out / "lines.csv", index=False)
    steps.to_csv(args.out / "line_steps.csv", index=False)
    balances.to_csv(args.out / "balances.csv", index=False)
    balance_steps.to_csv(args.out / "balance_steps.csv", index=False)
    pools.to_csv(args.out / "pools.csv", index=False)
    pool_steps.to_csv(args.out / "pool_steps.csv", index=False)
    thresholds.to_csv(args.out / "thresholds.csv", index=False)
    curves(cells, base, picks, lines, steps, balances, balance_steps, thresholds).to_csv(
        args.out / "curves.csv", index=False
    )
    inf.to_csv(args.out / "influence.csv", index=False)
    img.to_csv(args.out / "images.csv", index=False)
    det.to_csv(args.out / "image_detector.csv", index=False)
    if not inf.empty:
        harmful_pairs(inf).to_csv(args.out / "harmful_pairs.csv", index=False)
    stops, margins = stopping_tables(base, cells)
    stops.to_csv(args.out / "stops.csv", index=False)
    margins.to_csv(args.out / "margins.csv", index=False)
    summary(
        cells,
        lines,
        img,
        args.out,
        image_null(inf) if not inf.empty else None,
        balances=balances,
        pools=pools,
        stops=stops,
        margins=margins,
    )
    n_frames = "no rank frames" if frames.empty else f"{len(frames)} rank frames"
    print(f"{len(cells)} runs, {len(inf)} credited clicks, {len(img)} images, {n_frames} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

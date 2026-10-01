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
the candidate the check drew its picks from.  What the check *should* certify
is #4358's ruling; until then this reports what it does certify.

**How a click is credited (owner, 2026-09-23).** The harness scores the test
split after every click once a Good and a Bad exist; before that there is no
detector to score. So a scored step's change is split EQUALLY among every click
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

#: The two production paths, as the harness names them.
ARMS = {("siglip", "whole_image"): "SigLIP binary", ("siglip+dinov3_patch", "max_patch"): "DINOv3 region"}
#: The State of the App reports are one per production path (owner, 2026-09-24):
#: "Binary Photo" and "Region Photo" (and later e.g. "Document Logo").
PATHS = {"binary": ("siglip", "whole_image"), "region": ("siglip+dinov3_patch", "max_patch")}
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
        out[key] = {"text_ap": _f(r.get("text_AP")), **{c: _f(r.get(c)) for c in cols}}
    return out


def _text_for(ts: dict, ds: str, cat: str, emb: str, seed: int) -> dict[str, float]:
    # A paired region arm opens on its text half's sort, so its click-0 score is
    # the text embedder's when the pair has no row of its own.
    return ts.get((ds, cat, emb, seed)) or ts.get((ds, cat, emb.split("+")[0], seed)) or {"text_ap": np.nan}


def _text_line(text: dict[str, float], floor: float) -> dict[str, float]:
    return {m: text.get(f"text_{m}_{floor_tag(floor)}", np.nan) for m in LINE_METRICS}


def _text_balance(text: dict[str, float], beta: float) -> dict[str, float]:
    return {m: text.get(f"text_{m}_{beta_tag(beta)}", np.nan) for m in BALANCE_METRICS}


def _balance_at(frame: dict | None, beta: float) -> dict[str, float]:
    if frame is None:
        return {m: np.nan for m in BALANCE_METRICS}
    return balance_metrics(
        parse_ranks(frame["test_pos_ranks"]),
        int(frame["n_test"]),
        int(frame["n_test_pos"]),
        beta,
        frame_beta_k(frame, beta),
    )


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
    k = int(_f(end.get("floor_count"))) if np.isfinite(_f(end.get("floor_count"))) else -1
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(cells, lines, steps, balances, balance_steps, pools, pool_steps)``: one row per run, one per run x
    point x floor, one per run x recorded click x floor (every ``step`` rank frame, for the F1 curve), the
    same two per balance (#4413: the returned set's F-beta over the best cut, at each preset beta), and the
    same two over the session's own unvoted pool at the line the app kept (#4427: the positives in hand).

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
    skyd: dict[tuple, float] = {}
    for r in sky.to_dict("records") if not sky.empty else []:
        skyd[(r["dataset"], r["category"], r["embedder"], r["style"], int(r["seed"]))] = _f(r.get("average_precision"))

    keys = set(series) | set(clicks_by) | set(skyd)
    cells, lines, steps_out, balances, balance_steps, pools, pool_steps = [], [], [], [], [], [], []
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
            "session_beta": beta_of.get(key, float("nan")),
            "text_ap": text["text_ap"],
        }
        final_t = int(s.index.max()) if trained else (int(clicks["t"].max()) if clicks is not None else 0)
        for c in CHECKPOINTS:
            at = s[s.index <= c] if trained else None
            # Before a detector exists, what the user sees IS the text sort, so
            # the text score stands in -- never a gap, which would let the mean
            # at a checkpoint silently skip the runs that are still starving.
            row[f"ap_{c}"] = at.iloc[-1] if at is not None and len(at) else text["text_ap"]
            row[f"goods_{c}"] = _goods_by(clicks, c)
        row["final_t"] = final_t
        row["final_ap"] = s.iloc[-1] if trained else text["text_ap"]
        row["positives_found"] = _goods_by(clicks, final_t)
        row["ceiling_ap"] = skyd.get(key, np.nan)
        row["clicks_bought"] = row["final_ap"] - row["text_ap"]
        row["headroom"] = row["ceiling_ap"] - row["final_ap"]
        row.update(_check_columns(checks.get(key), last_frame, check_picks_by.get(key)))
        # The session's own preference, for the pool-side line (#4427): its beta, or F1 under a floor.
        own_beta = beta_of.get(key, 1.0)
        counts, end_count = _counts_by_t(ordinary_by.get(key), checks.get(key))
        cells.append(row)

        # The line, left to right.  Without rank frames only click 0 is known.
        steps = kinds.get("step")
        step_at = {int(r["t"]): r for r in steps.to_dict("records")} if steps is not None else {}
        sky_frame = kinds.get(CEILING)
        first_t = int(s.index.min()) if trained else None
        # (point, t, frame, read the text sort instead).  A point with neither is
        # unknown and stays blank rather than borrowing a neighbour's value.
        points: list[tuple[str, float, dict | None, bool]] = [("text", 0, None, True)]
        for c in CHECKPOINTS:
            if not have_frames:
                points.append((str(c), c, None, False))
            elif c in step_at:
                points.append((str(c), c, step_at[c], False))
            elif last_frame is not None and int(last_frame["t"]) <= c:
                # The pool ran out before click c: the line stays where it ended.
                points.append((str(c), c, last_frame, False))
            else:
                # No detector yet at click c: the user still has the text sort.
                points.append((str(c), c, None, first_t is None or c < first_t))
        points.append(("final", final_t, last_frame if have_frames else None, have_frames and not trained))
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
            for b in BETAS:  # the returned set at each balance (#4413)
                mb = _text_balance(text, b) if use_text else _balance_at(frame, b)
                balances.append({**ident, "point": point, "t": t, "beta": b, **mb})
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
                steps_out.append({**ident, "t": int(fr["t"]), "floor": x, **m})
            for b in BETAS:
                mb = balance_metrics(ranks, int(fr["n_test"]), int(fr["n_test_pos"]), b, frame_beta_k(fr, b))
                balance_steps.append({**ident, "t": int(fr["t"]), "beta": b, **mb})
            t_fr = int(fr["t"])
            mp = _pool_at(fr, counts.get(t_fr, float("nan")), _goods_by(clicks, t_fr), own_beta)
            pool_steps.append({**ident, "t": t_fr, "beta": own_beta, **mp})
    return (
        pd.DataFrame(cells),
        pd.DataFrame(lines),
        pd.DataFrame(steps_out),
        pd.DataFrame(balances),
        pd.DataFrame(balance_steps),
        pd.DataFrame(pools),
        pd.DataFrame(pool_steps),
    )


def curves(
    cells: pd.DataFrame,
    base: pd.DataFrame,
    picks: pd.DataFrame,
    lines: pd.DataFrame | None = None,
    steps: pd.DataFrame | None = None,
    balances: pd.DataFrame | None = None,
    balance_steps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Every run on a common click grid 0..horizon, as the USER would see it.

    Click 0 is the text-only AP; until the first scored click the user still
    sees the text sort, so it carries forward; after that the last scored value
    carries forward. A run that never trained stays at its text AP.  Goods found
    count the clicks only, never the spot check's picks.

    The returned set at each floor (precision, recall, the oracle's recall at
    that floor and F1; :func:`curve_col` names the columns) follows the
    same rule off the ``step`` rank frames: the text sort's until the first
    frame, then the last frame's.  Between two recorded frames that is a carried
    value, not a measurement, so a figure should read the curve only at the
    clicks ``line_steps.csv`` holds.  A trained run with no frames at all (a
    run recorded without them) is blank past click 0.
    """
    ordinary = base[~_is_check(base)] if not base.empty else pd.DataFrame(columns=[*RUN_KEY, "t"])
    by_run = {tuple(k): g.groupby("t")["average_precision"].mean() for k, g in ordinary.groupby(RUN_KEY)}
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
        for r in balances[balances["point"].astype(str) == "text"].to_dict("records"):
            text_val[(r["arm"], r["category"], int(r["seed"]), ("b", float(r["beta"])))] = {
                m: r[m] for m in BALANCE_CURVE_METRICS
            }
    if balance_steps is not None and not balance_steps.empty:
        for (arm, cat, seed, beta), g in balance_steps.groupby(["arm", "category", "seed", "beta"]):
            g = g.sort_values("t")
            run_val[(arm, cat, int(seed), ("b", float(beta)))] = (
                g["t"].to_numpy(),
                {m: g[m].to_numpy(dtype=float) for m in BALANCE_CURVE_METRICS},
            )
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
                elif not r.never_trained:
                    v[1:] = np.nan
                f1_cols[curve_col(metric, floor)] = v
        rows.append(
            pd.DataFrame(
                {"arm": r.arm, "category": r.category, "seed": r.seed, "t": grid, "ap": ap, "goods": goods, **f1_cols}
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
        else "The sessions' own floor is not recorded."
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
        b = balances.merge(own, on=[*by, "point", "beta", "category", "seed"], how="inner")
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
    return (
        b.groupby([*by, "beta", "point"], observed=True)
        .agg(
            k=("k", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            fbeta=("fbeta", "mean"),
            oracle_fbeta=("oracle_fbeta", "mean"),
            fb_share=("fb_share", "mean"),
            runs=("fbeta", "count"),
        )
        .round(3)
    )


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
        "The set the app returns when it aims for a balance (F-beta's beta: 0.5 precision-leaning, 1 "
        "balanced, 2 recall-leaning), on the fresh test half: its `fbeta` against `oracle_fbeta`, the best "
        "any cut of the same ranking reaches (`fb_share` = fbeta / oracle); `k`, `precision` and `recall` "
        "beside it. The text sort and the ceiling keep the balance's cap (32 at beta <= 1, 128 above). " + who,
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
) -> None:
    lines_md = ["# State of the App -- summary tables", ""]
    if not cells.empty and cells["never_trained"].any():
        nt = cells[cells["never_trained"]][["arm", "category", "seed", "positives_found", "text_ap", "ceiling_ap"]]
        lines_md += ["## Runs that never found a positive (no detector was ever trained)", "", _md(nt, index=False), ""]
    if not cells.empty:
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
    cells, lines, steps, balances, balance_steps, pools, pool_steps = run_tables(base, sky, ts, picks, frames)
    inf = attribute(base, picks, ts) if not base.empty else pd.DataFrame()
    img, det = roll_up(inf)
    cells.to_csv(args.out / "cells.csv", index=False)
    lines.to_csv(args.out / "lines.csv", index=False)
    steps.to_csv(args.out / "line_steps.csv", index=False)
    balances.to_csv(args.out / "balances.csv", index=False)
    balance_steps.to_csv(args.out / "balance_steps.csv", index=False)
    pools.to_csv(args.out / "pools.csv", index=False)
    pool_steps.to_csv(args.out / "pool_steps.csv", index=False)
    curves(cells, base, picks, lines, steps, balances, balance_steps).to_csv(args.out / "curves.csv", index=False)
    inf.to_csv(args.out / "influence.csv", index=False)
    img.to_csv(args.out / "images.csv", index=False)
    det.to_csv(args.out / "image_detector.csv", index=False)
    if not inf.empty:
        harmful_pairs(inf).to_csv(args.out / "harmful_pairs.csv", index=False)
    summary(cells, lines, img, args.out, image_null(inf) if not inf.empty else None, balances=balances, pools=pools)
    n_frames = "no rank frames" if frames.empty else f"{len(frames)} rank frames"
    print(f"{len(cells)} runs, {len(inf)} credited clicks, {len(img)} images, {n_frames} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

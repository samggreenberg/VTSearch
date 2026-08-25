"""Stage 2 (fold-count study, #2897): what does more cross-calibration cost, and buy?

Consumes ``results/cells/task_*.csv`` from a run launched with
``CALIB_FOLD_COUNTS`` set (see ``launch_folds_2897.sh``), where every trainable
step emits one ``folds_k{K}_xcal`` row per fold count - and, under safe
thresholds, a ``folds_k{K}_blend`` row and a ``folds_k{K}_anchored`` row (the
shipped fold-anchored rule, #3116) - carrying that K's regret and its measured
``fold_seconds``.

**What makes this analysis paired.** The fold arms are nested prefixes of one
Kmax calibration, so every K in a step re-cuts the *same* votes, the *same*
final model and the *same* held-out test scores.  Every contrast below is
therefore paired at ``(env, category, seed, t)``, and the noise that dominates
an across-runs fold-count comparison (different splits -> different trajectory)
is absent by construction.  The price of that pairing is the acquisition
feedback the screen cannot see, which is what the A/B stage measures; pass the
A/B run dirs as arguments to fold that check in.

Pre-registered deliverables (``docs/experiments/calibration-fold-count/REPORT.md``):

* **Benefit** - regret(K) and its paired delta vs production's K=2, per voting
  mode and vote window, with a Wilcoxon over cell means.
* **Cost** - ``fold_seconds(K)``, its ratio to K=2, and calibration's share of
  the whole per-step budget (the part a user waits through on every retrain).
* **Knee** - the smallest K statistically indistinguishable from the best K at
  the pre-registered :data:`MARGIN`, which is the number the study recommends.
* **Exchange rate** - regret bought per extra second of calibration, so a
  "significant but tiny" win is visible as such.
* **Mechanism** - ``sd(threshold)`` across seeds per K: does more calibration
  actually make the shipped cut less variable?  #3116 established that the
  regret decomposition cannot answer this (``rule_inefficiency`` is a signed
  cost gap, not a variance, and its reference grows with K), so the dispersion
  of the threshold is measured directly and the two decomposition terms are
  reported as arithmetic with a guard flag rather than as a mechanism.

Writes ``results/summary.json``, ``results/agg/*.csv`` and ``results/REPORT.md``.

Usage: ``python analyze_folds_2897.py [ab_run_dir ...]``
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import common
import folds_combine_3115

common.setup_env()

# After `setup_env`: `experiment_config` imports `vtscore` at module scope.
from experiment_config import region_voting_for  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import main_frame_files  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

#: The arms this analyzer owns.  ``xcal`` / ``blend`` / ``anchored`` are the
#: fold-**count** arms (#2897, #3116); the rest are the fold-**combine** arms
#: (#3115), which ride the same rows because they re-cut the same fold prefix.
#: ``anchored_qmedian`` must precede ``anchored`` in the alternation - a regex
#: alternation is first-match, not longest-match, so the other order silently
#: parses every ``anchored_qmedian`` row's arm as ``anchored`` and then fails to
#: anchor at ``$``, dropping the arm from the frame entirely.
FOLD_RE = re.compile(r"^folds_k(?P<k>\d+)_(?P<arm>xcal|blend|anchored_qmedian|anchored|tmean|tmedian|qmean|qmedian)$")

#: Production's fold count: the baseline every delta is measured against.
BASELINE_K = 2

#: Equivalence margin on regret, in ``FPR + FNR`` units (inclusion 0 weights
#: both at 1, so regret lives on 0..2).  0.005 = half a percentage point of
#: combined error rate: below this a fold-count change is not worth a user's
#: retrain latency however clean its p-value, and two fold counts within it are
#: treated as the same answer.  Pre-registered, not tuned to the result.
MARGIN = 0.005

#: How much extra calibration wall clock the study is willing to spend, as a
#: multiple of production's.  Cross-calibration is on the interactive retrain
#: path - the user waits through it after every vote - so an unbounded "more is
#: better" verdict is not adoptable however good the regret curve looks.
COST_CEILING_X = 4.0

#: Vote-count windows.  The cold start and the deep regime are different
#: questions: at 10 votes the calibration set is a handful of scores and extra
#: draws plausibly matter a lot; at 300 the quantile is already stable and they
#: should not.  Banding is what makes a crossover visible instead of averaged
#: away.
CHECKPOINTS = [int(c) for c in os.environ.get("CALIB_FOLD_CHECKPOINTS", "20,50,100,200,300").split(",") if c.strip()]

#: Windows the headline verdict reads (the regime a real search spends most of
#: its votes in).
DEEP_MIN = 100

#: Pairing unit: one step of one trajectory.  Every fold count re-cuts it.
STEP_KEYS = ["env", "category", "seed", "t"]
#: Aggregation unit for the significance tests - a cell, not a step.  Steps
#: within a trajectory are strongly autocorrelated (consecutive steps share
#: nearly all their votes), so testing over steps would count one trajectory's
#: luck hundreds of times.
CELL_KEYS = ["env", "category", "seed", "window"]


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a fixed-width dump."""
    try:
        return df.to_markdown(index=False, floatfmt=".5f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def load_cells(cells_dir: Path) -> pd.DataFrame:
    """Load the fold-count rows, reporting what was dropped rather than hiding it."""
    files = main_frame_files(cells_dir)
    frames, empty, unreadable, headers_only = [], 0, 0, []
    for p in files:
        if p.stat().st_size == 0:
            empty += 1
            continue
        try:
            f = pd.read_csv(p)
        except Exception as exc:  # noqa: BLE001 - a truncated cell must be counted, not crash the run
            common.log(f"  UNREADABLE {p.name}: {exc}")
            unreadable += 1
            continue
        # A cell whose simulation never reached a trainable step writes its
        # HEADER and nothing else.  That file is non-empty, parses cleanly, and
        # contributes zero rows - so it is invisible to a zero-byte check, to
        # `find -size 0`, and to any count of "cells present".  A rare category
        # can legitimately produce one (Autopilot never collects both classes),
        # which is exactly why it has to be *counted* rather than assumed away:
        # "208/208 cells" over a grid where 20 of them are empty is a different
        # study from the one that sentence describes.
        if f.empty:
            headers_only.append(p.name)
            continue
        frames.append(f)
    if headers_only:
        common.log(f"  {len(headers_only)} header-only cells (0 rows): {', '.join(sorted(headers_only)[:8])}...")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["env"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    # Which calibrator actually ran - bag-aware (region) or row-wise (binary) -
    # which is the axis this study splits on.
    #
    # Read per **cell**, from the same predicate the runner uses, not from the
    # dataset name.  Region voting needs *both* halves: boxes from the dataset
    # and a patch grid from the embedder.  `visual_genome_m x siglip` has the
    # first and not the second, so it silently trains and scores whole-image -
    # the trap behind #2877, #2905 and #2897's own errata, where a boxed dataset
    # on a single-vector embedder was reported as a region arm.  #2897's report
    # regrouped that cell into binary **by hand** while this column still called
    # it region; deriving it here is what stops the next study needing to know.
    df["voting"] = np.where(
        [region_voting_for(d, e) for d, e in zip(df["dataset"], df["embedder"], strict=True)],
        "region",
        "binary",
    )
    common.log(
        f"loaded {len(df):,} rows from {len(frames)}/{len(files)} cells "
        f"({empty} zero-byte, {unreadable} unreadable, {len(headers_only)} header-only, skipped)"
    )
    return df


def _optional(v: pd.DataFrame, col: str) -> pd.Series:
    """*col* when the run emitted it, else an all-NaN stand-in of the right shape.

    The honest-reference columns (#3116) postdate the #2897 cells, so an
    analyzer that must still read those runs cannot assume they are there.
    """
    return v[col] if col in v.columns else pd.Series(np.nan, index=v.index, dtype=float)


def fold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Just the fold-count arms, with K and arm parsed out and windows assigned."""
    m = df["gmm_variant"].str.extract(FOLD_RE)
    v = df[m["k"].notna()].copy()
    v["k"] = pd.to_numeric(m.loc[v.index, "k"]).astype(int)
    v["arm"] = m.loc[v.index, "arm"]
    # #3116: `calibration_shift` is measured against the *sample minimum* of the
    # test cost, which is optimistic, so the term is inflated by however much
    # that reference overfits.  Carry the cross-fitted version beside it where
    # the run emitted one; `rule_inefficiency` needs no such twin because it
    # never references the test oracle.
    v["calibration_shift_honest"] = _optional(v, "calibration_shift_honest")
    v["regret_honest"] = _optional(v, "regret_honest")
    # #3115: how many folds contributed a cut.  NaN on the pooled arm by design
    # (it never reads a per-fold cut) and on every pre-#3115 cell.
    v["n_folds_used"] = _optional(v, "n_folds_used")
    edges = [1, *sorted(CHECKPOINTS)]
    v["window"] = pd.cut(v["n_votes"], bins=edges, labels=[f"le_{c}" for c in sorted(CHECKPOINTS)])
    v["window_hi"] = pd.cut(v["n_votes"], bins=edges, labels=sorted(CHECKPOINTS)).astype("Int64")
    v = v[v["window"].notna()]
    # Calibration's share of the per-step wall clock the user waits through.
    other = v["train_seconds"] + v["pool_score_seconds"] + v["test_score_seconds"]
    v["step_seconds"] = v["fold_seconds"] + other
    v["cal_share"] = v["fold_seconds"] / v["step_seconds"].replace(0, np.nan)
    return v


def level_table(v: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Mean regret / cost / timing per (voting mode, arm, window, K)."""
    g = (
        v.groupby(["voting", "arm", "window", "k"], observed=True)
        .agg(
            regret=("regret", "mean"),
            cost=("cost", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            rule_inefficiency=("rule_inefficiency", "mean"),
            calibration_shift=("calibration_shift", "mean"),
            calibration_shift_honest=("calibration_shift_honest", "mean"),
            regret_honest=("regret_honest", "mean"),
            n_cal_scores=("n_cal_scores", "mean"),
            fold_seconds=("fold_seconds", "mean"),
            cal_share=("cal_share", "mean"),
            degenerate_rate=("degenerate", "mean"),
            n_steps=("regret", "size"),
        )
        .reset_index()
        .sort_values(["voting", "arm", "window", "k"])
    )
    g.to_csv(agg / "folds_levels.csv", index=False)
    return g


def threshold_dispersion(v: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """``sd(threshold)`` across **seeds**, per (voting, arm, window, K) - issue #3116.

    The quantity H4 was actually reaching for.  #2897 tried to read "did more
    folds make the cut less noisy" out of the regret decomposition, which cannot
    answer it: ``rule_inefficiency`` is a signed cost gap between two cuts, not a
    variance, and its reference moves with K (see :func:`verdicts`).  The
    dispersion of the shipped threshold itself has no such problem.

    Taken **across seeds at a fixed step**, then averaged over steps - not as one
    pooled standard deviation.  A pooled sd would mix the variation this study is
    asking about (same votes, different draw -> different cut) with the variation
    it is not (the threshold legitimately moving as the trajectory collects
    votes), and the second is far larger, so the pooled number would be
    dominated by an effect that has nothing to do with K.

    Steps carrying a single seed contribute nothing (an sd of one observation is
    undefined, not zero), so a single-seed run yields an empty frame rather than
    a column of zeros that would read as "perfectly stable".
    """
    keys = ["voting", "arm", "window", "k"]
    cols = [*keys, "sd_threshold", "n_seeds", "n_steps"]
    per_step = (
        v.groupby([*keys, "env", "category", "t"], observed=True)["threshold"]
        .agg(sd="std", n_seeds="count")
        .reset_index()
    )
    per_step = per_step[(per_step["n_seeds"] >= 2) & per_step["sd"].notna()]
    if per_step.empty:
        t = pd.DataFrame(columns=cols)
        t.to_csv(agg / "folds_threshold_sd.csv", index=False)
        return t
    t = (
        per_step.groupby(keys, observed=True)
        .agg(sd_threshold=("sd", "mean"), n_seeds=("n_seeds", "max"), n_steps=("sd", "size"))
        .reset_index()
        .sort_values(keys)
    )
    t.to_csv(agg / "folds_threshold_sd.csv", index=False)
    return t


def paired_vs_baseline(v: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Paired delta of every K against production's K, per (voting, arm, window).

    Pairs at the step - each K re-cut the same model and test scores - then
    aggregates to cell means before testing, so the unit of evidence is a
    trajectory rather than a step (see :data:`CELL_KEYS`).
    """
    rows = []
    for (voting, arm), g in v.groupby(["voting", "arm"], observed=True):
        base = g[g["k"] == BASELINE_K].set_index(STEP_KEYS)
        if base.empty:
            continue
        base = base[~base.index.duplicated()]
        for k, a in g.groupby("k", observed=True):
            if k == BASELINE_K:
                continue
            a = a.set_index(STEP_KEYS)
            a = a[~a.index.duplicated()]
            terms = ["rule_inefficiency", "calibration_shift", "calibration_shift_honest"]
            j = pd.concat(
                [
                    a[["regret", "fold_seconds", "window", *terms]].add_suffix("_a"),
                    base[["regret", "fold_seconds", *terms]].add_suffix("_b"),
                ],
                axis=1,
                join="inner",
            ).reset_index()
            if j.empty:
                continue
            j["d_regret"] = j["regret_a"] - j["regret_b"]
            j["d_seconds"] = j["fold_seconds_a"] - j["fold_seconds_b"]
            j["d_rule"] = j["rule_inefficiency_a"] - j["rule_inefficiency_b"]
            j["d_shift"] = j["calibration_shift_a"] - j["calibration_shift_b"]
            j["d_shift_honest"] = j["calibration_shift_honest_a"] - j["calibration_shift_honest_b"]
            j = j.rename(columns={"window_a": "window"})
            deltas = ["d_regret", "d_seconds", "d_rule", "d_shift", "d_shift_honest"]
            for window, w in j.groupby("window", observed=True):
                cells = w.groupby(CELL_KEYS, observed=True)[deltas].mean()
                d = cells["d_regret"].to_numpy()
                p = float("nan")
                if len(d) >= 6 and not np.allclose(d, 0):
                    p = float(wilcoxon(d, zero_method="zsplit").pvalue)
                d_sec = float(cells["d_seconds"].mean())
                rows.append(
                    {
                        "voting": voting,
                        "arm": arm,
                        "window": window,
                        "k": int(k),
                        "n_cells": int(len(cells)),
                        "n_steps": int(len(w)),
                        "d_regret": float(d.mean()),
                        "d_rule_inefficiency": float(cells["d_rule"].mean()),
                        "d_calibration_shift": float(cells["d_shift"].mean()),
                        "d_calibration_shift_honest": float(cells["d_shift_honest"].mean()),
                        "d_seconds": d_sec,
                        # Regret bought per extra second of calibration.  A win
                        # that is real but costs 10x the wall clock for 0.001
                        # regret should read as a bad trade, not as a win.
                        "regret_per_extra_second": float(d.mean() / d_sec) if d_sec > 0 else float("nan"),
                        "win_rate": float((cells["d_regret"] < 0).mean()),
                        "p_wilcoxon": p,
                    }
                )
    # A run carrying only the baseline count has nothing to contrast against
    # itself, so `rows` is empty and the frame has no columns to sort by.  That
    # is the *control* arm of the live A/B (`CALIB_FOLD_COUNTS="2,$K"` collapses
    # to a single value at K=2), not a broken run - its cells are perfectly good
    # and the cross-arm analysis reads them directly.  Return the empty shape
    # rather than dying on KeyError: 'voting'.
    cols = [
        "voting",
        "arm",
        "window",
        "k",
        "n_cells",
        "n_steps",
        "d_regret",
        "d_rule_inefficiency",
        "d_calibration_shift",
        "d_calibration_shift_honest",
        "d_seconds",
        "regret_per_extra_second",
        "win_rate",
        "p_wilcoxon",
    ]
    t = pd.DataFrame(rows, columns=cols) if not rows else pd.DataFrame(rows)
    if rows:
        t = t.sort_values(["voting", "arm", "window", "k"])
    t.to_csv(agg / "folds_paired_vs_k2.csv", index=False)
    return t


def knee_table(paired: pd.DataFrame, levels: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """The smallest K within :data:`MARGIN` of the best K, per (voting, arm, window).

    "Best" is the K with the lowest mean paired delta (K=2 enters at delta 0, so
    a curve that never improves correctly returns the baseline).  The knee is
    the recommendation; ``best_k`` is only there to show how much is left on the
    table by stopping at it.
    """
    rows = []
    for (voting, arm, window), g in paired.groupby(["voting", "arm", "window"], observed=True):
        curve = dict(zip(g["k"], g["d_regret"], strict=True))
        curve[BASELINE_K] = 0.0
        ks = sorted(curve)
        best_k = min(ks, key=lambda k: curve[k])
        knee = next(k for k in ks if curve[k] <= curve[best_k] + MARGIN)
        sec = levels[(levels["voting"] == voting) & (levels["arm"] == arm) & (levels["window"] == window)]
        sec = dict(zip(sec["k"], sec["fold_seconds"], strict=True))
        base_sec = sec.get(BASELINE_K, float("nan"))
        rows.append(
            {
                "voting": voting,
                "arm": arm,
                "window": window,
                "knee_k": int(knee),
                "knee_d_regret": float(curve[knee]),
                "best_k": int(best_k),
                "best_d_regret": float(curve[best_k]),
                "left_on_table": float(curve[knee] - curve[best_k]),
                "knee_cost_x": float(sec.get(knee, float("nan")) / base_sec) if base_sec else float("nan"),
            }
        )
    t = pd.DataFrame(rows).sort_values(["voting", "arm", "window"])
    t.to_csv(agg / "folds_knee.csv", index=False)
    return t


def verdicts(paired: pd.DataFrame, knee: pd.DataFrame, levels: pd.DataFrame, disp: pd.DataFrame) -> dict:
    """Mechanically apply the plan's decision rules.

    H1 (benefit): does any K beat K=2 by more than :data:`MARGIN` in the deep
    regime, with a paired Wilcoxon under 0.05?
    H2 (cost): is the winning K's calibration wall clock within
    :data:`COST_CEILING_X` of production's?
    H3 (recommendation): the smallest K satisfying both, per voting mode -
    ``2`` (keep production) when H1 fails.
    H4 (mechanism): **re-posed after #3116.**  It used to ask whether K's benefit
    lands on ``rule_inefficiency`` rather than on ``calibration_shift``, read as
    "sampling noise in the cut fell".  That question is not answerable from those
    two terms, for two independent reasons:

    * ``rule_inefficiency`` is a *signed cost gap between two cuts*, not a
      variance.  It was negative in every row of #2897 (-0.291 at K=1 ->
      -0.080 at K=16), i.e. the trained cut beating a calibration-set "oracle"
      that overfits a handful of scores.  Rising toward zero is not variance
      falling.
    * Its reference moves with the arm.  ``c_thr`` is estimated from the pooled
      calibration set, which grows linearly in K, so as K rises ``c_thr``
      converges on the test-oracle cut - shrinking ``calibration_shift`` and
      widening ``rule_inefficiency`` **from one cause, in opposite directions**,
      with the sum pinned to regret by construction.  The anti-correlation
      #2897 reported is algebra, not evidence.

    So the arithmetic comparison is still emitted, under a name that describes
    only the arithmetic (``h4_d_rule_below_d_shift``), together with
    ``h4_reference_moves_with_k`` recording that its reference is not fixed
    across the arms.  The mechanism question is answered instead by
    ``h4_sd_threshold_by_k`` (:func:`threshold_dispersion`), which measures the
    dispersion of the shipped threshold directly.
    """
    out: dict = {"margin": MARGIN, "cost_ceiling_x": COST_CEILING_X, "baseline_k": BASELINE_K, "by_voting": {}}
    # The shipped threshold is the blended one; fall back to the raw
    # cross-calibration arm on a run without safe thresholds.
    arm = shipped_arm(paired)
    out["arm_read"] = arm
    deep = paired[(paired["arm"] == arm) & (paired["window"].astype(str).map(_window_hi) >= DEEP_MIN)]

    for voting, g in deep.groupby("voting", observed=True):
        agg_k = g.groupby("k", observed=True).agg(
            d_regret=("d_regret", "mean"),
            d_rule=("d_rule_inefficiency", "mean"),
            d_shift=("d_calibration_shift", "mean"),
            d_shift_honest=("d_calibration_shift_honest", "mean"),
            p=("p_wilcoxon", "max"),
            d_seconds=("d_seconds", "mean"),
        )
        lv = levels[(levels["voting"] == voting) & (levels["arm"] == arm)]
        base_sec = float(lv[lv["k"] == BASELINE_K]["fold_seconds"].mean())
        sec_x = lv.groupby("k", observed=True)["fold_seconds"].mean() / base_sec if base_sec else None

        beats = agg_k[(agg_k["d_regret"] <= -MARGIN) & (agg_k["p"] < 0.05)]
        affordable = beats
        if sec_x is not None:
            affordable = beats[[sec_x.get(k, np.inf) <= COST_CEILING_X for k in beats.index]]
        recommended = int(min(affordable.index)) if len(affordable) else BASELINE_K

        best = agg_k["d_regret"].idxmin() if len(agg_k) else BASELINE_K

        # #3116's guard: the decomposition's reference is estimated from the
        # calibration set, so if that set's size moves with K the two terms are
        # not independent readings and must not be reported as if they were.
        # In this study it always does move - that is what K *is* - so this
        # flag is expected to be true, and its job is to make the caveat
        # travel with the number instead of living in someone's memory.
        n_cal_by_k = lv.groupby("k", observed=True)["n_cal_scores"].mean().dropna()
        reference_moves = bool(len(n_cal_by_k) > 1 and float(n_cal_by_k.max() - n_cal_by_k.min()) > 0.0)

        # The mechanism question, asked of the threshold directly.
        sd = disp[(disp["voting"] == voting) & (disp["arm"] == arm)]
        sd = sd[sd["window"].astype(str).map(_window_hi) >= DEEP_MIN]
        sd_by_k = sd.groupby("k", observed=True)["sd_threshold"].mean().dropna() if len(sd) else pd.Series(dtype=float)
        sd_falls = None
        if BASELINE_K in sd_by_k.index and best in sd_by_k.index and best != BASELINE_K:
            sd_falls = bool(sd_by_k[best] < sd_by_k[BASELINE_K])

        out["by_voting"][voting] = {
            "h1_any_k_beats_baseline": bool(len(beats)),
            "h1_ks_beating_baseline": [int(k) for k in beats.index],
            "h2_ks_also_affordable": [int(k) for k in affordable.index],
            "h3_recommended_k": recommended,
            "h3_kept_production": recommended == BASELINE_K,
            "best_k_ignoring_cost": int(best),
            "best_d_regret": float(agg_k["d_regret"].min()) if len(agg_k) else 0.0,
            "cost_x_at_recommended": float(sec_x.get(recommended, float("nan"))) if sec_x is not None else None,
            # Arithmetic only - see this function's docstring.  The old name for
            # this key asserted a mechanism the terms cannot carry (#3116).
            "h4_d_rule_below_d_shift": bool(len(agg_k) and agg_k.loc[best, "d_rule"] <= agg_k.loc[best, "d_shift"]),
            "h4_reference_moves_with_k": reference_moves,
            "h4_sd_threshold_by_k": {int(k): float(x) for k, x in sd_by_k.items()},
            "h4_sd_threshold_falls_at_best_k": sd_falls,
            "d_rule_at_best": float(agg_k.loc[best, "d_rule"]) if len(agg_k) else 0.0,
            "d_shift_at_best": float(agg_k.loc[best, "d_shift"]) if len(agg_k) else 0.0,
            "d_shift_honest_at_best": float(agg_k.loc[best, "d_shift_honest"]) if len(agg_k) else 0.0,
        }
    out["knee_by_window"] = knee[knee["arm"] == arm].to_dict(orient="records")
    return out


def _window_hi(label) -> int:
    try:
        return int(str(label).removeprefix("le_"))
    except ValueError:
        return 0


def shipped_arm(v: pd.DataFrame) -> str:
    """The arm the verdict reads: the closest thing in the run to what users get.

    ``anchored`` first (#3116): :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold`
    has been the shipped path since the 2026-08-05 population-anchored run, and
    it is the only arm in which K moves *both* halves of the threshold.  ``blend``
    is the retired ``cap50`` mix-in, kept so pre-#3116 runs still read; ``xcal``
    is the raw cut, the fallback for a run without safe thresholds.
    """
    arms = set(v["arm"])
    return next((name for name in ("anchored", "blend", "xcal") if name in arms), "xcal")


#: The fold-**count** axis reads one rule at a time.  #3115's challenger arms
#: re-cut the same prefix under a *different* rule, so leaving them in would make
#: every K-vs-K table a mixture of two questions - and the knee, the cost curve
#: and H4 would all be computed over rows that are not the arm they name.
COUNT_AXIS_ARMS: tuple[str, ...] = ("xcal", "blend", "anchored")


def ab_check(screen: pd.DataFrame, ab_dirs: list[Path], agg: Path) -> pd.DataFrame:
    """Does a run that *lives* at K reproduce the screen's delta for K?

    The screen holds the trajectory fixed, so it cannot see the votes a
    different fold count would have collected.  Each A/B arm is a full run at
    its own K, which means the arms are **not** paired with each other or with
    the screen: what is comparable is each run's own delta against its own K=2
    counterfactual, differenced across runs (Mann-Whitney over cell means).

    A screen that over- or under-states the live effect is still useful - but
    only if that is reported, so this table is part of the deliverable whether
    or not it agrees.
    """
    rows = []
    for d in ab_dirs:
        live = fold_frame(load_cells(Path(d) / "cells"))
        if live.empty:
            common.log(f"  A/B dir {d}: no fold rows; skipped")
            continue
        # The arm's live fold count is whichever K is not the baseline.
        ks = sorted(set(live["k"]) - {BASELINE_K})
        if not ks:
            common.log(f"  A/B dir {d}: only the baseline count present; skipped")
            continue
        k = ks[-1]
        arm = shipped_arm(live)
        live = live[live["arm"] == arm]
        for voting, g in live.groupby("voting", observed=True):
            live_d = _cell_deltas(g, k)
            screen_d = _cell_deltas(screen[(screen["voting"] == voting) & (screen["arm"] == arm)], k)
            if live_d.empty or screen_d.empty:
                continue
            p = float("nan")
            if min(len(live_d), len(screen_d)) > 5:
                p = float(mannwhitneyu(live_d, screen_d).pvalue)
            rows.append(
                {
                    "voting": voting,
                    "arm": arm,
                    "k": int(k),
                    "live_d_regret": float(live_d.mean()),
                    "screen_d_regret": float(screen_d.mean()),
                    "live_minus_screen": float(live_d.mean() - screen_d.mean()),
                    "n_live_cells": int(len(live_d)),
                    "n_screen_cells": int(len(screen_d)),
                    "p_unpaired": p,
                    "screen_agrees": bool(abs(live_d.mean() - screen_d.mean()) <= MARGIN),
                }
            )
    t = pd.DataFrame(rows)
    if not t.empty:
        t.to_csv(agg / "folds_ab_check.csv", index=False)
    return t


def _cell_deltas(v: pd.DataFrame, k: int) -> pd.Series:
    """Cell-mean paired regret delta of fold count *k* vs the baseline, deep windows."""
    v = v[v["window_hi"] >= DEEP_MIN]
    if v.empty:
        return pd.Series(dtype=float)
    a = v[v["k"] == k].set_index(STEP_KEYS)["regret"]
    b = v[v["k"] == BASELINE_K].set_index(STEP_KEYS)["regret"]
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    if j.empty:
        return pd.Series(dtype=float)
    j = j.reset_index()
    # One value per trajectory: the deep windows are pooled, so the cell is
    # (env, category, seed) rather than the four-key CELL_KEYS used elsewhere.
    return (j["a"] - j["b"]).groupby([j["env"], j["category"], j["seed"]]).mean()


def write_report(results: Path, levels, paired, knee, verd, ab, disp, combine, degen, checks, combine_verd) -> None:
    lines = [
        "# Calibration fold study (#2897 / #3116 / #3115)",
        "",
        "How much does raising `calibrate_count` above production's 2 reduce",
        "oracle-regret, and what does it cost in calibration wall clock?",
        "",
        "Every fold count is a **nested prefix** of one Kmax calibration, so each K",
        "re-cuts the same votes, the same model and the same held-out test scores:",
        "the contrasts below are paired at the step and are exactly what a live run",
        "at that K would have computed *for those votes*.  What they cannot show is",
        "the votes a different K would have collected - that is the A/B section.",
        "",
        "Design + decision rules: `docs/experiments/calibration-fold-count/REPORT.md`.",
        "",
        "Two axes share these rows, because every arm re-cuts the *same* trained fold",
        "prefix.  Sections up to the A/B check sweep the fold **count** at one rule",
        "(#2897, instrumented by #3116); the last section sweeps the **combine rule**",
        "at fixed count (#3115).",
        "",
        "## Verdicts (mechanical; read the tables before believing them)",
        "",
        "```json",
        json.dumps(verd, indent=2),
        "```",
        "",
        "## Levels per (voting mode, arm, window, K)",
        "",
        _md(levels),
        "",
        f"## Paired deltas vs K={BASELINE_K} (negative = more folds is better)",
        "",
        _md(paired),
        "",
        f"## Knee: smallest K within {MARGIN} regret of the best K",
        "",
        _md(knee),
        "",
        "## Threshold dispersion: sd(threshold) across seeds, per K",
        "",
        "The direct form of H4's question (#3116).  `rule_inefficiency` is a",
        "signed cost gap between two cuts, not a variance, and its reference is",
        "estimated from a calibration set that grows with K - so the two",
        "decomposition terms move in opposite directions from one cause and",
        "cannot answer 'did the cut get less noisy'.  This can: it is the spread",
        "of the shipped threshold across seeds at a fixed step, averaged over",
        "steps.",
        "",
        _md(disp) if len(disp) else "_No step carries >=2 seeds; dispersion is undefined for this run._",
        "",
        "## A/B check: does a run that lives at K reproduce the screen?",
        "",
        _md(ab) if len(ab) else "_No A/B run dirs passed; screen only._",
        "",
        *folds_combine_3115.report_lines(combine, degen, checks, combine_verd, DEEP_MIN),
    ]
    (results / "REPORT.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    results = common.RESULTS
    agg = results / "agg"
    agg.mkdir(parents=True, exist_ok=True)

    v = fold_frame(load_cells(results / "cells"))
    if v.empty:
        common.log("no fold-count rows found; was CALIB_FOLD_COUNTS set on the run?")
        return 1
    common.log(f"fold counts present: {sorted(set(v['k']))}; arms: {sorted(set(v['arm']))}")
    if BASELINE_K not in set(v["k"]):
        common.log(f"ERROR: no K={BASELINE_K} rows; every contrast here is against production's count")
        return 1

    # The fold-COUNT axis reads one rule per arm; #3115's challengers re-cut the
    # same prefix under a different rule and belong to the other axis entirely.
    count_v = v[v["arm"].isin(COUNT_AXIS_ARMS)]
    levels = level_table(count_v, agg)
    disp = threshold_dispersion(count_v, agg)
    if disp.empty:
        common.log("  sd(threshold): no step carries >=2 seeds; H4's direct instrument is unavailable")
    paired = paired_vs_baseline(count_v, agg)
    knee = knee_table(paired, levels, agg)
    verd = verdicts(paired, knee, levels, disp)
    ab = ab_check(count_v, [Path(d) / "results" for d in argv], agg) if argv else pd.DataFrame()
    verd["ab_check"] = ab.to_dict(orient="records") if len(ab) else None

    # The fold-COMBINE axis (#3115): same rows, contrasted across rules at fixed K.
    combine = folds_combine_3115.contrast_table(v, agg)
    degen = folds_combine_3115.degenerate_table(v, agg)
    checks = folds_combine_3115.control_checks(v)
    combine_verd = folds_combine_3115.verdicts(combine, DEEP_MIN, MARGIN)
    combine_verd["control_checks"] = checks
    verd["combine_3115"] = combine_verd
    if not checks.get("k1_score_space_is_pooled", True):
        common.log(
            f"  ACCEPTANCE FAILED: at K=1 the score-space arm must reproduce the pooled cut exactly; "
            f"{checks.get('k1_mismatches')} of {checks.get('k1_n_steps')} steps disagree"
        )

    (results / "summary.json").write_text(json.dumps(verd, indent=2))
    write_report(results, levels, paired, knee, verd, ab, disp, combine, degen, checks, combine_verd)
    common.log(f"wrote {results / 'summary.json'} and {results / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

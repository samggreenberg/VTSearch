#!/usr/bin/env python
"""#3314: does more cross-calibration ever pay, and what does it cost?

Reads the stage-A screen (``launch_folds_3314.sh screen``) and applies the
decision rules pre-registered in
``docs/experiments/calibration-fold-count-3310/PLAN.md``.  Stage B's arms, when
the gate books them, come in through ``--ab`` and are contrasted across runs.

**Why this is not `analyze_folds_2897.py`.**  That analyzer owns the #2897 /
#3116 / #3115 questions and archived runs are read through it; its verdicts are
cited in two merged reports.  This study asks a different question of the same
rows and answers it under different constants -- headline on ``cost`` rather
than ``regret``, the #3287 vote bands, a 1.5x wall-clock ceiling rather than
#2897's 4x calibration-only one, a pointwise harm gate, and standard errors
bootstrapped over cells rather than a Wilcoxon over them.  Changing those in
place would silently restate the older studies' verdicts.  The row loading and
the arm regex are imported from it, so the two cannot drift about what a
``folds_k{K}_*`` row means.

Three things this analyzer refuses to do.

**It does not price K off `fold_seconds`.**  That column is the fold *fits* plus
the conformal rule's overhead.  A live run at K also scores the sim set once per
fold and fits one anchored mixture per fold, both K-proportional, both paid
inside the safe-threshold block that no other timing column covers.  #3314 added
``cal_seconds`` for exactly this; reading ``fold_seconds`` as the price of K
under-states it, and an affordability ceiling read off it would wave through an
arm the user waits twice as long for.

**It does not decompose regret into ``rule_inefficiency + calibration_shift``.**
The two sum to regret by construction and ``calibration_shift``'s reference is
estimated from a calibration set whose size *is* the swept axis, so the two
terms slide against each other from one cause.  #2897 read that anti-correlation
as a mechanism; #3116 established it is algebra.  Levels are read off ``cost``
and ``regret_honest`` only, and the mechanism question -- did the cut get less
noisy -- is asked of ``sd(threshold)`` directly.

**It does not bootstrap over steps.**  Consecutive steps of one trajectory share
a model and nearly all their votes.  Every standard error here resamples
**cells**; steps are collapsed to a cell mean inside the band first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

common.setup_env()

import curves  # noqa: E402
from analyze_folds_2897 import FOLD_RE, load_cells  # noqa: E402

#: Production's fold count.  Every contrast is a difference from it, and it is
#: measured on THIS grid under THIS code: the screen lives at 2, so the K=2 arm
#: is the run's own trajectory rather than a level quoted from another study.
BASELINE_K = 2

#: The benefit margin, in ``FPR + FNR`` units (inclusion 0 weights both at 1, so
#: cost lives on 0..2).  0.005 is half a percentage point of combined error
#: rate: below it a fold-count change is not worth a user's retrain latency
#: however clean its interval.  Pre-registered in the PLAN, not tuned here.
MARGIN = 0.005

#: How much worse than K=2 a fold count may be in ANY band and still ship.  The
#: margin PR #2891 pre-registered for this family of decisions.  Pointwise on
#: purpose: a K can win early and lose late, and the adaptive schedule exists
#: precisely so it does not have to pay the late price -- but a FIXED K that
#: loses a band is out.
HARM_TOLERANCE = 0.01

#: The user-facing cost ceiling: median per-step wall clock at K, as a multiple
#: of K=2's, inside the bands where K applies.  This replaces #2897's 4x
#: calibration-only ceiling, which was pre-registered without reference to
#: `cal_share` and admitted an arm that spent 76% of every step calibrating --
#: the lesson recorded in that study's Limitations.  It is banded because the
#: adaptive schedule concentrates its extra folds exactly where fits are
#: cheapest, and a global ratio would hide that.
COST_CEILING_X = 1.5

#: Vote bands (the #3287 set).  The benefit is predicted to DECAY inside the
#: horizon -- variance reduction is worth most where per-fold noise is largest,
#: which is where the labelset is smallest -- so a pooled number would be an
#: average across the decay the adaptive schedule exists to exploit.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("early 1-25", 1, 25),
    ("mid 26-60", 26, 60),
    ("late 61-100", 61, 100),
    ("deep 101-150", 101, 150),
)

#: The adaptive family, pre-registered in the PLAN:
#: ``K(n_votes) = K_early while n_votes < N_cut, else 2``.
SCHEDULE_K_EARLY: tuple[int, ...] = (4, 6, 8)
SCHEDULE_N_CUT: tuple[int, ...] = (25, 60)

#: Bootstrap resamples for the paired standard errors.
N_BOOT = 2000

#: Pairing unit for stage A: one step of one trajectory.  Every K re-cuts it,
#: so the contrast is exact and the trajectory noise cancels.
STEP_KEYS = ("dataset", "category", "seed", "geometry", "t")
#: Bootstrap unit: a cell.  Never a step (see the module docstring).
CELL_KEYS = ("dataset", "category", "seed", "geometry")

#: The arm the decision reads: production's rule since the 2026-08-05
#: population-anchored run.  ``xcal`` is kept as the #2897 replication.
SHIPPED_ARM = "anchored"
POOLED_ARM = "xcal"


def geometry_of(row) -> str:
    """``siglip/whole_image``-style label for one row's (embedder, style) corner.

    The pair name collapses to its LEARN half: that is the space the detector
    trains, scores and sorts in, and this grid holds the opening fixed at a
    SigLIP text sort in every cell, so the SigLIP half of the pair is not a
    difference between arms.
    """
    emb = str(row["embedder"])
    learn = emb.partition("+")[2] or emb
    return f"{learn}/{row['style']}"


def voting_mode(geometry: str) -> str:
    """Region voting is a property of the STYLE, not of the dataset.

    ``max_patch`` on a boxed dataset pools a dragged ground-truth box;
    ``whole_image`` does not, whatever the embedder can emit.  Reading it off
    the dataset name is the trap behind #2877, #2905 and #2897's own errata.
    """
    return "region" if geometry.endswith("/max_patch") else "binary"


def band_of(votes: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=votes.index, dtype="object")
    for name, lo, hi in BANDS:
        out = out.mask((votes >= lo) & (votes <= hi), name)
    return out


BAND_ORDER = [b[0] for b in BANDS]


def band_bounds(name: str) -> tuple[int, int]:
    for label, lo, hi in BANDS:
        if label == name:
            return lo, hi
    raise KeyError(name)


def fold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The fold-count arms, with K, arm, geometry, band and the cost model attached."""
    if df.empty:
        return df
    m = df["gmm_variant"].astype(str).str.extract(FOLD_RE)
    v = df[m["k"].notna()].copy()
    if v.empty:
        return v
    v["k"] = pd.to_numeric(m.loc[v.index, "k"]).astype(int)
    v["arm"] = m.loc[v.index, "arm"]
    v["geometry"] = v.apply(geometry_of, axis=1)
    v["mode"] = v["geometry"].map(voting_mode)
    v["n_votes"] = v["n_good"] + v["n_bad"]
    v["band"] = band_of(v["n_votes"])
    v = v[v["band"].notna()]

    # The cost model, in the units the ceiling is written in.  `cal_seconds` is
    # the WHOLE calibration wall clock at K (fold fits + one haystack scoring
    # pass per fold + production's anchored fit + the conformal rule's
    # overhead); the other three columns are the rest of the step and none of
    # them overlaps it.  A run from before #3314 has no `cal_seconds`, so it
    # falls back to `fold_seconds` and SAYS SO in `cost_model`, rather than
    # quietly pricing K at a third of what it costs.
    if "cal_seconds" in v.columns and v["cal_seconds"].notna().any():
        v["cal_seconds_used"] = v["cal_seconds"]
        v["cost_model"] = "cal_seconds"
    else:
        v["cal_seconds_used"] = v["fold_seconds"]
        v["cost_model"] = "fold_seconds (pre-#3314 run: fold FITS only)"

    # THE DENOMINATOR IS THE APP'S RETRAIN, NOT THE HARNESS CELL.  A screen step
    # also computes six fold counts x eight arms of counterfactual rows, and a
    # user waits through none of that - a ratio taken over the cell's own wall
    # clock would divide by the study's instrumentation and report every K as
    # nearly free.  So the step is reconstructed from the pieces the app itself
    # performs on a retrain: fit the head, score the haystack with it, calibrate,
    # score the pool for display.
    #
    # `test_score_seconds` is deliberately EXCLUDED: scoring a held-out test set
    # is eval-only work that no app step does, and leaving it in would inflate
    # the denominator and make every K look cheaper than it is.  It is kept as
    # `harness_step_seconds` beside the app number so the two can be compared.
    final = _optional(v, "final_score_seconds").fillna(0.0)
    app_other = v["train_seconds"] + final + v["pool_score_seconds"]
    v["step_seconds"] = v["cal_seconds_used"] + app_other
    v["harness_step_seconds"] = v["step_seconds"] + v["test_score_seconds"]
    v["cal_share"] = v["cal_seconds_used"] / v["step_seconds"].replace(0, np.nan)
    return v


def _optional(v: pd.DataFrame, col: str) -> pd.Series:
    """*col* when the run emitted it, else an all-NaN stand-in of the right shape.

    The #3314 timing columns postdate every archived run, so an analyzer that
    must still read one cannot assume they are there.
    """
    return v[col] if col in v.columns else pd.Series(np.nan, index=v.index, dtype=float)


def _boot_se(d: np.ndarray, rng: np.random.Generator) -> float:
    if d.size < 2:
        return float("nan")
    return float(rng.choice(d, size=(N_BOOT, d.size), replace=True).mean(axis=1).std(ddof=1))


def paired_vs_baseline(v: pd.DataFrame, metric: str, rng: np.random.Generator) -> pd.DataFrame:
    """Paired Delta(K - 2) per (geometry, band, K), bootstrapped over cells.

    Paired **within the step**: the fold counts are nested prefixes of one Kmax
    calibration, so every K in a step re-cuts the same votes, the same final
    model and the same held-out test scores.  That is what makes the screen
    exact.  The steps are then collapsed to a cell mean inside the band before
    anything is resampled, because a trajectory - not a step - is the unit of
    evidence.
    """
    rows: list[dict] = []
    if v.empty:
        return pd.DataFrame()
    base = v[v["k"] == BASELINE_K].set_index(list(STEP_KEYS))[metric]
    base = base[~base.index.duplicated()]
    for (geom, k), g in v.groupby(["geometry", "k"], dropna=False):
        if k == BASELINE_K:
            continue
        a = g.set_index(list(STEP_KEYS))
        a = a[~a.index.duplicated()]
        common_idx = a.index.intersection(base.index)
        if len(common_idx) == 0:
            continue
        j = pd.DataFrame(
            {
                "d": a.loc[common_idx, metric].to_numpy(dtype=float) - base.loc[common_idx].to_numpy(dtype=float),
                "band": a.loc[common_idx, "band"].to_numpy(),
            },
            index=common_idx,
        ).reset_index()
        for band, w in j.groupby("band", dropna=False):
            cells = w.groupby(list(CELL_KEYS), dropna=False)["d"].mean().to_numpy(dtype=float)
            if cells.size == 0:
                continue
            se = _boot_se(cells, rng)
            rows.append(
                {
                    "geometry": geom,
                    "mode": voting_mode(str(geom)),
                    "band": band,
                    "k": int(k),
                    "metric": metric,
                    "delta": float(cells.mean()),
                    "se": se,
                    "n_cells": int(cells.size),
                    "n_steps": int(len(w)),
                    "win_rate": float((cells < 0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        # "Resolved" is the honest phrasing of significance here: a difference
        # smaller than twice its own standard error is not resolvable on this
        # sample, and saying so is more useful than a decimal that implies it is.
        out["resolved"] = out["delta"].abs() > 2 * out["se"]
        out["beats_margin"] = out["delta"] <= -np.maximum(MARGIN, 2 * out["se"])
        out = out.sort_values(["geometry", "band", "k"]).reset_index(drop=True)
    return out


def cost_table(v: pd.DataFrame) -> pd.DataFrame:
    """The banded price of K, in the units the ceiling is written in.

    The headline is ``step_ratio``: the MEDIAN, over steps in the band, of this
    K's whole per-step wall clock divided by K=2's on the same step.  Paired
    within the step for the same reason the benefit is - it is the same
    calibration, one machine, one process, one cache state - and a median rather
    than a mean because a single scheduler stall on a shared cluster moves a
    mean and cannot move a median.

    ``cal_share`` beside it is what the number means to a user: the fraction of
    each retrain they spend waiting for calibration.  #2897's ceiling was
    written on the calibration clock alone and admitted an arm at
    ``cal_share`` 0.885; a ratio of the STEP cannot do that.
    """
    if v.empty:
        return pd.DataFrame()
    keys = list(STEP_KEYS)
    base = v[v["k"] == BASELINE_K].set_index(keys)
    base = base[~base.index.duplicated()]
    rows: list[dict] = []
    for (geom, k), g in v.groupby(["geometry", "k"], dropna=False):
        a = g.set_index(keys)
        a = a[~a.index.duplicated()]
        idx = a.index.intersection(base.index)
        if len(idx) == 0:
            continue
        j = pd.DataFrame(
            {
                "step_ratio": a.loc[idx, "step_seconds"].to_numpy(dtype=float)
                / base.loc[idx, "step_seconds"].to_numpy(dtype=float),
                "cal_ratio": a.loc[idx, "cal_seconds_used"].to_numpy(dtype=float)
                / base.loc[idx, "cal_seconds_used"].to_numpy(dtype=float),
                "cal_share": a.loc[idx, "cal_share"].to_numpy(dtype=float),
                "cal_share_k2": base.loc[idx, "cal_share"].to_numpy(dtype=float),
                "cal_seconds": a.loc[idx, "cal_seconds_used"].to_numpy(dtype=float),
                "step_seconds": a.loc[idx, "step_seconds"].to_numpy(dtype=float),
                "band": a.loc[idx, "band"].to_numpy(),
            },
            index=idx,
        ).reset_index()
        for band, w in j.groupby("band", dropna=False):
            rows.append(
                {
                    "geometry": geom,
                    "mode": voting_mode(str(geom)),
                    "band": band,
                    "k": int(k),
                    "step_ratio": float(w["step_ratio"].median()),
                    "step_ratio_p90": float(w["step_ratio"].quantile(0.9)),
                    "cal_ratio": float(w["cal_ratio"].median()),
                    "cal_share": float(w["cal_share"].median()),
                    "cal_share_k2": float(w["cal_share_k2"].median()),
                    "cal_seconds": float(w["cal_seconds"].median()),
                    "step_seconds": float(w["step_seconds"].median()),
                    "n_steps": int(len(w)),
                    "affordable": bool(w["step_ratio"].median() <= COST_CEILING_X),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["geometry", "band", "k"]).reset_index(drop=True)
    return out


def ship_table(paired: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    """The three pre-registered rules, applied mechanically per (geometry, K).

    1. **Benefit** - beats K=2 by more than both :data:`MARGIN` and 2 SE in at
       least one band.
    2. **No harm** - not worse than K=2 by more than :data:`HARM_TOLERANCE` in
       ANY band.
    3. **Affordable** - banded median per-step wall clock within
       :data:`COST_CEILING_X` of K=2's, in every band the fixed K applies to,
       which for a fixed K is all of them.

    A K is a ship candidate only when all three hold.  Reported per K rather
    than collapsed to a winner, because "2 is fine everywhere, here is the price
    sheet" is one of the pre-registered outcomes and it is only readable off the
    full table.
    """
    if paired.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (geom, k), g in paired.groupby(["geometry", "k"], dropna=False):
        c = cost[(cost["geometry"] == geom) & (cost["k"] == k)] if not cost.empty else pd.DataFrame()
        best_i = g["delta"].idxmin()
        worst_i = g["delta"].idxmax()
        worst = float(g.loc[worst_i, "delta"])
        worst_se = float(g.loc[worst_i, "se"])
        benefit_bands = sorted(g.loc[g["beats_margin"], "band"].astype(str))
        ratios = c["step_ratio"].to_numpy(dtype=float) if not c.empty else np.array([np.nan])
        rows.append(
            {
                "geometry": geom,
                "mode": voting_mode(str(geom)),
                "k": int(k),
                "best_band": str(g.loc[best_i, "band"]),
                "best_delta": float(g.loc[best_i, "delta"]),
                "best_se": float(g.loc[best_i, "se"]),
                "worst_band": str(g.loc[worst_i, "band"]),
                "worst_delta": worst,
                "worst_se": worst_se,
                "max_step_ratio": float(np.nanmax(ratios)) if ratios.size else float("nan"),
                "rule1_benefit": bool(len(benefit_bands) > 0),
                "benefit_bands": ",".join(benefit_bands),
                "rule2_no_harm": bool(worst <= HARM_TOLERANCE),
                # A gate landing within 2 SE of its own constant decided
                # nothing; the boolean beside it is then where the noise fell.
                "harm_gate_indeterminate": bool(abs(worst - HARM_TOLERANCE) < 2 * worst_se),
                "rule3_affordable": bool(ratios.size and np.nanmax(ratios) <= COST_CEILING_X),
                "ship_candidate": bool(
                    len(benefit_bands) > 0
                    and worst <= HARM_TOLERANCE
                    and ratios.size
                    and np.nanmax(ratios) <= COST_CEILING_X
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["geometry", "k"]).reset_index(drop=True)


def schedule_table(v: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Score every ``(K_early, N_cut)`` in the pre-registered adaptive family.

    ``K(n_votes) = K_early while n_votes < N_cut, else 2``.

    Scored at the **step**, not by stitching banded aggregates together.  The
    screen carries every K's row for every step, so the schedule's
    trajectory-fixed counterfactual is exactly constructible: take the
    ``K_early`` row where the rule says ``K_early`` and the ``K=2`` row where it
    says 2, and difference against K=2 as usual.  Outside the early region the
    schedule *is* production, so its delta there is identically zero and its
    cost ratio identically one - which is the whole claim an adaptive count
    makes, and the reason it must not be approximated by "the bands below the
    cut".  It also keeps the PLAN's strict ``<`` exactly, instead of rounding it
    to a band edge one vote away.

    The pick is mechanical: among the schedules whose every band clears the
    ceiling and whose harm gate holds, the largest overall benefit.
    """
    rows: list[dict] = []
    if v.empty:
        return pd.DataFrame()
    keys = list(STEP_KEYS)
    base = v[v["k"] == BASELINE_K].set_index(keys)
    base = base[~base.index.duplicated()]
    for k_early in SCHEDULE_K_EARLY:
        early = v[v["k"] == k_early].set_index(keys)
        early = early[~early.index.duplicated()]
        idx = early.index.intersection(base.index)
        if len(idx) == 0:
            continue
        for n_cut in SCHEDULE_N_CUT:
            votes = base.loc[idx, "n_votes"].to_numpy(dtype=float)
            use_early = votes < n_cut
            cost_sched = np.where(use_early, early.loc[idx, "cost"], base.loc[idx, "cost"])
            secs_sched = np.where(use_early, early.loc[idx, "step_seconds"], base.loc[idx, "step_seconds"])
            j = pd.DataFrame(
                {
                    "d": cost_sched - base.loc[idx, "cost"].to_numpy(dtype=float),
                    "ratio": secs_sched / base.loc[idx, "step_seconds"].to_numpy(dtype=float),
                    "band": base.loc[idx, "band"].to_numpy(),
                },
                index=idx,
            ).reset_index()  # `geometry` is a STEP_KEYS index level, so it arrives as a column here
            for geom, g in j.groupby("geometry", dropna=False):
                per_band = []
                for band, w in g.groupby("band", dropna=False):
                    cells = w.groupby(list(CELL_KEYS), dropna=False)["d"].mean().to_numpy(dtype=float)
                    per_band.append(
                        {
                            "band": str(band),
                            "delta": float(cells.mean()),
                            "se": _boot_se(cells, rng),
                            "ratio": float(w["ratio"].median()),
                        }
                    )
                if not per_band:
                    continue
                overall = g.groupby(list(CELL_KEYS), dropna=False)["d"].mean().to_numpy(dtype=float)
                worst = max(b["delta"] for b in per_band)
                best = min(per_band, key=lambda b: b["delta"])
                max_ratio = max(b["ratio"] for b in per_band)
                rows.append(
                    {
                        "geometry": geom,
                        "mode": voting_mode(str(geom)),
                        "k_early": k_early,
                        "n_cut": n_cut,
                        "overall_delta": float(overall.mean()),
                        "overall_se": _boot_se(overall, rng),
                        "best_band": best["band"],
                        "best_delta": best["delta"],
                        "best_se": best["se"],
                        "worst_delta": worst,
                        "max_step_ratio": max_ratio,
                        "n_cells": int(overall.size),
                        "rule1_benefit": bool(
                            any(b["delta"] <= -max(MARGIN, 2 * b["se"]) for b in per_band if np.isfinite(b["se"]))
                        ),
                        "rule2_no_harm": bool(worst <= HARM_TOLERANCE),
                        "rule3_affordable": bool(max_ratio <= COST_CEILING_X),
                        "eligible": bool(
                            any(b["delta"] <= -max(MARGIN, 2 * b["se"]) for b in per_band if np.isfinite(b["se"]))
                            and worst <= HARM_TOLERANCE
                            and max_ratio <= COST_CEILING_X
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["geometry", "k_early", "n_cut"]).reset_index(drop=True)
    return out


def pick_stage_b(ship: pd.DataFrame, sched: pd.DataFrame) -> dict:
    """The gate, and the arms it books - decided by the rules, not by eye.

    The PLAN's gate is band-local: *"some K clearing the benefit margin in some
    band within the cost ceiling"*.  A fixed K has to clear the ceiling in every
    band because it applies in every band; a schedule only has to clear it where
    it raises the count, which is the entire reason the ceiling is banded and
    the entire reason the adaptive arm exists.  So the two are asked
    **independently** and either can open the gate.

    (An earlier draft of this function required a fixed-K candidate before it
    would even look at the schedules, which is stricter than the PLAN.  Fixed
    on 2026-08-28, *after* the screen had run - and the verdict is identical
    either way, because no schedule cleared rule 3 either.  Said out loud
    because a decision rule edited after seeing the data has to be.)

    ``k_best`` is the SMALLEST fixed K that is a ship candidate, from the
    geometry where the most K's clear (ties to the smaller K).  ``k_adaptive``
    is the eligible schedule with the largest overall benefit, ties to the
    smaller ``K_early`` then the smaller ``N_cut``.

    When nothing clears, the gate is closed and that is the study's answer:
    folds buy nothing worth their price even where they are cheapest.  The
    report then ships without stage B, which is a result and not a shortfall.
    """
    out: dict = {
        "gate_open": False,
        "k_best": None,
        "k_best_geometry": None,
        "schedule": None,
        "schedule_geometry": None,
        "reason": "",
    }
    cands = ship[ship["ship_candidate"]] if not ship.empty else pd.DataFrame()
    elig = sched[sched["eligible"]] if not sched.empty else pd.DataFrame()

    if not cands.empty:
        counts = cands.groupby("geometry")["k"].count().sort_values(ascending=False)
        geom = str(counts.index[0])
        out["k_best"] = int(cands[cands["geometry"] == geom]["k"].min())
        out["k_best_geometry"] = geom
    if not elig.empty:
        top = elig.sort_values(["overall_delta", "k_early", "n_cut"], ascending=[True, True, True]).iloc[0]
        out["schedule"] = f"{int(top['k_early'])}@{int(top['n_cut'])}"
        out["schedule_geometry"] = str(top["geometry"])

    out["gate_open"] = bool(len(cands) or len(elig))
    if out["gate_open"]:
        out["reason"] = (
            f"{len(cands)} (geometry, K) pairs and {len(elig)} (geometry, schedule) pairs cleared all three rules"
        )
    else:
        # Say WHICH rule closed it.  "Nothing cleared" is compatible with a flat
        # screen and with a large benefit priced out of reach, and those are
        # completely different findings - the first says stop, the second says
        # make the calibration cheaper.
        beat = ship[ship["rule1_benefit"]] if not ship.empty else pd.DataFrame()
        if beat.empty:
            out["reason"] = "no fold count beat K=2 by the margin in any band: the screen is flat"
        else:
            worst = float(beat["max_step_ratio"].min())
            out["reason"] = (
                f"{len(beat)} (geometry, K) pairs beat the margin, and every one of them "
                f"failed the {COST_CEILING_X}x step ceiling (cheapest was {worst:.2f}x); "
                f"no schedule cleared it either. The benefit is real and priced out of reach."
            )
    return out


def threshold_spread(v: pd.DataFrame) -> pd.DataFrame:
    """``sd(threshold)`` ACROSS SEEDS at a fixed (geometry, category, step), banded.

    The variance-reduction mechanism, observed directly.  The shipped combined
    quantile is a mean of K i.i.d. per-fold statistics, so its variance should
    fall like 1/K while its mean does not move - and that is a statement about
    the DISPERSION of the cut, which no regret decomposition can make (#3116).

    Taken across seeds at a fixed step and then averaged over steps, never
    pooled: a pooled sd would mix the variation this asks about (same votes,
    different draw) with the threshold legitimately moving as votes arrive, and
    the second is far larger.
    """
    d = v.dropna(subset=["threshold"]).copy()
    if d.empty:
        return pd.DataFrame()
    per_step = (
        d.groupby(["geometry", "mode", "band", "k", "dataset", "category", "t"], dropna=False)["threshold"]
        .agg(["std", "count"])
        .reset_index()
    )
    per_step = per_step[(per_step["count"] >= 2) & per_step["std"].notna()]
    if per_step.empty:
        return pd.DataFrame()
    return (
        per_step.groupby(["geometry", "mode", "band", "k"], dropna=False)["std"]
        .agg(sd_threshold="mean", n_steps="size")
        .reset_index()
        .sort_values(["geometry", "band", "k"])
    )


def degenerate_table(v: pd.DataFrame) -> pd.DataFrame:
    """``n_folds_used`` per K: how much of the fold budget actually contributed.

    A fold that saw one class contributes no cut to the combine, so a K that
    looks like 8 can be a 5 in the cold start.  Reported because a null at high
    K means something different when half the folds were dropped.
    """
    if "n_folds_used" not in v.columns:
        return pd.DataFrame()
    d = v.dropna(subset=["n_folds_used"])
    if d.empty:
        return pd.DataFrame()
    return (
        d.groupby(["geometry", "band", "k"], dropna=False)["n_folds_used"]
        .agg(mean_folds_used="mean", min_folds_used="min", n_steps="size")
        .reset_index()
        .sort_values(["geometry", "band", "k"])
    )


def worked_examples(v: pd.DataFrame, k: int, n: int = 12) -> pd.DataFrame:
    """The literal steps where fold count *k* moved the cut most, per geometry.

    An aggregate says a fold count is worth 0.006 of cost; it cannot say what
    that looked like on a screen.  These are individual steps - named category,
    seed, click, both thresholds and both operating points - so a reader can
    ask the question no mean answers: did the extra folds move the cut in the
    direction the story claims, and by enough to change what the user saw?

    Picked by the largest |Delta cost| rather than by the largest gain, because
    the worst steps are as much a part of the answer as the best ones, and a
    table of wins only is an advertisement.
    """
    keys = list(STEP_KEYS)
    cols = ["threshold", "cost", "fpr", "fnr", "n_flagged", "n_votes", "n_folds_used"]
    have = [c for c in cols if c in v.columns]
    base = v[v["k"] == BASELINE_K].set_index(keys)
    arm = v[v["k"] == k].set_index(keys)
    base, arm = base[~base.index.duplicated()], arm[~arm.index.duplicated()]
    idx = arm.index.intersection(base.index)
    if len(idx) == 0:
        return pd.DataFrame()
    j = base.loc[idx, have].add_suffix(f"_k{BASELINE_K}").join(arm.loc[idx, have].add_suffix(f"_k{k}"))
    j["d_cost"] = j[f"cost_k{k}"] - j[f"cost_k{BASELINE_K}"]
    j["d_threshold"] = j[f"threshold_k{k}"] - j[f"threshold_k{BASELINE_K}"]
    j = j.reset_index()
    out = []
    for geom, g in j.groupby("geometry", dropna=False):
        top = g.reindex(g["d_cost"].abs().sort_values(ascending=False).index).head(n)
        out.append(top)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def levels(v: pd.DataFrame) -> pd.DataFrame:
    """Mean level per (geometry, band, K) - the price sheet, whatever the verdict."""
    cols = {
        "cost": ("cost", "mean"),
        "regret_honest": ("regret_honest", "mean"),
        "cal_seconds": ("cal_seconds_used", "mean"),
        "step_seconds": ("step_seconds", "mean"),
        "cal_share": ("cal_share", "mean"),
        "n_cal_scores": ("n_cal_scores", "mean"),
        "n_steps": ("cost", "size"),
    }
    have = {name: spec for name, spec in cols.items() if spec[0] in v.columns}
    return (
        v.groupby(["geometry", "mode", "band", "k"], dropna=False)
        .agg(**have)
        .reset_index()
        .sort_values(["geometry", "band", "k"])
    )


def ab_contrast(screen: pd.DataFrame, ab_dirs: Sequence[Path], metric: str, rng: np.random.Generator) -> pd.DataFrame:
    """Stage B: does a run that LIVES at K reproduce the screen's delta for K?

    The arms share no votes with each other or with the screen - that is the
    acquisition feedback stage B exists to measure - so they are paired on
    ``(dataset, category, seed, geometry)`` only, and each arm's number is its
    own level against the screen's K=2 trajectory on the same cells.

    A screen that over- or under-states the live effect is still useful, but
    only if that is reported, so this table is part of the deliverable whether
    or not it agrees.
    """
    rows: list[dict] = []
    base = screen[(screen["k"] == BASELINE_K) & (screen["arm"] == SHIPPED_ARM)]
    if base.empty:
        return pd.DataFrame()
    base_cells = base.groupby([*CELL_KEYS, "band"], dropna=False)[metric].mean()
    for d in ab_dirs:
        live_raw = load_cells(Path(d) / "cells")
        live = fold_frame(live_raw)
        if live.empty:
            common.log(f"  A/B dir {d}: no fold rows; skipped")
            continue
        live = live[live["arm"] == SHIPPED_ARM]
        # The arm's live count is what the run lived at, read off the row rather
        # than off the directory name: a run that read a stale environment would
        # otherwise be indistinguishable from one that did what its name says.
        live_k = sorted(set(live["k"]))
        arm_k = max(live_k)
        live_cells = live[live["k"] == arm_k].groupby([*CELL_KEYS, "band"], dropna=False)[metric].mean()
        idx = live_cells.index.intersection(base_cells.index)
        if len(idx) == 0:
            continue
        j = pd.DataFrame(
            {"d": live_cells.loc[idx].to_numpy(dtype=float) - base_cells.loc[idx].to_numpy(dtype=float)},
            index=idx,
        ).reset_index()
        for (geom, band), w in j.groupby(["geometry", "band"], dropna=False):
            cells = w["d"].to_numpy(dtype=float)
            rows.append(
                {
                    "arm_dir": str(d),
                    "live_k": int(arm_k),
                    "geometry": geom,
                    "band": band,
                    "metric": metric,
                    "delta_vs_screen_k2": float(cells.mean()),
                    "se": _boot_se(cells, rng),
                    "n_cells": int(cells.size),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["resolved"] = out["delta_vs_screen_k2"].abs() > 2 * out["se"]
    return out


def figures(v: pd.DataFrame, outdir: Path, baseline_csv: Path | None) -> list[str]:
    """The mandatory quality-over-clicks pair, drawn by the one implementation.

    ``curves`` panels by ``dataset`` and colours by ``arm``.  This study has ONE
    dataset and three geometries, and the geometry is what must not be pooled -
    averaging ``max_patch`` with ``whole_image`` describes no system anyone could
    run - so the panel column carries the geometry and the hue is the fold
    count, which is the thing being compared.
    """
    d = v[v["arm"] == SHIPPED_ARM].copy()
    if d.empty:
        return []
    d["arm"] = "K=" + d["k"].astype(str)
    # `-`, not `/`: `per_run_figures` puts the panel column into the FILENAME,
    # so `dinov3_patch/max_patch` would ask matplotlib to save into a directory
    # that does not exist - a crash after every cell has been paid for.
    d["dataset"] = d["dataset"].astype(str) + " · " + d["geometry"].astype(str).str.replace("/", "-", regex=False)
    baseline = curves.text_sort_baseline(baseline_csv) if baseline_csv and Path(baseline_csv).exists() else None
    if baseline is not None:
        reps = []
        for geom in sorted(v["geometry"].unique()):
            b = baseline.copy()
            b["dataset"] = b["dataset"].astype(str) + " · " + geom.replace("/", "-")
            reps.append(b)
        baseline = pd.concat(reps, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = ["K=" + str(k) for k in sorted(set(v["k"]))]
    denominator = d[["dataset", "embedder", "category", "seed"]].drop_duplicates()
    written: list[str] = []
    for metric in ("cost", "average_precision"):
        if metric not in d.columns:
            continue
        written += curves.quality_vs_clicks(
            d,
            outdir,
            arms=arms,
            metric=metric,
            denominator=denominator,
            baseline=baseline,
            lower_is_better=(metric == "cost"),
        )
    return written


def _md(df: pd.DataFrame) -> str:
    """Markdown table at THREE significant digits, and only three.

    The standing rule is two, because four decimals do not make a table more
    rigorous - they make it harder to read and they invent findings: an unpaired
    0.0462 against 0.0508 reads as a trend a +/-0.03 standard error cannot
    support.  Three rather than two here because this study's decision constants
    are 0.005 and 0.01, so the third digit is exactly where the ship rules are
    read; rounding a -0.00512 delta to -0.005 would print a number that sits ON
    the margin it has to clear.  Every such number is printed beside its own
    standard error, and `resolved` says plainly when a difference is smaller
    than twice it.
    """
    if df is None or len(df) == 0:
        return "_(no rows)_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: float(f"{x:.3g}") if np.isfinite(x) else x)
    try:
        return d.to_markdown(index=False)
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + d.to_string(index=False) + "\n```"


def write_report(out: Path, blocks: dict) -> None:
    lines = [
        "# Fold count vs its wall-clock price (#3314, stage A screen)",
        "",
        "Mechanical output of the rules pre-registered in",
        "`docs/experiments/calibration-fold-count-3310/PLAN.md`.  The prose report",
        "lives beside that PLAN; this file is the analyzer's own record.",
        "",
        f"Cost model: **{blocks['cost_model']}**.  Margin {MARGIN}, harm tolerance",
        f"{HARM_TOLERANCE}, step wall-clock ceiling {COST_CEILING_X}x, baseline K={BASELINE_K}.",
        "",
        "Numbers are printed to three significant digits because the decision",
        "constants are 0.005 and 0.01 and the third digit is where the rules are",
        "read; each is beside its own bootstrapped standard error, and `resolved`",
        "says when a difference is smaller than twice it.  The standard errors",
        "resample **cells**, never steps.",
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(blocks["gate"], indent=2),
        "```",
        "",
        "## Ship rules per (geometry, K)",
        "",
        _md(blocks["ship"]),
        "",
        "## Paired delta vs K=2, per band (negative = more folds is better)",
        "",
        _md(blocks["paired"]),
        "",
        "## The price sheet: banded cost ratios",
        "",
        "`step_ratio` is the median per-step wall clock at K over K=2's, paired",
        "within the step.  `cal_share` is the fraction of a retrain the user",
        "spends waiting for calibration.",
        "",
        _md(blocks["cost"]),
        "",
        "## Levels",
        "",
        _md(blocks["levels"]),
        "",
        "## sd(threshold) across seeds - the variance-reduction mechanism, directly",
        "",
        _md(blocks["spread"]),
        "",
        "## Folds that actually contributed a cut",
        "",
        _md(blocks["degen"]),
        "",
        "## The adaptive family",
        "",
        _md(blocks["schedule"]),
        "",
        f"## The steps themselves: where K={blocks['example_k']} moved the cut most",
        "",
        "Individual steps, not means.  An aggregate says a fold count is worth",
        "some fraction of cost; only a row can say what that looked like on a",
        "screen.  Picked by the largest |delta| in either direction, so the worst",
        "steps are here beside the best ones.",
        "",
        _md(blocks["examples"]),
        "",
        "## Secondary: the pooled combine rule (#2897's axis, replicated)",
        "",
        "#2897's monotone worsening in K was a property of the POOLED rule, not",
        "of the fold count.  These rows are that replication under a text-sort",
        "opening, and they are not a decision input.",
        "",
        _md(blocks["pooled"]),
        "",
        "## Stage B: live arms vs the screen's K=2 trajectory",
        "",
        _md(blocks["ab"]) if blocks["ab"] is not None and len(blocks["ab"]) else "_Gate closed or no arms passed._",
        "",
    ]
    (out / "REPORT_folds3314.md").write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="the screen's results dir (default: $CALIB_RESULTS)")
    ap.add_argument("--out", default=None, help="where tables, figures and the report go (default: results/)")
    # Defaulted from the environment so the CHAINED analyze step gets the
    # click-0 anchor too: `launch_cells.sh` submits `python $CALIB_ANALYZE`
    # with no arguments, and a figure that silently loses its anchor is the
    # one thing `curves` exists to prevent.
    ap.add_argument(
        "--baseline",
        default=os.environ.get("CALIB_BASELINE") or None,
        help="text_baseline.py CSV: the click-0 anchor (default: $CALIB_BASELINE)",
    )
    ap.add_argument("--ab", action="append", default=[], help="a stage-B arm's results dir (repeatable)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(argv)

    results = Path(args.results) if args.results else common.RESULTS
    out = Path(args.out) if args.out else results
    agg = out / "agg"
    agg.mkdir(parents=True, exist_ok=True)

    raw = load_cells(results / "cells")
    v = fold_frame(raw)
    if v.empty:
        common.log("no fold-count rows found; was CALIB_FOLD_COUNTS set on the run?")
        return 1
    if BASELINE_K not in set(v["k"]):
        common.log(f"ERROR: no K={BASELINE_K} rows; every contrast here is against production's count")
        return 1
    cost_model = str(v["cost_model"].iloc[0])
    common.log(f"fold counts: {sorted(set(v['k']))}; arms: {sorted(set(v['arm']))}; cost model: {cost_model}")
    if cost_model.startswith("fold_seconds"):
        common.log(
            "  WARNING: this run predates #3314's `cal_seconds`, so the affordability rule "
            "prices K at its fold FITS only and under-states it."
        )

    ship_v = v[v["arm"] == SHIPPED_ARM]
    if ship_v.empty:
        common.log(f"ERROR: no `{SHIPPED_ARM}` rows - was the run launched with CALIB_SAFE_THRESHOLDS=1?")
        return 1
    rng = np.random.default_rng(3314)

    paired = paired_vs_baseline(ship_v, "cost", rng)
    paired_honest = paired_vs_baseline(ship_v, "regret_honest", rng)
    cost = cost_table(ship_v)
    ship = ship_table(paired, cost)
    sched = schedule_table(ship_v, rng)
    gate = pick_stage_b(ship, sched)
    lev = levels(ship_v)
    spread = threshold_spread(ship_v)
    degen = degenerate_table(v)
    pooled = paired_vs_baseline(v[v["arm"] == POOLED_ARM], "cost", rng)
    # Literal rows for whichever K the table says is the interesting one - the
    # gate's pick when it opened, else the K with the best banded delta, so a
    # flat screen still ships examples of what "flat" looked like.
    example_k = gate["k_best"] or (int(ship.loc[ship["best_delta"].idxmin(), "k"]) if not ship.empty else 1)
    examples = worked_examples(ship_v, int(example_k))

    ab = pd.DataFrame()
    if args.ab:
        ab = ab_contrast(ship_v, [Path(d) for d in args.ab], "cost", rng)

    for name, frame in (
        ("levels", lev),
        ("paired_cost", paired),
        ("paired_regret_honest", paired_honest),
        ("cost_ratios", cost),
        ("ship_rules", ship),
        ("schedule_family", sched),
        ("sd_threshold", spread),
        ("folds_used", degen),
        ("pooled_replication", pooled),
        (f"worked_examples_k{example_k}", examples),
        ("ab_arms", ab),
    ):
        frame.to_csv(agg / f"{name}.csv", index=False)

    figs: list[str] = []
    if not args.no_figures:
        figs = figures(ship_v, out / "figures", Path(args.baseline) if args.baseline else None)

    if not args.no_viewer:
        import viewer

        d = ship_v.copy()
        d["arm"] = "K=" + d["k"].astype(str)
        # Embedders are never averaged in the viewer, so the geometry takes that
        # slot: it is the dimension that must not be pooled here.
        d["embedder"] = d["geometry"]
        baseline = curves.text_sort_baseline(args.baseline) if args.baseline and Path(args.baseline).exists() else None
        viewer.build_viewer(
            d,
            out / "viewer.html",
            arms=["K=" + str(k) for k in sorted(set(ship_v["k"]))],
            baseline=baseline,
            title="calibration fold count (#3314)",
            subtitle="Colour = fold count K · one panel per geometry × category",
        )

    summary = {
        "margin": MARGIN,
        "harm_tolerance": HARM_TOLERANCE,
        "cost_ceiling_x": COST_CEILING_X,
        "baseline_k": BASELINE_K,
        "cost_model": cost_model,
        "fold_counts": sorted(int(k) for k in set(v["k"])),
        "geometries": sorted(set(v["geometry"])),
        "n_cells": int(v.groupby(list(CELL_KEYS)).ngroups),
        "gate": gate,
        "ship_candidates": ship[ship["ship_candidate"]].to_dict(orient="records") if not ship.empty else [],
        "figures": figs,
    }
    (out / "summary_folds3314.json").write_text(json.dumps(summary, indent=2))
    write_report(
        out,
        {
            "cost_model": cost_model,
            "gate": gate,
            "ship": ship,
            "paired": paired,
            "cost": cost,
            "levels": lev,
            "spread": spread,
            "degen": degen,
            "schedule": sched,
            "pooled": pooled,
            "examples": examples,
            "example_k": int(example_k),
            "ab": ab,
        },
    )
    common.log(f"wrote {out / 'REPORT_folds3314.md'} and {out / 'summary_folds3314.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

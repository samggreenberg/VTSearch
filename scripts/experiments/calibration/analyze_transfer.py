"""Stage 3 (#2883): is ``transfer`` a bias the fit could remove, or a variance?

``analyze_cut.py`` establishes *that* the last link of the #2836 chain dominates
- ``+0.037`` of a ``+0.056`` total on the production arm's ramp window, 67 %, on
the corrected (#3187) decomposition.  #2883 asks what it scales with before
anyone proposes a remedy for it.  This analyzer answers that, and it tests the
two readings of the term that the name "finite-sim-set estimation / transfer"
quietly assumes and nobody has checked:

**H1 - it is a variance, not a bias.**  ``D_sim`` and ``D_test`` are one random
partition of a single pool scored by one model, so there is no distribution to
transfer across and the term can only be estimation error.  In threshold units a
bias shows up as a mean comparable to its own mean-absolute; a variance shows up
as a mean near zero with a large absolute.  :func:`bias_or_variance` reports the
ratio for all four terms, so ``transfer`` is judged against its own siblings
rather than against a bar someone picked.

**H2 - the reference point is optimistic.**  ``oracle_cost`` is the minimum of
the empirical cost over the **test sample itself**, so it is biased low and every
gap measured against it is biased high.  :func:`transfer_bracket` puts the
cross-fitted reference beside it and reports the term as a bracket.

**H3 - it scales like an estimation error.**  :func:`learning_curve` fits
``a + b/m`` over the four subsample levels plus the full sim set, on two x-axes:
sample size and *positive count*.  A threshold estimated from labelled scores is
limited by the rarer class, so which axis fits better is itself the answer to
"what does it scale with".  The intercept ``a`` is a third estimate of the
reference point, taken from neither bound.

**H4 - the ``family_headroom_exhausted`` bound does not hold out of sample.**
``pooled_sim_oracle`` is the empirical rate-loss minimiser over the sim scores.
It bounds every rule's loss **on the sim set**; it is not a bound on **test**
loss, which is what every table in this line reports.
:func:`variance_reduction` pairs two regularised estimators of the same target
against it.  Both are reported: bagging the *argmin* and smoothing the *cost
curve* fail in different places (see ``test_transfer_rules``), so a null on one
is not a null on the idea, and quoting only one would say the wrong thing.

The label-free ``bagfit_*`` arms ride along and are reported here, but they are
in ``SWEEP_ONLY`` and cannot be a ship candidate: #2883 item 1 asks for the
characterisation before the remedy, and a remedy that wins in the run that
diagnoses the disease is the wrong-but-plausible result this line has paid for
twice.  Pre-registration: ``docs/experiments/2026-08-24-transfer-2883/PREREG.md``.

Writes ``results/summary_transfer.json``, ``results/agg/transfer_*.csv``,
``results/figures/transfer_*.png`` and a ``results/REPORT_TRANSFER.md`` draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# One implementation of the loaders, the windows and the pairing/statistics
# helpers, shared with the analyzer this one extends.  A second copy would drift,
# and the two analyzers reading the same cells have to agree about what a "cell"
# and a "window" are or their tables cannot be read side by side.
from analyze_cut import (  # noqa: E402
    BAGFIT_VARIANTS,
    CONTROL_ARM_SUBSTR,
    DECOMPOSITION,
    PRODUCTION_ARM_SUBSTR,
    WINDOWS,
    _fill_metric,
    _md,
    _pair_frames,
    _window,
    _wilcoxon,
    load_cells,
    load_cutdiag,
)

from vtscore.eval.transfer_rules import (  # noqa: E402
    TRANSFER_SUBSAMPLE_GRID,
    subsample_rule,
)

#: The empirical minimiser over the sim set - the estimator every arm here is
#: measured against, and the one ``family_headroom_exhausted`` treats as a bound.
ERM = "pooled_sim_oracle"

#: The regularised readings of that same target, with the unbagged sibling each
#: label-free arm is paired against.
VARIANCE_REDUCED: tuple[tuple[str, str], ...] = (
    ("pooled_sim_oracle_bag", ERM),
    ("pooled_sim_oracle_smooth", ERM),
    ("pooled_bagfit_mid", "pooled_mid"),
    ("pooled_bagfit_priorfree", "pooled_priorfree"),
)

#: Levels of the learning curve, smallest first, with the full sim set last.
#: ``1.0`` is ``pooled_sim_oracle`` itself: the curve's right-hand anchor is the
#: quantity the decomposition already reports, not a fifth measurement of it.
CURVE_LEVELS: tuple[tuple[float, str], ...] = (
    *((f, f"pooled_{subsample_rule(f)}") for f in TRANSFER_SUBSAMPLE_GRID),
    (1.0, ERM),
)

#: The rule-quality column.  ``cost`` is the *blended* threshold's cost, which is
#: what a user gets; every arm here is a diagnostic that production never blends,
#: so the honest column is the unblended one - the same choice ``COST_CHAIN``
#: makes in ``analyze_cut``.
METRIC = "raw_cut_cost"

CELL = ["arm", "category", "seed"]
STEP = ["arm", "category", "seed", "t"]


def _mean_sem(vals: np.ndarray) -> tuple[float, float, float | None, int]:
    """``(mean, sem, wilcoxon p, n)`` over the finite entries of *vals*."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), None, 0
    sem = float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
    return float(np.mean(v)), sem, (_wilcoxon(v) if v.size else None), int(v.size)


def _pooled_diag(diag: pd.DataFrame) -> pd.DataFrame:
    """The pooled geometry only - the one production infers through."""
    return diag[diag["geometry"] == "pooled"] if "geometry" in diag else diag


# ------------------------------------------------------------------
# H1: bias or variance
# ------------------------------------------------------------------


def bias_or_variance(diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Per term: the signed mean against the mean absolute, in threshold units.

    A term that encodes a wrong *assumption* moves the cut the same way every
    time, so its mean is a large fraction of its mean absolute.  A term that is
    finite-sample noise moves the cut both ways, so the mean collapses while the
    absolute does not.  ``symmetry`` is ``|mean| / mean_abs``: near 1 for a bias,
    near 0 for a variance.

    All four terms are reported, not just ``transfer``, because the number only
    means something relative to the siblings measured the same way on the same
    steps - a bar chosen in the abstract would be arbitrary.  Rows are complete
    chains only, matching ``analyze_cut.decomposition_table``: a step missing an
    oracle link must not contribute to one term and not another.
    """
    d = _pooled_diag(diag)
    cols = [c for _n, a, b in DECOMPOSITION for c in (a, b)]
    have = [c for c in dict.fromkeys(cols) if c in d.columns]
    if d.empty or len(have) != len(dict.fromkeys(cols)):
        return pd.DataFrame()
    d = d.copy()
    for name, a, b in DECOMPOSITION:
        d[f"term_{name}"] = d[a] - d[b]
    term_cols = [f"term_{n}" for n, _a, _b in DECOMPOSITION]
    complete = d[np.isfinite(d[term_cols].to_numpy(dtype=float)).all(axis=1)]

    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        w = _window(complete, lo, hi)
        for arm, sub in w.groupby("arm"):
            # Collapse to cells first: steps within a trajectory are
            # autocorrelated, so a per-step SEM would be far too small.
            per_cell = sub.groupby(CELL)[term_cols].mean()
            per_cell_abs = sub.groupby(CELL)[term_cols].agg(lambda s: np.nanmean(np.abs(s)))
            for name in (n for n, _a, _b in DECOMPOSITION):
                col = f"term_{name}"
                mean, sem, p, n = _mean_sem(per_cell[col].to_numpy())
                mabs = float(np.nanmean(per_cell_abs[col].to_numpy()))
                rows.append(
                    {
                        "window": wname,
                        "arm": arm,
                        "term": name,
                        "mean": mean,
                        "sem": sem,
                        "mean_abs": mabs,
                        "symmetry": abs(mean) / mabs if mabs > 0 else float("nan"),
                        "p": p,
                        "n_cells": n,
                    }
                )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "transfer_bias_or_variance.csv", index=False)
    return tbl


# ------------------------------------------------------------------
# H2: the reference point
# ------------------------------------------------------------------


def transfer_bracket(df: pd.DataFrame, diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """``transfer`` against both references, and the optimism between them.

    ``transfer_naive`` is what the decomposition reports today: the sim oracle's
    test cost minus the minimum of the empirical cost **on that same test
    sample**.  ``transfer_honest`` measures the same cut against a cross-fitted
    reference, which is an upper bound on the population optimum where the naive
    one is a lower bound - so the true term lies between them and
    ``optimism_share`` says how much of today's number is the reference
    overfitting rather than anything the fit could recover.

    Everything is paired within a step first (same cut, same test scores, two
    references) and collapsed to cells before any statistic is taken.
    """
    d = _pooled_diag(diag)
    needed = ("cost_test_oracle_naive", "cost_test_oracle_honest")
    if d.empty or not all(c in d.columns for c in needed):
        common.log(f"transfer bracket skipped - cutdiag has no {needed}; was the run made with this branch?")
        return pd.DataFrame()

    erm = df[df["gmm_variant"] == ERM].set_index([*STEP, "n_votes"])[[METRIC]]
    refs = d.set_index([*STEP, "n_votes"])[list(needed)]
    j = erm.join(refs, how="inner").reset_index()
    if j.empty:
        return pd.DataFrame()
    j["transfer_naive"] = j[METRIC] - j["cost_test_oracle_naive"]
    j["transfer_honest"] = j[METRIC] - j["cost_test_oracle_honest"]
    j["optimism"] = j["cost_test_oracle_honest"] - j["cost_test_oracle_naive"]

    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        w = _window(j, lo, hi)
        for arm, sub in w.groupby("arm"):
            cols = ["transfer_naive", "transfer_honest", "optimism", *needed]
            per_cell = sub.groupby(CELL)[cols].mean()
            entry: dict = {"window": wname, "arm": arm}
            # The reference *levels*, not just the gaps: H3's intercept is a
            # third estimate of the same reference and can only be read against
            # them if they are in the table beside it.
            entry["ref_naive"] = float(np.nanmean(per_cell["cost_test_oracle_naive"].to_numpy()))
            entry["ref_honest"] = float(np.nanmean(per_cell["cost_test_oracle_honest"].to_numpy()))
            # Step-weighted as well as cell-weighted, and the step count with it.
            # `analyze_cut.cost_decomposition`'s `cost_transfer` is the same
            # quantity computed two ways over: it averages *steps* rather than
            # cells, and it keeps only steps with a complete four-link oracle
            # chain, where this table needs the sim oracle alone.  Both
            # differences are legitimate and they do not cancel, so the two
            # numbers disagree by a small amount that a reader would otherwise
            # have to explain away.  Printing both is cheaper than the paragraph.
            entry["n_steps"] = int(len(sub))
            entry["transfer_naive_step_weighted"] = float(np.nanmean(sub["transfer_naive"].to_numpy()))
            for col in ("transfer_naive", "transfer_honest", "optimism"):
                mean, sem, p, n = _mean_sem(per_cell[col].to_numpy())
                entry[col] = mean
                entry[f"sem_{col}"] = sem
                entry[f"p_{col}"] = p
                entry["n_cells"] = n
            entry["optimism_share"] = (
                entry["optimism"] / entry["transfer_naive"] if entry["transfer_naive"] > 0 else float("nan")
            )
            rows.append(entry)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "transfer_bracket.csv", index=False)

    # The axis the user actually spends.  A window mean cannot show whether the
    # gap closes as votes accumulate, and "what do I get after 20 clicks" is the
    # question the ramp window exists to answer.
    by_votes = (
        j.groupby(["arm", "n_votes"])[["transfer_naive", "transfer_honest", "optimism"]]
        .agg(["mean", "sem", "size"])
        .reset_index()
    )
    by_votes.columns = ["_".join(c).rstrip("_") for c in by_votes.columns]
    by_votes.to_csv(agg_dir / "transfer_by_votes.csv", index=False)

    # One row per cell per step: the mean hides that some runs never leave the
    # floor, and on these trajectories the spread is usually the real finding.
    j[[*STEP, "n_votes", "transfer_naive", "transfer_honest"]].to_csv(agg_dir / "transfer_by_cell.csv", index=False)
    return tbl


def reference_sanity(df: pd.DataFrame, diag: pd.DataFrame) -> dict:
    """``cost_test_oracle_naive`` must reproduce the row's own ``oracle_cost``.

    The two are computed in different places from the same cut, so they agree or
    the join is wrong - the cheapest possible check that the new diagnostic
    columns are aligned to the rows they are read beside.  Free, and it is
    exactly the kind of misalignment that produces a clean table meaning nothing.
    """
    d = _pooled_diag(diag)
    if d.empty or "cost_test_oracle_naive" not in d.columns or "oracle_cost" not in df.columns:
        return {"n_steps": 0, "max_abs_diff": None, "ok": None}
    base = df[df["gmm_variant"] == ERM].set_index(STEP)[["oracle_cost"]]
    j = base.join(d.set_index(STEP)[["cost_test_oracle_naive"]], how="inner")
    if j.empty:
        return {"n_steps": 0, "max_abs_diff": None, "ok": None}
    diff = (j["oracle_cost"] - j["cost_test_oracle_naive"]).abs()
    return {
        "n_steps": int(len(j)),
        "max_abs_diff": float(diff.max()),
        # The columns are rounded independently on the way out of the harness.
        "ok": bool(diff.max() <= 2e-6),
    }


# ------------------------------------------------------------------
# H3: what it scales with
# ------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, NaN when either side is constant or too short.

    Rank rather than Pearson: the relationship under test is "does this slope
    move with prevalence at all", and prevalence spans two orders of magnitude
    across categories, so a linear correlation would be dominated by the densest
    few and answer a different question.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    rho, _p = spearmanr(a[ok], b[ok])
    return float(rho)


def _fit_inverse(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares ``y = a + b * (1/x)``; returns ``(a, b, r2)``."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    inv = 1.0 / x[ok]
    yy = y[ok]
    b, a = np.polyfit(inv, yy, 1)
    pred = a + b * inv
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    return float(a), float(b), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def learning_curve(df: pd.DataFrame, diag: pd.DataFrame, agg_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test cost of the sim oracle as a function of how much sim set it saw.

    The levels re-cut the *same* per-step model against the *same* test scores,
    so the only thing that moves across a curve is the number of labelled sim
    scores the cut was estimated from.  That is what makes the curve readable:
    re-running with a smaller ``sim_fraction`` would shrink the sim set and grow
    the test set at once, moving the estimator and the reference together.

    Two x-axes, because "what does it scale with" is the question:
    ``m`` (sample size) and ``n_pos`` (positives).  A threshold estimated from
    labelled scores is limited by the rarer class, so if the positives axis fits
    better, prevalence - not pool size - is the lever, and the remedy for the
    dominant term is to find more positives rather than to label more items.

    Returns ``(curve, fits)``: the per-level means, and the per-arm fits with the
    intercept ``a`` that estimates the reference point from neither bound.
    """
    d = _pooled_diag(diag)
    if d.empty or "sim_n" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()
    sizes = d.set_index([*STEP, "n_votes"])[["sim_n", "sim_prevalence"]]

    frames = []
    for frac, variant in CURVE_LEVELS:
        v = df[df["gmm_variant"] == variant].set_index([*STEP, "n_votes"])[[METRIC]]
        if v.empty:
            common.log(f"learning curve: no rows for {variant} - level {frac} missing from this run")
            continue
        j = v.join(sizes, how="inner").reset_index()
        j["frac"] = frac
        j["prevalence"] = j["sim_prevalence"]
        # The realised counts, from the run - a level's m is a fraction of the
        # sim set this cell actually had, not of the dataset's nominal size.
        j["m"] = j["sim_n"] * frac
        j["n_pos"] = j["sim_n"] * j["sim_prevalence"] * frac
        frames.append(j)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)

    curve_rows, fit_rows = [], []
    for wname, (lo, hi) in WINDOWS.items():
        w = _window(long, lo, hi)
        for arm, sub in w.groupby("arm"):
            per_cell = sub.groupby([*CELL, "frac"])[[METRIC, "m", "n_pos", "prevalence"]].mean().reset_index()
            for frac, lvl in per_cell.groupby("frac"):
                mean, sem, _p, n = _mean_sem(lvl[METRIC].to_numpy())
                curve_rows.append(
                    {
                        "window": wname,
                        "arm": arm,
                        "frac": frac,
                        "m": float(lvl["m"].mean()),
                        "n_pos": float(lvl["n_pos"].mean()),
                        "cost": mean,
                        "sem": sem,
                        "n_cells": n,
                    }
                )
            # Per-cell fits give the intercept an honest error bar; a single fit
            # to the five arm-level means would have none.
            #
            # **The axis question cannot be answered from these fits.**  Within a
            # cell the category is fixed, so prevalence is fixed, so
            # `n_pos = prevalence * m` exactly - the two candidate x-axes are the
            # same axis up to a constant and fit identically by construction
            # (R^2 agreed to four decimals in the selftest, on data planted to
            # scale with positives).  The discriminating information is *across*
            # cells: if cost really goes as `a + b/n_pos`, the n_pos-axis slope is
            # a constant of the problem and does not move with prevalence, while
            # the m-axis slope must go as `1/prevalence` to compensate.  So the
            # axis is decided by which slope is *independent* of prevalence.
            for axis in ("m", "n_pos"):
                a_s, b_s, r2_s, prev_s = [], [], [], []
                for _cell, g in per_cell.groupby(CELL):
                    if len(g) < 3:
                        continue
                    a, b, r2 = _fit_inverse(g[axis].to_numpy(), g[METRIC].to_numpy())
                    if np.isfinite(a):
                        a_s.append(a)
                        b_s.append(b)
                        r2_s.append(r2)
                        prev_s.append(float(g["prevalence"].mean()))
                if not a_s:
                    continue
                a_mean, a_sem, _p, n = _mean_sem(np.asarray(a_s))
                b_mean, b_sem, _pb, _n = _mean_sem(np.asarray(b_s))
                fit_rows.append(
                    {
                        "window": wname,
                        "arm": arm,
                        "axis": axis,
                        "intercept": a_mean,
                        "sem_intercept": a_sem,
                        "slope": b_mean,
                        "sem_slope": b_sem,
                        "median_r2": float(np.nanmedian(r2_s)),
                        "slope_prevalence_rho": _spearman(np.asarray(b_s), np.asarray(prev_s)),
                        "n_cells": n,
                    }
                )
    curve = pd.DataFrame(curve_rows)
    fits = pd.DataFrame(fit_rows)
    curve.to_csv(agg_dir / "transfer_learning_curve.csv", index=False)
    fits.to_csv(agg_dir / "transfer_curve_fits.csv", index=False)
    return curve, fits


def scaling_table(df: pd.DataFrame, diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """``transfer`` banded by the things #2883 item 1 asks it to be banded by.

    Positives, prevalence and position along the trajectory, each as quintiles of
    the run's own distribution rather than round numbers - a band chosen in
    advance lands unevenly and an average across a crossover is precisely the
    number that hides it.
    """
    d = _pooled_diag(diag)
    if d.empty or "cost_test_oracle_honest" not in d.columns:
        return pd.DataFrame()
    erm = df[df["gmm_variant"] == ERM].set_index([*STEP, "n_votes"])[[METRIC]]
    cols = ["cost_test_oracle_naive", "cost_test_oracle_honest", "sim_n", "sim_prevalence"]
    j = erm.join(d.set_index([*STEP, "n_votes"])[cols], how="inner").reset_index()
    if j.empty:
        return pd.DataFrame()
    j["transfer_naive"] = j[METRIC] - j["cost_test_oracle_naive"]
    j["transfer_honest"] = j[METRIC] - j["cost_test_oracle_honest"]
    j["n_pos"] = j["sim_n"] * j["sim_prevalence"]

    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        w = _window(j, lo, hi).copy()
        if w.empty:
            continue
        for axis in ("n_pos", "sim_prevalence", "n_votes"):
            try:
                w["_band"] = pd.qcut(w[axis], 5, duplicates="drop")
            except ValueError:
                continue
            for (arm, band), sub in w.groupby(["arm", "_band"], observed=True):
                per_cell = sub.groupby(CELL)[["transfer_naive", "transfer_honest", axis]].mean()
                naive, naive_sem, _p, n = _mean_sem(per_cell["transfer_naive"].to_numpy())
                honest, honest_sem, _ph, _n = _mean_sem(per_cell["transfer_honest"].to_numpy())
                rows.append(
                    {
                        "window": wname,
                        "arm": arm,
                        "axis": axis,
                        "band": str(band),
                        "band_mid": float(per_cell[axis].mean()),
                        "transfer_naive": naive,
                        "sem_naive": naive_sem,
                        "transfer_honest": honest,
                        "sem_honest": honest_sem,
                        "n_cells": n,
                    }
                )
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "transfer_scaling.csv", index=False)
    return tbl


# ------------------------------------------------------------------
# H4: is the empirical minimiser a bound?
# ------------------------------------------------------------------


def variance_reduction(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Each regularised estimator against the estimator it regularises.

    Paired within a step - same model, same sim scores, same test scores, one
    difference - then collapsed to cells.  A negative ``mean_d_raw_cut_cost``
    means the regularised reading is *cheaper on held-out data* than the
    empirical minimiser, which is the thing ``family_headroom_exhausted`` asserts
    cannot happen.
    """
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        for variant, baseline in VARIANCE_REDUCED:
            if not (df["gmm_variant"] == variant).any():
                continue
            w = _window(df, lo, hi)
            va = w[w["gmm_variant"] == variant].set_index(STEP)[[METRIC]]
            vb = w[w["gmm_variant"] == baseline].set_index(STEP)[[METRIC]]
            paired = _pair_frames(va, vb, [METRIC])
            if paired.empty:
                continue
            for arm, sub in paired.groupby("arm"):
                entry: dict = {
                    "window": wname,
                    "arm": arm,
                    "variant": variant,
                    "baseline": baseline,
                    "n_cells": int(len(sub)),
                    "label_free": variant in BAGFIT_VARIANTS,
                }
                _fill_metric(entry, METRIC, sub[f"d_{METRIC}"])
                rows.append(entry)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "transfer_variance_reduction.csv", index=False)
    return tbl


# ------------------------------------------------------------------
# Decisions
# ------------------------------------------------------------------


def _row(tbl: pd.DataFrame, **eq) -> pd.Series | None:
    if tbl.empty:
        return None
    sel = tbl
    for k, v in eq.items():
        sel = sel[sel[k] == v] if k != "arm_substr" else sel[sel["arm"].str.contains(v)]
    return None if sel.empty else sel.iloc[0]


def decisions(
    bov: pd.DataFrame,
    bracket: pd.DataFrame,
    fits: pd.DataFrame,
    vr: pd.DataFrame,
    window: str = "ramp_6_20",
) -> dict:
    """The four pre-registered readings, on the production arm's ramp window.

    Each is stated as the number *and* the verdict, because the verdict is what
    a later run will quote and the number is what it will need to check that the
    verdict still means the same thing.
    """
    out: dict = {"window": window, "production_arm": PRODUCTION_ARM_SUBSTR, "control_arm": CONTROL_ARM_SUBSTR}

    # H1
    t = _row(bov, window=window, term="transfer", arm_substr=PRODUCTION_ARM_SUBSTR)
    others = (
        bov[(bov["window"] == window) & bov["arm"].str.contains(PRODUCTION_ARM_SUBSTR) & (bov["term"] != "transfer")]
        if not bov.empty
        else pd.DataFrame()
    )
    if t is not None:
        out["h1_transfer_symmetry"] = float(t["symmetry"])
        out["h1_sibling_symmetry_min"] = float(others["symmetry"].min()) if not others.empty else float("nan")
        # Pre-registered: the term is a variance if it is symmetric in threshold
        # units AND unlike every sibling measured the same way.
        out["h1_transfer_is_variance"] = bool(
            t["symmetry"] < 0.10 and (others.empty or t["symmetry"] < others["symmetry"].min())
        )

    # H2
    b = _row(bracket, window=window, arm_substr=PRODUCTION_ARM_SUBSTR)
    if b is not None:
        out["h2_transfer_naive"] = float(b["transfer_naive"])
        out["h2_transfer_honest"] = float(b["transfer_honest"])
        out["h2_optimism"] = float(b["optimism"])
        out["h2_optimism_share"] = float(b["optimism_share"])
        out["h2_reference_is_optimistic"] = bool(b["optimism"] > 2.0 * b["sem_optimism"])
        out["h2_majority_is_reference"] = bool(b["optimism_share"] > 0.5)

    # H3
    for axis in ("m", "n_pos"):
        f = _row(fits, window=window, axis=axis, arm_substr=PRODUCTION_ARM_SUBSTR)
        if f is not None:
            out[f"h3_{axis}_slope"] = float(f["slope"])
            out[f"h3_{axis}_intercept"] = float(f["intercept"])
            out[f"h3_{axis}_median_r2"] = float(f["median_r2"])
            out[f"h3_{axis}_slope_prevalence_rho"] = float(f["slope_prevalence_rho"])
    # NOT by R^2: the two axes are proportional within a cell and fit equally
    # well by construction.  The axis whose slope does not move with prevalence
    # is the one the cost actually scales with.  See `learning_curve`.
    rho_m = out.get("h3_m_slope_prevalence_rho", float("nan"))
    rho_p = out.get("h3_n_pos_slope_prevalence_rho", float("nan"))
    if np.isfinite(rho_m) and np.isfinite(rho_p):
        out["h3_better_axis"] = "n_pos" if abs(rho_p) < abs(rho_m) else "m"

    # The three estimates of one reference point, side by side.  This is the
    # check the pre-registration commits to: the cross-fitted reference is an
    # upper bound that carries a cross-fitting penalty, the sample minimum is a
    # lower bound, and the curve's intercept is fitted from neither.  If the
    # intercept sits inside the bracket, H2 is real; if it sits outside by more
    # than the bracket's own width, H2 is reported UNRESOLVED, not won.
    if b is not None and "h3_n_pos_intercept" in out:
        intercept = out["h3_n_pos_intercept"]
        out["h3_reference_estimates"] = {
            "naive_lower_bound": float(b["ref_naive"]),
            "curve_intercept": float(intercept),
            "honest_upper_bound": float(b["ref_honest"]),
        }
        lo, hi = float(b["ref_naive"]), float(b["ref_honest"])
        width = hi - lo
        inside = lo - 1e-9 <= intercept <= hi + 1e-9
        out["h3_intercept_inside_bracket"] = bool(inside)
        out["h3_intercept_excess_over_bracket"] = (
            0.0 if inside else float(min(abs(intercept - lo), abs(intercept - hi)))
        )
        if not inside and width > 0 and out["h3_intercept_excess_over_bracket"] > width:
            out["h2_verdict"] = "unresolved - the curve intercept falls outside the bracket by more than its width"
        elif out.get("h2_majority_is_reference"):
            out["h2_verdict"] = "the majority of `transfer` is the reference point"
        else:
            out["h2_verdict"] = "the reference point is not the majority of `transfer`"

    # H4
    beaten = []
    if not vr.empty:
        sel = vr[(vr["window"] == window) & vr["arm"].str.contains(PRODUCTION_ARM_SUBSTR)]
        for _i, r in sel.iterrows():
            entry = {
                "variant": r["variant"],
                "baseline": r["baseline"],
                "mean_d_cost": float(r[f"mean_d_{METRIC}"]),
                "sem": float(r[f"sem_d_{METRIC}"]),
                "p": r[f"p_d_{METRIC}"],
                "label_free": bool(r["label_free"]),
            }
            out.setdefault("h4_arms", []).append(entry)
            if entry["baseline"] == ERM and entry["mean_d_cost"] < 0 and (entry["p"] or 1.0) < 0.01:
                beaten.append(entry["variant"])
    out["h4_erm_beaten_by"] = beaten
    # The claim under test is about the *bound*, so one estimator beating it is
    # enough to refute the bound - the size of the win is a separate question.
    out["h4_sim_oracle_is_not_a_bound"] = bool(beaten)
    return out


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------


def make_figures(
    curve: pd.DataFrame,
    bracket: pd.DataFrame,
    scaling: pd.DataFrame,
    fig_dir: Path,
    agg_dir: Path | None = None,
) -> list[str]:
    """The three figures the argument cannot be made without.

    Every one is drawn from the same CSVs as the tables above, so a reader can
    check a curve against the numbers rather than against the prose.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    made: list[str] = []
    win = "ramp_6_20"

    # 1. The learning curve, one panel per arm, on the axis that binds.
    if not curve.empty:
        c = curve[curve["window"] == win]
        arms = sorted(c["arm"].unique())
        if arms:
            fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4.2), squeeze=False)
            for ax, arm in zip(axes[0], arms, strict=True):
                sub = c[c["arm"] == arm].sort_values("n_pos")
                ax.errorbar(sub["n_pos"], sub["cost"], yerr=sub["sem"], marker="o", capsize=3)
                ax.set_xscale("log")
                ax.set_xlabel("labelled sim positives the cut was estimated from")
                ax.set_ylabel("test cost of the sim-oracle cut")
                ax.set_title(arm, fontsize=9)
                ax.grid(alpha=0.3)
            fig.suptitle(
                "Transfer is a sample-size effect: the same cut rule, the same test set,\n"
                "only the number of labelled sim scores changes (ramp 6-20)",
                fontsize=10,
            )
            fig.tight_layout()
            p = fig_dir / "transfer_learning_curve.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(p.name)

    # 2. The bracket: how much of today's number is the reference point.
    if not bracket.empty:
        b = bracket[bracket["window"] == win].sort_values("arm")
        if not b.empty:
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            y = np.arange(len(b))
            ax.barh(y, b["transfer_honest"], color="#3b6ea5", label="transfer vs a cross-fitted reference")
            ax.barh(
                y,
                b["optimism"],
                left=b["transfer_honest"],
                color="#c9553f",
                alpha=0.75,
                label="the reference's own optimism",
            )
            ax.set_yticks(y)
            ax.set_yticklabels(b["arm"], fontsize=8)
            ax.set_xlabel("excess cost vs the test-set optimum")
            ax.legend(fontsize=8)
            ax.grid(axis="x", alpha=0.3)
            ax.set_title("What the +0.037 'transfer' term is made of (ramp 6-20)", fontsize=10)
            fig.tight_layout()
            p = fig_dir / "transfer_bracket.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(p.name)

    # 3. The scaling axis, both references, one line per arm.
    if not scaling.empty:
        s = scaling[(scaling["window"] == win) & (scaling["axis"] == "n_pos")]
        if not s.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            for arm, sub in s.groupby("arm"):
                sub = sub.sort_values("band_mid")
                ax.errorbar(
                    sub["band_mid"],
                    sub["transfer_naive"],
                    yerr=sub["sem_naive"],
                    marker="o",
                    label=f"{arm} (naive ref)",
                )
                ax.errorbar(
                    sub["band_mid"],
                    sub["transfer_honest"],
                    yerr=sub["sem_honest"],
                    marker="s",
                    linestyle="--",
                    label=f"{arm} (honest ref)",
                )
            ax.set_xscale("log")
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xlabel("labelled sim positives (quintile midpoint)")
            ax.set_ylabel("transfer")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_title("Transfer against positives, under both reference points", fontsize=10)
            fig.tight_layout()
            p = fig_dir / "transfer_scaling.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            made.append(p.name)

    if agg_dir is None:
        return made

    # 4. The axis the user spends, both references, with a band.
    votes_csv = agg_dir / "transfer_by_votes.csv"
    if votes_csv.exists():
        bv = pd.read_csv(votes_csv)
        if not bv.empty:
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            for arm, sub in bv.groupby("arm"):
                sub = sub.sort_values("n_votes")
                for col, style, lbl in (
                    ("transfer_naive", "-", "naive reference (what the chain reports)"),
                    ("transfer_honest", "--", "cross-fitted reference"),
                ):
                    m, e = sub[f"{col}_mean"], sub[f"{col}_sem"]
                    (line,) = ax.plot(sub["n_votes"], m, style, label=f"{arm.split('/', 1)[1]} - {lbl}")
                    ax.fill_between(sub["n_votes"], m - e, m + e, alpha=0.18, color=line.get_color())
            ax.axvspan(6, 20, color="grey", alpha=0.10)
            ax.axhline(0.0, color="k", lw=0.8)
            ax.set_xlabel("votes (the axis a user spends)")
            ax.set_ylabel("transfer")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_title(
                "`transfer` over the vote budget, under both references\n(shaded band = the 6-20 ramp window)",
                fontsize=10,
            )
            fig.tight_layout()
            q = fig_dir / "transfer_by_votes.png"
            fig.savefig(q, dpi=130)
            plt.close(fig)
            made.append(q.name)

    # 5. One line per run: the spread, which the means above cannot show.
    cell_csv = agg_dir / "transfer_by_cell.csv"
    if cell_csv.exists():
        bc = pd.read_csv(cell_csv)
        arms = sorted(bc["arm"].unique())
        if arms:
            fig, axes = plt.subplots(1, len(arms), figsize=(6 * len(arms), 4.2), squeeze=False, sharey=True)
            for ax, arm in zip(axes[0], arms, strict=True):
                sub = bc[bc["arm"] == arm]
                for _key, g in sub.groupby(["category", "seed"]):
                    ax.plot(g["n_votes"], g["transfer_honest"], color="#3b6ea5", alpha=0.06, lw=0.8)
                mean = sub.groupby("n_votes")["transfer_honest"].mean()
                ax.plot(mean.index, mean.to_numpy(), color="#c9553f", lw=2.0, label="mean")
                ax.axhline(0.0, color="k", lw=0.8)
                ax.set_xlabel("votes")
                ax.set_title(arm, fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            axes[0][0].set_ylabel("transfer vs the cross-fitted reference")
            fig.suptitle(
                "One line per run. A mean over cells cannot show how far individual\n"
                "trajectories sit from it, and here the spread is the finding.",
                fontsize=10,
            )
            fig.tight_layout()
            q = fig_dir / "transfer_by_cell.png"
            fig.savefig(q, dpi=130)
            plt.close(fig)
            made.append(q.name)
    return made


# ------------------------------------------------------------------


def write_report(summary: dict, tables: dict[str, pd.DataFrame], path: Path) -> None:
    lines = [
        "# Transfer: bias or variance? (#2883) — draft",
        "",
        "Generated by `analyze_transfer.py`. Pre-registration:",
        "`docs/experiments/2026-08-24-transfer-2883/PREREG.md`.",
        "",
        "```json",
        json.dumps(summary["decisions"], indent=2, default=float),
        "```",
        "",
        f"Reference sanity (`cost_test_oracle_naive` vs the row's own `oracle_cost`): "
        f"`{json.dumps(summary['reference_sanity'], default=float)}`",
        "",
    ]
    for title, key in (
        ("H1 — bias or variance, in threshold units", "bias_or_variance"),
        ("H2 — the transfer bracket", "bracket"),
        ("H3 — learning curve", "curve"),
        ("H3 — curve fits", "fits"),
        ("H3 — scaling bands", "scaling"),
        ("H4 — variance reduction vs the empirical minimiser", "variance_reduction"),
    ):
        tbl = tables.get(key)
        if tbl is not None and not tbl.empty:
            lines += [f"## {title}", "", _md(tbl), ""]
    for fig in summary.get("figures", []):
        lines += [f"![{fig}](figures/{fig})", ""]
    path.write_text("\n".join(lines))


def main() -> int:
    cells_dir = common.RESULTS / "cells"
    agg_dir = common.RESULTS / "agg"
    fig_dir = common.RESULTS / "figures"
    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_cells(cells_dir)
    diag = load_cutdiag(cells_dir)
    if df.empty:
        common.log("no cell CSVs found; nothing to analyze")
        return 1
    if diag.empty:
        common.log("no __cutdiag frames - this analyzer is the decomposition half; nothing to do")
        return 1

    missing = [v for _f, v in CURVE_LEVELS if not (df["gmm_variant"] == v).any()]
    if missing:
        common.log(f"WARNING: {len(missing)} curve level(s) absent from the cells: {', '.join(missing)}")
        common.log("         -> the learning curve will be fitted on fewer points than designed")

    bov = bias_or_variance(diag, agg_dir)
    bracket = transfer_bracket(df, diag, agg_dir)
    curve, fits = learning_curve(df, diag, agg_dir)
    scaling = scaling_table(df, diag, agg_dir)
    vr = variance_reduction(df, agg_dir)
    sanity = reference_sanity(df, diag)
    if sanity.get("ok") is False:
        common.log(f"WARNING: reference sanity FAILED (max_abs_diff={sanity['max_abs_diff']}) - the join is suspect")

    figs = make_figures(curve, bracket, scaling, fig_dir, agg_dir)
    summary = {
        "n_variant_rows": int((df["gmm_variant"] != "").sum()),
        "n_diag_rows": int(len(diag)),
        "n_cells": int(df[["dataset", "embedder", "category", "seed"]].drop_duplicates().shape[0]),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "curve_levels": [v for _f, v in CURVE_LEVELS],
        "missing_curve_levels": missing,
        "reference_sanity": sanity,
        "decisions": decisions(bov, bracket, fits, vr),
        "figures": figs,
    }
    (common.RESULTS / "summary_transfer.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(
        summary,
        {
            "bias_or_variance": bov,
            "bracket": bracket,
            "curve": curve,
            "fits": fits,
            "scaling": scaling,
            "variance_reduction": vr,
        },
        common.RESULTS / "REPORT_TRANSFER.md",
    )
    common.log(f"wrote {common.RESULTS / 'REPORT_TRANSFER.md'}")
    common.log("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

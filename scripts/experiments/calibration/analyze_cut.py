"""Stage 2 (cut-rule study, #2836): which cut, and which term in the derivation is wrong.

Consumes a ``CALIB_SAFE_THRESHOLDS=1`` run's ``results/cells/task_*.csv`` (one row
per cut variant per step; see ``vtscore.eval.voting_iterations._SAFE_GMM_VARIANTS``)
together with the ``task_*__cutdiag.csv`` side frames (one row per step per fit
geometry, carrying the fitted mixture parameters and the whole decomposition
chain).  Produces two independent things:

**Which cut wins** — paired within-step contrasts between every shippable rule
and the production midpoint, on the blended threshold (what a user gets) *and*
on the raw cut (what the rule is worth before the conformal blend damps it).

**Why** — the four-term decomposition of today's error, per step:

``tau_cross - tau_priorfree``     prior/loss mismatch (the ``ln(w_lo/w_hi)`` term)
``tau_priorfree - tau_supervised``  component identification
``tau_supervised - tau_sim_oracle`` Gaussian misspecification
``tau_sim_oracle - tau_test_oracle`` finite-sim-set estimation / transfer

reported in threshold units and in excess-cost units, so "which rule scored
better" becomes "which assumption is wrong and what does it cost".  Plus the
three falsifiable predictions the issue pre-registers: that the realised
``cross - mid`` offset matches its closed form, that the per-step cost penalty
scales with that offset, and that the prior-free crossing beats both incumbents.

Writes ``results/summary_cut.json``, ``results/agg/cut_*.csv``,
``results/figures/cut_*.png`` and a ``results/REPORT_CUT.md`` draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Vote-count windows (inclusive).  Below 6 votes the blend is pure GMM (total
#: authority, but #2799 showed the app shows no trained detector before 7 votes);
#: 6-20 is the ramp, where the blend has partial authority and users are looking.
WINDOWS: dict[str, tuple[int, int]] = {"pure_gmm_2_5": (2, 5), "ramp_6_20": (6, 20)}

#: The arm the ship decision reads: production region voting.
PRODUCTION_ARM_SUBSTR = "dinov3_patch/max_patch"
#: The single-vector arm a winner must not regress (``calculate_gmm_threshold``
#: also backs the cosine/text sort, which has no region max-pool).
CONTROL_ARM_SUBSTR = "whole_image"

#: Rules that could actually ship: unsupervised, computable from the sim scores.
SHIPPABLE: tuple[str, ...] = (
    "pooled_mid",
    "pooled_cross",
    "pooled_priorfree",
    "pooled_rate",
    "pooled_gumbel_cross",
    "pooled_gumbel_priorfree",
    "pooled_gumbel_rate",
)
#: Label-reading diagnostics — bounds and decomposition anchors, never candidates.
ORACLE_VARIANTS: tuple[str, ...] = ("pooled_supervised", "pooled_sim_oracle")

#: The incumbent every candidate is measured against (production since #2833).
INCUMBENT = "pooled_mid"

#: Decomposition terms: name -> (tau_a, tau_b); each is ``tau_a - tau_b``, and
#: consecutive terms telescope to ``tau_cross - tau_test_oracle``.
DECOMPOSITION: tuple[tuple[str, str, str], ...] = (
    ("prior_loss", "tau_cross", "tau_priorfree"),
    ("identification", "tau_priorfree", "tau_supervised"),
    ("misspecification", "tau_supervised", "tau_sim_oracle"),
    ("transfer", "tau_sim_oracle", "tau_test_oracle"),
)

#: Cost-unit counterpart of DECOMPOSITION: the variant whose raw-cut cost stands
#: in for each chain link.  ``tau_test_oracle``'s cost is the row's ``oracle_cost``.
COST_CHAIN: tuple[str, ...] = ("pooled_cross", "pooled_priorfree", "pooled_supervised", "pooled_sim_oracle")


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a fixed-width dump."""
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def _wilcoxon(vals: np.ndarray) -> float | None:
    from scipy.stats import wilcoxon  # noqa: PLC0415

    if len(vals) < 3 or not np.any(vals != 0):
        return None
    _stat, p = wilcoxon(vals)
    return float(p)


def load_cells(cells_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in cells_dir.glob("task_*.csv") if "__sweep" not in p.name and "__cutdiag" not in p.name)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {len(df)} variant rows from {len(files)} cells")
    return df


def load_cutdiag(cells_dir: Path) -> pd.DataFrame:
    files = sorted(cells_dir.glob("task_*__cutdiag.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {len(df)} cut-diagnostic rows from {len(files)} cells")
    return df


def production_blend_sanity(df: pd.DataFrame) -> dict:
    """``pooled_mid`` must reproduce the production blended cut bit-for-bit.

    The fidelity check that licenses every within-step contrast: if the variant
    the harness *labels* as production does not equal the threshold the run
    actually used, the whole family is being re-cut against the wrong baseline.
    """
    keys = ["arm", "category", "seed", "t"]
    base = df[(df["pool_variant"] == "max") & (df["gmm_variant"] == "")].set_index(keys)["threshold"]
    prod = df[df["gmm_variant"] == INCUMBENT].set_index(keys)["threshold"]
    joined = base.to_frame("base").join(prod.to_frame("production_variant"), how="inner")
    if joined.empty:
        return {"n_steps": 0, "max_abs_diff": None, "ok": None}
    diff = (joined["base"] - joined["production_variant"]).abs()
    return {
        "n_steps": int(len(joined)),
        "max_abs_diff": float(diff.max()),
        "ok": bool(diff.max() <= 1e-6),  # thresholds are emitted rounded to 6 dp
    }


# ------------------------------------------------------------------
# Which cut wins
# ------------------------------------------------------------------


def _paired_cells(v: pd.DataFrame, a: str, b: str, lo: int, hi: int, metric_cols: list[str]) -> pd.DataFrame:
    """Per-(arm, category, seed) mean deltas of ``a - b`` on identical steps in [lo, hi].

    The t axis is collapsed first so the test's units are independent cells
    rather than autocorrelated steps within one trajectory.
    """
    keys = ["arm", "category", "seed", "t"]
    w = v[(v["n_votes"] >= lo) & (v["n_votes"] <= hi)]
    va = w[w["gmm_variant"] == a].set_index(keys)[metric_cols]
    vb = w[w["gmm_variant"] == b].set_index(keys)[metric_cols]
    j = va.join(vb, how="inner", lsuffix="_a", rsuffix="_b")
    if j.empty:
        return pd.DataFrame()
    for col in metric_cols:
        j[f"d_{col}"] = j[f"{col}_a"] - j[f"{col}_b"]
    j = j.reset_index()
    return j.groupby(["arm", "category", "seed"])[[f"d_{c}" for c in metric_cols]].mean().reset_index()


def window_table(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Per-window mean metrics per (arm, variant) — the headline table."""
    v = df[df["gmm_variant"] != ""]
    out = []
    for wname, (lo, hi) in WINDOWS.items():
        w = v[(v["n_votes"] >= lo) & (v["n_votes"] <= hi)]
        g = (
            w.groupby(["arm", "gmm_variant"])
            .agg(
                cost=("cost", "mean"),
                fpr=("fpr", "mean"),
                fnr=("fnr", "mean"),
                raw_cut_cost=("raw_cut_cost", "mean"),
                raw_cut_fpr=("raw_cut_fpr", "mean"),
                raw_cut_fnr=("raw_cut_fnr", "mean"),
                gmm_cut=("gmm_cut", "mean"),
                regret=("regret", "mean"),
                degenerate_rate=("degenerate", "mean"),
                fallback_rate=("cut_fallback", "mean"),
                n_steps=("cost", "size"),
            )
            .reset_index()
        )
        g.insert(0, "window", wname)
        out.append(g)
    tbl = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    tbl.to_csv(agg_dir / "cut_window_by_variant.csv", index=False)
    return tbl


def rule_contrasts(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Every shippable rule against the incumbent midpoint, per window per arm.

    ``d_cost`` is the blended threshold's cost (the ship number); ``d_raw_cut_cost``
    is the unblended cut's (the rule number).  They can disagree in magnitude by
    a lot on the ramp — the blend averages the cut with the conformal threshold —
    but a rule that only wins on the blended column is winning by being closer to
    the conformal cut, not by being a better rule.
    """
    v = df[df["gmm_variant"] != ""]
    metrics = ["cost", "fpr", "fnr", "raw_cut_cost", "raw_cut_fpr", "raw_cut_fnr", "gmm_cut"]
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        for cand in (*SHIPPABLE, *ORACLE_VARIANTS):
            if cand == INCUMBENT:
                continue
            for arm, sub in v.groupby("arm"):
                cells = _paired_cells(sub, cand, INCUMBENT, lo, hi, metrics)
                if cells.empty:
                    continue
                entry: dict = {
                    "window": wname,
                    "variant": cand,
                    "vs": INCUMBENT,
                    "arm": arm,
                    "n_cells": int(len(cells)),
                }
                for col in metrics:
                    vals = cells[f"d_{col}"].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    entry[f"mean_d_{col}"] = float(np.mean(vals)) if vals.size else float("nan")
                    entry[f"p_d_{col}"] = _wilcoxon(vals) if vals.size else None
                    if col == "cost":
                        entry["frac_cells_improved"] = float(np.mean(vals < 0)) if vals.size else float("nan")
                rows.append(entry)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "cut_contrasts.csv", index=False)
    return tbl


def oracle_distance(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Mean |raw cut − test-set oracle cut| per (arm, window, variant).

    The issue's conclusiveness criterion: the rule that should ship is the one
    closest to the cut that actually minimises the scored loss, not merely the
    one with the best mean cost — those come apart when a rule is right on
    average and wrong step by step.
    """
    v = df[df["gmm_variant"] != ""].copy()
    v["abs_oracle_gap"] = (v["gmm_cut"] - v["oracle_threshold"]).abs()
    v["signed_oracle_gap"] = v["gmm_cut"] - v["oracle_threshold"]
    out = []
    for wname, (lo, hi) in WINDOWS.items():
        w = v[(v["n_votes"] >= lo) & (v["n_votes"] <= hi)]
        g = (
            w.groupby(["arm", "gmm_variant"])
            .agg(
                mean_abs_gap=("abs_oracle_gap", "mean"),
                median_abs_gap=("abs_oracle_gap", "median"),
                mean_signed_gap=("signed_oracle_gap", "mean"),
                n=("abs_oracle_gap", "size"),
            )
            .reset_index()
        )
        g.insert(0, "window", wname)
        out.append(g)
    tbl = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    tbl.to_csv(agg_dir / "cut_oracle_distance.csv", index=False)
    return tbl


# ------------------------------------------------------------------
# Why: the decomposition and the pre-registered predictions
# ------------------------------------------------------------------


def decomposition_table(diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """The four-term chain in threshold units, per (arm, geometry, window).

    ``residual`` is the telescoping check: the four terms must sum to
    ``tau_cross - tau_test_oracle`` exactly, so a non-zero residual means a NaN
    dropped a link rather than a real disagreement.
    """
    d = diag.copy()
    for name, a, b in DECOMPOSITION:
        d[f"term_{name}"] = d[a] - d[b]
    d["total"] = d["tau_cross"] - d["tau_test_oracle"]
    d["residual"] = d["total"] - sum(d[f"term_{n}"] for n, _a, _b in DECOMPOSITION)

    out = []
    for wname, (lo, hi) in WINDOWS.items():
        w = d[(d["n_votes"] >= lo) & (d["n_votes"] <= hi)]
        agg = {f"term_{n}": (f"term_{n}", "mean") for n, _a, _b in DECOMPOSITION}
        agg |= {f"abs_term_{n}": (f"term_{n}", lambda s: float(np.nanmean(np.abs(s)))) for n, _a, _b in DECOMPOSITION}
        agg |= {
            "total": ("total", "mean"),
            "residual": ("residual", "mean"),
            "n": ("total", "size"),
        }
        g = w.groupby(["arm", "geometry"]).agg(**agg).reset_index()
        g.insert(0, "window", wname)
        out.append(g)
    tbl = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    tbl.to_csv(agg_dir / "cut_decomposition.csv", index=False)
    return tbl


def cost_decomposition(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """The same chain in excess-cost units, on the held-out test set.

    Threshold distance is not the quantity anyone pays; a term that moves the cut
    a long way through a flat region of the cost curve is cheap, and one that
    moves it a little across the elbow is not.  Each link is the difference of
    ``raw_cut_cost`` between consecutive rules on the same step, with the last
    link measured against the test-set oracle cost.
    """
    v = df[df["gmm_variant"].isin(COST_CHAIN)]
    keys = ["arm", "category", "seed", "t", "n_votes"]
    wide = v.pivot_table(index=keys, columns="gmm_variant", values="raw_cut_cost", aggfunc="first")
    oracle = (
        df[df["gmm_variant"] == INCUMBENT].set_index(keys)["oracle_cost"]
        if not df[df["gmm_variant"] == INCUMBENT].empty
        else None
    )
    if wide.empty or oracle is None:
        return pd.DataFrame()
    missing = [c for c in COST_CHAIN if c not in wide.columns]
    if missing:
        # A chain link that never emitted a row (e.g. an oracle variant with no
        # root anywhere) would make the whole decomposition silently wrong; say so.
        common.log(f"cost decomposition skipped - no rows for {missing}")
        return pd.DataFrame()
    wide = wide.join(oracle.to_frame("oracle_cost"), how="inner").reset_index()

    wide["cost_prior_loss"] = wide["pooled_cross"] - wide["pooled_priorfree"]
    wide["cost_identification"] = wide["pooled_priorfree"] - wide["pooled_supervised"]
    wide["cost_misspecification"] = wide["pooled_supervised"] - wide["pooled_sim_oracle"]
    wide["cost_transfer"] = wide["pooled_sim_oracle"] - wide["oracle_cost"]
    wide["cost_total"] = wide["pooled_cross"] - wide["oracle_cost"]

    cols = ["cost_prior_loss", "cost_identification", "cost_misspecification", "cost_transfer", "cost_total"]
    out = []
    for wname, (lo, hi) in WINDOWS.items():
        w = wide[(wide["n_votes"] >= lo) & (wide["n_votes"] <= hi)]
        g = w.groupby("arm")[cols].mean().reset_index()
        g["n"] = w.groupby("arm")[cols[0]].size().to_numpy()
        g.insert(0, "window", wname)
        out.append(g)
    tbl = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    tbl.to_csv(agg_dir / "cut_cost_decomposition.csv", index=False)
    return tbl


def offset_predictions(diag: pd.DataFrame, df: pd.DataFrame, agg_dir: Path) -> dict:
    """Predictions (1) and (2): the closed-form offset, and what it costs.

    (1) ``tau_cross - tau_mid`` should equal ``var*ln(w_lo/w_hi)/(mu_hi-mu_lo)``
        to fit error.  The closed form assumes equal variances, so the residual
        is expected to grow with ``|ln(var_lo/var_hi)|`` — that dependence is
        itself the check, not noise.
    (2) The per-step cost penalty of the crossing should *scale with* that
        offset.  If it does, the pooled-vs-image inversion is explained
        quantitatively (max-pooling inflates ``var_lo``, so the offset and the
        penalty are both larger) rather than by story.
    """
    d = diag[np.isfinite(diag["tau_cross"]) & np.isfinite(diag["tau_mid"])].copy()
    d["actual_offset"] = d["tau_cross"] - d["tau_mid"]
    d["offset_residual"] = d["actual_offset"] - d["pred_offset_equal_var"]
    d["log_var_ratio"] = np.log(d["var_lo"] / d["var_hi"])

    per_geom = (
        d.groupby(["arm", "geometry"])
        .agg(
            mean_actual=("actual_offset", "mean"),
            mean_predicted=("pred_offset_equal_var", "mean"),
            mean_abs_residual=("offset_residual", lambda s: float(np.nanmean(np.abs(s)))),
            corr=("actual_offset", lambda s: float("nan")),
            mean_log_var_ratio=("log_var_ratio", "mean"),
            n=("actual_offset", "size"),
        )
        .reset_index()
    )
    # Pearson r has to be computed pairwise, which the agg above cannot do.
    for i, row in per_geom.iterrows():
        sub = d[(d["arm"] == row["arm"]) & (d["geometry"] == row["geometry"])]
        pair = sub[["actual_offset", "pred_offset_equal_var"]].dropna()
        per_geom.loc[i, "corr"] = float(pair.corr().iloc[0, 1]) if len(pair) > 2 else float("nan")
    per_geom.to_csv(agg_dir / "cut_offset_identity.csv", index=False)

    # (2) join the per-step crossing penalty onto the same step's offset.
    keys = ["arm", "category", "seed", "t"]
    v = df[df["gmm_variant"].isin(("pooled_cross", INCUMBENT))]
    wide = v.pivot_table(index=keys, columns="gmm_variant", values="raw_cut_cost", aggfunc="first")
    if "pooled_cross" not in wide.columns or INCUMBENT not in wide.columns:
        return {"identity": per_geom.to_dict("records"), "scaling": {"n": 0}}
    penalty = pd.DataFrame(index=wide.index)
    penalty["penalty"] = wide["pooled_cross"] - wide[INCUMBENT]
    pooled = d[d["geometry"] == "pooled"].set_index(keys)
    joined = penalty.join(pooled[["actual_offset", "pred_offset_equal_var"]], how="inner").dropna()

    scaling: dict = {"n": int(len(joined))}
    if len(joined) > 10:
        scaling["corr_penalty_vs_offset"] = float(joined["penalty"].corr(joined["actual_offset"]))
        scaling["corr_penalty_vs_predicted"] = float(joined["penalty"].corr(joined["pred_offset_equal_var"]))
        slope, intercept = np.polyfit(joined["actual_offset"], joined["penalty"], 1)
        scaling["slope_penalty_per_offset"] = float(slope)
        scaling["intercept"] = float(intercept)
        # Penalty by offset quintile: a monotone increase is the prediction.
        joined["q"] = pd.qcut(joined["actual_offset"], 5, labels=False, duplicates="drop")
        by_q = joined.groupby("q").agg(offset=("actual_offset", "mean"), penalty=("penalty", "mean")).reset_index()
        by_q.to_csv(agg_dir / "cut_penalty_by_offset_quintile.csv", index=False)
        scaling["by_quintile"] = by_q.to_dict("records")
    return {"identity": per_geom.to_dict("records"), "scaling": scaling}


def evt_evidence(diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Is the Gaussian low component actually the wrong shape, and where?

    Pre-registered directional prediction: ``evt_loglik_gain`` should be positive
    on the **pooled** geometry (a max over ~24 region nodes is an extreme-value
    statistic) and near zero on the **image** geometry (a single vector's score is
    not a maximum of anything).  A gain that is uniform across geometries would
    mean the Gumbel is just a more flexible shape, not the *right* one.
    """
    g = (
        diag.groupby(["arm", "geometry"])
        .agg(
            evt_loglik_gain=("evt_loglik_gain", "mean"),
            frac_evt_better=("evt_loglik_gain", lambda s: float(np.nanmean(np.asarray(s, dtype=float) > 0))),
            evt_fit_rate=("evt_ok", "mean"),
            gmm_fit_rate=("gmm_ok", "mean"),
            n=("evt_ok", "size"),
        )
        .reset_index()
    )
    g.to_csv(agg_dir / "cut_evt_evidence.csv", index=False)
    return g


def tail_alpha_stability(diag: pd.DataFrame, agg_dir: Path) -> dict:
    """Is the oracle cut a *stable* quantile of the fitted Bad component?

    The fallback answer if no crossing rule wins: if the true optimum always sits
    at about the same survival level of the fitted low component, then "cut the
    Bad tail at alpha" is a principled rule with one constant to calibrate, and
    the mixture is being used as a tail model rather than as a classifier.
    Stability is judged on the spread across *cells*, not steps, so autocorrelated
    steps within one trajectory cannot manufacture it.
    """
    out: dict = {}
    for col in ("oracle_lo_sf_gauss", "oracle_lo_sf_evt"):
        sub = diag[(diag["geometry"] == "pooled") & np.isfinite(diag[col])]
        if sub.empty:
            out[col] = {"n": 0}
            continue
        per_cell = sub.groupby(["arm", "category", "seed"])[col].mean()
        q = per_cell.quantile([0.25, 0.5, 0.75])
        out[col] = {
            "n_cells": int(len(per_cell)),
            "median": float(q.loc[0.5]),
            "iqr_lo": float(q.loc[0.25]),
            "iqr_hi": float(q.loc[0.75]),
            # Spread ratio of the middle half; < 3 is the pre-registered bar for
            # "one constant would do", chosen so a rule calibrated on the median
            # stays within a factor of ~1.7 of correct for half the cells.
            "iqr_ratio": float(q.loc[0.75] / q.loc[0.25]) if q.loc[0.25] > 0 else float("inf"),
            "cv_across_cells": float(per_cell.std() / per_cell.mean()) if per_cell.mean() else float("nan"),
        }
        out[col]["stable"] = bool(out[col].get("iqr_ratio", float("inf")) < 3.0)
    (agg_dir / "cut_tail_alpha.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def estimator_variance(diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Step-to-step jitter of each cut within a cell — rule vs *estimator* quality.

    The crossing reads both variances and both weights; the midpoint reads only
    the means. Hypothesis 3 in the issue is that the crossing is the better rule
    and the worse estimator at these sample sizes, which shows up here as a larger
    within-cell standard deviation of consecutive cuts.
    """
    taus = [c for c in diag.columns if c.startswith("tau_")]
    w = diag[(diag["n_votes"] >= 6) & (diag["n_votes"] <= 20) & (diag["geometry"] == "pooled")]
    if w.empty:
        return pd.DataFrame()
    # Successive-difference SD is robust to the genuine drift a trajectory has:
    # a cut that tracks a moving model is not "jittery" just because it moves.
    rows = []
    for (arm, cat, seed), sub in w.sort_values("t").groupby(["arm", "category", "seed"]):
        entry = {"arm": arm, "category": cat, "seed": seed, "n_steps": int(len(sub))}
        for tau in taus:
            vals = sub[tau].to_numpy(dtype=float)
            diffs = np.diff(vals[np.isfinite(vals)])
            entry[tau] = float(np.std(diffs) / np.sqrt(2.0)) if diffs.size >= 2 else float("nan")
        rows.append(entry)
    per_cell = pd.DataFrame(rows)
    tbl = per_cell.groupby("arm")[taus].mean().reset_index()
    tbl.to_csv(agg_dir / "cut_estimator_variance.csv", index=False)
    return tbl


# ------------------------------------------------------------------
# Decisions
# ------------------------------------------------------------------


def decisions(contrasts: pd.DataFrame, gaps: pd.DataFrame, costs: pd.DataFrame, alpha: dict) -> dict:
    """The issue's pre-registered decision rules, evaluated.

    Ship the rule that is closest to the oracle cut **and** wins on cost on the
    production arm's ramp window, provided it does not regress the single-vector
    arm.  Anything else is a negative result with a named cause.
    """
    out: dict = {}
    if contrasts.empty:
        return {"error": "no contrasts"}

    prod = contrasts[
        (contrasts["window"] == "ramp_6_20")
        & (contrasts["arm"].str.contains(PRODUCTION_ARM_SUBSTR))
        & (contrasts["variant"].isin(SHIPPABLE))
    ]
    if prod.empty:
        return {"error": "no production-arm contrasts"}

    best = prod.sort_values("mean_d_cost").iloc[0]
    out["best_by_cost"] = {
        "variant": str(best["variant"]),
        "mean_d_cost": float(best["mean_d_cost"]),
        "p": None if best["p_d_cost"] is None else float(best["p_d_cost"]),
        "mean_d_raw_cut_cost": float(best["mean_d_raw_cut_cost"]),
        "p_raw": None if best["p_d_raw_cut_cost"] is None else float(best["p_d_raw_cut_cost"]),
        "frac_cells_improved": float(best["frac_cells_improved"]),
    }
    out["beats_midpoint"] = bool(
        best["p_d_cost"] is not None and float(best["p_d_cost"]) < 0.05 and float(best["mean_d_cost"]) < 0
    )

    # Closest to the oracle cut, among shippable rules, same arm and window.
    gp = gaps[
        (gaps["window"] == "ramp_6_20")
        & (gaps["arm"].str.contains(PRODUCTION_ARM_SUBSTR))
        & (gaps["gmm_variant"].isin(SHIPPABLE))
    ]
    out["closest_to_oracle"] = None if gp.empty else str(gp.sort_values("mean_abs_gap").iloc[0]["gmm_variant"])

    # Does the winner regress the single-vector arm the cosine/text sort uses?
    ctrl = contrasts[
        (contrasts["window"] == "ramp_6_20")
        & (contrasts["arm"].str.contains(CONTROL_ARM_SUBSTR))
        & (contrasts["variant"] == best["variant"])
    ]
    out["control_arm_delta"] = None if ctrl.empty else float(ctrl.iloc[0]["mean_d_cost"])
    out["regresses_control"] = bool(
        not ctrl.empty
        and float(ctrl.iloc[0]["mean_d_cost"]) > 0
        and ctrl.iloc[0]["p_d_cost"] is not None
        and float(ctrl.iloc[0]["p_d_cost"]) < 0.05
    )
    out["ship"] = bool(
        out["beats_midpoint"]
        and out["closest_to_oracle"] == out["best_by_cost"]["variant"]
        and not out["regresses_control"]
    )

    # Which term in the derivation dominates, in cost units.
    out["dominant_error_term"] = None
    if not costs.empty:
        c = costs[(costs["window"] == "ramp_6_20") & (costs["arm"].str.contains(PRODUCTION_ARM_SUBSTR))]
        if not c.empty:
            terms = {
                k: abs(float(c.iloc[0][f"cost_{k}"]))
                for k in ("prior_loss", "identification", "misspecification", "transfer")
                if f"cost_{k}" in c.columns and np.isfinite(c.iloc[0][f"cost_{k}"])
            }
            if terms:
                out["dominant_error_term"] = max(terms, key=lambda k: terms[k])
                out["error_terms_cost"] = terms
    # The leading hypothesis is confirmed only if the prior/loss term dominates.
    out["leading_hypothesis_confirmed"] = out["dominant_error_term"] == "prior_loss"
    out["tail_alpha_stable"] = bool(alpha.get("oracle_lo_sf_gauss", {}).get("stable", False))
    return out


def make_figures(df: pd.DataFrame, diag: pd.DataFrame, fig_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        common.log(f"matplotlib unavailable ({e}); skipping figures")
        return []
    saved = []

    v = df[df["gmm_variant"].isin((*SHIPPABLE, *ORACLE_VARIANTS))]
    for metric in ("cost", "raw_cut_cost"):
        curves = v.groupby(["arm", "gmm_variant", "n_votes"])[metric].mean().reset_index()
        n_arms = max(1, curves["arm"].nunique())
        fig, axes = plt.subplots(1, n_arms, figsize=(6.5 * n_arms, 4.5), sharey=True, squeeze=False)
        for ax, (arm, sub) in zip(axes[0], curves.groupby("arm"), strict=False):
            for variant, vs in sub.groupby("gmm_variant"):
                style = "--" if variant in ORACLE_VARIANTS else "-"
                ax.plot(vs["n_votes"], vs[metric], style, label=variant, lw=1.2)
            ax.axvspan(6, 20, alpha=0.08, color="gray")
            ax.set_title(arm, fontsize=8)
            ax.set_xlabel("votes")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(f"mean {metric}")
        axes[0][-1].legend(fontsize=6)
        p = fig_dir / f"cut_{metric}_vs_votes.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p.name)

    # The decomposition, as a stacked bar per arm/geometry.
    d = diag[(diag["n_votes"] >= 6) & (diag["n_votes"] <= 20)].copy()
    for name, a, b in DECOMPOSITION:
        d[f"term_{name}"] = d[a] - d[b]
    terms = [f"term_{n}" for n, _a, _b in DECOMPOSITION]
    g = d.groupby(["arm", "geometry"])[terms].mean()
    if not g.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        idx = np.arange(len(g))
        bottom_pos = np.zeros(len(g))
        bottom_neg = np.zeros(len(g))
        for term in terms:
            vals = g[term].to_numpy(dtype=float)
            base = np.where(vals >= 0, bottom_pos, bottom_neg)
            ax.bar(idx, vals, bottom=base, label=term.replace("term_", ""))
            bottom_pos = np.where(vals >= 0, bottom_pos + vals, bottom_pos)
            bottom_neg = np.where(vals < 0, bottom_neg + vals, bottom_neg)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{a}\n{gm}" for a, gm in g.index], fontsize=6)
        ax.set_ylabel("threshold units (cross − test oracle)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        p = fig_dir / "cut_decomposition.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p.name)
    return saved


def write_report(summary: dict, tables: dict, report_path: Path) -> None:
    lines = ["# GMM cut-point study — auto-generated summary (issue #2836)", ""]
    lines.append(f"Variant rows: {summary.get('n_variant_rows')} · diagnostic rows: {summary.get('n_diag_rows')}")
    lines.append("")
    lines.append("## Production-blend sanity (`pooled_mid` == the run's own threshold)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["sanity"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["decisions"], indent=2, default=float))
    lines.append("```")
    for title, key in (
        ("Window means by (arm, variant)", "window"),
        ("Rule contrasts vs the midpoint", "contrasts"),
        ("Distance to the oracle cut", "gaps"),
        ("Decomposition (threshold units)", "decomposition"),
        ("Decomposition (excess-cost units)", "cost_decomposition"),
        ("Extreme-value evidence", "evt"),
        ("Estimator variance (within-cell)", "estimator_variance"),
    ):
        tbl = tables.get(key)
        if tbl is None or (hasattr(tbl, "empty") and tbl.empty):
            continue
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_md(tbl))
    lines.append("")
    lines.append("## Offset predictions (1) and (2)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["offsets"], indent=2, default=float))
    lines.append("```")
    lines.append("")
    lines.append("## Bad-tail alpha stability")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["tail_alpha"], indent=2, default=float))
    lines.append("```")
    lines.append("")
    report_path.write_text("\n".join(lines))
    common.log(f"wrote {report_path}")


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
    if (df["gmm_variant"] != "").sum() == 0:
        common.log("no gmm_variant rows - was the run launched with CALIB_SAFE_THRESHOLDS=1?")
        return 1
    if diag.empty:
        common.log("WARNING: no __cutdiag frames; the decomposition half will be empty")

    sanity = production_blend_sanity(df)
    window = window_table(df, agg_dir)
    contrasts = rule_contrasts(df, agg_dir)
    gaps = oracle_distance(df, agg_dir)
    decomp = decomposition_table(diag, agg_dir) if not diag.empty else pd.DataFrame()
    costs = cost_decomposition(df, agg_dir)
    offsets = offset_predictions(diag, df, agg_dir) if not diag.empty else {}
    evt = evt_evidence(diag, agg_dir) if not diag.empty else pd.DataFrame()
    alpha = tail_alpha_stability(diag, agg_dir) if not diag.empty else {}
    est_var = estimator_variance(diag, agg_dir) if not diag.empty else pd.DataFrame()
    dec = decisions(contrasts, gaps, costs, alpha)
    figs = make_figures(df, diag, fig_dir) if not diag.empty else []

    summary = {
        "n_variant_rows": int((df["gmm_variant"] != "").sum()),
        "n_diag_rows": int(len(diag)),
        "n_cells": int(df[["dataset", "embedder", "category", "seed"]].drop_duplicates().shape[0]),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "sanity": sanity,
        "decisions": dec,
        "offsets": offsets,
        "tail_alpha": alpha,
        "figures": figs,
    }
    (common.RESULTS / "summary_cut.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(
        summary,
        {
            "window": window,
            "contrasts": contrasts,
            "gaps": gaps,
            "decomposition": decomp,
            "cost_decomposition": costs,
            "evt": evt,
            "estimator_variance": est_var,
        },
        common.RESULTS / "REPORT_CUT.md",
    )
    common.log("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Analysis for the #3329 goodness-of-fit run.

Scores the four pre-registered hypotheses in
``docs/experiments/2026-08-30-fit-quality-3329/PREREG.md`` and writes one CSV
per hypothesis under ``agg/`` plus a ``summary_fitq3329.json`` carrying the
decisions.  Every bar in the pre-registration is a module-level constant here,
so the selftest can plant an answer against the same number the report quotes.

Usage::

    python analyze_fitq_3329.py --results <CALIB_RESULTS> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import _cells_io

# --- Pre-registered bars (PREREG.md, do not retune after the run) -------------

#: H1: median tail ratio the Gaussian must miss by, region arm and binary control.
H1_TAIL_RATIO_REGION = 1.5
H1_TAIL_RATIO_BINARY = 1.2
#: H2: median Bad-mode skewness on the region arm.
H2_SKEW_REGION = 0.5
#: H3: the anchors count as inert below this movement in the fitted low mean,
#: and this M-step mass share.
H3_DMU_MAX = 0.01
H3_MASS_MAX = 1e-3
#: H4: the payoff gate - misfit must predict regret to license a fit-replacement
#: programme.
H4_PARTIAL_R2 = 0.05
#: Everything is "resolvable" at more than this many standard errors.
SE_BAR = 2.0

#: The scope the labelled statistics are read from: the geometry the cut is
#: actually taken in.  ``sim:image`` is retained for H2's paired contrast.
POOLED_SCOPE = "sim:pooled"
IMAGE_SCOPE = "sim:image"

#: Cell keys a paired contrast is taken within.
PAIR_KEYS = ["dataset", "category", "seed", "t"]

#: Below this many labelled items in either class the shape statistics decline
#: (they are third and fourth moments); mirrored from ``fit_quality.MIN_CLASS_N``
#: so the report can say what share of steps were unresolvable rather than
#: averaging over a filtered subset silently.
MIN_CLASS_N = 30


def _arm(frame: pd.DataFrame) -> pd.Series:
    """``embedder/style`` - the geometry a row belongs to."""
    return frame["embedder"].astype(str) + "/" + frame["style"].astype(str)


def load_fitq(results: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Every ``__fitq`` frame, concatenated, with a count of what was dropped."""
    files = _cells_io.side_frame_files(results / "cells", "__fitq")
    frames: list[pd.DataFrame] = []
    dropped = {"missing": 0, "empty": 0, "unreadable": 0, "header_only": 0}
    for p in files:
        try:
            if p.stat().st_size == 0:
                dropped["empty"] += 1
                continue
            f = pd.read_csv(p)
        except Exception:
            dropped["unreadable"] += 1
            continue
        if f.empty:
            dropped["header_only"] += 1
            continue
        frames.append(f)
    if not frames:
        return pd.DataFrame(), dropped
    out = pd.concat(frames, ignore_index=True)
    out["arm"] = _arm(out)
    return out, dropped


def load_main(results: Path) -> pd.DataFrame:
    """The base metric rows, for the regret column H4 regresses on."""
    out, _prov = _cells_io.load_cells(results / "cells")
    if out.empty:
        return out
    # The base row only: variant arms carry their own cuts and would enter the
    # regression as extra, non-independent rows for the same step.
    #
    # The two columns do NOT mark their base the same way, and assuming they did
    # silently emptied the whole frame on the first real run: `gmm_variant` is
    # blank on the base cut, but `pool_variant` is stamped with the base
    # pooling's own NAME, "max" (a repool arm carries "mean", "topk", ...), so
    # filtering it to blank dropped all 192 cells and H4 scored as a null it had
    # never actually computed. The selftest passed because its fixture planted a
    # blank `pool_variant`, which the harness never emits. Name each column's
    # base values rather than assuming blank means base.
    for col, base_values in (
        ("gmm_variant", ("", "nan")),
        ("pool_variant", ("", "nan", "max")),
    ):
        if col in out.columns:
            out = out[out[col].fillna("").astype(str).isin(base_values)]
    out["arm"] = _arm(out)
    return out


def _median_se(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Median and its bootstrap standard error.

    A median rather than a mean throughout: ``tail_ratio`` is a ratio with a
    heavy right tail, where a mean is dominated by the handful of steps whose
    predicted mass nearly vanished.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return float(np.median(x)), float(np.std(boots, ddof=1))


def h1_tail(fq: pd.DataFrame) -> pd.DataFrame:
    """Per-arm tail calibration at the cut: does the fit under-predict it?"""
    rows: list[dict[str, Any]] = []
    sub = fq[fq["scope"] == POOLED_SCOPE]
    for arm, g in sub.groupby("arm", sort=True):
        med, se = _median_se(g["tail_ratio"].to_numpy())
        bar = H1_TAIL_RATIO_REGION if "max_patch" in str(arm) else H1_TAIL_RATIO_BINARY
        rows.append(
            {
                "arm": arm,
                "n_steps": int(np.isfinite(g["tail_ratio"].to_numpy()).sum()),
                "tail_ratio_median": med,
                "tail_ratio_se": se,
                "bar": bar,
                # "Resolvably above 1" and "past the pre-registered bar" are two
                # different claims; both are reported because a ratio can be
                # significantly >1 and still far short of the bar.
                "resolvable_above_one": bool(
                    math.isfinite(med) and math.isfinite(se) and se > 0 and (med - 1.0) / se > SE_BAR
                ),
                "meets_bar": bool(math.isfinite(med) and med > bar),
            }
        )
    return pd.DataFrame(rows)


def h2_shape(fq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bad-mode skewness per arm, and the paired max-pooling contrast.

    Two readings, as pre-registered: across arms (``dinov3/max_patch`` against
    ``dinov3/whole_image``) and *within* a max_patch run (``sim:pooled`` against
    ``sim:image`` on the same media under the same model).  The paired reading
    governs if they disagree.
    """
    levels: list[dict[str, Any]] = []
    sub = fq[fq["scope"] == POOLED_SCOPE]
    for arm, g in sub.groupby("arm", sort=True):
        med, se = _median_se(g["shape_skew_neg"].to_numpy())
        n_ok = int((g["shape_n_neg"] >= MIN_CLASS_N).sum()) if "shape_n_neg" in g.columns else 0
        levels.append(
            {
                "arm": arm,
                "n_steps": int(np.isfinite(g["shape_skew_neg"].to_numpy()).sum()),
                "n_steps_above_class_floor": n_ok,
                "skew_neg_median": med,
                "skew_neg_se": se,
                "meets_bar": bool(math.isfinite(med) and med > H2_SKEW_REGION and "max_patch" in str(arm)),
            }
        )

    # The within-run paired contrast: same cell, same step, two poolings.
    pooled = fq[fq["scope"] == POOLED_SCOPE]
    image = fq[fq["scope"] == IMAGE_SCOPE]
    keys = [*PAIR_KEYS, "arm"]
    merged = pooled.merge(image, on=keys, suffixes=("_pooled", "_image"))
    paired: list[dict[str, Any]] = []
    for arm, g in merged.groupby("arm", sort=True):
        d = (g["shape_skew_neg_pooled"] - g["shape_skew_neg_image"]).to_numpy()
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        mean = float(np.mean(d))
        se = float(np.std(d, ddof=1) / math.sqrt(d.size)) if d.size > 1 else float("nan")
        paired.append(
            {
                "arm": arm,
                "n_pairs": int(d.size),
                "d_skew_pooled_minus_image": mean,
                "se": se,
                "resolvable": bool(math.isfinite(se) and se > 0 and abs(mean) / se > SE_BAR),
            }
        )
    return pd.DataFrame(levels), pd.DataFrame(paired)


def h3_anchoring(fq: pd.DataFrame) -> pd.DataFrame:
    """Do the anchors move the shipped fit at all?"""
    rows: list[dict[str, Any]] = []
    folds = fq[fq["scope"].astype(str).str.startswith("fold")]
    for arm, g in folds.groupby("arm", sort=True):
        dmu = np.abs(g["anchored_dmu_lo"].to_numpy())
        mass = g["anchor_mass_frac"].to_numpy()
        dmu_med, dmu_se = _median_se(dmu)
        mass_med, _ = _median_se(mass)
        rows.append(
            {
                "arm": arm,
                "n_fold_rows": int(len(g)),
                "abs_dmu_lo_median": dmu_med,
                "abs_dmu_lo_se": dmu_se,
                "anchor_mass_frac_median": mass_med,
                "kappa": float(g["anchor_kappa"].median()) if "anchor_kappa" in g.columns else float("nan"),
                # "Inert" is the H3 claim: the anchors neither carry mass nor
                # move the mean.  Both halves must hold.
                "inert": bool(
                    math.isfinite(dmu_med)
                    and dmu_med < H3_DMU_MAX
                    and math.isfinite(mass_med)
                    and mass_med < H3_MASS_MAX
                ),
            }
        )
    return pd.DataFrame(rows)


def h4_regret(fq: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Does misfit predict regret? The gate that decides whether the line continues.

    Regresses ``regret_honest`` on ``log(tail_ratio)`` and reports the partial
    R² over a model carrying ``log(n_test_pos)`` alone - so a correlation that
    is really "starved cells are both badly fitted and expensive" does not read
    as "misfit costs money".
    """
    rows: list[dict[str, Any]] = []
    if fq.empty or main.empty:
        return pd.DataFrame(rows)
    keys = [*PAIR_KEYS, "arm"]
    sub = fq[fq["scope"] == POOLED_SCOPE][[*keys, "tail_ratio"]]
    regret_col = "regret_honest" if "regret_honest" in main.columns else "regret"
    cols = [*keys, regret_col]
    if "n_test_pos" in main.columns:
        cols.append("n_test_pos")
    m = main[[c for c in cols if c in main.columns]]
    j = sub.merge(m, on=keys, how="inner")

    for arm, g in j.groupby("arm", sort=True):
        x = np.log(g["tail_ratio"].to_numpy())
        y = g[regret_col].to_numpy()
        npos = g["n_test_pos"].to_numpy() if "n_test_pos" in g.columns else np.full(x.shape, np.nan)
        ok = np.isfinite(x) & np.isfinite(y)
        if "n_test_pos" in g.columns:
            ok &= np.isfinite(npos) & (npos > 0)
        if ok.sum() < 10:
            continue
        x, y = x[ok], y[ok]
        controls = [np.ones_like(x)]
        if "n_test_pos" in g.columns:
            controls.append(np.log(npos[ok]))
        base = np.column_stack(controls)
        full = np.column_stack([*controls, x])
        r2_base = _ols_r2(base, y)
        r2_full = _ols_r2(full, y)
        beta, se = _ols_slope(full, y)
        rows.append(
            {
                "arm": arm,
                "n": int(ok.sum()),
                "slope": beta,
                "slope_se": se,
                "r2_base": r2_base,
                "r2_full": r2_full,
                "partial_r2": (r2_full - r2_base) / (1.0 - r2_base) if r2_base < 1.0 else float("nan"),
                "resolvable": bool(math.isfinite(se) and se > 0 and beta / se > SE_BAR),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["meets_bar"] = out["resolvable"] & (out["partial_r2"] >= H4_PARTIAL_R2)
    return out


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _ols_slope(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Last column's coefficient and its standard error."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return float(beta[-1]), float("nan")
    sigma2 = float(np.sum(resid**2)) / dof
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float(beta[-1]), float("nan")
    return float(beta[-1]), float(math.sqrt(max(0.0, cov[-1, -1])))


def _figures(fq: pd.DataFrame, main_frame: pd.DataFrame, out: Path, args: Any) -> list[str]:
    """Every figure this report owes, from the same frames as its tables.

    Kept behind ``--no-figures`` so a re-analysis that only wants the CSVs does
    not pay for matplotlib, and so the selftest (which plants numbers, not
    pictures) can skip it.
    """
    import figures_fitq_3329 as F  # noqa: PLC0415

    figdir = out / "figures"
    baseline = args.baseline or (out / "text_baseline.csv")
    written = F.quality_pair(main_frame, figdir, baseline)
    written += F.statistics_over_clicks(
        fq,
        figdir,
        bars={"h2": H2_SKEW_REGION, "h3_mass": H3_MASS_MAX, "h3_dmu": H3_DMU_MAX},
    )
    if args.worked:
        worked = Path(args.worked)
        written += F.worked_cell(
            {
                "siglip": worked / "worked_0.npz",
                "siglip+dinov3_patch": worked / "worked_12.npz",
            },
            figdir,
            checkpoints=(5, 20, 50, 100),
        )
    return written


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse the #3329 goodness-of-fit run.")
    ap.add_argument("--results", required=True, help="CALIB_RESULTS directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--baseline",
        default=None,
        help="text_baseline.py CSV: the click-0 anchor the quality curves are drawn from "
        "(default: <out>/text_baseline.csv)",
    )
    ap.add_argument(
        "--worked",
        default=None,
        metavar="DIR",
        help="directory of worked_cell_3329.py .npz captures, for the fit-overlay figure",
    )
    args = ap.parse_args(argv)

    results = Path(args.results)
    out = Path(args.out)
    (out / "agg").mkdir(parents=True, exist_ok=True)

    fq, dropped = load_fitq(results)
    main_frame = load_main(results)
    if fq.empty:
        print("no fit-quality rows found; nothing to analyse")
        return 1

    _cells_io.assert_one_opening(main_frame, "analyze_fitq_3329")

    h1 = h1_tail(fq)
    h2_levels, h2_paired = h2_shape(fq)
    h3 = h3_anchoring(fq)
    h4 = h4_regret(fq, main_frame)

    for name, frame in (
        ("h1_tail_calibration", h1),
        ("h2_shape_levels", h2_levels),
        ("h2_shape_paired", h2_paired),
        ("h3_anchoring", h3),
        ("h4_regret", h4),
    ):
        frame.to_csv(out / "agg" / f"{name}.csv", index=False)

    if not args.no_figures:
        figures = _figures(fq, main_frame, out, args)
        if figures:
            print("figures: " + ", ".join(figures))

    # The denominator, stated rather than implied.
    shape_floor_share = float("nan")
    pooled = fq[fq["scope"] == POOLED_SCOPE]
    if not pooled.empty and "shape_n_pos" in pooled.columns:
        shape_floor_share = float((pooled["shape_n_pos"] < MIN_CLASS_N).mean())
    fit_ok_share = float(pooled["fit_ok"].astype(bool).mean()) if not pooled.empty else float("nan")

    summary = {
        "n_fitq_rows": int(len(fq)),
        "n_cells_dropped": dropped,
        "arms": sorted(fq["arm"].unique().tolist()),
        "h1_any_arm_meets_bar": bool(h1["meets_bar"].any()) if not h1.empty else False,
        "h2_region_meets_bar": bool(h2_levels["meets_bar"].any()) if not h2_levels.empty else False,
        "h3_anchors_inert": bool(h3["inert"].all()) if not h3.empty else False,
        "h4_misfit_predicts_regret": bool(h4["meets_bar"].any()) if not h4.empty else False,
        # The interpretability guards from the pre-registration, reported
        # whether or not they bound - a filtered subset has to be visible.
        "share_steps_below_class_floor": shape_floor_share,
        "share_steps_with_a_fit": fit_ok_share,
        "shape_unresolved": bool(math.isfinite(shape_floor_share) and shape_floor_share > 0.5),
        "degenerate_fit_rate_high": bool(math.isfinite(fit_ok_share) and fit_ok_share < 0.8),
    }
    # The headline reading, spelled out so the report cannot quietly invert it.
    summary["verdict"] = (
        "misfit is real and predicts regret - a better fit is worth building"
        if summary["h4_misfit_predicts_regret"]
        else "misfit may be real but does not predict regret at the operating point"
    )
    (out / "summary_fitq3329.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

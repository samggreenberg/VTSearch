"""Stage 2 (cut-rule study, #2836): which cut, and which term in the derivation is wrong.

Consumes a ``CALIB_SAFE_THRESHOLDS=1`` run's ``results/cells/task_*.csv`` (one row
per cut variant per step; see ``vtscore.eval.arms_safe_gmm._SAFE_GMM_VARIANTS``)
together with the ``task_*__cutdiag.csv`` side frames (one row per step per fit
geometry, carrying the fitted mixture parameters and the whole decomposition
chain).  Produces two independent things:

**Which cut wins** — paired within-step contrasts between every shippable rule
and (a) the run's own production row, which is what "beats what we ship" means
and cannot go stale, and (b) the production midpoint, which is #2836's baseline
and did go stale (#2846).  Both on the blended threshold (what a user gets) *and*
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

**The tail sweep** (#2881) rides along in both halves: ``tail_alpha_stability``
says where the oracle cut sits in the fitted Bad tail, and ``tail_alpha_curve``
says what it costs to aim there, as a function of the one constant.  Its
pre-registration is ``docs/experiments/2026-08-04-gmm-cut/PREREG-2881.md``.

Writes ``results/summary_cut.json``, ``results/agg/cut_*.csv``,
``results/figures/cut_*.png`` and a ``results/REPORT_CUT.md`` draft.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import assert_one_opening, main_frame_files  # noqa: E402

from vtscore.eval.cut_rules import TAIL_ALPHA_PREREGISTERED, TAIL_RULES  # noqa: E402
from vtscore.eval.transfer_rules import BAGGED_FIT_RULES, TRANSFER_ORACLE_RULES  # noqa: E402

#: Vote-count windows (inclusive).  Below 6 votes the blend is pure GMM (total
#: authority, but #2799 showed the app shows no trained detector before 7 votes);
#: 6-20 is the ramp, where the blend has partial authority and users are looking.
WINDOWS: dict[str, tuple[int, int]] = {"pure_gmm_2_5": (2, 5), "ramp_6_20": (6, 20)}

#: The arm the ship decision reads: production region voting.
PRODUCTION_ARM_SUBSTR = "dinov3_patch/max_patch"
#: The single-vector arm a winner must not regress (``calculate_gmm_threshold``
#: also backs the cosine/text sort, which has no region max-pool).
CONTROL_ARM_SUBSTR = "whole_image"

#: #2881's tail-quantile sweep, as cell-variant names.  Imported rather than
#: respelled: the analyzer's allowlist going out of step with the rule table is
#: the exact failure ``unclassified_variants`` exists to shout about, and a rule
#: family that arrives seven at a time is where a hand-maintained list would
#: first lose one.
TAIL_VARIANTS: tuple[str, ...] = tuple(f"pooled_{r}" for r in TAIL_RULES)

#: #2883's label-free arm: the production Gaussian rules re-cut over bootstrap
#: refits of the haystack.  Shippable in principle - it reads no labels - but
#: pre-registered as **exploratory** and kept out of the ship gate below, because
#: #2883 item 1 asks for the characterisation of ``transfer`` before a remedy,
#: and a remedy that wins in the run that diagnoses the disease is the
#: wrong-but-plausible result this line has already paid for twice.
BAGFIT_VARIANTS: tuple[str, ...] = tuple(f"pooled_{r}" for r in BAGGED_FIT_RULES)

#: #2883's label-reading readings of the same sim set as ``pooled_sim_oracle``:
#: the four subsample levels and the two variance-reduced estimators.  They are
#: oracles, so they join ``ORACLE_VARIANTS`` rather than the allowlist - but they
#: are the arms that decide whether ``family_headroom_exhausted`` is measuring a
#: bound or one estimator, so ``analyze_transfer.py`` reads them by name.
TRANSFER_ORACLE_VARIANTS: tuple[str, ...] = tuple(f"pooled_{r}" for r in TRANSFER_ORACLE_RULES)

#: Rules that could actually ship: unsupervised, computable from the sim scores.
SHIPPABLE: tuple[str, ...] = (
    "pooled_mid",
    "pooled_cross",
    "pooled_priorfree",
    "pooled_rate",
    "pooled_gumbel_cross",
    "pooled_gumbel_priorfree",
    "pooled_gumbel_rate",
    "pooled_gumbel_any_cross",
    "pooled_gumbel_any_priorfree",
    "pooled_gumbel_any_rate",
    *TAIL_VARIANTS,
    *BAGFIT_VARIANTS,
)

#: Measured and reported, but **not eligible to be the ship candidate**.
#:
#: The tail sweep varies one free parameter, so handing all seven levels to a
#: 5 %-bar ship gate would be seven shots at it — and the sweep would then
#: "win" on this run at whatever alpha the noise favoured, which is precisely the
#: wrong-but-plausible result that has cost this study line two runs already.
#: The pre-registered constant (0.158, #2846's median over 511 cells) is the only
#: tail level that can ship; the other six exist to show the *shape* of the cost
#: curve in alpha, which is the actual claim the stability finding makes.  If the
#: curve turns out to peak somewhere else entirely, that is a finding for the
#: next pre-registration, not a rule to ship off this run.
SWEEP_ONLY: tuple[str, ...] = (
    *(v for v in TAIL_VARIANTS if v != f"pooled_tail_a{round(TAIL_ALPHA_PREREGISTERED * 1000):03d}"),
    # #2883's label-free bagged arms: measured and reported, deliberately unable
    # to win.  See BAGFIT_VARIANTS.
    *BAGFIT_VARIANTS,
)
#: What the ship decision, the oracle-distance tie-break and ``best_by_cost`` read.
SHIP_ELIGIBLE: tuple[str, ...] = tuple(v for v in SHIPPABLE if v not in SWEEP_ONLY)

#: Label-reading diagnostics — bounds and decomposition anchors, never candidates.
ORACLE_VARIANTS: tuple[str, ...] = ("pooled_supervised", "pooled_sim_oracle", *TRANSFER_ORACLE_VARIANTS)

#: ``pooled_tail_a158`` -> ``0.158``.  Parsed from the *data's* variant names
#: rather than read off the imported grid, so re-analyzing an older run whose
#: sweep used different levels still labels its own curve correctly.
_TAIL_RE = re.compile(r"^pooled_tail_a(\d{3})$")


def tail_alpha_of(variant: str) -> float | None:
    m = _TAIL_RE.match(str(variant))
    return int(m.group(1)) / 1000.0 if m else None


#: Deliberately not ship candidates: ``xcal_only`` is the no-blend control, and
#: the ``image_*`` family is the single-vector geometry arm, measured for
#: comparison rather than to ship.  Anything in the cells that is in *none* of
#: these three sets is a rule someone added to ``_SAFE_GMM_VARIANTS`` without
#: telling the analyzer, and :func:`unclassified_variants` says so — see there
#: for why that has to be loud rather than silent.
NON_CANDIDATES: tuple[str, ...] = ("xcal_only",)
#: ``skyline_`` is here for the same reason ``ORACLE_VARIANTS`` exists: the #3322
#: skyline arms read ground-truth labels the app can never see, so they are
#: diagnostics that decompose the cost rather than rules competing to ship.
NON_CANDIDATE_PREFIXES: tuple[str, ...] = ("image_", "skyline_")

#: The variant that *reconstructs* the production rule of #2836's era.  Kept as
#: the historical contrast, but it is a reconstruction and reconstructions go
#: stale: by #2846's re-measure production had moved to the fold-anchored
#: threshold and this rule reproduced it on 16 % of steps.  See ``BASE_ROW``.
INCUMBENT = "pooled_mid"

#: The run's *own* production row — no cut variant, the pooled max geometry the
#: app ships — identified by ``(pool_variant, gmm_variant)``.  Pairing against
#: this is what "does the rule beat what we ship" actually means, and unlike
#: ``INCUMBENT`` it cannot expire when the app changes its threshold (#2846).
BASE_ROW: tuple[str, str] = ("max", "")

#: Metrics the base row shares with a cut variant.  It has no ``gmm_cut`` and no
#: raw (unblended) cut of its own, so the rule-quality columns are meaningless
#: here; the blended cost *is* the ship number.
BASE_METRICS: tuple[str, ...] = ("cost", "fpr", "fnr")

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
    files = main_frame_files(cells_dir)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {len(df)} variant rows from {len(files)} cells")
    assert_one_opening(df, "analyze_cut.py")
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


def unclassified_variants(df: pd.DataFrame) -> list[str]:
    """Cut variants present in the cells that this analyzer does not classify.

    ``SHIPPABLE`` is an allowlist, and the ship decision, both contrast tables
    and the oracle-distance tie-break all read through it.  So a rule added to
    ``_SAFE_GMM_VARIANTS`` but not added here does not fail — it is *silently
    omitted from the verdict* while still appearing in the window means, which
    is the exact shape of wrong table this study line keeps producing.  #2881's
    ``tail_a*`` family was the seven-at-a-time case this was built for; it now
    imports its own names from the rule table, which is stronger than a warning,
    but the warning stays for the next rule added by hand.
    """
    known = {*SHIPPABLE, *ORACLE_VARIANTS, *NON_CANDIDATES}
    present = {str(v) for v in df["gmm_variant"].unique() if str(v) != ""}
    unknown = sorted(v for v in present - known if not any(v.startswith(p) for p in NON_CANDIDATE_PREFIXES))
    if unknown:
        common.log(f"WARNING: {len(unknown)} variant(s) not classified by this analyzer: {', '.join(unknown)}")
        common.log("         -> add them to SHIPPABLE (or NON_CANDIDATES); until then they cannot win or ship")
    return unknown


def production_blend_sanity(df: pd.DataFrame) -> dict:
    """``pooled_mid`` must reproduce the production blended cut bit-for-bit.

    The fidelity check that licenses every within-step contrast: if the variant
    the harness *labels* as production does not equal the threshold the run
    actually used, the whole family is being re-cut against the wrong baseline.

    When it fails, the *reason* is usually not a harness bug but the incumbent
    shipping out from under the study - this is exactly what #2846's re-measure
    found, where production had moved to the fold-anchored threshold and
    ``pooled_mid`` was still bit-for-bit correct on every step that took the old
    path.  Those two diagnoses call for opposite responses, and telling them
    apart is mechanical, so the breakdown by ``threshold_provenance`` is reported
    rather than left to whoever reads ``ok: false`` next.
    """
    keys = ["arm", "category", "seed", "t"]
    cols = ["threshold"] + (["threshold_provenance"] if "threshold_provenance" in df else [])
    base = df[(df["pool_variant"] == "max") & (df["gmm_variant"] == "")].set_index(keys)[cols]
    prod = df[df["gmm_variant"] == INCUMBENT].set_index(keys)["threshold"]
    joined = base.join(prod.to_frame("production_variant"), how="inner")
    if joined.empty:
        return {"n_steps": 0, "max_abs_diff": None, "ok": None}
    diff = (joined["threshold"] - joined["production_variant"]).abs()
    out = {
        "n_steps": int(len(joined)),
        "max_abs_diff": float(diff.max()),
        "ok": bool(diff.max() <= 1e-6),  # thresholds are emitted rounded to 6 dp
    }
    if not out["ok"] and "threshold_provenance" in joined:
        # Per production code path: which ones `pooled_mid` still reproduces, and
        # which ones it cannot because the app no longer computes it that way.
        by = joined.assign(mismatch=diff > 1e-6).groupby(joined["threshold_provenance"].fillna(""))
        out["by_provenance"] = {
            str(name): {
                "n_steps": int(len(g)),
                "n_mismatched": int(g["mismatch"].sum()),
                "max_abs_diff": float(diff.loc[g.index].max()),
            }
            for name, g in by
        }
    return out


# ------------------------------------------------------------------
# Which cut wins
# ------------------------------------------------------------------


def _pair_frames(va: pd.DataFrame, vb: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Per-(arm, category, seed) mean deltas of ``va - vb`` on identical steps.

    The t axis is collapsed first so the test's units are independent cells
    rather than autocorrelated steps within one trajectory.
    """
    j = va.join(vb, how="inner", lsuffix="_a", rsuffix="_b")
    if j.empty:
        return pd.DataFrame()
    for col in metric_cols:
        j[f"d_{col}"] = j[f"{col}_a"] - j[f"{col}_b"]
    j = j.reset_index()
    return j.groupby(["arm", "category", "seed"])[[f"d_{c}" for c in metric_cols]].mean().reset_index()


def _window(v: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    return v[(v["n_votes"] >= lo) & (v["n_votes"] <= hi)]


def _fill_metric(entry: dict, col: str, series: pd.Series) -> None:
    """Write one metric's mean / SEM / p (and cost's improved-cell share) into *entry*.

    The SEM is over **cells**, matching the pairing unit, and is what makes "these
    two rules are indistinguishable" a statement rather than an impression - the
    alpha sweep in :func:`tail_alpha_curve` needs exactly that to say whether its
    cost curve is flat near the optimum or a knife-edge.
    """
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    entry[f"mean_d_{col}"] = float(np.mean(vals)) if vals.size else float("nan")
    entry[f"sem_d_{col}"] = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
    entry[f"p_d_{col}"] = _wilcoxon(vals) if vals.size else None
    if col == "cost":
        entry["frac_cells_improved"] = float(np.mean(vals < 0)) if vals.size else float("nan")


def _paired_cells(v: pd.DataFrame, a: str, b: str, lo: int, hi: int, metric_cols: list[str]) -> pd.DataFrame:
    """:func:`_pair_frames` between two cut variants over the window [lo, hi]."""
    keys = ["arm", "category", "seed", "t"]
    w = _window(v, lo, hi)
    va = w[w["gmm_variant"] == a].set_index(keys)[metric_cols]
    vb = w[w["gmm_variant"] == b].set_index(keys)[metric_cols]
    return _pair_frames(va, vb, metric_cols)


def _paired_cells_vs_base(v: pd.DataFrame, a: str, lo: int, hi: int, metric_cols: list[str]) -> pd.DataFrame:
    """:func:`_pair_frames` between a cut variant and the run's own base row."""
    keys = ["arm", "category", "seed", "t"]
    w = _window(v, lo, hi)
    pool, gmm = BASE_ROW
    va = w[w["gmm_variant"] == a].set_index(keys)[metric_cols]
    vb = w[(w["pool_variant"] == pool) & (w["gmm_variant"] == gmm)].set_index(keys)[metric_cols]
    return _pair_frames(va, vb, metric_cols)


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
                    _fill_metric(entry, col, cells[f"d_{col}"])
                rows.append(entry)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "cut_contrasts.csv", index=False)
    return tbl


def base_row_contrasts(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Every rule against **the threshold the run actually used**, per window per arm.

    :func:`rule_contrasts` answers "does this rule beat the midpoint".  That was
    the same question as "does it beat what we ship" only for as long as the app
    computed the midpoint, and #2846 found out the hard way that it had stopped:
    ``pooled_mid`` was still bit-for-bit correct, production had simply moved to
    the fold-anchored threshold, and every "beats the shipped midpoint" line in
    the study silently stopped meaning what it said.

    The base row cannot go stale, because it is not a reconstruction — it is the
    step's own outcome under whatever production did that day.  ``base_provenance``
    records which path that was, so a contrast is never read without knowing what
    it was against.
    """
    metrics = list(BASE_METRICS)
    pool, gmm = BASE_ROW
    has_prov = "threshold_provenance" in df
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        for cand in (*SHIPPABLE, *ORACLE_VARIANTS):
            for arm, sub in df.groupby("arm"):
                cells = _paired_cells_vs_base(sub, cand, lo, hi, metrics)
                if cells.empty:
                    continue
                base = _window(sub, lo, hi)
                base = base[(base["pool_variant"] == pool) & (base["gmm_variant"] == gmm)]
                prov = base["threshold_provenance"].fillna("").astype(str) if has_prov else pd.Series(dtype=str)
                entry: dict = {
                    "window": wname,
                    "variant": cand,
                    "vs": "base_row",
                    "arm": arm,
                    "n_cells": int(len(cells)),
                    "base_provenance": "|".join(sorted(prov.unique())) if len(prov) else "",
                }
                for col in metrics:
                    _fill_metric(entry, col, cells[f"d_{col}"])
                rows.append(entry)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(agg_dir / "cut_contrasts_vs_base.csv", index=False)
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


#: Every cut the chain differences, in order.  A row missing any one of these
#: cannot contribute to *any* term without skewing the rest (see
#: :func:`_complete_chain_rows`).
DECOMPOSITION_CUTS: tuple[str, ...] = tuple(dict.fromkeys([c for _n, a, b in DECOMPOSITION for c in (a, b)]))


def _complete_chain_rows(frame: pd.DataFrame, chain: "list[str] | tuple[str, ...]") -> "pd.Series":
    """Mask of rows where every link in ``chain`` is present.

    The decomposition only telescopes when each term is averaged over the *same*
    steps.  ``DataFrame.mean()`` skips NaN per column, so a row missing one link
    silently shrinks that link's sample while leaving ``total`` at full size -
    the terms then sum to something other than the total, by an amount nobody
    can see.  Oracle variants are exactly the links that go missing: they do not
    fall back, so a step where the oracle cut is not finite emits no row at all
    (``vtscore.eval.arms_safe_gmm._safe_gmm_variant_rows``).  Dropping those
    rows wholesale is what keeps the arithmetic honest; the callers report how
    many were dropped rather than swallowing it.
    """
    present = [c for c in chain if c in frame.columns]
    if len(present) != len(chain):
        return pd.Series(False, index=frame.index)
    return frame[present].notna().all(axis=1)


def _with_incomplete_count(agg: pd.DataFrame, rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Attach ``n_incomplete`` (rows dropped for a missing chain link) to ``agg``.

    Outer-merged, so a group whose rows were *all* incomplete still appears -
    with empty terms and its true drop count - instead of vanishing from the
    table as though it had never been measured.
    """
    dropped = rows[~rows["_complete"]].groupby(keys).size().rename("n_incomplete").reset_index()
    out = agg.merge(dropped, on=keys, how="outer")
    for col in ("n", "n_incomplete"):
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    return out


def decomposition_table(diag: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """The four-term chain in threshold units, per (arm, geometry, window).

    Aggregated over **complete** rows only - those carrying every cut in the
    chain - so that all four terms and ``total`` average the same steps and the
    sum telescopes exactly.  ``n`` counts the rows that made it; ``n_incomplete``
    counts the rows dropped for a missing link, which is the number that used to
    disappear silently.  ``residual`` is then the arithmetic check it was always
    described as: it must be 0, and a non-zero value means the terms genuinely
    disagree rather than that a NaN ate a link.
    """
    d = diag.copy()
    d["_complete"] = _complete_chain_rows(d, DECOMPOSITION_CUTS)
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
        g = w[w["_complete"]].groupby(["arm", "geometry"]).agg(**agg).reset_index()
        g = _with_incomplete_count(g, w, ["arm", "geometry"])
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

    Like :func:`decomposition_table`, aggregated over complete rows only, with
    ``n_incomplete`` counting the dropped ones and ``cost_residual`` pinning the
    telescoping.  The oracle links are the ones that go missing here - a step
    where ``pooled_supervised`` or ``pooled_sim_oracle`` emitted no row pivots to
    NaN - and they feed three of the four terms, so averaging around them shrank
    exactly the terms this study reads.
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
    wide["_complete"] = _complete_chain_rows(wide, [*COST_CHAIN, "oracle_cost"])

    wide["cost_prior_loss"] = wide["pooled_cross"] - wide["pooled_priorfree"]
    wide["cost_identification"] = wide["pooled_priorfree"] - wide["pooled_supervised"]
    wide["cost_misspecification"] = wide["pooled_supervised"] - wide["pooled_sim_oracle"]
    wide["cost_transfer"] = wide["pooled_sim_oracle"] - wide["oracle_cost"]
    wide["cost_total"] = wide["pooled_cross"] - wide["oracle_cost"]
    terms = ["cost_prior_loss", "cost_identification", "cost_misspecification", "cost_transfer"]
    wide["cost_residual"] = wide["cost_total"] - sum(wide[c] for c in terms)

    cols = [*terms, "cost_total", "cost_residual"]
    out = []
    for wname, (lo, hi) in WINDOWS.items():
        w = wide[(wide["n_votes"] >= lo) & (wide["n_votes"] <= hi)]
        complete = w[w["_complete"]]
        g = complete.groupby("arm")[cols].mean().reset_index()
        g["n"] = complete.groupby("arm")[cols[0]].size().to_numpy()
        g = _with_incomplete_count(g, w, ["arm"])
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
    aggs = {
        "evt_loglik_gain": ("evt_loglik_gain", "mean"),
        "frac_evt_better": ("evt_loglik_gain", lambda s: float(np.nanmean(np.asarray(s, dtype=float) > 0))),
        "evt_fit_rate": ("evt_ok", "mean"),
        "gmm_fit_rate": ("gmm_ok", "mean"),
        "n": ("evt_ok", "size"),
    }
    # How often the Gumbel landed on the low mode.  #2836 assumed always and
    # discarded the rest; #2846 is the question of what that cost.  Absent when
    # re-analyzing a frame emitted before #2846, which is a supported thing to do
    # (this study's whole point is comparing against those numbers).
    if "evt_gumbel_is_low" in diag:
        aggs["gumbel_is_low_rate"] = ("evt_gumbel_is_low", "mean")
    g = diag.groupby(["arm", "geometry"]).agg(**aggs).reset_index()
    g.to_csv(agg_dir / "cut_evt_evidence.csv", index=False)
    return g


def fallback_reasons(df: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Why each rule declined to fire and what it cut instead, per arm per variant.

    Two columns, answering two different questions about the same event
    (issues #2846, #2900):

    ``cut_fail_reason`` is *which guard sent it there* - the whole difference
    between "the fit was sound but oriented the other way" (repairable, and what
    ``gumbel_any_*`` repairs) and "the two components collapsed onto each other"
    (a statement about that step's score distribution, which no solver can fix).
    Only the EVT rules have a reason vocabulary; the Gaussian ones report ``""``.

    ``cut_fallback_kind`` is *what was substituted*, which is not the same
    question and does not have the same answer in both families.  For the
    ``_SAFE_GMM_VARIANTS`` arms it is ``midpoint``: that family compares tilts
    against each other on one fit, so a rule with no root gets a neutral,
    rule-independent stand-in.  For the label-anchored arms it is the production
    rule's own branch - ``continued`` (the cut carried past the component mean
    at the rule's first-order slope, still moving with the cost tilt) or
    ``degenerate_midpoint`` (a fit too degenerate to express a boundary at all).
    The flag fires on the same fits in both families, so ``fallback_rate``
    aggregates and ``cut_fallback == 0/1`` filters stay comparable across them;
    the substituted *value* does not, which is what this column exposes.  **A
    contrast that reads a ``*_rate`` arm as "what the app would have done" must
    exclude the ``midpoint`` rows**, since on those steps the arm is scoring a
    stand-in the app never cuts at.

    Emitted **per window**, plus an ``all_steps`` row set.  Every other table
    here is windowed, so a bare all-steps count invited exactly the mistake
    #2846's report had to warn about in prose: reading these counts next to a
    ramp-window fallback *rate* and treating them as the same population.
    """
    has_reason = "cut_fail_reason" in df
    has_kind = "cut_fallback_kind" in df
    if not (has_reason or has_kind):
        return pd.DataFrame()
    df = df.copy()
    # A frame emitted before either column existed is still analyzable - this
    # study's whole point is comparing against those numbers - so fill the
    # missing side rather than dropping the table.
    for col in ("cut_fail_reason", "cut_fallback_kind"):
        df[col] = df[col].fillna("").astype(str) if col in df else ""
    fell = df[df["cut_fallback"] == 1]
    if fell.empty:
        return pd.DataFrame()
    out = []
    for wname, sub in (
        ("all_steps", fell),
        *((wname, _window(fell, lo, hi)) for wname, (lo, hi) in WINDOWS.items()),
    ):
        if sub.empty:
            continue
        g = (
            sub.groupby(["arm", "gmm_variant", "cut_fallback_kind", "cut_fail_reason"])
            .size()
            .reset_index(name="n_steps")
            .sort_values(["arm", "gmm_variant", "n_steps"], ascending=[True, True, False])
        )
        # Share within each (arm, variant), so a reason is readable without joining
        # back to that variant's own step count.
        g["share_of_fallbacks"] = g["n_steps"] / g.groupby(["arm", "gmm_variant"])["n_steps"].transform("sum")
        g.insert(0, "window", wname)
        out.append(g)
    tbl = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    tbl.to_csv(agg_dir / "cut_fallback_reasons.csv", index=False)
    return tbl


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


def tail_alpha_curve(base_contrasts: pd.DataFrame, agg_dir: Path) -> dict:
    """#2881's sweep: the cost of "cut the Bad tail at alpha", as a function of alpha.

    :func:`tail_alpha_stability` says the *oracle* cut sits at a stable survival
    level of the fitted Gumbel low component (median 0.158, IQR ratio 2.38 over
    511 cells).  That is a statement about where the optimum *is*, not about what
    it costs to aim there, and the two come apart if the cost curve is steep:
    a constant calibrated on a median is only transferable if being off by a
    factor of ~1.5 in alpha is cheap.  So the claim being tested here is
    **flatness**, and it needs the curve, not the argmin.

    Read against the run's own base row (production), production arm, ramp
    window — the only baseline that means "beats what we ship" (#2846).

    ``flat_alphas`` are the levels whose cost is within one standard error of the
    best level's; ``flat_alpha_ratio`` is how wide a band in alpha that spans.
    The pre-registered bar is a factor of 2: a constant that can be wrong by 2x
    and still land inside the noise will transfer to another dataset, and one
    that cannot is a number fitted to this run.
    """
    out: dict = {"n_levels": 0, "preregistered_alpha": TAIL_ALPHA_PREREGISTERED}
    if base_contrasts.empty:
        return out
    prod = _ramp_prod(base_contrasts)
    rows = []
    for _i, r in prod.iterrows():
        alpha = tail_alpha_of(r["variant"])
        if alpha is None:
            continue
        rows.append(
            {
                "alpha": alpha,
                "variant": str(r["variant"]),
                "ship_eligible": str(r["variant"]) not in SWEEP_ONLY,
                "mean_d_cost": float(r["mean_d_cost"]),
                "sem_d_cost": float(r["sem_d_cost"]),
                "p_d_cost": None if r["p_d_cost"] is None else float(r["p_d_cost"]),
                "frac_cells_improved": float(r["frac_cells_improved"]),
                "n_cells": int(r["n_cells"]),
                "base_provenance": str(r["base_provenance"]),
            }
        )
    if not rows:
        return out
    curve = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    curve.to_csv(agg_dir / "cut_tail_alpha_curve.csv", index=False)

    best = curve.loc[curve["mean_d_cost"].idxmin()]
    # One SEM of the *best* level: the band inside which another level is not
    # distinguishable from it.  A NaN SEM (one cell) leaves the band empty rather
    # than swallowing the whole curve.
    band = float(best["sem_d_cost"])
    flat = curve[curve["mean_d_cost"] <= float(best["mean_d_cost"]) + band] if np.isfinite(band) else curve.iloc[[]]
    prereg = curve[np.isclose(curve["alpha"], TAIL_ALPHA_PREREGISTERED)]

    out.update(
        n_levels=int(len(curve)),
        curve=curve.to_dict("records"),
        best_alpha=float(best["alpha"]),
        best_mean_d_cost=float(best["mean_d_cost"]),
        flat_alphas=[float(a) for a in flat["alpha"]],
        flat_alpha_ratio=(float(flat["alpha"].max() / flat["alpha"].min()) if len(flat) else float("nan")),
        preregistered=None if prereg.empty else prereg.iloc[0].to_dict(),
    )
    # The transferability claim, pre-registered at 2x.
    out["curve_is_flat"] = bool(np.isfinite(out["flat_alpha_ratio"]) and out["flat_alpha_ratio"] >= 2.0)
    # Is #2846's median still where this run's optimum is?  Membership in the
    # flat band, not equality with the argmin: "0.158 is indistinguishable from
    # the best level here" is the transferability claim, and demanding it be the
    # exact argmin would fail on noise alone.  If it falls *outside* the band that
    # is a real finding - the constant does not carry across runs - which is the
    # same conclusion as a steep curve, by another route.
    out["preregistered_in_flat_band"] = bool(
        not prereg.empty and float(prereg.iloc[0]["alpha"]) in set(out["flat_alphas"])
    )
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


def _wins(row: pd.Series | None) -> bool:
    """Significantly cheaper, on the pre-registered two-sided 5 % bar."""
    if row is None:
        return False
    p = row["p_d_cost"]
    return bool(p is not None and np.isfinite(float(p)) and float(p) < 0.05 and float(row["mean_d_cost"]) < 0)


def _ramp_prod(tbl: pd.DataFrame, variants: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The production arm's ramp window, optionally restricted to *variants*."""
    if tbl.empty:
        return tbl
    sel = (tbl["window"] == "ramp_6_20") & (tbl["arm"].str.contains(PRODUCTION_ARM_SUBSTR))
    if variants is not None:
        sel &= tbl["variant"].isin(variants)
    return tbl[sel]


def decisions(
    contrasts: pd.DataFrame,
    base_contrasts: pd.DataFrame,
    gaps: pd.DataFrame,
    costs: pd.DataFrame,
    alpha: dict,
    tail_curve: dict,
) -> dict:
    """The issue's pre-registered decision rules, evaluated.

    Ship the rule that is closest to the oracle cut **and** wins on cost on the
    production arm's ramp window, provided it does not regress the single-vector
    arm.  Anything else is a negative result with a named cause.

    "Wins on cost" is judged against the **run's own base row**, not against
    ``pooled_mid``: #2846 shipped a new production threshold mid-study and the
    midpoint contrast quietly became a comparison with a rule nobody runs.
    ``beats_midpoint`` is still reported, as the historical #2836 contrast, but
    it no longer gates the ship decision.

    Every candidate here is drawn from ``SHIP_ELIGIBLE``, not ``SHIPPABLE``: the
    swept tail levels are measured but cannot win the gate, because a sweep over
    a free parameter would otherwise get one shot per level at a 5 % bar.  See
    ``SWEEP_ONLY``.
    """
    out: dict = {}
    if contrasts.empty:
        return {"error": "no contrasts"}

    prod = _ramp_prod(contrasts, SHIP_ELIGIBLE)
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
    out["beats_midpoint"] = _wins(best)

    # --- Against what production actually did -------------------------------
    prod_base = _ramp_prod(base_contrasts, SHIP_ELIGIBLE)
    cand = None
    out["best_vs_production"] = None
    out["beats_production"] = False
    if not prod_base.empty:
        cand = prod_base.sort_values("mean_d_cost").iloc[0]
        out["best_vs_production"] = {
            "variant": str(cand["variant"]),
            "mean_d_cost": float(cand["mean_d_cost"]),
            "p": None if cand["p_d_cost"] is None else float(cand["p_d_cost"]),
            "frac_cells_improved": float(cand["frac_cells_improved"]),
            "base_provenance": str(cand["base_provenance"]),
        }
        out["beats_production"] = _wins(cand)

    # Is there anything left on this axis at all?  `pooled_sim_oracle` is the
    # empirical rate-loss minimiser over the sim scores *read with true labels*,
    # with no parametric form at all, so it bounds every rule that picks a
    # threshold from that sim set - Gaussian crossing, Gumbel crossing, and the
    # one-constant tail quantile of #2881 alike.  If *it* cannot beat production,
    # no unsupervised member of that set will, and the next study belongs on the
    # fit rather than the cut (#2846's closing finding, mechanised here so a
    # later run is told rather than having to rediscover it by hand).
    oracle_row = _ramp_prod(base_contrasts)
    oracle_row = oracle_row[oracle_row["variant"] == "pooled_sim_oracle"]
    out["family_headroom"] = None
    out["family_headroom_exhausted"] = None
    if not oracle_row.empty:
        o = oracle_row.iloc[0]
        out["family_headroom"] = {
            "mean_d_cost": float(o["mean_d_cost"]),
            "p": None if o["p_d_cost"] is None else float(o["p_d_cost"]),
        }
        out["family_headroom_exhausted"] = not _wins(o)

    # Closest to the oracle cut, among shippable rules, same arm and window.
    gp = gaps[
        (gaps["window"] == "ramp_6_20")
        & (gaps["arm"].str.contains(PRODUCTION_ARM_SUBSTR))
        & (gaps["gmm_variant"].isin(SHIP_ELIGIBLE))
    ]
    # Several rules in this family are *aliases* rather than competitors - at
    # inclusion 0 the cost weights are (1, 1), so `rate` reduces to `priorfree`
    # exactly, and the two Gumbel tilts collapse the same way.  Picking a single
    # argmin would tie-break arbitrarily between two names for one rule and then
    # fail the ship test by comparing them as if they disagreed.  Report the
    # whole tied set instead.
    out["closest_to_oracle"] = None
    out["closest_to_oracle_tied"] = []
    if not gp.empty:
        best_gap = float(gp["mean_abs_gap"].min())
        tied = sorted(gp[gp["mean_abs_gap"] <= best_gap + 1e-9]["gmm_variant"].astype(str))
        out["closest_to_oracle_tied"] = tied
        out["closest_to_oracle"] = tied[0]

    # The ship candidate is the one that beats *production*; the midpoint winner
    # is only the candidate when there is no base row to pair against.
    ship_candidate = str((cand if cand is not None else best)["variant"])
    out["ship_candidate"] = ship_candidate

    # Does the candidate regress the single-vector arm the cosine/text sort uses?
    # Measured against the same baseline as the ship test, for the same reason.
    ctrl_tbl = base_contrasts if cand is not None else contrasts
    ctrl = ctrl_tbl[
        (ctrl_tbl["window"] == "ramp_6_20")
        & (ctrl_tbl["arm"].str.contains(CONTROL_ARM_SUBSTR))
        & (ctrl_tbl["variant"] == ship_candidate)
    ]
    out["control_arm_delta"] = None if ctrl.empty else float(ctrl.iloc[0]["mean_d_cost"])
    out["regresses_control"] = bool(
        not ctrl.empty
        and float(ctrl.iloc[0]["mean_d_cost"]) > 0
        and ctrl.iloc[0]["p_d_cost"] is not None
        and float(ctrl.iloc[0]["p_d_cost"]) < 0.05
    )
    out["ship"] = bool(
        (out["beats_production"] if cand is not None else out["beats_midpoint"])
        and ship_candidate in out["closest_to_oracle_tied"]
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
                # The terms are only comparable to each other, and to `total`, if
                # every one of them averaged the same steps.  Both numbers say so
                # explicitly rather than leaving a reader to re-add the column.
                row = c.iloc[0]
                out["error_terms_residual"] = float(row.get("cost_residual", float("nan")))
                out["error_terms_n_incomplete"] = int(row.get("n_incomplete", 0))
    # The leading hypothesis is confirmed only if the prior/loss term dominates.
    out["leading_hypothesis_confirmed"] = out["dominant_error_term"] == "prior_loss"
    # One key per tail model, named after it.  The old single `tail_alpha_stable`
    # was keyed off the Gaussian row alone, so a reader who found it `false` next
    # to a passing EVT row concluded the EVT rule was unstable - which is the
    # opposite of what that table says (#2846 had to spend a paragraph on it).
    out["tail_alpha_stable_gauss"] = bool(alpha.get("oracle_lo_sf_gauss", {}).get("stable", False))
    out["tail_alpha_stable_evt"] = bool(alpha.get("oracle_lo_sf_evt", {}).get("stable", False))

    # #2881: the tail rule *as a rule*, plus what its sweep says about whether the
    # one constant transfers.  Named separately from the ship gate above because
    # only one level is eligible for it - a flat curve is evidence for the rule's
    # premise, not a licence to ship whichever level came out lowest.
    out["sweep_only_variants"] = list(SWEEP_ONLY)
    out["tail_alpha_curve"] = {
        k: tail_curve.get(k) for k in ("n_levels", "best_alpha", "best_mean_d_cost", "flat_alphas", "flat_alpha_ratio")
    }
    out["tail_curve_is_flat"] = tail_curve.get("curve_is_flat")
    out["tail_preregistered_alpha_in_flat_band"] = tail_curve.get("preregistered_in_flat_band")
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

    # The swept tail levels are left off these line plots: seven near-identical
    # curves would cost the legend its legibility and say nothing the alpha curve
    # in `cut_tail_alpha_curve.csv` does not say better.  The pre-registered level
    # is a ship candidate and stays.
    v = df[df["gmm_variant"].isin((*SHIP_ELIGIBLE, *ORACLE_VARIANTS))]
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
        ("Rule contrasts vs the run's own production row", "base_contrasts"),
        ("Rule contrasts vs the midpoint (the #2836 baseline)", "contrasts"),
        ("Distance to the oracle cut", "gaps"),
        ("Decomposition (threshold units)", "decomposition"),
        ("Decomposition (excess-cost units)", "cost_decomposition"),
        ("Extreme-value evidence", "evt"),
        ("Why each rule fell back, and to what (issues #2846, #2900)", "fallback_reasons"),
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
    curve = summary.get("tail_alpha_curve") or {}
    if curve.get("curve"):
        lines.append("## The tail rule as a rule: cost vs alpha (issue #2881)")
        lines.append("")
        lines.append("Production arm, ramp 6-20, paired against the run's own base row.")
        lines.append("Only the pre-registered level is a ship candidate; the rest show the curve's shape.")
        lines.append("")
        lines.append(_md(pd.DataFrame(curve["curve"])))
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({k: v for k, v in curve.items() if k != "curve"}, indent=2, default=float))
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

    unknown = unclassified_variants(df)
    sanity = production_blend_sanity(df)
    window = window_table(df, agg_dir)
    contrasts = rule_contrasts(df, agg_dir)
    base_contrasts = base_row_contrasts(df, agg_dir)
    gaps = oracle_distance(df, agg_dir)
    decomp = decomposition_table(diag, agg_dir) if not diag.empty else pd.DataFrame()
    costs = cost_decomposition(df, agg_dir)
    offsets = offset_predictions(diag, df, agg_dir) if not diag.empty else {}
    evt = evt_evidence(diag, agg_dir) if not diag.empty else pd.DataFrame()
    why = fallback_reasons(df, agg_dir)
    alpha = tail_alpha_stability(diag, agg_dir) if not diag.empty else {}
    tail_curve = tail_alpha_curve(base_contrasts, agg_dir)
    est_var = estimator_variance(diag, agg_dir) if not diag.empty else pd.DataFrame()
    dec = decisions(contrasts, base_contrasts, gaps, costs, alpha, tail_curve)
    # Carried in the decisions block, not only in the log: this is the one place
    # a reader checks before believing a verdict, and the verdict is exactly what
    # an unclassified rule is missing from.
    dec["unclassified_variants"] = unknown
    figs = make_figures(df, diag, fig_dir) if not diag.empty else []

    summary = {
        "n_variant_rows": int((df["gmm_variant"] != "").sum()),
        "n_diag_rows": int(len(diag)),
        "n_cells": int(df[["dataset", "embedder", "category", "seed"]].drop_duplicates().shape[0]),
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "unclassified_variants": unknown,
        "sanity": sanity,
        "decisions": dec,
        "offsets": offsets,
        "tail_alpha": alpha,
        "tail_alpha_curve": tail_curve,
        "figures": figs,
    }
    (common.RESULTS / "summary_cut.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(
        summary,
        {
            "window": window,
            "base_contrasts": base_contrasts,
            "contrasts": contrasts,
            "gaps": gaps,
            "decomposition": decomp,
            "cost_decomposition": costs,
            "evt": evt,
            "fallback_reasons": why,
            "estimator_variance": est_var,
        },
        common.RESULTS / "REPORT_CUT.md",
    )
    common.log("analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

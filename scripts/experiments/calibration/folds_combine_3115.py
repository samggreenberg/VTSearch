"""The combine-rule contrast (#3115): pooled vs averaged cross-calibration.

Two functions in the repo disagree about the same empirical fact.
``threshold_from_fold_orderings`` - the path the live app calls - **pools** every
fold's held-out scores into one bag and takes a single conformal quantile, on the
stated ground that "all folds' scores live on the same sigmoid scale".
``FoldAnchoredCut._combined_fold_quantile`` takes one cut per fold and averages
them in *quantile* space, on the stated ground that a cut must be read "in the
scale it was measured on" - i.e. that fold scores are **not** comparable.  One of
those premises is wrong and nobody has measured which.

This module reads the #2897/#3116 fold-count frame - which already carries every
arm, since all of them re-cut the *same* already-trained fold prefix - and turns
it into that measurement.  It is a different axis from the rest of
``analyze_folds_2897.py``: there the contrast is across **K** at a fixed rule,
here it is across **rules** at a fixed K, paired inside the step.

**The contrast is factored, not lumped.**  ``pooled -> qmean`` is what the issue
literally asks for, but it moves two things at once, so it is reported as its
legs and only then as a total:

======================  ======================================================
``tmean - pooled``      pooling vs **averaging**, both in score space.
``qmean - tmean``       score space vs **quantile** space, combine held fixed:
                        exactly the comparability premise the two docstrings
                        disagree on.
``*median - *mean``     **contamination**: a single-class fold pours its scores
                        straight into a pooled quantile, contributes 1/K to a
                        mean, and ~nothing to a median.
``anchored_qmedian``    the same contamination question on the **shipped** rule
``  - anchored``        rather than on the retired blend.
======================  ======================================================

Two structural facts shape every table here, and both are pinned by tests rather
than hoped for:

* **Below K=3 the median legs are identically zero.**  The mean and median of at
  most two numbers coincide, which is precisely why this question has never been
  askable at production's ``calibrate_count=2``.  Rows at K<3 are kept as a
  *control* (they must read exactly 0.0) and excluded from every headline.
* **At K=1 the score-space combine reproduces the pooled cut bit for bit.**
  Averaging one number is the identity.  That gives the study an exact,
  run-level acceptance check that is independent of anything the run measures:
  see :func:`control_checks`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

#: The pooled control.  ``xcal`` calls ``threshold_from_fold_orderings``
#: verbatim, so it *is* today's behaviour rather than a re-implementation of it.
POOLED = "xcal"

#: ``(name, arm, reference, collapses)`` - every contrast this module reports, in
#: the order it reads.  Each is paired inside the step, so the two arms share the
#: votes, the final model, the held-out test scores and the fold prefix; the only
#: thing that differs is the rule that turned those folds into a cut.
#:
#: ``collapses`` marks the contrasts whose two arms are **the same rule** below
#: three folds, because the mean and the median of at most two numbers coincide.
#: Those rows must read identically zero and are the control; the rest differ at
#: every K and are not.  The distinction is per-contrast rather than per-arm: a
#: median arm can appear on either side of a contrast that is *not* collapsed
#: (``tmedian`` vs the pooled control is the combine leg at K<3, not a no-op), so
#: inferring it from the arm names quietly mislabels half the table.
CONTRASTS: tuple[tuple[str, str, str, bool], ...] = (
    ("combine", "tmean", POOLED, False),
    ("combine_robust", "tmedian", POOLED, False),
    ("space", "qmean", "tmean", False),
    ("space_robust", "qmedian", "tmedian", False),
    ("contamination_t", "tmedian", "tmean", True),
    ("contamination_q", "qmedian", "qmean", True),
    ("total", "qmean", POOLED, False),
    ("shipped_contamination", "anchored_qmedian", "anchored", True),
)

#: Fold count below which a mean and a median of the same cuts are one rule.
COLLAPSE_BELOW_K = 3

#: Metrics carried through every contrast.  ``regret_honest`` rides along because
#: #3116 established the naive test oracle is optimistic; a *level* should be read
#: off the pair as a bracket.  Paired **differences** are far less sensitive to
#: which reference is used - both arms pay the same optimism on the same test
#: sample - which is why the headline stays on ``regret``.
METRICS = ("regret", "cost", "fpr", "fnr", "regret_honest")

#: Aggregation unit for the significance test: a trajectory, not a step.
#: Consecutive steps share nearly all their votes, so testing over steps counts
#: one trajectory's luck hundreds of times.
CELL_KEYS = ["env", "category", "seed", "window"]
STEP_KEYS = ["env", "category", "seed", "t"]


def _paired(v: pd.DataFrame, arm: str, ref: str) -> pd.DataFrame:
    """Step-paired join of *arm* against *ref*, keyed by (step, K).

    An inner join, so a step where either arm is missing - a fold prefix with no
    haystacks, a run without safe thresholds - drops out of *both* sides rather
    than being compared against a different set of steps.
    """
    keys = [*STEP_KEYS, "k"]
    cols = ["threshold", "n_folds_used", "window", "window_hi", "voting", *METRICS]
    a = v[v["arm"] == arm].set_index(keys)
    b = v[v["arm"] == ref].set_index(keys)
    a, b = a[~a.index.duplicated()], b[~b.index.duplicated()]
    have = [c for c in cols if c in a.columns and c in b.columns]
    j = pd.concat([a[have].add_suffix("_a"), b[have].add_suffix("_b")], axis=1, join="inner")
    if j.empty:
        return j
    j = j.reset_index().rename(columns={"window_a": "window", "window_hi_a": "window_hi", "voting_a": "voting"})
    for m in METRICS:
        if f"{m}_a" in j.columns:
            j[f"d_{m}"] = j[f"{m}_a"] - j[f"{m}_b"]
    j["moved"] = (j["threshold_a"] != j["threshold_b"]).astype(float)
    # A fold the combining arm could not use.  NaN on the pooled arm by design -
    # it never reads a per-fold cut, so it has no fold to drop - which is exactly
    # the asymmetry the contamination legs are about.
    j["dropped_a"] = (j["k"] - j["n_folds_used_a"]).clip(lower=0) if "n_folds_used_a" in j.columns else np.nan
    return j


def _summarise(j: pd.DataFrame, name: str, arm: str, ref: str, collapses: bool) -> list[dict]:
    """Cell-mean paired deltas per (voting, window, K), with a standard error.

    Quoted as ``mean +- SE`` over **cells** because that is the unit of evidence,
    and reported alongside ``resolvable`` - whether the difference exceeds twice
    its own standard error.  "Not resolvable here" is a finding; a fourth decimal
    that a +-0.03 SE cannot support is an invented one.
    """
    rows: list[dict] = []
    if j.empty:
        return rows
    deltas = [f"d_{m}" for m in METRICS if f"d_{m}" in j.columns]
    for (voting, window, k), g in j.groupby(["voting", "window", "k"], observed=True):
        cells = g.groupby(CELL_KEYS, observed=True)[[*deltas, "moved", "dropped_a"]].mean()
        d = cells["d_regret"].to_numpy()
        n = len(d)
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        p = float("nan")
        if n >= 6 and not np.allclose(d, 0):
            p = float(wilcoxon(d, zero_method="zsplit").pvalue)
        row = {
            "contrast": name,
            "arm": arm,
            "ref": ref,
            "voting": voting,
            "window": str(window),
            "window_hi": int(g["window_hi"].iloc[0]),
            "k": int(k),
            "n_cells": n,
            "n_steps": int(len(g)),
            "d_regret": float(d.mean()),
            "se_regret": se,
            "resolvable": bool(np.isfinite(se) and abs(d.mean()) > 2.0 * se),
            "p_wilcoxon": p,
            "win_rate": float((cells["d_regret"] < 0).mean()),
            # How often the two rules even disagree on the cut.  A contrast whose
            # arms produce the same threshold on 99% of steps is not "no effect",
            # it is "no exposure", and the two must not read alike.
            "moved_rate": float(cells["moved"].mean()),
            "mean_folds_dropped": float(cells["dropped_a"].mean()),
            "collapsed_by_construction": bool(collapses and k < COLLAPSE_BELOW_K),
        }
        for m in METRICS:
            if f"d_{m}" in cells.columns and m != "regret":
                row[f"d_{m}"] = float(cells[f"d_{m}"].mean())
        rows.append(row)
    return rows


def contrast_table(v: pd.DataFrame, agg) -> pd.DataFrame:
    """Every contrast in :data:`CONTRASTS`, per (voting, window, K)."""
    rows: list[dict] = []
    present = set(v["arm"])
    for name, arm, ref, collapses in CONTRASTS:
        if arm not in present or ref not in present:
            continue
        rows.extend(_summarise(_paired(v, arm, ref), name, arm, ref, collapses))
    t = pd.DataFrame(rows)
    if not t.empty:
        t = t.sort_values(["contrast", "voting", "window_hi", "k"])
    t.to_csv(agg / "combine_contrasts.csv", index=False)
    return t


def degenerate_table(v: pd.DataFrame, agg) -> pd.DataFrame:
    """How often a fold is too degenerate to contribute a cut, per (voting, K).

    The contamination hypothesis' own exposure term.  A single-class holdout is
    silently *inside* a pooled quantile and explicitly *outside* a mean, so the
    rate at which it happens bounds how much of any pooled-vs-averaged difference
    that channel could possibly explain - and a study reporting the difference
    without the rate cannot tell a real effect from an absent one.
    """
    if "n_folds_used" not in v.columns:
        return pd.DataFrame()
    w = v[v["arm"].isin(["tmean", "qmean"]) & v["n_folds_used"].notna()].copy()
    if w.empty:
        return pd.DataFrame()
    w["dropped"] = (w["k"] - w["n_folds_used"]).clip(lower=0)
    t = (
        w.groupby(["voting", "arm", "window", "k"], observed=True)
        .agg(
            any_dropped_rate=("dropped", lambda s: float((s > 0).mean())),
            mean_dropped=("dropped", "mean"),
            n_steps=("dropped", "size"),
        )
        .reset_index()
        .sort_values(["voting", "arm", "window", "k"])
    )
    t.to_csv(agg / "combine_degenerate.csv", index=False)
    return t


def control_checks(v: pd.DataFrame) -> dict:
    """Run-level acceptance checks that do not depend on what the run measures.

    Both are identities the arms satisfy by construction, so a failure means the
    harness is mis-wired (a mis-sliced fold prefix, haystacks out of step with
    their orderings) rather than that a rule performed badly - which is the whole
    point of having them.  A study whose only checks are its own headline columns
    cannot tell those two apart.

    * ``k1_score_space_is_pooled`` - averaging one number is the identity, so at
      K=1 ``tmean`` must reproduce ``xcal``'s threshold **exactly**, on every step.
    * ``median_collapses_below_k3`` - the mean and median of at most two numbers
      coincide, so every median contrast must be identically zero at K<3.
    """
    out: dict = {}
    k1 = _paired(v[v["k"] == 1], "tmean", POOLED)
    out["k1_n_steps"] = int(len(k1))
    out["k1_score_space_is_pooled"] = bool(len(k1)) and bool((k1["threshold_a"] == k1["threshold_b"]).all())
    if len(k1):
        out["k1_mismatches"] = int((k1["threshold_a"] != k1["threshold_b"]).sum())

    collapsed = []
    for arm, ref in (("tmedian", "tmean"), ("qmedian", "qmean"), ("anchored_qmedian", "anchored")):
        j = _paired(v[v["k"] < COLLAPSE_BELOW_K], arm, ref)
        if j.empty:
            continue
        collapsed.append(bool((j["threshold_a"] == j["threshold_b"]).all()))
    out["median_collapses_below_k3"] = bool(collapsed) and all(collapsed)
    return out


def verdicts(t: pd.DataFrame, deep_min: int, margin: float) -> dict:
    """Read the deep-regime, K>=3 rows into the answer #3115 asks for.

    Deliberately conservative about what counts as an answer.  A contrast is
    only called for a side when its paired mean exceeds **twice** its own
    standard error *and* the pre-registered margin; otherwise it is reported
    ``unresolved``, which on this question is a genuinely useful result - "the
    two docstrings' premises make no difference worth acting on to the threshold
    users get" would settle the disagreement as decisively as either winning.

    Both halves are load-bearing.  These runs pool hundreds of autocorrelated
    steps into each cell, so the standard error over cells gets small enough to
    resolve differences four orders of magnitude below the margin - real, and no
    reason whatsoever to change a shipped rule.  ``resolved`` and
    ``above_margin`` are both kept on every row so a reader can see which of the
    two a given contrast failed.
    """
    out: dict = {"deep_min": deep_min, "margin": margin, "by_voting": {}}
    if t.empty:
        return out
    deep = t[(t["window_hi"] >= deep_min) & (t["k"] >= COLLAPSE_BELOW_K)]
    for voting, g in deep.groupby("voting", observed=True):
        per: dict = {}
        for name, sub in g.groupby("contrast", observed=True):
            # Pool the K>=3 rows by cell count: every K re-cuts the same steps,
            # so an unweighted mean over K would let a sparsely-populated large K
            # count as much as the well-populated small ones.
            wts = sub["n_cells"].to_numpy(dtype=float)
            d = float(np.average(sub["d_regret"].to_numpy(), weights=wts)) if wts.sum() else float("nan")
            se = float(np.sqrt(np.average(np.square(sub["se_regret"].to_numpy()), weights=wts) / len(sub)))
            per[name] = {
                "d_regret": d,
                "se": se,
                "resolved": bool(np.isfinite(se) and abs(d) > 2 * se),
                "favours": ("arm" if d < 0 else "ref") if np.isfinite(se) and abs(d) > 2 * se else None,
                "above_margin": bool(abs(d) >= margin),
                "moved_rate": float(np.average(sub["moved_rate"].to_numpy(), weights=wts)) if wts.sum() else None,
                "n_cells": int(sub["n_cells"].sum()),
            }
        total = per.get("total", {})
        # The headline needs **both** halves: resolvable (the sample can see it)
        # and above the pre-registered margin (it is worth acting on).  Either
        # alone is a way to be wrong.  A huge sample resolves differences of
        # 1e-5, which is a real difference and not a reason to change the rule
        # users get; and a difference above the margin that the sample cannot
        # resolve is a coin flip wearing a decimal point.
        decided = bool(total.get("resolved") and total.get("above_margin"))
        out["by_voting"][voting] = {
            "contrasts": per,
            # The headline, in the issue's own terms.
            "pooling_is_wrong": bool(decided and total.get("favours") == "arm"),
            "averaging_is_wrong": bool(decided and total.get("favours") == "ref"),
            "unresolved": bool(total and not decided),
            # Distinguished from `unresolved` on purpose: "measured, and the
            # difference is too small to act on" and "never measured here"
            # are different facts, and three `false` flags in a row would read
            # as the first when it was the second.  The `q*` arms need a
            # haystack per fold, so a run without safe thresholds has no total.
            "total_absent": not total,
        }
    return out


def report_lines(t: pd.DataFrame, degen: pd.DataFrame, checks: dict, verd: dict, deep_min: int) -> list[str]:
    """The #3115 section of the run's REPORT.md."""

    def md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False, floatfmt=".4g")
        except Exception:  # noqa: BLE001 - tabulate not installed
            return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "## Combine rule: pooled vs averaged cross-calibration (#3115)",
        "",
        "`threshold_from_fold_orderings` pools every fold's held-out scores and takes",
        "one conformal quantile; `FoldAnchoredCut._combined_fold_quantile` averages",
        "per-fold cuts in quantile space *because* it holds that fold scores are not",
        "comparable.  Both premises cannot be right.  Every arm below re-cuts the same",
        "already-trained fold prefix, so each contrast is paired inside the step and",
        "differs only in the rule.",
        "",
        "The total (`qmean` vs pooled) moves two things at once, so read the legs:",
        "`tmean - pooled` is pooling vs averaging with the space held fixed, and",
        "`qmean - tmean` is score space vs quantile space with the combine held fixed.",
        "Median legs are identically zero below K=3 (mean and median of <=2 numbers",
        "coincide) - which is why production's `calibrate_count=2` could never have",
        "shown this - so the verdict reads K>=3 only.",
        "",
        "### Acceptance checks (identities, not results)",
        "",
        "```json",
        _json(checks),
        "```",
        "",
        "`k1_score_space_is_pooled` is the load-bearing one: averaging a single",
        "number is the identity, so at K=1 the score-space arm must reproduce the",
        "pooled threshold exactly.  It is independent of every column the study",
        "reports, which is what makes it a check rather than a restatement.",
        "",
        "### Verdicts (mechanical)",
        "",
        "```json",
        _json(verd),
        "```",
        "",
        f"### Paired contrasts (deep windows are >= {deep_min} votes; negative favours the arm)",
        "",
        "`d_regret +- se_regret` over cell means; `resolvable` is |d| > 2*se.  Where it",
        "is false the difference is not resolvable at this sample size, which is a",
        "result and not a gap.  `moved_rate` is how often the two rules even disagreed",
        "on the cut - a contrast with no exposure cannot have an effect.",
        "",
        md(t) if len(t) else "_No combine arms in this run._",
        "",
        "### Degenerate folds: the contamination channel's exposure",
        "",
        "A single-class holdout is silently *inside* a pooled quantile and explicitly",
        "*outside* a mean.  `any_dropped_rate` bounds how much of any pooled-vs-averaged",
        "difference that channel could explain; a near-zero rate means the median legs",
        "are testing robustness against a hazard this grid never presents.",
        "",
        md(degen) if len(degen) else "_No fold-usage counts in this run (pre-#3115 cells)._",
        "",
    ]
    return lines


def _json(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=float)

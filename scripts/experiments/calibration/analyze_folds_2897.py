"""Stage 2 (fold-count study, #2897): what does more cross-calibration cost, and buy?

Consumes ``results/cells/task_*.csv`` from a run launched with
``CALIB_FOLD_COUNTS`` set (see ``launch_folds_2897.sh``), where every trainable
step emits one ``folds_k{K}_xcal`` row per fold count - and, under safe
thresholds, a ``folds_k{K}_blend`` row - carrying that K's regret and its
measured ``fold_seconds``.

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
* **Mechanism** - whether K's benefit tracks the pooled calibration-set size and
  lands on the *rule inefficiency* term (sampling noise in the cut) rather than
  the calibration->test *shift* term, which K cannot touch.

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

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import main_frame_files  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

#: ``folds_k{K}_{xcal,blend}`` - the arms this analyzer owns.
FOLD_RE = re.compile(r"^folds_k(?P<k>\d+)_(?P<arm>xcal|blend)$")

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
    frames, empty, unreadable = [], 0, 0
    for p in files:
        if p.stat().st_size == 0:
            empty += 1
            continue
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:  # noqa: BLE001 - a truncated cell must be counted, not crash the run
            common.log(f"  UNREADABLE {p.name}: {exc}")
            unreadable += 1
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["env"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    # Region voting is a property of the dataset, and it selects which
    # calibrator ran (bag-aware vs row-wise) - the axis #2897 asks about.
    df["voting"] = np.where(df["dataset"].eq("visual_genome_m"), "region", "binary")
    common.log(
        f"loaded {len(df):,} rows from {len(frames)}/{len(files)} cells "
        f"({empty} zero-byte, {unreadable} unreadable, skipped)"
    )
    return df


def fold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Just the fold-count arms, with K and arm parsed out and windows assigned."""
    m = df["gmm_variant"].str.extract(FOLD_RE)
    v = df[m["k"].notna()].copy()
    v["k"] = pd.to_numeric(m.loc[v.index, "k"]).astype(int)
    v["arm"] = m.loc[v.index, "arm"]
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
            j = pd.concat(
                [
                    a[["regret", "fold_seconds", "window", "rule_inefficiency", "calibration_shift"]].add_suffix("_a"),
                    base[["regret", "fold_seconds", "rule_inefficiency", "calibration_shift"]].add_suffix("_b"),
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
            j = j.rename(columns={"window_a": "window"})
            for window, w in j.groupby("window", observed=True):
                cells = w.groupby(CELL_KEYS, observed=True)[["d_regret", "d_seconds", "d_rule", "d_shift"]].mean()
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


def verdicts(paired: pd.DataFrame, knee: pd.DataFrame, levels: pd.DataFrame) -> dict:
    """Mechanically apply the plan's decision rules.

    H1 (benefit): does any K beat K=2 by more than :data:`MARGIN` in the deep
    regime, with a paired Wilcoxon under 0.05?
    H2 (cost): is the winning K's calibration wall clock within
    :data:`COST_CEILING_X` of production's?
    H3 (recommendation): the smallest K satisfying both, per voting mode -
    ``2`` (keep production) when H1 fails.
    H4 (mechanism): does K's benefit land on the rule-inefficiency term rather
    than the calibration->test shift, as the variance story predicts?
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
        out["by_voting"][voting] = {
            "h1_any_k_beats_baseline": bool(len(beats)),
            "h1_ks_beating_baseline": [int(k) for k in beats.index],
            "h2_ks_also_affordable": [int(k) for k in affordable.index],
            "h3_recommended_k": recommended,
            "h3_kept_production": recommended == BASELINE_K,
            "best_k_ignoring_cost": int(best),
            "best_d_regret": float(agg_k["d_regret"].min()) if len(agg_k) else 0.0,
            "cost_x_at_recommended": float(sec_x.get(recommended, float("nan"))) if sec_x is not None else None,
            "h4_benefit_is_rule_inefficiency": bool(
                len(agg_k) and agg_k.loc[best, "d_rule"] <= agg_k.loc[best, "d_shift"]
            ),
            "d_rule_at_best": float(agg_k.loc[best, "d_rule"]) if len(agg_k) else 0.0,
            "d_shift_at_best": float(agg_k.loc[best, "d_shift"]) if len(agg_k) else 0.0,
        }
    out["knee_by_window"] = knee[knee["arm"] == arm].to_dict(orient="records")
    return out


def _window_hi(label) -> int:
    try:
        return int(str(label).removeprefix("le_"))
    except ValueError:
        return 0


def shipped_arm(v: pd.DataFrame) -> str:
    """The arm the verdict reads: the blended threshold users get, when present."""
    return "blend" if "blend" in set(v["arm"]) else "xcal"


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


def write_report(results: Path, levels, paired, knee, verd, ab) -> None:
    lines = [
        "# Calibration fold-count study (#2897)",
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
        "## A/B check: does a run that lives at K reproduce the screen?",
        "",
        _md(ab) if len(ab) else "_No A/B run dirs passed; screen only._",
        "",
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

    levels = level_table(v, agg)
    paired = paired_vs_baseline(v, agg)
    knee = knee_table(paired, levels, agg)
    verd = verdicts(paired, knee, levels)
    ab = ab_check(v, [Path(d) / "results" for d in argv], agg) if argv else pd.DataFrame()
    verd["ab_check"] = ab.to_dict(orient="records") if len(ab) else None

    (results / "summary.json").write_text(json.dumps(verd, indent=2))
    write_report(results, levels, paired, knee, verd, ab)
    common.log(f"wrote {results / 'summary.json'} and {results / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

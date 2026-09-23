#!/usr/bin/env python
"""How many paired cells does the trajectory A/B need to resolve a given Δcost? (#3840)

    python resolution_3840.py --steps <frames/steps.csv.gz> --out <analysis dir>

Input is the compact per-step frame ``frames_3840.py`` writes: one row per
(grid, cell, step), read through ``analyze_ab.load_base_rows`` so a cell here is
the cell every A/B report paired.

**The statistic is the one the reports quote.**  Per cell, the mean of cost over
the app-visible steps (``app_trained == 1``, at least two votes -
``analyze_ab``'s ``app_visible``/``all_steps``); per pair of grids, the paired
Δ = on − off over cells; SE = sd(Δ) / √n.  Before anything else this script
reproduces #3825's published ``ll1e-8`` line from that definition, and refuses
to go on if it cannot.

What it writes (all to ``--out``):

``pairs.csv``        every pair: n, mean Δ, sd, SE, how many cells never diverged
``curve.csv``        the subsampled SE at k = 8..114, with the SE estimator's own spread and coverage
``design.csv``       cells needed to resolve δ, for the σ each kind of pair has
``env.csv``          σ, mean and cell cost per environment, per pair
``alloc.csv``        what reweighting environments buys, per cell and per cell-hour
``components.csv``   category vs seed variance components (does a seed buy what a category buys?)
``steps.csv``        σ and cost when a trajectory is cut at T steps
``validation.json``  the pre-registered prediction against the 399-cell grid, drift and determinism
``TABLES.md``        all of the above as markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELL = ["arm", "category", "seed"]

#: Pairs built from the 2026-09-13 grids (seeds 0-1).  ``kind`` groups them for
#: the design table: an arm pair whose trajectories diverge, or a nominal A/A.
EXISTING_PAIRS = (
    ("native-sklearn", "native", "sklearn", "arm", "#3585: the native fit vs sklearn"),
    ("ll1e-3-baseline", "ll1e-3", "baseline", "arm", "#3825: 1e-3, a real +0.026 regression"),
    ("ll1e-6-baseline", "ll1e-6", "baseline", "arm", "#3825: 1e-6"),
    ("ll1e-8-baseline", "ll1e-8", "baseline", "arm", "#3825: 1e-8, the shipped rule"),
    ("ll1e-6-ll1e-8", "ll1e-6", "ll1e-8", "arm", "#3825: the two arms it could not separate"),
    ("baseline-native", "baseline", "native", "drift", "nominally the same fit, run by two studies"),
)

#: The pre-registered prediction (``docs/experiments/2026-09-22-ab-resolution-3840/PLAN.md``).
PREDICTED_SE = 0.0020
PREDICTED_SE_90 = (0.0015, 0.0024)
PREDICTED_SD_90 = (0.030, 0.048)
VALIDATION_SEEDS = tuple(range(2, 9))

KS = (8, 16, 32, 64, 114)
DELTAS = (0.002, 0.003, 0.004, 0.005, 0.01, 0.02)
Z_RESOLVE = 2.0  # "resolved" in every report here: |mean| > 2 SE
Z_POWER80 = 2.0 + 0.8416  # an 80% chance of clearing 2 SE when the effect is δ
STEP_CUTS = (20, 40, 60, 80, 100)


def cell_means(steps: pd.DataFrame, t_max: int | None = None) -> pd.DataFrame:
    """Per-(grid, cell) mean over the app-visible steps; wide on grid."""
    v = steps[(steps["app_trained"] == 1) & (steps["n_votes"] >= 2)]
    if t_max is not None:
        v = v[v["t"] <= t_max]
    return v.groupby(["grid", *CELL])["cost"].mean().unstack(0)


def paired(cm: pd.DataFrame, on: str, off: str, seeds=None) -> pd.Series:
    """Δ = on − off over the cells both grids hold (optionally only *seeds*)."""
    d = (cm[on] - cm[off]).dropna()
    if seeds is not None:
        d = d[d.index.get_level_values("seed").isin(seeds)]
    return d


def summarise(d: pd.Series) -> dict:
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": float(d.mean()),
        "sd": sd,
        "se": sd / np.sqrt(n) if n > 1 else float("nan"),
        "frac_zero": float(np.mean(np.abs(d.to_numpy()) < 1e-12)),
    }


def clustered_se(d: pd.Series) -> float:
    """SE of the mean treating each (environment, category) as one cluster.

    The naive SE treats every cell as independent.  Seeds of one category share
    whatever makes that category easy or hard for an arm, so when the question
    is "does the arm help on categories like these" rather than "on these
    categories", the cluster is the unit.  With S seeds per category this is the
    larger of the two whenever the category component is non-zero.
    """
    f = d.rename("d").reset_index()
    f["cl"] = f["arm"] + "|" + f["category"]
    m = f["d"].mean()
    tot = f.groupby("cl")["d"].apply(lambda x: float((x - m).sum()))
    g = len(tot)
    return float(np.sqrt(g / (g - 1) * (tot**2).sum()) / len(f)) if g > 1 else float("nan")


def divergence(steps: pd.DataFrame, on: str, off: str) -> pd.Series:
    """First step at which the two grids' thresholds differ, per cell (NaN = never)."""
    a = steps[steps["grid"] == on].set_index([*CELL, "t"])["threshold"]
    b = steps[steps["grid"] == off].set_index([*CELL, "t"])["threshold"]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    diff = j[~np.isclose(j["a"], j["b"], rtol=0, atol=1e-12, equal_nan=True)]
    first = diff.reset_index().groupby(CELL)["t"].min()
    cells = j.reset_index()[CELL].drop_duplicates().set_index(CELL).index
    return first.reindex(cells)


def curve(d: pd.Series, rng: np.random.Generator, reps: int = 4000) -> list[dict]:
    """SE of the paired mean at k cells, by drawing k of the n cells.

    With replacement: the n cells stand in for the population of cells, so the
    sd of the k-cell means *is* the SE a k-cell grid would have, and coverage
    of the full-sample mean by mean ± 2·SE is a fair test of whether the SE
    formula can be trusted at that k.  Without replacement is reported beside
    it for k < n; the two differ only by the finite-population factor.
    """
    x = d.to_numpy()
    full = x.mean()
    out = []
    for k in KS:
        if k > len(x):
            continue
        idx = rng.integers(0, len(x), size=(reps, k))
        s = x[idx]
        m = s.mean(axis=1)
        se_hat = s.std(axis=1, ddof=1) / np.sqrt(k)
        row = {
            "k": k,
            "se_true": float(m.std(ddof=1)),
            "se_formula": float(x.std(ddof=1) / np.sqrt(k)),
            "se_hat_median": float(np.median(se_hat)),
            "se_hat_p05": float(np.percentile(se_hat, 5)),
            "se_hat_p95": float(np.percentile(se_hat, 95)),
            "coverage_2se": float(np.mean(np.abs(m - full) <= 2 * se_hat)),
            "resolved_frac": float(np.mean(np.abs(m) > 2 * se_hat)),
        }
        if k < len(x):
            w = np.array([rng.choice(x, size=k, replace=False).mean() for _ in range(reps // 4)])
            row["se_true_wor"] = float(w.std(ddof=1))
        out.append(row)
    return out


def env_table(d: pd.Series, cost: pd.Series) -> pd.DataFrame:
    """Per environment: n, mean, sd, median cell seconds, share of the Δ variance."""
    f = d.rename("d").reset_index()
    g = f.groupby("arm")["d"]
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(ddof=1)})
    t["cell_seconds"] = cost.reindex(t.index)
    ss = f.assign(dev=(f["d"] - f["d"].mean()) ** 2).groupby("arm")["dev"].sum()
    t["var_share"] = ss / ss.sum()
    return t


def allocation(t: pd.DataFrame) -> dict:
    """Variance of the cell-weighted mean under three allocations of the same budget.

    The estimand keeps each environment's current weight W_e = n_e / N, so a
    reallocation changes the SE, not what is being estimated.  For a fixed
    number of cells N the variance is Σ W_e² σ_e² / n_e: proportional (today)
    gives (1/N) Σ W_e σ_e²; Neyman (n_e ∝ W_e σ_e) gives (1/N)(Σ W_e σ_e)².  For
    a fixed budget C of cell-seconds with cell cost c_e, the optimum is
    n_e ∝ W_e σ_e / √c_e and gives (Σ W_e σ_e √c_e)² / C, against
    (Σ W_e σ_e²)(Σ W_e c_e) / C for proportional.  Reported as the fraction of
    cells (or cell-seconds) the reallocation needs for the same SE.

    A floor keeps an environment with σ_e ≈ 0 from being allocated no cells at
    all: a design still has to *show* it is flat.  The numbers are therefore
    upper bounds on the saving, computed on the same cells that estimated σ_e.
    """
    t = t.dropna(subset=["sd", "cell_seconds"])
    w = t["n"] / t["n"].sum()
    s = t["sd"].clip(lower=1e-4)
    c = t["cell_seconds"]
    prop_cells = float((w * s**2).sum())
    ney_cells = float((w * s).sum() ** 2)
    prop_cost = float((w * s**2).sum() * (w * c).sum())
    ney_cost = float((w * s * np.sqrt(c)).sum() ** 2)
    share_cells = w * s / (w * s).sum()
    share_cost = (w * s / np.sqrt(c)) / (w * s / np.sqrt(c)).sum()
    return {
        "cells_needed_neyman": ney_cells / prop_cells,
        "cost_needed_neyman_by_cost": ney_cost / prop_cost,
        "share_proportional": w.round(3).to_dict(),
        "share_neyman_cells": share_cells.round(3).to_dict(),
        "share_neyman_cost": share_cost.round(3).to_dict(),
    }


def components(d: pd.Series) -> dict:
    """Category vs seed variance of Δ, within environment (one-way random effects).

    Δ_ecs = μ_e + a_ec + ε_ecs.  With S seeds per category the within-category
    mean square estimates σ²_seed and the between-category one estimates
    σ²_seed + S·σ²_cat, pooled over environments.  If σ²_cat is ~0, a seed buys
    what a new category buys; if not, seeds of one category are partly the same
    measurement taken twice.
    """
    f = d.rename("d").reset_index()
    f["cat"] = f["arm"] + "|" + f["category"]
    f["d"] = f["d"] - f.groupby("arm")["d"].transform("mean")
    g = f.groupby("cat")["d"]
    sizes = g.size()
    f = f[f["cat"].isin(sizes[sizes >= 2].index)]
    g = f.groupby("cat")["d"]
    k = g.ngroups
    n = len(f)
    if k < 3:
        return {}
    s_bar = n / k
    msw = float(((f["d"] - g.transform("mean")) ** 2).sum() / (n - k))
    n_env = f["arm"].nunique()
    msb = float((g.size() * (g.mean() - f["d"].mean()) ** 2).sum() / (k - n_env))
    var_cat = max(0.0, (msb - msw) / s_bar)
    return {
        "categories": int(k),
        "seeds_per_category": float(s_bar),
        "sd_seed": float(np.sqrt(msw)),
        "sd_category": float(np.sqrt(var_cat)),
        "icc": var_cat / (var_cat + msw) if (var_cat + msw) > 0 else float("nan"),
    }


def steps_table(steps: pd.DataFrame, on: str, off: str, seeds, cost_at: pd.Series) -> list[dict]:
    """σ of Δ when every trajectory is cut at T steps, and what the cut saves."""
    rows = []
    for T in STEP_CUTS:
        cm = cell_means(steps, T)
        if on not in cm or off not in cm:
            continue
        s = summarise(paired(cm, on, off, seeds))
        s.update({"T": T, "cost_fraction": float(cost_at.get(T, np.nan))})
        # SE per unit of compute: a grid of the same cost holds 1/cost_fraction as many cells.
        s["se_same_cost"] = s["se"] * np.sqrt(s["cost_fraction"]) if np.isfinite(s["cost_fraction"]) else np.nan
        rows.append(s)
    return rows


def _fmt(v, digits: int) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.{digits}g}"
    return str(v)


def _md(df: pd.DataFrame, digits: int = 3) -> str:
    """A markdown table without ``tabulate`` (not in the GRID venv)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--published", default="/expscratch/sgreenberg/anchem-3825/analysis/ab_ll1e-8/agg/ab_paired_cells.csv"
    )
    ap.add_argument("--seed", type=int, default=3840)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    steps = pd.read_csv(args.steps)
    cm = cell_means(steps)
    grids = set(cm.columns)
    md: list[str] = ["# #3840 tables (generated by `resolution_3840.py`)\n"]

    # 0. Reproduce the published line before trusting anything built on it.
    d825 = paired(cm, "ll1e-8", "baseline")
    s825 = summarise(d825)
    if abs(s825["mean"] - 0.0049) > 5e-5 or abs(s825["se"] - 0.0037) > 5e-5:
        raise SystemExit(f"cannot reproduce #3825's ll1e-8 line: {s825}")
    pub = Path(args.published)
    if pub.exists():
        p = pd.read_csv(pub)
        p = p[(p["scope"] == "app_visible") & (p["window"] == "all_steps")]
        p = p.set_index(CELL)["cost_on"] - p.set_index(CELL)["cost_off"]
        worst = float((p - d825.reindex(p.index)).abs().max())
        if worst > 1e-12:
            raise SystemExit(f"per-cell Δ disagrees with the published frame by {worst}")
        md.append(f"Reproduced #3825's `ll1e-8` line cell by cell ({len(p)} cells, max |diff| {worst:.1e}).\n")

    # 1. Every pair, seeds 0-1.
    pair_rows, curves, envs, allocs, comps = [], [], [], [], []
    cell_cost = steps.groupby(["grid", *CELL])["elapsed_seconds"].max().groupby("arm").median()
    pairs = [p for p in EXISTING_PAIRS if p[1] in grids and p[2] in grids]
    if {"v_ll1e-8", "v_baseline"} <= grids:
        pairs.append(("v:ll1e-8-baseline", "v_ll1e-8", "v_baseline", "validation", "seeds 2-8, the validation grid"))
    for label, on, off, kind, desc in pairs:
        seeds = VALIDATION_SEEDS if kind == "validation" else (0, 1)
        d = paired(cm, on, off, seeds)
        if d.empty:
            continue
        s = summarise(d)
        s["se_clustered"] = clustered_se(d)
        div = divergence(steps[steps["seed"].isin(seeds)], on, off)
        s.update(
            {
                "pair": label,
                "kind": kind,
                "what": desc,
                "never_diverge": float(div.isna().mean()),
                "median_divergence_step": float(div.median()),
            }
        )
        pair_rows.append(s)
        for r in curve(d, rng):
            curves.append({"pair": label, **r})
        t = env_table(d, cell_cost)
        envs.append(t.reset_index().assign(pair=label))
        allocs.append({"pair": label, **allocation(t)})
        c = components(d)
        if c:
            comps.append({"pair": label, **c})
    pairs_df = pd.DataFrame(pair_rows)[
        [
            "pair",
            "kind",
            "n",
            "mean",
            "sd",
            "se",
            "se_clustered",
            "frac_zero",
            "never_diverge",
            "median_divergence_step",
            "what",
        ]
    ]
    pairs_df.to_csv(out / "pairs.csv", index=False)
    md += ["## Pairs\n", _md(pairs_df), "\n"]
    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(out / "curve.csv", index=False)
    md += ["## Subsampled SE against k\n", _md(curve_df), "\n"]
    env_df = pd.concat(envs, ignore_index=True)
    env_df.to_csv(out / "env.csv", index=False)
    md += ["## Per environment\n", _md(env_df), "\n"]
    alloc_df = pd.DataFrame(allocs)
    alloc_df.to_json(out / "alloc.json", orient="records", indent=2)
    md += ["## Reallocation\n", _md(alloc_df[["pair", "cells_needed_neyman", "cost_needed_neyman_by_cost"]]), "\n"]
    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(out / "components.csv", index=False)
    md += ["## Category vs seed\n", _md(comp_df), "\n"]

    # 2. The design rule: cells needed for δ, at the σ each kind of pair has.
    arm_sd = pairs_df[pairs_df["kind"] == "arm"]["sd"]
    sigmas = {"low": float(arm_sd.min()), "median": float(arm_sd.median()), "high": float(arm_sd.max())}
    design = [
        {
            "delta": dl,
            **{f"N_2se_{k}": int(np.ceil((Z_RESOLVE * v / dl) ** 2)) for k, v in sigmas.items()},
            **{f"N_80pct_{k}": int(np.ceil((Z_POWER80 * v / dl) ** 2)) for k, v in sigmas.items()},
        }
        for dl in DELTAS
    ]
    design_df = pd.DataFrame(design)
    design_df.to_csv(out / "design.csv", index=False)
    md += [f"## Cells needed (σ low/median/high = {sigmas})\n", _md(design_df), "\n"]

    # 3. Steps: cut every trajectory at T.
    base_grid = "baseline"
    cum = steps[steps["grid"] == base_grid].groupby([*CELL, "t"])["elapsed_seconds"].max().reset_index()
    tot = cum.groupby(CELL)["elapsed_seconds"].max()
    cost_at = {}
    for T in STEP_CUTS:
        at = cum[cum["t"] <= T].groupby(CELL)["elapsed_seconds"].max()
        cost_at[T] = float(at.sum() / tot.reindex(at.index).sum())
    step_rows = []
    for label, on, off, kind, _ in pairs:
        seeds = VALIDATION_SEEDS if kind == "validation" else (0, 1)
        for r in steps_table(steps, on, off, seeds, pd.Series(cost_at)):
            step_rows.append({"pair": label, **r})
    steps_df = pd.DataFrame(step_rows)
    steps_df.to_csv(out / "steps.csv", index=False)
    md += ["## Trajectories cut at T steps\n", _md(steps_df), "\n"]

    # 4. The validation, drift and determinism.
    val: dict = {"predicted_se": PREDICTED_SE, "predicted_se_90": PREDICTED_SE_90, "predicted_sd_90": PREDICTED_SD_90}
    if {"v_ll1e-8", "v_baseline"} <= grids:
        v = summarise(paired(cm, "v_ll1e-8", "v_baseline", VALIDATION_SEEDS))
        v["se_clustered"] = clustered_se(paired(cm, "v_ll1e-8", "v_baseline", VALIDATION_SEEDS))
        v["se_pass"] = PREDICTED_SE_90[0] <= v["se"] <= PREDICTED_SE_90[1]
        v["sd_pass"] = PREDICTED_SD_90[0] <= v["sd"] <= PREDICTED_SD_90[1]
        val["validation"] = v
        allv = pd.concat([paired(cm, "v_ll1e-8", "v_baseline", VALIDATION_SEEDS), d825])
        val["pooled_with_0913_seeds"] = {**summarise(allv), "se_clustered": clustered_se(allv)}
        pd.concat(
            [
                d825.rename("delta").reset_index().assign(set="0913"),
                paired(cm, "v_ll1e-8", "v_baseline", VALIDATION_SEEDS)
                .rename("delta")
                .reset_index()
                .assign(set="validation"),
            ]
        ).to_csv(out / "cell_deltas.csv", index=False)
        val["validation_by_env"] = (
            env_table(paired(cm, "v_ll1e-8", "v_baseline", VALIDATION_SEEDS), cell_cost)
            .reset_index()
            .to_dict(orient="records")
        )
        for new, old in (("v_baseline", "baseline"), ("v_ll1e-8", "ll1e-8")):
            d = paired(cm, new, old, (0, 1))
            div = divergence(steps[steps["seed"].isin((0, 1))], new, old)
            val[f"drift_{old}"] = {**summarise(d), "never_diverge": float(div.isna().mean())}
    if {"v_rep", "v_baseline"} <= grids:
        d = paired(cm, "v_rep", "v_baseline", (0, 1))
        div = divergence(steps[steps["seed"].isin((0, 1))], "v_rep", "v_baseline")
        val["determinism"] = {**summarise(d), "never_diverge": float(div.isna().mean())}
    (out / "validation.json").write_text(json.dumps(val, indent=2, default=float))
    md += ["## Validation\n", "```json\n" + json.dumps(val, indent=2, default=float) + "\n```\n"]

    (out / "TABLES.md").write_text("\n".join(md))
    print((out / "TABLES.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

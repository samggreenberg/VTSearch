"""#3557: price the sign-dependent ("hinge") cut rule against the shipped ``mid_tilt``.

Reads two run-level arms of ``launch_hinge_3557.sh`` - ``incumbent`` (live rule
``mid_tilt``) and ``hinge`` (live rule ``hinge``) - and answers the
pre-registered decision rule in
``docs/experiments/2026-09-22-hinge-tilt-3557/PLAN.md``:

1. **Contract** - every re-cut rule's threshold must be non-increasing along the
   23-stop grid (seam stops ``-0.5``/``-0.25`` included) at every cell-step, on
   both arms.  Counted over *all* steps: nesting is not a deep-regime property.
2. **No material harm** - paired Δ against the incumbent at each (environment,
   k), upper 95% bound below +0.010, both for the *reporting-only* re-cut (arm A,
   ``hinge`` vs ``mid_tilt`` rows of the same step) and for the *full ship* (arm
   B's ``hinge`` rows vs arm A's ``mid_tilt`` rows, paired on cell).
3. **A real gain** - full-ship Δcost resolvably negative at some k < 0 in >= 3
   of 5 environments.
4. **The trajectory does not pay** - at k=0: end-of-session cost, cost-AUC and
   clicks-to-target, per environment.
5. **The knob survives** - distinct admitted sets across the 21 integer stops.

**Units.** Every cost and regret is on the rate scale (``/ 2**|k|``; #2865).
**Unit of analysis.** The *cell* (dataset x embedder x style x category x
seed): each cell is reduced to its deep-band (``n_votes > 100``) mean first,
then differenced and bootstrapped over cells.  Steps of one trajectory share a
model and are not independent.

The reduction is streamed one cell at a time (a cell's cut-inclusion frame is
~50k rows per style), so memory is bounded by the per-cell aggregates rather
than by the ~35M raw rows of the two arms.

Writes ``hinge3557_*.csv`` and ``hinge3557_summary.json`` into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

#: Deep regime, #2865's band: past this many votes the cut rule rather than the
#: anchor supply is what is measured.
DEEP_VOTES_MIN = 100
#: The pre-registered non-inferiority bar (#2891 / #3319), in rate-scale cost.
HARM_BAR = 0.010
#: Bootstrap resamples over cells.
N_BOOT = 2000
INCUMBENT = "mid_tilt"
HINGES = ("hinge", "hinge_raw", "hinge_cont")
ENV_KEYS = ["dataset", "embedder", "style"]
CELL_KEYS = [*ENV_KEYS, "category", "seed"]
#: Pre-registered environment count for rule 3 ("three of five").
N_ENVS_FOR_GAIN = 3

_CUTINCL_COLS = [
    "seed",
    "dataset",
    "category",
    "style",
    "t",
    "n_good",
    "n_bad",
    "embedder",
    "cut_rule",
    "inclusion_k",
    "fold_quantile",
    "cut_threshold",
    "cut_cost",
    "cut_fpr",
    "cut_fnr",
    "cut_regret",
    "admitted_frac",
    "n_admitted",
    "n_test",
    "seam_q_mid",
    "seam_q_cross0",
    "seam_q_rate0",
    "fit_log2_prior_odds",
    "fit_log2_var_ratio",
]
_MAIN_COLS = [
    "seed",
    "dataset",
    "category",
    "style",
    "t",
    "n_good",
    "n_bad",
    "embedder",
    "gmm_variant",
    "pool_variant",
    "threshold",
    "threshold_provenance",
    "cost",
    "fpr",
    "fnr",
    "average_precision",
    "live_cut_rule",
]


def _task_index(path: Path) -> int:
    m = re.search(r"task_(\d+)", path.name)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Per-cell reduction
# ---------------------------------------------------------------------------


def _contract(df: pd.DataFrame) -> pd.DataFrame:
    """Per (style, rule): steps audited, steps whose threshold rises anywhere
    along k, and the worst rise in threshold and in admitted count."""
    out = []
    for (style, rule), g in df.groupby(["style", "cut_rule"], sort=False):
        g = g.sort_values(["t", "inclusion_k"])
        thr = g["cut_threshold"].to_numpy()
        adm = g["n_admitted"].to_numpy()
        t = g["t"].to_numpy()
        same = t[1:] == t[:-1]
        d_thr = np.where(same, thr[1:] - thr[:-1], 0.0)
        d_adm = np.where(same, adm[:-1] - adm[1:], 0)  # > 0: fewer admitted at the larger k
        bad = (d_thr > 1e-9) | (d_adm > 0)
        viol_steps = np.unique(t[1:][bad])
        out.append(
            {
                "style": style,
                "cut_rule": rule,
                "n_steps": int(np.unique(t).size),
                "n_violating_steps": int(viol_steps.size),
                "max_threshold_rise": float(d_thr.max()) if d_thr.size else 0.0,
                "max_admitted_rise": int(d_adm.max()) if d_adm.size else 0,
            }
        )
    return pd.DataFrame(out)


def _liveness(deep: pd.DataFrame) -> pd.DataFrame:
    """Distinct admitted sets across the integer stops, and the dead-step share."""
    ints = deep[deep["inclusion_k"] == deep["inclusion_k"].round()]
    rows = []
    for (style, rule), g in ints.groupby(["style", "cut_rule"], sort=False):
        piv = g.pivot_table(index="t", columns="inclusion_k", values="n_admitted", aggfunc="first")
        arr = piv.to_numpy()
        distinct = np.array([np.unique(r[~np.isnan(r)]).size for r in arr])
        dead = (np.diff(arr, axis=1) == 0).mean(axis=1)
        rows.append(
            {
                "style": style,
                "cut_rule": rule,
                "distinct_sets": float(distinct.mean()),
                "dead_step_rate": float(np.nanmean(dead)),
            }
        )
    return pd.DataFrame(rows)


def _reduce_cell(args: tuple[str, str]) -> dict:
    """One cell's cut-inclusion + main frames -> the aggregates the study reads."""
    arm, cutincl_path = args
    cutincl_path = Path(cutincl_path)
    main_path = cutincl_path.with_name(cutincl_path.name.replace("__cutincl", ""))
    out: dict = {"arm": arm, "task": _task_index(cutincl_path), "ok": False}
    try:
        df = pd.read_csv(cutincl_path, usecols=lambda c: c in _CUTINCL_COLS)
    except (OSError, pd.errors.EmptyDataError, ValueError) as exc:
        out["error"] = f"cutincl unreadable: {exc}"
        return out
    if df.empty:
        out["error"] = "cutincl empty"
        return out
    df["n_votes"] = df["n_good"] + df["n_bad"]
    df["scale"] = 2.0 ** df["inclusion_k"].abs()
    df["rcost"] = df["cut_cost"] / df["scale"]
    df["rregret"] = df["cut_regret"] / df["scale"]
    ident = {c: df[c].iloc[0] for c in ["dataset", "embedder", "category", "seed"]}

    out["contract"] = _contract(df).assign(arm=arm, **ident)

    deep = df[df["n_votes"] > DEEP_VOTES_MIN]
    out["n_deep_steps"] = int(deep["t"].nunique())
    if not deep.empty:
        agg = (
            deep.groupby(["style", "cut_rule", "inclusion_k"], sort=False)
            .agg(
                rcost=("rcost", "mean"),
                rregret=("rregret", "mean"),
                fpr=("cut_fpr", "mean"),
                fnr=("cut_fnr", "mean"),
                admitted_frac=("admitted_frac", "mean"),
                fold_quantile=("fold_quantile", "mean"),
                n_steps=("t", "nunique"),
            )
            .reset_index()
        )
        out["deep"] = agg.assign(arm=arm, **ident)
        out["liveness"] = _liveness(deep).assign(arm=arm, **ident)

        # Guard binding: share of deep steps where the guarded hinge differs from
        # the literal one (C(k) < M(k): the guard took over), per k < 0.
        piv = deep[deep["cut_rule"].isin(["hinge", "hinge_raw"]) & (deep["inclusion_k"] < 0)].pivot_table(
            index=["style", "t", "inclusion_k"], columns="cut_rule", values="cut_threshold", aggfunc="first"
        )
        if {"hinge", "hinge_raw"} <= set(piv.columns):
            bind = (
                (piv["hinge"] != piv["hinge_raw"])
                .groupby(level=["style", "inclusion_k"])
                .mean()
                .rename("guard_binds")
                .reset_index()
            )
            out["guard"] = bind.assign(arm=arm, **ident)

    # Seam + mechanism: one row per step (the columns are per-step constants).
    seam = (
        df[(df["cut_rule"] == INCUMBENT) & (df["inclusion_k"] == 0)]
        .drop_duplicates(["style", "t"])[
            [
                "style",
                "t",
                "n_votes",
                "seam_q_mid",
                "seam_q_cross0",
                "seam_q_rate0",
                "fit_log2_prior_odds",
                "fit_log2_var_ratio",
            ]
        ]
        .copy()
    )
    out["seam"] = seam.assign(arm=arm, **ident)

    # Literal seam: raw hinge threshold at the stop just below 0 vs at 0.
    raw = df[(df["cut_rule"] == "hinge_raw") & df["inclusion_k"].isin([-1, -0.25, 0])]
    rp = raw.pivot_table(index=["style", "t"], columns="inclusion_k", values="n_admitted", aggfunc="first")
    if 0 in rp.columns:
        rp = rp.rename(columns={-1: "adm_m1", -0.25: "adm_m025", 0: "adm_0"}).reset_index()
        out["raw_seam"] = rp.assign(arm=arm, **ident)

    # Trajectory endpoints at the reporting inclusion (the live row).
    try:
        main = pd.read_csv(main_path, usecols=lambda c: c in _MAIN_COLS)
    except (OSError, pd.errors.EmptyDataError, ValueError) as exc:
        out["error"] = f"main unreadable: {exc}"
        return out
    live = main[main["gmm_variant"].isna() | (main["gmm_variant"].astype(str) == "")]
    live = live.drop_duplicates(["style", "t"], keep="first").copy()
    live["n_votes"] = live["n_good"] + live["n_bad"]
    out["traj"] = live[["style", "t", "n_votes", "n_good", "cost", "average_precision", "threshold_provenance"]].assign(
        arm=arm, **ident
    )
    out["live_rules"] = sorted(main["live_cut_rule"].dropna().astype(str).unique().tolist())
    out["ok"] = True
    return out


def load_arm(arm: str, results: Path, workers: int) -> list[dict]:
    files = sorted((results / "cells").glob("task_*__cutincl.csv"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_reduce_cell, [(arm, str(f)) for f in files], chunksize=4))


def _cat(parts: list[dict], key: str) -> pd.DataFrame:
    frames = [p[key] for p in parts if p.get("ok") and key in p and p[key] is not None and len(p[key])]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Mean, SE and 95% percentile CI of the mean, resampling cells."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), np.nan, np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    means = x[idx].mean(axis=1)
    return (
        float(x.mean()),
        float(means.std(ddof=1)),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def env_name(r) -> str:
    return f"{r['dataset']} x {r['embedder']} x {r['style']}"


def paired_table(a: pd.DataFrame, b: pd.DataFrame, value: str, rng: np.random.Generator) -> pd.DataFrame:
    """Per (env, k): mean/SE/CI of b - a over cells present in both."""
    keys = [*CELL_KEYS, "inclusion_k"]
    m = a[[*keys, value]].merge(b[[*keys, value]], on=keys, suffixes=("_a", "_b"))
    m["d"] = m[f"{value}_b"] - m[f"{value}_a"]
    rows = []
    for (ds, emb, sty, k), g in m.groupby([*ENV_KEYS, "inclusion_k"]):
        mean, se, lo, hi = boot_ci(g["d"].to_numpy(), rng)
        rows.append(
            {
                "dataset": ds,
                "embedder": emb,
                "style": sty,
                "inclusion_k": k,
                "n_cells": int(g["d"].notna().sum()),
                "d_mean": mean,
                "d_se": se,
                "d_lo": lo,
                "d_hi": hi,
                "a_mean": float(g[f"{value}_a"].mean()),
                "b_mean": float(g[f"{value}_b"].mean()),
            }
        )
    return pd.DataFrame(rows)


def clicks_to_target(traj: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """#3319's speed lens, per cell: first click at which each arm's k=0 cost is
    at or below the INCUMBENT's own end-of-session cost (median of its last 10
    steps).  Paired difference hinge - incumbent; a cell that never reaches it is
    given the horizon + 1 (censored), and counted."""
    rows = []
    for key, g in traj.groupby(CELL_KEYS):
        inc = g[g["arm"] == "incumbent"].sort_values("t")
        hin = g[g["arm"] == "hinge"].sort_values("t")
        if inc.empty or hin.empty:
            continue
        target = float(inc["cost"].tail(10).median())
        horizon = int(max(inc["t"].max(), hin["t"].max()))

        def first(df: pd.DataFrame) -> tuple[int, bool]:
            hit = df[df["cost"] <= target]
            return (int(hit["t"].iloc[0]), True) if len(hit) else (horizon + 1, False)

        ti, ri = first(inc)
        th, rh = first(hin)
        rows.append(
            dict(
                zip(CELL_KEYS, key, strict=True),
                target=target,
                clicks_incumbent=ti,
                clicks_hinge=th,
                reached_incumbent=ri,
                reached_hinge=rh,
                final_incumbent=target,
                final_hinge=float(hin["cost"].tail(10).median()),
                auc_incumbent=float(inc["cost"].mean()),
                auc_hinge=float(hin["cost"].mean()),
                ap_final_incumbent=float(inc["average_precision"].tail(10).median()),
                ap_final_hinge=float(hin["average_precision"].tail(10).median()),
                pos_incumbent=int(inc["n_good"].iloc[-1]),
                pos_hinge=int(hin["n_good"].iloc[-1]),
            )
        )
    per_cell = pd.DataFrame(rows)
    summary = []
    for env, g in per_cell.groupby(ENV_KEYS):
        rec = dict(zip(ENV_KEYS, env, strict=True), n_cells=len(g))
        for name, a, b in [
            ("final_cost", "final_incumbent", "final_hinge"),
            ("cost_auc", "auc_incumbent", "auc_hinge"),
            ("ap_final", "ap_final_incumbent", "ap_final_hinge"),
            ("positives", "pos_incumbent", "pos_hinge"),
            ("clicks_to_target", "clicks_incumbent", "clicks_hinge"),
        ]:
            mean, se, lo, hi = boot_ci((g[b] - g[a]).to_numpy(dtype=float), rng)
            rec.update({f"{name}_d": mean, f"{name}_se": se, f"{name}_lo": lo, f"{name}_hi": hi})
            rec[f"{name}_incumbent"] = float(g[a].median()) if name == "clicks_to_target" else float(g[a].mean())
            rec[f"{name}_hinge"] = float(g[b].median()) if name == "clicks_to_target" else float(g[b].mean())
        rec["reached_incumbent"] = float(g["reached_incumbent"].mean())
        rec["reached_hinge"] = float(g["reached_hinge"].mean())
        summary.append(rec)
    return per_cell, pd.DataFrame(summary)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--incumbent", type=Path, required=True, help="arm A results dir")
    ap.add_argument("--hinge", type=Path, required=True, help="arm B results dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3557)

    parts = {
        arm: load_arm(arm, path, args.workers) for arm, path in (("incumbent", args.incumbent), ("hinge", args.hinge))
    }
    coverage = {}
    for arm, ps in parts.items():
        bad = [p for p in ps if not p.get("ok")]
        live = sorted({r for p in ps if p.get("ok") for r in p.get("live_rules", [])})
        coverage[arm] = {
            "cells_found": len(ps),
            "cells_ok": len(ps) - len(bad),
            "cells_failed": [{"task": p["task"], "error": p.get("error")} for p in bad],
            "live_rules_recorded": live,
        }
        print(f"[{arm}] {len(ps)} cells, {len(bad)} unreadable/empty; live rule(s) recorded: {live}")
    allp = parts["incumbent"] + parts["hinge"]

    # 1. Contract.
    contract = _cat(allp, "contract")
    contract_sum = (
        contract.groupby(["arm", "dataset", "embedder", "style", "cut_rule"])
        .agg(
            n_steps=("n_steps", "sum"),
            n_violating_steps=("n_violating_steps", "sum"),
            max_threshold_rise=("max_threshold_rise", "max"),
            max_admitted_rise=("max_admitted_rise", "max"),
        )
        .reset_index()
    )
    contract_sum["violation_rate"] = contract_sum["n_violating_steps"] / contract_sum["n_steps"]
    contract_sum.to_csv(args.out / "hinge3557_contract.csv", index=False)

    # 2-3. Paired deltas.
    deep = _cat(allp, "deep")
    deep.to_csv(args.out / "hinge3557_deep_cells.csv.gz", index=False)
    a_inc = deep[(deep["arm"] == "incumbent") & (deep["cut_rule"] == INCUMBENT)]
    tables = []
    # Reporting-only: every rule on arm A against arm A's mid_tilt, same step.
    for rule in sorted(set(deep["cut_rule"]) - {INCUMBENT}):
        b = deep[(deep["arm"] == "incumbent") & (deep["cut_rule"] == rule)]
        t = paired_table(a_inc, b, "rregret", rng).assign(contrast="recut", arm_b=rule)
        tables.append(t)
    # Full ship: arm B's hinge rows against arm A's mid_tilt rows (cost AND regret).
    b_ship = deep[(deep["arm"] == "hinge") & (deep["cut_rule"] == "hinge")]
    tables.append(paired_table(a_inc, b_ship, "rcost", rng).assign(contrast="ship_cost", arm_b="hinge"))
    tables.append(paired_table(a_inc, b_ship, "rregret", rng).assign(contrast="ship_regret", arm_b="hinge"))
    # The trajectory alone: arm B's mid_tilt re-cut vs arm A's mid_tilt - same
    # rule, different votes.  Isolates what the acquisition change did.
    b_traj = deep[(deep["arm"] == "hinge") & (deep["cut_rule"] == INCUMBENT)]
    tables.append(paired_table(a_inc, b_traj, "rcost", rng).assign(contrast="trajectory_only", arm_b=INCUMBENT))
    paired = pd.concat(tables, ignore_index=True)
    paired["harmed"] = paired["d_hi"] >= HARM_BAR
    paired["sig_better"] = paired["d_hi"] < 0
    paired["sig_worse"] = paired["d_lo"] > 0
    paired.to_csv(args.out / "hinge3557_paired.csv", index=False)

    # 5. Liveness.
    live = _cat(allp, "liveness")
    live_sum = (
        live.groupby(["arm", "dataset", "embedder", "style", "cut_rule"])[["distinct_sets", "dead_step_rate"]]
        .mean()
        .reset_index()
    )
    live_sum.to_csv(args.out / "hinge3557_liveness.csv", index=False)

    # Guard binding and the seam / mechanism.
    guard = _cat(allp, "guard")
    if not guard.empty:
        guard.groupby(["arm", "dataset", "embedder", "style", "inclusion_k"])[
            "guard_binds"
        ].mean().reset_index().to_csv(args.out / "hinge3557_guard.csv", index=False)
    seam = _cat(allp, "seam")
    seam["cross_below_mid"] = seam["seam_q_cross0"] < seam["seam_q_mid"]
    seam["deep"] = seam["n_votes"] > DEEP_VOTES_MIN
    seam.to_csv(args.out / "hinge3557_seam_steps.csv.gz", index=False)
    seam_sum = (
        seam.groupby(["arm", "dataset", "embedder", "style", "deep"])
        .agg(
            n_steps=("t", "size"),
            cross_below_mid=("cross_below_mid", "mean"),
            q_mid=("seam_q_mid", "median"),
            q_cross0=("seam_q_cross0", "median"),
            q_rate0=("seam_q_rate0", "median"),
            prior_odds_bits=("fit_log2_prior_odds", "median"),
            var_ratio_bits=("fit_log2_var_ratio", "median"),
        )
        .reset_index()
    )
    seam_sum.to_csv(args.out / "hinge3557_seam.csv", index=False)
    raw_seam = _cat(allp, "raw_seam")
    if not raw_seam.empty:
        raw_seam.to_csv(args.out / "hinge3557_raw_seam_steps.csv.gz", index=False)

    # 4. Trajectory endpoints.
    traj = _cat(allp, "traj")
    traj.to_csv(args.out / "hinge3557_traj.csv.gz", index=False)
    per_cell, speed = clicks_to_target(traj, rng)
    per_cell.to_csv(args.out / "hinge3557_speed_cells.csv", index=False)
    speed.to_csv(args.out / "hinge3557_speed.csv", index=False)

    # ---- the decision rule ----
    envs = sorted({(r.dataset, r.embedder, r.style) for r in paired.itertuples()})
    ints = paired[paired["inclusion_k"] == paired["inclusion_k"].round()]
    ship = ints[ints["contrast"] == "ship_cost"]
    recut_h = ints[(ints["contrast"] == "recut") & (ints["arm_b"] == "hinge")]
    c_h = contract_sum[contract_sum["cut_rule"] == "hinge"]
    rule1 = bool((c_h["n_violating_steps"] == 0).all()) and len(c_h) > 0
    harm_ship = ship[ship["harmed"]]
    harm_recut = recut_h[recut_h["harmed"]]
    rule2 = harm_ship.empty and harm_recut.empty
    gain_envs = sorted(
        {(r.dataset, r.embedder, r.style) for r in ship.itertuples() if r.inclusion_k < 0 and r.sig_better}
    )
    rule3 = len(gain_envs) >= N_ENVS_FOR_GAIN
    rule4_rows = []
    for r in speed.itertuples():
        ok = (
            r.final_cost_hi < HARM_BAR
            and r.cost_auc_hi < HARM_BAR
            and not (np.isfinite(r.clicks_to_target_lo) and r.clicks_to_target_lo > 0)
        )
        rule4_rows.append({"env": f"{r.dataset} x {r.embedder} x {r.style}", "ok": bool(ok)})
    rule4 = bool(rule4_rows) and all(x["ok"] for x in rule4_rows)
    ls = live_sum.set_index(["arm", "dataset", "embedder", "style", "cut_rule"])["distinct_sets"]
    rule5_rows = []
    for ds, emb, sty in envs:
        try:
            inc = ls[("incumbent", ds, emb, sty, INCUMBENT)]
            hin = ls[("hinge", ds, emb, sty, "hinge")]
        except KeyError:
            continue
        rule5_rows.append(
            {"env": f"{ds} x {emb} x {sty}", "incumbent": inc, "hinge": hin, "ok": bool(hin >= inc - 0.5)}
        )
    rule5 = bool(rule5_rows) and all(x["ok"] for x in rule5_rows)

    crossover = (
        ints[ints["contrast"].isin(["ship_cost", "recut"])]
        .groupby(["contrast", "arm_b"])
        .agg(
            n=("d_mean", "size"),
            sig_better=("sig_better", "sum"),
            sig_worse=("sig_worse", "sum"),
            harmed=("harmed", "sum"),
        )
        .reset_index()
    )
    crossover.to_csv(args.out / "hinge3557_crossover.csv", index=False)

    summary = {
        "coverage": coverage,
        "deep_votes_min": DEEP_VOTES_MIN,
        "harm_bar": HARM_BAR,
        "rule1_contract": rule1,
        "rule2_no_harm": rule2,
        "rule2_harmed_ship": harm_ship[["dataset", "embedder", "style", "inclusion_k", "d_mean", "d_hi"]].to_dict(
            "records"
        ),
        "rule2_harmed_recut": harm_recut[["dataset", "embedder", "style", "inclusion_k", "d_mean", "d_hi"]].to_dict(
            "records"
        ),
        "rule3_gain": rule3,
        "rule3_gain_envs": [" x ".join(e) for e in gain_envs],
        "rule4_trajectory": rule4,
        "rule4_rows": rule4_rows,
        "rule5_liveness": rule5,
        "rule5_rows": rule5_rows,
        "ships": bool(rule1 and rule2 and rule3 and rule4 and rule5),
        "crossover": crossover.to_dict("records"),
    }
    (args.out / "hinge3557_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(
        json.dumps({k: v for k, v in summary.items() if k.startswith("rule") or k == "ships"}, indent=2, default=float)
    )


if __name__ == "__main__":
    main()

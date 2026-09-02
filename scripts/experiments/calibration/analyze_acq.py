"""Does decoupling the acquisition cut buy back the positives the fused threshold costs?

Design and pre-registered decision rules:
``docs/ML.md`` (threshold calibration).  Background: #2847 / PR #2873
found production finds half as many positives as the conformal path it replaced
(median 9 -> 4 per 100 votes, p=1e-20) with final cost unchanged (p=0.09),
because one number is both the reported decision line and the rank position
Autopilot's ``hard`` pick samples around.

Arms move **only** the selector's cut; reporting stays at inclusion 0 throughout,
so cost is comparable across arms.

Three things this analyzer refuses to do quietly:

1. **Report a result without checking the lever moved.** Every arm's
   ``acq_pool_percentile`` is compared against its ``report_pool_percentile``.
   If the sampling position did not shift, the arm measured nothing and says so.
2. **Call a null a null without power.** The ship rule turns on cost *not*
   regressing, so the analyzer reports the CI on that delta, not just its
   p-value - a wide null and a tight null are different findings, and #2847's
   near-miss came from treating the first as the second.
3. **Read the falsification arm as decoration.** ``acq_p2`` must move positives
   the wrong way; if it does not, the mechanism is wrong and the verdict is
   withheld.
4. **Pool two voting modes into one verdict.** Added for #2877's pile re-run,
   whose grid deliberately holds both.  The whole reason this question is still
   open is that the answer moved between environments, so a mean over a grid
   that spans them is precisely the number that would hide it: the ship rule is
   evaluated **per mode** and the pooled table is printed as descriptive only.

Two things it adds beyond the per-mode split, both from what the earlier runs
cost:

* **Sizing, as an output rather than an assumption.**  #2877 inherited #2876's
  seed count and came back with a decision-endpoint CI spanning two opposite
  shipping decisions.  The realized paired SD on ``final_cost`` is reported per
  mode together with the ``n`` a +/-0.010 half-width needs, so "run more seeds"
  is a number the run itself produces.
* **The mode contrast as a difference-in-differences, within one embedder.**
  ``siglip+dinov3_patch`` runs ``whole_image`` and ``max_patch`` inside one
  task, off one loaded pickle, on one sim/test split and one exemplar - so
  ``(arm - prod | region) - (arm - prod | binary)`` is paired cell-for-cell and
  carries no embedder with it.  #3115 reported a per-mode headline off a grid
  whose modes were disjoint embedders; this is the same contrast without that.

Writes ``agg/*.csv``, ``acq_summary.json``, ``figures/*.png`` and
``REPORT_acq.md`` under ``$CALIB_EXP/analysis``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze_spikes as sp  # noqa: E402  (reuse the #2847 loader + spike rule)

try:
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # noqa: BLE001
    _wilcoxon = None

#: Arms in sweep order; ``prod`` is the control.
#:
#: #2877's seven are the default so that run still reproduces byte-for-byte.
#: ``ACQ_ANALYZE_ARMS`` widens it for a grid that swept more of them - #3319 runs
#: twelve, including half steps.  An override is a *list*, in sweep order, and it
#: is the study's launcher that decides it; nothing here guesses from the
#: directory listing, because an arm that failed to produce cells would then
#: silently drop out of the frontier instead of being reported missing.
_DEFAULT_ARMS = ("acq_m4", "acq_m3", "acq_m2", "acq_m1", "prod", "acq_p2", "rank_pin")
ARMS: tuple[str, ...] = tuple(
    a.strip() for a in os.environ.get("ACQ_ANALYZE_ARMS", ",".join(_DEFAULT_ARMS)).split(",") if a.strip()
)
CONTROL = "prod"
FALSIFIER = "acq_p2"


def _arm_k(arm: str) -> float | None:
    """The nominal acquisition inclusion an arm's NAME declares.

    ``prod`` -> 0, ``acq_m3`` -> -3, ``acq_p2`` -> +2, and (#3319) the ``h``
    suffix is a HALF step: ``acq_m3h`` -> -3.5.  Derived rather than tabulated so
    that adding an arm to a launcher cannot leave it off the frontier's x-axis
    with no complaint - the failure mode a hand-maintained dict has.

    ``None`` means "not on the inclusion scale" (``rank_pin``), which is what
    keeps it out of the frontier fit and in its own marker.
    """
    if arm == CONTROL:
        return 0.0
    m = re.fullmatch(r"acq_([mp])(\d+)(h?)", arm)
    if m is None:
        return None
    return (-1.0 if m.group(1) == "m" else 1.0) * (float(m.group(2)) + (0.5 if m.group(3) else 0.0))


#: Nominal acquisition inclusion per arm, for the frontier's x-axis. ``rank_pin``
#: is not on the inclusion scale and is plotted separately.
ARM_K: dict[str, float] = {a: k for a in ARMS if (k := _arm_k(a)) is not None}
_EXPLICIT_LABEL: dict[str, str] = {
    "prod": "prod (k=0, shipped)",
    "acq_p2": "k=+2 (falsifier)",
    "rank_pin": "rank-pinned 0.959",
}
ARM_LABEL: dict[str, str] = {a: _EXPLICIT_LABEL.get(a, f"k={ARM_K[a]:g}" if a in ARM_K else a) for a in ARMS}

#: Ship rule (pre-registered): positives must rise, cost must not regress by more
#: than this at the 95% upper bound, and deep-spike incidence must not rise.
COST_REGRESSION_TOLERANCE = float(os.environ.get("ACQ_COST_TOL", "0.01"))
ALPHA = 0.05

#: The decision endpoint's target half-width, in cost units.  The sizing block
#: reports the ``n`` this needs; it is the same +/-0.010 #2877 derived against,
#: so the two runs' power statements are the same statement.
TARGET_HALF_WIDTH = float(os.environ.get("ACQ_TARGET_HALFWIDTH", "0.010"))

OUT = Path(os.environ.get("ACQ_OUT", str(common.EXP / "analysis")))


def load_all(root: Path) -> tuple[pd.DataFrame, dict]:
    parts, prov = [], {}
    for arm in ARMS:
        df, p = sp.load_arm(root / arm)
        prov[arm] = p
        if not df.empty:
            df = df.copy()
            df["arm"] = arm
            parts.append(df)
    if not parts:
        return pd.DataFrame(), prov
    return pd.concat(parts, ignore_index=True), prov


def load_halves(base: Path, halves: list[str]) -> tuple[pd.DataFrame, dict]:
    """Load ``base/<half>/<arm>/results`` for a study split into index spaces.

    The #2877 pile re-run submits one array per ``(half, arm)`` because a region
    cell holds a 2.4 GB patch pickle and a whole-image cell holds 26 MB, so they
    cannot share a memory request.  They are two index spaces and therefore two
    directory trees; they are *not* two studies, and nothing downstream should
    have to know which half a row came from -- the mode a row belongs to is read
    off its own ``(dataset, embedder, style)``, not off the path.

    Provenance keys stay per ``(half, arm)``, because "which cells did we fail
    to read" is a question about a submission.
    """
    parts, prov = [], {}
    for half in halves:
        for arm in ARMS:
            arm_dir = base / half / arm / "results"
            if not (arm_dir / "cells").is_dir():
                continue
            df, p = sp.load_arm(arm_dir)
            prov[f"{half}/{arm}"] = p
            if not df.empty:
                df = df.copy()
                df["arm"] = arm
                df["half"] = half
                parts.append(df)
    if not parts:
        return pd.DataFrame(), prov
    return pd.concat(parts, ignore_index=True), prov


def voting_mode(dataset: str, embedder: str, style: str) -> str:
    """``"region"`` or ``"binary"``, asserted from the cell rather than the flag.

    Both halves are required and neither is sufficient: a boxed dataset supplies
    the box to drag, a patch embedder supplies the grid to pool it over, and the
    ``whole_image`` style declines to use either.  Reading `region_voting=True`
    off the run's configuration instead is how #2877, #2897 and #2905 each came
    to report a binary environment as a region one.
    """
    import experiment_config as cfg  # noqa: PLC0415

    if str(style) == "whole_image":
        return "binary"
    return "region" if cfg.region_voting_for(str(dataset), str(embedder)) else "binary"


#: What makes a trajectory.  ``style`` is in it because a patch cell runs BOTH
#: styles inside one task: without it the whole-image and max_patch rows of the
#: same ``(category, seed)`` collapse into one group, and every endpoint below
#: becomes a mixture of two voting modes.  ``arm`` leads it; :data:`PAIR_KEYS`
#: is the same list with ``arm`` removed, which is what a paired test joins on.
TRAJ_KEYS: tuple[str, ...] = ("arm", "dataset", "embedder", "style", "category", "seed")
PAIR_KEYS: tuple[str, ...] = TRAJ_KEYS[1:]


def trajectory_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(arm, dataset, embedder, style, category, seed)``."""
    keys = [k for k in TRAJ_KEYS if k in df.columns]
    out = []
    for key, g in df.groupby(keys, dropna=False):
        g = g.sort_values("t")
        cost = g["cost"].to_numpy(dtype=float)
        orc = g["oracle_cost"].to_numpy(dtype=float)
        t = g["t"].to_numpy(dtype=float)
        warm = t >= sp.WARM_T
        excess = cost - orc
        deep = warm & (cost >= sp.DEEP_COST) & (excess >= sp.DEEP_EXCESS)
        # Ranking-limited steps are #2825's problem, not a threshold blip - the
        # #2847 report's distinction, kept so the guardrail means the same thing.
        genuine = deep & (orc <= 0.30)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=False))
        rec.update(
            n_steps=int(len(g)),
            # --- mechanism ---
            positives_100=int(g["n_good"].iloc[-1]),
            positives_50=int(g[g["t"] <= 50]["n_good"].max()) if (g["t"] <= 50).any() else 0,
            # --- decision ---
            final_cost=float(cost[-1]),
            mean_cost_warm=float(np.nanmean(cost[warm])) if warm.any() else np.nan,
            final_oracle_cost=float(orc[-1]),
            final_ap=float(g["average_precision"].iloc[-1]) if "average_precision" in g.columns else np.nan,
            # --- guardrails ---
            has_deep=bool(deep.any()),
            has_genuine=bool(genuine.any()),
            max_excess_warm=float(np.nanmax(excess[warm])) if warm.any() else np.nan,
            # --- did the lever actually move? ---
            acq_pct=float(g["acq_pool_percentile"].median()) if "acq_pool_percentile" in g.columns else np.nan,
            report_pct=float(g["report_pool_percentile"].median()) if "report_pool_percentile" in g.columns else np.nan,
            acq_moved_frac=(
                float((g["acq_threshold"] != g["threshold"]).mean()) if "acq_threshold" in g.columns else np.nan
            ),
        )
        out.append(rec)
    traj = pd.DataFrame(out)
    if not traj.empty:
        # Derived here rather than at load time so it exists on every path into
        # the analyzer -- including the self-test's fabricated frames, which
        # carry no `half` column and never saw a launcher.
        traj["mode"] = [
            voting_mode(r.get("dataset", ""), r.get("embedder", ""), r.get("style", "")) for _, r in traj.iterrows()
        ]
    return traj


def _paired(traj: pd.DataFrame, metric: str, arm: str, control: str = CONTROL) -> dict:
    """Wilcoxon + a bootstrap CI on the paired ``(category, seed)`` delta.

    The CI is what the ship rule reads: "cost did not regress" is a claim about
    an interval, not a p-value.
    """
    keys = [k for k in PAIR_KEYS if k in traj.columns]
    a = traj[traj["arm"] == control].set_index(keys)[metric]
    b = traj[traj["arm"] == arm].set_index(keys)[metric]
    j = pd.concat([a.rename("ctl"), b.rename("arm")], axis=1).dropna()
    if j.empty:
        return {"n_pairs": 0}
    d = (j["arm"] - j["ctl"]).to_numpy(dtype=float)
    rng = np.random.default_rng(12345)  # fixed: a CI must not move between runs
    boot = np.array([np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(4000)])
    res = {
        "n_pairs": int(len(j)),
        "control_median": float(j["ctl"].median()),
        "arm_median": float(j["arm"].median()),
        "median_delta": float(np.median(d)),
        "mean_delta": float(np.mean(d)),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "frac_arm_higher": float((d > 0).mean()),
    }
    if _wilcoxon is not None and np.any(d != 0):
        try:
            res["p"] = float(_wilcoxon(j["ctl"], j["arm"]).pvalue)
        except Exception:  # noqa: BLE001
            pass
    return res


def _mcnemar(traj: pd.DataFrame, arm: str, flag: str, control: str = CONTROL) -> dict:
    keys = [k for k in PAIR_KEYS if k in traj.columns]
    a = traj[traj["arm"] == control].set_index(keys)[flag]
    b = traj[traj["arm"] == arm].set_index(keys)[flag]
    j = pd.concat([a.rename("ctl"), b.rename("arm")], axis=1).dropna()
    if j.empty:
        return {"n_pairs": 0}
    n01 = int(((~j["ctl"].astype(bool)) & j["arm"].astype(bool)).sum())
    n10 = int((j["ctl"].astype(bool) & (~j["arm"].astype(bool))).sum())
    out = {
        "n_pairs": int(len(j)),
        "control_rate": float(j["ctl"].astype(bool).mean()),
        "arm_rate": float(j["arm"].astype(bool).mean()),
        "only_control": n10,
        "only_arm": n01,
    }
    n = n01 + n10
    if n:
        from math import comb

        k = min(n01, n10)
        out["p_exact"] = float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2.0**n)))
    return out


def verify_lever(traj: pd.DataFrame) -> dict:
    """Did each arm's sampling position actually move, and which way?

    Without this the whole run is uninterpretable: an arm whose acquisition cut
    never budged looks exactly like an arm where the lever does nothing.
    """
    out = {}
    ctl_pct = float(traj[traj["arm"] == CONTROL]["acq_pct"].median())
    for arm in ARMS:
        a = traj[traj["arm"] == arm]
        if a.empty:
            continue
        pct = float(a["acq_pct"].median())
        out[arm] = {
            "median_acq_pool_percentile": pct,
            "median_report_pool_percentile": float(a["report_pct"].median()),
            "shift_vs_control": pct - ctl_pct,
            "frac_steps_acq_differs": float(a["acq_moved_frac"].median()),
            "moved": bool(arm == CONTROL or abs(pct - ctl_pct) > 1e-6),
        }
    return out


def _core_summary(traj: pd.DataFrame) -> dict:
    per_arm = {}
    for arm in ARMS:
        a = traj[traj["arm"] == arm]
        if a.empty:
            continue
        per_arm[arm] = {
            "label": ARM_LABEL.get(arm, arm),
            "k_acq": ARM_K.get(arm),
            "n_trajectories": int(len(a)),
            "median_positives_100": float(a["positives_100"].median()),
            "median_positives_50": float(a["positives_50"].median()),
            "median_final_cost": float(a["final_cost"].median()),
            "median_mean_cost_warm": float(a["mean_cost_warm"].median()),
            "median_final_oracle_cost": float(a["final_oracle_cost"].median()),
            "median_final_ap": float(a["final_ap"].median()),
            "deep_spike_rate": float(a["has_deep"].mean()),
            "genuine_blip_rate": float(a["has_genuine"].mean()),
            "median_max_excess_warm": float(a["max_excess_warm"].median()),
        }

    contrasts = {}
    for arm in ARMS:
        if arm == CONTROL or arm not in per_arm:
            continue
        contrasts[arm] = {
            "positives_100": _paired(traj, "positives_100", arm),
            "positives_50": _paired(traj, "positives_50", arm),
            "final_cost": _paired(traj, "final_cost", arm),
            "mean_cost_warm": _paired(traj, "mean_cost_warm", arm),
            "final_oracle_cost": _paired(traj, "final_oracle_cost", arm),
            # Average precision is what separates "the extra labels taught the
            # model something" from "they were redundant" - the plan's H2. It was
            # only ever read as a marginal median, which cannot answer that; the
            # #2877 region-voting run is where the two readings come apart.
            "final_ap": _paired(traj, "final_ap", arm),
            "deep_incidence": _mcnemar(traj, arm, "has_deep"),
            "genuine_incidence": _mcnemar(traj, arm, "has_genuine"),
        }

    lever = verify_lever(traj)

    # --- the pre-registered ship rule ---
    falsifier_ok = False
    if FALSIFIER in contrasts:
        f = contrasts[FALSIFIER]["positives_100"]
        falsifier_ok = f.get("median_delta", 0) < 0 and f.get("p", 1.0) < ALPHA

    ship = {}
    for arm, c in contrasts.items():
        if arm == FALSIFIER:
            continue
        pos, cost, deep = c["positives_100"], c["final_cost"], c["deep_incidence"]
        c1 = pos.get("median_delta", 0) > 0 and pos.get("p", 1.0) < ALPHA
        c2 = cost.get("ci95_hi", float("inf")) < COST_REGRESSION_TOLERANCE
        c3 = not (deep.get("arm_rate", 0) > deep.get("control_rate", 0) and deep.get("p_exact", 1.0) < ALPHA)
        ship[arm] = {
            "positives_rose": bool(c1),
            "cost_did_not_regress": bool(c2),
            "spikes_did_not_rise": bool(c3),
            "lever_moved": bool(lever.get(arm, {}).get("moved")),
            "ADOPT": bool(c1 and c2 and c3 and lever.get(arm, {}).get("moved")),
        }

    adopt = [a for a, v in ship.items() if v["ADOPT"]]
    return {
        "n_trajectories": int(len(traj)),
        "lever_verification": lever,
        "falsifier_behaved": falsifier_ok,
        "per_arm": per_arm,
        "contrasts_vs_control": contrasts,
        "ship_rule": ship,
        "adopt": adopt,
    }


def sizing(traj: pd.DataFrame, metric: str = "final_cost") -> dict:
    """Realized paired SD on the DECISION endpoint, and the ``n`` it implies.

    #2877's one unrecoverable mistake was inheriting a seed count with an arm
    table: #2876's 8 seeds gave a decision-endpoint CI of [-0.014, +0.019],
    which is not a null - it is an interval containing both "ship it" and
    "revert it", reported as though it were one.  The input that would have
    caught it (the paired SD on ``final_cost`` in *this* environment) cannot be
    known before cells exist, so the fix is not a better guess up front but a
    number the run itself reports.

    Computed over the k<0 arms only.  The falsifier is a different distribution
    by construction and ``rank_pin`` is not on the inclusion scale, so pooling
    either would inflate the SD that sizes the arms the decision is about.
    """
    out: dict = {"metric": metric, "target_half_width": TARGET_HALF_WIDTH, "per_arm": {}}
    sds = []
    for arm in [a for a in ARMS if (ARM_K.get(a) or 0.0) < 0.0]:
        keys = [k for k in PAIR_KEYS if k in traj.columns]
        a = traj[traj["arm"] == CONTROL].set_index(keys)[metric]
        b = traj[traj["arm"] == arm].set_index(keys)[metric]
        j = pd.concat([a.rename("ctl"), b.rename("arm")], axis=1).dropna()
        if len(j) < 2:
            continue
        d = (j["arm"] - j["ctl"]).to_numpy(dtype=float)
        sd = float(np.std(d, ddof=1))
        need = int(np.ceil((1.96 * sd / TARGET_HALF_WIDTH) ** 2)) if TARGET_HALF_WIDTH > 0 else 0
        out["per_arm"][arm] = {
            "n_pairs": int(len(j)),
            "paired_sd": sd,
            "half_width_now": float(1.96 * sd / np.sqrt(len(j))),
            "n_for_target": need,
        }
        sds.append(sd)
    if sds:
        sd = float(max(sds))  # size for the worst arm, not the average one
        out["binding_sd"] = sd
        out["n_for_target"] = int(np.ceil((1.96 * sd / TARGET_HALF_WIDTH) ** 2)) if TARGET_HALF_WIDTH > 0 else 0
    return out


def mode_did(traj: pd.DataFrame) -> dict:
    """``(arm - prod | region) - (arm - prod | binary)``, paired within a cell.

    This is the mode question with the embedder taken out of it.  A patch cell
    runs ``whole_image`` and ``max_patch`` in one task off one loaded pickle, so
    for a given ``(embedder, category, seed)`` the two styles share the sim/test
    split and the startup exemplar and differ *only* in the scoring geometry.
    The difference of the two arm effects is therefore attributable to the mode.

    Restricted to embedders that appear in **both** modes.  #3115's per-mode
    headline came off a grid whose binary cells were all SigLIP and whose region
    cells were all DINOv3: the effect was real and its attribution was not, and
    check 13b exists because nothing had asserted otherwise.  Returning ``{}``
    when no embedder spans the modes is the honest answer to that grid.
    """
    if "mode" not in traj.columns or traj["mode"].nunique() < 2:
        return {}
    spanning = sorted(
        {emb for emb in traj["embedder"].unique() if traj[traj["embedder"] == emb]["mode"].nunique() >= 2}
    )
    if not spanning:
        return {"embedders": [], "note": "no embedder runs in both modes; a DiD here would be a per-embedder contrast"}

    t = traj[traj["embedder"].isin(spanning)]
    keys = [k for k in ("dataset", "embedder", "category", "seed") if k in t.columns]
    out: dict = {"embedders": spanning, "contrasts": {}}
    rng = np.random.default_rng(2877)
    for metric in ("final_cost", "final_ap", "positives_100", "final_oracle_cost"):
        if metric not in t.columns:
            continue
        per_metric = {}
        for arm in ARMS:
            if arm == CONTROL:
                continue
            frames = {}
            for mode in ("binary", "region"):
                m = t[t["mode"] == mode]
                a = m[m["arm"] == CONTROL].set_index(keys)[metric]
                b = m[m["arm"] == arm].set_index(keys)[metric]
                frames[mode] = (b - a).dropna()
            j = pd.concat([frames["binary"].rename("binary"), frames["region"].rename("region")], axis=1).dropna()
            if len(j) < 2:
                continue
            d = (j["region"] - j["binary"]).to_numpy(dtype=float)
            boot = np.array([np.mean(rng.choice(d, size=len(d), replace=True)) for _ in range(4000)])
            rec = {
                "n_pairs": int(len(j)),
                "binary_mean_delta": float(j["binary"].mean()),
                "region_mean_delta": float(j["region"].mean()),
                "did": float(np.mean(d)),
                "ci95_lo": float(np.percentile(boot, 2.5)),
                "ci95_hi": float(np.percentile(boot, 97.5)),
            }
            if _wilcoxon is not None and np.any(d != 0):
                try:
                    rec["p"] = float(_wilcoxon(j["region"], j["binary"]).pvalue)
                except Exception:  # noqa: BLE001
                    pass
            per_metric[arm] = rec
        if per_metric:
            out["contrasts"][metric] = per_metric
    return out


def build_summary(traj: pd.DataFrame, prov: dict) -> dict:
    """The pooled summary, plus one of the same shape per voting mode.

    The pooled numbers stay at the top level so every earlier caller and the
    self-test read what they always read - but for a grid holding both modes
    they are **descriptive**, not the verdict.  The whole reason this question
    survived three runs is that the answer moved between environments, and a
    mean over a grid spanning them is the number that hides it.
    """
    summary = {
        "config": {
            "cost_regression_tolerance": COST_REGRESSION_TOLERANCE,
            "alpha": ALPHA,
            "warm_t": sp.WARM_T,
            "deep_cost": sp.DEEP_COST,
            "deep_excess": sp.DEEP_EXCESS,
            "target_half_width": TARGET_HALF_WIDTH,
        },
        "provenance": prov,
        **_core_summary(traj),
    }
    summary["sizing"] = sizing(traj)

    modes = sorted(traj["mode"].dropna().unique()) if "mode" in traj.columns else []
    summary["modes_present"] = list(modes)
    if len(modes) > 1:
        summary["by_mode"] = {}
        for mode in modes:
            sub = traj[traj["mode"] == mode]
            summary["by_mode"][mode] = {**_core_summary(sub), "sizing": sizing(sub)}
        summary["mode_did"] = mode_did(traj)
        # The pooled verdict is not a verdict here.  Stated in the object rather
        # than only in the prose, so a reader of the JSON cannot take
        # `adopt` for an answer without meeting this key.
        summary["pooled_is_descriptive"] = True
    return summary


def make_figures(traj: pd.DataFrame, summary: dict, outdir: Path, prefix: str = "") -> list[str]:
    """The three figures, optionally name-spaced by *prefix*.

    A grid holding both voting modes draws them once per mode: the frontier's
    whole point is its SHAPE, and one curve averaged over two modes is a curve
    of neither.  *prefix* is what keeps the two sets from overwriting each other
    while the drawing code stays single-sourced.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    dpi = int(os.environ.get("ACQ_FIG_DPI", "200"))
    order = [a for a in ARMS if a in summary["per_arm"]]

    # Fig 1: the frontier - the deliverable. Positives against cost, per arm.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ks = [summary["per_arm"][a]["k_acq"] for a in order if summary["per_arm"][a]["k_acq"] is not None]
    karms = [a for a in order if summary["per_arm"][a]["k_acq"] is not None]

    axes[0].plot(ks, [summary["per_arm"][a]["median_positives_100"] for a in karms], "o-", color="#1f6f78")
    axes[0].set_xlabel("acquisition inclusion $k$"), axes[0].set_ylabel("median positives found by t=100")
    axes[0].set_title("mechanism: does the lever pull?", fontsize=9)
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.25, ls=":")

    axes[1].plot(ks, [summary["per_arm"][a]["median_final_cost"] for a in karms], "o-", color="#b0402e")
    axes[1].set_xlabel("acquisition inclusion $k$"), axes[1].set_ylabel("median final cost (FPR+FNR)")
    axes[1].set_title("decision: is it free?", fontsize=9)
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.25, ls=":")

    for a in order:
        x = summary["per_arm"][a]["median_positives_100"]
        y = summary["per_arm"][a]["median_final_cost"]
        m = "s" if a == "rank_pin" else ("D" if a == CONTROL else "o")
        axes[2].scatter(x, y, s=70, marker=m, zorder=4)
        axes[2].annotate(ARM_LABEL.get(a, a), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[2].set_xlabel("median positives found"), axes[2].set_ylabel("median final cost")
    axes[2].set_title("the frontier (down-and-right is better)", fontsize=9)
    axes[2].grid(alpha=0.25, ls=":")

    fig.tight_layout()
    p = outdir / f"{prefix}fig1_frontier.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    made.append(p.name)

    # Fig 2: the verification panel - where each arm actually sampled.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    pcts = [summary["lever_verification"][a]["median_acq_pool_percentile"] for a in order]
    ax.bar(range(len(order)), pcts, color="#4c72b0")
    ax.axhline(
        summary["lever_verification"][CONTROL]["median_acq_pool_percentile"],
        color="#b0402e",
        ls="--",
        lw=1.2,
        label="production (control)",
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ARM_LABEL.get(a, a) for a in order], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median acq_pool_percentile")
    ax.set_title("verification: where in the ranking each arm actually sampled", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y", ls=":")
    fig.tight_layout()
    p = outdir / f"{prefix}fig2_lever_verification.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    made.append(p.name)

    # Fig 3: guardrails.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(range(len(order)), [100 * summary["per_arm"][a]["genuine_blip_rate"] for a in order], color="#c44e52")
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([ARM_LABEL.get(a, a) for a in order], rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("% runs with a genuine threshold blip")
    axes[0].set_title("guardrail: #2847's fix must survive", fontsize=9)
    axes[0].grid(alpha=0.25, axis="y", ls=":")

    data = [traj[traj["arm"] == a]["final_cost"].dropna().to_numpy() for a in order]
    axes[1].boxplot(data, tick_labels=[ARM_LABEL.get(a, a) for a in order], showfliers=False)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].set_ylabel("final cost per trajectory")
    axes[1].set_title("decision endpoint, full distribution", fontsize=9)
    axes[1].grid(alpha=0.25, axis="y", ls=":")
    fig.tight_layout()
    p = outdir / f"{prefix}fig3_guardrails.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    made.append(p.name)
    return made


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _mode_sections(A, s: dict, heading: str, note: str = "") -> None:
    """The lever / per-arm / contrast / ship-rule block for ONE summary.

    Factored out of :func:`write_report` so a per-mode section and the pooled one
    are literally the same tables — a reader comparing them is comparing
    numbers, not two renderings that might differ.
    """
    A(f"\n{heading}\n")
    if note:
        A(f"{note}\n")

    if not s["falsifier_behaved"]:
        A(
            "> **VERDICT WITHHELD.** The falsification arm `acq_p2` (k=+2) did not "
            "significantly *reduce* positives found. The mechanism this run assumes — "
            "that the acquisition cut's rank position drives positive yield — is "
            "therefore unsupported, and the remaining arms cannot be read as evidence "
            "for it. Everything below is descriptive.\n"
        )

    bad_levers = [a for a, v in s["lever_verification"].items() if not v["moved"]]
    if bad_levers:
        A(f"> **Arms whose sampling position never moved: {', '.join(bad_levers)}.** These measured nothing.\n")

    A("\n### Lever verification — where each arm actually sampled\n")
    A("| arm | median `acq_pool_percentile` | shift vs control | steps where acq ≠ reporting |")
    A("|---|---:|---:|---:|")
    for a in ARMS:
        v = s["lever_verification"].get(a)
        if not v:
            continue
        A(
            f"| `{a}` — {ARM_LABEL.get(a, a)} | {_f(v['median_acq_pool_percentile'], 4)} | "
            f"{v['shift_vs_control']:+.4f} | {100 * v['frac_steps_acq_differs']:.0f}% |"
        )

    A("\n### Per-arm\n")
    A(
        "| arm | trajectories | positives @100 | positives @50 | final cost | mean warm cost | "
        "final AP | oracle cost | genuine blips |"
    )
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in ARMS:
        v = s["per_arm"].get(a)
        if not v:
            continue
        A(
            f"| `{a}` — {ARM_LABEL.get(a, a)} | {v['n_trajectories']} | "
            f"{_f(v['median_positives_100'], 1)} | {_f(v['median_positives_50'], 1)} | "
            f"{_f(v['median_final_cost'])} | {_f(v['median_mean_cost_warm'])} | "
            f"{_f(v['median_final_ap'])} | {_f(v['median_final_oracle_cost'])} | "
            f"{100 * v['genuine_blip_rate']:.1f}% |"
        )

    A(f"\n### Paired against `{CONTROL}` — cells are `(category, seed, style)`, never steps\n")
    A("| arm | metric | n | control | arm | median Δ | 95% CI on mean Δ | p |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        c = s["contrasts_vs_control"].get(arm)
        if not c:
            continue
        for metric in ("positives_100", "final_cost", "mean_cost_warm", "final_oracle_cost", "final_ap"):
            r = c[metric]
            if not r.get("n_pairs"):
                continue
            A(
                f"| `{arm}` | {metric} | {r['n_pairs']} | {_f(r['control_median'])} | {_f(r['arm_median'])} | "
                f"{r['median_delta']:+.3f} | [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {_f(r.get('p'), 5)} |"
            )

    A("\n### Ship rule (pre-registered)\n")
    A(
        f"Adopt iff positives rise (p<{ALPHA}) **and** the 95% upper bound on the "
        f"final-cost delta is below +{COST_REGRESSION_TOLERANCE} **and** deep-spike "
        "incidence does not rise **and** the lever actually moved.\n"
    )
    A("| arm | positives rose | cost did not regress | spikes did not rise | lever moved | **ADOPT** |")
    A("|---|:--:|:--:|:--:|:--:|:--:|")
    for arm, v in s["ship_rule"].items():
        y = lambda b: "yes" if b else "no"  # noqa: E731
        A(
            f"| `{arm}` | {y(v['positives_rose'])} | {y(v['cost_did_not_regress'])} | "
            f"{y(v['spikes_did_not_rise'])} | {y(v['lever_moved'])} | **{y(v['ADOPT'])}** |"
        )
    A(f"\n**Arms passing every criterion:** {', '.join(s['adopt']) if s['adopt'] else '_none_'}\n")

    sz = s.get("sizing") or {}
    if sz.get("per_arm"):
        A("\n### Power on the decision endpoint\n")
        A(
            "The realized paired SD on `final_cost`, and the `n` a "
            f"±{sz['target_half_width']:g} half-width would need. A CI wider than the "
            "tolerance is not a null — it is an interval that can contain both "
            "shipping decisions, which is what #2877 reported as a null.\n"
        )
        A("| arm | pairs | paired SD | current half-width | n for target |")
        A("|---|---:|---:|---:|---:|")
        for arm, v in sz["per_arm"].items():
            A(
                f"| `{arm}` | {v['n_pairs']} | {_f(v['paired_sd'], 4)} | "
                f"±{v['half_width_now']:.4f} | {v['n_for_target']} |"
            )
        if "n_for_target" in sz:
            A(
                f"\n**Binding: n ≈ {sz['n_for_target']} pairs** for ±{sz['target_half_width']:g} "
                f"(worst arm's SD {_f(sz.get('binding_sd'), 4)}).\n"
            )

    for fig in s.get("figures") or []:
        A(f"\n![{fig}](figures/{fig})\n")


def write_report(summary: dict, figs: list[str], outdir: Path) -> Path:
    L: list[str] = []
    A = L.append
    A("# Acquisition/reporting threshold decoupling — does it buy back the positives?\n")
    A(
        "Design: `docs/ML.md` (threshold calibration). Reporting is cut at "
        "inclusion 0 in **every** arm; only the selector's cut moves.\n"
    )

    by_mode = summary.get("by_mode") or {}
    if by_mode:
        A(
            "\n> **This grid holds both voting modes, so the verdict is per mode.** The "
            "acquisition offset has now been measured in four environments and the "
            "answer has moved between them — which is exactly what a mean over a grid "
            "spanning them would hide. The pooled tables are printed last and are "
            "**descriptive only**.\n"
        )
        A("\n## Verdict at a glance\n")
        A("| voting mode | trajectories | arms adopted | falsifier behaved |")
        A("|---|---:|---|:--:|")
        for mode, s in by_mode.items():
            A(
                f"| **{mode}** | {s['n_trajectories']} | "
                f"{', '.join(f'`{a}`' for a in s['adopt']) if s['adopt'] else '_none_'} | "
                f"{'yes' if s['falsifier_behaved'] else '**no**'} |"
            )
        for mode, s in by_mode.items():
            _mode_sections(A, s, f"## Voting mode: {mode}")

        did = summary.get("mode_did") or {}
        if did.get("contrasts"):
            A("\n## Is the offset actually mode-dependent? (difference-in-differences)\n")
            A(
                "`(arm − prod | region) − (arm − prod | binary)`, paired cell-for-cell "
                f"within **{', '.join(f'`{e}`' for e in did['embedders'])}** — one embedder "
                "running both styles inside one task, off one loaded pickle, on one "
                "sim/test split and one exemplar. The two modes therefore differ only in "
                "the scoring geometry, so the difference of the arm effects is "
                "attributable to the mode and not to the embedder (#3115 reported a "
                "per-mode headline off a grid where those two moved together).\n"
            )
            A("| metric | arm | n | Δ binary | Δ region | DiD | 95% CI | p |")
            A("|---|---|---:|---:|---:|---:|---:|---:|")
            for metric, per_arm in did["contrasts"].items():
                for arm, r in per_arm.items():
                    A(
                        f"| {metric} | `{arm}` | {r['n_pairs']} | {r['binary_mean_delta']:+.3f} | "
                        f"{r['region_mean_delta']:+.3f} | {r['did']:+.4f} | "
                        f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}] | {_f(r.get('p'), 5)} |"
                    )
            A(
                "\nA DiD whose CI covers 0 says the offset behaves the same way in both "
                "modes **in this environment** — which is a result about gating, not a "
                "failure to find one.\n"
            )
        elif did.get("note"):
            A(f"\n## Mode contrast\n\n> Not computed: {did['note']}.\n")

        _mode_sections(
            A,
            summary,
            "## Pooled across modes — descriptive only",
            "> Not a verdict. Kept because it is the shape every earlier "
            "environment's report had, and so the four runs stay comparable.",
        )
    else:
        _mode_sections(A, summary, "## Result")

    A("\n## Data read\n")
    A("| arm | cell files | trajectories | never found a positive | unreadable | zero-byte |")
    A("|---|---:|---:|---:|---:|---:|")
    for arm, v in summary["provenance"].items():
        A(
            f"| `{arm}` | {v.get('n_files', 0)} | {v.get('n_read', 0)} | "
            f"{len(v.get('no_positive_found', []))} | {len(v.get('unreadable', []))} | "
            f"{len(v.get('zero_byte', []))} |"
        )

    # Only figures no section already showed.  Each mode renders its own inside
    # its section (a frontier is a shape, and one a reader has to scroll to a
    # shared gallery to find is a table), so a trailing gallery that repeats
    # them would print every figure twice.
    shown = set(summary.get("figures") or [])
    for s_mode in (summary.get("by_mode") or {}).values():
        shown |= set(s_mode.get("figures") or [])
    extra = [f for f in figs if f not in shown]
    if extra:
        A("\n## Figures\n")
        for f in extra:
            A(f"![{f}](figures/{f})\n")

    out = outdir / "REPORT_acq.md"
    out.write_text("\n".join(L) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Acquisition-inclusion decoupling analysis.")
    # Two layouts, because two studies wrote them.  `--results-root` is the flat
    # `<root>/<arm>` every earlier environment used and is still the default;
    # `--base` + `--halves` is the #2877 pile re-run's `<base>/<half>/<arm>`,
    # which exists because a region cell and a whole-image cell cannot share a
    # memory request and therefore cannot share an array.
    ap.add_argument("--results-root", default=str(common.EXP / "results"))
    ap.add_argument("--base", default=None, help="study root holding <half>/<arm>/results")
    ap.add_argument("--halves", default="bin,reg", help="comma-separated halves under --base")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    outdir = Path(args.out)
    (outdir / "agg").mkdir(parents=True, exist_ok=True)

    if args.base:
        halves = [h.strip() for h in args.halves.split(",") if h.strip()]
        source = f"{args.base} (halves: {', '.join(halves)})"
        df, prov = load_halves(Path(args.base), halves)
    else:
        source = args.results_root
        df, prov = load_all(Path(args.results_root))
    if df.empty:
        print(f"no cells under {source}")
        return 1
    print(f"loaded {len(df)} base rows across {df['arm'].nunique()} arms from {source}")

    traj = trajectory_stats(df)
    traj.to_csv(outdir / "agg" / "trajectories.csv", index=False)
    summary = build_summary(traj, prov)

    # Figures per mode, then pooled.  Attached to the summary each one belongs
    # to so the report renders a mode's frontier inside that mode's section --
    # a frontier is a shape, and a reader who has to scroll to a shared gallery
    # to find it is reading the table instead.
    figdir = outdir / "figures"
    figs: list[str] = []
    for mode, sub in (summary.get("by_mode") or {}).items():
        made = make_figures(traj[traj["mode"] == mode], sub, figdir, prefix=f"{mode}_")
        sub["figures"] = made
        figs.extend(made)
    pooled = make_figures(traj, summary, figdir)
    summary["figures"] = pooled
    figs.extend(pooled)

    (outdir / "acq_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    rep = write_report(summary, figs, outdir)

    print(f"wrote {rep}")
    for label, s in [("pooled", summary), *((m, v) for m, v in (summary.get("by_mode") or {}).items())]:
        print(f"--- {label} ({s['n_trajectories']} trajectories)")
        for a in ARMS:
            v = s["per_arm"].get(a)
            if not v:
                continue
            print(
                f"  {a:9s} pos@100={v['median_positives_100']:5.1f}  final_cost={v['median_final_cost']:.3f}  "
                f"genuine_blips={100 * v['genuine_blip_rate']:5.1f}%  "
                f"acq_pct={s['lever_verification'][a]['median_acq_pool_percentile']:.4f}"
            )
        print(f"  falsifier behaved: {s['falsifier_behaved']}")
        print(f"  ADOPT: {s['adopt'] or 'none'}")
        sz = s.get("sizing") or {}
        if "n_for_target" in sz:
            print(
                f"  sizing: n\u2248{sz['n_for_target']} pairs for \u00b1{sz['target_half_width']:g} (SD {sz['binding_sd']:.4f})"
            )
    if summary.get("by_mode"):
        print("  NOTE: the pooled block above is descriptive; the verdict is per mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

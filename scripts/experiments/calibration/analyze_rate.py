"""Stage 2 (#2861): where is the anchor-mass optimum, and does it move?

The #2852/#2860 run swept anchor mass ``kappa`` over {1,3,10,30,100} and found
the best arm at ``kappa=1`` - the **bottom edge of the grid**, which is not an
optimum, only a boundary.  It also ran in exactly two environments (Visual
Genome x {siglip whole-image, dinov3 max-patch}), so nothing was known about
whether the answer travels.

This analyzer consumes a run that extends the grid two decades down
(0.01 .. 3) across six environments and answers three questions:

1. **Is the optimum interior, and where?**  Per (environment, family, rule,
   window) it plots the paired regret-vs-``xcal_only`` curve against kappa and
   locates the argmin.
2. **How sharp is it?**  A point argmin over 8 grid values is noise-prone, so
   the headline is a **plateau**: every kappa whose cell-level paired
   difference from the argmin is not significant at 0.05.  A wide plateau is
   itself the finding ("the knob barely matters"), a narrow one makes the
   argmin meaningful.
3. **Does it move across environments?**  Per-environment argmins side by side,
   plus the parameterization test: anchoring's actual strength is
   ``gamma = kappa*n / (kappa*n + N)`` for a haystack of size N, so if kappa*
   were really about gamma it would have to scale with N.  The six
   environments span N from ~400 to ~2500, which is enough leverage to tell a
   constant-kappa world from a constant-gamma one.

Pairing unit throughout: every arm re-cuts the *same* per-step model against
the *same* held-out test scores, so arms are paired within a step by
construction.  Steps inside one trajectory are strongly autocorrelated, so all
significance is computed on **cell means** (one number per
environment x category x seed x window), never on raw steps - the step counts
are reported only to show coverage.

Writes ``results/agg/rate_*.csv``, ``results/rate_summary.json`` and
``results/REPORT_rate.md``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import experiment_config as cfg  # noqa: E402

try:  # scipy is in the grid venv, but the analyzer must not die without it
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # noqa: BLE001
    _wilcoxon = None

#: Conformal FN budget at inclusion 0 - the H4 envelope.
FN_BUDGET = 0.25

#: ``fold_anchored_w{W}_{rule}_{combine}`` / ``anchored_w{W}_{rule}``.
FOLD_RE = re.compile(r"^fold_anchored_w(?P<w>[\d.]+)_(?P<rule>mid|rate)_(?P<combine>\w+)$")
LABEL_RE = re.compile(r"^anchored_w(?P<w>[\d.]+)_(?P<rule>mid|rate)$")

#: Non-swept arms kept in the frame as reference points.
CONTROLS = ("xcal_only", "pooled_mid", "rank_transfer", "pooled_sim_oracle", "pooled_rate", "pooled_priorfree")

#: Deep regime = every window whose upper edge is at least this, i.e.
#: le_100 / le_200 / le_300 = votes 51-300.  Kept identical to the #2860
#: analyzer's ``DEEP_WINDOWS_MIN`` so the two runs' headline numbers mean the
#: same thing.
DEEP_MIN = 100


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:  # noqa: BLE001 - tabulate missing
        return "```\n" + df.to_string(index=False) + "\n```"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_cells(cells_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in cells_dir.glob("task_*.csv") if "__sweep" not in p.name and "__cutdiag" not in p.name)
    frames = []
    for p in files:
        if p.stat().st_size == 0:
            continue
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:  # noqa: BLE001 - one truncated cell must not lose the run
            common.log(f"  skipping unreadable {p.name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["schedule"] = df["schedule"].fillna("")
    # A schedule row is the shipped blend under an alternative mix-in curve; it
    # is a legitimate comparison arm here, so give it a variant name.
    sched = df["gmm_variant"].eq("") & df["schedule"].ne("")
    df.loc[sched, "gmm_variant"] = "sched:" + df.loc[sched, "schedule"]
    df["env"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {len(df):,} rows from {len(frames)} cells ({len(files) - len(frames)} unreadable/empty)")
    return df


def assign_window(df: pd.DataFrame, checkpoints: list[int]) -> pd.DataFrame:
    edges = [1, *sorted(checkpoints)]
    labels = [f"le_{c}" for c in sorted(checkpoints)]
    df = df.copy()
    df["window"] = pd.cut(df["n_votes"], bins=edges, labels=labels)
    df["window_hi"] = pd.cut(df["n_votes"], bins=edges, labels=sorted(checkpoints)).astype("Int64")
    return df[df["window"].notna()]


def annotate_arms(df: pd.DataFrame) -> pd.DataFrame:
    """Split the swept arms into (family, kappa, rule); tag everything else."""
    fam, kap, rule = [], [], []
    for name in df["gmm_variant"]:
        m = FOLD_RE.match(name)
        if m:
            fam.append("fold_anchored")
            kap.append(float(m.group("w")))
            rule.append(m.group("rule"))
            continue
        m = LABEL_RE.match(name)
        if m:
            fam.append("label_anchored")
            kap.append(float(m.group("w")))
            rule.append(m.group("rule"))
            continue
        fam.append("other")
        kap.append(np.nan)
        rule.append("")
    out = df.copy()
    out["family"] = fam
    out["kappa"] = kap
    out["rule"] = rule
    return out


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

STEP_KEYS = ["env", "category", "seed", "t", "window"]
CELL_KEYS = ["env", "category", "seed", "window"]


def paired_vs_control(v: pd.DataFrame, control: str, metrics=("regret", "cost", "fnr")) -> pd.DataFrame:
    """Step-paired differences of every arm against *control*.

    Returns one row per (arm, env, category, seed, t, window) so downstream can
    aggregate to whatever unit it wants.  The join is an inner join on the step
    key, so a step where the control did not emit (a non-finite cut) drops out
    of every arm's comparison identically.
    """
    c = v[v["gmm_variant"] == control].set_index(STEP_KEYS)[list(metrics)]
    c = c[~c.index.duplicated()]
    out = []
    for name, a in v.groupby("gmm_variant", observed=True):
        if name == control:
            continue
        a = a.set_index(STEP_KEYS)
        a = a[~a.index.duplicated()]
        j = a[list(metrics)].join(c, how="inner", lsuffix="", rsuffix="_c")
        if j.empty:
            continue
        j = j.reset_index()
        for m in metrics:
            j[f"d_{m}"] = j[m] - j[f"{m}_c"]
        j["gmm_variant"] = name
        keep = [*STEP_KEYS, "gmm_variant", *[f"d_{m}" for m in metrics]]
        out.append(j[keep])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def to_cells(paired: pd.DataFrame) -> pd.DataFrame:
    """Collapse steps to one mean per (arm, env, category, seed, window).

    This is the unit every p-value in this report is computed on.  300 steps of
    a single trajectory are one experiment, not 300; treating them as
    independent is what makes a 1e-40 p-value out of a coin flip.
    """
    g = (
        paired.groupby(["gmm_variant", *CELL_KEYS], observed=True)
        .agg(d_regret=("d_regret", "mean"), d_cost=("d_cost", "mean"), d_fnr=("d_fnr", "mean"), n_steps=("t", "size"))
        .reset_index()
    )
    return g


def _wilcox_p(x: np.ndarray) -> float:
    """Two-sided paired-difference p-value; falls back to a sign test."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size < 6 or np.allclose(x, 0):
        return float("nan")
    if _wilcoxon is not None:
        try:
            return float(_wilcoxon(x, alternative="two-sided", zero_method="zsplit").pvalue)
        except Exception:  # noqa: BLE001
            pass
    pos = int((x > 0).sum())
    n = int((x != 0).sum())
    if n == 0:
        return float("nan")
    z = (pos - n / 2) / math.sqrt(n / 4)
    return float(math.erfc(abs(z) / math.sqrt(2)))


# --------------------------------------------------------------------------
# The kappa curve
# --------------------------------------------------------------------------


def rate_curve(cells: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Mean paired d_regret vs kappa, per (env, family, rule, window)."""
    c = cells.merge(arm_meta, on="gmm_variant", how="left")
    swept = c[c["family"].isin(["fold_anchored", "label_anchored"])]
    rows = []
    for (env, fam, rule, window), g in swept.groupby(["env", "family", "rule", "window"], observed=True):
        for kappa, k in g.groupby("kappa", observed=True):
            rows.append(
                {
                    "env": env,
                    "family": fam,
                    "rule": rule,
                    "window": str(window),
                    "kappa": float(kappa),
                    "d_regret": float(k["d_regret"].mean()),
                    "se": float(k["d_regret"].std(ddof=1) / math.sqrt(max(1, len(k)))),
                    # Step-weighted, i.e. the quantity the #2860 report printed;
                    # kept alongside the cell mean so the replicated kappa
                    # points can be compared to it like for like.
                    "d_regret_steps": float(np.average(k["d_regret"], weights=k["n_steps"])),
                    "d_cost": float(k["d_cost"].mean()),
                    "d_fnr": float(k["d_fnr"].mean()),
                    "n_cells": int(len(k)),
                    "n_steps": int(k["n_steps"].sum()),
                    "p_vs_zero": _wilcox_p(k["d_regret"].to_numpy()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["env", "family", "rule", "window", "kappa"])
    out.to_csv(agg / "rate_curve.csv", index=False)
    return out


def pooled_curve(cells: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path, deep_only: bool) -> pd.DataFrame:
    """The kappa curve pooled over environments (equal weight per cell)."""
    c = cells.merge(arm_meta, on="gmm_variant", how="left")
    c = c[c["family"].isin(["fold_anchored", "label_anchored"])]
    if deep_only:
        c = c[c["window"].astype(str).str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
    rows = []
    for (fam, rule), g in c.groupby(["family", "rule"], observed=True):
        for kappa, k in g.groupby("kappa", observed=True):
            rows.append(
                {
                    "family": fam,
                    "rule": rule,
                    "kappa": float(kappa),
                    "d_regret": float(k["d_regret"].mean()),
                    "se": float(k["d_regret"].std(ddof=1) / math.sqrt(max(1, len(k)))),
                    "d_regret_steps": float(np.average(k["d_regret"], weights=k["n_steps"])),
                    "d_cost": float(k["d_cost"].mean()),
                    "n_cells": int(len(k)),
                    "n_steps": int(k["n_steps"].sum()),
                    "p_vs_zero": _wilcox_p(k["d_regret"].to_numpy()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["family", "rule", "kappa"])
    out.to_csv(agg / ("rate_curve_pooled_deep.csv" if deep_only else "rate_curve_pooled_all.csv"), index=False)
    return out


def dataset_curve(cells: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Deep-regime kappa curve pooled within each dataset.

    Exists so the two Visual Genome environments can be pooled exactly the way
    the #2860 report pooled them - that table is the replication target for the
    kappa=1 and kappa=3 points this run re-measures.
    """
    c = cells.merge(arm_meta, on="gmm_variant", how="left")
    c = c[c["family"].isin(["fold_anchored", "label_anchored"])]
    c = c[c["window"].astype(str).str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
    c = c.assign(dataset=c["env"].str.split("/").str[0])
    rows = []
    for (ds, fam, rule, kappa), k in c.groupby(["dataset", "family", "rule", "kappa"], observed=True):
        rows.append(
            {
                "dataset": ds,
                "family": fam,
                "rule": rule,
                "kappa": float(kappa),
                "d_regret": float(k["d_regret"].mean()),
                "d_regret_steps": float(np.average(k["d_regret"], weights=k["n_steps"])),
                "n_cells": int(len(k)),
                "n_steps": int(k["n_steps"].sum()),
                "p_vs_zero": _wilcox_p(k["d_regret"].to_numpy()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["dataset", "family", "rule", "kappa"])
    out.to_csv(agg / "rate_curve_by_dataset.csv", index=False)
    return out


def plateau(cells: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Argmin kappa + the set of kappas statistically tied with it.

    Two kappas are compared **cell-paired against each other** (not each
    against its own control), which is the sharper test: the same trajectory
    scored two ways.  The plateau is every kappa with p >= 0.05 against the
    argmin, and it is the honest headline - with 8 grid points and correlated
    curves, a bare argmin over-reads the data.
    """
    c = cells.merge(arm_meta, on="gmm_variant", how="left")
    c = c[c["family"].isin(["fold_anchored", "label_anchored"])]
    rows = []
    groups: list[tuple[tuple, pd.DataFrame]] = []
    for key, g in c.groupby(["env", "family", "rule"], observed=True):
        groups.append((("per_env", *key), g))
        deep = g[g["window"].astype(str).str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
        if not deep.empty:
            groups.append((("per_env_deep", *key), deep))
    for key, g in c.groupby(["family", "rule"], observed=True):
        groups.append((("pooled", "ALL", *key), g))
        deep = g[g["window"].astype(str).str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
        if not deep.empty:
            groups.append((("pooled_deep", "ALL", *key), deep))

    for key, g in groups:
        scope, env, fam, rule = key
        means = g.groupby("kappa", observed=True)["d_regret"].mean()
        if means.empty:
            continue
        best_k = float(means.idxmin())
        wide = g.pivot_table(index=CELL_KEYS, columns="kappa", values="d_regret", observed=True)
        if best_k not in wide.columns:
            continue
        tied, pvals = [], {}
        for kappa in sorted(wide.columns):
            if kappa == best_k:
                tied.append(float(kappa))
                pvals[float(kappa)] = 1.0
                continue
            pair = wide[[kappa, best_k]].dropna()
            p = _wilcox_p((pair[kappa] - pair[best_k]).to_numpy())
            pvals[float(kappa)] = p
            if not np.isfinite(p) or p >= 0.05:
                tied.append(float(kappa))
        rows.append(
            {
                "scope": scope,
                "env": env,
                "family": fam,
                "rule": rule,
                "best_kappa": best_k,
                "best_d_regret": float(means.min()),
                "worst_d_regret": float(means.max()),
                "spread": float(means.max() - means.min()),
                "plateau": ",".join(f"{k:g}" for k in sorted(tied)),
                "plateau_lo": min(tied),
                "plateau_hi": max(tied),
                "n_cells": int(len(wide)),
                "p_by_kappa": json.dumps({f"{k:g}": (None if not np.isfinite(p) else round(p, 5)) for k, p in pvals.items()}),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(agg / "rate_plateau.csv", index=False)
    return out


# --------------------------------------------------------------------------
# Environment descriptors: the gamma test
# --------------------------------------------------------------------------


def env_table(df: pd.DataFrame, results: Path, agg: Path) -> pd.DataFrame:
    """One row per environment: what it is, how big its haystack is.

    ``n_fit`` is the size of the score population the mixture is actually
    fitted on - the N in ``gamma = kappa*n / (kappa*n + N)``.  Every style
    hands the estimator one score per sim-set media (``max_patch`` max-pools
    its patch grid first), so N is the sim-set size, ``n_medias *
    SIM_FRACTION``.  N is what would have to drag kappa* around if the thing
    being tuned were really the label *share* rather than the per-label mass.
    """
    medias: dict[tuple[str, str], int] = {}
    info_path = results / "prepare_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for ds, per_emb in info.get("datasets", {}).items():
            for emb, entry in per_emb.items():
                medias[(ds, emb)] = int(entry.get("n_medias", 0))
    rows = []
    for env, g in df.groupby("env", observed=True):
        ds, emb, style = env.split("/")
        n_med = medias.get((ds, emb), 0)
        rows.append(
            {
                "env": env,
                "dataset": ds,
                "embedder": emb,
                "style": style,
                "region_voting": bool(cfg.REGION_VOTING_BY_DATASET.get(ds, False)),
                "n_medias": n_med,
                "n_fit": round(n_med * cfg.SIM_FRACTION),
                "categories": int(g["category"].nunique()),
                "seeds": int(g["seed"].nunique()),
                "max_votes": int(g["n_votes"].max()),
                "median_prevalence": float(g["realized_prevalence"].median()),
            }
        )
    out = pd.DataFrame(rows).sort_values("env")
    out.to_csv(agg / "rate_environments.csv", index=False)
    return out


def gamma_test(plateaus: pd.DataFrame, envs: pd.DataFrame, agg: Path, n_ref: int = 150) -> pd.DataFrame:
    """Restate each environment's kappa* as the anchor share gamma it implies.

    If the estimator really wants a fixed *share* of the fit to come from
    labels, kappa* must fall as N rises (kappa* ~ N/n).  If instead kappa* is
    flat while gamma* swings with N, the per-anchor mass is the invariant and
    the tuned constant transfers as-is.
    """
    p = plateaus[(plateaus["scope"] == "per_env_deep") & (plateaus["family"] == "fold_anchored")]
    j = p.merge(envs[["env", "n_fit", "dataset", "embedder", "region_voting"]], on="env", how="left")
    j["gamma_at_ref"] = j["best_kappa"] * n_ref / (j["best_kappa"] * n_ref + j["n_fit"])
    out = j[
        [
            "env",
            "rule",
            "dataset",
            "embedder",
            "region_voting",
            "n_fit",
            "best_kappa",
            "plateau",
            "best_d_regret",
            "gamma_at_ref",
        ]
    ].sort_values(["rule", "n_fit"])
    out.to_csv(agg / "rate_gamma_test.csv", index=False)
    return out


# --------------------------------------------------------------------------
# Supporting tables
# --------------------------------------------------------------------------


def controls_table(cells: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Every non-swept arm's deep paired d_regret vs xcal, for context."""
    deep = cells[cells["window"].astype(str).str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
    rows = []
    for (name, env), g in deep.groupby(["gmm_variant", "env"], observed=True):
        if FOLD_RE.match(name) or LABEL_RE.match(name):
            continue
        rows.append(
            {
                "arm": name,
                "env": env,
                "d_regret": float(g["d_regret"].mean()),
                "d_cost": float(g["d_cost"].mean()),
                "n_cells": int(len(g)),
                "p": _wilcox_p(g["d_regret"].to_numpy()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["env", "d_regret"])
    out.to_csv(agg / "rate_controls.csv", index=False)
    return out


def kappa_by_window(cells: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Argmin kappa per vote window, pooled over environments.

    Worth a table of its own because the theory predicts movement: the labels'
    share of the fit is ``gamma = kappa*n/(kappa*n+N)``, so at a fixed kappa the
    anchors have almost no authority when n is small.  If the shallow windows
    prefer a *larger* kappa than the deep ones, a single constant kappa is a
    compromise between regimes rather than one right answer.
    """
    c = cells.merge(arm_meta, on="gmm_variant", how="left")
    c = c[c["family"].isin(["fold_anchored", "label_anchored"])]
    rows = []
    for (fam, rule, window), g in c.groupby(["family", "rule", "window"], observed=True):
        means = g.groupby("kappa", observed=True)["d_regret"].mean()
        if means.empty:
            continue
        rows.append(
            {
                "family": fam,
                "rule": rule,
                "window": str(window),
                "best_kappa": float(means.idxmin()),
                "best_d_regret": float(means.min()),
                "d_regret_at_1": float(means.get(1.0, np.nan)),
                "spread": float(means.max() - means.min()),
                "n_cells": int(g.groupby(CELL_KEYS, observed=True).ngroups),
            }
        )
    out = pd.DataFrame(rows).sort_values(["family", "rule", "window"])
    out.to_csv(agg / "rate_kappa_by_window.csv", index=False)
    return out


def vs_shipped_schedule(v: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Deep paired contrast against **today's shipped blend**, per environment.

    #2860 could only compare fusion to the 6->20 ramp, because PR #2849's
    per-mode schedules (`slow_cap50` for region voting, `cap50` for binary)
    merged after its run base.  This run scores both as counterfactual rows on
    the same trajectory, so the comparison the earlier report had to leave open
    is a plain paired difference here.

    Caveat carried from #2841: a counterfactual schedule row re-cuts *this*
    trajectory, so it cannot show what different acquisition would have done.
    It bounds the threshold-rule difference, not the whole-system difference.
    """
    rows = []
    for env, g in v.groupby("env", observed=True):
        ds = env.split("/")[0]
        control = "sched:slow_cap50" if cfg.REGION_VOTING_BY_DATASET.get(ds, False) else "sched:cap50"
        if control not in set(g["gmm_variant"]):
            continue
        paired = paired_vs_control(g, control)
        if paired.empty:
            continue
        cells = to_cells(paired)
        cells = cells[cells["window"].astype(str).str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
        for name, k in cells.groupby("gmm_variant", observed=True):
            rows.append(
                {
                    "env": env,
                    "control": control,
                    "arm": name,
                    "d_regret": float(k["d_regret"].mean()),
                    "n_cells": int(len(k)),
                    "p": _wilcox_p(k["d_regret"].to_numpy()),
                }
            )
    if not rows:
        out = pd.DataFrame(columns=["env", "control", "arm", "d_regret", "n_cells", "p", "family", "kappa", "rule"])
        out.to_csv(agg / "rate_vs_shipped_schedule.csv", index=False)
        return out
    out = pd.DataFrame(rows).merge(arm_meta.rename(columns={"gmm_variant": "arm"}), on="arm", how="left")
    out = out.sort_values(["env", "d_regret"])
    out.to_csv(agg / "rate_vs_shipped_schedule.csv", index=False)
    return out


def stability_by_kappa(v: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """H3 as a function of kappa: step-to-step |delta threshold| past 20 votes."""
    keys = ["env", "category", "seed", "gmm_variant"]
    w = v[v["n_votes"] > 20].sort_values([*keys, "t"])
    w = w.assign(d_thr=w.groupby(keys, observed=True)["threshold"].diff().abs())
    g = (
        w.dropna(subset=["d_thr"])
        .groupby(["env", "gmm_variant"], observed=True)
        .agg(mean_abs_dthr=("d_thr", "mean"), n=("d_thr", "size"))
        .reset_index()
        .merge(arm_meta, on="gmm_variant", how="left")
    )
    g.to_csv(agg / "rate_stability.csv", index=False)
    return g


def fnr_by_kappa(v: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """H4 as a function of kappa: realized FNR per window vs the 0.25 budget."""
    g = (
        v.groupby(["env", "gmm_variant", "window"], observed=True)
        .agg(fnr=("fnr", "mean"), fpr=("fpr", "mean"), cost=("cost", "mean"), n=("fnr", "size"))
        .reset_index()
        .merge(arm_meta, on="gmm_variant", how="left")
    )
    g["over_budget"] = g["fnr"] > FN_BUDGET
    g.to_csv(agg / "rate_fnr.csv", index=False)
    return g


def provenance_by_kappa(v: pd.DataFrame, arm_meta: pd.DataFrame, agg: Path) -> pd.DataFrame:
    """Does a tiny anchor mass silently turn the anchored fit into a plain one?"""
    s = v[v["gmm_variant"].str.startswith(("anchored_w", "fold_anchored_w"))].copy()
    s["path"] = np.where(
        s["threshold_provenance"].astype(str).str.contains("unanchored|gmm_failed", regex=True), "fallback", "anchored"
    )
    g = (
        s.groupby(["gmm_variant", "path"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .merge(arm_meta, on="gmm_variant", how="left")
    )
    for col in ("anchored", "fallback"):
        if col not in g.columns:
            g[col] = 0
    g["fallback_rate"] = g["fallback"] / (g["anchored"] + g["fallback"]).clip(lower=1)
    g = g.groupby(["family", "rule", "kappa"], observed=True)["fallback_rate"].mean().reset_index()
    g.to_csv(agg / "rate_provenance.csv", index=False)
    return g


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(results: Path, parts: dict) -> None:
    envs = parts["envs"]
    lines = [
        "# Anchor-mass (kappa) boundary sweep + environment generalization (#2861)",
        "",
        "Auto-generated by `analyze_rate.py`.  Every arm re-cuts the same per-step",
        "model against the same held-out test scores, so arms are paired within a",
        "step by construction; every p-value below is a paired Wilcoxon over **cell",
        "means** (one value per environment x category x seed x window), never over",
        "raw steps.",
        "",
        "## Environments",
        "",
        _md(envs, ".3f"),
        "",
        "## Pooled kappa curve, deep regime (51-300 votes), paired vs `xcal_only`",
        "",
        _md(parts["pooled_deep"]),
        "",
        "## Pooled kappa curve, all windows",
        "",
        _md(parts["pooled_all"]),
        "",
        "## Deep kappa curve pooled within each dataset",
        "",
        "`d_regret_steps` is the step-weighted mean - the quantity the #2860",
        "report printed, so its VG rows at kappa 1 and 3 are the replication check.",
        "",
        _md(parts["by_dataset"]),
        "",
        "## Optimum + plateau",
        "",
        "`plateau` = every kappa not significantly worse than the argmin when the",
        "two are compared cell-paired against each other (p >= 0.05).",
        "",
        _md(parts["plateau"].drop(columns=["p_by_kappa"])),
        "",
        "## Is kappa* or gamma* the invariant?",
        "",
        "`gamma_at_ref` is the share of the mixture fit that comes from labels at",
        "150 votes, `kappa*n/(kappa*n+N)`.  A constant kappa* across a 5x span of N",
        "means the per-anchor mass transfers and the share does not.",
        "",
        _md(parts["gamma"]),
        "",
        "## Per-environment kappa curves (deep regime, 51-300 votes)",
        "",
        _md(parts["curve_deep"]),
        "",
        "## Reference arms (deep, paired vs `xcal_only`)",
        "",
        _md(parts["controls"]),
        "",
        "## Argmin kappa by vote window (pooled over environments)",
        "",
        _md(parts["by_window"]),
        "",
        "## Deep paired contrast vs **today's shipped blend** (`slow_cap50` region / `cap50` binary)",
        "",
        _md(parts["shipped"]),
        "",
        "## Threshold stability by kappa (|delta threshold| per step, votes > 20)",
        "",
        _md(parts["stability"]),
        "",
        "## FNR vs the 0.25 conformal budget",
        "",
        _md(parts["fnr"]),
        "",
        "## Anchored-fit fallback rate by kappa",
        "",
        _md(parts["provenance"]),
        "",
    ]
    (results / "REPORT_rate.md").write_text("\n".join(lines))


def main() -> int:
    results = common.RESULTS
    agg = results / "agg"
    agg.mkdir(parents=True, exist_ok=True)

    df = load_cells(results / "cells")
    if df.empty:
        common.log("no cells found; nothing to analyze")
        return 1
    df = assign_window(df, cfg.ANCHORED_CHECKPOINTS)
    v = df[df["gmm_variant"] != ""].copy()
    v = annotate_arms(v)
    arm_meta = v[["gmm_variant", "family", "kappa", "rule"]].drop_duplicates("gmm_variant")

    envs = env_table(v, results, agg)
    common.log(f"{len(envs)} environments, {v['category'].nunique()} categories, {len(v):,} arm rows")

    paired = paired_vs_control(v, "xcal_only")
    if paired.empty:
        common.log("no paired rows against xcal_only; nothing to analyze")
        return 1
    cells = to_cells(paired)
    cells.to_csv(agg / "rate_cell_means.csv", index=False)
    common.log(f"{len(cells):,} arm-cell means")

    curve = rate_curve(cells, arm_meta, agg)
    curve_deep = curve[curve["window"].str.replace("le_", "", regex=False).astype(int) >= DEEP_MIN]
    pooled_deep = pooled_curve(cells, arm_meta, agg, deep_only=True)
    pooled_all = pooled_curve(cells, arm_meta, agg, deep_only=False)
    by_dataset = dataset_curve(cells, arm_meta, agg)
    plats = plateau(cells, arm_meta, agg)
    gamma = gamma_test(plats, envs, agg)
    controls = controls_table(cells, agg)
    by_window = kappa_by_window(cells, arm_meta, agg)
    shipped = vs_shipped_schedule(v, arm_meta, agg)
    stability = stability_by_kappa(v, arm_meta, agg)
    fnr = fnr_by_kappa(v, arm_meta, agg)
    prov = provenance_by_kappa(v, arm_meta, agg)

    head = pooled_deep[pooled_deep["family"] == "fold_anchored"]
    summary = {
        "n_cells_files": int(v.groupby(["env", "category", "seed"], observed=True).ngroups),
        "environments": envs.to_dict("records"),
        "pooled_deep_best": (
            head.loc[head["d_regret"].idxmin()].to_dict() if not head.empty else None
        ),
        "plateau_pooled_deep": plats[plats["scope"] == "pooled_deep"].drop(columns=["p_by_kappa"]).to_dict("records"),
        "per_env_deep_argmin": plats[plats["scope"] == "per_env_deep"]
        .drop(columns=["p_by_kappa"])
        .to_dict("records"),
    }
    (results / "rate_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    write_report(
        results,
        {
            "envs": envs,
            "pooled_deep": pooled_deep,
            "pooled_all": pooled_all,
            "by_dataset": by_dataset,
            "plateau": plats,
            "gamma": gamma,
            "curve_deep": curve_deep,
            "controls": controls,
            "by_window": by_window,
            "shipped": shipped[shipped["family"].isin(["fold_anchored", "label_anchored"]) | shipped["arm"].isin(["xcal_only", "rank_transfer", "sched:pure_gmm"])],
            "stability": stability[stability["family"].isin(["fold_anchored", "label_anchored"])]
            .groupby(["env", "family", "rule", "kappa"], observed=True)["mean_abs_dthr"]
            .mean()
            .reset_index(),
            "fnr": fnr[fnr["family"] == "fold_anchored"]
            .groupby(["window", "kappa", "rule"], observed=True)[["fnr", "fpr", "cost"]]
            .mean()
            .reset_index(),
            "provenance": prov,
        },
    )
    common.log(f"wrote {results / 'REPORT_rate.md'} and {results / 'rate_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

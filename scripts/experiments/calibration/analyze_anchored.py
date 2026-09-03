"""Stage 2 (anchored-mixture study, #2852): aggregate the anchored-arm cells.

Consumes the ``results/cells/task_*.csv`` files from a run with
``CALIB_SAFE_THRESHOLDS=1 CALIB_ANCHORED=1``, where every step emits - besides
the base blended row and the #2799 variant rows - one row per anchored arm:
``anchored_w{W}_{rule}`` (label-anchored mixture on the final model's haystack
scores), ``fold_anchored_w{W}_{rule}_{combine}`` (the cross-LabeledGMM repair:
per-fold anchored fits on honest held-out anchors, rank-transferred back), and
``rank_transfer`` (the scale-transfer-only attribution arm).  Controls ride in
the same frame: ``xcal_only`` (pure cross-cal) and ``pooled_mid`` (the shipped
safe-blend).

Computes the pre-registered deliverables of
``docs/plans/population-anchored-calibration.md``:

* Per-window (the plan's {20,50,100,200,300}-vote checkpoints) mean cost /
  regret / FPR / FNR per arm, and **paired** contrasts vs ``xcal_only`` (H1)
  and vs ``pooled_mid`` (H2) on identical (arm, category, seed, t) steps.
* Step-to-step threshold delta per arm (H3 - stability comes along free).
* FNR vs the conformal budget (0.25 at inclusion 0) per window (H4).
* Estimator-path tallies from ``threshold_provenance`` (how often the anchored
  EM ran vs fell back, and the fold-anchored per-fold tallies).

Writes ``results/summary.json``, ``results/agg/*.csv``, and a
``results/REPORT.md`` draft.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _cells_io import describe_load  # noqa: E402
from _cells_io import load_cells as _load_cells  # noqa: E402

import experiment_config as cfg  # noqa: E402

#: Conformal FN budget at inclusion 0 - the H4 envelope.
FN_BUDGET = 0.25

#: Arms whose H1/H2 verdicts the decision rules read (regexes over
#: ``gmm_variant``).  ``pooled_mid`` and ``xcal_only`` are the controls.
FUSION_ARM_RE = re.compile(r"^(anchored_|fold_anchored_|rank_transfer$)")

#: The deep-regime windows the plan keys its decisions on (votes > 100).
DEEP_WINDOWS_MIN = 100


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a fixed-width dump."""
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def load_cells(cells_dir: Path) -> pd.DataFrame:
    df, prov = _load_cells(cells_dir, where="analyze_anchored.py")
    if df.empty:
        return df
    df["gmm_variant"] = df["gmm_variant"].fillna("")
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    common.log(f"loaded {describe_load(prov)}")
    return df


def assign_window(df: pd.DataFrame, checkpoints: list[int]) -> pd.DataFrame:
    """Label each row with its checkpoint window ``(prev, c]`` -> ``"le_{c}"``."""
    edges = [1, *sorted(checkpoints)]
    labels = [f"le_{c}" for c in sorted(checkpoints)]
    df = df.copy()
    df["window"] = pd.cut(df["n_votes"], bins=edges, labels=labels)
    df["window_hi"] = pd.cut(df["n_votes"], bins=edges, labels=sorted(checkpoints)).astype("Int64")
    return df[df["window"].notna()]


def variant_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The comparison arms: fusion variants + the two controls."""
    v = df[df["gmm_variant"] != ""].copy()
    keep = v["gmm_variant"].str.match(FUSION_ARM_RE) | v["gmm_variant"].isin(["xcal_only", "pooled_mid"])
    return v[keep]


def window_table(v: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    g = (
        v.groupby(["arm", "gmm_variant", "window"], observed=True)
        .agg(
            cost=("cost", "mean"),
            regret=("regret", "mean"),
            fpr=("fpr", "mean"),
            fnr=("fnr", "mean"),
            threshold=("threshold", "mean"),
            oracle_threshold=("oracle_threshold", "mean"),
            degenerate_rate=("degenerate", "mean"),
            n=("cost", "size"),
        )
        .reset_index()
    )
    g.to_csv(agg_dir / "anchored_window_table.csv", index=False)
    return g


def paired_contrasts(v: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Per-window mean paired difference of each fusion arm vs each control.

    Pairing unit: identical (arm, category, seed, t) steps - every variant of a
    step re-cuts the same model against the same test scores, so the difference
    isolates the threshold rule.
    """
    keys = ["arm", "category", "seed", "t", "window"]
    rows = []
    fusion_names = sorted(n for n in v["gmm_variant"].unique() if FUSION_ARM_RE.match(n))
    for control in ("xcal_only", "pooled_mid"):
        c = v[v["gmm_variant"] == control].set_index(keys)[["cost", "regret", "fnr"]]
        for name in fusion_names:
            a = v[v["gmm_variant"] == name].set_index(keys)[["cost", "regret", "fnr"]]
            j = a.join(c, how="inner", lsuffix="_a", rsuffix="_c")
            if j.empty:
                continue
            j = j.reset_index()
            for window, w in j.groupby("window", observed=True):
                rows.append(
                    {
                        "variant": name,
                        "control": control,
                        "window": window,
                        "n_steps": int(len(w)),
                        "d_cost": float((w["cost_a"] - w["cost_c"]).mean()),
                        "d_regret": float((w["regret_a"] - w["regret_c"]).mean()),
                        "d_fnr": float((w["fnr_a"] - w["fnr_c"]).mean()),
                        # Sign test share: fraction of steps the fusion arm wins.
                        "win_rate_cost": float((w["cost_a"] < w["cost_c"]).mean()),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(agg_dir / "anchored_paired_contrasts.csv", index=False)
    return out


def stability_table(v: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """H3: per-arm step-to-step threshold delta, past the safe-blend ramp."""
    keys = ["arm", "category", "seed", "gmm_variant"]
    w = v[v["n_votes"] > 20].sort_values([*keys, "t"])
    w = w.assign(d_thr=w.groupby(keys, observed=True)["threshold"].diff().abs())
    g = (
        w.dropna(subset=["d_thr"])
        .groupby("gmm_variant", observed=True)
        .agg(mean_abs_dthr=("d_thr", "mean"), sd_dthr=("d_thr", "std"), n=("d_thr", "size"))
        .reset_index()
    )
    g.to_csv(agg_dir / "anchored_stability.csv", index=False)
    return g


def provenance_table(v: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Which estimator path each arm actually took, per window."""
    g = (
        v[v["gmm_variant"].str.match(FUSION_ARM_RE)]
        .groupby(["gmm_variant", "window", "threshold_provenance"], observed=True)
        .size()
        .reset_index()
        .rename(columns={0: "n"})
    )
    g.to_csv(agg_dir / "anchored_provenance.csv", index=False)
    return g


def hypothesis_verdicts(contrasts: pd.DataFrame, windows: pd.DataFrame, stability: pd.DataFrame) -> dict:
    """The plan's H1-H4, mechanically evaluated (the report still gets read)."""
    deep = contrasts[contrasts["window"].astype(str).str.replace("le_", "").astype(int) >= DEEP_WINDOWS_MIN]
    out: dict = {}

    # H1: at 100-300 votes at least one fusion arm beats pure x-cal on regret.
    h1 = deep[(deep["control"] == "xcal_only")]
    h1_best = h1.groupby("variant")["d_regret"].mean().sort_values() if not h1.empty else pd.Series(dtype=float)
    out["h1_deep_regret_vs_xcal"] = {k: float(v_) for k, v_ in h1_best.items()}
    out["h1_supported"] = bool((h1_best < 0).any()) if not h1_best.empty else None
    out["h1_best_arm"] = str(h1_best.index[0]) if not h1_best.empty else None

    # Attribution: rank-transfer absorbing the win names deficit 2; an anchored
    # arm winning where rank-transfer doesn't names deficit 1; fold_anchored
    # beating anchored names the train-anchor bias.
    if not h1_best.empty:
        rt = h1_best.get("rank_transfer", np.nan)
        anch = h1_best[h1_best.index.str.startswith("anchored_")].min() if len(h1_best) else np.nan
        fold = h1_best[h1_best.index.str.startswith("fold_anchored_")].min() if len(h1_best) else np.nan
        out["h1_attribution"] = {
            "rank_transfer_d_regret": None if pd.isna(rt) else float(rt),
            "best_anchored_d_regret": None if pd.isna(anch) else float(anch),
            "best_fold_anchored_d_regret": None if pd.isna(fold) else float(fold),
            "fold_beats_label_anchored": None if (pd.isna(fold) or pd.isna(anch)) else bool(fold < anch),
        }

    # H2: the winning fusion arm beats the shipped blend at matched steps.
    if out.get("h1_best_arm"):
        h2 = deep[(deep["control"] == "pooled_mid") & (deep["variant"] == out["h1_best_arm"])]
        out["h2_supported"] = bool((h2["d_regret"].mean() < 0)) if not h2.empty else None

    # H3: the winning arm cuts step-to-step threshold delta vs pure x-cal.
    if out.get("h1_best_arm") and not stability.empty:
        s = stability.set_index("gmm_variant")["mean_abs_dthr"]
        if out["h1_best_arm"] in s.index and "xcal_only" in s.index:
            out["h3_supported"] = bool(s[out["h1_best_arm"]] < s["xcal_only"])
            out["h3_mean_abs_dthr"] = {"winner": float(s[out["h1_best_arm"]]), "xcal_only": float(s["xcal_only"])}

    # H4: winner's FNR stays within the conformal budget at every checkpoint.
    if out.get("h1_best_arm") and not windows.empty:
        w = windows[windows["gmm_variant"] == out["h1_best_arm"]]
        out["h4_max_window_fnr"] = float(w["fnr"].max()) if not w.empty else None
        out["h4_supported"] = bool(w["fnr"].max() <= FN_BUDGET) if not w.empty else None

    return out


def write_report(
    results: Path, windows: pd.DataFrame, contrasts: pd.DataFrame, stability: pd.DataFrame, verdicts: dict
) -> None:
    lines = [
        "# Anchored-mixture calibration study (#2852) - draft report",
        "",
        "Auto-generated by `analyze_anchored.py`; numbers are means over paired",
        "within-step variants (identical models, votes, and test scores per step).",
        "Design + decision rules: `docs/plans/population-anchored-calibration.md`.",
        "",
        "## Hypothesis verdicts (mechanical; read the tables before believing them)",
        "",
        "```json",
        json.dumps(verdicts, indent=2),
        "```",
        "",
        "## Per-window means",
        "",
        _md(windows),
        "",
        "## Paired contrasts (fusion - control)",
        "",
        _md(contrasts),
        "",
        "## Threshold stability (|delta threshold| per step, votes > 20)",
        "",
        _md(stability),
        "",
    ]
    (results / "REPORT.md").write_text("\n".join(lines))


def main() -> int:
    results = common.RESULTS
    agg = results / "agg"
    agg.mkdir(parents=True, exist_ok=True)
    df = load_cells(results / "cells")
    if df.empty:
        common.log("no cells found; nothing to analyze")
        return 1
    df = assign_window(df, cfg.ANCHORED_CHECKPOINTS)
    v = variant_frame(df)

    windows = window_table(v, agg)
    contrasts = paired_contrasts(v, agg)
    stability = stability_table(v, agg)
    provenance_table(v, agg)
    verdicts = hypothesis_verdicts(contrasts, windows, stability)

    (results / "summary.json").write_text(json.dumps(verdicts, indent=2))
    write_report(results, windows, contrasts, stability, verdicts)
    common.log(f"wrote {results / 'summary.json'} and {results / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

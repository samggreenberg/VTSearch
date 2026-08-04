"""Stage 3 (safe-threshold study, #2799): the ON-vs-OFF trajectory A/B.

``analyze_safe.py`` compares GMM variants *within* a step: every variant re-cuts
the same model on the same votes, so the contrasts there are pure
threshold-rule effects.  That is the right lens for "which cut rule is better",
but it cannot answer "should ``safe_thresholds`` be on for every user", because
the threshold is **not** an output-only quantity: Autopilot's Hard phase picks
the unlabeled item nearest the decision threshold
(:func:`vtscore.eval.al_strategies._hard_pick_by_index`), so turning the blend
on changes *which items the user is asked to vote on*, and the two runs' label
sets diverge from the first Hard pick onward.

This analyzer therefore pairs two **separate runs** — one launched with
``CALIB_SAFE_THRESHOLDS=1``, one with ``0``, otherwise identical (same head,
arms, categories, seeds, steps) — on their ``(arm, category, seed)`` cells.
Because the trajectories diverge, rows are first collapsed to a per-cell mean
inside each vote window, and the paired test runs over cells (the independent
units), not over steps.

Env:
  ``CALIB_AB_ON``   results dir of the safe_thresholds=1 run (default ``$CALIB_RESULTS``)
  ``CALIB_AB_OFF``  results dir of the safe_thresholds=0 run (required)
  ``CALIB_AB_OUT``  where the A/B tables land (default the ON run's results dir)

Writes ``agg/ab_window_by_arm.csv``, ``agg/ab_paired_cells.csv``,
``summary_ab.json`` and a ``REPORT_AB.md`` draft.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: Vote-count windows the comparison aggregates over.  Below 6 votes the blend
#: is pure GMM (full authority), 6-20 is the ramp, above 20 it is pure
#: cross-cal — where the two arms should agree except through the vote history
#: the blend already steered.
WINDOWS: dict[str, tuple[int, int]] = {
    "pure_gmm_2_5": (2, 5),
    "ramp_6_20": (6, 20),
    "post_ramp_21_plus": (21, 10**6),
    "all_steps": (2, 10**6),
}

#: Metrics compared per window.  ``cost`` is the pre-registered decision metric;
#: ``fnr``/``fpr`` say *how* a cost change was bought (a needle-finding tool
#: cares about missed positives), and ``average_precision``/``auroc`` say
#: whether the ranking itself moved — which only selection feedback can do.
METRICS: tuple[str, ...] = ("cost", "fnr", "fpr", "regret", "average_precision", "auroc", "degenerate")

CELL_KEYS: tuple[str, ...] = ("arm", "category", "seed")

#: Row scopes the comparison is computed over.  ``app_visible`` keeps only the
#: steps at which the app would have had a trained detector on screen
#: (``app_trained``): before the Good/Bad quorum the app sorts by text/example
#: cosine, so neither the threshold the user sees nor the next Hard pick comes
#: from the detector — a difference measured there is a difference no user
#: experiences (the #2788 lesson).  ``all_steps`` keeps every trainable step,
#: which is what a purely numerical reading of the harness would report.
SCOPES: tuple[str, ...] = ("app_visible", "all_steps")


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False, floatfmt=".4f")
    except Exception:  # noqa: BLE001 - tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


def load_base_rows(results_dir: Path, label: str) -> pd.DataFrame:
    """Base (production) rows of a run: max pooling, no GMM-variant tag.

    The safe-ON run also emits the #2799 variant rows; those are
    ``analyze_safe.py``'s business.  Here only the row the run actually
    *operated at* counts — that is the threshold that drove the next Hard pick.
    """
    cells = results_dir / "cells"
    files = sorted(p for p in cells.glob("task_*.csv") if "__sweep" not in p.name)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    if "gmm_variant" in df.columns:
        df["gmm_variant"] = df["gmm_variant"].fillna("")
        df = df[df["gmm_variant"] == ""]
    df = df[df["pool_variant"] == "max"] if "pool_variant" in df.columns else df
    df["arm"] = df["dataset"] + "/" + df["embedder"] + "/" + df["style"]
    df["n_votes"] = df["n_good"] + df["n_bad"]
    df["run"] = label
    common.log(f"{label}: {len(df)} base rows from {len(files)} cells ({results_dir})")
    return df


def _cell_means(df: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Per-cell mean of each metric over the steps inside a vote window."""
    w = df[(df["n_votes"] >= lo) & (df["n_votes"] <= hi)]
    if w.empty:
        return pd.DataFrame()
    metrics = [m for m in METRICS if m in w.columns]
    return w.groupby(list(CELL_KEYS), as_index=False)[metrics].mean()


def _wilcoxon(delta: np.ndarray) -> tuple[float, str]:
    """Two-sided Wilcoxon signed-rank p-value for *delta* (NaN if degenerate)."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.allclose(d, 0):
        return float("nan"), "all-zero or empty"
    try:
        from scipy.stats import wilcoxon  # noqa: PLC0415

        return float(wilcoxon(d).pvalue), ""
    except Exception as exc:  # noqa: BLE001 - scipy missing or all-zero deltas
        return float("nan"), f"{type(exc).__name__}"


def _scoped(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Restrict to the steps *scope* counts (see :data:`SCOPES`)."""
    if scope == "app_visible" and "app_trained" in df.columns:
        return df[df["app_trained"] == 1]
    return df


def paired_window_table(on: pd.DataFrame, off: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Mean ON, mean OFF, and paired Δ (ON − OFF) per (scope, arm, window, metric)."""
    rows: list[dict] = []
    paired_cells: list[pd.DataFrame] = []
    for scope in SCOPES:
        on_s, off_s = _scoped(on, scope), _scoped(off, scope)
        for wname, (lo, hi) in WINDOWS.items():
            a, b = _cell_means(on_s, lo, hi), _cell_means(off_s, lo, hi)
            if a.empty or b.empty:
                continue
            j = a.merge(b, on=list(CELL_KEYS), suffixes=("_on", "_off"))
            if j.empty:
                continue
            j.insert(0, "window", wname)
            j.insert(0, "scope", scope)
            paired_cells.append(j)
            for arm, sub in j.groupby("arm"):
                for m in METRICS:
                    if f"{m}_on" not in sub.columns:
                        continue
                    delta = (sub[f"{m}_on"] - sub[f"{m}_off"]).to_numpy(dtype=float)
                    p, note = _wilcoxon(delta)
                    rows.append(
                        {
                            "scope": scope,
                            "arm": arm,
                            "window": wname,
                            "metric": m,
                            "n_cells": int(len(sub)),
                            "safe_on": float(np.nanmean(sub[f"{m}_on"])),
                            "safe_off": float(np.nanmean(sub[f"{m}_off"])),
                            "delta_on_minus_off": float(np.nanmean(delta)),
                            "win_rate_on": float(np.mean(delta < 0)) if np.isfinite(delta).any() else float("nan"),
                            "p_wilcoxon": p,
                            "note": note,
                        }
                    )
    tbl = pd.DataFrame(rows)
    agg_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(agg_dir / "ab_window_by_arm.csv", index=False)
    if paired_cells:
        pd.concat(paired_cells, ignore_index=True).to_csv(agg_dir / "ab_paired_cells.csv", index=False)
    return tbl


def curves(on: pd.DataFrame, off: pd.DataFrame, agg_dir: Path) -> pd.DataFrame:
    """Mean cost/FNR/FPR vs vote count for each run — the shape behind the windows."""
    frames = []
    for df in (on, off):
        metrics = [m for m in ("cost", "fnr", "fpr", "threshold", "degenerate", "app_trained") if m in df.columns]
        g = df.groupby(["arm", "run", "n_votes"], as_index=False)[metrics].mean()
        g["n_rows"] = df.groupby(["arm", "run", "n_votes"]).size().to_numpy()
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).sort_values(["arm", "n_votes", "run"])
    out.to_csv(agg_dir / "ab_curves_vs_votes.csv", index=False)
    return out


def verdict(tbl: pd.DataFrame, scope: str = "app_visible") -> dict:
    """The ship decision: force ``safe_thresholds`` on for every user, or not?

    Read on the production region-vote arm (``max_patch``) and, by default, on
    the steps a user would actually see a detector at (``app_visible``).
    Forcing a default on for everyone needs the blend to *help* where it has
    authority (the two sub-20-vote windows) and to not *hurt* once it has none
    (post-ramp, where only selection feedback can carry a difference).
    """
    out: dict = {"scope": scope}
    prod = tbl[(tbl["scope"] == scope) & ~tbl["arm"].str.contains("whole_image", na=False)]
    for wname in WINDOWS:
        w = prod[(prod["window"] == wname) & (prod["metric"] == "cost")]
        if w.empty:
            out[wname] = {"n_cells": 0, "reading": "no steps in this window at this scope"}
            continue
        d = float(w["delta_on_minus_off"].mean())
        p = float(w["p_wilcoxon"].mean())
        out[wname] = {
            "n_cells": int(w["n_cells"].max()),
            "delta_cost_on_minus_off": d,
            "p": p,
            "reading": ("safe ON better" if d < 0 else "safe ON worse") + (" (significant)" if p < 0.05 else " (n.s.)"),
        }
    helps = [out.get(w, {}).get("delta_cost_on_minus_off", 0.0) for w in ("pure_gmm_2_5", "ramp_6_20")]
    late = out.get("post_ramp_21_plus", {})
    harms_late = late.get("delta_cost_on_minus_off", 0.0) > 0 and late.get("p", 1.0) < 0.05
    out["force_on_for_all_users"] = bool(all(d <= 0 for d in helps) and not harms_late)
    return out


def main() -> int:
    on_dir = Path(os.environ.get("CALIB_AB_ON", str(common.RESULTS)))
    off_env = os.environ.get("CALIB_AB_OFF")
    if not off_env:
        common.log("ERROR: set CALIB_AB_OFF to the safe_thresholds=0 run's results dir")
        return 2
    off_dir = Path(off_env)
    out_dir = Path(os.environ.get("CALIB_AB_OUT", str(on_dir)))

    on = load_base_rows(on_dir, "safe_on")
    off = load_base_rows(off_dir, "safe_off")
    if on.empty or off.empty:
        common.log("no cells to compare (one of the runs produced no rows)")
        return 1

    agg_dir = out_dir / "agg"
    tbl = paired_window_table(on, off, agg_dir)
    curve = curves(on, off, agg_dir)
    verdicts = {scope: verdict(tbl, scope) for scope in SCOPES}
    v = verdicts["app_visible"]

    # Where the detector actually goes live: the app sorts by text/example
    # cosine until the Good/Bad quorum, so this is the first vote count at which
    # the blend can change anything a user sees.
    live = on[on["app_trained"] == 1]["n_votes"] if "app_trained" in on.columns else pd.Series(dtype=float)
    first_live = int(live.min()) if not live.empty else -1

    heads = sorted(set(on.get("head", pd.Series(dtype=str)).dropna().unique()))
    summary = {
        "on_dir": str(on_dir),
        "off_dir": str(off_dir),
        "heads": heads,
        "arms": sorted(on["arm"].unique()),
        "n_cells_on": int(on.groupby(list(CELL_KEYS)).ngroups),
        "n_cells_off": int(off.groupby(list(CELL_KEYS)).ngroups),
        "windows": {k: list(v_) for k, v_ in WINDOWS.items()},
        "first_app_visible_vote_count": first_live,
        "verdict": v,
        "verdict_by_scope": verdicts,
    }
    (out_dir / "summary_ab.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Safe thresholds ON vs OFF — trajectory A/B (#2799)",
        "",
        f"Head: `{', '.join(heads) or 'unknown'}` · cells ON/OFF: "
        f"{summary['n_cells_on']}/{summary['n_cells_off']} · ON `{on_dir}` vs OFF `{off_dir}`",
        "",
        "Both runs are full simulations; because the blended threshold drives Autopilot's",
        "Hard pick, the two arms vote on different items, so cells (category × seed), not",
        "steps, are the paired units.",
        "",
        f"The app shows a trained detector from **{first_live} votes** onward; below that it",
        "sorts by text/example cosine, so `scope=app_visible` is what users actually get and",
        "`scope=all_steps` is the purely numerical reading.",
        "",
        "## Per-window paired comparison (Δ = ON − OFF; negative = safe thresholds better)",
        "",
        _md(tbl),
        "",
        "## Verdict (read on the production max_patch arm, app-visible steps)",
        "",
        "```json",
        json.dumps(verdicts, indent=2),
        "```",
        "",
        f"Curves: `agg/ab_curves_vs_votes.csv` ({len(curve)} rows) · paired cells: `agg/ab_paired_cells.csv`",
    ]
    (out_dir / "REPORT_AB.md").write_text("\n".join(lines) + "\n")
    common.log(f"wrote {out_dir / 'REPORT_AB.md'}")
    common.log(json.dumps(v, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

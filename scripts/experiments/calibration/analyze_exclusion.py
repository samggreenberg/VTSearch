#!/usr/bin/env python
"""#3312: did the #3308 voted-media exclusion buy anything, and is the floor right?

Reads the two-stage grid `launch_exclusion_3308.sh` submits and answers the two
pre-registered questions separately, because they live in different regimes:

* **Stage A** (production scale, ~2100-media haystack, 150 clicks) - is the
  exclusion worth anything where a real user actually works?  The floor is inert
  here by construction, so the only contrast is `off` vs the shipped arm, and
  the pre-registered expectation is that it is **not resolvable**.  This
  analyzer is therefore built to report a *bound* rather than a winner: see
  `NULL_BOUND` and the "not resolvable" wording in the verdict table.
* **Stage B** (deep voting, ~420-media haystack, 380 clicks) - is 60 the right
  floor?  Here the remainder crosses every arm's floor during the run, so each
  arm switches its exclusion off at a different, *known* step, and the
  differences are attributable to the floor rather than to the arm.

The incumbent in both stages is `app`: the arm that pins nothing and resolves
the floor through the app's own `resolve_exclusion_floor`.  Every contrast is a
difference from it, measured on this grid under this code - a baseline quoted
from an earlier study is the one arm that cannot be paired.

Usage:
    python analyze_exclusion.py --base <BASE> --out <BASE>/analysis [--stages AB]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import curves  # noqa: E402
from _cells_io import assert_one_opening  # noqa: E402

#: The incumbent arm's directory name.  It pins nothing; its floor is whatever
#: `resolve_exclusion_floor(None)` returns, which is what a live detector uses.
INCUMBENT = "app"

#: Stage -> the arms that stage submits, in axis order (most permissive first).
STAGE_ARMS: dict[str, tuple[str, ...]] = {
    "A": ("off", "app"),
    "B": ("off", "always", "app", "f250"),
}

#: Bands on the **votes' share of the haystack**, which is the axis the
#: mechanism runs on: the exclusion can move the fitted population by at most
#: this fraction, so its effect is bounded by it.  Banding on raw vote count
#: instead would put stage A's 150 votes and stage B's 150 votes in the same
#: bucket while they consume 7% and 36% of their haystacks respectively - which
#: is precisely the confound this study exists to separate.
SHARE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("share 0-10%", 0.0, 0.10),
    ("share 10-25%", 0.10, 0.25),
    ("share 25-50%", 0.25, 0.50),
    ("share 50-100%", 0.50, 1.01),
)

#: How much worse than the incumbent an arm may be in ANY band and still be
#: shippable.  The margin PR #2891 pre-registered for this family of decisions,
#: kept here so a floor change is held to the same bar as a cut-rule change.
HARM_TOLERANCE = 0.01

#: Below this, a stage-A difference is reported as "no effect worth having"
#: rather than as a win.  Deliberately equal to `HARM_TOLERANCE`: an arm that
#: cannot move cost by as much as we would tolerate losing is not a difference
#: anyone should act on, whichever way it points.
NULL_BOUND = HARM_TOLERANCE

#: Bootstrap resamples for the paired standard errors.
N_BOOT = 2000

#: Cells are keyed by these; `geometry` replaces (embedder, style) so the three
#: corners of the mode/embedder square are one column.
CELL_KEYS = ("dataset", "category", "seed", "geometry")


def geometry_of(row) -> str:
    """``embedder/style`` - the representation AND the voting mode it votes in."""
    emb = str(row.get("embedder", "") or "")
    style = str(row.get("style", "") or "")
    return f"{emb}/{style}" if style else emb


def voting_mode(geometry: str) -> str:
    """``region`` when the geometry max-pools over patches, else ``binary``."""
    return "region" if geometry.endswith("/max_patch") else "binary"


def load_arms(base: Path, stages: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    """Every arm's base rows, tagged with its stage and arm, plus provenance."""
    import analyze_spikes as sp

    parts: list[pd.DataFrame] = []
    prov: dict[str, dict] = {}
    for stage in stages:
        for arm in STAGE_ARMS[stage]:
            arm_dir = base / f"stage{stage}" / arm / "results"
            df, p = sp.load_arm(arm_dir)
            prov[f"{stage}/{arm}"] = {**p, "dir": str(arm_dir)}
            if df.empty:
                continue
            df = df.copy()
            _assert_arm_matches_its_directory(df, arm_dir, arm)
            df["stage"] = stage
            df["arm"] = arm
            parts.append(df)
    if not parts:
        raise SystemExit(f"no cells under {base}/stage*/*/results/cells")
    frame = pd.concat(parts, ignore_index=True)
    frame["geometry"] = frame.apply(geometry_of, axis=1)
    frame["mode"] = frame["geometry"].map(voting_mode)
    # The share of the haystack these votes have consumed - the study's axis.
    # Checked for PRESENCE first: `frame.get` on a missing column returns None,
    # and `pd.to_numeric(None)` is a scalar NaN rather than a Series, so a
    # membership test is the only form of this guard that reports the stale
    # frame instead of dying on it two lines later.
    missing = [c for c in ("n_haystack", "n_remainder") if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"these cells carry no {'/'.join(missing)} column(s): they were produced by a harness "
            "that predates #3312, so the axis this study bands on cannot be reconstructed."
        )
    hay = pd.to_numeric(frame["n_haystack"], errors="coerce")
    rem = pd.to_numeric(frame["n_remainder"], errors="coerce")
    if hay.isna().all() or (hay <= 0).all():
        raise SystemExit("every cell reports an empty `n_haystack`; the study's axis is unreadable.")
    frame["voted_share"] = 1.0 - (rem / hay)
    # One opening for the whole study, or the pooled numbers below describe two
    # different experiments (#3278).
    assert_one_opening(frame, where="voted-exclusion study")
    return frame, prov


def _assert_arm_matches_its_directory(df: pd.DataFrame, arm_dir: Path, arm: str) -> None:
    """The arm is taken from the ROW and required to agree with the directory.

    A run that read a stale environment - the "cluster ran the previous commit"
    failure - is otherwise indistinguishable from one that did what its
    directory says, and this is the one place it would be silent.  `app` is
    checked against the *resolved* floor rather than a literal, because the
    whole point of that arm is that it is not pinned.
    """
    if "exclusion_arm" not in df.columns or "exclusion_min_remainder" not in df.columns:
        raise SystemExit(
            f"arm {arm_dir} has no `exclusion_arm`/`exclusion_min_remainder` column: it was "
            "produced by a run_cells.py that predates #3312, so which arm it is cannot be "
            "established."
        )
    stamped = sorted(set(df["exclusion_arm"].astype(str)))
    want_prefix = "app(" if arm == INCUMBENT else arm
    bad = [s for s in stamped if not s.startswith(want_prefix)]
    if bad:
        raise SystemExit(
            f"arm {arm_dir} holds rows stamped exclusion_arm={bad} but its directory says "
            f"{arm!r}. Refusing to analyse a mislabelled arm."
        )


def arm_floor(frame: pd.DataFrame, stage: str, arm: str) -> float:
    """The numeric floor an arm's cells actually ran at, read off the rows."""
    sel = frame[(frame["stage"] == stage) & (frame["arm"] == arm)]
    vals = pd.to_numeric(sel["exclusion_min_remainder"], errors="coerce").dropna().unique()
    if len(vals) != 1:
        raise SystemExit(f"stage {stage} arm {arm} ran at {sorted(vals)} floors; expected exactly one")
    return float(vals[0])


def band_of(share: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=share.index, dtype="object")
    for name, lo, hi in SHARE_BANDS:
        out = out.mask((share >= lo) & (share < hi), name)
    return out


def cell_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One number per (cell, stage, band, arm): that trajectory's mean over the band.

    Collapsing steps to a cell mean BEFORE any test is what makes the bootstrap
    honest.  Steps inside one trajectory share a model and a labelset prefix, so
    treating them as independent observations would shrink every standard error
    by roughly the square root of the band's width and turn noise into findings.
    """
    d = frame.dropna(subset=[metric]).copy()
    d["band"] = band_of(d["voted_share"])
    d = d[d["band"].notna()]
    keys = [*CELL_KEYS, "mode", "stage", "band", "arm"]
    return d.groupby(keys, dropna=False)[metric].mean().reset_index()


def paired_vs_incumbent(cells: pd.DataFrame, metric: str, rng: np.random.Generator) -> pd.DataFrame:
    """Paired Delta(arm - incumbent) per (stage, geometry, band), bootstrapped over cells.

    The pairing is on the CELL, not on the vote: two arms saw different votes by
    construction (that is the acquisition feedback this study exists to
    capture), but they saw the same category, the same seed and the same
    geometry, and that is what the difference is taken within.
    """
    rows: list[dict] = []
    for stage, stage_cells in cells.groupby("stage", dropna=False):
        inc = stage_cells[stage_cells["arm"] == INCUMBENT].set_index([*CELL_KEYS, "band"])[metric]
        for (geom, band, arm), g in stage_cells.groupby(["geometry", "band", "arm"], dropna=False):
            if arm == INCUMBENT:
                continue
            g = g.set_index([*CELL_KEYS, "band"])
            common = g.index.intersection(inc.index)
            # Cells the incumbent has and this arm does not (or vice versa) are
            # DROPPED from the paired difference and COUNTED here.  A starved
            # cell is a result about that arm; silently keeping the arm's other
            # cells would compute its mean over the subset that worked.
            d = (g.loc[common, metric] - inc.loc[common]).to_numpy(dtype=float)
            if d.size == 0:
                continue
            boot = rng.choice(d, size=(N_BOOT, d.size), replace=True).mean(axis=1)
            rows.append(
                {
                    "stage": stage,
                    "geometry": geom,
                    "mode": voting_mode(str(geom)),
                    "band": band,
                    "arm": arm,
                    "delta": float(d.mean()),
                    "se": float(boot.std(ddof=1)),
                    "n_cells_paired": int(d.size),
                    "n_cells_arm": int(len(g)),
                    "n_cells_incumbent": int((inc.index.get_level_values("band") == band).sum()),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["resolved"] = out["delta"].abs() > 2 * out["se"]
    return out


def floor_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Where each arm's exclusion was actually LIVE, per stage.

    The mechanism audit, and the thing that makes a stage-B difference readable:
    an arm only differs from the incumbent in the steps where their floors
    disagree about whether to exclude.  `n_remainder` is exactly the count
    `apply_vote_exclusion` compares against the floor, so this is reconstructed
    from the rows rather than reported by the harness - which means it also
    checks the harness, not just describes it.
    """
    rows: list[dict] = []
    for (stage, arm), g in frame.groupby(["stage", "arm"], dropna=False):
        floor = arm_floor(frame, str(stage), str(arm))
        rem = pd.to_numeric(g["n_remainder"], errors="coerce")
        live = (rem > 0) & (rem >= floor)
        rows.append(
            {
                "stage": stage,
                "arm": arm,
                "floor": floor,
                "steps": int(len(g)),
                "steps_excluding": int(live.sum()),
                "frac_excluding": float(live.mean()) if len(g) else float("nan"),
                "min_remainder": float(rem.min()) if len(g) else float("nan"),
                "max_remainder": float(rem.max()) if len(g) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["stage", "floor"]).reset_index(drop=True)


def trap_check(frame: pd.DataFrame) -> pd.DataFrame:
    """Two arms must agree exactly while their floors agree - a plumbing check.

    `always` (floor 0) and `app` (floor 60) differ in one thing only: what
    happens once the remainder falls under 60.  Above that line they are the
    same estimator, so on any step where BOTH have a remainder above the higher
    floor they must produce the *same threshold* - not a similar one.  Same for
    `f250` vs `app` above 250.

    Trajectories diverge after the first step where the floors disagree, so this
    is only asserted on the common prefix: steps before either arm's first
    disagreement.  A violation there means the arm plumbing is wrong (a stale
    environment, a mis-set env var) and no number in this report can be trusted,
    which is why it is computed and printed rather than left to inspection.
    """
    rows: list[dict] = []
    for stage in sorted(frame["stage"].dropna().unique()):
        st = frame[frame["stage"] == stage]
        arms = [a for a in STAGE_ARMS.get(str(stage), ()) if a in set(st["arm"])]
        if INCUMBENT not in arms:
            continue
        inc_floor = arm_floor(frame, str(stage), INCUMBENT)
        for arm in arms:
            if arm == INCUMBENT:
                continue
            f = arm_floor(frame, str(stage), arm)
            # The two agree while the remainder is at or above BOTH floors, and
            # `off` (inf) never agrees with anything - it is excluded from this
            # check by construction rather than by a special case.
            agree_above = max(f, inc_floor)
            if not math.isfinite(agree_above):
                rows.append(
                    {
                        "stage": stage,
                        "arm": arm,
                        "floor": f,
                        "checked": 0,
                        "identical": pd.NA,
                        "note": "floors never agree (off arm): nothing to check",
                    }
                )
                continue
            keys = [*CELL_KEYS, "t"]
            a = st[(st["arm"] == arm)].set_index(keys)["threshold"]
            b = st[(st["arm"] == INCUMBENT)].set_index(keys)["threshold"]
            rem = st[(st["arm"] == INCUMBENT)].set_index(keys)["n_remainder"]
            common = a.index.intersection(b.index)
            if common.empty:
                continue
            above = pd.to_numeric(rem.loc[common], errors="coerce") >= agree_above
            common = common[above.to_numpy()]
            if len(common) == 0:
                rows.append(
                    {
                        "stage": stage,
                        "arm": arm,
                        "floor": f,
                        "checked": 0,
                        "identical": pd.NA,
                        "note": f"no step had remainder >= {agree_above:g}",
                    }
                )
                continue
            same = np.isclose(
                a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float), rtol=0, atol=1e-12
            )
            rows.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "floor": f,
                    "checked": int(len(common)),
                    "identical": float(same.mean()),
                    "note": "must be 1.0: above both floors these arms ARE the same estimator",
                }
            )
    return pd.DataFrame(rows)


def verdict(paired: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The pre-registered decision, per (stage, voting mode, arm).

    An arm beats the incumbent when it wins by more than 2 SE pooled over the
    bands it is resolved in, AND is not worse by more than
    :data:`HARM_TOLERANCE` in ANY band.  The pointwise half is why this reads
    banded and not averaged: an arm can win overall while being worse everywhere
    a short session actually lives, and a short session is most sessions.

    The extra column this study needs is ``no_effect``.  Its headline question
    is not "which arm wins" but "is there anything here at all", and a
    difference smaller than :data:`NULL_BOUND` with a CI that excludes nothing
    interesting is a **result** - the honest outcome of "we improved rigor and
    it cost nothing".  Reporting that as a narrow win would be exactly the false
    precision the two-significant-digit rule exists to stop.
    """
    rows: list[dict] = []
    if paired.empty:
        return pd.DataFrame()
    for (stage, mode, arm), g in paired.groupby(["stage", "mode", "arm"], dropna=False):
        # Inverse-variance pooling across bands, which weights a band by how
        # well it was measured rather than by how many steps it happens to span.
        w = 1.0 / np.square(g["se"].to_numpy(dtype=float))
        d = g["delta"].to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(d) & (w > 0)
        pooled = float(np.sum(d[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
        pooled_se = float(np.sqrt(1.0 / np.sum(w[ok]))) if ok.any() else float("nan")
        worst_i = g["delta"].idxmax()
        worst = float(g["delta"].max())
        hi = pooled + 2 * pooled_se
        lo = pooled - 2 * pooled_se
        rows.append(
            {
                "stage": stage,
                "mode": mode,
                "arm": arm,
                "metric": metric,
                "pooled_delta": pooled,
                "pooled_se": pooled_se,
                "ci_lo": lo,
                "ci_hi": hi,
                # Three separate facts, deliberately not collapsed into one.
                # `resolved` asks whether the difference can be distinguished
                # from zero at all; `negligible` asks whether the whole interval
                # is too small to act on.  They are independent, and the
                # combination that this study most expects at production scale -
                # resolved AND negligible, i.e. "real but not worth anything" -
                # is exactly the one a single boolean would misreport.  Calling
                # that "no effect" would throw away the evidence that the
                # shipped arm is doing measurable work; calling it a "win" would
                # invent a reason to act.
                "resolved": bool(abs(pooled) > 2 * pooled_se),
                "beats_incumbent": bool(pooled < -2 * pooled_se),
                "worse_than_incumbent": bool(pooled > 2 * pooled_se),
                "negligible": bool(np.isfinite(lo) and np.isfinite(hi) and lo > -NULL_BOUND and hi < NULL_BOUND),
                "worst_band_delta": worst,
                "worst_band": str(g.loc[worst_i, "band"]),
                "harms_a_band": bool(worst > HARM_TOLERANCE),
                "candidate": bool(pooled < -2 * pooled_se and worst <= HARM_TOLERANCE),
            }
        )
    return pd.DataFrame(rows).sort_values(["stage", "mode", "arm"]).reset_index(drop=True)


def figures(frame: pd.DataFrame, outdir: Path, baseline_csv: Path | None) -> list[str]:
    """The mandatory quality-over-clicks pair, drawn by the one implementation.

    ``curves`` panels by ``dataset`` and colours by ``arm``.  This study has one
    dataset, two stages and three geometries, and neither the stage nor the
    geometry may be averaged away - a stage IS a haystack size, which is the
    study's axis, and averaging ``max_patch`` with ``whole_image`` describes no
    system anyone could run.  So the panel column carries both and the caption
    says so; the hue stays the arm, which is what is being compared.
    """
    d = frame.copy()
    # `-`, not `/`: `per_run_figures` puts the panel column into the FILENAME,
    # so a geometry written `dinov3_patch/max_patch` would ask matplotlib to
    # save into a directory that does not exist - a crash after every cell has
    # been paid for (the #3287 note).
    d["dataset"] = (
        d["dataset"].astype(str)
        + " · stage"
        + d["stage"].astype(str)
        + " · "
        + d["geometry"].astype(str).str.replace("/", "-", regex=False)
    )
    baseline = curves.text_sort_baseline(baseline_csv) if baseline_csv and Path(baseline_csv).exists() else None
    if baseline is not None:
        # The baseline is keyed by the real dataset name; re-key it onto the
        # per-panel names so every panel gets its own click-0 anchor rather than
        # silently losing it.
        reps = []
        for panel in sorted(d["dataset"].unique()):
            b = baseline.copy()
            b["dataset"] = panel
            reps.append(b)
        baseline = pd.concat(reps, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    denominator = d[[*CELL_KEYS, "dataset", "embedder"]].drop_duplicates()
    arms = sorted(set(d["arm"]))
    for metric in ("cost", "average_precision"):
        if metric not in d.columns:
            continue
        written += curves.quality_vs_clicks(
            d,
            outdir,
            arms=arms,
            metric=metric,
            denominator=denominator,
            baseline=baseline,
            lower_is_better=(metric == "cost"),
        )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="dir holding stageA/ and stageB/ arm dirs")
    ap.add_argument("--out", required=True, help="where the report, tables and figures go")
    ap.add_argument("--stages", default="AB", help="which stages to read (A, B or AB)")
    ap.add_argument("--baseline", default=None, help="text_baseline.py CSV: the click-0 anchor")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    base = Path(args.base)
    out = Path(args.out)
    (out / "agg").mkdir(parents=True, exist_ok=True)
    stages = [s for s in "AB" if s in args.stages.upper()]
    if not stages:
        raise SystemExit(f"--stages {args.stages!r} selected no stage (expected A, B or AB)")

    frame, prov = load_arms(base, stages)
    rng = np.random.default_rng(20260828)

    # --- completeness, counted before anything is averaged --------------------
    counts = (
        frame[["stage", "arm", *CELL_KEYS]]
        .drop_duplicates()
        .groupby(["stage", "arm", "geometry"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    counts.to_csv(out / "agg" / "cell_counts.csv", index=False)
    (out / "agg" / "provenance.json").write_text(json.dumps(prov, indent=2))

    # --- the mechanism: where was the exclusion actually live? ----------------
    regime = floor_regime(frame)
    regime.to_csv(out / "agg" / "floor_regime.csv", index=False)
    trap = trap_check(frame)
    trap.to_csv(out / "agg" / "trap_check.csv", index=False)

    # --- levels ---------------------------------------------------------------
    levels = []
    for metric in ("cost", "regret_honest", "regret", "average_precision"):
        if metric not in frame.columns:
            continue
        cm = cell_means(frame, metric)
        lv = (
            cm.groupby(["stage", "geometry", "mode", "band", "arm"], dropna=False)[metric]
            .agg(mean="mean", sd="std", n_cells="size")
            .reset_index()
        )
        lv["metric"] = metric
        levels.append(lv)
    by_band = pd.concat(levels, ignore_index=True) if levels else pd.DataFrame()
    by_band.to_csv(out / "agg" / "by_band.csv", index=False)

    # --- the contrasts --------------------------------------------------------
    paired_all, verdicts = [], []
    for metric in ("cost", "regret_honest"):
        if metric not in frame.columns:
            continue
        cm = cell_means(frame, metric)
        pv = paired_vs_incumbent(cm, metric, rng)
        if pv.empty:
            continue
        pv["metric"] = metric
        paired_all.append(pv)
        verdicts.append(verdict(pv, metric))
    paired = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
    paired.to_csv(out / "agg" / "paired_vs_incumbent.csv", index=False)
    vd = pd.concat(verdicts, ignore_index=True) if verdicts else pd.DataFrame()
    vd.to_csv(out / "agg" / "verdict.csv", index=False)

    figs: list[str] = []
    if not args.no_figures:
        figs = figures(frame, out / "figures", Path(args.baseline) if args.baseline else None)

    if not args.no_viewer:
        import viewer

        d = frame.copy()
        # Embedders are never averaged in the viewer, so the slot carries the
        # dimension that must not be pooled here: the geometry AND the stage,
        # since a stage is a haystack size and that is the study's axis.
        d["embedder"] = "stage" + d["stage"].astype(str) + " · " + d["geometry"].astype(str)
        baseline = curves.text_sort_baseline(args.baseline) if args.baseline and Path(args.baseline).exists() else None
        viewer.build_viewer(
            d,
            out / "viewer.html",
            arms=sorted(set(d["arm"])),
            baseline=baseline,
            title="voted-media exclusion floor (#3312)",
            subtitle="One chart per exclusion arm × stage × geometry × category; tick 'overlay on one chart' to compare them directly.",
        )

    write_report(out, frame, by_band, paired, vd, regime, trap, counts, prov, figs, stages)
    print(f"wrote {out / 'REPORT_exclusion.md'}")
    return 0


def _md(df: pd.DataFrame) -> str:
    """Markdown table, floats rounded to three significant digits on the way in."""
    if df is None or df.empty:
        return "_(no rows)_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: float(f"{x:.3g}") if pd.notna(x) else x)
    try:
        return d.to_markdown(index=False)
    except Exception:  # noqa: BLE001 - tabulate is optional
        return "```\n" + d.to_string(index=False) + "\n```"


def _fmt(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def write_report(out, frame, by_band, paired, vd, regime, trap, counts, prov, figs, stages) -> None:
    """The report.  Two significant digits, every difference paired with its SE."""
    n_read = sum(int(p.get("n_files", 0) or 0) for p in prov.values())
    n_bad = sum(len(p.get("unreadable", []) or []) for p in prov.values())
    n_empty = sum(len(p.get("empty", []) or []) for p in prov.values())

    lines: list[str] = []
    lines.append("# Did the voted-media exclusion buy anything? (#3312)\n")
    lines.append(
        "Measures PR #3311 (issue #3308) on real embeddings. The pre-registered design and "
        "decision rules are in [`PLAN.md`](PLAN.md); this file records the verdict they produce, "
        "**including the null**, which is the outcome the study most expects at production scale.\n"
    )
    lines.append("- Interactive slices: [`viewer.html`](viewer.html)\n")
    lines.append(
        f"- Cells read: **{n_read}**"
        + (f"; unreadable: **{n_bad}**; zero-byte: **{n_empty}**" if (n_bad or n_empty) else "; none dropped")
        + "\n"
    )

    lines.append("\n## Was the plumbing right?\n")
    lines.append(
        "Two arms whose floors agree above some remainder ARE the same estimator above it, so "
        "their thresholds must match exactly there - not approximately. `identical` below is the "
        "fraction of such steps that did match; anything under 1.0 means an arm ran under the "
        "wrong environment and **no number in this report can be trusted**.\n"
    )
    lines.append(_md(trap) + "\n")

    lines.append("\n## Where was the exclusion actually live?\n")
    lines.append(
        "Reconstructed from `n_remainder` (exactly the count `apply_vote_exclusion` compares "
        "against the floor), so this checks the harness rather than merely describing it. An arm "
        "that never excludes, or always does, cannot be a contrast about the floor.\n"
    )
    lines.append(_md(regime) + "\n")

    lines.append("\n## Verdict\n")
    if vd.empty:
        lines.append("_(no paired contrast could be formed)_\n")
    else:
        for stage in stages:
            sub = vd[vd["stage"] == stage]
            if sub.empty:
                continue
            lines.append(f"\n### Stage {stage}\n")
            lines.append(_md(sub.drop(columns=["stage"])) + "\n")
            for _, r in sub[sub["metric"] == "cost"].iterrows():
                lines.append("- " + _verdict_sentence(r) + "\n")

    lines.append("\n## Paired differences by band\n")
    lines.append(
        "Every difference is paired on (dataset, category, seed, geometry) and bootstrapped over "
        "cells. Bands are the **votes' share of the haystack** - the axis the mechanism runs on, "
        "since the exclusion can move the fitted population by at most that fraction. A difference "
        "smaller than twice its standard error is **not resolvable here**, which is a finding.\n"
    )
    lines.append(_md(paired) + "\n")

    lines.append("\n## Levels\n")
    lines.append(_md(by_band) + "\n")

    lines.append("\n## Cells per arm\n")
    lines.append(_md(counts) + "\n")

    if figs:
        lines.append("\n## Figures\n")
        for f in figs:
            name = Path(f).name
            lines.append(f"\n![{name}](figures/{name})\n")
        lines.append(
            "\nQuality over clicks, anchored at click 0 on the cell's own zero-click text sort. "
            "Panels are stage × geometry and are **not** averaged together: a stage is a haystack "
            "size, which is this study's axis, and two geometries are two representations whose "
            "mean describes no system anyone could run. Dashed segments bridge the un-measured "
            "stretch before an arm's first trained click.\n"
        )

    (out / "REPORT_exclusion.md").write_text("".join(lines))


def _verdict_sentence(r) -> str:
    """One arm's result, in the vocabulary the decision rules are written in.

    Reads `resolved` and `negligible` as two independent questions, because the
    four combinations are four different findings and three of them are easy to
    mistake for each other:

    * not resolved                -> the grid cannot tell; quote the bound.
    * resolved, negligible        -> real, and still not worth acting on.  The
      outcome this study expects at production scale, and the one a single
      "no effect" boolean would report as an absence of evidence rather than as
      evidence of smallness.
    * resolved, not negligible    -> an actionable win or loss.
    * not resolved, not negligible-> underpowered where it matters; say so.
    """
    arm, stage = r["arm"], r["stage"]
    d, se = float(r["pooled_delta"]), float(r["pooled_se"])
    lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
    where = f"stage {stage}, {r['mode']}, `{arm}` vs the shipped arm"
    resolved, negligible = bool(r.get("resolved")), bool(r.get("negligible"))
    # The shipped arm excludes; every other arm excludes less (or not at all),
    # so a POSITIVE delta means the exclusion helped.
    helped = d > 0

    if not resolved:
        tail = (
            f"The interval [{_fmt(lo, 4)}, {_fmt(hi, 4)}] sits inside ±{NULL_BOUND}, so whatever is "
            "there is too small to act on either way."
            if negligible
            else f"The interval [{_fmt(lo, 4)}, {_fmt(hi, 4)}] is wider than ±{NULL_BOUND}, so this grid "
            "cannot say whether the difference matters. Quote the bound, not a winner."
        )
        return f"**{where}: not resolvable here.** {_fmt(d, 4)} ± {_fmt(se, 4)} cost. {tail}"

    direction = "the exclusion helps" if helped else "the exclusion costs"
    if negligible:
        return (
            f"**{where}: real but negligible.** {direction} by {_fmt(abs(d), 4)} ± {_fmt(se, 4)} cost - "
            f"resolved at more than 2 SE, and the whole interval still inside ±{NULL_BOUND}. "
            "That is a *finding*, not a null: the effect exists and is not worth a decision."
        )
    if bool(r.get("beats_incumbent")):
        harm = " but it harms at least one band, so it does not ship" if bool(r.get("harms_a_band")) else ""
        return f"**{where}: beats the shipped arm** by {_fmt(-d, 4)} ± {_fmt(se, 4)} cost{harm}."
    return (
        f"**{where}: worse than the shipped arm** by {_fmt(d, 4)} ± {_fmt(se, 4)} cost - "
        "the exclusion is doing work worth keeping here."
    )


if __name__ == "__main__":
    raise SystemExit(main())

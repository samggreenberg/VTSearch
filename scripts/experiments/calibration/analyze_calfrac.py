#!/usr/bin/env python
"""#3287: is a 50/50 Train/Calibrate split of the calibration folds optimal?

Reads one results dir per ``calibration_fraction`` arm (see
``launch_calfrac_3287.sh``) and answers the issue's question **per voting mode
and per vote band**, which is the only shape in which it can be answered: the
trade-off it sits on -- fold-model quality against anchor/quantile resolution --
is predicted to REVERSE as the labelset grows, so a pooled winner would be an
average across a crossing.

Three things this analyzer deliberately does not do.

**It does not decompose regret into ``rule_inefficiency`` + ``calibration_shift``.**
Those two are not independent effects of this knob.  ``calibration_shift`` is
measured against ``cal_oracle_cost``, which is estimated *from the calibration
set* -- so moving ``calibration_fraction`` moves the yardstick itself, and the
two terms slide in opposite directions with their sum pinned to ``regret`` by
construction.  #2897 read exactly that anti-correlation as a finding when it was
algebra.  The decomposition is still computed here, but only as
``agg/trap_check.csv`` and only to SHOW the anti-correlation, with the report
stating in words that it is an identity rather than a result.

**It does not pair the arms on votes.**  ``calibration_fraction`` sets the
threshold, the threshold sets the acquisition cut, and the cut sets which item
Autopilot samples next -- so an arm at 0.3 has collected different votes by its
second trained step.  The arms are paired on ``(dataset, category, seed,
geometry)``, and the bootstrap resamples **cells**, never steps: consecutive
steps of one trajectory share a model and are nowhere near independent.

**It does not read a level off a mean over a partial grid.**  Every table
carries the cells it was computed over, and a fraction that starved on part of
the grid is reported as having starved rather than being quietly averaged over
the part that worked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import curves  # noqa: E402
from _cells_io import assert_one_opening  # noqa: E402

#: The incumbent.  Every contrast in this study is a difference from it, and it
#: is measured on this grid under this code rather than quoted from an earlier
#: study -- a baseline read off another run is the one arm that cannot be paired.
INCUMBENT = 0.5

#: Vote bands.  The mechanism runs on labelset SIZE: at a handful of votes the
#: fold models are the scarce thing and the argument says spend votes on Train,
#: while at 150 the quantile's resolution is what is scarce and the argument
#: says spend them on Calibrate.  If that is real the ordering of the arms
#: reverses somewhere in here, and only a banded read can see it.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("early 1-25", 1, 25),
    ("mid 26-60", 26, 60),
    ("late 61-100", 61, 100),
    ("deep 101-150", 101, 150),
)

#: How much worse than the incumbent an arm may be in ANY band and still be
#: shippable.  The margin PR #2891 pre-registered for this family of decisions.
HARM_TOLERANCE = 0.01

#: Bootstrap resamples for the paired standard errors.
N_BOOT = 2000

#: Cells are keyed by these; `geometry` replaces (embedder, style) so the three
#: corners of the mode/embedder square are one column.
CELL_KEYS = ("dataset", "category", "seed", "geometry")


def geometry_of(row) -> str:
    """``siglip/whole_image``-style label for one row's (embedder, style) corner.

    The pair name is collapsed to its LEARN half, because that is the space the
    detector trains, scores and sorts in -- ``siglip+dinov3_patch`` and a bare
    ``dinov3_patch`` arm learn in the same place and differ only in what ranked
    the opening, which this grid holds fixed at a SigLIP text sort.
    """
    emb = str(row["embedder"])
    learn = emb.partition("+")[2] or emb
    return f"{learn}/{row['style']}"


def voting_mode(geometry: str) -> str:
    """Region voting is a property of the STYLE, asserted per cell upstream.

    ``max_patch`` on a boxed dataset pools a dragged ground-truth box;
    ``whole_image`` does not, whatever the embedder can emit.  This is the
    distinction #2877/#2897/#2905 each got wrong by reading it off the dataset.
    """
    return "region" if geometry.endswith("/max_patch") else "binary"


def load_arms(base: Path, fractions: Sequence[float]) -> tuple[pd.DataFrame, dict]:
    """Every arm's base rows, tagged with its fraction, plus a provenance block."""
    import analyze_spikes as sp

    parts: list[pd.DataFrame] = []
    prov: dict[str, dict] = {}
    for f in fractions:
        arm_dir = base / f"f{round(f * 100):03d}" / "results"
        df, p = sp.load_arm(arm_dir)
        prov[f"{f:.2f}"] = {**p, "dir": str(arm_dir)}
        if df.empty:
            continue
        df = df.copy()
        # The fraction is taken from the ROW, not from the directory name, and
        # the two are then required to agree.  A run that read a stale
        # environment would otherwise be indistinguishable from one that did
        # what its directory says -- the "cluster ran the previous commit"
        # failure, in the one place it would be silent.
        if "calibration_fraction" in df.columns:
            stamped = pd.to_numeric(df["calibration_fraction"], errors="coerce").dropna().unique()
            bad = [s for s in stamped if abs(float(s) - f) > 1e-9]
            if bad:
                raise SystemExit(
                    f"arm {arm_dir} holds rows stamped calibration_fraction={sorted(bad)} "
                    f"but its directory says {f}. Refusing to analyse a mislabelled arm."
                )
        else:
            raise SystemExit(
                f"arm {arm_dir} has no `calibration_fraction` column: it was produced by a "
                "run_cells.py that predates #3287, so which arm it is cannot be established."
            )
        df["arm"] = f"{f:.2f}"
        df["fraction"] = f
        parts.append(df)
    if not parts:
        raise SystemExit(f"no cells under {base}/f*/results/cells")
    frame = pd.concat(parts, ignore_index=True)
    frame["geometry"] = frame.apply(geometry_of, axis=1)
    frame["mode"] = frame["geometry"].map(voting_mode)
    # One opening for the whole study, or the pooled numbers below describe two
    # different experiments (#3278).
    assert_one_opening(frame, where="calibration-fraction study")
    return frame, prov


def band_of(t: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=t.index, dtype="object")
    for name, lo, hi in BANDS:
        out = out.mask((t >= lo) & (t <= hi), name)
    return out


def cell_means(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One number per (cell, band, arm): that trajectory's mean over the band.

    Collapsing steps to a cell mean BEFORE any test is what makes the bootstrap
    honest.  Steps inside one trajectory share a model and a labelset prefix, so
    treating them as independent observations would shrink every standard error
    by roughly the square root of the band's width and turn noise into findings.
    """
    d = frame.dropna(subset=[metric]).copy()
    d["band"] = band_of(d["t"])
    d = d[d["band"].notna()]
    keys = [*CELL_KEYS, "mode", "band", "arm", "fraction"]
    return d.groupby(keys, dropna=False)[metric].mean().reset_index()


def paired_vs_incumbent(cells: pd.DataFrame, metric: str, rng: np.random.Generator) -> pd.DataFrame:
    """Paired Delta(arm - incumbent) per (geometry, band), bootstrapped over cells.

    The pairing is on the CELL, not on the vote: the two arms saw different
    votes by construction (that is the acquisition feedback the study exists to
    capture), but they saw the same category, the same seed and the same
    geometry, and that is what the difference is taken within.
    """
    inc = cells[cells["fraction"] == INCUMBENT]
    inc = inc.set_index([*CELL_KEYS, "band"])[metric]
    rows: list[dict] = []
    for (geom, band, frac), g in cells.groupby(["geometry", "band", "fraction"], dropna=False):
        if frac == INCUMBENT:
            continue
        g = g.set_index([*CELL_KEYS, "band"])
        common = g.index.intersection(inc.index)
        # Cells the incumbent has and this arm does not (or vice versa) are
        # DROPPED from the paired difference and COUNTED here.  A starved cell
        # is a result about that arm; silently keeping the arm's other cells
        # would compute its mean over the subset that worked and flatter it.
        d = (g.loc[common, metric] - inc.loc[common]).to_numpy(dtype=float)
        if d.size == 0:
            continue
        boot = rng.choice(d, size=(N_BOOT, d.size), replace=True).mean(axis=1)
        rows.append(
            {
                "geometry": geom,
                "mode": voting_mode(str(geom)),
                "band": band,
                "fraction": frac,
                "delta": float(d.mean()),
                "se": float(boot.std(ddof=1)),
                "n_cells_paired": int(d.size),
                "n_cells_arm": int(len(g)),
                "n_cells_incumbent": int(len(inc.loc[inc.index.get_level_values("band") == band])),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["resolved"] = out["delta"].abs() > 2 * out["se"]
    return out


def threshold_spread(frame: pd.DataFrame) -> pd.DataFrame:
    """``sd(threshold)`` ACROSS SEEDS at a fixed (geometry, category, step), banded.

    One of the issue's four metrics, and the one that says whether a fraction
    buys *stability* rather than level.  It needs >= 2 seeds at the same step to
    exist at all, which is why the grid runs four.
    """
    d = frame.dropna(subset=["threshold"]).copy()
    d["band"] = band_of(d["t"])
    d = d[d["band"].notna()]
    per_step = (
        d.groupby(["geometry", "mode", "band", "arm", "fraction", "dataset", "category", "t"], dropna=False)[
            "threshold"
        ]
        .agg(["std", "count"])
        .reset_index()
    )
    per_step = per_step[per_step["count"] >= 2]
    return (
        per_step.groupby(["geometry", "mode", "band", "arm", "fraction"], dropna=False)["std"]
        .agg(sd_threshold="mean", n_steps="size")
        .reset_index()
    )


def trap_check(frame: pd.DataFrame) -> pd.DataFrame:
    """The decomposition the issue warns about, computed only to be disbelieved.

    ``rule_inefficiency + calibration_shift == regret`` by construction, and
    ``calibration_shift`` is measured against a ``cal_oracle_cost`` estimated on
    the very set this knob resizes.  So the two terms are expected to move
    against each other with their sum pinned, and their correlation across arms
    is the evidence that a per-term reading would be algebra.  Reported as a
    correlation, not as a decomposition.
    """
    need = {"rule_inefficiency", "calibration_shift", "regret"}
    if not need <= set(frame.columns):
        return pd.DataFrame()
    d = frame.dropna(subset=list(need)).copy()
    d["band"] = band_of(d["t"])
    d = d[d["band"].notna()]
    keys = [*CELL_KEYS, "mode", "band", "arm", "fraction"]
    m = d.groupby(keys, dropna=False)[list(need)].mean().reset_index()
    rows: list[dict] = []
    for (geom, band), g in m.groupby(["geometry", "band"], dropna=False):
        per_arm = g.groupby("fraction")[["rule_inefficiency", "calibration_shift", "regret"]].mean()
        if len(per_arm) < 3:
            continue
        rows.append(
            {
                "geometry": geom,
                "band": band,
                "corr_terms_across_arms": float(per_arm["rule_inefficiency"].corr(per_arm["calibration_shift"])),
                "max_abs_sum_minus_regret": float(
                    (per_arm["rule_inefficiency"] + per_arm["calibration_shift"] - per_arm["regret"]).abs().max()
                ),
                "spread_rule_inefficiency": float(
                    per_arm["rule_inefficiency"].max() - per_arm["rule_inefficiency"].min()
                ),
                "spread_calibration_shift": float(
                    per_arm["calibration_shift"].max() - per_arm["calibration_shift"].min()
                ),
                "spread_regret": float(per_arm["regret"].max() - per_arm["regret"].min()),
            }
        )
    return pd.DataFrame(rows)


def verdict_by_geometry(paired: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The same rule, applied per GEOMETRY instead of per voting mode.

    Reported beside the per-mode table because the per-mode table can hide a
    disagreement inside itself.  "Binary" here is two geometries -
    ``siglip/whole_image`` and ``dinov3_patch/whole_image`` - and if they want
    different fractions then their pooled verdict is an average across exactly
    the kind of crossing this study bands its axis to avoid.  #3115 made the
    same discovery one level up: what looked like a law about voting mode was
    two cells that happened to agree.

    A per-mode default is only readable off the per-mode row when the
    geometries under it point the same way; when they do not, the honest
    finding is the disagreement.
    """
    rows: list[dict] = []
    if paired.empty:
        return pd.DataFrame()
    for (geom, frac), g in paired.groupby(["geometry", "fraction"], dropna=False):
        w = 1.0 / np.square(g["se"].to_numpy(dtype=float))
        d = g["delta"].to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(d) & (w > 0)
        pooled = float(np.sum(d[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
        pooled_se = float(np.sqrt(1.0 / np.sum(w[ok]))) if ok.any() else float("nan")
        worst = float(g["delta"].max())
        rows.append(
            {
                "geometry": geom,
                "mode": voting_mode(str(geom)),
                "fraction": frac,
                "metric": metric,
                "pooled_delta": pooled,
                "pooled_se": pooled_se,
                "beats_incumbent": bool(pooled < -2 * pooled_se),
                "worst_band_delta": worst,
                "worst_band": str(g.loc[g["delta"].idxmax(), "band"]),
                # How much room the pointwise gate had, signed: negative means
                # the arm cleared it, positive means it failed.  Printed because
                # "passed" and "passed by 2e-5" are different facts and only one
                # of them is a decision (see `_gate_margin_note`).
                "harm_margin": worst - HARM_TOLERANCE,
                "harms_a_band": bool(worst > HARM_TOLERANCE),
                "candidate": bool(pooled < -2 * pooled_se and worst <= HARM_TOLERANCE),
            }
        )
    return pd.DataFrame(rows).sort_values(["geometry", "fraction"]).reset_index(drop=True)


def gate_is_indeterminate(worst: float, worst_se: float) -> bool:
    """Is the pointwise harm gate actually decided, or did it land on the line?

    The gate asks whether an arm's worst band exceeds :data:`HARM_TOLERANCE`.
    That is a comparison between a measured number and a constant, and it is
    only a decision when the number is far enough from the constant to be
    distinguished from it.  An arm whose worst band is 0.00998 +/- 0.0069
    "passes" by 2e-5 - a margin four hundred times smaller than its own standard
    error - and reporting that as passing is precisely the false precision the
    two-significant-digit rule exists to stop.
    """
    return abs(worst - HARM_TOLERANCE) < 2 * worst_se


def verdict(paired: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The pre-registered decision, per voting mode.

    An arm is a candidate for that mode's default when it (a) beats 0.5 by more
    than 2 SE pooled over the bands it is resolved in, and (b) is not worse than
    0.5 by more than :data:`HARM_TOLERANCE` in ANY band.  (b) is the reason this
    reads pointwise and not on an average: an arm can win overall while being
    worse everywhere a short session actually lives, and a short session is most
    sessions.
    """
    rows: list[dict] = []
    if paired.empty:
        return pd.DataFrame()
    for (mode, frac), g in paired.groupby(["mode", "fraction"], dropna=False):
        # Inverse-variance pooling across bands, which weights a band by how
        # well it was measured rather than by how many steps it happens to span.
        w = 1.0 / np.square(g["se"].to_numpy(dtype=float))
        d = g["delta"].to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(d) & (w > 0)
        pooled = float(np.sum(d[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
        pooled_se = float(np.sqrt(1.0 / np.sum(w[ok]))) if ok.any() else float("nan")
        worst_i = g["delta"].idxmax()
        worst = float(g["delta"].max())
        worst_se = float(g.loc[worst_i, "se"])
        rows.append(
            {
                "mode": mode,
                "fraction": frac,
                "metric": metric,
                "pooled_delta": pooled,
                "pooled_se": pooled_se,
                "beats_incumbent": bool(pooled < -2 * pooled_se),
                "worst_band_delta": worst,
                "worst_band_se": worst_se,
                "worst_band": str(g.loc[worst_i, "band"]),
                "harm_margin": worst - HARM_TOLERANCE,
                "harms_a_band": bool(worst > HARM_TOLERANCE),
                # A gate that lands within 2 SE of its own threshold decided
                # nothing; the boolean beside it is then an artefact of where
                # the noise fell, not a result.
                "gate_indeterminate": gate_is_indeterminate(worst, worst_se),
                "candidate": bool(pooled < -2 * pooled_se and worst <= HARM_TOLERANCE),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "fraction"]).reset_index(drop=True)


def figures(frame: pd.DataFrame, outdir: Path, arms: Sequence[str], baseline_csv: Path | None) -> list[str]:
    """The mandatory quality-over-clicks pair, drawn by the one implementation.

    ``curves`` panels by ``dataset`` and colours by ``arm``.  This study has ONE
    dataset and three geometries, and the geometry is what a reader needs held
    apart -- averaging ``max_patch`` with ``whole_image`` describes no system
    anyone could run -- so the panel column is set to the geometry and the
    caption says so.  The hue stays the arm, which is the fraction, which is the
    thing being compared.
    """
    d = frame.copy()
    # `-`, not the `/` the tables use: `per_run_figures` puts the panel column
    # into the FILENAME (`cost_vs_clicks_runs__<dataset>.png`), so a geometry
    # written `dinov3_patch/max_patch` asks matplotlib to save into a directory
    # that does not exist.  It fails at `savefig`, i.e. after every cell has
    # been paid for and the averaged figure has already been written -- which is
    # exactly the kind of late crash the smoke test exists to move forward.
    d["dataset"] = d["dataset"].astype(str) + " · " + d["geometry"].astype(str).str.replace("/", "-", regex=False)
    baseline = curves.text_sort_baseline(baseline_csv) if baseline_csv and Path(baseline_csv).exists() else None
    if baseline is not None:
        # The baseline is keyed by the real dataset name; re-key it onto the
        # per-geometry panels so every panel gets its own click-0 anchor rather
        # than silently losing it.
        reps = []
        for geom in sorted(d["geometry"].unique()):
            b = baseline.copy()
            b["dataset"] = b["dataset"].astype(str) + " · " + geom.replace("/", "-")
            reps.append(b)
        baseline = pd.concat(reps, ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    denominator = d[[*CELL_KEYS, "dataset", "embedder"]].drop_duplicates()
    for metric in ("cost", "average_precision"):
        if metric not in d.columns:
            continue
        written += curves.quality_vs_clicks(
            d,
            outdir,
            arms=list(arms),
            metric=metric,
            denominator=denominator,
            baseline=baseline,
            lower_is_better=(metric == "cost"),
        )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="dir holding one f0NN/ arm dir per fraction")
    ap.add_argument("--out", required=True, help="where the report, tables and figures go")
    ap.add_argument("--fractions", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--baseline", default=None, help="text_baseline.py CSV: the click-0 anchor")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    base = Path(args.base)
    out = Path(args.out)
    (out / "agg").mkdir(parents=True, exist_ok=True)
    fractions = [float(x) for x in args.fractions.replace(",", " ").split()]
    if INCUMBENT not in fractions:
        raise SystemExit(f"{INCUMBENT} must be one of --fractions: every contrast here is a difference from it")

    frame, prov = load_arms(base, fractions)
    arms = [f"{f:.2f}" for f in fractions]
    rng = np.random.default_rng(20260827)

    # --- completeness, counted before anything is averaged --------------------
    counts = (
        frame[["arm", *CELL_KEYS]]
        .drop_duplicates()
        .groupby(["arm", "geometry"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    counts.to_csv(out / "agg" / "cell_counts.csv", index=False)
    (out / "agg" / "provenance.json").write_text(json.dumps(prov, indent=2))

    # --- levels ---------------------------------------------------------------
    levels = []
    for metric in ("cost", "regret_honest", "regret", "n_cal_scores"):
        if metric not in frame.columns:
            continue
        cm = cell_means(frame, metric)
        lv = (
            cm.groupby(["geometry", "mode", "band", "arm", "fraction"], dropna=False)[metric]
            .agg(mean="mean", sd="std", n_cells="size")
            .reset_index()
        )
        lv["metric"] = metric
        levels.append(lv)
    by_band = pd.concat(levels, ignore_index=True) if levels else pd.DataFrame()
    by_band.to_csv(out / "agg" / "by_band.csv", index=False)

    # --- the contrasts --------------------------------------------------------
    paired_all = []
    verdicts = []
    verdicts_geom: list[pd.DataFrame] = []
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
        verdicts_geom.append(verdict_by_geometry(pv, metric))
    paired = pd.concat(paired_all, ignore_index=True) if paired_all else pd.DataFrame()
    paired.to_csv(out / "agg" / "paired_vs_incumbent.csv", index=False)
    vd = pd.concat(verdicts, ignore_index=True) if verdicts else pd.DataFrame()
    vd.to_csv(out / "agg" / "verdict.csv", index=False)
    vg = pd.concat(verdicts_geom, ignore_index=True) if verdicts_geom else pd.DataFrame()
    vg.to_csv(out / "agg" / "verdict_by_geometry.csv", index=False)

    spread = threshold_spread(frame)
    spread.to_csv(out / "agg" / "sd_threshold.csv", index=False)
    trap = trap_check(frame)
    trap.to_csv(out / "agg" / "trap_check.csv", index=False)

    figs: list[str] = []
    if not args.no_figures:
        figs = figures(frame, out / "figures", arms, Path(args.baseline) if args.baseline else None)

    if not args.no_viewer:
        import viewer

        d = frame.copy()
        # Embedders are never averaged in the viewer, so the geometry goes in
        # that slot: it is the dimension that must not be pooled here.
        d["embedder"] = d["geometry"]
        # Same existence guard the figures use: a missing baseline costs the
        # click-0 anchor, and losing the anchor must not cost the whole page.
        baseline = curves.text_sort_baseline(args.baseline) if args.baseline and Path(args.baseline).exists() else None
        viewer.build_viewer(
            d,
            out / "viewer.html",
            arms=arms,
            baseline=baseline,
            title="calibration_fraction (#3287)",
            subtitle="One chart per calibration_fraction × geometry × category; tick 'overlay on one chart' to compare them directly.",
        )

    write_report(out, frame, by_band, paired, vd, vg, spread, trap, counts, prov, figs)
    print(f"wrote {out / 'REPORT_calfrac.md'}")
    return 0


def _md(df: pd.DataFrame) -> str:
    """Markdown table when ``tabulate`` is available, else a fixed-width dump.

    Floats are rounded to **three significant digits** on the way in rather than
    formatted on the way out, so the CSV keeps full precision and the report
    shows only what the sample supports.  Four decimals are not more rigorous;
    they are harder to read and they invent findings (`grid-experiments`).
    """
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: float(f"{v:.3g}") if np.isfinite(v) else v)
    try:
        return d.to_markdown(index=False)
    except Exception:  # noqa: BLE001 - tabulate not installed on the grid venv
        return "```\n" + d.to_string(index=False) + "\n```"


def _fmt(x: float, digits: int = 2) -> str:
    """Two significant digits by default: a third invents findings (#3129 lesson)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}g}"


def write_report(out, frame, by_band, paired, vd, vg, spread, trap, counts, prov, figs) -> None:
    L: list[str] = []
    A = L.append
    A("# What Train/Calibrate split should a detector use? (#3287)\n")
    A("Draft written by `analyze_calfrac.py`; every number below comes from `agg/*.csv`.\n")
    A("\nThe question: of the votes a user has cast, what share should **train** each\n")
    A("calibration fold's model, and what share should be held out to **read its threshold**?\n")
    A("The shipped answer is half and half. This measures what the user's cost would have been\n")
    A("at five different splits, under a simulation held as close to the app as the harness\n")
    A("allows — the same fused threshold path, the same production head, the same opening.\n")

    A("\n## What was run\n")
    A(f"- Arms: `calibration_fraction` ∈ {sorted(frame['fraction'].unique())}, one **full run** each.")
    A("- One dataset (`vg_scale_any`), 12 classes at identical prevalence, 4 seeds, 150 steps.")
    A("- Three geometries: `siglip/whole_image`, `dinov3_patch/whole_image`, `dinov3_patch/max_patch`.")
    A("  The middle one is what separates the **voting mode** from the **embedder**.")
    A("- Opening: a SigLIP text sort in every cell, including the region ones (the `siglip+dinov3_patch` pair).\n")

    A("\n### Cells read, and cells lost\n")
    A(_md(counts))
    A("\n")
    for arm, p in prov.items():
        lost = [k for k in ("unreadable", "zero_byte", "no_positive_found", "no_base_rows") if p.get(k)]
        if lost:
            A(f"- arm {arm}: " + ", ".join(f"{k}={len(p[k])}" for k in lost))
    A("\n")

    A("\n## The decision\n")
    if vd.empty:
        A("No arm produced a computable contrast.\n")
    else:
        A(_md(vd))
        A("\n")
        for mode in sorted(vd["mode"].unique()):
            g = vd[(vd["mode"] == mode) & (vd["metric"] == "cost")]
            cands = g[g["candidate"]]
            if cands.empty:
                A(
                    f"\n**{mode} voting: keep 0.5.** No fraction beats it by more than twice its "
                    f"standard error without harming a band by more than {HARM_TOLERANCE}."
                )
            else:
                best = cands.loc[cands["pooled_delta"].idxmin()]
                A(
                    f"\n**{mode} voting: {best['fraction']} is a candidate** "
                    f"({_fmt(best['pooled_delta'])} ± {_fmt(best['pooled_se'])} on cost, "
                    f"worst band {_fmt(best['worst_band_delta'])})."
                )
        A("\n")

    A("\n### The same rule, per geometry\n")
    if not vg.empty:
        A(_md(vg[vg["metric"] == "cost"].drop(columns=["metric"])))
        A("\n\nA per-mode default is only readable off the table above when the geometries under it\n")
        A("point the same way. Where they disagree, the disagreement is the finding — #3115 made the\n")
        A("same discovery one level up, where a law about voting mode turned out to be two cells that\n")
        A("happened to agree.\n")

    if not vd.empty and bool(vd.get("gate_indeterminate", pd.Series(dtype=bool)).any()):
        A("\n### Where the pointwise gate decided nothing\n")
        g = vd[vd["gate_indeterminate"]]
        A(_md(g[["mode", "fraction", "metric", "worst_band", "worst_band_delta", "worst_band_se", "harm_margin"]]))
        A("\n\nThese arms' worst band lands within 2 SE of the 0.01 tolerance itself, so the\n")
        A("`harms_a_band` boolean beside them is an artefact of where the noise fell rather than a\n")
        A("decision. Read them as **undecided on harm**, whichever way the boolean points.\n")

    A("\n## Read across vote bands, not pooled\n")
    A("The trade-off is predicted to reverse with labelset size, so the banded table is the result and\n")
    A("the pooled row above is a summary of it.\n\n")
    if not paired.empty:
        p = paired[paired["metric"] == "cost"].copy()
        p["Δcost ± SE"] = p.apply(lambda r: f"{_fmt(r['delta'])} ± {_fmt(r['se'])}", axis=1)
        A(_md(p[["mode", "geometry", "band", "fraction", "Δcost ± SE", "resolved", "n_cells_paired"]]))
        A("\n\nPaired within `(dataset, category, seed, geometry)`; the bootstrap resamples **cells**,\n")
        A("never steps, because consecutive steps of one trajectory share a model.\n")

    A("\n## Threshold stability\n")
    if not spread.empty:
        A(_md(spread))
        A("\n\n`sd(threshold)` is taken across the 4 seeds at a fixed (category, step) and then averaged\n")
        A("over the band. It answers a different question from the level: whether a fraction buys\n")
        A("*repeatability* even where it does not buy cost.\n")

    A("\n## The decomposition trap, shown rather than described\n")
    if trap.empty:
        A("Not computable on this frame.\n")
    else:
        A(_md(trap))
        A("\n\n`calibration_shift` is measured against `cal_oracle_cost`, which is estimated **from the\n")
        A("calibration set this knob resizes**. So the yardstick moves with the arm, the two terms slide\n")
        A("against each other, and their sum stays pinned to `regret` by construction — which is what\n")
        A("`max_abs_sum_minus_regret` (≈0) and the negative correlation above are showing. #2897 read\n")
        A("that anti-correlation as a finding. It is algebra, and no per-term claim is made here.\n")

    A("\n## Figures\n")
    for f in figs:
        A(f"\n![{f}](figures/{f})\n")
    A("\nPanels are **geometries**, not datasets (there is one dataset): `max_patch` and `whole_image`\n")
    A("must not be averaged, and the panel is what keeps them apart. Colour is the fraction. A line is\n")
    A("dashed wherever it describes fewer than 95% of that arm's cells — only a solid segment is a\n")
    A("level worth quoting. Click 0 is the free text sort.\n")
    A("\nEvery slice, including the ones these PNGs do not show: [`viewer.html`](viewer.html).\n")

    Path(out / "REPORT_calfrac.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

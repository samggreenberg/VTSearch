#!/usr/bin/env python3
"""#4864: the calibration split's rounding, ``nearest`` against the app's ``dither``, decided by the issue's rule.

Reads the four arms ``launch_split_rounding_4864.sh`` writes (binary ``dither`` /
``nearest`` on #4583's bench, region ``r_dither`` / ``r_nearest`` on #4583's
region bench) and answers the three measurements pre-registered on the issue:

1. **Δ-objective**, ``nearest - dither``: F-beta at beta 1 of the withheld half
   above the line the app holds (#4427), paired on (category, seed), SE
   clustered on category, at #4583's two readings (the filled curve's mean over
   votes 1-150, and after the check), with AP beside it.  Reuses #4583's
   per-arm tables (``analyze_calsplit_4583.arm_tables``) so the readings are
   literally #4583's.
2. **Ripple.**  Each arm's mean curve, high-passed by a centred 20-vote mean
   (20 votes hold a whole number of cycles at every frequency tested, so the
   filter leaves them untouched), and the paired Δ curve the same way.  Its
   amplitude at the split's own frequencies (0.3, 0.1, 0.4 cycles a vote at a
   0.3 split; 0.5 and 0.25 at 0.5, step 0's) is tested against a sign-flip
   null that flips every run of a category together.  The full periodogram is
   written with the null's median and 95th percentile per frequency.  The
   split-input decomposition (#3286's) runs on each arm's own recorded split
   sizes, ``model_n_pos`` / ``model_n_neg`` from the line-counts side frame.
3. **The band.**  Rows with 7-11 Goods (4-6 on the patch space): each run's
   mean objective there, paired, and each arm's switches between the count line
   (``model_n_pos < HEAD_COUNT_GOODS``) and the labels line per run, over the
   whole run and inside the band.

**Decision rule (the issue's, pre-registered).**  Ship ``nearest`` if, on both
paths and at both readings, (1) is no worse than -0.003 and (2) shows no periodic
component above 2x noise at the expected periods.  Keep the dither if (1) is
worse than -0.003 by more than 2 SE on either path.  Owner's call if quality
ties but (2) shows a ripple.

    python analyze_split_rounding_4864.py --base /expscratch/$USER/splitround-4864 \\
        --baseline /expscratch/$USER/progression-4184-h0.01/text_baseline.csv --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import analyze_calsplit_4583 as cs  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import split_cadence_4864 as sc  # noqa: E402

from vtscore.training.thresholds.labels_line import HEAD_COUNT_GOODS  # noqa: E402

RUN = ["category", "seed"]
HORIZON = cs.HORIZON
#: (arm dir under --base, path, split fraction): the shipped split of each path's space.
ARMS: dict[str, tuple[str, str, float]] = {
    "dither": ("dither", "binary", 0.3),
    "nearest": ("nearest", "binary", 0.3),
    "r_dither": ("region/r_dither", "region", 0.5),
    "r_nearest": ("region/r_nearest", "region", 0.5),
}
PAIRS: dict[str, tuple[str, str]] = {"binary": ("nearest", "dither"), "region": ("r_nearest", "r_dither")}
#: The split's own frequencies on each path (cycles a vote), from step 0.
TEST_FREQS: dict[str, list[float]] = {"binary": [0.3, 0.1, 0.4], "region": [0.5, 0.25]}
#: The Goods counts where the app's coin decides the line: the issue's 7-11 on a single-vector
#: space (the "ragged ends" trimmed), and around step 0's one row at 5 on the patch space.
BANDS: dict[str, tuple[int, int]] = {"binary": (7, 11), "region": (4, 6)}
#: The ship rule's tolerance on the objective.
TOLERANCE = 0.003
#: The ship rule's ripple bar: amplitude over the sign-flip null's median.
RIPPLE_X = 2.0
#: Centred high-pass window, votes: a whole number of cycles at 0.1, 0.25, 0.3, 0.4 and 0.5.
HIGHPASS = 20
N_FLIPS = 2000
N_BOOT = 1000
READS = ("objective, votes 1-150", "objective, after the check")
#: Post hoc (read after the first look at the curves): the votes right after the
#: first detector shows, where the two rules' lines differ most.
HANDOVER = {"votes 31-45 (post hoc)": (31, 45)}


def highpass(wide: pd.DataFrame) -> pd.DataFrame:
    """Each row minus its centred HIGHPASS-vote mean; the edges, where the window does not fit, dropped."""
    smooth = wide.T.rolling(HIGHPASS, center=True, min_periods=HIGHPASS).mean().T
    out = wide - smooth
    return out.loc[:, smooth.notna().all(axis=0)]


def ripple(wide: pd.DataFrame, freqs: Sequence[float], rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Shared periodic structure in the runs' curves (rows = runs, columns = votes) against category sign flips."""
    hp = highpass(wide.loc[:, 1:HORIZON])
    t = np.asarray(hp.columns, dtype=float)
    values = hp.to_numpy(dtype=float)
    present = ~np.isnan(values)
    filled = np.nan_to_num(values)
    counts = np.maximum(present.sum(axis=0), 1)
    cats, cat_index = np.unique(hp.index.get_level_values("category").to_numpy(), return_inverse=True)
    grid = np.arange(1, len(t) // 2 + 1) / len(t)
    test = np.asarray(freqs, dtype=float)
    all_f = np.concatenate([grid, test])

    def amp(signs: np.ndarray) -> np.ndarray:
        return sc._amplitude((filled * signs[:, None]).sum(axis=0) / counts, t, all_f)

    obs = amp(np.ones(len(values)))
    null = np.stack([amp(rng.choice([-1.0, 1.0], size=len(cats))[cat_index]) for _ in range(N_FLIPS)])
    # A 95% interval on each tested amplitude: categories resampled with replacement.
    by_cat = [np.flatnonzero(cat_index == k) for k in range(len(cats))]
    boot = np.empty((N_BOOT, len(test)))
    for k in range(N_BOOT):
        idx = np.concatenate([by_cat[c] for c in rng.integers(0, len(cats), len(cats))])
        boot[k] = sc._amplitude(filled[idx].sum(axis=0) / np.maximum(present[idx].sum(axis=0), 1), t, test)
    spectrum = pd.DataFrame(
        {
            "freq": grid,
            "amplitude": obs[: len(grid)],
            "null_median": np.median(null[:, : len(grid)], axis=0),
            "null_p95": np.quantile(null[:, : len(grid)], 0.95, axis=0),
        }
    )
    tests = {}
    for j, f in enumerate(test):
        a, nj = obs[len(grid) + j], null[:, len(grid) + j]
        tests[f"{f:g}"] = {
            "amplitude": float(a),
            "null_median": float(np.median(nj)),
            "x_noise": float(a / np.median(nj)),
            "p": float((1 + np.sum(nj >= a)) / (1 + N_FLIPS)),
            "ci_lo": float(np.quantile(boot[:, j], 0.025)),
            "ci_hi": float(np.quantile(boot[:, j], 0.975)),
        }
    return {"votes": [int(t[0]), int(t[-1])], "runs": int(len(values)), "tests": tests}, spectrum


def load_linecounts(cells: Path) -> pd.DataFrame:
    cols = ["seed", "category", "style", "t", "phase", "n_good", "n_bad", "model_n_pos", "model_n_neg"]
    frames = []
    import _cells_io  # noqa: PLC0415

    for f in _cells_io.side_frame_files(cells, "__linecounts"):
        try:
            frames.append(pd.read_csv(f, usecols=lambda c: c in cols))
        except (OSError, ValueError, EOFError, pd.errors.EmptyDataError):
            continue
    if not frames:
        return pd.DataFrame(columns=cols)
    lc = pd.concat(frames, ignore_index=True)
    for c in ("t", "n_good", "n_bad", "model_n_pos", "model_n_neg"):
        lc[c] = pd.to_numeric(lc[c], errors="coerce")
    lc = lc[lc["phase"].fillna("").astype(str).str.strip() != "check"]
    return lc[lc["t"] <= HORIZON].drop_duplicates(subset=[*RUN, "t"], keep="last").sort_values([*RUN, "t"])


def shown_only(lc: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """The line-count clicks where the app shows the trained detector (``app_trained``).

    The side frame records every click Test gives the trained head, which
    includes clicks the user still sees the typed query's set (the hold); a line
    switch there is not one anyone sees.
    """
    if lc.empty:
        return lc
    shown = frame[frame["phase"].fillna("").astype(str).str.strip() != "check"]
    shown = shown.drop_duplicates(subset=[*RUN, "t"], keep="last")[[*RUN, "t", "app_trained"]]
    out = lc.merge(shown, on=[*RUN, "t"], how="left")
    return out[out["app_trained"].fillna(0).astype(int) == 1].drop(columns="app_trained")


def in_sample(lc: pd.DataFrame) -> pd.Series:
    """Clicks whose class model is ``fit_labels_line``'s in-sample fallback, not the folds' held-out scores.

    When the held-out Goods do not outscore the held-out Bads the folds imply no
    class model, and the head's own scores of every vote stand in: the model then
    holds every Good and every Bad.  Its ``n_pos`` is all the Goods, so from
    ``HEAD_COUNT_GOODS`` Goods it passes the thin-model gate whatever the split
    held out.  On a 0.3 split a held-out model holds every vote only at 2 Goods
    and 2 Bads, which is left out of the tag.
    """
    return (lc["model_n_pos"] == lc["n_good"]) & (lc["model_n_neg"] == lc["n_bad"]) & (lc["n_bad"] > 2)


def _switch_count(kinds: np.ndarray) -> int:
    return int(np.abs(np.diff(kinds)).sum()) if len(kinds) > 1 else 0


def switches(lc: pd.DataFrame, band: tuple[int, int], tag_in_sample: bool = True) -> pd.DataFrame:
    """Per run: switches between the count line and the labels line, over the run and inside *band*.

    A click with no class model (``model_n_pos`` NaN: the fallback cut) is left
    out, so a switch is between the two lines the gate chooses from.  The
    ``held_out_`` columns count again over the clicks whose class model is the
    folds' (:func:`in_sample` left out): the switches the split's rounding can
    cause.  *tag_in_sample* is off on the patch space: at a 0.5 split a held-out
    model holds every vote whenever both counts are even, so the tag cannot tell
    the two apart there, and its ``held_out_`` columns repeat the totals.
    """
    rows = []
    lo, hi = band
    lc = lc.dropna(subset=["model_n_pos"])
    lc = lc.assign(in_sample=in_sample(lc) if tag_in_sample else False)
    for run, g in lc.groupby(RUN, sort=False):
        labels = (g["model_n_pos"].to_numpy() >= HEAD_COUNT_GOODS).astype(int)
        band_rows = ((g["n_good"] >= lo) & (g["n_good"] <= hi)).to_numpy()
        held = ~g["in_sample"].to_numpy()
        rows.append(
            {
                **dict(zip(RUN, run)),
                "switches": _switch_count(labels),
                "band_switches": _switch_count(labels[band_rows]),
                "band_clicks": int(band_rows.sum()),
                "labels_share": float(labels.mean()),
                "in_sample_clicks": int((~held).sum()),
                "held_out_switches": _switch_count(labels[held]),
                "held_out_band_switches": _switch_count(labels[held & band_rows]),
            }
        )
    return pd.DataFrame(rows)


def band_objective(frame: pd.DataFrame, band: tuple[int, int]) -> pd.Series:
    """Each run's mean objective over its own votes with *band* Goods and a trained detector on screen."""
    lo, hi = band
    v = frame[frame["phase"].fillna("").astype(str).str.strip() != "check"]
    v = v[(v["app_trained"].fillna(0).astype(int) == 1) & (v["t"] <= HORIZON)]
    v = v[(v["n_good"] >= lo) & (v["n_good"] <= hi)].copy()
    v["F"] = cs.fbeta(v["precision"], v["recall"], 1.0)
    return v.groupby(RUN)["F"].mean()


def coin_experiment(frame: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    """The dither's coin as a randomized trial of the two lines, at each count of Goods (the ``dither`` arm).

    Inside the band the coin, not the run, decides whether a click draws the
    labels line (the folds held out enough Goods) or the head's count line.  So
    at one count of Goods, a run's clicks on either line differ only by the
    coin.  Per run, the mean objective at its labels-line clicks minus at its
    count-line clicks, over the runs that drew both at that count; SE clustered
    on category.  Held-out class models only (:func:`in_sample` left out).
    """
    v = frame[frame["phase"].fillna("").astype(str).str.strip() != "check"]
    v = v[(v["app_trained"].fillna(0).astype(int) == 1) & (v["t"] <= HORIZON)]
    v = v.drop_duplicates(subset=[*RUN, "t"], keep="last").copy()
    v["F"] = cs.fbeta(v["precision"], v["recall"], 1.0)
    held = lc.dropna(subset=["model_n_pos"])
    held = held[~in_sample(held)]
    j = held.merge(v[[*RUN, "t", "F"]], on=[*RUN, "t"], how="inner")
    j["kind"] = np.where(j["model_n_pos"] >= HEAD_COUNT_GOODS, "labels", "count")
    rows = []
    per_g: dict[int, pd.Series] = {}
    for g, sub in j.groupby("n_good"):
        if g < 5 or g > 11:
            continue
        per = sub.groupby([*RUN, "kind"])["F"].mean().unstack("kind")
        both = per.dropna() if {"labels", "count"} <= set(per.columns) else per.iloc[0:0]
        d = (both["labels"] - both["count"]) if len(both) else pd.Series(dtype=float)
        per_g[int(g)] = d
        mean, se, sd, n = cs.clustered(d) if len(d) else (np.nan, np.nan, np.nan, 0)
        rows.append(
            {
                "n_good": int(g),
                "clicks": len(sub),
                "labels_share": float((sub["kind"] == "labels").mean()),
                "runs_both": n,
                "labels_minus_count": mean,
                "se": se,
            }
        )
    # Where nearest draws the count line and the dither sometimes the labels line: each run's
    # differences at 7 and 8 Goods averaged, so a run counts once.
    pooled = pd.concat([per_g[g] for g in (7, 8) if g in per_g and len(per_g[g])])
    if len(pooled):
        mean, se, _, n = cs.clustered(pooled.groupby(level=list(range(pooled.index.nlevels))).mean())
        sub = j[j["n_good"].isin([7, 8])]
        rows.append(
            {
                "n_good": "7-8",
                "clicks": len(sub),
                "labels_share": float((sub["kind"] == "labels").mean()),
                "runs_both": n,
                "labels_minus_count": mean,
                "se": se,
            }
        )
    return pd.DataFrame(rows)


def decomposition(lc: pd.DataFrame, fraction: float, freqs: Sequence[float], rng: np.random.Generator) -> dict:
    """#3286's split-input decomposition on the arm's own recorded split sizes (per fold).

    The class model pools every fold's held-out scores, so ``model_n_pos / FOLDS``
    is one fold's held-out Goods.  On the patch space it counts the held-out
    bags' scores the class model reads, which is not a count of votes, so this
    is read on the binary path only.
    """
    rows = lc.dropna(subset=["model_n_pos", "model_n_neg"]).copy()
    rows = rows[(rows["n_good"] >= 2) & (rows["n_bad"] >= 2) & ~in_sample(rows)]
    if rows.empty:
        return {}
    out = {}
    for q, held, count in (
        ("held_good", rows["model_n_pos"] / sc.FOLDS, rows["n_good"]),
        ("held_bad", rows["model_n_neg"] / sc.FOLDS, rows["n_bad"]),
        ("held_total", (rows["model_n_pos"] + rows["model_n_neg"]) / sc.FOLDS, rows["n_good"] + rows["n_bad"]),
    ):
        rows[f"res_{q}"] = held - fraction * count
        summary, _ = sc.cadence(rows, f"res_{q}", freqs, rng)
        out[q] = summary
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import _cells_io  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=4864)
    args = ap.parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    base = _cells_io.legacy_datasets(pd.read_csv(args.baseline)).groupby(RUN).mean(numeric_only=True)
    anchor = base["text_fbeta_b1"]
    tables: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    lcs: dict[str, pd.DataFrame] = {}
    prov: dict[str, dict] = {}
    problems: list[str] = []
    for arm, (sub, path, frac) in ARMS.items():
        d = args.base / sub / "results"
        if not (d / "cells").exists():
            problems.append(f"{arm}: no cells")
            continue
        frame, p = _cells_io.load_arm(d, keep_check=True)
        lost = len(p.get("zero_byte") or []) + len(p.get("unreadable") or [])
        expected = 720 if path == "binary" else 144
        complete = path == "binary" and p.get("n_files") == expected and lost == 0
        prov[arm] = {
            "load": _cells_io.describe_load(p),
            "n_files": p.get("n_files"),
            "expected": expected,
            "complete": complete,
        }
        if p.get("n_files") != expected or lost:
            problems.append(f"{arm}: {p.get('n_files')} of {expected} cells, {lost} lost")
        if frame.empty:
            problems.append(f"{arm}: no rows")
            continue
        # The arm ran what its name says, read back off its own rows.
        want = "nearest" if "nearest" in arm else "dither"
        got = sorted({str(x) for x in frame.get("split_rounding", pd.Series(dtype=str)).dropna().unique()})
        if got != [want]:
            problems.append(f"{arm}: split_rounding {got}, expected {want}")
        fr = sorted(
            {
                round(float(x), 6)
                for x in pd.to_numeric(frame["calibration_fraction"], errors="coerce").dropna().unique()
            }
        )
        if fr != [frac]:
            problems.append(f"{arm}: calibration_fraction {fr}, expected {frac}")
        betas = sorted({round(float(x), 6) for x in pd.to_numeric(frame["beta"], errors="coerce").dropna().unique()})
        if betas != [1.0]:
            problems.append(f"{arm}: beta {betas}, expected 1")
        tables[arm] = cs.arm_tables(frame, anchor, 1.0, complete)
        frames[arm] = frame
        lcs[arm] = shown_only(load_linecounts(d / "cells"), frame)
        print(f"{arm}: {prov[arm]['load']}; {len(lcs[arm])} line-count rows", flush=True)

    paired_rows, ripple_rows, spectra, decomp, sw_all = [], [], [], {}, []
    for path, (a, b) in PAIRS.items():
        if a not in tables or b not in tables:
            continue
        ta, tb = tables[a], tables[b]
        reads: dict[str, pd.Series] = {}
        for w, (lo, hi) in {**cs.WINDOWS, **HANDOVER}.items():
            reads[f"objective, {w}"] = cs.window_mean(ta["F"], lo, hi) - cs.window_mean(tb["F"], lo, hi)
            reads[f"AP, {w}"] = cs.window_mean(ta["ap"], lo, hi) - cs.window_mean(tb["ap"], lo, hi)
        reads["objective, after the check"] = ta["after"] - tb["after"]
        reads["returned set size, after the check"] = ta["k_after"] - tb["k_after"]
        reads["precision, after the check"] = ta["p_after"] - tb["p_after"]
        band = BANDS[path]
        reads[f"objective, {band[0]}-{band[1]} Goods"] = band_objective(frames[a], band) - band_objective(
            frames[b], band
        )
        for read, dlt in reads.items():
            strata = {"all": dlt}
            if path == "binary":
                cat = cs.band_of(dlt.index.get_level_values("category").to_series()).to_numpy()
                strata.update({f"band={s}": dlt[cat == s] for s in ("small", "medium", "large")})
            for stratum, dd in strata.items():
                mean, se, sd, n = cs.clustered(dd)
                paired_rows.append(
                    {"path": path, "read": read, "stratum": stratum, "n": n, "delta": mean, "se": se, "sd": sd}
                )
        # Ripple: each arm's own curve, and the paired difference.
        curves = {arm: tables[arm]["F"] for arm in (a, b)}
        curves["nearest - dither"] = (ta["F"] - tb["F"]).dropna(how="all")
        for name, wide in curves.items():
            summary, spec = ripple(wide, TEST_FREQS[path], rng)
            spectra.append(spec.assign(path=path, curve=name))
            for f, v in summary["tests"].items():
                ripple_rows.append({"path": path, "curve": name, "freq": float(f), **v, "votes": summary["votes"]})
        for arm in (a, b):
            if path == "binary":
                decomp[arm] = decomposition(lcs[arm], ARMS[arm][2], TEST_FREQS[path], rng)
            sw_all.append(switches(lcs[arm], band, tag_in_sample=path == "binary").assign(arm=arm, path=path))

    coin = coin_experiment(frames["dither"], lcs["dither"]) if "dither" in frames else pd.DataFrame()
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["resolved"] = paired["delta"].abs() > 2 * paired["se"]
    rip = pd.DataFrame(ripple_rows)
    sw = pd.concat(sw_all, ignore_index=True) if sw_all else pd.DataFrame()

    # The pre-registered rule.
    verdict: dict[str, object] = {}
    lines_rule = []
    worse_by_2se = False
    quality_ok = True
    for path in PAIRS:
        for read in READS:
            r = paired[(paired.path == path) & (paired.read == read) & (paired.stratum == "all")]
            if r.empty:
                quality_ok = False
                continue
            d, se = float(r["delta"].iloc[0]), float(r["se"].iloc[0])
            ok = d >= -TOLERANCE
            quality_ok &= ok
            # "worse than -0.003 by more than 2 SE": the gap below -0.003 exceeds 2 SE.
            if d < -TOLERANCE - 2 * se:
                worse_by_2se = True
            lines_rule.append(f"- {path}, {read}: Δ = {d:+.4f} ± {se:.4f} -> {'ok' if ok else 'WORSE than -0.003'}")
    ripple_hits = rip[(rip.curve.isin(["nearest", "r_nearest"])) & (rip.x_noise > RIPPLE_X)] if not rip.empty else rip
    for _, r in rip[rip.curve.isin(["nearest", "r_nearest"])].iterrows():
        lines_rule.append(
            f"- ripple, {r.path} `{r.curve}` at {r.freq:g}/vote: {r.x_noise:.1f}x noise (p = {r.p:.3g})"
            + (" -> ABOVE 2x" if r.x_noise > RIPPLE_X else "")
        )
    if worse_by_2se:
        verdict["rule"] = "keep the dither"
    elif quality_ok and ripple_hits.empty:
        verdict["rule"] = "ship nearest"
    elif quality_ok:
        verdict["rule"] = "owner's call: quality ties, ripple above 2x noise"
    else:
        verdict["rule"] = "no rule fires: quality below -0.003 but not by 2 SE"
    verdict["detail"] = lines_rule

    paired.to_csv(args.out / "paired.csv", index=False, float_format="%.5g")
    rip.to_csv(args.out / "ripple.csv", index=False, float_format="%.5g")
    pd.concat(spectra).to_csv(args.out / "spectra.csv", index=False, float_format="%.5g")
    sw.to_csv(args.out / "switches.csv", index=False)
    coin.to_csv(args.out / "coin.csv", index=False, float_format="%.5g")
    curves_out = pd.DataFrame({arm: t["F"].mean(axis=0) for arm, t in tables.items()})
    curves_out.index.name = "t"
    curves_out.to_csv(args.out / "curves.csv", float_format="%.5g")
    delta_curves = {}
    for path, (a, b) in PAIRS.items():
        if a in tables and b in tables:
            dlt = (tables[a]["F"] - tables[b]["F"]).dropna(how="all")
            g = dlt.groupby(level="category").mean()
            delta_curves[f"{path}:mean"] = dlt.mean(axis=0)
            delta_curves[f"{path}:se"] = g.std(ddof=1) / np.sqrt(len(g))
    pd.DataFrame(delta_curves).rename_axis("t").to_csv(args.out / "delta_curves.csv", float_format="%.5g")
    (args.out / "decomposition.json").write_text(json.dumps(decomp, indent=2) + "\n")
    (args.out / "provenance.json").write_text(
        json.dumps({"arms": prov, "problems": problems, "verdict": verdict}, indent=2)
    )

    pd.set_option("display.width", 250)
    lines = ["# #4864 - machine summary", ""]
    lines += [f"- `{a}`: {v['load']}" for a, v in prov.items()]
    if problems:
        lines += ["", "## Premise failures", "", *[f"- {x}" for x in problems]]
    lines += ["", f"## Decision rule: **{verdict['rule']}**", "", *lines_rule, ""]
    head = paired[paired.stratum == "all"]
    lines += ["## Paired nearest - dither (all bands)", "", head.round(4).to_string(index=False), ""]
    if not rip.empty:
        lines += [
            "## Ripple at the split's frequencies",
            "",
            rip.drop(columns=["votes"]).round(4).to_string(index=False),
            "",
        ]
    if not sw.empty:
        cols = ["switches", "band_switches", "held_out_switches", "held_out_band_switches", "band_clicks"]
        s = sw.groupby("arm")[[*cols, "labels_share", "in_sample_clicks"]].mean()
        s["runs"] = sw.groupby("arm").size()
        s["runs switching 2+ times"] = sw.assign(m=sw.switches >= 2).groupby("arm")["m"].mean()
        s["2+ on held-out clicks"] = sw.assign(m=sw.held_out_switches >= 2).groupby("arm")["m"].mean()
        s["runs with an in-sample click"] = sw.assign(m=sw.in_sample_clicks > 0).groupby("arm")["m"].mean()
        lines += ["## Count line <-> labels line switches per run", "", s.round(3).to_string(), ""]
    if not coin.empty:
        lines += [
            "## The coin as a trial: labels line - count line at one count of Goods (dither)",
            "",
            coin.round(4).to_string(index=False),
            "",
        ]
    for arm, dd in decomp.items():
        for q, v in dd.items():
            tests = ", ".join(f"{f}: {x['x_noise']:.1f}x (p={x['p']:.3g})" for f, x in v["tests"].items())
            lines.append(f"- split sizes, `{arm}` {q}: {tests}")
    (args.out / "REPORT_machine.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

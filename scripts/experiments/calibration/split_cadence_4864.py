#!/usr/bin/env python3
"""#4864 step 0: would the ``nearest`` split's sizes step together across the eval's runs?

#3286's ripple came from the split sizes alone: ``round(n * 0.5)`` stepped on a
fixed cadence in the vote count, and the eval casts one vote per step, so every
run stepped on the same vote and ~700 averaged runs kept the step.  #3286's
correction showed the split is the only place the vote count enters the
pipeline discretely, so a split whose sizes do not step together across runs
leaves nothing downstream to ripple.  That is checked here by arithmetic, with
no new run: replay every (run, vote) of a finished grid through each sizing rule
and ask whether the sizes' rounding residuals line up across runs.

Rules (each returns how many Goods and Bads train every fold):

* ``dither``  - the app's (:func:`conformal._split_train_counts`), its coin drawn
  per (run, vote) as the labelset-seeded RNG draws it;
* ``nearest`` - #4864's arm (:func:`arms_split_rounding._nearest_train_counts`);
* ``round``   - the rule before #3288, ``round()`` on the whole labelset then per
  class: the coherent reference, which the eval's one-vote-per-step clock
  phase-locks across every run.

For each rule and each split quantity (held-out Goods, Bads, and their total,
per fold), the residual ``held - fraction * count`` is the rounding's share.
Its mean over runs at each vote, detrended, is what a ripple would inherit.
Its amplitude at a frequency is tested against a **sign-flip null**: flipping
the sign of every run of a category at random keeps each run's own cadence but
breaks any alignment between categories, so an amplitude well above the flips'
is a cadence the runs share.

Also replays the labels line's gate (``2 * held-out Goods >= HEAD_COUNT_GOODS`` at
two folds) to count how often each rule switches a run between the count line
and the labels line, the user-facing cost the arm exists to remove.

    python split_cadence_4864.py --grid binary=CELLS_DIR:0.3 --grid region=CELLS_DIR:0.5 --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vtscore.eval.arms_split_rounding import _nearest_train_counts  # noqa: E402
from vtscore.training.thresholds import conformal  # noqa: E402
from vtscore.training.thresholds.labels_line import HEAD_COUNT_GOODS  # noqa: E402

RULES = ("dither", "nearest", "round")
QUANTITIES = ("held_good", "held_bad", "held_total")
RUN = ["category", "seed", "style"]
HORIZON = 150
#: The app's fold count (``DEFAULT_CALIBRATE_COUNT``): the class model pools every fold's held-out scores.
FOLDS = 2
#: Sign flips per null.
N_FLIPS = 2000
#: The earliest vote a cadence is read from: a vote where at least this share of runs shows a
#: trained detector.  On the 1% pool a rare class's run shows one late, so a high bar would read
#: only the last votes.
COVERAGE = 0.25


def round_train_counts(n: int, n_pos: int, n_neg: int, fraction: float) -> tuple[int, int] | None:
    """The split before #3288: ``round()`` (half to even) on the labelset, then per class (#3286)."""
    n_cal = max(1, round(n * fraction))
    n_train = n - n_cal
    if n_train < 2:
        return None

    def per_class(total: int) -> int:
        return max(1, min(total - 1, round(total * n_train / n)))

    return per_class(n_pos), per_class(n_neg)


def train_counts(rule: str, n_good: int, n_bad: int, fraction: float, key: str) -> tuple[int, int] | None:
    n = n_good + n_bad
    if rule == "dither":
        # The app seeds the coin from a digest of the labelset, so a draw is fixed for one
        # labelset and fresh at every vote; a per-(run, vote) seed draws the same way.
        rng = np.random.RandomState(zlib.crc32(key.encode()))
        return conformal._split_train_counts(n, n_good, n_bad, fraction, rng)
    if rule == "nearest":
        return _nearest_train_counts(n, n_good, n_bad, fraction, None)  # type: ignore[arg-type]
    return round_train_counts(n, n_good, n_bad, fraction)


def read_trajectories(cells: Path) -> tuple[pd.DataFrame, int]:
    """Every run's (vote, Goods, Bads) where the app shows a trained detector, from a grid's main cell files.

    Base rows only (:func:`_cells_io._base_rows`): a cell also writes a row per
    pool and GMM variant at every vote, and those carry ``app_trained = 1`` from
    the first trained vote whether or not the app shows a detector yet, so
    reading them would replay votes no user sees.
    """
    import _cells_io  # noqa: PLC0415

    files = _cells_io.main_frame_files(cells)
    cols = [
        "seed",
        "category",
        "style",
        "t",
        "n_good",
        "n_bad",
        "phase",
        "app_trained",
        "gmm_variant",
        "schedule",
        "pool_variant",
    ]
    frames = []
    unreadable = 0
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in cols)
        except (OSError, ValueError, EOFError, pd.errors.EmptyDataError):
            unreadable += 1
            continue
        frames.append(_cells_io._base_rows(df))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["phase"].fillna("").astype(str).str.strip() != "check"]
    if "app_trained" in rows:
        rows = rows[rows["app_trained"].fillna(0).astype(int) == 1]
    rows = rows[(rows["t"] >= 1) & (rows["t"] <= HORIZON)]
    rows = rows.drop_duplicates(subset=[*RUN, "t"], keep="last")
    rows = rows[(rows["n_good"] >= 2) & (rows["n_bad"] >= 2)]
    print(f"  {cells}: {len(files)} cell files, {unreadable} unreadable, {rows[RUN].drop_duplicates().shape[0]} runs")
    return rows[[*RUN, "t", "n_good", "n_bad"]].astype({"t": int, "n_good": int, "n_bad": int}), unreadable


def replay(rows: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Each row's held-out Goods, Bads and total per fold under every rule, and their rounding residuals."""
    out = rows.copy()
    keys = (
        rows["category"].astype(str)
        + "|"
        + rows["seed"].astype(str)
        + "|"
        + rows["style"].astype(str)
        + "|"
        + rows["t"].astype(str)
    ).tolist()
    goods = rows["n_good"].to_numpy()
    bads = rows["n_bad"].to_numpy()
    for rule in RULES:
        hg = np.full(len(rows), np.nan)
        hb = np.full(len(rows), np.nan)
        for i, (g, b, key) in enumerate(zip(goods, bads, keys)):
            tc = train_counts(rule, int(g), int(b), fraction, key)
            if tc is None:
                continue
            hg[i] = g - tc[0]
            hb[i] = b - tc[1]
        out[f"{rule}:held_good"] = hg
        out[f"{rule}:held_bad"] = hb
        out[f"{rule}:held_total"] = hg + hb
        out[f"{rule}:res_good"] = hg - fraction * goods
        out[f"{rule}:res_bad"] = hb - fraction * bads
        out[f"{rule}:res_total"] = hg + hb - fraction * (goods + bads)
    return out


def _detrend(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return y - np.polyval(np.polyfit(t, y, 2), t)


def _amplitude(d: np.ndarray, t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Amplitude of each frequency's sinusoid in *d* (cycles per vote)."""
    phase = np.exp(-2j * np.pi * np.outer(freqs, t))
    return 2.0 * np.abs(phase @ d) / len(t)


def cadence(
    table: pd.DataFrame, column: str, test_freqs: Sequence[float], rng: np.random.Generator
) -> tuple[dict, pd.DataFrame]:
    """The runs' shared cadence in one residual: amplitude at *test_freqs* against the sign-flip null."""
    wide = table.pivot_table(index=RUN, columns="t", values=column, aggfunc="last")
    covered = wide.notna().mean(axis=0)
    t_lo = int(covered[covered >= COVERAGE].index.min())
    wide = wide.loc[:, t_lo:HORIZON]
    # Each run's own level aside: a constant offset is not a cadence.
    wide = wide.sub(wide.mean(axis=1), axis=0)
    t = np.asarray(wide.columns, dtype=float)
    values = wide.to_numpy(dtype=float)
    present = ~np.isnan(values)
    filled = np.nan_to_num(values)
    counts = present.sum(axis=0)
    cats = wide.index.get_level_values("category").to_numpy()
    cat_codes, cat_index = np.unique(cats, return_inverse=True)

    def mean_curve(signs: np.ndarray) -> np.ndarray:
        return _detrend((filled * signs[:, None]).sum(axis=0) / np.maximum(counts, 1), t)

    freqs = np.arange(1, len(t) // 2 + 1) / len(t)
    test = np.asarray(test_freqs, dtype=float)
    all_freqs = np.concatenate([freqs, test])
    observed = _amplitude(mean_curve(np.ones(len(values))), t, all_freqs)
    null = np.empty((N_FLIPS, len(all_freqs)))
    for k in range(N_FLIPS):
        flip = rng.choice([-1.0, 1.0], size=len(cat_codes))[cat_index]
        null[k] = _amplitude(mean_curve(flip), t, all_freqs)
    spectrum = pd.DataFrame(
        {
            "freq": freqs,
            "amplitude": observed[: len(freqs)],
            "null_median": np.median(null[:, : len(freqs)], axis=0),
            "null_p95": np.quantile(null[:, : len(freqs)], 0.95, axis=0),
        }
    )
    tests = {}
    for j, f in enumerate(test):
        a = observed[len(freqs) + j]
        nj = null[:, len(freqs) + j]
        tests[f"{f:g}"] = {
            "amplitude": float(a),
            "null_median": float(np.median(nj)),
            "null_p95": float(np.quantile(nj, 0.95)),
            "x_noise": float(a / np.median(nj)) if np.median(nj) > 0 else float("nan"),
            "p": float((1 + np.sum(nj >= a)) / (1 + N_FLIPS)),
        }
    summary = {"t_lo": t_lo, "n_runs": int(len(values)), "n_categories": int(len(cat_codes)), "tests": tests}
    return summary, spectrum


def labels_band(table: pd.DataFrame) -> tuple[int, int]:
    """The Goods counts where the app's dither draws either line: P(labels line) strictly inside (0.02, 0.98)."""
    p = p_labels_by_goods(table).set_index("n_good")["dither"]
    inside = p[(p > 0.02) & (p < 0.98)].index
    return int(inside.min()), int(inside.max())


def line_switches(table: pd.DataFrame, band_goods: tuple[int, int]) -> pd.DataFrame:
    """Per run and rule: the votes the line is the labels line, and how often it switches kind.

    Over the run's votes with a trained detector on screen, and again over those
    inside *band_goods*, the band where the app's coin decides the line.
    """
    out = []
    lo, hi = band_goods
    for run, g in table.sort_values("t").groupby(RUN, sort=False):
        row = dict(zip(RUN, run))
        band = g[(g["n_good"] >= lo) & (g["n_good"] <= hi)]
        for rule in RULES:
            labels = (FOLDS * g[f"{rule}:held_good"].to_numpy() >= HEAD_COUNT_GOODS).astype(int)
            row[f"{rule}:switches"] = int(np.abs(np.diff(labels)).sum()) if len(labels) > 1 else 0
            row[f"{rule}:labels_votes"] = int(labels.sum())
            band_labels = (FOLDS * band[f"{rule}:held_good"].to_numpy() >= HEAD_COUNT_GOODS).astype(int)
            row[f"{rule}:band_switches"] = int(np.abs(np.diff(band_labels)).sum()) if len(band_labels) > 1 else 0
        row["band_votes"] = len(band)
        out.append(row)
    return pd.DataFrame(out)


def p_labels_by_goods(table: pd.DataFrame) -> pd.DataFrame:
    """P(labels line) at each count of Goods, per rule, over the grid's own rows."""
    rows = []
    for g, sub in table.groupby("n_good"):
        if g > 16:
            break
        row = {"n_good": int(g), "rows": len(sub)}
        for rule in RULES:
            row[rule] = float((FOLDS * sub[f"{rule}:held_good"] >= HEAD_COUNT_GOODS).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="append", required=True, help="NAME=CELLS_DIR:FRACTION (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=4864)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    report: dict = {}
    spectra = []
    switches_all = []
    plabels_all = []
    for spec in args.grid:
        name, rest = spec.split("=", 1)
        cells, frac = rest.rsplit(":", 1)
        fraction = float(frac)
        print(f"{name}: fraction {fraction}")
        rows, unreadable = read_trajectories(Path(cells))
        table = replay(rows, fraction)
        # The rule's own frequencies: a class's residual repeats every 1 / frac(fraction * k) votes.
        # At 0.3 that is 0.3 cycles a vote (period 10/3); at 0.5, 0.5 (period 2).  #3286's
        # round-half-even at 0.5 adds 0.25 (period 4).
        test_freqs = [0.3, 0.1, 0.4] if abs(fraction - 0.3) < 1e-9 else [0.5, 0.25]
        report[name] = {"cells": cells, "fraction": fraction, "unreadable": unreadable, "rows": len(table)}
        for rule in RULES:
            for q, col in zip(QUANTITIES, ("res_good", "res_bad", "res_total")):
                summary, spectrum = cadence(table, f"{rule}:{col}", test_freqs, rng)
                report[name][f"{rule}:{q}"] = summary
                spectra.append(spectrum.assign(grid=name, rule=rule, quantity=q))
                tests = ", ".join(
                    f"f={f} A={v['amplitude']:.4f} ({v['x_noise']:.1f}x noise, p={v['p']:.3g})"
                    for f, v in summary["tests"].items()
                )
                print(f"  {rule:8s} {q:10s} from vote {summary['t_lo']}: {tests}")
        band = labels_band(table)
        report[name]["band_goods"] = list(band)
        sw = line_switches(table, band).assign(grid=name)
        switches_all.append(sw)
        plabels_all.append(p_labels_by_goods(table).assign(grid=name))
        for rule in RULES:
            print(
                f"  {rule:8s} switches/run: mean {sw[f'{rule}:switches'].mean():.2f}, "
                f"in the {band[0]}-{band[1]} Goods band {sw[f'{rule}:band_switches'].mean():.2f}"
            )
    pd.concat(spectra).to_csv(args.out / "spectra.csv", index=False)
    pd.concat(switches_all).to_csv(args.out / "switches.csv", index=False)
    pd.concat(plabels_all).to_csv(args.out / "p_labels_by_goods.csv", index=False)
    (args.out / "cadence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

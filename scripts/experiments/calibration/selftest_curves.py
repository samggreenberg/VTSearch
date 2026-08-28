"""Planted-answer self-test for ``curves.py`` — the standard quality-over-clicks pair.

The figures are the one artefact a reader takes a *level* off, so the failure
mode that matters is not "it crashed" but "it drew a confident line over a
subset and said nothing".  Every check here is one of those:

* an arm that starves on two thirds of its cells must show **coverage below 1**,
  which is only possible if the caller's cell list is used as the denominator
  rather than the rows that happen to exist;
* the mean where coverage is low must fall under the dashed rule, not be drawn
  as though it described the grid;
* the mean must be computed over the cells that trained (a missing cell is NaN,
  never a zero, and never a forward-filled level);
* ``t=0`` must be the **zero-click text sort**, not the first trainable click,
  so the far left of the figure is what typing got for free;
* an arm that never beats that anchor must report **no crossover**, not the last
  click it happened to be measured at;
* the per-run panel must **count** the runs that drew no line at all.

Run: ``python selftest_curves.py``
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import curves as C  # noqa: E402

N_CAT, N_SEED, N_STEP, FIRST_T = 3, 8, 60, 5
#: What the typed query gets for free on every cell, before any click.
TEXT_COST = 0.30
#: Per arm: ``(final cost, fraction of cells that never train)``.
#:
#: ``starver`` is the arm the coverage rules exist for: a third of its cells
#: train and those are *better* than ``clean``'s, so any analysis that quietly
#: averages over the survivors reports it as the winner.  ``worse`` is the arm
#: the zero-click anchor exists for: it improves with clicks and still never
#: beats simply typing the query, which is invisible without the anchor.
PLANT = {"clean": (0.14, 0.0), "starver": (0.10, 2 / 3), "worse": (0.40, 0.0)}


def _cost(level: float, t: int) -> float:
    return level + 0.2 * np.exp(-t / 15.0)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, cells, base = [], [], []
    for arm, (level, starve) in PLANT.items():
        for ds in ("dsA", "dsB"):
            for cat in range(N_CAT):
                for seed in range(N_SEED):
                    cells.append({"arm": arm, "dataset": ds, "category": f"cat{cat}", "seed": seed})
                    if seed < round(starve * N_SEED):
                        continue  # never trained: no main row at all
                    for t in range(FIRST_T, N_STEP + 1):
                        rows.append(
                            {
                                "arm": arm,
                                "dataset": ds,
                                "category": f"cat{cat}",
                                "seed": seed,
                                "t": t,
                                "cost": _cost(level, t),
                            }
                        )
    # The baseline is a property of the CELL, not of the arm: every arm opens on
    # the same seed sort, so one row per (dataset, category, seed).
    for ds in ("dsA", "dsB"):
        for cat in range(N_CAT):
            for seed in range(N_SEED):
                base.append(
                    {
                        "dataset": ds,
                        "category": f"cat{cat}",
                        "seed": seed,
                        "supports_text": 1,
                        "text_cost": TEXT_COST,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(base)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail and not ok else ''}")
    return ok


def main() -> int:  # noqa: C901
    tmp = Path(tempfile.mkdtemp(prefix="curves-selftest-"))
    try:
        main_df, cells, base = _frames()
        arms = list(PLANT)
        written = C.quality_vs_clicks(main_df, tmp, arms=arms, denominator=cells, baseline=base)
        curve = pd.read_csv(tmp / "cost_vs_clicks.csv")

        ok = True
        print("planted-answer checks:")
        ok &= _check(
            "both standard figures are written, per dataset",
            {"cost_vs_clicks.png", "cost_vs_clicks_runs__dsA.png", "cost_vs_clicks_runs__dsB.png"} <= set(written),
            str(written),
        )

        clean = curve[curve["arm"] == "clean"]
        starver = curve[curve["arm"] == "starver"]
        worse = curve[curve["arm"] == "worse"]

        # --- the zero-click anchor -----------------------------------------
        ok &= _check(
            "the curve starts at click 0, not at the first trainable click",
            int(curve["t"].min()) == 0,
            f"min t {curve['t'].min()}",
        )
        ok &= _check(
            "click 0 IS the text sort's quality, on every arm",
            bool(np.allclose(curve.loc[curve["t"] == 0, "mean"], TEXT_COST)),
            str(curve.loc[curve["t"] == 0, "mean"].unique()),
        )
        ok &= _check(
            "every cell has a text sort, so click 0 is fully covered even for the starving arm",
            np.isclose(float(starver.loc[starver["t"] == 0, "coverage"].iloc[0]), 1.0),
        )
        ok &= _check(
            "the anchor is carried as a column so the far right can be read against it",
            bool(np.allclose(curve["baseline"].dropna(), TEXT_COST)),
        )

        # --- crossover: how many clicks before beating the typed query ------
        x = C.crossover(curve).set_index(["arm", "dataset"])
        ok &= _check(
            "an arm that ends better than the text sort reports a crossover click",
            bool(np.isfinite(x.loc[("clean", "dsA"), "crossover_t"])),
            str(x.loc[("clean", "dsA")].to_dict()),
        )
        # THE reason the anchor is worth computing: `worse` improves steadily
        # with clicks and is still never worth as much as typing the query.  A
        # curve with no anchor shows that as a nice descending line.
        ok &= _check(
            "an arm that never beats the text sort reports NO crossover",
            not np.isfinite(x.loc[("worse", "dsA"), "crossover_t"]),
            str(x.loc[("worse", "dsA")].to_dict()),
        )
        ok &= _check(
            "the crossover is the FIRST click that beats it, not the last measured",
            int(x.loc[("clean", "dsA"), "crossover_t"]) == FIRST_T,
            str(x.loc[("clean", "dsA"), "crossover_t"]),
        )

        # --- coverage: the denominator ------------------------------------
        ok &= _check(
            "the arm that trains everywhere reaches full coverage",
            np.isclose(clean["coverage"].max(), 1.0),
            f"max coverage {clean['coverage'].max():.3f}",
        )
        # THE wiring check.  Without the caller's cell list as the denominator
        # this is 1.0 by construction: an arm that starved on two thirds of its
        # grid would report a full-coverage curve over the third that worked.
        ok &= _check(
            "the starving arm never reaches full coverage after click 0",
            starver.loc[starver["t"] > 0, "coverage"].max() < 0.7,
            f"max coverage {starver.loc[starver['t'] > 0, 'coverage'].max():.3f}",
        )
        ok &= _check(
            "coverage counts the cells attempted, not the cells that trained",
            int(starver["n_cells"].max()) == N_CAT * N_SEED,
            f"n_cells {starver['n_cells'].max()}",
        )
        ok &= _check(
            "the starving arm's clicked curve falls under the solid-line rule",
            bool((starver.loc[starver["t"] > 0, "coverage"] < C.SOLID_COVERAGE).all()),
        )
        ok &= _check(
            "the healthy arm's warm curve does not",
            bool((clean.loc[clean["t"] >= FIRST_T, "coverage"] >= C.SOLID_COVERAGE).all()),
        )
        # The gap between the anchor and the first trainable click is dashed for
        # the same reason: nothing was measured in there.
        gap = clean[(clean["t"] > 0) & (clean["t"] < FIRST_T)]
        ok &= _check(
            "the stretch between the anchor and the first trained click is not drawn solid",
            bool((gap["coverage"] < C.SOLID_COVERAGE).all()),
            str(gap["coverage"].unique()),
        )

        # --- the level itself ----------------------------------------------
        # A missing cell must be absent from the mean, not a zero and not a
        # forward-filled level: both would move the level rather than the
        # denominator, and neither is visible in the figure.
        expect = _cost(PLANT["clean"][0], N_STEP)
        got = float(clean.loc[clean["t"] == N_STEP, "mean"].iloc[0])
        ok &= _check(
            "the mean is over the cells that trained (no zero-fill, no ffill)",
            abs(got - expect) < 1e-6,
            f"{got:.4f} vs {expect:.4f}",
        )
        ok &= _check(
            "the starving arm's level is its survivors', not diluted toward the anchor",
            abs(float(starver.loc[starver["t"] == N_STEP, "mean"].iloc[0]) - _cost(PLANT["starver"][0], N_STEP)) < 1e-6,
        )
        ok &= _check(
            "the never-crossing arm is above the anchor at the horizon", float(worse["mean"].iloc[-1]) > TEXT_COST
        )

        # --- the per-run panel ---------------------------------------------
        # A run that never trained draws only its t=0 anchor dot, so the count
        # has to be written on the panel or the arm merely looks like it has
        # fewer seeds.
        names = C.per_run_figures(main_df, tmp, arms=arms, denominator=cells, baseline=base)
        g = main_df[(main_df["arm"] == "starver") & (main_df["dataset"] == "dsA")]
        n_trained = int(g.groupby(["dataset", "category", "seed"]).ngroups)
        n_cells = int(cells[(cells["arm"] == "starver") & (cells["dataset"] == "dsA")].shape[0])
        ok &= _check(
            "the per-run panel has runs to count as never-trained",
            n_cells - n_trained == N_CAT * round(PLANT["starver"][1] * N_SEED),
            f"{n_cells - n_trained}/{n_cells}",
        )
        ok &= _check("per-run figures written for every dataset", len(names) == 2, str(names))

        # --- degrade gracefully with no baseline ---------------------------
        plain = C.mean_figure(main_df, tmp / "nobase", arms=arms, denominator=cells)[1]
        ok &= _check(
            "with no baseline the curve simply starts at the first click",
            int(plain["t"].min()) == 1 and not np.isfinite(plain["baseline"].iloc[0]),
        )
        ok &= _check("...and crossover then reports nothing rather than guessing", C.crossover(plain).empty)

        print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

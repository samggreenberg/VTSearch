"""Planted-answer self-test for ``stopping.py`` — the app's stopping rules, measured.

Each planted trajectory is one way the `phase` column can lie to a careless
reader, and each check is the corresponding refusal:

* a **clean stopper** must report the click it *first* went green, not the last
  one it stayed there — the app announces once, on the first;
* a **flapper** that crosses and falls back four times must still report the
  first crossing as ``t_stop`` (that is what the user saw) while reporting *no*
  ``t_sustained`` and an episode count above 1, so the two are never confused;
* a **never-stopper** must be counted as a run, not dropped: it is the run that
  makes the conditional mean flattering, and it is carried into the
  Kaplan-Meier fit as a censored observation rather than discarded;
* with **most runs censored** the KM median must come back ``NaN`` rather than
  quoting the median of the minority that finished;
* the **stopping cost** must be read at the stopping click and not at the
  budget, including when the extra clicks made the detector *worse* — the sign
  of ``cost_delta`` is the finding, so it must not be absolute-valued anywhere;
* the **binding indicator** must be named from the lights when a frame carries
  them, and must say so rather than guessing when the frame predates them.

Run: ``python selftest_stopping.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import stopping as S  # noqa: E402

BUDGET = 60

#: One planted trajectory per shape.  ``phases`` is a function from click to
#: phase; the metric is planted so that the value at click ``t`` is exactly
#: ``t / 1000``, which makes every "was this read at the right click?" check a
#: literal equality rather than a tolerance.
PLANTED: dict[str, dict] = {
    # Goes green at 20 and stays: t_stop == t_sustained == 20, one episode.
    "clean": {"done_at": lambda t: t >= 20, "lights": "green"},
    # Green at 15, then falls back four times, ending in `hard`: the app
    # announced at 15 and the run never settled.
    "flapper": {"done_at": lambda t: t in (15, 16, 25, 26, 35, 44), "lights": "green"},
    # Never green: censored at the budget.
    "never": {"done_at": lambda t: False, "lights": "amber-stable"},
}


def _frame() -> pd.DataFrame:
    """Three arms x four seeds, each arm one planted shape."""
    rows = []
    for arm, spec in PLANTED.items():
        for seed in range(4):
            for t in range(1, BUDGET + 1):
                done = bool(spec["done_at"](t))
                if spec["lights"] == "green":
                    smart, stable, span = ("green", "green", "green") if done else ("green", "yellow", "green")
                else:
                    # Stable is the culprit on every held step of this arm.
                    smart, stable, span = "green", "yellow", "green"
                rows.append(
                    {
                        "arm": arm,
                        "dataset": "ds",
                        "category": "cat",
                        "seed": seed,
                        "t": t,
                        "phase": "done" if done else "hard",
                        "smart": smart,
                        "stable": stable,
                        "span": span,
                        "n_good": t // 2,
                        "n_bad": t - t // 2,
                        # Planted so cost(t) == t/1000: rising, so the budget's
                        # extra clicks always make it WORSE and the delta's sign
                        # is checkable.
                        "cost": t / 1000.0,
                        "average_precision": 1.0 - t / 1000.0,
                    }
                )
    return pd.DataFrame(rows)


def _check(label: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  — ' + detail) if detail and not cond else ''}")
    return bool(cond)


def main() -> int:
    ok = True
    main_df = _frame()
    stops = S.stopping_points(main_df, keys=("arm", "dataset", "category", "seed"))

    by_arm = {a: g for a, g in stops.groupby("arm")}
    clean, flap, never = by_arm["clean"], by_arm["flapper"], by_arm["never"]

    # --- first fire is the app's announcement --------------------------------
    ok &= _check(
        "clean stopper fires at its first green click", set(clean["t_stop"]) == {20.0}, str(set(clean["t_stop"]))
    )
    ok &= _check("...and that is also where it settles", set(clean["t_sustained"]) == {20.0})
    ok &= _check("...one episode", set(clean["n_done_episodes"]) == {1})

    ok &= _check(
        "flapper reports its FIRST crossing, not its last", set(flap["t_stop"]) == {15.0}, str(set(flap["t_stop"]))
    )
    ok &= _check("...and reports no sustained stop at all", flap["t_sustained"].isna().all())
    ok &= _check(
        "...with the episode count that says why",
        set(flap["n_done_episodes"]) == {4},
        str(set(flap["n_done_episodes"])),
    )

    ok &= _check("never-stopper is kept as a run", len(never) == 4 and not never["stopped"].any())
    ok &= _check("...with no stopping click invented", never["t_stop"].isna().all())
    ok &= _check("...and its budget recorded for censoring", set(never["t_budget"]) == {BUDGET})

    # --- the stopping cost is read AT the stopping click ---------------------
    ok &= _check("stopping cost is the metric at the stopping click", set(clean["cost_at_stop"]) == {0.020})
    ok &= _check("...not the metric at the budget", set(clean["cost_final"]) == {BUDGET / 1000.0})
    ok &= _check(
        "...and the paired delta keeps its sign when the extra clicks hurt",
        np.allclose(clean["cost_delta"], (BUDGET - 20) / 1000.0),
        str(clean["cost_delta"].tolist()),
    )
    ok &= _check("clicks after the announcement are counted", set(clean["clicks_after_stop"]) == {float(BUDGET - 20)})
    ok &= _check("labelset at the stop is carried through", set(clean["n_good_at_stop"]) == {10.0})

    # --- censoring ------------------------------------------------------------
    summary = S.summarise(stops)
    srow = {r["arm"]: r for _, r in summary.iterrows()}
    ok &= _check(
        "fire rate leads, and is over ALL runs", srow["never"]["fire_rate"] == 0.0 and srow["clean"]["fire_rate"] == 1.0
    )
    ok &= _check("KM median equals the plain median when nothing is censored", srow["clean"]["km_t_stop"] == 20.0)
    ok &= _check("KM median is NaN when no run ever fired", not np.isfinite(srow["never"]["km_t_stop"]))

    # Pool all three shapes into one group: 8 of 12 runs fire, the other 4 are
    # censored at the budget.  The two medians then disagree, which is the whole
    # reason both are reported.
    pooled = stops.copy()
    pooled["arm"] = "pooled"
    ps = S.summarise(pooled).iloc[0]
    ok &= _check("...with the rate that qualifies it", np.isclose(ps["fire_rate"], 8 / 12))
    ok &= _check("...the conditional median is the firers' own", ps["median_t_stop"] == 17.5, str(ps["median_t_stop"]))
    # The whole point of carrying the censored runs: they push the survival
    # curve out, so the honest median is LATER than the median of the runs that
    # happened to finish.  Quoting the conditional number as "the stopping
    # point" understates it, and understates it more the worse the arm is.
    ok &= _check(
        "...and the censored median is later than it, never earlier", ps["km_t_stop"] == 20.0, str(ps["km_t_stop"])
    )

    # --- direct KM checks ----------------------------------------------------
    ok &= _check("KM: all events, odd n", S.km_median([1, 2, 3], []) == 2)
    ok &= _check("KM: censoring past the median leaves it where it was", S.km_median([1, 2, 3], [99]) == 2)
    ok &= _check(
        "KM: a minority of firers returns NaN, not the survivors' median",
        not np.isfinite(S.km_median([15] * 4, [60] * 8)),
    )
    ok &= _check(
        "KM: heavy censoring returns NaN, not a survivors' median", not np.isfinite(S.km_median([1], [9, 9, 9]))
    )

    # --- which rule was binding ----------------------------------------------
    ok &= _check("the held steps name Stable as the culprit", srow["never"]["blocked_stable"] == 1.0)
    ok &= _check(
        "...and clear Smart and Span", srow["never"]["blocked_smart"] == 0.0 and srow["never"]["blocked_span"] == 0.0
    )
    note = S.binding_note(summary)
    ok &= _check("the binding note names it", "stable" in note and "binding rule" in note, note)

    dark = main_df.drop(columns=list(S.LIGHT_COLUMNS))
    dark_summary = S.summarise(S.stopping_points(dark, keys=("arm", "dataset", "category", "seed")))
    ok &= _check(
        "a pre-#3560 frame says the binding rule is unrecoverable rather than guessing",
        "Re-run to answer it" in S.binding_note(dark_summary),
        S.binding_note(dark_summary),
    )
    ok &= _check(
        "...while still reporting the stopping point it CAN derive from `phase`",
        dark_summary.loc[dark_summary["arm"] == "clean", "median_t_stop"].iloc[0] == 20.0,
    )

    # --- the table renders every group ---------------------------------------
    table = S.stopping_table(summary)
    ok &= _check("the table has a row per arm", all(a in table for a in PLANTED), table)
    ok &= _check("...and quotes the fire rate beside every stopping click", "0%" in table and "100%" in table, table)
    ok &= _check(
        "an empty frame renders a sentence, not a broken table", "No stopping data" in S.stopping_table(pd.DataFrame())
    )
    ok &= _check("...and an empty frame in gives an empty frame out", S.stopping_points(pd.DataFrame()).empty)

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

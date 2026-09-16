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

And then the margins (issue #3560's second half), where the traps are different:

* a margin must be measured over the **held** steps only — a run that stopped
  sits at a comfortable positive margin for the rest of its budget, which would
  otherwise swamp the steps that actually held it;
* every gate must read in **one direction**: positive is "satisfied, with this
  much room", negative is "short by this much", whichever way the underlying
  rule's inequality points;
* when Stable blocks, the three gates it blocks *with* must be told apart and
  attributed in the rule's own short-circuit order — "Stable held it" is three
  findings, and the whole point of recording margins is to separate them;
* a frame that predates the margins must come back **empty**, not zeroed: unlike
  the lights, nothing can reconstruct these from old cells.

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
                        # Margins planted to AGREE with the lights above, and
                        # planted at constants so every margin is an exact
                        # subtraction: Smart clears its flatness cutoff by
                        # 0.005 and its t cutoff by 1.0; Span sits exactly ON
                        # its bar (a zero margin, which is green); Stable is
                        # over both of its cutoffs on every yellow step and
                        # under both on every green one.  The raw-rate halves
                        # never trip the still-falling gate (0.03 is not below
                        # half of 0.04), so this frame attributes every Stable
                        # block to the average - the other two gates get their
                        # own frame below.
                        "smart_slope": -0.010,
                        "smart_slope_t": -1.0,
                        "span_level": 40,
                        "span_target": 40,
                        "stable_flip_rate": 0.03,
                        "stable_confident_flip_rate": 0.001 if stable == "green" else 0.02,
                        "stable_max_confident_flip_rate": 0.002 if stable == "green" else 0.03,
                        "stable_flip_rate_early": 0.04,
                        "stable_flip_rate_late": 0.03,
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


#: One planted trajectory per **Stable gate**, so the short-circuit attribution
#: can be checked gate by gate.  All three never stop, so every step is held.
#: Each sets the gates ahead of its own to passing values, which is what makes
#: the attribution unambiguous — the rule credits the first gate that fails.
STABLE_GATES: dict[str, dict[str, float]] = {
    # Confident average over its cutoff: the first gate, so it takes the blame
    # even though the max is over its cutoff too.
    "avg": {"conf": 0.02, "max": 0.03, "early": 0.04, "late": 0.03},
    # Average fine, one bad step: the second gate.
    "max": {"conf": 0.001, "max": 0.05, "early": 0.04, "late": 0.03},
    # Both rate gates fine, but the raw rate is still falling — the later half
    # is 0.01, which is over the rate threshold and below half of 0.04.
    "falling": {"conf": 0.001, "max": 0.002, "early": 0.04, "late": 0.01},
}


def _gate_frame() -> pd.DataFrame:
    """Three arms, one per Stable gate, none of which ever stops."""
    rows = []
    for arm, g in STABLE_GATES.items():
        for seed in range(3):
            for t in range(1, 21):
                rows.append(
                    {
                        "arm": arm,
                        "dataset": "ds",
                        "category": "cat",
                        "seed": seed,
                        "t": t,
                        "phase": "hard",
                        "smart": "green",
                        "stable": "yellow",
                        "span": "green",
                        "smart_slope": -0.010,
                        "smart_slope_t": -1.0,
                        "span_level": 31,
                        "span_target": 40,
                        "stable_flip_rate": g["early"],
                        "stable_confident_flip_rate": g["conf"],
                        "stable_max_confident_flip_rate": g["max"],
                        "stable_flip_rate_early": g["early"],
                        "stable_flip_rate_late": g["late"],
                        "n_good": t // 2,
                        "n_bad": t - t // 2,
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

    # --- margins: how close, not merely whether -------------------------------
    KEYS = ("arm", "dataset", "category", "seed")
    mg = S.margins(main_df, keys=KEYS)
    mrow = {a: g for a, g in mg.groupby("arm")}

    ok &= _check("a post-#3560 frame is recognised as carrying margins", S.has_margins(main_df))
    ok &= _check(
        "margins are measured over the HELD steps only",
        set(mrow["clean"]["n_held"]) == {19} and set(mrow["never"]["n_held"]) == {BUDGET},
        str(set(mrow["clean"]["n_held"])),
    )
    # The clean arm's post-stop steps have Stable GREEN with a comfortable
    # positive margin.  If they leaked into the average this would come out
    # positive, which is the whole reason the slice exists.
    ok &= _check(
        "...so a stopper's comfortable post-stop margins do not wash out what held it",
        np.isclose(mrow["clean"]["stable_conf_median"].iloc[0], -0.015),
        str(mrow["clean"]["stable_conf_median"].tolist()),
    )

    # One sign convention: positive is room to spare, negative is short by that
    # much, whichever way the underlying rule's inequality points.
    ok &= _check(
        "a satisfied gate reads POSITIVE by the room it has",
        np.isclose(mrow["never"]["smart_slope_median"].iloc[0], 0.005)
        and np.isclose(mrow["never"]["smart_t_median"].iloc[0], 1.0),
        str((mrow["never"]["smart_slope_median"].iloc[0], mrow["never"]["smart_t_median"].iloc[0])),
    )
    ok &= _check(
        "...a blocking gate NEGATIVE by the amount it is short, in the same column",
        np.isclose(mrow["never"]["stable_conf_median"].iloc[0], -0.015)
        and np.isclose(mrow["never"]["stable_max_median"].iloc[0], -0.02),
        str((mrow["never"]["stable_conf_median"].iloc[0], mrow["never"]["stable_max_median"].iloc[0])),
    )
    ok &= _check(
        "...and a gate sitting exactly on its bar reads zero, and counts as green",
        mrow["never"]["span_median"].iloc[0] == 0.0 and mrow["never"]["span_green_share"].iloc[0] == 1.0,
    )
    ok &= _check(
        "the green share is over held steps, and separates a wall from a flap",
        mrow["never"]["smart_slope_green_share"].iloc[0] == 1.0
        and mrow["never"]["stable_conf_green_share"].iloc[0] == 0.0,
    )

    # --- which of Stable's three gates blocked --------------------------------
    gates = S.margins(_gate_frame(), keys=KEYS)
    grow = {a: g for a, g in gates.groupby("arm")}
    for gate in STABLE_GATES:
        ok &= _check(
            f"Stable's `{gate}` gate is attributed to itself and to nothing else",
            all(
                grow[gate][f"stable_block_{k}"].iloc[0] == (1.0 if k == gate else 0.0)
                for k in ("avg", "max", "falling")
            ),
            str({k: grow[gate][f"stable_block_{k}"].iloc[0] for k in ("avg", "max", "falling")}),
        )
    ok &= _check(
        "...and the shares over a blocked step sum to one",
        np.allclose(gates[["stable_block_avg", "stable_block_max", "stable_block_falling"]].sum(axis=1), 1.0),
    )
    ok &= _check(
        "Span's bar is the run's own, not an assumed 40",
        np.isclose(grow["avg"]["span_median"].iloc[0], -9.0),
        str(grow["avg"]["span_median"].iloc[0]),
    )

    # --- the margin table and the pre-#3560 refusal ---------------------------
    gsum = S.summarise_margins(gates)
    gtable = S.margin_table(gsum)
    ok &= _check("the margin table has a row per arm", all(a in gtable for a in STABLE_GATES), gtable)
    ok &= _check("...and carries no NaN identity columns from the run keys", "nan" not in gtable, gtable)
    ok &= _check(
        "a frame without the margins refuses rather than zeroing them",
        S.margins(main_df.drop(columns=["smart_slope"]), keys=KEYS).empty
        and not S.has_margins(main_df.drop(columns=["smart_slope"])),
    )
    ok &= _check(
        "...and says so in place of a table",
        "predate the per-step indicator margins" in S.margin_table(S.summarise_margins(pd.DataFrame())),
    )

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

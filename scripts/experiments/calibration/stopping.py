"""Where the app's stopping rules fired, and what the detector cost there.

Every simulated-user study reports a **final cost**: the metric at the last
click of a fixed budget (`CALIB_MAX_STEPS`, usually 150).  That number answers
"how good is the detector after 150 clicks", which is a question nobody in the
app is asking.  The app has stopping rules — the Smart / Stable / Span
indicators, all three green — and when they fire it tells the user *"All quality
indicators are green. You can continue labeling or export your results."*  The
number a user actually leaves with is the cost **there**, at a click count they
did not choose in advance.  Issue #3560.

This module turns the `phase` column every run already emits into that pair:

    stopping point  =  the click at which the rules first fired  (the width)
    stopping cost   =  the metric at that click                  (the height)

Nothing here needs a re-run.  `phase` has been on every metric row since the
harness adopted the app's phase machine (2026-07-31), so any study whose cells
are still on disk can be enriched by re-reading them.  What a *new* run adds is
the three indicator lights beside the phase (`smart` / `stable` / `span`, plus
`span_level` / `span_depth`), which say **which** rule held a run short of
stopping; those are absent from older frames and every function here degrades to
"unknown" rather than failing.

Three properties of the data decide the shape of this API, and all three were
measured before it was written:

**The rule fires far short of the budget.**  Not a rounding difference — on the
#3156 grid the `done` phase holds 24–37 of ~150 clicks per run, and a run that
stops has spent a quarter of its budget past the point the app said to stop.

**It flaps.**  The phase is *derived* from the current labelset every step, not
latched, so a run can go `done` on one vote and back to `hard` on the next; a
local probe saw up to five separate `done` episodes in one trajectory.  The app
announces completion on the **first** one and never re-announces
(`AutopilotStateService.completionAnnounced`), so *first fire* is the faithful
definition and is what `t_stop` reports.  `t_sustained` reports the stricter
one — the click from which it never goes back — and the gap between them is a
real finding about the rules, not noise to be smoothed away.

**It often never fires at all.**  Which makes every "average stopping point"
a **censored** statistic.  Averaging over the runs that stopped is the classic
survivorship error: the runs excluded are precisely the slow ones, so the mean
comes out flattering and gets more flattering the worse the arm is.  So
:func:`summarise` leads with the fire *rate*, quotes the conditional
distribution only behind it, and computes the median through Kaplan–Meier with
non-firing runs censored at their own budget — which returns `NaN`, honestly,
when fewer than half the runs ever fired.

Usage::

    import stopping
    stops = stopping.stopping_points(main, keys=curves.KEYS)
    print(stopping.stopping_table(stopping.summarise(stops)))

`selftest_stopping.py` is its planted-answer test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from vtscore.eval.autopilot_flow import STOPPING_PHASE

#: What identifies one trajectory in a pooled frame.  Mirrors ``curves.KEYS``
#: with ``arm`` in front: a study tags each arm's frame with an ``arm`` column
#: before concatenating, and two arms of the same cell are two trajectories.
#: Filtered to the columns actually present, so a single-arm frame keys on the
#: rest.
RUN_KEYS: tuple[str, ...] = ("arm", "dataset", "embedder", "category", "seed")

#: Metrics carried through to the stopping point by default: the one the ship
#: decision reads and the ranking metric that sits beside it in every report.
DEFAULT_METRICS: tuple[str, ...] = ("cost", "average_precision")

#: The indicator columns a post-#3560 run emits.  Absent from every earlier
#: frame, which is why every read of them is guarded.
LIGHT_COLUMNS: tuple[str, ...] = ("smart", "stable", "span")


def _present(df: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    return [k for k in keys if k in df.columns]


def _episodes(is_done: np.ndarray) -> int:
    """Number of maximal runs of consecutive ``done`` steps.

    One episode is a trajectory that reached ``done`` and stayed; five is one
    that crossed the line and fell back four times.  Reported because the two
    are the same `stopped=True` and mean entirely different things about
    whether the rules found a real convergence.
    """
    if is_done.size == 0:
        return 0
    return int(np.sum(is_done & ~np.concatenate(([False], is_done[:-1]))))


def stopping_points(
    main: pd.DataFrame,
    *,
    keys: Sequence[str] = RUN_KEYS,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """One row per trajectory: where the stopping rules fired and what it cost.

    *main* is a metric frame (base rows only — filter variants out first, as
    ``_cells_io.load_arm`` does) carrying at least ``t`` and ``phase``.

    Columns, per run:

    ``stopped``
        Whether the rules ever fired inside the budget.  **Read this first.**
        Every other stopping column is conditional on it, and the runs where it
        is ``False`` are systematically the slow ones.
    ``t_stop`` / ``t_sustained``
        First click at which the phase is ``done``, and the first from which it
        never leaves — the app's own announcement, and the stricter reading.
        Both ``NaN`` where they never happen.
    ``n_done_episodes``
        How many separate times the trajectory entered ``done``.
    ``t_budget`` / ``clicks_after_stop``
        The last click measured, and how many of them were spent past the
        announcement.  ``clicks_after_stop`` is the waste the issue is about.
    ``{metric}_at_stop`` / ``{metric}_final`` / ``{metric}_delta``
        The **stopping cost** (height at the stopping point), the number
        studies report today, and ``final - at_stop``.  For ``cost`` a positive
        delta means the extra clicks made the detector *worse*.
    ``n_good_at_stop`` / ``n_bad_at_stop``
        The labelset the user would have left with.
    ``blocked_smart`` / ``blocked_stable`` / ``blocked_span``
        Fraction of the run's not-yet-``done`` steps at which each light was not
        green — i.e. which rule was doing the holding.  ``NaN`` on a frame from
        before the lights were emitted.
    """
    kk = _present(main, keys)
    if main.empty or "phase" not in main.columns or "t" not in main.columns or not kk:
        return pd.DataFrame()

    metrics = [m for m in metrics if m in main.columns]
    out: list[dict[str, Any]] = []
    for run_key, g in main.groupby(kk, dropna=False, sort=True):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        is_done = (g["phase"].astype(str) == STOPPING_PHASE).to_numpy()

        first_i = int(np.argmax(is_done)) if is_done.any() else None
        # Sustained: the first index from which every LATER measured step is
        # also done.  A suffix scan, so it is the last falsification that
        # decides - not the first success.
        sust_i: int | None = None
        if is_done.size and is_done[-1]:
            k = is_done.size - 1
            while k > 0 and is_done[k - 1]:
                k -= 1
            sust_i = k

        row: dict[str, Any] = dict(zip(kk, run_key if isinstance(run_key, tuple) else (run_key,), strict=True))
        row["n_steps"] = int(len(g))
        row["t_budget"] = int(t[-1])
        row["stopped"] = bool(first_i is not None)
        row["t_stop"] = float(t[first_i]) if first_i is not None else float("nan")
        row["t_sustained"] = float(t[sust_i]) if sust_i is not None else float("nan")
        row["n_done_episodes"] = _episodes(is_done)
        row["clicks_after_stop"] = float(t[-1] - t[first_i]) if first_i is not None else float("nan")

        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            at_stop = float(vals[first_i]) if first_i is not None else float("nan")
            final = float(vals[-1])
            row[f"{m}_at_stop"] = at_stop
            row[f"{m}_final"] = final
            row[f"{m}_delta"] = final - at_stop
        for col, name in (("n_good", "n_good_at_stop"), ("n_bad", "n_bad_at_stop")):
            if col in g.columns and first_i is not None:
                row[name] = float(pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)[first_i])
            else:
                row[name] = float("nan")

        # Which light held the run back, over the steps where it was held.  A
        # run that never entered `done` is held at every step; one that did is
        # held only up to its first fire, because what happens after the app
        # said stop is a different question.
        held = slice(0, first_i if first_i is not None else len(g))
        for light in LIGHT_COLUMNS:
            key = f"blocked_{light}"
            if light not in g.columns:
                row[key] = float("nan")
                continue
            vals = g[light].astype(str).to_numpy()[held]
            # Blank = no phase machine ran (or a startup round owned the phase),
            # which is "not measured" rather than "not green".
            measured = vals[(vals != "") & (vals != "nan")]
            row[key] = float(np.mean(measured != "green")) if measured.size else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def km_median(t_event: Sequence[float], t_censor: Sequence[float]) -> float:
    """Kaplan-Meier median stopping click, censoring the runs that never fired.

    *t_event* are the clicks at which the rules fired; *t_censor* are the
    budgets of the runs that reached the end without firing — those runs are
    **not** dropped, they are carried as "would have fired at some click > this
    one", which is exactly what a censored observation is.

    Returns the first click at which the survival function falls to 0.5 or
    below, or ``NaN`` when it never does — the honest answer when most runs
    never stopped, and the reason this returns a number rather than printing
    a mean over the survivors.
    """
    ev = [float(x) for x in t_event if np.isfinite(x)]
    ce = [float(x) for x in t_censor if np.isfinite(x)]
    if not ev:
        return float("nan")
    at_risk = len(ev) + len(ce)
    surv = 1.0
    for tt in sorted(set(ev)):
        d = sum(1 for x in ev if x == tt)
        n = at_risk
        if n <= 0:
            break
        surv *= 1.0 - d / n
        if surv <= 0.5:
            return tt
        # Everyone who failed or was censored AT this click leaves the risk set.
        at_risk -= d + sum(1 for x in ce if x == tt)
    return float("nan")


def _q(x: pd.Series, p: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.quantile(p)) if len(x) else float("nan")


def summarise(
    stops: pd.DataFrame,
    *,
    by: Sequence[str] = ("arm",),
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Reduce per-run stopping points to one row per group.

    Leads with ``n_runs`` and ``fire_rate`` on purpose: every column after them
    is conditional on the rules having fired, and quoting a median stopping
    point without saying how many runs it describes is the survivorship error
    this whole module exists to avoid.  ``km_t_stop`` is the censoring-aware
    median — it uses the non-firing runs rather than dropping them, and is
    ``NaN`` when fewer than half of them ever fired.
    """
    if stops.empty:
        return pd.DataFrame()
    gb = _present(stops, by)
    groups: list[tuple[Any, pd.DataFrame]] = list(stops.groupby(gb, dropna=False, sort=True)) if gb else [((), stops)]
    metrics = [m for m in metrics if f"{m}_at_stop" in stops.columns]

    out: list[dict[str, Any]] = []
    for key, g in groups:
        fired = g[g["stopped"]]
        row: dict[str, Any] = dict(zip(gb, key if isinstance(key, tuple) else (key,), strict=True))
        row["n_runs"] = int(len(g))
        row["n_fired"] = int(len(fired))
        row["fire_rate"] = float(len(fired) / len(g)) if len(g) else float("nan")
        row["km_t_stop"] = km_median(fired["t_stop"], g.loc[~g["stopped"], "t_budget"])
        row["median_t_stop"] = _q(fired["t_stop"], 0.5)
        row["q25_t_stop"] = _q(fired["t_stop"], 0.25)
        row["q75_t_stop"] = _q(fired["t_stop"], 0.75)
        row["median_t_sustained"] = _q(fired["t_sustained"], 0.5)
        row["sustained_rate"] = (
            float(np.mean(np.isfinite(pd.to_numeric(fired["t_sustained"], errors="coerce"))))
            if len(fired)
            else float("nan")
        )
        row["median_episodes"] = _q(fired["n_done_episodes"], 0.5)
        row["median_clicks_after_stop"] = _q(fired["clicks_after_stop"], 0.5)
        for m in metrics:
            row[f"median_{m}_at_stop"] = _q(fired[f"{m}_at_stop"], 0.5)
            row[f"median_{m}_final"] = _q(fired[f"{m}_final"], 0.5)
            # Paired over the SAME runs (those that fired), so this is a
            # within-run difference and not a difference of two medians over
            # different denominators.
            row[f"median_{m}_delta"] = _q(fired[f"{m}_delta"], 0.5)
        for light in LIGHT_COLUMNS:
            row[f"blocked_{light}"] = (
                _q(g[f"blocked_{light}"], 0.5) if f"blocked_{light}" in g.columns else float("nan")
            )
        out.append(row)
    return pd.DataFrame(out)


def _fmt(v: Any, digits: int = 2) -> str:
    """Two significant digits by default — the report rule, applied here once."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "—"
    return f"{f:.{digits}f}"


def stopping_table(summary: pd.DataFrame, *, metric: str = "cost") -> str:
    """The stopping block of a REPORT.md, as markdown.

    One row per group.  The columns are the issue's two questions — *where* the
    rules fired and *what it cost there* — plus the two qualifications without
    which neither number can be read: how many runs ever fired, and how much
    the budget's extra clicks moved the metric afterwards.
    """
    if summary.empty:
        return "_No stopping data: no run carried a `phase` column._"
    ident = [c for c in summary.columns if c in ("arm", "dataset", "embedder", "category")]
    head = [*ident, "runs", "fired", "stop click (KM)", "stop click (median of fired)"]
    head += [f"{metric} at stop", f"{metric} at budget", f"Δ{metric} (paired)", "clicks after stop"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in ident]
        cells.append(str(int(r["n_runs"])))
        cells.append(f"{int(r['n_fired'])} ({r['fire_rate']:.0%})")
        cells.append(_fmt(r.get("km_t_stop"), 0))
        iqr = f"{_fmt(r.get('median_t_stop'), 0)} [{_fmt(r.get('q25_t_stop'), 0)}–{_fmt(r.get('q75_t_stop'), 0)}]"
        cells.append(iqr)
        cells.append(_fmt(r.get(f"median_{metric}_at_stop")))
        cells.append(_fmt(r.get(f"median_{metric}_final")))
        cells.append(_fmt(r.get(f"median_{metric}_delta")))
        cells.append(_fmt(r.get("median_clicks_after_stop"), 0))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def binding_note(summary: pd.DataFrame) -> str:
    """One sentence naming which rule held runs short of stopping, or why not.

    Only answerable on a post-#3560 frame; on an older one it says so instead of
    guessing, because the phase alone cannot separate Smart from Stable.
    """
    if summary.empty:
        return ""
    cols = [f"blocked_{c}" for c in LIGHT_COLUMNS]
    if not all(c in summary.columns for c in cols) or summary[cols].isna().all().all():
        return (
            "The binding indicator is not recoverable from these cells: they predate the "
            "`smart` / `stable` / `span` columns, and `phase` alone cannot tell Smart from "
            "Stable inside `hard`. Re-run to answer it."
        )
    means = {c: float(summary[f"blocked_{c}"].mean(skipna=True)) for c in LIGHT_COLUMNS}
    worst = max(means, key=lambda c: means[c] if math.isfinite(means[c]) else -1.0)
    parts = ", ".join(f"{c} {means[c]:.0%}" if math.isfinite(means[c]) else f"{c} —" for c in LIGHT_COLUMNS)
    return f"Share of held steps each indicator was not green: {parts}. **{worst}** is the binding rule."

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

Nothing in the *first* half needs a re-run.  `phase` has been on every metric
row since the harness adopted the app's phase machine (2026-07-31), so any study
whose cells are still on disk can be enriched by re-reading them.  What a *new*
run adds is the three indicator lights beside the phase (`smart` / `stable` /
`span`, plus `span_level` / `span_depth`), which say **which** rule held a run
short of stopping; those are absent from older frames and every function here
degrades to "unknown" rather than failing.

And then the question underneath that one
-----------------------------------------

"Stable held it" is three different findings.  A light is a threshold test, and
a study that records only the answer cannot say whether a run sat one noisy
window short of green for a hundred clicks or was never within reach of it —
which is the difference between a rule that needs de-flapping and a detector
that needs a better embedding.  So a run now also records, every step, the
continuous quantities the gates are thresholds **on**: the error-cost slope and
its t-statistic, the confident flip rate and its worst step, the two halves of
the raw flip window, and the atlas bar `span_level` is measured against.

:func:`margins`, :func:`summarise_margins` and :func:`margin_table` are that
half, on the same "margin to green" sign convention throughout
(:data:`GATES`): **negative is short by that much, positive is satisfied with
that much room.**  Recording them costs nothing — the rules fit all of it every
step already, because the vote order depends on the lights they produce — but
unlike `phase` and the lights it is *not* recoverable from old cells, because
the windows are per-step state the run discarded.  :func:`has_margins` is the
guard, and every margin function returns an empty frame rather than a table of
zeros on a frame that predates them.

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
    print(stopping.binding_note(stopping.summarise(stops)))

    # Post-#3560 cells only: how close the rules came, gate by gate.
    mg = stopping.margins(main, keys=curves.KEYS)
    print(stopping.margin_table(stopping.summarise_margins(mg)))

`selftest_stopping.py` is its planted-answer test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from vtscore.eval.autopilot_flow import (
    SMART_FLAT_THRESHOLD,
    SMART_SLOPE_T,
    STABLE_FALLING_RATIO,
    STABLE_MAX_THRESHOLD,
    STABLE_RATE_THRESHOLD,
    STOPPING_PHASE,
)

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

#: The five gates green is a conjunction over, each as a **margin to green**:
#: ``(name, column, threshold, sense)``, where the margin is
#: ``value - threshold`` when *sense* is ``+`` and ``threshold - value`` when it
#: is ``-``.  One sign convention for all five - **positive means the gate is
#: satisfied, with that much room; negative means the run is that far short** -
#: because the underlying rules point in both directions (Smart wants its slope
#: *above* a negative cutoff, Stable wants its flip rates *below* positive ones)
#: and a table that inherits that inconsistency cannot be read down a column.
#:
#: Smart's two are a **disjunction**: it is green when either is at or above
#: zero, so neither alone says a run was held.  Stable's two and Span's one are
#: conjunctive with each other and with Smart.
#:
#: Two gates are not in this table, for the same reason: their threshold is not
#: a constant.  **Span**'s bar is ``span_target``, which moves with the run's
#: diversity goal and with the atlas size, so its margin is a difference of two
#: columns; :func:`margins` computes it directly.  The third **Stable** gate -
#: the raw rate having stopped falling - compares two halves of the window
#: against each other, so it has no margin at all and is reported as a share of
#: held steps instead.
GATES: tuple[tuple[str, str, float, str], ...] = (
    ("smart_slope", "smart_slope", SMART_FLAT_THRESHOLD, "+"),
    ("smart_t", "smart_slope_t", -SMART_SLOPE_T, "+"),
    ("stable_conf", "stable_confident_flip_rate", STABLE_RATE_THRESHOLD, "-"),
    ("stable_max", "stable_max_confident_flip_rate", STABLE_MAX_THRESHOLD, "-"),
)

#: Names of every margin :func:`margins` reports, in table order - the four
#: constant-threshold gates above plus Span's moving bar.
GATE_NAMES: tuple[str, ...] = (*(g[0] for g in GATES), "span")

#: Every column :func:`margins` reads, so a caller can ask whether a frame
#: predates them without knowing how :data:`GATES` is spelled.  ``span_level``
#: is here because Span's margin needs it, even though it arrived with the
#: lights rather than with the margins.
MARGIN_COLUMNS: tuple[str, ...] = (
    "smart_slope",
    "smart_slope_t",
    "stable_flip_rate",
    "stable_confident_flip_rate",
    "stable_max_confident_flip_rate",
    "stable_flip_rate_early",
    "stable_flip_rate_late",
    "span_level",
    "span_target",
)


def has_margins(main: pd.DataFrame) -> bool:
    """Whether *main* carries the per-step margins, i.e. is a post-#3560 run.

    Older frames have the lights but not the numbers behind them, and every
    margin function here returns an empty frame on one rather than a table of
    zeros.  Unlike the lights, this cannot be back-filled by re-reading cells:
    the slope and flip-rate windows are per-step state that the run threw away.
    """
    return bool(len(main)) and all(c in main.columns for c in MARGIN_COLUMNS)


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
    if math.isnan(f):
        return "—"
    if math.isinf(f):
        # Distinct from "—": an infinite margin is a Smart window with no
        # residual at all (see `margins`), which is measured and extreme, not
        # missing.
        return "∞" if f > 0 else "-∞"
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


def _margin_frame(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-step margins to green for one trajectory's steps, by gate name."""
    out: dict[str, np.ndarray] = {}
    for name, col, thr, sense in GATES:
        v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        out[name] = (v - thr) if sense == "+" else (thr - v)
    lvl = pd.to_numeric(g["span_level"], errors="coerce").to_numpy(dtype=float)
    tgt = pd.to_numeric(g["span_target"], errors="coerce").to_numpy(dtype=float)
    # -1 is "no atlas was read", not "zero nodes short".
    out["span"] = np.where((lvl < 0) | (tgt < 0), np.nan, lvl - tgt)
    return out


def margins(
    main: pd.DataFrame,
    *,
    keys: Sequence[str] = RUN_KEYS,
) -> pd.DataFrame:
    """One row per trajectory: how *close* the rules came, not whether they fired.

    :func:`stopping_points` answers "did the app ever say stop, and where".  This
    answers the question underneath it — **how far short was the run, on which
    gate** — which is what separates an arm that sat one noisy window away from
    green all session from one that was never going to converge.  Both read the
    same rows; neither needs a re-run of anything that has the columns.

    Measured over each run's **held** steps: everything before its first fire,
    or the whole trajectory when it never fired.  What the indicators do after
    the app has said stop is a different question and would otherwise dominate
    the average, since a converged run sits at a comfortable positive margin for
    the rest of its budget.

    Returns an empty frame on a pre-#3560 frame (see :func:`has_margins`) rather
    than a table of zeros: unlike the lights, these cannot be reconstructed by
    re-reading old cells, because the slope and flip-rate windows were per-step
    state the run discarded.

    Columns, per run and per gate in :data:`GATE_NAMES`:

    ``{gate}_median`` / ``{gate}_best``
        The median and the **closest approach** — the largest margin the run
        ever reached — over its held steps.  Both in "margin to green" units:
        negative is short by that much, positive is satisfied with that much
        room (see :data:`GATES`).  ``NaN`` where the rule never fit that gate.
    ``{gate}_green_share``
        Fraction of held steps at which that gate was satisfied.  A gate at 0.9
        with a median margin barely under zero is a rule flapping on noise; one
        at 0.0 is a rule that was never close.
    ``stable_block_avg`` / ``stable_block_max`` / ``stable_block_falling``
        Which of Stable's three gates was doing the blocking, as a share of the
        held steps where Stable was measured and not green.  Attributed in the
        rule's own short-circuit order, so the shares over a blocked step sum to
        one and ``falling`` is only credited when the other two passed.  This is
        the resolution the light cannot give: "Stable held it" is three
        different findings, and they want three different fixes.
    ``n_held`` / ``n_measured``
        Held steps, and how many of those the phase machine actually evaluated
        (a startup schedule's rounds own the phase without consulting the
        indicators, and leave every margin blank).
    """
    kk = _present(main, keys)
    if not has_margins(main) or "t" not in main.columns or "phase" not in main.columns or not kk:
        return pd.DataFrame()

    out: list[dict[str, Any]] = []
    for run_key, g in main.groupby(kk, dropna=False, sort=True):
        g = g.sort_values("t")
        is_done = (g["phase"].astype(str) == STOPPING_PHASE).to_numpy()
        first_i = int(np.argmax(is_done)) if is_done.any() else None
        held = slice(0, first_i if first_i is not None else len(g))
        gh = g.iloc[held]

        row: dict[str, Any] = dict(zip(kk, run_key if isinstance(run_key, tuple) else (run_key,), strict=True))
        row["n_held"] = int(len(gh))
        if not len(gh):
            # Fired on its very first measured step: nothing was ever held.
            row["n_measured"] = 0
            for name in GATE_NAMES:
                row[f"{name}_median"] = row[f"{name}_best"] = row[f"{name}_green_share"] = float("nan")
            for k in ("avg", "max", "falling"):
                row[f"stable_block_{k}"] = float("nan")
            out.append(row)
            continue

        per_step = _margin_frame(gh)
        for name in GATE_NAMES:
            v = per_step[name]
            # `~isnan`, not `isfinite`: a Smart window with no residual at all
            # gives a t of +-inf, which is a measurement (an infinitely
            # significant trend) and not a missing one.  Every statistic below
            # is order-based or a comparison, so all three survive it.
            ok = v[~np.isnan(v)]
            row[f"{name}_median"] = float(np.median(ok)) if ok.size else float("nan")
            row[f"{name}_best"] = float(np.max(ok)) if ok.size else float("nan")
            # Span and Smart's slope gate are green AT zero; Smart's t gate and
            # Stable's two are strict.  The difference is one step in a float
            # margin and never decides a reading, so >= is used throughout.
            row[f"{name}_green_share"] = float(np.mean(ok >= 0.0)) if ok.size else float("nan")

        # Which Stable gate blocked, in the rule's own short-circuit order.
        avg = pd.to_numeric(gh["stable_confident_flip_rate"], errors="coerce").to_numpy(dtype=float)
        mx = pd.to_numeric(gh["stable_max_confident_flip_rate"], errors="coerce").to_numpy(dtype=float)
        early = pd.to_numeric(gh["stable_flip_rate_early"], errors="coerce").to_numpy(dtype=float)
        late = pd.to_numeric(gh["stable_flip_rate_late"], errors="coerce").to_numpy(dtype=float)
        measured = np.isfinite(avg) & np.isfinite(mx) & np.isfinite(early) & np.isfinite(late)
        row["n_measured"] = int(measured.sum())
        fail_avg = measured & (avg >= STABLE_RATE_THRESHOLD)
        fail_max = measured & ~fail_avg & (mx >= STABLE_MAX_THRESHOLD)
        fail_falling = (
            measured & ~fail_avg & ~fail_max & (late >= STABLE_RATE_THRESHOLD) & (late < STABLE_FALLING_RATIO * early)
        )
        blocked = fail_avg | fail_max | fail_falling
        n_blocked = int(blocked.sum())
        for k, arr in (("avg", fail_avg), ("max", fail_max), ("falling", fail_falling)):
            row[f"stable_block_{k}"] = float(arr.sum() / n_blocked) if n_blocked else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def summarise_margins(
    mg: pd.DataFrame,
    *,
    by: Sequence[str] = ("arm",),
) -> pd.DataFrame:
    """Reduce per-run margins to one row per group, by median across runs.

    Median rather than mean throughout: ``smart_t`` is ``±inf`` on a window with
    no residual at all — a real and not especially rare state, since a short
    window of identical costs has nothing to scatter — and one such run would
    take a mean with it.
    """
    if mg.empty:
        return pd.DataFrame()
    gb = _present(mg, by)
    groups: list[tuple[Any, pd.DataFrame]] = list(mg.groupby(gb, dropna=False, sort=True)) if gb else [((), mg)]
    # Numeric, and not one of the columns that *identify* a run rather than
    # measure it - a median of the `seed` column is a number with no referent,
    # and `dataset` would come back NaN and render as a column of dashes.
    skip = {*gb, *RUN_KEYS, "n_held", "n_measured"}
    cols = [c for c in mg.columns if c not in skip and pd.api.types.is_numeric_dtype(mg[c])]

    out: list[dict[str, Any]] = []
    for key, g in groups:
        row: dict[str, Any] = dict(zip(gb, key if isinstance(key, tuple) else (key,), strict=True))
        row["n_runs"] = int(len(g))
        row["median_n_held"] = _q(g["n_held"], 0.5) if "n_held" in g.columns else float("nan")
        for c in cols:
            row[c] = _q(g[c], 0.5)
        out.append(row)
    return pd.DataFrame(out)


def margin_table(summary: pd.DataFrame) -> str:
    """The "how close did it come" block of a REPORT.md, as markdown.

    One row per group.  Each gate is reported as the median margin to green over
    held steps, with the share of held steps it was satisfied at in brackets —
    the pair is the point: a gate whose median margin is ``-0.01`` and whose
    share is 48% is flapping, and a gate at ``-0.01`` and 0% is a wall.
    """
    if summary.empty:
        return "_No margin data: these cells predate the per-step indicator margins (issue #3560)._"
    ident = [c for c in summary.columns if c in ("arm", "dataset", "embedder", "category")]
    labels = {
        "smart_slope": "Smart slope",
        "smart_t": "Smart t",
        "stable_conf": "Stable avg",
        "stable_max": "Stable max",
        "span": "Span nodes",
    }
    head = [*ident, "runs", "held", *(labels[n] for n in GATE_NAMES), "Stable blocked by"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in ident]
        cells.append(str(int(r["n_runs"])))
        cells.append(_fmt(r.get("median_n_held"), 0))
        for name in GATE_NAMES:
            m = r.get(f"{name}_median")
            share = r.get(f"{name}_green_share")
            digits = 0 if name == "span" else 3
            share_txt = "—" if share is None or not math.isfinite(float(share)) else f"{float(share):.0%}"
            cells.append(f"{_fmt(m, digits)} [{share_txt}]")
        parts = []
        for k, lab in (("avg", "avg"), ("max", "max"), ("falling", "falling")):
            v = r.get(f"stable_block_{k}")
            if v is not None and math.isfinite(float(v)):
                parts.append(f"{lab} {float(v):.0%}")
        cells.append(", ".join(parts) if parts else "—")
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

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

All of that is descriptive: it says where the rules fire, not whether firing
there was right.  So each stopping point is also held against **the run's own
best** (``{metric}_from_best``, ``{metric}_clicks_past_best``): a rule that
fires long after the run's best costs users clicks, one that fires long before
it costs them quality, and both read off the same rows.

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
import objective
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

#: The columns a table names its groups by, in the order they are printed: the
#: run keys a study groups on, and ``band``, which the State of the App review
#: reads every stratum by (it is the size half of a ``class@band`` category).
IDENT_COLUMNS: tuple[str, ...] = ("arm", "dataset", "embedder", "category", "band")

#: Metrics carried through to the stopping point by default: the two a ship
#: decision reads - the objective on a balance run, cost without one (#4584;
#: each is dropped where the frame cannot carry it) - and the ranking metric
#: that sits beside them in every report.
DEFAULT_METRICS: tuple[str, ...] = (objective.OBJECTIVE, objective.COST, "average_precision")

#: The indicator columns a post-#3560 run emits.  Absent from every earlier
#: frame, which is why every read of them is guarded.
LIGHT_COLUMNS: tuple[str, ...] = ("smart", "stable", "span")

#: Phases a row can carry that are **not** a reading of the phase machine.  A
#: spot check Autopilot prompts mid-session (``spot_check="weak"``, #4496, the
#: default since 2026-10-05) owns the ``phase`` column for its rounds, which
#: are clicks and stay in the frame.  The phase machine's answer on those
#: steps is not recorded, so they can neither fire the rules nor end a
#: ``done`` stretch: the state is carried across them from the last step that
#: was read.  Without that, a check prompted inside a ``done`` stretch splits
#: it into two episodes and moves ``t_sustained`` past the check.
UNREAD_PHASES: frozenset[str] = frozenset({"prompt"})

#: Autopilot's walk to Done, as the phases each stretch of it is made of, in the
#: order it walks them.  The **opening** is on the text sort; **Hard** refines the
#: boundary until Smart and Stable are both green; **New** walks the coverage
#: atlas until Span is green too.  The two checks are band walks whose rounds
#: are clicks: a **check** Autopilot prompts mid-session on a weak separation
#: (#4496), and the **done check** it runs once all three lights are green,
#: before it reports Done.  Each stretch waits on a different thing, which is
#: why the question "which rule held the run" is only answerable per stretch:
#: Span is never green in Hard, because New is the walk that fills it, so a
#: count over every held click names Span as binding in every session there is.
#: The two checks share their phases and are told apart by what precedes them
#: (see :func:`_click_stages`).
STAGES: tuple[tuple[str, frozenset[str]], ...] = (
    ("opening", frozenset({"good", "bad", "more"})),
    ("hard", frozenset({"hard"})),
    ("new", frozenset({"new"})),
    ("check", frozenset({"spot", "prompt"})),
    ("done_check", frozenset()),
)

#: How a table heads each of :data:`STAGES`.
STAGE_LABELS: dict[str, str] = {
    "opening": "opening",
    "hard": "Hard",
    "new": "New",
    "check": "mid-session check",
    "done_check": "check before Done",
}

#: The stretch each light is read in.  Smart and Stable decide when Hard ends,
#: Span when New does; outside its own stretch a light is either not consulted
#: yet or green by definition of the phase.
LIGHT_STAGE: dict[str, str] = {"smart": "hard", "stable": "hard", "span": "new"}

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


def _done_states(phase: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(fired, held)`` per step of one sorted trajectory.

    ``fired`` is true where the phase machine read ``done``, which is where the
    app could announce: the first one is the stopping point.  ``held`` is the
    state the run was in, read steps as read and :data:`UNREAD_PHASES` steps
    carrying the last read state forward (not ``done`` before any read).  It is
    what episodes and the sustained stop are counted on.
    """
    p = phase.fillna("").astype(str).str.strip()
    fired = (p == STOPPING_PHASE).to_numpy()
    read = ~p.isin(UNREAD_PHASES).to_numpy()
    state = pd.Series(np.where(read, fired.astype(float), np.nan))
    held = state.ffill().fillna(0.0).to_numpy(dtype=float) > 0.5
    return fired, held


def _stage_of(phase: str) -> str:
    for name, phases in STAGES:
        if phase in phases:
            return name
    return ""


def _click_stages(phase: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(stage, clicks, picked_at)`` for the clicks between consecutive rows of one sorted run.

    Interval *i* is the clicks ``(t[i-1], t[i]]`` (``(0, t[0]]`` for the
    first), and its stage is the phase that **picked** them: the previous row's,
    since a row's phase is read after its vote.  The exceptions all come from
    how a prompted check is recorded.  A ``prompt`` row closes one round of a
    check's picks, so its interval is the check's.  The check is the **done
    check** when the run of ``prompt`` rows follows a ``spot`` row (all three
    lights green, the check before Done due), and a mid-session **check**
    otherwise (#4496 prompts it from Hard or New directly).  The row after the
    last round was picked in the state the check left, which the ``prompt``
    row does not record, so it takes its own row's stage, or the check's when
    that row is ``done`` (the check before Done closed on it; the app said Done
    one click before the frame can show it).  *picked_at* is the row whose
    lights were on screen when the interval was picked (``-1`` for the first,
    which nothing preceded).
    """
    n = len(phase)
    stage = np.empty(n, dtype=object)
    picked = np.arange(n) - 1
    check = "check"  # which check the current run of `prompt` rows belongs to
    for i in range(n):
        prev = phase[i - 1] if i else ""
        if phase[i] == "prompt":
            if prev != "prompt":
                check = "done_check" if prev == "spot" else "check"
            stage[i] = check
        elif i == 0:
            stage[i] = "opening"
        elif prev == "prompt":
            stage[i] = _stage_of(phase[i]) or check
        elif prev == "spot":
            # The check before Done was skipped (one ended moments ago): its one click.
            stage[i] = "done_check"
        else:
            stage[i] = _stage_of(prev)
    clicks = np.diff(np.concatenate(([0.0], t.astype(float))))
    return stage, clicks, picked


def _stages(g: pd.DataFrame, first_i: int | None, metrics: Sequence[str]) -> dict[str, Any]:
    """Where one run's clicks went before the stop, and which light each stretch waited on.

    Columns (all over the clicks up to the stop, or the whole run when it never
    fired):

    ``clicks_{stage}``
        Clicks in each of :data:`STAGES`.  Read off ``phase`` alone, so defined
        on any frame since 2026-07-31.
    ``hard_clicks_{light}`` / ``hard_clicks_both``
        Hard's clicks at which Smart, or Stable, was not green when they were
        picked, and those at which both were not.  Click-weighted, so a summary
        can pool them over runs.
    ``hard_exit_light``
        Which of Smart and Stable the last Hard click waited on: ``smart``,
        ``stable`` or ``both``; blank where the run spent no click in Hard.  On
        a run that fired, the light that went green last before it.
    ``lapses_{light}``
        How many times the light went from green to not green before the stop:
        the rule flapping, light by light.  Rows of the opening do not count.
    ``done_lapses_{light}``
        After the stop, how many times ``done`` fell back with this light not
        green.  ``n_done_episodes`` says how often it fell back; this says what
        broke it.
    ``t_hard_exit`` / ``{metric}_at_hard_exit``
        The click where Hard last ended before the stop, and the metric there.
        Against ``t_stop`` and ``{metric}_at_stop`` they are what New and the
        checks after Hard cost and bought.
    """
    ph = g["phase"].fillna("").astype(str).str.strip().to_numpy()
    t = g["t"].to_numpy(dtype=float)
    end = first_i + 1 if first_i is not None else len(g)
    stage, clicks, picked = _click_stages(ph, t)
    stage, clicks, picked = stage[:end], clicks[:end], picked[:end]
    row: dict[str, Any] = {f"clicks_{name}": float(clicks[stage == name].sum()) for name, _ in STAGES}

    has_lights = all(c in g.columns for c in LIGHT_COLUMNS)
    green = {c: (g[c].fillna("").astype(str).to_numpy() == "green") for c in LIGHT_COLUMNS} if has_lights else {}
    hard = np.flatnonzero((stage == "hard") & (picked >= 0))
    if has_lights:
        red_smart = ~green["smart"][picked[hard]]
        red_stable = ~green["stable"][picked[hard]]
        row["hard_clicks_smart"] = float(clicks[hard][red_smart].sum())
        row["hard_clicks_stable"] = float(clicks[hard][red_stable].sum())
        row["hard_clicks_both"] = float(clicks[hard][red_smart & red_stable].sum())
        if hard.size:
            last = picked[hard[-1]]
            s, b = not green["smart"][last], not green["stable"][last]
            row["hard_exit_light"] = "both" if s and b else "smart" if s else "stable" if b else ""
        else:
            row["hard_exit_light"] = ""
        outside = ~np.isin(ph, list(dict(STAGES)["opening"]))
        is_done = ph == STOPPING_PHASE
        for c in LIGHT_COLUMNS:
            seq = green[c][:end][outside[:end]]
            row[f"lapses_{c}"] = int(np.sum(seq[:-1] & ~seq[1:])) if seq.size > 1 else 0
            # After the stop: a `done` row followed by a read row that is not done.
            after = 0
            if first_i is not None:
                read = np.flatnonzero(~np.isin(ph, list(UNREAD_PHASES)))
                read = read[read >= first_i]
                for a, b in zip(read[:-1], read[1:], strict=True):
                    if is_done[a] and not is_done[b] and not green[c][b]:
                        after += 1
            row[f"done_lapses_{c}"] = after if first_i is not None else float("nan")
    else:
        for c in ("smart", "stable", "both"):
            row[f"hard_clicks_{c}"] = float("nan")
        row["hard_exit_light"] = ""
        for c in LIGHT_COLUMNS:
            row[f"lapses_{c}"] = row[f"done_lapses_{c}"] = float("nan")

    row["t_hard_exit"] = float(t[hard[-1]]) if hard.size else float("nan")
    for m in metrics:
        vals = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
        row[f"{m}_at_hard_exit"] = float(vals[hard[-1]]) if hard.size else float("nan")
    return row


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
        delta means the extra clicks made the detector *worse*; for the
        objective, ``fbeta``, a negative one does.
    ``{metric}_best`` / ``{metric}_t_best``
        The best the run ever reached, in the metric's own direction, and the
        first click it reached it at.  Defined for every run, fired or not.
    ``{metric}_from_best`` / ``{metric}_clicks_past_best``
        How far short of that best the stopping point was (never negative), and
        ``t_stop - t_best``: positive is a rule that fired after the run's best
        and cost the user clicks, negative one that fired before it and cost
        them quality.  ``NaN`` where the rules never fired.  The best is the
        extreme of a noisy series, so it is optimistic: ``from_best`` overstates
        what stopping gave away by about the step-to-step noise, and on a flat
        stretch ``clicks_past_best`` is a coin toss.  Read the two together.
    ``n_good_at_stop`` / ``n_bad_at_stop``
        The labelset the user would have left with.
    ``blocked_smart`` / ``blocked_stable``
        Fraction of the run's Hard clicks before the stop at which each light
        was not green: which of the two Hard waits on was doing the holding.
        ``NaN`` on a frame from before the lights were emitted, or with no Hard
        click.  There is no ``blocked_span``: Span is read in New, which only
        exists while Span is not green, so it would be 1 by construction.  Until
        2026-10-14 these were shares of *every* held step, which counted the
        opening and Hard against Span and named it binding in every review.
    ``clicks_{stage}``, ``hard_exit_light``, ``lapses_{light}``, ...
        Where the clicks before the stop went, stretch by stretch, and which
        light each waited on: see :func:`_stages`.
    """
    kk = _present(main, keys)
    if main.empty or "phase" not in main.columns or "t" not in main.columns or not kk:
        return pd.DataFrame()

    if any(m.startswith(objective.OBJECTIVE) for m in metrics):
        # A balance-era frame written before the objective's columns (#4584).
        main = objective.with_objective(main)
    if objective.OBJECTIVE in metrics and not objective.carries_beta(main):
        # No balance drew the line: there is no objective to carry, only NaN.
        metrics = [m for m in metrics if m != objective.OBJECTIVE]
    metrics = [m for m in metrics if m in main.columns]
    lower = {m: objective.lower_is_better(m) for m in metrics}
    out: list[dict[str, Any]] = []
    for run_key, g in main.groupby(kk, dropna=False, sort=True):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        fired, is_done = _done_states(g["phase"])

        first_i = int(np.argmax(fired)) if fired.any() else None
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
            # The run's own best (#3560's "is the stopping cost near the run's
            # best?"): the first click at the extreme, in the metric's direction.
            if np.isnan(vals).all():
                best, t_best = float("nan"), float("nan")
            else:
                i_best = int(np.nanargmin(vals) if lower[m] else np.nanargmax(vals))
                best, t_best = float(vals[i_best]), float(t[i_best])
            row[f"{m}_best"] = best
            row[f"{m}_t_best"] = t_best
            row[f"{m}_from_best"] = (at_stop - best) if lower[m] else (best - at_stop)
            row[f"{m}_clicks_past_best"] = row["t_stop"] - t_best
        for col, name in (("n_good", "n_good_at_stop"), ("n_bad", "n_bad_at_stop")):
            if col in g.columns and first_i is not None:
                row[name] = float(pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)[first_i])
            else:
                row[name] = float("nan")

        # Where the clicks before the stop went, and which light each stretch
        # waited on.  A run that never entered `done` is held at every click;
        # one that did only up to its first fire, because what happens after
        # the app said stop is a different question.
        row.update(_stages(g, first_i, metrics))
        for light in ("smart", "stable"):
            hard = row["clicks_hard"]
            row[f"blocked_{light}"] = row[f"hard_clicks_{light}"] / hard if hard else float("nan")
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
    groups: list[tuple[Any, pd.DataFrame]] = (
        list(stops.groupby(gb, dropna=False, sort=True, observed=True)) if gb else [((), stops)]
    )
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
            # Against each run's own best, over the same fired runs.  Absent
            # from a frame stopping_points did not write (a hand-built one).
            for col in ("from_best", "clicks_past_best"):
                if f"{m}_{col}" in fired.columns:
                    row[f"median_{m}_{col}"] = _q(fired[f"{m}_{col}"], 0.5)
        row.update(_summarise_stages(g, fired, metrics))
        out.append(row)
    return pd.DataFrame(out)


def _pooled(num: pd.Series, den: pd.Series) -> float:
    """``sum(num) / sum(den)`` over the runs where both are measured: a share of clicks, not a median of shares."""
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    ok = n.notna() & d.notna()
    total = float(d[ok].sum())
    return float(n[ok].sum()) / total if total > 0 else float("nan")


def _summarise_stages(g: pd.DataFrame, fired: pd.DataFrame, metrics: Sequence[str]) -> dict[str, Any]:
    """The stage columns of one :func:`summarise` row.

    The clicks per stage describe the runs that fired, like every column after
    ``fire_rate``.  What held Hard pools every run's held clicks, fired or not:
    a run that never stopped is the one most worth asking that of.
    """
    row: dict[str, Any] = {}
    if "clicks_hard" not in g.columns:
        return row
    for name, _ in STAGES:
        row[f"median_clicks_{name}"] = _q(fired[f"clicks_{name}"], 0.5)
        row[f"mean_clicks_{name}"] = float(fired[f"clicks_{name}"].mean()) if len(fired) else float("nan")
    for light in ("smart", "stable", "both"):
        row[f"hard_held_{light}"] = _pooled(g[f"hard_clicks_{light}"], g["clicks_hard"])
    # Kept under their old name for a reader of an older summary; now Hard's clicks.
    for light in ("smart", "stable"):
        row[f"blocked_{light}"] = row[f"hard_held_{light}"]
    exits = (
        fired["hard_exit_light"].fillna("").astype(str) if "hard_exit_light" in fired.columns else pd.Series(dtype=str)
    )
    exits = exits[exits != ""]
    for light in ("smart", "stable", "both"):
        row[f"hard_exit_{light}"] = float(np.mean(exits == light)) if len(exits) else float("nan")
    for light in LIGHT_COLUMNS:
        row[f"lapses_{light}"] = float(pd.to_numeric(g[f"lapses_{light}"], errors="coerce").mean())
        row[f"done_lapses_{light}"] = (
            float(pd.to_numeric(fired[f"done_lapses_{light}"], errors="coerce").mean()) if len(fired) else float("nan")
        )
    for m in metrics:
        col = f"{m}_at_hard_exit"
        if col in fired.columns:
            row[f"median_{col}"] = _q(fired[col], 0.5)
            # Paired within run: what New and the check before Done bought.
            row[f"median_{m}_hard_exit_to_stop"] = _q(fired[f"{m}_at_stop"] - fired[col], 0.5)
    return row


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


def stopping_table(summary: pd.DataFrame, *, metric: str | None = None) -> str:
    """The stopping block of a REPORT.md, as markdown.

    One row per group.  The columns are the issue's two questions — *where* the
    rules fired and *what it cost there* — plus the two qualifications without
    which neither number can be read: how many runs ever fired, and how much
    the budget's extra clicks moved the metric afterwards.  The last two hold
    the stop against each run's own best: how far short of it, and how many
    clicks after it (negative: before).  *metric* defaults to the objective
    where the runs carried one, else cost (#4584).
    """
    if summary.empty:
        return "_No stopping data: no run carried a `phase` column._"
    if metric is None:
        col = f"median_{objective.OBJECTIVE}_at_stop"
        has_objective = col in summary.columns and bool(summary[col].notna().any())
        metric = objective.OBJECTIVE if has_objective else objective.COST
    ident = [c for c in summary.columns if c in IDENT_COLUMNS]
    head = [*ident, "runs", "fired", "stop click (KM)", "stop click (median of fired)"]
    head += [f"{metric} at stop", f"{metric} at budget", f"Δ{metric} (paired)", "clicks after stop"]
    head += ["short of run's best", "clicks past best"]
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
        cells.append(_fmt(r.get(f"median_{metric}_from_best")))
        cells.append(_fmt(r.get(f"median_{metric}_clicks_past_best"), 0))
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

    Measured over each run's **held** steps — everything before its first fire,
    or the whole trajectory when it never fired — and, within them, only over
    the stretch that waits on each gate (:data:`LIGHT_STAGE`): Smart's and
    Stable's over Hard, Span's over New.  What the indicators do after the app
    has said stop is a different question and would otherwise dominate the
    average, since a converged run sits at a comfortable positive margin for
    the rest of its budget.  And a gate read outside its own stretch is either
    not consulted yet (Span in Hard, which no walk has filled) or green by
    definition of the phase (Smart and Stable in New); until 2026-10-14 both
    were counted, which put Span's margin at -35 nodes in a review whose
    sessions all walked it to green.

    Returns an empty frame on a pre-#3560 frame (see :func:`has_margins`) rather
    than a table of zeros: unlike the lights, these cannot be reconstructed by
    re-reading old cells, because the slope and flip-rate windows were per-step
    state the run discarded.

    Columns, per run and per gate in :data:`GATE_NAMES`:

    ``{gate}_median`` / ``{gate}_best``
        The median and the **closest approach** — the largest margin the run
        ever reached — over its held steps in the gate's stretch.  Both in "margin to green" units:
        negative is short by that much, positive is satisfied with that much
        room (see :data:`GATES`).  ``NaN`` where the rule never fit that gate.
    ``{gate}_green_share`` / ``{gate}_n``
        Fraction of those steps at which that gate was satisfied, and how many
        steps carried a margin (the weight :func:`summarise_margins` pools the
        share by).  A gate at 0.9 with a median margin barely under zero is a
        rule flapping on noise; one at 0.0 is a rule that was never close.
    ``stable_block_warmup`` / ``stable_block_avg`` / ``stable_block_max`` / ``stable_block_falling``
        What kept Stable from green, as a share of the Hard steps at which it
        was not green (``stable_blocked_n`` of them).  ``warmup`` is Stable not
        yet measuring: fewer than five labels of a class, or fewer than five
        retrains of history, so no gate was read.  The other three are its
        gates, attributed in the rule's own short-circuit order, so
        ``falling`` is only credited when the other two passed.  This is the
        resolution the light cannot give: "Stable held it" is four different
        findings, and they want four different fixes.  On a frame without the
        ``stable`` light, the denominator is the steps a gate failed and there
        is no ``warmup``.
    ``n_held`` / ``n_hard`` / ``n_new`` / ``n_measured``
        Held steps, those of them in Hard and in New, and how many Hard steps
        Stable actually measured (a startup schedule's rounds own the phase
        without consulting the indicators, and leave every margin blank).
    """
    kk = _present(main, keys)
    if not has_margins(main) or "t" not in main.columns or "phase" not in main.columns or not kk:
        return pd.DataFrame()

    out: list[dict[str, Any]] = []
    for run_key, g in main.groupby(kk, dropna=False, sort=True):
        g = g.sort_values("t")
        fired, _ = _done_states(g["phase"])
        first_i = int(np.argmax(fired)) if fired.any() else None
        held = slice(0, first_i if first_i is not None else len(g))
        gh = g.iloc[held]

        row: dict[str, Any] = dict(zip(kk, run_key if isinstance(run_key, tuple) else (run_key,), strict=True))
        row["n_held"] = int(len(gh))
        ph = gh["phase"].fillna("").astype(str).str.strip().to_numpy()
        in_stage = {stage: ph == stage for stage in set(LIGHT_STAGE.values())}
        row["n_hard"] = int(in_stage["hard"].sum())
        row["n_new"] = int(in_stage["new"].sum())

        per_step = _margin_frame(gh)
        for name in GATE_NAMES:
            light = "span" if name == "span" else name.split("_", 1)[0]
            v = per_step[name][in_stage[LIGHT_STAGE[light]]]
            # `~isnan`, not `isfinite`: a Smart window with no residual at all
            # gives a t of +-inf, which is a measurement (an infinitely
            # significant trend) and not a missing one.  Every statistic below
            # is order-based or a comparison, so all three survive it.
            ok = v[~np.isnan(v)]
            row[f"{name}_n"] = int(ok.size)
            row[f"{name}_median"] = float(np.median(ok)) if ok.size else float("nan")
            row[f"{name}_best"] = float(np.max(ok)) if ok.size else float("nan")
            # Span and Smart's slope gate are green AT zero; Smart's t gate and
            # Stable's two are strict.  The difference is one step in a float
            # margin and never decides a reading, so >= is used throughout.
            row[f"{name}_green_share"] = float(np.mean(ok >= 0.0)) if ok.size else float("nan")

        # What kept Stable from green over Hard, in the rule's own short-circuit order.
        gs = gh[in_stage["hard"]]
        avg = pd.to_numeric(gs["stable_confident_flip_rate"], errors="coerce").to_numpy(dtype=float)
        mx = pd.to_numeric(gs["stable_max_confident_flip_rate"], errors="coerce").to_numpy(dtype=float)
        early = pd.to_numeric(gs["stable_flip_rate_early"], errors="coerce").to_numpy(dtype=float)
        late = pd.to_numeric(gs["stable_flip_rate_late"], errors="coerce").to_numpy(dtype=float)
        measured = np.isfinite(avg) & np.isfinite(mx) & np.isfinite(early) & np.isfinite(late)
        row["n_measured"] = int(measured.sum())
        fail_avg = measured & (avg >= STABLE_RATE_THRESHOLD)
        fail_max = measured & ~fail_avg & (mx >= STABLE_MAX_THRESHOLD)
        fail_falling = (
            measured & ~fail_avg & ~fail_max & (late >= STABLE_RATE_THRESHOLD) & (late < STABLE_FALLING_RATIO * early)
        )
        if "stable" in gs.columns:
            not_green = gs["stable"].fillna("").astype(str).to_numpy() != "green"
            warmup = not_green & ~measured
            blocked = warmup | fail_avg | fail_max | fail_falling
        else:
            warmup = np.zeros(len(gs), dtype=bool)
            blocked = fail_avg | fail_max | fail_falling
        n_blocked = int(blocked.sum())
        row["stable_blocked_n"] = n_blocked
        for k, arr in (("warmup", warmup), ("avg", fail_avg), ("max", fail_max), ("falling", fail_falling)):
            row[f"stable_block_{k}"] = float(arr.sum() / n_blocked) if n_blocked else float("nan")
        out.append(row)
    return pd.DataFrame(out)


def summarise_margins(
    mg: pd.DataFrame,
    *,
    by: Sequence[str] = ("arm",),
) -> pd.DataFrame:
    """Reduce per-run margins to one row per group.

    Margins by median across runs, rather than mean: ``smart_t`` is ``±inf`` on
    a window with no residual at all — a real and not especially rare state,
    since a short window of identical costs has nothing to scatter — and one
    such run would take a mean with it.

    Shares are **pooled** over the group's steps instead (each run's share
    weighted by the steps it was taken over), so a share reads as "of all the
    Hard steps in this group" and Stable's four reasons sum to one.  A median of
    per-run shares does neither: it is taken over a different subset of runs
    for each column, and put a gate green on 100% of held steps beside the same
    gate blamed for 75% of Stable's blocks in the 2026-10-08 review.
    """
    if mg.empty:
        return pd.DataFrame()
    gb = _present(mg, by)
    groups: list[tuple[Any, pd.DataFrame]] = (
        list(mg.groupby(gb, dropna=False, sort=True, observed=True)) if gb else [((), mg)]
    )
    # Numeric, and not one of the columns that *identify* a run rather than
    # measure it - a median of the `seed` column is a number with no referent,
    # and `dataset` would come back NaN and render as a column of dashes.
    counts = {"n_held", "n_hard", "n_new", "n_measured", "stable_blocked_n", *(f"{n}_n" for n in GATE_NAMES)}
    skip = {*gb, *RUN_KEYS, *counts}
    cols = [c for c in mg.columns if c not in skip and pd.api.types.is_numeric_dtype(mg[c])]
    weight = {f"{n}_green_share": f"{n}_n" for n in GATE_NAMES}
    weight.update({c: "stable_blocked_n" for c in cols if c.startswith("stable_block_")})

    out: list[dict[str, Any]] = []
    for key, g in groups:
        row: dict[str, Any] = dict(zip(gb, key if isinstance(key, tuple) else (key,), strict=True))
        row["n_runs"] = int(len(g))
        for c in ("n_held", "n_hard", "n_new"):
            row[f"median_{c}"] = _q(g[c], 0.5) if c in g.columns else float("nan")
        for c in cols:
            w = weight.get(c)
            if w is not None and w in g.columns:
                share = pd.to_numeric(g[c], errors="coerce")
                row[c] = _pooled(share * pd.to_numeric(g[w], errors="coerce"), g[w].where(share.notna()))
            else:
                row[c] = _q(g[c], 0.5)
        out.append(row)
    return pd.DataFrame(out)


def margin_table(summary: pd.DataFrame) -> str:
    """The "how close did it come" block of a REPORT.md, as markdown.

    One row per group.  Each gate is reported as the median margin to green over
    the held steps of its own stretch (Smart's and Stable's over Hard, Span's
    over New), with the pooled share of those steps it was satisfied at in
    brackets — the pair is the point: a gate whose median margin is ``-0.01``
    and whose share is 48% is flapping, and a gate at ``-0.01`` and 0% is a
    wall.  Span's margin is the walk still to go, not a gate failing: New is
    the walk that fills it.
    """
    if summary.empty:
        return "_No margin data: these cells predate the per-step indicator margins (issue #3560)._"
    ident = [c for c in summary.columns if c in IDENT_COLUMNS]
    labels = {
        "smart_slope": "Smart slope",
        "smart_t": "Smart t",
        "stable_conf": "Stable avg",
        "stable_max": "Stable max",
        "span": "Span nodes",
    }
    head = [*ident, "runs", "Hard steps", "New steps", *(labels[n] for n in GATE_NAMES), "Stable blocked by"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in ident]
        cells.append(str(int(r["n_runs"])))
        cells.append(_fmt(r.get("median_n_hard"), 0))
        cells.append(_fmt(r.get("median_n_new"), 0))
        for name in GATE_NAMES:
            m = r.get(f"{name}_median")
            share = r.get(f"{name}_green_share")
            digits = 0 if name == "span" else 3
            share_txt = "—" if share is None or not math.isfinite(float(share)) else f"{float(share):.0%}"
            cells.append(f"{_fmt(m, digits)} [{share_txt}]")
        parts = []
        for k, lab in (("warmup", "warm-up"), ("avg", "avg"), ("max", "max"), ("falling", "falling")):
            v = r.get(f"stable_block_{k}")
            if v is not None and math.isfinite(float(v)):
                parts.append(f"{lab} {float(v):.0%}")
        cells.append(", ".join(parts) if parts else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage_table(summary: pd.DataFrame, *, metric: str | None = None) -> str:
    """Where the clicks before the stop went, stretch by stretch, as markdown.

    One row per group, over the runs that fired.  Means rather than medians, so
    the stretches add up to the stop click beside them.  The last three columns
    are the metric where Hard last ended, at the stop, and the paired change
    between them: what New's atlas walk and the check before Done bought.
    """
    if summary.empty or "mean_clicks_hard" not in summary.columns:
        return "_No stage data: no run carried a `phase` column._"
    if metric is None:
        col = f"median_{objective.OBJECTIVE}_at_stop"
        has_objective = col in summary.columns and bool(summary[col].notna().any())
        metric = objective.OBJECTIVE if has_objective else objective.COST
    ident = [c for c in summary.columns if c in IDENT_COLUMNS]
    head = [*ident, "fired", "stop click (mean of fired)", *(STAGE_LABELS[name] for name, _ in STAGES)]
    head += [f"{metric} at Hard's end", f"{metric} at stop", "Δ (paired)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in summary.iterrows():
        cells = [str(r[c]) for c in ident]
        cells.append(f"{int(r['n_fired'])} ({r['fire_rate']:.0%})")
        total = sum(float(r.get(f"mean_clicks_{name}", float("nan"))) for name, _ in STAGES)
        cells.append(_fmt(total, 0))
        cells += [_fmt(r.get(f"mean_clicks_{name}"), 0) for name, _ in STAGES]
        cells.append(_fmt(r.get(f"median_{metric}_at_hard_exit")))
        cells.append(_fmt(r.get(f"median_{metric}_at_stop")))
        cells.append(_fmt(r.get(f"median_{metric}_hard_exit_to_stop")))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(v: Any) -> str:
    return f"{float(v):.0%}" if v is not None and math.isfinite(float(v)) else "—"


def binding_note(summary: pd.DataFrame) -> str:
    """Which rule held runs short of stopping, read stretch by stretch, or why that is unknowable.

    Hard waits on Smart and Stable, so which of the two held the run is a
    question about Hard's clicks.  New waits on Span alone, and New *is* the
    walk that fills Span, so Span is never the light that fails: it is a walk
    of so many clicks, and the note says how many.  Only answerable on a
    post-#3560 frame; on an older one it says so instead of guessing, because
    the phase alone cannot separate Smart from Stable inside Hard.
    """
    if summary.empty:
        return ""
    cols = ["hard_held_smart", "hard_held_stable"]
    if not all(c in summary.columns for c in cols) or summary[cols].isna().all().all():
        return (
            "The binding indicator is not recoverable from these cells: they predate the "
            "`smart` / `stable` / `span` columns, and `phase` alone cannot tell Smart from "
            "Stable inside `hard`. Re-run to answer it."
        )
    ident = [c for c in summary.columns if c in IDENT_COLUMNS]
    notes = []
    for _, r in summary.iterrows():
        held = {c: float(r[f"hard_held_{c}"]) for c in ("smart", "stable")}
        top = max((v for v in held.values() if math.isfinite(v)), default=float("nan"))
        # A tie is reported as one: naming the first of two equal lights would
        # pin the block on whichever this module happens to list first.
        worst = [c for c in held if held[c] == top]
        if len(worst) > 1:
            verdict = f"{' and '.join(f'**{c}**' for c in worst)} tie as the binding rule in Hard"
        elif worst:
            verdict = f"**{worst[0]}** is the binding rule in Hard"
        else:
            verdict = "no Hard click was measured"
        prefix = ("/".join(str(r[c]) for c in ident) + ": ") if len(summary) > 1 and ident else ""
        notes.append(
            f"{prefix}{verdict}: Smart was not green on {_pct(held['smart'])} of Hard's clicks and Stable on "
            f"{_pct(held['stable'])} (both on {_pct(r.get('hard_held_both'))}). Hard last ended waiting on Smart in "
            f"{_pct(r.get('hard_exit_smart'))} of the sessions that stopped, on Stable in "
            f"{_pct(r.get('hard_exit_stable'))}, on both in {_pct(r.get('hard_exit_both'))}. New waits on Span "
            f"alone and is the atlas walk that turns it green, so Span is not a gate that fails: it is "
            f"{_fmt(r.get('mean_clicks_new'), 0)} clicks of walk, and the check before Done "
            f"{_fmt(r.get('mean_clicks_done_check'), 0)} more. Before the stop a light went from green to not "
            f"green {_fmt(r.get('lapses_smart'), 1)} times a session for Smart, "
            f"{_fmt(r.get('lapses_stable'), 1)} for Stable and {_fmt(r.get('lapses_span'), 1)} for Span; after "
            f"it, `done` fell back {_fmt(r.get('done_lapses_smart'), 1)} / {_fmt(r.get('done_lapses_stable'), 1)} "
            f"/ {_fmt(r.get('done_lapses_span'), 1)} times a session on Smart / Stable / Span."
        )
    return "\n\n".join(notes)

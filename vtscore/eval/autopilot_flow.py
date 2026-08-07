"""The app's Autopilot phase machine, ported for the voting-iterations harness.

The eval simulates a VTSearch user driving Autopilot.  For that simulation to
mean anything, the simulated user has to move through the *same phases*, on the
same triggers, as the real one — otherwise a study measures a flow nobody takes.
This module is the port of that logic, kept deliberately small and pure so it
can be diffed against its two sources by eye:

* :func:`next_phase` mirrors ``AutopilotStateService.checkPhaseTransition``
  (``frontend/src/app/services/autopilot-state.service.ts``).
* :func:`smart_status`, :func:`stable_status`, and :func:`span_status` mirror
  ``_compute_smart_status`` / ``_compute_stable_status`` / ``_compute_span_status``
  in :mod:`vtscore.detectors.labeling_progress`, which is what the app's
  ``/api/labeling-status`` poll feeds into the phase machine.

Why a port rather than a call: the phase machine itself lives in TypeScript, so
there is nothing to import; and the indicator functions in
``labeling_progress`` are wrapped in a module-level, lock-guarded MLP cache
built for an interactive session (one detector, advancing one vote at a time).
A simulation runs thousands of independent trajectories and already holds every
per-step model it needs, so it feeds those inputs in directly.  The rules —
every threshold and constant below — are copied verbatim, and
``tests_lib/detectors/test_autopilot_flow.py`` pins them against the sources.

Because this is a copy, it can go stale silently: change a phase trigger in the
app and the simulated user keeps taking the old route, so every study run after
that measures a flow nobody takes.  ``scripts/check-eval-app-sync.py`` — a
``./run-tests.sh`` gate — digests both sources and fails when either moves; see
"The Eval Default Arm IS the App" in ``docs/EVAL.md``.

The phase ordering the app implements, and the harness therefore reproduces:

1. ``good`` until ``good_target`` positives exist (text sort, take the top).
2. ``bad`` until ``bad_target`` negatives exist — still on the **text/example
   sort**, taking the item nearest that sort's cutoff.  No detector is trained
   in this phase, which is why the app never computes a threshold from a
   1-vs-1 fit (see issue #2788).
3. ``hard`` — the first learned sort.  Refine the boundary until the detector
   is both *smart* (error cost has levelled off) and *stable* (predictions have
   stopped flipping).
4. ``new`` — explore the coverage atlas until *span* goes green too.
5. ``done`` (all three green) or ``exhausted`` (nothing left to label).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

Phase = Literal["idle", "good", "bad", "hard", "new", "done", "exhausted"]
Status = Literal["red", "yellow", "green"]

# Autopilot's initial-phase vote targets (``INITIAL_STATE.goodToStart`` /
# ``badToStart``).  The first learned sort therefore happens at 3 + 4 = 7 votes,
# which always clears the calibrator's >=2-per-class fold-split guard.
GOOD_TARGET = 3
BAD_TARGET = 4

# ``_compute_smart_status`` / ``_compute_stable_status``: both indicators stay
# red until the labelset has at least this many of each class.
MIN_PER_CLASS = 5

# ``_compute_smart_status``: window of recent steps the error-cost trend is
# regressed over, the minimum number of points that makes a trend meaningful,
# and the relative-slope cutoff below which the cost is still falling.
SMART_WINDOW = 10
SMART_MIN_POINTS = 3
SMART_FLAT_THRESHOLD = -0.015

# ``_compute_stable_status``: the flip-rate window and its two cutoffs — the
# average must be under 0.5% and no single recent step above 1%.
STABLE_WINDOW = 10
STABLE_MIN_ENTRIES = 5
STABLE_RATE_THRESHOLD = 0.005
STABLE_MAX_THRESHOLD = 0.01

# ``_compute_span_status``: yellow once this many atlas nodes carry evidence;
# green at ``autopilot_goal_diversity`` (default 40), capped at the tree size.
SPAN_YELLOW = 10
SPAN_GREEN_DEFAULT = 40


def smart_status(recent_error_costs: list[float], good: int, bad: int) -> Status:
    """Port of ``_compute_smart_status``: has the error cost levelled off?

    *recent_error_costs* are the per-step costs of the last :data:`SMART_WINDOW`
    cached models, each scored against the **current** labelset (never the
    held-out test split — the app has no test labels, and using them here would
    leak into the vote order).  Green once the least-squares slope, normalised
    by the mean cost, stops falling faster than :data:`SMART_FLAT_THRESHOLD`.
    """
    if good < MIN_PER_CLASS or bad < MIN_PER_CLASS:
        return "red"
    costs = list(recent_error_costs)[-SMART_WINDOW:]
    if len(costs) < SMART_MIN_POINTS:
        return "yellow"

    n_pts = len(costs)
    x_vals = list(range(n_pts))
    x_mean = sum(x_vals) / n_pts
    y_mean = sum(costs) / n_pts
    numer = sum((x_vals[i] - x_mean) * (costs[i] - y_mean) for i in range(n_pts))
    denom = sum((x_vals[i] - x_mean) ** 2 for i in range(n_pts))
    slope = numer / denom if denom != 0 else 0.0
    relative_slope = slope / y_mean if y_mean > 0 else slope

    return "yellow" if relative_slope < SMART_FLAT_THRESHOLD else "green"


def stable_status(stability_entries: list[dict[str, Any]], good: int, bad: int) -> Status:
    """Port of ``_compute_stable_status``: have predictions stopped flipping?

    Each entry is ``{"num_flips": int, "num_unlabeled": int}`` for one step —
    how many unlabeled items changed predicted class since the previous step.
    Keyed off the flip *rate* rather than a raw count so the cutoff means the
    same thing on a 1k and a 1M item collection.
    """
    if good < MIN_PER_CLASS or bad < MIN_PER_CLASS:
        return "red"
    if len(stability_entries) < STABLE_MIN_ENTRIES:
        return "yellow"

    recent = stability_entries[-STABLE_WINDOW:]
    flip_rates: list[float] = []
    for s in recent:
        n_unlabeled = s.get("num_unlabeled", 0)
        flip_rates.append(s["num_flips"] / n_unlabeled if n_unlabeled > 0 else 0.0)

    avg_flip_rate = sum(flip_rates) / len(flip_rates)
    max_flip_rate = max(flip_rates)
    if avg_flip_rate < STABLE_RATE_THRESHOLD and max_flip_rate < STABLE_MAX_THRESHOLD:
        return "green"
    return "yellow"


def span_status(level: int, depth: int, green_at: int = SPAN_GREEN_DEFAULT) -> Status:
    """Port of ``_compute_span_status``: how much of the atlas carries evidence?

    *level* is the atlas's count of consecutive BFS-order evidence-bearing
    nodes and *depth* its total node count (i.e.
    :meth:`CoverageAtlas.span_info`'s ``level`` / ``depth``).  A degenerate tree
    (no nodes) reads green, matching the app — there is no diversity left to
    chase.
    """
    if depth <= 0:
        return "green"
    green = min(green_at, depth)
    yellow = min(SPAN_YELLOW, green)
    if level >= green:
        return "green"
    if level >= yellow:
        return "yellow"
    return "red"


def next_phase(
    good_count: int,
    bad_count: int,
    *,
    remaining_unlabeled: float,
    smart: Status,
    stable: Status,
    span: Status,
    good_target: int = GOOD_TARGET,
    bad_target: int = BAD_TARGET,
) -> Phase:
    """Port of ``AutopilotStateService.checkPhaseTransition``.

    *remaining_unlabeled* is how many pool items still carry no vote; pass
    ``math.inf`` when the collection size is unknown (the app's "size unknown"
    case, which leaves the targets uncapped and never exhausts).

    The phase is derived from counts and indicator statuses rather than
    accumulated, so it can move backwards — un-toggling votes regresses the
    phase, exactly as it does in the app.
    """
    # Cap each target at the most votes of that class the collection could still
    # yield, so a tiny dataset can still advance past the initial phases instead
    # of stranding in ``good`` forever.
    eff_good_target = min(good_target, good_count + remaining_unlabeled)
    eff_bad_target = min(bad_target, bad_count + remaining_unlabeled)

    if good_count < eff_good_target:
        return "good"
    if bad_count < eff_bad_target:
        return "bad"
    if smart == "green" and stable == "green" and span == "green":
        return "done"
    if remaining_unlabeled == 0:
        return "exhausted"
    if smart == "green" and stable == "green":
        return "new"
    return "hard"


#: Phases in which the app has run a learned sort, so a trained detector's
#: scores and threshold are on screen and driving the next pick.  Before this
#: the user is still on the text/example sort (see the module docstring).
TRAINED_PHASES: frozenset[str] = frozenset({"hard", "new", "done", "exhausted"})


def app_has_detector(phase: str) -> bool:
    """Whether the app would have a trained detector on screen in *phase*.

    The harness records this per step as ``app_trained``: a threshold computed
    at a step where this is false is a number no user ever sees, and studies
    about threshold quality (issue #2788) must filter on it rather than
    counting every simulated step.
    """
    return phase in TRAINED_PHASES


class AutopilotFlow:
    """Per-trajectory phase state for one simulated Autopilot session.

    Accumulates the two histories the indicators need — per-step error costs
    (Smart) and per-step prediction-flip counts (Stable) — and recomputes the
    phase after every vote.  The atlas span is read from the harness's atlas,
    which it labels in lock-step with the votes.
    """

    def __init__(self, *, good_target: int = GOOD_TARGET, bad_target: int = BAD_TARGET, span_green: int | None = None):
        self.good_target = good_target
        self.bad_target = bad_target
        self.span_green = SPAN_GREEN_DEFAULT if span_green is None else span_green
        self.phase: Phase = "good"
        self.error_costs: list[float] = []
        self.stability: list[dict[str, Any]] = []
        self._prev_predictions: Optional[dict[int, int]] = None

    def record_step(self, error_cost: float | None, predictions: dict[int, int] | None) -> None:
        """Fold one step's model into the Smart / Stable histories.

        *error_cost* is that step's weighted FPR/FNR on the current labelset
        (``None`` when no model was trained).  *predictions* maps each still-
        unlabeled pool id to its predicted class, from which the flip count
        against the previous step is derived.
        """
        if error_cost is not None:
            self.error_costs.append(error_cost)
        if predictions is None:
            return
        if self._prev_predictions is not None:
            shared = predictions.keys() & self._prev_predictions.keys()
            num_flips = sum(1 for cid in shared if predictions[cid] != self._prev_predictions[cid])
            self.stability.append({"num_flips": num_flips, "num_unlabeled": len(predictions)})
        self._prev_predictions = dict(predictions)

    def update(self, good_count: int, bad_count: int, remaining_unlabeled: float, span: dict[str, Any] | None) -> Phase:
        """Recompute and return the phase after a vote."""
        smart = smart_status(self.error_costs, good_count, bad_count)
        stable = stable_status(self.stability, good_count, bad_count)
        sp: Status = span_status(int(span["level"]), int(span["depth"]), self.span_green) if span is not None else "red"
        self.phase = next_phase(
            good_count,
            bad_count,
            remaining_unlabeled=remaining_unlabeled,
            smart=smart,
            stable=stable,
            span=sp,
            good_target=self.good_target,
            bad_target=self.bad_target,
        )
        return self.phase

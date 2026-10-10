# Reporting the app's stopping rules in eval studies (issue #3560)

**Status:** the measurement layer has shipped, and the State of the App
analyzer reports it. The first reading of a real review is done: the 2026-10-08
Binary Photo review's [stopping section](../experiments/2026-10-08-state-of-the-app-binary-photo/REPORT.md)
(#4611). The stop fires in about 80% of sessions, around click 80 to 90, and comes early:
Fβ rises 0.05 to 0.08 after it. What is owed is adopting the convention, and the
"are the rules any good" questions.

## Background

Every simulated-user study quotes a **final cost**: the metric at the last click
of a fixed budget (`CALIB_MAX_STEPS`, usually 150). The app has its own notion of
finished — the Smart / Stable / Span indicators all green, at which point the
phase becomes `done` and the panel offers the export hand-off. So "final" in
every published report is a click count nobody chose, and it is not the click
at which any user would actually stop.

What already exists, and is therefore *not* owed:

- The harness runs the app's phase machine on every autopilot-fidelity run and
  has emitted **`phase`** on every metric row since 2026-07-31.
- Since #3560 it also emits the three indicator lights (`smart` / `stable` /
  `span`) and the raw span counts (`span_level` / `span_depth` / `span_target`),
  which say *which* rule held a run short of stopping — and, beside them, the
  **margins**: the continuous quantities those gates are thresholds on
  (`smart_slope`, `smart_slope_t`, and the four Stable flip rates), which say
  *how close* each came. `stopping.margins()` / `summarise_margins()` /
  `margin_table()` report them, and attribute a Stable block to one of its three
  gates. See [`docs/EVAL.md`](../EVAL.md#stopping-point-and-stopping-cost-issue-3560).
- [`scripts/experiments/calibration/stopping.py`](../../scripts/experiments/calibration/stopping.py)
  derives the stopping point and stopping cost from those columns, with the
  censoring and flapping handled, holds each stop against the run's own best
  (`{metric}_from_best`, `{metric}_clicks_past_best`), and reads through a spot
  check prompted mid-session (`phase == "prompt"`); `curves.quality_vs_clicks(..., stops=…)`
  marks the stopping click on the mandatory averaged figure.
- The State of the App analyzer
  ([`state_of_app/analyze.py`](../../scripts/experiments/state_of_app/analyze.py))
  writes `stops.csv`, `margins.csv` and a "Where the app said stop" block in
  `summary.md`, per production path and per band, at the review's objective.
- [`docs/EVAL.md`](../EVAL.md) documents the columns and the derivation.

**The marginal compute cost of all of this is zero.** The phase machine's inputs
— the Smart error-cost window, the Stable prediction flips, the Span atlas — are
computed every step already, because the vote order depends on them; the lights
were being discarded. A local measurement puts the whole Smart-window rescoring
at 1–4% of a run's wall clock, and that 1–4% was already being paid before this
issue. No study gets slower for reporting a stopping point.

<!-- item-sep -->

- **Adopt the reporting convention.** Once one study has been through it and the
  table has survived a reading, fold it into the mandatory set: the
  `grid-experiments` skill already requires the quality-over-clicks pair and the
  interactive viewer of every simulated-user study, and the stopping block
  belongs in the same list. Proposed shape, which
  `stopping.stopping_table` emits:

  | arm | runs | fired | stop click (KM) | stop click (median of fired) | fbeta at stop | fbeta at budget | Δfbeta (paired) | clicks after stop | short of run's best | clicks past best |

  On a post-margins study a second block goes beside it, from
  `stopping.margin_table` — the same arms, read as how close each gate came
  rather than whether it fired:

  | arm | runs | held | Smart slope | Smart t | Stable avg | Stable max | Span nodes | Stable blocked by |

  The two are complementary and neither substitutes for the other: the first
  is about the runs that stopped, the second is about the steps that were held.
  An arm can look identical in the first (nothing fired) and completely
  different in the second (one arm was a hair short throughout, the other never
  close).

  The two qualifications are not optional decoration. **`fired`** comes before
  every other column because each of them is conditional on it, and the runs it
  excludes are systematically the slow ones. **Δfbeta** is paired within run and
  keeps its sign: a negative Δ means the clicks spent past the app's advice made
  the detector *worse*, which is a finding about the stopping rule and not a
  wrinkle to average away.

<!-- item-sep -->

- **Then ask whether the rules are any good.** The measurement above is
  descriptive: it says where the app's rules fire, not whether firing there was
  right. The questions it makes askable, in the order they get cheaper to
  answer:

  - **Is the stopping cost near the run's own best?** `stopping_points` now
    carries it (`{metric}_from_best`, `{metric}_clicks_past_best`, the last two
    columns of the table above). A rule that fires 40 clicks after the run's
    floor is a rule that costs users clicks; one that fires 40 clicks before it
    is a rule that costs them quality. What is owed is the reading, which
    the 2026-10-08 Binary Photo review's stopping block began (#4611). The best is the extreme
    of a noisy series, so the two columns are read together, never alone.
  - **Which indicator is binding, and is it the right one?** Needs the lights,
    which a State of the App run carries, so the block's binding note and margin
    table answer it from the same read. A local
    probe on synthetic data had Stable holding runs far more often than Smart or
    Span, which if it reproduces means the stopping rule is in practice a
    prediction-flip rule with two decorations. The margins sharpen the same
    answer: `stable_block_avg` / `stable_block_max` /
    `stable_block_falling` say *which of Stable's three gates* was doing it, and
    a gate with a median margin barely short of zero and a green share near half
    is a rule flapping rather than a detector failing.
  - **Does the rule flap because the rule is noisy, or because the detector
    is?** `n_done_episodes` above 1 is common. Answered for Smart in #3832 - the
    rule was - and the fix went into the app's indicator rather than the eval:
    a decline now has to clear the cost window's own scatter (see
    `vtscore.detectors.cost_trend`), not a fixed slope. Still open for Stable
    and Span, and still the question to ask of any `n_done_episodes` above 1.

<!-- item-sep -->

- **Do not truncate runs at `done`.** Recorded as a decision so it is not
  re-litigated: the simulated user keeps clicking to the budget. The stretch
  past the stopping point is what says whether stopping there was right, and
  arms can only be compared at a fixed `t`. If a study ever wants the honest
  end-to-end cost of a user who obeys the app, that is an opt-in arm, not a
  change to the default.

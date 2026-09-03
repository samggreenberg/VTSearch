# Reporting the app's stopping rules in eval studies (issue #3560)

**Status:** the measurement layer has shipped; what is owed is the *reporting*
— re-analysing the influential finished studies and adopting the convention.

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
  `span`) and the raw span counts (`span_level` / `span_depth`), which say
  *which* rule held a run short of stopping.
- [`scripts/experiments/calibration/stopping.py`](../../scripts/experiments/calibration/stopping.py)
  derives the stopping point and stopping cost from those columns, with the
  censoring and flapping handled; `curves.quality_vs_clicks(..., stops=…)` marks
  the stopping click on the mandatory averaged figure.
- [`docs/EVAL.md`](../EVAL.md) documents the columns and the derivation.

**The marginal compute cost of all of this is zero.** The phase machine's inputs
— the Smart error-cost window, the Stable prediction flips, the Span atlas — are
computed every step already, because the vote order depends on them; the lights
were being discarded. A local measurement puts the whole Smart-window rescoring
at 1–4% of a run's wall clock, and that 1–4% was already being paid before this
issue. No study gets slower for reporting a stopping point.

<!-- item-sep -->

- **Re-analyse the influential finished studies** — no re-runs. `phase` is in
  the cells of every study since 2026-07-31, so the enrichment is a read, not a
  grid. In rough order of influence:
  [#3156 vg-scale](../experiments/2026-08-25-vg-scale/REPORT.md) (the overview
  everything else is read against),
  [#2877 acquisition-inclusion](../experiments/2026-08-07-acquisition-inclusion/REPORT.md),
  [#3267 good-mining](../experiments/2026-08-27-good-mining-3267/REPORT.md).
  For each: `stopping.stopping_points` over the arm frames `_cells_io.load_arm`
  already returns, `stopping.summarise`, `stopping.stopping_table` into the
  report, and a re-generated `cost_vs_clicks.png` with `stops=` passed.

  Two things to check before quoting a number off an old grid, both of which the
  code will tell you rather than assume: that the cells are still on
  `/expscratch` (nothing here can rebuild them), and that `phase` is actually
  populated — a study run with `autopilot_fidelity=False` for byte-reproduction
  of a pre-fidelity result has an empty column, and `stopping_points` returns an
  empty frame rather than a table of zeros.

  What the re-analysis **cannot** answer is which of Smart and Stable was
  binding, since those cells predate the lights. `stopping.binding_note` says so
  in as many words instead of guessing. That question needs a re-run, and is the
  only thing here that does.

<!-- item-sep -->

- **Adopt the reporting convention.** Once one study has been through it and the
  table has survived a reading, fold it into the mandatory set: the
  `grid-experiments` skill already requires the quality-over-clicks pair and the
  interactive viewer of every simulated-user study, and the stopping block
  belongs in the same list. Proposed shape, which
  `stopping.stopping_table` emits:

  | arm | runs | fired | stop click (KM) | stop click (median of fired) | cost at stop | cost at budget | Δcost (paired) | clicks after stop |

  The two qualifications are not optional decoration. **`fired`** comes before
  every other column because each of them is conditional on it, and the runs it
  excludes are systematically the slow ones. **Δcost** is paired within run and
  keeps its sign: a positive Δ means the clicks spent past the app's advice made
  the detector *worse*, which is a finding about the stopping rule and not a
  wrinkle to average away.

<!-- item-sep -->

- **Then ask whether the rules are any good.** The measurement above is
  descriptive: it says where the app's rules fire, not whether firing there was
  right. The questions it makes askable, in the order they get cheaper to
  answer:

  - **Is the stopping cost near the run's own best?** Compare `cost_at_stop`
    against `min(cost)` over the trajectory. A rule that fires 40 clicks after
    the run's floor is a rule that costs users clicks; one that fires 40 clicks
    before it is a rule that costs them quality. Both are readable off cells
    that already exist.
  - **Which indicator is binding, and is it the right one?** Needs the lights,
    so it needs a re-run — but only of a grid small enough to answer it. A local
    probe on synthetic data had Stable holding runs far more often than Smart or
    Span, which if it reproduces means the stopping rule is in practice a
    prediction-flip rule with two decorations.
  - **Does the rule flap because the rule is noisy, or because the detector
    is?** `n_done_episodes` above 1 is common. Whether the fix is hysteresis in
    the app's indicator (a real product change, not an eval one) depends on
    which.

<!-- item-sep -->

- **Do not truncate runs at `done`.** Recorded as a decision so it is not
  re-litigated: the simulated user keeps clicking to the budget. The stretch
  past the stopping point is what says whether stopping there was right, and
  arms can only be compared at a fixed `t`. If a study ever wants the honest
  end-to-end cost of a user who obeys the app, that is an opt-in arm, not a
  change to the default.

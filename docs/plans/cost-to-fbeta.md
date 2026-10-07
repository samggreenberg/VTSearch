# Cost → F-beta: which Cost-era decisions to re-measure (issue #4581)

**Background.** Until late September 2026 every detector decision was read on one
metric: `cost = w_fp · FPR + w_fn · FNR`, with weights `(1, 2^k)` at an Inclusion
`k ≥ 0` and `(2^-k, 1)` below it (`inclusion_cost_weights` in
`vtscore/training/thresholds/knobs.py`). Nearly every verdict was taken at
`k = 0`, so at `FPR + FNR`, as a paired Δcost against a 0.01 tolerance; the
calibration runner still hard-codes `INCLUSION = 0`
(`scripts/experiments/calibration/experiment_config.py`). The preference then
became a precision floor (#4224, #4269, #4272) and, on 2026-10-01, an F-beta
balance (#4413; presets 1/4, 1 and 4 since #4448). The objective since #4427 is
**the F-beta, at the preset's beta, of the withheld half above the threshold the
app holds**, and since #4452 that threshold is the labels line
([`ML.md`](../ML.md#threshold-calibration); the default arm is described in
[`EVAL.md`](../EVAL.md)).

#4581 asks whether the Cost-era decisions would have gone the same way on F-beta,
and which to rerun. Two answers already exist:

- **The ladder.** #4184 scored the deck's seven calibration rungs in cost at 0.44%
  and 5% ([report](../experiments/2026-09-25-progression-4184/REPORT.md)); the
  steps read as progress. #4519 redrew it in F1 at 1%
  ([report](../experiments/2026-10-05-ladder-fbeta-4519/REPORT.md)): rungs 2 to 7
  (mixture midpoint, blend, fused cuts, 70/30, second cut) end between 0.05 and
  0.20 against cross-calibration's 0.48, and only the labels line (0.51) recovers
  it. That re-run changed the metric, the prevalence and the last rung at once,
  so it cannot say which flipped the reading.
- **Follow Suit (#4548).** Today's app against cross-calibration, each preset
  scored at its own beta: +0.17 at beta 1/4, +0.03 at 1, 0 at 4. Mean F1 at beta
  1 hides most of what the later work bought.

## What decides whether a decision is owed a re-measurement

A Cost-era decision is owed a run on F-beta when both hold:

1. **It is still on the shipped path.** Since #4452 the line is drawn from the
   labels alone, so a rule that only ever placed the *line* (the GMM midpoint,
   the blend, the fused rank transfer) is retired as a line rule, and the F1
   ladder has already read it. What survives from the Cost era is whatever the
   labels line still reads (the calibration folds), whatever the fused estimator
   still feeds (acquisition, the Smart light, the no-class-model fallback), and
   the opening.
2. **What it moved is where a cut lands or what gets asked**, not ranking
   quality alone. A verdict read on oracle cost, AUROC or AP agrees with any
   monotone metric and is settled. A verdict whose ship rule was a cost
   tolerance, or whose gain was an FNR drop against flat FPR (a lower cut), is
   the kind F-beta re-weighs, and at three betas can re-weigh three ways.

Two lessons from the switch bound every rerun: the *reading* of F-beta decides
as much as the metric (#4409 shipped on a rank-count re-draw of the line and was
reverted on the threshold the app holds, #4427), and a bench's prevalence is a
knob of its own (#4201): the 5% pool made the ladder's midpoint rungs look like
progress whatever the metric.

## The Cost-era decisions still on the shipped path

| decision, where it lives | set by, read on | what it still does | disposition |
|---|---|---|---|
| Acquisition offset −4 (`ACQUISITION_INCLUSION_OFFSET`, `knobs.py`), re-cut through the fused estimator | #3319: cost plateau from −2 to −5; −4 chosen on hard-pick precision, at 7.1% prevalence | Autopilot's `hard` picks every click. The one shipping mechanism still parameterised in cost's units. #4409 / #4428 priced rank-depth *alternatives* on the objective and kept line − 4; the value itself, κ and the cut rule were never re-swept | **#3546** (open) owns it; the offset sweep and the region cross-check are added there |
| Fused estimator: κ = 0.3, `mid_tilt`, GMM midpoint, `qmean` (`anchored.py`, `gmm.py`) | #2852, #2865, #2836 on regret at Inclusion 0 | The acquisition re-cut above, and the line when no class model can be fitted | **#3546** and the re-scoring (#4582) |
| Smart light: `FPR + FNR` at each model's Inclusion 0 cut (`vtscore/detectors/cost_trend.py`) | #3832, #4243, by reasoning; never A/B'd | Ends `hard`, gates `done` | **#4359** (open), re-keyed from the floor to the balance |
| Calibration split 0.3 / 0.5 (`PRODUCTION_SPLIT_BY_SPACE`) and fold count 2 (`DEFAULT_CALIBRATE_COUNT`) | #3287, #3314 on Δcost of the *mixture* line | The folds whose held-out scores are the labels line's class model | **#4583**: the one Cost-era knob the new line reads that nobody has measured |
| Text-sort line `gmm_midpoint` (`TEXT_SORT_CUT_RULE`; the guarded rule ships off) | #3826: the A/B failed a Δcost ship rule; the study's F1 reading favoured the guarded line | The threshold the app holds for the opening's ~23 votes, and where the Bad phase samples | **#4136** (open): the display half is now measurable on the objective without a trajectory A/B |
| Band schedule 8 / 8 / 16 / 32, 5 picks a band; the 32 / 128 cap (`spot_check.py`) | #4267, #4383 on the floor's promise; the walk arms were a wash on the objective (#4427) | The check's picks are training votes; the cap holds only the no-class-model fallback | Settled by #4427, #4452 and #4496; **#4482** (open) owns what a band pick is worth |
| Blend fallback schedules `corridor20` / `slow_cap50` (`blend_schedules.py`) | #2841, #3551 on cost under re-weighted losses | Fires on about 1% of steps, when no fold can be fitted | Not worth a run: the effect cannot reach the objective at that rate |
| Anchored EM numerics (`_ANCHORED_EM_*`), the native 1-D EM | #3825, #3839, #3585 on convergence; a cost A/B as the guard | Every fused fit | Settled: numerical, not a cut decision |
| Head: linear SVM at `C = 1` | #3197, #4114, #4219 on oracle cost, AUROC and AP | The ranking | Settled on ranking; re-scored only where AP and AUROC disagreed (#4582) |
| Opening `g3@top,b4@mid,g20+dry1/16@top` (`PRODUCTION_STARTUP`) | #4222, #4282, #4303 on AP | Autopilot's first ~23 votes | Settled: read on AP |
| Stopping-point reporting | #3560's plan proposes `cost at stop` / `cost at budget` columns | How every future study reports where a user would have stopped | **#3560** (open): the tables carry the objective |
| A/B sizing σ ≈ 0.04 per cell, `analyze_ab.py` deciding on `cost` | #3840, #4111 in cost units | Sizes and decides every rerun on this list | **#4584**: on balance-era pairs the objective's σ is about 0.08 at beta 1 and 0.16 at beta 1/4, so a grid sized for Δcost 0.01 resolves 0.02 to 0.04 of F-beta |

Not on the list, and why: the Inclusion knob studies (#2693, #3196, #2865's knob
yield, #3557) priced a control that is retired (#4269); the floor-era rules
(#4220, #4256, #4257, #4267) priced a promise the balance gave up on purpose
(#4413), and the spot check they produced has since been re-priced on the
objective (#4427, #4452, #4496); the Gumbel cut (#2836, #2846) never shipped and
today's region line is the labels line, reviewed on the objective in the
2026-10-06 State of the App; the data, infrastructure and timing studies move no
cut.

## How a rerun is run

- **The default arm is today's app**, as `docs/EVAL.md` requires: the balance,
  the labels line, the weak check. Each preset runs its own sessions
  (`CALIB_BETA` ∈ {0.25, 1, 4}) and is scored at its own beta.
- **Score the objective**: F-beta of the withheld half above the threshold the
  app holds, unchecked and after the check, with AP beside it (a cut decision
  should leave AP alone) and the returned set's size and precision, which is
  what the user sees. Never a rank-count re-draw of the line (#4409's reversal).
- **Bench and pools**: `coco_better`, 144 cells × 5 seeds, 150 clicks, the
  user's pool at 1% (as #4519) with the withheld half at the bench's own 0.44%.
  Where a Cost-era decision was taken at 5 to 7% prevalence, add the 5% pool
  once, so the metric and the prevalence are read apart.
- **Pair by cell and seed**, same commit for every arm, and size with the
  objective's σ (#4584), not the cost σ: 720 runs resolve about 0.006 at beta
  1 and about 0.012 at beta 1/4. #4583 and #3546 are A/Bs of this shape.
- **Re-score before re-running.** Every Cost-era row carries `precision` and
  `recall`, so F-beta at the three presets is derivable from the cells that
  still exist on `/expscratch`; #4582 does that first and hands only the
  decisions that flip to a grid.

## Open work

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- [ ] #4584 — The A/B tooling decides and sizes on cost: give it the objective, measure its σ (Sonnet)

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- [ ] #3560 — Stopping-point tables carry the objective beside cost

<!-- item-sep -->

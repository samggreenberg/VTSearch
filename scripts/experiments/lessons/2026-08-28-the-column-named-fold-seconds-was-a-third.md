# 2026-08-28 — the column named `fold_seconds` was a fifteenth of what a fold count costs (#3314)

**Study:** #3314 fold-count cost/benefit, stage A sizing. **Cost:** near-miss —
caught on the sizing cell, before the array. Had it not been, the study's whole
cost half would have been wrong, with clean numbers and a plausible table.

## What happened

#3314's third pre-registered ship rule is an affordability ceiling: a fold count
may raise the user's per-step retrain latency by at most 1.5×.  The obvious
column to read it off is `fold_seconds`, which #2897 and #3115 both used and
which the harness documents as "the calibration wall clock this K would have
cost".

It is not.  `fold_seconds` is the fold *fits* plus the conformal rule's
overhead.  The shipped threshold also, per fold:

- scores the whole sim set with that fold's model, so the fold's mixture can be
  anchored on its own haystack; and
- fits one anchored EM.

Both scale with K.  Both are paid inside `_safe_threshold_for_step`, which sits
between the calibration block and `t_test` - so neither lands in
`train_seconds`, `xcal_seconds`, `pool_score_seconds` **or**
`test_score_seconds`.  A cost model assembled from those four plus
`fold_seconds` therefore sees none of it.

Measured on this study's own sizing cell (`vg_scale_any x siglip`,
`whole_image`, 150 steps), per fold:

| term | seconds per fold | share |
|---|---|---|
| fold fit | 0.010 | 7% |
| fold haystack scoring | 0.010 | 7% |
| **anchored EM** | **0.128** | **86%** |

So `fold_seconds` at K=8 reports 0.079 s where the calibration actually costs
1.16 s - **15× under**.  Read through the old column every fold count sits
comfortably under any ceiling; read honestly, the per-step ratio is ≈ K/2 and
the 1.5× rule caps K at **3**.  The verdict would have been the opposite one,
with clean numbers and a plausible table.

#3115's own launcher header quotes `fold_seconds` as 2% of a binary cell and 31%
of a region cell.  Both are true of that column.  Neither is the price of K.

## Why it was invisible

The column has an honest name for what it measures and a misleading one for what
it gets used for, and the two studies that used it were asking about the *fold
fits* (does the combine rule change, does the count change the estimate), not
about what a user waits through.  #3314 is the first study whose decision rule is
a latency, and a latency has to be measured against something a user actually
pays.

The second half of the trap: a screen step's own wall clock is **not** that
denominator either.  A `CALIB_FOLD_COUNTS` step computes six fold counts × eight
arms of counterfactual rows, and a user waits through none of it - the cell runs
4.7 s/step where the app's retrain inside it is 0.33 s.  Dividing by the cell
would have reported every K as nearly free, which is the same wrong answer from
the other direction.

## Now prevented

- `vtscore/eval/voting_iterations.py` emits `fold_fit_seconds`,
  `fold_score_seconds`, `anchored_seconds` and their sum `cal_seconds` beside
  the old `fold_seconds`, which is left untouched so archived runs keep meaning.
  `final_score_seconds` joins them for the final model's own haystack pass -
  K-independent, and so a denominator term rather than a numerator one.
- `tests_lib/detectors/test_fold_count_variant_rows.py::TestFoldCountCostColumns`
  pins the arithmetic (`cal_seconds` is the sum of its parts), that the scoring
  term is billed per fold, and that a run with no anchored fit reports NaN
  rather than silently claiming the anchored rule is free.
- `analyze_folds_3314.py` prices K off `cal_seconds`, reconstructs the
  denominator from the **app's** retrain (fit + haystack scoring + calibration +
  pool scoring, deliberately excluding the eval-only test scoring), and *names
  the cost model in its own report*.  A pre-#3314 run falls back to
  `fold_seconds` and says so, loudly, in the log and in `summary.json`.
- `selftest_analyze_folds_3314.py` plants the two failures: `fold_seconds` that
  would put every K under the ceiling, and a large `test_score_seconds` that
  would do the same by inflating the denominator.

## Still only advice

**A timing column is only a cost if something a user waits for is inside it.**
Before writing a decision rule on wall clock, list the work the app does on that
action and check that every piece of it lands in some column - and that nothing
the *harness* does lands there too.  Neither direction is visible in the numbers.

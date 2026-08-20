### Iteration 1 — measured

## Right in the limit, wild at the start

- Conformal-anchor fix (#2784): clean-data regret 0.010–0.016 → **~0.000**
- But folds redraw every vote, and the cut is a **low quantile over tens of positives**
- Cold start: "admit nothing" spikes, budget overshoot below ~20 votes

<!-- The first calibration study (docs/experiments/calibration/REPORT.md,
     #2781) had a happy half and a structural half; give them in that order.
     The happy half: the runaway-threshold bug was the conformal walk pinning
     its cut to the lowest calibration positive, and #2784's fix removed
     essentially all regret in the clean binary-voting regime — so the rule,
     when fed, works.

     Then the structural half, which no bug fix touches: three deficits that
     do not decay with more of the same data. The cut is a low quantile over
     tens of positives, so its variance is dominated by the handful of rarest
     points; the folds are redrawn on every vote, so the threshold jumps
     retrain to retrain; and fold-model scores must transfer to the final
     model's scale. Below roughly 20 votes this shows up as degenerate "admit
     nothing" cuts and budget overshoot. This slide is the setup for the whole
     talk: iteration 1 is right in the limit and unusable at the start. -->

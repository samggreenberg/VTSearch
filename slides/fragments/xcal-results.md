### Iteration 1 — measured

## Right in the limit, wild at the start

- Conformal-anchor fix (#2784): clean-data regret 0.010–0.016 → **~0.000**
- But folds redraw every vote, and the cut is a **low quantile over tens of positives**
- Cold start: "admit nothing" spikes, budget overshoot below ~20 votes

<!-- docs/experiments/calibration/REPORT.md (#2781). The three structural
     deficits that never decay with label count: tiny positive sample, fold-to-
     final scale transfer, per-retrain variance. Sets up everything after. -->

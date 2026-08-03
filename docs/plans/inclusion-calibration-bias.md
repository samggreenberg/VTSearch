# Inclusion threshold calibration under non-exchangeable votes

**Background.** The Inclusion knob maps to a decision threshold via
split-conformal quantiles of held-out calibration scores
(`vtscore/training/thresholds.py::conformal_threshold`). Those semantics — "at
Inclusion `k` you miss at most `alpha(k) = 0.25 * 2^-k` of true matches" —
assume the calibration examples are **exchangeable** with the inference set, and
VTSearch's votes are chosen by the detector's own sort rather than at random.

That concern was measured and came back mostly negative: under the real
Autopilot vote order the budget converges at about the same rate as random
sampling (FNR excess 0.004 at 100 votes), because Autopilot's Hard phase biases
the cut conservatively *low* and its New phase already injects atlas diversity.
See [`docs/experiments/inclusion-knob/SELECTION-BIAS.md`](../experiments/inclusion-knob/SELECTION-BIAS.md).
**So no bias-correction work is warranted right now.** What remains is a
cold-start gap and some documentation.

The harness (`scripts/experiments/inclusion_knob/run_autopilot_sweep.py`) drives
the repo's own `vtscore.eval.al_strategies` selector, so any follow-up can reuse
it rather than re-simulating the vote order.

## Open work

<!-- item-sep -->

- **Scope the budget's meaning in `docs/ML.md`.** The Inclusion documentation
  should state that the `alpha(k)` guarantee is calibration-relative and needs a
  couple of dozen votes before it means much: at ~12 votes the measured miss
  rate exceeds the cap by ~0.3, converging to ~0.004 by 100 votes. Also worth
  saying that beyond k≈3 the cap is finer than a few dozen calibration
  positives can certify (the oracle reference violates it too), so the
  halving-per-step semantics should not be read literally at high k on a small
  vote set. No code change; smallest useful item here.

<!-- item-sep -->

- **Cold-start calibration (the only measured weak spot).** Below ~20 votes both
  random and Autopilot voting overshoot the budget badly (excess 0.208 and 0.298
  respectively) — the calibration set is too small for a 25th-percentile read,
  and Autopilot is at its least exchangeable before the New phase has
  diversified anything. Options worth measuring: widening the quantile's
  effective support at low counts (e.g. interpolating against the population
  score distribution rather than the current hard GMM blend), raising
  `calibrate_count` when votes are scarce so the pooled calibration set grows,
  or simply suppressing the confident framing of the Inclusion budget in the UI
  until enough votes exist. Measure with the existing harness before changing
  production; the 12-vote cell is the one that matters. The pre-registered
  measurement spec for these options is
  [`coldstart-threshold-experiment.md`](coldstart-threshold-experiment.md)
  (issue #2788).

<!-- item-sep -->

- **Region-bag (grouped) calibration arm — measured; bias present at high k.**
  The calibration study (#2781, `docs/experiments/calibration/REPORT.md`) ran the
  grouped path (`_compute_fold_orderings_grouped`) on Visual Genome region voting
  and compared its Inclusion-budget compliance to the ungrouped single-vector
  path. The grouped path **overshoots the FNR cap materially at k ≥ 1** (measured
  excess FNR +0.09 to +0.13 at t ≥ 100, vs ~+0.00 to +0.05 ungrouped), and the
  #2784 fix barely moves it (the high-k tail is governed by the FN-budget cap,
  not the k=0 anchor). So on a grouped/region detector the halving-per-step
  Inclusion semantics should not be read literally at high k. Remaining work: if
  region detectors become a primary workflow, either widen the effective
  calibration support at high k or scope the guarantee down for grouped
  calibration in `docs/ML.md`.

<!-- item-sep -->

- **Propensity-weighted conformal quantiles (shelved, not planned).** The
  textbook repair for non-exchangeable calibration data is weighted conformal
  prediction (Tibshirani et al.): log each vote's surfacing context (score/rank
  when shown, plus the Autopilot phase that requested it — already tracked in
  `AutopilotStateService`) and take weighted quantiles. **Do not build this on
  current evidence**: selection bias is the smaller term next to finite-sample
  noise and the irreducible-overlap floor, so weighting would optimize the wrong
  thing, and weights estimated from tens of votes are high-variance. Revisit
  only if a genuinely exploitative labeling flow ships — the study's `toplist`
  arm (greedy top-of-sort, excess growing to 0.410) shows what that failure
  looks like and would be the trigger.

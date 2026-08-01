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
  diversified anything. The 12-vote cell is the one that matters.

  Scope this to the **post-quorum** window, t ∈ [5, 15] measured in votes. That
  window is fully Autopilot-reachable: the app's first learned sort fires at the
  Hard phase, 3 good + 4 bad, so vote 7 lands in the middle of it. (The *pre*-
  quorum `too_few_default` = 0.5 path is not reachable on the Autopilot flow at
  all — the Bad phase stays on the text sort and trains nothing — so it is not
  part of this item; see `docs/EVAL.md` on harness fidelity and the `app_trained`
  column.)

  The concrete lever worth measuring is a **low-vote `calibrate_count` boost**:
  8 folds when the vote count is below 10, else the current 2. At 5–15 votes two
  folds pool only ~4–6 held-out scores, which is where both the residual
  conformal-provenance degenerate cuts and the budget overshoot live; fold fits
  at n ≤ 10 are milliseconds, so it is nearly free. Adopt iff it cuts
  conformal-provenance degenerate steps at t ∈ [5, 15] by ≥ 50% **or** shrinks
  the t = 12 budget excess, without regressing regret over t ∈ [20, 30]
  (late-window non-inferiority). Check `threshold_percentile` at t ∈ {5, 6} too:
  a non-degenerate but mid-mass cut that admits half the dataset is not a win.
  Measure with the existing harness before changing production — run it with
  `autopilot_fidelity=True` and filter on `app_trained`, or the pre-quorum steps
  will dominate the counts again.

  Independently defensible with no measurement: suppressing the confident
  framing of the Inclusion budget in the UI until enough votes exist.

<!-- item-sep -->

- **Region-bag (grouped) calibration arm.** The study's arms are all
  single-vector. The grouped path (`_compute_fold_orderings_grouped`) max-pools
  each voted image's region rows to one calibration score, and region voting
  floods a Bad vote with every region of the image. Both reshape the calibration
  score distribution, so the measured drift may differ. Reuse the harness with a
  patch dataset and `region_voting=True` semantics if region detectors become a
  primary workflow.

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

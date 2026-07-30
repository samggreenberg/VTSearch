# Inclusion threshold calibration under non-exchangeable votes

**Background.** The Inclusion knob maps to a decision threshold via
split-conformal quantiles of held-out calibration scores
(`vtscore/training/thresholds.py::conformal_threshold`). Those semantics — "at
Inclusion `k` you miss at most `alpha(k) = 0.25 * 2^-k` of true matches" —
assume the calibration examples are **exchangeable** with the inference set.
VTSearch's votes are not a random sample: Autopilot picks them (seed from text
sort, Bads from the bottom, then a Hard/New margin-plus-diversity interleave),
so the calibration positives are drawn from a policy-shaped distribution.

The measurement study lives in
[`docs/experiments/inclusion-knob/SELECTION-BIAS.md`](../experiments/inclusion-knob/SELECTION-BIAS.md);
its harness (`scripts/experiments/inclusion_knob/run_autopilot_sweep.py`) drives
the repo's own `vtscore.eval.al_strategies` selector, so any follow-up can reuse
it rather than re-simulating the vote order.

## Open work

<!-- item-sep -->

- **Propensity-weighted conformal quantiles.** The principled repair for
  non-exchangeable calibration data is weighted conformal prediction
  (Tibshirani et al.): weight each calibration score by the inverse probability
  that its item was surfaced for voting. VTSearch could log a vote's surfacing
  context (the item's score and rank at the moment it was shown, plus which
  Autopilot phase asked for it — the phase is already tracked in
  `AutopilotStateService`) and take weighted quantiles in
  `conformal_threshold`. Two open questions before this is shippable: whether
  propensity weights estimated from tens of votes are stable enough to beat the
  unweighted rule (high-variance weights can be worse than none), and whether
  the vote-origin metadata can be persisted without breaking the
  "no persisted vectors" rule (it is scalar per-vote metadata, so it should fit
  in the detector JSON's `LabeledElement`, but the schema change wants review).
  Measure with the existing harness before changing production.

<!-- item-sep -->

- **Reconsider the safe-blend ramp.** `calculate_safe_threshold` blends the
  cross-calibration threshold with a GMM threshold fit on the *full population's*
  score distribution — the one threshold input that labeling policy cannot bias
  — but ramps it to zero weight at 20 labels
  (`MIN_LABELS = 6`, `MAX_LABELS = 20`). If the study's per-vote-count numbers
  show calibration drift persisting or growing past 20 votes, a permanent floor
  on the GMM weight is a cheap partial mitigation. Note this *dilutes* the
  budget semantics rather than restoring them (the blended cut no longer
  corresponds to any quantile), so it is a robustness tweak, not a fix, and it
  needs a measured recall/precision trade before shipping.

<!-- item-sep -->

- **Document the budget's scope in the user-facing docs.** Whatever the fix,
  `docs/ML.md`'s description of Inclusion should say plainly that the `alpha(k)`
  guarantee is *calibration-relative* — it bounds misses among items distributed
  like the ones the user voted on, not among all true matches in the dataset.
  Blocked on nothing; do this once the study's final numbers are in so the
  wording can cite the measured gap.

<!-- item-sep -->

- **Region-bag (grouped) calibration arm.** The study's arms are all
  single-vector. The grouped path
  (`_compute_fold_orderings_grouped`) max-pools each voted image's region rows
  to one calibration score, and region voting floods a Bad vote with every
  region of the image. Both change the calibration score distribution's shape,
  so the measured drift may differ. Reuse the harness with a patch dataset and
  `region_voting=True` semantics if region detectors become a primary workflow.

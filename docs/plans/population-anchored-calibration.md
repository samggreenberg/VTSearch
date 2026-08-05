# Population-anchored calibration — fuse the haystack distribution into the trained threshold instead of scheduling it out

**Status:** Design (pre-registered). Estimators and harness first (#2852, #2853);
any production change is gated on the decision rules below.

## Background

The shipped threshold treats the GMM (population) cut and the cross-calibration
(labeled) cut as **rivals on a hand-tuned schedule**: `calculate_safe_threshold`
ramps GMM weight from 1 at 6 labels to 0 at 20, after which pure x-cal ships.
Three results say that framing is wrong:

- An ongoing owner-side experiment finds the naive GMM threshold **still
  competitive with x-cal at ~300 votes** — 15× past the ramp's expiry, and the
  experiment ended there, so plausibly beyond. The safe-thresholds study
  ([`docs/experiments/safe-thresholds/REPORT.md`](../experiments/safe-thresholds/REPORT.md),
  #2799) had already measured safe-ON still winning past the ramp at its
  30-vote horizon, attributing the residual to selection feedback; its scope
  note excludes the deeper regime this contradicts.
- The selection-bias study
  ([`docs/experiments/inclusion-knob/SELECTION-BIAS.md`](../experiments/inclusion-knob/SELECTION-BIAS.md))
  cleared the *labels* of blame under Autopilot: vote-collection bias is
  conservative and converges. Whatever keeps x-cal from beating an
  unsupervised bimodality assumption at 300 labels, it is not label quality.
- The residual-violation analysis there and the #2790 instability work point at
  the same three structural deficits, none of which decay with label count:

  1. **Sample size.** The conformal FN cap is a low quantile over tens of
     held-out positives; the GMM fits on up to 50k scores from the whole
     haystack. An order statistic over dozens of points carries irreducible
     noise a population-scale fit doesn't have.
  2. **Scale transfer.** The x-cal cut is measured on **fold models'** score
     scales (half the votes) but applied to the **final model's** scores — S3
     in [`threshold-stability-experiment.md`](threshold-stability-experiment.md).
     The GMM has no transfer step: it is fitted on the exact distribution
     being cut.
  3. **Per-retrain variance.** Fold splits and fold fits redraw every vote; the
     x-cal cut is a fresh noisy estimate each step, while a 250k-score GMM
     barely moves. Part of "GMM helps" is "GMM is a stabilizer."

**The reframe:** labels and haystack hold complementary information — labels
know *which quantile matters and which side is which*; the haystack knows
*where that lives on the final model's actual score scale*. They should feed
one estimator, not two rivals averaged on a label-count schedule.

## Candidate estimators

- **Rank-transfer.** Compute the conformal cut on the fold orderings as today,
  carry it over as a **quantile** of the pooled fold scores, and realize that
  quantile on the final model's full population score distribution. Labels pick
  the operating point; the haystack supplies the scale. Kills deficit 2
  exactly. Already specified as the `rank-transfer-k2` arm of
  [`threshold-stability-experiment.md`](threshold-stability-experiment.md)
  (there as an S3 diagnostic; here as a production candidate).
- **Label-anchored mixture** (#2852). Fit the 2-component mixture on the full
  haystack scores with the voted items' component responsibilities anchored to
  their labels. Population-scale sample size and native scale at every label
  count (GMM's strengths) with the "hope the modes are Good and Bad" failure
  mode removed (x-cal's strength). Replaces the ramp with an implicit,
  data-driven label/population weighting. Falls back to the unanchored GMM on
  degenerate fits, never to 0.5.
- **Never-expiring blend** (control). The shipped blend with a permanent GMM
  floor weight instead of the 20-label expiry. Not a fusion — kept as the
  cheapest possible fix and as the arm that tests whether *any* scheduling
  tweak suffices before adopting a new estimator.

## Pre-registered experiment

**Harness:** #2853 — the `scripts/experiments/calibration/` + 
`vtscore/eval/voting_iterations` machinery, checkpoints extended to
{20, 50, 100, 200, 300} votes, with every candidate rule run as a
**within-step paired variant** (the `_SAFE_GMM_VARIANTS` pattern) so arms see
identical models, votes, and steps.

**Arms:** pure x-cal (status quo past 20) · shipped safe-blend ·
never-expiring blend · rank-transfer · label-anchored mixture.

**Metrics per step, paired:** inclusion-weighted cost, FNR/FPR, regret vs the
oracle cut, step-to-step threshold delta, estimator path taken.

**Hypotheses.**

- **H1 (the deficit has a name):** at 100–300 votes, at least one fusion arm
  beats pure x-cal on regret. Which one attributes the deficit: rank-transfer
  absorbing GMM's late value ⇒ scale transfer (deficit 2) dominated;
  anchored-mixture winning where rank-transfer doesn't ⇒ quantile sample size
  (deficit 1) dominated.
- **H2 (fusion beats scheduling):** the winning fusion arm beats the
  never-expiring blend at matched steps — i.e. the ramp's problem is its
  *form*, not its expiry point.
- **H3 (stability comes along free):** the winning arm also cuts step-to-step
  threshold delta vs pure x-cal, since both fusions lean on the slow-moving
  population distribution.
- **H4 (no recall regression):** the winning arm's FNR at Inclusion 0 stays
  within the conformal budget's measured envelope at every checkpoint.

**Decision rules.**

1. A fusion arm satisfies H1 + H4 and beats the shipped blend on regret at
   both ≤20 and ≥100 votes → adopt it as the production threshold path,
   retiring the ramp (`calculate_safe_threshold` becomes the fallback for
   datasets too small to fit the population estimator).
2. Only the never-expiring blend clears the bar → keep the blend, re-key its
   schedule (floor weight, or expiry keyed to effective calibration-positive
   count rather than raw labels), and record that fusion lost.
3. Nothing beats pure x-cal at depth → the ongoing experiment's GMM result
   needs reconciling with this harness before any production change; report
   the contradiction as the headline finding.
4. Whatever the outcome, write
   `docs/experiments/population-anchored-calibration/REPORT.md` with the
   paired tables, and fold the verdict into the safe-thresholds guidance in
   `docs/ML.md`.

## Relation to other plans

- [`threshold-stability-experiment.md`](threshold-stability-experiment.md)
  owns rank-transfer as an S3 *diagnostic* on the `evaluation-framework`
  harness; this plan is where it graduates (or not) to a production candidate.
  The two measurements are complementary, not duplicated: that one measures
  temporal stability on the realistic loop, this one measures deep-regime
  accuracy on paired within-step variants.
- [`inclusion-calibration-bias.md`](inclusion-calibration-bias.md)'s cold-start
  item ("interpolating against the population score distribution") is the
  ≤20-vote special case of the same idea; a winning fusion arm here would
  subsume it.
- [`provenance-partitioned-calibration.md`](provenance-partitioned-calibration.md)
  is orthogonal: it filters *which labels* enter calibration; this plan changes
  *what the labels are fused with*. Both can ship.

## Open work

<!-- item-sep -->

- [ ] #2852 — Label-anchored mixture threshold as an eval variant (Opus 4.8)

<!-- item-sep -->

- [ ] #2853 — Deep-vote calibration harness: checkpoints to 300+, paired
  blend-schedule and threshold-rule arms (Sonnet 5)

<!-- item-sep -->

- **Run the deep-regime sweep + write the report.** Owner-gated on compute;
  verdict flows through the pre-registered decision rules above.

<!-- item-sep -->

- **Production adoption (gated on decision rule 1 or 2).** Either the winning
  fusion estimator replaces the safe-blend path in
  `vtscore/detectors/training.py` / `thresholds.py`, or the blend's schedule is
  re-keyed; cache-key and Stats-chart implications included; tests in
  `tests_lib/detectors/`.

<!-- item-sep -->

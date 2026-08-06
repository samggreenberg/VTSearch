# Population-anchored calibration — fuse the haystack distribution into the trained threshold instead of scheduling it out

**Status:** Adopted; one measurement still owed before the winner's settings can
be called final.

## Background

The threshold used to treat the GMM (population) cut and the cross-calibration
(labeled) cut as **rivals on a hand-tuned schedule** — `calculate_safe_threshold`
ramping GMM weight down as labels accumulated. Three structural deficits of the
conformal cut motivated replacing that framing with a *fusion*: the quantile's
tiny sample size, the fold→final scale transfer, and per-retrain variance (none
of which decay with label count).

The 2026-08-05 deep-regime run measured the candidates and the **fold-anchored
mixture** ("cross-LabeledGMM") at κ=1 with the rate-optimal cut won — see
[`docs/experiments/population-anchored-calibration/REPORT.md`](../experiments/population-anchored-calibration/REPORT.md)
for the numbers and
[`docs/ML.md`](../ML.md) for what production now computes. The schedule blend
survives only as the fallback for label sets too small to form calibration
folds.

## Open work

<!-- item-sep -->

- [ ] #2852 — Label-anchored mixture threshold as an eval variant (Opus 4.8)

<!-- item-sep -->

- [ ] #2853 — Deep-vote calibration harness: checkpoints to 300+, paired
  blend-schedule and threshold-rule arms (Sonnet 5)

<!-- item-sep -->

- **Boundary sweep on the adopted settings.** κ ∈ {0.1, 0.3, 1} ×
  folds ∈ {2, 4}, fold-anchored arms only, plus the `slow_cap50` blend as a
  control arm. The shipped κ=1 sits at the *edge* of the measured grid and
  performance was still improving as κ fell, 2 folds degenerate the
  qmean/qmedian combine comparison (they are byte-identical there), and the
  run's blend control predates the #2841 `slow_cap50` schedule — so "fusion
  beats the ramp" is measured but "fusion beats every schedule" is not. The
  harness arms for all of this already exist; this is a rerun off tip, not new
  code.

<!-- item-sep -->

- **Deeper-than-inclusion-0 evidence for the cut rule.** The run scored every
  arm at inclusion 0, where the rate cut and the midpoint coincide for
  equal-variance fits. The shipped rule reads inclusion as its cost weights and
  is clamped at the inter-mean interval's edges to stay monotone; the clamp's
  *cost* is unmeasured because no arm ran at a non-zero inclusion. A sweep over
  inclusion would say whether the rate rule's tilt is worth what it pays at the
  ends of the knob.

<!-- item-sep -->

- **Inclusion resolution on cleanly separated haystacks.** Because the cut is
  carried to the final model as a *quantile*, every cut inside an empty band
  between two well-separated modes realizes to the same threshold — so on a
  cleanly separated dataset the Inclusion knob moves the cut without moving the
  admitted set. This is the same "band the calibration data cannot resolve" the
  conformal rule names, and it shrinks as the modes overlap (the realistic
  case), but it is worth measuring how often real datasets sit in the flat
  regime before deciding whether the knob needs a tie-break inside the band.

<!-- item-sep -->

## Relation to other plans

- [`threshold-stability-experiment.md`](threshold-stability-experiment.md)
  owns rank-transfer as an S3 *diagnostic* on the `evaluation-framework`
  harness. The two measurements are complementary: that one measures temporal
  stability on the realistic loop, this one measured deep-regime accuracy on
  paired within-step variants.
- [`inclusion-calibration-bias.md`](inclusion-calibration-bias.md)'s cold-start
  item ("interpolating against the population score distribution") is the
  ≤20-vote special case of the same idea, now subsumed by the shipped fusion.
- [`provenance-partitioned-calibration.md`](provenance-partitioned-calibration.md)
  is orthogonal: it filters *which labels* enter calibration; this plan changed
  *what the labels are fused with*. Both can ship.

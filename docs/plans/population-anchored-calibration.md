# Population-anchored calibration — fuse the haystack distribution into the trained threshold instead of scheduling it out

**Status:** Adopted at κ=0.3 with the `mid_tilt` cut (the measured midpoint at
inclusion 0, rate-rule tilt away from it), and the tilt is now measured across
the whole knob — it held (#2865,
[`REPORT.md`](../experiments/inclusion-cut-rule/REPORT.md)). One known gap
remains: the fused path covers binary voting, where it does not beat the blend
it replaced.

## Background

The threshold used to treat the GMM (population) cut and the cross-calibration
(labeled) cut as **rivals on a hand-tuned schedule** — `calculate_safe_threshold`
ramping GMM weight down as labels accumulated. Three structural deficits of the
conformal cut motivated replacing that framing with a *fusion*: the quantile's
tiny sample size, the fold→final scale transfer, and per-retrain variance (none
of which decay with label count).

The 2026-08-05 deep-regime run measured the candidates and the **fold-anchored
mixture** ("cross-LabeledGMM") won; the 2026-08-06 anchor-mass sweep moved its
operating point to the interior optimum, **κ=0.3 with the midpoint cut** — see
[`docs/experiments/population-anchored-calibration/REPORT.md`](../experiments/population-anchored-calibration/REPORT.md)
for the numbers and
[`docs/ML.md`](../ML.md) for what production now computes. The schedule blend
survives only as the fallback for label sets too small to form calibration
folds.

## Open work

<!-- item-sep -->

<!-- item-sep -->

- [ ] #2853 — Deep-vote calibration harness: checkpoints to 300+, paired
  blend-schedule and threshold-rule arms (Sonnet 5)

<!-- item-sep -->

<!-- item-sep -->

- **Give binary voting a path back to `cap50`.** The fused threshold covers
  binary-voting detectors too, unconditionally since #2863, and there it is at
  best a dead heat with the `cap50` blend it replaced (−0.0004 n.s. at the
  shipped `κ=0.3, mid`; the `κ=1, rate` that #2861 shipped was +0.0063 *worse*).
  Either a voting-mode split (mirroring #2841) or the positive-count gate below.
  Low positive counts want spread control, not a better-located cut.

<!-- item-sep -->

<!-- item-sep -->

- **Price a sign-dependent tilt (`cross_tilt`'s asymmetry).** The #2865 sweep
  found the one rule that genuinely reads the acquisition-biased mixture weights
  is *better* than the shipped `mid_tilt` below inclusion 0 — by up to
  −0.034±0.005 at k=−1 on binary COCO, the largest effect anywhere on that
  table — and worse above it (up to +0.073±0.012). Those weights push the cut in
  the "admit more" direction, which is what the knob wants when it asks for
  fewer false alarms and the opposite of what it wants above zero. A rule that
  reads them only on one side of the knob is not obviously wrong, but it is a
  *new* rule: it needs its own pre-registration, and a hinge at k=0 has to be
  shown not to break the nesting contract
  `test_inclusion_slide_recut.py::test_slide_is_monotone_across_the_whole_knob`
  pins.

<!-- item-sep -->

- **Explain the k=0 loss on `coco_val × dinov3_patch`.** `rate` is worse than
  `mid` there by 0.015±0.002 — five times its inclusion-0 gap in the other three
  environments, and the single reason `rate` did not ship in #2865. If the
  variance-asymmetry mechanism in [`docs/ML.md`](../ML.md#threshold-calibration)
  is right, that environment should show the widest component-width asymmetry of
  the four, which is checkable from the `__cutdiag` frame the run already wrote.

<!-- item-sep -->

- **Gate fusion on positive-anchor count.** The effect scales with positives
  (24 → −0.093, 8 → −0.019, 3 → −0.002), not with dataset size. "Use fusion
  once the fold anchors hold ≥ k positives, else the blend" is directly
  supported by the six-environment data; k is not yet estimated.

<!-- item-sep -->

- **Test `κ ∝ 1/n` before pinning any constant.** The per-window argmin falls
  3 → 0.1 from 20 to 300 votes, so a fixed κ is a compromise costing ~0.008 at
  each end. A fixed *total* anchor mass (κ = M/n, or M/n_good) is a one-line
  change to the caller.

<!-- item-sep -->

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

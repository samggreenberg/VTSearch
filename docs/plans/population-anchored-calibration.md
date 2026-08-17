# Population-anchored calibration — fuse the haystack distribution into the trained threshold instead of scheduling it out

**Status:** Adopted at κ=0.3 with the `mid_tilt` cut (the measured midpoint at
inclusion 0, rate-rule tilt away from it). Two known gaps remain: the fused
path covers binary voting, where it does not beat the blend it replaced, and
the inclusion tilt is unmeasured away from inclusion 0 — the sweep that prices
it is built and awaiting a GRID run (#2865).

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

- [ ] #2865 — Inclusion-aware cut rule: `mid_tilt` (the issue's candidate 1) is
  shipped and the sweep apparatus is built (`launch_incl_2865.sh`,
  `analyze_cutincl.py`, the `cross_tilt` / `q_tilt` eval-only arms). What
  remains is **running it on the GRID and writing the report** (Opus 4.8).
  It absorbs two items this plan used to carry separately: *deeper-than-
  inclusion-0 evidence for the cut rule* (that is the sweep itself — both runs
  scored every arm at inclusion 0, the one setting where the rule choice cannot
  matter), and *inclusion resolution on cleanly separated haystacks*, which the
  analyzer answers as a by-product via the per-environment knob-yield ceiling:
  because the cut is carried to the final model as a quantile, a cut inside an
  empty band between two well-separated modes realizes to the same threshold
  however far it moves, and the ceiling table measures how often real data sits
  in that flat regime.
  Two things the run should settle beyond picking a rule: `q_tilt`'s step size
  is a free parameter that has to be *fitted*, not assumed, and the incumbent's
  own knob yield is unknown — `mid_tilt` may already be delivering most of the
  slider, in which case the honest outcome is "keep it".

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

# Population-anchored calibration — the fused cut as the fallback line

**Status:** Shipped at κ=0.3 with the `mid_tilt` cut (#2865,
[`REPORT.md`](../experiments/2026-08-21-inclusion-cut-rule/REPORT.md)). **Rescoped
2026-10-10:** since #4452 a trained head's line is the labels line
(`fit_labels_line`; [`ML.md`](../ML.md#the-labels-line)), which never reads κ or
the fused mixture. The fused cut now decides only two fallbacks: the line when no
class model can be fitted (too few votes, one class, held-out Goods below the
Bads), and the acquisition cut for a detector with no labels line. The Inclusion
knob it was tuned on is internal at 0 (#4269). So the open work is priced on
**F-beta at the balance** (the shipped objective), **inside the regime where the
fallback fires**, and only after the gate item below shows that regime matters.

## Background

The threshold used to treat the GMM (population) cut and the cross-calibration
(labeled) cut as rivals on a hand-tuned schedule (`calculate_safe_threshold`).
The **fold-anchored mixture** replaced that schedule; the anchor-mass sweep put
it at κ=0.3 with the midpoint cut. The numbers are in
[`docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md`](../experiments/2026-08-05-population-anchored-calibration/REPORT.md).
The schedule blend survives as the fallback below the fused cut. Every
measurement on record was cost at an Inclusion setting, over the whole vote
range: none of it isolates the few-vote, one-class states where the fused cut
still decides anything.

## Open work

<!-- item-sep -->

- **Measure how much the fallback decides (gate for every item below).** On the
  shipped path (the eval default arm under a balance; the State of the App
  datasets), count the steps where the reported line comes from the fused cut
  rather than the labels line, and the Autopilot picks that read the offset
  acquisition fallback rather than the target precision. Also price what the
  fallback costs there: F-beta at the row's beta against `oracle_fbeta`. If the
  fallback covers only the first few clicks and its F-beta gap is within noise,
  retire this plan: its remaining items cannot move the shipped line.

<!-- item-sep -->

- **Give binary voting a path back to `cap50`, inside the fallback regime.** On
  the whole vote range the fused cut was at best a dead heat with the `cap50`
  blend on binary voting (−0.0004 n.s. at `κ=0.3, mid`). Re-measure only on the
  steps the gate item finds the fallback deciding, on F-beta. A voting-mode split
  (mirroring #2841) or the positive-count gate below are the candidate fixes.

<!-- item-sep -->

- **Gate fusion on positive-anchor count.** The fusion gain scaled with
  positives (24 → −0.093, 8 → −0.019, 3 → −0.002). The fallback regime has few
  positives by construction, so this may simply say "use the blend there".
  Estimate k (fusion once the fold anchors hold ≥ k positives) on F-beta within
  that regime.

<!-- item-sep -->

- **Test `κ ∝ 1/n` at small n only.** The per-window argmin fell 3 → 0.1 from 20
  to 300 votes. Only the low-vote end now matters, so test a fixed total anchor
  mass (κ = M/n, or M/n_good) over the fallback's vote range, not the full curve.

<!-- item-sep -->

## Relation to other plans

- [`provenance-partitioned-calibration.md`](provenance-partitioned-calibration.md)
  is orthogonal: it filters *which labels* enter calibration; this plan is about
  *what the labels are fused with*.
- [`region-vs-binary-kappa-mechanism.md`](region-vs-binary-kappa-mechanism.md)
  explains *why* κ\* differs by voting mode; it waits on this plan's gate item.

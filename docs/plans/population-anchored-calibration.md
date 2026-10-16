# Population-anchored calibration — the fused cut as the fallback line

**Status:** Shipped at κ=0.3 with the `mid_tilt` cut (#2865,
[`REPORT.md`](../experiments/2026-08-21-inclusion-cut-rule/REPORT.md)).
**Rescoped (owner, 2026-10-10):** since #4452 a trained head's line under a
balance is the labels line (`fit_labels_line`;
[`ML.md`](../ML.md#threshold-calibration)), which never reads κ or the fused
mixture, and the Inclusion knob the fused cut was tuned on is internal at 0
(#4269). So the open work is priced on **F-beta at the balance** (the shipped
objective), **inside the regime where the fallback fires**, and only after the
gate item below shows that regime matters.

**Where the fused cut still decides, read from the code.** Less than the rescope
assumed. When the calibration folds hold out no Good, or rank their Goods no
higher than their Bads (#4899), the labels line stands in with the head's own
in-sample scores of the labels. A trained head therefore draws no labels line
only when even those in-sample scores put its Goods no higher than its Bads
(training needs a Good and a Bad, so a one-class labelset never gets this far).
Only then does the line come from the fused cut (`_fused_threshold` in
`vtscore/detectors/training.py`), and only then does Autopilot's acquisition
leave the labels line's even-odds target (`ACQUISITION_TARGET_PRECISION`, #3546)
for the line − 4 re-cut. The app always sets a balance; a library caller that
sets none (`beta=None`) still gets the fused cut everywhere.

## Background

The threshold used to treat the GMM (population) cut and the cross-calibration
(labeled) cut as rivals on a hand-tuned schedule (`calculate_safe_threshold`).
Three structural deficits of the conformal cut motivated a *fusion* instead: the
quantile's tiny sample size, the fold→final scale transfer, and per-retrain
variance (none of which decay with label count). The **fold-anchored mixture**
replaced that schedule; the anchor-mass sweep put
it at κ=0.3 with the midpoint cut. The numbers are in
[`docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md`](../experiments/2026-08-05-population-anchored-calibration/REPORT.md).
The schedule blend survives as the fallback below the fused cut. Every
measurement on record was cost at an Inclusion setting, over the whole vote
range: none of it isolates the states where the fused cut still decides
anything.

## Open work

<!-- item-sep -->

- **Measure how much the fallback decides (gate for every item below).** On the
  shipped path (the eval default arm under a balance; the State of the App
  datasets), count the clicks with a trained head where no labels line was
  drawn. It may need no new run: a calibration base row under a balance records
  `line_head_count = -1` when no labels line was drawn (#4878), so the runs since
  then may already hold the count. The same clicks are the Autopilot picks that
  read the line − 4 re-cut. Where any exist, price what the fallback costs
  there: F-beta at the row's beta against the best cut
  (`oracle_fbeta_cut`). If the count is near zero, or its F-beta gap is within
  noise, retire this plan: its remaining items cannot move the shipped line.

<!-- item-sep -->

- **Give binary voting a path back to `cap50`, inside the fallback regime.** On
  the whole vote range the fused cut was at best a dead heat with the `cap50`
  blend on binary voting (−0.0004 n.s. at `κ=0.3, mid`). Re-measure only on the
  clicks the gate item finds the fallback deciding, on F-beta. A voting-mode
  split (mirroring #2841) or the positive-count gate below are the candidate
  fixes.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Gate fusion on positive-anchor count.** The fusion gain scaled with
  positives (24 → −0.093, 8 → −0.019, 3 → −0.002). Estimate k (fusion once the
  fold anchors hold ≥ k positives, else the blend) on F-beta within the fallback
  regime, if the gate item leaves one.

<!-- item-sep -->

- **Test `κ ∝ 1/n` inside the fallback regime only.** The per-window argmin
  fell 3 → 0.1 from 20 to 300 votes. Test a fixed total anchor mass (κ = M/n,
  or M/n_good) over the votes the fallback actually fires at, not the full
  curve.

<!-- item-sep -->

<!-- item-sep -->

## Relation to other plans

- [`provenance-partitioned-calibration.md`](provenance-partitioned-calibration.md)
  is orthogonal: it filters *which labels* enter calibration; this plan is about
  *what the labels are fused with*.
- [`region-vs-binary-kappa-mechanism.md`](region-vs-binary-kappa-mechanism.md)
  explains *why* κ\* differs by voting mode; its real-data items wait on this
  plan's gate item.

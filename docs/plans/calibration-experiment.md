# Calibration experiment — trained vs oracle thresholds (issue #2781)

**Background.** The Grid study ran (Autopilot simulation, Visual Genome
region voting + Caltech binary voting) on both the pre- and post-#2784
calibration code. Results, verdict, and the pre-registered decision-rule
outcomes are in
[`docs/experiments/2026-07-31-calibration/REPORT.md`](../experiments/2026-07-31-calibration/REPORT.md);
the harness lives
in `vtscore/eval/calibration_metrics.py` + the `emit_calibration_metrics` path of
`vtscore/eval/voting_iterations.py` (+ the provenance/node-score surface in
`vtscore/training/thresholds/`), and the runner in
`scripts/experiments/calibration/`.

Headline: the #2781 runaway-threshold bug was **not** the `NO_GOOD_THRESHOLD`
sentinel (never fired) but the conformal walk's `k=0` anchor — fixed in **PR
#2784**, which this study confirms clears the clean-separation regime (Caltech
regret → ≈0) while being ~a no-op under class overlap (VG region voting).
Calibration is provably the raw-patch tree's bottleneck (regret significantly
larger than `max_patch`, ties/beats at the oracle), but no pre-registered remedy
(`topk`, sign-corrected `pnorm`) recovers it, so `max_patch` stays the region-vote
strategy.

**Rescoped 2026-10-10.** That study measured *cost regret* (the conformal cut
against the cost oracle) at an Inclusion setting. Both are retired from the
shipped path. Since #4452 the line is the labels line at the detector's balance,
Inclusion is internal at 0 (#4269), and the harness now carries an F-beta oracle
(`oracle_fbeta`, #4654). So the question this plan still owes is the same one,
asked of today's line: **how far is the labels line from the best F-beta cut**,
by voting mode and patch style, on trajectories a user would actually walk.

## Open follow-ups

<!-- item-sep -->

- **Re-run the study on today's line.** Same cells and pickles; the default arm
  (labels line at the balance, `autopilot_fidelity=True`) at the pre-registered
  4 seeds, reporting F-beta over `oracle_fbeta` per step, instead of cost regret.
  This one run folds in the old "4 seeds" and "Autopilot fidelity" follow-ups.
  The tree geometry and re-pools are opt-in since #3400, so add
  `CALIB_PATCH_STYLES=max_patch,max_patch_pca_hac
  CALIB_REPOOL_VARIANTS=topk,pnorm` and declare `patch_style` to preflight.

<!-- item-sep -->

- **Max-pool-aware calibration for the raw-patch tree, if the re-run still shows
  the gap.** The 2026-07-31 study found the tree ranks best but is
  calibration-bottlenecked, and neither `topk` nor the sign-corrected `pnorm`
  recovered it. The labels line fits its class model on the same max-pooled
  scores, so the gap may persist; measure it in the re-run first. Only if it does
  is a rule that models the max-over-N tail (or a per-node-count line) worth
  building, measured against `max_patch` on F-beta. The `max_patch_hac` /
  `max_patch_pca_hac` styles survive in `vtscore/eval/patch_styles.py` for this;
  production is tree-free since #2886.

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

- **Patch styles under binary voting.** Caltech × {`max_patch`,
  `max_patch_pca_hac`} measures the line's F-beta gap when every Good vote is
  image-level (the "user ignores region voting" mode). Run it as an extra cell of
  the re-run above.

<!-- item-sep -->

- **Plain `max_patch_hac` arm.** Isolates PCA merge-ordering from node-count
  effects, if a verdict from the re-run hinges on something PCA-specific.

<!-- item-sep -->

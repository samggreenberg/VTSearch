# Calibration experiment — trained vs oracle thresholds (issue #2781)

**Status: shipped.** The Grid study ran (Autopilot simulation, Visual Genome
region voting + Caltech binary voting) on both the pre- and post-#2784
calibration code. Results, verdict, and the pre-registered decision-rule
outcomes are in **`docs/experiments/calibration/REPORT.md`**; the harness lives
in `vtscore/eval/calibration_metrics.py` + the `emit_calibration_metrics` path of
`vtscore/eval/voting_iterations.py` (+ the provenance/node-score surface in
`vtscore/training/thresholds.py`), and the runner in
`scripts/experiments/calibration/`.

Headline: the #2781 runaway-threshold bug was **not** the `NO_GOOD_THRESHOLD`
sentinel (never fired) but the conformal walk's `k=0` anchor — fixed in **PR
#2784**, which this study confirms clears the clean-separation regime (Caltech
regret → ≈0) while being ~a no-op under class overlap (VG region voting).
Calibration is provably the raw-patch tree's bottleneck (regret significantly
larger than `max_patch`, ties/beats at the oracle), but no pre-registered remedy
(`topk`, sign-corrected `pnorm`) recovers it, so `max_patch` stays the region-vote
strategy.

## Open follow-ups

<!-- item-sep -->

- **Max-pool-aware calibration for the raw-patch tree.** The study proves the
  tree ranks best (lowest oracle cost) but is calibration-bottlenecked, and that
  neither `topk` nor the sign-corrected `pnorm` (closes only ~21% of the gap)
  recovers it. A calibration rule that models the max-over-N tail directly — or
  a per-node-count threshold — is the remaining lever; measure it against
  `max_patch`'s trained cost before any production change.

<!-- item-sep -->

- **Re-run at the pre-registered 4 seeds with corrected `pnorm`.** This run used
  2 seeds (4-GPU QOS cap) and only measured the sign-corrected `pnorm` on the
  post-#2784 pass. A 4-seed replication would tighten the tree regret Wilcoxon
  (currently p = 0.013 at 2 seeds) and give `pnorm` a clean, non-inverted read
  from t = 0.

<!-- item-sep -->

- **Cold-start degenerate defaults.** Post-#2784 the residual degenerate
  thresholds are dominated by the `too_few_default` (0.5) path at < ~4 votes,
  which the fix does not touch — filed separately as a concrete bug.

<!-- item-sep -->

- **Patch styles under binary voting** — Caltech × {`max_patch`,
  `max_patch_pca_hac`} would measure grouped-calibration regret when every Good
  vote is image-level (the "user ignores region voting" mode).

<!-- item-sep -->

- **Plain `max_patch_hac` arm** — isolates PCA merge-ordering from node-count
  effects if a follow-up verdict hinges on something PCA-specific.

<!-- item-sep -->

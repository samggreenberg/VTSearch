# Calibration experiment — trained vs oracle thresholds (issue #2781)

**Status:** Design (pre-registered). This is the Grid study for #2781
("Calibrating calibration"), image datasets only. The code (harness
extensions + `scripts/experiments/calibration/` runner) is not yet written;
this document is the spec it will be written to. Owner runs it on the HLTCOE
Grid when access is available.

## Question

Every retrain picks a decision threshold via cross-calibration + the conformal
inclusion rule (`vtscore/training/thresholds.py`). How much error cost does
that choice lose relative to the *best possible* threshold for the same
ranking — and does that loss explain why the multi-scale raw-patch tree
(`max_patch_pca_hac`) failed to beat plain `max_patch` in the Max-Patch study?

The regret metric, per #2781: at each step compute the test-set
`cost = FPR + FNR` (inclusion 0 weights) at the **trained** threshold and at
the **oracle** threshold (the cut minimizing the same cost on the held-out
test scores). The difference is the calibration regret — the part of our error
that better thresholding could remove. The Max-Patch report already showed the
shape this study formalizes: `max_patch_pca_hac`'s tree *ranks* best of any arm
(AP 0.492) but over-fires at the trained threshold (highest FPR), plausibly
because a max over ~392 tree nodes has a heavier tail than a max over ~197 raw
patches. If its oracle-threshold cost beats (or ties) MaxPatch while its regret
is significantly larger, calibration is the bottleneck and a max-pool-aware fix
can save the tree; if it loses at the oracle too, no threshold rule can save it
and the line of inquiry closes.

Secondary question, same instrumentation: the runaway-threshold bug. The
issue reports a learned threshold landing far above *all* positives and
negatives (FNR 1.0 / FPR 0, cost 1.0), self-healing one vote later. The prime
suspect is the `NO_GOOD_THRESHOLD = 2.0` sentinel (above every sigmoid score
by construction) leaking out of an early-return in
`compute_fold_orderings` / `threshold_from_fold_orderings`. The harness runs
thousands of independent train-and-calibrate steps, so it can measure the
incidence, the vote counts at which it fires, which code path produced it, and
whether the next step recovers — matching or refuting the "jumps back one
click later" observation.

## Arms

Every arm is an `(embedder, style)` pair on the existing style machinery
(`vtscore/eval/patch_styles.py`), plus pooling variants of the tree style
(remedial arms, below).

**Region-voting dataset — `visual_genome_m`** (ground-truth boxes; the
simulated user drags real region votes):

| Arm | Embedder | Style | Calibration path |
|---|---|---|---|
| SigLIP | `siglip` | `whole_image` | row-wise (ungrouped) |
| SigLIP-L | `siglip_l` | `whole_image` | row-wise (ungrouped) |
| DINOv3-MaxPatch | `dinov3_patch` | `max_patch` | grouped bag max-pool, ~197 rows/image |
| MaxPatchPcaHac | `dinov3_patch` | `max_patch_pca_hac` | grouped bag max-pool, ~392 nodes/image |

Deliberately **no** plain `max_patch_hac` arm: the four arms above are the
pre-registered set, and the PCA-vs-node-count confound is accepted (the
Max-Patch run showed the PCA and non-PCA trees behave alike; if this study's
verdict ends up hinging on the distinction, a follow-up adds the arm).

**Binary-voting dataset — `caltech101_m`** (boxless; every Good vote is
image-level): **whole-image arms only** — `siglip` and `siglip_l`. This side
measures the plain row-wise conformal path under ordinary binary voting, the
regime most users are in. Patch styles under image-level voting are out of
scope for v1 (noted under Open follow-ups).

**Remedial arms (v1, pre-registered)** — pooling variants of
`max_patch_pca_hac` that attack the hypothesized failure (max-over-N tail
grows with N) without touching training. Each reuses the base arm's per-step
trained fold models and per-node scores and only re-pools, so the marginal
cost is scoring-only:

- **`..._topk`** — image score = mean of the top-k node sigmoids (k = 4).
  Softer than a hard max; the same pooling is applied to calibration bags and
  test images so the geometry stays consistent.
- **`..._pnorm`** — extreme-value normalization: image score = probability
  that a null image with the same node count N would reach the observed max,
  i.e. `1 − F̂_neg(max)^N`, with `F̂_neg` the empirical CDF of node scores
  over the calibration *negative* bags. This is the "calibration can save it"
  arm in its purest form — it explicitly corrects for N, so an image with
  twice the nodes no longer gets an inflated max for free.

Both are style subclasses gated behind the runner's arm list; if either needs
a tuning constant beyond the pre-registered ones (k = 4), that is a design
change, not a knob to sweep post hoc.

## Metrics (per step t, new columns over the existing harness CSV)

The harness already records `cost`, `fpr`, `fnr`, `auroc`,
`average_precision`, and timings per step. Added:

- **`threshold`** — the trained threshold the step actually used (currently
  computed but never written).
- **`threshold_provenance`** — which path produced it: `conformal` (the
  normal quantile rule), `no_good_sentinel` (`NO_GOOD_THRESHOLD`),
  `too_few_default` (the 0.5 early-returns), `gmm_blend` (safe-threshold
  ramp; off in this study's production-faithful config, recorded anyway).
  Requires a small provenance out-param in `thresholds.py` (or a
  harness-side wrapper that re-derives it).
- **`oracle_threshold`, `oracle_cost`, `oracle_fpr`, `oracle_fnr`** — the cut
  minimizing `fpr_weight·FPR + fnr_weight·FNR` over the held-out test scores
  (O(n log n) sweep over observed cuts), and its costs.
- **`regret`** = `cost − oracle_cost`.
- **`cal_oracle_threshold`, `cal_oracle_cost`** — the cut minimizing the same
  weighted cost over the **pooled calibration fold orderings** (the exact
  information the conformal rule saw), evaluated on the test set. This splits
  regret into two addends: *rule inefficiency* (`cost − cal_oracle_cost`: the
  conformal quantile rule vs the best use of its own data) and
  *calibration→test shift* (`cal_oracle_cost − oracle_cost`: finite-sample +
  geometry mismatch between calibration bags and test images).
- **`threshold_percentile`** — where the trained threshold sits in the test
  score distribution; **`degenerate`** — flag for `threshold` above the max
  (or below the min) test score. FNR 1.0 / FPR 0 with `degenerate = 1` is the
  #2781 bug signature.
- **`n_pool_rows`** — rows/nodes max-pooled per image for the arm (197 vs 392
  vs 1), so the FPR-tail-vs-N relationship is plottable directly.
- **Inclusion sweep** (near-free; long-format side CSV): after computing the
  step's fold orderings once, re-threshold at
  k ∈ {−4, −2, −1, 0, 1, 2, 4} via `threshold_from_fold_orderings` and record
  each k's threshold + test FPR/FNR. Measured FNR vs the conformal cap
  `alpha(k) = 0.25·2^-k` checks the Inclusion budget's semantics under the
  **grouped** calibration path — closing the "Region-bag (grouped)
  calibration arm" open item in
  [`inclusion-calibration-bias.md`](inclusion-calibration-bias.md).

Fixed config is production-faithful and matches Max-Patch: inclusion 0
(cost = FPR + FNR exactly, per `_inclusion_weights`), `calibrate_count = 2`,
`calibration_fraction = 0.5`, `safe_thresholds = False`, MLP trainer, 150
votes per trajectory.

## Hypotheses (pre-registered, honest priors)

- **Regret is real and biggest for the tree.** Ordering of regret at t = 150:
  `max_patch_pca_hac` > `max_patch` > whole-image arms. The grouped max-pool
  calibrates in inference geometry (post-#2732), so the *mean* threshold
  should be roughly right; the loss should show up as variance + a heavy
  upper tail of FPR, growing with `n_pool_rows`.
- **The tree survives at the oracle.** `max_patch_pca_hac`'s oracle cost ties
  or beats `max_patch`'s (its AP was the best of any arm); the trained-cost
  gap between them is mostly regret. This is the load-bearing hypothesis —
  if it fails, calibration cannot save the tree.
- **`pnorm` closes most of the gap; `topk` some of it.** Explicitly
  normalizing for N should remove the node-count tax; a fixed top-k softens
  the tail but doesn't adapt to N.
- **Decomposition:** early steps (few votes) are dominated by
  calibration→test shift (tiny calibration sets, quantiles on a handful of
  scores); by ~50+ votes the residual regret is mostly rule inefficiency,
  and small.
- **Whole-image arms are well-calibrated.** Regret < 0.05 at t ≥ 50 on both
  datasets; SigLIP-L matches SigLIP's regret (calibration quality should not
  depend on embedding dimensionality). Caltech (binary voting, clean classes)
  shows the least regret of all cells.
- **The bug is the sentinel.** Degenerate thresholds occur at low vote counts,
  carry `no_good_sentinel` provenance, and vanish on the next step. If instead
  degenerate steps show `conformal` provenance, the bug is in the quantile
  rule itself and the study's per-step orderings will localize it.

## Decision rules (pre-registered)

1. If `max_patch_pca_hac` ties/beats `max_patch` at the **oracle** threshold
   *and* its regret is significantly larger (paired Wilcoxon over
   (category, seed) cells at t = 150, Holm-corrected): calibration is the
   bottleneck. Adopt the best remedial arm **iff** it closes ≥ half the
   trained-cost gap *and* its trained cost beats `max_patch`'s trained cost;
   then file the production-adoption issue.
2. If `max_patch_pca_hac` loses at the oracle too: the tree's extra nodes
   genuinely rank worse at the operating region; close the multi-scale-tree
   line and record the verdict in the Max-Patch report's follow-ups.
3. Any degenerate-threshold steps observed → file a bug issue with the
   provenance histogram and the smallest reproducing (n_good, n_bad, grouped?)
   configuration; a fix is its own PR, not part of this study.
4. Inclusion-budget violations under the grouped path materially worse than
   the single-vector study's (excess FNR > 0.05 at t ≥ 100 for k ≥ 1) →
   reopen the cold-start item in `inclusion-calibration-bias.md` with the
   grouped numbers.

## Sizing

Reuses the Max-Patch category machinery: scale-band selection on VG
(4 bands × 6 = up to 24 categories), prevalence spread on Caltech (6
categories), 4 seeds, 150 votes. Cells (one SLURM task per
(dataset, embedder, category, seed); all of an embedder's styles inside):

- VG: 3 embedders × ≤24 categories × 4 seeds = **≤288 cells**
- Caltech: 2 embedders × 6 categories × 4 seeds = **48 cells**

Same memory envelope as Max-Patch (~64G for VG × patch styles); remedial arms
add scoring-only overhead inside the dinov3 cells. Env knobs `CALIB_*`
mirroring the `MAXPATCH_*` set.

## Deliverables

1. Harness extensions in `vtscore/eval/voting_iterations.py` (+ a provenance
   surface in `vtscore/training/thresholds.py`, + pooling-variant styles in
   `vtscore/eval/patch_styles.py`), all default-off so `style=None` history
   and existing studies stay byte-for-byte.
2. Runner `scripts/experiments/calibration/` in the standard stage layout
   (prepare → SLURM array cells → summarize), sharing the Max-Patch prepare
   pickles where the (dataset, embedder) pair coincides.
3. `docs/experiments/calibration/REPORT.md` in the standard report style
   (BLUF, verdict, figures from the per-cell CSVs). Planned figures: regret
   curves vs t per arm; trained-vs-oracle cost bars with the two-addend
   decomposition; FPR tail vs `n_pool_rows`; degenerate-threshold incidence
   vs vote count with provenance; inclusion-budget compliance
   (measured FNR vs `alpha(k)`) for grouped vs ungrouped arms.

## Open work

<!-- item-sep -->

- **Build the harness extensions** — threshold/provenance/oracle/regret
  columns, calibration-set oracle, inclusion-sweep side CSV, degenerate flag
  (per the Metrics section). Includes the `thresholds.py` provenance surface.

<!-- item-sep -->

- **Build the remedial pooling styles** — `max_patch_pca_hac` variants
  `topk` (k = 4) and `pnorm` (extreme-value normalization), reusing the base
  arm's fold models and node scores per step.

<!-- item-sep -->

- **Build the runner** — `scripts/experiments/calibration/` (prepare /
  `run_cells.py` / `summarize.py` / `queue_all.sh`), `CALIB_*` env knobs,
  CPU-smoke-tested before Grid submission.

<!-- item-sep -->

- **Run on the Grid + write the report** — owner-gated on Grid access;
  verdict flows through the pre-registered decision rules above.

<!-- item-sep -->

## Open follow-ups (out of v1 scope)

<!-- item-sep -->

- **Patch styles under binary voting** — Caltech × {`max_patch`,
  `max_patch_pca_hac`} would measure grouped-calibration regret when every
  Good vote is image-level (the "user ignores region voting" mode). Add only
  if the VG verdict makes the interaction matter.

<!-- item-sep -->

- **Plain `max_patch_hac` arm** — isolates PCA merge-ordering from node-count
  effects if the v1 verdict hinges on something PCA-specific.

<!-- item-sep -->

## Known limitations (accepted for v1)

- The oracle uses test labels by definition; it is a bound, not an achievable
  rule. `cal_oracle` is the achievable-information bound.
- The harness measures the same `thresholds.py` code paths the app calls, but
  not app-tier caching (`cross_calibration_threshold_cached`) — if the #2781
  bug lives in cache-key staleness rather than the sentinel, the harness will
  show clean provenance and the bug hunt moves to the app tier.
- Max-Patch harness carryovers: the exemplar image may land in the test split
  (tiny, equal across arms), and pool acquisition scores candidates by
  whole-image vector in every style.
- `pnorm` changes the score's scale (it becomes a p-value-like quantity);
  AP/AUROC remain comparable (monotone per image count N only when N is
  constant per arm, which it is within an arm), but cross-arm score
  distributions are not directly comparable — comparisons stay at the
  cost/regret level.

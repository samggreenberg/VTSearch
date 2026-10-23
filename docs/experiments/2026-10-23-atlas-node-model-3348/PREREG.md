# Pre-registration: replacing the atlas's per-node vMF model (#3348)

Written 2026-10-23, before any task of the run array started. Instrument:
`scripts/experiments/structure/node_models_3348.py`, driven by
`run_node_models_3348.py`; planted-answer check `selftest_node_models_3348.py`
(passed: every split-conformal variant 0.040-0.051 below 0.05 on held-out
synthetic data, planted shift fires, shipped arithmetic reproduced to 3e-8).

## Question

`domain_shift_report` is refused on patch embedders because, on `dinov3_patch`,
it fires on 80 % of the atlas's own held-out data and separates in-domain from a
different corpus by only 0.13 (#3329 part 2, B4-B6). Both threshold repairs
were priced and failed. #3348 asks whether a **different per-node model** fixes
it: a mixture per node, a different sufficient statistic, or a non-parametric
per-node score, measured the #3329 way (held-out in-domain PIT, cross-corpus
separation) with the kNN conformal rule as the positive control.

## Grid

5 embedders (`siglip`, `dinov3_patch`, `clip`, `clip_l`, `siglip2_l`) x
3 seeds (the #3329 build/holdout split, `default_rng(seed)`, 20 % held out).
Each task builds an atlas per dataset and scores the held-out split (75 PIT
cells) and every ordered dataset pair (125 pairs per seed, 375 in all).

**Datasets differ from #3329, and this is forced:** `vg_scale_any` and
`vg_box_large` have no `dinov3_patch` cell any more (retired with the VG
machinery, #4038). They are replaced by `coco_better` and `o365_better`;
`coco_val`, `caltech101_m` and `visual_genome_m` stay. Comparability with
#3329 therefore comes from re-running the shipped model on this grid
(`grid_mean_shipped`), not from its old numbers. Sources for the
same-/different-source split: COCO = {`coco_val`, `coco_better`}, VG, Caltech,
Objects365. That is 18 different-source and 2 same-source ordered pairs per
embedder per seed.

## Candidates

Same tree for all (shipped k = 3, `auto_max_depth`), same node eligibility
(n >= 20, r̄ >= 0.1). They vary along three axes:

- **statistic**: `vmf` (shipped `mu . x`), `knn` (mean cosine to the 5 nearest
  node points), `mix` (max over 1-3 vMF directions per node), `ppca`
  (Gaussian on the node's top 10 principal directions + isotropic residual)
- **calibration**: `loo` (shipped: tree on all build points, rank against the
  node's own leave-out scores) or `split` (tree on half, rank against the other
  half routed to the node, which makes the deepest node exactly conformal)
- **combiner**: `deepest`, `mean`, `median`

References: `grid_mean_shipped` (production code, the baseline), `grid_median`,
`grid_deepest`; `knn_global` (tree-free split-conformal kNN, the positive
control); `class_conformal` (#3329's per-class control, KS 0.047 there).

## Measures

Per embedder, over its 15 cells / 15 diagonal pairs / 54 different-source pairs:

- **KS**: median over cells of the KS distance of held-out p-values from U(0,1)
- **rate@0.05**: mean held-out fraction below 0.05 (nominal 0.05)
- **false alarm**: share of diagonal pairs where the shipped rule
  (z > 3 and frac >= 2 alpha) fires
- **detection**: share of different-source pairs where it fires
- **separation** = detection - false alarm (the #3329 B4 quantity)

## Predictions

- **P1 (replication).** `grid_mean_shipped` on `dinov3_patch`: rate@0.05 > 0.10
  and separation < 0.30. Every other embedder: false alarm <= 0.2.
- **P2 (positive control).** `knn_global` KS <= 0.05 and rate@0.05 within
  0.035-0.065 on every embedder. If it fails, the harness is wrong and nothing
  else is read.
- **P3 (where the defect lives).** On `dinov3_patch`, `vmf_split_deepest` is
  calibrated (rate@0.05 within 0.035-0.065). If so, the patch failure is the
  **calibration scheme**, not the vMF statistic: the k-means cells are drawn
  around the very points `loo` ranks against. If `vmf_loo_deepest` (same
  statistic, exact ranks, no grid) is still off, the grid is not the cause
  either.
- **P4 (statistic).** Among calibrated variants on `dinov3_patch`, `knn` beats
  `vmf` on separation by at least 0.10. A non-parametric node score should see
  a diffuse space better than a mean direction.

## Decision rule (fixed now)

A variant **passes** when, on every embedder: KS <= 0.05, rate@0.05 within
0.03-0.075, false alarm <= 0.10; **and** separation on `dinov3_patch` >= 0.50;
**and** on each other embedder, separation >= the shipped model's minus 0.10.

- **If one or more variants pass**, the one with the highest minimum separation
  across embedders goes into `vtscore/coverage/atlas.py`, with a new
  persistence format, and the patch-embedder refusal is lifted. A tie within
  0.05 goes to the cheaper one: a tree variant whose node state is scalars
  beats one that must read node vectors at query time, which beats
  `knn_global`.
- **If only `knn_global` passes**, the report recommends it, and that is a
  replacement of the atlas's guard, not a repair of its node model. It goes to
  the owner, because it changes what the endpoint needs at query time (the
  reference dataset's vectors).
- **If nothing passes**, the refusal stays, and the report says which
  condition failed and on which embedder.

# Pre-registration, part 2: the rest of the #3329 inventory

**Written before the run**, on the night of 2026-08-30, after part 1 (the score
mixture) merged as PR #3333. Part 1's pre-registration is sealed in
[`PREREG.md`](PREREG.md) and is not edited by this; this file covers the
families part 1 explicitly deferred — the issue's parts **B** (embedding-space
structure) and **C** (browse projection), plus the one cheap item from **D**.

Part 1's method carries over: every statistic is **absolute** — a fit against
its own data, never against a rival model — and no p-values are reported on the
distributional statistics, because at these sample sizes every test rejects
every model. The bars are effect sizes.

## Why these need a different shape of run

Part 1's fits are recomputed at every click, so they rode the 192-cell voting
loop for free. **These do not.** The Coverage Atlas, the kNN conformal support
rule, the UMAP layout and the HDBSCAN compaction radius are each fitted **once
per dataset**, so the axis that buys information here is not clicks — it is
**dataset × embedder**. The grid is 5 × 5 rather than 192 × 1.

## BLUF — the prediction

**The atlas's stated null is false, and the reason is its own aggregation
rather than its fit.** The guard will look well-behaved in the one place anyone
would check it (the fraction below alpha) while being wrong in the place that
decides its power (the spread).

| # | claim | pre-registered bar |
|---|---|---|
| B1 | in-domain typicality p-values are **not** uniform | median `ks_uniform` on the **holdout** > 0.05 across the grid |
| B2 | they are specifically **under-dispersed**, and path averaging is why | `sd` < 0.27 (U(0,1) is 0.289); and `sd` rises toward 0.289 when scored at the deepest node only, on ≥ 4 of 5 embedders |
| B3 | the under-dispersion tracks the averaging, not the data | Spearman ρ(`path_len_mean`, −`sd`) > 0.5 across the 25 cells |
| B4 | the guard is **conservative**, not trigger-happy | median self-`z_score` ≤ 0, and `shifted` fires on **0** of the 25 self pairs |
| B5 | it still separates real domains | `shifted` fires on the majority of off-diagonal (build ≠ query) pairs |
| C1 | UMAP preserves neighbourhoods locally and not globally | median `trustworthiness_k10` > 0.95 **and** median `shepard_spearman` < 0.6 |
| C2 | the projection costs class purity | `knn_class_purity_k10` lower in the layout than in the embedding, on ≥ 4 of 5 embedders |
| C3 | the fitted 90th-percentile cluster radius contains ~90 % | median `containment_mean` within 0.85–0.95 |

**B4 and B5 are the pair that matters**, and they are pre-registered together on
purpose. A guard that never fires on its own data looks correct; a guard that
never fires on *anything* is broken. Only the two readings together say which
one this is, and either result is reportable.

**C1 is pre-registered as a split verdict** so that "the layout is good at what
it is for and bad at what it is not for" is a finding rather than a hedge
written afterwards. The browse canvas is a neighbourhood-inspection tool, so
local trustworthiness is the property that matters and global distance is not
one it ever promised — but nothing in the tree measures either, and a reader who
believes the picture shows global structure is being misled.

## The grid

| axis | values |
|---|---|
| datasets | `vg_scale_any`, `coco_val`, `caltech101_m`, `vg_box_large`, `visual_genome_m` |
| embedders | `siglip`, `dinov3_patch`, `clip`, `clip_l`, `siglip2_l` |

**25 cells** for the per-dataset families, plus **5 cross-dataset tasks** (one
per embedder) giving every ordered (build, query) dataset pair — 125 pairs for
B4/B5. Five datasets spanning three sources (COCO, Visual Genome, Caltech) is
what makes B5 answerable at all: an "off-diagonal" pair between two Visual
Genome slices is a much weaker shift than one between Caltech and COCO, and the
report has to read them apart.

The atlas is built exactly as production builds it — `k=3`,
`max_depth=auto_max_depth(n, k=3)`, the call in
`vtscore/detectors/labeling_progress.py` — so the null under test is the shipped
estimator's, not a variant of it.

## Two design choices, stated in advance

**Every PIT reading is on a held-out 20 % split.** The build points sit inside
their own calibration quantiles, and although the atlas LOO-corrects them
(`t_loo` in `_make_node`), an in-sample reading would still be the fit's own
best case. The build-set reading is emitted beside the holdout one so the size
of that optimism is visible rather than assumed.

**The conformal family is partly a positive control.** Split-conformal support
p-values are uniform under exchangeability *by construction*, so if the
`conformal / in_class_holdout` scope does not read uniform, the first suspect is
this harness and not the app. A part-1 lesson, applied: three of that run's
findings were instrument defects that emitted plausible numbers, so this run
carries a family whose right answer is known in advance.

## What would make this uninterpretable

- **Fewer than 40 in-class items** for the conformal split on a given cell — its
  p-value resolution is then `1/(m+1)` with m tiny, and the scope is reported as
  unresolved rather than as a null.
- **An atlas shallower than 3 calibrated levels**, which makes B2/B3 vacuous:
  with a path length of 1 there is no averaging to blame. `path_len_mean` is
  recorded per cell so this is visible, and the smoke run on `caltech101_m`
  already showed 2.9, so the small datasets are near this edge.
- **UMAP failing to the PCA fallback** (under 10 points, or a degenerate
  matrix), which would make C1–C3 describe a different projector entirely.

## Part D, and what is still not covered

`vtscore/timing/fit.py` computes an OLS r² in `affine_fit` and **throws it
away** at the call site (`fit_step`). That is the one place in the tree that
already computes a goodness-of-fit statistic and discards it, so it is fixed
here rather than measured: r² is kept on `StepCoeffs` and reported.

**Not covered by this run, and deliberately:** the SIFT/VLAD RANSAC
reprojection-error distributions and the MatchStats verification MLP (§11 of the
inventory) are off the default path and need structural-search fixtures this
grid does not build; and Toponymy's suppressed warnings (§10) need the captioner
stack. Both are named here so the inventory's remaining surface is explicit
rather than quietly dropped.

## Analysis, fixed in advance

`analyze_structure_3329.py`, with `selftest_analyze_structure_3329.py` (planted
answers) green **before any result is read** — the bars above are fixed by this
file before the grid is submitted, which is the thing pre-registration protects,
and the analyzer is written against them while the grid runs. Two significant
digits; every bar is a module-level constant in the analyzer, so the selftest
plants against the same number the report quotes.

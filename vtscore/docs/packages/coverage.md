# `vtscore.coverage`

The Coverage Atlas structure: a hierarchical partition of a dataset's
embedding space that remembers, per region, how much labeled evidence of
each class the user has provided.

This package is the **algorithm**, and nothing else. It holds no state, takes
no lock, and never touches a `DatasetContext` — every entry point here is a
function of the embedding matrix handed to it. The *wiring* that builds an
atlas for the active dataset, replays a detector's votes into it and caches
it on a context is separate, in `vtscore/state/coverage.py`; see
[`state.md`](state.md#coverage-atlas) for that half and for the
`build_coverage_atlas*` / `coverage_atlas_*` API the app calls.

The split is deliberate. The two halves used to sit side by side as
`state/coverage.py` and `state/coverage_atlas.py` — a near-homograph pair in
which only one of the two was actually state.

Related docs: [`state.md`](state.md) for the wiring and the atlas's place on
`DatasetContext`; [`detectors.md`](detectors.md) for the evidence-coverage
report built on the same idea.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/coverage/atlas.py` | `CoverageAtlas`, `auto_max_depth`, `domain_shift_report` — the partition, its evidence channels, moments and calibrated typicality |

## What the structure keeps

The atlas replaces the old **diversity tree**, keeping what the tree threw
away. Three differences matter to callers:

- **Evidence channels, not a "seen" bit.** Each node counts labeled evidence
  per class (`n_pos` / `n_neg`), so "verified good here", "verified bad here"
  and "never exercised" are distinguishable.
- **Mean-centered geometry.** Vectors are mean-centered and re-normalised
  before partitioning: contrastive embeddings concentrate in a narrow cone,
  so raw cosines are uniformly high and the centering is what restores
  contrast. The centering vector is part of the structure.
- **Moments and calibrated typicality.** Each node stores its mean direction
  `mu` and resultant length `rbar` — the sufficient statistics of a von
  Mises–Fisher component — plus a quantile grid of its own points'
  typicality `t(x) = mu · x`. So `typicality_pvalues(matrix)` returns *ranks
  on a p-value-like scale* rather than raw distances, which is what
  `domain_shift_report` uses to detect a detector being pointed at a dataset
  it was not trained on.

## Entry points

| Symbol | Description |
|--------|-------------|
| `CoverageAtlas` | The partition itself: build, lookup, evidence counting, coverage level, next-sample selection, typicality |
| `auto_max_depth(n, ...)` | Depth to build to for a dataset of *n* items |
| `domain_shift_report(atlas, matrix, alpha=0.05)` | Dataset-level shift report: how much of *matrix* looks atypical under *atlas* |
| `COVERAGE_ATLAS_DEFAULT_K` / `COVERAGE_ATLAS_MAX_DEPTH` / `COVERAGE_ATLAS_MIN_NODE_SIZE` | Partition-shape defaults |

`CoverageAtlas.typicality_pvalues` returns ranks, not calibrated p-values —
its docstring records the measured deviation and what it costs.

## Compatibility

`vtscore.state.coverage_atlas` remains as a deprecated alias that re-exports
this package and warns on import. Import from `vtscore.coverage` instead.

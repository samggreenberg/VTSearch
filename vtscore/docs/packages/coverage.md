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

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/coverage/atlas.py` | `CoverageAtlas`, `auto_max_depth`, `domain_shift_report` — the partition, its evidence channels, moments and calibrated typicality |

## What the structure keeps

Three properties matter to callers:

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
| `CoverageAtlas(vectors, k=3, max_depth=10, min_node_size=20, on_progress=None)` | The partition itself, built by recursive k-means over `{id: vector}` (`k` must be 2-9). See the method groups below |
| `auto_max_depth(n, k=3, min_node_size=20)` | Depth to build to for *n* items: `COVERAGE_ATLAS_MAX_DEPTH` unless `n / min_node_size` exceeds the 4 000-leaf budget, then clamped so `k**depth` stays under it. At least 1 |
| `domain_shift_report(atlas, matrix, alpha=0.05)` | Dataset-level shift report: how much of *matrix* looks atypical under *atlas* |
| `COVERAGE_ATLAS_DEFAULT_K` / `COVERAGE_ATLAS_MAX_DEPTH` / `COVERAGE_ATLAS_MIN_NODE_SIZE` | Partition-shape defaults: `3` / `10` / `20` |

`CoverageAtlas` methods:

- **Lookup / evidence:** `lookup(vector_id)` (leaf name), `label(vector_id, good)`,
  `unlabel(vector_id)`, `n_pos(name)` / `n_neg(name)`, `reset_labeled()`,
  `labeled_ids()`.
- **Coverage:** `coverage_level()`, `next_sample(...)`, `total_nodes()`,
  `depth()`, `span_info()`.
- **Typicality:** `typicality_pvalues(matrix)`, `typicality_pvalue(vector)`.
- **Persistence:** `to_serializable()` / `CoverageAtlas.from_serializable(data)`
  (skips the k-means build; raises `ValueError` on an unknown or missing
  format so callers can rebuild), `structural_clone()`.

`CoverageAtlas.typicality_pvalues` returns ranks, not calibrated p-values —
its docstring records the measured deviation and what it costs.

## Compatibility

`vtscore.state.coverage_atlas` remains as a deprecated alias that re-exports
this package and warns on import. Import from `vtscore.coverage` instead.

---

## Cross-references

- [`state.md`](state.md#coverage-atlas) - building, replaying and caching an atlas for a live dataset.
- [`detectors.md`](detectors.md) - the Span indicator that reads atlas coverage, and the labelset-kNN evidence-coverage report.

# `vtscore.coverage`

The Coverage Atlas structure: a hierarchical partition of a dataset's
embedding space that remembers, per region, how much labeled evidence of
each class the user has provided. And the domain-shift report: how much of
a dataset looks unlike a reference dataset.

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
| `vtscore/coverage/atlas.py` | `CoverageAtlas`, `auto_max_depth` — the partition, its evidence channels and moments |
| `vtscore/coverage/domain_shift.py` | `domain_shift_report`, `typicality_pvalues` — split-conformal kNN against a reference dataset's vectors (#3348) |

## What the structure keeps

Three properties matter to callers:

- **Evidence channels, not a "seen" bit.** Each node counts labeled evidence
  per class (`n_pos` / `n_neg`), so "verified good here", "verified bad here"
  and "never exercised" are distinguishable.
- **Mean-centered geometry.** Vectors are mean-centered and re-normalised
  before partitioning: contrastive embeddings concentrate in a narrow cone,
  so raw cosines are uniformly high and the centering is what restores
  contrast. The centering vector is part of the structure.
- **Moments.** Each node stores its mean direction `mu` and resultant
  length `rbar` — the sufficient statistics of a von Mises–Fisher
  component — and orders its `ids` by typicality `mu · x`, which
  `next_sample` uses to probe a node's typical half. The atlas no longer
  calibrates that typicality into p-values: #3348 measured the calibration
  as wrong on every embedder and moved domain shift to
  `vtscore/coverage/domain_shift.py`.

## Entry points

| Symbol | Description |
|--------|-------------|
| `CoverageAtlas(vectors, k=3, max_depth=10, min_node_size=20, on_progress=None)` | The partition itself, built by recursive k-means over `{id: vector}` (`k` must be 2-9). See the method groups below |
| `auto_max_depth(n, k=3, min_node_size=20)` | Depth to build to for *n* items: `COVERAGE_ATLAS_MAX_DEPTH` unless `n / min_node_size` exceeds the 4 000-leaf budget, then clamped so `k**depth` stays under it. At least 1 |
| `domain_shift_report(reference, matrix, alpha=0.05)` | Dataset-level shift report: how much of *matrix* looks atypical against the *reference* vectors. Split-conformal kNN (k = 5), so a reference item's p-value is uniform; scores at most 4 000 rows of *matrix*; raises `ReferenceTooSmall` under 40 reference vectors |
| `typicality_pvalues(reference, matrix, k=5)` | The per-row p-values behind the report, every row scored |
| `COVERAGE_ATLAS_DEFAULT_K` / `COVERAGE_ATLAS_MAX_DEPTH` / `COVERAGE_ATLAS_MIN_NODE_SIZE` | Partition-shape defaults: `3` / `10` / `20` |

`CoverageAtlas` methods:

- **Lookup / evidence:** `lookup(vector_id)` (leaf name), `label(vector_id, good)`,
  `unlabel(vector_id)`, `n_pos(name)` / `n_neg(name)`, `reset_labeled()`,
  `labeled_ids` (property).
- **Coverage:** `coverage_level()`, `next_sample(...)`, `total_nodes`
  (property), `depth()`, `span_info()`.
- **Persistence:** `to_serializable()` / `CoverageAtlas.from_serializable(data)`
  (skips the k-means build; raises `ValueError` on an unknown or missing
  format so callers can rebuild; a pre-#3348 snapshot's `t_quantiles` are
  ignored), `structural_clone()`.

## Compatibility

`vtscore.state.coverage_atlas` remains as a deprecated alias that re-exports
the atlas and warns on import. Import from `vtscore.coverage` instead. It no
longer re-exports `domain_shift_report`: the atlas-based report it named was
removed in #3348, and the new one takes vectors, not an atlas.

---

## Cross-references

- [`state.md`](state.md#coverage-atlas) - building, replaying and caching an atlas for a live dataset.
- [`detectors.md`](detectors.md) - the Span indicator that reads atlas coverage, and the labelset-kNN evidence-coverage report.

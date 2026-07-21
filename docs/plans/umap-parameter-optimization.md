# Plan: per-embedder UMAP parameter optimization

**Goal:** empirically choose UMAP projection parameters (VTSBrowse Stage 1,
`vtscore/projection/umap_projection.py`) **per embedder**, by sweeping a
parameter grid on a GPU cluster (GRID) and scoring each candidate layout with a
taxonomy-separability metric. Ships as a per-embedder defaults table in code
(current globals remain the fallback; `ServerSettings` still override).

This plan supersedes **Phase A (quantitative sweep)** of
`docs/plans/vtsbrowse-empirical-tuning.md`; that plan retains the
browser-required qualitative review (Phase B) and the pyramid/canvas/hover
knobs, which stay out of scope here.

## Locked scope decisions

- **Backend strategy: cuML sweep + CPU verify.** Run the full grid on
  `cuml.manifold.UMAP` (the GRID/GPU production path), then re-fit **only the
  winning params per embedder** with CPU `umap-learn` and confirm the metric
  ranking transfers. The two backends produce different layouts for identical
  params (different optimizers; cuML lacks densmap and some `init` modes), so
  the verify step is what licenses shipping one defaults table for both. If a
  winner does *not* transfer, fall back to the best param set that scores well
  on both backends rather than introducing backend-conditional defaults.
- **Embedders in scope (first pass): `siglip`, `clip`, `siglip_l` (image) and
  `clap` (audio).** Patch embedders don't feed the browse projection.
  Everything else (`siglip2`, DINO singles, EUPE, SIFT-VLAD, other CLAP
  variants, AST, text/video/document embedders) inherits its media type's
  closest tuned value for now and can be added in a later pass with the same
  harness.
- **Datasets: demo registry + 1–2 deep-taxonomy additions** (see §Datasets).
- **Primary metric: ceiling-normalized taxonomy separability**, with
  label-free structure-preservation guards and multi-seed stability as
  tie-breaker (see §Metric).

## Parameter grid

Primary axes (full grid, every cell):

| Knob | Values | Rationale |
|------|--------|-----------|
| `n_neighbors` | {5, 10, 15, 30, 50, 100, 200}, clamped to N−1 | The local-vs-global knob; the one most likely to differ per embedder — and per N (see §Analysis). |
| `min_dist` | {0.0, 0.05, 0.1, 0.25, 0.5} | Layout packing / boundary margins; doesn't change the neighbor graph, so expect weaker metric response. |

Secondary axes (sweep only where the primary grid is flat or unstable):

- `init` ∈ {spectral, random} — spectral can degrade on disconnected graphs;
  restrict to modes cuML supports.
- `n_epochs` ∈ {auto, 500, 1000} — convergence on large N.
- Optional: PCA pre-reduction to 50-d before UMAP (denoise/speed; changes
  results, so it must be swept, not assumed).

Held fixed: `metric="euclidean"` (settled — embeddings are L2-normalized at
ingest, so euclidean is cosine-monotonic), `spread`, `repulsion_strength`,
`negative_sample_rate`, `set_op_mix_ratio`, `local_connectivity`. densmap is
unavailable in cuML and stays out.

Seeds: **3 seeded fits per cell** (seeded in the harness only; production
stays unseeded). The stamped `random_state` makes cells comparable; the seed
spread measures the run-to-run variance the unseeded production fit will see.

## Metric

**Primary — ceiling-normalized taxonomy separability.** For a dataset with a
class taxonomy (a tree of labeled subsets: animals ⊃ mammals ⊃ dogs …):

1. Build one kNN graph (k ≈ 20) of the **final 2-D layout** — i.e.
   *post-compaction* coordinates, since `compact_layout` is part of what ships.
2. For each taxonomy node with member set S (one-vs-rest): score every point
   by the fraction of its 2-D neighbors inside S; the node's score is the
   **AUROC** of that fraction against actual membership. AUROC ≈ 1 ⇔ a clean
   boundary exists at the k scale. Position/scale-invariant and
   multi-island-tolerant by construction (a node may form several clean blobs
   without penalty). One kNN graph serves all nodes.
3. Aggregate: mean over nodes **per taxonomy level**, then average the levels,
   so coarse levels (animal/not) aren't swamped by many leaf classes.
4. **Normalize by the high-D ceiling:** compute the same score on the original
   embedding matrix (cosine-space kNN) and report the **ratio** 2-D/high-D.
   The ratio is the quantity UMAP params actually control; the raw high-D
   score would punish the projection for the embedder's own inability to
   separate a class.

**Guards (label-free)** — protect against layouts that game class purity by
shattering the space:

- kNN-recall@k: overlap between each point's 2-D and high-D neighbor sets.
- Trustworthiness & continuity (`sklearn.manifold.trustworthiness`).

**Tie-breaker — stability:** across the 3 seeds, report metric mean ± std and
inter-seed layout agreement (neighbor-set overlap between runs). Prefer params
on a plateau over a knife-edge winner; production is unseeded, so a chosen
default must be robust to seed variation.

Sanity check (once, not per cell): confirm compaction is metric-neutral by
scoring pre- vs post-compaction on a few cells — rigid cluster motion should
barely move kNN purity, but islands slid adjacent could.

## Datasets

Image and audio only (matching the embedder scope). Use demo-registry sources
so the experiment exercises the real ingest path, with registry S/M/L/A size
variants supplying the N axis:

| Media | Dataset | Taxonomy | Role |
|-------|---------|----------|------|
| Audio | ESC-50 | 5 super-categories × 10 classes (animals is a branch) | primary audio set |
| Audio | UrbanSound8K | urban-sound taxonomy (2 levels) | second audio set |
| Audio | GTZAN | flat genres | secondary; leaf-level only |
| Image | Places365 | indoor/outdoor/nature → 365 scenes | primary image set, large N |
| Image | Caltech-256 | broad object categories | second image set |

**Deep-taxonomy additions (1–2 new downloaders):** nothing in-registry has a
4+ level tree, which is exactly the dogs → canines → mammals → animals stress
case. Candidates:

- **iNaturalist (mini/subset)** — image; kingdom→phylum→class→order→family→
  genus→species, the literal use case. First choice.
- **FSD50K** — audio; ~51k Zenodo-downloadable clips labeled against the
  hierarchical AudioSet ontology (multi-label is fine — the metric is
  one-vs-rest per node). First-choice audio deep tree.
- (ImageNet-subset + WordNet hierarchy is the fallback if iNaturalist proves
  awkward to subset.)

Multi-label datasets need no special handling; each taxonomy node is scored
one-vs-rest independently.

## Experiment mechanics

- Lives in **`scripts/experiments/umap_params/`**, following the
  `scripts/experiments/toponymy_image/` GRID precedent (`setup_node.sh`,
  `prepare_dataset.py`, `queue_all.sh`, `evaluate.py`, `summarize.py`,
  `visualize.py`). Dev-only: outside the shipped package, outside deptry's
  prod surface, must not break `./run-tests.sh`.
- **Embed once, sweep many.** Embedding each (dataset × embedder) dominates
  cost; the sweep itself is cheap re-fits. Snapshot embedded datasets as
  **dataset pickles** (the sanctioned persistence exception — pickles ARE the
  dataset) and fan the grid out over cached matrices via
  `get_embedding_matrix`.
- Budget estimate: 35-cell primary grid × 3 seeds × ~10 dataset-embedder cells
  (incl. size variants) ≈ 1–2k cuML fits — small next to the one-time
  embedding cost of Places365/iNaturalist/FSD50K.
- Each cell emits one row (dataset, embedder, N, params, seed, all metrics,
  fit seconds) to CSV/JSON; `summarize.py` produces per-(embedder, dataset)
  heatmaps and the per-embedder recommendation; `visualize.py` renders
  thumbnail scatter grids colored by top-level taxonomy class for eyeballing.

## Analysis & decision procedure

1. Per (embedder, dataset, N): heatmap of the primary metric over the grid;
   identify the plateau, not just the argmax.
2. **The N question:** test whether the `n_neighbors` optimum tracks the
   embedder or tracks N. Compare a per-embedder constant against a simple
   `n_neighbors(N)` rule (e.g. `clip(round(c·N^α), lo, hi)`) fit across the
   size variants. If N dominates, the deliverable becomes a per-embedder
   *rule*, not a per-embedder *constant* — the plumbing below supports either.
3. Pick per-embedder winners subject to: on-plateau, guard metrics not
   degraded vs current defaults, stable across seeds.
4. **CPU verify:** re-fit winners (and the current defaults as baseline) with
   `umap-learn`, 3 seeds; accept if the winner still beats the baseline on the
   primary metric. On failure, choose the best both-backend performer.

## Deliverables (open work)

<!-- item-sep -->

- **Sweep harness** — `scripts/experiments/umap_params/` per §Experiment
  mechanics: metric implementation (taxonomy AUROC + ceiling normalization +
  guards), grid runner over cached embedded pickles, seeded fits, CSV/JSON
  emission, summarize/visualize scripts, short README. Taxonomy definitions
  for each dataset (node → member classes) live beside the harness as data
  files.

<!-- item-sep -->

- **Deep-taxonomy downloaders** — add iNaturalist-mini (image) and FSD50K
  (audio) demo sources (or the chosen fallbacks), wired through the standard
  downloader/demo-registry path with size variants.

<!-- item-sep -->

- **Cluster run + write-up** — execute the sweep on GRID, run the CPU verify
  pass, and record the winning per-embedder params, heatmaps, and
  metric/plot evidence in §Results below.

<!-- item-sep -->

- **Per-embedder defaults plumbing** — a defaults map keyed by embedder name
  (constant or `n_neighbors(N)` rule per §Analysis), consulted where
  `PROJECTION_N_NEIGHBORS` / `PROJECTION_MIN_DIST` are read today, keyed off
  the embedder that produced the projected matrix (the dataset's *primary*
  embedder — `get_embedding_matrix` semantics), falling back to the current
  globals for untuned embedders; `ServerSettings` overrides still win. The
  persisted projection is already keyed on effective params, so new defaults
  force recompute instead of serving stale layouts — no migration needed.

<!-- item-sep -->

## Risks & notes

- **Backend transfer** is the main scientific risk; the CPU verify step is the
  mitigation, and "best both-backend performer" the fallback.
- `min_dist` may barely move the separability metric (it doesn't alter the
  neighbor graph). If so, choose it in the Phase B qualitative review
  (`vtsbrowse-empirical-tuning.md`) on visual grounds and let this experiment
  settle `n_neighbors` only.
- Metric gaming: a projection could shatter space to boost one-vs-rest purity;
  the label-free guards exist precisely to veto such cells.
- Tiny-N cells fall into the PCA/trivial fallback (`min_n_for_umap=10`) —
  exclude N below the UMAP boundary from the grid; the fallback isn't tunable.
- Pyramid/hex readability (`base_cols`, `tile_span`, canvas radius) is
  deliberately out of scope; it consumes whatever layout Stage 1 produces and
  stays with the old plan.

## Results

_(empty — fill in when the cluster run happens.)_

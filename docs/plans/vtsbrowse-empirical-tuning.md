# Plan: VTSBrowse empirical tuning

**Goal:** empirically choose the VTSBrowse projection-and-rendering defaults,
in three parts:

1. **Per-embedder UMAP parameters** (Stage 1,
   `vtscore/projection/umap_projection.py`) — a parameter sweep on a GPU
   cluster (GRID), scored with a taxonomy-separability metric. The bulk of
   this plan.
2. **Pyramid parameters** (Stage 2, `build_pyramid`) — a headless sweep of
   `base_cols` / `tile_span` on top of the tuned projections.
3. **Qualitative review** (browser required) — visual judgment of the
   shortlisted candidates on the rendered canvas, plus hover-feel and
   colormap checks the metrics can't capture.

The UMAP sweep ships as a per-embedder defaults table in code (current
globals remain the fallback; `ServerSettings` still override).

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
- **Compaction is an evaluated boolean, not an assumption.** Every fitted
  layout is scored twice — raw UMAP coordinates and post-`compact_layout`
  coordinates — so the sweep quantifies whether compaction helps or hurts
  separability, and by how much (see §Metric).

## Part 1 — per-embedder UMAP sweep (GRID)

### Parameter grid

Primary axes (full grid, every cell):

| Knob | Values | Rationale |
|------|--------|-----------|
| `n_neighbors` | {5, 10, 15, 30, 50, 100, 200}, clamped to N−1 | The local-vs-global knob; the one most likely to differ per embedder — and per N (see §Analysis). |
| `min_dist` | {0.0, 0.05, 0.1, 0.25, 0.5} | Layout packing / boundary margins; doesn't change the neighbor graph, so expect a weaker metric response (extremes may still matter — swollen clusters can abut). |
| `compact` | {false, true} | **Free axis:** compaction is a post-process, so both variants are scored from the *same* fit — no extra UMAP runs. |

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

### Metric

**Primary — ceiling-normalized taxonomy separability.** For a dataset with a
class taxonomy (a tree of labeled subsets: animals ⊃ mammals ⊃ dogs …):

1. Build one kNN graph (k ≈ 20) of the 2-D layout. Compute this **twice per
   fit** — once on the raw UMAP coordinates and once on the
   post-`compact_layout` coordinates — emitting a full metric set for each, so
   the `compact` boolean is evaluated everywhere, not spot-checked.
   Compaction moves clusters as rigid bodies, so any score change isolates its
   between-cluster effects (islands slid adjacent can bleed neighbors across
   a boundary).
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

Both guards are also computed for both `compact` variants.

**Tie-breaker — stability:** across the 3 seeds, report metric mean ± std and
inter-seed layout agreement (neighbor-set overlap between runs). Prefer params
on a plateau over a knife-edge winner; production is unseeded, so a chosen
default must be robust to seed variation.

### Datasets

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

### Experiment mechanics

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
  embedding cost of Places365/iNaturalist/FSD50K. The `compact` axis adds
  scoring passes, not fits.
- Each cell emits rows (dataset, embedder, N, params, seed, **compact
  boolean**, all metrics, fit seconds) to CSV/JSON; `summarize.py` produces
  per-(embedder, dataset) heatmaps, a compaction-delta report (Δ-separability
  raw→compacted, aggregated per dataset and overall), and the per-embedder
  recommendation; `visualize.py` renders thumbnail scatter grids colored by
  top-level taxonomy class for eyeballing.

### Analysis & decision procedure

1. Per (embedder, dataset, N): heatmap of the primary metric over the grid;
   identify the plateau, not just the argmax.
2. **The N question:** test whether the `n_neighbors` optimum tracks the
   embedder or tracks N. Compare a per-embedder constant against a simple
   `n_neighbors(N)` rule (e.g. `clip(round(c·N^α), lo, hi)`) fit across the
   size variants. If N dominates, the deliverable becomes a per-embedder
   *rule*, not a per-embedder *constant* — the plumbing below supports either.
3. **The compaction question:** from the compaction-delta report, decide
   whether `compact=True` stays the production default. Compaction exists for
   canvas readability (closing empty oceans), so a small separability cost may
   be acceptable — but a consistent, material hit flips the default (or sends
   `compact_layout` back for rework, e.g. a minimum inter-island margin).
   Quantify "by how much" either way.
4. Pick per-embedder winners subject to: on-plateau, guard metrics not
   degraded vs current defaults, stable across seeds.
5. **CPU verify:** re-fit winners (and the current defaults as baseline) with
   `umap-learn`, 3 seeds; accept if the winner still beats the baseline on the
   primary metric. On failure, choose the best both-backend performer.

## Part 2 — pyramid-parameter sweep (headless, scriptable)

For each `(dataset, params)` cell, take a fixed projection (the tuned UMAP
defaults once Part 1 lands), run `build_pyramid` with trial pyramid params,
and emit metrics to CSV/JSON:

- **Build cost:** pyramid build seconds, peak RSS.
- **Aggregation health per level:** for each pyramid level — number of
  non-empty hexes, count distribution (min/median/p95/max items per hex),
  fraction of single-item hexes, tile fan-out (hexes per tile, tiles per
  level). Tells whether `base_cols`/`tile_span`/`n_levels` produce a readable,
  evenly-loaded pyramid or a few mega-hexes + dust.
- **Small-N behavior:** confirm the PCA-2 / trivial fallbacks trigger at the
  intended `min_n_for_umap` boundary and produce sane bounds.

**Parameter grid (starting point, widen if flat):**

- `base_cols ∈ {4, 6, 9}`
- `tile_span ∈ {8, 16, 32}`

**Datasets** — span media type and N (use the S/M/L/A size variants the demo
registry provides, e.g. `esc50_s/m/l/a`, `gtzan_a`); at least one small
(hundreds), one medium (few thousand), and one large (tens of thousands)
dataset to find the performance ceiling.

## Part 3 — qualitative review (browser required)

For a shortlist of param sets that scored well in the quantitative sweeps,
load each dataset in the actual Browse canvas (`/browse/:datasetId`) and judge
what the metrics can't:

- **Cluster separation & shape:** do semantically-distinct groups (genres,
  news categories, image classes) actually land in visually-separable blobs?
  Compare `n_neighbors` (local vs global structure) and `min_dist` (tight vs
  spread clusters) side by side. `min_dist` in particular may be decided
  *here* rather than by Part 1, if the metric response is as weak as expected.
- **Compaction eyeball check:** compare raw vs compacted shortlist layouts on
  the canvas; the Part 1 delta report says what separability costs, this says
  what readability buys.
- **Hex readability across zoom:** at the auto-picked level, is the on-screen
  hex count comfortable (not 3 giant hexes, not 10k specks)? Does the
  `targetScreenRadius=28` level-picker land on the right level? Sweep it (e.g.
  22 / 28 / 36) if levels feel off.
- **Density colormap:** is **log** the right compression, or does **sqrt** read
  better for the actual count distributions seen in the sweeps? The ramp is now
  darkred→yellow (black left free to mean empty/"None"); revisit the stop count
  and endpoints if the sweep says so.
- **Hover feel:** debounce window (does sweeping machine-gun previews at 30 ms?
  try 50–80 ms), audio loop hard-cut vs a short fade, text-snippet length,
  popup placement.
- **Performance:** pan/zoom smoothness at the largest dataset; confirm Canvas
  2D culling holds (the design's WebGL escape hatch stays deferred unless this
  fails).

Capture before/after screenshots for the write-up.

## Deliverables (open work)

<!-- item-sep -->

- **UMAP sweep harness** — `scripts/experiments/umap_params/` per §Experiment
  mechanics: metric implementation (taxonomy AUROC + ceiling normalization +
  guards, scored per `compact` variant), grid runner over cached embedded
  pickles, seeded fits, CSV/JSON emission, summarize/visualize scripts
  (including the compaction-delta report), short README. Taxonomy definitions
  for each dataset (node → member classes) live beside the harness as data
  files.

<!-- item-sep -->

- **Deep-taxonomy downloaders** — add iNaturalist-mini (image) and FSD50K
  (audio) demo sources (or the chosen fallbacks), wired through the standard
  downloader/demo-registry path with size variants.

<!-- item-sep -->

- **Cluster run + write-up** — execute the sweep on GRID, run the CPU verify
  pass, and record the winning per-embedder params, the compaction verdict
  (keep/flip/rework, with the measured delta), heatmaps, and metric/plot
  evidence in §Results below.

<!-- item-sep -->

- **Per-embedder defaults plumbing** — a defaults map keyed by embedder name
  (constant or `n_neighbors(N)` rule per §Analysis), consulted where
  `PROJECTION_N_NEIGHBORS` / `PROJECTION_MIN_DIST` are read today, keyed off
  the embedder that produced the projected matrix (the dataset's *primary*
  embedder — `get_embedding_matrix` semantics), falling back to the current
  globals for untuned embedders; `ServerSettings` overrides still win. Apply
  the compaction verdict to `fit_projection`'s `compact` default. The
  persisted projection is already keyed on effective params, so new defaults
  force recompute instead of serving stale layouts — no migration needed.

<!-- item-sep -->

- **Pyramid sweep harness + run** — Part 2 above; can share scaffolding with
  the UMAP harness under `scripts/experiments/`. Dev-only, outside the shipped
  package and deptry's prod surface. Chosen `base_cols`/`tile_span` defaults
  landed in `pyramid.py` with a one-line rationale.

<!-- item-sep -->

- **Qualitative review + final defaults** — Part 3 above; canvas/hover/
  colormap knobs landed in the frontend components, `min_dist` confirmed or
  chosen visually, screenshots captured for the write-up. Update
  `docs/plans/vtsbrowse.md` *§Open problems → Empirical* to "settled — see
  §Results".

<!-- item-sep -->

## Risks & notes

- **Backend transfer** is the main scientific risk; the CPU verify step is the
  mitigation, and "best both-backend performer" the fallback.
- `min_dist` may barely move the separability metric (it doesn't alter the
  neighbor graph). If so, choose it in the Part 3 qualitative review on visual
  grounds and let Part 1 settle `n_neighbors` (and `compact`) only.
- Metric gaming: a projection could shatter space to boost one-vs-rest purity;
  the label-free guards exist precisely to veto such cells.
- Tiny-N cells fall into the PCA/trivial fallback (`min_n_for_umap=10`) —
  exclude N below the UMAP boundary from the grid; the fallback isn't tunable.
- Tune with fixed seeds for comparability, but the chosen defaults must be
  robust to the run-to-run variation of the *unseeded* production fit — prefer
  params stable across seeds, not a knife-edge winner.
- Any container built before the params-in-match fix carries a projection fit
  under old params and must recompute (the persistence-key fix is what makes
  re-tuning safe). If the largest target N stutters even after tuning, that
  triggers the deferred **WebGL renderer** follow-up (out of scope).

## Reference: current defaults and where each knob lives

The knobs the sweeps vary, their current values, and their source of truth.

### UMAP (Stage 1) — `vtscore/projection/umap_projection.py:fit_projection`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_neighbors` | `15` | Clamped to `N-1`. A `ServerSettings` (`projection_n_neighbors`), plumbed through the route. |
| `min_dist` | `0.1` | A `ServerSettings` (`projection_min_dist`), plumbed through. |
| `compact` | `True` | Post-fit `compact_layout` rigid-body packing; evaluated as a boolean by the Part 1 sweep. |
| `min_n_for_umap` | `10` | Below this → PCA-2 fallback. Constant for now. |
| `random_state` | `None` (unseeded) | Keep unseeded in prod. Seed only inside the sweep harness. |
| `metric` | `"euclidean"` | Settled (ingest-normalized vectors); not a tuning target. |

### Pyramid (Stage 2) — `vtscore/projection/pyramid.py:build_pyramid`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_levels` | **auto-depth** (`None`) | Descends until every occupied hex holds a single clip, stopping early on co-located clips. Derived, not exposed. |
| `max_useful_levels(N)` | `1 + ceil(log2 N)`, clamped `[1, 14]` | Runaway-guard **ceiling** for the auto-depth descent only, not the operating depth. |
| `base_cols` | `6.0` | Level-0 spans ~6 hex columns across the larger extent. Drives `base_radius`. Constant for now, revisit if sweep says so. |
| `base_radius` | derived (`None` → `_base_radius_for`) | Override path exists but unused. |
| `tile_span` | `16.0` | Hex columns/rows per tile. Payload-size vs round-trip-count tradeoff. Constant for now. |

`build_pyramid(proj)` is called with **no** `n_levels` (auto-depth) in
`vtsearch/routes/projection.py` and `vtscore/datasets/load_pipeline.py`;
`base_cols`/`tile_span` use the function defaults.

### Canvas renderer (Stage 3) — `frontend/.../browse-canvas/browse-canvas.component.ts`

| Knob | Current value | Location |
|------|---------------|----------|
| Target on-screen hex radius (level picker) | `28` px (`DEFAULT_TARGET_RADIUS`), scaled by thumbnail-size buttons (XS–XL → ×0.5–×2.5) | `levelForEffZoom` / `setThumbnailRadius` |
| Density scale | log (`log(count)/log(maxCount)`) | `drawHex` |
| Colormap | darkred→yellow, 8-stop LUT (black left free for empty) | `HEATMAP`, `hex-render.util.ts` |
| Pile cell shape | inscribed disc (rounded); singleton keeps the sharp hex | `traceCellPath`, `hex-render.util.ts` |
| Minimap overview level | hexes ≈ `5` px | `overviewLevel`, `browse-minimap.component.ts` |
| Hover debounce | `30` ms | `onCanvasMouseMove` |
| Hex hit radius | `1 × radius` | `hitTest` |

### Hover preview — `frontend/.../browse-hover-preview/browse-hover-preview.component.ts`

| Knob | Current value | Location |
|------|---------------|----------|
| Audio | `loop=true`, hard-cut on move | `playAudio` |
| Text truncation | `300` chars | `loadText` |
| Popup offset | `+16 / -8` px from cursor | `show` |

## Results

_(empty — fill in when the sweeps run.)_

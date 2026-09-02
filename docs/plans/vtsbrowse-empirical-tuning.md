# Plan: VTSBrowse empirical tuning

**Background.** VTSBrowse's projection-and-rendering defaults were to be chosen
empirically in three parts: per-embedder UMAP parameters (Stage 1), pyramid
parameters (Stage 2), and a qualitative canvas review (Stage 3). **Part 1 is
done** — a GRID/cuML sweep over `n_neighbors` × `min_dist` × `compact` × 3 seeds
across 23 embedded (dataset × embedder) matrices, scored with a ceiling-normalized
taxonomy-separability metric and label-free structure guards, CPU-verified with
`umap-learn`. The chosen per-embedder values live in
`vtscore/config/runtime.py`'s `PROJECTION_DEFAULTS_BY_EMBEDDER` (+
`PROJECTION_COMPACT_DEFAULT`), resolved by
`vtscore/projection/params.py:resolve_projection_params` (the one resolver every
projection fit path calls); the harness is
`scripts/experiments/umap_params/` and the write-up is
[`docs/experiments/2026-07-22-vtsbrowse-umap-tuning/`](../experiments/2026-07-22-vtsbrowse-umap-tuning/REPORT.md).

Three facts from that run shape the work still owed: `n_neighbors` tracks the
embedder, not N (so a per-embedder *constant* is right, and no `n_neighbors(N)`
rule is needed); `min_dist` barely moves separability, so it is a visual call;
and `compact_layout` cost separability on every dataset × embedder (−2.0% mean,
5–6% on the structure guards), which is why compaction now ships **off** and why
the rework below exists.

## Open work

<!-- item-sep -->

- **Pyramid-parameter sweep (Part 2)** — headless and scriptable. For each
  `(dataset, params)` cell, take a fixed projection under the tuned UMAP defaults,
  run `build_pyramid` with trial parameters, and emit to CSV/JSON: build seconds
  and peak RSS; per-level aggregation health (non-empty hexes, items-per-hex
  min/median/p95/max, single-item-hex fraction, hexes per tile and tiles per
  level); and small-N behaviour (confirm the PCA-2 / trivial fallbacks trigger at
  the `min_n_for_umap` boundary with sane bounds). Starting grid, widen if flat:
  `base_cols ∈ {4, 6, 9}` × `tile_span ∈ {8, 16, 32}`. Datasets should span media
  type and N using the registry's S/M/L/A variants (`esc50_s/m/l/a`, `gtzan_a`, …)
  — at least one small (hundreds), one medium (few thousand) and one large (tens
  of thousands) set, so the performance ceiling shows. Can share scaffolding with
  the UMAP harness under `scripts/experiments/`; dev-only, outside the shipped
  package and deptry's prod surface. Land the chosen `base_cols`/`tile_span` in
  `pyramid.py` with a one-line rationale.

<!-- item-sep -->

- **Qualitative review (Part 3, browser required; partly done)** — `min_dist` was
  settled from the sweep (weak effect; 0.05 for images, 0.10 for audio). Still
  owed, on the real Browse canvas (`/browse/:datasetId`):
  - **Cluster separation & shape**, and the **compaction eyeball check** — the
    sweep says what compaction costs in separability; this says what it buys in
    readability. (The live capture was blocked by an environment issue last pass.)
  - **Hex readability across zoom** — at the auto-picked level, is the on-screen
    hex count comfortable (not 3 giant hexes, not 10k specks)? Does
    `targetScreenRadius=28` land on the right level? Sweep 22 / 28 / 36 if not.
  - **Density colormap** — is **log** the right compression, or does **sqrt** read
    better for the count distributions the sweeps saw? Revisit the stop count and
    endpoints of the darkred→yellow ramp if so.
  - **Hover feel** — debounce window (30 ms may machine-gun previews; try 50–80),
    audio loop hard-cut vs a short fade, text-snippet length, popup placement.
  - **Performance** — pan/zoom smoothness at the largest dataset; confirm Canvas 2D
    culling holds. If it doesn't, that triggers the deferred WebGL renderer
    follow-up in [`vtsbrowse.md`](vtsbrowse.md) (out of scope here).

  Capture before/after screenshots for the write-up, and update
  [`vtsbrowse.md`](vtsbrowse.md) *§Empirical knobs* as each knob settles.

<!-- item-sep -->

- **Rework `compact_layout` with a minimum inter-island margin.** Compaction is
  off by default because it consistently bled neighbours across cluster
  boundaries, and the cost grew with layout density (worst on 21k-image
  Places365). A minimum inter-island margin would keep the screen-fill benefit
  without the boundary bleed; if it works, re-score it with the Part 1 harness
  before flipping the default back.

<!-- item-sep -->

- **Deep-taxonomy demo sources (optional follow-up)** — the sweep built the
  iNaturalist (156-species) and FSD50K-eval loaders inside the harness
  (`prepare_dataset.py` / `prepare_fsd50k.py`); wiring them through the standard
  downloader/demo-registry path as first-class demo sources with size variants
  is still owed if they are wanted beyond the experiment.

<!-- item-sep -->

- **Widen the tuned-embedder set (optional).** Part 1 covered `siglip`, `clip`,
  `siglip_l` (image) and `clap` (audio). Everything else (`siglip2`, DINO singles,
  EUPE, SIFT-VLAD, other CLAP variants, AST, text/video/document embedders)
  currently inherits its media type's closest tuned value and can be added with
  the same harness. Patch embedders don't feed the browse projection.

<!-- item-sep -->

## Notes for the remaining work

- Tune with fixed seeds for comparability, but the chosen defaults must be robust
  to the run-to-run variation of the *unseeded* production fit — prefer params
  stable across seeds, not a knife-edge winner. Large `n_neighbors` was the least
  seed-stable region of the Part 1 grid.
- Tiny-N cells fall into the PCA/trivial fallback (`min_n_for_umap=10`); exclude
  N below the UMAP boundary, the fallback isn't tunable.
- The persisted projection is keyed on **effective params**, so changing a default
  forces a clean recompute — which is what makes re-tuning safe.
- The cuML (GPU) and CPU `umap-learn` backends produce different layouts for
  identical params. Part 1's CPU-verify step showed the ranking transfers, which
  is what licenses one defaults table for both; keep that step in any re-tune, and
  on a failure prefer the best both-backend performer over backend-conditional
  defaults.

## Reference: current defaults and where each knob lives

### UMAP (Stage 1) — `vtscore/projection/umap_projection.py:fit_projection`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_neighbors` | per-embedder (`PROJECTION_DEFAULTS_BY_EMBEDDER`): 10 for `clip`/`siglip`/`siglip_l`, 15 for `clap`; global fallback `15` | Clamped to `N-1`. A `ServerSettings` (`projection_n_neighbors`) still overrides. Resolved by `vtscore/projection/params.py:resolve_projection_params`, which every fit path calls (route, ingest pre-build, positives map). |
| `min_dist` | per-embedder: 0.05 image, 0.10 audio; global fallback `0.1` | A `ServerSettings` (`projection_min_dist`) still overrides. |
| `compact` | `False` (`PROJECTION_COMPACT_DEFAULT`), including `fit_projection`'s own signature default | Post-fit `compact_layout` rigid-body packing; off since the Part 1 sweep. Stamped on the `Projection`, so a layout packed under the old default fails the freshness check and is refit. |
| `min_n_for_umap` | `10` | Below this → PCA-2 fallback. Constant for now. |
| `random_state` | `None` (unseeded) | Keep unseeded in prod. Seed only inside a sweep harness. |
| `metric` | `"euclidean"` | Settled (ingest-normalized vectors); not a tuning target. |

### Pyramid (Stage 2) — `vtscore/projection/pyramid.py:build_pyramid`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_levels` | **auto-depth** (`None`) | Descends until every occupied hex holds a single clip, stopping early on co-located clips. Derived, not exposed. |
| `max_useful_levels(N)` | `1 + ceil(log2 N)`, clamped `[1, 14]` | Runaway-guard **ceiling** for the auto-depth descent only, not the operating depth. |
| `base_cols` | `6.0` | Level-0 spans ~6 hex columns across the larger extent. Drives `base_radius`. Part 2's target. |
| `base_radius` | derived (`None` → `_base_radius_for`) | Override path exists but unused. |
| `tile_span` | `16.0` | Hex columns/rows per tile. Payload-size vs round-trip-count tradeoff. Part 2's target. |

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

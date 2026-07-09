# Plan: VTSBrowse empirical tuning pass

**Status:** The remaining work is the quantitative sweep (Phase A) and the
qualitative review (Phase B), detailed below — the deferred empirical-tuning
work from `docs/plans/vtsbrowse.md`.

The remaining steps need a **stronger environment** (a browser for visual
judgment, ideally GPU + real demo datasets) because choosing good defaults
requires *looking at the rendered canvas*, which the headless cloud container
can't do (no Chrome/Chromium). The two most impactful UMAP knobs are exposed as
`ServerSettings`, so both phases retune purely by changing two settings
(`projection_n_neighbors`, `projection_min_dist`).

| Step | Needs a browser? | Needs GPU? | Headless here? |
|------|------------------|-----------|----------------|
| Quantitative sweep harness (impl + run) | no | helps (faster embed) | **yes** (small N) |
| Visual layout / hex-readability review | **yes** | no | no |
| Hover-preview feel (debounce, audio loop) | **yes** | no | no |
| Final default selection + write-up | partly | no | partly |

## Phase A — quantitative sweep (headless, scriptable)

A standalone script (suggested: `scripts/projection_sweep.py`, dev-only, not
shipped) that, for each `(dataset, params)` cell:

1. Loads the demo dataset's embedding matrix via `get_embedding_matrix(ctx)`
   (reuse the real ingest path; embeddings are already L2-normalized).
2. Runs `fit_projection(..., random_state=SEED)` — **seeded here only**, so
   trials are comparable; production stays unseeded.
3. Runs `build_pyramid` with the trial pyramid params.
4. Emits a row of metrics to CSV/JSON.

**Metrics to capture** (all computable without a browser):

- **Build cost:** UMAP fit seconds, pyramid build seconds, peak RSS.
- **Layout quality (neighborhood preservation):** trustworthiness &
  continuity of the 2-D embedding vs the original space (`sklearn.manifold.
  trustworthiness`); optionally kNN-overlap @k. Lets `n_neighbors`/`min_dist`
  be ranked numerically before any eyeballing.
- **Aggregation health per level:** for each pyramid level — number of
  non-empty hexes, count distribution (min/median/p95/max items per hex),
  fraction of single-item hexes, tile fan-out (hexes per tile, tiles per
  level). Tells whether `base_cols`/`tile_span`/`n_levels` produce a readable,
  evenly-loaded pyramid or a few mega-hexes + dust.
- **Small-N behavior:** confirm the PCA-2 / trivial fallbacks trigger at the
  intended `min_n_for_umap` boundary and produce sane bounds.

**Parameter grid (starting point, widen if flat):**

- `n_neighbors ∈ {5, 15, 30, 50, 100}` (clamped to `N-1`)
- `min_dist ∈ {0.0, 0.1, 0.25, 0.5}`
- `base_cols ∈ {4, 6, 9}`
- `tile_span ∈ {8, 16, 32}`

**Datasets** — span media type and N (use the S/M/L/A size variants the demo
registry provides, e.g. `esc50_s/m/l/a`, `gtzan_a`):

| Media type | Demo source(s) | Why |
|------------|----------------|-----|
| Audio | ESC-50 (S→A), GTZAN | clipped → many points per file; tests density + sibling-scatter geometry |
| Text | AG News, BBC, IMDB | large N, E5/BGE embeddings; stresses level count |
| Image | image sources | SigLIP/DINO; visual-cluster sanity |
| Document | UCSF, arXiv | longer docs, fewer items |
| Video | video demos | X-CLIP; smallest N, fallback-boundary check |

Capture at least one **small** (hundreds), one **medium** (few thousand), and
one **large** (tens of thousands if downloadable) dataset to find the
performance ceiling.

## Phase B — qualitative review (browser required)

For a shortlist of param sets that scored well in Phase A, load each dataset in
the actual Browse canvas (`/browse/:datasetId`) and judge what the metrics
can't:

- **Cluster separation & shape:** do semantically-distinct groups (genres,
  news categories, image classes) actually land in visually-separable blobs?
  Compare `n_neighbors` (local vs global structure) and `min_dist` (tight vs
  spread clusters) side by side.
- **Hex readability across zoom:** at the auto-picked level, is the on-screen
  hex count comfortable (not 3 giant hexes, not 10k specks)? Does the
  `targetScreenRadius=28` level-picker land on the right level? Sweep it (e.g.
  22 / 28 / 36) if levels feel off.
- **Density colormap:** is **log** the right compression, or does **sqrt** read
  better for the actual count distributions seen in Phase A? The ramp is now
  darkred→yellow (black left free to mean empty/"None"); revisit the stop count
  and endpoints if the sweep says so.
- **Hover feel:** debounce window (does sweeping machine-gun previews at 30 ms?
  try 50–80 ms), audio loop hard-cut vs a short fade, text-snippet length,
  popup placement.
- **Performance:** pan/zoom smoothness at the largest dataset; confirm Canvas
  2D culling holds (the design's WebGL escape hatch stays deferred unless this
  fails).

Capture before/after screenshots for the write-up.

## Remaining deliverables

2. **Sweep harness** `scripts/projection_sweep.py` + a short README block on
   running it. (Dev tool; keep it out of the shipped package and out of
   `deptry`'s prod-dependency surface — if it needs `sklearn.manifold` etc.,
   confirm those are already deps.)
3. **Chosen defaults** landed in `umap_projection.py` / `pyramid.py` / the
   canvas component / settings, each with a one-line rationale comment or a row
   in the write-up.
4. **Write-up** appended to *§Results* (below, currently empty): the grid that
   was run, the winning values, and the metric/screenshot evidence. Update
   `docs/plans/vtsbrowse.md` *§Open problems → Empirical* to "settled — see plan
   §Results".

**Acceptance:** `./run-tests.sh` green (the sweep script is dev-only and must
not break the build, `deptry`, or `ruff`); new defaults committed with
rationale; no hardcoded UMAP params left unreachable; design doc §Empirical
marked settled and pointing here.

**Risks & notes.** Tune with a fixed seed for comparability, but the chosen
defaults must be robust to the run-to-run variation of the *unseeded* production
fit — prefer params stable across a few seeds, not a knife-edge winner. Any
container built before the params-in-match fix carries a projection fit under
old params and must recompute (the persistence-key fix is what makes re-tuning
safe). If the largest target N stutters even after
tuning, that triggers the deferred **WebGL renderer** follow-up (out of scope).
GPU only speeds up embedding the demo datasets; the projection/pyramid math is
CPU NumPy/numba.

## Reference: current defaults and where each knob lives

The knobs the sweep varies, their current values, and their source of truth.

### UMAP (Stage 1) — `vtscore/projection/umap_projection.py:fit_projection`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_neighbors` | `15` | Clamped to `N-1`. Now a `ServerSettings` (`projection_n_neighbors`), plumbed through the route. |
| `min_dist` | `0.1` | Now a `ServerSettings` (`projection_min_dist`), plumbed through. |
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
| Singleton cell shape | inscribed disc, hex otherwise | `traceCellPath`, `hex-render.util.ts` |
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

_(empty — fill in when the pass is run.)_

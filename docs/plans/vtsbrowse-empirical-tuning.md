# Plan: VTSBrowse empirical tuning pass

> **Status:** Planned, not started. This is the deferred *§Open problems →
> Empirical* work from `docs/plans/vtsbrowse.md`. It is written to be picked
> up on a **stronger environment** (one with a browser for visual judgment,
> and ideally GPU + real demo datasets downloaded) because the core
> deliverable — choosing good defaults — requires *looking at the rendered
> canvas*, which the headless cloud container can't do (no Chrome/Chromium).
>
> The other VTSBrowse follow-ups (sibling highlighting, text-seeded
> navigation, detector handoff) have been **cut** — see *§Open follow-ups* in
> the design doc. Empirical tuning is the remaining v1 polish item.

## Goal

VTSBrowse shipped with **defensible-but-unvalidated defaults** for the UMAP
fit, the hex-tile pyramid, and the canvas renderer. The design doc
(*§Open problems*) deliberately left these as tunable parameters rather than
baking in constants, on the explicit understanding that good values can only
be chosen by running the pipeline on real datasets and judging the output.

This pass:

1. Makes the parameters that are currently hardcoded **reachable without
   editing code** (server settings or build-route params) so a tuning loop
   doesn't require a recompile per trial.
2. Runs a **quantitative sweep** (headless, scriptable) over demo datasets ×
   parameter grid, capturing build cost and layout/aggregation metrics.
3. Runs a **qualitative review** (browser, requires a display) of the rendered
   canvas to pick final defaults for the knobs that only a human eye can
   settle.
4. Lands the chosen defaults (and any newly-exposed settings) with a short
   write-up of *why* each value was picked, so the next person doesn't re-open
   settled questions.

## Why a stronger environment

| Step | Needs a browser? | Needs GPU? | Can do headless here? |
|------|------------------|-----------|----------------------|
| Expose knobs as settings/params (impl) | no | no | **yes** |
| Quantitative sweep harness (impl + run) | no | helps (faster embed) | **yes** (small N) |
| Visual layout / hex-readability review | **yes** | no | no |
| Hover-preview feel (debounce, audio loop) | **yes** | no | no |
| Final default selection + write-up | partly | no | partly |

The impl steps (expose knobs, write the sweep harness) are environment-neutral
and could even be done first in the cloud container. The *judgment* steps need
a real display.

## Current defaults and where each knob lives

The knobs, their current values, and their source of truth as of this plan:

### UMAP (Stage 1) — `vtscore/projection/umap_projection.py:fit_projection`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_neighbors` | `15` | Clamped to `N-1`. **Not plumbed** from the route — see *§Gap*. |
| `min_dist` | `0.1` | **Not plumbed** from the route. |
| `min_n_for_umap` | `10` | Below this → PCA-2 fallback. |
| `random_state` | `None` (unseeded) | Locked decision — keep unseeded in prod. Seed only inside the sweep harness for run-to-run comparability. |
| `metric` | `"euclidean"` | **Settled** (ingest-normalized vectors); not a tuning target. |

### Pyramid (Stage 2) — `vtscore/projection/pyramid.py:build_pyramid`

| Knob | Current default | Notes |
|------|-----------------|-------|
| `n_levels` | **auto-depth** (`None`) | Default `None` makes `build_pyramid` descend until every occupied hex holds a single clip (one-clip-per-hex at deepest zoom), stopping early on co-located clips. Pass an int for fixed depth. |
| `max_useful_levels(N)` | `1 + ceil(log2 N)`, clamped `[1, 14]` | Runaway-guard **ceiling** for the auto-depth descent only — not the operating depth. Replaced the old `log4(N / base_cols²)` heuristic, which assumed uniform 2-D fill and bottomed out ~1 level short of single clips for clustered embeddings (e.g. ~3 levels / ~5 clips-per-hex for the 245-clip ESC-50 demo). |
| `base_cols` | `6.0` | Level-0 spans ~6 hex columns across the larger extent. Drives `base_radius`. |
| `base_radius` | derived (`None` → `_base_radius_for`) | Override path exists but unused. |
| `tile_span` | `16.0` | Hex columns/rows per tile. Payload-size vs round-trip-count tradeoff. |

`build_pyramid(proj)` is called with **no** `n_levels` (auto-depth) in
`vtsearch/routes/projection.py:131` and `vtscore/datasets/load_pipeline.py:1104`;
`base_cols`/`tile_span` use the function defaults. Auto-depth resolves the
former `n_levels` tuning question — the descent self-terminates at single-clip
resolution rather than relying on a closed-form estimate.

### Canvas renderer (Stage 3) — `frontend/src/app/components/browse-canvas/browse-canvas.component.ts`

| Knob | Current value | Location |
|------|---------------|----------|
| Target on-screen hex radius (level picker) | `28` px | `updateActiveLevel`, `:163` |
| Density scale | log (`log(count)/log(maxCount)`) | `drawHex`, `:284` |
| Colormap | darkred→yellow, 8-stop LUT (black left free for "None"/empty space) | `HEATMAP`, `hex-render.util.ts` |
| Singleton cell shape | inscribed disc (`HEX_INRADIUS_RATIO`), hex otherwise | `traceCellPath`, `hex-render.util.ts` |
| Minimap overview level | level whose hexes ≈ `5` px (finer-grained than level 0) | `overviewLevel`, `browse-minimap.component.ts` |
| Hover debounce | `30` ms | `onCanvasMouseMove`, `:445` |
| Hex hit radius | `1 × radius` | `hitTest`, `:490` |

### Hover preview — `frontend/src/app/components/browse-hover-preview/browse-hover-preview.component.ts`

| Knob | Current value | Location |
|------|---------------|----------|
| Audio | `loop=true`, hard-cut on move | `playAudio`, `:93` |
| Text truncation | `300` chars | `loadText`, `:126` |
| Popup offset | `+16 / -8` px from cursor | `show`, `:53` |

## Gap to fix first: `n_neighbors` / `min_dist` are unreachable

The build route (`vtsearch/routes/projection.py:126`) calls `fit_projection`
with **no** `n_neighbors`/`min_dist`, so the two most impactful UMAP knobs are
pinned to the function defaults and can't be tuned without a code edit + the
container being rebuilt (the projection is frozen + persisted, so a stale
container also has to be invalidated). **Step 1 of the impl is to plumb these
through** so the sweep — and any future re-tune — is a config change, not a
patch.

**Decision to make (resolve at impl, ask if non-obvious):** which knobs become
*server settings* (persisted, user-visible) vs *build-route params* (advanced /
internal) vs *stay constants*. A reasonable cut:

- **Server settings** (`ServerSettings` + `CoreConfig`, like `dataset_max_age_days`):
  `projection_n_neighbors`, `projection_min_dist`. These materially change the
  map and a user/operator might want to set them per deployment.
- **Derived, not exposed:** `n_levels` (auto-depth — descends to single-clip
  resolution; `max_useful_levels` is only the guard ceiling), `base_radius`
  (derived from `base_cols`).
- **Constants for now, revisit only if the sweep says so:** `base_cols`,
  `tile_span`, `min_n_for_umap`.

Because the projection is **persisted and frozen**, changing a UMAP setting
must **not** silently reuse an old container's projection. The build route
already guards on the media-id set (`_try_load_persisted`), but it does **not**
key on the UMAP params. **Add the active UMAP params to the persisted
projection's metadata and to the `_try_load_persisted` match**, so flipping a
setting forces a recompute instead of serving a layout fit under the old
params. (This is a real correctness fix the sweep depends on, not just tuning.)

## Phase A — quantitative sweep (headless, scriptable)

A standalone script (suggested: `scripts/projection_sweep.py`, dev-only, not
shipped in the app) that, for each `(dataset, params)` cell:

1. Loads the demo dataset's embedding matrix via
   `get_embedding_matrix(ctx)` (reuse the real ingest path; embeddings are
   already L2-normalized).
2. Runs `fit_projection(..., random_state=SEED)` — **seeded here only**, so
   trials are comparable; production stays unseeded.
3. Runs `build_pyramid` with the trial pyramid params.
4. Emits a row of metrics to CSV/JSON.

**Metrics to capture** (all computable without a browser):

- **Build cost:** UMAP fit seconds, pyramid build seconds, peak RSS.
- **Layout quality (neighborhood preservation):** trustworthiness &
  continuity of the 2-D embedding vs the original space (`sklearn.manifold.
  trustworthiness`); optionally kNN-overlap @k between original and projected
  neighbors. These are the standard quantitative proxies for "did UMAP keep
  the structure?" and let `n_neighbors`/`min_dist` be ranked numerically
  before any eyeballing.
- **Aggregation health per level:** for each pyramid level — number of
  non-empty hexes, count distribution (min/median/p95/max items per hex),
  fraction of single-item hexes, tile fan-out (hexes per tile, tiles per
  level). This tells whether `base_cols`/`tile_span`/`n_levels` produce a
  readable, evenly-loaded pyramid or a few mega-hexes + dust.
- **Small-N behavior:** confirm the PCA-2 / trivial fallbacks trigger at the
  intended `min_n_for_umap` boundary and produce sane bounds.

**Parameter grid (starting point, widen if flat):**

- `n_neighbors ∈ {5, 15, 30, 50, 100}` (clamped to `N-1`)
- `min_dist ∈ {0.0, 0.1, 0.25, 0.5}`
- `base_cols ∈ {4, 6, 9}`
- `tile_span ∈ {8, 16, 32}`

**Datasets** — span media type and N (use the S/M/L/A size variants the demo
registry already provides, e.g. `esc50_s/m/l/a`, `gtzan_a`):

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

## Deliverables

1. **Knob plumbing** (impl, env-neutral): UMAP params reachable via settings +
   build-route, persisted-projection match keyed on UMAP params so a setting
   change forces recompute. Tests in `tests/api/test_projection.py` /
   `tests_lib/projection/`.
2. **Sweep harness** `scripts/projection_sweep.py` + a short README block on
   running it. (Dev tool; keep it out of the shipped package and out of
   `deptry`'s prod-dependency surface — if it needs `sklearn.manifold` etc.,
   confirm those are already deps.)
3. **Chosen defaults** landed in `umap_projection.py` / `pyramid.py` / the
   canvas component / settings, each with a one-line rationale comment or a row
   in the write-up.
4. **Write-up** appended to this plan's *§Results* (below, currently empty):
   the grid that was run, the winning values, and the metric/screenshot
   evidence. Update `docs/plans/vtsbrowse.md` *§Open problems → Empirical* to
   "settled — see plan §Results".

## Acceptance

- `./run-tests.sh` green (the plumbing + persistence-match changes are the only
  shippable code; the sweep script is dev-only and must not break the build,
  `deptry`, or `ruff`).
- New defaults committed with rationale; no hardcoded UMAP params left
  unreachable in the route.
- Design doc *§Open problems → Empirical* marked settled and pointing here.

## Risks & notes

- **Unseeded production fit vs seeded sweep.** Tune with a fixed seed for
  comparability, but the chosen defaults must be robust to the run-to-run
  variation of the *unseeded* production fit — prefer params whose quality is
  stable across a few seeds, not a knife-edge winner.
- **Frozen + persisted projection.** Any container built before the
  params-in-match fix carries a projection fit under old params; re-tuning
  means those must recompute. The persistence-key fix (above) is what makes
  re-tuning safe; without it the sweep results won't reflect the served map.
- **Performance ceiling defines scope, not features.** If the largest target N
  stutters even after tuning, that's the trigger to open the deferred **WebGL
  renderer** follow-up — out of scope for this pass.
- **GPU is a convenience, not a requirement.** It only speeds up embedding the
  demo datasets; the projection/pyramid math is CPU NumPy/numba.

## Results

_(empty — fill in when the pass is run.)_

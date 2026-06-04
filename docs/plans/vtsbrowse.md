# Design: VTSBrowse — a UMAP hexbin Dataset Browser in VTSearch

> **Status:** Prerequisite shipped (app-wide L2 normalization at ingest),
> **Flask-free projection backend** shipped (`vtscore/projection/`: UMAP fit +
> hex-tile pyramid), **Browse routes** shipped
> (`vtsearch/routes/projection.py`: build, meta, tiles endpoints), **the
> Angular browse canvas** shipped (Canvas 2D renderer, pan/zoom, hover
> preview, tile caching, `/browse/:datasetId` route), **projection
> persistence** shipped (inside the dataset container), **ZIP container
> format** shipped (single `.pkl` file containing `medias.pkl` + `meta.json`
> + optional `projection.npz`; all legacy raw-pickle and sidecar support
> removed), **dataset age-off** shipped (server setting
> `dataset_max_age_days`, `expires_at` timestamp on datasets, auto-removal
> on load/list), **and opt-in projection-at-creation** shipped (a "Build
> 2-D Browse projection now" checkbox in every dataset importer that runs
> the UMAP + pyramid build inline as a load stage instead of lazily on
> first Browse visit; defaults off).
> This doc scopes VTSBrowse as a **module within
> VTSearch** — new routes, services, and Angular components that add a Browse
> mode alongside the existing Find and Train modes. See *§What shipped* and
> *§Open follow-ups* at the bottom for where things stand.
>
> **Direction change (this revision):** the original sketch was a
> *hover-to-hear grid* that reused VTSearch's left-panel media-list. The
> Browse experience is now a **UMAP 2-D projection rendered as a
> pannable/zoomable hexbin density map**, where hovering a hex previews a
> representative item from that region. The **frontend and a new projection +
> tiling backend** are the substantive additions, captured in
> *§Browse-canvas architecture* below.
>
> **Name change:** this feature was originally called "VictoryTones" and
> scoped to audio only. It is now **VTSBrowse** and designed to support
> browsing datasets of any MediaType (audio, text, image, video, document).
>
> **Architecture change:** the original design proposed VTSBrowse as a
> *second app tier* (a separate Flask app and Angular build target). It is
> now a **module within VTSearch** — Browse routes under the existing Flask
> app, Browse components in the existing Angular app. VTSearch already has
> "Find" and "Train" as dataset/detector workflows; Browse will be a
> dataset-only workflow added to the same app.
>
> **Decisions locked** (see *§Locked decisions*): single-dataset v1,
> **server-side hex tiling**, **Canvas 2D** rendering, **browse-only** scope,
> and a **projection frozen + persisted at ingest** (unseeded UMAP, computed
> once).

## Problem / Goal

We want a **Dataset Browser** ("VTSBrowse") for quickly exploring a
collection of media by *content*. The center of the app is a **Browse
canvas**: a 2-D UMAP projection of the content-embedding space, drawn as a
tiled field of **hexagons**.

- The canvas is **pannable and zoomable**.
- Each hex's **color encodes how many items fall in that region** (density).
- **Hovering a hex previews a representative ("central") item** from that
  region. The preview behavior depends on the dataset's MediaType:
  - **Audio:** plays the representative clip on repeat until the cursor
    moves to another hex.
  - **Text:** shows the text snippet in a mouse popup/tooltip.
  - **Image:** shows a thumbnail of the image in a mouse popup.
  - **Video:** shows a thumbnail (or a short looping clip) in a mouse popup.
  - **Document:** shows a text excerpt or title in a mouse popup.
- **Zooming in subdivides** the hexes into finer-grained cells over a
  narrower slice of the projection (level-of-detail), so the same screen
  always shows a readable number of hexes but reveals more structure as you
  descend.

It reuses the heavy lifting VTSearch already has:

- the **DatasetImporters** (load media from folders, pickles, archives, ...)
  via the `PluginRegistry`;
- the **MediaEmbedders** for all supported types (LAION-CLAP, CLAP-Music,
  AST, Whisper for audio; SigLIP, CLIP, DINOv2/v3 for images; E5, BGE for
  text; X-CLIP for video) and the cached **embedding matrix** that already
  backs sorting in VTSearch.

On top of that reused stack it adds two new pieces: a **UMAP projection** of
the embedding matrix to 2-D, and a **server-side hex-tile pyramid** that the
canvas streams as the user pans and zooms.

It deliberately **does not** want most of VTSearch's apparatus: no
LabelSets, no labels/votes, no detector training, no MLP, no eval, no
achievements. It is a browser, not a trainable searcher.

## Decision: module within VTSearch (not a separate app)

**VTSBrowse is a module added to VTSearch**, not a standalone app or a
separate app tier. Browse routes live under `vtsearch/routes/`, Browse
Angular components live in `frontend/src/app/`, and the projection backend
stays in `vtscore/projection/` (library tier, Flask-free).

VTSearch already has two dataset/detector workflows — **Find** (text-seeded
or detector-based search) and **Train** (vote on items to train a detector).
Browse adds a third workflow that operates on a **dataset alone** (no
detector required): explore the dataset's embedding space via a 2-D
projection. The Browse canvas replaces the media-list and center panel with
a hex-tile map; the rest of the VTSearch shell (header, dataset picker,
settings) stays.

This is simpler than the original "second app tier" design (no separate
Flask app, no second Angular build target, no separate `vtsbrowse/` package)
and is the right fit because Browse is a mode of VTSearch, not a different
product.

## What VTSBrowse reuses

- **Dataset importers** — `vtscore/datasets/importers/*` (server_folder,
  local_folder, pickle, http_archive, demo, ...) via the `PluginRegistry`.
  Unchanged; any MediaType the importer supports is available to browse.
- **Media types + embedders** — all registered media types and their
  embedders via the media/embedder registries in `vtscore/media/__init__.py`.
  Audio (LAION-CLAP, CLAP-Music, AST, Whisper), image (SigLIP, CLIP,
  DINOv2/v3), text (E5, BGE), video (X-CLIP). Unchanged. Embeddings are
  **L2-normalized at ingest** (see *§Prerequisite*), so the projection
  consumes unit vectors directly regardless of media type.
- **The embedding matrix** — `vtscore/embedding/matrix.py`
  `get_embedding_matrix(ctx) -> (sorted_ids, (N, d) float32)`, lazily built
  and cached on `DatasetContext` (`_emb_matrix` / `_emb_matrix_ids` in
  `vtscore/state/core.py`), invalidated when the media-id set changes. This
  is the **direct input to UMAP**; row `i` <-> `sorted_ids[i]`.
- **Media loading + embedding** — `vtscore/datasets/loader*.py`,
  `vtscore/embedding/{helpers,matrix,loader}.py`, and the
  `load_pipeline -> ingest -> embed_media_bulk` flow.
- **Media serving** — `GET /api/medias/<id>/audio`
  (`vtsearch/routes/media/list.py`) streams WAV bytes for audio; analogous
  endpoints serve image thumbnails, text content, etc. The browse canvas
  uses whichever endpoint matches the dataset's MediaType for hover preview.
- **Clipper system** (audio datasets) — `vtscore/media/audio/clipper.py`
  (tiling / silence / VAD splitters). Each clip is embedded independently
  and carries `clip_start`/`clip_end`, so **clipping directly controls map
  density** (more clips -> more points -> a denser projection).
- **Concurrency/progress** — `vtscore/concurrency/{progress,async_jobs}.py`
  for background dataset loads **and the background UMAP/tiling build**.

## What Browse mode does not use

- **No LabelSets / labels / votes** — no `good_votes`/`bad_votes`, no
  `label_history`, no `LabelSet`, no label importers/exporters.
- **No detectors / training** — no `DetectorContext`, no
  `vtscore/training/*` (MLP, thresholds), no `vtscore/detectors/*`
  workflow, no scoring routes.
- **No eval** — no `vtscore/eval`.
- **No achievements**, no detector/labelset persistence.
- **No left-panel media-list, no center panel** — the Browse canvas
  replaces both. There is no per-item list, no waveform detail, no
  transport. (This supersedes the earlier "reuse the media-list grid"
  sketch.)

## New routes and components

Browse adds routes to the existing VTSearch Flask app and components to the
existing Angular app. The projection backend stays in `vtscore/projection/`
(library tier). v1 is **single-dataset** — Browse operates on whichever
dataset is loaded in the active `DatasetContext`.

### New API surface

- **`POST /api/projection/build`** — kick the background job that computes
  UMAP + the hex-tile pyramid for the loaded dataset; reports progress.
- **`GET /api/projection/meta`** — projection bounds, available zoom
  levels, hex sizing per level, point count, build status, **dataset
  MediaType** (so the client knows which hover-preview behavior to use).
- **`GET /api/projection/tiles/<level>/<tx>/<ty>`** — the hex aggregates
  for one tile at one zoom level (see *§Browse-canvas architecture*).

Existing endpoints (`/api/medias/<id>/audio`, `/api/medias/<id>`, dataset
loading, etc.) are reused as-is. The Browse canvas uses whichever media
endpoint matches the dataset's MediaType for hover preview.

No `/api/order` text-query endpoint in the Browse flow (the projection *is*
the ordering); a text-seeded "fly to this region" affordance is a
follow-up.

## Prerequisite: normalize embeddings at ingest (app-wide)

The projection wants unit vectors, and it turns out VTSearch is *already*
direction-only nearly everywhere — it just normalizes lazily, at every
comparison. We make that canonical: **L2-normalize each embedding once, at
ingest, in every embedder's `embed` / `embed_text`** (audio, image, and the
CLAP/CLIP text-query paths), then **drop the per-comparison normalization**.
This is a VTSearch-wide change, not VTSBrowse-local — done as a
prerequisite so VTSBrowse can consume unit vectors and use plain
Euclidean UMAP.

Why it's safe and arguably overdue:

- **Magnitude is already discarded for all similarity.** The cosine-sort
  path (`vtscore/training/region_similarity.py:135-140` and `:49-75`)
  re-normalizes query and media on every call. Normalizing at ingest just
  moves that work earlier and makes it the stored form. Several embedders
  already normalize at output (BGE/E5 text, DINOv2/v3 patch); this makes the
  rest (CLAP, CLAP-Music, CLIP, SigLIP, and their text-query paths) match.

Two real behavior changes to validate (not just no-ops):

- **Diversity tree** (`vtscore/state/diversity_tree.py`) k-means currently
  clusters on **raw** vectors (Euclidean = magnitude + direction) — the one
  consumer that uses magnitude. Normalizing switches it to **angular**
  clustering, which is the standard, more-correct choice for these
  embeddings and makes it consistent with every other comparison, **but it
  changes diversity ordering** users will observe.
- **MLP** (`vtscore/training/mlp.py`) trains on raw embeddings; unit-norm
  inputs change the input scale (not direction), which shifts **threshold
  calibration**. Generally neutral-to-helpful, but retest convergence and
  thresholds.

**Sites to touch:** add an L2 step to the non-normalizing embedders' `embed`
and `embed_text` (CLAP, CLAP-Music, CLIP, SigLIP, ...); guard against the
zero-vector divide; remove the now-redundant normalization in
`region_similarity.py`; refresh the `svm.py` docstring caveat. **Retest:**
full `./run-tests.sh`, with attention to diversity ordering, MLP
convergence, and threshold calibration. Breaks backwards compatibility of
stored-embedding magnitude (acceptable per repo policy) — call it out in the
PR.

> **Shipped — see *§What shipped*.** Implemented as a single helper,
> `vtscore/embedding/normalize.py:l2_normalize`, applied at the
> `MediaEmbedder` base wrappers (`embed_media`, `embed_media_bulk`, and a new
> `embed_text` -> `_embed_text_impl` indirection so every subclass is covered
> at one layer) **and** at the pickle/re-ingest write paths
> (`loader_pickle._build_pickle_{full,thin}_media`, `ingest._build_media_data`).
> `region_similarity` now scores by plain dot product. The chosen chokepoint
> was "normalize wherever a vector enters `medias` (plus query vectors at the
> embedder)", not the matrix boundary.

**Chokepoint placement (resolve at implementation).** Normalizing *only*
inside `embed`/`embed_text` covers fresh embeds but **not pickle-loaded
datasets**, which write stored (possibly raw, old) vectors straight into
`medias[cid]["embedding"]` without re-embedding. To keep the invariant
"every embedding in `medias` is unit-norm" we need a single ingest
chokepoint that also covers the pickle/import write path — e.g. normalize
wherever an embedding is written into `medias`, or normalize at the
`get_embedding_matrix` boundary and route every similarity consumer through
the matrix. Pick one canonical chokepoint rather than scattering L2 calls
across each embedder; otherwise pickle loads silently bypass it.

## Browse-canvas architecture

This is the heart of VTSBrowse and the part with no precedent in
`vtsearch`. It splits cleanly into a **projection stage** (run once per
dataset at ingest, server-side, then frozen and persisted with the dataset),
a **tile-pyramid stage** (server-side aggregation, persisted alongside the
projection, streamed to the client), and a **canvas renderer** (Canvas 2D,
client-side).

### Stage 1 — UMAP projection (server-side, batch, **computed once at ingest, persisted**)

Run `umap-learn` on the cached `(N, d)` embedding matrix to produce an
`(N, 2)` array of projected coordinates.

- **New dependency:** `umap-learn` (pulls `pynndescent`; `numba` is already
  present transitively via `librosa`, and `scikit-learn` is a direct dep).
- **Metric:** plain **Euclidean**. Because embeddings are L2-normalized at
  ingest (see *§Prerequisite*), Euclidean distance on the unit sphere is a
  monotonic function of cosine distance, so UMAP needs no special metric and
  no per-fit normalization step.
- **Determinism:** **not seeded.** We accept a non-deterministic fit (no
  fixed `random_state`), which keeps UMAP's numba parallelism on and the fit
  faster. This is only acceptable *because the projection is frozen at ingest*
  (below): the layout is computed exactly once and then reused verbatim
  forever, so its non-reproducibility never surfaces — you never see a
  "different map" for the same dataset, because the fit never runs a second
  time. (If we recomputed on every load, unseeded would mean a new layout each
  session; freezing is what buys back stability without paying for a seeded,
  parallelism-disabled fit.)
- **Cost & scheduling:** UMAP is O(N) with heavy constants — seconds at a
  few thousand points, into minutes at 10^5. It runs as a **background
  async job with progress** (`vtscore/concurrency`), never inline in a
  request. The canvas shows a building state until it's ready.
- **Small-N edge cases:** `n_neighbors` must be `< N`; clamp for tiny
  datasets. Decide a minimum-N below which we skip UMAP and fall back to a
  trivial layout (e.g. PCA-2 or a grid).
- **Datasets are immutable input piles for the projection — by design, not
  just v1.** A UMAP fit locks to the exact set of points it was trained on.
  We deliberately **never add to or remove from a loaded dataset**; there is
  no incremental layout and no out-of-sample `umap.transform()`. If you want
  to combine datasets (or otherwise change the pile), you **re-run UMAP from
  scratch** on the new full set, producing a *new* dataset artifact with its
  own frozen projection — the old map is not migrated or transformed. Because
  the pile is immutable, the projection it induces is too: computed once when
  the dataset is ingested, then stored and treated as a fixed property of the
  dataset (like its embeddings), never recomputed for the life of that
  artifact.

> **Carve-out from "No Persisted Vectors or MLPs" — the projection IS
> persisted.** This is a deliberate, scoped exception to the repo rule, and
> it needs to be called out because that rule is otherwise strict. The
> rationale: the rule exists so that embeddings and MLP weights can never
> drift from the active embedder/labels — they are *re-derivable* from
> origins, so caching them on disk only risks staleness. The UMAP projection
> is different in the one way that matters: with an **unseeded** fit it is
> **not reproducible**, so "re-derive on load" does not reproduce the same
> artifact — it produces a *different* map. To deliver "compute once, never
> again" (the locked behavior), the `(N, 2)` coordinates **and** the tile
> pyramid must be stored, not recomputed.
>
> Scope of the exception, to keep it honest:
> - It covers **only** the 2D projection coordinates and the derived hex
>   pyramid — *not* embeddings and *not* MLP weights, which stay in-memory /
>   re-derived exactly as the rule demands.
> - Storage rides with the **dataset artifact** (the pickle is already the
>   sanctioned snapshot of media + embeddings; the projection is the same
>   kind of frozen-at-ingest property). A dataset ingested without a
>   projection (e.g. a legacy `vtsearch` pickle) computes UMAP **once on
>   first open** and persists it back, after which it is never recomputed.
> - It is still **never** written to `settings.json` or to detector/labelset
>   JSON — those remain origin-only.
> - Each artifact carries a `projection_id` (minted at the one-time fit) so a
>   tile can always be checked against the projection it belongs to; because
>   the projection never changes after ingest, this id is effectively a
>   stable per-dataset constant rather than a moving invalidation token.

### Stage 2 — Hex-tile pyramid (server-side aggregation)

Per the locked decision, **the server bins points into hexes and serves
them as tiles**, rather than shipping the raw point cloud for the client to
bin. The server builds a **multi-resolution pyramid** over the 2-D
projection:

- **Zoom levels `z = 0 ... Zmax`.** Level 0 is the coarsest (whole projection
  in a handful of big hexes); each deeper level **halves the hex edge
  length** in projection space, so a fixed screen always shows a comparable
  number of hexes while revealing finer structure as you descend — this is
  the "zoom in -> hexes subdivide" behavior.
- **Hex lattice anchored in projection (data) space**, using axial/cube
  coordinates with nearest-center rounding (the d3-hexbin algorithm,
  reimplemented — no d3 dependency). Anchoring in data space means **pure
  panning never changes hex membership**; only crossing a zoom-level
  boundary re-bins. (The canvas still pans continuously; it just swaps which
  precomputed level it draws as the scale crosses thresholds.)
- **Per hex, the server precomputes:** axial coords (-> center), **count**
  (for the density color), and a **representative media id** = the item
  whose projected point is nearest the hex centroid (this is the item
  hover-previews). Representative selection is per level, so the preview
  naturally generalizes/specializes as you zoom.
- **Tiles** group hexes into a spatial grid per level: a tile is
  `(level, tx, ty)` -> the list of non-empty hexes inside it. This makes the
  payload **viewport-bounded** (the client fetches only the tiles covering
  what's on screen) and **trivially cacheable**: because the projection is
  frozen at ingest, a tile is **immutable for the life of the dataset**, so
  the client (and any CDN) can cache it with a long `max-age` and the
  `projection_id` in the URL — no revalidation, no invalidation logic (see
  *§Tile-cache invalidation*).
- **Build & storage:** the pyramid is computed in the same one-time ingest
  job as the UMAP fit (one O(N * levels) pass after the projection) and
  **persisted with the dataset artifact** alongside the coordinates — it is
  part of the carve-out above, not recomputed on load.

Why server-side tiling (the chosen path) over client-side binning: it keeps
the per-interaction payload bounded by the **viewport**, not by **N**, so
the design scales to very large collections (the cost of binning is paid
once at build time, on the server, and amortized across every pan/zoom and
every client). The price is more machinery (a pyramid + a tile endpoint +
client-side tile caching) and round-trips on zoom/pan, mitigated by an LRU
tile cache and prefetching neighbors.

#### Tile-cache invalidation (a non-problem, by construction)

Freezing the projection at ingest **dissolves** what would otherwise be the
trickiest caching question, so it is worth recording why there is nothing to
solve here. A tile is keyed by `(level, tx, ty)`, and its contents (which
hexes, their counts, their representative items) are only meaningful relative
to one projection. In a world where the projection could be *re-fit* — and
especially with our unseeded fit, where a re-fit is **guaranteed** to move
every coordinate — a cached tile would silently become stale geometry, and
the canvas would render a Frankenstein mix of two layouts. That is the
classic `ETag`/invalidation problem.

We sidestep it entirely: **the projection is computed exactly once and never
re-fit**, so a tile's contents are fixed for the entire life of the dataset
artifact. Mechanically:

- Each artifact has a `projection_id` minted at its one-time fit. It goes in
  the tile URL (e.g. `.../{projection_id}/{level}/{tx}/{ty}`), so tiles
  from different datasets can never collide in any cache.
- Within a dataset that `projection_id` is a **stable constant**, so tiles
  are served with a long-lived `Cache-Control: max-age` (effectively
  immutable). No `If-None-Match` round-trips, no revalidation, no
  version-bumping — the client, an LRU, and any CDN can all cache forever.
- The only event that introduces a *new* `projection_id` is ingesting a
  **different** dataset (a new artifact), which the user reaches by loading
  it — a fresh metadata fetch under a fresh id. There is no in-place rebuild
  of an existing dataset's projection to invalidate against.

So the `projection_id` survives as a clean cache-namespacing key, but the
hard part — invalidating live caches when a projection changes underneath
them — simply does not arise, because projections don't change underneath
anyone.

### Bin shape (hex/square)

Not everyone likes hexagons.  The pyramid can tile the projection as either
**hexagons** (the default d3-hexbin lattice) or **squares**, and the browse
canvas exposes a toggle to switch between them.  The design keeps this cheap by
factoring binning out of the rest of the pipeline:

- **The UMAP layout is shape-independent and computed once.**  Hex and square
  are two *binnings* of the same frozen `(N, 2)` coordinates, so switching only
  re-bins (an O(N·levels) NumPy pass), never re-fits UMAP.  Both binnings carry
  the **same `projection_id`**.
- **`build_pyramid(projection, *, bin_shape=...)`** dispatches the three
  lattice-specific operations — point→cell `assign`, cell `center`, and
  `tile_index` — through a small `_BinGeometry` record selected by `bin_shape`
  (`"hex"` | `"square"`).  The square lattice uses a cell side of `radius·√3`
  (the hex column spacing) so a square and a hexagon at the same pyramid level
  share the renderer's level-of-detail picker and have a comparable on-screen
  footprint.  Every other knob (`base_radius`, `tile_span`, auto-depth) is
  shared.
- **The pyramid records its `bin_shape`** (in the dataclass, in `meta()`, and in
  the persisted JSON), so a loaded projection always knows which lattice it was
  binned with.  Containers written before the toggle have no `bin_shape` and are
  hex by construction, so they default to hex.
- **Persistence is per-shape.**  Each binning is stored in its own ZIP entry —
  `projection.npz` for hex (unchanged legacy name) and `projection_{shape}.npz`
  for the rest — sharing the coordinates (a small duplication next to the tile
  JSON).  Hex and square coexist in one container; appending one leaves the
  other intact.
- **The context caches both.**  `DatasetContext._pyramids` maps `bin_shape →
  Pyramid`, so toggling back and forth within a session is instant after the
  first build of each shape.  Hex is built at ingest (the projection-at-creation
  stage); square is derived lazily on first toggle, then cached and persisted.
- **The HTTP surface is shape-keyed.**  `POST /api/projection/build` takes
  `{"shape": ...}`, `GET /api/projection/meta?shape=...` reports that shape's
  status/metadata (including `bin_shape`), and tiles move to
  `GET /api/projection/tiles/<shape>/<level>/<tx>/<ty>`.  The client tile cache
  keys on shape too, so both binnings stay cached side by side under the one
  shared `projection_id`.
- **Every other control is shape-agnostic.**  Pan, zoom, on-screen cell size,
  the minimap, and hover all run off the shared `radius`/`tile_span` metadata
  and a shared `BinGeometry`, so they work identically in either mode.  Toggling
  preserves the current pan/zoom: because the projection id is unchanged, the
  canvas re-bins in place instead of re-framing to data.

### Stage 3 — Canvas renderer (client-side, **Canvas 2D**)

- **Transform model.** Maintain a projection-space <-> screen-space affine
  transform: **pan = translate**, **zoom = scale about the cursor**. From
  the continuous scale, pick the **nearest discrete pyramid level** to
  fetch; render its hexes through the current transform. Pan within a level
  is a cheap retransform of already-fetched tiles.
- **Tile streaming.** On each viewport change, compute the covering tiles
  for the active level, fetch any missing ones (`GET .../tiles/z/tx/ty`),
  keep them in an **LRU cache**, and prefetch adjacent tiles + the
  neighboring levels for snappy zoom.
- **Drawing.** Raw **Canvas 2D** (matching the existing
  `charts.service.ts` / audio-waveform code — no d3/three/pixi
  dependency). Draw one filled hexagon per non-empty visible hex,
  **viewport-culled**. Canvas 2D comfortably handles low tens-of-thousands
  of on-screen hexes; because the screen-visible hex count is roughly
  constant by construction (level-of-detail), this stays well within
  budget. WebGL is
  the escape hatch if a future requirement blows past it.
- **Density color.** Map `count` -> a perceptually-uniform ramp (viridis /
  magma) on a **log or sqrt scale**, because density is heavy-tailed. Wire
  the ramp endpoints to theme CSS variables as the charts service does.
- **Hover preview.** On hex hover, preview the hex's representative item.
  The preview strategy is **media-type-dependent**:
  - **Audio:** play via `/api/medias/<id>/audio` with `loop = true`;
    **hard-cut/replace** on moving to the next hex; **debounce** so sweeping
    the canvas doesn't machine-gun audio. Browsers block autoplay until a
    user gesture, so the first canvas click **unlocks** the audio element /
    `AudioContext`.
  - **Text:** show the text content (or a truncated snippet) in a tooltip
    anchored to the cursor or hex. No audio playback.
  - **Image:** show a thumbnail of the image in a popup anchored to the
    cursor or hex.
  - **Video:** show a thumbnail (or a short looping preview clip) in a popup.
  - **Document:** show the document title or a text excerpt in a tooltip.
- **Initial framing.** Fit-to-bounds of the projection on first paint, at
  the level whose hex count best fills the viewport.

## Locked decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Dataset scope (v1) | **Single dataset** | No registry / `X-Dataset-Id` headers; one `DatasetContext`. |
| Media type | **Any supported MediaType** | The browser works with whatever MediaType the dataset uses; hover preview adapts per type. |
| Hex binning location | **Server-side tiles** | Pyramid + tile endpoint; viewport-bounded payload; scales to large N. |
| Frontend packaging | **In-app components** | Browse components live in the existing Angular app; no second build target. The Browse canvas replaces the media-list + center panel when Browse mode is active. |
| Renderer | **Canvas 2D** | No new viz dependency; WebGL deferred. |
| Embedding normalization | **App-wide L2 at ingest** | Normalize in every `embed`/`embed_text`; drop per-comparison normalization. See *§Prerequisite*. |
| Canvas point = one item | **Item (clip for audio), not file** | Each point is a media item; for audio, sibling clips of one source file land independently on the map. |
| Hover preview | **Local item only** | Hovering a point previews *that item* only, not its parent file — even when other items from the same source sit elsewhere on the canvas. |
| Product scope (v1) | **Browse-only** | Pan / zoom / hover-to-preview + sibling highlight. No selection, voting, detector training, or ranking in v1; handoff to VTSearch's train-a-detector flow is a deferred follow-up. |
| Hex color | **Density (count)** | Color = item count per hex (log/sqrt-scaled), which the tile pyramid already aggregates for free. No detector/query needed; a second visual channel (opacity/outline) is reserved for hover + sibling highlight. |
| Sibling highlight | **On hover** | Hovering an item highlights the other items from the same source file wherever they fall on the canvas. Requires a per-point source-file group id; see *§Interaction model*. |
| UMAP seeding | **Unseeded (fast)** | No `random_state`; numba parallelism stays on. Safe only because the projection is frozen at ingest, so the fit never re-runs for a given dataset. |
| Projection lifetime | **Frozen at ingest, persisted** | UMAP + pyramid computed once and stored with the dataset artifact (carve-out from "No Persisted Vectors/MLPs"; covers 2D coords + pyramid only). Never recomputed; legacy datasets compute-once on first open. Dissolves tile-cache invalidation. |
| Bin shape | **Hex (default) or square, user-toggleable** | The projection can be tiled as hexagons or squares. The UMAP layout is shape-independent and shared; switching only re-bins the frozen coords (no re-fit). Each binning is its own `Pyramid` (carrying `bin_shape`), cached per-shape on the context and persisted in its own container entry (`projection.npz` for hex, `projection_{shape}.npz` for square). See *§Bin shape (hex/square)*. |

### Notes on frontend integration

Browse components (the hex-tile canvas, hover-preview overlays, projection
build status) are standard Angular components in `frontend/src/app/`. They
share the existing SCSS tokens, services, and build pipeline — no second
build target. When Browse mode is active, the Browse canvas replaces the
media-list and center panel; the VTSearch shell (header, dataset picker,
settings sidebar) stays visible.

## Interaction model (v1)

VTSBrowse v1 is **browse-only**: the only interactions are spatial
navigation (pan/zoom) and two hover behaviors. There is no selection, voting,
detector, or ranking — those belong to the deferred VTSearch handoff (see
*§Open follow-ups*).

- **Hover -> preview.** Hovering a point previews that item (audio plays the
  clip; text/image/video/document show a popup). See *§Stage 3* for the
  per-media-type preview strategy.
- **Hover -> highlight siblings.** Simultaneously, the other items from the
  same source file are highlighted wherever they sit. This needs each rendered
  point to carry a **source-file group id** (derivable from the item's origin
  / source path — items already record their parent), so the renderer can
  light up matching points without a round-trip.
- **Color -> density.** Hex fill encodes item count, taken straight from the
  pyramid's per-hex aggregation; hover/sibling state rides a separate channel
  (outline or opacity bump) so it reads on top of the density fill.

**Binning interaction (resolve at implementation).** Sibling highlight and
per-point hover are only literally per-point at the deepest zoom, where the
renderer draws individual items. At aggregated zoom levels a "point" is a
hex of many items, so both behaviors must degrade gracefully: hovering a hex
should highlight the **hexes that contain** sibling items (not invisible
points), and the preview target becomes a representative item of the hovered
hex (exact pick is part of the empirical hover policy). The tile payload
therefore needs enough per-hex identity to answer "which hexes hold a sibling
of this group?" — either ship source-file group ids down to the hex level, or
serve sibling lookups from a small server endpoint keyed by group id.

## Open problems to resolve before/at scaffold

**Empirical (deferred to experimentation, not design blockers).** These have
no defensible a-priori value; we set them by running UMAP/the pyramid on
real datasets and looking at the output. The design must leave them as
tunable parameters, not bake in constants:

- **UMAP knobs:** `n_neighbors`, `min_dist`, and the small-N fallback
  threshold. (Metric is settled: Euclidean on ingest-normalized vectors.)
- **Pyramid parameters:** `Zmax`, base hex size at level 0, tile size
  (hexes per tile), and the density color scale (log vs sqrt, colormap).
- **Performance ceiling / target N** for v1 — sets whether Canvas 2D
  culling is sufficient or WebGL is needed sooner.
- **Hover preview policy:** debounce window; for audio: loop, hard-cut vs.
  crossfade, volume source, autoplay-unlock gesture; for text/image/video:
  popup size, truncation, fade timing.

**Design (resolve before scaffold).** All resolved — see *§Locked decisions*.
For the record:

- **UMAP determinism** — settled: unseeded (fast), safe because the
  projection is frozen at ingest.
- **Projection lifetime / rebuild** — settled: computed once at ingest and
  persisted with the dataset artifact; never re-fit. Datasets remain
  immutable input piles (combining them produces a new artifact, not an
  in-place rebuild).
- **Tile-cache invalidation** — settled: a non-problem by construction (frozen
  projection => immutable tiles); `projection_id` survives only as a stable
  cache-namespacing key. See *§Tile-cache invalidation*.

## Library-tier placement

The UMAP + tiling code is **Flask-free** and lives in `vtscore/projection/`
(library tier), covered by `./run-tests.sh vtscore-clean`. Browse routes
in `vtsearch/routes/` call into the library tier for projection/pyramid
computation. This keeps the heavy compute code testable without the app
and available to any future consumer of `vtscore`.

## What shipped

- **Prerequisite: app-wide L2 normalization at ingest.** Every embedding is
  now unit-norm at the point it enters `medias`, and every text-query vector
  is unit-norm at the embedder, so all similarity is a plain dot product.
  - New leaf helper `vtscore/embedding/normalize.py:l2_normalize` (numpy-only;
    zero / non-finite norms pass through untouched; idempotent).
  - Applied at the `MediaEmbedder` base: `embed_media` and `embed_media_bulk`
    normalize fresh outputs; `embed_text` became a thin wrapper over a new
    `_embed_text_impl` hook (every embedder subclass renamed `embed_text ->
    _embed_text_impl`) so the normalization lives in one place.
  - Applied at the stored-vector write paths so pickle/legacy loads and
    re-ingest-from-origin also hold the invariant:
    `loader_pickle._build_pickle_full_media` / `_build_pickle_thin_media` and
    `ingest._build_media_data`.
  - `vtscore/training/region_similarity.py` dropped per-comparison
    normalization (both the per-media `score_against_query` path and the
    vectorized `cosine_sort_with_boxes` fast path); the zero-query guard
    stays. Contract change: callers must pass a unit-norm `query_vec` (all
    in-tree sources already do).
  - `svm.py` standardize-caveat docstring refreshed.
  - **Behavior changes (per design):** the diversity tree's k-means now
    clusters on unit vectors (angular, not magnitude+direction), so diversity
    ordering shifts; the MLP trains on unit-scale inputs, which can shift
    threshold calibration. **Breaks backwards compatibility** of
    stored-embedding magnitude — legacy pickles re-normalize on load.
  - Tests: `tests_lib/detectors/test_embedding_normalization.py`,
    `tests_lib/datasets/test_pickle_normalization.py`.

- **Projection backend (Stages 1 + 2), Flask-free.** The browse-canvas compute
  core landed under `vtscore/projection/` (library tier — covered by
  `./run-tests.sh vtscore-clean`), with no app/HTTP wiring yet.
  - `umap_projection.py` — `fit_projection(matrix, ids, ...) -> Projection`
    (Stage 1). Plain Euclidean UMAP on the ingest-normalized matrix; **unseeded
    by default** (`random_state=None`) per the locked decision, with an optional
    seed for reproducible/test fits. `n_neighbors` is clamped to `N-1`; below
    `min_n_for_umap` it falls back to a deterministic **PCA-2** layout, and the
    degenerate `N <= 2` / scalar-embedding cases fall back to a trivial layout.
    Each fit mints a `projection_id`. An optional `on_progress` callback matches
    the ingest `(status, message, current, total)` convention.
  - `hexbin.py` — the **d3-hexbin** assignment rule reimplemented vectorized in
    NumPy (no d3 dependency): `hexbin_assign(points, radius) -> (q, r)` integer
    cell keys (`q = round(2*pi)` to keep d3's half-integer column index
    integral) and `hex_center(q, r, radius)` to invert.
  - `pyramid.py` — `build_pyramid(projection, ...) -> Pyramid` (Stage 2). By
    default (**auto-depth**, `n_levels=None`) it descends — each level halving the
    hex radius — until every occupied hex holds a single clip, so the deepest
    level resolves one-clip-per-hex (enabling hover-to-hear of individual
    sounds); it stops early when a halving no longer separates any co-located
    clips, capped by `max_useful_levels()` as a runaway guard. Pass an int
    `n_levels` for fixed depth. Each level aggregates per hex: axial key, center,
    **count** (density), and a **representative media id** (member nearest the
    cell centroid, ties broken to the smaller id). Hexes are grouped into
    `(level, tx, ty)` **tiles**; `Tile.to_payload()` / `Pyramid.meta()` emit
    JSON-friendly dicts for the tile/meta endpoints. Tunable knobs
    (`base_cols`/`base_radius`, `tile_span`) are parameters, not baked constants,
    per *§Open problems*.
  - **New dependency:** `umap-learn` (added to `[project.dependencies]` with the
    `umap-learn -> umap` deptry module-name mapping). Imported lazily so the
    package import never triggers numba's JIT until a fit runs.
  - Tests: `tests_lib/projection/` (new `projection` group/marker) —
    `test_hexbin.py`, `test_umap_projection.py`, `test_pyramid.py`.
  - **Not yet built (next phases):** persistence of the projection + pyramid
    with the dataset artifact (the carve-out).

- **Browse routes (HTTP layer).** The three projection endpoints landed under
  `vtsearch/routes/projection.py` with a `projection_bp` Blueprint:
  - `POST /api/projection/build` — kicks off a background UMAP + pyramid build
    via the `projection_jobs` :class:`JobManager` singleton. Returns immediately
    with a `job_id`; if a projection is already cached on the `DatasetContext`,
    returns `"ready"` without rebuilding. Signature-keyed caching means
    unchanged datasets short-circuit.
  - `GET /api/projection/meta` — returns projection/pyramid metadata (bounds,
    zoom levels, hex sizing, point count, `projection_id`, dataset `media_type`,
    projection `method`) when ready; build progress (job_id, current, total,
    message) when building; `"idle"` when no build has been requested.
  - `GET /api/projection/tiles/<level>/<tx>/<ty>` — serves a tile's hex cells
    as JSON (delegates to `Pyramid.get_tile` + `Tile.to_payload`). Empty tiles
    return `{"cells": []}`. Route uses `signed=True` on `tx`/`ty` converters
    since tile indices can be negative.
  - Schemas live in `vtsearch/schemas/projection.py`.
  - `DatasetContext` gained `_projection` and `_pyramid` cache slots (in-memory
    only; reset per test by `conftest.reset_state`).
  - `projection_jobs` added to `JOB_MANAGERS` so it participates in
    `/api/jobs/active` introspection and `reset_all_async_jobs_for_tests()`.
  - Tests: `tests/api/test_projection.py` (7 tests covering meta, build, tiles).

- **Browse canvas frontend (Stage 3).** Angular components for the Canvas 2D
  hex-tile renderer, coded against the `Tile.to_payload()` / `Pyramid.meta()`
  contracts from the projection backend.
  - `BrowseCanvasComponent` — Canvas 2D renderer with:
    - Affine projection-space ↔ screen-space transform (pan = translate,
      zoom = scale about cursor).
    - Automatic level-of-detail: picks the pyramid zoom level whose hex
      screen radius is ~28px.
    - Viewport culling: only draws hexes inside the visible area.
    - Density colormap: viridis (14-stop LUT), log-scaled count.
    - Hover hit-test: finds the nearest hex within one radius of the cursor.
    - Tile prefetch: neighbors + adjacent zoom levels.
    - ResizeObserver for responsive canvas sizing; devicePixelRatio-aware.
  - `BrowseHoverPreviewComponent` — media-type-dependent hover previews:
    audio playback (looped, hard-cut on move), image/video thumbnails,
    text snippets (fetched via `/api/medias/<id>/paragraph`).
  - `BrowseViewComponent` — routed view with status states (loading,
    building, ready, empty, error), projection build trigger, dataset
    info overlay.
  - `ProjectionApiService` — API calls for
    `/api/projection/{build,meta,tiles}` endpoints.
  - `TileCacheService` — LRU tile cache (512 entries) with in-flight
    request dedup, shareReplay, and neighbor/level prefetching.
  - `browseContextGuard` — dataset-only route guard; Browse requires no
    detector.
  - Route: `/browse/:datasetId` (lazy-loaded `BrowseViewComponent`).
  - App shell integration: `isOnLabelView` recognizes `/browse`;
    incompatible-pair explainer skipped (no detector on browse).
  - `ContextSwitchService` updated to navigate within `/browse` when the
    dataset pulldown changes on the browse route.
  - **Not yet wired:** sibling highlighting (needs source-file group ids
    in the tile payload or a server-side lookup endpoint).

- **Projection persistence (in-container).** The `(N, 2)` projection
  coordinates and the full hex-tile pyramid are persisted inside the
  dataset ZIP container as `projection.npz`, using NumPy's compressed
  `.npz` format (coords and ids as native arrays; all pyramid metadata,
  levels, tiles, and cells as a JSON blob).
  - `vtscore/projection/persistence.py` — serialization helpers
    `_pyramid_to_meta` and `_rebuild_from_npz_arrays`, shared by the
    container module.
  - `vtscore/datasets/container.py` — `append_projection(path, proj, pyr)`
    and `read_projection(path)`.  `allow_pickle=False` on load for safety.
  - `vtsearch/routes/projection.py` — the build route calls
    `_try_load_persisted()` before computing: if a container has a
    projection and its media-id set matches, it is restored instantly.
    After a fresh UMAP + pyramid build, `_persist_projection()` appends
    the result.  Both are best-effort (registry lookup or I/O failures
    log a warning and fall back to recompute/skip).
  - Tests: `tests_lib/projection/test_persistence.py` (round-trip, empty
    projection, negative tile indices, overwrite) and
    `tests/api/test_projection.py::TestProjectionPersistence` (route
    loads from container, skips stale projection, persists after build).

- **ZIP container format.** Dataset files are ZIP containers (the `.pkl`
  extension is unchanged). The container holds `medias.pkl` (the pickled
  media dict) + `meta.json` (embedder, clipper, media_type, name,
  timestamps, `expires_at`) + optional `projection.npz` (appended after
  browse build). **No legacy support:** raw pickle files and sidecar
  files (`.embedder`, `.clipper`, `.projection`) are not supported.
  - `vtscore/datasets/container.py` — `write_container`, `read_container`,
    `append_projection`, `read_projection`, `read_meta`.
  - `export_dataset_to_file()` produces ZIP containers with metadata.
  - `_read_pickle_dataset()` reads only ZIP containers.
  - `read_pkl_embedder/clipper` read from container `meta.json`.
  - Demo dataset cache (`loader_demo.py`) writes ZIP containers.
  - Find endpoint (`_load_find_dataset_medias`, `_load_pkl_for_check`)
    reads via `read_container`.
  - Stage-file endpoint extracts `medias.pkl` from the ZIP for peeking.
  - Tests: `tests_lib/datasets/test_container.py`.

- **Opt-in projection-at-creation.** A "Build 2-D Browse projection now"
  checkbox in the Advanced block of every dataset importer lets the user
  spend the UMAP + hex-tile compute up front (so Browse opens instantly)
  instead of paying for it lazily on first Browse visit. Defaults **off**:
  building the projection is a cost the user explicitly opts into.
  - Frontend: the checkbox lives in the shared `vt-import-advanced`
    component, so it appears uniformly across the generic-form,
    server-folder, local-folder/files, and demo import flows. Its value
    rides each flow's existing submit path (`runImporter` params,
    `import-local-{folder,files}` multipart, `load-file` multipart,
    `load-demo` body) as a `build_projection` `"true"`/`"false"` string.
  - Backend: `build_projection` is a framework-level pass-through key
    (like `clipper` / `embedder` / `dataset_name`), threaded through
    `_run_importer_in_background` → `_run_origin_load_in_background` to a
    new inline `_build_projection_stage`. The stage runs **after** the
    dataset is registered, fits UMAP on the cached embedding matrix,
    builds the hex-tile pyramid, caches both on the `DatasetContext`, and
    persists them into the container via `_persist_projection_to_container`
    (resolving the pkl path through the dataset registry). It is
    **best-effort by contract**: the dataset is already saved and usable
    before it runs, so a failure — or a cancel during the fit — leaves the
    dataset intact and just defers the projection to the lazy Browse-time
    build. Empty / embedding-less datasets are a silent no-op.
  - Tests: `tests/datasets/test_load_projection_stage.py`.

- **Dataset age-off.** Server setting `dataset_max_age_days` (default: None
  = never expire) stamps new datasets with an `expires_at` timestamp in
  both the container `meta.json` and the dataset registry entry.
  - Setting added to `ServerSettings` model, API schemas, `CoreConfig`.
  - `_auto_register_dataset` (load pipeline) computes `expires_at =
    now + max_age_days * 86400` and passes it to both the container and
    the registry.
  - Enforcement: loading an expired dataset returns HTTP 410 and
    auto-unregisters it; listing datasets filters and auto-unregisters
    expired entries.

- **Hex/square bin-shape toggle.** The browse canvas can tile the projection as
  hexagons (default) or squares; a segmented toggle in the top-right control
  cluster switches between them and the choice persists in the
  `browse_bin_shape` server setting.
  - Backend: new `vtscore/projection/squarebin.py` (`squarebin_assign` /
    `square_center`, square side `radius·√3`); `build_pyramid(..., bin_shape=)`
    dispatches assign/center/tile-index through a `_BinGeometry` record and the
    `Pyramid` carries a `bin_shape` field (surfaced in `meta()` and the
    persisted JSON).  `BIN_SHAPES` / `DEFAULT_BIN_SHAPE` exported from
    `vtscore.projection`.
  - Persistence: `container.append_projection` / `read_projection` store each
    shape in its own entry (`projection.npz` for hex, `projection_{shape}.npz`
    otherwise), sharing coords; both shapes coexist in one container.
  - State: `DatasetContext._pyramid` became `_pyramids: dict[bin_shape ->
    Pyramid]`; `clear_medias` empties it.
  - Routes: `build` takes `{"shape"}`, `meta` takes `?shape=`, tiles move to
    `/api/projection/tiles/<shape>/<level>/<tx>/<ty>`.  When the shared layout
    already exists, a new shape is re-binned inline and returned `ready` (no
    UMAP); only the first build of a dataset runs UMAP in the background.
  - Frontend: shared `bin-geometry.ts` (`BinGeometry` for hex/square: grid
    spacing, cell tracing, hit-test predicate) drives both the canvas and the
    minimap; toggling preserves pan/zoom (same `projection_id` → re-bin in
    place); `TileCacheService` keys tiles by shape so both binnings cache side
    by side.
  - Setting: `browse_bin_shape` (`"hex"` | `"square"`, default hex), persisted
    like the minimap prefs (not a Settings-modal widget — driven by the canvas
    toggle).
  - Tests: `tests_lib/projection/test_squarebin.py`, square cases in
    `test_pyramid.py` / `test_persistence.py`, and `TestBinShapeToggle` in
    `tests/api/test_projection.py`.

## Open follow-ups

**Shipped:**
- **Browse a Find run's positives as their own UMAP.** The Find right panel has
  a `Browse` button next to `Export` (find mode only, disabled when the good
  list is empty). It UMAPs *only* the positive items' high-d vectors — not the
  whole-dataset layout filtered to the subset — into a fresh, ephemeral
  projection. Backend: `POST /api/projection/build` takes an optional `ids`
  list; the result lives in `_subset_*` slots on `DatasetContext` and is
  **never persisted**; `meta`/`tiles` take `?subset=1` to serve it alongside
  the full projection. Frontend: `BrowseSubsetService` hands the positive ids
  (the current "good" list) from the Find view to the Browse view, which
  threads subset mode through build/meta/tiles.
  - *Known limitation:* the id handoff is in-memory, so a hard reload of
    `/browse/:datasetId?subset=1` loses the subset and shows a "re-run Find"
    message (the ephemeral projection would have to be recomputed anyway). If
    reload-survival is wanted, persist the subset id list against a token in
    the URL and rebuild from it on load.

**Active:**
- **Dataset pickle size — drop inline `media_bytes` when the media is reachable
  on disk.** The dataset container still bakes a full copy of every audio/image/
  video blob into `medias.pkl` even when the media also carries a `media_path`
  (or lives under an external `*_dir`). For filesystem-backed datasets this is
  the dominant size cost. A save-time "thin if resolvable" mode (mirror of the
  existing load-time `thin=True`) would shrink those containers dramatically, at
  the cost of making the `.pkl` no longer fully self-contained — so it needs a
  portability decision (always inline, never inline, or a per-export flag).
  *Shipped alongside this note:* embeddings now serialize as compact `float32`
  ndarrays instead of Python float lists, and the dataset pickles use protocol 5
  — ~20% smaller containers and a faster load (no per-vector `PyFloat`
  reconstruction). The peek summarizer (`peek_pickle_dataset_summary`) was
  updated to stub numpy reconstruction so it stays cheap on the new format.
- **Empirical tuning pass:** choose validated defaults for the UMAP fit, the
  hex-tile pyramid, and the canvas renderer (the knobs *§Open problems →
  Empirical* deliberately left unset). Planned in detail, ready to execute on
  an environment with a browser (visual layout/hover judgment can't be done in
  the headless cloud container) — see **`docs/plans/vtsbrowse-empirical-tuning.md`**.
- **WebGL renderer** if the Canvas 2D ceiling is hit (a trigger from the tuning
  pass's performance review, not a standalone feature).
- If/when independent distribution of the projection backend is required, open
  a companion plan for **`vtscore` decoupling + PyPI publish**.

**Cut (decided not to pursue):**
- **Sibling highlighting** — *cut.* Would have highlighted hexes containing
  items from the same source file on hover (needs source-file group ids in the
  tile payload or a server-side lookup endpoint). The canvas has the rendering
  path stubbed (hovered-hex stroke in `drawHex`), but the data plumbing will not
  be wired.
- **VTSearch detector handoff** — *cut.* Would have let a user select/vote items
  on the canvas to seed the train-a-detector flow and recolor hexes by detector
  score. Browse stays dataset-only; no bridge into a `DetectorContext`.
- **Text-seeded navigation** — *cut.* Would have added a query box that embeds
  text and pans the canvas to the nearest region. The projection remains the
  only ordering in Browse.

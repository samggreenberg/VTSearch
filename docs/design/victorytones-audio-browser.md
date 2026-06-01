# Design: VictoryTones — a UMAP hexbin Audio Browser on `vtscore`

> **Status:** Proposed (design only; no code yet). This doc scopes
> VictoryTones as a **second app tier** in the VTSearch repo, built on
> the existing `vtscore` library alongside `vtsearch`.
>
> **Direction change (this revision):** the original sketch was a
> *hover-to-hear grid* that reused VTSearch's left-panel media-list. The
> Browse experience is now a **UMAP 2-D projection rendered as a
> pannable/zoomable hexbin density map**, where hovering a hex auditions a
> representative clip from that region. The repo-structure decision and the
> reuse/omit story are unchanged; the **frontend and a new projection +
> tiling backend** are the substantive additions, captured in
> *§Browse-canvas architecture* below.
>
> **Decisions locked** (see *§Locked decisions*): single-dataset v1,
> MediaType fixed to Audio, **server-side hex tiling**, a **second Angular
> build target**, **Canvas 2D** rendering.

## Problem / Goal

We want an **Audio Browser** ("VictoryTones") for quickly auditioning and
exploring a collection of audio by *content*. The center of the app is a
**Browse canvas**: a 2-D UMAP projection of the audio content-embedding
space, drawn as a tiled field of **hexagons**.

- The canvas is **pannable and zoomable**.
- Each hex's **color encodes how many audio clips fall in that region**
  (density).
- **Hovering a hex plays a representative ("central") clip** from that
  region, on repeat, until the cursor moves to another hex.
- **Zooming in subdivides** the hexes into finer-grained cells over a
  narrower slice of the projection (level-of-detail), so the same screen
  always shows a readable number of hexes but reveals more structure as you
  descend.

It reuses the heavy lifting VTSearch already has:

- the **DatasetImporters** (load audio from folders, pickles, archives, …),
  with **MediaType fixed to Audio**;
- the **audio MediaEmbedders** (LAION-CLAP, CLAP-Music, AST, Whisper) and the
  cached **embedding matrix** that already backs sorting in VTSearch.

On top of that reused stack it adds two new pieces: a **UMAP projection** of
the embedding matrix to 2-D, and a **server-side hex-tile pyramid** that the
canvas streams as the user pans and zooms.

It deliberately **does not** want most of VTSearch's apparatus: no
LabelSets, no labels/votes, no detector training, no MLP, no eval, no
achievements. It is a browser, not a trainable searcher.

## Decision: in-repo second app tier (not a separate repo)

**Add `victorytones/` (an app tier) to this repo, consuming the same
`vtscore`.** Do *not* spin up a separate repository that "references
vtscore" — at least not now. Reasoning below.

### Why in-repo wins

1. **The reused code lives in `vtscore` and changes there.** Importers and
   audio embedders are exactly what VictoryTones leans on, and exactly the
   code most likely to evolve. Same-repo keeps both consumers in lockstep
   under one test run (`./run-tests.sh`); a separate repo guarantees
   version skew on the shared surface.
2. **The repo is already "library tier + app tier."** `vtscore` is the
   Flask-free library; `vtsearch` is the Flask app on top of it. Adding a
   *second* app tier that also consumes `vtscore` is the grain of the
   existing architecture — see `docs/ARCHITECTURE.md` §Directory map and
   §Dependency graph — not a fork of it.
3. **The Browse canvas is not in `vtscore` at all.** It's Angular, in the
   app tier (`frontend/`). Either path reuses or re-implements frontend;
   same-repo lets VictoryTones share Angular services and SCSS instead of
   copy-pasting them into another repo.
4. **Dropping labels/detectors is subtractive and trivial in-repo.** Those
   are per-**detector** concerns (`DetectorContext`: votes, training,
   model, threshold). VictoryTones simply never instantiates a
   `DetectorContext`; it uses only `DatasetContext` (`medias`) + the
   embedding matrix + a new projection + media serving. Nothing has to be
   removed — it just isn't wired up.

### The fact that rules out "separate repo referencing vtscore" today

**`vtscore` is not currently a standalone, distributable package.** It is
Flask-free at *import* time (enforced by `./run-tests.sh vtscore-clean`),
but several `vtscore` modules carry **lazy runtime back-imports into
`vtsearch`**:

| `vtscore` module | imports from `vtsearch` (lazily) |
|------------------|----------------------------------|
| `datasets/load_pipeline.py` | `vtsearch.auth`, `vtsearch.state` |
| `datasets/ingest.py` | `vtsearch.state.next_media_id` |
| `detectors/workflow.py`, `media_seeding.py`, `labelset_elements.py` | `vtsearch.state` |
| `labels/sync.py` | `vtsearch.auth`, `vtsearch.achievements` |
| `concurrency/async_jobs.py` | `vtsearch.auth` |
| `exporters/_template.py` | `vtsearch.auth` |
| `cli.py` | `vtsearch.achievements` |
| `embedding/loader.py` | `vtsearch.logging_config` (optional bridge) |

So "a separate repo that *references* vtscore" presumes an artifact that
doesn't exist yet. To make it real you'd have to either vendor the **whole**
VTSearch repo as a git dependency (you inherit `vtsearch` anyway), or first
**sever those back-edges and publish `vtscore` to PyPI** — a real project
that an audio browser does not by itself justify.

Notably, the paths VictoryTones touches most are in that table:
`datasets/load_pipeline.py` (dataset loading) reaches into `vtsearch.auth`
and `vtsearch.state`. So even the in-repo version benefits from tidying
those edges (see *§vtscore back-edges* below), but in-repo it keeps working
as-is because `vtsearch` is present.

### When to revisit (separate repo / published `vtscore`)

Reconsider extraction the day independent distribution becomes a concrete
requirement: a separate release cadence, a separate deploy target, a
different owning team, or external consumers who should `pip install
vtscore`. At that point the right sequence is **decouple + publish
`vtscore` first, then build VictoryTones against the package** — and that
work pays for itself because you'd need it regardless. Until then, in-repo
avoids paying the decoupling tax up front for a benefit you don't yet need.

## What VictoryTones reuses (from `vtscore`)

- **Dataset importers** — `vtscore/datasets/importers/*` (server_folder,
  local_folder, pickle, http_archive, demo, …) via the `PluginRegistry`.
  Unchanged, **but constrained to MediaType Audio** in the UI.
- **Audio media type + embedders** — `vtscore/media/audio/*`
  (`embedder_clap`, `embedder_clap_music`, `embedder_ast`,
  `embedder_whisper`) via the media/embedder registries in
  `vtscore/media/__init__.py`. Unchanged. Output dimensionality `d` is
  **512** (CLAP, CLAP-Music, Whisper) or **768** (AST); embeddings are now
  **L2-normalized at ingest** (see *§Prerequisite*), so the projection
  consumes unit vectors directly.
- **The embedding matrix** — `vtscore/embedding/matrix.py`
  `get_embedding_matrix(ctx) → (sorted_ids, (N, d) float32)`, lazily built
  and cached on `DatasetContext` (`_emb_matrix` / `_emb_matrix_ids` in
  `vtscore/state/core.py`), invalidated when the media-id set changes. This
  is the **direct input to UMAP**; row `i` ↔ `sorted_ids[i]`.
- **Media loading + embedding** — `vtscore/datasets/loader*.py`,
  `vtscore/embedding/{helpers,matrix,loader}.py`, and the
  `load_pipeline → ingest → embed_media_bulk` flow.
- **Audio serving** — `GET /api/medias/<id>/audio`
  (`vtsearch/routes/media/list.py`) streams WAV bytes; reused verbatim for
  hover playback.
- **Clipper system** — `vtscore/media/audio/clipper.py` (tiling / silence /
  VAD splitters). Each clip is embedded independently and carries
  `clip_start`/`clip_end`, so **clipping directly controls map density**
  (more clips → more points → a denser projection).
- **Concurrency/progress** — `vtscore/concurrency/{progress,async_jobs}.py`
  for background dataset loads **and the background UMAP/tiling build**.

## What VictoryTones omits (vs. `vtsearch`)

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

### Package name

The package is `victorytones/` (not `vtvictorytones/`). The `vt` prefix on
`vtscore`/`vtsearch` (and the `vt-` Angular selectors) plausibly derives
from **V**ictory**T**ones itself, so `vtvictorytones` would read as
"VictoryTones VictoryTones." VictoryTones is its own product, so it takes
the brand name directly rather than a `vt`+role name like `vtsearch`. (If
family-grouping under `vt*` ever matters more than the brand, `vtbrowser`
was the runner-up.)

## App-tier shape (`victorytones/`)

A thin Flask app paralleling `vtsearch`, importing only the `vtscore`
slices above. v1 is **single-user** (`DefaultLoginProvider`) and
**single-dataset** — no multi-dataset registry, no `X-Dataset-Id` headers.
Minimum surface:

- `POST /api/datasets/load` (or registry load) — load a dataset via an
  importer / pickle (MediaType Audio), populate the one `DatasetContext`.
  Reuses the existing load pipeline + progress.
- `GET /api/medias/<id>/audio` — stream audio bytes (reused from
  `vtsearch`, audio-only).
- **`POST /api/projection/build`** — kick the background job that computes
  UMAP + the hex-tile pyramid for the loaded dataset; reports progress.
- **`GET /api/projection/meta`** — projection bounds, available zoom
  levels, hex sizing per level, point count, build status.
- **`GET /api/projection/tiles/<level>/<tx>/<ty>`** — the hex aggregates
  for one tile at one zoom level (see *§Browse-canvas architecture*).

No `/api/order` text-query endpoint in v1 (the projection *is* the
ordering); a text-seeded "fly to this region" affordance is a follow-up.

## Prerequisite: normalize embeddings at ingest (app-wide)

The projection wants unit vectors, and it turns out VTSearch is *already*
direction-only nearly everywhere — it just normalizes lazily, at every
comparison. We make that canonical: **L2-normalize each embedding once, at
ingest, in every embedder's `embed` / `embed_text`** (audio, image, and the
CLAP/CLIP text-query paths), then **drop the per-comparison normalization**.
This is a VTSearch-wide change, not VictoryTones-local — done as a
prerequisite so VictoryTones can consume unit vectors and use plain
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
and `embed_text` (CLAP, CLAP-Music, CLIP, SigLIP, …); guard against the
zero-vector divide; remove the now-redundant normalization in
`region_similarity.py`; refresh the `svm.py` docstring caveat. **Retest:**
full `./run-tests.sh`, with attention to diversity ordering, MLP
convergence, and threshold calibration. Breaks backwards compatibility of
stored-embedding magnitude (acceptable per repo policy) — call it out in the
PR.

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

This is the heart of VictoryTones and the part with no precedent in
`vtsearch`. It splits cleanly into a **projection stage** (run once per
dataset, server-side, in memory), a **tile-pyramid stage** (server-side
aggregation the client streams), and a **canvas renderer** (Canvas 2D,
client-side).

### Stage 1 — UMAP projection (server-side, batch, **in-memory only**)

Run `umap-learn` on the cached `(N, d)` embedding matrix to produce an
`(N, 2)` array of projected coordinates.

- **New dependency:** `umap-learn` (pulls `pynndescent`; `numba` is already
  present transitively via `librosa`, and `scikit-learn` is a direct dep).
- **Metric:** plain **Euclidean**. Because embeddings are L2-normalized at
  ingest (see *§Prerequisite*), Euclidean distance on the unit sphere is a
  monotonic function of cosine distance, so UMAP needs no special metric and
  no per-fit normalization step.
- **Determinism:** seed `random_state` so the map is reproducible run to
  run. Tradeoff: a fixed `random_state` disables UMAP's numba parallelism,
  so the fit is slower. The map is the canonical, shareable view, so
  **determinism wins** over fit speed.
- **Cost & scheduling:** UMAP is O(N) with heavy constants — seconds at a
  few thousand points, into minutes at 10⁵. It runs as a **background
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
  scratch** on the new full set — the layout is expected to change wholesale,
  and that's correct. This makes the projection a pure function of (the
  embedding set + UMAP params + seed), which is also why it's safe to treat
  as a recompute-on-load cache.

> **CRITICAL — do not persist the projection.** UMAP coordinates are a
> *derived artifact* of the embeddings, so the repo's **"No Persisted
> Vectors or MLPs"** rule applies in full: the `(N, 2)` array (and the tile
> pyramid below) live **only in memory**, on the `DatasetContext` (a new
> lazy field alongside `_emb_matrix`), and are **recomputed on load**. They
> are never written to `settings.json`, to a detector/dataset JSON, or to
> any other store. The one allowed exception elsewhere — dataset pickles —
> stores *embeddings*, from which the projection is re-derived; it never
> stores the projection itself.

### Stage 2 — Hex-tile pyramid (server-side aggregation)

Per the locked decision, **the server bins points into hexes and serves
them as tiles**, rather than shipping the raw point cloud for the client to
bin. The server builds a **multi-resolution pyramid** over the 2-D
projection:

- **Zoom levels `z = 0 … Zmax`.** Level 0 is the coarsest (whole projection
  in a handful of big hexes); each deeper level **halves the hex edge
  length** in projection space, so a fixed screen always shows a comparable
  number of hexes while revealing finer structure as you descend — this is
  the "zoom in → hexes subdivide" behavior.
- **Hex lattice anchored in projection (data) space**, using axial/cube
  coordinates with nearest-center rounding (the d3-hexbin algorithm,
  reimplemented — no d3 dependency). Anchoring in data space means **pure
  panning never changes hex membership**; only crossing a zoom-level
  boundary re-bins. (The canvas still pans continuously; it just swaps which
  precomputed level it draws as the scale crosses thresholds.)
- **Per hex, the server precomputes:** axial coords (→ center), **count**
  (for the density color), and a **representative media id** = the clip
  whose projected point is nearest the hex centroid (this is the clip
  hover-plays). Representative selection is per level, so the audition clip
  naturally generalizes/specializes as you zoom.
- **Tiles** group hexes into a spatial grid per level: a tile is
  `(level, tx, ty)` → the list of non-empty hexes inside it. This makes the
  payload **viewport-bounded** (the client fetches only the tiles covering
  what's on screen) and **cacheable** (immutable per projection build; an
  `ETag`/version keyed on the build id lets the browser and any CDN cache
  them, and invalidate atomically when the dataset/projection rebuilds).
- **Build:** the pyramid is computed in the same background job as the UMAP
  fit (one O(N · levels) pass after the projection), and held in memory on
  the `DatasetContext` (again, never persisted).

Why server-side tiling (the chosen path) over client-side binning: it keeps
the per-interaction payload bounded by the **viewport**, not by **N**, so
the design scales to very large collections (the cost of binning is paid
once at build time, on the server, and amortized across every pan/zoom and
every client). The price is more machinery (a pyramid + a tile endpoint +
client-side tile caching) and round-trips on zoom/pan, mitigated by an LRU
tile cache and prefetching neighbors.

### Stage 3 — Canvas renderer (client-side, **Canvas 2D**)

- **Transform model.** Maintain a projection-space ↔ screen-space affine
  transform: **pan = translate**, **zoom = scale about the cursor**. From
  the continuous scale, pick the **nearest discrete pyramid level** to
  fetch; render its hexes through the current transform. Pan within a level
  is a cheap retransform of already-fetched tiles.
- **Tile streaming.** On each viewport change, compute the covering tiles
  for the active level, fetch any missing ones (`GET …/tiles/z/tx/ty`),
  keep them in an **LRU cache**, and prefetch adjacent tiles + the
  neighboring levels for snappy zoom.
- **Drawing.** Raw **Canvas 2D** (matching the existing
  `charts.service.ts` / audio-waveform code — no d3/three/pixi
  dependency). Draw one filled hexagon per non-empty visible hex,
  **viewport-culled**. Canvas 2D comfortably handles low tens-of-thousands
  of on-screen hexes; because the screen-visible hex count is roughly
  constant by construction (LOD), this stays well within budget. WebGL is
  the escape hatch if a future requirement blows past it.
- **Density color.** Map `count` → a perceptually-uniform ramp (viridis /
  magma) on a **log or sqrt scale**, because density is heavy-tailed. Wire
  the ramp endpoints to theme CSS variables as the charts service does.
- **Hover-to-hear.** On hex hover, play the hex's representative id via
  `/api/medias/<id>/audio` with `loop = true`; **hard-cut/replace** on
  moving to the next hex; **debounce** so sweeping the canvas doesn't
  machine-gun audio. Browsers block autoplay until a user gesture, so the
  first canvas click **unlocks** the audio element / `AudioContext`.
- **Initial framing.** Fit-to-bounds of the projection on first paint, at
  the level whose hex count best fills the viewport.

## Locked decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Dataset scope (v1) | **Single dataset** | No registry / `X-Dataset-Id` headers; one `DatasetContext`. |
| Media type | **Audio, fixed** | Importer UI constrained to Audio; no media-type picker. |
| Hex binning location | **Server-side tiles** | Pyramid + tile endpoint; viewport-bounded payload; scales to large N. |
| Frontend packaging | **Second Angular build target** | Separate app, its own `outputPath`/`index`/`main`; served by the `victorytones` Flask app. Shares SCSS/services where practical, not the VTSearch shell. |
| Renderer | **Canvas 2D** | No new viz dependency; WebGL deferred. |
| Embedding normalization | **App-wide L2 at ingest** | Normalize in every `embed`/`embed_text`; drop per-comparison normalization. See *§Prerequisite*. |
| Canvas point = one clip | **Clip, not file** | Each point is a clip media item (`clip_start`/`clip_end` from the audio clipper); sibling clips of one source file land independently on the map. |
| Hover audio | **Local clip only** | Hovering a point plays *that clip's* `[clip_start, clip_end]` slice, never the whole source file — even when other clips of the same file sit elsewhere on the canvas. |
| Product scope (v1) | **Browse-only** | Pan / zoom / hover-to-listen + sibling highlight. No selection, voting, detector training, or ranking in v1; handoff to VTSearch's train-a-detector flow is a deferred follow-up. |
| Hex color | **Density (count)** | Color = clip count per hex (log/sqrt-scaled), which the tile pyramid already aggregates for free. No detector/query needed; a second visual channel (opacity/outline) is reserved for hover + sibling highlight. |
| Sibling highlight | **On hover** | Hovering a clip highlights the other clips of the same source file wherever they fall on the canvas. Requires a per-point source-file group id; see *§Interaction model*. |

### Notes on the second Angular build target

`frontend/angular.json` currently defines a single `frontend` application
that builds to `../static` (served by `vtsearch`). VictoryTones adds a
**second project** (e.g. `victorytones`) with its own `browser` entry
(`main.ts`), `index.html`, and `outputPath` (e.g. `../static-vt`, served by
the `victorytones` Flask app), reusing the shared SCSS tokens and any
framework-agnostic services. Two build targets means two `npm run build`
outputs; `run-tests.sh`'s frontend build check must cover both. This is
heavier than a single in-app route but keeps the browser free of the
VTSearch shell (left panel, center panel, detector/dashboard UI), which it
does not use.

## Interaction model (v1)

VictoryTones v1 is **browse-only**: the only interactions are spatial
navigation (pan/zoom) and two hover behaviors. There is no selection, voting,
detector, or ranking — those belong to the deferred VTSearch handoff (see
*§Open follow-ups*).

- **Hover → listen.** Hovering a point plays that clip's
  `[clip_start, clip_end]` slice (locked above).
- **Hover → highlight siblings.** Simultaneously, the other clips of the same
  source file are highlighted wherever they sit. This needs each rendered
  point to carry a **source-file group id** (derivable from the clip's origin
  / source path — clips already record their parent), so the renderer can
  light up matching points without a round-trip.
- **Color → density.** Hex fill encodes clip count, taken straight from the
  pyramid's per-hex aggregation; hover/sibling state rides a separate channel
  (outline or opacity bump) so it reads on top of the density fill.

**Binning interaction (resolve at implementation).** Sibling highlight and
per-point hover are only literally per-point at the deepest zoom, where the
renderer draws individual clips. At aggregated zoom levels a "point" is a
hex of many clips, so both behaviors must degrade gracefully: hovering a hex
should highlight the **hexes that contain** sibling clips (not invisible
points), and the listen target becomes a representative clip of the hovered
hex (exact pick is part of the empirical hover policy). The tile payload
therefore needs enough per-hex identity to answer "which hexes hold a sibling
of this group?" — either ship source-file group ids down to the hex level, or
serve sibling lookups from a small server endpoint keyed by group id.

## Open problems to resolve before/at scaffold

**Empirical (deferred to experimentation, not design blockers).** These have
no defensible a-priori value; we set them by running UMAP/the pyramid on
real audio datasets and looking at the output. The design must leave them as
tunable parameters, not bake in constants:

- **UMAP knobs:** `n_neighbors`, `min_dist`, and the small-N fallback
  threshold. (Metric is settled: Euclidean on ingest-normalized vectors.)
- **Pyramid parameters:** `Zmax`, base hex size at level 0, tile size
  (hexes per tile), and the density color scale (log vs sqrt, colormap).
- **Performance ceiling / target N** for v1 — sets whether Canvas 2D
  culling is sufficient or WebGL is needed sooner.
- **Hover audio policy:** debounce window, loop, hard-cut vs. crossfade,
  volume source, autoplay-unlock gesture.

**Design (resolve before scaffold).**

- **Determinism vs. fit speed:** confirm we accept the seeded (slower) fit.
- **Rebuild semantics:** full refit on dataset change is locked (datasets are
  immutable input piles; no incremental `umap.transform`); the open part is
  the projection-version/`ETag` scheme for tile-cache invalidation.

## `vtscore` back-edges to address

Even in-repo, the dataset-load path VictoryTones depends on
(`datasets/load_pipeline.py`, `datasets/ingest.py`) reaches into
`vtsearch.auth` / `vtsearch.state`. Two acceptable approaches:

1. **Leave as-is for v1.** Because VictoryTones is in the same repo,
   `vtsearch` is importable, so the lazy back-imports resolve. Cheapest;
   defers the cleanup.
2. **Parameterize the seams.** Where `vtscore` reaches back for user
   context (`get_current_user`), id allocation (`next_media_id`), or the
   active context, accept these as injected callables/params instead of
   importing `vtsearch`. This is the same work a future published
   `vtscore` needs, done incrementally and motivated by a real second
   consumer.

Recommendation: start with (1) to ship the browser, but treat each
back-edge VictoryTones actually exercises as a candidate for (2) — it
both de-risks a future extraction and removes a hidden `vtsearch`
dependency from the load path. The UMAP + tiling code itself is new and
should be written **Flask-free** (it belongs in `vtscore`, e.g.
`vtscore/projection/`), so it can live under `./run-tests.sh
vtscore-clean` from day one.

## Open follow-ups

- Nothing shipped yet — this is a proposal. When the first slice lands,
  add a *What shipped* section and update the status header.
- **VTSearch detector handoff:** v1 is browse-only. The deferred feature is
  letting a user select/vote clips on the canvas to seed VTSearch's existing
  train-a-detector flow (and, once a detector exists, optionally recolor hexes
  by detector score instead of density). Adds selection + vote UI and a
  bridge into the detector context; explicitly out of v1.
- **Text-seeded navigation:** a query box that embeds text and flies the
  canvas to the nearest region (reuses `embed_text_query` + cosine). Out of
  v1 scope; the projection is the only ordering in v1.
- **WebGL renderer** if the Canvas 2D ceiling is hit.
- If/when independent distribution is required, open a companion plan for
  **`vtscore` decoupling + PyPI publish** (sever the back-edges in the
  table above) and migrate VictoryTones to depend on the package.

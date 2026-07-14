# Design: VTSBrowse — a UMAP hexbin Dataset Browser in VTSearch

**Status:** Thin open work remains (thin-pickle save mode, empirical tuning,
WebGL escape hatch, compaction fill ceiling); the living design spec follows the
open work as reference for it.

VTSBrowse is a **module within VTSearch** — Browse routes under the existing
Flask app (`vtsearch/routes/projection.py`), Browse components in the existing
Angular app (`frontend/src/app/`), projection backend in `vtscore/projection/`
(library tier, Flask-free). It adds a **dataset-only** workflow (no detector,
no labels/votes, no MLP/eval) alongside Find and Train: a 2-D UMAP projection
of the content-embedding space, drawn as a pannable/zoomable tiled field of
**hexagons** (audio/text) or **squares** (image/video/document), where hovering
a bin previews a representative item. Bin color encodes item count (density);
zooming subdivides bins (level-of-detail). It reuses VTSearch's importers,
embedders, embedding matrix, media serving, clipper, and concurrency stack.

## Open follow-ups

**Active:**

- **Dataset pickle size — drop inline `media_bytes` when the media is reachable
  on disk.** The dataset container still bakes a full copy of every audio/image/
  video blob into `medias.pkl` even when the media also carries a `media_path`
  (or lives under an external `*_dir`). For filesystem-backed datasets this is
  the dominant size cost. A save-time "thin if resolvable" mode (mirror of the
  existing load-time `thin=True`) would shrink those containers dramatically, at
  the cost of making the `.pkl` no longer fully self-contained — so it needs a
  portability decision (always inline, never inline, or a per-export flag).
- **Empirical tuning pass:** choose validated defaults for the UMAP fit, the
  hex-tile pyramid, and the canvas renderer (the knobs *§Empirical knobs*
  deliberately left unset). Planned in detail, ready to execute on an
  environment with a browser (visual layout/hover judgment can't be done in
  the headless cloud container) — see **`docs/plans/vtsbrowse-empirical-tuning.md`**.
- **WebGL renderer** if the Canvas 2D ceiling is hit (a trigger from the tuning
  pass's performance review, not a standalone feature).
- **Compaction fill ceiling (crispness-vs-coverage).** `compact_layout` deliberately
  trades raw canvas coverage for crisp, separable islands. Because it only *translates*
  clusters, it can close the oceans *between* islands but cannot fill the gaps *within*
  a cluster or reach the frame corners — so its grid-fill tops out around ~0.21 on the
  ESC-50 audio benchmark, below what a `min_dist≈0.9` UMAP fit reaches (~0.38). The
  latter wins on fill only by *inflating* blobs (spreading points across more cells),
  which blurs clusters and starts merging neighbouring classes (Procrustes disparity
  ~0.22). Two unexplored levers if more coverage is wanted without the blur: (a) pack
  noise as its own bounded set of micro-units so they fill inter-island gaps (the
  prototype hit ~0.27 this way, but at O(N) units — would need a grid/merge cap to
  scale); (b) a mild anisotropic stretch of the *packed* layout toward the frame
  aspect ratio to claim the corners. Current defaults (`margin_frac=0.15`,
  `attract=0.02`, `iters=400`) were tuned on ESC-50 only — fold into the empirical
  tuning pass once a browser environment is available to judge layouts visually.
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

### Empirical knobs deliberately left unset (feed the tuning pass)

These have no defensible a-priori value; they are set by running UMAP/the
pyramid on real datasets and looking at the output. The design leaves them as
tunable parameters, not baked-in constants:

- **UMAP knobs:** `n_neighbors`, `min_dist`, and the small-N fallback
  threshold. (Metric is settled: Euclidean on ingest-normalized vectors.)
- **Pyramid parameters:** `Zmax`, base hex size at level 0, tile size
  (hexes per tile), and the density color scale (log vs sqrt, colormap).
- **Performance ceiling / target N** for v1 — sets whether Canvas 2D culling is
  sufficient or WebGL is needed sooner.
- **Hover preview policy:** debounce window; for audio: loop, hard-cut vs.
  crossfade, volume source, autoplay-unlock gesture; for text/image/video:
  popup size, truncation, fade timing.

## Deferred sub-items within shipped areas

Pieces of the live feature left open, to pick up alongside the follow-ups above:

- **Canvas selection actions** — selection is tracked (`BrowseSelectionService`)
  but not acted on: export / seed detector / subset projection, plus a minimap
  overlay.
- **Detector-positives browse** — clipped/region positives are embedded
  image-level (not patch-pooled); no server-side TTL for the ephemeral
  `__detpos__<id>` context; preview bytes held in memory for the session.
- **Bin-popup ordering** — true 1-D-UMAP member ordering (currently a Hilbert
  order over the 2-D coords) would need a second fit + a persisted `order` field
  on `Projection`.
- [ ] #2402 — Browse: re-centre a bin's representative after a lopsided partial removal

## Design spec (living)

The completed feature's contract. Kept below the open work because it mostly
documents shipped mechanics; retained because it is the living spec a
future contributor needs when picking up the open follow-ups.

### What Browse reuses / does not use

Reuses: dataset importers (`vtscore/datasets/importers/*`), all media types +
embedders, the cached embedding matrix (`vtscore/embedding/matrix.py`
`get_embedding_matrix(ctx) -> (sorted_ids, (N,d) float32)`, the direct UMAP
input), media-serving endpoints (`/api/medias/<id>/{audio,image}`, ...), the
audio clipper (each clip is a point, so clipping controls density), and
`vtscore/concurrency` for the background build.

Does **not** use: LabelSets/labels/votes, detectors/training/MLP, eval,
achievements, or the left-panel media-list + center panel (the Browse canvas
replaces both). It is a browser, not a trainable searcher.

### API surface

- **`POST /api/projection/build`** — kicks the background UMAP + pyramid job for
  the loaded dataset; reports progress.
- **`GET /api/projection/meta`** — bounds, zoom levels, per-level hex sizing,
  point count, build status, media type, resolved `bin_shape`.
- **`GET /api/projection/tiles/<level>/<tx>/<ty>`** — the bin aggregates for one
  tile at one level.

Shape is derived server-side from the dataset's media type, so no endpoint takes
a `shape` parameter. Existing media endpoints are reused as-is for hover preview.

### Architecture — three stages

**Stage 1 — UMAP projection** (server, batch, computed **once at ingest,
frozen, persisted**). `umap-learn` on the `(N,d)` matrix → `(N,2)`. Metric is
plain **Euclidean** (safe because every embedding is L2-normalized at ingest,
app-wide, so all similarity is a plain dot product). Fit is **unseeded** (numba
parallelism on, faster) — safe
*only because* the projection is frozen at ingest and never re-fit, so its
non-reproducibility never surfaces. Runs as a background async job with
progress. Small-N: clamp `n_neighbors < N`, PCA-2/grid fallback below a
threshold. Datasets are **immutable input piles by design**: no incremental
layout, no out-of-sample `transform()`; combining datasets produces a *new*
artifact with its own frozen projection.

> **Carve-out from "No Persisted Vectors or MLPs."** The 2-D coordinates **and**
> the tile pyramid are persisted (never embeddings or MLP weights), because an
> unseeded fit is not reproducible, so "re-derive on load" would produce a
> *different* map. Storage rides with the dataset artifact (the sanctioned
> snapshot); never in `settings.json` or detector/labelset JSON. Each artifact
> carries a `projection_id` minted at the one-time fit — effectively a stable
> per-dataset constant. A legacy pickle with no projection computes UMAP once on
> first open and persists it back.

**Stage 2 — tile pyramid** (server aggregation, persisted alongside coords).
Zoom levels `z = 0..Zmax`; each deeper level halves the bin edge length in
projection space (screen always shows a comparable bin count). Lattice anchored
in **data space**, so pure panning never re-bins; only crossing a level boundary
re-bins. Per bin: axial coords (→ center), **count** (density color), and a
**representative media id**. Reps are chosen **bottom-up** (`_assign_reps`) so
`reps(z) ⊆ reps(z+1)` — thumbnails persist as you zoom in; the square quadtree
persists reps exactly, hex has a small centroid-nearest fallback minority. Tiles
group bins into a spatial grid per level so the payload is **viewport-bounded**
and — because the projection is frozen — **immutable** for the dataset's life,
served with a long `Cache-Control: max-age` and `projection_id` in the URL. Tile-
cache invalidation is a non-problem by construction: projections never change
underneath anyone; `projection_id` survives only as a cache-namespacing key.

**Bin shape** is a fixed property of the media type, not a user choice
(`bin_shape_for_media_type`): browsable-thumbnail media (image/video/document)
tile as **squares** (pack edge-to-edge, perfectly zoom-persistent quadtree
reps); audio/text tile as **hexes** (density map). The UMAP layout is
shape-independent (both binnings share coords + `projection_id`);
`build_pyramid(projection, *, bin_shape=...)` dispatches the three
lattice-specific ops through a `_BinGeometry` record; the square side is
`radius·√3` so both shapes share the renderer's LOD picker. Persistence is
per-shape (`projection.npz` for hex, `projection_{shape}.npz` otherwise);
`DatasetContext._pyramids` caches per shape.

**Stage 3 — Canvas 2D renderer** (client). Projection↔screen affine transform
(pan = translate, zoom = scale about cursor); pick the nearest discrete pyramid
level from the continuous scale. On each viewport change, fetch covering tiles
(LRU cache, prefetch neighbors + adjacent levels). Draw one filled bin per
non-empty visible cell, viewport-culled — WebGL is the escape hatch past the
Canvas 2D ceiling. Density color = `count` → perceptually-uniform ramp on a
log/sqrt scale, endpoints wired to theme CSS variables. Hover preview is
media-type-dependent: audio plays via `/api/medias/<id>/audio` (loop, hard-cut,
debounce, first-click autoplay unlock); text/image/video/document show a
tooltip/popup. Initial framing fits-to-bounds at the level that best fills the
viewport.

### Locked decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Dataset scope (v1) | **Single dataset** | One `DatasetContext`; no `X-Dataset-Id`. |
| Media type | **Any supported** | Hover preview adapts per type. |
| Bin location | **Server-side tiles** | Viewport-bounded payload; scales to large N. |
| Frontend packaging | **In-app components** | No second build target; canvas replaces media-list + center panel. |
| Renderer | **Canvas 2D** | WebGL deferred to the escape-hatch trigger. |
| Embedding normalization | **App-wide L2 at ingest** | Drop per-comparison normalization. |
| Canvas point | **Item (clip for audio), not file** | Sibling clips land independently. |
| Product scope (v1) | **Browse-only** | Pan/zoom/hover-preview. No voting/training/ranking (handoff cut). |
| Bin color | **Density (count)** | Log/sqrt-scaled; aggregated by the pyramid for free. |
| UMAP seeding | **Unseeded (fast)** | Safe only because the projection is frozen at ingest. |
| Projection lifetime | **Frozen at ingest, persisted** | Carve-out from "No Persisted Vectors/MLPs" (coords + pyramid only). Dissolves tile-cache invalidation. |
| Bin shape | **Fixed by media type** | Square for image/video/document, hex for audio/text; no user toggle. |

### Library-tier placement

UMAP + tiling code is Flask-free in `vtscore/projection/`, covered by
`./run-tests.sh vtscore-clean`. Browse routes in `vtsearch/routes/` call into it.

# `vtscore.projection`

The Flask-free backend of **VTSBrowse**, the browse canvas: take a
dataset's `(N, d)` embedding matrix, reduce it to a frozen 2-D layout,
aggregate that layout into a multi-resolution tile pyramid the canvas
streams while panning and zooming, and letter the map with named
regions.

Nothing here is interactive. The package computes structures; the HTTP
endpoints that serve them, and the persistence that stores them, live in
the VTSearch Browse routes and in `vtscore.datasets.container`.

Related docs: [`embedding.md`](embedding.md) for the matrix this
consumes; [`state.md`](state.md) for the `DatasetContext` fields that
cache a projection and its pyramids; [`concurrency.md`](concurrency.md)
for the background-job runner the builds ride on.

## Contents

**The two stages**

| Module | Concern |
|--------|---------|
| `vtscore/projection/umap_projection.py` | Stage 1 - `fit_projection`, the `Projection` dataclass, `remove_ids` |
| `vtscore/projection/compaction.py` | Stage 1.5 - `compact_layout`, slide clusters together to close the empty oceans |
| `vtscore/projection/pyramid.py` | Stage 2 - `build_pyramid`, `Pyramid` / `Tile` / `HexCell` / `LevelMeta`, `rebin_like`, `tile_member_ids` |
| `vtscore/projection/hexbin.py` | Vectorised d3-style hexagonal binning (no d3 dependency) |
| `vtscore/projection/squarebin.py` | Vectorised square-grid binning |
| `vtscore/projection/persistence.py` | Serialisation helpers shared with the ZIP container |

**The lifecycle** - what turns those stages into a browsable map.

| Module | Concern |
|--------|---------|
| `vtscore/projection/store.py` | Where a layout lives on disk: `pkl_path_for`, `persist_projection`, `load_persisted_layout`, and the `projection_params_match` freshness guard |
| `vtscore/projection/service.py` | The state machine: `build_layout`, `fit_and_install_layout`, the subset fit and cull, and the meta / labels / tile payloads |

**Signposts** - the "street sign" naming layer.

| Module | Concern |
|--------|---------|
| `vtscore/projection/labels.py` | The data contract: `RegionLabel`, `RegionLabelSet`, `make_label_set` |
| `vtscore/projection/signpost_prep.py` | The one entry point every build path calls: texts → fit → cache |
| `vtscore/projection/signpost_texts.py` | The `object_to_text` layer: one cached text per media, provider per media type |
| `vtscore/projection/signpost_captioners.py` | The generative text tier (image VLM, audio captioner) |
| `vtscore/projection/signpost_build.py` | Fit Toponymy over a frozen layout and flatten its topic tree into labels |
| `vtscore/projection/demo_signposts.py` | Ground-truth signposts read from a dataset's hierarchical `category` |
| `vtscore/projection/signpost_serve.py` | The read side: which set to letter a layout with, and the background self-heal of a stale one |

---

## Stage 1: `fit_projection`

```python
def fit_projection(
    matrix: np.ndarray,          # (N, d), from get_embedding_matrix
    ids: list[int],              # ids[i] labels row i
    *,
    n_neighbors: int = PROJECTION_N_NEIGHBORS,
    min_dist: float = PROJECTION_MIN_DIST,
    min_n_for_umap: int = 10,
    random_state: int | None = None,
    compact: bool = True,
    on_progress: ProgressCallback | None = None,
) -> Projection: ...
```

UMAP runs with the plain `"euclidean"` metric and no per-fit
normalisation. That is correct rather than sloppy: embeddings are
L2-normalised at ingest (see [`embedding.md`](embedding.md#normalisation)),
so Euclidean distance on the unit sphere is monotonic in cosine
distance.

**The fit is unseeded by default** (`random_state=None`), which keeps
UMAP's numba parallelism on. That is only safe because a projection is
computed exactly once per dataset and then frozen and persisted - it
never re-runs, so its non-reproducibility never surfaces. Pass an int
for a reproducible fit, at the cost of parallelism; tests do.

Small datasets can't support a neighbour graph (UMAP needs
`n_neighbors < N`), so `n_neighbors` is clamped to `N - 1`, and below
`min_n_for_umap` points the layout falls back to a deterministic PCA-2
- or a trivial layout for `N ≤ 1` - rather than failing.

`umap` is imported lazily, so importing this package never pays numba's
JIT until an actual fit runs.

### `Projection`

| Field | Description |
|-------|-------------|
| `projection_id` | Minted at the one-time fit; tiles derived from this layout are cached against it |
| `ids` / `coords` | `coords[i]` is the 2-D point for media id `ids[i]`; `(N, 2)` float32 |
| `method` | `"umap"`, `"pca"`, or `"trivial"` |
| `n_neighbors` / `min_dist` | The knobs this layout was fit under, stamped so a persisted projection can be invalidated when the settings change. `None` on the fallbacks and on legacy containers |
| `bounds` (property) | `(xmin, ymin, xmax, ymax)`, zeros when empty |

`remove_ids(projection, remove)` returns a new `Projection` without
those ids - it does not re-fit, so the surviving points keep their exact
positions. That is what the Find→Browse flow uses when the user removes
false positives: the layout stays put and the viewport is preserved.

## Stage 1.5: compaction

UMAP leaves wide empty gaps between islands, so a zoom-to-fit on the raw
layout is mostly dead water. `compact_layout` clusters the points and
slides each cluster together **as a rigid body**, preserving its
internal shape exactly - only the between-cluster gaps shrink.

It runs by default (`compact=True`) and only on the UMAP path; the PCA
and trivial fallbacks are too small to be worth packing.

## Stage 2: `build_pyramid`

The canvas can't stream `N` individual points, so the layout is
aggregated into cells, at several zoom levels, grouped into tiles.

| Type | What it is |
|------|-----------|
| `HexCell` | One aggregated cell at one level: lattice keys `(q, r)`, centre `(cx, cy)`, `count` (density), and `rep_id` |
| `Tile` | The non-empty cells inside one `(level, tx, ty)` spatial cell - the unit the tile endpoint serves |
| `LevelMeta` | Per-level sizing: `level`, `radius`, `n_cells` |
| `Pyramid` | The whole thing: `projection_id`, `bounds`, `base_radius`, `tile_span`, `point_count`, `levels`, `tiles`, `bin_shape` |

`rep_id` is the media the canvas draws for that cell: at the deepest
level it is the clip nearest the cell centroid, and coarser levels
inherit a finer level's representative rather than picking a new one, so
a thumbnail doesn't change identity as you zoom out.

Useful methods: `level_radius(level)` (`base_radius / 2**level`),
`get_tile(level, tx, ty)`, `meta()` (the JSON summary the meta endpoint
returns). `max_useful_levels(point_count)` caps how deep it is worth
going. `tile_member_ids(...)` resolves which ids sit in a cell, backed
by a lazily-filled, process-scoped membership cache that is excluded
from equality and never persisted - the frozen coords re-imply it.

### Hex or square?

`BIN_SHAPES` is `("hex", "square")`, and the choice is **per media
type, not a user setting**: `bin_shape_for_media_type(t)` returns
squares for media with browsable thumbnails (image, video, document)
and hexes for the rest (audio, text). A square grid tiles a thumbnail
grid without gaps; a hex lattice packs abstract density better.

`rebin_like(projection, template, *, preserve_reps=True)` builds a
pyramid for a new projection using an existing one's geometry, so two
layouts stay comparable.

---

## Signposts

The naming layer that letters the map - "continents" at level 0, then
"countries", then "states". The canvas shows each sign only while the
zoom is near its level.

`RegionLabel` is the whole contract: a `text` anchored at `(x, y)` in
projection space, tagged with the pyramid `level` it belongs to (which
may be **fractional** - the canvas interpolates visibility on a
continuous axis), plus a `score` used as the de-clutter tiebreak and a
`source` naming which namer produced it. A `RegionLabelSet` carries the
`projection_id` it was fit against, so a set can never be served over a
layout it doesn't belong to; a stale set is inert, never wrong.

Signs are **derived text**: names, 2-D anchors, scalar scores. No
keyphrase vectors or model state persist - the topic model and the
clusterable UMAP are dropped on the floor at the end of the build.

### The pipeline, and why it splits where it does

`signpost_prep` is the single entry point, called by the ingest
projection stage, by the lazy Browse build, and by the Find→Browse
subset build. It splits the work by cost profile:

- **Per-media texts** are the only full-corpus model cost - Toponymy's
  contrastive keyphrase mining reads *every* object's text, not just the
  exemplars it shows the LLM - and they are clustering-independent. So
  they are computed once and **cached on the media dicts**, which means
  a text computed at ingest persists inside the dataset pickle and every
  later browse or subset re-fit reuses it. Cached strings are derived
  text, which the No-Persisted-Vectors rule explicitly allows.
- **Clustering and naming** are layout-scoped and cheap (a ~5-D UMAP
  plus the fit), so they re-run fresh per layout, full or subset.

That is also why a Find→Browse subset **re-fits** its signs rather than
filtering the dataset-level ones: contrastive keyphrases recompute
against the subset's own siblings, which the image study showed beats
filtering - and it stays interactive because the expensive half is
already cached.

### Text providers

Providers are registered per media type. The zero-shot tier matches each
media against a fixed vocabulary by cosine: CLAP against AudioSet-527
for audio, SigLIP against OpenImages-600 for images, top-5 each. The
generative tier (`signpost_captioners`) instead produces free text -
Qwen2.5-VL-3B-Instruct for images, `whisper-small-audio-captioning` for
audio - and is **opt-in per media type** through the
`browse_signpost_captioner` setting. A captioner always wraps the tag
provider as a fallback, so a failed model download or a per-item decode
failure degrades to tags rather than leaving the map blank.

Because the text is per media and already paid for, it is also shown:
`signpost_metadata_entry` hands it to the media type's
`display_metadata`, titled by kind ("AI Caption" / "AI Tags") so it
reads as machine-generated rather than curated truth.

### Two vector spaces, deliberately

`signpost_build` feeds Toponymy two matrices that are allowed to differ:

- **`clusterable_vectors`** - a dedicated ~5-D cosine UMAP of the
  **score** embedder's matrix, the same space the frozen 2-D layout was
  fit in, so a named cluster is a compact region *on the map*. A cluster
  scattered across the layout would put its anchor in the noise.
- **`embedding_vectors`** - the **text-capable** embedder's matrix, the
  space keyphrase strings embed into, so the keyphrase↔cluster alignment
  means something.

For a single cross-modal embedder (CLAP, SigLIP) both are the same
matrix - the configuration the signpost studies validated.

### Degradation, and the one case that isn't silent

Everything is best-effort: a missing prerequisite returns `None` and the
map stays unlettered. The prerequisites are not equal, though. A missing
text-capable embedder, or a media type with no provider, is a routine
data-dependent skip and stays silent. A missing `toponymy` install is
**not**: `scripts/install.sh` installs it unconditionally, so its
absence means a broken environment. Build paths therefore gate on
`require_signposting()`, which logs a one-time error, while the serve
and signature paths use the quiet `signposting_available()` probe, since
`None` there is expected on every poll.

### What the fit swallows, and where it says so

Toponymy narrates naming hiccups through `warnings.warn`, one per topic,
which floods a CLI run. `_fit_topic_layers` therefore suppresses
toponymy-origin warnings for the duration of the fit - but **counts**
them rather than discarding them, and logs one line when the fit
returns: the suppressed count with a per-message breakdown, how many
topics fell back to the literal name `"unnamed"`, and how many
duplicate-name disambiguation passes ran. The line is `debug` on a clean
fit and `warning` as soon as either count is non-zero.

The count is the only signal a broken prompt parse would ever give.
`KeyphraseNamer` reads Toponymy's own prompt layouts with two regexes, so
a library bump can invalidate them silently; when it does, the naming
retry path burns three `wait_random_exponential(4, 10)` sleeps per
colliding cluster and says nothing. Warnings from other modules are
forwarded untouched.

### Demo signposts

`demo_signposts` is the deliberate cheat: instead of clustering,
keyphrases and an LLM, it reads each media's hierarchical `category` - a
`/`-separated path like `"Europe/France/Île-de-France/Paris"` - and
letters the map straight from those ground-truth labels. Each distinct
path prefix becomes one named region (`"Europe"` at depth 0,
`"Europe/France"` at depth 1, …), anchored at the medoid of its
members' points.

It exists so the sign *display* - zoom-band fading, multi-level
hand-off, de-cluttering - can be exercised end to end without the heavy
naming pipeline, and so a demo dataset shipping a path-encoded taxonomy
lights up the moment it is browsed.

---

## Persistence

`persistence.py` holds the serialisation helpers only; the ZIP container
module (`vtscore.datasets.container`) does the actual writing; `store.py`
resolves which container a dataset writes to and judges whether what is
already stored is still fresh; and the Browse routes own nothing but the
HTTP surface.

Storing a projection and its pyramid **is** a carve-out from the
No-Persisted-Vectors rule, and a narrow one: what gets written is the
2-D layout and the aggregated cells, not the `(N, d)` embeddings they
were derived from. The projection is frozen at ingest precisely so it
never re-runs; that is what makes the unseeded fit acceptable, and it is
why `projection_id` exists as a cache key for every tile derived from
it.

---

## Cross-references

- [`embedding.md`](embedding.md) - `get_embedding_matrix` produces the
  input, and `l2_normalize` is why the euclidean metric is correct.
- [`state.md`](state.md) - `DatasetContext._projection` / `_pyramids` /
  `_region_labels` and their subset twins, which `service.py` is the one
  place that drives.
- [`datasets.md`](datasets.md) - the ingest projection stage, and the
  container format that persists the result.

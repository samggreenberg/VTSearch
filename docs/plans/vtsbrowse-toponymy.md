# Design: VTSBrowse Toponymy — street signs for the browse map

> **Status:** Design only — nothing implemented yet. This doc scopes adding
> **named region labels ("street signs")** to the VTSBrowse canvas so a user
> panning a UMAP map of audio (or any media) sees human-readable names on the
> dense regions — "dog barking", "jazz piano", "applause" — instead of an
> anonymous density field. It builds directly on the shipped VTSBrowse
> pipeline (`docs/plans/vtsbrowse.md`): UMAP projection → hex/square pyramid →
> Canvas 2D renderer.
>
> **Key decision (locked with the user):** the label *source* is **pluggable**
> between two backends, selectable per environment, because some deployments
> have an LLM available and some don't:
>
> - **Backend A — zero-shot vocabulary (no LLM, fully local, default).** Match
>   each region's centroid against a vocabulary of candidate text labels using
>   the embedder's *own* shared text space (CLAP/CLIP/SigLIP `embed_text`).
>   This is the CLIP zero-shot-classification trick applied to map regions.
> - **Backend C — multimodal LLM over representative clips (optional).** Send a
>   region's representative items (or their transcripts) to a configured LLM
>   and ask for a short label. Richer, free-form names; requires outbound API
>   access + config; not available unless an endpoint is set.
>
> Both backends sit behind a common `RegionLabeler` interface and are chosen
> via a server setting. The frontend renders whatever labels exist and is
> agnostic to which backend produced them.

## Problem / Goal

VTSBrowse already answers *"where is the structure?"* — UMAP clusters similar
media, the hexbin pyramid draws density. It does **not** answer *"what is each
region?"*. The user has to hover-audition a representative clip per hex to find
out. **Toponymy** = giving names to places on a map. We want a "place" on the
browse map to carry a readable **sign**, shown at an appropriate zoom level-of-
detail, so the map is legible at a glance.

This is purely additive: a dataset-only, read-only enrichment of an existing
frozen projection. No labels/votes/training/detector apparatus is introduced
(consistent with VTSBrowse's browse-only scope).

### What this is NOT

- Not per-hex captions. A "place" is coarser than a hex; signs sit on regions.
- Not a new clustering UI, not interactive re-labeling, not search-by-label
  (those are possible follow-ups; see *§Open follow-ups*).
- Not mobile/responsive (VTSearch is desktop-only).

## What already exists (and what we reuse)

| Piece | Where | Reuse |
|-------|-------|-------|
| Frozen 2-D layout | `vtscore/projection/umap_projection.py` `Projection` | Anchor positions for signs come from here. |
| Hex/square pyramid | `vtscore/projection/pyramid.py` `Pyramid`/`HexCell` | Multi-resolution structure; `rep_id` gives representative items for Backend C. |
| Embedding matrix (in-memory) | `vtscore/embedding/matrix.py` `get_embedding_matrix(ctx)` | Region centroids (Backend A) and representative selection are computed from this **at build time**. |
| Shared text space | `MediaEmbedder.supports_text` / `embed_text` (`vtscore/media/embedder.py:492,678`), `embed_text_query` (`vtscore/embedding/helpers.py:86`) | Backend A scores vocabulary terms against region centroids in the embedder's own space. |
| Whisper ASR | `vtscore/converters/audio2text.py` | Optional transcript input for Backend C on speech datasets. |
| Build job + persistence | `vtsearch/routes/projection.py` (`_start_umap_build`, `_persist_projection`), `vtscore/projection/persistence.py`, `vtscore/datasets/container.py` (`append_projection`/`read_projection`) | Labels are computed inside the same background build and persisted alongside the pyramid. |
| Canvas renderer | `frontend/src/app/components/browse-canvas/browse-canvas.component.ts` | Where signs get drawn with LOD + collision handling. |
| Plugin registry pattern | `vtscore/plugins/__init__.py` `PluginRegistry` | Template for a `RegionLabeler` registry. |
| Server settings | `vtsearch/settings_models.py` `ServerSettings` (Pydantic) | Where the labeler choice + LLM endpoint config live. |

## The two genuinely new pieces

Everything above is reused. Two things do not exist yet:

1. **A notion of a labeled *region* per zoom level** (coarser than a hex).
2. **The signs on screen** (text overlays with LOD + de-clutter).

Plus the pluggable labeling itself. The rest of this doc specs those.

---

## Stage 0 — Define "places": regions per zoom level

The hex pyramid is *geometric* binning, not semantic clustering: one perceptual
blob spans many adjacent hexes, so we can't just label every hex. We need a
coarser, nested decomposition that maps onto zoom levels.

**Chosen approach: a hierarchical region tree, computed once at build time.**

- Run a hierarchy over the embedding matrix (recommended: **recursive
  bisecting k-means** on the high-d embeddings, or agglomerative if N is small;
  HDBSCAN is a candidate but its variable cluster count is harder to map to
  fixed zoom levels — see *§Decisions to lock*). Each tree node is a region.
- Each region records, **in memory only**:
  - `member_ids` (the media ids it contains),
  - `centroid` (mean of member embeddings, L2-normalized) — **never persisted**
    (it is an embedding; see the No-Persisted-Vectors rule),
  - `anchor` = the 2-D coords of the medoid (member nearest the centroid) — a
    plain coordinate, persistable,
  - `level` — the depth at which this region's sign should appear, aligned to a
    pyramid zoom level so signs thin out as you zoom out.
- Map tree depth → pyramid level so the top of the tree (a handful of big
  regions) shows at coarse zoom, leaves at fine zoom. Reuse
  `max_useful_levels(N)` as the depth cap.

The region tree is a transient build-time artifact. **Only the labeled output
survives** (Stage 2): a flat list of `RegionLabel(level, anchor_x, anchor_y,
text, score, source)`. Centroids and member lists are discarded after
labeling, keeping us compliant with the No-Persisted-Vectors rule.

---

## Stage 1 — The pluggable labeler

```python
# vtscore/projection/labeling/base.py  (new package)

@dataclass(frozen=True)
class RegionLabel:
    level: int          # pyramid zoom level this sign belongs to
    anchor_x: float     # placement in projection (data) space
    anchor_y: float
    text: str           # the street sign, e.g. "jazz piano"
    score: float        # confidence (cosine for A; model/heuristic for C)
    source: str         # labeler id that produced it ("zeroshot" | "llm")


class RegionLabeler(ABC):
    """Turns regions (centroids + representative ids) into RegionLabels."""

    @property
    @abstractmethod
    def name(self) -> str: ...          # "zeroshot" | "llm" | ...

    @abstractmethod
    def available(self, *, media_type: str, embedder: MediaEmbedder) -> bool:
        """Can this labeler run for this dataset right now?
        A: embedder.supports_text. C: an LLM endpoint is configured."""

    @abstractmethod
    def label_regions(
        self,
        regions: list[Region],          # in-memory: centroid + member/rep ids + level + anchor
        *,
        ctx: DatasetContext,
        media_type: str,
        embedder: MediaEmbedder,
        on_progress: ProgressCallback | None = None,
    ) -> list[RegionLabel]: ...
```

Discovery mirrors the embedder system: a `PluginRegistry` over
`vtscore/projection/labeling/` keyed on a `LABELER = <instance>` sentinel, so
adding a third backend later is drop-in. A `signature()` per labeler (vocab
hash for A; model id + prompt version for C) feeds cache invalidation (Stage 2).

### Backend A — zero-shot vocabulary (`labeling/zeroshot.py`, default)

1. `available()` ⇔ `embedder.supports_text` (true for CLAP/CLAP-Music,
   ParaSpeechCLAP, CLIP/SigLIP, X-CLIP; **false** for Whisper, AST, DINOv2/v3 —
   for those, A is unavailable and we fall back to C or no signs).
2. Resolve a **vocabulary** for the media type (Stage 1a). Embed each term once
   via `embed_text` (reuses the `embed_text_query` LRU), forming a `(V, d)`
   matrix in the embedder's own space.
3. Per region: `scores = centroid @ vocab_matrix.T`; take argmax. Attach the
   term as `text` and the cosine as `score`. Optionally require a margin/
   threshold; below it, emit no sign (a "mixed" region stays anonymous rather
   than mislabeled).
4. Fully local, deterministic, cheap (one matmul over V terms × R regions).

#### Stage 1a — vocabularies

- Ship a **default vocabulary per media type** as a small data file
  (e.g. `vtscore/projection/labeling/vocab/audio.txt`). For audio, an AudioSet/
  ESC-50-style class list (~300–500 sound classes) is a strong starting point;
  music → genres + instruments; image → an open-vocab noun list; text/document
  → TBD (keyword extraction may beat a fixed vocab — see *§Decisions to lock*).
- Allow a **user override**: a dataset- or server-level custom vocabulary
  (so a bird-call corpus can ship species names). Surfaced as a setting /
  optional uploaded list. Vocab content is hashed into the labeler signature so
  changing it invalidates cached labels.

### Backend C — multimodal LLM over representatives (`labeling/llm.py`, optional)

1. `available()` ⇔ an LLM endpoint is configured (Stage 1b). If not, C never
   appears as a choice.
2. Per region, select a few **representatives** (region medoid + nearest
   members; the pyramid already computes a `rep_id` we can reuse). Cap at ~3–5
   per region to bound cost.
3. Build the model input. Two sub-modes by media type / capability:
   - **Audio-capable LLM:** attach the representative clips directly, prompt
     *"These short audio clips all come from one cluster of a sound map. Reply
     with a 1–4 word label naming what they have in common."*
   - **Text fallback (speech / no audio support):** run the existing Whisper
     converter on the representatives and send transcripts instead.
4. Parse the short label; store as `text`, `source="llm"`, a heuristic
   `score`. Batch across regions to amortize latency; this runs **once** at
   build, never on the request path.

#### Stage 1b — LLM configuration (provider-agnostic)

- Define a thin `LlmCaptioner` seam with **one shipped implementation: an
  OpenAI-compatible HTTP client** (works against many hosted providers and
  local servers — vLLM, llama.cpp, Ollama's OpenAI-compat endpoint — so a
  fully-local LLM deployment is also "has an LLM"). Keep it dependency-light
  (plain `requests`/`httpx`, no new heavy SDK).
- Config split across the two settings tiers / env:
  - `ServerSettings.browse_llm_endpoint` (URL), `browse_llm_model` (model id) —
    non-secret, in `data/settings.json`.
  - **API key via environment variable** (e.g. `VTSEARCH_BROWSE_LLM_API_KEY`) —
    never written to settings/JSON/disk, consistent with secret-handling norms.
- Outbound network: requires the environment's network policy to permit the
  endpoint. Document this; degrade gracefully (no signs / fall back to A) when
  the call fails.

### Choosing a backend

- New server setting `browse_labeler: "none" | "zeroshot" | "llm"`
  (`vtsearch/settings_models.py`, Literal + default).
- **Default = `"zeroshot"`** when the active embedder supports text, else
  `"none"`. `"llm"` is only selectable when an endpoint is configured.
- The build job asks the registry for the chosen labeler; if it's unavailable
  for the dataset (e.g. `zeroshot` on a text-less embedder, or `llm` with no
  endpoint), it falls back: `llm → zeroshot → none`, logging the downgrade.
- The meta endpoint advertises which labelers are *available* for the active
  dataset so the settings UI can disable impossible choices.

---

## Stage 2 — Compute at build, persist strings only

Labeling runs **inside the existing projection build job** (`_start_umap_build`
in `vtsearch/routes/projection.py`), right after `build_pyramid`, because that
is exactly where the in-memory embedding matrix is already in hand. Flow:

```
fit_projection → build_pyramid → build_region_tree(matrix, projection)
              → labeler.label_regions(...) → list[RegionLabel]
              → persist (pyramid + labels + labeler signature)
```

**Persistence** extends the container's projection record
(`vtscore/projection/persistence.py` + `vtscore/datasets/container.py`):

- Add a `labels` block to the projection meta JSON: the `RegionLabel` list
  (all plain scalars/strings) plus a `labeler_signature`
  (`{labeler, vocab_hash | model_id, prompt_version}`).
- **No vectors persisted.** `RegionLabel` carries only text + 2-D anchors +
  scalar score. Centroids/member lists are build-time-only. This is allowed:
  the No-Persisted-Vectors rule forbids embeddings and MLP weights, not derived
  text — same category as the already-persisted pyramid geometry.
- **Invalidation:** labels are valid only while `(projection_id,
  labeler_signature)` matches the current setting. On Browse load, if the
  persisted labels' signature differs from the active labeler/vocab/model, treat
  labels as absent and recompute (mirrors how a never-persisted bin shape is
  re-binned). Switching `browse_labeler` therefore re-labels lazily on next
  visit without touching the (still-valid) frozen layout.
- **Subset projections** (Find→Browse) are ephemeral and never persisted;
  labels for them are computed in-memory and dropped, same as their pyramids.

---

## Stage 3 — API

Two small additions to `vtsearch/routes/projection.py`:

1. **Meta** (`GET /api/projection/meta`) gains:
   - `available_labelers: ["zeroshot", ...]` (capability for this dataset),
   - `labeler: "zeroshot"` (which produced the current labels, or `null`),
   - `has_labels: bool`.
2. **Labels endpoint** (new): `GET /api/projection/labels?shape=&subset=` →
   `{ labels: [{level, x, y, text, score}, ...] }`. Returned whole (label
   counts are tiny vs. tiles — tens to low hundreds), so no tiling needed; the
   client filters by level client-side. Schema in
   `vtsearch/schemas/projection.py`.

A `POST /api/projection/relabel` (recompute with the current setting without
re-fitting UMAP) is a nice-to-have; not required for v1 since switching the
setting triggers lazy recompute on next load.

---

## Stage 4 — Frontend: draw the signs

In `frontend/src/app/components/browse-canvas/browse-canvas.component.ts`:

- Fetch labels once via a new `ProjectionApiService.getLabels(shape, subset)`;
  cache like meta.
- Render `RegionLabel`s whose `level` matches (or brackets) the current LOD
  level, transformed through the same projection→screen affine the hexes use.
- **De-clutter:** greedy collision avoidance (drop/ζfade lower-`score` signs
  that would overlap a higher one); fade signs in/out across zoom transitions so
  coarse names dissolve into finer ones as you descend.
- **Style:** a subtle semi-transparent "sign" pill with a text label, legible
  over the density colormap; respect theme (`BrowseColormap`/theme tokens).
  Desktop-only, no touch sizing.
- A show/hide-signs toggle in the Browse toolbar (user setting), defaulting on
  when labels exist.

Models: extend `frontend/src/app/models/projection.models.ts` with a
`RegionLabelPayload` interface.

---

## Decisions to lock (before coding)

These are the open design choices; each wants a call before implementation:

1. **Region decomposition algorithm.** Recursive bisecting k-means (predictable
   per-level counts, maps cleanly to zoom) vs. HDBSCAN (finds natural clusters,
   variable count, harder LOD mapping) vs. just labeling coarse-level hexes
   (cheapest, least semantic). *Leaning: recursive k-means.*
2. **Text/document datasets.** For text, classic toponymy uses TF-IDF/keyword
   extraction over the items themselves, which may beat a fixed vocab. Do we
   special-case text (keyword labeler) or force the vocab path? *Leaning: a
   third `keywords` labeler later; out of scope for the audio-first v1.*
3. **Default vocabulary for audio.** AudioSet (~500 classes) vs. a smaller
   curated list. Larger = more coverage but noisier argmax.
4. **Sign density / LOD policy.** How many signs on screen at once; one level of
   signs at a time vs. blending two.
5. **LLM transport.** Confirm OpenAI-compatible HTTP is the one shipped client
   (covers hosted + local servers) and that no new SDK dependency is added.

## Phasing

- **Phase 1 (audio-first, no LLM): Stages 0 + 1(A) + 2 + 3 + 4.** Region tree,
  zero-shot CLAP-vocab labeler, persistence, API, canvas signs. Ships a working
  "street signs" experience on any text-capable audio embedder with zero new
  network/infra. This is the recommended first slice.
- **Phase 2: Backend C (LLM) + the labeler switch + config.** Adds
  `labeling/llm.py`, the OpenAI-compatible client, settings, capability
  advertising, and fallback. Lights up only where an LLM endpoint is configured.
- **Phase 3 (follow-ups):** text/document `keywords` labeler; search-by-sign
  (click a sign → seed a Find); user-editable signs; relabel endpoint.

## Testing notes

- Library-tier tests (`tests_lib/projection/`) for: region-tree determinism
  (seed all RNG), zero-shot labeler against a stubbed text-embedder, persistence
  round-trip of `RegionLabel`s, and signature-based invalidation. These must
  stay Flask-free (`./run-tests.sh vtscore-clean`).
- App-tier tests (`tests/`) for the meta/labels endpoints and the
  build→label→persist→reload path; stub the LLM client (no real network).
- Seed every RNG (k-means init, any sampling) per the flaky-test rules.

## Open follow-ups

- (none yet — populated as phases ship)

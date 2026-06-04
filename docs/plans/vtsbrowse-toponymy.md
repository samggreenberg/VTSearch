# Design: VTSBrowse Toponymy — distinguishing street signs for the browse map

> **Status:** Design only — nothing implemented yet. This doc scopes adding
> **named region labels ("street signs")** to the VTSBrowse canvas so a user
> panning a UMAP map of audio (or any media) sees human-readable names on the
> regions. It builds on the shipped VTSBrowse pipeline
> (`docs/plans/vtsbrowse.md`): UMAP projection → hex/square pyramid → Canvas 2D
> renderer.
>
> **Revision history.**
> - *v1* proposed a from-scratch labeler with a fixed-vocabulary argmax. That
>   was flat tagging, not toponymy.
> - *v2* corrected the model to contrastive, hierarchy-aware naming but still
>   hand-rolled the clustering / keyphrase / naming machinery.
> - **v3 (this revision): adopt the Tutte Institute `toponymy` library**
>   (<https://github.com/TutteInstitute/toponymy>, Healy & McInnes) rather than
>   reimplement it. The library *is* the contrastive, hierarchy-aware,
>   LLM-naming pipeline we were describing; it is explicitly pluggable for
>   non-text data. We supply the audio-specific glue and the map-rendering
>   layer. This collapses three of our hand-rolled stages into "configure
>   Toponymy."

## What toponymy actually is (unchanged — the requirement the library meets)

A place-name distinguishes a place from its neighbors. For clusters that means
naming region `01` by what separates it from its **parent** `0` and its
**siblings** `00`, `02` — not by its absolute-top feature (which it usually
*inherited* from the parent and *shares* with siblings). Naming is **top-down**
and **collision-aware**, and the LLM's job is to **invent a distinguishing
axis** a fixed vocabulary can't contain. The `toponymy` library implements
exactly this: balanced multiresolution clustering → contrastive
(`information_weighted`) keyphrase extraction per cluster → central exemplar
selection → an LLM that synthesizes keyphrases + exemplars + sub-topic names
into a concise distinguishing name, layer by layer.

## Using the `toponymy` library

The entry point is:

```python
from toponymy import Toponymy, KeyphraseBuilder
from toponymy.clustering import ToponymyClusterer
from toponymy.llm_wrappers import OpenAINamer  # or Ollama/VLLM/LlamaCpp/Anthropic/...

topic_model = Toponymy(
    llm_wrapper=<a LLMWrapper>,
    text_embedding_model=<TextEmbedderProtocol>,
    clusterer=ToponymyClusterer(),                 # default; multiresolution
    keyphrase_builder=KeyphraseBuilder(object_to_text=<callable>),
    object_description="audio clips",
    corpus_description="<dataset description>",
)
topic_model.fit(objects, embedding_vectors, clusterable_vectors)
```

`fit(objects, embedding_vectors, clusterable_vectors, exemplar_method="central",
keyphrase_method="information_weighted", subtopic_method="central")`:

- **`clusterable_vectors`** → `clusterer.fit_predict(...)` (the multiresolution
  clustering).
- **`embedding_vectors`** → keyphrase/exemplar alignment.
- **`objects`** → `keyphrase_builder.fit_transform(objects)` and exemplar
  display.

### Requirements → how VTSearch meets each

| Toponymy requires | What it's for | How we supply it |
|---|---|---|
| `clusterable_vectors: np.ndarray (n, k)` | The multiresolution clustering (docs say use UMAP/t-SNE here). | **We already have it:** the frozen `Projection.coords` from `vtscore/projection/umap_projection.py`. Reuse the existing browse layout directly. *(Decision: cluster in the 2-D map vs. a separate higher-D UMAP — quality tradeoff; see below.)* |
| `embedding_vectors: np.ndarray (n, d)` | Aligning keyphrases/exemplars to clusters. | **We already have it:** the in-memory CLAP matrix from `vtscore/embedding/matrix.py:get_embedding_matrix(ctx)`. |
| `text_embedding_model: TextEmbedderProtocol` | Embeds keyphrase *strings*; alignment to clusters. | A thin adapter over the active embedder's text branch (`MediaEmbedder.embed_text`, `vtscore/media/embedder.py:678`). For **CLAP this is ideal**: keyphrase strings land in the *same* space as `embedding_vectors`, so cross-modal keyphrase→audio-cluster alignment is meaningful. Requires `embedder.supports_text`. |
| `clusterer` | Hierarchy. | Use the bundled `ToponymyClusterer()` (multiresolution, `fast_hdbscan`-based). **This replaces our hand-rolled region tree.** |
| `keyphrase_builder` | Contrastive keyphrases. | `KeyphraseBuilder(object_to_text=<callable>)` — see the gap below. `information_weighted` (default) does the contrastive selection. **Replaces our hand-rolled evidence extraction.** |
| `llm_wrapper: LLMWrapper` (required) | The actual naming. | Map our `browse_llm_*` settings to a bundled namer (see *§The LLM*). |
| `objects: List[Any]` | Keyphrase source + exemplar display. | Our list of audio media (ids/dicts), paired with `object_to_text`. |

So **two of the three `fit` inputs we already produce** (the projection coords
and the embedding matrix), the clusterer/keyphrase-builder/prompting are
provided, and the contrastive + hierarchical naming we were going to build is
the library's whole point.

### The one real gap: `object_to_text` (audio → a little text)

`KeyphraseBuilder(object_to_text: Callable[[Any], str])` and the exemplar
functions' `object_to_text_function: Callable[[List[Any]], List[str]]` are the
**official non-text hook** (default: identity, i.e. objects-are-strings).
Toponymy needs *some text per object* to (a) mine contrastive keyphrases and
(b) show exemplars to the LLM. Audio has no words; this callable is where we
turn a clip into a short text. In availability order:

1. **CLAP zero-shot tags (general audio, no LLM, no captioner) — primary.** For
   each clip, the top-k vocabulary terms by CLAP similarity, e.g. `"dog,
   barking, animal, outdoors"`. CLAP already gives us this for free; it shrinks
   the dreaded "audio→text" problem to per-clip tagging. Toponymy's
   `information_weighted` keyphrases then do the *contrastive* work of finding
   which of those tags distinguish each cluster, and the LLM names it. The LLM
   never has to caption audio — it names from tags + sibling context.
2. **Whisper transcripts (speech) — when present.** Reuse
   `vtscore/converters/audio2text.py`; transcripts are natural per-clip text.
3. **A dedicated audio-captioning model — optional, heavy.** Richest per-clip
   text; a follow-up, not v1.

This also cleanly generalizes: for image datasets the same hook yields CLIP/
SigLIP zero-shot tags; for text datasets `object_to_text` is the identity and
Toponymy works as designed.

### The LLM

`Toponymy.__init__` **requires** an `llm_wrapper` (no default), and naming
quality comes from it. The library ships wrappers covering every deployment
shape, so we don't write our own client:

- **Self-hosted / OpenAI-compatible:** `OpenAINamer(base_url=..., api_key=...,
  model=...)`, `OllamaNamer(host=...)`, `VLLMNamer(...)`,
  `LlamaCppNamer(model_path=...)`, `HuggingFaceNamer(model=...)` (in-process).
- **Hosted APIs:** `OpenAINamer`, `AnthropicNamer`, `CohereNamer`,
  `GoogleGeminiNamer`, `AzureAINamer` (+ async/batch variants).

The client libraries are **optional extras** of `toponymy` (install only the
one an environment uses), which matches "some environments have an LLM, some
don't."

**No-LLM environments.** The `LLMWrapper` ABC is tiny — two methods,
`_call_llm(prompt, temperature, max_tokens) -> str` and
`_call_llm_with_system_prompt(...) -> str`. We can ship a **`KeyphraseNamer`**:
a trivial in-process `LLMWrapper` whose `_call_llm` returns the cluster's top
`information_weighted` keyphrase instead of calling a model. This gives an
honest no-LLM fallback that **still uses Toponymy's contrastive clustering and
keyphrase machinery** — just without the LLM's phrasing/abstraction. (Or, for a
small local model, point `LlamaCppNamer`/`HuggingFaceNamer` at a GGUF/HF model;
also fully local.) The `browse_labeler` setting therefore becomes a choice of
**namer**, not of pipeline.

### Dependency footprint (a real decision)

`toponymy==0.5.2` (Python ≥3.10; we run 3.11 ✓). Core deps **already in
VTSearch**: numpy, scikit-learn, transformers (→tokenizers), pandas, scipy,
numba (via umap-learn), tqdm. **New transitive deps it adds:** `datasets`
(HuggingFace — the heaviest surprise), `vectorizers`, `fast_hdbscan`,
`apricot-select`, `tenacity`, `httpx`. Plus the per-environment LLM client
extra (`openai`/`anthropic`/`ollama`/…). Moderate but non-trivial; `deptry`
will require adding `toponymy` to `pyproject.toml` dependencies. **This weight
is the main argument against adoption** and is the first decision to lock.

---

## What we still build (the glue + the map layer)

Adopting Toponymy removes the clustering/keyphrase/naming code we'd otherwise
write. What remains is VTSearch-specific:

### G1 — Adapters & config
- `object_to_text` provider (CLAP zero-shot tags / Whisper) — see the gap above.
- `TextEmbedderProtocol` adapter over `MediaEmbedder.embed_text`.
- Namer selection from settings (`browse_labeler` + `browse_llm_*`), including
  the `KeyphraseNamer` no-LLM fallback. Fallback order
  `llm → keyphrase → none`.

### G2 — Run inside the build job, extract a sign list
Run Toponymy in the existing background build (`_start_umap_build` in
`vtsearch/routes/projection.py`), right after `build_pyramid`, where the
embedding matrix and projection are in hand. From Toponymy's fitted topic tree
(`topic_tree.py` / cluster layers) extract, per layer:
- the topic **name** (string),
- the cluster **membership** → compute an **anchor** = the projected coords of
  the cluster medoid (using our frozen layout),
- a **level** = layer index mapped to a pyramid zoom level.

Produce a flat `RegionLabel(level, anchor_x, anchor_y, text, score, source)`
list. *(To verify during implementation: the exact attributes Toponymy exposes
for per-layer cluster membership + names; `topic_tree.py`/`cluster_layer.py`.)*

### G3 — Persist strings only
Extend the projection record (`vtscore/projection/persistence.py`,
`vtscore/datasets/container.py`): add a `labels` block (the `RegionLabel` list
+ a `labeler_signature` = `{namer, object_to_text_mode, model_id,
toponymy_version}`).
- **No vectors persisted** — only text + 2-D anchors + scalar score. Centroids,
  keyphrases, the topic model itself are build-time-only. Allowed: the
  No-Persisted-Vectors rule forbids embeddings/MLP weights, not derived text.
- **Invalidation:** labels valid only while `(projection_id,
  labeler_signature)` matches the active setting; otherwise recompute on next
  Browse load.
- **Subset projections** (Find→Browse): labels in-memory only, never persisted.

### G4 — API
In `vtsearch/routes/projection.py`: meta gains `available_labelers`, `labeler`,
`has_labels`; new `GET /api/projection/labels?shape=&subset=` returns the whole
`RegionLabel` list (tiny — one per topic node). Schema in
`vtsearch/schemas/projection.py`.

### G5 — Frontend signs
In `browse-canvas.component.ts`: fetch labels once; render those whose `level`
matches the current LOD through the existing projection→screen affine; greedy
collision de-clutter; fade across zoom (coarse names dissolve into finer ones —
correct now, since a child topic is a refinement of its parent). Subtle
semi-transparent sign pill, theme-aware, desktop-only; show/hide toolbar
toggle. Add `RegionLabelPayload` to `models/projection.models.ts`.

---

## Decisions to lock (before coding)

1. **Adopt `toponymy` at all** vs. a slim in-house reimplementation. The
   library gives us the correct contrastive/hierarchical pipeline for free; the
   cost is the dependency footprint (esp. `datasets`, `fast_hdbscan`,
   `vectorizers`, `apricot-select`). *Leaning: adopt — reimplementing it well is
   a lot of subtle work.*
2. **`clusterable_vectors`: reuse the frozen 2-D map vs. compute a separate
   higher-D UMAP for clustering.** Toponymy examples often cluster in ~5–10D and
   name with full embeddings; clustering in 2-D is cheaper and keeps signs
   consistent with what's drawn, but is a quality compromise. *Leaning: start
   with the 2-D map; revisit if cluster quality is poor.*
3. **Default `object_to_text` for audio:** CLAP top-k zero-shot tags (and the
   vocabulary + k behind them) vs. Whisper-first for speech. *Leaning: CLAP tags
   as the general default, Whisper when the dataset is speech.*
4. **No-LLM fallback:** ship the `KeyphraseNamer` passthrough vs. require a
   small local model (LlamaCpp/HF). *Leaning: `KeyphraseNamer` — zero infra.*
5. **Async/batch namers** for large datasets (the library has `Async*`/`Batch*`
   variants) — wire later if naming latency matters.

## Phasing

- **Phase 1 (no LLM): Toponymy + `KeyphraseNamer` + CLAP `object_to_text` +
  G1–G5.** A full contrastive, hierarchical, library-backed sign layer with
  zero external infra — names are the top contrastive keyphrase per topic.
- **Phase 2: real LLM namers + the setting switch + config.** Swap
  `KeyphraseNamer` for `OpenAINamer(base_url=...)` / `OllamaNamer` / local
  model, selected by `browse_labeler`/`browse_llm_*`. Everything else
  (clustering, keyphrases, persistence, API, canvas) is unchanged.
- **Phase 3 (follow-ups):** audio-captioning `object_to_text`; image/text
  datasets (the same hooks generalize); search-by-sign; user-editable signs.

## Testing notes

- Library-tier (`tests_lib/projection/`, Flask-free): the `object_to_text`
  provider (CLAP tags, seeded), the `TextEmbedderProtocol` adapter, the
  `KeyphraseNamer`, `RegionLabel` extraction from a fitted topic tree (stub or
  tiny fixture), persistence round-trip, signature invalidation. Toponymy's own
  clustering/keyphrase correctness is the library's responsibility — we test
  *our glue*, mocking the LLM.
- App-tier (`tests/`): meta/labels endpoints; build→fit→persist→reload; LLM
  namer stubbed (no real network).
- Seed every RNG; never hit a real LLM endpoint in tests.
- `deptry` will fail until `toponymy` is added to `pyproject.toml`; the LLM
  client extras stay optional/per-deployment.

## Open follow-ups

- (none yet — populated as phases ship)

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

**Guiding principle: follow Toponymy's examples wherever possible.** Adopt their
example/default configuration rather than inventing our own knobs —
`ToponymyClusterer(min_clusters=4, verbose=...)`, `keyphrase_method=
"information_weighted"`, `exemplar_method="central"`, `metric="cosine"` on the
clustering UMAP, `ENGLISH_STOP_WORDS`, etc. We only diverge where VTSearch
*forces* it (the `object_to_text` hook for audio, the namer selection, and the
map-rendering layer). This keeps our decision surface small and tracks upstream.

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
| `clusterable_vectors: np.ndarray (n, k)` | The multiresolution clustering (docs say use UMAP/t-SNE here). | **A dedicated higher-D UMAP**, computed at build time from the CLAP matrix (`umap.UMAP(n_components≈5, metric="cosine")`, mirroring Toponymy's examples) — *not* the frozen 2-D browse layout. The 2-D layout stays for rendering + sign anchors; clustering gets the richer ~5-D map. Both reductions derive from the same embedding matrix. (Our embeddings are L2-normalized at ingest, so cosine ≡ euclidean here; we keep `cosine` to match their example.) |
| `embedding_vectors: np.ndarray (n, d)` | Aligning keyphrases/exemplars to clusters. | **We already have it:** the in-memory CLAP matrix from `vtscore/embedding/matrix.py:get_embedding_matrix(ctx)`. |
| `text_embedding_model: TextEmbedderProtocol` | Embeds keyphrase *strings*; alignment to clusters. | A thin adapter over the active embedder's text branch (`MediaEmbedder.embed_text`, `vtscore/media/embedder.py:678`). For **CLAP this is ideal**: keyphrase strings land in the *same* space as `embedding_vectors`, so cross-modal keyphrase→audio-cluster alignment is meaningful. Requires `embedder.supports_text`. |
| `clusterer` | Hierarchy. | Use the bundled `ToponymyClusterer()` (multiresolution, `fast_hdbscan`-based). **This replaces our hand-rolled region tree.** |
| `keyphrase_builder` | Contrastive keyphrases. | `KeyphraseBuilder(object_to_text=<callable>)` — see the gap below. `information_weighted` (default) does the contrastive selection. **Replaces our hand-rolled evidence extraction.** |
| `llm_wrapper: LLMWrapper` (required) | The actual naming. | Map our `browse_llm_*` settings to a bundled namer (see *§The LLM*). |
| `objects: List[Any]` | Keyphrase source + exemplar display. | Our list of audio media (ids/dicts), paired with `object_to_text`. |

So `embedding_vectors` we already have (the CLAP matrix), `clusterable_vectors`
is one extra higher-D UMAP fit at build time, the
clusterer/keyphrase-builder/prompting are provided, and the contrastive +
hierarchical naming we were going to build is the library's whole point.

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
embedding matrix and projection are in hand. Fit the dedicated higher-D
clustering UMAP here (one extra reduction from the same matrix), then pass it
as `clusterable_vectors`. From Toponymy's fitted topic tree
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

### G4 — Labeler-selection meta
Once namer selection (G1) exists, projection meta gains `available_labelers`
and `labeler`.

> **Background (what G2 plugs into).** The serving + rendering layers already
> exist, so G2's output has a ready-made sink: build a
> `RegionLabelSet(projection_id, labels)` (`vtscore/projection/labels.py`) and
> assign it to `ctx._region_labels` (or `ctx._subset_region_labels`) at the end
> of the build job. `GET /api/projection/labels?subset=` serves it — only over
> a matching `projection_id` — meta reports `has_labels`, and the canvas
> (`sign-layout.ts` + the `browse_signposts` toolbar toggle) letters the signs
> with zoom-band sizing and de-cluttering. Each `RegionLabel.level` is a
> (possibly fractional) **pyramid zoom level** — 0 = coarsest layer — so G2's
> topic-layer→level mapping should target that scale.

---

## Decisions to lock (before coding)

> **Experimental evidence:** every unresolved decision below was exercised
> end-to-end on the grid (ESC-50 / Speech Commands / Clotho, 4,445 clips, real
> CLAP embeddings, toponymy 0.5.2, no-LLM + Qwen2.5-7B namers) — see
> **`docs/reports/2026-07-12-toponymy-audio-signposts.html`** and the reusable
> framework at `scripts/experiments/toponymy_audio/`.
>
> **Image evidence (2026-07-12):** the image half of Phase 3 was exercised the
> same way (Caltech-101 / Stanford Dogs / Enrico / RVL-CDIP / mixed, 8,874
> images, real SigLIP embeddings, 75 fits incl. Find→Browse subset re-fits) —
> see **`docs/reports/2026-07-12-toponymy-image-signposts.html`** and
> `scripts/experiments/toponymy_image/`. Headline resolutions: image
> `object_to_text` should be an **instructed ~3B VLM captioner** (cached at
> ingest), *not* a fixed tag vocabulary — zero-shot SigLIP tags remain only a
> no-new-models fallback (they collapse on fine-grained subsets: 0–14%
> breed-sign hit on a 10-breed Find result, vs 56–78% for the captioner /
> in-vocab tags; and 93–94% of their emitted vocabulary on documents and
> screenshots is photo-term distractors). Subset re-fits are interactive
> (~16 s for 150 images + ~12 naming calls). Duplicate signs were ~0–3% on
> image maps, so KeyphraseNamer's audio dedup weakness doesn't carry over.
> The LLM namer needs a "preserve exact rare terms" instruction (it rewrote
> exact breed keyphrases into generic names, 76%→50% hit). Caption-text-space
> browsing does NOT reproduce the audio transcript rescue (flat ARI on
> document types) — taxonomy-grade document structure would need a
> layout-aware embedder instead.

1. **Adopt `toponymy` at all** — **RESOLVED: adopt, installed with
   `--no-deps` + explicitly declared real dependencies.** The pipeline works
   (fine-layer signs on ESC-50: ARI 0.87 / purity 0.97 / 91% name–category
   hit; ~3–5 min end-to-end for 2k clips on one GPU). A plain
   `pip install toponymy` is still off the table — toponymy 0.5.2 pins
   `transformers<5.0.0` and a resolver dry-run downgrades the app's
   transformers 5.12→4.57 and huggingface-hub 1.20→0.36 — but all three
   workarounds were **tested end-to-end on the grid (2026-07-12)** in a venv
   mirroring the app (transformers 5.12.1, hf-hub 1.x), rerunning the full
   ESC-50 fit from cached embeddings/texts:

   - **Install `--no-deps` (recommended — tested, passes):** unmodified
     toponymy 0.5.2 imports and runs the *entire* pipeline under
     transformers 5.12.1 — including its own `HuggingFaceNamer` driving
     Qwen2.5-7B through the transformers-5 pipeline API — with an identical
     topic tree (70/34/13), identical names, and timing parity (fit 113 s vs
     108 s on transformers 4.57). The `<5` pin (upstream rationale: a
     sentence-transformers compat concern) is empirically unnecessary for our
     usage. Static analysis agrees: `transformers` is only *used* inside the
     already-guarded HuggingFaceNamer block, `tokenizers` is imported but
     never used, and `datasets` is declared but **never imported anywhere**.
     Integration recipe: pin `toponymy==0.5.2`, install with `--no-deps` in
     `scripts/install.sh`, and declare its actually-used deps in
     `pyproject.toml`: `fast_hdbscan`, `vectorizers`, `apricot-select`,
     `tenacity`, `jinja2` (numpy/pandas/scikit-learn/scipy/numba/tqdm are
     already VTSearch deps; `httpx` arrives via huggingface-hub 1.x;
     `tokenizers` via transformers). Add a smoke test that imports toponymy
     and fits a tiny corpus, since `--no-deps` bypasses resolver protection
     for future toponymy versions; deptry will need `toponymy` in
     dependencies plus the usual per-rule config.
   - **Vendor the core (tested, passes — fallback):** a 5-line patch (drop
     the vestigial top-level `import tokenizers` / `import transformers` in
     `llm_wrappers.py`, wrap `import httpx` in try/except there and in
     `embedding_wrappers.py`) makes the whole package import-clean; full fit
     reproduces the baseline exactly. ~13k LOC to carry if upstream ever
     hard-breaks.
   - **Sidecar venv/process (tested, passes — last resort):** toponymy in its
     own 752 MB venv (its own transformers 4.57.6, **no torch**), embeddings
     and texts passed by file, keyphrase/topic-name embedding served by the
     app's CLAP text branch over localhost HTTP — only **4 round-trips for
     4,352 encoded strings** end-to-end; fit parity (180 s). Working demo:
     `scripts/experiments/toponymy_audio/sidecar/`. Operationally heaviest
     (second venv to build/ship, process lifecycle) — keep as escape hatch.

   Upstream: `main` still pins `<5` and has *added* `litellm` as a hard dep
   (trending heavier), and #150 deliberately made transformers a base dep —
   so don't block on upstream; file an issue offering the transformers-5
   runtime evidence and the unused `datasets`/`tokenizers` finding, and
   re-check before implementation in case a fixed release lands.
2. ~~`clusterable_vectors` source~~ **RESOLVED:** cluster on a **dedicated
   higher-D UMAP** (`n_components≈5`, `metric="cosine"`), not the 2-D browse
   layout. Confirmed empirically (26 s for 2k clips; multiresolution tree
   70→34→13 topics recovers ESC-50's taxonomy without labels).
3. **Default `object_to_text` for audio** — **RESOLVED: CLAP top-5 zero-shot
   tags against the AudioSet-527 vocabulary** (template "The sound of {}"),
   shipping the label list as a data file, with music-genre terms filtered
   for non-music datasets (they cause the worst sign pollution). Evidence
   against alternatives: a small domain-matched vocabulary is *worse* than the
   big generic one (runner-up tags become semantically distant junk); Whisper
   is mandatory for speech datasets and useless elsewhere — and for speech the
   CLAP *map* itself is near-random (ARI 0.05), so the better long-term answer
   there is browsing transcript-text embeddings via the existing `audio2text`
   converter. **Promote the audio captioner** (was Phase 3 "optional, heavy"):
   `MU-NLPC/whisper-small-audio-captioning` is 1 GB, captions 2k clips in
   ~90 s, and produced the cleanest signs on both ESC-50 and uncurated Clotho.
4. **No-LLM fallback** — **RESOLVED: ship `KeyphraseNamer`** (signs are
   genuinely usable, e.g. "sheep goat bleat livestock"), but add a cheap
   sibling-dedup pass before rendering — without an LLM, duplicate sibling
   signs reach 38% on hard maps.
5. **Async/batch namers** for large datasets — still open, but now with a cost
   model: naming ≈ 1 LLM call/topic ≈ 2 topics/s on an a100 with Qwen2.5-7B;
   topic count ≈ n / base_min_cluster_size, so the 20k+ regime needs
   `base_min_cluster_size` scaled up (≈50–100) and/or the async namers.
   Also lock at implementation: raise `lowest_detail_level` (default fine-layer
   prompts request 8–15-word names — too long for map signs) and set
   `llm_specific_instructions` (English-only, ≤6 words, no filler names —
   Qwen produced "日语学习指令" and "Sounds" without them).

## Phasing

- **Phase 1 (no LLM): Toponymy + `KeyphraseNamer` + CLAP `object_to_text` +
  G1–G4.** A full contrastive, hierarchical, library-backed sign layer with
  zero external infra — names are the top contrastive keyphrase per topic.
  (The serving API and the canvas sign layer are already live — see the G4
  background note — so Phase 1's frontend work is done.)
- **Phase 2: real LLM namers + the setting switch + config.** Swap
  `KeyphraseNamer` for `OpenAINamer(base_url=...)` / `OllamaNamer` / local
  model, selected by `browse_labeler`/`browse_llm_*`. Everything else
  (clustering, keyphrases, persistence, API, canvas) is unchanged.
- **Phase 3 (follow-ups):** audio-captioning `object_to_text`; image/text
  datasets; search-by-sign; user-editable signs. For images the
  `object_to_text` question is now **resolved by experiment** (see the image
  evidence note above): default = instructed ~3B VLM captioner
  (Qwen2.5-VL-3B class; prompt states type + subject + key visible text),
  computed once at ingest and cached like thumbnails; fallback = SigLIP
  zero-shot tags vs OpenImages-600 for photo corpora / no-VLM deployments;
  never a curated per-domain label list. Subset (Find→Browse) fits recompute
  from cached texts (~16 s at n=150).

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

## Eval scaffold (shipped): ground-truth signposts + a synthetic demo

Ahead of the real pipeline, a **cheating** signpost path exists so the map
*display* (zoom-band fading, multi-level hand-off, de-clutter) can be evaluated
hands-on:

- **`vtscore/projection/demo_signposts.py`** derives a hierarchical
  `RegionLabelSet` straight from each media's `/`-separated `category` path
  (one sign per distinct prefix; anchor = the region's medoid in the frozen
  layout; level = depth × a fixed zoom step). It's `source="ground-truth"`, no
  clustering/LLM. The projection route builds+caches it lazily in
  `_label_set_for` whenever a browsed dataset has hierarchical categories — so
  Places365-style path taxonomies light up for free too.
- **The `synthetic_world_audio` / `synthetic_world_image` demo datasets**
  (`vtscore/media/_toponymy_demo.py`) ship a hand-authored 4-level geographic
  taxonomy (Continent → Country → State → City, 108 leaf cities) with *pre-baked
  hierarchical embeddings* — no download, no model, no GPU. UMAP recovers the
  nested clusters and the map letters ~160 signs across the four levels.

This is scaffolding for evaluating the display, **not** a substitute for the
real contrastive/LLM naming below; the real pipeline should populate
`_region_labels` from the build job (G2), at which point it wins over the
lazy ground-truth fallback (a set already pinned to the layout is never
overwritten).

## Open follow-ups

<!-- item-sep -->

- **LLM namer "preserve exact terms" instruction** — image study: the 7B
  namer rewrote exact breed keyphrases into generic names (hit 76%→50% on
  Stanford Dogs) and caption-derived names drifted to the appearance axis
  ("Fluffy White Dogs"); add prompt guidance to keep rare exact terms
  (breeds, brands, form titles) and re-measure.

<!-- item-sep -->

- **Captioner default validation on an uncurated image dump** — the image
  study's datasets are curated/single-domain (mixed is synthetic); run the
  Qwen-3B captioner default + prompt-sensitivity check on a real-world image
  dump before shipping it as the image `object_to_text` default.

<!-- item-sep -->

- **Layout-aware document embedder** — caption-text-space clustering did NOT
  recover document-type structure (ARI flat at 0.38 on RVL-CDIP); if
  taxonomy-grade doc browsing becomes a requirement, evaluate a
  Donut/LayoutLM-class embedder as a selectable image embedder.

<!-- item-sep -->

- **Image demo dataset fixes** — RVL-CDIP single-class mirror (#2291), Enrico
  URL/layout rot (#2292); working replacements prototyped in
  `scripts/experiments/toponymy_image/`.

<!-- item-sep -->

- **Vocab-filtering ablation** — drop AudioSet's music-genre terms from the
  tag vocabulary for non-music datasets and confirm the "Flamenco/Bluegrass"
  sign pollution disappears (framework rerun, minutes).

<!-- item-sep -->

- **Captioner validation on a second uncurated corpus + music** — the
  captioner's Clotho result is flattered by its training data; check GTZAN and
  one more real-world set before making it a default for user data.

<!-- item-sep -->

- **Scale run** — urbansound8k_a or a ~20k mixed corpus to validate the
  `base_min_cluster_size ∝ n` rule and the naming-cost model.

<!-- item-sep -->

- **KeyphraseNamer sibling dedup** — cheap pass that appends the first
  non-shared keyphrase when sibling signs collide (38% dups on speech maps).

<!-- item-sep -->

- **Detail-level sweep** — pick `lowest_detail_level` / prompt instructions
  that keep signs ≤6 words without losing specificity.

<!-- item-sep -->

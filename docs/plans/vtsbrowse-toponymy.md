# Plan: VTSBrowse Toponymy — remaining signpost work

> **Background.** The signpost infrastructure is live (Phase 1: no-LLM,
> zero-shot-tag texts). `vtscore/projection/signpost_texts.py` computes and
> caches one text per media on the media dict (persisted in the dataset
> pickle; audio = CLAP top-5 AudioSet-527 tags, image = SigLIP top-5
> OpenImages-600 tags, text = content — providers are registered per media
> type); `signpost_build.py` wraps the Tutte Institute `toponymy` library
> (5-D cosine clusterable UMAP, `base_min_cluster_size ∝ n`, `KeyphraseNamer`
> no-LLM namer) and flattens the fitted topic tree into a `RegionLabelSet`
> (medoid anchors in the frozen layout, coarsest layer at zoom band 0);
> `signpost_prep.py` orchestrates texts → fit → cache and is hooked into all
> three build paths — the opt-in ingest stage (texts cached **before** the
> registry pickle write; rides the `build_projection` opt-in), the lazy
> Browse build, and the Find→Browse subset build (in-memory only).
> Full-dataset label sets persist in the dataset container next to the
> projection, stamped with a `labeler_signature`
> (`namer|texts_provider:embedder|toponymy=version`) and served only over a
> matching `projection_id` + signature. `toponymy==0.5.2` installs
> `--no-deps` (its `transformers<5` pin is empirically unnecessary; real deps
> declared in `pyproject.toml`; `apricot-select` builds as a plain sdist —
> no SETUPTOOLS_USE_DISTUTILS shim, which setuptools >= 74 rejects at
> import); the `slow`-marked smoke test in
> `tests_lib/projection/test_toponymy_smoke.py` guards that bypass.
> Decision evidence lives in the experiment reports:
> **`docs/experiments/2026-07-12-toponymy-audio-signposts/`** and
> **`docs/experiments/2026-07-12-toponymy-image-signposts/`**, with reusable
> frameworks at `scripts/experiments/toponymy_{audio,image}/`.

## Phase 2 — real LLM namers + labeler settings

<!-- item-sep -->

- **LLM namer selection** — swap `KeyphraseNamer` for a bundled toponymy
  namer (`OpenAINamer(base_url=...)` / `OllamaNamer` / local
  `HuggingFaceNamer`), selected by new `browse_labeler` / `browse_llm_*`
  settings with fallback order `llm → keyphrase → none`. Set
  `llm_specific_instructions` (English-only, ≤6 words, no filler names —
  Qwen produced "日语学习指令" and "Sounds" without them). Everything else
  (clustering, keyphrases, texts, persistence, API, canvas) is unchanged;
  the `labeler_signature` already carries the namer name, so switching
  labelers invalidates persisted signs automatically.

<!-- item-sep -->

- **Labeler-selection meta (was G4)** — once namer selection exists,
  projection meta gains `available_labelers` and `labeler`.

<!-- item-sep -->

- **LLM namer "preserve exact terms" instruction** — image study: the 7B
  namer rewrote exact breed keyphrases into generic names (hit 76%→50% on
  Stanford Dogs) and caption-derived names drifted to the appearance axis
  ("Fluffy White Dogs"); add prompt guidance to keep rare exact terms
  (breeds, brands, form titles) and re-measure.

<!-- item-sep -->

- **Detail-level sweep** — pick `lowest_detail_level` / prompt instructions
  that keep signs ≤6 words without losing specificity (default fine-layer
  prompts request 8–15-word names — too long for map signs).

<!-- item-sep -->

- **Async/batch namers for the 20k+ regime** — naming ≈ 1 LLM call/topic ≈
  2 topics/s (Qwen2.5-7B on an a100); `_base_min_cluster_size` already
  scales ∝ n, but large datasets with a real LLM namer likely also want
  toponymy's async/batch namer variants.

## Phase 3 — richer `object_to_text` providers

<!-- item-sep -->

> **Shipped (opt-in):** the image (`Qwen2.5-VL-3B`) and audio
> (`MU-NLPC/whisper-small-audio-captioning`) captioner providers now exist in
> `vtscore/projection/signpost_captioners.py`, wired per media type behind the
> `browse_signpost_captioner` setting with the zero-shot tag provider retained
> as a `FallbackTextProvider` (model-load or per-item decode failure degrades
> to tags). Default stays **tags**; the items below are the validation still
> owed before either captioner can become the *default*.

<!-- item-sep -->

- **Audio captioner: validate + promote to default** — the whisper captioner
  (1 GB, ~90 s / 2k clips) produced the cleanest signs on ESC-50 and Clotho;
  still owed before it becomes the audio default: validate on GTZAN + one more
  real-world set (its Clotho result is flattered by training data).

<!-- item-sep -->

- **Image VLM captioner: validate + promote to default** — the Qwen2.5-VL-3B
  provider (214 s / 1k images, ~8 GB alongside SigLIP) beats the SigLIP tag
  fallback on fine-grained subsets (0–14% breed-sign hit vs 56–78%). Still
  owed before making it the image default: validate the prompt on a real-world
  uncurated image dump, and measure caption cache size vs pickle bloat.

<!-- item-sep -->

- **Captioner total-failure cache staleness** — when a captioner is enabled but
  its model can't load at all, `FallbackTextProvider` serves tags for every
  item yet stamps them under the *captioner* signature, so a later run with the
  model repaired won't recompute (the signature still matches). Acceptable
  best-effort for now; revisit alongside the "Post-hoc text persistence"
  follow-up if captioner adoption makes it bite.

<!-- item-sep -->

- **Speech routing** — Whisper transcripts as the `object_to_text` for
  speech datasets (reuse `vtscore/converters/audio2text.py`); but the CLAP
  *map* itself is near-random for speech (ARI 0.05), so the better long-term
  answer is browsing transcript-text embeddings via the existing `audio2text`
  converter. Needs a routing decision (per-dataset heuristic or user pick),
  not just a provider.

<!-- item-sep -->

- **Vocab-filtering ablation** — drop AudioSet's music-genre terms from the
  tag vocabulary for non-music datasets and confirm the "Flamenco/Bluegrass"
  sign pollution disappears (framework rerun, minutes).

## Pipeline follow-ups

<!-- item-sep -->

<!-- item-sep -->

- **Post-hoc text persistence** — signpost texts computed during the *lazy*
  Browse build (dataset ingested without prep) are stamped on the in-memory
  media dicts but never re-pickled, so the next process recomputes them.
  Cheap for tags; matters once captioner providers land — consider rewriting
  the container pickle (or a sidecar text entry) after a lazy-build labeling.

<!-- item-sep -->

- **Scale run** — urbansound8k_a or a ~20k mixed corpus to validate the
  `base_min_cluster_size ∝ n` rule and the naming-cost model.

<!-- item-sep -->

- **Layout-aware document embedder** — caption-text-space clustering did NOT
  recover document-type structure (ARI flat at 0.38 on RVL-CDIP); if
  taxonomy-grade doc browsing becomes a requirement, evaluate a
  Donut/LayoutLM-class embedder as a selectable image embedder.

<!-- item-sep -->

- **Search-by-sign / user-editable signs** — original Phase 3 UI ideas;
  unscoped.

<!-- item-sep -->

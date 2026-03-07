# Plan: New MediaEmbedders

> **Status (March 2026):** Three embedders from this plan have been implemented:
> `ImageSiglipEmbedder`, `TextBGEEmbedder`, and `AudioClapMusicEmbedder`.
> The remaining proposals are still open for future work.

## Current State

We now have 7 embedders across the non-document media types (4 defaults + 3 alternatives):

| Embedder | Model | Media Type | Dim | Library |
|---|---|---|---|---|
| `AudioClapEmbedder` | `laion/clap-htsat-unfused` | audio | 512 | transformers |
| `AudioClapMusicEmbedder` | `laion/larger_clap_music_and_speech` | audio | 512 | transformers | **IMPLEMENTED** |
| `ImageClipEmbedder` | `openai/clip-vit-base-patch32` | image | 768 | transformers |
| `ImageSiglipEmbedder` | `google/siglip-base-patch16-224` | image | 768 | transformers | **IMPLEMENTED** |
| `TextE5Embedder` | `intfloat/e5-base-v2` | paragraph | 768 | sentence-transformers |
| `TextBGEEmbedder` | `BAAI/bge-base-en-v1.5` | paragraph | 768 | sentence-transformers | **IMPLEMENTED** |
| `VideoXClipEmbedder` | `microsoft/xclip-base-patch32` | video | 768 | transformers |

The architecture already supports multiple embedders per media type (registry keyed by `name`, `embedders_for_type()` returns a list). Adding new embedders requires:

1. A new class extending `MediaEmbedder` in `vtsearch/media/<type>/`
2. A config constant for the model ID in `vtsearch/config.py`
3. Registration via `register_embedder()` in `vtsearch/media/__init__.py`

---

## Proposed New Embedders

### Image

#### 1. `ImageSiglipEmbedder` (Priority: HIGH) — IMPLEMENTED
- **Model**: `google/siglip-base-patch16-224` (ViT-B-16-SigLIP)
- **Dimension**: 768
- **Library**: transformers (`SiglipModel`, `SiglipProcessor`)
- **Why**: SigLIP uses sigmoid loss instead of softmax contrastive loss, which improves zero-shot classification accuracy over CLIP on many benchmarks. Better calibrated similarity scores (no need for temperature scaling). Trained on WebLI dataset.
- **Description wrappers**: Same style as CLIP ("a photo of {text}", etc.)
- **Notes**: Drop-in replacement pattern — same `embed_media`/`embed_text` flow as `ImageClipEmbedder`. Should also expose `embed_pil_image()` for PDF/CIFAR use cases.

#### 2. `ImageDINOv2Embedder` (Priority: MEDIUM)
- **Model**: `facebook/dinov2-base`
- **Dimension**: 768
- **Library**: transformers (`Dinov2Model`, `AutoImageProcessor`)
- **Why**: Self-supervised vision-only model. Excellent for visual similarity search where text queries aren't needed. Produces richer visual features than contrastive models for tasks like near-duplicate detection and style similarity.
- **Limitation**: No text encoder — `embed_text()` returns `None`. Text-sort would be unavailable; only example-sort and learned-sort would work.
- **Description wrappers**: Empty (no text embedding).

#### 3. `ImageEVA02ClipEmbedder` (Priority: LOW)
- **Model**: `QuanSun/EVA-CLIP-8B` or a smaller variant like `EVA02-CLIP-B-16` via open_clip
- **Dimension**: 768 (base) or larger
- **Library**: open_clip (`create_model_and_transforms`)
- **Why**: EVA-02 CLIP achieves state-of-the-art on many vision-language benchmarks. Significant improvement over vanilla CLIP.
- **Notes**: Would require adding `open_clip` as a dependency. Worth considering only if we already use open_clip for SigLIP (we don't — SigLIP is in transformers).

### Audio

#### 4. `AudioClapMusicEmbedder` (Priority: MEDIUM) — IMPLEMENTED
- **Model**: `laion/larger_clap_music_and_speech`
- **Dimension**: 512
- **Library**: transformers (`ClapModel`, `ClapProcessor`)
- **Why**: Larger CLAP variant trained on music and speech data. Better for music retrieval and genre classification than the unfused model. Same API as existing CLAP embedder.
- **Notes**: Nearly identical implementation to `AudioClapEmbedder` — could share a base class or parameterize the model ID.

#### 5. `AudioWhisperEmbedder` (Priority: LOW)
- **Model**: `openai/whisper-base` (encoder only)
- **Dimension**: 512
- **Library**: transformers (`WhisperModel`, `WhisperProcessor`)
- **Why**: Whisper's encoder produces speech-aware features. Useful for spoken-word audio where semantic speech content matters more than acoustic properties.
- **Limitation**: No text encoder in the same space — `embed_text()` returns `None`. Speech-specific; poor for environmental sounds or music.
- **Notes**: Would use encoder output pooling (mean over time dimension).

### Text

#### 6. `TextBGEEmbedder` (Priority: MEDIUM) — IMPLEMENTED
- **Model**: `BAAI/bge-base-en-v1.5`
- **Dimension**: 768
- **Library**: sentence-transformers (`SentenceTransformer`)
- **Why**: BGE consistently ranks at or near the top of the MTEB leaderboard. Asymmetric retrieval with "Represent this sentence: " prefix for queries.
- **Notes**: Very similar implementation to `TextE5Embedder` — different model ID and query prefix.

#### 7. `TextGTEEmbedder` (Priority: LOW)
- **Model**: `thenlper/gte-base`
- **Dimension**: 768
- **Library**: sentence-transformers (`SentenceTransformer`)
- **Why**: GTE is another strong MTEB performer. Symmetric — no special prefix needed for queries vs passages, which simplifies the implementation.

#### 8. `TextJinaEmbedder` (Priority: LOW)
- **Model**: `jinaai/jina-embeddings-v2-base-en`
- **Dimension**: 768
- **Library**: sentence-transformers (`SentenceTransformer`)
- **Why**: Supports up to 8192 token context (vs 512 for E5/BGE), making it better for long documents. Good general-purpose retrieval model.

### Video

#### 9. `VideoLanguageBindEmbedder` (Priority: LOW)
- **Model**: `LanguageBind/LanguageBind_Video_FT`
- **Dimension**: 768
- **Library**: transformers or custom LanguageBind package
- **Why**: LanguageBind aligns video, audio, and text in a shared space. Could provide better video-text alignment than X-CLIP.
- **Notes**: Dependency footprint is larger. May require the `languagebind` package.

### Multi-Modal

#### 10. `ImageBindEmbedder` (Priority: LOW — exploratory)
- **Model**: `facebook/imagebind-huge`
- **Dimension**: 1024
- **Library**: Custom (Meta's imagebind package)
- **Why**: Single model that embeds image, text, audio, video, depth, and thermal into one shared space. Could theoretically replace all four current embedders with a single model.
- **Limitation**: Large model (~4.6GB), requires the `imagebind` package (not on PyPI, installed from GitHub). The shared embedding space may sacrifice per-modality quality compared to specialized models.
- **Notes**: More of a research exploration than a production candidate.

---

## Implementation Order (Recommended)

1. ~~**`ImageSiglipEmbedder`** — Highest value, straightforward, no new dependencies~~ **DONE**
2. ~~**`TextBGEEmbedder`** — Easy win, same sentence-transformers pattern as E5~~ **DONE**
3. ~~**`AudioClapMusicEmbedder`** — Nearly identical to existing CLAP, immediate value for music datasets~~ **DONE**
4. **`ImageDINOv2Embedder`** — Interesting vision-only alternative, good for visual similarity
5. Everything else as needed

## Implementation Checklist (per embedder)

- [ ] Add `MODEL_ID` constant to `vtsearch/config.py`
- [ ] Create embedder class in `vtsearch/media/<type>/embedder_<name>.py`
- [ ] Implement `name`, `media_type_id`, `load_models()`, `embed_media()`, `embed_text()`
- [ ] Add `description_wrappers` if text embedding is supported
- [ ] Register in `vtsearch/media/__init__.py`
- [ ] Add model to `requirements-cpu.txt` / `requirements-gpu.txt` if new dependency needed
- [ ] Add tests (embed a test media file, embed text, check dimensions)
- [ ] Update `vtsearch/models/loader.py` preload logic if needed
- [ ] Update docs

## Open Questions

1. **Multiple embedders per type in the UI**: The frontend currently assumes one embedder per media type for text-sort. How should the user select which embedder to use? A dropdown in the sort dialog? A setting?
2. **Embedding cache key**: Currently the cache is keyed by media file hash. With multiple embedders, the cache key must include the embedder name to avoid collisions. Need to verify this is already handled.
3. **Detector compatibility**: Detectors (neural nets) are trained on embeddings of a specific dimension from a specific embedder. If a user switches embedders, existing detectors become incompatible. Should we store the embedder name with the detector and enforce matching?

# VTSearch Feature Brainstorm

*A wide-ranging idea backlog covering internal improvements, new features, new media types, converters, clippers, demo datasets, and experiments.*

This is intentionally broad and unfiltered. Items are tagged with rough priority hints (★★★ high, ★★ medium, ★ exploratory) and effort hints (XS / S / M / L / XL). Pick what's interesting and turn the rest into separate plan docs as they mature.

---

## 1. New Media Types

### 1.1 Code / source files ★★ M
A first-class `code` media type for browsing and clustering source files. Embedder: **CodeBERT**, **UniXcoder**, **StarEncoder**, or **Voyage-Code-3**. Display with syntax highlighting (Prism/Shiki). Extensions: every common source ext. Use case: search "files that handle authentication", "places we call SSE", "TODO clusters by topic". This is a natural fit for VTSearch's "rank → vote → train" loop applied to code review or refactoring queues.

### 1.2 Tabular / spreadsheet ★ M
`tabular` for CSV/TSV/Parquet/XLSX. Embedder: column-name + sample-row text passed to E5/BGE; or **TaPaS**/**TAPEX**. Display: rendered grid with column types. Voting use case: "which CSVs look like time-series sensor data?", "which sheets have PII?".

### 1.3 3D mesh / point cloud ★ L
`mesh` for `.obj`, `.stl`, `.ply`, `.glb`. Embedder: **OpenShape**, **Uni3D**, **PointBERT**. Display: Three.js viewer. Converter: `mesh→image` (multi-angle renders) so we get free fallback through SigLIP. Use case: 3D asset libraries.

### 1.4 Time series / signals ★ M
Generic 1-D signal type (ECG, accelerometer, finance ticks). Embedder: **TS2Vec**, **MOMENT**, **Chronos**. Converter: `timeseries→image` (line plot, recurrence plot, GAF) → reuse SigLIP. Display: Plotly chart with brush-to-zoom.

### 1.5 Music / MIDI ★ S–M
Specialised `midi` type separate from `audio`. Embedder: **MERT**, **Jukebox** features. Or convert: `midi→audio` via fluidsynth → CLAP-Music. Display: piano roll.

### 1.6 Spectrogram-as-image ★ XS
Not a new type per se, but an "audio in image-mode" via `audio→image` converter (see §2.1). Lets users vote on visual structure of audio (e.g. "find recordings with car-horn pattern").

### 1.7 Email / message ★ M
`.eml`, `.mbox`, Slack JSON. Embedder: E5 on `subject + body` plus structured headers as metadata. Display: threaded view. Use case: triage inbox/support queues.

### 1.8 Geospatial raster ★ M
GeoTIFF/Sentinel/Landsat tiles. Embedder: **Prithvi**, **SatlasNet**, **Clay**. Use case: satellite imagery curation.

### 1.9 Web pages / HTML ★ S
`.html`/`.mhtml`/URL fetch. Render to Markdown via Readability before embedding with E5. Display: rendered iframe + extracted text side-by-side. Useful for bookmark organisation, OSINT.

### 1.10 Bio-signal / EEG ★ exploratory
Niche but cool: **LaBraM**, **BENDR**. Probably blocked by data availability for demos.

---

## 2. New MediaConverters

### 2.1 `audio → image` (spectrogram) ★★★ S ✅ shipped
Mel-spectrogram or CQT plot as PNG. Unlocks SigLIP/DINOv3 over audio data — huge for cross-model ensembling and for users who want to *visually* spot recurring spectral patterns. Params: `spectrogram_type` (mel/cqt), `n_mels`, `time_window_s`, `colormap`. See `vtsearch/converters/audio2image.py`.

### 2.2 `image → text` (OCR) ★★★ S ✅ shipped
PaddleOCR-backed converter that flattens detected text regions into a single text media. Params: `language`, `threshold`. See `vtsearch/converters/image2text.py`.

### 2.3 `audio → text` (ASR) ★★ M
Whisper-tiny/base. Converts podcasts/voice-notes into searchable transcript. Param: `language`, `model_size`. Pairs with `video→audio→text` chain.

### 2.4 `video → text` (transcription) ★★ M
Composition of `video→audio→text`, but worth exposing as a single converter so users don't have to chain two manually.

### 2.5 `text → audio` (TTS preview) ★ S
For voting on summaries: hear them played out. Param: voice (Coqui/Piper).

### 2.6 `image → text` (caption) ★★ M
**BLIP-2**, **LLaVA-1.6**, **Florence-2** captioning. Provides a caption that can be edited and re-embedded; useful for archives without metadata.

### 2.7 `image → image` (super-resolution / denoise) ★ S
Real-ESRGAN. Useful for surveillance/historical photo collections.

### 2.8 `pdf → table` (table extraction) ★ M
Camelot/Tabula extracts CSVs from PDF tables → tabular media type.

### 2.9 `code → text` (docstring extraction) ★ XS
For language-agnostic code search via natural language.

### 2.10 `video → keyframe-image` ★★ S
Already partially covered by `video→image`, but with **TransNet V2** scene boundary picking instead of uniform sampling. Far better for content-aware clipping.

### 2.11 `email → text` ★ XS
Strip MIME, dedupe quoted replies, normalise headers.

### 2.12 Composability ★★ M (cross-cutting)
Today the framework allows one converter per source; consider explicit chains: `video → audio → spectrogram → image embedder`. The plumbing already exists in `effective_source_specs`; what's missing is the UI.

---

## 3. New MediaClippers

### 3.1 Audio
- **`sound_silence`** ★★ S — split on silence (librosa `effects.split`). Drop intro/outro silence. Param: `top_db`.
- **`sound_beat`** ★ S — split on detected beats / downbeats (madmom). For music datasets.
- **`sound_speech_activity`** ★★ S — VAD-based segmentation (Silero VAD). Useful for podcasts.
- **`sound_energy_envelope`** ★ S — segment around energy peaks (find drum hits, claps, gunshots).

### 3.2 Image
- **`image_saliency`** ★★ M — crop to salient region (U²-Net or SAM auto-mask). Improves embedding signal on "find dogs" by focusing on subject.
- **`image_object`** ★★ M — clip per detected object using a lightweight YOLO/RT-DETR. Each clip becomes its own scoreable element.
- **`image_face`** ★ S — face-cropped clips (RetinaFace).
- **`image_window`** ★ XS — sliding window with stride, complements existing tiling.
- **`image_color_palette`** ★ exploratory — clip into colour-region masks.

### 3.3 Text
- **`text_paragraph`** ★★ XS — split on `\n\n`. Free wins on prose.
- **`text_token_window`** ★★ S — token-aware windows (tiktoken) with overlap.
- **`text_semantic`** ★★ M — embedding-similarity-based chunking (sentence groups whose adjacent cosine drops below threshold).
- **`text_heading`** ★ S — split on Markdown/HTML headings.

### 3.4 Video
- **`video_action`** ★ M — clip around detected action peaks (motion histogram or VideoMAE attention).
- **`video_audio_cue`** ★ M — clip around audio-event peaks (loud noise, vocal onset).
- **`video_uniform_with_keyframes`** ★★ S — uniform sampling biased to nearest I-frame.
- **`video_dialogue`** ★ M — clip per speech turn (VAD on demuxed audio).

### 3.5 Document
- **`document_section`** ★★ S — split per H1/H2, or per chapter for ePub.
- **`document_page_range`** ★ XS — manual page-window clips for long PDFs.

### 3.6 Cross-cutting
A **`clipper_chain`** abstraction so we can run e.g. `document_section → text_token_window` without writing a custom clipper.

---

## 4. New Embedders

### 4.1 Image
- **MetaCLIP-Huge** ★★ — drop-in upgrade from CLIP for English-heavy tasks.
- **Multilingual SigLIP** ★★★ — important: today's text embedders are English-leaning. Multilingual SigLIP unlocks non-English UI users.
- **EVA-CLIP-18B** ★ — flagship-quality, but heavy.
- **Florence-2** ★★ — captioning + grounding + features in one model; great for the patch-region voting story.
- **SAM2 features** ★★ — for `image_object`/`image_saliency` clippers.
- **NaFlex SigLIP-2** ★★ — supports native aspect ratios; better for non-square images and document pages.

### 4.2 Text
- **NV-Embed-v2** ★★, **Stella-en-1.5B** ★★ — top of MTEB; tiny enough to run on consumer GPU.
- **mE5** / **multilingual-e5-large** ★★★ — same multilingual-UI argument as SigLIP.
- **ColBERT** ★ — late-interaction retrieval would change the storage shape; experimental.
- **GTE-Qwen2-1.5B** ★ — strong, larger.
- **Voyage-Code-3** / **Jina-Code-v2** ★★ — pair with the `code` media type (§1.1).

### 4.3 Audio
- **AST (Audio Spectrogram Transformer)** ★★ — strong general audio.
- **CLAP-General-2024** ★★ — newer LAION CLAP checkpoint.
- **MERT-v1-330M** ★★ — music-specialised; complements CLAP-Music.
- **Whisper-encoder** ★ — speech-rich datasets; use the encoder hidden states only.

### 4.4 Video
- **InternVideo2** ★★ — current SOTA for video-text retrieval.
- **VideoMAE-v2** ★ — single-modality but very strong action features.

### 4.5 Multimodal-shared
- **ImageBind / LanguageBind family** ★★★ — *one* embedder that handles audio, image, text, video, depth, IMU. Would let us run cross-modal queries like "audio clips that match this photo" without converter chains. Big architectural win for the cross-modal voting story.

### 4.6 Embedder ergonomics
- **LoRA-on-embedder** ★ XL — fine-tune the last few layers of CLIP/SigLIP/CLAP on user votes. Risky (no-persisted-vectors rule applies to weights too — need to think through).
- **Reranker stage** ★★ M — after retrieval, rerank top-K with a cross-encoder. BGE-Reranker is small and fast.
- **Quantised embedders** ★ S — int8/binary with **MRL** truncation; saves RAM at near-zero quality loss.

---

## 5. New Detectors / Trainable Models

Today there's exactly one detector type: a tiny binary MLP. Rich opportunities here.

### 5.1 Multi-class detector ★★★ L
A softmax head with N classes instead of one binary head. Lets users build "genre classifier", "scene type", "speaker ID" detectors without spinning up N separate binary detectors. Voting UI shifts from green/red to N coloured buttons or a chip picker.

### 5.2 Hierarchical / taxonomy detector ★★ L
For datasets that have natural taxonomy (Stanford Dogs → breed → group). Train multi-level classifier; UI shows breadcrumb of predicted ancestors.

### 5.3 Regression detector ★★ M
Continuous score 0–1: "aesthetic quality", "explicitness", "loudness". Train MSE/Huber instead of BCE. Users vote with a slider rather than green/red. Infrastructure exists (`inclusion`-as-slider), generalises naturally.

### 5.4 Few-shot prototype detector ★★ M
No MLP — just store class centroids and use cosine. Instant predictions, zero training. Great for "I have 3 examples, find more". Calibration via temperature scaling.

### 5.5 Outlier detector ★ M
Isolation Forest / one-class SVM over embeddings. Single-button workflow: "show me weird stuff in this dataset". Pairs beautifully with diversity tree.

### 5.6 Ensemble of MLPs ★★ S
Train 5 small MLPs with different seeds; report mean + std. Std is a free uncertainty estimate that improves the "Hard" select-mode.

### 5.7 Calibrated probabilities ★★ S
Add Platt scaling / isotonic regression on top of MLP logits. Shows users a real probability instead of an unbounded score; fixes the "what does 0.7 mean?" UX confusion.

### 5.8 Patch-level region detectors ★★ M
Extend region-aware MLP scoring to a region-level binary detector. "Find images where *part of the image* contains a face" rather than the whole-image score. Plumbing exists via `patch_grid`.

---

## 6. New Sorting Modes

Beyond text-sort, learned-sort, example-sort:

### 6.1 Multi-anchor sort ★★ S
Sort by mean cosine to N anchor items rather than 1. Quick-and-dirty "find more like these 5 things" without spinning up a detector.

### 6.2 Negative prompt sort ★★ XS
"Like X but not Y": `cos(item, X) - α · cos(item, Y)`. Single-line addition to text-sort.

### 6.3 Cross-modal sort ★★ M
"Sort audio by similarity to this image" via shared-embedder (ImageBind) or via converter chain (image→text→audio). One UI flow, two backends.

### 6.4 Metadata sort ★★ XS
Sort by file size, duration, date, dimensions, bitrate, MD5 (debug). Currently only score sort exists; add a "Sort by" dropdown that lists available metadata columns.

### 6.5 Cluster view / group-by ★★★ M
Render the diversity-tree top-level clusters as collapsible groups. Click a cluster → drill in. Feels much more like an "explorer" than a flat list.

### 6.6 Density / typicality sort ★ M
Sort by local density (k-NN distance in embedding space). Show prototypical examples first, or anomalous ones first.

### 6.7 Reverse / shuffle / random ★ XS
Quality-of-life: random shuffle for blind voting, reverse to inspect bottom of list.

### 6.8 Compose sorts ★ S
"Filter by inclusion threshold then sort by text". Today everything is replace-only; allow stacking.

### 6.9 Diff sort ★ M
Sort by `score_modelA − score_modelB`. Find items where two detectors disagree — highest-leverage label candidates.

---

## 7. New Importers

### 7.1 Hugging Face Datasets ★★★ M
`datasets.load_dataset(name, split=...)`. Massive unlock — every benchmark dataset becomes a one-click import. HF Hub auth token reuse from existing `HF_TOKEN`.

### 7.2 YouTube / Vimeo / podcast feed ★★ M
yt-dlp + cookies. Bulk-import a channel or playlist. Combine with `video→audio→text` for instant podcast search.

### 7.3 S3 / GCS / Azure Blob ★★ M
For team-scale media libraries. Use boto3/google-cloud-storage; respect existing folder importer's contract.

### 7.4 Arxiv / S2 / OpenReview ★★ S
Fetch by query → PDF → existing pdf chain.

### 7.5 RSS / Atom ★ S
Subscribe to feeds; periodic re-pull via cron.

### 7.6 Reddit / Mastodon / Bluesky export ★ S
For social-media analysis users.

### 7.7 GitHub repo ★ M
Clone repo → walk → embed source files (pairs with `code` type §1.1).

### 7.8 IMAP / POP3 inbox ★ M
For email clustering use case (§1.7).

### 7.9 SFTP / FTP / WebDAV ★ S
NAS users.

### 7.10 Browser-extension importer ★ L
Right-click → "Send to VTSearch". Bookmark organiser angle.

### 7.11 Wallabag / Pocket / raindrop.io ★ XS
For personal-knowledge-base users.

### 7.12 ImageNet / COCO / LAION shard ★★ M
Streaming WebDataset support; only embed what you actually open.

---

## 8. New Exporters

### 8.1 Hugging Face Dataset push ★★★ M
Mirror of §7.1. Closes the loop: import HF, label, push back to HF as a labeled split.

### 8.2 Parquet / Arrow ★★ S
ML-friendly tabular format with embeddings (when policy allows).

### 8.3 SQLite snapshot ★★ S
Self-contained `.sqlite` of media + labels + scores. Simpler to share than JSON for analytics use.

### 8.4 WebDataset / TFRecord shard ★ M
For training pipelines downstream.

### 8.5 Slack / Discord / Telegram ★ S
Notification-style: post top-N hits or training summary to a channel.

### 8.6 Google Sheets / Notion / Airtable ★ M
Collaboration-friendly outputs. Embed thumbnails inline.

### 8.7 Webhook batched / streaming ★ XS
Existing webhook is one-shot; add a streaming variant that POSTs as items finish.

### 8.8 Excel with embedded thumbnails ★ S
For non-technical reviewers; uses openpyxl image insert.

### 8.9 PDF report ★ M
"Top 20 hits from detector X on dataset Y, with thumbnails and scores." Reportlab.

---

## 9. Sync Sources & Settings I/O

### 9.1 Cloud-blob sync source ★★ M
S3/GCS-backed `LabelsetSource`/`SettingsSource` so multiple containers can share state without a shared filesystem.

### 9.2 GitOps sync source ★ M
Settings/labelsets as `.json` in a Git repo, auto-commit on change. Free audit log + rollback.

### 9.3 Settings profiles ★★ S
Named saved bundles ("Aesthetic-grading workflow", "Audio-tagging workflow"). One-click switch instead of fiddling with N toggles.

### 9.4 Per-user settings ★★★ M
Today settings are global even with multi-user auth (HANDOFF.md flags this). Move to per-user settings file under `get_user_data_dir()`.

### 9.5 Detector-coupled settings ★ S
Some settings (calibrate_count, inclusion default, autopilot config) are arguably per-detector. Allow opt-in override.

---

## 10. Demo Datasets to Add

### 10.1 Audio
- **NSynth** — instruments × pitches × velocities. Perfect CLAP-Music demo.
- **FSD50K** — 50k Freesound clips, 200 sound classes. Better than ESC-50 for label-rich demos.
- **MagnaTagATune** — music tagging benchmark.
- **VoxCeleb1 mini** — speaker ID; great for "few-shot prototype detector" §5.4.
- **CommonVoice** subsets per language — demonstrates multilingual sort.
- **RAVDESS / CREMA-D** — emotional speech; classic detector targets.

### 10.2 Image
- **CIFAR-100**, **STL-10** — small, fast, classic.
- **Tiny-ImageNet** — light ImageNet-flavoured classification.
- **iNaturalist mini** — fine-grained species; pairs with hierarchical detector §5.2.
- **ArtBench** — paintings by genre; aesthetic-grading regression playground §5.3.
- **DocVQA / FUNSD / RVL-CDIP** — for the OCR/document story.
- **WikiArt** — multi-style visual.
- **NSFW-detector test set** — for content-safety demo (clearly labelled, opt-in).

### 10.3 Text
- **Wikipedia 20-topics subset** — bigger 20NG cousin.
- **arXiv abstracts** — multilingual scientific search.
- **Reuters-21578** — historical baseline.
- **Hugging Face daily-papers** RSS — feels alive.
- **GitHub issues** sample — pairs with code type §1.1.
- **Songs-lyrics dataset** — fun for music-info-retrieval demos.

### 10.4 Video
- **Kinetics-400 mini** — action recognition standard.
- **HMDB-51** — older but small.
- **AVA-Speech** — for speech-activity clipper §3.4.
- **Breakfast / 50Salads** — long-form action sequences.
- **Charades** — multi-action labels.

### 10.5 Document
- **CUAD legal contracts** — clauses-as-labels; great for hierarchical detector.
- **Recipe1M** — recipes as PDFs/HTML.
- **ProjectGutenberg** sample — long-form text/document hybrid.

### 10.6 Multimodal
- **MS-MARCO video** subset — text+video retrieval.
- **CLEVR** — synthetic compositional reasoning.
- **AudioSet eval** subset — multi-label audio.

### 10.7 Synthetic
- Procedural **shape grammar** images (Spirograph, fractal trees) — free DINOv3 stress test.
- Procedural **drum patterns** — free CLAP-Music stress test.
- Procedural **physics scenes** (rolling balls) — for video.

---

## 11. UI / UX Improvements

### 11.1 Mobile-responsive layout ★★★ XL
HANDOFF + frontend audit confirm: desktop-only today. Phone & tablet support is the single biggest reach extension. Touch-swipe to vote falls out for free.

### 11.2 Keyboard-shortcut overlay ★★ XS
`?` opens a help sheet listing every binding. Fixes onboarding fragility.

### 11.3 Undo last vote ★★★ XS
`Cmd/Ctrl-Z`. Mis-clicks happen constantly when speed-labelling.

### 11.4 Bulk vote ★★★ S
Shift-click range select on the list. "Mark all 14 of these as good." Hugely speeds up obvious-class labelling.

### 11.5 Multi-select + lasso (grid) ★★ M
Drag-to-select on grid view.

### 11.6 Comparison mode ★★ M
Side-by-side A/B on two media items. For tie-breaking similar scores or model-vs-model output.

### 11.7 Comments / notes per item ★★ S
Free-text per-media note. Round-trips through label export. Useful for "why I voted bad on this one".

### 11.8 Tag/multi-label system ★★★ L
First-class multi-label support orthogonal to the binary detector. A media item can have any subset of `{"speech", "outdoor", "echoey"}`. Pairs with multi-class detector §5.1.

### 11.9 Saved views / saved searches ★★ S
"My top-10 hits for detector X on dataset Y" as a bookmark.

### 11.10 Vote history scrubber ★ S
Rewind through last N votes; see how your distribution shifted as labels grew.

### 11.11 Achievements polish ★ XS
Service exists; surface it more (toasts, weekly recap).

### 11.12 Onboarding tour ★★ M
Driver.js / Shepherd-style overlay walking new users through Dashboard → Load → Vote → Train. Today the first-time experience drops you in deep.

### 11.13 Region-of-interest voting UI ★★ M
Already partially supported on the backend (patch grids); finish the box-draw UX so users can vote "this *part* of the image is a positive".

### 11.14 Audio loop A-B / waveform clips ★ S
Set A and B markers, loop between them. For close auditioning.

### 11.15 PDF page jump + highlight ★ S
For document review.

### 11.16 Drag-and-drop import directly to dashboard ★★ S
Skip the modal. Drop a folder → start importing.

### 11.17 Recent items / history pane ★ S
Persistent across sessions.

### 11.18 Light improvements
- Toast notifications instead of alert dialogs ★ XS
- Empty-state illustrations for "no datasets yet" ★ XS
- Skeleton loaders (today: spinners) ★ S
- Per-detector colour accent ★ XS

### 11.19 Voice annotations ★ exploratory M
Hold a key, dictate a note that's transcribed (Whisper) and attached to the media. For mobile-style fast review.

### 11.20 Sketch search ★ exploratory M
Draw a sketch, use it as image-sort query. CLIP+sketch models exist.

---

## 12. Internal / Architectural Improvements

### 12.1 Streaming embeddings (lazy) ★★ L
Today everything is in RAM. Memory-mapped or DB-backed embedding store (DuckDB / LanceDB / Qdrant) so we can hold 1M items. Important for the HF / S3 importers (§7) to be useful.

### 12.2 GPU batched embedding ★★ M — **mostly DONE**
Phase A (image + text bulk overrides), Phase B (bulk `patch_forward`),
and Phase C (clip re-embed via `embed_media_bulk` with no tempfile)
all landed. Deferred follow-ups:

- **Audio CLAP + CLAP-Music bulk override.** Decode is the bottleneck
  and adds I/O complexity; smaller GPU win than image but still
  meaningful for big audio imports. `librosa` is happy to decode a
  list serially while the model batches.
- **Video X-CLIP / LanguageBind bulk override.** Tricky because
  X-CLIP at batch 32 with 8 frames each is ~640 MB of activations
  and can OOM on 8 GB cards. Likely wants a smaller default
  `embed_batch_size` (e.g. 8) on the video embedders.
- **Fuse single-vector + patch forward on DINOv2/DINOv3/EUPE.** Today
  the backbone runs twice per image (once for `embed_media_bulk`,
  once for `patch_forward_bulk`). Fusing requires changing the loader
  to call a single combined hook and split the outputs — worth it if
  profiling shows the backbone forward is the dominant cost.

### 12.3 Mixed-precision training ★ XS
`torch.cuda.amp` for the MLP; trivial change.

### 12.4 Model preload manager ★★ S — **DONE**
Replaced the static `autoload_media_embedders` setting with
`preload_predicted_embedders()` in `vtsearch/embedding/loader.py`. Startup
walks the dataset and detector registries and warms each unique embedder
referenced (`entry["embedder"]`, falling back to the default embedder
for `entry["media_type"]`). `register_dataset()` also fires
`smart_preload_in_background()` so a newly-implied embedder is warmed
mid-session. Empty registry preloads nothing.

### 12.5 ~~WebSocket~~ SSE for live progress ★★ M — **DONE**
Was: progress polled via REST (`/api/dataset/progress`, `/api/sort/progress`,
`/api/find/progress`, `/api/dataset/loading-tasks`,
`/api/detectors/loading-tasks`, `/api/eval/voting-iterations`).

Replaced by a single Server-Sent Events stream at **`GET /api/events`**
(see [`docs/api/events.md`](../api/events.md)). Channels: `dataset`,
`sort`, `find`, `eval`, `loading-tasks`, `detector-loading-tasks`. The
first frame on every channel is the current snapshot, so clients don't
need a bootstrap REST call.

**Why SSE over WebSocket:**
- Progress is one-way (server → client), text-only — exactly SSE's
  sweet spot, where WebSocket's bidirectionality is wasted.
- Flask is sync/WSGI — SSE works as a plain streaming response, no
  `flask-sock` / ASGI / Upgrade-handshake surface area.
- Auth, per-request context, and proxy compatibility come for free
  because it's still an HTTP GET.
- `EventSource` reconnects automatically; we don't have to build
  heartbeat / resume / backoff.

WebSocket can be revisited later if a genuinely bidirectional feature
(live multi-user voting, presence) lands.

The old §18.9 ("SSE event stream") is subsumed by this item — they
referred to the same idea.

### 12.6 Background prefetch of next likely media ★ S
For speed-labelling, preload the next 3 items' previews.

### 12.7 Resume interrupted training ★ S
Checkpoint MLP state every N epochs.

### 12.8 Centralised plugin registry CLI ★ S — **shipped**
`python app.py --list-plugins` shows every importer/exporter/embedder/converter/clipper across all auto-discovered plugin families. Supports `--format plain|json|names` and `--plugin-family <name>` for shell-completion scripts. Backed by `vtsearch.plugins.inventory.gather_plugins()`. See [CLI.md § Inspecting plugins and the API schema](../CLI.md#inspecting-plugins-and-the-api-schema).

### 12.9 OpenAPI schema ★★ M — **shipped (minimal); deeper migration in progress**
`GET /openapi.json` and `python app.py --openapi-schema` return an OpenAPI 3.0 doc generated from Flask's `url_map` — every route, method, path parameter, and view docstring. Request/response schemas are intentionally left permissive (`{type: object}`); the route inventory alone is enough to power Swagger UI and a generated TS client and to gate API surface changes in CI. See [API.md § Machine-readable schema](../API.md#machine-readable-schema). The open question — "do we want to add per-route schemas later?" — is now being answered: a deeper migration to flask-smorest + marshmallow with real request/response schemas is in progress at [openapi-schema.md](openapi-schema.md), serving its richer spec at `/api/openapi.json` and a Swagger UI at `/api/docs`. Both implementations coexist today; consolidate on flask-smorest once enough blueprints are migrated and delete the `vtsearch.openapi` walker + the `/openapi.json` route + the `--openapi-schema` CLI flag.

### 12.10 Python client library ★★ M
`pip install vtsearch-client` so notebooks can drive the same endpoints headlessly.

### 12.11 Plugin discovery via importlib.metadata ★★ M — **shipped**
Every `PluginRegistry` now also scans an `importlib.metadata` entry-point group (`vtsearch.importers`, `vtsearch.exporters`, …). Third-party packages can register plugins from their own `pyproject.toml` without monkey-patching `vtsearch`. Built-ins still win on name clashes; broken entry points warn and are skipped. See [EXTENDING-plugins.md § Third-party plugins via importlib.metadata entry points](../EXTENDING-plugins.md#third-party-plugins-via-importlibmetadata-entry-points).

### 12.12 Ruff → Ruff format CI gate ★ XS
We have ruff; add a CI step that fails on unformatted code.

### 12.13 Type-checking with mypy/pyright ★ M — **Stage 1 shipped; in progress**
Pyright in basic mode is now a hard CI gate. Rolled out package-by-package per [pyright-type-checking.md](pyright-type-checking.md). Stage 1 (foundation: `auth`, `cli`, `concurrency`, `config`, `exporters`, `labels`, `plugins`, `settings_io`, `sync`, `utils`) shipped in PR #1349 along with an advisory job that reports residual error counts over the full `vtsearch/` package.

### 12.14 Async-friendly Flask routes ★ M (not recommended)
Original idea: migrate hot paths (sorting, scoring) to Quart or FastAPI for true async; long-running training already uses background threads so impact is bounded — but the polling endpoints would benefit.

**Assessment: not worth doing as currently scoped.** The benefit is thin and the migration cost is real:

- Sort/score are CPU-bound (embedding dot products, MLP inference, ranking). Async only helps a handler yield during I/O waits — there is no I/O to wait on here, so true async produces no speedup on the named hot paths.
- Training and dataset loading are already off the request thread (`JobManager`, background threads in `vtsearch/concurrency/`). Async wouldn't change that either.
- The "polling endpoints would benefit" argument only matters under many concurrent clients starving Flask's thread pool. VTSearch is a single-user / small-team tool; threaded Flask (or a modest gunicorn worker count) handles `/api/progress/*` polling without issue.
- Migration cost is non-trivial: per-request context (`g.dataset_context`, `g.detector_context`) set in `before_request`, the `_ProxyDict`/`_ProxyList` proxies in `vtsearch/state/core.py` with their `g`-first / thread-local-fallback logic, `_state_lock` (a sync `RLock`), every blueprint, every `request.json` call — all built around sync Flask semantics. Quart is the cheaper port but still requires auditing every handler and lock; FastAPI is closer to a rewrite.

**Revisit if:** VTSearch grows into a multi-tenant hosted service with hundreds of concurrent pollers, or genuinely I/O-bound routes appear (streaming LLM calls, fan-out to remote vector DBs). Until then, leave as-is.

### 12.15 Structured logging + request IDs ★★ S
Today logs are print-style. JSON logs with `dataset_id`/`detector_id`/`request_id` make production debugging tractable.

### 12.16 Prometheus metrics ★★ S
`/metrics` endpoint with vote count, embedding latency, training time, RAM usage by dataset.

### 12.17 Pydantic models for settings ★ S
The `_SETTING_SPECS` table is clever but custom; Pydantic v2 would generalise it and produce JSON schemas for free.

### 12.18 Concurrency-gate observability ★ XS
The download/embed gates already exist; expose their queue depth in the UI so users can see "3 datasets waiting for embedding".

### 12.19 Richer error surfaces ★★ S
Today most user errors come back as plain JSON. A central error component in the frontend with copy-to-clipboard + relevant context would dramatically improve self-service debugging.

### 12.20 Vector DB optional backend ★ L
For "I have 5M items" use cases — Qdrant/LanceDB as a drop-in `EmbeddingStore` interface. Keep the in-RAM store as default.

---

## 13. ML Improvements

### 13.1 Inclusion-aware loss weighting ★★ XS
Today inclusion adjusts class weights. Try also adjusting the threshold post-hoc (already partially in safe_thresholds) plus focal loss variants.

### 13.2 Hard-negative mining loop ★★ M
After first MLP pass, find unlabelled items closest to the boundary, surface them in `Hard` select mode (already exists but heuristic). Use uncertainty from MLP ensemble §5.6.

### 13.3 Triplet/contrastive fine-tune of embedder ★ L
LoRA-on-embedder using `(anchor=good, positive=good, negative=bad)` triplets sampled from votes. Risk: embedder drift across detectors. Mitigation: per-detector LoRA adapters loaded on demand.

### 13.4 Pseudo-labelling ★ M
Auto-label high-confidence unlabeled items (above e.g. p>0.95), retrain. Classic semi-supervised win.

### 13.5 Active-learning strategy comparison ★★ experiment
We have `Top`/`Hard`/`New` select modes. Add **BALD**, **EIG**, **CoreSet**, **margin** variants. Run as an `eval/` experiment to pick a winner.

### 13.6 Threshold optimisation per metric ★ S
Today threshold is calibrated for accuracy/F1. Allow user to specify desired precision *or* recall and back-solve threshold.

### 13.7 Multi-task heads ★★ M
Share embedder, train multiple MLP heads (one per labelset) at once. Faster than independent training; weak-positive transfer between related labelsets.

### 13.8 Self-supervised continued pretraining ★ XL
Domain-adapt the embedder on the user's unlabeled corpus (MAE/BYOL style). Heavy. Park behind a feature flag.

### 13.9 Distillation ★ M
Train a tiny CNN/MLP to mimic SigLIP scores on user's data. Useful for edge deployment / latency-sensitive batch scoring.

### 13.10 Model evaluation card ★★ S
Per-detector dashboard: precision/recall/F1 from a held-out vote split, calibration plot, confusion matrix, top-K errors. Currently you have to leave the app to get this.

### 13.11 Vote-noise robustness ★★ experiment
Add synthetic label noise, measure detector quality degradation. Inform UI for "warn user when their vote disagrees with a confident model prediction" feature.

### 13.12 Cross-embedder ensembling ★★ M
Train one MLP per embedder, average. Often beats best single embedder. Pairs with the smart-preload manager (§12.4) so every used embedder is already warm.

---

## 14. Evaluation Framework Extensions

### 14.1 Active-learning curves ★★ M
Plot model-quality-vs-vote-count for each select mode. Picks winning strategy quantitatively.

### 14.2 ROC-AUC, PR-AUC, ECE ★★ S
Currently AP / P@k / R@k. Add the standard binary classifier suite.

### 14.3 Calibration plot ★★ S
Reliability diagram. Pairs with calibrated detector §5.7.

### 14.4 Inter-rater agreement ★ M
For multi-user mode (when it lands), compute Cohen's κ / Krippendorff's α between users on shared items.

### 14.5 Embedding-quality probe ★★ M
Linear probe on standard tasks (ImageNet, ESC-50) for any registered embedder. Auto-rank embedders against each other for the user's media type.

### 14.6 Cross-dataset transfer matrix ★ M
Train on dataset A, evaluate on dataset B. Useful for "does my detector generalise?" question.

### 14.7 Cluster-purity ★ S
Diversity tree quality metric: how often do same-label items end up in the same leaf?

### 14.8 Voting-effort ROI ★★ M
"Each vote you cast moves F1 by Δ on average." Surface in UI to motivate users.

---

## 15. Processors / Extractors / Localizers

### 15.1 Audio
- **Speaker diarisation** (`pyannote`) — labels regions by speaker.
- **BPM / key / genre** (`librosa`, `essentia`) — adds metadata for sort-by.
- **Music emotion** (Audionomy / MTG-Jamendo).
- **Loudness LUFS** — for podcast levelling.

### 15.2 Image
- **Depth estimation** (Depth-Anything-v2) — adds a depth thumbnail overlay.
- **Segmentation mask** (SAM2) — pairs with `image_object` clipper.
- **NSFW score** (CLIP-NSFW or custom).
- **Aesthetic score** (LAION-Aesthetics-v2).
- **Colour palette extraction** (k-means in LAB).
- **EXIF / metadata extraction** (camera, GPS, timestamp).
- **Watermark detection**.

### 15.3 Video
- **Action recognition** (VideoMAE labels per clip).
- **Object tracking** (ByteTrack); per-object trail metadata.
- **Scene-change list** — already used by clipper, expose as metadata.

### 15.4 Text
- **Language detection** (langid/fasttext).
- **NER** (spaCy/GLiNER).
- **Sentiment** (siebert/sentiment-roberta).
- **Topic modelling** (BERTopic over the dataset).
- **PII detection** (Presidio) — surface as a flag.

### 15.5 Document
- **Layout analysis** (LayoutLMv3 / Surya).
- **Form extraction** — turn `(field, value)` pairs into searchable metadata.

---

## 16. Collaboration / Multi-user Features

### 16.1 Per-user settings ★★★ M
Already mentioned (§9.4). The single highest-leverage multi-user fix.

### 16.2 Shared workspaces ★★ L
Datasets and detectors can be marked shared with a list of users (the `readers` field already exists in the API). UI for managing shares is missing.

### 16.3 Vote provenance ★★ S
Track which user cast which vote. Pairs with multi-user mode.

### 16.4 Vote conflict resolution ★ M
When two users vote opposite on the same item, surface as a "needs review" queue. Optionally weight by inter-rater reliability.

### 16.5 Activity feed ★ M
"Alice trained X", "Bob loaded Y dataset". Useful in team settings.

### 16.6 @mentions in comments ★ S
Pairs with §11.7 if comments land.

### 16.7 Permissions: read-only viewer role ★ S
For demo or stakeholder accounts that shouldn't accidentally alter labels.

---

## 17. CLI Improvements

### 17.1 `python app.py --list-importers` / `--list-exporters` etc. ★★ XS
Discoverability without grepping source.

### 17.2 Detector input-spec auto-detect ★★ M
Already designed in `docs/design/cli-detector-converter.md`; ship it.

### 17.3 Pipeline file ★ M
`python app.py --pipeline pipeline.yaml` runs an importer → clipper → embedder → detector → exporter sequence declared in YAML. Replaces N CLI flags with one config file. Repeatable for cron.

### ~~17.4 Watch mode ★ S~~ — unnecessary
~~`--watch /path/to/inbox` re-runs autodetect whenever new files appear.~~

Dropped: not worth building. External schedulers (cron, systemd timers, file-watcher daemons like `inotifywait` / `entr`) can re-invoke `--autodetect` on new files without bloating the CLI surface or adding a long-running-process code path to maintain.

### 17.5 Dry-run mode ★ XS ✅ shipped
`--dry-run` prints what would be embedded/scored/exported without doing it. Validates importer/exporter names, settings file, dataset pickle existence, and required CLI field values, but loads no media and trains no models. See [CLI.md § Dry-run mode](../CLI.md#dry-run-mode).

### 17.6 ~~Progress JSON output~~ — **DONE**
`python app.py --autodetect --progress-format json` emits NDJSON on stdout — one event per line, shapes documented in `vtsearch/cli_progress.py`. Errors are routed to the same stream so a single pipe captures the full run; tqdm bars remain on stderr (discard with `2>/dev/null`).

### 17.7 Embedded interactive REPL ★ exploratory
`python app.py --repl` drops into IPython with `medias`/`good_votes`/`detector` already imported. Power-user analysis.

---

## 18. Productionisation / Observability

### 18.1 `/healthz` and `/readyz` ★★ S
Distinguish "process is up" from "models are loaded and DB is reachable".

### 18.2 Backup CLI ★★ S — NOT NECESSARY
`python app.py --backup data-snapshot.tar.gz` and `--restore`. Includes datasets-pkl, settings, detectors.

### 18.3 Audit log ★ M
Append-only log of every label change, detector creation, dataset import. Compliance need for regulated industries.

### 18.4 Rate limiting ★ S
Flask-Limiter on `/api/find-label` and other CPU-heavy endpoints.

### 18.5 API key auth provider ★★ M
A new `LoginProvider` that maps `Authorization: Bearer <key>` → username. For headless integrations.

### 18.6 Container image slim ★ M
`Dockerfile.labbench` already does this for SigLIP-only. Add similarly slim variants per media type.

### 18.7 ARM64 image ★ S
Apple Silicon + Graviton tier deployments.

### 18.8 Model-cache warmer init container ★ S
Compose pattern that pulls model weights once, sidecar reuses.

### 18.9 ~~SSE event stream~~ — **subsumed by §12.5 (DONE)**
The progress feed shipped as `GET /api/events` — see §12.5 and
[`docs/api/events.md`](../api/events.md). An "activity feed" channel
(votes, label changes, etc.) for §16.5 can be added as another event
name on the same endpoint when needed.

---

## 19. Security & Privacy

### 19.1 Auto-blur faces ★ M
Privacy-preserving mode for image datasets. Composes with `image_face` clipper §3.2.

### 19.2 PII redaction in text ★ S
Strip emails/SSNs from text previews based on a Presidio-style detector.

### 19.3 Per-dataset access policy ★★ M
"This dataset can only be loaded by users with role X". Built on existing `readers` field.

### 19.4 Encrypted at rest ★ M
Optional Fernet-encryption for `data/` contents.

### 19.5 Signed URLs for thumbnails ★ S
For multi-user setups so a `/api/media/...` URL can't be guessed by another user.

### 19.6 Content filtering on import ★ S
NSFW/CSAM safety filter integrated into folder-importer pipeline (gated by setting).

### 19.7 SSRF tests already exist — keep adding ★ XS
Webhook exporter, http_archive importer; just hygiene.

---

## 20. Suggested Experiments

These are testable hypotheses that fit naturally into `docs/experiments/` alongside `hac-tree-sweep`.

### 20.1 Embedder bake-off per media type ★★★
For each media type, train detector on the same labels with each registered embedder. Plot mAP / F1 / training time. Output: a recommended-default per type, and a "use this if you care about latency" tier.

### 20.2 MLP architecture sweep ★★
Sweep `(hidden_dim, n_layers, dropout, weight_decay, lr)`. Confirm or refute current `_auto_hidden_dim` heuristic.

### 20.3 Active-learning strategy comparison ★★★
`Top` vs `Hard` vs `New` vs **BALD** vs **CoreSet** vs **margin**. Run on N demo datasets, measure votes-to-target-F1.

### 20.4 Calibration-set-size impact ★
Sweep `calibrate_count` from 1 to 50. Measure threshold stability.

### 20.5 Inclusion bias calibration ★
Sweep inclusion -10..+10 on labelled holdout. Validate that the exponential curve matches user mental model.

### 20.6 Diversity-tree depth/k sweep ★
Same idea as hac-tree-sweep but for the global `DiversityTree`. Measure vote-budget-to-max-coverage trade-off.

### 20.7 Patch-vs-single embedder accuracy ★★
Hold MLP and dataset constant; switch DINOv2-patch ↔ DINOv2-single; measure detector quality.

### 20.8 Cross-embedder ensemble lift ★★
Linearly average detector logits across embedders; report ensemble lift over best single.

### 20.9 Vote-noise robustness ★★
Inject 5/10/20% label flip; measure detector degradation. Justifies any "warn user" UX.

### 20.10 Few-shot prototype vs MLP ★
At what N labels does MLP overtake cosine-prototype detector? Useful as a "use cosine until you have N votes" UX rule.

### 20.11 Reranker stage value ★
With/without BGE-Reranker on top of text-sort retrievals. Measure P@10.

### 20.12 Multilingual-embedder utility ★
On a multi-language demo, mE5 vs E5. Justifies §4.2 priority.

### 20.13 Spectrogram converter cross-modal ★★
Train a SigLIP detector on `audio→image` spectrograms vs CLAP detector on raw audio. Are they complementary in ensemble?

### 20.14 Document-as-image vs document-as-text ★
Page-render-and-SigLIP vs PDF-text-and-E5 for legal/scientific documents. Possibly ensemble both.

### 20.15 Quantised-embedder quality loss ★
int8/binary embedding tradeoff. RAM savings vs detector F1.

### 20.16 Diversity-tree-driven autopilot vs current heuristic ★★
Replace autopilot's diversity step with raw `DiversityTree.next_sample()`. Measure votes-to-coverage.

### 20.17 Calibrator (Platt/Isotonic) impact on UX comprehension ★ user-study
Show same scores, with and without calibration; user-study of which feels more interpretable.

---

## 21. "Spice" Ideas (mildly weird; included for inspiration)

- **"Surprise me"** button: serves the highest-uncertainty media in a random cluster.
- **Daily mix**: 20 items at session start drawn from underexplored clusters.
- **"Why?" overlay**: highlight the patches most responsible for an MLP score (already feasible with patch_grid + saliency).
- **Embedding-space teleport**: click anywhere in a 2D PCA/UMAP plot, jump to nearest media.
- **Voice voting**: say "good"/"bad"/"skip" aloud. Whisper VAD + tiny ASR. Pairs with §11.19.
- **Label-from-screenshot**: take a screenshot of an image elsewhere, paste into VTSearch, instantly text-sort the dataset by it.
- **Notebook export**: dump current `(dataset, detector, votes, scores)` to a Jupyter notebook with reproducible setup.
- **Public detector gallery**: import others' detectors by hash; pairs with HF Datasets §7.1 + §8.1.
- **Voting battle royale**: two users speed-label the same dataset; live leaderboard of disagreements.
- **Auto-generated detector names**: small LLM names a detector from its top-10 positives.
- **Score-over-time chart per item**: as you label more, per-item scores drift; visualise the trace.
- **Auto-generated demo from any folder**: drop a folder, app produces a one-pager showing top clusters, sample items, suggested labelsets.
- **Time-lapse of clustering**: animate `DiversityTree` rebuilds as a dataset grows.
- **Detector "recipe" sharing**: export not just the labels but the full pipeline (embedder, clipper, converter chain, MLP arch) as a single JSON.

---

## 22. Top-10 "if I had to pick" (priority synthesis)

If the goal is maximum user-visible impact for moderate engineering cost, this is roughly where I'd start:

1. **Multi-class / multi-label detectors (§5.1, §11.8)** — biggest expressive-power jump.
2. **Bulk vote + undo (§11.3, §11.4)** — every active user benefits within an hour.
3. **`audio→image` spectrogram converter (§2.1)** — unlocks cross-model ensembling and visual audio search.
4. **Hugging Face Datasets importer + exporter (§7.1, §8.1)** — closes a huge loop.
5. **Cluster / group-by view (§6.5)** — repositions VTSearch from "list with sort" to "true explorer".
6. **OCR converter (§2.2)** + **document-section / token-window clippers (§3.3, §3.5)** — unlock document datasets fully.
7. **Per-user settings (§9.4)** — flagged in HANDOFF; cheap; required for real multi-user.
8. **Mobile-responsive layout (§11.1)** — biggest reach extension.
9. **Embedder bake-off experiment (§20.1)** — informs every default we ship.
10. **OpenAPI + generated TS client (§12.9)** — long-term velocity multiplier.

# VTSearch Brainstorm

*Wide-ranging idea backlog covering new features to build and UX friction to reduce in the current app. Items are tagged with rough priority (★★★ high, ★★ medium, ★ exploratory) and effort (XS / S / M / L / XL). Pick what's interesting and turn the rest into separate plan docs as they mature.*

Sections:

- **Part I — New Capabilities** (1–10)
- **Part II — UX Friction** (11–19)
- **Part III — Architecture, Tooling, Ops** (20–26)
- **Part IV — Experiments & Inspiration** (27–28)

---

# Part I — New Capabilities

## 1. New Media Types

### ~~1.1 Code / source files~~ — Not going to happen

### ~~1.2 Tabular / spreadsheet~~ — Not going to happen

### ~~1.3 3D mesh / point cloud~~ — Not going to happen

### ~~1.4 Time series / signals~~ — Not going to happen

### ~~1.5 Music / MIDI~~ — Not going to happen

### ~~1.6 Spectrogram-as-image~~ — Not going to happen

### ~~1.7 Email / message~~ — Not going to happen

### ~~1.8 Geospatial raster~~ — Not going to happen

### ~~1.9 Web pages / HTML~~ — Not going to happen

### ~~1.10 Bio-signal / EEG~~ — Not going to happen

---

## 2. New MediaConverters

### ~~2.1 `video → text` (transcription)~~ — Not going to happen

### ~~2.2 `text → audio` (TTS preview)~~ — Not going to happen

### ~~2.3 `image → text` (caption)~~ — Not going to happen

### ~~2.4 `image → image` (super-resolution / denoise)~~ — Not going to happen

### ~~2.5 `pdf → table` (table extraction)~~ — Not going to happen

### ~~2.6 `code → text` (docstring extraction)~~ — Not going to happen

### ~~2.7 `video → keyframe-image`~~ — Not going to happen

### ~~2.8 `email → text`~~ — Not going to happen

### 2.9 Composability ★★ M (cross-cutting)
Today the framework allows one converter per source; consider explicit chains: `video → audio → spectrogram → image embedder`. The plumbing already exists in `effective_source_specs`; what's missing is the UI. (See also [clipper-chain.md](clipper-chain.md), which generalises this for clippers.)

### 2.10 `audio → text` (ASR) follow-ups
The base converter ships; the deferred improvements:
- **Switch to `faster-whisper` backend** — CTranslate2-based reimplementation, ~4× faster on CPU at the same WER, supports int8 quantization (small drops from ~480 MB → ~150 MB on disk and ~1 GB → ~500 MB RAM). Would also benefit the `SpeechExtractor` processor. Add as an optional backend the converter picks if available, falling back to `openai-whisper`.
- **`large-v3-turbo` option** — OpenAI's 809M-param distilled variant of large-v3 (~6× faster than large-v3, ~1–2% WER hit). Not exposed today because `openai-whisper` versions vary in which size strings they accept; bundle with the faster-whisper switch since CTranslate2 has stable turbo support.
- **VAD pre-filter** — Whisper hallucinates on silence/music. `faster-whisper`'s built-in Silero VAD filter eliminates most of this; add a `vad_filter: bool` param once the backend supports it.
- **Per-segment timestamps** — current converter flattens to a single transcript blob. For long podcasts, emitting one text media per segment (with `start_s` / `end_s` metadata) would let users vote on individual passages. Defer until there's a UX story for "vote on a 30-second chunk of audio".

---

## 3. New MediaClippers

### 3.1 Audio
- **`sound_beat`** ★ S — split on detected beats / downbeats (madmom). For music datasets.
- **`sound_energy_envelope`** ★ S — segment around energy peaks (find drum hits, claps, gunshots).
- **`sound_speech_activity` follow-ups** — expose advanced Silero knobs (`min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`) if users hit edge cases; reuse the loaded VAD model for `video_dialogue` (§3.4) when that ships.

### 3.2 Image
- **`image_saliency`** ★★ M — crop to salient region (U²-Net or SAM auto-mask). Improves embedding signal on "find dogs" by focusing on subject.
- **`image_face`** ★ S — face-cropped clips (RetinaFace).
- **`image_window`** ★ XS — sliding window with stride, complements existing tiling.
- **`image_color_palette`** ★ exploratory — clip into colour-region masks.
- **`image_object` follow-ups** — experiment with SAM2 boxes for the same UI; consider a per-class confidence threshold instead of one global value if users ask for it.

### 3.3 Text
- ~~**`text_paragraph`** ★★ XS — split on `\n\n`. Free wins on prose.~~ **Shipped** — `TextParagraphClipper` in `vtsearch/media/text/clipper.py`.
- **`text_token_window`** ★★ S — token-aware windows (tiktoken) with overlap.
- **`text_semantic`** ★★ M — embedding-similarity-based chunking (sentence groups whose adjacent cosine drops below threshold).
- **`text_heading`** ★ S — split on Markdown/HTML headings.

### 3.4 Video
- **`video_action`** ★ M — clip around detected action peaks (motion histogram or VideoMAE attention).
- **`video_audio_cue`** ★ M — clip around audio-event peaks (loud noise, vocal onset).
- **`video_dialogue`** ★ M — clip per speech turn (VAD on demuxed audio).

### 3.5 Document
- **`document_section`** ★★ S — split per H1/H2, or per chapter for ePub.
- **`document_page_range`** ★ XS — manual page-window clips for long PDFs.

### 3.6 Cross-cutting
A **`clipper_chain`** abstraction so we can run e.g. `document_section → text_token_window` without writing a custom clipper. See [clipper-chain.md](clipper-chain.md) — Phase 1 in flight.

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
- ~~**VideoMAE-v2** ★ — single-modality but very strong action features.~~ Shipped (`OpenGVLab/VideoMAEv2-Base`, vision-only, mean-pooled patch features, L2-normalised); registered as the `videomae` embedder on the `video` media type.

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

### 9.4 Detector-coupled settings ★ S
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

# Part II — UX Friction

## 11. Auto-fill questions we currently ask

The biggest UX cost in VTSearch right now is the volume of decisions the user makes *before* they see any value. Many of these decisions have a clearly best default and could be auto-filled.

### 11.1 Dataset name from path/folder ★★★ XS — shipped
Almost every dataset importer requires a `name`, but the user almost always wants the folder/file basename. The local-folder importer already derives this in the frontend (`lfDatasetName` in `dataset-importer-modal.component.ts`), but **server_folder**, **server_files**, **http_archive**, and **pickle** importers all leave it blank. Auto-derive name from `os.path.basename(path)`, the URL's last path segment, or the pickle's stem. The field stays editable.

**What shipped:** the `Dataset Name` input is now pre-filled live in every importer view, with a per-importer dirty flag that freezes the value once the user types.

- *server_folder* — its dedicated picker view uses `sfDatasetName` + `sfDerivedDatasetName()` to extract the last segment of the browsed path whenever the user navigates folders.
- *server_files / http_archive / pickle* (generic-form importers) — `formDerivedDatasetName()` in `dataset-importer-modal.component.ts` runs on every source-field change and matches by `field_type`: `url` → last URL path segment with `.zip` / `.tar.gz` / `.tar.bz2` / `.tar.xz` / `.tar` / `.rar` stripped; `server_path` and `file` → basename with extension stripped; a field literally keyed `path` → final non-empty path segment. The pickle file-picker calls `maybeApplyDerivedDatasetName()` directly from `onFileSelected()` so picking a `.pkl` pre-fills the name.
- *Backend fallback* — each importer overrides `default_display_name()` (`server_folder/__init__.py:476`, `server_files/__init__.py:508`, `http_archive/__init__.py:344`, `pickle/__init__.py:91`) so CLI imports and any path that skips the modal still get a sensible default; tests in `tests/io/test_importers.py` cover all four.

### 11.2 Media type from file extensions ★★★ S
Today the user manually picks `media_type` for every folder/file/server import. We already own `vtsearch/media/` extension maps. After the user picks a path or URL, sample the first ~50 entries and auto-select the dominant media type. Show a "Detected: image (47 of 50 files)" hint with a dropdown to override.

### 11.3 Embedder default per media type ★★★ XS
The `embedder` dropdown is populated by `/api/datasets/embedders/<media_type>` but defaults to the first option, which is just whatever Python returns first. The user has no basis to choose between `siglip`, `dinov2_patch`, `dinov3_patch`, etc. Pick a *recommended* embedder per media type (the one used by demos), highlight it in the dropdown, and mark others as `Advanced ▼`.

### 11.4 OS dark mode for `theme` ★★ XS
`theme` defaults to `"dark"` in `_SETTING_SPECS`. On first load, read `prefers-color-scheme` from the browser and store that as the initial value. (See `vtsearch/settings.py`.)

### 11.5 Concurrency limits from hardware ★★ S — shipped
Was: `max_concurrent_dataset_downloads` and `max_concurrent_dataset_embeddings` both defaulted to `1` and sat untouched.

**What shipped:** `vtsearch/embedding/loader.py` exposes two hardware-derived defaults — `default_concurrent_downloads()` returns `max(1, min(4, os.cpu_count() or 1))`, and `default_concurrent_embeddings()` returns `1` on CPU-only boxes or `min(2, torch.cuda.device_count())` when CUDA is available. `ServerSettings` in `vtsearch/settings_models.py` wires them in via `default_factory=` (with lazy imports so settings-model import doesn't pull torch), so the values are computed on first read of an unset key rather than being baked into `data/settings.json` — a manual override in the file always wins.

**Deviation from the brief:** the brief suggested `torch.cuda.mem_get_info()` for VRAM probing; we use `torch.cuda.device_count()` instead. `mem_get_info()` reports *currently free* VRAM, which fluctuates with whatever else is on the GPU; device count is stable and gives the same "one task per visible device, capped at 2" behaviour without depending on box state at startup.

### 11.6 Detector media type from selected dataset ★★★ XS
New-detector modal forces a `media_type` pick. If the user already has a dataset selected on the dashboard, pre-fill it (already partially done — extend to *lock* and gray-out unless they explicitly unlock).

### 11.7 Output filenames with timestamps ★★ XS — shipped
`server_csv_file` / `server_json_file` exporters now default to `data/autodetect_results_{YYYYMMDD-HHMMSS}.{csv,json}`, so consecutive runs no longer silently overwrite. `vtsearch/exporters/_template.py` resolves `{YYYYMMDD-HHMMSS}` (UTC), `{detector_name}`, and `{username}`, with the latter two routed through `sanitize_template_value` so user-controlled values cannot escape the admin-implied directory. Covered by `tests/io/test_csv_webhook_exporters.py`.

### 11.8 Demo embedder from prior demo ★ XS
Demo dataset picker re-asks for embedder every time. Remember the last embedder used for each media type per user (cheap: piggyback on per-user settings).

### 11.9 Calibrate count from dataset size ★ S
`calibrate_count` defaults to a constant. For tiny datasets (< 200 items) it can be too big; for huge datasets it can be too small. Auto-scale: `min(50, max(10, len(medias) // 20))`.

### 11.10 Detector name from query/seed ★ XS
The new-detector modal asks for a `name` and a text/media seed. If the user typed a seed query first, pre-fill `name` with it (sanitized). They almost always type the same thing twice today.

### 11.11 Importer category from URL/path ★ S
If the user pastes a URL into the importer modal, jump them directly to `http_archive`. If they paste a server path, jump to `server_folder` (or `server_files` if it ends in `.txt`/`.npz`). Today they hunt for the right tab first.

---

## 12. Hints to add to the UI

The labels exist but they read like API names. The user needs micro-copy that explains *what* and *why*.

### 12.1 Empty-state guidance on the dashboard ★★★ S
The empty dashboard shows two empty tables and a row of disabled buttons. Add a first-run banner:
> "Welcome — load a dataset to get started. Try the **Demo** tab for a one-click example, or **Local Folder** to use your own files."
with a "Load demo dataset" CTA that opens the importer pre-pinned to Demo. (See `dashboard.component.html`.)

### 12.2 First-vote tooltip in label view ★★★ XS — shipped
Was: when the labeling view opens with zero votes, overlay a faint hint near the Good/Bad buttons. Dismiss on first vote, persist dismiss in user settings.

**What shipped:** the voting overlay (`frontend/src/app/components/center-panel/voting-overlay/`) now renders a faint `Use ← / → or click. Autopilot will find the next question.` hint above the Good/Bad buttons whenever the parent `vt-center-panel` reports zero votes and the user hasn't previously dismissed it. The center panel subscribes to `voteState.goodVotes$` / `badVotes$`, so the hint also retires when a vote arrives via the left-panel hover-vote or grid click — not just via the center buttons. Dismiss state persists as a new per-user setting `label_hint_dismissed` (`vtsearch/settings_models.py:UserSettings`, default `False`), wired through `AppSettingsSchema` / `SettingsUpdateSchema` and the PUT `/api/settings` dispatch map.

### 12.3 Explainer below jargon settings ★★★ S
`enrich_descriptions`, `safe_thresholds`, `calibrate_count`, `calibration_fraction`, the autopilot phase thresholds — none of these have any explainer in the settings modal. Add a one-sentence helper below each, the way Mac System Settings does. Pull from the docstrings already in `settings_models.py` / `settings.py`.

### 12.4 "What is an embedder?" tooltip ★★ XS
Add a `?` icon next to the embedder dropdown in every importer that exposes one. Hover shows a 2-line definition + link to docs. Same treatment for *clipper*, *detector*, *labelset*, *autorun*, *diversity tree*, *inclusion*.

### 12.5 Inline format hints for path/URL fields ★★ XS
- `paths_file` (server_files importer): show "Accepts `.txt`, `.list`, or `.npz`. One path per line, or a NumPy archive of pre-computed vectors."
- CSV label importer: show the expected column schema (`md5,label`) with a sample.
- JSON label importer: show a 3-line sample of the expected structure.

### 12.6 Loading-state context in the progress modal ★★ S — shipped
Was: dashboard loading rows showed cryptic step numbers like "[Step 3/4] Loading embedding model…" with no plain-English context for what step 3 actually meant.

**What shipped:** `frontend/src/app/utils/format-progress.ts` gained `formatProgressHeader(progress, kind, embedder?)`, which returns a `{ header, subtitle, detail }` triple derived from the event's `status` + `message` (+ optional `embedder` woven into model-load / embedding-files / embedding-labels subtitles). The dataset card, detector card, and orphan-task row in `dashboard.component.html` were restructured to render the header line ("Loading dataset · embedding model") and a one-line subtitle ("Loading SigLIP weights. First-time only — cached on disk afterwards.") above the existing progress bar; the `[Step S/T]` prefix is stripped from the detail because the header now conveys the phase. Phase vocabulary covers the dataset-load flow (downloading source / unpacking archive / embedding model / warming text encoder / embedding files / slicing clips / embedding clips / converting media / removing duplicates / building diversity index / saving to registry) and the detector-load flow (restoring labels / seeding examples / embedding labels). Embedder names are pretty-printed via a small lookup table (`siglip → SigLIP`, `clap → LAION-CLAP`, `xclip → X-CLIP`, etc.) so the subtitle matches user-facing terminology.

**Open follow-ups:** byte-rate / ETA on the first-run model download (§14.2) and per-file embedding progress (§14.1) remain — the header is now in place but the bar itself still shows a single indeterminate spinner inside `embedding model` and `embedding files`. Folding those in will make the subtitle's "First-time only — cached on disk afterwards." reassurance land harder.

### 12.7 Inclusion slider tick labels ★★ XS
The slider is `[-10, +10]` with no anchors. Add tick labels: `-10 strict` / `0 default` / `+10 lenient`, plus a one-line caption "Trades off precision (left) vs recall (right)."

### 12.8 Autopilot phase intent ★★ S — shipped
The collapsed Autopilot bar shows four dots. Hovering should reveal phase intent: *"Phase 3: Boundary refinement — votes on uncertain items train the model fastest."* Already exists in long-form docs, but not in UI.

**What shipped:** `AutopilotPanelComponent` now computes a `phaseIntent(phase, stepNumber)` string of the form *"Phase N: <short name> — <why this phase matters>"* for each of the four phases (plus a "done" variant), and binds it as the `title` tooltip on every collapsed-mode dot and every expanded-view step label. The active dot/label tooltip leads with the same intent and appends *"Click to reselect recommendation."* so the existing affordance is preserved. Spec coverage added for both the collapsed dots and the expanded labels.

### 12.9 Keyboard shortcut discoverability ★★ XS
The keyboard help modal exists but is only reachable via a button most users never click. Show shortcuts inline as tooltips on the Good/Bad buttons (`Good (→)`, `Bad (←)`), and surface "press `?` for keyboard help" as a one-time toast after the third labeling session.

### 12.10 Region-vote affordance ★ S
Region voting requires holding `Shift`. When a patch-region embedder is detected for the current dataset, show a thin info strip above the centre panel: *"Hold Shift to draw a region. Releases vote good on that region only."* (A marquee-mode toggle button complements but doesn't replace this hint.)

### 12.11 Cross-dataset scoring warning ★★ S
When the user selects Dataset B + Detector trained on Dataset A and clicks Find/Train, today nothing flags this. Show a non-blocking note: *"This detector was trained on a different dataset (Dataset A). Scoring will still work but may be less accurate."*

### 12.12 What "smart"/"stable" mean ★★ XS — shipped
The labeling status bar shows colored dots for `smart` and `stable`. Add a hover tooltip explaining each ("Smart: the model fits your votes consistently. Stable: predictions stopped shifting between retrains.").

**What shipped:** plain-English `title` tooltips on every Smart/Stable/Diverse indicator the user sees:
- *Labeling status bar.* `ProgressIndicatorsComponent` exposes `smartTooltip`, `stableTooltip`, and `spanTooltip` getters that lead with the plain-English meaning, append live subtext (cost / flips / level) when present, and end with the green/yellow/red legend. Wired into the `[title]` attribute on each `.labeling-indicator` button.
- *Autopilot mini-icons.* The matching mini-icons (`.ap-status-icon`) shown next to the active step's detail during the boundary/diversity phases previously bound `title` to the bare ariaLabel (`Smart: green`). The `StatusIcon` shape now carries a richer `title` field populated by `phaseStatusIcons()` with the same explanatory text style, and the template binds it. Spec coverage extended to assert the explanation is present.

---

## 13. Pauses we can speed up

Latency surfaces are the second-biggest UX cost. Some are real (model training); others are spurious (synchronous downloads in request handlers).

### 13.1 Eliminate first-vote retrain stall ★★★ M — shipped
Voting calls `train_and_score()` synchronously in the request handler (`routes/sorting.py`). For >100 labels this can block 5–10s per vote. Move retraining to a background job (mirror `learned_sort_jobs`) and return the *previous* score map immediately, then push an updated sort over SSE when the new model is ready. The user keeps voting on stale scores for ~5s instead of staring at a spinner. Effort: M because we need to coalesce rapid votes and decide when the live UI shows new scores.

**What shipped:** training was lifted off the vote path before this item was written. The vote endpoints (`/api/medias/<id>/vote`, `/api/detectors/<name>/labels/<id>/vote`) only toggle state and return `{"ok": true}` — they never called `train_and_score()`. Retraining lives behind `/api/learned-sort`, which hands off to the `learned_sort_jobs` `JobManager` and returns a `job_id` immediately (`vtsearch/routes/sorting.py:356-487`); the manager keeps one running + one pending slot and coalesces rapid retrains in-place (latest signature wins) so a burst of votes collapses to one extra training run after the current one finishes (`vtsearch/concurrency/async_jobs.py:125-199`). A signature cache short-circuits the no-op case so re-sorting without new votes is free (`vtsearch/routes/sorting.py:417-419`). The front-end debounces with a 300ms `scheduleLearnedSort()` (`frontend/src/app/components/label-view/label-view.component.ts:651-658`) and never clears `sortState.sortOrder` while the job runs, so the user keeps voting against the previous score map — only the bottom progress-indicators bar swaps to a "Training…" status (the media list and vote inputs are not gated by `sortBusy`). One deviation from the original text: the client polls `/api/learned-sort/result` every 500ms (`label-view.component.ts:500-522`) rather than receiving the new sort over SSE — UX-equivalent, different transport. Ship commits: `ca166ab1 Move learned-sort + eval train-and-score to background jobs` and `82183bd9 Coalesce rapid learned-sort requests into a single pending slot`.

### 13.2 Eager-preload the next-likely embedder ★★ S
`predict_embedders_to_preload()` already runs at startup. Extend it to also fire when:
- The user selects a media type on the importer form (preload that media type's default embedder).
- The user selects a dataset row on the dashboard (preload its embedder so the Train click is instant).

### 13.3 Skip the demo-picker double round-trip ★★ XS
Picking a demo today: open importer → pick Demo tab → pick demo card → fill embedder → submit → wait for download. Most users want the *recommended* setup. Add a one-click "Quick load" button on each demo card that uses the recommended embedder and skips the params form.

### 13.4 Parallel-load multiple selected datasets ★★ S — backend ready, frontend pending
Original brief: *"Selecting 3 datasets and clicking the bulk-load action loads them serially due to default concurrency of 1. The `_download_gate` already supports concurrent loads; bump the default (see §11.5) and most users immediately see 3x faster bulk loads."*

The "default concurrency of 1" premise is now stale: §11.5 shipped hardware-derived defaults, so `_download_gate` already lets `max(1, min(4, os.cpu_count() or 1))` loads run in parallel (typically 2–4 on user boxes) and `_embed_gate` already lets `min(2, torch.cuda.device_count())` loads embed in parallel on multi-GPU hosts. The gate also reads its limit fresh on every `acquire()` (`vtsearch/datasets/load_pipeline.py:31-32, 77-78`), so a user-bumped setting takes effect immediately for queued tasks. Test coverage at limit=1 and limit=2 lives in `tests/datasets/test_parallel_loading.py:587-918`.

What's actually missing is the *bulk-load action*. The dashboard supports multi-select (`selectedDatasetIds: Set<string>` in `frontend/src/app/components/dashboard/dashboard.component.ts`) and renders side-action buttons for **Combine selected** and **Delete selected** (`dashboard.component.html:150-168`), but there is no **Load selected** counterpart — the only load path is the per-row `▶` button calling `loadDataset()` (`dashboard.component.ts:796-800` → `POST /api/datasets/registry/<id>/load`).

**Open follow-ups:**
- Add a "Load selected" side-action button next to Combine/Delete in `dashboard.component.html`, plus a `loadSelectedDatasets()` method that fires `loadRegistered(id)` for every entry of `selectedDatasetIds` in a tight loop (no need for a new bulk endpoint — the existing per-id endpoint plus `_download_gate` give the parallelism for free). Disable when `selectedDatasetIds.size === 0` or every selected dataset is already loaded.
- No new backend tests needed; add a frontend spec asserting the new button calls `loadRegistered` once per selected id.

### 13.5 Don't block voting on labelset-source export ★★ XS — shipped
`LabelsetSource.sync_to_labelset_source()` runs synchronously on every vote change to push to the external store. For slow targets (webhook, slow disk) this stalls the vote. Run it in a debounced background thread (200ms debounce coalesces rapid voting bursts).

**What shipped:** `sync_to_labelset_source()` (`vtsearch/labels/sync.py`) now schedules a `threading.Timer` keyed by `detector_id` that fires after `_DEBOUNCE_DELAY` (200ms) and runs the actual push on a background thread. Rapid calls inside the window cancel the prior timer and overwrite the captured contexts (user / dataset ctx / detector ctx), so a burst of votes collapses to one save with the latest state. Two new helpers: `flush_pending_label_syncs()` drains the queue synchronously (used by tests and available for graceful shutdown) and `reset_label_sync_for_tests()` cancels pending pushes without running them (wired into conftest's `reset_state` fixture so a sync scheduled by test A can't fire after test A's contexts are gone). The `_workers_lock` serializes the worker against flush so a mid-write push is waited out instead of racing with the assertion. Five new tests in `tests/io/test_sync_sources.py` cover: non-blocking scheduling (slow `save` doesn't stall the caller), 20-call burst → 1 write coalesce, per-detector keying (A's vote doesn't cancel B's pending push), latest-state-wins semantics, and reset-drops-without-writing.

### 13.6 Lazy-create per-media-type panel preferences ★ XS
First time a user opens a new media type, the panel settings (`view_mode_*`, `grid_icon_size_*`, `focus_mode_*`, `panel_pct_*`) all write to disk. Coalesce into one save. (Minor but the first-image-open feels janky on slow disks.)

### 13.7 Skip diversity-tree rebuild on small updates ★ M
When a few medias are added/removed (e.g. clip fix-up), today the whole diversity tree rebuilds. For incremental changes < 1% of dataset size, do an incremental insert/delete instead. Saves seconds on every clip-aware import.

### 13.8 Async embedder warm-up after import ★ XS — shipped
After a dataset loads, the "warming up text encoder…" step blocks task completion. Move it to fire-and-forget so the dataset is usable for grid-browsing immediately and Text sort just waits on first use.

**What shipped:** the synchronous `_warmup_embedder_stage` was replaced with a fire-and-forget `_warmup_embedder_async(media_dict)` daemon thread (`vtsearch/datasets/load_pipeline.py:800`). Both load paths — the importer-driven `_run_origin_load_in_background` and the registry-driven `load_registered_dataset` (`vtsearch/routes/datasets/registry.py`) — now kick off the warm-up after `_register_and_migrate` / `_reg_add_loaded` and return immediately, so the dashboard row goes green the moment the dataset is in memory. `load_registered_dataset`'s `_LOAD_STEPS` dropped from 3 to 2 (read pickle + build diversity index). The warmup itself still calls `emb.load_models()` then `emb.embed_text("warmup")` on a daemon thread named `warmup-embedder`; if the user clicks Text Sort before warmup finishes, the existing `_embedder_load_lock` in `vtsearch/routes/sorting.py:107` serialises the wait behind the regular "Loading embedder…" sort-progress bar — no race, no double-load (`load_models` is idempotent via `_model_load_lock`, `vtsearch/media/embedder.py:551-555`). The now-orphaned `_load_embedder_with_progress` / `_load_embedder_for_clips` helpers in `vtsearch/datasets/load_pipeline.py` were deleted (the routes/sorting variant of the same name is a separate function).

---

## 14. Long processes to clarify

Where speed isn't possible, *perceived* speed comes from honest progress.

### 14.1 Per-file progress during embedding ★★★ M
The dataset-load progress reports step 1-4 ("downloading", "embedding", "deduping", "diversity tree") but inside step 2 (embedding) the user sees a single bar that's stuck at "embedding…" for minutes. The embedder already iterates per-file; thread an `on_progress(current, total, filename)` callback through `MediaEmbedder.embed_*` so the modal shows "Embedding 437 / 1284: kitchen-mic-02.wav". This is the single highest-impact clarity fix.

### 14.2 Bytes/sec for first-run model downloads ★★★ M
First-run model downloads (CLAP ~1.1 GB, X-CLIP ~600 MB) currently show "Loading embedding model…" with no bar. HuggingFace's `tqdm`-style progress is available — pipe it through `update_progress()` to show `Downloading SigLIP (412 / 860 MB, 18 MB/s, ~25s left)`.

### 14.3 ETA estimates on long bars ★★ S — shipped

Every progress bar now shows a remaining-time estimate once it has been running long enough for the rate to mean something.

**What shipped:** `ProgressTracker._compute_eta` in `vtsearch/concurrency/progress.py` records a per-phase start time, computes `raw = (elapsed / completed) * (total - current)` once `elapsed > 5s` with `current > 0` and a known `total`, and smooths it with an EMA (α = 0.3) against the previous sample. The phase clock resets whenever `status` or `total` changes, or when `current` decreases (a new bar starting), so phase transitions don't pollute the estimate with stale rate from the previous phase. The result lives in a new `eta_seconds` field added to `_PROGRESS_COMMON_EXTRAS`, so every singleton tracker (dataset / sort / eval / find) and every per-task tracker created by `LoadingTasksTracker` carries it for free.

On the frontend, `ProgressEvent.eta_seconds` lands on `frontend/src/app/models/api.models.ts`, and `formatEta()` + a tail in `formatProgressMessage()` in `frontend/src/app/utils/format-progress.ts` render it as `· ~Hh Mm left` / `· ~Mm Ss left` / `· ~Ss left` appended to the existing `(current/total) message` detail line. Because the dashboard cards, find view, label view, and detector card all flow through `formatProgressMessage` / `formatProgressHeader.detail`, no per-component change is needed — the ETA chip appears automatically on every long-running bar that has a `current` and `total`. Short bars (≤5s) keep showing only `(C/T)` so they don't flash a meaningless estimate.

### 14.4 Per-detector progress during auto-detect ★★ S
`/api/auto-detect` runs N detectors in parallel and reports a single aggregated bar. Switch to a list of mini-bars in `AutodetectResultsModalComponent` ("Detector A: ✓ done · Detector B: 47% · Detector C: queued"). The frontend already gets per-detector results; just expose progress per-id over SSE.

### 14.5 Per-dataset progress during multi-load ★★ S
When the user bulk-loads 3 datasets, the dashboard shows 3 stacked task rows but the SSE channel emits a single aggregate. Tag each progress event with `dataset_id` so each row's bar moves independently.

### 14.6 Cancel buttons everywhere ★★★ S
`learned_sort_jobs`, `eval_jobs`, and auto-detect all support `cancel()` in the backend but the UI doesn't expose it. Add a small X button next to every running progress bar. (Dataset cancel already works — use it as the pattern.)

### 14.7 Voting iterations: progress breakdown ★ S
Eval voting-iterations modal shows `step X/Y`. Add a sub-line "(dataset 2 of 5: gtzan, category 3 of 4: jazz)" so the user knows what's currently running.

### 14.8 First-run banner about model downloads ★★ XS
On the very first import of any media type, prepend the progress modal with a one-shot info strip: *"First time loading audio — VTSearch will download the CLAP model (~1.1 GB). This happens once and is cached locally."* Dismiss for that media type forever.

### 14.9 Stream training fold-level progress ★ M
`train_and_score()` does N folds + optional safe-threshold blending. Surface fold-level progress through the existing SSE `sort` channel. Especially valuable during long autopilot phase 3 retrains.

### 14.10 Replace indeterminate spinners with named phases ★★ XS
Several spinners say "Loading…" with no context (export modal, label-importer-modal, find sort modal). Pass the current operation name to the spinner: "Exporting 142 labels to CSV…", "Importing labels from server CSV…".

---

## 15. Confusing UI to streamline

These are surfaces where the *labels exist*, but the mental model is broken or the controls duplicate-and-conflict.

### 15.1 Importer category vs importer type two-level tabs ★★★ M
The dataset importer modal has *category tabs* and then *type subtabs*, with the same importer sometimes appearing in two places. Most users can't tell `Local Folder` from `Local Files`. Flatten to a single-level grid of importer cards with badges (`📁 folder` / `📄 files` / `🌐 url` / `📦 archive` / `▶ demo`). Same UI works for label and processor importers.

### 15.2 Blank-vs-trained tabs in new-detector modal ★★ S
"Blank" and "Trained" are labels for the developer, not the user. Rename to **Start with examples** vs **Import a trained model**. Lift the embedder/media-type pickers above the tabs so they're shared.

### 15.3 The three sort radio buttons ★★★ M
The Manual mode shows `Text` / `Learned` / `Load` as radio buttons. Rename to make intent obvious:
- `Text` → **Search** (with a magnifying glass icon)
- `Learned` → **Use my votes** (with a thumb icon)
- `Load` → **Use a saved detector** (with an open-folder icon)
And collapse `Learned` into a passive state of `Search`: once you've voted, the search results re-rank silently by your votes. The user never thinks "should I switch sort modes?"

### 15.4 Twin panel-settings asymmetry ★★ S
Left and right panel each have independent `view_mode`, `focus_mode`, `grid_icon_size`, `panel_pct`. Most users want them in sync. Default to mirrored, add a "Mirror left/right" toggle (on by default), and only show the second column when the toggle is off.

### 15.5 "Inclusion" vs "threshold" ★★ S
The slider is labeled `Inclusion` but most users think in terms of confidence/threshold. Either rename to **Threshold (precision ↔ recall)** with the same scale, or replace with a confidence-based slider (0–1) directly. Inclusion is internal jargon.

### 15.6 Selection-strategy buttons (Top/Hard/New) ★★ M
In Manual mode these are 3 unlabeled jargon buttons. Either hide them by default (only Autopilot users need them) or rename: `Top` → **Most likely match**, `Hard` → **Most uncertain**, `New` → **Most novel**.

### 15.7 The "Load" sort mode is buried ★★ S
"Load" requires a `+` click that opens another modal where the user picks a detector. Instead expose recently-used detectors as a dropdown in the sort row, with a `Manage…` link for the modal. (Mirrors how Word/Photoshop handle recent files.)

### 15.8 Vote-pile right panel ★ S
The right panel shows "Good" and "Bad" as two stacked stacks. There's no drag-to-reorder, no batch operations (`select all → un-vote`), and no obvious way to remove an item from a pile (must reopen the centre, find it, vote the other way). Add multi-select + a context menu (remove, re-vote, copy ID).

### 15.9 Crop modal optionality ★ S
After picking an example media, a crop modal appears even if the user wants to use the full file. Add a clear "Use full file" button alongside "Crop and confirm", and skip the modal entirely for text and audio < 5s.

### 15.10 "Achievements" tab discoverability ★ XS
Achievements live in Settings, where users go for *settings*, not gamification rewards. Either move to its own menu item or a small trophy icon in the header.

### 15.11 Disabled-button reasons hidden ★★ XS
Train / Find buttons disable for non-obvious reasons (media-type mismatch, nothing selected, etc.) with a hidden hint that's `visibility:hidden`. Show the reason inline at all times — disabled buttons should always say *why*. (Pattern: GitHub's merge button.)

---

## 16. Inconsistencies to normalize

### 16.1 "Detector" vs "Model" vs "Classifier" ★★ XS
Code uses `detector`. Dashboard table is labeled "Detectors". Some UI elements (sort modes, the Models dashboard column) call them "models". User Guide uses both interchangeably. Pick one (suggest **detector** to keep alignment with the codebase) and lint the frontend strings.

### 16.2 Path-style fields ★★ S
The "where do I put this file?" concept appears across plugins as `filepath`, `paths_file`, `path`, `url`, `file`. Plus the underlying field types differ (`server_path`, `text`, `file`). Standardize labels: **Save to (server path)**, **Path or URL**, **Upload a file**.

### 16.3 Toast / banner / inline error styling ★★ S
Errors appear in at least three styles: red banner inline, modal-level red text, console-log only. Add a single toast service and route all `error` SSE events + HTTP failure responses through it.

### 16.4 Saved-state indicator ★ XS
Settings auto-save but show no "saved" feedback. Some forms (export modal) require an explicit Save. Pick a convention: either always auto-save with a tiny `✓ saved` indicator, or always require explicit Save with a Cancel.

### 16.5 Embedder display names ★★ XS
The dropdown shows raw IDs (`siglip`, `dinov3_patch`, `e5`). Map each to a human label (`SigLIP (general images)`, `DINOv3 patch (region-aware)`, `E5 (text)`). Keep the raw ID as a secondary `<small>` line for power users.

### 16.6 Destructive-confirm follow-ups
The `confirmDestructive` standard shipped, with two leftovers:
- *Delete label entry* and *Clear votes* don't have any UI surface today (only the `clearVotes()` API client exists, with no caller). When those actions get a button, wire them through `confirmDestructive` — e.g. `"Clear all votes for detector 'X'? This deletes every saved label for this model and cannot be undone."`
- The destructive primary button has no distinct danger styling yet — it reuses `.btn--primary`. A red variant would make the modal even harder to dismiss-by-accident.

---

## 17. Non-standard UIs to normalize

VTSearch has a few interactions that are clever but un-Google-able — replace with conventions users already know.

### 17.1 Stripe histogram in left panel ★★ M
The mini-histogram below the media list lets users click to jump to a score range. Most users never figure this out. Either remove or replace with a standard horizontal slider that filters the list. (Slider supports drag-to-zoom-window, is keyboard-friendly, and is familiar from price filters.)

### 17.2 Hover-to-reveal delete confirmation ★★ XS
Hover-only confirmation for destructive actions is unfamiliar and fragile. Replace with a standard modal confirm (matches §16.6).

### 17.3 Resize-cursor on small drag handles ★ XS
Panel dividers and column-resize handles are ~2px wide. Make them 8px hit targets with a `cursor: col-resize` on hover (standard).

### 17.4 Folder-browser Phase 2
The unified `<vt-folder-browser>` shipped (Phase 1). Phase 2 remaining:
- **Left sidebar with Pinned / Bookmarks / Recent.** *Pinned* anchors derived from `SERVER_ROOTS`, `saved_datasets_dir`, and `detectors_dir`; *Bookmarks* the user explicitly stars via a ⭐ button on the breadcrumb (new per-user setting key `bookmarked_browse_paths: list[str]`); *Recent* MRU of the last 8 visited folders (new per-user setting key `recent_browse_paths: list[str]`). Settings would slot into `vtsearch/settings.py`'s per-user tier using the existing `_SETTING_SPECS` factory.
- **Address-bar mode.** Click the breadcrumb whitespace to turn it into an editable text input pre-filled with the current path; `Enter` navigates to whatever the user typed. Mirrors the macOS Finder Cmd+Shift+G / GNOME Files Ctrl+L pattern.
- **New-folder button.** Not currently exposed by the backend (`/api/browse` is read-only). Would need a `POST /api/browse/mkdir` route guarded by the same path-validation chain.

### 17.5 Drag-and-drop in the examples-editor ★ XS
The reusable `vt-drop-zone` shipped in the main importer + new-detector flows. The Edit Examples → + Add Good / + Add Bad widget still uses small button-driven file inputs. Same component can drop in next to or in place of those buttons.

### 17.6 Region rectangle interaction ★ S
After drawing, the rectangle is editable via 8 handles — good. But the "click on the rectangle to restore" interaction is non-discoverable. A standard "✓ confirm region" / "✗ clear" button overlay on the rectangle would replace the current "press ← twice to discard" pattern.

### 17.7 Sort-bar "+" to add a sort source ★ XS
The `+` icon to load a saved detector for sort is non-standard. Replace with a labeled button **"Load saved detector"** in the sort dropdown.

---

## 18. Long flows to shorten

Workflows the user *can* do today but that take too many steps and clicks.

### 18.1 First-time dataset → labelled export ★★★ M
Today: open menu → pick importer category → pick importer → fill form → pick media type → pick embedder → submit → wait → close modal → select dataset → click "New detector" → fill form → click Train → vote 7 items → export → pick exporter → fill form → submit. That's ~15 clicks before the user has anything to show.
**Compressed flow:** "Quick start" CTA on empty dashboard → pick a media type → upload a folder → app auto-creates a detector with the folder's name and drops the user into the labeling view. Export becomes a single header button with a recent-target fallback.

### 18.2 Cross-dataset training with a re-used labelset ★★ M
The labelset-source machinery lets a detector pull labels from a different dataset, but using it requires:
1. Add a labelset source to detector A (configure plugin, write filepath template).
2. Vote on dataset X.
3. Load dataset Y.
4. Train detector A.
Compress to a "Use these labels on another dataset" button on the right panel.

### 18.3 Re-running auto-detect after edits ★ S
Tweaking a label, then re-running auto-detect, is currently: edit → save → navigate to dashboard → re-pick dataset+detector → click Find → wait → reopen results modal. Add a "Re-run with current settings" button inside the existing results modal.

### 18.4 Pre-computed embedding import follow-ups
The base `.npz` flow shipped for `server_folder`, `server_files`, `local_folder`, `local_files`. Open:
- `http_archive`: still no `.npz` option. Matching filenames inside an extracted archive is hairy and rarely useful, so it was deliberately left out — revisit if a user asks.
- Server-side `.npz` field for `server_folder` is a plain text input; could be upgraded to the same server-path browser used for the folder picker if the typing friction shows up in testing.

### 18.5 Configure & test a webhook exporter ★ M
Currently: open export modal → pick webhook → fill URL → fill auth → submit → realize the URL was wrong → repeat. Add a "Send test ping" button next to the URL field that fires a single test payload.

### 18.6 Combining detectors ★ M
The "combine detectors" feature exists but requires multi-select + a non-obvious icon button. Surface as a clear "Merge detectors" CTA with a preview of what the merged detector would look like (count, intersection vs union choice, name).

### 18.7 Audio-segment-to-detector follow-ups
The right-click "Use as detector seed" menu shipped for left-panel (click focus-mode) items. Open:
- **Video / document / text crop overlays.** The crop modal only supports audio + image today; the right-click menu hides the crop entries for everything else. If we want time-range cropping on video or page-range cropping on documents, the bounded clippers and `vt-media-crop-modal` overlay both need new variants.
- **Context-pulldown parity.** The right-click menu only exists on left-panel items; right-panel labelset / label items still bind right-click to vote-good. Worth considering if we want detector-card-style actions there too.

### 18.8 Resuming a labelling session ★★ S
There's no "recent sessions" surface. Add a "Recent sessions" list on the dashboard (dataset + detector pair + last activity timestamp) so the user gets back into work in one click.

### 18.9 Bulk-importing multiple folders at once ★ M
Currently the server-folder importer is one folder per import job. Allow multi-folder selection (server folder browser + multi-select) and create one dataset per folder in the same job.

### 18.10 "I want a detector exactly like this one, but trained from scratch" ★ XS
Useful for experimentation. Add a "Clone" action in the detector row that duplicates the labelset (or labelset source) but resets the trained model.

---

## 19. UI feature additions

Larger UI bets that aren't pure friction reduction.

### 19.1 Mobile-responsive layout ★★★ XL
HANDOFF + frontend audit confirm: desktop-only today. Phone & tablet support is the single biggest reach extension. Touch-swipe to vote falls out for free.

### 19.2 Keyboard-shortcut overlay ★★ XS
`?` opens a help sheet listing every binding. Fixes onboarding fragility.

### 19.3 Undo last vote ★★★ XS
`Cmd/Ctrl-Z`. Mis-clicks happen constantly when speed-labelling.

### 19.4 Bulk vote ★★★ S
Shift-click range select on the list. "Mark all 14 of these as good." Hugely speeds up obvious-class labelling.

### 19.5 Multi-select + lasso (grid) ★★ M
Drag-to-select on grid view.

### 19.6 Comparison mode ★★ M
Side-by-side A/B on two media items. For tie-breaking similar scores or model-vs-model output.

### 19.7 Comments / notes per item ★★ S
Free-text per-media note. Round-trips through label export. Useful for "why I voted bad on this one".

### 19.8 Tag/multi-label system ★★★ L
First-class multi-label support orthogonal to the binary detector. A media item can have any subset of `{"speech", "outdoor", "echoey"}`. Pairs with multi-class detector §5.1.

### 19.9 Saved views / saved searches ★★ S
"My top-10 hits for detector X on dataset Y" as a bookmark.

### 19.10 Vote history scrubber ★ S
Rewind through last N votes; see how your distribution shifted as labels grew.

### 19.11 Achievements polish ★ XS
Service exists; surface it more (toasts, weekly recap).

### 19.12 Onboarding tour ★★ M
Driver.js / Shepherd-style overlay walking new users through Dashboard → Load → Vote → Train. Today the first-time experience drops you in deep.

### 19.13 Region-of-interest voting UI ★★ M
Backend already supports patch grids and v2 of patch-embedder.md shipped Shift-drag + the marquee toggle. Remaining polish lives in §17.6 (rectangle confirm/clear overlay).

### 19.14 Audio loop A-B / waveform clips ★ S
Set A and B markers, loop between them. For close auditioning.

### 19.15 PDF page jump + highlight ★ S
For document review.

### 19.16 Drag-and-drop import directly to dashboard ★★ S
Skip the modal. Drop a folder → start importing.

### 19.17 Recent items / history pane ★ S
Persistent across sessions.

### 19.18 Light improvements
- Empty-state illustrations for "no datasets yet" ★ XS
- Skeleton loaders (today: spinners) ★ S
- Per-detector colour accent ★ XS

### 19.19 Voice annotations ★ exploratory M
Hold a key, dictate a note that's transcribed (Whisper) and attached to the media. For mobile-style fast review.

### 19.20 Sketch search ★ exploratory M
Draw a sketch, use it as image-sort query. CLIP+sketch models exist.

---

# Part III — Architecture, Tooling, Ops

## 20. Internal / Architectural Improvements

### 20.1 Streaming embeddings (lazy) ★★ L
Today everything is in RAM. Memory-mapped or DB-backed embedding store (DuckDB / LanceDB / Qdrant) so we can hold 1M items. Important for the HF / S3 importers (§7) to be useful.

### 20.2 GPU batched embedding follow-ups
The image + text + clip-re-embed bulk paths shipped. Remaining:
- **Audio CLAP + CLAP-Music bulk override.** Decode is the bottleneck and adds I/O complexity; smaller GPU win than image but still meaningful for big audio imports. `librosa` is happy to decode a list serially while the model batches.
- **Video X-CLIP / LanguageBind bulk override.** Tricky because X-CLIP at batch 32 with 8 frames each is ~640 MB of activations and can OOM on 8 GB cards. Likely wants a smaller default `embed_batch_size` (e.g. 8) on the video embedders.
- **Fuse single-vector + patch forward on DINOv2/DINOv3/EUPE.** Today the backbone runs twice per image (once for `embed_media_bulk`, once for `patch_forward_bulk`). Fusing requires changing the loader to call a single combined hook and split the outputs — worth it if profiling shows the backbone forward is the dominant cost.

### 20.3 Mixed-precision training ★ XS
`torch.cuda.amp` for the MLP; trivial change.

### 20.4 Background prefetch of next likely media ★ S
For speed-labelling, preload the next 3 items' previews.

### 20.5 Resume interrupted training ★ S
Checkpoint MLP state every N epochs.

### 20.6 Python client library ★★ M
`pip install vtsearch-client` so notebooks can drive the same endpoints headlessly.

### 20.7 Ruff format CI gate ★ XS
We have ruff; add a `./run-tests.sh` step that fails on unformatted code. (Format is currently `ruff format --check` in the lint phase — confirm coverage and tighten if needed.)

### 20.7.1 Burn down the C901 noqa list ★ S (opportunistic)
60 legacy functions still carry `# noqa: C901` markers (down from 77 — see `git grep "# noqa: C901"`). Each one is a candidate for incremental refactoring; the markers can be deleted as functions are simplified under complexity 10. *Burned down so far:* `multi_find` and `find_check_labels` (the two worst offenders, CC 62 and 22, broken into named helpers in `vtsearch/routes/detectors/find.py`); `load_dataset_from_folder` / `load_dataset_from_folder_chunked` / `load_dataset_from_pickle` / `load_dataset_from_pickle_chunked` (CC 50/51/37/29 — collapsed onto a shared per-file helper layer in `vtsearch/datasets/loader_folder.py` and `loader_pickle.py`); `import_local_folder` (36 → 11 — extracted `_parse_clipper_params`, `_save_uploaded_files_to_temp`, `_read_optional_vectors_file`, `_build_local_folder_field_values`, `_extract_clipper_config`, `_make_local_folder_loader`); `run_converters_on_folder` (34 → 12) and `apply_converter_to_demo` (24 → 7) in `vtsearch/converters/runner.py` — collapsed onto shared `_build_converted_media_dict` / `_emit_converted_outputs` helpers; `demo_dataset_list` (31 → 10 — extracted `_initial_demo_status`, `_downgrade_for_mismatch`, `_calculate_demo_num_files`, `_calculate_demo_download_size_mb`); the 2026-05 D-grade sweep tracked in [c901-refactor-triage.md](c901-refactor-triage.md) — eight functions (`_run_pipeline` 31→3, `export_labels` 29→5, `_apply_clip_and_embed` 25→6, `resolve_label_embeddings` 25→6, `Audio2ImageMediaConverter.convert` 25→8, `Image2TextMediaConverter.convert` 22→7, `label_file_sort` 21→9, `populate_label_embeddings` 21→9), all dropped under the threshold by splitting into named per-stage helpers. *Worst offenders remaining:* `DatasetImporter.effective_source_specs` (27), `load_pipeline_file` (26), `_resolve_or_train_detector` (25), `combine_detectors` (24), `load_demo_dataset` (23), `sync_labels_to_loaded_detector` (23), `train_and_score` (23), `_eval_cached_models` (22), `CombineDatasetsImporter.run` (21). The 2026-05 triage marked five of those as **Skip** (complexity is honest dispatch — see [c901-refactor-triage.md](c901-refactor-triage.md) for the rationale); the rest deserve a fresh triage pass. No deadline — refactor opportunistically when touching the code.

### 20.7.2 Periodic pre-commit autoupdate ★ XS
Run `pre-commit autoupdate` on a quarterly cadence so pinned hook versions don't drift too far from the latest releases.

### 20.7.3 Coverage-delta gate ★ S (deferred)
If we ever want a coverage-delta gate again it would have to live in `./run-tests.sh` and run against `git merge-base origin/dev HEAD`. Not a priority — the local opt-in coverage report (`VTSEARCH_COVERAGE=1 ./run-tests.sh`) is enough for now.

### 20.8 Structured logging + request IDs ★★ S
Today logs are print-style. JSON logs with `dataset_id`/`detector_id`/`request_id` make production debugging tractable.

### 20.9 Pydantic models for settings ★ S
The `_SETTING_SPECS` table is clever but custom; Pydantic v2 would generalise it and produce JSON schemas for free.

### 20.10 Concurrency-gate observability ★ XS
The download/embed gates already exist; expose their queue depth in the UI so users can see "3 datasets waiting for embedding".

### 20.11 Richer error surfaces ★★ S
Today most user errors come back as plain JSON. A central error component in the frontend with copy-to-clipboard + relevant context would dramatically improve self-service debugging.

### 20.12 Vector DB optional backend ★ L
For "I have 5M items" use cases — Qdrant/LanceDB as a drop-in `EmbeddingStore` interface. Keep the in-RAM store as default.

---

## 21. ML Improvements

### 21.1 Inclusion-aware loss weighting ★★ XS
Today inclusion adjusts class weights. Try also adjusting the threshold post-hoc (already partially in safe_thresholds) plus focal loss variants.

### 21.2 Hard-negative mining loop ★★ M
After first MLP pass, find unlabelled items closest to the boundary, surface them in `Hard` select mode (already exists but heuristic). Use uncertainty from MLP ensemble §5.6.

### 21.3 ~~Triplet/contrastive fine-tune of embedder~~ ★ L — **won't do (2026-05-19)**
LoRA-on-embedder using `(anchor=good, positive=good, negative=bad)` triplets sampled from votes. Risk: embedder drift across detectors. Mitigation: per-detector LoRA adapters loaded on demand.

**Decision: won't do.** Adapting the embedder backbone clashes with VTSearch's "frozen embedder, swappable detector head" design — per-detector LoRA adapters are effectively persisted learned weights, which the project explicitly avoids (see CLAUDE.md "No Persisted Vectors or MLPs"). The expected gain over the current frozen-embedder + MLP path doesn't justify the new artefact lifecycle, the per-detector apply-on-demand plumbing, or the training cost. Pursue §21.2 (hard-negative mining) and §21.4 (pseudo-labelling) instead for label-driven quality wins.

### 21.4 Pseudo-labelling ★ M
Auto-label high-confidence unlabeled items (above e.g. p>0.95), retrain. Classic semi-supervised win.

### 21.5 Active-learning strategy comparison ★★ experiment
We have `Top`/`Hard`/`New` select modes. Add **BALD**, **EIG**, **CoreSet**, **margin** variants. Run as an `eval/` experiment to pick a winner.

### 21.6 Threshold optimisation per metric ★ S
Today threshold is calibrated for accuracy/F1. Allow user to specify desired precision *or* recall and back-solve threshold.

### 21.7 Multi-task heads ★★ M
Share embedder, train multiple MLP heads (one per labelset) at once. Faster than independent training; weak-positive transfer between related labelsets.

### 21.8 Self-supervised continued pretraining ★ XL
Domain-adapt the embedder on the user's unlabeled corpus (MAE/BYOL style). Heavy. Park behind a feature flag.

### 21.9 Distillation ★ M
Train a tiny CNN/MLP to mimic SigLIP scores on user's data. Useful for edge deployment / latency-sensitive batch scoring.

### 21.10 Model evaluation card ★★ S — **Won't do**
~~Per-detector dashboard: precision/recall/F1 from a held-out vote split, calibration plot, confusion matrix, top-K errors. Currently you have to leave the app to get this.~~ Declined 2026-05-19: not pursuing an in-app evaluation dashboard; users who need these metrics can keep computing them out-of-app from the existing label export.

### 21.11 Vote-noise robustness ★★ experiment
Add synthetic label noise, measure detector quality degradation. Inform UI for "warn user when their vote disagrees with a confident model prediction" feature.

### 21.12 Cross-embedder ensembling ★★ M
Train one MLP per embedder, average. Often beats best single embedder. Pairs with the smart-preload manager so every used embedder is already warm.

---

## 22. Evaluation Framework Extensions

### 22.1 Active-learning curves ★★ M
Plot model-quality-vs-vote-count for each select mode. Picks winning strategy quantitatively.

### 22.2 ROC-AUC, PR-AUC, ECE ★★ S
Currently AP / P@k / R@k. Add the standard binary classifier suite.

### 22.3 Calibration plot ★★ S
Reliability diagram. Pairs with calibrated detector §5.7.

### 22.4 Inter-rater agreement ★ M
For multi-user mode (when it lands), compute Cohen's κ / Krippendorff's α between users on shared items.

### 22.5 Embedding-quality probe ★★ M
Linear probe on standard tasks (ImageNet, ESC-50) for any registered embedder. Auto-rank embedders against each other for the user's media type.

### 22.6 Cross-dataset transfer matrix ★ M
Train on dataset A, evaluate on dataset B. Useful for "does my detector generalise?" question.

### 22.7 Cluster-purity ★ S
Diversity tree quality metric: how often do same-label items end up in the same leaf?

### 22.8 Voting-effort ROI ★★ M
"Each vote you cast moves F1 by Δ on average." Surface in UI to motivate users.

---

## 23. Processors / Extractors / Localizers

### 23.1 Audio
- **Speaker diarisation** (`pyannote`) — labels regions by speaker.
- **BPM / key / genre** (`librosa`, `essentia`) — adds metadata for sort-by.
- **Music emotion** (Audionomy / MTG-Jamendo).
- **Loudness LUFS** — for podcast levelling.

### 23.2 Image
- **Depth estimation** (Depth-Anything-v2) — adds a depth thumbnail overlay.
- **Segmentation mask** (SAM2) — pairs with `image_object` clipper.
- **NSFW score** (CLIP-NSFW or custom).
- **Aesthetic score** (LAION-Aesthetics-v2).
- **Colour palette extraction** (k-means in LAB).
- **EXIF / metadata extraction** (camera, GPS, timestamp).
- **Watermark detection**.

### 23.3 Video
- **Action recognition** (VideoMAE labels per clip).
- **Object tracking** (ByteTrack); per-object trail metadata.
- **Scene-change list** — already used by clipper, expose as metadata.

### 23.4 Text
- **Language detection** (langid/fasttext).
- **NER** (spaCy/GLiNER).
- **Sentiment** (siebert/sentiment-roberta).
- **Topic modelling** (BERTopic over the dataset).
- **PII detection** (Presidio) — surface as a flag.

### 23.5 Document
- **Layout analysis** (LayoutLMv3 / Surya).
- **Form extraction** — turn `(field, value)` pairs into searchable metadata.

---

## 24. Collaboration / Multi-user Features

### 24.1 Per-user settings ★★★ M
Today settings are global even with multi-user auth (HANDOFF.md flags this). Move to per-user settings file under `get_user_data_dir()`.

### 24.2 Shared workspaces ★★ L
Datasets and detectors can be marked shared with a list of users (the `readers` field already exists in the API). UI for managing shares is missing.

### 24.3 Vote provenance ★★ S
Track which user cast which vote. Pairs with multi-user mode.

### 24.4 Vote conflict resolution ★ M
When two users vote opposite on the same item, surface as a "needs review" queue. Optionally weight by inter-rater reliability.

### 24.5 Activity feed ★ M
"Alice trained X", "Bob loaded Y dataset". Useful in team settings. (Can be added as a new SSE channel on the existing `/api/events` endpoint.)

### 24.6 @mentions in comments ★ S
Pairs with §19.7 if comments land.

### 24.7 Permissions: read-only viewer role ★ S
For demo or stakeholder accounts that shouldn't accidentally alter labels.

---

## 25. CLI Improvements

### 25.1 `python app.py --list-importers` / `--list-exporters` etc. ★★ XS
Per-family convenience aliases over the existing `--list-plugins --plugin-family <name>`. Discoverability without grepping source.

### 25.2 Detector input-spec auto-detect ★★ M
Already designed in `docs/design/cli-detector-converter.md`; ship it.

### 25.3 Embedded interactive REPL ★ exploratory
`python app.py --repl` drops into IPython with `medias`/`good_votes`/`detector` already imported. Power-user analysis.

---

## 26. Productionisation / Observability

### 26.1 `/healthz` and `/readyz` ★★ S
Distinguish "process is up" from "models are loaded and DB is reachable".

### 26.2 Audit log ★ M
Append-only log of every label change, detector creation, dataset import. Compliance need for regulated industries.

### 26.3 Rate limiting ★ S
Flask-Limiter on `/api/find-label` and other CPU-heavy endpoints.

### 26.4 API key auth provider ★★ M
A new `LoginProvider` that maps `Authorization: Bearer <key>` → username. For headless integrations.

### 26.5 Container image slim ★ M
`Dockerfile.labbench` already does this for SigLIP-only. Add similarly slim variants per media type.

### 26.6 ARM64 image ★ S
Apple Silicon + Graviton tier deployments.

### 26.7 Model-cache warmer init container ★ S
Compose pattern that pulls model weights once, sidecar reuses.

---

## 27. Security & Privacy

### 27.1 Auto-blur faces ★ M
Privacy-preserving mode for image datasets. Composes with `image_face` clipper §3.2.

### 27.2 PII redaction in text ★ S
Strip emails/SSNs from text previews based on a Presidio-style detector.

### 27.3 Per-dataset access policy ★★ M
"This dataset can only be loaded by users with role X". Built on existing `readers` field.

### 27.4 Encrypted at rest ★ M
Optional Fernet-encryption for `data/` contents.

### 27.5 Signed URLs for thumbnails ★ S
For multi-user setups so a `/api/media/...` URL can't be guessed by another user.

### 27.6 Content filtering on import ★ S
NSFW/CSAM safety filter integrated into folder-importer pipeline (gated by setting).

### 27.7 SSRF tests already exist — keep adding ★ XS
Webhook exporter, http_archive importer; just hygiene.

---

# Part IV — Experiments & Inspiration

## 28. Suggested Experiments

These are testable hypotheses that fit naturally into `docs/experiments/` alongside `hac-tree-sweep`.

### 28.1 Embedder bake-off per media type ★★★
For each media type, train detector on the same labels with each registered embedder. Plot mAP / F1 / training time. Output: a recommended-default per type, and a "use this if you care about latency" tier.

### 28.2 MLP architecture sweep ★★
Sweep `(hidden_dim, n_layers, dropout, weight_decay, lr)`. Confirm or refute current `_auto_hidden_dim` heuristic.

### 28.3 Active-learning strategy comparison ★★★
`Top` vs `Hard` vs `New` vs **BALD** vs **CoreSet** vs **margin**. Run on N demo datasets, measure votes-to-target-F1.

### 28.4 Calibration-set-size impact ★
Sweep `calibrate_count` from 1 to 50. Measure threshold stability.

### 28.5 Inclusion bias calibration ★
Sweep inclusion -10..+10 on labelled holdout. Validate that the exponential curve matches user mental model.

### 28.6 Diversity-tree depth/k sweep ★
Same idea as hac-tree-sweep but for the global `DiversityTree`. Measure vote-budget-to-max-coverage trade-off.

### 28.7 Patch-vs-single embedder accuracy ★★
Hold MLP and dataset constant; switch DINOv2-patch ↔ DINOv2-single; measure detector quality.

### 28.8 Cross-embedder ensemble lift ★★
Linearly average detector logits across embedders; report ensemble lift over best single.

### 28.9 Vote-noise robustness ★★
Inject 5/10/20% label flip; measure detector degradation. Justifies any "warn user" UX.

### 28.10 Few-shot prototype vs MLP ★
At what N labels does MLP overtake cosine-prototype detector? Useful as a "use cosine until you have N votes" UX rule.

### 28.11 Reranker stage value ★
With/without BGE-Reranker on top of text-sort retrievals. Measure P@10.

### 28.12 Multilingual-embedder utility ★
On a multi-language demo, mE5 vs E5. Justifies §4.2 priority.

### 28.13 Spectrogram converter cross-modal ★★
Train a SigLIP detector on `audio→image` spectrograms vs CLAP detector on raw audio. Are they complementary in ensemble?

### 28.14 Document-as-image vs document-as-text ★
Page-render-and-SigLIP vs PDF-text-and-E5 for legal/scientific documents. Possibly ensemble both.

### 28.15 Quantised-embedder quality loss ★
int8/binary embedding tradeoff. RAM savings vs detector F1.

### 28.16 Diversity-tree-driven autopilot vs current heuristic ★★
Replace autopilot's diversity step with raw `DiversityTree.next_sample()`. Measure votes-to-coverage.

### 28.17 Calibrator (Platt/Isotonic) impact on UX comprehension ★ user-study
Show same scores, with and without calibration; user-study of which feels more interpretable.

---

## 29. "Spice" Ideas (mildly weird; included for inspiration)

- **"Surprise me"** button: serves the highest-uncertainty media in a random cluster.
- **Daily mix**: 20 items at session start drawn from underexplored clusters.
- **"Why?" overlay**: highlight the patches most responsible for an MLP score (already feasible with patch_grid + saliency).
- **Embedding-space teleport**: click anywhere in a 2D PCA/UMAP plot, jump to nearest media.
- **Voice voting**: say "good"/"bad"/"skip" aloud. Whisper VAD + tiny ASR. Pairs with §19.19.
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

## Top picks (priority synthesis)

If the goal is maximum user-visible impact for moderate engineering cost, this is roughly where I'd start:

1. **Multi-class / multi-label detectors (§5.1, §19.8)** — biggest expressive-power jump.
2. **Bulk vote + undo (§19.3, §19.4)** — every active user benefits within an hour.
3. **Hugging Face Datasets importer + exporter (§7.1, §8.1)** — closes a huge loop.
4. **Cluster / group-by view (§6.5)** — repositions VTSearch from "list with sort" to "true explorer".
5. **Document-section / token-window clippers (§3.3, §3.5)** — unlock document datasets fully.
6. **Per-user settings (§24.1)** — flagged in HANDOFF; cheap; required for real multi-user.
7. **Mobile-responsive layout (§19.1)** — biggest reach extension.
8. **Embedder bake-off experiment (§28.1)** — informs every default we ship.
9. **Per-file embedding progress (§14.1)** + **bytes/sec for downloads (§14.2)** — biggest "is it hung?" fix.
10. **Active-context UX one-two: friendlier sort-mode naming (§15.3) + cancel buttons everywhere (§14.6)**.

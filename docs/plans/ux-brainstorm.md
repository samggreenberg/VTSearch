# VTSearch UX Brainstorm

*A wide-ranging audit of friction points in the VTSearch web app, with concrete suggestions for auto-fill, hints, speed-ups, clarity improvements, streamlining, and consistency fixes.*

Items are tagged with rough priority (★★★ high impact, ★★ medium, ★ exploratory) and effort (XS / S / M / L / XL). File paths are concrete so each idea can be turned into its own plan doc.

The structure mirrors the original questions:

1. [Questions we can auto-fill](#1-questions-we-can-auto-fill)
2. [Hints we should add to the UI](#2-hints-we-should-add-to-the-ui)
3. [Pauses we can speed up](#3-pauses-we-can-speed-up)
4. [Long processes we can make clearer](#4-long-processes-we-can-make-clearer)
5. [Confusing UI to streamline](#5-confusing-ui-to-streamline)
6. [Inconsistencies across the app](#6-inconsistencies-across-the-app)
7. [Non-standard UIs to normalize](#7-non-standard-uis-to-normalize)
8. [Long-but-possible flows to shorten](#8-long-but-possible-flows-to-shorten)
9. [Quick wins (prioritized)](#9-quick-wins-prioritized)

---

## 1. Questions we can auto-fill

The biggest UX cost in VTSearch right now is the volume of decisions the user makes *before* they see any value. Many of these decisions have a clearly best default and could be auto-filled.

### 1.1 Dataset name from path/folder ★★★ XS
Almost every dataset importer requires a `name`, but the user almost always wants the folder/file basename. The local-folder importer already derives this in the frontend (`lfDatasetName` in `dataset-importer-modal.component.ts`), but **server_folder**, **server_files**, **http_archive**, and **pickle** importers all leave it blank. Auto-derive name from `os.path.basename(path)`, the URL's last path segment, or the pickle's stem. The field stays editable.

### 1.2 Media type from file extensions ★★★ S
Today the user manually picks `media_type` for every folder/file/server import. We already own `vtsearch/media/` extension maps. After the user picks a path or URL, sample the first ~50 entries and auto-select the dominant media type. Show a "Detected: image (47 of 50 files)" hint with a dropdown to override.

### 1.3 Embedder default per media type ★★★ XS
The `embedder` dropdown is populated by `/api/datasets/embedders/<media_type>` but defaults to the first option, which is just whatever Python returns first. The user has no basis to choose between `siglip`, `dinov2_patch`, `dinov3_patch`, etc. Pick a *recommended* embedder per media type (the one used by demos), highlight it in the dropdown, and mark others as `Advanced ▼`.

### 1.4 OS dark mode for `theme` ★★ XS
`theme` defaults to `"dark"` in `_SETTING_SPECS`. On first load, read `prefers-color-scheme` from the browser and store that as the initial value. (See `vtsearch/settings.py`.)

### 1.5 Concurrency limits from hardware ★★ S
`max_concurrent_dataset_downloads` and `max_concurrent_dataset_embeddings` both default to 1, then sit untouched. On startup, detect cores/VRAM via `torch.cuda.mem_get_info()` (already available in `embedding/loader.py`) and set sensible defaults — e.g. downloads = `min(4, cpu_count)`, embeddings = `1` if CPU-only else `min(2, gpu_count)`. Bake this into the spec's default callable, not a one-time write.

### 1.6 Detector media type from selected dataset ★★★ XS
New-detector modal forces a `media_type` pick. If the user already has a dataset selected on the dashboard, pre-fill it (already partially done — extend to *lock* and gray-out unless they explicitly unlock).

### 1.7 Output filenames with timestamps ★★ XS
`server_csv_file` / `server_json_file` exporters default to `data/autodetect_results.csv` — running twice silently overwrites. Default to `data/autodetect_results_{YYYYMMDD-HHMMSS}.csv` with `{detector_name}` and `{username}` template support (matches the existing `LabelsetSource` placeholder pattern).

### 1.8 Demo embedder from prior demo ★ XS
Demo dataset picker re-asks for embedder every time. Remember the last embedder used for each media type per user (cheap: piggyback on per-user settings).

### 1.9 Calibrate count from dataset size ★ S
`calibrate_count` defaults to a constant. For tiny datasets (< 200 items) it can be too big; for huge datasets it can be too small. Auto-scale: `min(50, max(10, len(medias) // 20))`.

### 1.10 Clipper default from media type + duration ★ M
Most users don't know what a clipper is. For audio, default to `null_clipper` for clips < 30s and `overlap_chunks` for longer. For video, default to `chunks` with frame-aware step size. The "Clipper" button can hide entirely until the user opens `Advanced`.

→ Promoted to [smart-clipper-defaults.md](smart-clipper-defaults.md). Phase 1 (per-dataset "Auto (recommended)" picker entry) shipped; Phase 2 (per-media routing via MediaClipper options) deferred.

### 1.11 Detector name from query/seed ★ XS
The new-detector modal asks for a `name` and a text/media seed. If the user typed a seed query first, pre-fill `name` with it (sanitized). They almost always type the same thing twice today.

### 1.12 Importer category from URL/path ★ S
If the user pastes a URL into the importer modal, jump them directly to `http_archive`. If they paste a server path, jump to `server_folder` (or `server_files` if it ends in `.txt`/`.npz`). Today they hunt for the right tab first.

---

## 2. Hints we should add to the UI

The labels exist but they read like API names. The user needs micro-copy that explains *what* and *why*.

### 2.1 Empty-state guidance on the dashboard ★★★ S
The empty dashboard shows two empty tables and a row of disabled buttons. Add a first-run banner:
> "Welcome — load a dataset to get started. Try the **Demo** tab for a one-click example, or **Local Folder** to use your own files."
with a "Load demo dataset" CTA that opens the importer pre-pinned to Demo. (See `dashboard.component.html`.)

### 2.2 First-vote tooltip in label view ★★★ XS
When the labeling view opens with zero votes, overlay a faint hint near the Good/Bad buttons: *"Use ← / → or click. Autopilot will guide you."* Dismiss on first vote, persist dismiss in user settings.

### 2.3 Explainer below jargon settings ★★★ S
`enrich_descriptions`, `safe_thresholds`, `calibrate_count`, `calibration_fraction`, the autopilot phase thresholds — none of these have any explainer in the settings modal. Add a one-sentence helper below each, the way Mac System Settings does. Pull from the docstrings already in `settings_models.py` / `settings.py`.

### 2.4 "What is an embedder?" tooltip ★★ XS
Add a `?` icon next to the embedder dropdown in every importer that exposes one. Hover shows a 2-line definition + link to docs. Same treatment for *clipper*, *detector*, *labelset*, *autorun*, *diversity tree*, *inclusion*.

### 2.5 Inline format hints for path/URL fields ★★ XS
- `paths_file` (server_files importer): show "Accepts `.txt`, `.list`, or `.npz`. One path per line, or a NumPy archive of pre-computed vectors."
- CSV label importer: show the expected column schema (`md5,label`) with a sample.
- JSON label importer: show a 3-line sample of the expected structure.

### 2.6 Loading-state context in the progress modal ★★ S
Today the dataset loading modal shows step numbers like "[Step 3/4] Loading embedding model…". The user has no model for what step 3 means. Add a header `Loading dataset · embedding model` and a subtitle with a one-liner like "Downloading SigLIP weights (~860 MB). First-time only — cached afterwards."

### 2.7 Inclusion slider tick labels ★★ XS
The slider is `[-10, +10]` with no anchors. Add tick labels: `-10 strict` / `0 default` / `+10 lenient`, plus a one-line caption "Trades off precision (left) vs recall (right)."

### 2.8 Autopilot phase intent ★★ S
The collapsed Autopilot bar shows four dots. Hovering should reveal phase intent: *"Phase 3: Boundary refinement — votes on uncertain items train the model fastest."* Already exists in long-form docs, but not in UI.

### 2.9 Keyboard shortcut discoverability ★★ XS
The keyboard help modal exists but is only reachable via a button most users never click. Show shortcuts inline as tooltips on the Good/Bad buttons (`Good (→)`, `Bad (←)`), and surface "press `?` for keyboard help" as a one-time toast after the third labeling session.

### 2.10 Region-vote affordance ★ S
Region voting requires holding `Shift`. There is no visual hint of this. When a patch-region embedder is detected for the current dataset, show a thin info strip above the centre panel: *"Hold Shift to draw a region. Releases vote good on that region only."*

### 2.11 Cross-dataset scoring warning ★★ S
When the user selects Dataset B + Detector trained on Dataset A and clicks Find/Train, today nothing flags this. Show a non-blocking note: *"This detector was trained on a different dataset (Dataset A). Scoring will still work but may be less accurate."*

### 2.12 What "smart"/"stable" mean ★★ XS
The labeling status bar shows colored dots for `smart` and `stable`. Add a hover tooltip explaining each ("Smart: the model fits your votes consistently. Stable: predictions stopped shifting between retrains.").

---

## 3. Pauses we can speed up

Latency surfaces are the second-biggest UX cost. Some are real (model training); others are spurious (synchronous downloads in request handlers).

### 3.1 Eliminate first-vote retrain stall ★★★ M
Voting calls `train_and_score()` synchronously in the request handler (`routes/sorting.py`). For >100 labels this can block 5–10s per vote. Move retraining to a background job (mirror `learned_sort_jobs`) and return the *previous* score map immediately, then push an updated sort over SSE when the new model is ready. The user keeps voting on stale scores for ~5s instead of staring at a spinner. Effort: M because we need to coalesce rapid votes and decide when the live UI shows new scores.

### 3.2 Eager-preload the next-likely embedder ★★ S
`predict_embedders_to_preload()` already runs at startup. Extend it to also fire when:
- The user selects a media type on the importer form (preload that media type's default embedder).
- The user selects a dataset row on the dashboard (preload its embedder so the Train click is instant).

### 3.3 Skip the demo-picker double round-trip ★★ XS
Picking a demo today: open importer → pick Demo tab → pick demo card → fill embedder → submit → wait for download. Most users want the *recommended* setup. Add a one-click "Quick load" button on each demo card that uses the recommended embedder and skips the params form.

### 3.4 Parallel-load multiple selected datasets ★★ S
Selecting 3 datasets and clicking the bulk-load action loads them serially due to default concurrency of 1. The `_download_gate` already supports concurrent loads; bump the default (see §1.5) and most users immediately see 3x faster bulk loads.

### 3.5 Don't block voting on labelset-source export ★★ XS
`LabelsetSource.sync_to_labelset_source()` runs synchronously on every vote change to push to the external store. For slow targets (webhook, slow disk) this stalls the vote. Run it in a debounced background thread (200ms debounce coalesces rapid voting bursts).

### 3.6 Lazy-create per-media-type panel preferences ★ XS
First time a user opens a new media type, the panel settings (`view_mode_*`, `grid_icon_size_*`, `focus_mode_*`, `panel_pct_*`) all write to disk. Coalesce into one save. (Minor but the first-image-open feels janky on slow disks.)

### 3.7 Don't re-embed text queries on every keystroke ★★ XS — **shipped**
Was: 400ms debounced live-search on the Text-sort input (`sort-bar.component.ts`) firing `POST /api/sort` → `embed_text_query()` on every pause, with no caching frontend or backend.

Now:
- Frontend: live-search removed. The input only triggers a search on Enter or the new "Search" button (disabled while the trimmed query is empty). See `frontend/src/app/components/left-panel/sort-bar/sort-bar.component.{ts,html,scss}`.
- Backend: 32-entry in-memory LRU in `vtsearch/embedding/helpers.py` keyed by `(embedder_name, media_type, enrich, text)`. Repeat queries — re-submitting the same string, toggling sort modes and back, or `eval/runner.py` calls — skip the text encoder. Cache is process-scoped (no persistence, per "No Persisted Vectors") and cleared between tests via the existing `reset_state` autouse fixture.

### 3.8 Skip diversity-tree rebuild on small updates ★ M
When a few medias are added/removed (e.g. clip fix-up), today the whole diversity tree rebuilds. For incremental changes < 1% of dataset size, do an incremental insert/delete instead. Saves seconds on every clip-aware import.

### 3.9 Async embedder warm-up after import ★ XS
After a dataset loads, the "warming up text encoder…" step blocks task completion. Move it to fire-and-forget so the dataset is usable for grid-browsing immediately and Text sort just waits on first use.

---

## 4. Long processes we can make clearer

Where speed isn't possible, *perceived* speed comes from honest progress.

### 4.1 Per-file progress during embedding ★★★ M
The dataset-load progress reports step 1-4 ("downloading", "embedding", "deduping", "diversity tree") but inside step 2 (embedding) the user sees a single bar that's stuck at "embedding…" for minutes. The embedder already iterates per-file; thread an `on_progress(current, total, filename)` callback through `MediaEmbedder.embed_*` so the modal shows "Embedding 437 / 1284: kitchen-mic-02.wav". This is the single highest-impact clarity fix.

### 4.2 Bytes/sec for first-run model downloads ★★★ M
First-run model downloads (CLAP ~1.1 GB, X-CLIP ~600 MB) currently show "Loading embedding model…" with no bar. HuggingFace's `tqdm`-style progress is available — pipe it through `update_progress()` to show `Downloading SigLIP (412 / 860 MB, 18 MB/s, ~25s left)`.

### 4.3 ETA estimates on long bars ★★ S
Every progress bar should show an ETA once we've seen >5s and have a current/total. ETA = `(elapsed / current) * (total - current)`, smoothed. Eliminates the "is this hung?" fear.

### 4.4 Per-detector progress during auto-detect ★★ S
`/api/auto-detect` runs N detectors in parallel and reports a single aggregated bar. Switch to a list of mini-bars in `AutodetectResultsModalComponent` ("Detector A: ✓ done · Detector B: 47% · Detector C: queued"). The frontend already gets per-detector results; just expose progress per-id over SSE.

### 4.5 Per-dataset progress during multi-load ★★ S
When the user bulk-loads 3 datasets, the dashboard shows 3 stacked task rows but the SSE channel emits a single aggregate. Tag each progress event with `dataset_id` so each row's bar moves independently.

### 4.6 Cancel buttons everywhere ★★★ S
`learned_sort_jobs`, `eval_jobs`, and auto-detect all support `cancel()` in the backend but the UI doesn't expose it. Add a small X button next to every running progress bar. (Dataset cancel already works — use it as the pattern.)

### 4.7 Voting iterations: progress breakdown ★ S
Eval voting-iterations modal shows `step X/Y`. Add a sub-line "(dataset 2 of 5: gtzan, category 3 of 4: jazz)" so the user knows what's currently running.

### 4.8 First-run banner about model downloads ★★ XS
On the very first import of any media type, prepend the progress modal with a one-shot info strip: *"First time loading audio — VTSearch will download the CLAP model (~1.1 GB). This happens once and is cached locally."* Dismiss for that media type forever.

### 4.9 Stream training fold-level progress ★ M
`train_and_score()` does N folds + optional safe-threshold blending. Surface fold-level progress through the existing SSE `sort` channel. Especially valuable during long autopilot phase 3 retrains.

### 4.10 Replace indeterminate spinners with named phases ★★ XS
Several spinners say "Loading…" with no context (export modal, label-importer-modal, find sort modal). Pass the current operation name to the spinner: "Exporting 142 labels to CSV…", "Importing labels from server CSV…".

---

## 5. Confusing UI to streamline

These are surfaces where the *labels exist*, but the mental model is broken or the controls duplicate-and-conflict.

### 5.1 Importer category vs importer type two-level tabs ★★★ M
The dataset importer modal has *category tabs* and then *type subtabs*, with the same importer sometimes appearing in two places. Most users can't tell `Local Folder` from `Local Files`. Flatten to a single-level grid of importer cards with badges (`📁 folder` / `📄 files` / `🌐 url` / `📦 archive` / `▶ demo`). Same UI works for label and processor importers.

### 5.2 Blank-vs-trained tabs in new-detector modal ★★ S
"Blank" and "Trained" are labels for the developer, not the user. Rename to **Start with examples** vs **Import a trained model**. Lift the embedder/media-type pickers above the tabs so they're shared.

### 5.3 The three sort radio buttons ★★★ M
The Manual mode shows `Text` / `Learned` / `Load` as radio buttons. Rename to make intent obvious:
- `Text` → **Search** (with a magnifying glass icon)
- `Learned` → **Use my votes** (with a thumb icon)
- `Load` → **Use a saved detector** (with an open-folder icon)
And collapse `Learned` into a passive state of `Search`: once you've voted, the search results re-rank silently by your votes. The user never thinks "should I switch sort modes?"

### 5.4 Twin panel-settings asymmetry ★★ S
Left and right panel each have independent `view_mode`, `focus_mode`, `grid_icon_size`, `panel_pct`. Most users want them in sync. Default to mirrored, add a "Mirror left/right" toggle (on by default), and only show the second column when the toggle is off.

### 5.5 "Inclusion" vs "threshold" ★★ S
The slider is labeled `Inclusion` but most users think in terms of confidence/threshold. Either rename to **Threshold (precision ↔ recall)** with the same scale, or replace with a confidence-based slider (0–1) directly. Inclusion is internal jargon.

### 5.6 Selection-strategy buttons (Top/Hard/New) ★★ M
In Manual mode these are 3 unlabeled jargon buttons. Either hide them by default (only Autopilot users need them) or rename: `Top` → **Most likely match**, `Hard` → **Most uncertain**, `New` → **Most novel**.

### 5.7 The "Load" sort mode is buried ★★ S
"Load" requires a `+` click that opens another modal where the user picks a detector. Instead expose recently-used detectors as a dropdown in the sort row, with a `Manage…` link for the modal. (Mirrors how Word/Photoshop handle recent files.)

### 5.8 "Add Labels" vs "Import Labels" vs "Label importer" ★★ S — shipped
Standardised on **Import Labels** (Title Case, matching the existing `Export Labels` / `Export Detector` siblings):

- Models dashboard detector-card button: `Add Labels` → `Import Labels` (aria-label + tooltip).
- New Detector → Trained tab form field: `Label Importer` → `Import Labels From`.
- Label importer modal title: now `Import Labels` standalone, and `Import Labels into <detectorname>` when launched against a specific detector (per user preference: keep contextual variant).
- Right-panel button was already `Import Labels`; left as-is.

Internal symbols (`addLabelsModalOpen`, `onAddLabels`, `.add-labels-btn`) were intentionally left alone — out of scope for a UI-string fix and would balloon the diff.

### 5.9 Vote-pile right panel ★ S
The right panel shows "Good" and "Bad" as two stacked stacks. There's no drag-to-reorder, no batch operations (`select all → un-vote`), and no obvious way to remove an item from a pile (must reopen the centre, find it, vote the other way). Add multi-select + a context menu (remove, re-vote, copy ID).

### 5.10 Crop modal optionality ★ S
After picking an example media, a crop modal appears even if the user wants to use the full file. Add a clear "Use full file" button alongside "Crop and confirm", and skip the modal entirely for text and audio < 5s.

### 5.11 Settings tab nesting ★ S ❌
The Settings modal has tabs → sub-tabs (per media type) → twin columns (left/right panel). That's three levels of nesting in a modal. Flatten by adopting (§5.4)'s mirror toggle and grouping per-media settings under a single accordion per type.

**Rejected:** Not pursuing. The current Settings layout stays as-is — tabs → media-type sub-tabs → twin Left/Right columns.

### 5.12 "Achievements" tab discoverability ★ XS
Achievements live in Settings, where users go for *settings*, not gamification rewards. Either move to its own menu item or a small trophy icon in the header.

### 5.13 Disabled-button reasons hidden ★★ XS
Train / Find buttons disable for non-obvious reasons (media-type mismatch, nothing selected, etc.) with a hidden hint that's `visibility:hidden`. Show the reason inline at all times — disabled buttons should always say *why*. (Pattern: GitHub's merge button.)

---

## 6. Inconsistencies across the app

### 6.1 `media_type` naming ★★ XS ✅
Plural in dropdown labels ("images") vs singular in API/IDs ("image"). Pick one (singular) and use it everywhere visible.

**Shipped:** Removed the `tab_title` plural-form override from `MediaType` (base + all five subclasses) and from `/api/media-types`. The Angular frontend now uses `name` (singular) everywhere — dropdowns, tabs, demo tabs, and detector picker all read "Audio", "Image", "Text", "Video", "Document". Removed the `tab_title?` field from `MediaTypeInfo` in `frontend/src/app/models/api.models.ts`.

### 6.2 "Detector" vs "Model" vs "Classifier" ★★ XS
Code uses `detector`. Dashboard table is labeled "Detectors". Some UI elements (sort modes, the Models dashboard column) call them "models". User Guide uses both interchangeably. Pick one (suggest **detector** to keep alignment with the codebase) and lint the frontend strings.

### 6.3 Path-style fields ★★ S
The "where do I put this file?" concept appears across plugins as `filepath`, `paths_file`, `path`, `url`, `file`. Plus the underlying field types differ (`server_path`, `text`, `file`). Standardize labels: **Save to (server path)**, **Path or URL**, **Upload a file**.

### 6.4 Modal back-button conventions ★★ XS — SHIPPED
Convention: every nested-modal flow renders a left-aligned `&larr; Back` button using the shared `btn btn--secondary btn--sm back-btn` class combination (styled by `.back-btn` in `frontend/src/scss/_components.scss`). Documented under "Nested-modal back buttons" in `CLAUDE.md`. All current nested flows (processor / settings / label importer modals, settings exporter, load-sort, resort-prompt, new-detector → media picker, new-detector → trained-importer form) already follow this convention.

### 6.5 Confirm-on-destructive ★★ S — shipped
Delete dataset, delete detector, delete label entry, clear votes, reset settings — currently a mix of inline-hover-confirm, modal-confirm, and no-confirm. Standardize on a single confirm pattern, with the operation name in the confirmation: *"Delete detector 'cats'? This removes its labelset and training metadata. The dataset is unaffected."*

**What shipped**: Added `VtDialogService.confirmDestructive(question, detail, actionLabel?)` and routed every destructive action with a UI surface through it: delete dataset (single + bulk), delete model (single + bulk), reset settings. Messages follow the `<Action>? <What is removed; what is unaffected>.` template, and the primary button uses the action verb ("Delete" / "Reset") instead of "OK".

**Open follow-ups**:
- *Delete label entry* and *Clear votes* were listed in the original item but don't have any UI surface today (only the `clearVotes()` API client exists, with no caller). When those actions get a button, wire them through `confirmDestructive` — e.g. `"Clear all votes for detector 'X'? This deletes every saved label for this model and cannot be undone."`
- The labeling-view "press ← twice to vote no and discard the box" inline-confirm is a different UX (two-key chord, not a delete) and was intentionally left as-is. If we ever consolidate it, treat it as its own pattern rather than forcing it through the modal.
- The destructive primary button has no distinct danger styling yet — it reuses `.btn--primary`. A red variant would make the modal even harder to dismiss-by-accident; deferred.

### 6.6 Toast / banner / inline error styling ★★ S
Errors appear in at least three styles: red banner inline, modal-level red text, console-log only. Add a single toast service and route all `error` SSE events + HTTP failure responses through it.

### 6.7 Saved-state indicator ★ XS
Settings auto-save but show no "saved" feedback. Some forms (export modal) require an explicit Save. Pick a convention: either always auto-save with a tiny `✓ saved` indicator, or always require explicit Save with a Cancel.

### 6.8 Embedder display names ★★ XS
The dropdown shows raw IDs (`siglip`, `dinov3_patch`, `e5`). Map each to a human label (`SigLIP (general images)`, `DINOv3 patch (region-aware)`, `E5 (text)`). Keep the raw ID as a secondary `<small>` line for power users.

### 6.9 SSE channel/progress schemas ★ S — SHIPPED
Each long-running operation used to emit a slightly different progress payload (`step`/`total_steps` for dataset, `current`/`total` for eval, plain status strings for sort). ~~Standardize on a single `ProgressEvent` interface so the frontend can render any of them with the same component.~~ Shipped:

- **Backend** (`vtsearch/concurrency/progress.py`): every singleton `ProgressTracker` (dataset, sort, eval, find) and every per-task tracker created by `LoadingTasksTracker` now exposes the same `_PROGRESS_COMMON_EXTRAS = {"step": None, "total_steps": None, "error": None}` on top of the four base fields. `dataset_progress` additionally carries `staging_result` (used only by combine-datasets staging). `update_sort_progress` and `update_eval_progress` accept the new extras as optional kwargs, with the merge semantics shared through a single `_common_extras_kwargs` helper.
- **Frontend** (`frontend/src/app/models/api.models.ts`): replaced `DatasetProgress` and the untyped `SortProgressResponse` with a single `ProgressEvent` interface. `LoadingTask extends ProgressEvent` and re-narrows the base fields as required. `ProgressEventsService` now types every channel subject as `BehaviorSubject<ProgressEvent>` (or `LoadingTask[]`); the `find$` and `eval$` channels are no longer `Record<string, unknown>`.
- **Frontend** (`frontend/src/app/utils/format-progress.ts`): added `formatProgressMessage(progress, defaultMessage)` and `isProgressIndeterminate(progress)` — the single source of truth for the `[Step S/T] (C/T) message` layout that previously lived inline in `dashboard.component.ts` (×2), `dataset-card.component.ts`, `detector-card.component.html`, `find-view.component.ts`, and `label-view.component.ts`. All six call sites were converted.

Wire-format note: no breaking change for current callers — the new sort/eval `step`/`total_steps`/`error` keys default to `null` and just become available for future multi-step phases.

### 6.10 Date formatting ★ XS — SHIPPED
"last_trained / created" columns format dates differently from the version string in the footer. ~~Use one formatter (relative for < 7d ago, absolute YYYY-MM-DD otherwise) everywhere.~~ Shipped: a single `frontend/src/app/utils/format-date.ts` is now called from all four sites (detector card, dataset card, dataset stats modal, settings-modal version footer). Always-absolute — no relative branch. Columns render `YYYY-MM-DD HH:MM` (local), version renders `YYYY-MM-DD` (UTC). Relative time was dropped deliberately: it drifts on every reload and the coarse buckets (`2w ago`) throw away precision that matters in a model-management dashboard.

### 6.11 Active-dataset/detector indicator ★★ S — shipped
The read-only top-bar fields became click-to-switch `<vt-context-pulldown>` instances with media-type compatibility dimming, "+ Add New" footers that open the importer / new-detector modals in-place, an `vt-incompatible-pair-explainer` overlay for half-set / incompatible pairs, URL-encoded active pair (`/label/:datasetId/:detectorId` and `/find/:datasetId/:detectorId` gated by `activeContextGuard`), per-pair spinner glyphs driven by `GET /api/jobs/active`, learned-sort rehydration via the `JobManager` signature cache, and a re-embed task path that fires `"Re-resolving labels for X's embedder…"` when the active dataset's embedder differs from the loaded detector's. This entry also closed [§8.2](#82-switching-active-datasetdetector--s) below.

### 6.12 Plugin field types ★ S — shipped
Free-text fields that should be numbers (`n_mels`, `time_window_s`, `size` for synthetic), enums that should be selects (`colormap`, `language` for OCR), and password fields disguised as text (`auth_header`). Tighten the `PluginField` type system and migrate.

**What shipped:** added `"number"` to `FieldType` (with new `min`/`max`/`step` attrs and a `PluginField.is_integer_number()` heuristic so the CLI parses with `int` vs `float`). Migrated:
- `n_mels`, `time_window_s` (audio2image): `text` → `number`
- `colormap` (audio2image): `text` → `select` (curated matplotlib colormap list)
- `language`, `threshold` (image2text): `text` → `select` / `number`
- `n_clips` (video2image), `ffmpeg_timeout` (video2audio): `text` → `number`
- `size` (synthetic importer): `text` → `number`
- `url` (webhook exporter): `text` → `url` (the auth header was already `password`)

Frontend: every plugin-field renderer (dataset/processor/label/settings importer modals, settings exporter modal, export modal, autodetect results modal, new-detector modal, plus the three converter-param sections inside dataset-importer-modal) now switches on `field_type` and renders `number`/`url`/`password`/`email` inputs with `min`/`max`/`step` attributes where applicable.

Backend coercion is unchanged — values still arrive as strings from the web and as `int`/`float` from argparse (driven by the new `is_integer_number()` heuristic over `step`/`default`/`min`/`max`).

---

## 7. Non-standard UIs to normalize

VTSearch has a few interactions that are clever but un-Google-able — replace with conventions users already know.

### 7.1 Stripe histogram in left panel ★★ M
The mini-histogram below the media list lets users click to jump to a score range. Most users never figure this out. Either remove or replace with a standard horizontal slider that filters the list. (Slider supports drag-to-zoom-window, is keyboard-friendly, and is familiar from price filters.)

### 7.2 Side-toggle bulk-select on dashboard ★ S — SHIPPED
The triangle-shaped mixed-state checkbox on the side of the dashboard tables is unusual. Move to a standard left-column checkbox per row with a top-of-column master checkbox (Gmail / Linear / GitHub pattern).

**What shipped:** Per-row checkboxes moved to a new pinned leftmost `select-col` in both the Datasets and Detectors tables; the tri-state master checkbox now lives in that column's header. The right-side sidebar retains the Combine and Delete bulk-action buttons (no longer paired with a checkbox). Files: `frontend/src/app/components/dashboard/{dashboard,dataset-card/dataset-card,detector-card/detector-card}.component.{html,scss}`, plus `dashboard.component.ts` (removed unused `spinDataset/DetectorSelectToggle` animation state).

### 7.3 Hover-to-reveal delete confirmation ★★ XS
Hover-only confirmation for destructive actions is unfamiliar and fragile. Replace with a standard modal confirm (matches §6.5).

### 7.4 Resize-cursor on small drag handles ★ XS
Panel dividers and column-resize handles are ~2px wide. Make them 8px hit targets with a `cursor: col-resize` on hover (standard).

### 7.5 Folder-tree breadcrumb browser ★ M
The server-folder browser has a custom breadcrumb + table UI. It's functional but every modern OS has standardized on the same "left sidebar with starred/recent locations, breadcrumb on top, list in middle, double-click to enter, Enter key to confirm". Bring ours closer to that.

### 7.6 Drag-and-drop affordances ★ S — SHIPPED
Several places accept drag-drop (file import, example media into detector) but show no drop zone until the drag is over the page. Add visible dashed drop zones with `"Drop a folder here to import"` text.

**What shipped:** New reusable `vt-drop-zone` component (`frontend/src/app/components/drop-zone/`) renders a dashed-bordered, clickable target that doubles as a drag-drop receiver. It handles folder drops by walking `webkitGetAsEntry()` recursively and synthesising `webkitRelativePath` on each File so the existing upload code sees the same shape as a `<input type=file webkitdirectory>` selection. Wired into three places: (1) the dataset importer modal's Local Folder / Local Files views (`"Drop a folder here to import"` / `"Drop files here to import"`); (2) the new-detector modal's media-picker Local Folder / Local Files views (`"Drop a folder here to use as example"` / `"Drop a media file here to use as example"`); (3) the new-detector main form (tab=blank), where it replaces the inline "Upload File…" button alongside the existing "Browse Media…" button. The previous bare `<input type=file>` widgets in those locations were removed.

**Open follow-ups:** The examples-editor modal (Edit Examples → + Add Good / + Add Bad) still uses small button-driven file inputs. Adding compact drop zones there is straightforward — the same `vt-drop-zone` component can drop in next to or in place of the `+ Add Good` / `+ Add Bad` buttons — but was descoped from this pass at the user's request.

### 7.7 Modal stacking ★★ XS
Some flows can stack 3 modals deep (importer → demo picker → embedder picker). Stacking modals is a known anti-pattern. Either convert nested modals to in-place sub-views with a back button (§6.4), or use a single multi-step wizard.

### 7.8 Shift-drag to draw region ★★ XS
The image region-vote uses Shift+drag with no UI cue. Standard image-region tools use a dedicated mode toggle (a button that turns the cursor into a marquee). Keep Shift+drag as a power-user shortcut, but also expose a Marquee button.

### 7.9 Region rectangle interaction ★ S
After drawing, the rectangle is editable via 8 handles — good. But the "click on the rectangle to restore" interaction is non-discoverable. A standard "✓ confirm region" / "✗ clear" button overlay on the rectangle would replace the current "press ← twice to discard" pattern.

### 7.10 Sort-bar "+" to add a sort source ★ XS
The `+` icon to load a saved detector for sort is non-standard. Replace with a labeled button **"Load saved detector"** in the sort dropdown.

---

## 8. Long-but-possible flows to shorten

Workflows the user *can* do today but that take too many steps and clicks.

### 8.1 First-time dataset → labelled export ★★★ M
Today: open menu → pick importer category → pick importer → fill form → pick media type → pick embedder → submit → wait → close modal → select dataset → click "New detector" → fill form → click Train → vote 7 items → export → pick exporter → fill form → submit. That's ~15 clicks before the user has anything to show.
**Compressed flow:** "Quick start" CTA on empty dashboard → pick a media type → upload a folder → app auto-creates a detector with the folder's name and drops the user into the labeling view. Export becomes a single header button with a recent-target fallback.

### 8.2 Switching active dataset/detector ★★ S — shipped
Shipped together with §6.11: the top-bar pulldowns make this a 1-click switch from any view. Sort mode is per-user-tier and survives the switch by design; view-local state (scroll, partial Find query) resets on switch (treated as a re-entry to the view). See §6.11 for the full set of affordances.

### 8.3 Cross-dataset training with a re-used labelset ★★ M
The labelset-source machinery lets a detector pull labels from a different dataset, but using it requires:
1. Add a labelset source to detector A (configure plugin, write filepath template).
2. Vote on dataset X.
3. Load dataset Y.
4. Train detector A.
Compress to a "Use these labels on another dataset" button on the right panel.

### 8.4 Re-running auto-detect after edits ★ S
Tweaking a label, then re-running auto-detect, is currently: edit → save → navigate to dashboard → re-pick dataset+detector → click Find → wait → reopen results modal. Add a "Re-run with current settings" button inside the existing results modal.

### 8.5 Importing pre-computed embeddings ★ S — shipped
`.npz` is now a top-level option in every importer that loads files:

- `server_folder` and `server_files` declare a separate `vectors_file` `PluginField` (server_path, accepts `.npz`, optional). Files in the folder / listed paths whose basename or relative path matches a key in the archive reuse the supplied vector instead of running the embedder.
- `local_folder` now exposes the same `.npz` upload widget that `local_files` already had — vectors are forwarded to `/api/dataset/import-local-folder` regardless of picker kind.
- `local_files`'s existing widget stays put and is shared across both browser-side flows.

Open follow-ups:

- `http_archive`: still no `.npz` option. Matching filenames inside an extracted archive is hairy and rarely useful, so it was deliberately left out — revisit if a user asks.
- Server-side `.npz` field for `server_folder` is a plain text input; could be upgraded to the same server-path browser used for the folder picker if the typing friction shows up in testing.

### 8.6 Configure & test a webhook exporter ★ M
Currently: open export modal → pick webhook → fill URL → fill auth → submit → realize the URL was wrong → repeat. Add a "Send test ping" button next to the URL field that fires a single test payload.

### 8.7 Combining detectors ★ M
The "combine detectors" feature exists but requires multi-select + a non-obvious icon button. Surface as a clear "Merge detectors" CTA with a preview of what the merged detector would look like (count, intersection vs union choice, name).

### 8.8 Renaming + re-syncing a detector ★ XS
Renaming a detector with a labelset source filepath template (`{detector_name}.labels.json`) leaves the old file on disk. After rename, prompt: "Move existing labelset file to new name?" with a one-click yes.

### 8.9 Audio segment example → trained detector ★★ M
"I want a detector for this 3-second cough sound": today requires opening a centre-panel item, opening the crop modal, dragging selection, confirming, then navigating to new-detector with that example. Add a right-click "Use this as a detector seed" on any media in the left panel.

### 8.10 Resuming a labelling session ★★ S
There's no "recent sessions" surface. Add a "Recent sessions" list on the dashboard (dataset + detector pair + last activity timestamp) so the user gets back into work in one click.

### 8.11 Bulk-importing multiple folders at once ★ M
Currently the server-folder importer is one folder per import job. Allow multi-folder selection (server folder browser + multi-select) and create one dataset per folder in the same job.

### 8.12 "I want a detector exactly like this one, but trained from scratch" ★ XS
Useful for experimentation. Add a "Clone" action in the detector row that duplicates the labelset (or labelset source) but resets the trained model.

---

## 9. Quick wins (prioritized)

A short cut of the highest-leverage, lowest-effort items. If we had a single afternoon, do these.

| # | Idea | Section | Effort | Impact |
|---|------|---------|--------|--------|
| 1 | Detect media type from file extensions | §1.2 | S | ★★★ |
| 2 | Recommended embedder per media type | §1.3 | XS | ★★★ |
| 3 | Auto-derive dataset name from path | §1.1 | XS | ★★★ |
| 4 | Empty-state CTA on dashboard | §2.1 | S | ★★★ |
| 5 | First-vote tooltip in label view | §2.2 | XS | ★★★ |
| 6 | Show disabled-button reasons | §5.13 | XS | ★★ |
| 7 | Active dataset/detector header strip | §6.11 | S | ★★★ |
| 8 | Settings explainers under each control | §2.3 | S | ★★★ |
| 9 | Per-file embedding progress | §4.1 | M | ★★★ |
| 10 | Bytes/sec for model downloads | §4.2 | M | ★★★ |
| 11 | ETA on every long bar | §4.3 | S | ★★ |
| 12 | Cancel buttons everywhere | §4.6 | S | ★★★ |
| 13 | OS dark-mode default for theme | §1.4 | XS | ★★ |
| 14 | Human embedder display names | §6.8 | XS | ★★ |
| 15 | Quick-load button on demo cards | §3.3 | XS | ★★★ |
| 16 | Show shortcuts on Good/Bad buttons | §2.9 | XS | ★★ |
| 17 | Toast service for all errors | §6.6 | S | ★★ |
| 18 | Detect URL/path → jump to importer | §1.12 | S | ★★ |
| 19 | Timestamps in default export filenames | §1.7 | XS | ★★ |
| 20 | Flatten importer category/type tabs | §5.1 | M | ★★★ |

---

*Total opportunities surfaced: ~75. Many are independent and can be shipped incrementally without coordination. The biggest single mental-model unlock is §6.11 + §5.3 + §4.1: persistent active-context indicator, friendlier sort-mode naming, and honest per-file embedding progress. Together they remove the three biggest "what is happening right now?" moments in the app.*

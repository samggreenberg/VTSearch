# `vtscore` Public Surface Inventory

This is the docstring-only sketch of the public API a future `vtscore` library exposes — the contract the [extract-library refactor](plans/extract-library.md) must preserve. Internal helpers, route wiring, and Flask glue are deliberately omitted.

It is **not** a tutorial. The shape here tells a refactor whether a given rename or reorganisation is a breaking change for external consumers.

Names are listed at their **current** `vtsearch.<pkg>.<name>` paths. After Phase 8 they will be reachable as `vtscore.<pkg>.<name>` (re-export shims may bridge one release; see Phase 6).

Anything flagged **⚠️ seam** has Flask, settings, or filesystem coupling that Phases 1–4 must remove before this name is library-clean.

---

## 1. Datasets — `vtsearch.datasets`

Loading, importing, and labelling media collections.

### Core types

- `vtsearch.datasets.origin.Origin` — Records the source importer and parameters used to load an element. `to_dict()` / `from_dict()` round-trip; `display()` returns a human label. Subclasses live alongside (`FolderOrigin`, `HTTPArchiveOrigin`, `PickleOrigin`, etc.).
- `vtsearch.datasets.labelset.LabeledElement` — A single labeled element: `md5`, `label`, `origin`, `filename`, `category`, optional `metadata` dict, optional `region_box`.
- `vtsearch.datasets.labelset.LabelSet` — Collection of `LabeledElement`s plus optional detector metadata (name, media_type, embedder).

### Loaders

- `vtsearch.datasets.loader.load_dataset_from_pickle(path, …)` — Load a previously-saved dataset pickle.
- `vtsearch.datasets.loader.load_dataset_from_folder(folder, media_type, …)` — Walk a folder tree and embed media in place.
- `vtsearch.datasets.loader.export_dataset_to_file(…)` — Serialise a loaded dataset to JSON/CSV for inspection.

### Importer plugin family

- `vtsearch.datasets.importers.base.DatasetImporter` — ABC. Declares `name`, `display_name`, `fields: list[PluginField]`, optional `multi_media: bool`, and `run(field_values, progress)` which yields media dicts.
- `vtsearch.datasets.importers.get_importer(name)` — Fetch importer by name.
- `vtsearch.datasets.importers.list_importers()` — Enumerate registered importers.

### Media-source plugin family

- `vtsearch.datasets.sources.base.MediaSource` — ABC for resolving an `Origin` to a fetchable file. Implements `list_items()`, `fetch_item()`, `resolve_path()`, `cleanup()`.
- `vtsearch.datasets.sources.base.MediaItem` — One discoverable media file within a source (`key`, `filename`, `source_name`).

---

## 2. Embedding — `vtsearch.embedding`, `vtsearch.media.embedder`

### Per-media-type helpers (`vtsearch.embedding.helpers`)

- `embed_audio_file(path) -> np.ndarray`
- `embed_image_file(path) -> np.ndarray`
- `embed_video_file(path) -> np.ndarray`
- `embed_paragraph_file(path) -> np.ndarray`
- `embed_text_query(text, media_type) -> np.ndarray` — Embed a text query into the media-type's joint space; results cached in-memory.
- `clear_text_query_cache()` — Drop all cached query embeddings.

### Runtime setup (`vtsearch.embedding.loader`)

- `initialize_models()` — Prepare cache dirs and PyTorch threading config. Idempotent; called once at startup.
- `get_torch_device() -> torch.device` — Selects `cuda` / `mps` / `cpu`.
- `default_concurrent_downloads() -> int` — Hardware-derived default for parallel dataset downloads.
- `default_concurrent_embeddings() -> int` — Hardware-derived default for parallel embeddings.

### Embedding matrix cache (`vtsearch.embedding.matrix`)

- `get_embedding_matrix(ctx) -> (list[int], np.ndarray)` — Return `(sorted_media_ids, (N, D) float32 matrix)`. Cached on `DatasetContext`; rebuilt on dataset changes.

### Embedder ABC (`vtsearch.media.embedder`)

- `MediaEmbedder` — ABC. Implements `embed_media(media)` and `embed_text(text)`.
- `media_from_path(path)` — Build the minimal media dict an embedder consumes.
- `resolve_embed_batch_size(default)` — Reads `VTSEARCH_EMBED_BATCH_SIZE` env var, else returns default.

---

## 3. Detectors — `vtsearch.detectors`

### Training (`vtsearch.detectors.training`)

- `validate_good_bad_split(votes)` — Returns `(num_good, num_bad)`; raises if either is zero.
- `train_and_threshold(features, labels, …)` — Train MLP and compute a calibrated decision threshold.
- `train_and_score(ctx, …)` — Vote-aware online training: invoked from sort/vote handlers.
- `collect_media_origins(labelset)` — Gather the distinct origin types referenced by a labelset.
- `train_detector_from_origins(labelset, …)` — Load-time training: resolves origins to embeddings, then trains.
- `serialize_weights(state_dict) -> list` — JSON-serialisable nested-list form of a PyTorch state dict.

### Workflow

- `vtsearch.detectors.workflow.apply_and_retrain(detector_id, det_ctx, new_entries, detector_name)` — Resolve new label entries against the loaded dataset, apply them, retrain. Swaps the active detector via `vtsearch.state.core.override_detector_context()`.

### Registry, store, resolver

- `vtsearch.detectors.registry.{get_detector, list_detectors, create_detector, delete_detector}` — In-memory detector registry.
- `vtsearch.detectors.store.get_detectors_dir()` — Resolve the detectors directory. ⚠️ **seam — reads `vtsearch.settings.get_detectors_dir`. Phase 2 will route through `CoreConfig`.**
- `vtsearch.detectors.resolver.resolve_file_context(...)` — Context manager that materialises an `Origin` into a local file for embedding.
- `vtsearch.detectors.resolver.resolve_file_from_origin(entry)` — Resolve a single labelset entry to a `(file, embedding)` pair.
- `vtsearch.detectors.resolver.resolve_label_embeddings(labelset)` — Batch-resolve a whole labelset.
- `vtsearch.detectors.resolver.register_source_resolver(fn)` / `register_importer_resolver(fn)` — Plug in custom resolution strategies.

### Labeling progress

- `vtsearch.detectors.labeling_progress.get_progress_for_step(step)` — Retrieve cached model + metrics from a particular vote step.
- `vtsearch.detectors.labeling_progress.clear_progress_cache()` — Drop all cached progress data.
  - ⚠️ **seam — reads `get_autopilot_goal_diversity` from settings. Phase 2.**

---

## 4. Training primitives — `vtsearch.training`

Media-agnostic. Anything that takes a `(features, labels)` matrix and returns a model or a threshold lives here.

### MLP (`vtsearch.training.mlp`)

- `build_model(input_dim, hidden_dim=None) -> nn.Module` — Construct untrained MLP (`Linear → ReLU → Dropout → Linear`).
- `build_model_from_weights(weights) -> nn.Module` — Reconstruct a model from `serialize_weights()` output (handles a legacy format).
- `train_model(model, X, y, …) -> nn.Module` — Train MLP on a feature matrix.

### Thresholds (`vtsearch.training.thresholds`)

- `calculate_gmm_threshold(scores)` — Fit a 2-component GMM to a score distribution.
- `find_optimal_threshold(scores, labels)` — Threshold maximising F1 on a validation set.
- `calculate_cross_calibration_threshold(...)` — Cross-validated threshold using `calibrate_count` / `calibration_fraction`.
- `cross_calibration_threshold_cached(...)` — Memoised variant.
- `calculate_safe_threshold(...)` — Blend of cross-calibration and GMM clipped by a safety floor.

### Region similarity (`vtsearch.training.region_similarity`)

- `score_against_query(media, query_vec)` — Return `(max_cosine_sim, best_region_box)` for patch-region media.
- `score_all_against_query(medias, query_vec)` — Batch variant returning a `{media_id: (score, region)}` dict.

---

## 5. Media types — `vtsearch.media`

The plugin family for audio / image / text / video / document support.

- `vtsearch.media.base.MediaType` — ABC. Declares `type_id`, `folder_name`, `load_media_data()`, plus embedder and processor registries.
- `vtsearch.media.base.MediaResponse` — Framework-agnostic media payload (`data`, `mimetype`, `download_name`). Routes convert to `flask.Response` outside the library boundary.
- `vtsearch.media.base.DemoDataset` — Metadata for one demo dataset.
- `vtsearch.media.base.ProgressCallback` — Type alias for progress callbacks.
- `vtsearch.media.clipper.MediaClipper` — ABC for clippers; `clip(media)` splits long media into chunks.
- `vtsearch.media.cropping.crop_file_bytes(...)` — Apply a bounded clipper to a file and return cropped bytes (audio + image today).

### Processor plugin family (`vtsearch.media.processors`)

- `Processor` — ABC. Implements `process()` and optional `load_model()`.
- `Detector` — Binary processor returning `bool`.
- `Localizer` — Returns a list of bounding boxes.
- `Extractor` — Returns a list of detail dicts.

---

## 6. Converters — `vtsearch.converters`

Cross-media conversion (document→image, video→audio, image→text via OCR, etc.).

- `vtsearch.converters.base.MediaConverter` — ABC. Declares `source_type`, `target_type`, `fields`, and `convert(media, params)`.
- `vtsearch.converters.{get_converter, list_converters, list_converters_for_source, list_converters_for_target}`.

---

## 7. Results exporters — `vtsearch.exporters`

- `vtsearch.exporters.base.LabelsetExporter` — ABC. Implements `export(labelset, params)` with declared `fields`, `icon`, `display_name`.
- `vtsearch.exporters.base.ExporterField` — Alias for `PluginField`.
- `vtsearch.exporters.{get_exporter, list_exporters}`.

---

## 8. Labels — `vtsearch.labels`

### Label importers

- `vtsearch.labels.importers.base.LabelImporter` — ABC. Implements `run()` with declared `fields`, `name`, `display_name`.
- `vtsearch.labels.importers.base.LabelImporterField` — Alias for `PluginField`.

### Bidirectional labelset sources

- `vtsearch.labels.sources.base.LabelsetSource` — ABC (inherits the generic `SyncSource[LoadT, SaveT]`). Implements `load()`, `save()`, `load_full()`.
- `vtsearch.labels.sources.base.LabelsetSourceField` — Alias for `PluginField`.

### Sync helpers

- `vtsearch.labels.sync.sync_to_labelset_source(detector_ctx)` — Auto-export on vote change.
- `vtsearch.labels.sync.sync_from_labelset_source(detector_ctx)` — Manual import on demand.

---

## 9. Evaluation — `vtsearch.eval`

- `vtsearch.eval.runner.eval_text_sort(queries, …)` — Text-sort eval: AP, P@k, R@k per query.
- `vtsearch.eval.runner.eval_learned_sort(…)` — Learned-sort eval: train/test split simulation.
- `vtsearch.eval.metrics.QueryMetrics` — Per-query text-sort metrics.
- `vtsearch.eval.metrics.LearnedSortMetrics` — Per-fold learned-sort metrics.
- `vtsearch.eval.metrics.DatasetResult` — Aggregated dataset-level result.
- `vtsearch.eval.metrics.compute_metrics(rankings, predictions, …)` — IR + classification metrics.

`vtsearch.eval.visualize` is matplotlib-only and will move to a `vtscore[viz]` extra (see plan §Risks).

---

## 10. Plugin registry — `vtsearch.plugins`

Backs every other plugin family above.

- `vtsearch.plugins.PluginRegistry` — Reusable registry. Walks a package for the family's sentinel attribute (`IMPORTER`, `EXPORTER`, `LABELSET_SOURCE`, etc.) and supports `importlib.metadata` entry-point discovery (`vtsearch.<family>` groups). Built-ins win on name clashes; broken entry points warn and are skipped.
- `vtsearch.plugins.PluginBase` — Common base. Declares `name`, `display_name`, `description`, `icon`, `fields`.
- `vtsearch.plugins.PluginField` — One configurable input: `key`, `label`, `field_type`, `description`, `options`, `default`, etc.
- `vtsearch.plugins.FieldType` — Literal union over `file | folder | url | text | password | email | number | select | server_path | checkbox`.
- `vtsearch.plugins.make_plugin_registry(...)` — Factory returning `(get_fn, list_fn)` for a family.

---

## 11. Context objects — `vtsearch.state.core`

The two structs the library hands back to callers. The module-level `medias` / `good_votes` / etc. proxies are an **app-layer** concern (Phase 3) — library callers operate on contexts directly.

- `vtsearch.state.core.DatasetContext(dataset_id="")` — Per-dataset state: `dataset_id`, `medias: dict[int, dict]`, `diversity_tree`, `dataset_display_name`, plus cached `_emb_matrix_ids` / `_emb_matrix`.
- `vtsearch.state.core.DetectorContext(detector_id, name, media_type, embedder)` — Per-detector state: `detector_id`, `name`, `media_type`, `embedder`, `good_votes`, `bad_votes`, `label_history`, `vote_region_boxes`, `vote_click_times`, `model`, `threshold`, `labelset_source`, `training_medias`, `find_initial_labels`, etc.
- `vtsearch.state.core.{get_context, get_detector_context}` — Lookups by ID.
- `vtsearch.state.core.{register_context, unregister_context, list_loaded_dataset_ids, clear_all_contexts}` — Dataset registry surface.
- `vtsearch.state.core.{register_detector_context, unregister_detector_context, list_loaded_detector_ids, clear_all_detector_contexts}` — Detector registry surface.
- `vtsearch.state.core.{set_thread_dataset_context, get_thread_dataset_context, set_thread_detector_context, get_thread_detector_context}` — Thread-local context overrides (used by background jobs and tests).
- `vtsearch.state.core.{register_dataset_context_resolver, register_detector_context_resolver}` — Install per-request resolvers. The Flask shim (`vtsearch.shim.register_flask_context_resolvers`) calls these at app startup so `g._dataset_context` / `g._detector_context` feed the proxies.
- `vtsearch.state.core.override_detector_context(ctx)` — Context manager taking the top tier of `get_active_detector_context()`'s resolution chain. Used by `apply_and_retrain` to swap detectors regardless of whether the caller is inside a Flask request.

---

## 12. Concurrency — `vtsearch.concurrency`

- `vtsearch.concurrency.async_jobs.AsyncJob` — Background-job state: `job_id`, `status`, `result`, `error`, `progress`, `started_at`, `user`, `dataset_id`, `detector_id`.
- `vtsearch.concurrency.async_jobs.JobManager` — Single-runner async job manager with one pending slot; memoises results by signature.
- `vtsearch.concurrency.progress.ProgressTracker` — Thread-safe `(status, message, current, total)` tracker with cancellation and subscriber hooks.
- `vtsearch.concurrency.progress.CancelledError` — Raised inside a tracked task when cancellation is requested.

---

## 13. Security — `vtsearch.security`

- `vtsearch.security.pickle.RestrictedUnpickler` — Allowlist-based unpickler.
- `vtsearch.security.pickle.safe_pickle_load(stream)` — Deserialise a pickle stream safely.
- `vtsearch.security.pickle.peek_pickle_dataset_summary(path)` — Read just the metadata header of a dataset pickle without instantiating heavy objects.
- `vtsearch.security.path_validation.{validate_server_filepath, sanitize_template_value, rglob_follow_symlinks, glob_top_level}` — Filesystem path safety.
- `vtsearch.security.url_validation.*` — SSRF guards for outbound HTTP (importers fetching from URLs).

---

## 14. Sync abstraction — `vtsearch.sync`

- `vtsearch.sync.SyncSource[LoadT, SaveT]` — Generic `PluginBase` subclass parametrised by what it loads and saves. Both `SettingsSource` (app-side) and `LabelsetSource` (library-side) inherit this.

---

## 15. CLI — `vtsearch.cli`

- `vtsearch.cli.autodetect(...)` — Programmatic entry point: load a dataset, train/score detectors from a settings file, optionally export results. Used by `python app.py --autodetect`.
- `vtsearch.cli.import_labels_into_detector_from_file(...)` — One-shot label import path used by CLI and tests.
- `vtsearch.cli.main(argv)` — Top-level dispatcher (argument parsing, dry-run, orchestration).
  - ⚠️ **seam — calls `set_settings_path` and `get_autorun_detectors`. Phase 2 will pass these in via `CoreConfig` instead.**

---

## Seams to fix before this surface is library-clean

Cross-referenced with the [extract-library plan](plans/extract-library.md):

| Module / name                                       | Coupling                       | Phase | Status |
|-----------------------------------------------------|--------------------------------|-------|--------|
| `state.core._request_*_context()`                   | `flask.g`                      | 1     | ✅ shipped — pluggable resolver hook |
| `detectors.workflow.apply_and_retrain`              | `flask.g`                      | 1     | ✅ shipped — `override_detector_context()` |
| `cli.main` / `cli.autodetect`                       | `vtsearch.settings`            | 2     | ✅ shipped — `CoreConfig.from_settings(settings_path=…)` |
| `cli_pipeline`                                      | `vtsearch.settings`            | 2     | ✅ shipped — same pattern |
| `datasets.load_pipeline`                            | `vtsearch.settings`            | 2     | ✅ shipped — gates via `CoreConfig`, writes via app hook |
| `datasets.registry.get_saved_datasets_dir`          | `vtsearch.settings`            | 2     | ✅ shipped — routes through `CoreConfig.from_settings()` |
| `detectors.store.get_detectors_dir`                 | `vtsearch.settings`            | 2     | ✅ shipped — routes through `CoreConfig.from_settings()` |
| `detectors.labeling_progress`                       | `vtsearch.settings`            | 2     | ✅ shipped — routes through `CoreConfig.from_settings()` |
| `state/__init__` (inclusion, calibrate_*, etc.)     | `vtsearch.settings`            | 2     | ✅ shipped — reads via `CoreConfig`, writes via persister hook |
| `config.CoreConfig.from_settings()` bridge          | `vtsearch.settings`            | 8     | open — relocates to app-side shim at the physical move |
| Module-level proxies (`medias`, `good_votes`, …)    | global state                   | 3     | open   |
| `autorun_extractors` / `autorun_localizers`         | module globals in `state/core` | 3     | open   |
| Hardcoded `data/` paths (if any)                    | filesystem                     | 4     | open   |

The contract for the refactor: every name listed above keeps the same call signature and semantics. Phases 1–4 introduce the seams (parameters, resolvers, config objects) without breaking the surface.

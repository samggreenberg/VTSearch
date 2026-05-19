# Changelog — `vtscore`

All notable changes to the `vtscore` library are documented here. The
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`vtscore` versions are tracked manually in `vtscore/__init__.py` and move
only when a release is cut — there is no auto-bump on commit. (The companion
[`vtsearch`](../README.md) application uses a git-derived timestamp version
instead, since every commit on `dev` is effectively a new app release.)

## [Unreleased]

_No changes yet._

## [0.1.0] — Initial release

The library was carved out of the `vtsearch` monolith over phases 0–8 of
[`docs/plans/extract-library.md`](../docs/plans/extract-library.md). The
0.1.0 release captures that work as the first publishable snapshot.

### Added

- **`vtscore.config.CoreConfig`** — dataclass for the knobs library code reads
  (`safe_thresholds`, `calibrate_count`, `calibration_fraction`,
  `enrich_descriptions`, `data_dir`, `saved_datasets_dir`, `detectors_dir`,
  `autopilot_goal_diversity`, `max_concurrent_dataset_downloads`,
  `max_concurrent_dataset_embeddings`, `inclusion`). `CoreConfig.from_settings()`
  is the app-side bridge; library-only consumers construct `CoreConfig`
  directly.
- **`vtscore.datasets`** — `Origin`, `LabeledElement`, `LabelSet`,
  `DatasetImporter`, folder / pickle / demo dataset loaders, the
  `MediaSource` abstraction (local_folder, http_archive, pullwrest), the
  `IMPORTER`-sentinel auto-discovery scanner, and bidirectional split / dedup
  helpers.
- **`vtscore.media`** — `MediaType`, `MediaEmbedder`, `MediaClipper`,
  `Processor` / `Detector` / `Localizer` / `Extractor` ABCs, the audio /
  image / text / video / document plugins, and `MediaResponse` (framework-
  agnostic HTTP response wrapper).
- **`vtscore.embedding`** — Lazy embedder loader (LAION-CLAP, SigLIP, X-CLIP,
  E5, DINOv2/v3, BGE, EUPE, LanguageBind), torch device selector, smart
  preload scheduler, cached `(N, D)` embedding matrix.
- **`vtscore.training`** — Generic learned-sort primitives: MLP build / train /
  weight-serialise, GMM and cross-calibration threshold solvers, safe-
  threshold blending, region-similarity scoring, SVM prototype.
- **`vtscore.detectors`** — Detector registry, JSON-backed store, vote-aware
  online training (`train_and_score`), origin resolver, label-sync,
  cross-dataset label restoration, labelset materialisation, and the
  labeling-session analyzer (`analyze_labeling_progress` + helpers).
- **`vtscore.eval`** — Offline text-sort / learned-sort evaluation, voting-
  iteration simulator, metric dataclasses (`QueryMetrics`,
  `LearnedSortMetrics`, `DatasetResult`), and `format_results_json`.
- **`vtscore.converters`** — `MediaConverter` ABC and the seven built-in
  converters: audio↔image / text, video→audio / image, document→image / text,
  image→text.
- **`vtscore.exporters`** — `LabelsetExporter` ABC and built-ins
  (`server_json_file`, `server_csv_file`, `webhook`, `email_smtp`, `gui`,
  `holder`).
- **`vtscore.labels`** — `LabelImporter` + `LabelsetSource` ABCs, registries,
  and bidirectional sync helpers.
- **`vtscore.plugins`** — `PluginRegistry` with sentinel-based discovery,
  eager construction by default, and `importlib.metadata` entry-point
  support so third-party packages can register plugins without
  monkey-patching.
- **`vtscore.concurrency`** — `AsyncJob` / `JobManager`,
  `cap_workers_by_memory`, the long-running-operation progress trackers
  (`ProgressTracker`, `LoadingTasksTracker`, dataset / sort / eval / find
  variants), and the per-thread progress hook (`set_thread_progress`).
- **`vtscore.state`** — Per-dataset `DatasetContext` and per-detector
  `DetectorContext`, context registries, pluggable
  `register_*_context_resolver()` hooks for app integration,
  `with_dataset_context` / `with_detector_context` thread-local bindings.
- **`vtscore.sync`** — `SyncSource[L,S]` ABC shared by labelset sources
  (library) and settings sources (app-side).
- **`vtscore.security`** — Path validation (`validate_server_filepath`,
  symlink-aware globbing), SSRF guard (`validate_url`), allowlist-based
  `safe_pickle_load` + `peek_pickle_dataset_summary`.
- **`vtscore.utils`** — `build_media_hit` (the canonical scored-media hit
  dict) and offline synthetic-media generators
  (`generate_audio_dataset` / `generate_image_dataset` /
  `generate_video_dataset`).
- **`vtscore.cli`** — Flask-free CLI entry points: `autodetect_main`,
  `autodetect_importer_main`, plus chunked variants. Reads `CoreConfig`,
  not `vtsearch.settings`.

### Architecture invariants

- **No Flask imports** anywhere under `vtscore/`. Verified by the
  `./run-tests.sh vtscore-clean` mode (installs a meta-path import hook that
  refuses `flask` / `werkzeug` / `flask_smorest` before collection).
- **No `vtsearch.settings` imports** in library-candidate modules.
  Configuration arrives via `CoreConfig` or a context object.
- **No hardcoded `data/` paths.** Every reference routes through
  `vtscore.config.DATA_DIR` (honouring `$VTSEARCH_DATA_DIR`), which is
  snapshotted into `CoreConfig.data_dir`.
- **No persisted embeddings or MLP weights.** Origins are the canonical
  persisted form; the library re-derives `origin → file → embedding → MLP`
  on demand. Detector JSON files store only `LabeledElement`s; dataset
  pickles are the one sanctioned vector store.

[Unreleased]: https://github.com/samggreenberg/vtsearch/compare/vtscore-0.1.0...HEAD
[0.1.0]: https://github.com/samggreenberg/vtsearch/releases/tag/vtscore-0.1.0

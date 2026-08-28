# Changelog - `vtscore`

All notable changes to the `vtscore` library are documented here. The
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`vtscore` versions are tracked manually in `vtscore/__init__.py` and move
only when a release is cut - there is no auto-bump on commit. (The companion
[`vtsearch`](../README.md) application uses a git-derived timestamp version
instead, since every commit on `dev` is effectively a new app release.)

## [Unreleased]

### Changed

- **Voted media are excluded from the calibrated threshold's haystacks**
  (issue #3308). The fold-anchored threshold estimator drops the voted
  items from every population sample it touches - each calibration fold
  model's corpus scores and the final model's realization sample - because
  those models were trained on the votes, so the votes' own scores under
  them are optimistically shifted (and the calibration votes previously sat
  in the haystack twice: once as free points, once as anchors). All the
  distributions in the quantile transfer now cover one identical
  population, the unlabeled remainder. New optional `voted_ids` parameter
  on `vtscore.detectors.training.train_and_threshold` (and the internal
  `_train_and_score_xy` / `_fused_threshold`); `train_and_score` and the
  labelset/model-loading pipelines pass it automatically, and omitting it
  keeps the historical include-everything behaviour. Thresholds move only
  where votes are a nontrivial share of the corpus (small datasets); on
  large corpora the change is bounded by the votes' share of the ≤50k
  haystack sample.

- **`calibration_fraction` defaults are now per-embedder, and `None` means
  "resolve it".** The shipped Train/Calibrate split of each calibration fold
  is keyed on the space the detector learns in (issue #3287):
  `PRODUCTION_SPLIT_BY_SPACE` in `vtscore.training.thresholds` maps
  `single_vector` → 0.3 and `patch` → 0.5, with `PRODUCTION_SPLIT = 0.5` as
  the unknown-space fallback, resolved by `production_split_for(patch_space=…)`
  (a three-state contract mirroring `production_schedule_for`) and, with the
  explicit-setting precedence, by
  `vtscore.detectors.training.resolve_calibration_fraction`. Accordingly the
  `calibration_fraction` parameters of `train_and_score`,
  `labelset_train_and_score`, `train_detector_from_origins`,
  `simulate_voting_iterations`, `run_voting_iterations_eval[_from_pickles]`,
  `eval_learned_sort`, and `run_eval` changed from `float = 0.5` to
  `float | None = None`, where `None` resolves to the per-space production
  default (the eval keys it on `patch_grid` presence, matching the app).
  `CoreConfig.calibration_fraction` is now `float | None` (`None` = no
  explicit user setting), and `vtscore.state.get_calibration_fraction()` /
  `set_calibration_fraction()` pass that tri-state through. Callers that
  passed an explicit fraction are unaffected; callers that *relied* on the
  implicit 0.5 default and want it back should pass `0.5` explicitly.

- **Train/Calibrate split sizes are dithered rather than rounded.**
  `calibration_folds` / `calibration_folds_cached` (and the grouped,
  bag-aware path behind them) now round a fractional split size up with
  probability equal to its fractional part, instead of calling `round`.
  The count is unbiased either way; what changes is that a *tie* no longer
  resolves the same way for every labelset of a given size. `round` is
  round-half-to-even, so at the default `calibration_fraction=0.5` the odd
  label's destination alternated with the label count - Train at
  `n % 4 == 1`, Calibrate at `n % 4 == 3` - and every threshold read off
  the fold models inherited that period-4 seesaw. Harmless for one
  detector, but any study that advances one label at a time saw it
  phase-locked across every run, where it survived averaging as a
  spurious 4-label ripple (issue #3286). Thresholds remain a pure,
  reproducible function of the labelset: the tie-break RNG is seeded from
  a digest of the labels and training vectors, so the same votes still
  give the same cut, and the calibration cache stays valid.

- **Results exporters declare which payload kinds they handle, instead of
  sniffing the dict shape.** `LabelsetExporter` is now `ResultsExporter`
  (the old name stays as a permanent module-level alias), and it exposes
  `export_find_results()` and `export_labelset()` alongside the existing
  `export_cli_detectors()`. `supported_payloads` is derived from which of
  those a subclass overrides, so each picker offers an exporter only for the
  kinds it can actually read and the export route answers 400 - rather than
  letting an exporter be handed a shape it doesn't understand and deliver an
  empty export while reporting success. `email_smtp` gained a labelset mode
  it never had.

  **Existing exporters keep working with no changes.** The default
  `export_find_results()` / `export_labelset()` both delegate to `export()`,
  so a plugin written against the single-method contract still runs and still
  does its own `if "labels" in results` check; it is credited with both
  payload kinds and logs one line at import pointing at the named methods.
  Migrating is a mechanical split of that `if` into two methods, and buys
  accurate picker filtering. See
  [`vtscore/docs/extending/results-exporters.md`](docs/extending/results-exporters.md).

- **`RemoteUnreachableError` now also covers a retryable HTTP status that
  outlives the retry budget.** `download_file_with_progress` and
  `fetch_text_with_retry` used to end a run of 500/502/503/504/429 responses by
  calling `raise_for_status()`, so the caller got a raw `requests.HTTPError`
  naming whichever CDN node the redirect landed on. Both now raise
  `RemoteUnreachableError` with a sentence naming the host and the status.
  Non-retryable statuses (404 and friends) still raise `HTTPError` unchanged,
  and gated 401/403 still raise `GatedResourceError`.
- **A multi-file demo download tolerates individual files the host refuses.**
  The per-file sets (Apollo 11, the Nixon tapes) set a failed file aside, retry
  it once after the rest of the set, then skip it with a `notify()` warning,
  failing the download only when more than a quarter of the set is missing.

- **`LINEAR_SVM_HEAD` is the production detector head**, replacing
  `LINEAR_HEAD`. Both sentinels build the same `Linear(D, 1)`; the new one is
  fitted by `vtscore.training.svm.fit_linear_svm_head` (squared hinge + L2 via
  liblinear, `class_weight="balanced"`) rather than by `train_model`'s balanced
  BCE loop. The fit delegates to `train_svm(kernel="linear")` — the very call
  the eval harness scores as its `svm_linear` arm — so the shipped head and the
  measured arm cannot drift apart, and `vtscore.eval.voting_iterations`'
  `PRODUCTION_HEAD` moves to `"linear_svm"` with it. `head="linear"` and
  `head="mlp"` remain as named eval arms. Callers that passed `LINEAR_HEAD` to
  match production must now pass `LINEAR_SVM_HEAD`; scores from a detector
  trained on the same labels will differ.
- **`train_svm` accepts `sample_weight`** (`decision_sigmoid` calibration
  only), mirroring `train_model`'s contract: per-row weights replace the
  `class_weight` balance rather than stacking on it. This is how region
  flooding weights a Bad image's many region rows down to one image's worth.

### Added

- **`vtscore.concurrency.notifications`** - `notify()`, `Notification` and
  `NotificationBroker`: a fan-out channel for one-off user-facing messages.
  The progress trackers publish *state* and an exception ends the operation;
  neither fits "we skipped 3 unreadable files but the other 900 imported
  fine". `notify()` is the third option — keep going, and say so. Consumers
  subscribe to the process-wide `notifications` broker (the app's SSE stream
  and the CLI printer both do); with no subscriber the message is still
  logged at a severity matching its level. `PluginBase.notify()` wraps it
  with the plugin's `display_name` as the source. The call never raises: bad
  levels degrade, long messages truncate, broken subscribers are swallowed.

- **`vtscore.security.login`** - the `LoginProvider` ABC,
  `DefaultLoginProvider`, the process-wide `set_login_provider` /
  `get_login_provider` registry, `get_user_data_dir()` and
  `is_safe_username()`, moved down from the `vtsearch` app tier. Path
  confinement (`vtscore.security.path_validation`) asks the active provider
  where the current user's data lives, so the abstraction had to be reachable
  in a process with no Flask in it — previously `get_file_access_base_dir()`
  raised `ImportError` there. An embedder can now opt into per-user
  confinement by registering a provider whose `get_user_data_dir()` returns a
  per-user subtree; the default stays single-user and unconfined.

- **`beats` audio embedder** - Microsoft's BEATs iter3+ AudioSet-2M
  self-supervised encoder, exposed as a 768-d audio-only embedder
  (`supports_text=False`). There is no `transformers` implementation, so the
  architecture is vendored in `vtscore.media.audio._beats_model` (DeepNorm
  residuals, a shared gated relative position bias, a weight-normalised
  convolutional position embedding) and the released MIT-licensed checkpoint
  is loaded onto it. Its Kaldi `compute-fbank-feats` front-end is ported from
  torchaudio's pure-PyTorch implementation rather than taken as a dependency,
  since torchaudio's wheels are built against a pinned torch.

### Changed

- **`clap_general` is the default audio embedder**, replacing `clap`.
  Measured on the full ESC-50 (2000 clips, all 50 categories),
  `laion/larger_clap_general` wins every comparison against
  `laion/clap-htsat-unfused`: text-sort mAP 0.869-0.895 vs 0.850-0.866,
  learned-sort mean F1 0.523-0.564 vs 0.457-0.529, and leave-one-out 1-NN
  accuracy 0.973 vs 0.958. `embedders_for_type("audio")[0]` and any caller
  that passes `embedder=""`/`None` now resolve to `clap_general`.
  `clap` is **not** removed - it stays a first-class explicit choice, ~2.1x
  faster and ~20% smaller, and existing pickles and detector JSONs recording
  `embedder: "clap"` keep resolving. Both are 512-d, so vectors from the two
  are dimension-compatible but not interchangeable.
- **The two general CLAP display names are now distinguishable.** `clap` reads
  "CLAP (general, faster)" (was "CLAP (general audio)") and `clap_general`
  reads "CLAP (general, larger)" (was "CLAP (general 2024)"); their progress
  labels are "CLAP Fast" and "CLAP General".
- **`fold_anchored_gmm_threshold` is the shipped decision threshold.** Per
  calibration fold, a semi-supervised 2-component mixture is fitted to that
  fold model's scores over the whole collection with the fold's *held-out*
  labels clamped to their component; each fold's cut is carried to the final
  model as a quantile and the folds are combined in quantile space. Anchor mass
  is 0.3 (each vote counts as three tenths of a haystack point) and the cut
  rule is `mid_tilt`: at Inclusion 0 the midpoint between the fitted component
  means — the interior optimum of a six-environment κ sweep — and away from 0
  that midpoint's combined quantile shifted by the rate-optimal cut's own
  displacement from its inclusion-0 position, so the fused threshold answers
  the Inclusion knob monotonically while reproducing the measured midpoint arm
  exactly where it was measured. The
  label-count-scheduled blend (`calculate_safe_threshold`) is now only the
  fallback for label sets too small to form calibration folds.
- **`gmm_cut_from_fit(rule="rate")` continues past the component means instead
  of falling back to the midpoint** when the density crossing has no root
  between them. The cut is read as the highest score at which the low component
  still out-densities the high one under the cost tilt, which makes it monotone
  in the Inclusion knob (and so keeps the included sets nested); it equals the
  old root wherever a crossing exists, including at every equal-weight cut.
  Once the crossing runs off the inter-mean interval the cut keeps moving,
  continuing past the edge by the log-cost excess times the mixture-weighted
  variance over the mean gap - the equal-variance crossing's own slope, so for
  equal-variance fits the continuation extends the interior crossing line
  seamlessly. Returning the bare edge there (the first form of this change)
  made the cut *constant* in the cost ratio, which flattened the composed
  `mid_tilt` quantile over whole bands of the Inclusion knob and silently
  collapsed the acquisition offset to a no-op inside them.
- **`vtscore.eval` defaults `safe_thresholds=True`**, matching the app; `False`
  is the no-fusion control arm. `eval_learned_sort` / `run_eval` lost the
  parameter entirely - they delegate to the production trainer, which has no
  such mode.
- **`gmm_cut_from_fit` returns `(cut, kind)` instead of `(cut, flag)`**, where
  *kind* is one of `CUT_KIND_INTERIOR` (`""`), `CUT_KIND_CONTINUED` or
  `CUT_KIND_DEGENERATE_MIDPOINT`. It is empty exactly when the old flag was 0,
  so `bool(kind)` is the previous "no interior stationary point" boolean; the
  non-empty values distinguish a cut *continued* past a component mean (still
  moving with the cost tilt, still the rate rule) from a fit too degenerate to
  express a boundary at all (a midpoint, constant in the tilt). Those two were
  indistinguishable before, which made a fallback countable but not
  attributable. **Breaking:** a caller comparing the second element to `0`/`1`
  should compare to `""` or wrap in `bool()`.

### Removed

- **`CoreConfig.safe_thresholds`**, `vtscore.state.get_safe_thresholds` /
  `set_safe_thresholds`, and the `safe_thresholds` parameter of
  `train_and_score`, `labelset_train_and_score`, and `run_learned_sort`. The
  fused threshold measured better than the alternative at every label count, so
  the switch was deleted rather than kept as a way to opt into a worse cut.
  **Breaking:** construct `CoreConfig` without the field, and drop the keyword
  from any `train_and_score` call.
- **`vtscore.training.thresholds.xcal_is_discarded`** - it existed to skip the
  fold training where the blend schedule zeroed the cross-cal cut, and the
  fold-anchored estimator needs those fold models at every label count.
- **`cross_calibration_threshold_cached`** - superseded by
  `calibration_folds_cached`, which returns the fold models alongside the
  orderings (plus `threshold_from_folds` for the cross-calibration cut).

### Added

- **`vtscore.state.current_user`** - Flask-free resolution of "who is this
  work for": a pluggable request-user resolver
  (`register_request_user_resolver`), the `thread_user` thread-local scope
  background jobs use to inherit a requester's identity, and the `"default"`
  fallback. Library code that needs a username now calls
  `vtscore.state.current_user.get_current_user()` instead of importing
  `vtsearch.auth`, which made `JobManager.start()` (and label sync, dataset
  loading, exporters, plugin templating) hard-require Flask at call time.
  `vtsearch.auth` re-exports every name, so there is still exactly one
  thread-local behind the app.
- **`FoldAnchoredCut`** / **`fit_fold_anchored_cut`** - the fitted estimator,
  split from the cut so a new Inclusion value can be re-cut arithmetically
  without refitting or re-scoring.
- **`inclusion_cost_weights`** - the single definition of what an Inclusion
  value costs in `(fpr_weight, fnr_weight)`, read by both the shipped threshold
  rule and the eval harness.

- **`LabelsetExporter.opens_url`** and the `"open_url"` response key - an
  exporter can return an `http(s)` URL for the frontend to open in a new
  browser tab, which is how a third-party site with no ingest API receives a
  labelset. Setting `opens_url = True` advertises it on `to_dict()` so the UI
  can label the button before the export runs.
- **`vtscore.security.url_validation.validate_browser_url`** - scheme
  allowlist for URLs the *user's browser* opens. Deliberately not the
  `validate_url` SSRF guard: no server-side request is made, so private hosts
  are legitimate targets and only the scheme is dangerous.
- **`open_url` exporter** - formats the labelset into a user-supplied URL
  template (`{ids}`, `{count}`), URL-encoding the joined identifiers,
  truncating to `max_items`, and refusing a URL past the ~2000-character
  practical limit.

## [0.1.0] - Initial release

The library was carved out of the `vtsearch` monolith and shipped as a
separate package. The 0.1.0 release captures that work as the first
publishable snapshot. See [`docs/architecture.md`](docs/architecture.md)
for the seven seams the refactor cut between vtscore and vtsearch.

### Added

- **`vtscore.config.CoreConfig`** - dataclass for the knobs library code reads
  (`safe_thresholds`, `calibrate_count`, `calibration_fraction`,
  `enrich_descriptions`, `data_dir`, `saved_datasets_dir`, `detectors_dir`,
  `autopilot_goal_diversity`, `max_concurrent_dataset_downloads`,
  `max_concurrent_dataset_embeddings`, `inclusion`). `CoreConfig.from_settings()`
  is the app-side bridge; library-only consumers construct `CoreConfig`
  directly.
- **`vtscore.datasets`** - `Origin`, `LabeledElement`, `LabelSet`,
  `DatasetImporter`, folder / pickle / demo dataset loaders, the
  `MediaSource` abstraction (local_folder, http_archive, pullwrest), the
  `IMPORTER`-sentinel auto-discovery scanner, and bidirectional split / dedup
  helpers.
- **`vtscore.media`** - `MediaType`, `MediaEmbedder`, `MediaClipper`,
  `Processor` / `Detector` / `Localizer` / `Extractor` ABCs, the audio /
  image / text / video / document plugins, and `MediaResponse` (framework-
  agnostic HTTP response wrapper).
- **`vtscore.embedding`** - Lazy embedder loader (LAION-CLAP, SigLIP, X-CLIP,
  E5, DINOv2/v3, BGE, EUPE, LanguageBind), torch device selector, smart
  preload scheduler, cached `(N, D)` embedding matrix.
- **`vtscore.training`** - Generic learned-sort primitives: MLP build / train /
  weight-serialise, GMM and cross-calibration threshold solvers, safe-
  threshold blending, region-similarity scoring, SVM prototype.
- **`vtscore.detectors`** - Detector registry, JSON-backed store, vote-aware
  online training (`train_and_score`), origin resolver, label-sync,
  cross-dataset label restoration, labelset materialisation, and the
  labeling-session analyzer (`analyze_labeling_progress` + helpers).
- **`vtscore.eval`** - Offline text-sort / learned-sort evaluation, voting-
  iteration simulator, metric dataclasses (`QueryMetrics`,
  `LearnedSortMetrics`, `DatasetResult`), and `format_results_json`.
- **`vtscore.converters`** - `MediaConverter` ABC and the seven built-in
  converters: audio↔image / text, video→audio / image, document→image / text,
  image→text.
- **`vtscore.exporters`** - `LabelsetExporter` ABC and built-ins
  (`server_json_file`, `server_csv_file`, `webhook`, `email_smtp`, `gui`,
  `holder`).
- **`vtscore.labels`** - `LabelImporter` + `LabelsetSource` ABCs, registries,
  and bidirectional sync helpers.
- **`vtscore.plugins`** - `PluginRegistry` with sentinel-based discovery,
  eager construction by default, and `importlib.metadata` entry-point
  support so third-party packages can register plugins without
  monkey-patching.
- **`vtscore.concurrency`** - `AsyncJob` / `JobManager`,
  `cap_workers_by_memory`, the long-running-operation progress trackers
  (`ProgressTracker`, `LoadingTasksTracker`, dataset / sort / eval / find
  variants), and the per-thread progress hook (`set_thread_progress`).
- **`vtscore.state`** - Per-dataset `DatasetContext` and per-detector
  `DetectorContext`, context registries, pluggable
  `register_*_context_resolver()` hooks for app integration,
  `with_dataset_context` / `with_detector_context` thread-local bindings.
- **`vtscore.sync`** - `SyncSource[L,S]` ABC shared by labelset sources
  (library) and settings sources (app-side).
- **`vtscore.security`** - Path validation (`validate_server_filepath`,
  symlink-aware globbing), SSRF guard (`validate_url`), allowlist-based
  `safe_pickle_load` + `peek_pickle_dataset_summary`.
- **`vtscore.utils`** - `build_media_hit` (the canonical scored-media hit
  dict) and offline synthetic-media generators
  (`generate_audio_dataset` / `generate_image_dataset` /
  `generate_video_dataset`).
- **`vtscore.cli`** - Flask-free CLI entry points: `autodetect_main`,
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

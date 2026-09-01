# Codebase audit — September 2026 (structure & organization)

**Background.** A structure-and-organization audit was run at `0de6fb63` (dev):
six specialist reviewers over disjoint areas (vtscore core, vtscore ML,
eval/experiments, vtsearch app tier, Angular frontend, tests/tooling/docs),
each instructed to verify every claim by reading the code and to skip anything
already tracked in [`codebase-audit-2026-08.md`](codebase-audit-2026-08.md).
Unlike the August audit (defects), this one targets **tech debt**: god modules,
duplicated logic, dead code, layering violations, and unnecessary complexity.
The August audit's improvement proposals remain open and are complementary;
a few items below note when they should ship together with one of them.

**How to work this file.** Each item is self-contained: evidence with
file:line pointers, a remediation direction, a size (S/M/L), and a
**recommended Claude model** for the implementer (per the CLAUDE.md ladder —
Haiku 4.5 mechanical, Sonnet 5 normal, Opus 4.8 regression-prone/design-heavy).
Ground rules for implementer sessions:

- Base on `dev`; one item (or one flagged bundle) per PR; run a **full**
  `./run-tests.sh` before pushing. Regenerate the OpenAPI snapshot
  (`cd frontend && npm run regenerate-openapi-snapshot`) whenever a route or
  schema changes.
- When you pick up an item, either file a GitHub issue for it per the
  CLAUDE.md promotion rules (`claude` label, model line, then replace the
  body here with a one-line `- [ ] #N — title` pointer) or just ship it and
  delete the item. Never delete a `<!-- item-sep -->` sentinel.
- **External extension APIs must be preserved** (plugin ABCs, registry
  sentinels, `vtscore.*` entry-point groups, the documented public `vtscore`
  API). Items that touch that surface say so and name the shim to keep.
  Internal surfaces, the REST API (with a same-PR frontend update), and
  persisted data may break freely.
- Items flagged **[re-pin]** move logic that `scripts/check-eval-app-sync.py`
  pins; update the `Mirror` paths and run `--update` after reconciling.
- Items flagged **[owner decision]** need a product call from the user before
  implementing — put the question through `AskUserQuestion`, don't guess.

**Suggested first wave** (high value, low risk): the dead-code deletions
(frontend services, one-off scripts, dead documented vtscore API), the
mechanical Haiku batches (progress-callback dedup, `.sr-only`, empty test
dirs, stale allowlists), the eval-harness default fixes
(`CALIB_SAFE_THRESHOLDS`, stale study defaults, re-pool arms), and the
converters-`print()` sweep. The big splits (`thresholds.py`,
`state/core.py`, `voting_iterations.py`, `browse-canvas`) and the settings
rework are the highest-payoff items but need Opus-tier care.

---

## Library tier — god modules & misplaced code

<!-- item-sep -->

- **`vtscore/config.py` is five unrelated modules in one file** — `vtscore/config.py` (933 lines) (high impact)

  Contents in order: data paths (21-26); thread/decode-worker sizing (32-86); CUDA probing + device cache (89-260); precision/autocast (262-330); a ~180-line prose block plus six functions on transformers image-processor backend negotiation (330-591); scalar constants (601-660); UMAP projection defaults (662-692); ~95 lines of HF model IDs and per-checkpoint audio constants (694-788); `CoreConfig` and its builder hook (790-933). Nothing shares state except the module name, and this is the single most-imported module in `vtscore`.

  *Direction:* Split into a `vtscore/config/` package along those seams (paths, runtime, device, processor_backend, models, core_config), with the package `__init__` re-exporting every public name so `vtscore.config.X` keeps resolving (it is documented public API in `vtscore/docs/packages/config.md` — update that file in step). The groups have no cross-references except `DEVICE` → `resolve_device`.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`vtscore/state/core.py`: two god objects plus four unrelated subsystems** — `vtscore/state/core.py` (1501 lines) (high impact)

  `DatasetContext.__slots__` (385-506) spans five independent concerns: medias/revision, embedding-matrix cache, region-matrix cache, secondary lookup indexes, ten VTSBrowse projection slots, embedder binding. `DetectorContext` (722+) is similar. Around them sit the frozen-sentinel machinery (101-221), hook registration (233-298), and the two context stores (931-1368) — none of which import each other.

  *Direction:* Split into sentinel, hooks, dataset-context, and detector-context modules under `vtscore/state/`; within `DatasetContext`, group the ten browse slots into a `BrowseState` sub-object and the matrix-cache slots into a caches sub-object (both are already accessed as groups). `vtscore/state/__init__.py` already re-exports everything, so no external import changes. Do together with the cache-invalidation item below.

  *Size:* L · *Model:* Opus 4.8 (`__slots__` reshuffling around `_freeze_into`'s slot walk is regression-prone)

<!-- item-sep -->

- **`DatasetContext` cache invalidation is hand-copied across four modules by poking private slots** — `vtsearch/routes/projection.py:428` (medium impact)

  `clear_medias()` in `vtscore/state/__init__.py:161-176` assigns 12 private slots directly; the same pattern recurs in `vtscore/embedding/matrix.py:466`, `vtscore/state/media_lookup.py:128-131`, and — across the tier boundary — `vtsearch/routes/projection.py:428-429` and `:576-577`. The clear-list and `__slots__` have already drifted: `clear_medias` misses `_emb_sidecar_disabled`, `_origin_key_index`, `_md5_index`, `_name_index`, `_lookup_index_revision`, and the four `_subset_*` slots.

  *Direction:* Add `DatasetContext.reset_derived_caches(*, matrices=True, lookups=True, projection=True, subset=True)` defined next to `__slots__` so the two cannot drift; rewrite the five call sites to use it.

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **`vtscore/training/thresholds.py` — 2680 lines, five clean seams** — `vtscore/training/thresholds.py` (high impact) **[re-pin]**

  Five non-overlapping concerns: knob semantics (1-262), the 1-D Gaussian mixture (264-960), population anchoring (961-1500), conformal/cross-calibration folds (1501-2550), and the blend tail (2553-2680, already a thin shim over `vtscore/training/blend_schedules.py`). The GMM layer has zero references to the conformal layers.

  *Direction:* Split into a `vtscore/training/thresholds/` package (knobs, gmm, anchored, conformal, blend) whose `__init__` re-exports everything so `vtscore.training.thresholds.X` keeps resolving. `scripts/check-eval-app-sync.py:268/291/320` pin `resolve_exclusion_floor`, `fold_anchored_gmm_threshold`, and `_rate_cut` by dotted path — verify the gate resolves through the package re-export and update the `app=` strings in the same PR. Fold in the tiny duplicate: `calculate_gmm_threshold` (1493-1499) duplicates `fit_gmm_threshold` (2563-2568) verbatim; make the former call the latter (behaviour must stay bit-identical — it is a documented public export).

  *Size:* L · *Model:* Opus 4.8

<!-- item-sep -->

- **`vtscore/media/embedder.py` is two unrelated modules welded together** — `vtscore/media/embedder.py:449` (high impact)

  Lines 449-891 are a self-contained monkey-patch subsystem (tqdm interception, `torch.load`/safetensors patches, thread-local progress) and 104-232 are torch tensor/device helpers; the `MediaEmbedder` ABC only starts at line 894. All 21 embedder modules pay the read/import cost of ~450 lines that have nothing to do with the ABC they implement.

  *Direction:* Extract a load-progress module (233-450 + 449-891) and a torch-ops module (104-232) under `vtscore/media/`, re-exporting the public names (`intercept_tqdm_progress`, `intercept_weight_loading_progress`, `timed_progress`, `embedder_load_setup`, `load_pretrained_local_first`, `IMPORT_MODULE_ESTIMATES`) from `vtscore.media.embedder`. **The re-export shim is mandatory**: `vtscore/docs/extending/embedders.md:185-192` shows third-party embedders importing these names from `vtscore.media.embedder`.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Demo-source machinery is extracted for `image` but inlined for `audio`/`video`/`text`** — `vtscore/media/audio/media_type.py:512` (medium impact)

  The image media type delegates demo catalogs to `vtscore/media/image/_demo_sources.py` and `vtscore/media/image/_demo_categories.py`, keeping `media_type.py` at 213 lines. The other three types inline the identical machinery: `vtscore/media/audio/media_type.py:512-1290` (file is 1378 lines), `vtscore/media/video/media_type.py:386-730`, `vtscore/media/text/media_type.py:169-628`. The per-category slice loop is written three times (`_slice_by_category` in `image/_demo_sources.py:656`, closures at `audio/media_type.py:1082` and `text/media_type.py:471`).

  *Direction:* Mirror the image layout for audio/video/text (a `_demo_sources.py` per type); promote the three slice-loop copies to one `demo_slice_by_category(...)` next to `demo_slice` in `vtscore/media/base.py:205`. Pure moves, no logic change.

  *Size:* L · *Model:* Haiku 4.5

<!-- item-sep -->

- **`labeling_progress.py`: a single-slot cache spread across 15 module globals** — `vtscore/detectors/labeling_progress.py:53` (medium impact)

  Lines 53-124 declare 15 module-level mutable cache variables threaded through eleven `global` statements; the identity stamp (`_bind_cache_identity`) "must be called with `_progress_lock` held, at the top of every entry point" — an invariant nothing enforces (the class of bug #2914 fixed once already). The cache holds exactly one (dataset, detector) pair, so switching detectors discards work.

  *Direction:* A `_ProgressCache` dataclass holding the twelve fields plus its key, with a small bounded `OrderedDict` of caches keyed by (dataset, detector). Entry points become `cache = _cache_for(key)` under the lock; `_bind_cache_identity` and the globals disappear. Keep `_compute_smart_status` / `_compute_stable_status` / `_compute_span_status` in this file — `scripts/check-eval-app-sync.py:149/169/183` pin them by dotted path.

  *Size:* M · *Model:* Opus 4.8 (lock ordering vs `_state_lock` is documented and fragile)

<!-- item-sep -->

- **`vtscore/state/` holds ~1,400 lines of pure algorithms that touch no state** — `vtscore/state/coverage_atlas.py` (medium impact)

  `coverage_atlas.py` (832 lines, hierarchical k-means / vMF moments), `near_dupes.py` (398 lines, pHash/SimHash), and `sort_results_cache.py` (164 lines, a generic LRU) reference no `DatasetContext` and no `_state_lock`. The atlas *wiring* is a separate module, `vtscore/state/coverage.py` — a near-homograph pair where one is wiring and one is the algorithm.

  *Direction:* Move the three algorithm modules out of `state/` (e.g. under `vtscore/utils/` or a new clustering home), keeping the `vtscore.state` re-exports (`build_coverage_atlas*`, `collapse_near_duplicates`, `phash_image`, `simhash_text` are re-exported from both `vtscore.state` and `vtsearch.state`). Rename or fold `state/coverage.py` so the wiring/algorithm distinction is visible.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **`vtscore/datasets/loader.py` is a re-export façade pinned in place by test monkeypatch targets** — `vtscore/datasets/loader.py:123` (medium impact)

  The module re-exports metadata parsers and security names, then imports `loader_folder` / `loader_pickle` / `loader_demo` at the *bottom* of the file with `# noqa: E402` while `loader_pickle.py:12` imports back up from it — a deliberate circular-import dance. `loader_demo.py` re-imports the parent at call time purely "to keep the existing test patches working", and consumers import parsers through the façade instead of their real home (`vtscore/datasets/metadata.py`).

  *Direction:* Repoint the ~10 lazy imports in `vtscore/media/` at `vtscore.datasets.metadata` directly; repoint `loader_demo.py`'s `_loader.` lookups (and the corresponding test `patch(...)` targets) at the real modules; delete the `_apply_converter_to_demo` back-compat alias (only referenced by `tests/converters/test_document_and_converters.py:769-791` — point those at `vtscore.converters.runner`). Then the bottom imports can move to the top.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`evt_mixture.py`: 487 lines of unshipped research arm inside the semver'd library tier** — `vtscore/training/evt_mixture.py` (low impact)

  Only importers are two `tests_lib/` files and `scripts/experiments/calibration/`; no production module imports it and it is not in `vtscore/training/__init__.py`'s exports. External consumers pip-install an unshipped hypothesis indistinguishable from the production `thresholds.py` beside it.

  *Direction:* Move under a research/ subpackage (or into the experiment tree) and update the three import sites; at minimum add a module banner mirroring the `eval_only` embedder convention.

  *Size:* S · *Model:* Haiku 4.5

---

## Library tier — duplication

<!-- item-sep -->

- **Two independent implementations of "reproduce a clip's bytes from `origin.params`"** — `vtscore/media/lazy_clip.py:172` and `vtscore/detectors/resolver.py:469` (high impact)

  `clip_recipe`/`_apply_recipe` and `_clip_audio_to_bytes`/`_clip_image_to_bytes`/`_clip_text_to_bytes` both rederive clip bytes from origins and have already drifted three ways: the archive-member guard exists only in `lazy_clip`, the text `clip_index` branch only in `resolver`, and converter replay diverges completely (`_converter_recipe` vs `_converter_origin_to_chain`). Both also import the private `_wav_slice` from `vtscore/media/audio/clipper.py` across packages. This is the load-bearing path of the origins-are-canonical invariant, implemented twice.

  *Direction:* Promote one public clip-replay module under `vtscore/media/` covering all four kinds (audio, image, text, converter) plus the archive-member guard; have `resolver._apply_clip_and_embed` call it. Move `_wav_slice`/`_wav_duration` out of the clipper module into a public audio decode home. Each divergence may be load-bearing for its own callers — check before unifying.

  *Size:* M · *Model:* Opus 4.8

<!-- item-sep -->

- **`image/_demo_sources.py`: five copies of the same clip-dict builder** — `vtscore/media/image/_demo_sources.py:1088` (medium impact)

  `_embed_file_images` (1088), `_embed_pil_pages` (1139), `_embed_cifar_arrays` (1185), `_embed_vg_images` (1253), `_embed_openlogo_images` (1312) each repeat the embedder-load guard, the clip-id seed, the progress preamble, and a 15-key media-dict literal. VG and OpenLogo differ by one filename expression.

  *Direction:* One `_emit_image_clip(...)` builder plus one `_embed_loop(...)` driver taking a per-source render callable; collapse VG/OpenLogo into a multi-label variant parameterized by the filename rule.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Clipper family: duplicated tiling math, segment emission ×3, six identical no-op clippers** — `vtscore/media/audio/clipper.py:145` (medium impact)

  `SoundTilingClipper` and `VideoTilingClipper` (`vtscore/media/video/clipper.py:66-125`) share identical constructor validation, tile-start arithmetic, and `parameters` dicts. The segment→tiles emission is byte-identical in `SoundSilenceClipper.clip` (343-366) and `SoundSpeechActivityClipper.clip` (682-705) with a third copy in the tiling clipper. Six `*DefaultClipper` classes across the media types are all `clip() -> [media]` with a per-type name.

  *Direction:* (a) a shared `tile_starts(total, duration, min_overlap)` helper + shared parameters block in `vtscore/media/clipper.py`; (b) a module-private `_emit_wav_segments` in the audio clipper used by all three; (c) a concrete `DefaultClipper(name, media_type, description)` base. Don't change the `MediaClipper` ABC's abstract method set (documented plugin API), and keep every registered clipper `name` string (persisted in `origin.params`).

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Six near-synonymous "which embedder?" wrappers over two canonical resolvers; one is dead** — `vtscore/detectors/training.py:31` (medium impact)

  On top of `score_marker_embedder_for_snap` / `keying_embedder_for_snap` (`vtscore/embedding/binding.py:400/413`) sit `_score_embedder_for_snap`, `detector_score_embedder`, `_patch_embedder_for_snap` (`vtscore/detectors/training.py:31/54/72`), `_detector_embedder` (`vtscore/detectors/labelset_training.py:67`), and `_embedder_for_active_dataset` (`labelset_training.py:51`, **zero callers repo-wide** — dead). Five call sites conjure `SimpleNamespace(embedder_type=...)` to feed `keying_embedder_for_snap`.

  *Direction:* Delete the dead one; inline the two trivial delegators; add `slot_embedders_for_snap` and `keying_embedder_for_type` to `binding.py` and drop the `SimpleNamespace` carriers. Resolved values must stay identical (`_blend_schedule_for_snap` and `resolve_calibration_fraction` depend on them).

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **Four copy-paste registries in `vtscore/media/__init__.py`** — `vtscore/media/__init__.py:58` (low impact)

  Media types, clippers, cleaners, and embedders each get a five-function registry block; clippers and cleaners are character-identical modulo the word. ~130 lines where ~40 would do.

  *Direction:* A private generic `_PluginRegistry` with optional filter/sort hooks, instantiated four times; keep every public function name and exception shape (this is the plugin registration API used by discovery and entry points).

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **`_default_progress()` copy-pasted into six modules; `ProgressCallback` declared ten times** — `vtscore/datasets/loader.py:73` (low impact)

  Byte-identical bodies in `vtscore/datasets/loader.py:73-86`, `vtscore/datasets/downloader/core.py:337-346`, `vtscore/datasets/ingest.py:31-39`, `vtscore/datasets/archive.py:95-99`, `vtscore/datasets/importers/http_archive/__init__.py:38`, `vtscore/converters/runner.py:56`; the type alias is redeclared in ten files.

  *Direction:* Export `ProgressCallback` and a `resolve_progress_callback()` from `vtscore/concurrency/progress.py`; replace all definitions with imports. Purely mechanical; also a prerequisite for the legacy-progress removal below.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **Two background-import pipelines with a copy-pasted error taxonomy** — `vtscore/datasets/load_pipeline.py:417` (medium impact)

  `_run_origin_load_in_background` (417-665) and `_stage_importer_in_background` (838-966) duplicate the tracker-creation / thread-user / daemon-thread / finally-cleanup harness, and `stage_task:932-950` reproduces `_handle_load_failure:352-359`'s missing-dependency and OOM strings verbatim.

  *Direction:* Extract one `_run_import_task(...)` harness holding tracker creation, user scoping, **gate acquisition**, the except-taxonomy, timing recorder, and cleanup; both flows supply only their body. This also closes the August audit's "staging imports bypass the concurrency gates" item as a side effect — delete that item there when this ships.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Streaming atomic-write ritual and JSON label extraction hand-rolled repeatedly** — `vtscore/exporters/server_csv_file/__init__.py:126` (low impact)

  The tmp-name/fsync/`os.replace`/unlink ritual is duplicated in the CSV and JSON file exporters (the exact ritual `vtscore/io.py` exists to own, which lacks a streaming variant); JSON label extraction is implemented three times across `vtscore/labels/importers/server_json_file/__init__.py:70-76` and `vtscore/labels/sources/server_json_file/__init__.py:38-60`.

  *Direction:* Add an `atomic_write_stream(path)` context manager to `vtscore/io.py` and adopt it; extract one shared label-extraction helper.

  *Size:* S · *Model:* Haiku 4.5

---

## Library tier — dead code, unkept promises, misc

<!-- item-sep -->

- **Dead documented `vtscore` API: four promises the code doesn't keep** — `vtscore/embedding/loader.py:623` (medium impact)

  (1) `get_clap_model`/`get_xclip_model`/`get_e5_model` (`loader.py:623-653`) claim "existing callers … continue to work"; there are zero callers repo-wide. (2) `MediaClipper.resolve_for_durations` (`vtscore/media/clipper.py:158`) is documented in three extension guides as a load-time hook but is never called by anything (only a no-op test at `tests/detectors/test_clippers.py:2459`). (3) `vtscore.timing.profile_covers` (`vtscore/timing/profile.py:568`) — zero callers. (4) `apply_converter_to_demo(embedder_name=...)` carries a noqa'd dead parameter.

  *Direction:* Delete all four plus their doc entries (`vtscore/docs/packages/embedding.md`, `vtscore/docs/packages/timing.md`, `vtscore/docs/extending/clippers.md`, `docs/EXTENDING-media.md`) in the same commit. If `resolve_for_durations` is a planned hook, wire it next to `resolve_for_media` and test it instead. This is a documented-API break — deliberate, flagged here per policy.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **`detectors/resolver.py`'s pluggable-resolver registry has zero registrants and never has** — `vtscore/detectors/resolver.py:51` (medium impact)

  Two Protocols, two globals, two public `register_*_resolver` functions, and an auto-wire dance (~85 lines) exist to defer a same-tier import; the `except ImportError: pass` converts a broken install into "every label fails to resolve". No caller of either register function exists anywhere; the only references are in `vtscore/docs/packages/detectors.md`.

  *Direction:* Inline the two default resolvers as direct imports and delete the machinery, updating `detectors.md` in the same change; or keep the two `register_*` functions as documented overrides, delete only the auto-wire dance, and add a test registering a fake resolver.

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **`labelset_ops.py` façade is bypassed by 19 of 20 call sites, including for private symbols** — `vtscore/detectors/labelset_ops.py:1` (low impact)

  Only `vtsearch/routes/detectors/registry.py` uses the façade; everything else imports the sibling modules directly, and two route modules import the *private* `_label_sync_write_lock` / `_merge_labelsets_across_datasets` from `vtscore/detectors/label_sync.py`.

  *Direction:* Delete the façade and let the siblings be the API (matches actual usage), promoting the two private symbols to public names; or make the façade real with a lint/test. Deleting is less work.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **Every shipped converter reports errors with `print()`** — `vtscore/converters/audio2image.py:20` (low impact)

  21 `print()` calls across `vtscore/converters/`, all in `except` blocks that then `return []`; no converter module except `base.py` imports `logging`. The library tier's shipped templates teach plugin authors the wrong pattern, and a failing corpus floods stdout with unlevel'd lines.

  *Direction:* Per-module `logging.getLogger(__name__)`; `log.warning` for per-item failures, `log.error(..., exc_info=True)` where an exception is in hand; note the convention in `vtscore/docs/extending/converters.md`.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **`image_response` is a load-bearing media-type hook the ABC never declares** — `vtscore/media/base.py:474` (medium impact)

  Four media types implement `image_response(media)`; the route layer consumes it via `getattr(mt, "image_response", None)` (`vtsearch/routes/media/list.py:792`, `vtsearch/routes/detectors/labels.py:517`); neither it nor `ensure_thumbnail_bytes` appears in `vtscore/docs/extending/media-types.md`. `load_demo_source`'s declared `**kwargs` tail is contradicted by all four implementations, which spell out the same four extra parameters.

  *Direction:* Declare `image_response` on `MediaType` with a `return None` default, replace the `getattr` sites with direct calls, document both hooks, and give `load_demo_source` an explicit typed signature (keep a trailing `**kwargs` to avoid breaking third-party overrides, or treat as a documented break).

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **`_emit_converted_demo_outputs` skips the sub-output disambiguators its folder-path twin applies** — `vtscore/converters/runner.py:468` (low impact)

  The folder path stamps `converter_out_index`/`converter_n_out`/`converter_content_hash` onto each of a source's N outputs (258); the demo path (468-497) passes the flat origin through, so N converted demo medias share one origin and lazy replay cannot distinguish them.

  *Direction:* Call `_origin_with_disambiguators` from the demo path too, or collapse the two emitters into one (their bodies differ only in which fields supply source path and category).

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **Never-implemented Holder / ReCaller / PullWrest integrations ship as registered plugins (~865 lines)** — `vtscore/exporters/holder/__init__.py:44` (medium impact) **[owner decision]**

  Every I/O function in four plugin packages raises `NotImplementedError("TODO: implement ...")`; all are registered (sentinels present) and merely set `hidden_from_picker = True`. The payload-shaping code around the stubs can never be exercised, and `holder` still appears in the generated exporter listing in `vtscore/docs/packages/exporters.md`.

  *Direction:* Ask the owner: delete (git history preserves them), move out of the auto-discovered directories, or keep-as-templates. If they stay registered, add a test asserting no registered plugin's entry point raises `NotImplementedError`, so the stub count can't grow.

  *Size:* S · *Model:* Haiku 4.5 once decided

<!-- item-sep -->

- **Small vtscore batch (one PR)** — `vtscore/concurrency/async_jobs.py:523` (low impact)

  (a) `JOB_MANAGERS` (523-527) lists only user-visible managers, so `reset_all_async_jobs_for_tests` re-lists hidden ones by hand and misses `archive_thumbnail_jobs` (516) — one registry with a `user_visible` flag. (b) Two families construct `PluginRegistry` directly (`vtscore/datasets/sources/__init__.py:27`, `vtscore/converters/__init__.py:25`) while ten use `make_plugin_registry` — pick one. (c) `SAVED_DATASETS_DIR` (`vtscore/datasets/registry.py:43`) is referenced only by `.vulture-whitelist.py` and misstates the live value — delete both.

  *Size:* S · *Model:* Haiku 4.5

---

## Concurrency & progress

<!-- item-sep -->

- **Two parallel dataset-progress systems; the "legacy" one is still load-bearing** — `vtscore/concurrency/progress.py:799` (high impact)

  The global `dataset_progress` singleton coexists with the per-task `loading_tasks` registry; `get_progress()` falls back to the global, which then needs bespoke support everywhere (`LEGACY_PROGRESS_TARGET`, special cases in `_cancel_acknowledged` and `cancel_dataset_progress`, and `_park_global_progress_if_orphaned` in `vtscore/datasets/load_pipeline.py:399-414` — an admission that phantom progress bars are a recurring bug class, #3167). Six library modules default their callback to the global `update_progress`.

  *Direction:* After the `resolve_progress_callback` dedup lands, make the resolution return a no-op when no thread callback is bound, then delete `dataset_progress`, `update_progress`, and every special case; the SSE `"dataset"` channel becomes a task channel.

  *Size:* M · *Model:* Opus 4.8 (cancellation semantics and the SSE contract)

<!-- item-sep -->

- **`AsyncJob` re-implements `ProgressTracker` with a third cancellation mechanism** — `vtscore/concurrency/async_jobs.py:45` (medium impact)

  `AsyncJob` duplicates the tracker's field set without ETA smoothing, step weights, or `subscribe()` — which is why training/projection jobs poll over REST while imports push over SSE. Cancellation exists three ways, all raising the same `CancelledError`.

  *Direction:* Give `AsyncJob` a `ProgressTracker` field and forward to it; register a jobs SSE channel in `vtscore/concurrency/events.py`; make `bind_job_cancellation` set the tracker's cancel event rather than a second one.

  *Size:* M · *Model:* Opus 4.8 (pending-slot/coalescing semantics in `_run_inner` are easy to break)

<!-- item-sep -->

- **Three thread-spawn idioms in the app tier** — `vtsearch/routes/datasets/staging.py:376` (low impact)

  `JobManager`, `vtsearch.threading.spawn` (replays user + dataset + detector thread-locals), and a raw `threading.Thread` at `staging.py:376` that hand-rolls the context replay.

  *Direction:* The raw thread should use `spawn`; note in `vtsearch/threading.py`'s docstring when each of the two remaining idioms applies.

  *Size:* S · *Model:* Haiku 4.5

---

## Layering & host seams

<!-- item-sep -->

- **Nine ad-hoc app→library hook seams, each hand-rolled** — `vtscore/state/core.py:240` (medium impact)

  Nine `register_*` seam registrations (request predicate, dataset/detector resolvers, request-user resolver, setting persisters, CoreConfig builder, achievement recorders, last-embedder hook, plugin families), each a module global plus a setter, with two different validation strategies and no test reset; wiring is spread over five `register_app_*` functions in `vtsearch/shim/__init__.py`.

  *Direction:* One typed host-seams registry in `vtscore` with `install(**seams)`, introspection, and `reset_seams()` for tests; keep the existing `register_*` names as one-line forwarders (referenced from `vtsearch/shim` and `vtscore/docs`).

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`vtscore` imports `vtsearch` in the embedding loader** — `vtscore/embedding/loader.py:302` (low impact)

  `from vtsearch.logging_config import install_transformers_logging_bridge` inside `initialize_models`, wrapped in a bare try/except. Same layering violation as the August audit's `vtscore/cli.py:332` achievements item — ship the two together: a `vtscore`-side hook that `vtsearch` registers at startup (or fold both into the host-seams registry above).

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **`CoreConfig` is used as a global getter, not the snapshot it was designed as** — `vtscore/config.py:793` (medium impact)

  All 14 call sites call `CoreConfig.from_settings()` ad hoc, each invoking ~18 settings getters through the shim; the design comment at 793-816 still says "this class is unused at runtime" — stale. The frozen-value-object abstraction buys nothing while costing a full settings snapshot per lookup.

  *Direction:* Either pass one snapshot down per operation (the original design) or replace with direct getter calls; delete the stale comment either way.

  *Size:* M · *Model:* Opus 4.8

<!-- item-sep -->

- **`PluginBase` auto-derivation driven by three hardcoded tables of client names and emoji** — `vtscore/plugins/__init__.py:262` (medium impact)

  `_PLUGIN_NAME_SUFFIXES` (16 order-dependent literals), `_PLUGIN_FAMILY_BASE_NAMES` (14 class names that must skip derivation), and `_FAMILY_STOCK_ICONS` (seven emoji compared by codepoint to detect "author didn't pick an icon"). Adding a plugin family means editing three literal tables; forgetting one silently stamps a derived name onto an abstract base.

  *Direction:* Mark family bases with the existing `_is_plugin_family_base` opt-in on each base class; derive the strippable suffix from the nearest family base's `__name__`; replace the icon table with an explicit stock-icon marker. **Derived `name` values are registry keys and appear in entry-point configs and persisted settings — add a golden-list test asserting every in-tree plugin's derived name is unchanged.**

  *Size:* M · *Model:* Opus 4.8

---

## App tier — settings

<!-- item-sep -->

- **Settings keys are declared in four hand-synced places; the sync is already broken** — `vtsearch/settings_models.py:226` (high impact)

  The pydantic models (57 fields) are mirrored by 66 hand-written `TYPE_CHECKING` stubs in `vtsearch/settings.py:40-148`, two marshmallow schemas in `vtsearch/schemas/settings.py`, and a derived setter table in `vtsearch/routes/settings/api.py:345-369`. Measured drift: 8 fields lack stubs (hence 28 type-ignore suppressions), 4 fields are absent from `AppSettingsSchema` so `GET /api/settings` silently drops them (`projection_n_neighbors`, `projection_min_dist`, `recent_sessions`, `default_settings_source`), and the stub block's maintenance instruction names symbols that no longer exist.

  *Direction:* Generate both marshmallow schemas from the pydantic models at import time; replace the `TYPE_CHECKING` block with a generated `.pyi` or `__getattr__` typing; add a `client_only` marker so pure-UI prefs (the `popup_*`/`browse_details_*` family, which the backend never reads) skip accessor generation and ride one passthrough dict.

  *Size:* L · *Model:* Opus 4.8

<!-- item-sep -->

- **Migration shims for old persisted formats, which CLAUDE.md explicitly forbids** — `vtsearch/settings_store.py:744` (medium impact)

  Four live violations: `_maybe_migrate_legacy_settings` (744-812), `coerce_animation_mode` (`vtsearch/settings_models.py:59-80`), the duplicate `@pre_load` coercion (`vtsearch/schemas/settings.py:260-275`), and the legacy merge-order re-routing in `get_all()` (`vtsearch/settings.py:600-610`) — plus `_DEFAULT_USER_FALLBACK_KEYS` and the `_read_value` read-through branch that exist only to serve the unmigrated shape.

  *Direction:* Delete all of it; on load, drop unrecognized/uncoercible values and let pydantic defaults apply. Persisted data may break freely per policy — mention the break in the PR.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **The settings-source sync engine: ~600 lines and 8 REST endpoints with no SPA consumer** — `vtsearch/routes/settings/sources.py:52` (high impact) **[owner decision]**

  None of the eight `/api/settings-sources*` / labelset-source management endpoints is called by the frontend (verified against path literals and generated-client method names; the SPA's only adjacent call is `moveLabelsetSourceFile`, which lives in the detectors-registry routes). Behind them, `vtsearch/settings_store.py` carries the dirty-key/sync-marker state machine on the hot path of every settings read. Sub-finding: the `.syncmark` cross-worker dedup layer (92-155) guards against multi-worker races, but `gunicorn.conf.py:28` hardcodes `workers = 1` and all Dockerfiles use it — dead by configuration. Note the *labelset*-source sync itself (detector label sync at load) is live vtscore machinery; the orphan is the settings-source half plus the REST management surface.

  *Direction:* Ask the owner whether settings-source sync is a kept capability. If yes: delete the `.syncmark` layer and keep the lazy pull. If no: delete `routes/settings/sources.py`, the settings-sources plugin family registration, and roughly half of `settings_store.py`.

  *Size:* M–L · *Model:* Opus 4.8 (the sync state machine interlocks with file-lock ordering)

<!-- item-sep -->

- **Six CLI-override knobs, each implemented four times** — `vtsearch/settings.py:974` (medium impact)

  Six `set_cli_X`/`get_cli_X`/`get_effective_X` triads (~390 lines of identical two-step precedence), re-plumbed in `vtsearch/cli_main.py:504-625` (`_apply_X` helpers), `app.py:280-350` (env-var variants — for only three of the six knobs), and `vtsearch/routes/settings/api.py:187-222` (`_with_effective` overlay for four of six). Which knobs work under Docker is arbitrary.

  *Direction:* One declarative `AdminOverride` descriptor next to the pydantic field carrying flag name + env name; argparse args, env fallbacks, effective-getters, and the response overlay all derive from it.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`inclusion` has four owners and three copies of its clamp** — `vtsearch/routes/sorting.py:661` (medium impact)

  The `[-10, 10]` clamp lives in `vtsearch/settings_models.py:338`, `vtsearch/routes/settings/api.py:85`, and `vtsearch/routes/sorting.py:671`; two live write endpoints (`PUT /api/settings` and `POST /api/inclusion`) both funnel into `vtscore.state.set_inclusion`, which persists *back* through the shim's persister hook. `calibrate_count`/`calibration_fraction` have the same triangle.

  *Direction:* Delete the bespoke inclusion endpoints and have the find slider PUT `{inclusion}` to `/api/settings` (frontend change in the same PR); one clamp via `settings.validate_setting`; make the state-tier side effect a settings observer registered in the shim.

  *Size:* S–M · *Model:* Sonnet 5

---

## App tier — routes, schemas, facades

<!-- item-sep -->

- **`routes/projection.py` is a state machine wearing a route module, mutating library privates** — `vtsearch/routes/projection.py:55` (high impact)

  The first route decorator is at line 654; lines 55-652 are 21 orchestration helpers driving UMAP fits, pyramid builds, signposts, and persistence. Route code makes ~70 private-attribute accesses on `DatasetContext` (`_projection`, `_pyramids`, `_subset_*`, `_*_job_id`, …), so neither side can be refactored independently, and the lifecycle is untestable without a Flask client.

  *Direction:* Move the orchestration into a `vtscore/projection/` service operating on an explicit context; promote the projection slots into a public state object with real methods (pairs with the `BrowseState` grouping in the `state/core.py` item). The route module drops to ~250 lines of HTTP.

  *Size:* L · *Model:* Opus 4.8

<!-- item-sep -->

- **`routes/sorting.py` holds ML pipeline logic that belongs in vtscore** — `vtsearch/routes/sorting.py:690` (medium impact)

  `_example_sort_from_paths` (690-746), `_parse_label_file` (842-853), `_embed_external_labels` (855-904), `_train_and_score_dataset` (906-923), `_cosine_sort` (104-140), `_apply_crop_or_keep` (766-787) — no `request`, no HTTP, unreachable from the CLI. The precedent exists: `get_embedder_for_medias` was already moved to vtscore (issue #2931) leaving a compat alias at `vtsearch/routes/_shared.py:759`.

  *Direction:* Move the six functions into `vtscore/training/` as named library entry points; delete the `get_embedder_for_medias` alias (2 call sites) — it is exactly the legacy re-export the repo policy forbids on internal surfaces.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`routes/_shared.py` is nine unrelated modules in one file** — `vtsearch/routes/_shared.py:28` (medium impact)

  866 lines mixing mimetype sniffing, context-header decorators, find-progress control, the error envelope, body parsing, a route-registration factory, a ~400-line plugin block, media response shaping, sort-result windowing, embedder resolution, policy guards, path scrubbing, and mtime formatting. `find_idle` (178-217) is the public twin of `sorting.py`'s private `_sort_idle:99` — the same concern implemented twice.

  *Direction:* Split into http/context/plugins/policy modules under `vtsearch/routes/`; push `windowed_sort_*` toward the sort-results cache and `find_idle*` next to `_sort_idle`. Mechanical move + import fixes.

  *Size:* M · *Model:* Haiku 4.5

<!-- item-sep -->

- **Two incompatible error envelopes; the frontend pays for both** — `vtsearch/routes/_shared.py:219` (medium impact)

  The hand-rolled `{"error", "detail", "request_id"}` envelope (used by every global handler + 13 route sites) coexists with flask-smorest's `{"message", "errors", "code"}` (**349** `abort()` calls). smorest's carries no `request_id`; the client interceptor maintains a `known` set of both spellings (`frontend/src/app/interceptors/error.interceptor.ts:146-153`).

  *Direction:* Standardize on the smorest envelope (27× more common, documented in the OpenAPI spec); inject `request_id` via an error-handler override or `after_request`; delete `error_response` and simplify the interceptor in the same PR. Also unblocks the August audit's 404-message item.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`schemas/`: copy-pasted validators, passthrough hooks, and a hand-mirrored plugin descriptor** — `vtsearch/schemas/datasets.py:43` (medium impact)

  `_list_of_strings` is byte-identical in `datasets.py:43-51` and `detectors.py:86-93`; the `@post_dump` passthrough hook is identical in three files; 20 `unknown = "include"` declarations with no shared base; `ImporterFieldSchema` (`datasets.py:223-314`) hand-transcribes `PluginField.to_dict()` (`vtscore/plugins/__init__.py:213-232`) with no drift-guard test; 18 near-identical one-field list-response wrappers; 35 `fields.Raw` / 41 bare `fields.Dict()` degrade the generated TS client to `any`.

  *Direction:* Add a shared base schema + validators module under `vtsearch/schemas/`; generate `ImporterFieldSchema` from `PluginField`'s dataclass fields or add a key-set drift test; replace the wrappers with a `list_response(...)` factory. (`PluginField` itself is external API — its *serialization* transcript is not.)

  *Size:* M · *Model:* Haiku 4.5

<!-- item-sep -->

- **Plugin routes are registered twice per plugin and validate every body twice** — `vtsearch/routes/_shared.py:251` (medium impact)

  `register_plugin_typed_routes` mints a static rule per plugin whose typed view exists only to attach `@arguments` for spec generation, then calls the dynamic fallback which re-validates via `validate_plugin_args`. Entry-point plugins discovered after import never get a typed route, so OpenAPI coverage depends on import timing; file-upload plugins are excluded anyway.

  *Direction:* Keep one dynamic route and generate the per-plugin request bodies at spec-build time (a custom apispec walk over `list_plugins()`; `vtsearch/openapi_postprocess.py` establishes the post-processing seam). Validate once.

  *Size:* M · *Model:* Opus 4.8 (apispec internals; generated TS client shape changes)

<!-- item-sep -->

- **`state_proxies.py` + `vtsearch/state/__init__.py`: 375 lines of facade for one production call site** — `vtsearch/state_proxies.py:43` (medium impact)

  The eight proxy globals have exactly one production consumer (`vtsearch/routes/detectors/registry.py:301`); everything else is tests (49 files) plus a `TYPE_CHECKING` block in `app.py`. The proxies keep their own storage permanently empty so `isinstance` passes — the August audit already logged the silent-wrong-answer failure mode. The 78-name re-export list silently drops new `vtscore.state` exports.

  *Direction:* Rewrite the one production site to explicit context access; migrate tests to `thread_dataset_context(ctx)` + explicit access (mechanical sweep); delete `state_proxies.py` and collapse the re-export list. Supersedes the August audit's "forward the remaining dunders" item — delete that item there when this ships.

  *Size:* L · *Model:* Sonnet 5 for the app change, Haiku 4.5 for the test sweep

<!-- item-sep -->

- **`achievements.py`: response shape written twice; reaches into three settings privates** — `vtsearch/achievements.py:527` (medium impact)

  `get_full_state` hand-builds the full payload in both its disabled (575-596) and enabled (598-658) branches, already divergent on `next_threshold`. The module imports `_ensure_user_loaded`, `_settings_lock`, `_user_caches` from `vtsearch/settings.py` — a coupling that `settings.py:305-316` documents as the reason its state must stay module-global.

  *Direction:* Build the payload once from a state dict; give settings a public `snapshot_user(username)` read API and drop the private imports (which then frees `UserSettingsStore` to own its containers); split the static achievement catalog into its own module.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **CLI autodetect: a 2×2 entry-point matrix with hand-copied bodies that have already drifted** — `vtscore/cli.py:1130` (medium impact)

  `autodetect_main` / `autodetect_importer_main` / `autodetect_main_chunked` / `autodetect_importer_main_chunked` share one body modulo the loader; the chunked pair accepts `stream_results`/`keep_negatives`, the whole pair doesn't; `_load_importer_whole:764` double-applies `apply_custom_metadata_md5`. On the app side `_dispatch_autodetect` (`vtsearch/cli_main.py:723-790`) repeats the same 8-argument call four times.

  *Direction:* One private `_autodetect(source_spec, **opts)` with a `SourceSpec` (pickle/importer × whole/chunked) carrying its own description payload; **keep all four public names as thin shims** — they are documented library API (`vtscore/docs/packages/cli.md:51-92`) and called from `cli_main.py`. Add the missing flags to the whole variants; drop the duplicate md5 call.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Small app-tier batch (one PR)** — `vtsearch/hooks.py:37` (low impact)

  (a) `_STATE_SYNC_EXEMPT_PREFIXES` hardcodes URL prefixes that silently stop matching on a rename — make it a route attribute. (b) Six endpoints have full test suites and zero frontend references (`/api/dashboard/dataset-info`, `/api/dashboard/dataset-rename`, `/api/dataset/load-folder`, `/api/votes/seed-from-examples`, `/api/find/queue-ids`, `/api/find/boundary-next`) — get an explicit keep-as-API/delete decision from the owner for each rather than drift.

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **The autorun extractor/localizer surface is fully orphaned: ~1000 lines, 12 endpoints, no UI** — `vtsearch/autorun_processors.py:38` (high impact) **[owner decision]**

  `vtsearch/autorun_processors.py` (whose extractor and localizer halves are verbatim copy-paste twins), `vtsearch/routes/processors/crud.py` (10 routes), `vtsearch/routes/processors/scoring.py` (`/api/extract`, `/api/auto-extract`, `/api/localize`, `/api/auto-localize`), `vtsearch/schemas/processors.py`, and `frontend/src/app/services/processors-api.service.ts` — the Angular service is referenced by nothing (verified), and no template calls any of the endpoints. (The "autorun" strings in `dashboard.component.ts` are the unrelated Auto-Find tab.)

  *Direction:* Confirm with the owner, then delete the app-tier registry, routes, schemas, and the dead Angular service (+ OpenAPI snapshot regen). **Keep the `Extractor`/`Localizer` ABCs in `vtscore/media/processors.py`** — they are documented external extension API (`docs/EXTENDING-processors.md`) and the converter/processor application path stays.

  *Size:* M · *Model:* Haiku 4.5 once decided

---

## Eval harness & experiments

<!-- item-sep -->

- **`CALIB_SAFE_THRESHOLDS` defaults to the retired #2781 control and preflight can't see it** — `scripts/experiments/calibration/experiment_config.py:459` (high impact)

  The config defaults safe-thresholds **off** while the app ships them on (`vtscore/eval/voting_iterations.py:3722` defaults `True`, "matching the app"); `scripts/experiments/preflight.sh:612-615` emits no DIVERGES row when the var is unset. 21 of 32 launchers carry a manual `export CALIB_SAFE_THRESHOLDS=1`; 8 set nothing and silently measure the unfused control — the exact failure family in `scripts/experiments/lessons/2026-08-12-a-study-default-is-not-a-shipped-default.md`.

  *Direction:* Default to `"1"`; delete the now-redundant exports from the 21 launchers; a study wanting the control sets `=0` and declares the divergence.

  *Size:* S · *Model:* Sonnet 5 (must audit all 32 launchers)

<!-- item-sep -->

- **Stale study defaults: anchored grid, cut rules, and patch styles no longer contain the shipped values** — `scripts/experiments/calibration/experiment_config.py:471` (medium impact)

  `ANCHORED_WEIGHTS` defaults `"1,3,10,30,100"` (shipped `FOLD_ANCHOR_WEIGHT = 0.3` absent); `ANCHORED_RULES` defaults `"mid,rate"` (shipped `mid_tilt` absent); `PATCH_STYLES` defaults include `max_patch_pca_hac`, a geometry #2886 removed from ingest — doubling the GPU cost of every default calibration cell.

  *Direction:* Make each default contain the shipped value first (`"0.3,1,3,10,30"`, `"mid_tilt,mid,rate"`, `"max_patch"`); update `preflight.sh:658`'s default argument to match.

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **Concluded re-pool arms (`topk`/`pnorm`) run by default; every analyzer discards their rows** — `scripts/experiments/calibration/experiment_config.py:344` (medium impact)

  `REPOOL_VARIANTS` defaults `"topk,pnorm"`; the verdict is recorded in [`set-scorer-experiment.md`](set-scorer-experiment.md) (both failed). Every analyzer filters `pool_variant == "max"` back out, so the arms cost a re-calibration + re-pool per step per cell producing rows nothing reads.

  *Direction:* Flip the default to `""` now (one line). If nothing re-runs it within a release, delete the emitter block (`vtscore/eval/voting_iterations.py:2484-2528`), the `repool_*` parameters, and the four `calibration_metrics` helpers (only `voting_iterations` and one test reference them).

  *Size:* S · *Model:* Haiku 4.5 (flip) / Sonnet 5 (deletion)

<!-- item-sep -->

- **`simulate_voting_iterations`: 45 positional parameters and a 786-line body that is the pin target for two mirrors** — `vtscore/eval/voting_iterations.py:3716` (high impact) **[re-pin]**

  None of the 45 parameters is keyword-only, so inserting one mid-list silently rebinds positional callers; the two `training.*_default` mirrors name this whole function as their harness side, so a trip points the reconciler at 1058 lines.

  *Direction:* Insert `*` after `seed` (all real callers already pass by keyword); extract the production-default resolution block (4141-4160) into a small named helper and re-point the two mirrors at it; extract the validation block similarly. Run `--update` after re-pointing.

  *Size:* M · *Model:* Opus 4.8 (default-arm semantics)

<!-- item-sep -->

- **Split `voting_iterations.py` (4999 lines): schema tuples, trainers, and per-study arm emitters out** — `vtscore/eval/voting_iterations.py:232` (high impact) **[re-pin]**

  Measured seams: seven output-schema tuples (232-673), production-faithful core (674-1455), ~1270 lines of concluded-study arm emitters (1456-2724), trainers (2725-3401), the #3322 skyline family (3402-3710), and the loop (3716-4773). The package already extracted five study families (`vtscore/eval/cut_rules.py`, `transfer_rules.py`, `fit_quality.py`, `patch_styles.py`, `calibration_metrics.py`) — the pattern is established and unfinished. The "private" column tuples are imported from four external files, so drop the leading underscores when moving.

  *Direction:* Staged: (1) columns module — pure move, Haiku-safe; (2) step-trainers module; (3) one arm-rows module per study family with a small `(enabled, emitter)` dispatch replacing the seven inline blocks. Steps 2-3 move five pinned symbols (`_style_train_and_calibrate`, `_safe_threshold_for_step`, `_cut_inclusion_arms`) — update the `Mirror.harness` paths and `--update`.

  *Size:* L · *Model:* Haiku 4.5 (step 1), Opus 4.8 (steps 2-3)

<!-- item-sep -->

- **`check-eval-app-sync` is one-directional: harness-side edits never trip the gate** — `scripts/check-eval-app-sync.py:527` (medium impact)

  Only the app side is digested; the harness side is a substring existence check. If the harness's default resolution moves, the gate is silent — the direction drift actually happened in (#2923).

  *Direction:* Add an optional harness-side digest pin (same `_normalize_python`), tripping with a distinct "harness-changed" reason; gate it on the `ported` mirrors first, where the harness side is a genuine hand copy. Combines well with the August audit's substring→AST item.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Eight hand-rolled `load_cells` copies with eight robustness levels — and one live regression** — `scripts/experiments/calibration/bench_cells.py:28` (medium impact)

  Independent cell loaders in eight analyzers diverge on zero-byte skip, unreadable-file catch, and header-only counting — failures documented by three lessons files. Live bug: `bench_cells.py:28` declares `_SIDECARS` missing `picks` and `fitq` while `EMIT_PICKS` defaults on, so four bench analyzers silently concatenate the per-click pick log into the metric frame — exactly what `_cells_io.py:32-38` documents `SIDE_FRAME_SUFFIXES` exists to prevent. The README claim that every analyzer discovers input through `_cells_io.main_frame_files` is false for six files.

  *Direction:* One `load_cells(cells_dir)` in `scripts/experiments/calibration/_cells_io.py` implementing the union of the guards + side-frame exclusion; convert all nine call sites; delete `_SIDECARS`; fix the README claim.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`run_autopilot_sweep.py` re-implements the harness vote loop against a retired configuration** — `scripts/experiments/inclusion_knob/run_autopilot_sweep.py:185` (medium impact)

  Hand-rebuilds the loop with the pre-#2877 interleave, the retired `"mlp"` head, no acquisition offset, and a hardcoded calibration fraction; imports the harness-private `_build_eval_atlas`; nothing under `scripts/` is covered by the sync gate. The study concluded and has a successor run through `scripts/experiments/calibration/`.

  *Direction:* Delete it plus `summarize_autopilot.py` (and the self-declared-superseded `run_selection_sweep.py` + `summarize_selection.py`), keeping the reports in `docs/experiments/`; or rewrite as a thin `simulate_voting_iterations(...)` call. Never leave a second uncovered copy.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`scripts/experiments/calibration/` is 124 files in one flat directory with cross-study import chains** — `scripts/experiments/calibration/README.md` (medium impact)

  81 Python files (30,886 lines), 32 launchers (6,406 lines of shell), ≥18 concluded studies with reports under `docs/experiments/`. Analyzers import each other across studies (`analyze_transfer.py` ← `analyze_cut.py`; `analyze_folds_3314.py` ← `analyze_folds_2897.py`; `analyze_acq.py`/`analyze_startup.py` ← `analyze_spikes`; …), so archival is non-local. Two files are referenced by nothing: `figures_tail_3281.py`, `text_vs_detector.py`.

  *Direction:* (1) delete the two zero-reference files; (2) extract the genuinely shared pieces the cross-imports reach for (the unified `load_cells`, a fold-frames module, the spikes arm loader), then `git mv` each concluded study's analyze/selftest/figures/launch files into a per-study subdirectory, leaving a pointer in its report. Per-study duplication of planted-answer selftests is fine — the flat namespace and accidental coupling are the debt.

  *Size:* L · *Model:* Sonnet 5

<!-- item-sep -->

- **`build_pile.py`: a seven-subcommand multi-tool with a 303-line dataset loader** — `scripts/experiments/pile/build_pile.py:579` (medium impact)

  1885 lines interleaving dataset loaders (including `_load_vg_scale`, 303 lines), provenance/fingerprinting, and five audit modes; its own docstring records that `--rebuildable` "sat broken for eleven days behind a pile that verified clean (#3297)".

  *Direction:* Split into loaders (one module per source kind, registered by `spec["kind"]`), a provenance module, and an audit module; inside `_load_vg_scale`, name the correction-application and banding passes first (where the #3281 double-normalization lesson lives).

  *Size:* L · *Model:* Opus 4.8 (the pile feeds every study; loader regressions are silent and expensive)

<!-- item-sep -->

- **`common.py` forked seven ways; `_cells_io.py` forked twice** — `scripts/experiments/calibration/common.py:12` (medium impact)

  Seven per-experiment `common.py` copies differ by 3 lines (env-var names); the load-bearing `_neutralise_editable_finder` (the #2846 wrong-worktree fix) now lives in seven files. The two `_cells_io.py` copies differ only in docstrings.

  *Direction:* A shared experiments-lib module with `setup_env(exp_env_var, results_env_var, ...)` and one `cells_io`; each study's `common.py` shrinks to its constants plus a call. (`scripts/experiments/docmarks/sources/_common.py` is unrelated — leave it.)

  *Size:* M · *Model:* Haiku 4.5

<!-- item-sep -->

- **The Smart-indicator FP/FN cost loop is a mirror that doesn't need to be one** — `vtscore/eval/voting_iterations.py:2826` (medium impact) **[re-pin]**

  `_labelset_error_costs` (2826-2839) and `labeling_progress._score_step` (`vtscore/detectors/labeling_progress.py:691-699`) copy the same weighted-error arithmetic — pure math with none of the can't-delegate reasons the sync-gate docstring lists. Also: `voting_iterations._inclusion_weights` (685) and `calibration_metrics.inclusion_weights` are two wrappers in one package over the same production function.

  *Direction:* Add `weighted_error_cost(scores, labels, threshold, fpr_weight, fnr_weight)` to `vtscore/training/thresholds.py`; both callers delegate (each keeps its own plumbing). Delete `_inclusion_weights` in favour of the `calibration_metrics` import it already has. Coordinates with the August audit's `_eval_cached_models` delegation item — ship together.

  *Size:* S · *Model:* Opus 4.8 (touches the shipped stopping-advice and every study's vote order simultaneously)

---

## Frontend — god components & extraction seams

<!-- item-sep -->

- **`browse-canvas`: extract the thumbnail store and the animation controller** — `frontend/src/app/components/browse-canvas/browse-canvas.component.ts:216` (high impact)

  The 3062-line component already has five extracted pure modules with specs (`view-transform.ts`, `sign-layout.ts`, `bin-geometry.ts`, `hex-render.util.ts`, `render-perf.ts`). Two blocks were left behind: the thumbnail cache/LRU/preload scheduler (~400 lines: fields at 216-256, methods 691, 1929-2265) and the zoom/pan/settle animation controller (~500 lines: 1022-1360, 2659-2807) — both nearly Angular-free.

  *Direction:* A thumb-store class over `(mediaUrl, onLoaded)` owning both caches, the tier flag, and the preload scheduler; an animation class over `{getCanvas, draw, setTransform}` owning snapshot buffers and both eased transitions. Mirror the `render-perf.ts` shape so the spec pattern transfers. Fold in the August audit's two thumbnail items (full-res preload bytes; permanent `thumbFailed`) while the code is open.

  *Size:* L · *Model:* Opus 4.8 (the RAF/redraw notification path is the zoneless-staleness surface)

<!-- item-sep -->

- **`browse-bin-popup`: 1726 lines, zero signals, 18 dual-presentation branches, SCSS ~800 bytes from the budget** — `frontend/src/app/components/browse-bin-popup/browse-bin-popup.component.ts:222` (high impact)

  The only large DOM-rendering component with no signals (0 vs 32 in browse-view), compensating with 14 manual CD calls; `docked()` forks 18 branches in TS, 18 in the template, plus a 100-line SCSS override block; the stripped stylesheet is 7,159 bytes against the 8 kB `anyComponentStyle` warning ceiling that `run-tests.sh` treats as a hard failure.

  *Direction:* (a) Split out a presentational bin-member-grid component (virtual-scrolled rows, thumbnails, selection, keyboard nav — shared by both presentations), leaving the floating shell (drag/place/nudge/clamp) and a thin docked host; this halves the SCSS per component. (b) Signalize `ids`/`rows`/`previewId` and the three dicts; delete the manual CD calls.

  *Size:* L · *Model:* Sonnet 5 for (a), Opus 4.8 for (b)

<!-- item-sep -->

- **`SortStateService` is an anemic store; ~530 lines of sort/autopilot orchestration live in `label-view`** — `frontend/src/app/components/label-view/label-view.component.ts:642` (high impact)

  The service is 20 getters + 12 setters with no transitions; the component owns the sort block (642-930) and autopilot block (1193-1432), and find-view/dashboard re-derive slices of it. `autoSelectNext` (1456) is mirrored by the Python eval harness, and the August audit flags it as an *unpinnable* mirror precisely because it lives inside a god component.

  *Direction:* Extract `autoSelectNext` to a pure `frontend/src/app/utils/` function first (independently valuable — makes the eval-sync mirror pinnable; update the mirror path and `--update`). Then move the learned-sort/text-sort/window transitions onto `SortStateService` and the autopilot transitions onto `AutopilotStateService`, leaving the component as event forwarding.

  *Size:* L · *Model:* Opus 4.8

<!-- item-sep -->

- **Seven hand-rolled divider drags against one shared directive used by exactly one view** — `frontend/src/app/components/label-view/panel-resize.directive.ts:1` (medium impact)

  `PanelResizeDirective` (spec'd, `runOutsideAngular`, clamping, destroy cleanup) has two consumers, both in label-view. Seven re-implementations: three in `browse-view.component.ts` (786-940), one in `browse-bin-popup.component.ts:992-1037`, two in `find-view.component.ts` (497-573), one in `browse-minimap.component.ts:509-545` — the last omits `runOutsideAngular`, and browse-view's three only detach listeners on mouseup.

  *Direction:* Promote the directive to `frontend/src/app/directives/`, generalize side/clamp as inputs, adopt at all seven sites (`snapPanelWidthToGridColumns` already handles release-snap for three).

  *Size:* L · *Model:* Opus 4.8 (drag handlers deliberately run outside Angular; the template-bound output is the CD notification)

---

## Frontend — duplication & dead code

<!-- item-sep -->

- **The audio-audition state machine is triplicated near-verbatim across three Browse components** — `frontend/src/app/components/browse-hover-preview/browse-hover-preview.component.ts:160` (medium impact)

  `browse-hover-preview` (160-249) and `browse-selection-panel.component.ts:299-394` differ only in comments; `browse-bin-popup.component.ts:1359-1459` is the same machine already diverged (pooled element). The dwell debounce, stale-event guard, rAF sweep, and buffering tri-state were fixed once and copied twice; all three emit the `NowPlaying` interface that *is* shared.

  *Direction:* Extract a browse-audio-audition class next to `frontend/src/app/utils/clip-window.ts` exposing `hover(id)` / `leave()` / `stop()`; components keep only their `nowPlaying` output wiring.

  *Size:* M · *Model:* Opus 4.8 (the output is the zoneless notification path)

<!-- item-sep -->

- **Verified dead frontend code: two whole services and five orphan inputs/outputs** — `frontend/src/app/services/detectors-scoring-api.service.ts:1` (medium impact)

  `DetectorsScoringApiService` (53 lines) and `ProcessorsApiService` (97 lines) are referenced only by their own construct-and-assert-truthy specs (verified). Orphans: `FolderBrowserComponent.initialPath` / `.autoFocus` / `.loadError` (the last means load failures are silently dropped — decide whether consumers should bind it before deleting), `SkeletonComponent.borderRadius`, `BrowseMinimapComponent.resized`.

  *Direction:* Delete the service+spec pairs (`ProcessorsApiService` deletion pairs with the autorun-surface item above) and the five declarations with their now-constant call sites.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **Utilities exist but are bypassed; helpers reimplemented per component** — `frontend/src/app/utils/clip-window.ts:9` (medium impact)

  `clip-window.ts` is used by the three Browse previews while `audio-player.component.ts:211-262` and `video-player.component.ts:150-195` hand-roll the same clip snap-and-loop (~60 lines each, differing only in the element ref); `formatSize` is byte-identical in four components; `formatMetadataValue` byte-identical in two; `sortRowsByColumn` is bypassed by three bespoke table sorters; a seven-case `sortEntries` ladder is triplicated across `label-list`, `labelset-list`, and `browse-selection-panel`.

  *Direction:* Widen `clip-window.ts` to `HTMLMediaElement` and adopt in both players; add `formatBytes`/`formatMetadataValue` to `frontend/src/app/utils/`; give `sortRowsByColumn` an optional key-extractor and adopt; extract one `sortListEntries(entries, mode)`.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`.sr-only` is a byte-for-byte duplicate of the global rule in three component stylesheets** — `frontend/src/scss/_components.scss:949` (low impact)

  Identical 10-declaration blocks in `browse-selection-panel.component.scss:134-144`, `label-sort.component.scss:15-25`, `center-panel.component.scss:147-157`; other components use the global class with no local copy, proving it applies through encapsulation.

  *Direction:* Delete the three local blocks. (The repeated `.loading-text`/`.error-msg`/`.stat-*` selectors are third-copy promotion candidates under style-guide §6 — separate, smaller pass.)

  *Size:* S · *Model:* Haiku 4.5

---

## Frontend — state & idiom consistency

<!-- item-sep -->

- **Dashboard selection: mirrored dataset/detector logic, and state duplicated between component and service** — `frontend/src/app/components/dashboard/dashboard.component.ts:525` (medium impact)

  The dataset (525-604) and detector (709-762) selection ladders are structural twins (~90 mirrored lines); the source of truth is two plain `Set`s on the component that must be manually pushed into `DashboardSelectionService` via `pushTopBarLabels()` after every mutation (7 call sites, no guard), while the pulldown pushes back through a `selectRequest$` Subject — so the top bar depends on the Dashboard being mounted.

  *Direction:* Move the two Sets into the service as signals with one parameterized `toggle(kind, id, additive)`; `pushTopBarLabels` collapses into a `computed`; `selectRequest$` disappears.

  *Size:* M · *Model:* Opus 4.8 (top-bar reactivity across a route boundary)

<!-- item-sep -->

- **Two projection pollers in two idioms (already drifting) and three verbatim find-progress blocks** — `frontend/src/app/services/browse-prep.service.ts:239` (medium impact)

  `BrowsePrepService` (BehaviorSubject) and `BrowseSubsetPrepService` (signals) duplicate the same 22-line `schedulePoll` with independent `MAX_POLL_ERRORS`; the subset copy carries `overall_step_end`, the full copy doesn't. A verbatim 25-line find-progress subscription appears in `find-view.component.ts:419-442`, `label-view.component.ts:858-881`, and a variant in `dashboard.component.ts:1468-1483`. The mandated `adaptivePoll()` helper has 3 adopters against 7 hand-rolled poll loops.

  *Direction:* (a) Fold the find-progress subscription into `SortStateService` (it owns every field written) — small and safe. (b) Extract a `pollUntil(fetch, isDone, opts)` helper returning a signal; rebuild both prep services on it, converting `BrowsePrepService` to signals.

  *Size:* S for (a), M for (b) · *Model:* Sonnet 5 (a), Opus 4.8 (b)

<!-- item-sep -->

- **Per-media-type settings preferences hand-rolled in 14 components** — `frontend/src/app/components/browse-view/browse-view.component.ts:642` (medium impact)

  The `{media_type: value}` read/clamp/merge-write dance is written longhand at each site (browse-view inlines it five times over five value types; `grid_icon_size_right` is mirrored in three components), and `label-view-panel-state.service.ts:43-58` holds five such dicts behind **plain getters with no signal underneath** — not the sanctioned getter-over-signal shape.

  *Direction:* Add a `perMediaType<T>(key, {default, clamp})` helper on `SettingsStateService` returning a value-signal + merge-preserving setter; convert the panel service first (also fixes the getter shape), then browse-view, then the three `grid_icon_size_right` copies.

  *Size:* L · *Model:* Opus 4.8

<!-- item-sep -->

- **`find-view` and `label-view` duplicate the pair-change reset and inclusion seeding verbatim** — `frontend/src/app/components/find-view/find-view.component.ts:337` (medium impact)

  `reloadForNewPair` opens with an identical supersede/clear/reload sequence in both views, and `seedInclusion()` is byte-identical (`label-view.component.ts:969-974` / `find-view.component.ts:359-364`). This is the correctness-critical teardown for the intent/active split — the ordering rule is documented twice and enforced nowhere.

  *Direction:* A component-provided pair-scope service owning `scope$`, `supersede()`, and the shared reset; both views call it then run their view-specific tail. `seedInclusion` moves into `SortStateService`.

  *Size:* M · *Model:* Opus 4.8

<!-- item-sep -->

- **`ActiveDetectorService` abandoned at 15 call sites; no dataset counterpart; no by-id index** — `frontend/src/app/services/active-detector.service.ts:11` (medium impact)

  The service exists to retire the imperative `detectors.find(...)` lookup (#2819) but has 4 consumers against 15 remaining imperative sites; `DatasetStateService` exposes arrays only, so all 15 are O(n) scans; `app.component.ts:202` wants an `ActiveDatasetService` that doesn't exist.

  *Direction:* Add `datasetById`/`detectorById` computed Maps to `DatasetStateService`; add an `ActiveDatasetService` mirroring the detector one; migrate the component call sites (guards and pre-reactive services legitimately keep the imperative read).

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Four coexisting subscription-teardown idioms (20 / 7 / 3 / 3 files)** — `frontend/src/app/components/label-view/label-view.component.ts:1 ` (medium impact)

  `destroy$`+`takeUntil` (20 files), manual `Subscription` field (7), `subs[]` array (3, the leakiest), and the modern `takeUntilDestroyed()` (3). With 3/33 adoption of the current idiom, a new component picks by coin flip. (For calibration: the `input()`/`output()`, `inject()`, and `@if` migrations are 100% complete — don't re-open those.)

  *Direction:* Sweep the 10 single-subscription/array cases to `takeUntilDestroyed()` (mechanical); for the 20 `destroy$` files, convert those whose subject is teardown-only, and rename the ~5 that double as scope resets (`pairScope$`) to say so.

  *Size:* M · *Model:* Sonnet 5 for the mechanical 10; Opus 4.8 for the `destroy$` files

---

## Tests & tooling

<!-- item-sep -->

- **`tests_lib/` is not the tier it claims: repo-meta tests silted in, and its conftest imports the app tier** — `tests_lib/conftest.py:157` (high impact)

  `tests_lib/__init__.py` promises no `vtsearch` imports, but `tests_lib/conftest.py:157` and `tests_lib/fixtures/medias.py:12` import `vtsearch.state` (binding the app-side proxy, not the library object). ~10 tests under `tests_lib/core/` exercise no library code (Dockerfile syntax, user-guide anchors, SCSS parsing, gate self-tests, `pyproject.toml` checks), so `./run-tests.sh core` runs a Dockerfile text parser; symmetrically 8 files in `tests/` touch no app-tier module.

  *Direction:* Add a third tree for repo/tooling meta-tests with its own marker; move the ~10 over; move the 8 app-tree files to `tests_lib/`; fix the two imports to `vtscore.state` and add a layering test asserting nothing under `tests_lib/` imports `vtsearch.*`.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **The two conftests are a 90% copy that has already drifted in the embedding stub** — `tests/conftest.py:70` (medium impact)

  Near-verbatim duplication of the group-marker hook, fake embedders, `_allow_test_tmp_paths`, `_stub_embedding_models`, most of the reset fixture, and the entire 54-line summary printer — with real drift in the load-bearing stub: `tests/conftest.py:70-93` seeds in-memory medias off `media_bytes` while `tests_lib/conftest.py:118-124` falls back to a process-salted `hash(str(path))`, so the two suites assert against different fake-embedder semantics.

  *Direction:* Extract the genuinely shared pieces into a shared module both conftests import (unlike `helpers.py`, these are not policy-duplicated); unify the fake-embedder semantics on the `tests/` version.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **~300 test assertions read global state through a conftest-installed alias layer on `app`** — `tests/conftest.py:135` (medium impact)

  The conftest monkeypatches six names onto the app module "for backward compatibility"; 161 `app_module.good_votes` + 137 `app_module.bad_votes` references depend on test infrastructure rather than the real import surface, and four files carry `import app as app_module  # noqa: F401` comments claiming a side effect that doesn't exist.

  *Direction:* Codemod to `from vtsearch.state import ...`, delete the shim block and the side-effect-only imports.

  *Size:* M · *Model:* Haiku 4.5 (codemod) + Sonnet 5 (verify proxy identity semantics per site)

<!-- item-sep -->

- **The vulture audit scans 23% of the Python and 13% of its whitelist is unfalsifiable** — `.vulture-whitelist.py:7` (medium impact)

  The documented command (written down in both `.vulture-whitelist.py:7` and `docs/RELEASE.md`) scans `vtsearch/ app.py tests/` — never `vtscore/` (100k lines, the externally-consumed tier), `tests_lib/`, or `scripts/`. 13 of 102 whitelist entries name symbols absent from the scanned scope, so they can never fire. Nothing runs vulture as a gate.

  *Direction:* Extend the command to cover `vtscore/` + `tests_lib/`, move it into a script so it's written down once, triage the first pass, drop the 13 dead entries (keep `with_dataset_context`/`with_detector_context` — documented public API), and consider wiring it into `run-tests.sh` so the whitelist is load-bearing.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **`Dockerfile.image-embedders` and its GPU twin are a 90% copy; their requirements differ by one line** — `docker/Dockerfile.image-embedders:1` (medium impact)

  The Dockerfile pair has already diverged (the CPU file's import canary is absent from the GPU file); `requirements/image-embedders.txt` vs `requirements/image-embedders-gpu.txt` differ by exactly the torch index URL, both carrying the same 20-line rationale comment verbatim; no test compares either pair.

  *Direction:* Collapse the requirements pair (a `-r` wrapper or a build-arg'd index URL); extract the shared Dockerfile body behind a `BASE_IMAGE` build arg, or at minimum add a parity test beside the existing no-agpl mirror check.

  *Size:* M · *Model:* Sonnet 5

<!-- item-sep -->

- **Repo-hygiene batch (one PR)** — `pyproject.toml:185` (low impact)

  (a) The `S104` ignore cites `tests/api/test_ssrf_validation.py`, which moved to `tests_lib/core/test_ssrf_validation.py`. (b) codespell's `ignore-words-list` carries `ans` and `ser`, neither occurring as a whole word anywhere (the comment's justification is stale). (c) `tests/downloads/` and `tests/gpu/` contain only empty `__init__.py` files — delete; `tests_lib/projection/` is the only group dir *missing* an `__init__.py` — add it. (d) Four `requirements/*.txt` headers claim `tests/core/test_requirements.py` enforces mirror sync; the real check is `tests_lib/core/test_agpl_optional_extra.py:85` — fix the pointers.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **Delete three unreferenced one-off scripts (~950 lines)** — `scripts/overfitting_probe.py:1` (low impact)

  `scripts/run_mlp_ensemble_sweep.py` (self-described "throwaway experiment script for issue #2497"; its output dir was never committed), `scripts/overfitting_probe.py`, and `scripts/scan_exif_orientation.py` have zero inbound references anywhere. (`scripts/spike_structural_roxford.py` is still referenced by [`structural-embedder.md`](structural-embedder.md) — keep.)

  *Direction:* Delete all three; git history is the archive.

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **`gridenv.sh` contradicts itself: the "untracked" shim is tracked, and the "self-locating" file hardcodes one venv** — `gridenv.sh:2` (low impact)

  The comment says the `.shadow/` editable-finder shim is untracked (so fresh worktrees regenerate it); `git ls-files` shows it committed, making the defensive creation block dead. Three lines below the "never hardcode a path" warning, the file hardcodes `module load python/3.12.3` and an absolute venv path.

  *Direction:* Pick one story: either gitignore + `git rm --cached` the shim (restoring the documented design) or delete the creation block and fix the comment. Move the venv/module lines behind env vars with the current values as defaults.

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **A 206 KB generated punch-card PNG is hand-refreshed every release and nothing displays it** — `scripts/punchcard/vtsearch_pr_punchcard.png` (low impact) **[owner decision]**

  19 commits of the PNG (+ the hand-appended `pr_merges.txt`) with zero embeds anywhere; `docs/RELEASE.md` makes regenerating both a mandatory release step including manual data entry.

  *Direction:* Ask the owner: delete the PNG from the tree and drop the release step, or embed it somewhere and derive `pr_merges.txt` from `git log --merges` inside `scripts/punchcard/punchcard.py`.

  *Size:* S · *Model:* Haiku 4.5 / Sonnet 5 if automating

<!-- item-sep -->

- **`slides/Makefile` is missing both hardening fixes `render.sh` documents** — `slides/Makefile:36` (low impact)

  Two Marp wrappers exist; `slides/render.sh` carries `--no-stdin` (hang fix) and the `PIPESTATUS` check (#3301's silent-success fix), the Makefile carries neither — so `make` from a script reproduces the exact hang the shell script was fixed to prevent.

  *Direction:* Make the Makefile's pattern rule and `watch` delegate to `render.sh` (the `speaker` target already does).

  *Size:* S · *Model:* Haiku 4.5

<!-- item-sep -->

- **`@angular-devkit/build-angular` is an unused devDependency and the reason `npm audit` runs `--omit=dev`** — `frontend/package.json:33` (low impact)

  `frontend/angular.json` uses only `@angular/build:*` builders (required, per `frontend/README.md`); the devkit package drags in the webpack toolchain, forces the `webpack-dev-server` override pin, and is the stated justification for narrowing the audit gate in `run-tests.sh`.

  *Direction:* Remove it, drop the override, re-run `npm ci` + `build:prod` + `test:ci`, then try widening the audit gate.

  *Size:* S · *Model:* Sonnet 5

<!-- item-sep -->

- **The ensure-test-deps `PreToolUse` hook reads `$TOOL_INPUT` only; its sibling prefers stdin** — `.claude/settings.json:33` (low impact)

  The bash matcher greps `$TOOL_INPUT` while `.claude/hooks/require-issue-labels.py:82-88` documents the payload as stdin-first with env fallback. If the harness delivers on stdin, the dep-install hook greps an empty string and silently never fires (masked today because `run-tests.sh` invokes the installer directly).

  *Direction:* Move the matcher into a small hook script mirroring `_read_payload` (stdin first, env fallback); verify against a live session rather than guessing the contract.

  *Size:* S · *Model:* Sonnet 5

---

## Documentation

<!-- item-sep -->

- **Two independently written extension-authoring doc sets cover the same plugin families (6,600 lines)** — `docs/EXTENDING.md:9` (high impact)

  `docs/EXTENDING-plugins.md` + `docs/EXTENDING-media.md` + `docs/EXTENDING-processors.md` (3,869 lines) and `vtscore/docs/extending/` (2,762 lines, 8 files) each state the contract for the same nine extension points in their own words; the stated front door (`docs/EXTENDING.md`) links to zero of the vtscore set. Independent prose is worse for drift than a copy would be.

  *Direction:* Declare a division of labour — `vtscore/docs/extending/` owns the library contract (it ships with the semver'd package); `docs/EXTENDING-*.md` keeps only app-tier wiring and links across. At minimum add the cross-links to `docs/EXTENDING.md`'s table today.

  *Size:* L · *Model:* Opus 4.8 (an audience judgement call, not a mechanical merge)

<!-- item-sep -->

- **`vtscore/docs/packages/exporters.md` teaches the deprecated exporter contract** — `vtscore/exporters/base.py:604` (medium impact)

  The package doc mentions `LabelsetExporter` 14 times and `ResultsExporter` zero, tells authors to implement the legacy `export()` (which triggers a definition-time warning per `base.py:178-210`), and never mentions `PAYLOAD_KINDS` / `supported_payloads` / `export_find_results` / `export_labelset`; the correct contract lives only in `vtscore/docs/extending/results-exporters.md`. `vtscore/docs/extending/README.md:40` also still lists the old name.

  *Direction:* Docs-only: rewrite the package doc around `ResultsExporter`, demote `LabelsetExporter` to a one-line permanent-alias note, and have the package doc link to the extending guide rather than restate the contract (the restatement is what rotted). **Preserve the alias and legacy path in code** — external API.

  *Size:* S · *Model:* Sonnet 5

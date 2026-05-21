# Logical-Bug Audit — 2026-05

**Status:** In progress — most findings still open; resolved items
are marked inline as struck-through headings.

**Scope:** Multi-agent audit (10 per-subsystem + 5 cross-section
interaction passes) of the entire VTSearch codebase, focused on
**logical** bugs — missed expectations between modules, race conditions,
silent miscompute, data corruption, and broken invariants. Syntax,
typing, and lint issues were explicitly out of scope (those are covered
by `ruff` / `pyright` / `tsc`).

**Total distinct findings (after dedup):** ~95.

## How to read this doc

- Findings are grouped first by **severity**, then by subsystem within
  each tier.
- Each open finding has a stable ID (`C#` Critical, `H#` High,
  `M#` Medium, `L#` Low) so it can be referenced from other docs / PRs.
- Shipped findings are reduced to a struck-through heading. The full
  fix summary is the PR that landed it — git log, not this doc.
- The "Recurring patterns" section at the bottom is the actionable
  starting point — most findings collapse into a small number of root
  causes; fixing the pattern usually fixes many findings at once.
- The "Suggested fix order" section recommends which root-cause PRs to
  land first for maximum leverage.
- File:line references are approximate; line numbers may shift slightly
  as the codebase evolves.

## Audit coverage

Per-subsystem agents:

1. State management & concurrency (`vtsearch/state/`, `concurrency/`)
2. Detector training pipeline (`vtsearch/detectors/`, `training/`)
3. Datasets & importers (`vtsearch/datasets/`)
4. Routes & API contracts (`vtsearch/routes/`)
5. Settings & sync sources (`vtsearch/settings.py`, `settings_io/`,
   `sync/`)
6. Embedding & training infrastructure (`vtsearch/embedding/`,
   `training/`)
7. Security validation (`vtsearch/security/`)
8. Plugins, converters, eval, exporters
9. Auth, CLI, app entry (`vtsearch/auth/`, `app.py`, `cli.py`)
10. Frontend Angular logic (`frontend/src/`)

Cross-section interaction agents:

11. Dataset ↔ Detector ↔ Embedder triple
12. Settings ↔ everything (user vs server tier, sync sources)
13. Background-jobs / context propagation
14. Error / exception flow across layers
15. Frontend ↔ backend contract seams

---

## Critical — data corruption / loss / hangs / silent miscompute

### ~~C1. Download gate is never released if importer skips the `"embedding"` status~~

### ~~C2. JobManager never sets dataset/detector thread-local context~~

### ~~C3. Dataset background load tasks don't set thread-local dataset context~~

### ~~C4. Embedding-matrix cache is not invalidated after clip/dedup~~

### ~~C5. `find_label` allows body field to override the request's dataset context~~

### ~~C6. Zip-slip in HTTP archive importer (zip AND tar)~~

### ~~C7. NaN/Infinity threshold leaks through safe-threshold blending~~

### ~~C8. Bulk label paths skip achievement recording entirely~~

### ~~C9. Path-template substitution missing post-resolution validation~~

### ~~C10. MD5 / metadata cache collision returns wrong-dataset media on switch~~

### ~~C11. `fill_labels_from_sort` silently swallows sync failures~~

### ~~C12. Orphaned dataset registry entry on activation failure~~

---

## High — likely-encountered correctness or security bugs

### State / concurrency

- ~~**`record_vote()` called after releasing `_state_lock`**~~
- ~~**H1. Vote-progress invalidation / `record_vote` race**~~ — fixed by
  replacing the toggle contract on `POST /api/medias/<id>/vote` with an
  absolute-target contract.  Body takes `target: "good" | "bad" | "none"`
  instead of `vote: "good" | "bad"`; handler delegates to a new
  `vtscore.state.votes.set_vote(media_id, target, region_box)` that
  no-ops on idempotent re-applies (no `label_history` append, no
  achievement credit, no click-time bump, no progress-cache churn), so
  two stale-view tabs racing the same target collapse into a single
  transition on the server instead of alternating ADD/REMOVE.  Progress
  cache now invalidates on **any** training-set membership change for
  the media (was: polarity flips only — left un-vote cached models
  stale).  Response shape grew `state` + `click_time`; the Angular
  `VoteStateService` was rewritten around a single `submitToggleVote()`
  entry point that computes the target from local state and clears
  `pendingOptimistic` deterministically on every POST return, closing
  the persistent prediction-vs-server desync half of the bug.  Covered
  by new regressions across `tests/core/test_votes.py`,
  `tests/api/test_error_recovery.py`, `tests/core/test_achievements.py`,
  `tests/api/test_api_contracts.py`, `tests/detectors/test_patch_embedder.py`,
  and `frontend/src/app/services/vote-state.service.spec.ts`.  See
  Open follow-ups for the parallel labelset-element vote endpoint.

### Detector / training

- ~~**H2. Cross-dataset region-box loss**~~ — fixed in
  `vtscore/detectors/labelset_training.py`. `_embed_one` now resolves the
  origin, runs `patch_forward` on the file when the active embedder
  supports patch regions, and pools the box via `box_to_vote_vector` so a
  region vote's training intent survives a dataset switch. Logs a
  warning and falls back to a full-file embedding only when the embedder
  has no patch path (single-vector models) or the forward pass produces
  no output. Covered by `TestRegionAwareTrainingCrossDataset` in
  `tests/detectors/test_patch_embedder.py`.
- ~~**H3. Embedder drift on save → reload**~~
- ~~**H5. Detector embedder not revalidated on dataset switch**~~ —
  fixed in `vtscore/detectors/dataset_sync.py` +
  `vtsearch/routes/detectors/scoring.py` +
  `vtsearch/routes/detectors/find.py`.
  `invalidate_detector_model_on_embedder_mismatch()` drops
  `DetectorContext.model` / `threshold` / `last_learned_scores` /
  `training_medias` / `calibration_cache` when the dataset about to be
  scored uses a different embedder than the one the cached MLP was
  trained on.  `before_request` calls
  `ensure_detector_model_matches_active_embedder()` for the active
  ctx so a dataset switch invalidates immediately; the scoring fast
  paths (`_resolve_or_train_detector` for find-label / auto-detect,
  `_select_scorer` in multi-dataset Find) repeat the check
  per-detector to cover autorun loops and cross-dataset Find where the
  active ctx wrapper alone isn't enough.  The helper deliberately
  leaves `label_embeddings` and `embedder` alone so the load
  endpoint's progress-tracked `_maybe_start_label_reembed()` flow
  still detects the mismatch and schedules its visible re-embed task;
  the next training pass restamps the marker via
  `populate_label_embeddings` and `record_detector_embedder`.  Also
  fixed the secondary mixed-embedder bug in the resolver:
  `resolve_label_embeddings()` now accepts an `embedder_name` kwarg
  and the scoring / find callers pass the dataset's embedder so
  origin-resolved label vectors share one space with the snap-matched
  ones (previously the origin path defaulted to the media type's first
  registered embedder, mixing two spaces into a single MLP).  Covered
  by `TestEmbedderMismatchInvalidatesStaleModel` in
  `tests/detectors/test_detectors.py` and the new
  `test_forwards_embedder_name_to_*` cases in
  `tests/detectors/test_resolver.py`.
- ~~**H6. `train_model` produces degenerate single-class model**~~
- ~~**H7. Vote applied before retrain; retrain failure leaves vote live**~~

### Datasets

- ~~**H8. Origin dict shared by reference across medias**~~
- ~~**H9. Single-item split has 0 test samples**~~
- ~~**H10. Clipped media re-ingest reads whole file for MD5**~~ —
  fixed (2026-05-20). `vtscore/datasets/ingest.py` (formerly
  `vtsearch/datasets/ingest.py`) now routes both `_ingest_via_source`
  and `_ingest_via_resolver` through a new
  `_resolve_clip_content_and_embedding` helper that pairs the clip
  embedding with the clip's actual content bytes — so `md5`,
  `file_size`, and `media_bytes` describe the clip, not the parent.
  Video metadata-only clips (and other fall-through paths) fall back
  to the load-pipeline's `MD5(parent_bytes) + boundary_tag` scheme so
  distinct clips of the same parent still hash uniquely. The clip-bytes
  return value is plumbed through `_apply_clip_and_embed` /
  `_replay_chain` / `replay_chain_on_file`; see
  `tests/io/test_label_import_ingestion.py::TestClippedReingest`.
- ~~**H11. Multi-media import with empty form yields empty dataset**~~

### Routes / API

- ~~**H12. `add_media_to_pile` race**~~ — investigation closed
  (2026-05-20): not a real race in the current architecture. The
  audit's framing assumes a global "active dataset" pointer that
  could be switched mid-handler between `snapshot_medias()` and
  `apply_label()`, but `before_request` in `app.py` pins both
  `g._dataset_context` and `g._detector_context` for the duration
  of each request (resolver in `vtsearch/shim/__init__.py:19-39`),
  and `flask.g` is request-local — so the two calls inside one
  `add_media_to_pile` invocation always resolve to the same
  contexts. Note also that `apply_label` writes to the *detector*
  context's `good_votes` / `bad_votes`, not to a dataset, so the
  audit's "labels applied to wrong dataset" framing is a category
  error. The same line window (L600-694) does contain real bugs,
  but they are different from the one H12 names — see H32 / H33 /
  H34.
- ~~**H13. `vote_media` silent-mistarget on dropped header**~~
- ~~**H14. `export_labels` leaks votes across datasets**~~
- ~~**H15. File-browser symlink metadata leak**~~ — investigation
  revised and fixed (2026-05-20). The audit's literal claim was
  incorrect: the `target.resolve().relative_to(root)` check at
  `vtsearch/routes/file_browser.py:90` does block drill-through into
  symlinked-out directories (resolve canonicalises through the link,
  and `relative_to` then raises). The real bug was in the listing
  loop at L105–118: `entry.is_dir()` / `entry.is_file()` / `stat()`
  all follow symlinks by default, so an in-root symlink pointing
  outside the root showed up as an ordinary entry and leaked the
  external target's existence, name, `size_bytes`, and `modified_at`.
  In multi-user mode this violated the data-dir isolation contract
  asserted by `TestMultiUserBrowseIsolation`. Fix: in the listing
  loop, call `entry.resolve(strict=True).relative_to(root)` on
  symlinks and skip any that escape root or are broken — intra-root
  symlinks remain visible. Covered by `TestBrowseSymlinks` in
  `tests/api/test_file_browser.py`
  (`test_symlink_to_external_dir_hidden`,
  `test_symlink_to_external_file_hidden`,
  `test_intra_root_symlink_still_listed`,
  `test_broken_symlink_skipped`,
  `test_symlink_drill_through_blocked`). Note: a separate
  symlink-follow content escape exists in
  `vtscore/security/path_validation.py:rglob_follow_symlinks`
  (importers walk with `followlinks=True` and per-file paths
  outside the validated root are not re-checked) — that's a
  different code path, not covered by this fix.
- ~~**H16. Header refers to unloaded dataset → silent fallback**~~ —
  also closes the "header points to an unloaded id" half of H34 by the
  same mechanism. The "header absent → thread-local leak" half of H34
  remains open.
- ~~**H32. `add_media_to_pile` TOCTOU between md5 check and insertion**~~ —
  fixed in `vtsearch/routes/media/list.py`. After the initial
  outside-the-lock MD5 lookup (fast path for an existing match) and
  the unlocked embed step, the route now re-runs
  `build_media_lookup(medias)` under `_state_lock` immediately before
  assigning `new_id` (L700-718). On collision it routes into the
  existing-cid branch (`is_new=False`); otherwise it inserts. Two
  concurrent uploads of identical bytes therefore produce exactly one
  new media, with the loser voting the winner's id. Covered by
  `TestAddToPile.test_concurrent_uploads_same_md5_no_duplicate` in
  `tests/core/test_medias.py`, which uses a `threading.Barrier(2)`
  patched into the embedder to deterministically hold both requests
  inside the unlocked window.
- ~~**H33. `add_media_to_pile` label not synced to disk**~~ — fixed
  in `vtsearch/routes/media/list.py`. Both branches of
  `add_media_to_pile` (existing-MD5 match and new-media insertion)
  now call `_sync_pile_label_to_storage()` after `apply_label`,
  which mirrors `vote_media`'s tail by invoking
  `sync_labels_to_loaded_detector()` + `sync_to_labelset_source()`.
  The label reaches the detector's on-disk labelset and any
  configured `LabelsetSource`, so a subsequent
  `ensure_votes_match_active_dataset` rehydration restores it
  instead of silently dropping it. Covered by
  `TestAddToPile.test_existing_media_label_synced_to_disk`,
  `test_new_media_label_synced_to_disk`, and
  `test_label_survives_rehydration` in `tests/core/test_medias.py`.
- ~~**H34. Missing `X-Detector-Id` → silent detector fallback**~~ —
  fixed in `vtsearch/routes/_shared.py`. Added two stateless route
  decorators, `require_detector_header` and `require_dataset_header`,
  that reject 400 when the corresponding header (or its `?detector_id=`
  / `?dataset_id=` query-param fallback) is absent. Applied to every
  vote-mutating endpoint: `vote_media` and `add_media_to_pile`
  (`media/list.py`), `import_labels` and `fill_labels_from_sort`
  (`labels/vote.py`), `run_label_import` and `ingest_missing`
  (`labels/importers.py`), `clear_votes_route` (detector only) and
  `seed_votes_from_examples` (`sorting.py`), and `find_label`
  (`detectors/scoring.py`). The decorators run *before* the resolver
  chain so a thread-local leak on a Flask worker can't silently
  mistarget a vote even if a future code path were to leave a
  thread-local detector pinned across requests. Pure-read endpoints
  (registry listings, dashboard, file browser, settings GET) keep the
  fall-through behaviour so background scripts and the standalone CLI
  still work without headers. Test plumbing: `tests/conftest.py` now
  wraps `client.open()` to auto-inject `X-Dataset-Id` /
  `X-Detector-Id` from the thread-local active context (mimicking
  Angular's `activeContextInterceptor`), so the existing test corpus
  keeps working unchanged; tests that need to exercise the
  header-absent path drop the thread-local first.

### Plugins / converters / exporters

- ~~**H17. Plugin scanner silently shadows duplicate names**~~
- ~~**H18. CSV exporter doesn't escape embedded newlines**~~ —
  investigation closed (2026-05-20): not a real bug. The audit's
  path is also stale (exporters moved to `vtscore/exporters/`); the
  actual file is `vtscore/exporters/server_csv_file/__init__.py`.
  The code at L120–141 does not hand-roll CSV rows — it dispatches
  dict vs scalar cells and hands the row to `writer.writerow(row)`
  (L141), where `writer` is a stdlib `csv.writer` (L36) using the
  default `QUOTE_MINIMAL`, which always quotes fields containing
  `\n`, `\r`, or the delimiter. The file is opened with
  `newline=""` (L39), so no newline translation corrupts the
  quoted content. The matching importer
  (`vtscore/labels/importers/server_csv_file/__init__.py` L96)
  uses `csv.DictReader`, which reads multi-line quoted fields
  back correctly. Verified via round-trip with `\n` embedded in
  the `label`, `origin_name`, and `category` fields plus `"` in
  the label: the exporter wrote a valid quoted CSV and the
  importer returned the original strings byte-for-byte
  (`.strip().lower()` on the label only trims leading/trailing
  whitespace — interior newlines pass through).
- ~~**H19. Required select fields silently accept empty string**~~ —
  investigation closed (2026-05-20): not a real bug. The audit's
  framing assumes `_presence_kwargs()` in `vtscore/plugins/schema.py`
  produces `{"required": True, "load_default": ""}` for a required
  select, but the function is a mutually-exclusive 3-way branch that
  never emits both keys (required-with-no-default → `{"required":
  True}` only; default present → `{"load_default": <default>}`
  only). More importantly, `_build_select` adds
  `_non_empty_after_strip` to the validator list for every required
  select (schema.py:116-117) — empty strings and whitespace-only
  strings are rejected with `"Field may not be empty."`, and the
  OneOf at L119 excludes `""` from its allowed set for required
  fields too (double defence). Empirical check via
  `schema.load({"choice": ""})` confirms a 422 for required selects
  with static options, dynamic options, and with a non-empty default
  alike. The parallel text-field case is already covered by
  `tests_lib/core/test_plugin_schema.py::test_required_text_rejects_whitespace_only`.
- ~~**H20. `audio2image` has no upper bound on `n_mels`**~~ — fixed
  systemically in `vtscore/plugins/schema.py:_build_number()`, which
  now attaches `validate.Range` whenever a `PluginField` declares
  numeric `min` / `max` (so every plugin family that goes through
  `validate_plugin_args` benefits). Converters bypass that route-layer
  schema because their `params` ride inside `source_specs` /
  `clipper_chain` pass-through dicts, so they got their own entry
  point: `MediaConverter.validate_params()` runs the params through
  the converter's own plugin schema, and the two upstream parsers
  (`_parse_multi_media_specs` and `clipper_chain.validate_chain`) call
  it before any expensive work runs. The `audio2image` declared range
  is now actually enforced — `n_mels=10_000_000` is rejected at
  request time instead of building a ~41 GB mel filter bank inside
  librosa. Audited every numeric `PluginField` in the codebase; no
  declared `max` was tighter than real usage, so attaching `Range`
  everywhere was safe.

### CLI / app entry

- ~~**H21. Successful `--autodetect` falls through to Flask startup**~~
  — Not a bug. The audit misread `elif args.local or not args.autodetect:`
  as an independent `if`; it is an `elif` paired with `if args.autodetect:`
  above it, so the two branches are mutually exclusive and Flask never
  starts after a successful autodetect. Simplified the redundant `elif`
  condition to a plain `else:` in `app.py` since reaching it already
  implies `not args.autodetect`.

### Embedding

- ~~**H22. `predict_embedders_to_preload` mismatched media type**~~ —
  shipped in commit `92e27a39` (PR #1561, "H22: persist detector
  embedder for accurate preload prediction"). The detector registry
  now carries an `embedder` field stamped by
  `record_detector_embedder()` during training
  (`vtscore/detectors/workflow.py:174`,
  `vtscore/detectors/labelset_training.py:281`), and both dataset
  and detector paths in `predict_embedders_to_preload()`
  (`vtscore/embedding/loader.py`) go through a shared `_resolve()`
  that prefers `entry["embedder"]` and falls back to the media
  type's default only when the field is unset or unrecognised.
  Legacy detector entries written before the field existed remain on
  the default-embedder fallback until the next retrain stamps them —
  documented as expected behaviour in the function's docstring.

### Security / sync

- ~~**H23. Settings-source filepath template not path-validated**~~ —
  duplicate of C9 (already struck through above); shipped in commit
  `988dca3b` ("logical-bug-audit C9 — validate resolved sync-source
  paths"). Both sync sources now call
  `validate_server_filepath(resolved, base_dir=get_file_access_base_dir())`
  at the end of `_resolve_filepath()`
  (`vtsearch/settings_io/sources/server_json_file/__init__.py:89`,
  `vtscore/labels/sources/server_json_file/__init__.py:121`/L150), so
  the background sync site at `vtsearch/settings.py:932` is covered
  alongside route handlers. Regression tests:
  `tests/io/test_sync_sources.py::test_resolved_template_path_outside_base_dir_rejected`
  (one per source).

### Frontend

- ~~**H24. Vote-state polling chain dies on a single error**~~
- ~~**H25. Active dataset pair is set before load completes**~~ —
  shipped in commit `9470cf1a` (PR #1563, "Fix H25: split
  ActiveContextService into intent + active layers").
  `ActiveContextService` now exposes two layers:
  **intent** (what the user just picked, flips immediately for UI
  affordances like the pulldown highlight) and **active** (the loaded
  pair — what `activeContextInterceptor` reads when attaching
  `X-Dataset-Id` / `X-Detector-Id`).  `ContextSwitchService.flipAndLoad`
  calls `setIntent()` on entry
  (`frontend/src/app/services/context-switch.service.ts:147`) and
  only promotes to `setActive()` inside `finishIfCurrent()` (L297)
  after any required dataset / detector load endpoint has resolved
  *and* the corresponding `loadingTasks` SSE channel has gone idle.
  The latest-wins request-id check on the same line guards against a
  stale switch racing past a newer one.  `setActivePair()` keeps its
  atomic-both-layers semantics so cleanup paths
  (`ActiveContextWatcherService` clearing a removed half,
  `ActiveContextService.clear()`) work unchanged.  Regression tests:
  `frontend/src/app/services/context-switch.service.spec.ts` cover
  (a) intent flips immediately while active stays pinned mid-load,
  (b) active promotion only after both the HTTP load and the
  loading-tasks idle signal arrive, (c) cancel-and-replace via
  request-id mismatch when a second switch starts before the first
  finishes.
- ~~**H26. `recordVote` runs before the HTTP vote returns**~~ — fixed
  by gating the undo-stack push on the POST's success.  A new
  `VoteStateService.submitToggleVoteAndRecord(id, vote, mediaName,
  regionBox?)` captures `previousPolarity` synchronously (before the
  optimistic flip), then calls `submitToggleVote(...)` and pushes the
  undo entry inside the resulting Observable's `tap` — so an entry
  only lands on `past` after the server confirms.  On error the `tap`
  doesn't run, leaving the undo / redo stacks untouched.  The three
  callers (`center-panel`, `find-view`, `label-view`) were migrated to
  the new helper; `recordVote` stays public as a low-level primitive
  (used by the spec) with a docstring pointing future callers at the
  wrapper.  Regressions in `vote-state.service.spec.ts` cover (a) the
  no-entry-on-error path, (b) `previousPolarity` capture before the
  flip, and (c) redo-stack preservation on a failed POST.
- ~~**H27. Binary media endpoints bypass the `activeContextInterceptor`**~~
  — Not a bug. Functional `HttpInterceptorFn`s registered via
  `provideHttpClient(withInterceptors([...]))` apply to **every**
  `HttpClient` call — typed-client wrappers and raw `this.http.get(...)`
  share the same client and the same interceptor chain. The real
  bypass surface is native `<img src>` / `<audio src>` / `<video src>` /
  `<iframe>` / `fetch()`, which don't go through `HttpClient` at all;
  those are already handled by `ActiveContextService.mediaUrl()`
  appending `?dataset_id=…&detector_id=…` query params, with the
  backend reading them as a fallback (`app.py` L238, L252). Also
  removed the four dead `getAudio/getVideo/getImage/getMedia` methods
  on `MediasApiService` — they had no callers; every binary-stream
  consumer goes through `mediaUrl()`.

### Multi-process / settings

- ~~**H28. Per-user cache is process-local; concurrent worker writes lose
  updates**~~ — fixed in `vtsearch/settings.py`. Every per-user (and
  server-tier) write now goes through `_mutate_*_locked`, which holds a
  cross-process `fcntl.flock` on a sibling `.lock` file, re-reads the
  on-disk JSON, applies the mutator in place, and atomic-writes with
  a per-writer `<file>.<pid>.<uuid>.tmp` name. The legacy "mutate cache
  then save whole dict" pattern was removed (including from
  `vtsearch/achievements.py`, which now uses the new public
  `settings.mutate_user(mutator)` RMW helper for nested-dict updates).
  Canonical lock order is `file_lock → settings_lock` everywhere — the
  outer `with _settings_lock:` was removed from setter wrappers and
  from read paths that previously held it across `_ensure_user_loaded`
  (which can transitively trigger setter writes via sync-from-source).
  Covered by `TestConcurrentWrites` in `tests/core/test_settings.py`
  (RMW key-preservation, unique tmp filename pattern, two-thread
  no-deadlock, `mutate_user` nested-dict RMW, `add_autorun_detector`
  cross-process merge).

  **Open follow-ups:**
  - `_synced_users` is still process-local, so on a fresh container
    every worker independently runs sync-from-source for each user
    once. With the RMW fix this is no longer corrupting (just
    duplicate I/O against the source) but worth de-duplicating with
    an mtime-marker if it shows up in profiles.
  - The legacy-settings migration in `_maybe_migrate_legacy_settings_locked`
    still calls `_atomic_write` without the cross-process lock. It is a
    one-shot startup step so the race window is small, but for full
    correctness it should also use `_mutate_server_locked`.
  - Windows has no `fcntl`, so the cross-process lock silently degrades
    to the in-process lock only. Not a regression (the codebase ships
    Linux-only Docker images), but worth a note if a contributor ever
    wants to test on Windows.
- ~~**H29. `_save_user` holds `_settings_lock` across sync I/O**~~ —
  was at `vtsearch/settings.py` ~L331–338. H28's fix already moved
  `_sync_to_source` outside both the file lock and `_settings_lock`,
  which neutralised the worst case (a hung NFS/webhook source could
  no longer freeze every settings read/write process-wide). The H29
  follow-up closes the remaining surface: `_atomic_write` and
  `_load_path` inside `_mutate_server_locked` / `_mutate_user_locked`
  now run under the cross-process `_file_lock` only, and
  `_settings_lock` is acquired briefly at the end just to swap the
  in-memory cache.  A slow local fsync (NFS data dir, full disk,
  hung disk controller) therefore no longer blocks unrelated
  settings reads, and only blocks writes to the *same* user's file
  via the per-file lock — different users' writes proceed in
  parallel.  Regression tests:
  `tests/integration/test_thread_safety.py::TestSlowSettingsIODoesNotBlockOthers`
  (slow `_sync_to_source` doesn't block a reader; slow
  `_atomic_write` for user A doesn't block user B's write; slow
  `_atomic_write` doesn't block settings reads).

### Error flow

- ~~**H30. Detector save's `os.replace` failure leaves in-memory state
  "saved"**~~ — fixed by surfacing persistence failures at the previously-
  unprotected call sites (`POST /api/medias/<id>/vote`,
  `POST /api/votes/seed-from-examples`) with the same try/except + abort
  500 pattern that already guards `fill_labels_from_sort` (C11), and by
  making `vtscore.detectors.workflow.apply_and_retrain` snapshot the
  detector context's vote dicts / region boxes / label history / click
  state before the sync and restore them when `_write_detector` raises.
  A failed save now never leaves votes live in memory while the on-disk
  labelset omits them; the next save no longer inherits ghost state from
  a prior failed write.  Regression tests:
  `tests/detectors/test_workflow.py::TestPersistenceFailureIsTransactional`,
  `tests/api/test_api_contracts.py::TestVotesContract::test_disk_sync_failure_surfaces_as_500`,
  and `tests/detectors/test_detectors.py::TestSeedVotesFromExamples::test_seed_disk_sync_failure_surfaces_as_500`.
- ~~**H31. Partial label-import has no rollback**~~ — fixed by
  isolating each entry inside `_apply_labels` with a per-entry
  try/except and surfacing the per-entry failures in the response
  (`failed` / `failed_count`).  Downstream syncs
  (`sync_labels_to_loaded_detector`, `sync_to_labelset_source`,
  `record_detector_import`) still fire on partial success so the
  in-memory detector and the labelset source stay consistent with what
  actually landed.

---

## Medium — real bugs but lower frequency or non-corrupting impact

### State / concurrency

- ~~**M1.** Lock held during cross-lock callbacks in `toggle_vote` — state ↔
  progress lock ordering creates a narrow but real deadlock window.~~ —
  investigated and closed (2026-05-20).  The literal deadlock the audit
  named was no longer reachable in the current code (post-H1 refactor:
  nothing inside `_progress_lock` calls back into `_state_lock`-acquiring
  code, and route callers of progress functions do not hold `_state_lock`
  when entering the progress module).  But the design was fragile —
  four sites (`vtscore/state/votes.py:_set_vote_locked` + `clear_votes`,
  `vtscore/state/__init__.py:clear_medias` + `set_inclusion`) held
  `_state_lock` while calling into the progress module, while two others
  (`register_detector_context` / `unregister_detector_context`) explicitly
  released it first.  Standardised all six sites on release-first: the
  progress-cache calls now run strictly outside `_state_lock` everywhere,
  so the canonical order is one-directional and adding a new state→progress
  callsite is harder to get wrong.  `_set_vote_locked` no longer touches
  `_progress_lock`; `set_vote` / `toggle_vote` invalidate the progress
  cache after releasing `_state_lock`.  `clear_all` no longer wraps both
  inner clears in a single `_state_lock`, since each inner now releases
  before calling `clear_progress_cache` (the sole caller, dataset-load,
  immediately repopulates state so the loss of cross-clear atomicity is
  acceptable).  Lock-order invariant documented in the
  `vtscore/detectors/labeling_progress.py` module docstring.
- ~~**M2.** `combine_datasets.run_chunked` re-issues IDs starting at 1 on every
  call → cid collision when consumed twice.~~ — fixed in
  `vtscore/cli.py`. Investigation showed this is not unique to
  `combine_datasets`: every chunked importer/loader emits IDs `1..N`
  per yielded chunk (the convention paired with
  `consume_chunks_into`'s renumbering in the in-process loader). The
  HTTP/UI path is safe — `vtscore/datasets/load_pipeline.py:1057-1073`
  already renumbers. The CLI path (`_run_live_pipeline`) scored each
  chunk independently and merged the per-chunk hit lists, so the
  exported JSON's `id` fields collided across chunks (the CSV
  exporter doesn't include `id`, so was unaffected). Fix: added
  `_renumber_chunks()` at the CLI boundary and wrapped both
  `_load_pickle_chunked` and `_load_importer_chunked` with it, giving
  every media a globally unique id in the CLI flow regardless of
  which chunked importer fed it. Covered by
  `tests_lib/cli/test_chunk_renumber.py` (unit) and
  `tests/cli/test_chunked_id_renumber.py` (end-to-end via pickle and
  combine_datasets, including the JSON exporter).
- ~~**M3.** `importers/base.py` L407–410 — skipped records leave `next_id`
  unincremented, ID collisions on first-record-skip.~~

### Detector

- ~~**M4.** `populate_label_embeddings` cache not invalidated when a
  `region_box` is removed from an element; stale pooled vector
  continues to be used.~~
- ~~**M5.** `labelset_elements.resolve_current_dataset_cid` can return a
  colliding-MD5 cid in cross-dataset labelsets → clicks vote the
  wrong media.~~ — closed as not-a-bug on `dev`. The function returns
  `cids[0]` from the origin+name ∪ md5 union, which is only ambiguous
  when two cids in the active dataset share an MD5. Both Flask
  dataset-load paths (`vtscore/datasets/load_pipeline.py` and
  `vtsearch/routes/datasets/registry.py`) run `collapse_duplicates`,
  which collapses same-MD5 medias into a single `dupe_set`
  representative — so the md5 lookup never yields more than one cid.
  Cross-dataset MD5 match is the intended semantic (same content →
  same logical media), not a miscompute. Docstring on
  `resolve_current_dataset_cid` records the invariant and points at
  the regression test
  (`tests/datasets/test_duplicates.py::test_collapse_duplicates_yields_unique_md5_lookup`).
  Related but distinct issues are left as their own items: M4 (region
  box cache invalidation, since closed upstream) and the
  `vote_detector_label` handler dropping `region_box` when mirroring
  into in-memory votes (`vtsearch/routes/detectors/labels.py:601`).
- ~~**M6.** `restore_labels_from_detector` resolves by MD5 only on the
  second pass; dedup-collapsed cids can land votes on the wrong cid
  after reload.~~ — investigated and closed as not a real bug. Pass 1
  already consults `md5_lookup` via `resolve_media_ids`
  (`vtscore/state/media_lookup.py:62`), and the dedup-collapse reload
  path lands on the correct rep cid in every case (deduped,
  un-deduped, cross-dataset). The forward-pointer to **M5** in the
  original M6 note is now also closed (see M5 above); a separately
  worrying latent issue is that pass 2 recomputes the *parent*
  file's md5 for `converter` origins
  (`vtscore/detectors/resolver.py:_resolve_converter`) while the
  dataset stores the converted-output md5 — track under detector
  findings if it bites.
- ~~**M7. `safe_thresholds` read at training time, cached on DetectorContext, never refreshed**~~
  — partial close. The audit's "baked into the detector JSON" claim was
  wrong: the threshold is never serialised (see the "No Persisted Vectors
  or MLPs" rule in `CLAUDE.md`) — every detector JSON write site lists
  only `name` / `text_query` / `media_example` / `media_type` / `examples`
  / `created_at` / `labelset` / `input_spec`. But a narrower staleness
  was real: the in-memory `DetectorContext.model` / `threshold` cached on
  detector load were not invalidated when the user changed
  `safe_thresholds` / `inclusion` / `calibrate_count` /
  `calibration_fraction`, so `/api/find-label`, `/api/find`, and
  `/api/auto-detect` (the three consumers that short-circuit on the
  cached MLP) kept scoring with the prior setting. Sort / vote paths
  retrained every call and so were already correct. Fixed by a new
  `vtscore.state.core.invalidate_loaded_detector_models()` that walks
  every loaded `DetectorContext` and clears `model` + `threshold`; the
  setters for all four training-relevant settings in
  `vtscore/state/__init__.py` call it on actual change. The
  `/api/settings PUT` route now dispatches those three keys through
  `vtsearch.state` (matching the existing `inclusion` path) so both the
  dedicated endpoints (`/api/safe-thresholds` etc.) and the bulk
  settings endpoint trigger invalidation. Regression test:
  `tests/sorting/test_safe_thresholds.py::TestTrainingSettingsInvalidateLoadedDetector`.

### Datasets / loaders

- **M8.** ~~Thin-mode pickle loader treats `embedding: None` as present, then
  `np.array(None)` produces an object-dtype row.~~ **Shipped (with M12).**
  `_convert_one_pickle_media` now treats missing key and explicit `None`
  identically for both thin and full modes — entry is skipped and the
  "missing media" warning fires.
- ~~**M9.** `loader_folder._has_override` doesn't warn when both `rel_path`
  and `file_name` override entries exist with different embeddings.~~
- **M10.** ~~`clipper_chain._run_clipper_step` assumes deterministic output count
  across calls — no validation.~~ **Shipped.** `_run_clipper_step` /
  `_run_converter_step` now stamp `n_out`, `clip_index`, and a short
  `content_hash` on every trail entry. `_select_chain_output` prefers
  content matching over positional, logs warnings on output-count
  drift / no-match / ambiguous match, and returns `None` instead of
  silently picking `outputs[0]` (was a regression vs. the legacy
  `_clip_text_to_bytes` resolver path).
- **M11.** ~~Stale media in `cli._score_medias_with_detectors` when some
  embeddings are `None` (zip truncates silently).~~ **Shipped.**
  `_score_medias_with_detectors` now uses `zip(all_ids, scores,
  strict=True)` so a partial embedding matrix raises `ValueError`
  instead of silently dropping hits. With the M12 fix in place this
  loop should never be partial, but the strict zip guards against
  future regressions.
- **M12.** ~~`loader_pickle._build_pickle_full_media` has no null-check before
  `np.array(media_info["embedding"])`.~~ **Shipped.**
  `_convert_one_pickle_media` skips any entry whose `embedding` is
  missing or `None` before the build helpers run — symmetric with the
  thin-mode path (M8) and with the folder loader's drop-on-no-embed
  behaviour. The registry load path doesn't re-embed after
  `load_dataset_from_pickle`, so preserving `None` would have just
  pushed the crash into the first sort/find call; dropping the
  poisoned entry keeps the dataset usable and surfaces the loss via
  the existing `missing_media` warning.

### Routes / API

- ~~**M13.** `learned_scores` in `/api/votes` can serialize as JSON
  `NaN`/`Infinity` if the MLP destabilizes — invalid JSON to strict
  clients.~~ — fixed by routing every sigmoid→score path through
  `vtscore.utils.scores.sigmoid_to_finite_scores` (NaN/±Inf → `-1.0`
  sentinel) and adding a defensive `finite_or` guard at
  `GET /api/votes`. Sanitised sites: `labelset_train_and_score`,
  `train_and_threshold`, `_score_all_media`, `/api/learned-sort`,
  `/api/label-file-sort`, `/api/find-label`, `/api/find` (live + cold
  paths), `/api/auto-detect`, and CLI autodetect. `-1.0` sits outside
  the `[0, 1]` sigmoid range so `score >= threshold` is always False
  for sanitised scores and they sink to the bottom of any sort —
  matches the frontend's existing `learnedScores[id] ?? -1` fallback.
  Regression test in `tests/api/test_api_contracts.py` parses the
  response with a `parse_constant` that rejects `NaN`/`Infinity`.
- ~~**M14.** `diversity_tree_next_sample` references stale media IDs after
  `/api/dataset/clear`.~~ — investigated and closed.  The literal scenario
  the audit named is no longer reachable: `clear_medias` (called via
  `clear_dataset → clear_all`) sets `ctx.diversity_tree = None`, and the
  active-dataset path of `/api/dataset/clear` unregisters the context so
  subsequent requests either raise `DatasetNotLoadedError` (H16) or hit the
  request-missing sentinel (whose tree is `None`).  A regression test pins
  the behavior.  Two adjacent bugs surfaced during the investigation and
  were fixed in the same PR: `clear_votes()` did not reset the dataset's
  diversity-tree `seen` / `_labeled` sets, so `/api/votes/clear` left
  `diversity_tree_next_sample` skipping previously-voted nodes and the
  diversity-level chip stuck above zero; and
  `ensure_votes_match_active_dataset` rehydrated detector votes via
  `apply_label(silent=True)` (which skips the per-vote tree update)
  without replaying onto the tree, so swapping detectors on the same
  dataset left the tree reflecting the previous detector's seen state.
  Both now reset the tree under `_state_lock` via a new
  `DiversityTree.reset_seen()` / `resync_diversity_tree_to_detector`
  helper pair that also dedupes the replay loop in `build_diversity_tree`.

### Settings / sync

- ~~**M15.** Pending labelset sync stores `dataset_ctx = None` without
  checking; later `_run_pending_sync` triggers `AttributeError`.~~ —
  investigated and closed as not a real bug.
  `sync_to_labelset_source` captures `dataset_ctx = get_active_context()`
  (`vtscore/labels/sync.py:97`), and `get_active_context()` /
  `get_active_detector_context()` are invariantly non-None: their
  resolution chain (`vtscore/state/core.py:461`, `:614`) falls back to a
  `_request_missing_*` sentinel inside a Flask request and to
  `_empty_*_context` outside one. The dead `is None` halves at
  `vtscore/labels/sync.py:89` and `:207` are what tipped the audit toward
  this finding, but the `not detector_ctx.labelset_source` half already
  screens out both sentinels (their `labelset_source` is `None`). When
  the timer fires, `validated_vote_snapshot` reads `medias` /
  `dataset_id` off whatever ctx-shaped object was captured, so no
  `AttributeError` can fire. Two adjacent real-but-milder concerns were
  noted during the investigation and left open: a misconfigured request
  with `X-Detector-Id` but no `X-Dataset-Id` silently no-ops the sync
  (captured sentinel → `validated_vote_snapshot` returns `safe=False`),
  and the captured ctx references survive an unload-before-fire race
  inside the 200ms debounce window. Both are silent inconsistencies, not
  crashes; file separate findings if they ever bite.
- ~~**M16.** Sync to source on first read after `_synced_users` marker can
  still return stale local config if source changes silently.~~ — fixed
  alongside two adjacent latent defects in `_ensure_user_loaded`.
  Replaced the `_synced_users: set[str]` "claim-then-sync" marker with
  per-user `_UserSyncState` bookkeeping (`last_version`,
  `last_check_monotonic`, `last_sync_succeeded`, `dirty_keys`) protected
  by a per-user `_per_user_sync_lock` RLock, so:
  (a) the TOCTOU race where a concurrent reader saw the marker before
  the actual sync had populated the cache is gone — the lock now
  serialises the decide-and-sync slow path,
  (b) a transient first-sync failure no longer permanently locks the
  user out of sync (`last_sync_succeeded` stays `False` and the slow
  path retries past a 1-second rate-limit window),
  (c) a new `SyncSource.peek_version` hook (default `None`, implemented
  for `server_json_file` via `st_mtime_ns`) makes an upstream change
  visible automatically on the next read after the freshness window
  elapses, instead of requiring manual `POST /api/settings-sources/sync`
  or process restart.  Auto re-sync respects local `dirty_keys` so a
  freshly clicked toggle isn't silently overwritten by an upstream
  value; manual `sync_from_settings_source` ignores dirty markers
  (explicit user pull) and clears them.  `_sync_to_source` clears
  dirty markers on a successful export (source now matches local).
  Regression tests:
  `tests/io/test_sync_sources.py::TestSyncFromSourceFreshness`
  (six cases — version-bump detection, first-failure retry, concurrent
  reader sees post-sync cache, dirty-key skip on auto re-sync, manual
  sync clears dirty, freshness window avoids repeat probes).
- ~~**M17.** Legacy migration `_maybe_migrate_legacy_settings_locked` pops keys
  from in-memory cache before per-user disk write; per-user-write
  failure leaves cache and disk diverged.~~ — investigated and partially
  closed (2026-05-21). The literal ordering claim is **inverted**:
  `_maybe_migrate_legacy_settings_locked` writes the user file
  *before* popping `_server_cache`, and returns early on user-write
  failure, so the divergence described is impossible. A separate,
  narrower divergence existed in the *server*-file rewrite step: if
  `_atomic_write(_server_settings_path(), _server_cache)` raised
  after the in-memory pop, `_server_cache` (legacy keys gone) and
  the on-disk server file (legacy keys still present) would silently
  disagree until the next `_mutate_server_locked()` re-read the disk.
  Fix: compute the server-tier-only candidate first, write it, and only
  pop the in-memory cache after the disk write returns successfully
  (early-return on failure mirrors the user-write branch). Failure-path
  coverage added in `tests/core/test_per_user_settings.py` for both
  the user-write and server-rewrite branches.

### Embedding / training

- **M18.** `_PeekUnpickler` doesn't override `FLOAT` / `SETITEMS` opcodes →
  falls back to slow real unpickle for older protocols.
- ~~**M19.** `embed_text_enriched` crashes (`np.mean` on empty) when text encoder
  fails and all wrappers return None.~~ — investigated and closed as not a
  real bug (2026-05-21).  `vtscore/media/embedder.py:727-750` explicitly
  guards the empty-list case with `if not embeddings: return
  self.embed_text(text)` immediately before the `np.mean` call, so the
  crash described is unreachable.  Git archaeology confirms the guard has
  been in place since the function was introduced in PR #334 (commit
  `b2c7bb4a`, 2026-02-28) — the audit's claim was a false positive.
  `tests/sorting/test_enrich_descriptions.py::test_enriched_falls_back_when_all_fail`
  already exercises this path.  All other `np.mean` call sites in the
  codebase (`vtscore/eval/metrics.py:53,60`,
  `vtscore/eval/label_curve.py:390`) carry their own empty-input guards
  too, so no related real bugs to fix.
- **M20.** XCLIP single-frame video: `linspace(0, 0, 1)` + padding gives 8
  identical frames → degenerate embedding.
- **M21.** Empty paragraph clip survives to dataset with `None` embedding;
  embedding-matrix builder later misbehaves.

### Auth / context

- **M22.** `set_thread_user()` cleanup relies on every caller's `finally`; a
  future `ThreadPoolExecutor` reuse would leak user identity across
  requests.
- **M23.** `setup_logging` re-running can leave duplicate handlers on
  non-root loggers.
- **M24.** `CoreConfig.from_settings()` raises if a blueprint's module-level
  code runs before `vtsearch/shim/__init__.py` registers the
  builder.

### Frontend

- **M25.** `LeftPanelComponent` lacks `OnDestroy` / `takeUntil` on init
  subscriptions; subscriptions leak across dataset switches.
- **M26.** `labelset-state.service` `startPolling()` is not tied to
  `destroy$`; rapid switches leak polls.
- **M27.** `progress-events.service` doesn't reconcile stale `task_id`s after
  backend restart.
- **M28.** Audio waveform fetch's `catch {}` silently shows "Unable to load
  waveform" with no UI state propagation.
- **M29.** `AudioContext` not cleaned up on rapid navigation → resource
  exhaustion.
- **M30.** Autopilot phase transitions can oscillate
  (`hard → new → hard → new`) when smart/stable status flickers.
- **M31.** `settings-importer-modal` auto-closes after 1.5s timeout regardless
  of operation duration.

### Security

- **M32.** `sanitize_template_value` allows `...` (and worse — any non-`.` /
  `..` token of dots).
- **M33.** `rglob_follow_symlinks` doesn't detect cycles → CPU/RAM DoS on
  circular link layouts.
- **M34.** Email exporter validates "@" only; `"@example.com"` passes and
  fails at SMTP time.

### Eval / exporters

- **M35.** `voting_iterations.py` reports F1/FPR with one good + one bad vote
  as if reliable.

---

## Low — latent / cosmetic / hypothetical

- **L1.** `_run_pipeline` returns `None` silently in text mode (no "done"
  signal).
- **L2.** Pipeline-vs-server logic in `app.py` L792 is masked by `sys.exit()`
  but breaks if the function is ever refactored to return.
- **L3.** Frontend cross-pane settings: `getViewMode()` falls back to
  hard-coded defaults when the per-media-type entry isn't loaded.
- **L4.** `get_settings_source_config` reads cache without re-checking sync
  state.
- **L5.** Frontend `clip_box` exports: list-of-floats CSV-joined without
  quote-protection (edge case).
- **L6.** Empty `Origin.params` not always dropped consistently across
  importers.
- **L7.** `eval/metrics.py` returns F1=0 for empty test set without flagging
  the degenerate case.
- **L8.** Logging of vote-related events truncates achievement traceback for
  UI display.
- **L9.** `get_param` returns `""` for unknown keys; silently swallows
  renamed parameters.

---

## Recurring patterns (fix-in-batches root causes)

These themes show up across many findings. Addressing each pattern in
one PR is far more effective than one-off fixes.

1. **Background-thread context propagation.** Every `Thread(target=…)`
   spawn in the repo should set user + dataset + detector
   thread-locals (where applicable) before invoking the target, in a
   `try/finally`. Candidates:
   - `JobManager._run`
   - `_run_importer_in_background`
   - `_stage_importer_in_background`
   - `_warmup_embedder_async`
   - `smart_preload_in_background`
   - `preload_embedder_for_dataset`
   - `timed_progress` ticker.

   A small helper
   `with run_with_context(user, dataset_id, detector_id): …` removes
   the repetition.

2. **Template-path substitution → re-validate.** Every source /
   exporter that interpolates `{username}`, `{detector_id}`,
   `{detector_name}` must call `validate_server_filepath()` on the
   **resolved** path before opening a file. Affected:
   `labels/sources/server_json_file`,
   `settings_io/sources/server_json_file`, and any future plugin in
   the same family.

3. **Achievement recording at one site.** Move `record_vote()` (and
   all achievement hooks) into `apply_label_with_click_time` so bulk
   paths (`fill-from-sort`, label import, find-label) credit the user
   uniformly. Also: call it *inside* the state lock, not after.

4. **Embedding-matrix cache invalidation.** Bump a `media_revision`
   counter on every `medias` mutation; the matrix accessor checks the
   counter. Same fix neutralizes both clip/dedup invalidation (C4)
   and dynamic seeding races.

5. **Embedder identity is its own first-class attribute.** Track the
   detector's *training* embedder separately from the active
   dataset's embedder. Every train/score path must compare them and
   refuse / clear caches when they differ — don't only compare
   against the dataset.

6. **`X-Dataset-Id` / `X-Detector-Id` headers must be required, not
   silently defaulted.** When the header is missing or refers to an
   unloaded context, the `before_request` middleware should return
   400 / 410, not fall back to the empty proxy. Catches C5, the
   silent-mistarget routes, and the binary-streaming bypass.

7. **Frontend cache keys must be dataset-qualified.**
   `MediaMetadataCacheService` and any other id-keyed cache should
   key on `${datasetId}:${mediaId}`.

8. **Vote endpoints should be transactional.**
   apply-then-retrain-then-sync should either all-succeed or
   all-rollback; surface failures to the frontend instead of
   returning 200 and logging server-side.

9. **Background jobs need a clear error state.** Every failure path
   on a load / embed / train job should call
   `update_progress(task_id, error="…")` so the frontend can stop
   spinning and show what went wrong.

---

## Suggested fix order (highest leverage first)

1. **C1, C2, C3** — gate hand-off + thread-local context
   propagation. One small helper (pattern #1) unblocks 3+ bug
   classes.
2. **C4** + pattern #4 — embedding-matrix cache invalidation via a
   `media_revision` counter.
3. **C7** — clamp `xcal_threshold` to a finite sentinel and assert
   no `NaN` in `safe_threshold`.
4. **C9** + pattern #2 — re-validate every resolved template path
   in `SyncSource._resolve_filepath()`.
5. **C5, C10, C11, C12** — straightforward request-handling and
   frontend-cache fixes once the above patterns are in place.
6. Pattern #6 — make header presence + context-loaded a hard
   precondition.
7. Detector-embedder identity (pattern #5) + C8 (achievement
   uniformity).

---

## Open follow-ups

Cross-cutting open items that don't fit any single finding above:

- **Pattern #4 (media_revision counter)** is still unimplemented.
  The C4 stage-level invalidation closes the known clip/dedup hole,
  but any future mutation site that changes embeddings without
  changing the id set will reintroduce the same class of bug. A
  `media_revision` counter on `DatasetContext` bumped from every
  `medias` mutation (or a `MediasDict` subclass that does so
  transparently) would neutralise the whole category and let the
  matrix accessor compare a single int instead of two id lists.

- **H1 follow-up: labelset element vote endpoint still toggles.**
  `POST /api/detectors/<name>/labels/<element_id>/vote` (and the
  underlying `apply_element_vote_in_data` in
  `vtscore/detectors/labelset_elements.py`) still takes
  `vote: "good" | "bad"` with toggle-on-same-direction semantics — a
  stale-view tab voting against an already-good labelset element
  removes the element from the on-disk labelset, same kind of
  inflation race the media-vote H1 fix eliminated.  Migrating that
  endpoint to absolute-target (`target: "good" | "bad" | "remove"`)
  would let the in-memory `set_vote()` mirror run idempotently too,
  closing the same class of race on the labelset side.  Deferred
  because the labelset element CRUD is a separate user surface
  (`vt-labels-list` modal, not the centre-pane click flow that H1
  was scoped to).  The on-disk labelset write is idempotent already
  via `_write_detector`; the race is on the in-memory mirror and
  the `num_training` registry counter.

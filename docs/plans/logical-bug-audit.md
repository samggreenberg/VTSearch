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
- **H5. Detector embedder not revalidated on dataset switch** —
  `vtsearch/routes/detectors/scoring.py` + `dataset_sync.py`.
  `ensure_votes_match_active_dataset()` rehydrates votes but doesn't
  check the new dataset's embedder against the detector's training
  embedder. An MLP trained on CLAP can score SigLIP vectors with no
  warning.
- ~~**H6. `train_model` produces degenerate single-class model**~~
- ~~**H7. Vote applied before retrain; retrain failure leaves vote live**~~

### Datasets

- ~~**H8. Origin dict shared by reference across medias**~~
- ~~**H9. Single-item split has 0 test samples**~~
- **H10. Clipped media re-ingest reads whole file for MD5** —
  `vtsearch/datasets/ingest.py` L275. MD5 of the full parent
  ≠ MD5 of the clip, so dedup never re-matches the original clip.
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
- **H32. `add_media_to_pile` TOCTOU between md5 check and insertion** —
  `vtsearch/routes/media/list.py` L607–608 vs. L687–690. The md5
  lookup (`snap = snapshot_medias()` + `build_media_lookup(snap)`)
  runs outside `_state_lock`; the new-media insertion happens under
  the lock ~80 lines later. Two concurrent uploads of the same file
  can both miss the existing-cid hit, both embed, and both insert,
  producing duplicate medias with identical md5. Fix sketch:
  re-check `md5_lookup` (or recompute it from `medias`) under
  `_state_lock` immediately before assigning `new_id` and writing
  `medias[new_id]`, and dispatch to the existing-cid branch if a
  match appears in the recheck.
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
- **H34. Missing `X-Detector-Id` → silent detector fallback** —
  `vtscore/state/core.py` `get_active_detector_context()` falls
  through to the thread-local (then `_empty_detector_context`) when
  `g._detector_context` is unset. The H16 fix closed the sub-case
  where the header is *present* but names an unloaded id (now 409);
  the *header-absent* case remains open. Any route that mutates votes
  without requiring the header — `add_media_to_pile`, `vote_media`,
  bulk-label endpoints — silently writes to whatever the thread-local
  resolves to, which on a Flask threaded-server worker can be the
  detector left over from a previous request on the same thread.
  Detector-side analog of H13; the fix is the same shape (reject the
  request when the header is absent for any vote-mutating endpoint).

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

- **H22. `predict_embedders_to_preload` mismatched media type** —
  `vtsearch/embedding/loader.py` L162. Registry filters by
  `media_type_id`, but dataset metadata may carry a different
  `media_type` string for a custom embedder; preloads the wrong
  model.

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
- **H25. Active dataset pair is set before load completes** —
  `frontend/src/app/services/context-switch.service.ts` L132–134.
  `setActivePair()` runs before the load is verified; interceptor
  immediately tags subsequent requests with a context that may not
  be loaded → cascade of 404s.
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
- **H29. `_save_user` holds `_settings_lock` across sync I/O** —
  `vtsearch/settings.py` ~L331–338. Slow source (NFS, webhook) blocks
  all other settings reads/writes globally. (Per-user side is
  incidentally addressed by H28's fix — `_sync_to_source` now runs
  outside the file lock and outside `_settings_lock`. Server-tier
  paths through `_atomic_write` no longer hold `_settings_lock` across
  file I/O either, but a dedicated audit of every remaining sink is
  still warranted.)

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
- **H31. Partial label-import has no rollback** —
  `vtsearch/routes/labels/importers.py` L143. Failure mid-loop leaves
  a half-applied labelset; user has to manually figure out which
  ones landed.

---

## Medium — real bugs but lower frequency or non-corrupting impact

### State / concurrency

- **M1.** Lock held during cross-lock callbacks in `toggle_vote` — state ↔
  progress lock ordering creates a narrow but real deadlock window.
- **M2.** `combine_datasets.run_chunked` re-issues IDs starting at 1 on every
  call → cid collision when consumed twice.
- **M3.** `importers/base.py` L407–410 — skipped records leave `next_id`
  unincremented, ID collisions on first-record-skip.

### Detector

- **M4.** `populate_label_embeddings` cache not invalidated when a
  `region_box` is removed from an element; stale pooled vector
  continues to be used.
- **M5.** `labelset_elements.resolve_current_dataset_cid` can return a
  colliding-MD5 cid in cross-dataset labelsets → clicks vote the
  wrong media.
- **M6.** `restore_labels_from_detector` resolves by MD5 only on the second
  pass; dedup-collapsed cids can land votes on the wrong cid after
  reload.
- **M7.** `safe_thresholds` is read at *training* time and baked into the
  detector JSON; per-user threshold preference cannot change after
  save.

### Datasets / loaders

- **M8.** Thin-mode pickle loader treats `embedding: None` as present, then
  `np.array(None)` produces an object-dtype row.
- **M9.** `loader_folder._has_override` doesn't warn when both `rel_path`
  and `file_name` override entries exist with different embeddings.
- **M10.** `clipper_chain._run_clipper_step` assumes deterministic output count
  across calls — no validation.
- **M11.** Stale media in `cli._score_medias_with_detectors` when some
  embeddings are `None` (zip truncates silently).
- **M12.** `loader_pickle._build_pickle_full_media` has no null-check before
  `np.array(media_info["embedding"])`.

### Routes / API

- **M13.** `learned_scores` in `/api/votes` can serialize as JSON
  `NaN`/`Infinity` if the MLP destabilizes — invalid JSON to strict
  clients.
- **M14.** `diversity_tree_next_sample` references stale media IDs after
  `/api/dataset/clear`.

### Settings / sync

- **M15.** Pending labelset sync stores `dataset_ctx = None` without checking;
  later `_run_pending_sync` triggers `AttributeError`.
- **M16.** Sync to source on first read after `_synced_users` marker can still
  return stale local config if source changes silently.
- **M17.** Legacy migration `_maybe_migrate_legacy_settings_locked` pops keys
  from in-memory cache before per-user disk write; per-user-write
  failure leaves cache and disk diverged.

### Embedding / training

- **M18.** `_PeekUnpickler` doesn't override `FLOAT` / `SETITEMS` opcodes →
  falls back to slow real unpickle for older protocols.
- **M19.** `embed_text_enriched` crashes (`np.mean` on empty) when text encoder
  fails and all wrappers return None.
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

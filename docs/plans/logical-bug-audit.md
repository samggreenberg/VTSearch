# Logical-Bug Audit: 2026-05

**Status:** All findings resolved; kept as the audit record. C1–C12 (critical)
and H1–H34 (high) shipped; the security trio M32–M34 and data-integrity M21
shipped 2026-06-24; the L1–L9 low batch was triaged the same day (L7/L9 fixed,
the rest closed as stale / hypothetical / by-design); the last mediums M29 +
M35 shipped 2026-06-25. No open findings remain. A handful of cross-cutting
open follow-ups (below) are the only work still owed.

**Scope:** Multi-agent audit (10 per-subsystem + 5 cross-section interaction
passes) of the entire VTSearch codebase, focused on **logical** bugs — missed
expectations between modules, race conditions, silent miscompute, data
corruption, broken invariants. Syntax, typing, and lint were out of scope
(covered by `ruff` / `pyright` / `tsc`). ~95 distinct findings after dedup.

## How to read this doc

- Findings had stable IDs (`C#` Critical, `H#` High, `M#` Medium, `L#` Low) so
  they could be referenced from other docs / PRs.
- Each resolved finding is now a single line: what it was + outcome, with the
  landing PR / test refs in parentheses. The full fix summary is the PR that
  landed it (git log, not this doc). "Closed" means investigated and found to be
  not-a-bug / by-design / stale; "shipped" means a fix landed.
- File:line references are approximate; line numbers drift as the code evolves.

**Audit coverage.** Per-subsystem agents: (1) state & concurrency, (2) detector
training, (3) datasets & importers, (4) routes & API, (5) settings & sync,
(6) embedding & training infra, (7) security validation, (8) plugins /
converters / eval / exporters, (9) auth / CLI / app entry, (10) frontend Angular
logic. Cross-section agents: (11) dataset↔detector↔embedder, (12)
settings↔everything, (13) background-jobs / context propagation, (14)
error/exception flow, (15) frontend↔backend contract seams.

---

## Open follow-ups

Cross-cutting items still owed. Everything ID'd (C/H/M/L) is resolved; these are
the only remaining work.

- **Pattern #4 — `media_revision` counter is still unimplemented.** The C4
  stage-level invalidation closes the known clip/dedup hole, but any future
  mutation site that changes embeddings without changing the id set will
  reintroduce the same class of bug. A `media_revision` counter on
  `DatasetContext` bumped from every `medias` mutation (or a `MediasDict`
  subclass that does so transparently) would neutralise the whole category and
  let the matrix accessor compare a single int instead of two id lists.

- **H28 residual multi-process items** (from the per-user settings RMW fix) —
  **all three resolved.**
  - ~~Duplicate cross-worker sync-from-source I/O.~~ **Shipped.** A best-effort
    per-user on-disk marker (`<user_settings.json>.syncmark`, written by
    `_write_sync_marker`) records the source `peek_version` last applied. A
    fresh worker whose first read finds the source unchanged at that version
    adopts it and skips the duplicate `source.load()` — the values are already
    on disk (`_adopt_sync_marker_if_current`; `test_sync_sources.py::TestSyncDedupMarker`).
  - ~~Legacy-settings migration wrote without the cross-process lock.~~
    **Shipped.** `ensure_server_loaded` now does its first load + one-shot
    migration inside `_load_and_migrate_server` under the server `file_lock`,
    and `_maybe_migrate_legacy_settings` takes the default user's `file_lock`
    around its write — so both `_atomic_write` calls are cross-process safe.
    The migration was hoisted out of `settings_lock` (callers invoke
    `ensure_server_loaded` outside the lock) to keep the canonical
    file→settings order (`test_per_user_settings.py::TestMigrationCrossProcessLock`).
  - ~~Windows `fcntl` degradation was untested.~~ **Covered.** The fallback to
    the in-process lock is now exercised by
    `test_settings.py::TestFileLockWithoutFcntl`. Still Linux-only in
    production; the fallback only matters for a Windows contributor running
    multiple processes against one settings file (which they shouldn't).

---

## Findings resolved

### Critical — data corruption / loss / hangs / silent miscompute

- **C1.** Download gate never released if importer skips the `"embedding"` status. **Shipped.**
- **C2.** JobManager never sets dataset/detector thread-local context. **Shipped.**
- **C3.** Dataset background load tasks don't set thread-local dataset context. **Shipped.**
- **C4.** Embedding-matrix cache not invalidated after clip/dedup. **Shipped** (stage-level; see Pattern #4 follow-up).
- **C5.** `find_label` body field could override the request's dataset context. **Shipped.**
- **C6.** Zip-slip in HTTP archive importer (zip and tar). **Shipped.**
- **C7.** NaN/Infinity threshold leaked through safe-threshold blending. **Shipped.**
- **C8.** Bulk label paths skipped achievement recording. **Shipped.**
- **C9.** Path-template substitution missing post-resolution validation (commit `988dca3b`). **Shipped.**
- **C10.** MD5 / metadata cache collision returned wrong-dataset media on switch. **Shipped.**
- **C11.** `fill_labels_from_sort` silently swallowed sync failures. **Shipped.**
- **C12.** Orphaned dataset registry entry on activation failure. **Shipped.**

### High — likely-encountered correctness or security bugs

- **`record_vote()` called after releasing `_state_lock`.** **Shipped.**
- **H1.** Vote-progress / `record_vote` race — replaced the toggle contract on `POST /api/medias/<id>/vote` with an absolute `target` contract; `set_vote()` no-ops on idempotent re-applies; progress cache invalidates on any training-set membership change (`test_votes.py`, `test_error_recovery.py`, `test_achievements.py`, `test_api_contracts.py`, `test_patch_embedder.py`, `vote-state.service.spec.ts`).
- **H2.** Cross-dataset region-box loss — `_embed_one` runs `patch_forward` + `box_to_vote_vector` so a region vote survives a dataset switch (`TestRegionAwareTrainingCrossDataset`).
- **H3.** Embedder drift on save → reload. **Shipped.**
- **H5.** Detector embedder not revalidated on dataset switch — `invalidate_detector_model_on_embedder_mismatch()` + `resolve_label_embeddings(embedder_name=…)` (`test_detectors.py`, `test_resolver.py`).
- **H6.** `train_model` produced degenerate single-class model. **Shipped.**
- **H7.** Vote applied before retrain; retrain failure left vote live. **Shipped.**
- **H8.** Origin dict shared by reference across medias. **Shipped.**
- **H9.** Single-item split had 0 test samples. **Shipped.**
- **H10.** Clipped media re-ingest read whole file for MD5 — clip-aware hashing via `_resolve_clip_content_and_embedding` (`test_label_import_ingestion.py::TestClippedReingest`).
- **H11.** Multi-media import with empty form yielded empty dataset. **Shipped.**
- **H12.** `add_media_to_pile` race — **closed, not a real race** (`before_request` pins request-local contexts); the real bugs at that line window are H32/H33/H34.
- **H13.** `vote_media` silent-mistarget on dropped header. **Shipped.**
- **H14.** `export_labels` leaked votes across datasets. **Shipped.**
- **H15.** File-browser symlink metadata leak — listing loop now resolves symlinks and skips escapees (`TestBrowseSymlinks`). A distinct importer `followlinks=True` escape is noted but separate.
- **H16.** Header refers to unloaded dataset → silent fallback — **shipped** (also closes half of H34).
- **H32.** `add_media_to_pile` TOCTOU between md5 check and insertion — re-runs `build_media_lookup` under `_state_lock` before assigning the id (`test_medias.py::TestAddToPile`).
- **H33.** `add_media_to_pile` label not synced to disk — both branches now call `_sync_pile_label_to_storage()` (`test_medias.py`).
- **H34.** Missing `X-Detector-Id` → silent detector fallback — added `require_detector_header` / `require_dataset_header` decorators on every vote-mutating endpoint (`_shared.py`); read endpoints keep fall-through.
- **H17.** Plugin scanner silently shadowed duplicate names. **Shipped.**
- **H18.** CSV exporter doesn't escape embedded newlines — **closed, not a bug** (stdlib `csv.writer` QUOTE_MINIMAL; path was also stale).
- **H19.** Required select fields silently accept empty string — **closed, not a bug** (`_non_empty_after_strip` + OneOf already reject empty).
- **H20.** `audio2image` had no upper bound on `n_mels` — `_build_number()` attaches `validate.Range`; `MediaConverter.validate_params()` enforces it for converter params.
- **H21.** Successful `--autodetect` falls through to Flask startup — **not a bug** (`elif` misread); simplified to plain `else`.
- **H22.** `predict_embedders_to_preload` mismatched media type — persist the detector's training embedder (commit `92e27a39`, PR #1561).
- **H23.** Settings-source filepath template not path-validated — duplicate of C9 (commit `988dca3b`; `test_sync_sources.py`).
- **H24.** Vote-state polling chain died on a single error. **Shipped.**
- **H25.** Active dataset pair set before load completes — split `ActiveContextService` into intent + active layers (commit `9470cf1a`, PR #1563; `context-switch.service.spec.ts`).
- **H26.** `recordVote` ran before the HTTP vote returned — `submitToggleVoteAndRecord` pushes the undo entry only inside the success `tap` (`vote-state.service.spec.ts`).
- **H27.** Binary media endpoints bypass `activeContextInterceptor` — **not a bug** (functional interceptors apply to every `HttpClient` call); removed four dead methods.
- **H28.** Per-user cache process-local; concurrent worker writes lost updates — cross-process `flock` RMW via `_mutate_*_locked` (`test_settings.py::TestConcurrentWrites`). The three residual multi-process items (sync dedup marker, migration under the file lock, Windows fallback coverage) are now also resolved — see Open follow-ups.
- **H29.** `_save_user` held `_settings_lock` across sync I/O — I/O moved under the per-file lock only (`test_thread_safety.py::TestSlowSettingsIODoesNotBlockOthers`).
- **H30.** Detector save's `os.replace` failure left in-memory state "saved" — transactional snapshot/restore in `apply_and_retrain` + abort-500 at the unprotected call sites (`test_workflow.py`, `test_api_contracts.py`, `test_detectors.py`).
- **H31.** Partial label-import had no rollback — per-entry try/except surfacing `failed` / `failed_count`.

### Medium — real but lower-frequency / non-corrupting

- **M1.** Lock held during cross-lock callbacks in `toggle_vote` — **closed** (deadlock unreachable post-H1); standardised all six state→progress sites on release-first.
- **M2.** `combine_datasets.run_chunked` re-issued IDs from 1 → cid collision — added `_renumber_chunks()` at the CLI boundary (`tests_lib/cli/test_chunk_renumber.py`, `tests/cli/test_chunked_id_renumber.py`).
- **M3.** `importers/base.py` skipped records left `next_id` unincremented. **Shipped.**
- **M4.** `populate_label_embeddings` cache not invalidated when a `region_box` is removed. **Shipped.**
- **M5.** `resolve_current_dataset_cid` could return a colliding-MD5 cid — **closed, not a bug** (`collapse_duplicates` guarantees unique-MD5 lookup).
- **M6.** `restore_labels_from_detector` MD5-only on second pass — **closed, not a bug** (pass 1 consults `md5_lookup`).
- **M7.** `safe_thresholds` cached on `DetectorContext`, never refreshed — **partial close** (never serialised); added `invalidate_loaded_detector_models()` on setting change (`test_safe_thresholds.py`).
- **M8 / M12.** Pickle loader treated `embedding: None` as present → object-dtype / null-deref — `_convert_one_pickle_media` skips missing/None uniformly. **Shipped.**
- **M9.** `loader_folder._has_override` didn't warn on conflicting override embeddings. **Shipped.**
- **M10.** `clipper_chain._run_clipper_step` assumed deterministic output count — stamps `n_out`/`clip_index`/`content_hash`, prefers content match. **Shipped.**
- **M11.** Stale media in `cli._score_medias_with_detectors` with `None` embeddings — matrix builder raises `ValueError`; `_drop_none_embeddings_stage`; `zip(strict=True)` on every id↔score callsite (`test_embedding_matrix.py`, `test_load_stage_matrix_cache.py`, `test_export_options.py`).
- **M13.** `learned_scores` could serialize as JSON `NaN`/`Infinity` — routed through `sigmoid_to_finite_scores` (`-1.0` sentinel) + `GET /api/votes` guard (`test_api_contracts.py`).
- **M14.** `diversity_tree_next_sample` referenced stale media IDs after `/api/dataset/clear` — **closed** (tree is `None`); fixed two adjacent tree-reset bugs via `reset_seen()` / `resync_diversity_tree_to_detector`.
- **M15.** Pending labelset sync stored `dataset_ctx = None` → `AttributeError` — **closed, not a bug** (contexts invariantly non-None via sentinels).
- **M16.** Sync-on-first-read could return stale local config — replaced `_synced_users` set with per-user `_UserSyncState` + `peek_version` hook (`test_sync_sources.py::TestSyncFromSourceFreshness`).
- **M17.** Legacy migration popped cache before disk write → divergence — **partially closed** (ordering was inverted); fixed the narrower server-file rewrite path (`test_per_user_settings.py`).
- **M18.** `_PeekUnpickler` missing opcode overrides — added `FLOAT` / `BYTEARRAY8` handlers; staging surfaces peek `error`. **Shipped.**
- **M19.** `embed_text_enriched` crashed (`np.mean` on empty) — **closed, not a bug** (guard present since PR #334).
- **M20.** XCLIP single-frame video degenerate embedding — **closed, not a bug** (padding is correct); fixed a real adjacent silent partial-read via a warning in `sample_video_frames`.
- **M21.** Empty paragraph clip survived with `None` embedding — `_clip_content_bytes` treats blank text as no content (`test_clip_reembed_bulk.py::TestBlankTextClipNotEmbedded`). **Shipped.**
- **M22.** `set_thread_user()` cleanup relied on caller `finally` → pool identity leak — added `thread_user` / `thread_dataset_context` / `thread_detector_context` context managers (`test_thread_context_scopes.py`).
- **M23.** `setup_logging` re-run could leave duplicate handlers — **closed, not a bug** (root cleared; named libs only `.setLevel`).
- **M24.** `CoreConfig.from_settings()` raised if a blueprint ran before the shim registered the builder — shim hooks now install before any route import.
- **M25.** `LeftPanelComponent` lacked `OnDestroy`/`takeUntil` — style alignment, not a real leak; added the pattern anyway.
- **M26.** `labelset-state.service.startPolling()` not tied to `destroy$` — **false positive** (singleton + idempotent guard); `destroy$` was dead code.
- **M27.** `progress-events.service` didn't reconcile stale `task_id`s after backend restart — SSE `boot_id` frame + `serverReset$`. **Shipped.**
- **M28.** Audio waveform fetch's `catch {}` swallowed errors — `AbortController` + `response.ok` + `console.warn`. **Shipped.**
- **M29.** `AudioContext` not cleaned up on rapid navigation — decode via throwaway `OfflineAudioContext` (`utils/decode-audio.ts`). **Shipped.**
- **M30.** Autopilot phase transitions could oscillate — **won't fix** (instantaneous derivation is intentional; oscillation is rate-limited and self-settling).
- **M31.** `settings-importer-modal` auto-closes after 1.5s — **resolved** (fires only on success); cleared the zombie-`close()` timer across all four auto-closing modals.
- **M32.** `sanitize_template_value` allowed all-dots tokens — collapses `.`/`..`/`...`/… to `_` (`test_template_sanitization.py`). **Shipped.**
- **M33.** `rglob_follow_symlinks` didn't detect cycles → DoS — tracks `(st_dev, st_ino)` and prunes visited dirs (`test_importer_symlinks.py`). **Shipped.**
- **M34.** Email exporter validated "@" only — pragmatic regex before MX lookup (`test_exporters.py::TestEmailLabelsetExporter`). **Shipped.**
- **M35.** `voting_iterations.py` reported metrics at 1-vs-1 as reliable — each row carries `n_good`/`n_bad` (`test_eval_voting_iterations.py`). **Shipped.**

### Low — latent / cosmetic / hypothetical (triaged 2026-06-24)

- **L1.** `_run_pipeline` returns `None` silently in text mode — **closed** (every branch emits a terminal signal).
- **L2.** Pipeline-vs-server fall-through in `app.py` — **closed** (hypothetical; stale ref, now `cli_main.py`).
- **L3.** `getViewMode()` hard-coded fallbacks — **closed, by design** (matches server defaults; benign transient).
- **L4.** `get_settings_source_config` reads cache without re-checking sync — **closed, by design** (getter makes no freshness claim).
- **L5.** `clip_box` CSV join without quote-protection — **closed** (written through `csv.writer.writerow`).
- **L6.** Empty `Origin.params` not dropped consistently — **closed** (round-trips identically; aliasing handled where it matters).
- **L7.** `eval/metrics.py` returned F1=0 for empty test set silently — logs a warning on `total == 0` (`test_eval.py::TestBinaryClassification`). **Shipped.**
- **L8.** Vote-event logging truncates achievement traceback — **closed, stale** (logs via `logger.exception`; the "truncation" is unrelated vote-step history).
- **L9.** `get_param` returns `""` for unknown keys silently — now logs a warning naming key + converter (`test_converter_selection.py`). **Shipped.**

### Late follow-ups (shipped 2026-06-25)

- **M18 pickle-peek hardening** — `_codecs.encode` allowlisted; `_PeekUnpickler` size-bounds `BINUNICODE`/`BINUNICODE8` (`_PEEK_MAX_INLINE_STR = 4096`).
- **H1 labelset-element vote endpoint** — `POST /api/detectors/<name>/labels/<element_id>/vote` takes an absolute `target` instead of a toggle; `apply_element_vote_in_data` is idempotent, closing the same inflation race as the media-vote H1 fix.

---

## Root-cause patterns (reference)

The findings collapsed into a small number of root causes; addressing each in
one PR fixed many findings at once. All are resolved except Pattern #4 (see Open
follow-ups).

1. **Background-thread context propagation** — every `Thread(target=…)` sets
   user/dataset/detector thread-locals in `try/finally` (helper:
   `run_with_context`). (C1–C3.)
2. **Template-path substitution → re-validate** — every source/exporter
   interpolating `{username}`/`{detector_id}`/… calls `validate_server_filepath`
   on the resolved path. (C9, H23.)
3. **Achievement recording at one site** — `record_vote()` moved into
   `apply_label_with_click_time`, inside the state lock. (C8.)
4. **Embedding-matrix cache invalidation** — a `media_revision` counter bumped
   on every `medias` mutation. **(Open — see follow-ups.)** (C4.)
5. **Embedder identity is first-class** — track the detector's training embedder
   separately; every train/score path compares and clears caches on mismatch. (H5, H22.)
6. **`X-Dataset-Id` / `X-Detector-Id` required, not silently defaulted** — 400
   when missing/unloaded on mutating routes. (C5, H13, H34.)
7. **Frontend cache keys dataset-qualified** — `${datasetId}:${mediaId}`. (C10.)
8. **Vote endpoints transactional** — apply/retrain/sync all-succeed-or-rollback,
   failures surfaced. (C11, H30, H31.)
9. **Background jobs need a clear error state** — every load/embed/train failure
   calls `update_progress(task_id, error=…)`.

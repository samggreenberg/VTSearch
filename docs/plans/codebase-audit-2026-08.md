# Codebase audit — August 2026

**Background.** A full-codebase defect audit was run at `00664df5`; its confirmed
bugs were filed as issues and have all shipped. What remains are the improvement
proposals it surfaced, kept here rather than as issues because each is a judgement
call about direction rather than a defect with a right answer. Promote one to an
issue (and delete its item here) when it becomes concrete enough to ship on its
own; delete it when it ships or is rejected.

---

## Improvement proposals

### Flask API layer

<!-- item-sep -->

- **Detector listing re-reads full detector JSON files from disk on every request for legacy entries** — `vtsearch/routes/detectors/registry.py` (medium impact)

  GET /api/detectors/registry backfills two fields for entries that predate them, and does so on every single request: `list_registry` calls `_read_detector(_detector_path(name))` whenever `entry.get('embedder_type')` is falsy, reads the same file a second time whenever 'examples' is missing, and a third time for the `test_verdict` of an autofind entry. `_read_detector` parses the whole detector JSON — including a labelset that can hold thousands of label dicts — so a dashboard that polls the registry pays O(legacy_detectors × labelset_size) JSON parsing per poll, twice per entry. Neither computed value is written back via update_detector, so the cost never amortizes. Benefit: one-time lazy migration turns a recurring disk+parse cost into a single write.

  *Direction:* Read the file once per entry, derive both embedder_type and examples from that single read, and persist them back with update_detector(did, embedder_type=..., examples=...) so subsequent listings hit only the registry entry.

<!-- item-sep -->

- **Readers-endpoint maps error to status by substring-matching the message** — `vtsearch/routes/detectors/registry.py` (low impact)

  update_detector_readers (and its dataset twin at routes/datasets/registry.py) chooses the HTTP status with `status = 403 if "creator" in err else 404` — coupling status-code semantics to the wording of set_detector_readers' human-readable error string. Rewording the message (e.g. 'Only the owner can modify readers') would silently turn permission failures into 404s. The registry helper already distinguishes the two cases internally (registry.py); it just flattens them into prose.

  *Direction:* Have set_detector_readers/set_readers return a typed error (enum or exception class) and map it explicitly to 403/404 in the routes.

<!-- item-sep -->

- **Two parallel detector APIs (file-based /api/detectors vs registry) with divergent integrity rules** — `vtsearch/routes/detectors/crud.py` (medium impact)

  The codebase exposes two overlapping detector CRUD surfaces operating on the same on-disk files: the file-keyed /api/detectors family (crud.py: list by directory scan, create/delete/rename by name) and the id-keyed /api/detectors/registry family (registry.py: ACLs, loaded flags, owner checks). They enforce different invariants — crud.py checks name collisions on create/rename but knows nothing about ACLs or the registry's loaded-ids, while the registry routes enforce ownership but skip collision checks; crud's DELETE /api/detectors/<name> unlinks the file while leaving any registry entry pointing at nothing, and crud's combine writes a detector file that never gets a registry entry. Any invariant fixed in one family has to be re-fixed in the other. Consolidating on the registry family (with the file store as its private persistence) — or routing crud handlers through shared helpers that own collision/ACL/registry consistency — would eliminate this class of drift; backwards-compat breaks are acceptable per repo policy.

  *Direction:* Fold the /api/detectors file-keyed routes into the registry blueprint (or shared service functions) so name-collision, ownership, registry-entry, and file lifecycle are enforced in exactly one place.

<!-- item-sep -->

### App core (settings, auth, CLI)

<!-- item-sep -->

- **Achievements persist a full settings-file RMW plus a source push on every single vote** — `vtsearch/achievements.py` (medium impact)

  `record_vote` runs `mutate_user(...)` per vote (achievements.py), and `mutate_user` is heavyweight by design: cross-process flock on the user settings file, fresh `_load_path` re-read, full-dict `json.dumps` + `fsync` + rename (`_atomic_write`, settings_store.py), then a dirty-marking pass over *every* exportable key and a `_sync_to_source` push (settings.py) — which for a configured source means a second full serialize/write (plus a `peek_version` stat and a `.syncmark` write). A user hand-labeling in the Train flow votes multiple times per second, so each click costs 3-4 fsync'd file writes and lock round-trips on the request path, and the achievement counters share a file (and its lock) with every other settings read/write, amplifying contention on the settings lock. The `days_seen`/`docs_read_ids`/`trained_detector_ids` lists also use O(n) `in` checks on every vote (achievements.py), which grows linearly with days active. Benefit: batching (e.g. accumulate credits in-process and flush on a short timer or every N votes, as the counters are approximate milestones anyway) or moving `achievement_state` to its own small file outside the settings-sync machinery would remove the per-click fsync+push cost and cut settings-lock traffic substantially.

  *Direction:* Accumulate vote credits in a process-local per-user buffer and flush to disk on a debounce (e.g. 5s or on get_full_state), and/or store achievement_state in a dedicated per-user file excluded from settings-source export.

<!-- item-sep -->

### State & concurrency

<!-- item-sep -->

- **Promoted job whose dataset/detector was unloaded silently runs against the empty context and caches a bogus 'done' result** — `vtscore/concurrency/async_jobs.py` (medium impact)

  JobManager._run resolves `ds_ctx = get_context(job.dataset_id)` / `get_detector_context(job.detector_id)` at spawn time and, when either returns None (dataset/detector unloaded while the job sat in the pending slot), simply skips binding the thread-local (lines 332-335). The target then resolves the global _empty_dataset_context/_empty_detector_context, computes over zero medias/votes, and completes with status "done" — which _run_inner caches in _last_done keyed by the job's signature, so a subsequent start() with the same signature (same dataset id!) can short-circuit to the empty result via cached_for(). The job's pollers see success with an empty/garbage payload instead of a clear failure.

  *Direction:* In _run, when job.dataset_id is non-empty but get_context returns None (or likewise for detector_id), mark the job status="error" with a "dataset/detector no longer loaded" message, set done_event, promote pending, and return without invoking the target.

<!-- item-sep -->

- **build_coverage_atlas holds the global _state_lock across the entire hierarchical k-means fit** — `vtscore/state/coverage.py` (medium impact)

  build_coverage_atlas() wraps the full CoverageAtlas(...) construction — hierarchical k-means over up to 50k vectors, which the module's own comment describes as costing minutes at scale — inside `with _state_lock:` (coverage.py). _state_lock serializes essentially every endpoint (votes, media reads, before_request state sync), so any caller of this exported public API (it is re-exported from both vtscore.state and vtsearch.state) freezes the whole app for the duration. Production routes currently dodge this by using build_coverage_atlas_for_context (lock-free), so today only tests call the locked variant — but the exported function is a loaded gun for the next plugin/route author, and the codebase already has the right pattern (get_embedding_matrix, cached_media_lookups: snapshot under lock, build unlocked, store under lock with a revision re-check).

  *Direction:* Restructure build_coverage_atlas to snapshot the medias + media_revision under _state_lock, build the atlas unlocked, then re-acquire the lock, verify the revision (and active context identity) still match, and only then assign ctx.coverage_atlas + resync votes; or delete the locked variant and route all callers through build_coverage_atlas_for_context plus an explicit locked resync.

<!-- item-sep -->

- **SortResultsCache bounds entry count but not bytes: 8 cached rankings of a 1M-item dataset pin gigabytes** — `vtscore/state/sort_results_cache.py` (low impact)

  The cache exists for 100k-1M item datasets (its own docstring), keeps up to max_entries=8 full result lists, and each list is a Python list of per-row dicts ({"id", "score"} or {"id", "similarity", "best_region"}) held by reference. At ~150-250 bytes per small dict, 8 rankings x 1M rows is roughly 1-2 GB of steady-state heap from a user simply re-sorting a large dataset a few times (each re-sort mints a new token, so distinct entries accumulate up to the cap even for the same dataset). The LRU bound protects against unbounded growth but not against exactly the large-N case the cache was built for.

  *Direction:* Bound the cache by total rows (e.g. evict oldest until sum(len(results)) <= ~2M) in store(), and/or store rankings columnar (an int64 id array + float32 score array, materializing row dicts only in page()) which cuts memory ~20x and also makes the stored list immune to caller mutation.

<!-- item-sep -->

### Media & embedding

<!-- item-sep -->

- **Whisper and PaddleOCR converters reload their model for every single media item** — `vtscore/converters/audio2text.py` (medium impact)

  `Audio2TextMediaConverter.convert` calls `whisper.load_model(model_size, ...)` inside convert(), i.e. once per audio file — the runner invokes convert per source file (vtscore/converters/runner.py), so transcribing a folder of 500 clips loads the ~145 MB (base) to ~3 GB (large) checkpoint from disk into RAM/VRAM 500 times, dominating wall-clock cost. `Image2TextMediaConverter` has the same shape: `_make_paddleocr` (vtscore/converters/image2text.py, 57-75) constructs a fresh PaddleOCR engine (det+rec+cls model loads) per image. The codebase already has the right pattern in the same package: `Image2FaceMediaConverter` caches its MTCNN detector on the instance across a conversion run (vtscore/converters/image2face.py, explicitly documented as 'reused across every image in a conversion run'). Caching keyed by the params that affect the model (model_size / language) would make both converters usable on realistic corpus sizes.

  *Direction:* Cache the loaded model on the converter instance keyed by (model_size,) / (language,) — mirroring Image2FaceMediaConverter._make_detector — and reuse it across convert() calls.

<!-- item-sep -->

- **Audio import decodes every file twice and ignores the bytes it was handed** — `vtscore/media/audio/media_type.py` (medium impact)

  `AudioMediaType.load_media_data` accepts `media_bytes` precisely so callers that already read the file avoid a second read (per the base-class contract, vtscore/media/base.py), but then (a) fully decodes the audio from `file_path` anyway just to compute duration (`decode_audio(str(file_path), sr=None, mono=True)` — a complete PCM decode of, say, an hour-long podcast merely for its length), and (b) decodes the same payload a second time inside `generate_waveform_thumbnail(media_bytes)` (line 1053 → decode at line 169). So every imported audio file is decoded end-to-end twice per import, and the provided bytes are never used for the duration path. Duration is available near-free from the container header via `soundfile.SoundFile(...).frames / samplerate` (with an ffmpeg probe fallback for AAC/M4A), and even without that, decoding once and reusing the array for both duration and the waveform render halves import decode cost.

  *Direction:* Decode once from `media_bytes` (already in hand) and derive both duration and the waveform thumbnail from that single `(samples, sr)`; or read duration from `sf.info`-style header metadata instead of a full decode.

<!-- item-sep -->

- **Audio media_response always claims audio/wav and a .wav filename regardless of actual codec** — `vtscore/media/audio/media_type.py` (low impact)

  `AudioMediaType.media_response` serves whatever `_resolve_media_bytes` returns — which for MP3/FLAC/OGG/M4A imports (all in `file_extensions`, line 328) is the original container bytes — but hard-codes `mimetype="audio/wav"` and `download_name=f"media_{id}.wav"`. Browsers usually sniff their way through playback, but a user downloading the clip gets an `.wav` file containing MP3/M4A bytes (which some players refuse by extension), and strict clients/proxies that trust Content-Type can mis-handle the stream. The bytes' real container is cheaply detectable from magic bytes (RIFF/ID3-or-0xFFEx/fLaC/OggS/ftyp).

  *Direction:* Sniff the first bytes of the payload (RIFF→wav, fLaC→flac, OggS→ogg, ID3/0xFFEx→mp3, ....ftyp→mp4/m4a) and set mimetype + download extension accordingly, defaulting to audio/wav.

<!-- item-sep -->

### Datasets & IO

<!-- item-sep -->

- **sync_from_labelset_source applies labels with an O(labels × medias) scan** — `vtscore/labels/sync.py` (low impact)

  For every imported label entry, the apply loop does a linear scan `for mid, media in ds_medias.items(): if media.get("md5") == md5` — O(L × N) while holding `_sync_lock` (which blocks every concurrent debounced push). With a 100k-item dataset and a few thousand imported labels this is hundreds of millions of dict lookups on the sync path that runs at detector load. Concrete benefit: building a one-pass `{md5: mid}` index before the loop makes the apply O(L + N) and shrinks the window during which `_sync_lock` starves `_push_to_labelset_source`.

  *Direction:* Before the loop: `md5_to_mid = {m.get("md5"): mid for mid, m in ds_medias.items()}` (first-wins to preserve current semantics), then `mid = md5_to_mid.get(md5)` per entry.

<!-- item-sep -->

### Eval harness

<!-- item-sep -->

- **check-eval-app-sync's Python normalizer erases the `(x,)` vs `(x)` difference** — `scripts/check-eval-app-sync.py` (low impact)

  `_normalize_python` drops every comma that precedes a closing bracket so that `ruff format` re-wrapping a call is not read as a logic change. That also erases the semantic difference between a one-element tuple `(x,)` and a parenthesised expression `(x)`, so that one real logic change cannot trip a pin.

  *Direction:* Keep the trailing comma when the bracket pair holds a single element (or only strip inside call/collection contexts with >1 element).

<!-- item-sep -->

- **eval_learned_sort thresholds on a haystack that includes the held-out test set** — `vtscore/eval/runner.py` (low impact)

  `eval_learned_sort` calls `train_and_score(medias, ...)` over the FULL media dict, so the safe-threshold population estimator (fold-anchored GMM / blend) is fitted on a score distribution that includes the held-out test items, and the resulting threshold is then evaluated on those same items. `simulate_voting_iterations` explicitly refuses this (voting_iterations.py: "Restrict to the simulation set so the held-out test_ids never feed into the GMM ... otherwise the test scores leak into calibration and the reported metrics are biased upward"), so the two eval components apply inconsistent leakage policy: the runner's accuracy/precision/recall/F1 numbers are mildly transductive while the voting-iterations numbers are not, making them non-comparable. (Unsupervised leakage — no test labels are read — but the same kind voting_iterations guards against.)

  *Direction:* Restrict the snapshot passed to `train_and_score` to the train split (train_good + train_bad + train-side unlabeled), then score the test items separately with the returned model.

<!-- item-sep -->

- **Bad-phase text-sort cutoff and rank geometry include held-out test items** — `vtscore/eval/al_strategies.py` (low impact)

  `build_seed_scores` scores EVERY media id in the dataset (seed_scores.py), and `_pick_bad_phase` uses that map directly as the ranking: `_sort_threshold(ranking)` fits the text-sort GMM cutoff over the full dataset's cosine distribution (sim + test), and `_hard_pick_by_index` measures rank distance through interleaved test items that can never be picked. The simulated user's universe is supposed to be D_sim (the app's counterpart is the loaded dataset, which for the simulation is the sim half). The example-sort fallback is inconsistent with it too: `_centroid_similarities(ctx, list(ctx.embeddings))` ranks over sim embeddings only, so the two seeding modes run over different universes. No label leakage (cosines are query-based), but the bad-phase cutoff position and the resulting Bad votes depend on the test half, which shifts trajectories relative to a run at a different sim_fraction and blurs the sim/test separation the harness otherwise maintains.

  *Direction:* In `simulate_voting_iterations`, filter `seed_scores` to `sim_ids` before constructing the ALContext, so the text ranking, its GMM cutoff, and the rank distances all live in the simulated user's visible universe.

<!-- item-sep -->

### Projection, plugins & security utils

<!-- item-sep -->

- **Hilbert ordering recomputed per level in _level_membership despite being level-independent** — `vtscore/projection/pyramid.py` (low impact)

  `_level_membership` calls `perm = _hilbert_order(coords)` inside the per-level cache-miss path. The Hilbert permutation depends only on the frozen coords — it is identical for every level — yet each level's first tile fetch pays a fresh O(N) quantize + 16-iteration bit-twiddle + O(N log N) stable argsort. On a large dataset with a deep pyramid (up to 14 levels), the browse canvas re-derives the exact same permutation up to 14 times as the user zooms through levels, each time on the request thread serving the first tile of that level.

  *Direction:* Memoize the permutation once per Pyramid (e.g. a `_hilbert_perm` field alongside `_member_index`, or key the member-index cache computation to compute the perm once and reuse across levels).

<!-- item-sep -->

### Frontend — dashboard & modals

<!-- item-sep -->

- **Three near-identical dynamic plugin-field form engines should collapse into one shared component** — `frontend/src/app/components/modals/label-importer-modal/label-importer-modal.component.ts` (medium impact)

  The plugin-field form machinery — default seeding (`field.default`, first static option for strict selects), dynamic-options fetching with per-key loading/error maps, `depends_on` cascades, free-text datalist vs strict select rendering, file-field capture, and the full template branch ladder for server_path/file/password/email/url/select/number/text — is implemented three times with only cosmetic differences: label-importer-modal.component.ts (~125–225 + template), new-detector-modal.component.ts (trained tab, ~992–1097 + ~130 template lines with `nmm-` id prefixes), and plugin-import-form.component.ts (~59–170). new-detector-modal's own comment admits it exists "mirroring label-importer-modal ... with full parity". Divergence has already crept in (plugin-import-form validates required fields in `canSubmit`; the trained tab does not, so a missing required file only fails server-side), though the stale-response guard for dynamic options is already shared (`frontend/src/app/utils/dynamic-field-options.ts`). A single `vt-plugin-fields-form` component taking `fields` + an options-fetch fn and emitting `{values, file, fileFieldKey}` would delete roughly 600 lines and make future field types land once.

  *Direction:* Extract a shared standalone component (fields input, getFieldOptions fn input, values/file outputs); adopt it in all three call sites. Backwards-compat breakage is acceptable per repo policy.

<!-- item-sep -->

- **Add a lint/audit gate for the zoneless anti-pattern: plain template-bound fields mutated in async callbacks** — `frontend/src/app/components/folder-browser/folder-browser.component.ts` (medium impact)

  This audit found five components in one area (folder-browser, login, progress-modal, combine-detectors-modal, settings-modal's exporter flag) that missed the zoneless migration's signalization pass, each producing invisible-until-next-click UI. The codebase clearly knows the rule — dozens of fields carry "signalized so the unpatched HTTP callbacks schedule CD under zoneless" comments, and local-folder/server-folder pickers even document the ancestor-marking subtleties around `markForCheck()` — but nothing enforces it, and component specs mask it by feeding synchronous `of(...)` observables so subscribe callbacks run inside an existing CD pass. Concrete benefit: a mechanical gate would have caught all five bugs. Two practical options: (a) a lint rule (the frontend has no ESLint setup today, so this means adding one — a typescript-eslint custom rule or `no-restricted-syntax` approximation) flagging `this.<identifier> =` assignments inside `.subscribe(...)` callbacks in `@Component` classes unless the property is a signal; (b) a test-infra convention requiring async fakes (`delay(0)` / Subjects) in specs that assert rendered output, which makes the missing repaint fail in Vitest.

  *Direction:* Add a frontend lint step carrying the rule, wire it into `run-tests.sh` as a gate, and sweep remaining plain template-bound fields to signals.

<!-- item-sep -->

- **Bulk delete fires N parallel requests each triggering its own registry refresh** — `frontend/src/app/components/dashboard/dashboard.component.ts` (low impact)

  `deleteSelectedDatasets()` and `deleteSelectedDetectors()` loop over targets, issuing one DELETE per item and calling `this.datasetState.refresh()` in every `next` callback — deleting 20 datasets produces 20 registry refetches racing each other, and intermediate refreshes can repopulate the table mid-delete (rows flicker back before their own DELETE lands). Partial failures are also silent: individual errors are swallowed ("Global error interceptor surfaces the failure in the banner") with no summary of which items survived. Benefit: `forkJoin` over the delete observables with a single `refresh()` (and one toast noting any failures) removes the refetch storm and the flicker, and gives the user an accurate outcome for bulk operations.

  *Direction:* Wrap the per-item deletes in `forkJoin([...ids.map(id => api.delete(id).pipe(catchError(err => of({id, err}))))])`, then do one selection prune + one `datasetState.refresh()` + one summary toast.

<!-- item-sep -->

### Frontend — services & views

<!-- item-sep -->

- **LabelsetStateService lacks the out-of-order read guard that VoteStateService has** — `frontend/src/app/services/labelset-state.service.ts` (medium impact)

  The labelset piles are fed by two independent readers of GET labels-detail: the adaptive poll (startPolling) and the on-vote refresh() called from vote()'s POST continuation and from label-view.onMediaVoted() (label-view.component.ts line 1061). adaptivePoll never overlaps its own GETs, but a poll GET issued just before a vote commits can resolve AFTER the post-vote refresh(), reverting the just-moved element to its old pile until the next poll tick (~1.5s flicker, longer once the poll has backed off to its 10s heartbeat). This is exactly the staleness class VoteStateService closed with its votesSeq / lastAppliedVotesSeq issue-order guard (vote-state.service.ts lines 74-87 documents the identical bug for /api/votes); the labelset store predates that fix and never got it. The optimistic applyOptimisticState here also has no pending-entry reconciliation, so the stale overwrite is not corrected until the next full read.

  *Direction:* Stamp every labels-detail read (poll and refresh) with a monotonic issue id and drop responses older than the newest applied, mirroring VoteStateService.applyVotesFresh; optionally keep per-element pending entries until the server response agrees, as vote-state does for votes and region boxes.

<!-- item-sep -->

- **Find-view duplicates label-view's panel drag/snap machinery by hand** — `frontend/src/app/components/find-view/find-view.component.ts` (medium impact)

  Label-view uses PanelResizeDirective for its divider drags and has icon-size auto-pop plus snap-on-load; find-view still carries its own divider-drag and grid-snap code and neither of those behaviours. Every future panel fix has to be made twice, and because the two views share the same settings keys the drift is user-visible — a width saved and snapped in Label restores un-snapped in Find.

  *Direction:* Reuse PanelResizeDirective for find-view's dividers and lift the auto-pop / snap-on-load behaviour alongside it, deleting the duplicated drag code.

  *Background:* the settings half of this item is done — both views now resolve their per-media-type panel preferences through `SettingsStateService.perMediaType` (#3447), so the shadow dicts and the mirror effects are gone from find-view. What remains is the drag/snap duplication.

<!-- item-sep -->

- **RunningJobsService clears all busy-pair spinners on any transient poll failure** — `frontend/src/app/services/running-jobs.service.ts` (low impact)

  The catchError in the /api/jobs/active poll emits `{busy_pairs: []}` on any error or 10s timeout, which wipes the busyPairs map. A single dropped request or slow response while jobs are genuinely running makes every pulldown spinner vanish for one 5s poll interval and then reappear — flickering UI that misreports 'no jobs running' during exactly the load conditions (heavy training) when jobs ARE running and the backend is slow to answer. Since the map is advisory UI state, holding the last known value through a transient failure is strictly better than clearing it; genuinely finished jobs are corrected on the next successful poll anyway.

  *Direction:* On error, re-emit the current busyPairsSubject.value (or emit nothing via EMPTY) instead of an empty payload, reserving the clear for stopPolling().

<!-- item-sep -->

### Frontend — styles & templates

<!-- item-sep -->

- **Global `.importer-picker`/`.exporter-picker` grid rule is overridden by every single consumer** — `frontend/src/scss/_components.scss` (medium impact)

  _components.scss defines `.importer-picker, .exporter-picker { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--space-lg); }`, and its comment block (lines 632-635) says the settings/label pickers use this responsive grid. In reality all four consumers of these classes (settings-importer-modal.component.scss, settings-exporter-modal.component.scss, label-importer-modal.component.scss, dataset-importer-modal's source-picker.component.scss) redeclare the class locally as `display: flex; flex-direction: column` — and component-scoped rules win via Angular's encapsulation specificity bump — so the documented grid never renders anywhere. The next modal that applies the shared class trusting _components.scss (or style-guide §2.5) will get a grid layout that matches no existing picker. The shared rule's body has drifted into fiction.

  *Direction:* Make the global rule match universal usage (flex column with the agreed gap), delete the four local redeclarations, and fix the misleading comment. Any picker that genuinely wants a grid can opt in with a modifier class.

<!-- item-sep -->

- **`.form-actions` rule duplicated verbatim in six component SCSS files** — `frontend/src/app/components/modals/settings-importer-modal/settings-importer-modal.component.scss` (medium impact)

  The identical rule `display: flex; justify-content: flex-end; gap: var(--space-md); margin-top: var(--space-md)` is declared in settings-importer-modal:24, settings-exporter-modal:24, label-importer-modal:29, clipper-chooser:78, dataset-importer-modal:22, and (minus margin-top) combine-datasets-modal:192 — plus a seventh dead copy in export-modal:316. Style-guide §6 says to promote a pattern on its third copy; this one has seven. Worse, every one of these `.form-actions` divs sits in the `[modal-footer]` slot, where the global `.modal-footer` (_components.scss) already provides `display:flex; justify-content:flex-end; gap:var(--space-md)` — so the duplicated body is ~redundant with the slot it lives in, and any future tune of footer spacing will drift across seven files.

  *Direction:* Either drop the `.form-actions` wrapper divs entirely (project the buttons directly into `[modal-footer]`, which already lays them out) or promote one `.form-actions` rule to _components.scss and delete all local copies.

<!-- item-sep -->

- **Toast/list button labels break Title Case** — `frontend/src/app/components/toast-container/toast-container.component.html` (low impact)

  In the toast template, the action buttons read "Dismiss all" (line 12), "Hide details"/"Details" (line 62), and "Copy debug info" (line 65); §4.1 puts button labels in the Title Case bucket ("Dismiss All", "Hide Details", "Copy Debug Info"). Same lapse in media-list.component.html ("Load more" → "Load More"). These are hand-review-only rules per the guide, which is exactly why an audit should catch them.

  *Direction:* Title-Case the four button labels.

<!-- item-sep -->

- **Shared `_picker-shared.scss` itself violates the single-disabled-opacity rule** — `frontend/src/scss/_picker-shared.scss` (low impact)

  `.demo-table tbody tr.disabled { opacity: 0.55 }` hand-picks a disabled opacity in a *shared* stylesheet, contradicting §1.9 ("There is only one disabled opacity" — `--opacity-disabled: 0.5` — for `.disabled` states, added precisely because a rendered audit found seven ad-hoc values, 0.55 among them). Since this file is one of the guide's four canonical references, an off-token value here gets copied as precedent.

  *Direction:* Change to `opacity: var(--opacity-disabled);`.

<!-- item-sep -->

- **Residual token violations accumulate because the style audit is not gated** — `frontend/src/app/components/modals/settings-modal/auto-find/auto-find-settings.component.scss` (medium impact)

  The repo's own `.claude/scripts/style-check.py` currently reports 16 un-annotated raw px/rem spacing/font hits, 9 raw z-indexes, 5 raw `opacity: 0.7`s, and 2 heading restyles — i.e. the guide is drifting despite the tooling existing. Concrete examples verified by reading the code: this file's `gap: 0.35rem` (the token `--space-sm` is 0.375rem) and `opacity: 0.7` on `.help-icon` at rest (line 20 — the exact "dimmed icon" case §1.10 created `--opacity-dim` for; center-panel.component.scss and autopilot-panel.component.scss have the same raw 0.7); `gap: 2px` in browse-legend.component.scss and label-list.component.scss (`--space-2xs`); and label-list.component.scss restyling `h3` to `--font-sm`/regular weight — anti-pattern 7, where the element is really a small section label, not an h3. Unlike ruff/codespell, this checker runs only when someone remembers `/style-check`, so violations land silently in PRs whose authors never open the guide.

  *Direction:* Sweep the current curated hits (most are one-line token swaps; the intentional off-scale ones already carry `// kept exact` annotations the script honors), then wire `style-check.py` into run-tests.sh as a gate the way the screenshots wiring-check already is, with an annotation/whitelist mechanism for the deliberate exceptions.

<!-- item-sep -->

### Security

<!-- item-sep -->

- **Hardcoded default Flask secret_key with no startup guard for multi-user deployments** — `app.py` (low impact)

  `app.secret_key` falls back to the literal `"vtsearch-dev-key-change-in-production"` when `VTSEARCH_SECRET_KEY` is unset (app.py), and nothing refuses to start (or warns loudly) when a non-DefaultLoginProvider is active with this public key. Signed session cookies (used by `TrivialLoginProvider` for `vtsearch_user`, and by the HF OAuth handshake for state/PKCE) are then forgeable by anyone who knows the shipped default. The practical blast radius today is limited because the only session-cookie-based provider, `TrivialLoginProvider`, is passwordless-by-design (an attacker can just POST /api/auth/login with any username), but the moment any real session-backed provider is added, or the HF state cookie is relied upon, a forgeable key becomes a live impersonation/CSRF-token-forgery vector. There is no defense-in-depth check tying 'multi-user provider selected' to 'a real secret key must be present'.

  *Direction:* At startup, if the active login provider is not DefaultLoginProvider and `VTSEARCH_SECRET_KEY` is unset (secret_key == the default), refuse to boot (or generate a random per-process key and log a prominent warning that sessions won't survive restart). Never ship a usable default secret in a mode that trusts signed cookies.

<!-- item-sep -->

### Tests & tooling

<!-- item-sep -->

- **Cross-dataset clip re-embedding tests assert nothing — the documented behavior can never fail** — `tests/detectors/test_clipper_workflow.py` (medium impact)

  All four tests in TestCrossDatasetClipEmbedding (lines 482-559) call `_apply_clip_and_embed(...)` and assert nothing — comments say "Result may be None if no embedder is loaded, which is fine". Their docstrings claim they verify that clip params cause slicing/cropping/sentence-extraction before embedding, but a regression that ignores clip params entirely (or returns None always) passes all four. This is the load-bearing path of the repo's core invariant ("origins are canonical; the system rederives origin → file → embedding on demand") for clipped labels, so it is effectively untested. Notably the excuse is stale: the session-scoped `_stub_embedding_models` fixture (tests/conftest.py) patches every embedder's `embed_media` with a deterministic content-seeded fake, so a real assertion is easy — the clipped call should return a non-None unit vector that differs from the unclipped `_apply_clip_and_embed` result on the same file (the fake seeds off file/clip bytes).

  *Direction:* Assert the return is a non-None float32 vector of the right dim, and that clipped vs. no-clip origins produce different vectors (and identical clip params produce identical vectors).

<!-- item-sep -->

- **check-dockerfiles.py misses python invocations on RUN continuation lines and `python3`** — `scripts/check-dockerfiles.py` (low impact)

  The ordering gate skips every physical line that continues a previous instruction (lines 38-41: `if is_continuation: continue`), so `RUN set -e \\\n && python scripts/foo.py` is never inspected — only a `python` on the RUN's first physical line is caught. It also matches only `\bpython\b` (line 62), so `python3 ...` (the actual interpreter name in Dockerfile.gpu, where `/usr/bin/python` is only a symlink created in the same layer) slips through. A future Dockerfile edit that runs Python in a multi-line RUN before vtsearch/+vtscore/ are copied would pass the gate the gate exists to catch. Relatedly, .dockerignore excludes `tests/` but not `tests_lib/`, so the library-tier test tree (with fixtures) is baked into every image for no reason.

  *Direction:* Accumulate logical instructions (join continuation lines before matching), broaden the regex to `python[0-9.]*`, and add tests_lib/ to .dockerignore.

<!-- item-sep -->

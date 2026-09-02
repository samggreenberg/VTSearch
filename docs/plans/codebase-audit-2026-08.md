# Codebase audit — August 2026

**Background.** A full-codebase inspection was run at `00664df5` (dev): fourteen
specialist reviewers over disjoint areas (~118 k lines Python, ~44 k lines
TypeScript), each reported finding then handed to an independent verifier
prompted to *refute* it against the code and defaulting to "not real" when
uncertain. 69 bugs survived that pass; 3 claims were refuted and dropped, and one
more (#2972) was withdrawn after a second suite run disproved it. The
repo's own gates were green throughout (7 860 tests, ruff, format, frontend
build), so everything below came from reading code, not from a broken build.

**What is still owed** is the two lists below: the confirmed defects, which live
as GitHub issues and are referenced here only by number, and the improvement
proposals, which were *not* filed as issues because they are design directions
rather than discrete shippable defects — their bodies live here and nowhere else.
Delete a pointer line when its issue closes; delete a proposal when it ships or is
rejected.

---

## Filed issues

All 69 confirmed defects from the audit have shipped (the closed issues, with
their fixes, live in the tracker — grep `label:claude` closed issues in the
audit's date range if you need the history). What still lives in this file is
the "Improvement proposals" section below.


---

## Improvement proposals (40)

Design and architecture directions surfaced by the same review. These are
deliberately **not** issues: each is a judgement call about direction rather than
a defect with a right answer, and several are alternatives to one another. Promote
one to an issue (and delete its item here) when it becomes concrete enough to
ship on its own.

### Flask API layer

<!-- item-sep -->

- **Global NotFound handler discards every abort(404, message=...) body on /api/ routes** — `vtsearch/errors.py:43` (medium impact)

  _handle_404 renders `error_response(exc.name, 404)`, i.e. always the literal 'Not Found', for every NotFound raised under /api/ — including flask-smorest `abort(404, message=...)` calls, whose message/extra kwargs ride on exc.data and are simply dropped. Routes across the codebase craft specific 404 messages that no client can ever see: `_abort_find(404, f"Dataset file missing for '{name}'")` (routes/detectors/find.py:183), 'Detector not found', 'File not found: <name>' (media/server.py:253), 'Job not found', etc. The learned-sort docstring (routes/sorting.py:486-492) even documents the message loss as a known quirk. The result is that any 404 with actionable detail (which dataset's pkl vanished, which of several filenames was missing) degrades to a generic banner. Benefit: one small change restores meaningful diagnostics to dozens of endpoints without touching them.

  *Direction:* In _handle_404 (and _handle_405), prefer the smorest payload when present: `msg = (getattr(exc, 'data', None) or {}).get('message') or exc.description or exc.name; return error_response(msg, 404)`.

<!-- item-sep -->

- **Detector listing re-reads full detector JSON files from disk on every request for legacy entries** — `vtsearch/routes/detectors/registry.py:127` (medium impact)

  GET /api/detectors/registry backfills two fields for entries that predate them, and does so on every single request: line 127-129 calls `_read_detector(_detector_path(name))` whenever `entry.get('embedder_type')` is falsy, and lines 133-135 read the same file a second time whenever 'examples' is missing. `_read_detector` parses the whole detector JSON — including a labelset that can hold thousands of label dicts — so a dashboard that polls the registry pays O(legacy_detectors × labelset_size) JSON parsing per poll, twice per entry. Neither computed value is written back via update_detector, so the cost never amortizes. Benefit: one-time lazy migration turns a recurring disk+parse cost into a single write.

  *Direction:* Read the file once per entry, derive both embedder_type and examples from that single read, and persist them back with update_detector(did, embedder_type=..., examples=...) so subsequent listings hit only the registry entry.

<!-- item-sep -->

- **Readers-endpoint maps error to status by substring-matching the message** — `vtsearch/routes/detectors/registry.py:1009` (low impact)

  update_detector_readers (and its dataset twin at routes/datasets/registry.py:640) chooses the HTTP status with `status = 403 if "creator" in err else 404` — coupling status-code semantics to the wording of set_detector_readers' human-readable error string. Rewording the message (e.g. 'Only the owner can modify readers') would silently turn permission failures into 404s. The registry helper already distinguishes the two cases internally (registry.py:322-329); it just flattens them into prose.

  *Direction:* Have set_detector_readers/set_readers return a typed error (enum or exception class) and map it explicitly to 403/404 in the routes.

<!-- item-sep -->

- **Two parallel detector APIs (file-based /api/detectors vs registry) with divergent integrity rules** — `vtsearch/routes/detectors/crud.py:104` (medium impact)

  The codebase exposes two overlapping detector CRUD surfaces operating on the same on-disk files: the file-keyed /api/detectors family (crud.py: list by directory scan, create/delete/rename by name) and the id-keyed /api/detectors/registry family (registry.py: ACLs, loaded flags, owner checks). They enforce different invariants — crud.py checks name collisions on create/rename but knows nothing about ACLs or the registry's loaded-ids, while the registry routes enforce ownership but skip collision checks (see the two bugs filed above); crud's DELETE /api/detectors/<name> unlinks the file while leaving any registry entry pointing at nothing, and crud's combine writes a detector file that never gets a registry entry. Any invariant fixed in one family has to be re-fixed in the other. Consolidating on the registry family (with the file store as its private persistence) — or routing crud handlers through shared helpers that own collision/ACL/registry consistency — would eliminate this class of drift; backwards-compat breaks are acceptable per repo policy.

  *Direction:* Fold the /api/detectors file-keyed routes into the registry blueprint (or shared service functions) so name-collision, ownership, registry-entry, and file lifecycle are enforced in exactly one place.

<!-- item-sep -->

### App core (settings, auth, CLI)

<!-- item-sep -->

<!-- item-sep -->

- **Achievements persist a full settings-file RMW plus a source push on every single vote** — `vtsearch/achievements.py:371` (medium impact)

  `record_vote` runs `mutate_user(...)` per vote (achievements.py:371), and `mutate_user` is heavyweight by design: cross-process flock on the user settings file, fresh `_load_path` re-read, full-dict `json.dumps` + `fsync` + rename (`_atomic_write`, settings_store.py:69-89), then a dirty-marking pass over *every* exportable key and a `_sync_to_source` push (settings.py:420-431) — which for a configured source means a second full serialize/write (plus a `peek_version` stat and a `.syncmark` write). A user hand-labeling in the Train flow votes multiple times per second, so each click costs 3-4 fsync'd file writes and lock round-trips on the request path, and the achievement counters share a file (and its lock) with every other settings read/write, amplifying contention on the very lock structure that finding #1 shows is fragile. The `days_seen`/`docs_read_ids`/`trained_detector_ids` lists also use O(n) `in` checks on every vote (achievements.py:389-409), which grows linearly with days active. Benefit: batching (e.g. accumulate credits in-process and flush on a short timer or every N votes, as the counters are approximate milestones anyway) or moving `achievement_state` to its own small file outside the settings-sync machinery would remove the per-click fsync+push cost and cut settings-lock traffic substantially.

  *Direction:* Accumulate vote credits in a process-local per-user buffer and flush to disk on a debounce (e.g. 5s or on get_full_state), and/or store achievement_state in a dedicated per-user file excluded from settings-source export.

<!-- item-sep -->

### State & concurrency

<!-- item-sep -->

- **Promoted job whose dataset/detector was unloaded silently runs against the empty context and caches a bogus 'done' result** — `vtscore/concurrency/async_jobs.py:324` (medium impact)

  JobManager._run resolves `ds_ctx = get_context(job.dataset_id)` / `get_detector_context(job.detector_id)` at spawn time and, when either returns None (dataset/detector unloaded while the job sat in the pending slot), simply skips binding the thread-local (lines 332-335). The target then resolves the global _empty_dataset_context/_empty_detector_context, computes over zero medias/votes, and completes with status "done" — which _run_inner caches in _last_done keyed by the job's signature, so a subsequent start() with the same signature (same dataset id!) can short-circuit to the empty result via cached_for(). The job's pollers see success with an empty/garbage payload instead of a clear failure.

  *Direction:* In _run, when job.dataset_id is non-empty but get_context returns None (or likewise for detector_id), mark the job status="error" with a "dataset/detector no longer loaded" message, set done_event, promote pending, and return without invoking the target.

<!-- item-sep -->

- **build_coverage_atlas holds the global _state_lock across the entire hierarchical k-means fit** — `vtscore/state/coverage.py:87` (medium impact)

  build_coverage_atlas() wraps the full CoverageAtlas(...) construction — hierarchical k-means over up to 50k vectors, which the module's own comment describes as costing minutes at scale — inside `with _state_lock:` (coverage.py:87-109). _state_lock serializes essentially every endpoint (votes, media reads, before_request state sync), so any caller of this exported public API (it is re-exported from both vtscore.state and vtsearch.state) freezes the whole app for the duration. Production routes currently dodge this by using build_coverage_atlas_for_context (lock-free), so today only tests call the locked variant — but the exported function is a loaded gun for the next plugin/route author, and the codebase already has the right pattern (get_embedding_matrix, cached_media_lookups: snapshot under lock, build unlocked, store under lock with a revision re-check).

  *Direction:* Restructure build_coverage_atlas to snapshot the medias + media_revision under _state_lock, build the atlas unlocked, then re-acquire the lock, verify the revision (and active context identity) still match, and only then assign ctx.coverage_atlas + resync votes; or delete the locked variant and route all callers through build_coverage_atlas_for_context plus an explicit locked resync.

<!-- item-sep -->

- **SortResultsCache bounds entry count but not bytes: 8 cached rankings of a 1M-item dataset pin gigabytes** — `vtscore/state/sort_results_cache.py:87` (low impact)

  The cache exists for 100k-1M item datasets (its own docstring), keeps up to max_entries=8 full result lists, and each list is a Python list of per-row dicts ({"id", "score"} or {"id", "similarity", "best_region"}) held by reference. At ~150-250 bytes per small dict, 8 rankings x 1M rows is roughly 1-2 GB of steady-state heap from a user simply re-sorting a large dataset a few times (each re-sort mints a new token, so distinct entries accumulate up to the cap even for the same dataset). The LRU bound protects against unbounded growth but not against exactly the large-N case the cache was built for.

  *Direction:* Bound the cache by total rows (e.g. evict oldest until sum(len(results)) <= ~2M) in store(), and/or store rankings columnar (an int64 id array + float32 score array, materializing row dicts only in page()) which cuts memory ~20x and also makes the stored list immune to caller mutation.

<!-- item-sep -->

### Training & detectors

<!-- item-sep -->

- **Progress-cache stability chain mixes thresholds and geometries from two different training pipelines** — `vtscore/detectors/labeling_progress.py:459` (medium impact)

  The Stable indicator compares consecutive steps' predictions (`_compute_step_stability`), but consecutive steps can come from different pipelines with incompatible thresholds and input geometry: a step resolved from `_live_models` (injected by `run_learned_sort`) carries the production model — trained on region-flooded patch rows with the fold-anchored population threshold — while a step resolved by `_train_step` trains a fresh linear head on image-level `media_embedding` vectors only (`_collect_training_data`, line 302) and thresholds it with an in-sample `conformal_threshold` over training scores (line 459). The monitored pool is likewise scored on image-level primary vectors (line 355) even though the production detector on a patch dataset max-pools ~197 rows per media. So (a) a single vote can flip many predictions purely because the threshold provenance switched between steps (fused population cut vs in-sample conformal cut), inflating the flip rate and delaying the green 'Stable' light, and (b) on patch datasets the Smart/Stable indicators measure a preview model whose training set (no flooding, no region pooling) and scoring geometry differ from the shipped detector, so the stopping-condition advice ('you can likely stop labeling') is calibrated against the wrong model. The comment at line 450 ('matching the production detector this previews') only holds for the head architecture, not the data or threshold.

  *Direction:* Route `_train_step` through the same `_build_vote_xy`/`train_and_threshold` assembly used by production (or at minimum reuse the same threshold rule for both live and replayed steps), and score the monitored pool with max-pooled region scores on patch datasets.

<!-- item-sep -->

- **_eval_cached_models re-implements inclusion_cost_weights instead of delegating to the declared single definition** — `vtscore/detectors/labeling_progress.py:678` (low impact)

  `vtscore.training.thresholds.inclusion_cost_weights` (thresholds.py:76) documents itself as 'the single definition' of the FPR/FNR cost weights, with eval modules explicitly delegating 'so a measured arm and the shipped path can never disagree'. `_eval_cached_models` (lines 678-683) inlines the identical `2.0**inclusion` arithmetic instead of calling it. The values match today, but this is exactly the mirrored-logic drift pattern the repo's eval-app-sync gate exists to prevent: if the knob's cost semantics ever change in `inclusion_cost_weights`, the Smart indicator's error-cost curve silently keeps the old pricing and the indicator diverges from every other consumer of the knob.

  *Direction:* Replace lines 678-683 with `fpr_weight, fnr_weight = inclusion_cost_weights(inclusion_value)` (the module already imports from vtscore.training.thresholds).

<!-- item-sep -->

### Media & embedding

<!-- item-sep -->

- **Whisper and PaddleOCR converters reload their model for every single media item** — `vtscore/converters/audio2text.py:103` (medium impact)

  `Audio2TextMediaConverter.convert` calls `whisper.load_model(model_size, ...)` inside convert(), i.e. once per audio file — the runner invokes convert per source file (vtscore/converters/runner.py:366), so transcribing a folder of 500 clips loads the ~145 MB (base) to ~3 GB (large) checkpoint from disk into RAM/VRAM 500 times, dominating wall-clock cost. `Image2TextMediaConverter` has the same shape: `_make_paddleocr` (vtscore/converters/image2text.py:50, 57-75) constructs a fresh PaddleOCR engine (det+rec+cls model loads) per image. The codebase already has the right pattern in the same package: `Image2FaceMediaConverter` caches its MTCNN detector on the instance across a conversion run (vtscore/converters/image2face.py:89-114, explicitly documented as 'reused across every image in a conversion run'). Caching keyed by the params that affect the model (model_size / language) would make both converters usable on realistic corpus sizes.

  *Direction:* Cache the loaded model on the converter instance keyed by (model_size,) / (language,) — mirroring Image2FaceMediaConverter._make_detector — and reuse it across convert() calls.

<!-- item-sep -->

- **Audio import decodes every file twice and ignores the bytes it was handed** — `vtscore/media/audio/media_type.py:1049` (medium impact)

  `AudioMediaType.load_media_data` accepts `media_bytes` precisely so callers that already read the file avoid a second read (per the base-class contract, vtscore/media/base.py:511-514), but then (a) fully decodes the audio from `file_path` anyway just to compute duration (`decode_audio(str(file_path), sr=None, mono=True)` — a complete PCM decode of, say, an hour-long podcast merely for its length), and (b) decodes the same payload a second time inside `generate_waveform_thumbnail(media_bytes)` (line 1053 → decode at line 169). So every imported audio file is decoded end-to-end twice per import, and the provided bytes are never used for the duration path. Duration is available near-free from the container header via `soundfile.SoundFile(...).frames / samplerate` (with an ffmpeg probe fallback for AAC/M4A), and even without that, decoding once and reusing the array for both duration and the waveform render halves import decode cost.

  *Direction:* Decode once from `media_bytes` (already in hand) and derive both duration and the waveform thumbnail from that single `(samples, sr)`; or read duration from `sf.info`-style header metadata instead of a full decode.

<!-- item-sep -->

- **Audio media_response always claims audio/wav and a .wav filename regardless of actual codec** — `vtscore/media/audio/media_type.py:1127` (low impact)

  `AudioMediaType.media_response` serves whatever `_resolve_media_bytes` returns — which for MP3/FLAC/OGG/M4A imports (all in `file_extensions`, line 328) is the original container bytes — but hard-codes `mimetype="audio/wav"` and `download_name=f"media_{id}.wav"`. Browsers usually sniff their way through playback, but a user downloading the clip gets an `.wav` file containing MP3/M4A bytes (which some players refuse by extension), and strict clients/proxies that trust Content-Type can mis-handle the stream. The bytes' real container is cheaply detectable from magic bytes (RIFF/ID3-or-0xFFEx/fLaC/OggS/ftyp).

  *Direction:* Sniff the first bytes of the payload (RIFF→wav, fLaC→flac, OggS→ogg, ID3/0xFFEx→mp3, ....ftyp→mp4/m4a) and set mimetype + download extension accordingly, defaulting to audio/wav.

<!-- item-sep -->

### Datasets & IO

<!-- item-sep -->

- **Staging imports bypass the download/embed concurrency gates** — `vtscore/datasets/load_pipeline.py:837` (medium impact)

  `_stage_importer_in_background` runs `importer.run(...)` and `embed_missing(...)` directly on its daemon thread without acquiring `_download_gate` or `_embed_gate`, unlike `_run_origin_load_in_background` which carefully sequences both. The combine flow can stage several datasets at once, so N stagings download and embed fully in parallel with each other *and* with gated regular loads — defeating the user-configurable `max_concurrent_dataset_downloads` / `max_concurrent_dataset_embeddings` limits whose whole purpose is bounding bandwidth/RAM/GPU pressure (and, per the embedder-singleton finding, making the `_on_progress` race reachable even when the embed gate is 1). Concrete benefit: staging a handful of image datasets on a RAM-constrained host would no longer multiply resident model weights and working sets past the configured budget.

  *Direction:* Reuse `_LoadGateController` in `stage_task`: acquire the download gate before `importer.run`, swap to the embed gate before `embed_missing`, release in the finally block.

<!-- item-sep -->

- **sync_from_labelset_source applies labels with an O(labels × medias) scan** — `vtscore/labels/sync.py:343` (low impact)

  For every imported label entry, the apply loop does a linear scan `for mid, media in ds_medias.items(): if media.get("md5") == md5` — O(L × N) while holding `_sync_lock` (which blocks every concurrent debounced push). With a 100k-item dataset and a few thousand imported labels this is hundreds of millions of dict lookups on the sync path that runs at detector load. Concrete benefit: building a one-pass `{md5: mid}` index before the loop makes the apply O(L + N) and shrinks the window during which `_sync_lock` starves `_push_to_labelset_source`.

  *Direction:* Before the loop: `md5_to_mid = {m.get("md5"): mid for mid, m in ds_medias.items()}` (first-wins to preserve current semantics), then `mid = md5_to_mid.get(md5)` per entry.

<!-- item-sep -->

### Eval harness

<!-- item-sep -->

- **check-eval-app-sync gate does not pin several ported surfaces, including the two that actually drifted** — `scripts/check-eval-app-sync.py:86` (medium impact)

  MIRRORS pins the phase machine, vote targets, three indicator rules, and four training defaults — but the harness ports more app logic than that, and the unpinned surfaces are exactly where drift was found: (1) `al_strategies._hard_pick_by_index` says it "Mirrors LabelViewComponent.autoSelectNext" (rank-space cutoff pick) and `_select_phase_faithful` mirrors the phase->Sort/Select pairing (`restoreAutopilotSortSelect`) — neither TS block is pinned, so changing the app's select rule or pairing silently detaches every simulated pick; (2) `_labelset_error_costs` / `AutopilotFlow.record_step` mirror `labeling_progress._eval_cached_models` / `_score_step` / `_compute_step_stability` (the Smart/Stable input semantics) — unpinned, and the Smart plumbing had in fact drifted (fixed in #2923, but nothing stops it drifting again); the `progress.smart_status` mirror pins only the rule function, so its `divergence=` text is the only thing standing in for a pin on the plumbing. Two smaller mechanism gaps: `_check_harness_side` (line 359) verifies the harness symbol by raw substring (`symbol in path.read_text()`), so a symbol surviving only in a comment passes; and `_normalize_python`'s trailing-comma stripping (line 265) erases the semantic difference between `(x,)` and `(x)`, so that one real logic change cannot trip a pin.

  *Direction:* Add Mirror entries for `ts:...label-view.component.ts::autoSelectNext(`, the sort/select restore block, and `py:vtscore.detectors.labeling_progress._eval_cached_models` / `_score_step` / `_compute_step_stability`; AST-check the harness symbol instead of substring; keep the trailing comma when the next token is `)` and the previous token is not an argument (or only strip inside call/collection contexts with >1 element).

<!-- item-sep -->

- **eval_learned_sort thresholds on a haystack that includes the held-out test set** — `vtscore/eval/runner.py:197` (low impact)

  `eval_learned_sort` calls `train_and_score(medias, ...)` over the FULL media dict, so the safe-threshold population estimator (fold-anchored GMM / blend) is fitted on a score distribution that includes the held-out test items, and the resulting threshold is then evaluated on those same items. `simulate_voting_iterations` explicitly refuses this (voting_iterations.py:2408-2414: "Restrict to the simulation set so the held-out test_ids never feed into the GMM ... otherwise the test scores leak into calibration and the reported metrics are biased upward"), so the two eval components apply inconsistent leakage policy: the runner's accuracy/precision/recall/F1 numbers are mildly transductive while the voting-iterations numbers are not, making them non-comparable. (Unsupervised leakage — no test labels are read — but the same kind voting_iterations guards against.)

  *Direction:* Restrict the snapshot passed to `train_and_score` to the train split (train_good + train_bad + train-side unlabeled), then score the test items separately with the returned model.

<!-- item-sep -->

- **Bad-phase text-sort cutoff and rank geometry include held-out test items** — `vtscore/eval/al_strategies.py:296` (low impact)

  `build_seed_scores` scores EVERY media id in the dataset (seed_scores.py:81-94), and `_pick_bad_phase` uses that map directly as the ranking: `_sort_threshold(ranking)` fits the text-sort GMM cutoff over the full dataset's cosine distribution (sim + test), and `_hard_pick_by_index` measures rank distance through interleaved test items that can never be picked. The simulated user's universe is supposed to be D_sim (the app's counterpart is the loaded dataset, which for the simulation is the sim half). The example-sort fallback is inconsistent with it too: `_centroid_similarities(ctx, list(ctx.embeddings))` ranks over sim embeddings only, so the two seeding modes run over different universes. No label leakage (cosines are query-based), but the bad-phase cutoff position and the resulting Bad votes depend on the test half, which shifts trajectories relative to a run at a different sim_fraction and blurs the sim/test separation the harness otherwise maintains.

  *Direction:* In `simulate_voting_iterations`, filter `seed_scores` to `sim_ids` before constructing the ALContext, so the text ranking, its GMM cutoff, and the rank distances all live in the simulated user's visible universe.

<!-- item-sep -->

### Projection, plugins & security utils

<!-- item-sep -->

- **vtscore CLI scoring imports vtsearch.achievements, breaking standalone library use** — `vtscore/cli.py:332` (medium impact)

  `_score_medias_with_detectors` unconditionally does `from vtsearch.achievements import record_find` whenever scoring produced results. `vtsearch.achievements` imports `vtsearch.auth` and `vtsearch.settings` (Flask-dependent app tier). vtscore is documented as the standalone, semver-released library tier ("Library tier; no Flask dependency" in docs/ARCHITECTURE.md, and vtscore/__init__.py positions it for external consumers), so an external consumer driving the autodetect pipeline through vtscore.cli gets an ImportError (or a Flask import) the moment scoring succeeds — the failure only appears on the success path, after models are trained and embeddings computed. Unlike the similar lazy hooks in state/votes.py and datasets/load_pipeline.py (which only run inside the app), this one sits on the library-tier CLI path that is meant to work without the app.

  *Direction:* Wrap the import in try/except ImportError (skip achievements when the app tier is absent), or better, expose a module-level `on_find_recorded` callback hook in vtscore that vtsearch registers at startup — keeping the dependency arrow pointing app→library only.

<!-- item-sep -->

- **Hilbert ordering recomputed per level in _level_membership despite being level-independent** — `vtscore/projection/pyramid.py:640` (low impact)

  `_level_membership` calls `perm = _hilbert_order(coords)` inside the per-level cache-miss path. The Hilbert permutation depends only on the frozen coords — it is identical for every level — yet each level's first tile fetch pays a fresh O(N) quantize + 16-iteration bit-twiddle + O(N log N) stable argsort. On a large dataset with a deep pyramid (up to 14 levels), the browse canvas re-derives the exact same permutation up to 14 times as the user zooms through levels, each time on the request thread serving the first tile of that level.

  *Direction:* Memoize the permutation once per Pyramid (e.g. a `_hilbert_perm` field alongside `_member_index`, or key the member-index cache computation to compute the perm once and reuse across levels).

<!-- item-sep -->

### Frontend — browse surface

<!-- item-sep -->

- **Idle thumbnail preloader fetches full-resolution originals once the full-res tier engages, unbounded in bytes** — `frontend/src/app/components/browse-canvas/browse-canvas.component.ts:1950` (medium impact)

  `useFullResThumbs` (line 668) flips `startThumbLoad` to the uncapped `/image` endpoint (line 1950), justified by the comment "Only a handful of such giant cells fit on screen at once, so the LRU still bounds memory". But the idle preloader (`runThumbPrefetch` → `warmThumbsForTiles` → `startThumbLoad(cell.rep_id, true)`, line 2165) shares the same tier and the same 2048-entry `MAX_THUMBS` cap: at a large thumbnail size (4XL/5XL crosses the 384px threshold at dpr 1), every idle pass warms up to 64 OFF-SCREEN cells — the pan ring plus the finer level's cells — with full-resolution originals, up to 2048 of them. For a photo dataset that is potentially gigabytes of image data fetched and retained for cells the user may never see; the cache bound is a count, not bytes, so the stated memory reasoning doesn't hold for the preload path. The benefit of fixing this is bounded memory/network at high zoom, where the app is otherwise most responsive.

  *Direction:* Have preload (`preload === true`) always fetch the capped `/thumbnail` regardless of tier (a later on-screen paint upgrades it), or shrink the LRU cap sharply while `thumbsAreFullRes` is active.

<!-- item-sep -->

- **A transient thumbnail load failure permanently blanks that cell until the projection changes** — `frontend/src/app/components/browse-canvas/browse-canvas.component.ts:1939` (low impact)

  `img.onerror` (line 1939-1942) adds the rep id to `thumbFailed`, and every subsequent `getThumb`/preload skips it forever — `thumbFailed` is only cleared on a projection switch or a resolution-tier crossing. A single transient failure (server restart, brief network blip, one 502 during a burst of 64 preload fetches) therefore leaves that bin rendered as flat density shading among thumbnails for the rest of the session, with no retry path and no user-visible way to recover short of leaving the view. The `onerror` also fires no redraw, relying on the 12s first-view backstop timer for the opening view.

  *Direction:* Treat failures as retryable: store a failure timestamp and retry after a backoff (or cap retries per id), and/or clear `thumbFailed` on `zoomToFit`/manual refresh actions.

<!-- item-sep -->

### Frontend — dashboard & modals

<!-- item-sep -->

- **Three near-identical dynamic plugin-field form engines should collapse into one shared component** — `frontend/src/app/components/modals/label-importer-modal/label-importer-modal.component.ts:125` (medium impact)

  The plugin-field form machinery — default seeding (`field.default`, first static option for strict selects), dynamic-options fetching with per-key loading/error maps, `depends_on` cascades, free-text datalist vs strict select rendering, file-field capture, and the full template branch ladder for server_path/file/password/email/url/select/number/text — is implemented three times with only cosmetic differences: label-importer-modal.component.ts (~125–225 + template), new-detector-modal.component.ts (trained tab, ~992–1097 + ~130 template lines with `nmm-` id prefixes), and plugin-import-form.component.ts (~59–170). new-detector-modal's own comment admits it exists "mirroring label-importer-modal ... with full parity". Divergence has already crept in (plugin-import-form validates required fields in `canSubmit`; the trained tab does not, so a missing required file only fails server-side), and the stale-response race reported separately must be fixed in three places. A single `vt-plugin-fields-form` component taking `fields` + an options-fetch fn and emitting `{values, file, fileFieldKey}` would delete roughly 600 lines and make future field types (and the race fix) land once.

  *Direction:* Extract a shared standalone component (fields input, getFieldOptions fn input, values/file outputs); adopt it in all three call sites. Backwards-compat breakage is acceptable per repo policy.

<!-- item-sep -->

- **Add a lint/audit gate for the zoneless anti-pattern: plain template-bound fields mutated in async callbacks** — `frontend/src/app/components/folder-browser/folder-browser.component.ts:133` (medium impact)

  This audit found five components in one area (folder-browser, login, progress-modal, combine-detectors-modal, settings-modal's exporter flag) that missed the zoneless migration's signalization pass, each producing invisible-until-next-click UI. The codebase clearly knows the rule — dozens of fields carry "signalized so the unpatched HTTP callbacks schedule CD under zoneless" comments, and local-folder/server-folder pickers even document the ancestor-marking subtleties around `markForCheck()` — but nothing enforces it, and component specs mask it by feeding synchronous `of(...)` observables so subscribe callbacks run inside an existing CD pass. Concrete benefit: a mechanical gate would have caught all five bugs. Two practical options: (a) an ESLint rule (typescript-eslint custom rule or `no-restricted-syntax` approximation) flagging `this.<identifier> =` assignments inside `.subscribe(...)` callbacks in `@Component` classes unless the property is a signal; (b) a test-infra convention requiring async fakes (`delay(0)` / Subjects) in specs that assert rendered output, which makes the missing repaint fail in Vitest.

  *Direction:* Add the lint rule to the frontend ESLint config and sweep remaining plain template-bound fields to signals; run-tests.sh already fails on lint errors so this becomes a durable gate.

<!-- item-sep -->

- **Bulk delete fires N parallel requests each triggering its own registry refresh** — `frontend/src/app/components/dashboard/dashboard.component.ts:691` (low impact)

  `deleteSelectedDatasets()` and `deleteSelectedDetectors()` loop over targets, issuing one DELETE per item and calling `this.datasetState.refresh()` in every `next` callback — deleting 20 datasets produces 20 registry refetches racing each other, and intermediate refreshes can repopulate the table mid-delete (rows flicker back before their own DELETE lands). Partial failures are also silent: individual errors are swallowed ("Global error interceptor surfaces the failure in the banner") with no summary of which items survived. Benefit: `forkJoin` over the delete observables with a single `refresh()` (and one toast noting any failures) removes the refetch storm and the flicker, and gives the user an accurate outcome for bulk operations.

  *Direction:* Wrap the per-item deletes in `forkJoin([...ids.map(id => api.delete(id).pipe(catchError(err => of({id, err}))))])`, then do one selection prune + one `datasetState.refresh()` + one summary toast.

<!-- item-sep -->

### Frontend — services & views

<!-- item-sep -->

- **LabelsetStateService lacks the out-of-order read guard that VoteStateService has** — `frontend/src/app/services/labelset-state.service.ts:124` (medium impact)

  The labelset piles are fed by two independent readers of GET labels-detail: the adaptive poll (startPolling) and the on-vote refresh() called from vote()'s POST continuation and from label-view.onMediaVoted() (label-view.component.ts line 1061). adaptivePoll never overlaps its own GETs, but a poll GET issued just before a vote commits can resolve AFTER the post-vote refresh(), reverting the just-moved element to its old pile until the next poll tick (~1.5s flicker, longer once the poll has backed off to its 10s heartbeat). This is exactly the staleness class VoteStateService closed with its votesSeq / lastAppliedVotesSeq issue-order guard (vote-state.service.ts lines 74-87 documents the identical bug for /api/votes); the labelset store predates that fix and never got it. The optimistic applyOptimisticState here also has no pending-entry reconciliation, so the stale overwrite is not corrected until the next full read.

  *Direction:* Stamp every labels-detail read (poll and refresh) with a monotonic issue id and drop responses older than the newest applied, mirroring VoteStateService.applyVotesFresh; optionally keep per-element pending entries until the server response agrees, as vote-state does for votes and region boxes.

<!-- item-sep -->

- **Find-view duplicates label-view's per-media-type panel-preference machinery by hand** — `frontend/src/app/components/find-view/find-view.component.ts:87` (medium impact)

  Label-view extracted its per-media-type panel bookkeeping into LabelViewPanelStateService (grid-size dicts, focus-mode dicts, panel_pct_left/right persistence, applyPanelPx clamping) and uses PanelResizeDirective for divider drags, but find-view still carries a parallel hand-rolled copy: gridIconSizeLeftDict / focusModeLeftDict / focusModeRightDict / panelPxLeftDict / panelPxRightDict fields (lines 87-91), a near-identical settings-mirror effect (lines 137-177), applyPanelPx (line 821), savePanelPx (line 812), and duplicated divider-drag + grid-snap logic (lines 407-483). The two implementations have already drifted — find-view lacks the icon-size auto-pop and snap-on-load behaviors label-view gained — and every future panel fix must be made twice. Since the settings keys are shared between the views, drift produces user-visible inconsistency (e.g. a width saved and snapped in Label restores un-snapped in Find).

  *Direction:* Provide LabelViewPanelStateService (renamed to a view-agnostic PanelStateService) in find-view too, and reuse PanelResizeDirective for its dividers, deleting the duplicated dict/effect/drag code.

<!-- item-sep -->

- **RunningJobsService clears all busy-pair spinners on any transient poll failure** — `frontend/src/app/services/running-jobs.service.ts:117` (low impact)

  The catchError in the /api/jobs/active poll emits `{busy_pairs: []}` on any error or 10s timeout, which wipes the busyPairs map. A single dropped request or slow response while jobs are genuinely running makes every pulldown spinner vanish for one 5s poll interval and then reappear — flickering UI that misreports 'no jobs running' during exactly the load conditions (heavy training) when jobs ARE running and the backend is slow to answer. Since the map is advisory UI state, holding the last known value through a transient failure is strictly better than clearing it; genuinely finished jobs are corrected on the next successful poll anyway.

  *Direction:* On error, re-emit the current busyPairsSubject.value (or emit nothing via EMPTY) instead of an empty payload, reserving the clear for stopPolling().

<!-- item-sep -->

### Frontend — styles & templates

<!-- item-sep -->

- **Global `.importer-picker`/`.exporter-picker` grid rule is overridden by every single consumer** — `frontend/src/scss/_components.scss:637` (medium impact)

  _components.scss:637-642 defines `.importer-picker, .exporter-picker { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--space-lg); }`, and its comment block (lines 632-635) says the settings/label pickers use this responsive grid. In reality all four consumers of these classes (settings-importer-modal.component.scss:1, settings-exporter-modal.component.scss:1, label-importer-modal.component.scss:1, dataset-importer-modal's source-picker.component.scss:21/30) redeclare the class locally as `display: flex; flex-direction: column` — and component-scoped rules win via Angular's encapsulation specificity bump — so the documented grid never renders anywhere. The next modal that applies the shared class trusting _components.scss (or style-guide §2.5) will get a grid layout that matches no existing picker. The shared rule's body has drifted into fiction.

  *Direction:* Make the global rule match universal usage (flex column with the agreed gap), delete the four local redeclarations, and fix the misleading comment. Any picker that genuinely wants a grid can opt in with a modifier class.

<!-- item-sep -->

<!-- item-sep -->

- **`.form-actions` rule duplicated verbatim in six component SCSS files** — `frontend/src/app/components/modals/settings-importer-modal/settings-importer-modal.component.scss:24` (medium impact)

  The identical rule `display: flex; justify-content: flex-end; gap: var(--space-md); margin-top: var(--space-md)` is declared in settings-importer-modal:24, settings-exporter-modal:24, label-importer-modal:29, clipper-chooser:78, dataset-importer-modal:22, and (minus margin-top) combine-datasets-modal:192 — plus a seventh dead copy in export-modal:316. Style-guide §6 says to promote a pattern on its third copy; this one has seven. Worse, every one of these `.form-actions` divs sits in the `[modal-footer]` slot, where the global `.modal-footer` (_components.scss:565) already provides `display:flex; justify-content:flex-end; gap:var(--space-md)` — so the duplicated body is ~redundant with the slot it lives in, and any future tune of footer spacing will drift across seven files.

  *Direction:* Either drop the `.form-actions` wrapper divs entirely (project the buttons directly into `[modal-footer]`, which already lays them out) or promote one `.form-actions` rule to _components.scss and delete all local copies.

<!-- item-sep -->

<!-- item-sep -->

- **Toast countdown uses forbidden `&hellip;` entity; toast/list button labels break Title Case** — `frontend/src/app/components/toast-container/toast-container.component.html:44` (low impact)

  Style-guide §4.2 is explicit: use the Unicode `…` character, "never the HTML entity `&hellip;`" — line 44 of the toast countdown is the single `&hellip;` in the codebase. In the same template, the action buttons read "Dismiss all" (line 12), "Hide details"/"Details" (line 62), and "Copy debug info" (line 65); §4.1 puts button labels in the Title Case bucket ("Dismiss All", "Hide Details", "Copy Debug Info"). Same lapse in media-list.component.html:32 ("Load more" → "Load More"). These are hand-review-only rules per the guide, which is exactly why an audit should catch them.

  *Direction:* Replace `&hellip;` with `…` and Title-Case the four button labels.

<!-- item-sep -->

- **Shared `_picker-shared.scss` itself violates the single-disabled-opacity rule** — `frontend/src/scss/_picker-shared.scss:138` (low impact)

  `.demo-table tbody tr.disabled { opacity: 0.55 }` hand-picks a disabled opacity in a *shared* stylesheet, contradicting §1.9 ("There is only one disabled opacity" — `--opacity-disabled: 0.5` — for `.disabled` states, added precisely because a rendered audit found seven ad-hoc values, 0.55 among them). Since this file is one of the guide's four canonical references, an off-token value here gets copied as precedent.

  *Direction:* Change to `opacity: var(--opacity-disabled);`.

<!-- item-sep -->

- **Residual token violations accumulate because the style audit is not gated** — `frontend/src/app/components/modals/settings-modal/auto-find/auto-find-settings.component.scss:12` (medium impact)

  The repo's own `.claude/scripts/style-check.py` currently reports 16 un-annotated raw px/rem spacing/font hits, 9 raw z-indexes, 5 raw `opacity: 0.7`s, and 2 heading restyles — i.e. the guide is drifting despite the tooling existing. Concrete examples verified by reading the code: this file's `gap: 0.35rem` (the token `--space-sm` is 0.375rem) and `opacity: 0.7` on `.help-icon` at rest (line 20 — the exact "dimmed icon" case §1.10 created `--opacity-dim` for; center-panel.component.scss:79/127 and autopilot-panel.component.scss:69 have the same raw 0.7); `gap: 2px` in browse-legend.component.scss:35 and label-list.component.scss:40 (`--space-2xs`); and label-list.component.scss:50 restyling `h3` to `--font-sm`/regular weight — anti-pattern 7, where the element is really a small section label, not an h3. Unlike ruff/codespell, this checker runs only when someone remembers `/style-check`, so violations land silently in PRs whose authors never open the guide.

  *Direction:* Sweep the current curated hits (most are one-line token swaps; the intentional off-scale ones already carry `// kept exact` annotations the script honors), then wire `style-check.py` into run-tests.sh as a gate the way the screenshots wiring-check already is, with an annotation/whitelist mechanism for the deliberate exceptions.

<!-- item-sep -->

### Security

<!-- item-sep -->

- **Hardcoded default Flask secret_key with no startup guard for multi-user deployments** — `app.py:118` (low impact)

  `app.secret_key` falls back to the literal `"vtsearch-dev-key-change-in-production"` when `VTSEARCH_SECRET_KEY` is unset (app.py:118), and nothing refuses to start (or warns loudly) when a non-DefaultLoginProvider is active with this public key. Signed session cookies (used by `TrivialLoginProvider` for `vtsearch_user`, and by the HF OAuth handshake for state/PKCE) are then forgeable by anyone who knows the shipped default. The practical blast radius today is limited because the only session-cookie-based provider, `TrivialLoginProvider`, is passwordless-by-design (an attacker can just POST /api/auth/login with any username), but the moment any real session-backed provider is added, or the HF state cookie is relied upon, a forgeable key becomes a live impersonation/CSRF-token-forgery vector. There is no defense-in-depth check tying 'multi-user provider selected' to 'a real secret key must be present'.

  *Direction:* At startup, if the active login provider is not DefaultLoginProvider and `VTSEARCH_SECRET_KEY` is unset (secret_key == the default), refuse to boot (or generate a random per-process key and log a prominent warning that sessions won't survive restart). Never ship a usable default secret in a mode that trusts signed cookies.

<!-- item-sep -->

### Tests & tooling

<!-- item-sep -->

- **Cross-dataset clip re-embedding tests assert nothing — the documented behavior can never fail** — `tests/detectors/test_clipper_workflow.py:482` (medium impact)

  All four tests in TestCrossDatasetClipEmbedding (lines 482-559) call `_apply_clip_and_embed(...)` and assert nothing — comments say "Result may be None if no embedder is loaded, which is fine". Their docstrings claim they verify that clip params cause slicing/cropping/sentence-extraction before embedding, but a regression that ignores clip params entirely (or returns None always) passes all four. This is the load-bearing path of the repo's core invariant ("origins are canonical; the system rederives origin → file → embedding on demand") for clipped labels, so it is effectively untested. Notably the excuse is stale: the session-scoped `_stub_embedding_models` fixture (tests/conftest.py:276-297) patches every embedder's `embed_media` with a deterministic content-seeded fake, so a real assertion is easy — the clipped call should return a non-None unit vector that differs from the unclipped `_apply_clip_and_embed` result on the same file (the fake seeds off file/clip bytes).

  *Direction:* Assert the return is a non-None float32 vector of the right dim, and that clipped vs. no-clip origins produce different vectors (and identical clip params produce identical vectors).

<!-- item-sep -->

- **check-dockerfiles.py misses python invocations on RUN continuation lines and `python3`** — `scripts/check-dockerfiles.py:40` (low impact)

  The ordering gate skips every physical line that continues a previous instruction (lines 38-41: `if is_continuation: continue`), so `RUN set -e \\\n && python scripts/foo.py` is never inspected — only a `python` on the RUN's first physical line is caught. It also matches only `\bpython\b` (line 62), so `python3 ...` (the actual interpreter name in Dockerfile.gpu, where `/usr/bin/python` is only a symlink created in the same layer) slips through. A future Dockerfile edit that runs Python in a multi-line RUN before vtsearch/+vtscore/ are copied would pass the gate the gate exists to catch. Relatedly, .dockerignore excludes `tests/` but not `tests_lib/`, so the library-tier test tree (with fixtures) is baked into every image for no reason.

  *Direction:* Accumulate logical instructions (join continuation lines before matching), broaden the regex to `python[0-9.]*`, and add tests_lib/ to .dockerignore.

<!-- item-sep -->

- **ensure-test-deps.sh keeps the SETUPTOOLS_USE_DISTUTILS=stdlib shim that install.sh explicitly documents as broken** — `.claude/hooks/ensure-test-deps.sh:59` (low impact)

  Line 59 installs apricot-select under `SETUPTOOLS_USE_DISTUTILS=stdlib`, while scripts/install.sh's vts_install_toponymy comment (added later) states verbatim: "Do NOT wrap it in SETUPTOOLS_USE_DISTUTILS=stdlib: pip's isolated build env installs the latest setuptools, and setuptools >= 74 refuses to even import with that value set, so the build dies with 'BackendUnavailable' (and Python >= 3.12 has no stdlib distutils for the shim to point at anyway)... verified on Python 3.12 and 3.14". The hook works today only because the remote container runs Python 3.11 (stdlib distutils still exists — I confirmed the build succeeds there); the day the remote image moves to Python 3.12+, first-run dependency install breaks at this line and every test/app command is blocked. The two installers for the same package now embody contradictory conclusions about the same workaround.

  *Direction:* Drop the env-var prefix in ensure-test-deps.sh to match install.sh's vts_install_toponymy (plain `pip install apricot-select`).

<!-- item-sep -->

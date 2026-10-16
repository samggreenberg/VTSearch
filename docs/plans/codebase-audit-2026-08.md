# Codebase audit — August 2026

**Background.** A full-codebase inspection was run at `00664df5` (dev): fourteen
specialist reviewers over disjoint areas (~118 k lines Python, ~44 k lines
TypeScript), each reported finding then handed to an independent verifier
prompted to *refute* it against the code and defaulting to "not real" when
uncertain. 69 bugs survived that pass; 3 claims were refuted and dropped, and one
more (#2972) was withdrawn after a second suite run disproved it. The
repo's own gates were green throughout (7 860 tests, ruff, format, frontend
build), so everything below came from reading code, not from a broken build.

**What is still owed** is the improvement proposals below. The audit's confirmed
defects were filed as GitHub issues and have all shipped. The proposals that turned
out to be discrete defects were promoted to issues in October 2026 (#4972) and stay
here only as pointer lines; the rest are design directions, whose bodies live here
and nowhere else. Delete a proposal when it ships or is rejected.

---

## Improvement proposals

Design and architecture directions surfaced by the same review. The bodies here
are deliberately **not** issues: each is a judgement call about direction rather
than a defect with a right answer, and several are alternatives to one another.
Promote one to an issue (and replace its body with a pointer line) when it becomes
concrete enough to ship on its own.

### Flask API layer

<!-- item-sep -->

<!-- item-sep -->

- **Detector listing re-reads full detector JSON files from disk on every request for legacy entries** — `vtsearch/routes/detectors/registry.py` (medium impact)

  GET /api/detectors/registry backfills two fields for entries that predate them, and does so on every single request: line 127-129 calls `_read_detector(_detector_path(name))` whenever `entry.get('embedder_type')` is falsy, and lines 133-135 read the same file a second time whenever 'examples' is missing. `_read_detector` parses the whole detector JSON — including a labelset that can hold thousands of label dicts — so a dashboard that polls the registry pays O(legacy_detectors × labelset_size) JSON parsing per poll, twice per entry. Neither computed value is written back via update_detector, so the cost never amortizes. Benefit: one-time lazy migration turns a recurring disk+parse cost into a single write.

  *Direction:* Read the file once per entry, derive both embedder_type and examples from that single read, and persist them back with update_detector(did, embedder_type=..., examples=...) so subsequent listings hit only the registry entry.

<!-- item-sep -->

- **Readers-endpoint maps error to status by substring-matching the message** — `vtsearch/routes/detectors/registry.py` (low impact)

  update_detector_readers (and its dataset twin at routes/datasets/registry.py) chooses the HTTP status with `status = 403 if "creator" in err else 404` — coupling status-code semantics to the wording of set_detector_readers' human-readable error string. Rewording the message (e.g. 'Only the owner can modify readers') would silently turn permission failures into 404s. The registry helper already distinguishes the two cases internally (registry.py); it just flattens them into prose.

  *Direction:* Have set_detector_readers/set_readers return a typed error (enum or exception class) and map it explicitly to 403/404 in the routes.

<!-- item-sep -->

- **Two parallel detector APIs (file-based /api/detectors vs registry) with divergent integrity rules** — `vtsearch/routes/detectors/crud.py` (medium impact)

  The codebase exposes two overlapping detector CRUD surfaces operating on the same on-disk files: the file-keyed /api/detectors family (crud.py: list by directory scan, create/delete/rename by name) and the id-keyed /api/detectors/registry family (registry.py: ACLs, loaded flags, owner checks). They enforce different invariants — crud.py checks name collisions on create/rename but knows nothing about ACLs or the registry's loaded-ids, while the registry routes enforce ownership but skip collision checks (see the two bugs filed above); crud's DELETE /api/detectors/<name> unlinks the file while leaving any registry entry pointing at nothing, and crud's combine writes a detector file that never gets a registry entry. Any invariant fixed in one family has to be re-fixed in the other. Consolidating on the registry family (with the file store as its private persistence) — or routing crud handlers through shared helpers that own collision/ACL/registry consistency — would eliminate this class of drift; backwards-compat breaks are acceptable per repo policy.

  *Direction:* Fold the /api/detectors file-keyed routes into the registry blueprint (or shared service functions) so name-collision, ownership, registry-entry, and file lifecycle are enforced in exactly one place.

<!-- item-sep -->

### App core (settings, auth, CLI)

<!-- item-sep -->

<!-- item-sep -->

- **Achievements persist a full settings-file RMW plus a source push on every single vote** — `vtsearch/achievements.py` (medium impact)

  `record_vote` runs `mutate_user(...)` per vote (achievements.py), and `mutate_user` is heavyweight by design: cross-process flock on the user settings file, fresh `_load_path` re-read, full-dict `json.dumps` + `fsync` + rename (`_atomic_write`, settings_store.py), then a dirty-marking pass over *every* exportable key and a `_sync_to_source` push (settings.py) — which for a configured source means a second full serialize/write (plus a `peek_version` stat and a `.syncmark` write). A user hand-labeling in the Train flow votes multiple times per second, so each click costs 3-4 fsync'd file writes and lock round-trips on the request path, and the achievement counters share a file (and its lock) with every other settings read/write, amplifying contention on the very lock structure that finding #1 shows is fragile. The `days_seen`/`docs_read_ids`/`trained_detector_ids` lists also use O(n) `in` checks on every vote (achievements.py), which grows linearly with days active. Benefit: batching (e.g. accumulate credits in-process and flush on a short timer or every N votes, as the counters are approximate milestones anyway) or moving `achievement_state` to its own small file outside the settings-sync machinery would remove the per-click fsync+push cost and cut settings-lock traffic substantially.

  *Direction:* Accumulate vote credits in a process-local per-user buffer and flush to disk on a debounce (e.g. 5s or on get_full_state), and/or store achievement_state in a dedicated per-user file excluded from settings-source export.

<!-- item-sep -->

### State & concurrency

<!-- item-sep -->

- [ ] #4976 — A queued job whose dataset or detector was unloaded runs against the empty context and is marked done (Sonnet)

<!-- item-sep -->

- [ ] #4977 — build_coverage_atlas holds the global state lock across the whole k-means fit (Opus)

<!-- item-sep -->

- [ ] #4978 — SortResultsCache bounds entry count but not size, so large re-sorts pin gigabytes (Sonnet)

<!-- item-sep -->

### Training & detectors

<!-- item-sep -->

<!-- item-sep -->

<!-- item-sep -->

### Media & embedding

<!-- item-sep -->

- [ ] #4979 — Whisper and PaddleOCR converters reload their model for every media item (Sonnet)

<!-- item-sep -->

- [ ] #4980 — Audio import decodes every file twice and ignores the bytes it was given (Sonnet)

<!-- item-sep -->

- [ ] #4981 — Audio responses always claim audio/wav and a .wav filename whatever the codec (Sonnet)

<!-- item-sep -->

### Datasets & IO

<!-- item-sep -->

- **sync_from_labelset_source applies labels with an O(labels × medias) scan** — `vtscore/labels/sync.py` (low impact)

  For every imported label entry, the apply loop does a linear scan `for mid, media in ds_medias.items(): if media.get("md5") == md5` — O(L × N) while holding `_sync_lock` (which blocks every concurrent debounced push). With a 100k-item dataset and a few thousand imported labels this is hundreds of millions of dict lookups on the sync path that runs at detector load. Concrete benefit: building a one-pass `{md5: mid}` index before the loop makes the apply O(L + N) and shrinks the window during which `_sync_lock` starves `_push_to_labelset_source`.

  *Direction:* Before the loop: `md5_to_mid = {m.get("md5"): mid for mid, m in ds_medias.items()}` (first-wins to preserve current semantics), then `mid = md5_to_mid.get(md5)` per entry.

<!-- item-sep -->

### Eval harness

<!-- item-sep -->

- **check-eval-app-sync gate does not pin several ported surfaces, including the two that actually drifted** — `scripts/check-eval-app-sync.py` (medium impact)

  MIRRORS pins the phase machine, vote targets, three indicator rules, and four training defaults — but the harness ports more app logic than that, and the unpinned surfaces are exactly where drift was found: (1) `_select_phase_faithful` mirrors the phase->Sort/Select pairing (`restoreAutopilotSortSelect`), which is not pinned, so changing the app's pairing silently detaches every simulated pick (`autoSelectNext` itself is now pinned); (2) `_labelset_error_costs` / `AutopilotFlow.record_step` mirror `labeling_progress._eval_cached_models` / `_compute_step_stability` (the Smart/Stable input semantics) — unpinned (`_score_step` now is), and the Smart plumbing had in fact drifted once (fixed in #2923). One smaller mechanism gap remains: `_normalize_python`'s trailing-comma stripping erases the semantic difference between `(x,)` and `(x)`, so that one real logic change cannot trip a pin.

  *Direction:* Add Mirror entries for the sort/select restore block and `py:vtscore.detectors.labeling_progress._eval_cached_models` / `_compute_step_stability`; keep the trailing comma when the next token is `)` and the previous token is not an argument (or only strip inside call/collection contexts with >1 element).

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

<!-- item-sep -->

- [ ] #4982 — Pyramid recomputes the level-independent Hilbert permutation for every level (Sonnet)

<!-- item-sep -->

### Frontend — dashboard & modals

<!-- item-sep -->

- **Three near-identical dynamic plugin-field form engines should collapse into one shared component** — `frontend/src/app/components/modals/label-importer-modal/label-importer-modal.component.ts` (medium impact)

  The plugin-field form machinery — default seeding (`field.default`, first static option for strict selects), dynamic-options fetching with per-key loading/error maps, `depends_on` cascades, free-text datalist vs strict select rendering, file-field capture, and the full template branch ladder for server_path/file/password/email/url/select/number/text — is implemented three times with only cosmetic differences: label-importer-modal.component.ts (~125–225 + template), new-detector-modal.component.ts (trained tab, ~992–1097 + ~130 template lines with `nmm-` id prefixes), and plugin-import-form.component.ts (~59–170). new-detector-modal's own comment admits it exists "mirroring label-importer-modal ... with full parity". Divergence has already crept in (plugin-import-form validates required fields in `canSubmit`; the trained tab does not, so a missing required file only fails server-side), and the stale-response race reported separately must be fixed in three places. A single `vt-plugin-fields-form` component taking `fields` + an options-fetch fn and emitting `{values, file, fileFieldKey}` would delete roughly 600 lines and make future field types (and the race fix) land once.

  *Direction:* Extract a shared standalone component (fields input, getFieldOptions fn input, values/file outputs); adopt it in all three call sites. Backwards-compat breakage is acceptable per repo policy.

<!-- item-sep -->

- **Add a lint/audit gate for the zoneless anti-pattern: plain template-bound fields mutated in async callbacks** — `frontend/src/app/components/folder-browser/folder-browser.component.ts` (medium impact)

  This audit found five components in one area (folder-browser, login, progress-modal, combine-detectors-modal, settings-modal's exporter flag) that missed the zoneless migration's signalization pass, each producing invisible-until-next-click UI. The codebase clearly knows the rule — dozens of fields carry "signalized so the unpatched HTTP callbacks schedule CD under zoneless" comments, and local-folder/server-folder pickers even document the ancestor-marking subtleties around `markForCheck()` — but nothing enforces it, and component specs mask it by feeding synchronous `of(...)` observables so subscribe callbacks run inside an existing CD pass. Concrete benefit: a mechanical gate would have caught all five bugs. Two practical options: (a) an ESLint rule (typescript-eslint custom rule or `no-restricted-syntax` approximation) flagging `this.<identifier> =` assignments inside `.subscribe(...)` callbacks in `@Component` classes unless the property is a signal; (b) a test-infra convention requiring async fakes (`delay(0)` / Subjects) in specs that assert rendered output, which makes the missing repaint fail in Vitest.

  *Direction:* Add the lint rule to the frontend ESLint config and sweep remaining plain template-bound fields to signals; run-tests.sh already fails on lint errors so this becomes a durable gate.

<!-- item-sep -->

- [ ] #4984 — Dashboard bulk delete refreshes the registry once per item and never reports which deletes failed (Sonnet)

<!-- item-sep -->

### Frontend — services & views

<!-- item-sep -->

- [ ] #4985 — LabelsetStateService can apply an older labels-detail read over a newer one (Sonnet)

<!-- item-sep -->

- **Find-view duplicates label-view's panel drag/snap machinery by hand** — `frontend/src/app/components/find-view/find-view.component.ts` (medium impact)

  Label-view uses PanelResizeDirective for its divider drags and has icon-size auto-pop plus snap-on-load; find-view still carries its own divider-drag and grid-snap code and neither of those behaviours. Every future panel fix has to be made twice, and because the two views share the same settings keys the drift is user-visible — a width saved and snapped in Label restores un-snapped in Find.

  *Direction:* Reuse PanelResizeDirective for find-view's dividers and lift the auto-pop / snap-on-load behaviour alongside it, deleting the duplicated drag code.

  *Background:* the settings half of this item is done — both views now resolve their per-media-type panel preferences through `SettingsStateService.perMediaType` (#3447), so the shadow dicts and the mirror effects are gone from find-view. What remains is the drag/snap duplication.

<!-- item-sep -->

- [ ] #4986 — RunningJobsService clears every busy-pair spinner when a single poll fails or times out (Sonnet)

<!-- item-sep -->

### Frontend — styles & templates

<!-- item-sep -->

- [ ] #4987 — Global .importer-picker / .exporter-picker grid rule is overridden by every consumer (Sonnet)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #4990 — .form-actions is redeclared in six component stylesheets and the copies have already diverged (Sonnet)

<!-- item-sep -->

<!-- item-sep -->

- [ ] #4989 — Toast and Load more button labels are in sentence case, not Title Case (Haiku)

<!-- item-sep -->

- [ ] #4988 — Raw disabled-state opacities bypass --opacity-disabled, including one in the shared picker stylesheet (Haiku)

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

- **check-dockerfiles.py misses `python3` invocations** — `scripts/check-dockerfiles.py` (low impact)

  The ordering gate matches only `\bpython\b`, so `python3 ...` (the actual interpreter name in Dockerfile.gpu, where `/usr/bin/python` is only a symlink created in the same layer) slips through. A future Dockerfile edit that runs `python3` before vtsearch/+vtscore/ are copied would pass the gate the gate exists to catch. Relatedly, .dockerignore excludes `tests/` but not `tests_lib/`, so the library-tier test tree (with fixtures) is baked into every image for no reason.

  *Direction:* Broaden the regex to `python[0-9.]*`, with a carve-out for Dockerfile.gpu's `python3 -m venv`, which runs before the COPYs and needs no package; and add tests_lib/ to .dockerignore.

<!-- item-sep -->

<!-- item-sep -->

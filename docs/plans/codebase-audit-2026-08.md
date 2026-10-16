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

- [ ] #5001 — Detector registry listing re-parses detector JSON on every request for entries whose embedder type or examples were never stored (Sonnet)

<!-- item-sep -->

- [ ] #5000 — Readers endpoints pick 403 vs 404 by searching the error message for "creator" (Sonnet)

<!-- item-sep -->

- **Two parallel detector APIs (file-based /api/detectors vs registry) with divergent integrity rules** — `vtsearch/routes/detectors/crud.py` (medium impact)

  The codebase exposes two overlapping detector CRUD surfaces operating on the same on-disk files: the file-keyed /api/detectors family (crud.py: list by directory scan, create/delete/rename by name) and the id-keyed /api/detectors/registry family (registry.py: ACLs, loaded flags, owner checks). They enforce different invariants — crud.py knows nothing about ACLs or the registry's loaded-ids, and only some of its routes keep the registry in step: rename and combine update or create the registry entry, but create writes a detector file that never gets one, and DELETE /api/detectors/<name> unlinks the file while leaving any registry entry pointing at nothing. Any invariant fixed in one family has to be re-fixed in the other. Consolidating on the registry family (with the file store as its private persistence) — or routing crud handlers through shared helpers that own collision/ACL/registry consistency — would eliminate this class of drift; backwards-compat breaks are acceptable per repo policy.

  *Direction:* Fold the /api/detectors file-keyed routes into the registry blueprint (or shared service functions) so name-collision, ownership, registry-entry, and file lifecycle are enforced in exactly one place.

<!-- item-sep -->

### App core (settings, auth, CLI)

<!-- item-sep -->

<!-- item-sep -->

- **Achievements persist a full settings-file RMW plus a source push on every single vote** — `vtsearch/achievements.py` (medium impact)

  `record_vote` runs `mutate_user(...)` per vote (achievements.py), and `mutate_user` is heavyweight by design: cross-process flock on the user settings file, fresh `_load_path` re-read, full-dict `json.dumps` + `fsync` + rename (`_atomic_write`, settings_store.py), then a dirty-marking pass over *every* exportable key and a `_sync_to_source` push (settings.py) — which for a configured source means a second full serialize/write (plus a `peek_version` stat and a `.syncmark` write). A user hand-labeling in the Train flow votes multiple times per second, so each click costs 3-4 fsync'd file writes and lock round-trips on the request path, and the achievement counters share a file (and its lock) with every other settings read/write, amplifying contention on the settings lock. The `days_seen`/`docs_read_ids`/`trained_detector_ids` lists also use O(n) `in` checks on every vote (achievements.py), which grows linearly with days active. Benefit: batching (e.g. accumulate credits in-process and flush on a short timer or every N votes, as the counters are approximate milestones anyway) or moving `achievement_state` to its own small file outside the settings-sync machinery would remove the per-click fsync+push cost and cut settings-lock traffic substantially.

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

<!-- item-sep -->

### Eval harness

<!-- item-sep -->

- [ ] #5005 — check-eval-app-sync does not pin the Smart window and Stable-input plumbing the harness ports (Sonnet)

<!-- item-sep -->

- [ ] #5006 — check-eval-app-sync normalizer makes (x,) and (x) hash the same (Sonnet)

<!-- item-sep -->

- [ ] #5003 — eval_learned_sort fits its threshold on the very items it then evaluates (Opus)

<!-- item-sep -->

- [ ] #5004 — Autopilot simulation's text-sort cut and rank geometry include the held-out test half (Opus)

<!-- item-sep -->

### Projection, plugins & security utils

<!-- item-sep -->

<!-- item-sep -->

- [ ] #4982 — Pyramid recomputes the level-independent Hilbert permutation for every level (Sonnet)

<!-- item-sep -->

### Frontend — dashboard & modals

<!-- item-sep -->

- **Three near-identical dynamic plugin-field form engines should collapse into one shared component** — `frontend/src/app/components/modals/label-importer-modal/label-importer-modal.component.ts` (medium impact)

  The plugin-field form machinery — default seeding (`field.default`, first static option for strict selects), dynamic-options fetching with per-key loading/error maps, `depends_on` cascades, free-text datalist vs strict select rendering, file-field capture, and the full template branch ladder for server_path/file/password/email/url/select/number/text — is implemented three times with only cosmetic differences: label-importer-modal.component.ts, new-detector-modal.component.ts (the trained tab, with `nmm-` id prefixes in its template), and plugin-import-form.component.ts. new-detector-modal's own comment admits it exists "mirroring label-importer-modal ... with full parity". Divergence has already crept in (plugin-import-form validates required fields in `canSubmit`; the trained tab does not, so a missing required file only fails server-side), though the stale-response guard for dynamic options is already shared (`frontend/src/app/utils/dynamic-field-options.ts`). A single `vt-plugin-fields-form` component taking `fields` + an options-fetch fn and emitting `{values, file, fileFieldKey}` would delete roughly 600 lines and make future field types land once.

  *Direction:* Extract a shared standalone component (fields input, getFieldOptions fn input, values/file outputs); adopt it in all three call sites. Backwards-compat breakage is acceptable per repo policy.

<!-- item-sep -->

- **Add a lint/audit gate for the zoneless anti-pattern: plain template-bound fields mutated in async callbacks** — `frontend/src/app/components/folder-browser/folder-browser.component.ts` (medium impact)

  This audit found five components in one area (folder-browser, login, progress-modal, combine-detectors-modal, settings-modal's exporter flag) that missed the zoneless migration's signalization pass, each producing invisible-until-next-click UI. The codebase clearly knows the rule — dozens of fields carry "signalized so the unpatched HTTP callbacks schedule CD under zoneless" comments, and local-folder/server-folder pickers even document the ancestor-marking subtleties around `markForCheck()` — but nothing enforces it, and component specs mask it by feeding synchronous `of(...)` observables so subscribe callbacks run inside an existing CD pass. Concrete benefit: a mechanical gate would have caught all five bugs. Two practical options: (a) a lint rule (the frontend has no ESLint setup today, so this means adding one — a typescript-eslint custom rule or `no-restricted-syntax` approximation) flagging `this.<identifier> =` assignments inside `.subscribe(...)` callbacks in `@Component` classes unless the property is a signal; (b) a test-infra convention requiring async fakes (`delay(0)` / Subjects) in specs that assert rendered output, which makes the missing repaint fail in Vitest.

  *Direction:* Add a frontend lint step carrying the rule, wire it into `run-tests.sh` as a gate, and sweep remaining plain template-bound fields to signals.

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

  The repo's own `.claude/scripts/style-check.py` still reports un-annotated raw px/rem spacing/font hits, raw z-indexes, raw decorative-dim opacities and a heading restyle — i.e. the guide is drifting despite the tooling existing. Concrete examples verified by reading the code: this file's `gap: 0.35rem` (the token `--space-sm` is 0.375rem) and `opacity: 0.7` on `.help-icon` at rest (the exact "dimmed icon" case §1.10 created `--opacity-dim` for; center-panel.component.scss and autopilot-panel.component.scss have the same raw 0.7); `gap: 2px` in browse-legend.component.scss and label-list.component.scss (`--space-2xs`); and label-list.component.scss restyling `h3` to `--font-sm`/regular weight — anti-pattern 7, where the element is really a small section label, not an h3. Unlike ruff/codespell, this checker runs only when someone remembers `/style-check`, so violations land silently in PRs whose authors never open the guide.

  *Direction:* Sweep the current curated hits (most are one-line token swaps; the intentional off-scale ones already carry `// kept exact` annotations the script honors), then wire `style-check.py` into run-tests.sh as a gate the way the screenshots wiring-check already is, with an annotation/whitelist mechanism for the deliberate exceptions.

<!-- item-sep -->

### Security

<!-- item-sep -->

- [ ] #5002 — Flask secret_key defaults to a public constant and nothing warns when signed session cookies are in use (Sonnet)

<!-- item-sep -->

### Tests & tooling

<!-- item-sep -->

- [ ] #5007 — Cross-dataset clip re-embedding tests call _apply_clip_and_embed and assert nothing (Sonnet)

<!-- item-sep -->

- [ ] #5008 — check-dockerfiles.py does not match python3, and tests_lib/ ships in every image (Sonnet)

<!-- item-sep -->

<!-- item-sep -->

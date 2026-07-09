# `httpResource` / reactive-resource migration for the data layer

Status: **Phases 1–4 shipped** (`SettingsStateService`, `MediaStateService`,
left-panel media-type/embedder reads, importer/exporter picker-modal reads all
on `rxResource`; pilot compat shims removed). Pollers and `forkJoin` aggregates
deferred by design — open follow-ups below.

## Goal & scope

Trim the `switchMap` / manual-subscription / `BehaviorSubject` boilerplate in
the **read** path of the data layer by moving GET-style fetches onto Angular's
reactive resource primitives (`rxResource` / `httpResource`). Reads become
signal-driven: a resource re-fetches when its request signals change, exposes
`value()`/`status()`/`error()`/`isLoading()`, and drops the hand-rolled
"subscribe in `ngOnInit`, store in a field, unsubscribe in `ngOnDestroy`"
ceremony.

**Explicitly NOT in scope:** mutations (POST/PUT/DELETE — votes, sort kicks,
imports, renames, deletes) stay imperative `HttpClient` calls. Resources are
for reads. Server-Sent Events (`ProgressEventsService`) stay as-is. This is a
*read-path* refactor, adoptable incrementally — never a big-bang rewrite. Each
step is independently shippable and revertible; there is no point where the app
must be half-migrated.

## Open follow-ups

- **Expand to more read services** (the standing next step). Done for the four
  services in "What shipped". A still-open refinement: if media-types /
  embedders ever need to **re-fetch on dataset switch**, give the resource a
  `toSignal(activeContext.datasetId$)` request key instead of the eager load —
  today the component is recreated on switch, so eager-once suffices.
- **Remaining component-local read subscribes** (not yet converted). The next
  candidates after the picker modals are heavier: `dataset-importer-modal`
  (interdependent clipper/embedder/demo loads + dynamic field-option fetches)
  and `label-importer-modal` (importer list is a clean read, but field-options
  and imports are mutations). Convert the clean list reads when those modals are
  next touched; leave the dynamic field-option fetches imperative.
- **Pollers and `forkJoin` aggregates** (`VoteStateService`, labeling status,
  `DatasetStateService.refresh()`) are still imperative by design. Resources
  fetch-on-request-change, not on an interval, and don't map cleanly onto
  aggregate composition. Options if revisited: drive a resource's reload from a
  timer signal (pollers), or a resource per call composed via `computed`
  (aggregates). Revisit only if a resource clearly wins — decide per call site;
  don't force it.

### Decision points still open

- **Depth of signal adoption**: thin `toSignal` bridges (minimal, keep
  `BehaviorSubject` services) vs converting hot state services to signal-backed
  (bigger, cleaner). Started thin (recommended); convert services
  opportunistically.
- **Relationship to zoneless.** Going zoneless (the other deferred carrot from
  `angular-21-upgrade.md`) wants broad signal adoption too; this migration is a
  natural on-ramp. Decide whether to treat them as one track or keep separate.

### Out of scope (record only)

- Mutations, SSE, polling intervals (above).
- Replacing `ng-openapi-gen` — the generated client stays; `rxResource` wraps it.
- Going zoneless — its own effort (`angular-21-upgrade.md` → Deferred).

## Conversion pattern (reference for the remaining reads)

The shipped phases proved a repeatable recipe. Follow it for the next
conversions.

- **`rxResource` over raw `httpResource`.** `rxResource({ params, stream })`
  (from `@angular/core/rxjs-interop`) wraps an **Observable loader** that can
  call the existing typed `*-api.service.ts` methods unchanged, so we **keep the
  generated client** (`ng-openapi-gen`, `apiService: false`) and its types, and
  just gain signal-driven reactivity. Raw `httpResource(() => url)` bypasses the
  generated client — you'd hand-build URLs/params it already generates,
  duplicating the typed contracts and losing the OpenAPI-snapshot guarantee.
  Reserve `httpResource` only for genuinely new raw reads where no generated
  function exists. Both share `provideHttpClient` + the interceptor chain
  (`activeContext` header injection, achievements-refresh, error/circuit-breaker
  chokepoint), so headers and error side effects keep working either way.
- **Request trigger.** For an imperative "load/refresh on demand" read, use a
  **monotonic counter signal** as the request: `params: () => count === 0 ?
  undefined : count` keeps the resource idle until the first `load()`
  (`undefined` request = no fetch); each `load()` bumps the counter to refetch;
  `clear()` is `resource.set([])`/`set(undefined)`. For a **component-local
  `ngOnInit` fetch that loads once**, use an **eager** resource (no request
  signal). For an **input-derived** read (request depends on an `@Input` set in
  `ngOnInit`), use a **trigger/gate signal** flipped at the end of `ngOnInit`
  (export-modal's `labelsReady` pattern).
- **Signals are the canonical surface.** Expose `…Signal()` / `isLoading()` /
  `error()` from the resource; templates and consumers read the signals.
  Per-service compat shims (a sync getter, a `toObservable` bridge) are **not**
  kept — migrate caller and callee together (repo "no shims" rule).
- **Consumer migration.** Every `x$.subscribe(v => …)` becomes a **constructor
  `effect()`** that reads the signal; effects auto-dispose with the component, so
  the matching `takeUntil(destroy$)` / `destroy$` / `OnDestroy` plumbing is
  dropped where it was the last user. Synchronous getter reads become signal
  calls, including in templates.
- **`inject()` over constructor params.** Resource field initializers reference
  the api services, and a field initializer can't safely read a constructor
  parameter property, so the services must be `inject()`ed.
- **Error merging.** Where a component's `error` was set by *both* a list-load
  failure and an action failure, keep a writable `signal` for the action error
  and expose `error` as a `computed` that ORs in `resource.error()`.
- **Mutations stay imperative** — still `HttpClient` POST/PUT; write the
  server's response back via `resource.set(...)`. Document side-effects that used
  a manual `emit()` move to an `effect()` watching the signal.

### Testing `rxResource` (the real cost)

The loader is **effect-scheduled and promise-based**, which changes Vitest +
`HttpTestingController` timing:

- The loader runs in a **root effect**, so `fixture.detectChanges()` does *not*
  issue the GET. Call **`TestBed.tick()`** after the action that triggers the
  load and before `httpMock.expectOne(...)`. (This was the bulk of consumer-spec
  churn — one `TestBed.tick()` per call site.)
- The value **commits on a microtask**, so reading `resource.value()`
  synchronously after `flush()` misses it. Drain with the shared
  `settleResource()` helper (`frontend/src/app/testing/settle-resource.ts`:
  `await macrotask` + `TestBed.tick()`).
- **Values do not commit under `fakeAsync` at all** — the virtual clock doesn't
  drive resolution. A fakeAsync spec can still *flush* the request to satisfy
  `verify()`, but any spec asserting a resource-derived value must be a
  real-async (`async`/`await`) test.
- Match trigger-gated / query-string reads by an `r.url` predicate, not a plain
  `expectOne(string)` (e.g. export-modal's labels GET carries `?enrich=true`).
- At **runtime** none of this matters: zone change detection flushes the loader
  effect immediately after the load, so the GET fires as before. Timing change
  is test-only.

## What shipped

**Phase 1 (pilot) — `SettingsStateService`**
(`frontend/src/app/services/settings-state.service.ts`): reimplemented on
`rxResource` wrapping `SettingsApiService.getSettings()`, monotonic-counter
request trigger, `animations-off` document side-effect moved to an `effect()`.
Pilot initially kept compat shims (sync getter + `settings$` `toObservable`
bridge) so its ~16 consumers were untouched.

**Phase 2 — `MediaStateService` + shim removal**: media-stub list
(`GET /api/medias/ids`) migrated to `rxResource` (same monotonic-counter
trigger; `mediasSignal()` / `isLoading()`; `selectedId` a plain `signal`;
`selectedMedia`/`getMedia()` stay synchronous accessors). Pilot's
`SettingsStateService` shims (`settings$`, sync `settings` getter,
`BehaviorSubject`s, `OnDestroy`/`destroy$`) **removed**. ~18 consumer files
across both services migrated from `.subscribe()` to constructor `effect()`
reading the signals (templates too).

**Phase 3 — left-panel reads + housekeeping**: `LeftPanelComponent`
media-type / embedder `ngOnInit` subscribes → two **eager** `rxResource`s
(metadata as `computed`s, derived `mediaTypeName`/`textSortAvailable` recomputed
by constructor `effect()`s). Shared `settleResource()` test helper extracted to
`frontend/src/app/testing/settle-resource.ts`. Dead `DetectorStateService`
(no external consumers) deleted with its spec; the
`ProcessorsApiService.getAutorunExtractors`/`getAutorunLocalizers` wrappers stay.

**Phase 4 — picker-modal reads**: four importer/exporter picker modals moved
eager list reads to `rxResource` following the Phase 3 pattern —
`LabelExporterModalComponent` (`getExporters()`; dropped its whole
`destroy$`/`OnDestroy`), `SettingsImporterModalComponent` /
`SettingsExporterModalComponent` (`listImporters()`/`listExporters()`;
`OnDestroy` stays for a success-message timer), and `ExportModalComponent` — all
three init reads: eager `getStatus()` + `getExporters()`, plus the input-derived
`exportLabels(...)` on a **trigger signal** (`labelsReady`) with `buildColumns`
moved to a constructor `effect()`. Added an `export-modal` spec (was untested);
updated the `label-exporter` spec for loader timing.

## Behaviour changes (backwards-compat notes)

- `SettingsStateService` / `MediaStateService` no longer expose Observable
  surfaces (`settings$`, `medias$`, `loading$`, `selectedId$`) or sync getters —
  callers read signals (`settingsSignal()`, `mediasSignal()`, `selectedId()`).
- `DetectorStateService` removed.
- Effect-driven side effects (medias→media-type sync, deferred autopilot sort)
  are now microtask-scheduled rather than synchronous.

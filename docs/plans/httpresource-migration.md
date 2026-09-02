# `httpResource` / reactive-resource migration for the data layer

Status: Remaining work is expanding the `rxResource` read-path migration to more
services and components; pollers and `forkJoin` aggregates stay deferred by
design. Open follow-ups below.

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

<!-- item-sep -->

- **Expand to more read services** (the standing next step). Already done for
  `SettingsStateService`, `MediaStateService`, the left-panel media-type/embedder
  reads, and the importer/exporter picker modals. A still-open refinement: if media-types /
  embedders ever need to **re-fetch on dataset switch**, give the resource a
  `toSignal(activeContext.datasetId$)` request key instead of the eager load —
  today the component is recreated on switch, so eager-once suffices.

<!-- item-sep -->

- **Remaining component-local read subscribes** (not yet converted). The next
  candidates after the picker modals are heavier: `dataset-importer-modal`
  (interdependent clipper/embedder/demo loads + dynamic field-option fetches)
  and `label-importer-modal` (importer list is a clean read, but field-options
  and imports are mutations). Convert the clean list reads when those modals are
  next touched; leave the dynamic field-option fetches imperative.

<!-- item-sep -->

- **Pollers and `forkJoin` aggregates** (`VoteStateService`, labeling status,
  `DatasetStateService.refresh()`) are still imperative by design. Resources
  fetch-on-request-change, not on an interval, and don't map cleanly onto
  aggregate composition. Options if revisited: drive a resource's reload from a
  timer signal (pollers), or a resource per call composed via `computed`
  (aggregates). Revisit only if a resource clearly wins — decide per call site;
  don't force it.

<!-- item-sep -->

### Decision points still open

- **Depth of signal adoption**: thin `toSignal` bridges (minimal, keep
  `BehaviorSubject` services) vs converting hot state services to signal-backed
  (bigger, cleaner). Started thin (recommended); convert services
  opportunistically. The app is already zoneless
  (`provideZonelessChangeDetection()`, `zone.js` dropped end to end), so every
  remaining `BehaviorSubject` read path is change-detected via its `toSignal`
  bridge rather than by a zone tick — which raises the value of converting the
  hot ones, and makes a missed bridge a stale-view bug rather than a slow one.

### Out of scope (record only)

- Mutations, SSE, polling intervals (above).
- Replacing `ng-openapi-gen` — the generated client stays; `rxResource` wraps it.

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
  the matching `takeUntilDestroyed()` / `DestroyRef` / `OnDestroy` plumbing is
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
- **A virtual clock does not commit values.** Advancing Vitest fake timers
  flushes the *request* but not the resource's promise resolution, so any spec
  asserting a resource-derived value must `await` a real macrotask (that is what
  `settleResource()` does) rather than relying on timer advancement alone.
- Match trigger-gated / query-string reads by an `r.url` predicate, not a plain
  `expectOne(string)` (e.g. export-modal's labels GET carries `?enrich=true`).
- At **runtime** none of this matters: the zoneless scheduler flushes the loader
  effect as part of the change-detection pass that follows the load, so the GET
  fires as before. The timing change is test-only.

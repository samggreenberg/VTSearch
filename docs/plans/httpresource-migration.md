# `httpResource` / reactive-resource migration for the data layer

Status: **Phase 1 (pilot) shipped.** The `SettingsStateService` read path now
runs on `rxResource`; the pattern and its test story are proven (see "What
shipped" below). Remaining read services are deferred — see "Open follow-ups".
Scoped after the Angular 21 + Vitest work landed (see `angular-21-upgrade.md`,
which lists this as a deferred carrot). This doc decides *how* VTSearch should
adopt Angular's resource primitives and in what order.

## What shipped (Phase 1 pilot)

`SettingsStateService` (`frontend/src/app/services/settings-state.service.ts`)
was reimplemented on `rxResource` while keeping its public surface
(`settings`, `settings$`, `load`, `update`, `clear`) intact, so none of its ~16
consumers changed. The concrete, repeatable pattern:

- **Wrap the existing generated-client read in `rxResource`.** The loader's
  `stream` calls the existing typed method (`SettingsApiService.getSettings()`),
  so the generated client and the interceptor chain are untouched.
- **A monotonic counter signal is the request.** `params: () => count === 0 ?
  undefined : count` keeps the resource idle until the first `load()`
  (`undefined` request = no fetch); each `load()` bumps the counter to refetch.
  This maps an imperative "load/refresh on demand" trigger onto a reactive
  resource.
- **Signals are the new canonical API** (`settingsSignal`, `isLoading`,
  `error`), exposed from the resource. Backwards-compat shims bridge to existing
  consumers: the sync getter reads the signal; `settings$` is
  `toObservable(settingsSignal)` (replays latest like the old `BehaviorSubject`,
  but emits asynchronously).
- **Mutations stay imperative.** `update()` is still a `HttpClient` PUT; it
  writes the server's response back into the resource via `resource.set(...)`.
  `clear()` is `resource.set(undefined)`.
- **Document side-effects move to an `effect()`** (here, the `animations-off`
  class) that watches the settings signal, replacing the old manual `emit()`.

### Testing `rxResource` (important — this is the real cost)

The resource's loader is **effect-scheduled and promise-based**, which changes
test timing. Concretely, for the Vitest + `HttpTestingController` suite:

- The loader runs in a **root effect**, so `fixture.detectChanges()` does *not*
  issue the GET. Call **`TestBed.tick()`** after the action that triggers
  `load()` and before `httpMock.expectOne(...)`. (This is the bulk of the
  consumer-spec churn: ~40 call sites across the settings consumers each needed
  one `TestBed.tick()`.)
- The value **commits on a microtask** (the loader is promise-based), so reading
  `resource.value()` synchronously after `flush()` misses it. In a normal
  (non-fakeAsync) test, drain with `await new Promise(r => setTimeout(r))` then
  `TestBed.tick()`.
- **`rxResource` values do not commit under `fakeAsync` at all** — the virtual
  clock doesn't drive the resource's resolution (looping `tick()/TestBed.tick()`
  does not help). A fakeAsync spec can still *flush* the request to satisfy
  `verify()`, but any spec that asserts a **settings-derived value** must be
  converted to a real-async (`async`/`await`) test. Only one right-panel test
  needed this; the rest only flush the request.
- At **runtime** none of this matters: zone change detection flushes the loader
  effect immediately after `load()`, so the GET fires as before. The timing
  change is test-only.

## Goal

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
*read-path* refactor, adoptable incrementally — never a big-bang rewrite.

## Current state (as of this plan)

- **Generated client**: `ng-openapi-gen` (config `apiService: false`) emits
  standalone **Observable-returning functions** under
  `src/app/generated/api-client/fn/**`, e.g.
  `getVotes(http, rootUrl, params, ctx): Observable<StrictHttpResponse<T>>`.
- **~26 hand-written `*-api.service.ts`** wrap those functions into typed,
  `Observable`-returning methods (e.g. `SortingApiService.getLabelingStatus()`).
  ~60 GET-style read methods across them.
- **~29 state services** (`*-state.service.ts`) hold UI state in
  `BehaviorSubject`s and orchestrate fetches/polling (e.g. `VoteStateService`
  polls `/api/votes` via `timer(0, ms)`; `DatasetStateService` `forkJoin`s the
  registries).
- **Consumption is imperative, not reactive**: **66** `.subscribe()` call sites
  in non-spec code vs **2** `| async` pipes. Components subscribe in `ngOnInit`
  and write to plain fields. So this migration is as much about *consumption*
  (template-level signal reads) as about the services.
- **HTTP wiring**: `provideHttpClient(withInterceptors([activeContext,
  achievementsRefresh, error]))`. The `activeContext` interceptor injects
  `X-Dataset-Id` / `X-Detector-Id` on every request from
  `ActiveContextService`; `error` is the circuit-breaker/connection-state
  chokepoint. **`ActiveContextService` is `BehaviorSubject`-based** (no signals).
- **Signals today**: essentially unused (1 component). Zone-based change
  detection is still on.

## Key design decision: `rxResource` over raw `httpResource`

Angular 21 ships two relevant primitives:

- **`httpResource(() => url | request)`** — issues the GET itself given a URL
  (or request object). It **bypasses the generated client**: you'd hand-build
  URLs/params that `ng-openapi-gen` already generates, duplicating the typed
  param/response contracts and losing the OpenAPI-snapshot guarantee.
- **`rxResource({ params, stream })`** (from `@angular/core/rxjs-interop`) —
  wraps an **Observable loader**. The loader can call the existing typed
  `*-api.service.ts` methods unchanged, so we **keep the generated client** and
  its types, and just gain signal-driven reactivity + `value()/status()/error()`.

**Recommendation: lead with `rxResource`** wrapping existing api-service
methods. It is the incremental-friendly choice — no generated-client churn, no
URL duplication, interceptors still apply (same `HttpClient`). Reserve
`httpResource` only for genuinely new raw reads where no generated function
exists. (Both share `provideHttpClient` + the interceptor chain, so headers,
achievements-refresh, and the error/circuit-breaker all keep working either
way — verified against the current functional-interceptor setup.)

## Obstacles to resolve before/during rollout

1. **Signal adoption is a prerequisite.** Resources are signal-driven, but the
   reactivity sources (`ActiveContextService`, sort/vote/settings state) are
   `BehaviorSubject`s, and consumers `.subscribe()` rather than read signals.
   Two sub-decisions:
   - Bridge the existing `Observable`s to signals with `toSignal(...)` at the
     edges (cheap, non-invasive, lets resources key off
     `toSignal(activeContext.datasetId$)`), **or** convert the hot state
     services to `signal()`-backed (larger, cleaner long-term). Start with
     `toSignal` bridges; convert services opportunistically.
   - Resources must be created in an **injection context** (field initializer
     or `inject()`-time). Polling/orchestration that lives in services needs the
     resource owned by the right scope.
2. **Polling endpoints don't map to resources.** `VoteStateService` /
   labeling-status poll on a timer. Resources fetch-on-request-change, not on an
   interval. Options: keep pollers imperative, or drive a resource's reload from
   a timer signal. Recommend leaving pollers alone in phase 1.
3. **`forkJoin` aggregates** (e.g. `DatasetStateService.refresh()`) — a resource
   per call composed via `computed`, or keep the aggregate imperative. Decide
   per call site; don't force it.
4. **Experimental API.** `httpResource`/`rxResource` are still marked
   experimental in 21; the signature may shift in 22/23. Keep the surface small
   and centralized (a thin helper) so an API change is a one-file fix.
5. **Error semantics.** Today errors flow through `errorInterceptor` (toasts,
   circuit breaker, connection state). Resources surface errors via `error()`
   instead of an Observable `error` callback — confirm the interceptor-driven
   side effects still fire (they do, since the request still goes through the
   chain) and decide how components render `error()` vs the current toasts.
6. **Test story.** The Vitest suite drives `HttpTestingController` +
   `fakeAsync`. Resources resolve on microtasks/effects; spec patterns will need
   a small, documented helper (flush + `await`/`tick` + read `value()`), since
   the current specs assert on imperative subscribe results.

## Suggested sequencing (incremental, reversible)

1. **Pilot one read** end-to-end — **done** (`SettingsStateService`, see "What
   shipped"). The pattern and its test story are proven.
2. **Write the pattern down** — **done** (the "What shipped" + "Testing
   `rxResource`" sections above are the reference for the next conversions).
3. **Expand by area**, one PR per service/feature, only where it removes real
   boilerplate. Leave pollers, `forkJoin` aggregates, and mutations alone unless
   a conversion is clearly a win.
4. **Revisit consumption**: as reads become resources, prefer template signal
   reads / `@if (resource.value(); as x)` over `.subscribe()`-into-field.

Each step is independently shippable and revertible; there is no point where the
app must be half-migrated.

## Open follow-ups

- **Expand to more read services** (step 3). The settings pilot kept the old
  public API for zero consumer churn; future conversions can go further and
  expose the resource signals directly to consumers (then drop the
  `settings$`/getter shims) where it removes real boilerplate. Natural next
  candidates: context-keyed list reads (media-types, embedders) where a
  `toSignal(activeContext)` request key earns the reactivity.
- **Pollers and `forkJoin` aggregates** (`VoteStateService`, labeling status,
  `DatasetStateService.refresh()`) are still imperative by design; revisit only
  if a resource clearly wins.
- **A shared test helper** for the `TestBed.tick()` / real-async drain dance was
  not extracted yet (the settings specs inline it). Extract one once a second
  service is converted and the shape is confirmed to repeat.

## Decision points needing the user

- **Depth**: thin `toSignal` bridges (minimal, keep `BehaviorSubject` services)
  vs converting hot state services to signal-backed (bigger, cleaner). Recommend
  starting thin.
- **Relationship to zoneless.** Going zoneless (the other deferred carrot)
  wants broad signal adoption too; this migration is a natural on-ramp. Decide
  whether to treat them as one track or keep separate.
- **Pilot target** — which first read to convert (media-types vs settings vs an
  active-context-keyed list).

## Out of scope (record only)

- Mutations, SSE, polling intervals (above).
- Replacing `ng-openapi-gen` — the generated client stays; `rxResource` wraps it.
- Going zoneless — its own effort (`angular-21-upgrade.md` → Deferred).

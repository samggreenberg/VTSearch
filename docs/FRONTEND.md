# VTSearch Frontend Architecture

This document maps the Angular single-page app in `frontend/`. It is the
front-end counterpart to [ARCHITECTURE.md](ARCHITECTURE.md), which covers the
Python tiers.

It is a **map, not an inventory.** The SPA holds roughly a hundred components and
seventy-five services; enumerating them here would be a list that rots on the next
commit. Instead this document explains the handful of mechanisms everything
else is built on — the change-detection model, the service layer, the
active-dataset/detector context, the generated API client, and the component
composition conventions — so that any given file can be placed by reading its
neighbours.

**What lives elsewhere:**

| For | See |
|-----|-----|
| Build commands, dev server, Angular upgrade mechanics | [`frontend/README.md`](../frontend/README.md) |
| SCSS tokens, shared classes, copy style | [style-guide.md](style-guide.md) |
| REST endpoint reference | [API.md](API.md) and [api/](api/) |
| Backend state model, per-dataset contexts, auth | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Modal back-button rules, desktop-only scope, budget policy | [`CLAUDE.md`](../CLAUDE.md) |

## Table of contents

1. [Shape of the app](#1-shape-of-the-app)
2. [Directory layout](#2-directory-layout)
3. [Feature areas and their boundaries](#3-feature-areas-and-their-boundaries)
4. [The service layer](#4-the-service-layer)
5. [Reactivity: the zoneless change-detection model](#5-reactivity-the-zoneless-change-detection-model)
6. [The active dataset/detector context](#6-the-active-datasetdetector-context)
7. [Talking to the backend](#7-talking-to-the-backend)
8. [Component composition conventions](#8-component-composition-conventions)
9. [Styling](#9-styling)
10. [Testing](#10-testing)
11. [Build and serve](#11-build-and-serve)
12. [Invariants](#12-invariants)

---

## 1. Shape of the app

Angular 21, TypeScript 5.9, **standalone components** (no `NgModule`s),
**zoneless change detection** (no `zone.js` anywhere — app, tests, or
`package.json`), built by the esbuild-based `@angular/build:application`
builder. Desktop-only: there are no responsive breakpoints and none should be
added.

Bootstrap is three files deep:

```
src/main.ts            bootstrapApplication(AppComponent, appConfig)
src/app/app.config.ts  the entire provider graph (below)
src/app/app.component.ts  persistent chrome + <router-outlet>
```

`app.config.ts` is the whole composition root:

```ts
providers: [
  provideZonelessChangeDetection(),
  provideRouter(routes),
  provideHttpClient(withInterceptors([
    timezoneInterceptor,        // X-Timezone-Offset
    activeContextInterceptor,   // X-Dataset-Id / X-Detector-Id
    achievementsRefreshInterceptor,
    errorInterceptor,           // toasts + offline circuit breaker
  ])),
]
```

Every service is `@Injectable({ providedIn: 'root' })` unless it is
deliberately component-scoped (see
[§8](#8-component-composition-conventions)), so there is no other provider
list to keep in sync.

`AppComponent` owns everything that outlives a route: the header (burger menu
of recent sessions, dataset/detector pulldowns, achievements/help/settings
buttons), the toast container, the offline banner, the dialog host, the
achievement-unlock host, and the app-level modals (Settings, Achievements,
Keyboard help, Add Dataset, New Detector, Find Results). Route content renders into
`<router-outlet>` inside `.main-content`.

**Routes** (`app.routes.ts`) are all lazy (`loadComponent`) and encode the
active context in the URL:

| Path | Guard | View |
|------|-------|------|
| `/dashboard` | — | Dataset & detector management |
| `/label/:datasetId/:detectorId` | `activeContextGuard` | Labeling / training |
| `/tune/:datasetId/:detectorId` | `activeContextGuard` | Tune (`FindViewComponent`) |
| `/browse/:datasetId` | `browseContextGuard` | VTSBrowse projection canvas |

Bare `/label`, `/tune`, `/browse` (and anything unmatched) redirect to
`/dashboard`: a half-specified pair is not a representable state, so there is
nothing to render. The legacy `/find/…` and `/test/…` pair paths redirect to
`/tune/…`.

---

## 2. Directory layout

```
frontend/
├── angular.json            Builder wiring, production budgets, the dedicated `test` build config
├── ng-openapi-gen.json     Generated-client config (output: src/app/generated/api-client)
├── openapi.json            Committed snapshot of the Flask spec (see §7)
├── proxy.conf.json         Dev server: /api and /static → localhost:5000
├── vitest.config.ts        isolate: false — the cascade guard is in test-setup.ts (see §10)
├── docs-assets/            Symlink to docs/user/ — the in-app user guide is served from here
├── public/                 Favicons, logo, Toasty's faces (copied verbatim into the build output)
├── scripts/
│   ├── openapi-gen-cached.mjs   Hash-stamped wrapper around ng-openapi-gen
│   └── build-stamp.mjs          Writes generated/build-stamp.ts (the bundle's version, §7)
└── src/
    ├── main.ts, index.html, test-setup.ts
    ├── styles.scss         Global stylesheet entry; @use's the scss/ partials
    ├── scss/               _variables (design tokens), _layout, _components,
    │                       _data-table, _picker-shared — global, class-only rules
    └── app/
        ├── app.component.*, app.config.ts, app.routes.ts
        ├── components/     One directory per component; feature areas nest (see §3)
        ├── services/       API services, state services, coordination services (§4)
        ├── guards/         Route guards that resolve the URL pair into context (§6)
        ├── interceptors/   The four HTTP interceptors (§7)
        ├── directives/     Cross-cutting DOM behaviour (`no-focus-steal`, `panel-resize`)
        ├── models/         Hand-written types (§7): api.models.ts, projection.models.ts
        ├── utils/          Pure functions — no Angular DI, trivially unit-testable
        ├── testing/        Shared TestBed fragments and zoneless helpers (§10)
        └── generated/      **gitignored**; api-client/ (from openapi.json) and
                            build-stamp.ts, both rewritten on prebuild/pretest
```

Two rules keep this navigable:

- **A component's directory holds its children.** `dashboard/` contains
  `dataset-card/`, `new-detector-modal/`, `combine-datasets-modal/`, …
  because nothing outside the Dashboard uses them. A component used by two
  feature areas moves up to `components/`.
- **`utils/` is Angular-free.** Anything in there is a pure function over
  plain data (clip windows, progress formatting, grid-icon sizing, keyboard
  shortcut matching). If it needs `inject()`, it is a service, not a util.

---

## 3. Feature areas and their boundaries

Four routed views, plus the shared chrome. Their boundaries are worth knowing
because they determine which service a new piece of state belongs to.

### Dashboard (`components/dashboard/`)

The largest area by code volume. Dataset and detector cards, the sortable
managed-column tables behind them, per-row loading progress, and the entry
points into every creation flow (Add Dataset, New Detector, Combine…). It is
deliberately a **thin layout/wiring shell**: its state was lifted into
singleton services so the top-bar pulldowns can mirror it without depending on
the component —

- `DashboardColumnsService` — the two `ManagedColumns` instances (sort/visibility).
- `DashboardModalsService` — which row-action modal is open, for what target.
- `DashboardLoadingTasksService` — per-task loading rows, and the
  poll-until-settled-then-refresh bookkeeping.
- `DashboardSelectionService` — the highlighted table rows (which drive
  Train / Tune / Find / Combine / Delete), *owned* here rather than mirrored from the
  component, so the top-bar pulldowns show what is *selected* while the
  Dashboard is on screen (not merely what is loaded) by reading signals
  directly, and the selection survives a round trip to another view. Both
  grids share one `toggle(kind, id, additive)` ladder, and the ids the top bar
  reads are `computed` over the selection and the registry — the pulldown
  needs no push, and there is no per-mutation mirror call to forget.

The Add-Dataset and New-Detector flows are *not* owned by the Dashboard at
all: `NewThingFlowsService` is a singleton opener, and the modals are rendered
by `AppComponent` inside `@defer` blocks, so any surface can start those flows.

### Label view (`components/label-view/`)

The training loop. A three-panel layout (`vt-left-panel`, `vt-center-panel`,
`vt-right-panel`) with draggable dividers (the shared
`directives/panel-resize.directive.ts`). The three panels are shared with the
Tune view:

- **Left** — media list (virtual scroller), sort bar, the Threshold
  (`vt-balance`), stripe overview, select mode, and the **Autopilot panel**
  that drives the automated vote → train → re-sort loop.
- **Center** — the media viewer, one child per media type (image, text, video,
  audio, document) plus the voting overlay.
- **Right** — labels, labelsets, vote grid, and the detector context bar.

**Component-provided services.** The view's non-chrome work lives in services
provided on `vt-label-view`, never `providedIn: 'root'`: their requests are
cancelled by the *component's* pair scope, which a singleton has no way to
name (see the note on `SortStateService` in `PairScopeService.seedBeta`).

- `PairScopeService` — the pair's lifetime and the reset that ends it.
- `SortRunnerService` — the sorts, the `autoSelectNext` each one ends on, the
  balance's write path, and the typed query's hold (#4736).
- `SpotCheckRunnerService` — Autopilot's Spot Check step, run inside the
  ordinary vote flow (#5136).
- `LabelViewPanelStateService` — per-media-type panel preferences.

The component keeps one-line forwarders for the handlers its template binds,
so what is left in `label-view.component.ts` is panel chrome, Autopilot's
phase wiring (including the effects that start a spot check) and the re-sort
prompt. What outlives the view is root: `SortStateService` (the ranking, the
line, the balance), `VoteStateService`, `AutopilotStateService` (the phase
machine) and `PanelHideStateService` (the side-panel folds, shared with Tune).

**Ordering rules.** These keep what the view serves independent of which
request answers first (#4318):

- **A newer sort ends an older one**, so the ranking on screen answers the
  sort asked for last. The backstop that ranks a pair nothing else ranked
  (`seedRankingIfUnranked`) reads the kind of the sort started last
  (`SortRunnerService.newestSortKind`), not `sortMode`.
- **The centre is the view's until the user acts.** On entry or after a pair
  switch the centre item is re-picked against every ranking that lands, until
  the user clicks, votes or steps; from then on no re-rank moves it.
- **Balance writes are `switchMap`-ed and pair-scoped**, so a balance the user
  moved past can never land last. A learned re-sort after a balance change
  starts from the POST's response, never beside it: the learned sort caches by
  balance server-side.
- **The Threshold's two values travel separately.** `SortStateService.beta` is
  what the radios show; `SortStateService.balance` is the state of the line on
  screen and only arrives *with* that line.
- **Autopilot asks from `askWindow()`**: the learned ranking held behind the
  typed query's set while there is one, else the ranking on screen. The line,
  count and list the user sees never read the held ranking.
- **A folded right panel is unmounted** (#4673), so the views, not
  `vt-right-panel`, own the votes poll.

**Where the behaviour is written down:** the Threshold's states, presets and
wording in `vt-balance` and `utils/line-balance.ts`; the balance write path,
the line redraw and the hold in `SortRunnerService`; the spot check's
lifecycle in `SpotCheckRunnerService`, when it is due in label-view's
constructor effects and `AutopilotStateService.doneChecked`; the folds in
`PanelHideStateService`.

### Tune view (`components/find-view/`)

The Dashboard's **Tune** button opens it at `/tune/:datasetId/:detectorId`.
Its code keeps an older name, as the Train view's keeps `label`: it was Find
until #4525 and Test until #5098, hence `FindViewComponent`,
`panelMode: 'find'`, `VoteStateService.findMode`, and on the API side
`/api/find-label`, `find_mode` and the `find` SSE channel (the legacy paths
redirect, [§1](#1-shape-of-the-app)). **Find** is now the Dashboard's third
button, which opens no view: it starts a background AutoFind run
(`AutoFindService.find`), and its Find Results dialog hands items to Browse
through the same `BrowseSubsetPrepService` Tune's Browse uses.

The view scores the pair's dataset with its detector (`POST /api/find-label`)
and has two tabs in the left panel (`findTab`, owned by the view because the
centre and right panes switch with it; #4524): **Autopilot**, the Tune
autopilot, and **Review**, the ranked result in full (work queue, boundary
walk, Verified Good / Verified Bad piles, To Dataset / Export / Browse).

**The Tune autopilot** tests the line on the corpus in front of the user. The
server owns the sample and derives the phase on every read (`line_phase`;
statistics in `vtscore/training/thresholds/line_test.py`, routes in
`vtsearch/routes/line_test.py`), so nothing in the frontend accumulates it.
Its panes are `vt-line-test-panel` (left, projected as `[findAutopilot]`),
`vt-line-test-stage` (centre, and the verdict with its three exits at Done)
and `vt-line-test-result` (right). The ranked list is not shown while a test
runs: a pick is drawn from a band, and the list would show its rank (#4267).

**Component-provided services:** `PairScopeService` and
`LineTestSessionService` (the last `/api/line-test` response plus the round as
dealt), so the test lives exactly as long as the pair.

**Ordering rules:**

- **Test replies can land out of order.** Each is stamped with its request's
  sequence, and an older one never overwrites a newer.
- **The balance is frozen within a phase** (`LineTestSessionService.locked`).
  Tune runs no spot check, and its Threshold's state line is the host's
  (`lineState`), never Train's check range.
- **Tune's balance write** (`betaRequests$`) is debounced and `switchMap`-ed:
  only the newest POST can install a line.
- **The vote piles are read as `?scope=verified`** (#5115) and polled only
  while Review is up (`syncPiles`). Unverified items are placed by the frozen
  ranking and the cutoff, never from `/api/votes`.
- **One `MediaPrefetchService` effect** warms the next images of whichever
  queue the tab votes through (`LineTestSessionService.upcoming` on
  Autopilot, the boundary walk on Review).

### Browse view (`components/browse-*`)

VTSBrowse: a UMAP projection rendered on a canvas as a hex-tile pyramid. This
area is unusual and mostly self-contained — `browse-canvas` is the single
largest component in the app because it owns the render loop. Its state is
split across small services rather than living in the canvas:
`BrowseViewportService` (visible region, shared with the minimap) and
`BrowseSelectionService` (selection at *item* granularity, so it stays coherent
as bins split and merge across zoom levels) — both provided on `browse-view`,
not root — plus the root `TileCacheService`,
`BrowsePrepService` (load + project before navigating), and
`ProjectionApiService`.

### Shared chrome

`AppComponent`'s header, `context-pulldown` (the dataset and detector
pickers, plus the incompatible-pair explainer), `toast-container`,
`dialog-host`, `offline-banner`, `achievement-*`. Anything here must work on
every route.

**Where does new code go?** If it is used by one view, nest it under that
view's directory. If two views need it, promote it to `components/` and give
its state a root service. If it is state two *services* need, it belongs in a
service of its own — the pattern the Dashboard and Browse areas already
follow.

---

## 4. The service layer

`services/` holds three distinct kinds of file. The suffix tells you which.

### `*-api.service.ts` — thin typed HTTP wrappers

One method per endpoint, returning an `Observable` of a generated type. They
hold **no state**. Each wraps a function from the generated client rather than
hand-building a URL, unless the spec cannot type the call; those stay on plain
`HttpClient` with a comment saying why (a plugin-field route whose body is the
plugin's, a multipart upload, a blob download, the Browse projection
endpoints):

```ts
getMediaIds(): Observable<MediaIdsListResponse[]> {
  return listMediaIds(this.http, this.config.rootUrl).pipe(map((r) => r.body));
}
```

They are split to roughly mirror the backend's route modules
(`vtsearch/routes/datasets/{listings,registry,ui}.py` →
`datasets-{listings,registry,ui}-api.service.ts`, and so on). The mirror is
approximate and does not need to be exact; the split exists to keep any one
file readable.

### `*-state.service.ts` — client-side state

The stores the views bind to: `MediaStateService` (the dataset-wide media stub
list and current selection), `VoteStateService` (the per-pile vote sets, undo
and redo), `SortStateService`, `SettingsStateService`, `DatasetStateService`
(the registry listing), `LabelsetStateService`, `AutopilotStateService`.

They own state and the reactive surface over it; they call API services for
I/O. A component should almost never call an API service *and* keep the result
in a field that another component also needs — that is a state service.

### Coordination services — everything else

The ones worth knowing before changing anything:

| Service | Responsibility |
|---------|----------------|
| `ActiveContextService` | The active (dataset, detector) pair, split into intent/active layers ([§6](#6-the-active-datasetdetector-context)) |
| `ContextSwitchService` | Drives a pair change end to end: intent → load → promote to active |
| `ActiveContextWatcherService` | Reacts when an active id disappears from the registry (deleted elsewhere) |
| `PairRecoveryService` | Loads the active pair again when the backend has lost it (a restart), so an open Train, Tune or Browse view comes back instead of 409ing; sends a lost detector-positives map back to the Dashboard ([§6](#6-the-active-datasetdetector-context)) |
| `ProgressEventsService` | The single `EventSource` over `/api/events`, fanned out per channel |
| `ConnectionStateService` | Online/offline, and the probe that reopens the circuit breaker |
| `ToastService` | Toasts: four levels, the structured `ErrorContext` from failed requests, and the backend's `notification` channel |
| `VtDialogService` | `confirm()` / `prompt()` as promises, rendered by `dialog-host` |
| `NewThingFlowsService` | Singleton openers for the Add-Dataset / New-Detector flows |
| `AutoFindService` | The end of a background AutoFind: opens the results of a Find this tab started (the Dashboard's big **Find** button, one run per ticked dataset) unless the dialog already shows another, toasts any other run of the user's (an import's, a dataset's ⋯ **Run AutoFind**, or a Find that landed while the dialog was open) with a **View Results** action, and holds the run the app-root Find Results dialog shows. AutoFind is the unattended path and never opens the dialog on its own (#4615) |
| `MediaMetadataCacheService` | Lazy batched fetch of full metadata for whatever is in the viewport |
| `PairScopeService` | **Component-provided** (`find-view`, `label-view`): the active pair's lifetime, the `scoped()` teardown operator, and the pair-change reset in its one correct order |
| `SortRunnerService` | **Component-provided** (`label-view`): runs the sorts — text, learned (with its job poll), detector, example — and advances the selection they end on. Lives beside the view rather than on the root-singleton `SortStateService` because every call in it is torn down by `pairScope.scoped()` |
| `KeyboardService`, `ThemeService`, `AchievementsService`, `AuthService` | App-wide concerns wired in `AppComponent` |

`adaptive-poll.ts` is not a service but a shared operator; see
[§7](#7-talking-to-the-backend).

---

## 5. Reactivity: the zoneless change-detection model

**This is the section to read before touching any state.**

The app runs `provideZonelessChangeDetection()` and `zone.js` is gone
end-to-end. There is no monkey-patched `setTimeout`, no zone tick after an
HTTP callback, no implicit "something happened, re-render everything". Angular
refreshes a view only when it is *notified*, and the notifications are:

1. A **signal** read during that view's template evaluation is written.
2. A **template-bound event** fires (`(click)`, `(widthChange)`, …).
3. `AsyncPipe` emits, or an `@Input`/signal input changes.
4. Something explicitly calls `markForCheck()` / `ApplicationRef.tick()`.

Anything else that mutates rendered state produces a **stale view** — the
value changed, the DOM did not. That is the characteristic frontend bug in
this codebase, and it is silent.

### The rules

- **State that a template reads must be a signal**, or must reach the template
  through `AsyncPipe` / `toSignal`. Writing a plain field from an HTTP
  continuation, a timer, an `EventSource` callback, or a promise `.then()`
  notifies nobody.
- **Getter-over-signal is the sanctioned migration shape.** Several state
  services expose `get sortBusy() { return this._sortBusy(); }` over a private
  signal, so existing `sortState.sortBusy` bindings stayed byte-identical while
  becoming reactive. A signal read through a getter during template evaluation
  *is* tracked as a dependency of that view —
  `testing/getter-signal-zoneless.spec.ts` pins this.
- **Per-media-type settings prefs go through `SettingsStateService.perMediaType`.**
  A `{media_type: value}` settings key bound to a media-type signal returns a
  `computed` value plus a merge-preserving setter; do not hand-roll a shadow
  `Record` field kept in sync by `effect()`s. The value is a real signal (a
  template binding on it repaints on its own), and there is no mirror to go
  stale when a key disappears server-side. The setter **merges**; a setter
  that replaced the dict would wipe every other media type's preference. It takes a plain key, or a *signal* of a key where
  the key itself varies (`vt-view-controls` picks `focus_mode_left` /
  `_right` / `_popup` from its `side` input). `LabelViewPanelStateService` is
  the reference use; `find-view`, `browse-view`, `browse-bin-popup`,
  `browse-selection-panel`, `right-panel` and `view-controls` are the other
  consumers.

  Two notes on where the seams are. **The Settings modal is deliberately not a
  consumer**: it edits a *draft* settings signal across *every* media type via
  a `typeId` loop, not "the current one", so the helper's shape does not fit.
  And a preference whose control derives its next value from the current one —
  the `vt-view-controls` size buttons, which also grey out at the ends of the
  ladder — keeps a **local optimistic signal** written on click, with the
  preference behind it; binding straight to the preference would make a rapid
  second click read a pre-round-trip value.

- **Consumers use `effect()`, not `subscribe()`.** A constructor `effect()`
  reading a signal auto-disposes with the component, which is why the
  teardown plumbing has been dropped wherever it was the last user.
- **Where a subscription is still needed, tear it down with
  `takeUntilDestroyed()`** — see "Subscription teardown vs. cancellation"
  below for the one distinction that decides it.
- **Reads go through `rxResource`.** The read path is being migrated onto
  Angular's reactive resource primitives (`SettingsStateService`,
  `MediaStateService`, several picker modals so far). `rxResource({ params,
  stream })` wraps the *existing* generated-client method, so the typed client
  and interceptor chain are untouched and the service's public surface becomes
  a value signal (e.g. `settingsSignal`, a `computed` over `resource.value()`)
  plus `isLoading()` / `error()`. Prefer it over raw
  `httpResource`, which would bypass the generated client. The remaining
  conversion recipe and the list of still-open call sites live in
  [`docs/plans/httpresource-migration.md`](plans/httpresource-migration.md).
- **Mutations, SSE and pollers stay imperative** — POST/PUT/DELETE remain
  plain `HttpClient` calls; write the server's response back into the store.
  This is a deliberate boundary, not an unfinished migration.
- **Every component is `ChangeDetectionStrategy.OnPush`.** All of them, with
  no exceptions today. A new component should be too.
- **Callbacks from outside Angular must write a signal.** The SSE pump, canvas
  RAF loops, and the drag handlers that deliberately run outside Angular
  (`directives/panel-resize.directive.ts`) all reach the UI either by writing a
  signal or by emitting through a template-bound output. `NgZone.run()` is a
  no-op here and is not the fix.
- **A `BehaviorSubject` service is not wrong, but its bridge is load-bearing.**
  Roughly a third of the services still hold RxJS subjects; each is
  change-detected via its `toSignal`/`AsyncPipe` bridge. A missed bridge is a
  stale-view bug, not merely a slow one.

`AsyncPipe`, `toSignal`, and `toObservable` are all in use and all fine. What
is never fine is a rendered value with no notification path.

---

## 6. The active dataset/detector context

The backend resolves per-dataset and per-detector state from the
`X-Dataset-Id` / `X-Detector-Id` request headers (see
[ARCHITECTURE.md § Multi-dataset support](ARCHITECTURE.md#multi-dataset-support)).
The frontend's job is to make sure those headers name a pair the backend has
actually loaded. Five pieces cooperate.

### `ActiveContextService` — two layers, not one

```
intent   what the user just picked (pulldown click, deep link)  → UI affordances
active   what the backend has loaded                            → HTTP headers
```

The split exists to fix a real cascade: tagging requests with the new ids the
instant a pulldown was clicked produced a storm of `409 dataset_not_loaded`
until the load finished. `intent` updates immediately so the pulldown
highlights; `active` lags until a load completes and is promoted explicitly.

- Read the pair as a whole via `pair$` / `intentPair$` (subscribing to the two
  halves separately fires twice for one atomic change).
- `setActivePair()` writes both layers — for cleanup paths only (registry
  watcher, `clear()`, tests).
- `nextRequestId()` / `currentRequestId` implement **latest-caller-wins**: a
  prep step captures the id at start and discards its result if the id has
  moved. Cancelling in-flight work is best-effort; this check is the
  correctness guarantee.

### `ActiveDatasetService` / `ActiveDetectorService` — the ids as *entries*

`ActiveContextService` carries **ids**, on an RxJS layer a `computed` can't
track. Almost every consumer wants the registry *entry* behind the id — its
name, its media type, its embedder — so each one used to redo the same
`datasets.find(d => d.id === …)` by hand. That read whatever happened to be
loaded at call time and never updated when the registry landed a moment later:
the lifecycle gap that left an export filename detector-less when the modal
opened first (#2819).

The two services close it, one per half, with the same five members:

```
activeId      the id the backend has loaded ('' when none)
intentId      the id the user picked
datasetId /   activeId || intentId — name the user's pick immediately
  detectorId  rather than blanking for the duration of a load
dataset /     the registry entry, or null (nothing selected, or the
  detector    registry fetch is still in flight)
datasetName / the entry's name, or ''
  detectorName
```

Reach for these from **components**: a `computed`/`effect`/template read
repopulates on its own once the registry or an in-flight switch settles.

Two places deliberately keep the imperative lookup, and should:

- **Route guards** run once, before activation, and have already awaited the
  registry — there is nothing left to arrive.
- **Pre-reactive services** (`ActiveContextWatcherService`) resolve the pair
  from the arrays their own `combineLatest` emitted; reading a signal instead
  would resolve against a *different* snapshot than the one being handled.

Both still index rather than scan: `DatasetStateService` exposes `datasetById`
/ `detectorById`, `computed` Maps over the two registries. They are the shared
lookup the services are built on, and reading one is signal-tracked exactly
like reading `datasets` — so swapping a `find` for a `.get` is a readability
change, never a reactivity one.

Note what these services are *not* for. A lookup that resolves something other
than the active pair — the Dashboard's multi-select, the id being switched
*to*, an entry from an HTTP response body — is a different question that
happens to share the `find(d => d.id === …)` shape. Use the Maps there if the
predicate is a plain id match; do not route it through the active-context
services.

### `ContextSwitchService` — the only way to *change* the pair

`switchTo()` tags intent, kicks off whatever dataset/detector loads are
needed, and promotes to active only once they settle. It has cancel-and-replace
semantics and a `switching$` observable for loading UI. A failed load must not
promote — that would re-open the 409 cascade.

### `activeContextInterceptor` — the header attachment

Reads the **active** layer only and sets each header when its id is non-empty.
That is the whole interceptor; all the difficulty lives upstream in the
intent/active split.

**Native element requests bypass it entirely.** `<img src>`, `<audio src>`, and
`<video src>` are browser requests, not `HttpClient` requests, so they carry no
headers. Build those URLs with `ActiveContextService.mediaUrl(path, extra?)`,
which appends `dataset_id` / `detector_id` query params from the active layer.

**Reading an item of another dataset.** Show in photo (#4750) shows a face
crop's photo out of the Image dataset the same import produced, in an overlay,
while the Face pair stays active: switching the pair would land on the
incompatible-pair explainer (a face detector cannot pair with an Image dataset)
and reset the face ranking. Such a request names the other dataset **and no
detector**: `DATASET_OVERRIDE` on an `HttpClient` request (the interceptor sets
`X-Dataset-Id` to it and drops `X-Detector-Id`), `mediaUrlIn(datasetId, path)`
for an `<img src>`. Never send the active detector with another dataset's id.
The backend rehydrates a detector's votes whenever a request pairs it with a
dataset other than the one they were derived for
(`ensure_votes_match_active_dataset`), so one such request clears the session
the user is in.

Collapse to photos (#4751) follows the same rule for a whole grid: the left
panel swaps `vt-media-list` for `vt-photo-grid` while `CollapsePhotosService`
is active (switched on, on a dataset where Show in photo is offered), and the
grid draws each photo with `mediaUrlIn(photoDatasetId, …)`. The grouping is the
server's (`GET /api/sort/collapsed`, keyed by `SortStateService.sortToken`, so
Tune's `setSortResults` keeps find-label's token); the grid only pages rows and
leaves out the faces it is told to hide. A face on a photo emits the same
`mediaSelect` / `mediaVote` / `mediaContextRequest` a card does, so the host's
vote flow is unchanged. The Show in photo overlay and the grid both answer the
409 for an unloaded Image dataset with `SiblingLoadService`
(component-provided), which loads it and calls back.

### Routes and guards — the URL is the source of truth

`/label/:datasetId/:detectorId` and `/tune/:datasetId/:detectorId` carry the
pair so reload, share links, and browser back/forward all work.
`activeContextGuard` resolves the URL pair before the view renders: it awaits
the registry fetch (a cold deep-link can arrive first), toasts and redirects to
`/dashboard` for an unknown id, fast-paths when the pair already matches, and
otherwise holds activation on `ContextSwitchService.applyActivePair(...)`. An
*incompatible* pair (mismatched media types) is allowed through — the
`vt-incompatible-pair-explainer` overlay is a legitimate UI state.

`browseContextGuard` is the dataset-only variant, with one carve-out:
ephemeral detector-positive contexts (`__detpos__<id>`) are built server-side,
are deliberately absent from the registry, and skip the checks. The guard
still makes the map the active dataset, so a reload names it as the
Dashboard's Browse action does.

Finally, `ActiveContextWatcherService` watches the registry against the active
pair and, when an id disappears (deleted from another tab, the CLI, or another
session), clears the affected half, toasts, and bounces off any now-broken
view.

`PairRecoveryService` covers the other way a pair goes missing: still
registered, but no longer loaded, because the backend restarted under an open
tab (#5155). Every request then answers 409 `dataset_not_loaded` /
`detector_not_loaded`, and nothing else would load the pair again: the guard
let the view in long ago, and polls skip a failed tick. `errorInterceptor`
hands it each failed response, including the ones that opt out of the toast,
as Browse's tile requests do; a 409 naming the active pair, with no switch in
flight and the tab on that pair's `/label` or `/tune` view or on `/browse` of
its dataset (#5158), starts `ContextSwitchService.reloadActivePair()`, and its
`reloaded$` makes the view run its pair-change reset. For Browse that reset
has to ask for a build: a dataset load leaves the pyramid out of memory, and
the build restores the layout persisted in the dataset's container rather than
fitting it again. The reload loads the dataset *before* the detector,
because the detector load resolves its dataset from the request's
`X-Dataset-Id`, which during a reload names the unloaded dataset. A 409 that
lands mid-switch or mid-reload, or for a request sent before the last reload
landed, is absorbed without a toast; one off the pair views still toasts. A
reload that fails, or a pair lost again within a minute of one, sends the tab
to the Dashboard with a message saying to reopen it.

A detector-positives map has nothing to load it again from: its context lived
only in the memory of the process that built it. After a restart the backend
answers 404 `dataset_not_found` for its id, not 409, so `PairRecoveryService`
answers that 404 apart (#5159): with the tab on `/browse/__detpos__<id>`, it
goes to the Dashboard with a message saying to browse the detector's positives
again from there. The 404 is absorbed off that view too, where it can only be
a request sent before the tab left. `dataset_not_found` for any other dataset
is untouched.

---

## 7. Talking to the backend

### The generated OpenAPI client

`frontend/openapi.json` is a **committed snapshot** of the Flask-smorest spec.
`ng-openapi-gen` turns it into `src/app/generated/api-client/` — typed models
plus one function per operation. The generated tree is **gitignored** and
rebuilt automatically:

```
prebuild / prebuild:prod / pretest / pretest:ci
    → node scripts/openapi-gen-cached.mjs && node scripts/build-stamp.mjs
```

The wrapper hashes the spec, the generator config, and the generator's own
version, and skips regeneration when nothing moved (`--force`, or
`npm run generate-api-client`, regenerates unconditionally).
`build-stamp.mjs` writes the bundle's version into
`generated/build-stamp.ts`; see
[the build stamp and the skew check](#the-build-stamp-and-the-skew-check) below.

The snapshot itself is gated: `./run-tests.sh` re-dumps the spec from the
running Flask app and **fails if `frontend/openapi.json` is stale**. When you
change a backend schema or route, run `npm run regenerate-openapi-snapshot`
and commit the result. That gate is what makes a backend field rename a
TypeScript compile error instead of a runtime surprise.

The client is configured with `apiService: false` — there is no generated
god-service. Hand-written `*-api.service.ts` files call the generated
functions, which is what lets them stay thin while keeping the typed contract.

### The build stamp and the skew check

`static/` is a gitignored build artifact, so `git pull && python app.py` can
leave a new server serving an old SPA, with nothing to say so (#2898). To catch
that, the bundle carries the same version stamp as the server.
`frontend/scripts/build-stamp.mjs` runs from the `prebuild` / `pretest` hooks
above. It writes `src/app/generated/build-stamp.ts` (gitignored, beside the
generated client) with the value of `vtsearch.__version__`, which is the UTC
timestamp of `HEAD`'s commit. It reads that value from `VTSEARCH_VERSION` when
the variable is set, and from git otherwise.

At startup `BuildSkewService` compares the stamp with `GET /api/version`. On a
mismatch it raises a toast that does not dismiss, and the Settings footer adds
a `⚠ bundle v …` chip beside the server version.

Nothing here is edited by hand. If you change how the Python version is
derived (`vtsearch/__init__.py`), change `build-stamp.mjs` to match.
CLAUDE.md's "Versioning" section carries that rule.

### `models/api.models.ts` — what stays hand-written

Everything the spec describes is **re-exported** from the generated tree, not
mirrored. Only three categories are hand-written, because the spec cannot
describe them:

- **SSE payloads** (`ProgressEvent`, `LoadingTask`, …) — `/api/events` is a raw
  streaming response flask-smorest does not model.
- **Client-only shapes** that no endpoint returns.
- **Request-side settings shapes** the frontend builds and posts.

Adding a hand-written interface that duplicates a generated one is a
regression: it can drift, and the compile-time guarantee is exactly what is
lost.  `models/projection.models.ts` follows the same rule for the Browse
payloads: it re-exports the generated `ProjectionMeta`, `TileResponse`,
`RegionLabel`, … and hand-writes only `ProjectionLayout` (a ready meta, whose
layout fields the one shared schema leaves optional), `BinShape` and the
canvas's `ViewTransform`.

### The interceptor chain

Order matters; it is fixed in `app.config.ts`.

1. **`timezoneInterceptor`** — attaches `X-Timezone-Offset` so the backend can
   bucket achievement milestones by the user's wall clock, not UTC.
2. **`activeContextInterceptor`** — the context headers ([§6](#6-the-active-datasetdetector-context)).
3. **`achievementsRefreshInterceptor`** — after a POST to any endpoint that
   could unlock a tier (vote, find, label import, dataset load), schedules a
   coalesced achievements refresh.
4. **`errorInterceptor`** — the failure chokepoint. Turns every failed response
   into a structured `ErrorContext` toast (endpoint, status, `request_id`, and
   the active pair at the time), deduped by status + message so one backend
   condition surfacing from several in-flight requests shows once. It also
   implements the **offline circuit breaker**: while offline, requests
   short-circuit to a synthetic network error without touching the wire, which
   is what stops the console filling with `ERR_CONNECTION_REFUSED`.
   `ConnectionStateService`'s probe (tagged with the `CONNECTION_PROBE` context
   token) is the one request allowed through.

Opt a single request out of the global toast with the `SKIP_ERROR_TOAST`
`HttpContextToken` — for probes that expect a 404, retry loops, and forms that
render their own inline validation. The error still propagates to
`catchError`/`error:` handlers; only the toast is suppressed.

### Server push, and polling when push won't do

`ProgressEventsService` holds **one** `EventSource` on `/api/events` and fans
it out to every consumer.
Channels: `server` (a per-connect `boot_id`, so a backend restart fires
`serverReset$` and consumers can drop state keyed on dead `task_id`s);
`loading-tasks` and `detector-loading-tasks` (`LoadingTask[]` lists); and
`sort`, `find`, `eval`, which carry the `ProgressEvent` shape that
`utils/format-progress.ts` renders. A payload-less `heartbeat` keeps the
circuit breaker online during long silent operations. The channel state is held in
signals — a signal write inside the SSE callback notifies Angular's scheduler
directly, which is precisely what makes the pump work without zone.js.

One channel breaks that mould: **`notification`**, carrying one-off messages
any server-side code (most usefully a plugin that hit a recoverable problem
and carried on) pushes with `notify()`. Those are *events*, not state — there
is no "current notification" to re-read and none is replayed to a late
listener — so the service exposes them as `notifications$`, a `Subject`
rather than a signal, and `ToastService` subscribes to turn each into a toast
of the matching level. See [`api/events.md`](api/events.md) for the payload
and [`EXTENDING-plugins.md`](EXTENDING-plugins.md#notifying-the-user-toasts)
for the producing side.

### Subscription teardown vs. cancellation

Two different jobs get done with the same RxJS machinery, and conflating them
is how a working poller gets swept into a bug. Name which one you are doing
before reaching for an idiom.

**Teardown** — "stop this when the component dies." There is exactly one
sanctioned idiom, and new code must use it:

```ts
private destroyRef = inject(DestroyRef);
...
this.someService.thing$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(...);
```

Drop the explicit `destroyRef` argument only in an injection context (a field
initializer or the constructor body); anywhere else — `ngOnInit`,
`ngAfterViewInit`, an event handler — it must be passed. The point of the
idiom is that teardown is *declared at the subscription*, so adding a
subscription can never forget to add a matching `unsubscribe()`. It is now the
**only** teardown idiom in the SPA: the hand-rolled `destroy$ = new
Subject<void>()` fired from `ngOnDestroy` and the `subs: Subscription[]` array
drained in `ngOnDestroy` are both gone, and neither may be reintroduced. Both
were strictly worse — the bookkeeping was manual and the compiler did not check
it — and the subject form had a hazard the sanctioned idiom does not: RxJS
`takeUntil` never fires on a pre-completed notifier, so a subscription armed
*after* `ngOnDestroy` ran was never torn down at all. `takeUntilDestroyed`
completes such a subscription immediately.

Because teardown rides on the `DestroyRef`, a **spec drives it with
`fixture.destroy()`**, never by calling `ngOnDestroy()` by hand — a bare hook
call runs only what the hook itself still does.

**Cancellation** — "stop the *previous* one because a newer one supersedes
it." This is not teardown and `takeUntilDestroyed()` cannot express it; the
component is very much still alive. It legitimately takes one of two shapes,
and both are correct as written:

- A **scope subject** fired on each reset, for a family of streams that share
  a lifetime shorter than the component's: `polling$`
  (`DashboardLoadingTasksService`), `stopPolling$` (`VoteStateService`) and
  `sortSuperseded$` (`SortRunnerService`) cancel superseded work. Name it for the
  scope it bounds, never `destroy$`.

  The dataset/detector pair's scope is the one case where the subject is not a
  component field: `PairScopeService` (component-provided on `find-view` and
  `label-view`) keeps it private and exposes `pairScope.scoped()` as the pipe
  operator, because the *ordering* around firing it is load-bearing. A pair
  switch must supersede **before** any of the new pair's state is installed, or
  a late response repaints the old pair's ranking over the new one's — a silent
  wrong-results bug, not a crash. `resetForNewPair()` is the only caller that
  can fire the scope, so the order is not a caller's to get wrong. The service's
  own `ngOnDestroy` covers destroy-time teardown, so a view must not (and need
  not) fire the scope by hand.
- A **re-assigned `Subscription` field** that unsubscribes the previous value
  before storing the next: `text-viewer`'s `sub`, `folder-browser`'s
  `currentSub`, `browse-bin-member-grid`'s `scrollSub` (re-keyed to the
  viewport *instance*), `label-importer-modal`'s `ingestSub`.

A `Subscription` field is only a teardown idiom when it is written once and
read only by `ngOnDestroy`. If it is re-assigned anywhere, it is cancellation
— leave it alone.

Where a real poll is still needed, reach for one of the two primitives rather
than `timer(0, n)` + `switchMap`. Both run each request to completion before
scheduling the next — a `switchMap` timer cancels in-flight requests, so a
backend slower than the interval froze the panel permanently (#2572) — and
which one you want is decided by whether the thing being polled *ends*:

- **`adaptivePoll()`** (`services/adaptive-poll.ts`) for open-ended background
  polling. Eases from `fastMs` to `slowMs` after N unchanged responses, and
  suspends entirely while the tab is hidden. Callers own teardown via
  `takeUntil(stop$)`.
- **`pollUntil()`** (`services/poll-until.ts`) for a job the user is waiting on:
  a projection build, anything with a terminal state. The caller's `apply`
  returns `'continue'` or `'stop'`, and the loop tears itself down when it
  settles. A failed request is absorbed and retried with exponential backoff
  (2s → 30s); only five consecutive failures give up, via `onLostContact`.
  Teardown is the returned handle's `stop()`.

`adaptivePoll`'s cadence easing and pause-while-hidden are precisely wrong for
the second case — progress is never stale while a build runs, and pausing in a
background tab would strand the user behind a bar that stopped moving — which
is why they are two helpers rather than one with a flag.

---

## 8. Component composition conventions

- **Standalone, `OnPush`, `vt-` selector prefix.** (`AppComponent` is the sole
  `app-` selector, as the bootstrap root.)
- **Signal inputs and outputs** (`input()`, `output()`) for new components;
  older decorators survive in places and are converted opportunistically.
- **Presentational children, stateful parents.** The larger pickers are
  explicit about this: `vt-source-picker` takes precomputed lists as inputs,
  emits user actions as outputs, and lets the parent decide what they mean —
  which is how the same widget serves both the Add-Dataset modal (run an
  importer) and the New-Detector modal (materialise one example file). Layout
  that differs between hosts is projected into named content slots rather than
  branched on inside the child.
- **Forms use `ngModel` as a value binding and nothing more.** Each control
  binds `[ngModel]` + `(ngModelChange)` (or `[(ngModel)]`) to a signal or a
  field. There is no `NgForm`, no `#ref="ngModel"` and no reactive form, so no
  template reads Angular's field state (`valid`, `dirty`, `touched`). Validity
  is a hand-written check (the importer pickers' `canSubmit`), and the Settings
  modal saves on every change. Signal Forms (`@angular/forms/signals`, stable
  from Angular 22) was evaluated for these forms in #5292 and not adopted.
  Three reasons: its field state and schema validators replace machinery
  nothing here uses; most of these forms render fields a plugin declares at
  runtime, where a schema bound to static paths fits badly; and the package
  imports `@angular/forms` itself, so adopting it removes nothing from the
  bundle. Reconsider it for a new hand-written form with a fixed shape that
  needs per-field errors or touched/dirty state.

### Modals

`vt-modal` (`components/modal/`) is the shared shell: backdrop, CDK focus
trap, close button, and a module-level **stack** of open instances so Escape
dismisses only the topmost one. Modals genuinely nest here (New Detector →
media crop, Settings → importer picker, anything → a confirm dialog), and
without the stack one keypress closed them all and lost the outer form.

- `dialog-host` + `VtDialogService` provide promise-returning `confirm()` /
  `prompt()`, including the standardised `confirmDestructive()` phrasing.
  Purely informational messages go to `ToastService`, never a modal.
- **Back vs Cancel is a rule, not a preference.** An inner view that *replaces*
  an outer one gets a left-aligned `← Back` chevron; `Cancel` in the footer
  means "abandon the whole dialog". A picker whose tab bar stays visible has no
  outer view to return to and correctly has no back button. The full rule,
  including the canonical markup and the persistent-tab exception, is in
  [`CLAUDE.md`](../CLAUDE.md) under "Nested-modal back buttons".
- **Plugin-field forms preview their template variables.** A modal that builds
  a form from a plugin's `fields` (Export, Find Results) seeds each
  value through `PluginTemplateVarsService`, which resolves the *declared*
  `template_vars` — `{detector_name}`, `{username}`, the date parts — so the
  user sees and can edit the value the server would substitute rather than a
  raw placeholder (issue #3199). Only declared names are touched, and anything
  unresolvable stays templated for the server. Never do this in a form whose
  values are **persisted** (AutoFind's saved exporter fields, a labelset-sync
  source): those templates are meant to re-resolve on every later run. See
  `utils/plugin-template-vars.ts`.

### Lazy loading and the bundle budget

Routes are lazy by construction. Beyond that, the heavy app-level modals are
imported by `AppComponent` but used **only inside `@defer` blocks**, so Angular
splits them into separate chunks — they drag in the file browser, the crop
modal and `marked`, which together push the eager bundle over budget.

The production `initial` budget is deliberately tight and the rationale is
written into `angular.json` next to the number. `./run-tests.sh` treats **any**
`▲ [WARNING]` from `build:prod` as a hard failure, including the 8 kB
`anyComponentStyle` budget. Fix the bloat — split the component, extract shared
styles, delete dead rules. Raising a budget needs the user's explicit approval.

What actually sits in the eager bundle is mostly **framework**: 317 kB of
Angular against 121 kB of app code when it was last audited (#3811). That is
why the budget carries visible headroom — Angular's own patch releases move the
number a few kB at a time, and when the margin was under 1 kB a routine
security bump turned the build gate red for every branch (#3795). Treat the
headroom as reserved for the framework, not as room for app code.

The corollary is that the budget cannot catch an app-code regression on its own
any more, so one specific trap is covered by a test:
**do not import `FormsModule` into anything on the eager path.** `vt-login` and
`vt-dialog-host` each render a single uncontrolled text field and used to pull
all ~37 kB of `@angular/forms` for it; both now use plain `[value]` + `(input)`.
`TestEagerBundleComposition` in `tests/core/test_frontend.py` walks the built
bundle's *static* imports from `main.js` and fails if the package reappears.

### Other primitives

`components/icon/` maps names (and backend-supplied emoji) to sanitised inline
SVG, cached per process. `components/context-menu/`, `drop-zone/`,
`skeleton/`, `progress-bar/`, `job-progress/`, `clipboard-copy/` are the small
shared widgets. `toasty-hint/` is one of Toasty's hints (#4680): Toasty and a
speech bubble beside the control to click next, with "Hide this hint" / "Hide
all hints" boxes that `HintsService` keeps in the user's settings. Drop it
under any positioned container; it measures itself against its anchor and
flips the bubble to the other side when it would run off the edge. Inside a
`vt-modal` (however deep) the anchor may be left out: the hint is then about
the dialog box, below it or beside its top corner when the box fills the
window. At most one hint is on screen (#4807): the component shows a hint in
a dialog only while that dialog is the topmost open one (`topOpenModal` in
`modal/modal.component.ts`), and a hint on a view only while no dialog is
open; within a view the host picks one (the Dashboard's `dashboardHint`
ladder, the voting overlay's `hint` input). Its `face`
(happy / sad / surprised, `utils/toasty-faces.ts`) is the same set the toasts
use. [`hints.md`](hints.md) charts every hint: its face, its text and when it fires;
a new hint adds a row there. `directives/no-focus-steal.directive.ts` stops toolbar buttons
next to the Browse canvas from swallowing keyboard focus on mousedown.

Services are root-provided by default; provide one on a component only when
per-instance state or a component-bound lifetime is the point.  The complete
list today: `LabelViewPanelStateService` and `SortRunnerService`
(`label-view`), `PairScopeService` (`label-view`, `find-view`),
`LineTestSessionService` (`find-view`), `BrowseViewportService` /
`BrowseSelectionService` (`browse-view`), and `SiblingLoadService`
(`photo-grid`, `source-photo-modal`).

---

## 9. Styling

[style-guide.md](style-guide.md) is authoritative for design tokens, shared
classes, patterns, and copy style. The structural facts:

- `src/styles.scss` is the global entry and `@use`s the partials in
  `src/scss/`: `_variables` (the `--space-*` / `--font-*` / `--radius-*` /
  color / z-index token set), `_layout`, `_components` (the shared `.btn`,
  `.back-btn`, card, form and modal classes), `_data-table`, `_picker-shared`
  (the `.tab-bar` / `.tab` primitive).
- Everything global is **class-only**, so it applies everywhere without
  bleeding through component encapsulation.
- Component `.scss` files hold only what is genuinely local. A rule that two
  components want belongs in `_components.scss` — that is also how the
  8 kB per-component style budget stays satisfiable.
- Themes (dark / light / high-viz / system) are token swaps driven by
  `ThemeService`; components should read tokens, not hardcode colors.

If your change alters a GUI surface framed by a screenshot in the user docs
or the slides, don't reshoot it: queue the shot in `docs/reshoot-queue/` and
the release run re-renders it. The full rule is "Screenshot reshoots" in
`CLAUDE.md`.

---

## 10. Testing

The suite runs on **Vitest + jsdom** via the `@angular/build:unit-test`
builder. No browser is needed, which is why it works in the cloud container.

```bash
cd frontend && npm run test:ci     # headless, one shot
cd frontend && npm test            # watch
./run-tests.sh frontend            # build:prod + npm audit + Vitest
```

Shared fragments live in `src/app/testing/` and compose freely in a
`providers` array:

- `provideHttpTesting(...interceptors)` — the
  `provideHttpClient` + `provideHttpClientTesting` pair, optionally with
  interceptors under test.
- `provideZoneless()` / `configureZoneless()` — put the `TestBed` under
  zoneless change detection.
- `mocks.ts` — shared service stubs.
- `settle-resource.ts` — `settleResource()` / `settleZoneless(fixture)`, the
  awaits the two subsections below rely on.
- `line-balance.ts` / `line-test.ts` — wire-shaped fixtures for the balance and
  Tune-autopilot payloads.

### The zoneless oracle

A normal spec that pumps `fixture.detectChanges()` by hand **masks** exactly
the staleness bug zoneless introduces: it force-renders regardless of whether
anything was notified. So a spec auditing that property must (1) add
`provideZonelessChangeDetection()`, (2) **remove** the manual pumps — a
leftover pump both re-masks the bug and can race auto-detect into NG0100, (3)
drive updates through the same channel the app uses (write the service
signal, dispatch the bound event), (4) `await settleZoneless(fixture)`, and
(5) assert on rendered DOM, not component fields. The `*.zoneless.spec.ts`
files on the main views are the worked examples.

### `rxResource` timing

The loader is effect-scheduled and promise-based, which changes test timing
even though runtime behaviour is unaffected:

- `fixture.detectChanges()` does **not** issue the GET — call `TestBed.tick()`
  before `httpMock.expectOne(...)`.
- The value commits on a microtask — `await settleResource()` after
  `flush()` (advancing fake timers alone does not commit it).

### Other conventions

`src/test-setup.ts` installs the inert jsdom stubs (`EventSource`,
`ResizeObserver`, `scrollIntoView`, `CSS.escape`, media element methods,
canvas `getContext`) and resets the `TestBed` before each test so a throwing
teardown cannot cascade — that per-test reset *is* the cascade guard. It also
installs the real-XHR guard (`src/app/testing/real-xhr-guard.ts`). `HttpClient`
is provided in root, so a spec without `provideHttpTesting()` gets no missing
provider error, and its requests would go to jsdom's `localhost:3000`. The
guard stops every real `XMLHttpRequest` before it reaches the network and fails
the test that sent it, naming the request.
`vitest.config.ts` deliberately leaves `isolate: false` (the builder's
default): per-file isolation cost 3–4× the runtime and the guard in
`test-setup.ts` covers the same failure mode. No spec uses
`fakeAsync`/`tick`: time is driven with `vi.useFakeTimers()` plus real
macrotask drains.

---

## 11. Build and serve

```bash
cd frontend
npm install
npm start          # dev server on :4200, proxying /api and /static to :5000
npm run build:prod # → ../static/ (index.html, main.js, styles.css)
```

Flask serves the built output from `static/`, whose build artifacts are
gitignored — every deployment builds them. `public/` is copied verbatim into
the output alongside them (favicons, logo); `docs-assets/` is a
symlink to `docs/user/`, so the in-app user guide (rendered with `marked` in
the keyboard-help modal) is the same file the repo ships.

`angular.json` pins the canonical `@angular/build:*` builders (`@angular/build`
is a direct devDependency; the `@angular-devkit/build-angular` alias package is
deliberately not installed) and gives the test target its own
`build:test` configuration, so spec polyfills stay decoupled from production.
Both polyfill arrays are empty and must stay that way — reintroducing
`zone.js` would invalidate every rule in [§5](#5-reactivity-the-zoneless-change-detection-model).
The Angular upgrade runbook is in [`frontend/README.md`](../frontend/README.md).

---

## 12. Invariants

The short list of things that break silently if violated:

- **No `zone.js`.** Not in the app, not in the test build, not in
  `package.json`.
- **Rendered state has a notification path** — signal write, bound event, or
  async-pipe emission. A plain field written from a callback is a stale view.
- **Only `ContextSwitchService` changes the active pair.** Promoting a pair the
  backend has not loaded reopens the 409 cascade.
- **Native `src` URLs go through `mediaUrl()`.** Interceptors do not run for
  `<img>` / `<audio>` / `<video>`.
- **`openapi.json` is regenerated when the backend spec changes**, and the
  generated client is never edited or committed.
- **Hand-written types never duplicate generated ones** — re-export instead.
- **New components are standalone + `OnPush`**, with a `vt-` selector.
- **Build warnings are failures.** Fix the bloat rather than the budget.
- **Desktop only.** No responsive breakpoints, no touch affordances.

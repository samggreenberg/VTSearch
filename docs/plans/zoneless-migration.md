# Zoneless change detection — detailed migration plan

**Status: all phases (0–5) shipped; production is zoneless; kept as the
migration reference.** The provider flip (Phase 3) landed, Phase 4 human browser
QA passed live against the GRID (2026-06-22, no staleness, zero console errors),
and Phase 5 cleanup is complete: `zone.js` is dropped end to end (empty
`build:test` polyfills, removed from `package.json`, surviving only as
`@angular/core`'s optional peer), all 43 fakeAsync specs run native `async`, and
`ChangeDetectionStrategy.OnPush` is explicit on all 86 components.
`./run-tests.sh` is green (5026 passed; `frontend` 860 tests). **No open work
remains** — this doc is retained for the reusable conversion recipes, the
zoneless mental model, and the risks/verification guidance that inform any future
change-detection work.

VTSearch's Angular 21 frontend runs on `provideZonelessChangeDetection()` instead
of zone.js. The migration was unusually careful because the frontend cannot run
in a browser in the Claude-Code-on-the-web container (no Chrome), and the failure
mode zoneless introduces — "a value changed but the view silently went stale" —
does not show up in a normal headless unit run (component specs drive
`fixture.detectChanges()` by hand, which force-renders and masks staleness). So
the effort front-loaded a test-harness phase that made the Vitest suite catch
staleness, converted the reactivity surface to patterns correct under *both* zone
and zoneless, and treated the production flip as the last step behind a human
browser-QA pass. Every Angular API claim below was verified against angular.dev /
the angular source (see Appendix: verified facts).

---

## How zoneless changes the rules (the mental model)

Under zone.js, *any* async callback (timer, XHR, DOM event, promise) triggers a
global change-detection (CD) pass, so a component can mutate a plain field in any
callback and the view repaints. Under `provideZonelessChangeDetection()`, **CD
runs only when something explicitly notifies Angular's scheduler.** The canonical
list of notifications (verbatim from the official zoneless guide) is:

- `ChangeDetectorRef.markForCheck` — **called automatically by `AsyncPipe`**
- `ComponentRef.setInput`
- **Updating a signal that is read in a template**
- **Bound** host or template listener callbacks (`(click)="…"`, `@HostListener`)
- Attaching a view that was marked dirty by one of the above

Everything else that used to "just work" no longer schedules CD:

| Pattern | Zoneless behavior | Action |
|---|---|---|
| Signal read in template, signal written anywhere | **Schedules CD** | ✅ preferred target |
| `obs \| async` in template | `AsyncPipe` calls `markForCheck` → **schedules CD** | ✅ safe, even from raw callbacks |
| `cdr.markForCheck()` in a callback | **Schedules CD** (notifies scheduler directly) | ✅ escape hatch |
| `(click)` / `@HostListener` handler | **Schedules CD** | ✅ safe |
| `.subscribe(v => this.field = v)`, `field` read in template | field updates, **view does NOT repaint** | ❌ must convert |
| `setTimeout`/`setInterval`/`Promise.then` writing a template field | writes, **no repaint** | ❌ must convert |
| raw `element.addEventListener(...)` / `ResizeObserver` / `MutationObserver` writing a template field | writes, **no repaint** | ❌ must convert |
| `effect()` writing a **plain** (non-signal) template-bound field | runs, but **does not mark host dirty** | ❌ latent bug — make the field a signal |
| `effect()` whose template reads the **signal** it updates | template's signal read marks dirty | ✅ safe |
| `requestAnimationFrame` loop drawing to `<canvas>` | canvas draws imperatively, no Angular view involved | ✅ safe, no change |
| `ngZone.run(() => …)` purely to re-enter Angular for CD | `NgZone` is a `NoopNgZone`; `.run()` **does not** drive CD | ❌ must replace the CD trigger |
| `ngZone.runOutsideAngular(...)` perf wrapper | harmless no-op; **callable, safe to keep** | ✅ leave as-is |

Two corrections earlier informal audits got backwards:

- **`AsyncPipe` and `markForCheck()` are zoneless-safe.** A `BehaviorSubject.next()`
  fired from inside a raw `addEventListener`/`setTimeout` *does* update an
  `obs | async` binding, because the pipe calls `markForCheck`. So a service that
  pushes through a Subject is fine **as long as its consumer reads it via `| async`
  (or a signal), not via `.subscribe()`-into-a-plain-field.** The cheapest lever.
- **`NgZone.run`/`runOutsideAngular` do not have to be deleted.** The methods stay
  callable; what breaks is relying on `.run()` to *cause* a CD pass.

---

## The conversion recipes (canonical patterns)

Every site falls into one of these. Each is written so the result is correct
under **both** zone and zoneless (so each could land while prod was still zoned).

### Recipe A — `subscribe`-into-plain-field → `| async` pipe (cheapest)

When a component subscribes to a service Observable in `ngOnInit` only to mirror
it into a field the template reads, delete the subscription and bind the
Observable through `async` (which handles subscribe/unsubscribe and calls
`markForCheck` on emit).

```ts
// before
toasts: Toast[] = [];
ngOnInit() { this.sub = this.toastService.toasts$.subscribe(t => this.toasts = t); }
ngOnDestroy() { this.sub?.unsubscribe(); }
```
```html
<!-- before -->  @for (t of toasts; track …) { … }
<!-- after  -->  @for (t of (toastService.toasts$ | async) ?? []; track …) { … }
```
Drop the field, the `sub`, and the `ngOnDestroy` if it was its only user. Use for
clean single-source reads.

### Recipe B — `subscribe`/`BehaviorSubject` → signal (preferred for hot/shared state)

For service state read in many places, or where you also need synchronous
"latest value" access, convert the `BehaviorSubject` to a `signal`. Consumers
read the signal in the template (or via `computed`); imperative consumers read
`sig()`.

```ts
// service: before
private readonly statusSubject = new BehaviorSubject<ConnectionStatus>('online');
readonly status$ = this.statusSubject.asObservable();
goOffline() { this.statusSubject.next('offline'); }
// service: after
readonly status = signal<ConnectionStatus>('online');
goOffline() { this.status.set('offline'); }
```
Template reads `connection.status()`. A `.set()` from a raw `addEventListener`
schedules CD because it is a signal write read in a template. Bridge with
`toObservable(sig)` for RxJS consumers, `toSignal(obs$)` for the reverse.
**Note:** a signal read *through a value-returning getter* during template
evaluation is still tracked (pinned by
`frontend/src/app/testing/getter-signal-zoneless.spec.ts`), so
`sortState.sortBusy` / `voteState.goodVotes`-style getter bindings stay
byte-for-byte identical yet become reactive — how `SortStateService` /
`VoteStateService` were signalized without touching a single template.

### Recipe C — imperative callback → `markForCheck()` (escape hatch)

When a value genuinely must be mutated from a timer/raw-listener and converting to
a signal is disproportionate, inject `ChangeDetectorRef` and call `markForCheck()`
after the mutation. Prefer A/B; reserve C for local, leaf, animation-ish state.

### Recipe D — `ngZone.run()` re-entry → drop the run, trigger CD properly

Remove the `ngZone.run(...)` wrapper (keep any surrounding `runOutsideAngular`).
Replace the CD purpose with a **signal write** (best) or `markForCheck()`.
**Caveat (corrected):** a bare `output()` emit from an unpatched callback will not
by itself schedule the *parent's* CD **only if the parent subscribes to it
imperatively in TS**. A template `(event)` binding is a *bound listener* and *is*
on the notification path — so the four `setTimeout(() => this.close())` auto-close
paths needed no rework (proven by
`frontend/src/app/testing/output-emit-zoneless.spec.ts`).

### Recipe E — leave canvas rAF / `runOutsideAngular` alone

`requestAnimationFrame` loops that only draw to `<canvas>` (`browse-canvas`,
`browse-minimap`, `audio-crop-overlay`, the progress-modal chart) need no CD and
must **not** be "fixed" into triggering it (perf regression). `runOutsideAngular`
perf wrappers stay; only their inner `.run()` re-entries (Recipe D) change.

### Recipe F — `effect()` writing template-bound state → signalize the field

An `effect()` that writes a plain template-bound field (e.g. `center-panel`'s
settings mirror) must write a **signal** instead, and the template must read that
signal. Either make the destination a `signal()`, or — if the source is already a
signal — replace the effect+field with a `computed()`.

---

## Risks & gotchas (carry these into future change-detection work)

- **Silent fakeAsync breakage.** If a test target loses zone.js *while still using
  fakeAsync*, the ProxyZone shim no-ops silently and every fakeAsync occurrence
  throws. (Now moot: all specs are native `async` and zone.js is gone.)
- **`rxResource` is invisible to `fakeAsync`.** The virtual clock doesn't drive
  promise-based resolution — use real-async + `settleResource()`. A loading
  `rxResource` also holds the app *unstable*, so `await fixture.whenStable()`
  before flushing the GET **deadlocks**; issue the GET with `TestBed.tick()` then
  `settleResource()`.
- **`effect()` plain-field trap (Recipe F).** Audit every `effect()` for plain
  template-bound writes; `center-panel`'s settings mirror was the canonical case.
- **Output emits from non-Angular callbacks (Recipe D).** Bound template
  `(event)` listeners schedule CD; imperative TS subscribes do not.
- **Do not "fix" canvas rAF loops (Recipe E)** into triggering CD.
- **`ExpressionChangedAfterItHasBeenChecked` (NG0100)** can surface under stricter
  CD timing and in `whenStable()` specs; contained by `isolate:true` + the cascade
  guard but still real failures to fix.
- **`@Input()` setters** that mutate template state need the same treatment as a
  subscribe-assign.
- **`ResizeObserver`/`MutationObserver`** are as un-patched as raw
  `addEventListener` — audit them as a first-class class, not as effects.
- **CDK virtual scroll** can flicker/stale under zoneless — browser-QA only (the
  jsdom suite cannot exercise real scroll geometry). Confirmed clean in Phase 4.
- **`ngModel` two-way bindings** are safe via `(ngModelChange)`, but a
  programmatic plain-field write from a non-event callback won't refresh the input.
- **Zoneless async leaks in tests.** Without zone.js the framework no longer
  auto-cleans timers/microtasks; an SSE poller's `timer(0, N)` first emission or a
  root-singleton rxResource reload can fire after teardown and throw NG0205. Drain
  one macrotask in `afterEach` *while the injector is still alive* (a global drain
  in `test-setup.ts` was rejected — it let a pending re-check crash teardown-state
  leaf components).

---

## Verification strategy (how we knew it worked without a browser)

The plan made the headless suite a genuine zoneless oracle rather than relying on
eyeballing:

1. **Per-component proof.** Each migrated component's spec runs its `TestBed`
   under `provideZonelessChangeDetection()`, drives state through the production
   channel, `await`s `whenStable()`, and asserts the **rendered DOM** — not
   `component.someField`. A missing notification ⇒ stale DOM ⇒ failing assertion.
   `detectChanges()` force-runs CD even when Angular would not have scheduled it,
   masking the exact staleness bug; `whenStable()` flushes only *scheduled* CD.
2. **Canary specs** (`.zoneless.spec.ts`) explicitly test the scheduling path with
   zero manual CD pumping, so a forgotten signal write / `markForCheck` fails loud.
3. **Lockstep migration.** The zoneless `TestBed` was adopted per component during
   Phases 1–2, so the suite was fully green under zoneless semantics *before* the
   prod flip — making the flip a near-no-op for behavior.
4. **Human QA (Phase 4)** was the final backstop for what DOM-level unit
   assertions can't capture (real paint timing, focus, canvas interaction).

**Harness nuance:** adding `provideZonelessChangeDetection()` to a `TestBed` flips
`ComponentFixtureAutoDetect` **on** by default. The corollary is that manual
`fixture.detectChanges()` pumps must be *removed*, not supplemented (a leftover
pump both re-masks staleness and can throw NG0100 racing auto-detect). During the
final spec migration the oracle exposed and fixed **five real staleness bugs** —
plain fields written from HTTP `.subscribe()` yet read in the template:
`text-viewer.text`, `autodetect-results-modal`, `examples-editor-modal`,
`combine-datasets-modal`, and `label-list.sortedEntries`.

---

## Framework-surface notes (verified non-gaps + watch-items)

**Watch-items (required attention):**

- **CDK virtual scroll** (`@angular/cdk/scrolling` in `left-panel/media-list` +
  `browse-bin-popup`) has historically had zoneless repaint/flicker interactions.
  Browser-QA both viewports for scroll staleness (done in Phase 4, clean).
- **`ngModel` programmatic writes** (58 files): two-way binding is safe via
  `(ngModelChange)`; writing an `ngModel`-bound plain field from a non-event
  callback won't refresh — same rule as everywhere.

**Verified non-gaps (no action needed):**

- **Router / lazy routes / `@defer`** — framework-driven, on the notification path.
- **`HttpClient` / interceptors** — not a CD trigger; repaint flows through the
  consumer. The three interceptors don't touch template state.
- **Angular animations** — not used (CSS-only transitions).
- **`marked`** — called with `{ async: false }`, synchronous.
- **`onStable` / `onMicrotaskEmpty` / `afterRender`** — zero app usages.
- **SSR / hydration** — none.
- **`PendingTasks`** — not needed; `whenStable()` resolves correctly.
- **video/audio `(timeupdate)`/`(play)`/`(loadedmetadata)`** — bound listeners.

---

## Rollback plan (for the record)

- Phases 1–2 were individually revertible, behavior-neutral commits under zone, so
  a QA regression reverts just the offending cluster.
- Phase 3 was a 3-line revert (`app.config.ts` provider + `angular.json`
  build-polyfills + budget); because 1–2 are correct under zone too, reverting
  only Phase 3 returns to a working zone-based app with the modernized reactivity.
- Phase 0 test-target polyfills were independent of the flip.

---

## What shipped

Terse per-phase record; the full detail is in git log / the landing PRs.

- **Phase 0 — test harness.** `frontend/src/app/testing/zoneless-testbed.ts`
  (`provideZoneless()` / `configureZoneless()`), `settleZoneless()` next to
  `settleResource()`, and a reference/canary spec verified to fail on a
  plain-field-write bug and pass on the signal path. Test polyfills decoupled from
  the prod build via a dedicated `build:test` configuration; ProxyZone shim
  replaced by `zone.js/plugins/vitest-patch` (which forced a `zone.js` bump
  `~0.15` → `~0.16`, in Angular 21.2's peer range). No production code changed.
- **Phase 1 — hot shared services signalized.** 1.1 `ProgressEventsService` SSE
  pump (six channels → signals, dropped `zone.run`); 1.2 `KeyboardService`
  de-zoned + coupled `center-panel` state signalized; 1.3 `ConnectionStateService`
  (status/retrying → signals, offline-banner off `| async`); 1.4
  `BrowseSelectionService` (`changed$` → `version` signal). Each with consumer +
  canary specs.
- **Phase 2 — components, cluster by cluster.** 2.1 leaf utils (toast-container,
  clipboard-copy, voting-overlay, dialog-host + `VtDialogService`, browse-legend);
  2.2 stats/picker modals (find/detector/dataset-stats, label-importer onto
  `rxResource`, importer/exporter mutation fields); 2.3 center-panel viewers
  (image-viewer drag/key/`ResizeObserver`/shake; video-player verified DOM-only);
  2.4 label-view (own fields + `SortState`/`VoteState` bridged via `toSignal`);
  2.5 find-view + **signalized `SortStateService`/`VoteStateService`** (dropped
  their `$` observables, clean break); 2.6 browse cluster (browse-view/-hover-
  preview/-selection-panel; canvas/minimap/bin-popup verified safe); 2.7
  left-panel + media-list; 2.8 right-panel, context-pulldown, dashboard, the
  modals; 2.9 app.component + a pre-flip sweep (export-modal, examples-editor).
- **Phase 3 — production flip.** `app.config.ts` →
  `provideZonelessChangeDetection()`; base `build` polyfills emptied (`[]`);
  initial budget tightened 540 kB → 500 kB (measured eager bundle 527 → **488 kB**
  once zone.js left). Removed three obsolete `polyfills.js` Python tests. Full
  suite green (4919 passed).
- **Phase 4 — human browser QA (2026-06-22, live against the GRID via Chrome
  MCP).** Every staleness-prone surface repainted correctly; no "stale until I
  click"; zero console errors. Verified: SSE load-progress bars, vote → retrain →
  re-sort → minimap, dashboard timers, de-zoned keyboard shortcuts, offline
  banner + Retry, `VtDialogService` destructive confirm, CDK virtual scroll,
  browse canvas, modals. The one bug found (`vt-modal` `Esc` focus) was
  pre-existing and unrelated to zoneless.
- **Phase 5 — cleanup.** Removed all 8 no-op `ngZone.run` wrappers in
  browse-canvas + both in panel-resize.directive; migrated all 43 fakeAsync
  occurrences (11 files) to native `async` + Vitest fake timers and dropped
  `vitest-patch`; migrated the last 14 default-TestBed specs to the zoneless
  TestBed and **removed `zone.js` end to end** (empty `build:test` polyfills,
  gone from `package.json`); explicit `ChangeDetectionStrategy.OnPush` on all 86
  components — which surfaced that `DatasetStateService` /
  `DashboardLoadingTasksService` were read non-reactively in the dashboard
  template and had to be signalized (Recipe B); fixed `vt-modal` `Esc`-to-close by
  moving to `@HostListener('document:keydown.escape')`.

---

## Appendix: verified facts (with sources)

All confirmed against angular.dev and angular/angular(-cli) source.

1. **`markForCheck` schedules CD under zoneless** — marks the view dirty *and*
   notifies the scheduler (`NotificationSource.MarkForCheck`), independent of
   zones. https://angular.dev/guide/zoneless
2. **`AsyncPipe` is zoneless-safe** — calls `markForCheck` on emit.
3. **Canonical scheduling-trigger list** — `markForCheck` (via `AsyncPipe`),
   `ComponentRef.setInput`, updating a signal read in a template, bound
   host/template listener callbacks, attaching a dirty view.
   `afterRender`/`afterNextRender` run *during* a CD pass, not as triggers.
4. **Bound `(event)`/`@HostListener` trigger CD; raw `addEventListener` does not**
   — the word "Bound" in the trigger list is load-bearing.
5. **Effects do not auto-mark host dirty** — an `effect → plain field → template`
   binding has no notification path. `render3/reactivity/effect.ts`.
6. **`provideZonelessChangeDetection()` is STABLE** (since v20.2; default in
   v21+). Use this name, not `provideExperimentalZonelessChangeDetection`.
7. **`zone.js/plugins/vitest-patch` exists & is documented** as the bridge for
   `fakeAsync`/`flush`/`waitForAsync` under the Vitest builder, but Angular
   recommends converting to native `async` + Vitest fake timers.
   https://angular.dev/guide/testing/migrating-to-vitest
8. **Keeping zone.js in TEST polyfills while prod is zoneless is valid** — test
   polyfills and prod bootstrap are independent.
9. **`whenStable()` catches staleness that `detectChanges()` masks** — the guide
   says to avoid `detectChanges()` (it "forces change detection to run when
   Angular might otherwise have not") and shows
   `provideZonelessChangeDetection()` + `await fixture.whenStable()`.
10. **`NgZone` becomes `NoopNgZone`; `ngZone.run()` alone doesn't drive CD** — the
    methods stay callable ("do not need to be removed") but no longer cause a CD
    pass. `zoneless_scheduling_impl.ts`.

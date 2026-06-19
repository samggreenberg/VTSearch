# Zoneless change detection — detailed migration plan

Status: **Phase 0 shipped (test harness); Phase 1 complete — 1.1 + 1.3 + 1.4
(ProgressEventsService SSE pump + ConnectionStateService + BrowseSelectionService
signalized) and 1.2 (KeyboardService de-zoned + the coupled center-panel state
signalized, i.e. the center-panel slice of Phase 2.3); Phase 2.1 shipped (leaf
utility components — toast-container, clipboard-copy, voting-overlay,
dialog-host + VtDialogService, browse-legend); Phase 2.2 shipped (stats + picker
modals — find/detector/dataset-stats signalized; label-importer signalized +
moved onto rxResource; settings-importer/exporter + label-exporter mutation-result
fields signalized); Phase 2.3 shipped (center-panel viewers — image-viewer's
window drag/key handlers + shake `setTimeout` + `ResizeObserver` rendered-size
writes signalized; video-player verified DOM-only, no change); Phase 2.4 shipped
(label-view — its subscribe/timer/effect-written template state signalized and
the still-Observable `SortStateService`/`VoteStateService` channels it binds
bridged into signals via `toSignal`); Phase 2.5 shipped (find-view — signalized
its own subscribe/effect-written state + an `unverifiedSortOrder` computed, and
**signalized the shared `SortStateService` and `VoteStateService`** so every
binding of those two services repaints under zoneless with no per-consumer
bridges; this also migrated right-panel's vote piles → `computed`s,
center-panel's `goodVotes$`/`badVotes$` subscribes → an `effect`, and dropped
label-view's `toSignal` bridges); Phase 2.6 shipped (browse cluster —
browse-view signalized its 13 async/poller/effect-written template fields +
dropped the divider-drag `ngZone.run`; browse-hover-preview signalized
`textContent` written from the paragraph `fetch().then()`; browse-selection-panel
signalized `count`/`viewMode`/`gridGoalWidth`/`sortedEntries` written from the
selection + settings effects and the metadata `version$` subscribe — fixing a
latent effect-into-plain-field staleness bug; browse-canvas/-minimap/-bin-popup
verified safe as-is — canvas-only or already `markForCheck`/`@HostListener`
disciplined); Phase 2.7 shipped (left-panel + media-list — signalized
left-panel's effect-written `mediaTypeName`/`textSortAvailable`; media-list's
`loadingMedias` → `computed`, `markForCheck` for the metadata-hydration
subscribe, dropped the relayout `zone.run`); the rest of Phase 2 (2.8–2.9; note
2.8 right-panel's `LabelsetStateService`/settings mirror still remains) and
Phases 3–5 not started.**
This document
is the exceedingly-explicit, source-verified plan for taking VTSearch's Angular
21 frontend off zone.js and onto `provideZonelessChangeDetection()`. It
supersedes the earlier stub. Every count and file:line reference was checked
against the code; every Angular API claim was verified against angular.dev / the
angular source (Appendix C); and the plan itself was adversarially reviewed for
accuracy, feasibility, and completeness (which added the dialog-service,
CDK-virtual-scroll, observer-callback, and test-autoDetect items below).

**What shipped (Phase 0).** The Vitest suite can now run any spec under a
zoneless `TestBed` and catch staleness:
- `frontend/src/app/testing/zoneless-testbed.ts` — `provideZoneless()` /
  `configureZoneless()` helpers (per-component opt-in; not enabled globally).
- `settleZoneless(fixture)` added next to `settleResource()` in
  `frontend/src/app/testing/settle-resource.ts` (macrotask drain +
  `whenStable()`, no `detectChanges()`).
- `frontend/src/app/testing/zoneless-testbed.spec.ts` — reference/canary spec for
  the 0.3/0.4 pattern; verified to **fail** on a plain-field-write staleness bug
  and pass on the signal path (the suite is a genuine oracle).
- Test polyfills decoupled from the prod build via a dedicated
  `frontend:build:test` configuration in `angular.json` (see the 0.1 correction
  below). The hand-rolled ProxyZone shim in `test-setup.ts` was replaced by the
  Angular-maintained `zone.js/plugins/vitest-patch`; all 69 spec files / 767
  tests pass.

**Correction discovered while implementing 0.1** (the plan as written assumed
facts that the pinned toolchain did not match):
- `zone.js/plugins/vitest-patch` ships only in **zone.js ≥ 0.16**, not the pinned
  `~0.15.0`. Phase 0 bumped `zone.js` to `~0.16.0` (Angular 21.2 peer-depends on
  `~0.15.0 || ~0.16.0`, so this is in-range).
- The `@angular/build:unit-test` builder (21.2) has **no `polyfills` option**, so
  0.1's "give the test target its own `polyfills`" is not literally possible.
  Equivalent decoupling: a dedicated `build:test` configuration that pins the
  test polyfills, with the unit-test `buildTarget` pointed at it.
- `vitest-patch` needs `ProxyZoneSpec`/`SyncTestZoneSpec` (from
  `zone.js/testing`) to exist at load time, and the builder appends
  `zone.js/testing` *last*; so the polyfills array lists `zone.js/testing`
  **before** `vitest-patch`. See Open follow-ups for the Phase-3 implication.

No production code has changed yet; the only thing that shipped alongside the
original stub was the 540 kB initial-bundle budget bump and the one-component
`@defer` (the incompatible-pair explainer). The budget is unwound in Phase 3.5.

**Why this plan is unusually careful.** The frontend cannot be run in a browser
in the Claude-Code-on-the-web container (no Chrome/Chromium), and the failure
mode zoneless introduces — "a value changed but the view silently went stale" —
is exactly the kind of bug that does *not* show up in a normal headless unit run
(every component spec today drives `fixture.detectChanges()` by hand, which
force-renders and therefore *masks* staleness). So the plan front-loads a
**test-harness phase that makes the Vitest suite able to catch staleness**, and
treats the production flip as the *last* step, gated behind a human browser-QA
pass. Every technical assertion below was verified against angular.dev and the
angular/angular source; the citations live in "Appendix C: verified facts".

---

## 0. TL;DR / decision

- **Going zoneless is the right end state** (drops the ~35 kB zone.js polyfill,
  lets the initial-bundle budget fall back toward the framework floor, modern
  default, cleaner stack traces). But it is an **app-wide change-detection
  posture change**, not a provider flip: 0 of 86 components are `OnPush` today
  and ~278 imperative `.subscribe()` sites (≈221 in components) assign to plain
  fields that zone.js currently repaints "for free".
- **The migration is incremental and reversible.** Production stays zone-based
  until a single flip in Phase 3. Phases 1–2 convert the reactivity surface to
  patterns that are correct under *both* zone and zoneless, so each can ship on
  its own with zero behavior change while prod is still zoned.
- **The safety net is built first (Phase 0).** We make the unit suite run its
  `TestBed` under `provideZonelessChangeDetection()` and assert on the **rendered
  DOM** after `await fixture.whenStable()` (not on component fields after manual
  `detectChanges()`). This turns the headless suite into a real zoneless-staleness
  detector, component by component, *before* prod flips.
- **The flip + human QA is last (Phases 3–4).** Only after the suite is green
  under a zoneless `TestBed` across the interactive surfaces do we flip the
  production provider, and a human must browser-QA the interactive flows before
  merge.

---

## 1. How zoneless changes the rules (the mental model)

Under zone.js, *any* async callback (timer, XHR, DOM event, promise) triggers a
global change-detection (CD) pass, so a component can mutate a plain field in any
callback and the view repaints. Under `provideZonelessChangeDetection()`, **CD
runs only when something explicitly notifies Angular's scheduler.** The canonical
list of notifications (verbatim from the official zoneless guide — see Appendix C
#3) is:

- `ChangeDetectorRef.markForCheck` — **called automatically by `AsyncPipe`**
- `ComponentRef.setInput`
- **Updating a signal that is read in a template**
- **Bound** host or template listener callbacks (`(click)="…"`, `@HostListener`)
- Attaching a view that was marked dirty by one of the above

Everything else that used to "just work" no longer schedules CD. The practical
consequences, each verified (Appendix C):

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

Two corrections to be explicit about, because earlier informal audits got them
backwards or over-flagged:

- **`AsyncPipe` and `markForCheck()` are zoneless-safe.** A `BehaviorSubject.next()`
  fired from inside a raw `addEventListener`/`setTimeout` *does* update an
  `obs | async` binding, because the pipe calls `markForCheck`, which notifies
  the scheduler independent of zones. So a service that pushes through a Subject
  is fine **as long as its consumer reads it via `| async` (or a signal), not via
  `.subscribe()`-into-a-plain-field.** This is the cheapest conversion lever.
- **`NgZone.run`/`runOutsideAngular` do not have to be deleted** to be
  zoneless-compatible (the methods stay callable). What breaks is relying on
  `.run()` to *cause* a CD pass; that specific purpose must be replaced.

---

## 2. Current state (audited)

Counts are from a full sweep of non-spec `.ts`/`.html` under
`frontend/src/app` (86 components, ~52 services). See Appendix A for the
file-level catalog.

**Already zoneless-safe (the head start):**

- **Inputs/outputs are largely modernized.** `output()` is 100% signal-based
  (62 files, 164 calls; **zero** `@Output()` decorators). Signal `input()` is
  229 calls across 57 files; 39 files still carry decorator `@Input()` (these
  work under zoneless — only input *setters* that mutate template state need a
  look).
- **Two state services are signal/`rxResource`-driven** (`SettingsStateService`,
  `MediaStateService`) from the `httpresource-migration.md` work, plus the
  `left-panel` reads and the importer/exporter picker modals. These are the
  **reference pattern** for everything else.
- **Reactive primitives in use:** `signal(` ×8, `computed(` ×9, `effect(` ×20,
  `rxResource(` ×10. `toSignal`/`toObservable`/`linkedSignal`/`model`/
  `httpResource` are **not** used yet.
- **`AsyncPipe` consumers (already safe):** 4 bindings total — `offline-banner`
  (`connection.status$`, `connection.retrying$`) and `app.component`
  (`achievements.hasPending$`). These keep working unchanged.
- **Two components already discipline CD with `markForCheck()`:** `browse-bin-popup`
  (5 sites) and `icon` (1).

**The risk surface (what must change):**

- **`OnPush` adoption: 0 / 86.** Nothing is written to OnPush discipline today,
  so every component transitions from "checked on every zone tick" to "checked
  only when notified".
- **≈221 `.subscribe()` callbacks in 43 component files** (228 raw component
  matches; 278 across all non-spec `.ts`) assign to plain template-bound fields
  with no `markForCheck`/`async` — the bulk of the work.
  Concentrations: `dashboard` (32), `label-view` (24), `dataset-importer-modal`
  (22), `find-view` (13), `browse-view` (13), `new-detector-modal` (12),
  `context-pulldown` (11), `center-panel` (11), `right-panel` (10),
  `load-sort-modal` (8), the stats/importer/exporter modals (see Appendix A).
- **NgZone re-entry purely for CD** in 7 files — the hard blockers:
  `progress-events.service` (the SSE pump), `keyboard.service`, `media-list`,
  `panel-resize.directive`, `find-view`, `browse-view`, `browse-canvas`. These
  *will* silently stop triggering CD the moment zone.js is gone.
- **Timer/promise/raw-listener callbacks mutating template state** (≈ the
  `setTimeout` reset/flash idioms): `voting-overlay` flashes, `image-viewer`
  shake + the window drag/key handlers, `center-panel` vote-spin/undo-toast,
  `settings-modal` "Saved" badge, `toast-container` copied state,
  `clipboard-copy` button text, `browse-view` build poller, `browse-minimap`
  resize handle, `browse-hover-preview` fetch, the four "setTimeout(() =>
  this.close())" modal closers. (Counts: 38 `setTimeout`, 1 `setInterval`, 16
  `requestAnimationFrame` — most rAF is canvas and safe.) Full list: Appendix A §C.
- **One `effect()` latent bug pattern** already in the tree: `center-panel`'s
  constructor effect reads `settingsState.settingsSignal()` and writes **plain**
  fields (`this.volume`, `this.audioPlaying`, …) bound in the template. Correct
  under zone today; under zoneless those fields must become signals.
- **`VtDialogService` holds dialog state in plain fields** (`dialogOpen`,
  `dialogMessage`, `dialogType`, `dialogShowInput`, `dialogInputValue`,
  `dialogButtons`) that `dialog-host.component` binds directly, with **no**
  signal / `markForCheck` / `async` anywhere in the path. `show()` runs
  synchronously inside `confirm()`/`prompt()`/`confirmDestructive()`. Most call
  sites invoke these as the first statement of a bound `(click)` handler, so
  `show()` runs in the click's CD-scheduling stack and is fine — **but** any call
  made from a `.then()` / post-`await` continuation (e.g.
  `find-view.component.ts:674` `this.dialog.prompt(...).then(...)`) runs `show()`
  in a microtask outside any notification, so under zoneless **the dialog would
  not appear**. This is fragile-by-construction (correctness depends on the
  caller's stack). Fix defensively: signalize the dialog state (Recipe B) so it
  is correct from any call context. (Note: `dialog.service.ts` also imports
  `ApplicationRef`/`createComponent`/`EnvironmentInjector`/`ComponentRef` and
  declares `modalRef` but **uses none of them** — dead code; drop it while here.)
- **Raw `ResizeObserver` / `MutationObserver` callbacks** are the same un-patched
  callback class as raw `addEventListener` (zone.js never patched them either, so
  this is also latent today on default CD). Three write template-bound state:
  `browse-legend.component.ts:73` (`MutationObserver` → `this.theme`, read in the
  colormap key at L92), `image-viewer.component.ts` (`ResizeObserver` →
  `recomputeRenderedSize` writes `renderedW`/`renderedH` ~L332/341/348, read in
  region/overlay math), `media-list.component.ts:367` (column-count relayout).
  The observers feeding
  only canvas redraws (`browse-canvas`, `browse-minimap`) are Recipe E (safe).

---

## 3. The conversion recipes (canonical patterns)

Every site falls into one of these. The recipes are written so the result is
correct under **both** zone and zoneless, so they can land while prod is still
zoned.

### Recipe A — `subscribe`-into-plain-field → `| async` pipe (cheapest)

When a component subscribes to a service Observable in `ngOnInit` only to mirror
it into a field the template reads, delete the subscription and bind the
Observable through `async`. The `AsyncPipe` handles subscribe/unsubscribe and
calls `markForCheck` on emit.

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
Drop the field, the `sub`, and the `ngOnDestroy` if it was its only user. Use
this for clean single-source reads (`toast-container`, the stats modals' simple
cases, list reads not already on `rxResource`).

### Recipe B — `subscribe`/`BehaviorSubject` → signal (preferred for hot/shared state)

For service state read in many places, or where you also need synchronous
"latest value" access, convert the `BehaviorSubject` to a `signal` (mirror the
`SettingsStateService`/`MediaStateService` pattern). Consumers read the signal in
the template (or via `computed`); imperative consumers read `sig()`.

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
schedules CD because it is a signal write read in a template. For consumers that
still want an Observable (rare), bridge with `toObservable(sig)`; for the reverse
(an Observable you must keep, exposed as a signal) use `toSignal(obs$)`.

### Recipe C — imperative callback → `markForCheck()` (escape hatch)

When a value genuinely must be mutated from a timer/raw-listener and converting
to a signal is disproportionate, inject `ChangeDetectorRef` and call
`markForCheck()` after the mutation. Works regardless of `OnPush` and schedules
CD under zoneless. Prefer A/B; reserve C for local, leaf, animation-ish state
(e.g. a `setTimeout` that clears a flash class) where a signal is overkill — but
note a signal is usually *also* clean here, so default to signals and use C only
where it clearly reads better.

### Recipe D — `ngZone.run()` re-entry → drop the run, trigger CD properly

Remove the `ngZone.run(...)` wrapper (keep any surrounding `runOutsideAngular`,
which stays harmless). Replace the CD purpose with a **signal write** (best) or
`markForCheck()`. Outputs emitted from a non-Angular callback (`emit()` inside a
former `ngZone.run`) need the *parent* to be notified — model the value as a
signal the parent reads, or `markForCheck` the parent; a bare `output()` emit
from an unpatched callback will not, by itself, schedule the parent's CD.

### Recipe E — leave canvas rAF / `runOutsideAngular` alone

`requestAnimationFrame` loops that only draw to `<canvas>` (`browse-canvas`,
`browse-minimap`, `audio-crop-overlay`, the progress-modal chart) need no CD and
must **not** be "fixed" into triggering it — that would regress performance.
Likewise `runOutsideAngular` perf wrappers stay; only their inner `.run()`
re-entries (Recipe D) change.

### Recipe F — `effect()` writing template-bound state → signalize the field

An `effect()` that writes a plain template-bound field (e.g. `center-panel`'s
settings mirror) must write a **signal** instead, and the template must read that
signal. Either make the destination a `signal()` the template calls, or — if the
source is already a signal — replace the effect+field with a `computed()` and
read the computed in the template (no effect needed).

---

## 4. Phasing

Each phase is independently shippable and gated on `./run-tests.sh` (the only CI;
there is none in GitHub). Phases 1–2 are behavior-neutral under the still-zoned
production app. Phase 0 must come first; Phase 3 (the flip) must come last.

### Phase 0 — Make the test suite catch zoneless staleness (no production change)

This is the linchpin and the part the original stub correctly flagged as missing.
Today every component spec calls `fixture.detectChanges()` manually, which
*force-renders* and therefore stays green even if the real app would go stale.
We fix the harness so it exercises real zoneless scheduling, **before** touching
production.

0.1 **Decouple the test target's polyfills from the prod build.** ✅ **DONE.**
Today the `test` target inherited `polyfills: ["zone.js"]` from `buildTarget:
frontend:build:development`. The `@angular/build:unit-test` builder has no
`polyfills` option of its own (its schema rejects it), so the decoupling is done
with a dedicated **`build:test` configuration** that pins its own polyfills,
pointed at by the unit-test `buildTarget`. A later removal of zone.js from the
base production polyfills then cannot silently break fakeAsync:
```jsonc
// architect.build.configurations.test
"test": {
  "optimization": false,
  "extractLicenses": false,
  "sourceMap": true,
  // testing BEFORE vitest-patch: the patch needs ProxyZoneSpec at load time,
  // and the builder appends zone.js/testing last, so list it explicitly first.
  "polyfills": ["zone.js", "zone.js/testing", "zone.js/plugins/vitest-patch"]
},
// architect.test.options
"test": {
  "builder": "@angular/build:unit-test",
  "options": {
    "tsConfig": "tsconfig.spec.json",
    "buildTarget": "frontend:build:test",
    "runner": "vitest",
    "runnerConfig": "vitest.config.ts",
    "setupFiles": ["src/test-setup.ts"]
  }
}
```
`zone.js/plugins/vitest-patch` is the documented bridge that keeps
`fakeAsync`/`tick`/`waitForAsync` working under the Vitest builder (Appendix C
#7), but it ships only in **zone.js ≥ 0.16**, so Phase 0 bumped `zone.js` to
`~0.16.0` (in Angular 21.2's `~0.15.0 || ~0.16.0` peer range). It is explicitly a
**transitional** mechanism — Angular recommends migrating to native `async` +
Vitest fake timers long-term (deferred; Open follow-ups). The hand-rolled
ProxyZone shim in `test-setup.ts` was fully subsumed by `vitest-patch` and
removed **after** confirming all 43 fakeAsync occurrences (11 files) still pass.

0.2 **Add a zoneless `TestBed` helper.** Create
`frontend/src/app/testing/zoneless-testbed.ts` exporting a providers fragment
`[provideZonelessChangeDetection()]` (and a small `configureZoneless()` helper).
Specs opt in as their component migrates (see per-component recipe). Enabling it
**globally** in Phase 0 would turn the suite red en masse (none of the 221 sites
are ready yet), so it is adopted **per component, in lockstep with Phases 1–2**.

0.3 **Establish the staleness-catching spec pattern.** For a migrated interactive
component, its spec must (a) use the zoneless `TestBed`, (b) drive updates through
the *same channel the app uses* (push to the service subject/signal, dispatch a
bound event), (c) `await fixture.whenStable()` — **not** `fixture.detectChanges()`
— and (d) assert on `fixture.nativeElement.querySelector(...)` (rendered DOM),
**not** on `component.someField`. Rationale (Appendix C #9): `detectChanges()`
force-runs CD even when Angular would not have scheduled it, masking the exact
staleness bug; `whenStable()` flushes only *scheduled* CD, so a missing
notification surfaces as a failing DOM assertion. Add a `settleZoneless()` helper
next to the existing `settleResource()` if a shared drain is useful.

**Important harness nuance:** adding `provideZonelessChangeDetection()` to a
`TestBed` flips `ComponentFixtureAutoDetect` **on by default** (it defaults off
under zone-based TestBeds). That is what makes the canary work — the fixture
refreshes only what CD actually schedules. The corollary is that the **188
existing `fixture.detectChanges()` calls (39 files) must be *removed*, not
supplemented**, as each spec migrates: a leftover manual `detectChanges()` both
re-masks staleness *and* can throw `ExpressionChangedAfterItHasBeenChecked`
(NG0100) when it races auto-detect. So the per-component spec migration is
"add the zoneless provider, delete the manual `detectChanges()` pumps, assert DOM
after `whenStable()`", not "add provider on top of the existing pumps".

0.4 **Add one "staleness canary" spec per migrated interactive component:** mutate
the backing value through the production channel **without** any manual CD pump,
`await whenStable()`, assert the DOM reflects it. This directly exercises the
scheduling path and fails loudly if a component forgets the signal write /
`markForCheck`.

0.5 **Keep `isolate: true` and the TestBed cascade guard.** Both are load-bearing
and become *more* important as specs adopt `whenStable()`/auto-detect (which
raise the chance of teardown-time `ExpressionChangedAfterItHasBeenChecked` or
unflushed-resource throws). Do not relax either.

**Phase 0 gate:** `./run-tests.sh frontend` green with the new test polyfills and
the helper in place (no component migrated yet, so behavior is unchanged); the
ProxyZone shim either retained or cleanly replaced with all fakeAsync specs
passing.

### Phase 1 — Signalize the hot shared services (highest leverage)

These three services are read across the whole app and contain the NgZone CD
re-entries; converting them neutralizes the largest blast radius. Each lands with
its consumers and specs updated, under zone-based prod (behavior-neutral).

1.1 **`ProgressEventsService` (SSE pump) — the single most important rewrite.
✅ DONE.** Dropped the `this.zone.run(...)` wrappers in `listen()` and `onopen`
and removed `NgZone` from the service. The six `BehaviorSubject` channels
(`dataset`, `loadingTasks`, `detectorLoadingTasks`, `sort`, `find`, `eval`) are
now `signal`s exposed read-only; a signal write inside the EventSource callback
notifies the scheduler with no zone. The synchronous latest-value getters became
signal reads (callable — `loadingTasks()`/`detectorLoadingTasks()`; the unused
`find` getter was dropped). `votingIterations` is now a `computed`. `serverReset$`
stays a `Subject` (its only consumer, `dashboard-loading-tasks`, is imperative).
The many consumers that compose channels with RxJS operators
(takeUntil/filter/take — `dashboard-loading-tasks`, `toast`, `context-switch`,
`find-view`, `label-view`, `dashboard`, `progress-modal`) keep their `$`
observables, now `toObservable` bridges over the backing signals, so a signal
write still drives them; this is behavior-neutral under the still-zoned prod app.
Per-component `| async`→signal template reads are deferred to those components'
Phase 2 conversions. New zoneless staleness canary in
`progress-events.service.spec.ts` drives an SSE frame through the fake EventSource
and asserts the rendered DOM repaints with no manual `detectChanges`.

1.2 **`KeyboardService`. ✅ DONE.** Kept the `runOutsideAngular` keydown listener
(harmless no-op under zoneless, still avoids per-keystroke churn under zone) and
the injected `NgZone` it needs. Dropped all 10
`this.zone.run(() => this.action$.next(...))` re-entries — `action$` now emits
plainly (it stays a `Subject`; the consumer composes it, so a signal would have
bought nothing). The CD trigger moved to the **consumer** (`center-panel`),
whose shortcut/timer/HTTP-driven template-bound state was signalized:
`isVoting`, `volume`, `audioPlaying`, `showAnimations`, `showMetadata`,
`swipeClass`, `spinningVote`, `undoToastText`, `pendingBadConfirm`, and the
private `labelHintDismissed`. The Recipe-F settings-mirror `effect()` now writes
those signals (`.set()`) instead of plain fields, fixing the latent
plain-field-write trap in one stroke. This is the center-panel slice of Phase
2.3; `image-viewer`/`video-player` remain for the rest of 2.3. New zoneless DOM
canary `center-panel.zoneless.spec.ts` drives a real `keydown` through the live
`KeyboardService` listener (an un-bound document callback) and asserts the vote
renders + the undo toast appears with no manual `detectChanges`; the existing
contract spec was updated to the signal accessors.

1.3 **`ConnectionStateService`. ✅ DONE.** Converted `statusSubject`/
`retryingSubject` to signals exposed read-only (`status`/`retrying`, backed by
private `_status`/`_retrying` writables, so only the service flips them). The
`offline`-event raw listener's `goOffline()` and the interceptor's
`recordSuccess`/`recordNetworkFailure` now schedule CD via the signal write. The
`offline-banner` template dropped `| async` for `status()`/`retrying()` signal
reads (and lost its now-unused `CommonModule` import). The sole observable
consumer, `ProgressEventsService`, replaced its `status$.pipe(distinctUntilChanged())`
constructor subscription with an `effect(() => …connection.status()…)` (signals
are distinct by default). The `errorInterceptor` is untouched — it only uses the
imperative methods (`isOffline`/`recordSuccess`/`recordNetworkFailure`), which
keep working. New `offline-banner.component.spec.ts` is a zoneless DOM canary
driving the breaker through `recordNetworkFailure`/`recordSuccess`/`retry()`.

1.4 **`BrowseSelectionService` (`changed$`). ✅ DONE.** Replaced the `changed$`
`Subject` with a signal: the existing `_version` counter became
`signal(0)`/`bump()` and is exposed as a read-only `version` signal. The three
consumers were converted off `.changed$.subscribe(…)`: `browse-canvas` repaints
via an `effect(() => { selection.version(); requestRedraw(); })` (its `selStateFor`
memo now reads `version()`), and `browse-bin-popup` / `browse-selection-panel`
react via an `effect` on `version()` (popup keeps its `markForCheck` discipline;
panel re-runs `refreshSelection`). New `browse-selection.service.spec.ts` adds a
unit spec for the selection logic plus a zoneless signal-driven view canary.

**Phase 1 gate:** each service's spec + its consumers' specs migrated to the
zoneless `TestBed` (Recipe in 0.3) and green; `./run-tests.sh` full pass.

### Phase 2 — Convert components, cluster by cluster

Order by route so each cluster is shippable and QA-able as a unit. For every
component: apply Recipe A/B/C/D/F as appropriate, switch its spec to the zoneless
`TestBed` with DOM assertions (0.3) + a canary (0.4), and confirm red→green.

Suggested order (lightest/most-isolated first to build confidence):

1. **Leaf utility components. ✅ DONE (Phase 2.1).** `toast-container` binds
   `toasts$` via `| async` (Recipe A) and `copiedId` became a signal;
   `clipboard-copy`'s `buttonText` flash became a signal (it is set from a
   post-`await` continuation *and* a timer — the exact zoneless-sensitive path);
   `voting-overlay`'s `goodFlash`/`badFlash` became signals and its `@Input()`
   decorators were modernized to `input()`; `field-hint-icon` was already
   signal-based (no change); `dialog-host` + **`VtDialogService`** had all dialog
   state (`dialogOpen`/`dialogTitle`/`dialogMessage`/`dialogType`/
   `dialogShowInput`/`dialogInputValue`/`dialogButtons`) signalized so a
   `confirm`/`prompt` opened from a non-event callback still schedules CD, and the
   dead `ApplicationRef`/`createComponent`/`EnvironmentInjector`/`modalRef` code
   was dropped; `browse-legend`'s `theme` became a signal driven by the
   `data-theme` `MutationObserver`. Each spec now runs under the zoneless
   `TestBed` (`configureZoneless` + `settleZoneless`, no manual `detectChanges`)
   with a staleness canary; new specs added for `toast-container`,
   `clipboard-copy`, and `browse-legend`.
2. **Stats / picker modals. ✅ DONE (Phase 2.2).** `find-stats`/`detector-stats`/
   `dataset-stats` signalized `loading`/`error`/`stats` (Recipe B; templates use
   `@else if (stats(); as stats)`). `label-importer` moved its list read onto an
   eager `rxResource` and signalized all its subscribe-written fields (mutation
   results + the dynamic-field-option dicts). `settings-importer`/
   `settings-exporter`/`label-exporter` were already on `rxResource`+signals;
   their `submitting`/`successMessage` mutation-result fields were signalized to
   finish them. The four `setTimeout(close)` paths needed no change — a
   template-bound `(closed)` output emit schedules the parent's CD under zoneless
   (proven by `testing/output-emit-zoneless.spec.ts`; see Open follow-ups). Each
   has a zoneless DOM-canary spec.
3. **`center-panel` + viewers. ✅ DONE (Phase 2.3).** The center-panel voting
   state shipped with Phase 1.2. Phase 2.3 finished the viewers: `image-viewer`
   signalized the seven fields written from un-patched callbacks — `regionBox`,
   `shiftHeld`, `panX`/`panY` (window `mousemove`/`mouseup` drag + `keydown`/
   `keyup`/`blur` listeners), `renderedW`/`renderedH` (the `ResizeObserver`
   rendered-size writes), and `regionBoxShake` (the shake `setTimeout`); its
   getters (`imageTransform`/`regionBoxStyle`/`regionDrawActive`/`wrapCursor`)
   now read those signals and stay plain getters (so the contract spec keeps
   reading them without `()`). `zoom`/`rotation`/`zoomLabel`/`marqueeMode`/
   `highlightMode`/`imageReady`/`imageSrc` stayed plain — they are written only
   from bound handlers / `ngOnChanges`, which already schedule CD, and the parent
   `center-panel` reads `marqueeMode`/`highlightMode`/`zoom`/`zoomLabel` only via
   bound-event-driven re-checks. `video-player` was verified DOM-only and left
   unchanged: its `videoSrc`/`videoError` are written only in `ngOnChanges` and
   the bound `(error)` handler, and the `setInterval` clip-loop only writes
   `video.currentTime` (a DOM property, not template state); `(loadedmetadata)`/
   `(play)`/`(pause)`/`(error)` are bound listeners. The existing image-viewer
   contract spec was updated to the signal accessors (kept on the default
   TestBed, mirroring the center-panel split); a new
   `image-viewer.zoneless.spec.ts` canary drives a real window `keydown`/`keyup`/
   `blur` (Shift) through the live constructor listener and asserts the
   `.region-mode` crosshair affordance repaints with no manual `detectChanges`.
4. **`label-view`. ✅ DONE (Phase 2.4).** Signalized every template-bound field
   written from a non-CD-scheduling context: `datasetName`, `labelingStatus`
   (status-polling timer), `trainableModelName`, `leftWidth`/`rightWidth` (the
   settings-mirror `effect()` — Recipe F), `autopilotCollapsed`,
   `autopilotEnabled`, `showResortPrompt`, and `cropPending` (set from a
   `fetch().then()` continuation — the un-bound microtask path). The two
   constructor `effect()`s now read only their intended tracked signals and run
   the rest `untracked`, because `applyPanelPx`/`setAutopilotCollapsed` both read
   *and* write the width/collapse signals (an `effect` that reads a signal it
   also writes would loop, and would spuriously revert a manual collapse toggle
   on stale settings). label-view also binds `SortStateService` (11 reads) and
   `VoteStateService` (good/bad votes, label counts, derived
   `learnedSortAvailable`) getters directly in its template; those services stay
   `BehaviorSubject`-backed (un-signalized, per the per-cluster plan and the
   center-panel precedent of not signalizing shared services mid-migration), so
   label-view bridges the channels it binds into local signals via
   `toSignal(…$, { requireSync: true })` (a `computed` for `learnedSortAvailable`)
   — the cheapest way to make those reads repaint under zoneless without
   touching the shared services or the not-yet-migrated find-view/right-panel
   consumers. The learned-sort `setTimeout` needed no change beyond the bridge:
   it writes only the private `learnedSortPending` guard and re-fires
   `onLearnedSort`, whose sortState writes now repaint via the bridge. New
   `label-view.zoneless.spec.ts` canary drives the `/api/dataset/status`
   subscribe (an un-bound callback) and asserts the left panel's `.dataset-name`
   header repaints with no manual `detectChanges`; the existing contract spec was
   updated to the signal accessors (kept on the default TestBed, mirroring the
   center-panel/image-viewer split).
5. **`find-view`. ✅ DONE (Phase 2.5).** Signalized its own subscribe/effect-written
   template state (`datasetName`, `viewModeLeft`/`gridGoalWidthLeft`/`focusModeLeft`/
   `focusModeRight`) and replaced the `combineLatest(sortOrder$, verifiedIds$)`
   subscribe with an `unverifiedSortOrder` `computed`. The four divider-drag
   `ngZone.run` re-entries were dropped outright (the widths only drive `--left-width`/
   `--right-width` CSS custom properties imperatively — not template-bound — so no
   CD is needed); `runOutsideAngular` on the listeners stays. The two `.then`
   confirm flows route through HTTP and were already safe. As the centerpiece of
   2.5, `SortStateService`/`VoteStateService` were signalized (see Open follow-ups),
   so find-view's `sortState.*` getter bindings became reactive with **zero**
   template churn. New `find-view.zoneless.spec.ts` canary drives the
   `/api/dataset/status` subscribe AND a `SortStateService` setter and asserts the
   DOM repaints with no manual `detectChanges`.
6. **Browse cluster. ✅ DONE (Phase 2.6).** `browse-view` signalized its 13
   template-bound fields written from async subscribes / the build poller / the
   settings `effect()` (`status`, `meta`, `mediaType`, build progress/total/
   message, `errorMessage`, `panelWidth`, `colormap`, `thumbnailBorder`,
   `hexScaleIndex`, `binShape`, `datasetName`); the divider-drag `ngZone.run`
   was dropped (panelWidth is a signal — its `.set()` schedules CD from the
   out-of-zone mousemove listener under both zoned and zoneless, mirroring the
   Phase 1.1 SSE-pump precedent). `browse-hover-preview` signalized `textContent`
   (written from the paragraph `fetch().then()` microtask). `browse-selection-panel`
   signalized `count`/`viewMode`/`gridGoalWidth`/`sortedEntries` — these were
   written from the selection-refresh + settings `effect()`s and the metadata
   `version$` subscribe, and the effect-into-plain-field writes were a **latent
   staleness bug** (Recipe F). `browse-canvas`, `browse-minimap`, and
   `browse-bin-popup` were verified zoneless-safe **as-is** and left unchanged:
   browse-canvas has no template-bound plain fields (its `ngZone.run`-wrapped
   output emits / `selection.*` calls are harmless no-ops under zoneless and
   load-bearing under the still-zoned prod, so they stay until Phase 5);
   browse-minimap's `width`/`height` feed only the canvas (not template-bound);
   browse-bin-popup is already `markForCheck`-disciplined and its drag is on
   `@HostListener` (a bound host listener, on the zoneless notification path).
   New zoneless canaries: `browse-selection-panel.zoneless.spec.ts` (selection
   signal bump + metadata `version$` emit), `browse-hover-preview.zoneless.spec.ts`
   (async fetch resolves), `browse-view.zoneless.spec.ts` (projection-load
   subscribe errors → error-state repaint).
7. **`left-panel` + `media-list`. ✅ DONE (Phase 2.7).** `media-list`:
   `loadingMedias` (a plain field written from a constructor `effect()` — Recipe F)
   → a `computed` over `mediaState.isLoading()`; added `cdr.markForCheck()` at the
   end of `rebuildOrderedItems()` so the list repaints from the async
   `metadataCache.version$` subscribe (the `cachedOrderedItems`/`gridRows` arrays
   stay plain — hot virtual-scroll state, repainted via the markForCheck); dropped
   the `zone.run(...)` wrapper around the relayout `cdr.detectChanges()`
   (`detectChanges` is zone-independent). `left-panel`: signalized `mediaTypeName`
   + `textSortAvailable`, plain template-bound fields written from constructor
   `effect()`s reacting to late-arriving media-type/embedder metadata (Recipe F).
   `panel-resize.directive` was left as-is: its `ngZone.run`-wrapped
   `widthChange`/`resizeEnd` emits are no-ops under zoneless, and the emit → the
   parent label-view's `(widthChange)`/`(resizeEnd)` handler writes the
   signalized `leftWidth`/`rightWidth` (Phase 2.4), which schedules CD and
   rechecks the directive's `[class.dragging]` host binding. (No new canary: the
   effect→signal fix is proven by the identical browse-selection-panel canary and
   the `left-panel` container deadlocks `whenStable()` via its media-grid
   rxResources; the existing specs cover behavior.)
8. **`right-panel`** (vote piles ✅ DONE in Phase 2.5 — six `subscribe` mirrors →
   `computed`s over the signalized `VoteStateService`; **remaining**: its
   `LabelsetStateService` mirror `goodElements`/`badElements`/`currentMediaType`
   and the settings-derived `viewMode`/`gridGoalWidth`, all still
   subscribe/effect-written plain fields), **`context-pulldown`**, **`dashboard`**
   (32 — mostly `ngOnInit` list reads, good `async`/signal candidates),
   **`new-detector-modal`** / **`dataset-importer-modal`** (heaviest; convert the
   clean list reads, leave dynamic field-option fetches imperative with Recipe C
   where a signal is awkward), **`settings-modal`** (forkJoin init → signals;
   "Saved" badge timer → signal), **`load-sort-modal`** / **`resort-prompt-modal`**.
9. **`app.component`** and remaining shell pieces.

**Phase 2 gate (per cluster):** the cluster's specs run under the zoneless
`TestBed`, assert on the DOM, include canaries, and pass; `./run-tests.sh` full
pass. After the *whole* of Phase 2, **every interactive component spec asserts
DOM under a zoneless `TestBed`** — that is the readiness bar for Phase 3.

### Phase 3 — Flip production to zoneless

Only after Phases 0–2 are complete and the interactive surfaces are green under
the zoneless `TestBed`:

3.1 In `frontend/src/app/app.config.ts`, replace
`provideZoneChangeDetection({ eventCoalescing: true })` with
`provideZonelessChangeDetection()`. Note the current provider's
`eventCoalescing: true` is **not** silently lost: zoneless schedules CD (it does
not run per-event), so event coalescing is the default behavior — if anything the
flip preserves/improves it. Still, re-verify the high-frequency drag handlers
(`find-view`, `browse-view`, `image-viewer`, `browse-minimap`) in Phase 4, since
they previously leaned on coalesced single-CD-per-frame.

3.2 In `frontend/angular.json`, remove `"zone.js"` from the **base** `build`
options `polyfills` (the production array). Leave the **`build:test`
configuration's** polyfills (set in 0.1) untouched — tests still need zone.js +
`zone.js/testing` + `vitest-patch` for fakeAsync. Because `build:test` pins its
own array, it is already decoupled and needs no change at the flip.

3.3 Keep `zone.js` in `package.json` (the test run imports it). It is no longer
bundled into prod because it is out of the build polyfills; no `npm audit`
concern (zone.js has no advisories).

3.4 Re-run `./run-tests.sh` (full). Expect possibly a handful of
`ExpressionChangedAfterItHasBeenChecked` surfacing now that CD timing is stricter;
fix them (CLAUDE.md "Fix All Errors" — no waving off).

3.5 **Budget:** with zone.js gone (~35 kB raw / ~11 kB transfer), drop the
initial budget from 540 kB back toward the framework floor (~495 kB), tightening
to the real measured size with a small headroom, and delete the `"//"` rationale
block in `angular.json` that points here. Do not pre-commit a number — measure
the post-flip `build:prod` initial size and set warn just above it.

**Phase 3 gate:** `./run-tests.sh` full pass; `build:prod` clean (0 `▲ [WARNING]`)
under the new budget.

### Phase 4 — Human browser QA (mandatory, cannot be skipped)

The container has no browser; the unit suite (even hardened) cannot 100% prove
real rendering. A human runs `python app.py --local`, builds the frontend, and
exercises every interactive surface, watching specifically for "stale until I
click" symptoms:

- Voting (good/bad, keyboard shortcuts, undo/redo, vote spinner, swipe anim).
- Sorting / autopilot (sort bar, progress bars driven by SSE).
- Train-and-score / find (eval progress modal, find progress).
- Dataset load progress (loading-tasks bars), offline banner + Retry.
- Browse canvas (pan/zoom, hover preview, bin popup, selection panel, minimap).
- All modals (open/close, the four `setTimeout(close)` paths, "Saved" badge,
  copied-to-clipboard text, importer/exporter flows).
- Left/right panel resizers (drag widths), context pulldown, dashboard lists,
  new-detector + importer modals.
- **Confirm/prompt dialogs** triggered from non-click paths (e.g. find-view's
  rename-after-`.then`), to prove the signalized `VtDialogService` opens reliably.
- **CDK virtual-scroll viewports** (`left-panel/media-list`, `browse-bin-popup`):
  scroll fast and check for item flicker / blank rows / stale viewport — a known
  zoneless interaction the jsdom suite cannot see (see Framework-surface notes).

A checklist version of the above goes in the PR description for the QA pass.

### Phase 5 — Cleanup / follow-ups

- Migrate the residual fakeAsync/timer specs to native `async` + Vitest fake
  timers and drop `zone.js/plugins/vitest-patch` (and finally `zone.js` from
  `package.json`) — Angular's recommended long-term direction (Appendix C #7).
  Optional, separable, no prod impact.
- Adopt `ChangeDetectionStrategy.OnPush` explicitly on components for clarity
  (under zoneless they already behave OnPush-like; this is documentation/intent,
  not a functional change).

### Open follow-ups

- **Phase 1 complete; Phases 2.1 + 2.2 + 2.3 + 2.4 + 2.5 shipped; Phases 2.6–5 remain.**
  Shipped so far: Phase 0 (harness); all of Phase 1 — 1.1 + 1.3 + 1.4
  (ProgressEventsService SSE pump + ConnectionStateService + BrowseSelectionService
  signalized) and 1.2 (KeyboardService de-zoned + the coupled center-panel state
  signalized); Phase 2.1 (leaf utility components — toast-container, clipboard-copy,
  voting-overlay, dialog-host + VtDialogService, browse-legend); **Phase 2.2**
  (stats / picker modals); **Phase 2.3** (center-panel viewers — image-viewer's
  window drag/key handlers + shake `setTimeout` + `ResizeObserver` rendered-size
  writes signalized; video-player verified DOM-only, no change); **Phase 2.4**
  (`label-view` — own subscribe/timer/effect-written fields signalized; the bound
  `SortStateService`/`VoteStateService` reads bridged via `toSignal`); and
  **Phase 2.5** (`find-view` + the shared-service signalization — see next bullet),
  each with its consumers and a zoneless canary spec. Remaining in **Phase 2**:
  clusters 2.6–2.9 in the suggested order (next up: 2.6 browse cluster). Note 2.8
  shrank to right-panel's **`LabelsetStateService`/settings mirror** only
  (`goodElements`/`badElements`/`mediaType`/`viewMode`/`gridGoalWidth`), since its
  vote piles landed with 2.5. Then the prod flip (Phase 3), human browser QA
  (Phase 4), and cleanup (Phase 5).
- **`SortStateService`/`VoteStateService` are now signal-backed (Phase 2.5).**
  Rather than continue the per-consumer `toSignal` bridge from 2.4, Phase 2.5
  signalized both services: each value is a private `signal` exposed via a
  *value-returning getter*, with the `set*` methods writing the signal. The `$`
  observables were dropped entirely (clean break — no shims). Because a signal
  read **through a getter** during template evaluation is tracked (pinned by
  `frontend/src/app/testing/getter-signal-zoneless.spec.ts`), existing
  `sortState.sortBusy` / `voteState.goodVotes` template bindings stayed
  byte-for-byte the same yet became reactive under zoneless. All consumers were
  migrated in the same change: find-view (template getters now reactive; its own
  `datasetName`/`viewModeLeft`/… signalized; `unverifiedSortOrder` is a
  `computed`; the divider `ngZone.run` re-entries dropped — widths are CSS-var
  only), label-view (its 15 `toSignal` bridges removed; binds the getters
  directly; the one-shot `labelsetGoodCount$` rehydrate sub → a counts-tracking
  `effect`), right-panel (six `subscribe`-into-field vote mirrors → `computed`s
  over the getters), and center-panel (`goodVotes$`/`badVotes$` subscribes → an
  `effect`; `toast$` stays a `Subject` — it is a fire-once event, not state).
  `toast$` is the only remaining observable on either service. **Done** — this
  was the "natural cleanup" the 2.4 follow-up anticipated; no per-consumer
  bridges remain for these two services.
- **What Phase 2.2 covered.** `find-stats`/`detector-stats`/`dataset-stats`:
  `loading`/`error`/`stats` plain fields → signals (templates read them via
  `@else if (stats(); as stats)` so the bodies were untouched); the stat-derived
  getters now read `stats()`. `label-importer-modal`: moved the list read onto an
  eager `rxResource` (mirroring `settings-importer`), and signalized every
  subscribe-written template field (`submitting`, `error` via an `importError`
  signal merged with the resource error, `successMessage`, `addingGood`,
  `addingBad`, and the three `dynamicFieldOptions`/`Loading`/`Error` dicts via
  `signal<Record<…>>` + `.update`). `settings-importer`/`settings-exporter`/
  `label-exporter` were already on `rxResource`+signals from the httpresource work;
  Phase 2.2 finished them by signalizing `submitting`/`successMessage`. New
  zoneless DOM-canary specs added for all three stats modals + the two settings
  modals; the existing `label-importer`/`label-exporter` specs migrated to the
  zoneless `TestBed` (no manual `detectChanges`). **rxResource test gotcha:** a
  loading `rxResource` holds the app *unstable*, so `await fixture.whenStable()`
  before flushing the GET deadlocks — the rxResource specs issue the GET with
  `TestBed.tick()` then `settleResource()` instead (a plain `ngOnInit` HTTP
  subscribe does *not* block `whenStable`, so the stats specs use it freely).
- **`setTimeout(close)` finding — Recipe D caveat corrected for template-bound
  outputs.** The four `setTimeout(() => this.close())` auto-close paths
  (`settings-importer`, `label-importer`, `settings-exporter`, …) needed **no**
  rework: `close()` emits a bound `output()`, and a parent's `(closed)="…"` is a
  *bound template listener*, which is on the zoneless notification path — so the
  emit schedules the parent's CD and its `@if` gate drops the modal even though
  the emit fired from an unpatched timer. Proven by a new framework canary,
  `frontend/src/app/testing/output-emit-zoneless.spec.ts`. Recipe D's warning that
  "a bare `output()` emit from an unpatched callback will not schedule the
  parent's CD" holds only for an output the parent subscribes to **imperatively**
  in TS (a raw callback), *not* for a template `(event)` binding. So only the
  components' own template-bound state (`submitting`/`successMessage`/…) was
  signalized; the timer-driven close was left as-is.
- **Full consumer specs for the browse cluster.** Phase 1.4 added a service unit
  spec + a signal-driven canary, but `browse-canvas`, `browse-bin-popup`, and
  `browse-selection-panel` still have **no** component specs (they are heavy to
  mock: CDK virtual scroll, metadata cache, canvas 2D context). Real per-component
  zoneless DOM specs for them are deferred to their Phase 2.6 conversion.
- **Drop the `vitest-patch` bridge (Phase 5).** Migrate the 43 fakeAsync
  occurrences (11 files) to native `async` + Vitest fake timers, then remove
  `zone.js/testing` + `zone.js/plugins/vitest-patch` from the `build:test`
  polyfills and finally `zone.js` from `package.json`. Angular's recommended
  long-term direction; separable, no prod impact.
- **Adopt the zoneless `TestBed` per component.** `provideZoneless()` /
  `configureZoneless()` and `settleZoneless()` exist but are used only by the
  reference spec. Each migrated component (Phases 1–2) must switch its spec to
  them, delete the manual `fixture.detectChanges()` pumps, and add a canary
  (0.4). 188 `detectChanges()` calls across 39 files still to convert.

---

## 4.5 Framework-surface notes (verified non-gaps + watch-items)

Each Angular/CDK/3rd-party surface was checked against this codebase; recording
the verdicts so a reviewer doesn't have to re-derive them.

**Watch-items (do require attention):**

- **CDK virtual scroll.** `@angular/cdk` is used only via `@angular/cdk/scrolling`
  (`CdkVirtualScrollViewport`) in `left-panel/media-list` and `browse-bin-popup`
  (no overlay / portal / drag-drop / a11y / `LiveAnnouncer` usage). CDK virtual
  scroll has historically had zoneless repaint/flicker interactions (it leans on
  `NgZone`/`onStable` internally). The `media-list` `zone.run(() =>
  cdr.detectChanges())` relayout (Appendix A §A) sits right next to the viewport.
  Action: confirm the pinned CDK 21.x carries the virtual-scroll zoneless fix,
  and **browser-QA both viewports for scroll staleness/flicker** (Phase 4) — the
  jsdom suite cannot exercise real scroll geometry.
- **`ngModel` programmatic writes.** 58 files use `FormsModule`/`ngModel`.
  Two-way binding is safe because `(ngModelChange)` is a *bound* listener that
  schedules CD. The one caveat: writing an `ngModel`-bound **plain field** from a
  non-event callback won't refresh the input — the same rule as everywhere, but
  worth calling out given the breadth. Covered by the general subscribe/observer
  conversions; no separate phase.

**Verified non-gaps (no action needed):**

- **Router / lazy routes / `@defer`.** `app.routes.ts` uses `loadComponent` lazy
  routes; `app.component.html` has 5 `@defer (when …)` modals. Router events and
  `@defer` triggers are framework-driven and on the zoneless notification path.
  The `@defer (when <plainField>)` gates (`showSettings`, importer-flow flags…)
  flip inside bound `(click)` handlers, which schedule CD. Safe.
- **`HttpClient` / interceptors.** `HttpClient` is not itself a CD trigger;
  HTTP-result repaint flows through the *consumer* (signal/`async`/`markForCheck`)
  — which is exactly what the subscribe-assign conversion handles. The three
  interceptors (active-context, achievements-refresh, error/circuit-breaker) run
  in the request chain and don't touch template state. Safe (Phase 1.3 already
  notes the circuit breaker keeps working).
- **Angular animations.** Not used — no `@angular/animations`, no
  `provideAnimations`, no `trigger()`/`[@…]` bindings. Transitions are CSS-only.
- **`marked`.** Called with `{ async: false }` in `keyboard-help-modal` — fully
  synchronous, no CD interaction.
- **`onStable` / `onMicrotaskEmpty` / `afterRender`.** Zero usages in app code,
  so nothing to migrate to `afterNextRender`. (CDK may use `onStable` internally
  — folded into the CDK watch-item above.)
- **SSR / hydration.** None (no `provideClientHydration`, no server rendering).
- **`PendingTasks`.** Not needed: there's no SSR/test-stability requirement, and
  the SSE pump / pollers don't need to hold the app "unstable". `whenStable()` in
  tests resolves correctly.
- **Bootstrap.** No special handling beyond the Phase 3.1 provider swap.
- **video/audio `(timeupdate)`/`(play)`/`(loadedmetadata)`.** Bound template
  listeners → schedule CD. Safe.

## 5. Verification strategy (how we know it works without a browser)

The crux of "be very careful, we can't check the frontend here": we do **not**
rely on eyeballing. The plan makes the headless suite a genuine zoneless oracle:

1. **Per-component proof.** Each migrated component's spec runs its `TestBed`
   under `provideZonelessChangeDetection()`, drives state through the production
   channel, `await`s `whenStable()`, and asserts the **rendered DOM**. A missing
   notification ⇒ stale DOM ⇒ failing assertion. (Appendix C #9.)
2. **Canary specs** (0.4) explicitly test the scheduling path with zero manual
   CD pumping, so a forgotten signal write / `markForCheck` cannot pass.
3. **Lockstep migration.** Because the zoneless `TestBed` is adopted per component
   in Phases 1–2, the suite is fully green under zoneless semantics *before* the
   prod flip — the flip itself (Phase 3) is then a near-no-op for behavior.
4. **Human QA (Phase 4)** is the final backstop for anything the DOM-level unit
   assertions can't capture (real paint timing, focus, canvas interaction).

This is why the order matters: harness first, components second (validated under
zoneless tests), flip third, human QA last.

---

## 6. Rollback plan

- Phases 1–2 are individually revertible commits and are behavior-neutral under
  zone, so a regression found in QA reverts just the offending cluster.
- Phase 3 is a 3-line revert (`app.config.ts` provider + the `angular.json`
  build-polyfills line + budget). Because Phases 1–2 are correct under zone too,
  reverting only Phase 3 returns to a fully working zone-based app with the
  modernized reactivity intact.
- The test-target polyfills (Phase 0) are independent of the prod flip and need
  not be reverted.

---

## 7. Risks & gotchas (carry these into every PR)

- **Silent fakeAsync breakage.** If the test target loses zone.js (e.g. someone
  "cleans up" polyfills), the ProxyZone shim no-ops *silently* and all 43
  fakeAsync occurrences throw. Phase 0.1 decouples test polyfills precisely to
  prevent this; never let the test target depend on the build target for zone.js.
- **`rxResource` is invisible to `fakeAsync`** (already documented in
  `httpresource-migration.md`): the virtual clock doesn't drive promise-based
  resource resolution. As more reads become resources/signals, *more* specs must
  leave fakeAsync for real-async + `settleResource()`. The fakeAsync count should
  **shrink** over the migration, not grow.
- **`effect()` plain-field trap** (Recipe F): the existing `center-panel` effect
  is the canonical example. Audit every `effect()` (20 of them) for plain
  template-bound writes during migration.
- **Output emits from non-Angular callbacks** (Recipe D): a bare `output()` emit
  from a former `ngZone.run` won't schedule the parent's CD — model as a signal
  or `markForCheck` the parent.
- **Do not "fix" canvas rAF loops** into triggering CD (Recipe E) — performance
  regression.
- **`ExpressionChangedAfterItHasBeenChecked`** may surface post-flip and in
  `whenStable()` specs; contained by `isolate:true` + the cascade guard but still
  real failures to fix (no waving off, per CLAUDE.md).
- **`@Input()` setters** (39 files still on decorator inputs): any setter that
  mutates template state needs the same treatment as a subscribe-assign.
- **`ResizeObserver`/`MutationObserver`** are as un-patched as raw
  `addEventListener` — audit them as a first-class class, not as effects.
- **`VtDialogService` fragility:** dialog visibility currently depends on the
  caller's stack; signalize it so confirm/prompt are correct from any context.
- **CDK virtual scroll** can flicker/stale under zoneless — browser-QA only.
- **`ngModel` two-way bindings** (58 files): safe via `(ngModelChange)`, but a
  programmatic plain-field write from a non-event callback won't refresh.

---

## 8. Open questions for the user

These are choices the plan can proceed without (defaults noted) but that change
scope at the margins:

- **Conversion bias:** default is "`| async` where the source stays an Observable
  and is read in one place (Recipe A); signalize hot/shared service state (Recipe
  B); `markForCheck` only as a leaf escape hatch (Recipe C)." If you'd rather go
  *all-signals* (convert every Observable service to signals) for uniformity, the
  scope grows but the end state is cleaner.
- **Phase 5 timing:** drop the vitest-patch bridge and convert timer specs to
  Vitest fake timers as part of this effort, or defer to its own follow-up
  (default: defer).
- **Explicit `OnPush`:** annotate components with `ChangeDetectionStrategy.OnPush`
  for intent (default: skip — redundant under zoneless).

---

## Appendix A: file-level work catalog

### §A — NgZone `.run()` re-entry purely for CD (Recipe D) — hard blockers

| File | Sites | What it protects | Zoneless fix |
|---|---|---|---|
| `services/progress-events.service.ts` | `listen()` per-frame (~6 channels) + `onopen` | SSE frames fire outside zone; `.run()` repaints progress UI | Signalize 6 channels; drop `zone.run`; remove `NgZone` |
| `services/keyboard.service.ts` | 10 (`action$.next`) | shortcut dispatch repaints center-panel | Drop `zone.run`; signalize consumer state |
| `components/left-panel/media-list/media-list.component.ts` | 1 (`zone.run(() => cdr.detectChanges())`) | grid relayout flush (OnPush) | Drop `zone.run`, keep `markForCheck()` |
| `components/label-view/panel-resize.directive.ts` | 2 (`widthChange`/`resizeEnd` emit) | parent panel width | Signal width / `markForCheck` parent |
| `components/find-view/find-view.component.ts` | 4 (`leftWidth`/`rightWidth`) | divider drag widths | Signal widths |
| `components/browse-view/browse-view.component.ts` | 1 (`panelWidth`) | divider drag width | Signal width |
| `components/browse-canvas/browse-canvas.component.ts` | ~8 (`densityMaxChanged`/`contextMenu`/`hexHover`/`selection.*`) | cross-boundary emits to parent/bin-popup | Recipe D; canvas rAF unchanged |

### §B — `runOutsideAngular` / canvas rAF (Recipe E) — leave as-is

`keyboard.service:28`, `panel-resize:55`, `find-view:359/401`, `media-list:366`,
`browse-view:421`, `browse-canvas:350/360/885/997`, `browse-minimap:151/367`
(perf wrappers — keep). Canvas rAF draw loops: `browse-canvas` (multiple),
`browse-minimap:238`, `media-crop-modal/audio-crop-overlay`, `progress-modal`
chart — keep; do not make them trigger CD.

### §C — Timer / promise / raw-listener mutating template state (Recipe B/C)

`voting-overlay:34,43` (flash classes); `image-viewer:296` (shake) and
`:420–512` (window drag/key → `panX/panY/regionBox/zoomLabel/shiftHeld` — the
largest single offender); `center-panel:203` (undo toast), `:310` (spinningVote),
`:311` (isVoting + emit), `:147` (visibilitychange playback);
`settings-modal:537` ("Saved" badge); `toast-container:97` (copiedId);
`clipboard-copy:123` (buttonText); `browse-view:830,843,847` (build poller →
status/progress/meta); `browse-minimap:408–409` (resize-handle listener, NOT in
`runOutsideAngular`, mutates bound width/height); `browse-hover-preview:115–116`
(`fetch().then` → textContent); `label-view:615` (learned-sort toggle);
`settings-importer:125`/`label-importer:213`/`settings-exporter:124`/
`examples-editor:82` (`setTimeout(() => this.close())` emit). Service-level
subjects pushed from timers/raw-listeners that are consumed via **plain-field
subscribe** (so the *consumer* needs A/B): `toast.service:160`,
`browse-prep.service:220`, `connection-state.service:83` (the last is safe for
`offline-banner` because that consumer uses `| async`, but signalize anyway per
Phase 1.3). Raw `ResizeObserver`/`MutationObserver` writing template state (same class as
raw `addEventListener`): `browse-legend:73` (`MutationObserver` → `theme`),
`image-viewer` (`ResizeObserver` → `recomputeRenderedSize` ~L332 →
`renderedW`/`renderedH`), `media-list:367` (column-count relayout). Observer callbacks feeding only canvas
redraws (`browse-canvas`, `browse-minimap`) are Recipe E (safe). Plain-field
dialog state in `VtDialogService` (consumed by `dialog-host`) → signalize
(Recipe B). DOM-only `setTimeout`s (focus/scroll/select: `context-pulldown:216`,
`detector-context-bar:26`, `dataset-card:102`, `detector-card:113`,
`folder-browser:154,386`, `import-config:142` queueMicrotask) are **safe** — no
template field.

### §D — `subscribe`-into-plain-field reads (Recipe A/B), by component

High-count `ngOnInit`/event list reads (good `async`/signal candidates):
`dashboard` (32), `label-view` (24), `dataset-importer-modal` (22), `find-view`
(13), `browse-view` (13), `new-detector-modal` (12), `context-pulldown` (11),
`center-panel` (11), `right-panel` (10), `load-sort-modal` (8), plus the modal
catalog: `find-stats`/`detector-stats`/`dataset-stats` (stats reads, Recipe A);
`label-importer` (list + dynamic field options), `examples-editor`,
`autodetect-results`, `resort-prompt`, `auto-find-settings`,
`import-defaults-settings`, `settings-modal` (forkJoin init), `progress-modal`
(timer poller + SSE iterations). Mutation-result subscribes (POST/PUT) keep their
imperative `HttpClient` call but their *template-bound result fields*
(`submitting`/`successMessage`/`status`/`error`) become signals or `markForCheck`.

### §E — `effect()` audit (Recipe F)

20 `effect()` calls across 16 files. Known plain-field-write trap:
`center-panel:91` (settings mirror → `volume`/`audioPlaying`/`showAnimations`/
`showMetadata`/`labelHintDismissed`). Audit the other 19 (`settings-state` ×2,
`left-panel` ×2, `label-view` ×2, `find-view` ×2, and one each in
`achievements.service`, `view-controls`, `right-panel`, `export-modal`,
`media-list`, `dashboard`, `browse-view`, `browse-selection-panel`,
`browse-bin-popup`, `achievements-tab`, `app.component`) for plain template-bound
writes vs. signal-read-in-template (the latter are fine). Audit the
`ResizeObserver`/`MutationObserver` callbacks (Appendix C §C) **separately** —
they are a raw-callback class, not effects, even though both can write
template-bound fields.

## Appendix B: test-infra catalog

- 69 spec files. `fakeAsync` 43 occurrences / 11 files; `tick(` (zone) subset of
  those; `TestBed.tick(` 21 / 9 files; `discardPeriodicTasks` 5 / 2 files;
  `fixture.detectChanges(` 188 / 39 files; `fixture.whenStable` 0;
  `fixture.autoDetectChanges` 0; zone `flush()` 0 (all `flush(` matches are
  `HttpTestingController.flush`).
- The 11 fakeAsync files (preserve via vitest-patch in Phase 0):
  `vote-state.service`, `browse-prep.service`, `dataset-state.service`,
  `right-panel` (heaviest, 15), `detector-context-bar`, `label-view`,
  `settings-modal`, `progress-modal`, `dashboard`, `detector-card`,
  `dataset-card`.
- Today zone.js enters tests via the **static strategy**: `"zone.js"` in the
  build target's `polyfills` makes `@angular/build:unit-test` inject
  `zone.js/testing` before `setupFiles`. The `test-setup.ts` ProxyZone shim then
  hand-wires `fakeAsync` for Vitest (zone.js's jest patch bails on
  `typeof jest === 'undefined'`). Phase 0.1 replaces this coupling with explicit
  test-target polyfills.

## Appendix C: verified facts (with sources)

All confirmed against angular.dev and angular/angular(-cli) source. Verbatim
quotes abbreviated; URLs are authoritative.

1. **`markForCheck` schedules CD under zoneless — TRUE.** It marks the view dirty
   *and* notifies the scheduler (`NotificationSource.MarkForCheck`), independent
   of zones, so it fires from unpatched callbacks. `zoneless_scheduling_impl.ts`;
   https://angular.dev/guide/zoneless
2. **`AsyncPipe` is zoneless-safe — TRUE.** It calls `markForCheck` on emit; the
   guide lists "`ChangeDetectorRef.markForCheck` (called automatically by
   `AsyncPipe`)". https://angular.dev/guide/zoneless
3. **Canonical scheduling-trigger list — confirmed verbatim:** `markForCheck`
   (via `AsyncPipe`), `ComponentRef.setInput`, updating a signal read in a
   template, bound host/template listener callbacks, attaching a dirty view.
   `afterRender`/`afterNextRender` run *during* a CD pass, not as triggers.
   https://angular.dev/guide/zoneless
4. **Bound `(event)`/`@HostListener` trigger CD; raw `addEventListener` does not
   — TRUE.** The word "Bound" in the trigger list is load-bearing.
   https://angular.dev/guide/zoneless
5. **Effects do not auto-mark host dirty — TRUE.** Only `markForCheck` *within*
   an effect body, or a signal read in the template, reaches the component; an
   `effect → plain field → template` binding has no notification path.
   `render3/reactivity/effect.ts`; https://angular.dev/guide/zoneless
6. **`provideZonelessChangeDetection()` is STABLE — TRUE.** Stable since v20.2;
   zoneless is the default in v21+. Use this name (not the old
   `provideExperimentalZonelessChangeDetection`).
   https://angular.dev/api/core/provideZonelessChangeDetection
7. **`zone.js/plugins/vitest-patch` exists & is documented — NUANCED.** It is the
   documented way to keep `fakeAsync`/`flush`/`waitForAsync` working under the
   Vitest builder, *but* Angular "strongly recommend[s] you start planning to
   convert … to native `async` and Vitest fake timers". Treat as a bridge, not
   the long-term path. https://angular.dev/guide/testing/migrating-to-vitest
8. **Keeping zone.js in TEST polyfills while prod is zoneless — TRUE.** Test
   polyfills and prod bootstrap are independent; loading the patch in the test
   target keeps fakeAsync working. https://angular.dev/guide/testing/migrating-to-vitest
9. **`whenStable()` catches staleness that `detectChanges()` masks — TRUE.** The
   guide says to "avoid using `fixture.detectChanges()` when possible … This
   forces change detection to run when Angular might otherwise have not scheduled
   change detection," and shows `provideZonelessChangeDetection()` +
   `await fixture.whenStable()`. https://angular.dev/guide/zoneless
10. **`NgZone` becomes `NoopNgZone`; `ngZone.run()` alone doesn't drive CD —
    TRUE.** `provideZonelessChangeDetection()` provides
    `{provide: NgZone, useClass: NoopNgZone}`. The methods stay *callable* (the
    guide notes `run`/`runOutsideAngular` "do not need to be removed"), but no
    longer cause a CD pass. `zoneless_scheduling_impl.ts`;
    https://angular.dev/guide/zoneless

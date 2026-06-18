# Zoneless change detection (and the initial-bundle budget)

Status: **Not started — gated on the signal/resource migration.** This doc
records (a) why the production initial-bundle budget sits at 540 kB rather than
something tighter, and (b) what it would take to go zoneless and reclaim the
~35 kB zone.js polyfill. No code change has been made toward zoneless itself;
the only thing that shipped alongside this doc is the budget bump and a
one-component `@defer` (the incompatible-pair explainer), described below.

## Why the initial bundle is "always" over budget

The recurring `bundle initial exceeded maximum budget` warning is **not** a
symptom of UI/dev bloat. An audit of the eager path found the architecture is
already well split:

- **All routes are lazy** (dashboard, label, find, browse are each their own
  chunk).
- **The five heavy modals are `@defer`-loaded** (settings ~80 kB, importer
  ~89 kB, new-detector ~44 kB, keyboard-help ~46 kB, achievements ~18 kB) — they
  download only when opened.
- **`marked` (the one heavy third-party dep) is confined to the deferred
  keyboard-help modal** and never touches the eager path.
- The eager shell components are all tiny (offline-banner, dialog-host,
  toast-container, the achievement-unlock logic host with an empty template).

So the ~530 kB initial bundle is essentially **irreducible framework cost**:
~240 kB Angular (core + router + forms + http + CDK + platform-browser),
~35 kB zone.js, ~20 kB styles, and the necessary app shell. A 525 kB budget
sits right against that floor, so practically *any* shell change — one more
injected service, a few more framework symbols pulled in transitively — tips it
over by a few kB. History bears this out: the budget was bumped to 540 kB once,
pulled back to 525 kB, and then started tripping again.

The budget now sits at **540 kB by design** (see the `"//"` note in
`frontend/angular.json`). This is the documented, user-approved interim state,
not a reflex bump: it acknowledges the framework floor and stops the warning
from being a tripwire on every unrelated shell tweak. The real lever for going
lower is dropping zone.js, which means going zoneless — see below.

### Incidental trim shipped with the bump

`AppComponent`'s template now wraps `<vt-incompatible-pair-explainer />` in
`@defer (when showIncompatibleExplainer)`. The explainer is a rare
error-recovery view (shown only when the active dataset/detector pair is
incompatible), never present at first paint, so deferring it moves it out of the
eager bundle at zero functional cost. This matches the existing `@defer` pattern
already used for the shell's modals.

## What going zoneless would require (and why it's gated)

Dropping zone.js (`provideZonelessChangeDetection()` + removing `"zone.js"` from
`polyfills` in `angular.json`) reclaims ~35 kB raw / ~11 kB transfer from the
eager bundle and is the modern Angular default. But an audit shows VTSearch is
**not** ready for it:

| Pattern | Count | Zoneless-safe? |
|---|---|---|
| `\| async` pipe | 4 usages (2 files) | yes |
| `signal()` / `computed()` / `toSignal()` | 3 (pre-migration baseline) | yes |
| imperative `.subscribe(x => this.field = x)` in components | ~237 | **no** |
| `setTimeout` / `setInterval` / `requestAnimationFrame` | ~78 | **risk** |
| components using `ChangeDetectionStrategy.OnPush` | 0 of 87 | — |

The app relies almost entirely on Zone's "any async tick → re-check everything"
model. The SSE progress service (`progress-events.service.ts`) and keyboard
service (`keyboard.service.ts`) both call `NgZone.run()` purely to trigger CD
after pushing into RxJS subjects, and ~237 component subscriptions assign into
plain fields that Zone then repaints. Under zoneless, every one of those sites
needs to become a signal, an `async` pipe, or an explicit
`ChangeDetectorRef.markForCheck()`, across all 87 components. That is an
**app-wide change-detection rewrite**, not a provider flip.

**Critically, there is no safety net for it in this repo's tooling.** The Vitest
specs drive `fixture.detectChanges()` manually, so they stay green even when the
real app's UI silently stops updating — the exact failure mode zoneless
introduces ("screen goes stale until you click something") is invisible to the
unit suite. There is also no Chrome/Chromium in the Claude-Code-on-the-web
container to QA by hand. So a zoneless flip can only be verified by a human
running the app in a real browser.

### The path to zoneless

Zoneless is the natural *end state* of the signal/resource migration already in
flight — see **`httpresource-migration.md`** (Phases 1–3 shipped: settings,
media, and left-panel reads now run on `rxResource`/signals). The remaining
read services (pollers, `forkJoin` aggregates) and the bulk of the ~237
imperative component subscriptions still need converting. Sequencing:

1. Continue the `httpResource`/signal migration until the imperative
   subscribe-and-assign count is near zero and components are `OnPush`-clean.
2. Convert the `NgZone.run()` re-entry sites (SSE, keyboard) to signal writes /
   `markForCheck`; keep `runOutsideAngular` blocks (those are zoneless-friendly
   already — they intentionally avoid CD).
3. Only then flip `provideZonelessChangeDetection()`, drop `zone.js` from
   `polyfills` and from `package.json`, and update `test-setup.ts` (the
   `fakeAsync`/`ProxyZone` bootstrap assumes zone.js is loaded).
4. After the flip, a human must browser-QA every interactive surface (voting,
   sorting, progress bars, browse canvas, modals) before merge.

## Open follow-ups

- Drive the imperative-subscribe count down via `httpresource-migration.md` so
  step 1 above becomes tractable.
- Once zoneless lands and zone.js is gone, drop the 540 kB budget back toward
  the framework floor (~495 kB) and delete the `"//"` rationale in
  `angular.json` that points here.

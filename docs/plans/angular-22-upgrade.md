# Angular 21 → 22 upgrade

**Status:** Researched, not started. This doc captures the v22-specific
findings and the scoped-work options; the reusable step-by-step upgrade
mechanics (staged `ng update`, Vitest wiring gotchas, `overrides` drift,
budget rules) live in [`angular-21-upgrade.md`](angular-21-upgrade.md) and are
not duplicated here.

## Background

Angular 22 released June 2026. VTSearch is on 21.2.x and has already
pre-adopted every headline default v22 ships for new apps: **zoneless**
(`provideZonelessChangeDetection()`, `zone.js` dropped end-to-end),
**Vitest** (`@angular/build:unit-test` + jsdom), **standalone** components,
and the **esbuild `@angular/build:application`** builder. So the upgrade is a
dependency bump gated on TypeScript 6.0, not a code migration — nearly every
v22 breaking change is a no-op here (see the audit below).

## The upgrade (required work)

- **TypeScript 5.9.3 → 6.0 is the hard gate.** v22 drops support for TS ≤ 5.9.
  Bump `frontend/tsconfig.json` (`typescript: ~5.9.3` → `~6.0`) and expect a
  handful of TS-6.0 strictness/deprecation fixes across the ~90-component app
  and the generated OpenAPI client types.
- **Node:** v22's floor is Node 22 (drops Node 20). The container is already on
  Node 22.x, so no image change is needed — but re-verify at bump time.
- **Run `ng update @angular/core @angular/cli @angular/cdk`**, one major step,
  gate on `build:prod` (no `▲ [WARNING]`) + `./run-tests.sh`. Re-pin the
  `overrides` block (esbuild/vite/postcss/…) to whatever the new
  `@angular/build` pulls — this is the likeliest source of a broken
  `npm ci`/build.

<!-- item-sep -->

- **Breaking-change audit (done — nearly all no-ops).** Confirmed against the
  v22 changelog and this codebase:
  - Components default to `OnPush`: **no-op**, all 91/91 components already set
    `ChangeDetectionStrategy.OnPush` explicitly.
  - `paramsInheritanceStrategy` defaults to `'always'`: **no-op**, routes are
    flat (no `children:`), nothing reads inherited parent params.
  - `data-*` no longer property-binds: **no-op**, every usage is `[attr.data-*]`.
  - `provideRoutes` / `createNgModuleRef` / `ComponentFactory[Resolver]` /
    `checkNoChanges()` removed: **no-op**, none used.
  - `FetchBackend` becomes the default HTTP backend: **low risk** —
    `app.config.ts` uses `provideHttpClient(withInterceptors(...))` with no
    upload-progress reliance (only `HttpEventType.Response` in the error
    interceptor), so the XHR→Fetch default switch is safe. Verify uploads
    (`drop-zone`) at runtime anyway.
  - Duplicate input binding / multiple matching selectors now compile-error:
    would surface at `build:prod`; none known.

<!-- item-sep -->

## Optional follow-ups (each its own opt-in effort; not required by v22)

- **Signal-API modernization (umbrella)** — the codebase still uses decorator
  `@Input()` (127 sites), `@Output()` (12), and `@ViewChild` (40) across ~60
  component files rather than signal `input()`/`output()`/`viewChild()`/`model()`;
  constructor injection is already gone bar one util site. These are **not** a
  v22 requirement (available since v17.3, work fine in 21 and 22), so this is a
  "modernize toward signal-first authoring" project decoupled from the version
  bump. Sliced into per-cluster, ~PR-sized issues (each folds any coupled
  `ngOnChanges` into `computed`/`effect`, since signal inputs don't fire
  `ngOnChanges`), tagged with a recommended Claude model by difficulty:

  - [ ] #2540 — Importer pickers (Fable 5)
  - [ ] #2541 — Progress widgets (Fable 5)
  - [ ] #2542 — Dashboard cards + create/combine modals (Fable 5 → Sonnet 5)
  - [ ] #2543 — Importer modal shell (Sonnet 5)
  - [ ] #2544 — Left panel · controls (Sonnet 5)
  - [ ] #2545 — Right panel (Sonnet 5)
  - [ ] #2546 — Top-level modals (Sonnet 5)
  - [ ] #2547 — Misc shared leaf components (Sonnet 5)
  - [ ] #2548 — Left panel · lists / virtual scroll (Sonnet 5 → Opus 4.8)
  - [ ] #2549 — Center-panel media viewers (Opus 4.8)
  - [ ] #2550 — Browse canvas cluster (Opus 4.8)

<!-- item-sep -->

- **Angular Aria (now stable)** — headless, styleable a11y components. Directly
  relevant: `vt-modal` hand-rolls focus management via CDK's `CdkTrapFocus`
  across ~24 dialogs (the ~7 kB eager cost noted in the `angular.json` budget
  comment). Evaluate whether Aria can replace some of that.

<!-- item-sep -->

- **Signal Forms (now stable)** — a signals-native forms story for the
  settings/import/config forms. Was "low value, experimental" at 21; now stable,
  so worth a re-look if forms churn.

<!-- item-sep -->

- **Async Signals stable** (`resource()`/`httpResource()`/`rxResource()`) —
  overlaps the existing [`httpresource-migration.md`](httpresource-migration.md)
  plan; production-ready in 22.

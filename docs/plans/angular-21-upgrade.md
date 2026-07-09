# Angular 19 → 21 upgrade + Vitest spec migration

**Status: all phases shipped.** Angular is on 21.2.17 (`npm audit --omit=dev` = 0 vulnerabilities, was 6 high) and the Vitest spec migration is complete — all 68 spec files / 760 `it` blocks run headless under Vitest, pass, and are wired into `./run-tests.sh`. This doc is **kept as the migration reference for future Angular bumps**; the reusable upgrade path, risks, and non-obvious gotchas are preserved below, with the completed work collapsed under "What shipped".

## Open follow-ups

Both the Angular bump and the Vitest migration are done; remaining loose ends are small:

- **Nothing quarantined.** The Jasmine→Vitest port ended with all 760 `it` blocks passing, so there is no quarantine/triage list owed.
- **`test-setup.ts` ProxyZone shim is a workaround.** It reimplements zone.js's jest patch for Vitest. If a future `zone.js`/Angular release ships first-class Vitest support for `fakeAsync`, delete the shim and rely on the builder. Watch the `@angular/build:unit-test` release notes (still marked experimental in 21).
- **Migrate fakeAsync specs to Vitest fake timers (optional).** Longer term, the 11 `fakeAsync` specs could move to `vi.useFakeTimers()` and drop the zone-testing dependency entirely; only worth it if/when the app goes zoneless (see "Deferred / out of scope").
- **Resolved during the upgrade:** Node/TS floors (container's Node 22 + TS 5.9.3 satisfy v21, no setup-script change); overrides drift (none needed); budget regression (initial crept 521.97 → 524.30 kB but stays under the 525 kB warn budget — watch it on future bumps); `@types/jasmine` removed, `tsconfig.spec.json` now uses `vitest/globals`.

## Migration reference (for future bumps)

### Upgrade path (staged, one major at a time)

Angular's supported migration is **one major per step** via `ng update`; do not skip a version. Run each step on the working branch, commit between steps, and gate on `npm run build:prod` (no `▲ [WARNING]`) + `./run-tests.sh core` before proceeding. The exact peer-dep floors are whatever `ng update` prints — treat it as authoritative.

```
cd frontend && npx ng update @angular/core@<N> @angular/cli@<N> @angular/cdk@<N>
```

Per step, expect: a **TypeScript floor** bump and a possible **Node floor** bump (verify the container satisfies it — if not, that's a setup-script/image change outside `frontend/`, surface it to the user); automated schematics for renamed/removed APIs; and a possible `vite`/`esbuild` `overrides` revisit if the new `@angular-devkit` pulls a different range (stale overrides can break `npm ci` or the build). Gate after each: `build:prod` clean, TypeScript clean, app boots; after the final major also smoke the core flow (load dataset → train detector → sort).

### Vitest wiring — gotchas future bumps must preserve

The version bump alone does **not** enable Vitest. The pieces that landed and must survive a re-bump:

- **Builder wiring.** `build`/`serve`/`extract-i18n`/`test` all use the canonical `@angular/build:*` builders (not the `@angular-devkit/build-angular:*` aliases) — **required**, because the `@angular/build:unit-test` runner warns and fails to inherit polyfills (notably `zone.js/testing`) when `buildTarget` points at a devkit alias. `test` (watch) / `test:ci` (`ng test --no-watch`) scripts each have a `pretest*` hook regenerating the API client.
- **jsdom polyfills** (`src/test-setup.ts`, via `setupFiles`): inert stubs for `EventSource`, `HTMLMediaElement.play/pause/load`, `HTMLCanvasElement.getContext` — jsdom omits these and throws "Not implemented".
- **fakeAsync / ProxyZone bootstrap** (the subtle one). `zone.js/testing` only auto-establishes the ProxyZone that `fakeAsync()`/`tick()` need for Jasmine/Mocha/Jest runners, and Vitest is none of those, so every `fakeAsync` spec threw "Expected to be running in 'ProxyZone'". `test-setup.ts` replicates zone.js's jest patch against Vitest's globals (describe → sync zone, it/hooks → proxy zone). See the ProxyZone follow-up above.
- **Isolation + cascade guard.** The unit-test builder defaults to `isolate: false`; `vitest.config.ts` flips it to `isolate: true` so a dirty `TestBed` singleton can't poison later files, and `test-setup.ts` resets the TestBed at the start of each test so a throwing teardown doesn't cascade "test module already instantiated".
- **run-tests.sh wiring.** Vitest runs on the full `./run-tests.sh` (the real gate — no CI) and on the frontend-only `./run-tests.sh frontend` group; intentionally **not** on the fast `core` path (which keeps only the compile-only build check).

### Risks / watch-items

- **Node/TS floors** may exceed what the container ships (a setup-script/image change outside `frontend/` — surface it, don't assume).
- **`overrides` drift** (`vite`, `esbuild`, `postcss`, …) is the likeliest source of a broken `npm ci`/build after a bump — review at each step.
- **Budget regressions** from a major bump — fix bloat, don't raise budgets (CLAUDE.md rule).
- **Vitest first-run red is expected on new specs** — realign to current behavior, don't weaken.
- **No mobile/responsive concerns** — desktop-only app; ignore viewport deprecation noise.

### Deferred / out of scope (record only)

Angular 21 modernization carrots, each its own opt-in effort — give any picked-up item its own plan file and link it here:

- **Zoneless change detection** — drops `zone.js` from polyfills (bundle/budget win) but effectively needs broad signal adoption first (VTSearch barely uses signals).
- **`httpResource` / `resource`** — could trim `switchMap`/manual-subscription boilerplate across the ~67 RxJS API services. **Scoped in its own plan: `httpresource-migration.md`** (recommends `rxResource` wrapping the generated client; not started).
- **Signal Forms** — low value (no forms here) and experimental in 21.
- **ARIA directives, CLI MCP server** — marginal for this app.

## What shipped

- **Angular 19 → 20 → 21**, one major per step via `ng update`. All `@angular/*` at `21.2.17`, `@angular-devkit/build-angular` + `@angular/cli` at `21.2.16`, TypeScript `5.9.3` (container's Node 22 satisfies the floor). Removed the never-imported `@angular/platform-browser-dynamic` (a dead dep confusing `ng update`'s peer resolver).
- **Block control-flow migration** — mandatory on 21 (legacy `*ngIf`/`*ngFor`/`*ngSwitch` removed); `ng update` converted 51 templates to `@if`/`@for`/`@switch`.
- **Out-of-root docs asset fixed** — Angular 21 forbids `assets[].input` outside the workspace root, breaking `../docs/user → /assets/docs`; replaced with a committed symlink `frontend/docs-assets → ../docs/user` (docs stay single-sourced). Overrides (`vite`/`esbuild`) needed no changes.
- **Gates green:** `build:prod` clean (0 warnings, initial 524.30 kB < 525 kB budget), `npm audit --omit=dev` = 0, full `./run-tests.sh` passes (4841 tests).
- **Vitest builder + scripts** — `test`/`test:ci` targets on `@angular/build:unit-test` (jsdom); all `build`/`serve`/`extract-i18n` moved to canonical `@angular/build:*`; `@types/jasmine` removed, `vitest`/`jsdom` added dev-only. jsdom polyfills, ProxyZone shim, and isolation guard added to `test-setup.ts` (details under Migration reference).
- **Jasmine → Vitest port** (mechanical, all specs): `toBeTrue()`→`toBe(true)`, `spyOn`→`vi.spyOn`, `.and.returnValue`→`.mockReturnValue` (+ `callFake`/`resolveTo` equivalents), `spy.calls.mostRecent().args`→`spy.mock.lastCall`, `done`-callback tests → Promise; `tsconfig.spec.json` types `jasmine`→`vitest/globals`.
- **Real spec drift fixed.** The 754 `it` blocks had only ever typechecked; ~186 stale specs were realigned to current behavior (new `/api/embedders`, `/api/media-types`, `/api/settings`, SSE-vs-HTTP, vote body `{vote}`→`{target}`, etc.) — no tests deleted/skipped/weakened.
- **One real bug surfaced + fixed.** `DocumentViewerComponent` bound a plain string into `<object [data]>` (a RESOURCE_URL sink, NG0904 in prod — document viewing was latently broken); fixed by wrapping the same-origin URL with `DomSanitizer.bypassSecurityTrustResourceUrl`.

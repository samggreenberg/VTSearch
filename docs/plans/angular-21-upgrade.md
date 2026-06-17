# Angular 19 → 21 upgrade + Vitest spec migration

Status: **Upgrade shipped (Steps 1 & 2); Vitest migration deferred.** Angular is
now on 21.2.17 and `npm audit --omit=dev` reports 0 vulnerabilities (was 6
high). The Vitest spec migration phase has **not** started — see "What shipped"
and "Open follow-ups" below.

## What shipped

- **Angular 19 → 20 → 21**, one major per step via `ng update`. All `@angular/*`
  packages at `21.2.17`, `@angular-devkit/build-angular` + `@angular/cli` at
  `21.2.16`, TypeScript at `5.9.3`. (Node floor satisfied by the container's
  Node 22; no setup-script change needed.)
- **`@angular/platform-browser-dynamic` removed** — it was never imported
  (`main.ts` uses standalone `bootstrapApplication`), and as a dead dep it
  confused `ng update`'s peer resolver. Removing it unblocked the family bump.
- **Block control-flow migration applied** — Angular 21 makes this migration
  mandatory (the legacy `*ngIf`/`*ngFor`/`*ngSwitch` structural directives are
  removed in v21), so `ng update` converted 51 component templates to
  `@if`/`@for`/`@switch`. This was listed as "optional/deferred" in the original
  plan but is no longer optional on 21; it rode along with the bump.
- **Out-of-root docs asset fixed.** Angular 21 forbids `assets[].input` paths
  outside the workspace root, which broke the `../docs/user → /assets/docs`
  mapping (in-app user guide). Replaced with a **committed symlink**
  `frontend/docs-assets → ../docs/user`; angular.json now references
  `docs-assets`. The builder's `isSubDirectory` check is pure path math, so the
  symlink resolves cleanly under the workspace root while the docs stay
  single-sourced.
- **Overrides** (`vite`, `esbuild`, …) needed no changes — Angular 21 pulls
  `vite@6.4.3`/`esbuild@0.28.1`, both already within the existing ranges.
- **Gates green:** `build:prod` clean (0 warnings, initial bundle 524.30 kB <
  525 kB budget), spec files typecheck, `npm audit --omit=dev` = 0
  vulnerabilities, full `./run-tests.sh` passes (4841 tests).

The Vitest spec migration (below) is intentionally deferred to its own effort.

## Original design (for reference)

## Why

`npm audit` (run in `frontend/`) reports **6 high-severity advisories**, all
rooted in a single package: `@angular/core <= 19.2.25`
([GHSA-rgjc-h3x7-9mwg](https://github.com/advisories/GHSA-rgjc-h3x7-9mwg),
"Angular Client Hydration DOM Clobbering & Response-Cache Poisoning"). The
other five flagged packages (`common`, `forms`, `platform-browser`,
`platform-browser-dynamic`, `router`) are transitive dependents of `core`.
The only remediation npm offers is `npm audit fix --force`, which jumps to
Angular 21 — a breaking multi-major upgrade.

**Live exposure is effectively nil.** The advisory is specific to Angular's
SSR / client-hydration path. VTSearch is a pure client-side SPA built to
`../static/` and served by Flask: no `provideClientHydration`, no
`provideServerRendering`, no `platformServer`. So this is a "stay current /
keep the audit clean" upgrade, not an urgent patch.

**The real VTSearch-specific payoff is Vitest.** Angular 21 ships Vitest as
the standard unit-test runner, replacing Karma/Jasmine. Karma was removed from
this repo because the cloud container (Ubuntu 24.04) has no Chrome/Chromium, so
our **68 spec files / 754 `it` blocks currently only typecheck and never run**
(`package.json`'s `test` script is a no-op echo). Vitest runs in Node via
jsdom/happy-dom — no browser — so the upgrade can turn the frontend unit suite
back on in the same environment that runs `./run-tests.sh`. That is a genuine
capability gain beyond clearing the audit.

The other Angular 21 headline features are **nice-to-have but not free** and
are explicitly out of scope here (see "Deferred / out of scope"). None of them
arrive automatically with the version bump.

## Current state (as of this plan)

- **Angular:** `^19.2.x` across `core/common/compiler/forms/platform-browser{,-dynamic}/router/cdk`.
- **Change detection:** zone-based — `provideZoneChangeDetection({ eventCoalescing: true })` in `src/app/app.config.ts`; `zone.js ~0.15.0`; `polyfills: ["zone.js"]` in `angular.json`.
- **Signals:** barely used (1 component: `field-hint-icon`). No `linkedSignal`/`resource` usage.
- **Forms:** **none.** No `ReactiveFormsModule`/`FormGroup`/`FormControl` anywhere in non-spec code.
- **Data layer:** ~67 RxJS-based API service files using `provideHttpClient(withInterceptors([...]))` with three interceptors (active-context, achievements-refresh, error).
- **Build:** `@angular-devkit/build-angular:application` (esbuild/Vite under the hood), output to `../static`, prod budgets `initial` 525kB warn / 1MB error and `anyComponentStyle` 8kB warn / 10kB error.
- **Tests:** `@types/jasmine` only; `tsconfig.spec.json` includes `src/**/*.spec.ts` for typecheck; no runner wired. `prebuild`/`prebuild:prod` run `ng-openapi-gen` to regenerate the API client.
- **Toolchain pins via `overrides`:** `vite ^6.4.2`, `esbuild ^0.28.1`, `postcss`, `uuid`, `qs`, etc. — these will likely need revisiting because Angular 21 pins its own vite/esbuild ranges.
- **TypeScript:** `~5.7.2`.

## Upgrade path (staged, one major at a time)

Angular's supported migration is **one major per step** via `ng update`; do not
skip 20. Run each step on the working branch, commit between steps, and gate on
`npm run build:prod` + `./run-tests.sh core` (which runs the frontend build
check) before proceeding.

The exact peer-dep floors below are from memory and **must be confirmed by the
`ng update` output**, which is authoritative — treat the bullets as "expect
something in this area", not as gospel version numbers.

### Step 1 — Angular 19 → 20

```
cd frontend && npx ng update @angular/core@20 @angular/cli@20 @angular/cdk@20
```

Expect:
- **TypeScript bump** (~5.8) and a **Node floor** bump — verify the container's
  Node satisfies it; if not, that's a setup-script/image change, flag to the user.
- Automated schematics for renamed/removed APIs (Angular 20 stabilized several
  signal APIs and removed long-deprecated ones).
- `provideExperimentalZonelessChangeDetection` → `provideZonelessChangeDetection`
  rename lands here (only relevant if/when we go zoneless — we are not, yet).
- Revisit the `vite`/`esbuild` `overrides` if the new `@angular-devkit` pulls a
  different range; stale overrides can break the build or `npm ci`.

Gate: `npm run build:prod` clean (no `▲ [WARNING]`), TypeScript clean, app boots.

### Step 2 — Angular 20 → 21

```
cd frontend && npx ng update @angular/core@21 @angular/cli@21 @angular/cdk@21
```

Expect:
- Another TS floor (~5.9) and possible Node floor bump — verify against the container.
- Zoneless becomes the **default for new projects**; our existing
  `provideZoneChangeDetection` is untouched, so behavior is preserved. (Going
  zoneless is a separate, deferred effort.)
- Builder/runner changes around testing (see Vitest phase).
- Re-check budgets: a major Angular bump can shift baseline bundle size; if
  `initial` creeps past 525kB, fix the bloat (per CLAUDE.md, do **not** just
  raise the budget without explicit approval).

Gate: `npm run build:prod` clean, `./run-tests.sh core` green, app boots and a
manual smoke of the core flow (load dataset → train detector → sort) works.

## Vitest spec migration (the payoff — separate phase, after the bump)

The version bump alone does **not** enable Vitest. After Angular is on 21:

1. **Wire the Vitest unit-test builder.** Add the `unit-test` target to
   `angular.json` pointing at the Angular Vitest builder, with an
   `application`-builder-compatible config. Use jsdom (or happy-dom) as the DOM
   environment so it runs headless in the container.
2. **Replace the no-op `test` script** with the real runner (`ng test`, now
   Vitest-backed) and add a non-watch CI variant (e.g. `test:ci`).
3. **Port Jasmine → Vitest in the 68 spec files.** Mechanical but broad:
   `jasmine.createSpy`/`createSpyObj` → `vi.fn`/Vitest mocks, `spyOn(...)` call
   sites, `jasmine.objectContaining` → `expect.objectContaining`, matcher
   differences, and the `TestBed`/zoneless harness wiring. Swap
   `@types/jasmine` for Vitest types in `tsconfig.spec.json`.
4. **Expect real failures.** These 754 `it` blocks have never executed — only
   typechecked. Turning them on will surface stale assertions, drifted mocks,
   and over-mocked tests that pass by accident. Budget time to triage/fix or
   quarantine genuinely-broken specs (with a tracked list, not silent skips).
5. **Hook into `./run-tests.sh`.** Decide with the user whether the frontend
   unit run joins the `core` group (alongside the existing build check) or runs
   as its own step. `run-tests.sh` is the source of truth (no CI backstop), so
   the Vitest run must be wired in for it to actually guard anything.

Gate: Vitest runs headless in the cloud container with 0 failures (or an
explicit, agreed quarantine list), wired into `run-tests.sh`.

## Deferred / out of scope (record only — do not do as part of this work)

These are the modernization carrots Angular 21 unlocks. They are **opt-in
migrations on top of the upgrade**, not part of clearing the audit, and each is
its own effort:

- **Zoneless change detection.** Smaller bundle (drops `zone.js` from
  polyfills, helps the budget) and cleaner stack traces, but effectively
  requires adopting signals broadly first — VTSearch barely uses them today.
- **`httpResource` / `resource`.** Could trim `switchMap`/manual-subscription
  boilerplate across the ~67 RxJS API services, adoptable incrementally. Worth
  a focused pass later if the data layer is being touched anyway.
- **Signal Forms.** Low value here — VTSearch uses no forms — and experimental
  in 21 (not stable until ~22/23).
- **ARIA directives, CLI MCP server.** Marginal for this app.

If any of these get picked up, give them their own plan file and link it here.

## Risks / watch-items

- **Node/TS floors** may exceed what the cloud container ships; that's a
  setup-script/image change outside `frontend/`, so surface it to the user
  rather than assuming.
- **`overrides` drift** (`vite`, `esbuild`, `postcss`, …) is the most likely
  source of a broken `npm ci` or build after the bump — review them at each step.
- **Budget regressions** from the major bump — fix bloat, don't raise budgets
  (CLAUDE.md rule).
- **Vitest failures are expected, not exceptional** — the suite has never run.
  Don't treat first-run red as a blocker on the upgrade itself; it's the
  migration phase's job to resolve.
- **No mobile/responsive concerns** — desktop-only app (CLAUDE.md), so ignore
  any viewport-related deprecation noise.

## Suggested sequencing

1. Branch + `ng update` to 20, build/test gate, commit.
2. `ng update` to 21, build/test gate, commit. **Audit is now clean** — this is
   a valid stopping point if Vitest is deferred.
3. Wire Vitest builder + scripts, commit.
4. Port specs in batches (by folder), fixing as you go, committing per batch.
5. Wire Vitest into `run-tests.sh`, final gate, open PR to `dev`.

## Open follow-ups

The Angular bump is done; everything below is the **deferred Vitest phase** and
its loose ends:

- **Vitest spec migration (whole phase, not started).** Wire the Vitest
  `unit-test` builder + scripts, port the 68 Jasmine spec files / 754 `it`
  blocks (which have only ever typechecked, never run), triage the real
  failures that surface, and hook the run into `./run-tests.sh`. See the
  "Vitest spec migration" section above for the step list.
- Decide whether the Vitest run joins `run-tests.sh`'s `core` group or stands
  alone (needs user input) — defer until the migration is actually picked up.
- Triage list for any specs quarantined during the Jasmine→Vitest port.
- `@types/jasmine` is still in `devDependencies` and `tsconfig.spec.json` still
  lists `"types": ["jasmine"]`; the specs typecheck against Jasmine globals for
  now. Swap to Vitest types as part of the migration.
- **Resolved during the upgrade:** Node/TS floors (container's Node 22 + TS
  5.9.3 satisfy v21, no setup-script change); overrides drift (none needed);
  budget regression (initial crept 521.97 → 524.30 kB but stays under the
  525 kB warn budget — watch it on future bumps).

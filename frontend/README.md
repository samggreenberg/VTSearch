# VTSearch Frontend

Angular SPA for the VTSearch media explorer. Built with Angular CLI 21.2 and TypeScript 5.9, standalone components, zoneless change detection, and the esbuild `@angular/build:application` builder.

## Prerequisites

- **Node.js 22+** (LTS recommended)
- **npm** (bundled with Node.js)

## Development server

Start the Flask backend first (`python app.py --local` from the project root), then:

```bash
npm install   # first time only
npm start
```

This starts the Angular dev server at `http://localhost:4200/` with a proxy that forwards `/api/*` requests to the Flask backend at `localhost:5000` (configured in `proxy.conf.json`). The application automatically reloads when source files change.

## Building for production

```bash
npm run build:prod
```

This compiles the Angular app and outputs the build artifacts to `../static/` (the project root's `static/` directory), where Flask serves them. Output files: `index.html`, `main.js`, `polyfills.js`, `styles.css`.

## Architecture

The SPA's architecture lives in **[`../docs/FRONTEND.md`](../docs/FRONTEND.md)**:
directory layout and feature-area boundaries, the service layer, the
**zoneless change-detection rules**, active dataset/detector propagation and
the `X-Dataset-Id` / `X-Detector-Id` headers, the generated OpenAPI client, and
the component/modal composition conventions. Styling conventions are in
[`../docs/style-guide.md`](../docs/style-guide.md).

This file stays scoped to the build: commands, test wiring, and the Angular
upgrade runbook.

## Running unit tests

```bash
npm run test:ci   # headless, one shot
npm test          # watch mode
```

Specs run on **Vitest + jsdom** via the `@angular/build:unit-test` builder — no
browser required, so they work in the cloud container. Karma is gone. The suite
also runs from the repo root as part of `./run-tests.sh` (full) and
`./run-tests.sh frontend` (build + `npm audit` + Vitest); the fast
`./run-tests.sh core` path keeps only the compile-only `build:prod` check.

## Upgrading Angular

The reusable mechanics for the next major bump. Version-specific findings for a
given bump live in their own plan under `docs/plans/` (e.g.
`angular-22-upgrade.md`).

### Upgrade path (staged, one major at a time)

Angular's supported migration is **one major per step** via `ng update`; do not
skip a version. Run each step on the working branch, commit between steps, and
gate on `npm run build:prod` (no `▲ [WARNING]`) + `./run-tests.sh core` before
proceeding. The exact peer-dep floors are whatever `ng update` prints — treat it
as authoritative.

```bash
npx ng update @angular/core@<N> @angular/cli@<N> @angular/cdk@<N>
```

Per step, expect: a **TypeScript floor** bump and a possible **Node floor** bump
(verify the container satisfies it — if not, that's a setup-script/image change
outside `frontend/`, surface it to the user); automated schematics for
renamed/removed APIs; and a possible `vite`/`esbuild` `overrides` revisit if the
new `@angular-devkit` pulls a different range (stale overrides can break
`npm ci` or the build). Gate after each: `build:prod` clean, TypeScript clean,
app boots; after the final major also smoke the core flow (load dataset → train
detector → sort).

### Test wiring a bump must preserve

- **Builder wiring.** `build`/`serve`/`extract-i18n`/`test` all use the canonical
  `@angular/build:*` builders (not the `@angular-devkit/build-angular:*`
  aliases) — **required**, because the `@angular/build:unit-test` runner warns
  and fails to inherit polyfills when `buildTarget` points at a devkit alias.
  `test` (watch) / `test:ci` (`ng test --no-watch`) each have a `pretest*` hook
  regenerating the API client.
- **Dedicated `test` build configuration.** `angular.json`'s `test` target points
  at its own `build:test` configuration rather than `development`, so the spec
  polyfills stay decoupled from the production build. Both polyfill arrays are
  empty today (zoneless end to end); keep them explicit rather than implicit.
- **jsdom polyfills** (`src/test-setup.ts`, via `setupFiles`): inert stubs for
  `EventSource`, `ResizeObserver`, `Element.scrollIntoView`, `CSS.escape`,
  `HTMLMediaElement.play/pause/load`, and `HTMLCanvasElement.getContext` — jsdom
  omits these and throws "Not implemented".
- **Zoneless specs, native async.** No spec uses `fakeAsync`/`tick`; time is
  driven with `vi.useFakeTimers()` + real macrotask drains, and fixture-creating
  specs install `provideZonelessChangeDetection()`. A bump must not reintroduce
  `zone.js` — it is absent from the polyfills and from `package.json`.
- **Isolation + cascade guard.** The unit-test builder defaults to
  `isolate: false`; `vitest.config.ts` flips it to `isolate: true` so a dirty
  `TestBed` singleton can't poison later files, and `test-setup.ts` resets the
  TestBed at the start of each test so a throwing teardown doesn't cascade
  "test module already instantiated".

### Risks / watch-items

- **Node/TS floors** may exceed what the container ships (a setup-script/image
  change outside `frontend/` — surface it, don't assume).
- **`overrides` drift** (`vite`, `esbuild`, `postcss`, …) is the likeliest source
  of a broken `npm ci`/build after a bump — review at each step.
- **Budget regressions** from a major bump — fix bloat, don't raise budgets
  (CLAUDE.md rule).
- **Vitest first-run red is expected on new specs** — realign to current
  behavior, don't weaken.
- **No mobile/responsive concerns** — desktop-only app; ignore viewport
  deprecation noise.

## Code scaffolding

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```

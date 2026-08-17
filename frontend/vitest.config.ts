import { defineConfig } from 'vitest/config';

/**
 * Vitest config consumed by the `@angular/build:unit-test` builder (wired via
 * `runnerConfig` in angular.json). The builder merges this on top of its own
 * defaults.
 *
 * `isolate` is left at the builder's default (`false`): all 155 spec files
 * share one module graph per worker rather than each getting a fresh one.
 *
 * This file previously forced `isolate: true`, to stop a spec that leaves the
 * `TestBed` instantiated (a component whose teardown throws, an unverified
 * `HttpTestingController`) from cascading "test module already instantiated"
 * into every file that ran after it. That cascade is now handled directly, and
 * far more cheaply, by the defensive `getTestBed().resetTestingModule()` that
 * `src/test-setup.ts` runs in a `beforeEach` — it fires before each spec's own
 * `beforeEach`, so a dirty TestBed is cleared at the start of the next test
 * instead of being prevented by rebuilding the whole module graph 155 times.
 *
 * Per-file isolation was costing more than the entire rest of the suite: it
 * turned ~89s of actual test execution into ~155s of jsdom environment
 * construction plus ~92s of setup-file re-evaluation. Dropping it takes the
 * unit run from ~88s wall / ~258 CPU-seconds to ~33s / ~68 — the single
 * largest saving in the whole test pipeline.
 *
 * The trade-off worth knowing: with isolation off, spec files also share
 * *application* module-level state (service singletons, module-scoped caches),
 * so a spec that mutates such state can in principle leak into a later file.
 * Nothing in the suite does today — the change was validated with three
 * consecutive shuffled-order runs (`sequence.shuffle`), all 155 files and 2051
 * tests passing. If a file-order-dependent failure ever does appear, fix the
 * leaking spec rather than re-enabling isolation for all 155; setting
 * `isolate: true` here again would buy back correctness at 3-4x the runtime.
 */
export default defineConfig({
  test: {
    isolate: false,
  },
});

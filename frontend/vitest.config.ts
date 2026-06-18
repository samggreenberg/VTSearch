import { defineConfig } from 'vitest/config';

/**
 * Vitest config consumed by the `@angular/build:unit-test` builder (wired via
 * `runnerConfig` in angular.json). The builder merges this on top of its own
 * defaults.
 *
 * We override the builder's default `isolate: false` (which it picks to mimic
 * the old Karma/Jasmine single-context run). The Angular `TestBed` is a
 * module-level singleton; with isolation off, every spec file shares one
 * instance, so a single spec that leaves the TestBed instantiated (e.g. a
 * component whose teardown throws, or an unverified HttpTestingController)
 * cascades "test module already instantiated" into every file that runs after
 * it. Per-file isolation gives each spec a fresh module graph — and thus a
 * fresh TestBed — so failures stay contained to the spec that caused them.
 */
export default defineConfig({
  test: {
    isolate: true,
  },
});

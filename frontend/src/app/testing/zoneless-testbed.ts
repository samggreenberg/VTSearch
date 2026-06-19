import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';

/**
 * Zoneless `TestBed` helpers for the zoneless-migration effort.
 *
 * See `docs/plans/zoneless-migration.md` (Phase 0). The migration converts
 * VTSearch's frontend off zone.js. The failure mode it introduces — "a value
 * changed but the view silently went stale" — does NOT surface in a normal spec
 * today, because every component spec drives `fixture.detectChanges()` by hand,
 * which force-renders and therefore masks the staleness. These helpers let a
 * spec run its `TestBed` under `provideZonelessChangeDetection()` so the fixture
 * refreshes only what change detection actually *schedules* — turning the
 * headless suite into a real zoneless-staleness oracle, component by component.
 *
 * How to use, as a component migrates (Phase 1–2):
 *  1. Add `provideZonelessChangeDetection()` to the spec's providers — either via
 *     `provideZoneless()` spread into `configureTestingModule({ providers: [...] })`
 *     or by calling `configureZoneless({ ... })` instead of `configureTestingModule`.
 *  2. **Remove** the manual `fixture.detectChanges()` pumps (do not supplement
 *     them — a leftover manual pump both re-masks staleness and can race
 *     auto-detect into `ExpressionChangedAfterItHasBeenChecked`/NG0100). Adding
 *     `provideZonelessChangeDetection()` flips `ComponentFixtureAutoDetect` on by
 *     default, so the fixture refreshes itself on scheduled CD.
 *  3. Drive updates through the *same channel the app uses* (push to the service
 *     subject/signal, dispatch a bound event), then `await settleZoneless()` (or
 *     `await fixture.whenStable()`) — NOT `fixture.detectChanges()`.
 *  4. Assert on `fixture.nativeElement.querySelector(...)` (rendered DOM), not on
 *     `component.someField`. A missing notification ⇒ stale DOM ⇒ failing
 *     assertion.
 *
 * Enabling zoneless **globally** is intentionally NOT done here: none of the
 * ~221 subscribe sites are ready yet, so a global flip would turn the suite red
 * en masse. It is adopted per component, in lockstep with the conversions.
 */

/**
 * Providers fragment that puts a `TestBed` under zoneless change detection.
 * Spread it into a spec's `configureTestingModule({ providers: [...] })`.
 */
export function provideZoneless(): unknown[] {
  return [provideZonelessChangeDetection()];
}

/**
 * Convenience wrapper around `TestBed.configureTestingModule` that prepends the
 * zoneless provider to the supplied module definition. Returns the `TestBed` so
 * callers can chain (mirrors `configureTestingModule`'s return).
 */
export function configureZoneless(
  moduleDef: Parameters<typeof TestBed.configureTestingModule>[0] = {},
) {
  return TestBed.configureTestingModule({
    ...moduleDef,
    providers: [...provideZoneless(), ...(moduleDef.providers ?? [])],
  });
}
